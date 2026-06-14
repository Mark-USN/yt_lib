"""YouTube transcript tool for FastMCP.

This module fetches transcripts/subtitles for a YouTube video using
`youtube-transcript-api`, with a best-effort on-disk cache.

Key behaviors:
- Accepts either a full YouTube URL or a bare video id.
- Tries preferred languages first (descending priority).
- Falls back to translating the first available transcript when possible.
- Writes/reads a JSON cache under a configurable cache directory.

Environment variables:
- MCP_CACHE_DIR: override the base cache directory.

Notes for AI agents:
- The primary public entrypoints (MCP tools) are `yt_json()` and
  `yt_text()`.
- `fetch_transcript()` is the core I/O function; it returns raw transcript
  snippets as JSON-serializable dictionaries.
"""

from __future__ import annotations

import json
import os
import secrets
import threading
import time
from datetime import timedelta
from contextlib import contextmanager
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol
# from urllib.parse import parse_qs, urlparse
from youtube_transcript_api import (
    FetchedTranscript,
    NoTranscriptFound,
    NotTranslatable,
    TranscriptsDisabled,
    TranslationLanguageNotAvailable,
    YouTubeTranscriptApi,
)
from yt_lib.utils.log_utils import get_logger
from yt_lib.yt_types import extract_video_id, Snippet

logger = get_logger(__name__)

class TranscriptPathProvider(Protocol):
    """ Prototype to translate app_context methods into a simple protocol for this module,
        to avoid a hard dependency on the full app context. 
    """
    def transcript_path(self, video_id: str) -> Path:
        """ Directory and file name to 'cache' transcripts for a given video ID.
            Args:
                video_id: The YouTube video ID for which to provide a transcript cache path.
            Returns:
                A Path object representing the file path where the transcript for the given video
                ID should be cached.
        """


# Global context for cache path provider; must be set by MCP at runtime before use.
_CONTEXT: TranscriptPathProvider | None = None


def set_context(context: TranscriptPathProvider) -> None:
    """ Set the global context for transcript cache path provision.
        Args:
            context: An object implementing the `TranscriptPathProvider` protocol, which provides
                    a method `transcript_path(video_id: str) -> Path` to determine where to cache
                    transcripts for a given video ID.
    """
    global _CONTEXT             #pylint: disable=global-statement
    _CONTEXT = context


def _get_transcript_cache_path(video_id: str) -> Path:
    """ Get the cache path for a given video ID using the global context.
        Args:
            video_id: The YouTube video ID.
        Returns:
            File path to the cached transcript JSON for the video.
        Raises:
            RuntimeError: If the global context has not been set.
    """
    if _CONTEXT is None:
        msg = "yt_transcript runtime context has not been initialized."
        raise RuntimeError(msg)

    return _CONTEXT.transcript_path(video_id)

# A small, process-wide throttle to avoid overwhelming upstream services when a
# workflow engine fans out work in parallel.
_MAX_CONCURRENT_FETCHES = int(os.environ.get("MCP_TRANSCRIPT_MAX_CONCURRENCY", "6"))
_FETCH_SEM = threading.BoundedSemaphore(value=max(1, _MAX_CONCURRENT_FETCHES))


# class TranscriptSnippet(TypedDict):
#     """ Raw transcript snippet shape returned by `FetchedTranscript.to_raw_data()`."""
#     text: str
#     start: float
#     duration: float


#: Default language priority (descending).
PREFERRED_LANGS: tuple[str, ...] = (
    "en",
    "en-US",
    "en-GB",
    "es",
    "es-419",
    "es-ES",
)

@contextmanager
def _file_lock(lock_path: Path):
    """ Best-effort cross-platform advisory file lock.
        Args:
            lock_path: Path to the lock file (e.g. "<cachefile>.lock").

        We use a sidecar lock file (e.g. "<video>.json.lock") so multiple workers
        don't concurrently read/write the same cache file.

        On Windows this uses msvcrt.locking; on POSIX it uses fcntl.flock.
        If neither is available, the lock becomes a no-op (still safe with atomic
        writes, but may do duplicate work).
    """

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "a+b")  # noqa: SIM115 (must stay open during lock)
    try:
        try:
            if os.name == "nt":
                import msvcrt  # pylint: disable=import-outside-toplevel

                # Lock 1 byte from the start of the file.
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl  # pylint: disable= import-error, import-outside-toplevel

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        except Exception as err:  # pylint: disable=broad-exception-caught
            logger.debug("Cache lock unavailable (%s): %s", lock_path, err)

        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt  # pylint: disable=import-outside-toplevel

                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl  # pylint: disable= import-error, import-outside-toplevel

                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        fh.close()


def _atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """ Atomically write text to `path`.
        Args:
            path: The destination file path.
            text: The text content to write.
            encoding: The text encoding to use (default: "utf-8").

        Writes to a temp file in the same directory, then replaces the destination.
        This prevents readers from seeing partial writes.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(6)}"
    tmp_path = path.with_name(tmp_name)
    try:
        tmp_path.write_text(text, encoding=encoding)
        os.replace(tmp_path, path)
    finally:
        # If anything failed before replace, clean up.
        try:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        except OSError:
            pass

def _as_raw_snippets(transcript: FetchedTranscript |
                        list[Snippet]) -> list[Snippet]:
    """ Normalize transcript output to raw JSON-serializable snippet dicts.
        Args:
            transcript: Either a `FetchedTranscript` object or a list of raw snippet dicts.
        Returns:
            A list of raw snippet dictionaries.
    """

    if isinstance(transcript, list):
        return transcript
    return transcript.to_raw_data()  # already JSON-friendly


def transcript_to_list_and_cache(
    transcript: FetchedTranscript | list[Snippet] | None,
    cache_path: Path,
) -> list[Snippet] | None:
    """ Convert a fetched transcript into raw snippet dictionaries and cache to disk.
        This is "best effort" caching: any write failure is logged and the transcript
        is still returned.
        Args:
            transcript: Transcript object from `youtube-transcript-api` (or raw list)
                or None.
            cache_path: Destination JSON path.
        Returns:
            The transcript as `list[Snippet]`, or None.
    """

    if transcript is None:
        return None

    transcript_list = _as_raw_snippets(transcript)

    try:
        _atomic_write_text(
            cache_path,
            json.dumps(transcript_list, ensure_ascii=False, indent=2),
        )
        logger.info("Saved transcript cache to %s", cache_path)
    except OSError as exc:
        logger.warning("Failed to write transcript cache %s: %s", cache_path, exc)

    return transcript_list


def fetch_transcript(
    url_or_id: str,
    prefer_langs: Sequence[str] | None = None,
) -> list[Snippet] | None:
    """ Fetch a transcript for a YouTube video and return raw snippet dicts.
        Args:
            url_or_id: YouTube URL or video id.
            prefer_langs: Preferred language codes (descending priority). If None,
                defaults to :data:`PREFERRED_LANGS`.
        Returns:
            A list of transcript snippets or None when no transcript exists.
        Raises:
            TranscriptsDisabled: When transcripts are disabled for the video.
            NoTranscriptFound: When no transcript is found for the video.
    """

    langs = list(prefer_langs) if prefer_langs is not None else list(PREFERRED_LANGS)
    video_id = extract_video_id(url_or_id)
    cache_path = _get_transcript_cache_path(video_id)
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")

    # A per-video lock prevents cache corruption and redundant upstream calls
    # when a workflow fans out transcript fetches in parallel.
    with _file_lock(lock_path):
        # 1) Best-effort cache read.
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if isinstance(cached, list):
                    cache_path.touch()
                    logger.info("Using cached transcript for %s", video_id)
                    return cached  # type: ignore[return-value]
            except (OSError, json.JSONDecodeError) as exc:  # pylint: disable=broad-exception-caught
                logger.warning(
                    "Failed to load cached transcript %s: %s; recomputing.",
                    cache_path,
                    exc,
                )

        ytt_api = YouTubeTranscriptApi()

        # 2) Try preferred languages (descending priority) via `fetch()`.
        try:
            with _FETCH_SEM:
                fetched = ytt_api.fetch(
                    video_id,
                    languages=langs or None,
                    preserve_formatting=True,
                )
            return transcript_to_list_and_cache(fetched, cache_path)
        except TranscriptsDisabled:
            logger.info("Transcripts disabled for %s", video_id)
            return None
        except NoTranscriptFound:
            # Preferred languages unavailable; we may still be able to fetch some other
            # language or translate.
            pass
        except Exception as err:  # pylint: disable=broad-exception-caught
            logger.warning("Transcript fetch failed for %s: %s", video_id, err)
            return None

        # 3) Fallback: list available transcripts; try translating the first available.
        try:
            with _FETCH_SEM:
                transcript_list = ytt_api.list(video_id)
        except (TranscriptsDisabled, NoTranscriptFound) as err:
            logger.info("No transcripts for %s: %s", video_id, err)
            return None
        except Exception as err:  # pylint: disable=broad-exception-caught
            logger.warning("Transcript list failed for %s: %s", video_id, err)
            return None

        # Log available language codes for debugging.
        try:
            available_langs = [getattr(tr, "language_code", "?") for tr in transcript_list]
            logger.debug("Available languages for %s: %s", video_id, available_langs)
        except Exception:  # pylint: disable=broad-exception-caught
            pass

        first_tr = next(iter(transcript_list), None)
        if first_tr is None:
            return None

        try:
            with _FETCH_SEM:
                if langs:
                    logger.info("Translating first available transcript to %s", langs[0])
                    fetched = first_tr.translate(langs[0]).fetch(preserve_formatting=True)
                else:
                    fetched = first_tr.fetch(preserve_formatting=True)
        except (NotTranslatable, TranslationLanguageNotAvailable):
            logger.warning(
                "Translation failed; returning subtitles in original language %s.",
                getattr(first_tr, "language_code", "?"),
            )
            with _FETCH_SEM:
                fetched = first_tr.fetch(preserve_formatting=True)
        except Exception as err:  # pylint: disable=broad-exception-caught
            logger.warning("Transcript fallback fetch failed for %s: %s", video_id, err)
            return None

        return transcript_to_list_and_cache(fetched, cache_path)


def yt_json(
    url_or_id: str,
    prefer_langs: Sequence[str] | None = None,
) -> list[Snippet] | None:
    """ Return the raw transcript snippets (typed), or None.
        This is the "structured" variant intended for typed workflow engines.
        It returns the same data that `fetch_transcript()` produces (a list of
        Snippet dicts), without JSON serialization.
        Args:
            url_or_id: YouTube URL or video id.
            prefer_langs: Preferred language codes (descending priority).
        Returns:
            A JSON string or None.
    """
    return fetch_transcript(url_or_id, prefer_langs)

# ---------------------------------------------------------------------------
# Test / CLI entry point (not a formal unit test, just a quick way to run and see results)q
# ---------------------------------------------------------------------------

def test() -> None:
    """CLI entry point to test transcript retrieval (outside MCP)."""


    yt_url = "https://www.youtube.com/watch?v=ulebPxBw8Uw"

    while not yt_url:
        yt_url = input("Enter YouTube URL: ").strip()
        if not yt_url:
            logger.warning("Please paste a valid YouTube URL.")

    start = time.perf_counter()
    trans = yt_json(yt_url)
    elapsed = time.perf_counter() - start
    print("\n\n--- JSON TRANSCRIPT ---\n")
    # `yt_jason()` already returns a JSON string.
    print(trans)
    print(f"\nTranscribed in {timedelta(seconds=elapsed)}.\n")



if __name__ == "__main__":
    test()
