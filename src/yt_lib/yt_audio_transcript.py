from __future__ import annotations
# import datetime
import os
# import re
import json
# import logging
import time
import whisper
import asyncio
import concurrent.futures
import threading
from pathlib import Path
from typing import Any, List, Dict, Optional, TypeVar
from yt_dlp import YoutubeDL
from youtube_transcript_api import FetchedTranscript
from dataclasses import dataclass
from collections.abc import Callable
from yt_lib.utils.ffmpeg_bootstrap import ensure_ffmpeg_on_path, get_ffmpeg_binary_path
from yt_lib.utils.paths import resolve_cache_paths
from yt_lib.utils.yt_ids import extract_video_id
from yt_lib.utils.log_utils import get_logger # , log_tree


# -----------------------------
# Logging setup
# -----------------------------
logger = get_logger(__name__)

PREFERRED_LANGS = ["en", "en-US", "en-GB", "es", "es-419", "es-ES"]

ProgressCallback = Callable[[float, str], None]

# ----------------- Whisper chunking configuration -----------------
# Duration (in seconds) for each Whisper chunk
CHUNK_DURATION_SECONDS = 30.0
# Overlap (in seconds) between consecutive chunks
CHUNK_OVERLAP_SECONDS = 5.0


# # ----------------- Helpers -----------------

# ----------------- Output management -----------------

def _get_audio_dir() -> Path:
    """Folder for temporary storage of yt_dlp audio cache.
    We delete these if they are over a day old in the code below.
    """
    return resolve_cache_paths(
        app_name = "audio",
        start = Path(__file__)).app_cache_dir 


def _get_transcript_cache_path(video_id: str) -> Path:
    """Return the path to the cached transcript JSON for this video."""

    return resolve_cache_paths(
        app_name = "transcripts",
        start = Path(__file__)).app_cache_dir / f"{video_id}.json"


# ----------------- Audio + Whisper -----------------


def download_audio(url: str, video_id: str) -> Path:
    """Download audio from a YouTube video using yt-dlp, with simple caching."""
    audio_dir = _get_audio_dir()

    # If we already have any file named <video_id>.* reuse it
    existing = sorted(audio_dir.glob(f"{video_id}.*"))
    if existing:
        audio_path = existing[0]
    else:
        output_template = str(audio_dir / "%(id)s.%(ext)s")

        # Make sure we have an ffmpeg directory for yt_dlp to use.
        ffmpeg_dir = get_ffmpeg_binary_path()

        ydl_opts = {
            "extract-audio": True,               # Extract audio from video.
            "verbose": False,
            "quiet": True,                       # Suppress normal output.
            "no_warnings": True,
            "skip_download": False,
            "extract_flat": False,
            "format": "bestaudio/best",
            "outtmpl": output_template,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                    "preferredquality": "192",
                }
            ],
        }

        if ffmpeg_dir:
            # Explicitly tell yt_dlp where ffmpeg lives
            ydl_opts["ffmpeg_location"] = ffmpeg_dir

        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            # FFmpegExtractAudio will produce <id>.wav in the same dir
            audio_path = audio_dir / f"{info['id']}.wav"

    return audio_path


# =========================
# ASYNC (cooperative) SECTION
# =========================
#
# Goal: allow MCP long-jobs to be cancelled while transcribing.
# Strategy:
#   * keep the MCP tool function async
#   * yield to the event loop between chunks (await sleep(0))
#   * do the heavy Whisper work in a dedicated single worker thread so the loop
#     stays responsive and cancellation can be observed between chunks.
#
# IMPORTANT LIMITATION:
#   Cancellation is *cooperative* and takes effect BETWEEN chunks.
#   If a single chunk transcription takes a long time, cancellation will be
#   delayed until that chunk completes. To improve responsiveness, use a smaller
#   chunk_duration (e.g., 10-15s).

_WHISPER_THREAD_LOCAL = threading.local()

def _get_thread_local_whisper_model(model_name: str):
    """Load/cache Whisper model in the *worker thread* (thread-local)."""
    model = getattr(_WHISPER_THREAD_LOCAL, "model", None)
    cached_name = getattr(_WHISPER_THREAD_LOCAL, "model_name", None)
    if model is None or cached_name != model_name:
        _WHISPER_THREAD_LOCAL.model = whisper.load_model(model_name)
        _WHISPER_THREAD_LOCAL.model_name = model_name
    return _WHISPER_THREAD_LOCAL.model

def _transcribe_chunk_in_worker_thread(model_name: str, segment_audio):
    """Runs inside the dedicated worker thread."""
    model = _get_thread_local_whisper_model(model_name)
    return whisper.transcribe(model, segment_audio)


@dataclass
class ProgressInfo:
    fraction: float          # 0.0 .. 1.0
    message: str = ""        # optional human-readable status


def make_sample_progress_fn(num_samples: int) -> Callable[[int, int, str], ProgressInfo]:
    """
    Returns a function that converts (processed_samples, total_samples, phase) into ProgressInfo.
    Uses samples, not chunk count, so it stays accurate even for a partial last chunk.
    """
    total = max(1, int(num_samples))

    def _progress(processed_samples: int, total_samples: int = total, phase: str = "") -> ProgressInfo:
        done = min(max(int(processed_samples), 0), total_samples)
        frac = done / total_samples
        return ProgressInfo(frac, f"{phase} {frac*100:.1f}%".strip())

    return _progress

async def transcribe_with_whisper_async(
    audio_path: Path,
    model_name: str = "small",
    chunk_duration: float = CHUNK_DURATION_SECONDS,
    overlap: float = CHUNK_OVERLAP_SECONDS,
    *,
    yield_every_n_chunks: int = 1,
    progress_cb: Optional[ProgressCallback] = None,
) -> Optional[List[Dict]]:
    """Async/transcribable variant of `transcribe_with_whisper`.

    This function yields control back to the event loop between chunks so the job
    can be cancelled by the caller.

    Args:
        audio_path: Path to the audio file.
        model_name: Whisper model name to use.
        chunk_duration: Duration (in seconds) of each chunk.
        overlap: Overlap (in seconds) between chunks.
        yield_every_n_chunks: Yield to loop every N chunks (default: 1).

    Returns:
        List of Whisper chunk dicts, or None if transcription fails.
    """
    logger.info(
        "🎧 (async) Transcribing %s with Whisper model '%s' in %.1fs chunks (overlap %.1fs)",
        audio_path,
        model_name,
        chunk_duration,
        overlap,
    )

    # Load and resample audio (this is blocking but usually fast-ish; keep it sync).
    audio = whisper.load_audio(str(audio_path))
    sample_rate = 16000

    samples_per_chunk = int(chunk_duration * sample_rate)
    overlap_samples = int(overlap * sample_rate)
    num_samples = audio.shape[0]

    if samples_per_chunk <= 0:
        raise ValueError("chunk_duration must be > 0")

    if overlap_samples >= samples_per_chunk:
        logger.warning(
            "Overlap (%.1fs) >= chunk duration (%.1fs); disabling overlap.",
            overlap,
            chunk_duration,
        )
        overlap_samples = 0

    step = samples_per_chunk - overlap_samples
    if step <= 0:
        step = samples_per_chunk

    chunks: List[Dict] = []
    chunk_index = 0

    # progress fn based on samples:
    prog = make_sample_progress_fn(num_samples)


    def report(processed_samples: int, phase: str):
        if progress_cb:
            info = prog(processed_samples, num_samples, phase)
            progress_cb(info.fraction, info.message)

    report(0, "loading")


    # Dedicated single worker thread: keeps model pinned to one thread and avoids
    # 'random thread' issues that can happen if the default threadpool hops threads.
    try:
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper") as ex:
            for start_sample in range(0, num_samples, step):
                # Checkpoint: if the caller cancelled, this is where CancelledError lands.
                if yield_every_n_chunks > 0 and (chunk_index % yield_every_n_chunks) == 0:
                    await asyncio.sleep(0)

                end_sample = min(start_sample + samples_per_chunk, num_samples)
                segment_audio = audio[start_sample:end_sample]

                if segment_audio.size == 0:
                    break
                report(start_sample, f"transcribing chunk {chunk_index}")

                start_time = start_sample / sample_rate
                end_time = min(num_samples, end_sample) / sample_rate

                logger.debug(
                    "🧩 (async) Chunk %d: samples [%d:%d] -> time [%.2f, %.2f]s",
                    chunk_index,
                    start_sample,
                    end_sample,
                    start_time,
                    end_time,
                )

                try:
                    chunk = await loop.run_in_executor(
                        ex, _transcribe_chunk_in_worker_thread, model_name, segment_audio
                    )
                except asyncio.CancelledError:
                    # Best effort cleanup; note: a chunk already running in the worker
                    # thread will still run to completion, but we stop awaiting more work.
                    logger.info("🛑 Transcription cancelled during chunk %d.", chunk_index)
                    raise

                chunks.append(chunk)
                chunk_index += 1

                # after finishing this chunk, we’ve processed up through end_sample
                report(end_sample, f"completed chunk {chunk_index-1}")

                if end_sample >= num_samples:
                    break
            report(num_samples, "done")
        return chunks
    finally:
        audio_path.unlink(missing_ok=True)



async def fetch_audio_transcript_async(
        url: str,
        prefer_langs: Optional[List[str]] = None,
        *,
        model_name: str = "small",
        chunk_duration: float = CHUNK_DURATION_SECONDS,
        overlap: float = CHUNK_OVERLAP_SECONDS,
        progress_cb: Optional[ProgressCallback] = None, 
    ) -> Optional[List[Dict]]:
    """Async wrapper around `fetch_audio_transcript` with cooperative cancellation.

    Major differences from the sync version:
      * audio download happens in a worker thread
      * transcription uses `transcribe_with_whisper_async()` so cancellation can be
        observed between chunks

    Cancellation behavior:
      * Cancelling the MCP long-job raises `asyncio.CancelledError`.
      * Cancellation takes effect BETWEEN chunks (see async transcribe notes).
    """
    if prefer_langs is None:
        prefer_langs = ["en", "es"]

    if progress_cb:
        progress_cb(0.0, "starting")

    video_id = extract_video_id(url)
    cache_path = _get_transcript_cache_path(video_id)

    if cache_path.exists():
        logger.info("✅ Using cached Whisper transcript for %s", video_id)
        try:
            transcript = json.loads(cache_path.read_text(encoding="utf-8"))
            if progress_cb:
                progress_cb(1.0, "finished")
            return transcript
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("⚠️ Failed to load cached transcript %s: %s; recomputing.", cache_path, exc)

    # Ensure ffmpeg on PATH
    if progress_cb:
        progress_cb(0.01, "downloading audio")

    ffmpeg_dir = ensure_ffmpeg_on_path()
    logger.info("✅ Using ffmpeg from: %s", ffmpeg_dir)

    # Download audio (blocking) -> worker thread
    try:
        audio_path = await asyncio.to_thread(download_audio, url, video_id)
    except asyncio.CancelledError:
        logger.info("🛑 Cancelled during audio download.")
        raise

    # Yield after download so cancellation can be observed immediately.
    await asyncio.sleep(0)
    if progress_cb:
        progress_cb(0.05, "audio downloaded")

    # Transcribe with cooperative cancellation
    chunks: Optional[List[Dict]] = None
    try:
        chunks = await transcribe_with_whisper_async(
                audio_path,
                model_name=model_name,
                chunk_duration=chunk_duration,
                overlap=overlap,
                yield_every_n_chunks=1,
                progress_cb=lambda frac, msg: progress_cb(0.05 + 0.95 * frac, msg) if progress_cb else None,
            )
    except asyncio.CancelledError:
        # If cancelled, best effort cleanup of the downloaded audio file
        audio_path.unlink(missing_ok=True)
        raise

    # Cache result
    if chunks is not None:
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
            logger.info("💾 Saved Whisper transcript cache to %s", cache_path)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("⚠️ Failed to write transcript cache %s: %s", cache_path, exc)
    if progress_cb:
        progress_cb(1.0, "finished")

    return chunks


async def youtube_audio_json_async(url: str,
                                   prefer_langs: Optional[List[str]] = None,
                                   *,
                                   progress_cb: Any = None,
                                   ) -> str | None:
    """Async version of youtube_audio_json (supports cooperative cancellation)."""

    cb = progress_cb if callable(progress_cb) else None

    chunks = await fetch_audio_transcript_async(url, prefer_langs, progress_cb=cb)
    if chunks is None:
        return None
    return json.dumps(chunks, ensure_ascii=False, indent=2)


async def youtube_audio_text_async(url: str,
                                   prefer_langs: Optional[List[str]] = None,
                                   *,
                                   progress_cb: Any = None,
                                   ) -> str | None:

    """Async version of youtube_audio_text (supports cooperative cancellation)."""

    cb = progress_cb if callable(progress_cb) else None

    chunks = await fetch_audio_transcript_async(url, prefer_langs, progress_cb=cb)
    if chunks is None:
        return None
    # Same logic as youtube_audio_text: join segment texts
    full_text_parts: List[str] = []
    for chunk in chunks:
        text_part = (chunk.get("text") or "").strip()
        if text_part:
            full_text_parts.append(text_part)
    return " ".join(full_text_parts).strip()



# ----------------- CLI -----------------

# ---------------------------------------------------------------------------
# Test / CLI entry point (not a formal unit test, just a quick way to run and see results)q
# ---------------------------------------------------------------------------

def test() -> None:
    """ CLI entry point to test the YouTube to text tool. """
    import fastmcp, torch
    from datetime import timedelta
    print("\nfastmcp:", fastmcp.__version__)
    print("torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Device count:", torch.cuda.device_count())
    print("Whisper models available:", whisper.available_models())

    # CLI for testing the YouTube to text tool.
    yt_url = "https://www.youtube.com/watch?v=DAYJZLERqe8"    # 6:32 comedy
    # yt_url = "https://www.youtube.com/watch?v=_uQrJ0TkZlc"    # 6 + hours!
    # yt_url = "https://www.youtube.com/watch?v=Ro_MScTDfU4"    # 30:34 Python tutorial < 30 Mins
    # yt_url = "https://www.youtube.com/watch?v=gJz4lByMHUg"    # Just music
    # yt_url = "https://youtu.be/N23vXA-ai5M?list=PLC37ED4C488778E7E&index=1"
    # yt_url = "https://youtu.be/N23vXA-ai5M"
    # yt_url = "https://www.youtube.com/watch?v=ulebPxBw8Uw"

    while not yt_url:
        yt_url = input("Enter YouTube URL: ").strip()
        if not yt_url:
            logger.warning("⚠️ Please paste a valid YouTube URL.")

    ffmpeg_path = ensure_ffmpeg_on_path()
    if not ffmpeg_path:
        raise SystemExit(
            "❌ FFmpeg is not available. Please install "
            "FFmpeg and ensure it is on the system PATH."
        )
    logger.info("✅ Using ffmpeg at %s", get_ffmpeg_binary_path())

    start = time.perf_counter()
    json_trans = youtube_audio_json(yt_url)
    elapsed = time.perf_counter()-start
    print("\n\n--- JSON AUDIO TRANSCRIPT ---\n")
    print(f"{json_trans}")
    print(f"\n✅ Transcribed in {str(timedelta(seconds=elapsed))} seconds.\n")

    start = time.perf_counter()
    text_trans = youtube_audio_text(yt_url)
    elapsed = time.perf_counter()-start
    print("\n\n--- TEXT AUDIO TRANSCRIPT ---\n")
    print(f"{text_trans}")
    print(f"\n✅ Transcribed in {str(timedelta(seconds=elapsed))} seconds.\n")
    
if __name__ == "__main__":
    test()

