""" Module for transcribing YouTube video audio using yt_dlp to download the audio, ffmpeg to
    convert it to a suitable format, and Whisper to perform the transcription.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Protocol
from yt_lib.audio.ffmpeg_audio import convert_to_16k_mono_wav
from yt_lib.audio.audio_types import AudioTranscript, AUDIO_SETTINGS
from yt_lib.audio.whisper_audio import transcribe_wav_in_chunks
from yt_lib.audio.ytdlp_audio import download_audio_source
from yt_lib.yt_types import ProgressReporter # ,Snippet
from yt_lib.utils.log_utils import get_logger


# -----------------------------
# Logging setup
# -----------------------------
logger = get_logger(__name__)

class PathProvider(Protocol):
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

    def audio_dir(self) -> Path:
        """ Directory to temperary store audio files for a given video ID.
            Returns:
                A Path object representing the directory where audio files for the given video
                ID should be stored.
        """

    def audio_path(self, audio_file: str) -> Path:
        """ Directory and file name to temperarily store audio files for a given video ID.
            Args:
                audio_file: The name of the audio file.
            Returns:
                A Path object representing the file path where the audio file for a video
                ID should be stored.
        """

# Global context for cache path provider; must be set by MCP at runtime before use.
_CONTEXT: PathProvider | None = None


def set_context(context: PathProvider) -> None:
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


def _get_audio_dir() -> Path:
    """ Folder for temporary storage of yt_dlp audio cache.
        We delete these if they are over a day old in the code below.
        Returns:
            Path to the audio cache directory.
    """
    if _CONTEXT is None:
        msg = "yt_transcript runtime context has not been initialized."
        raise RuntimeError(msg)

    return _CONTEXT.audio_dir()



async def transcribe_youtube_audio_async(
    url: str,
    *,
    model_name: str = AUDIO_SETTINGS.whisper_model,
    chunk_duration_s: float = AUDIO_SETTINGS.chunk_duration_seconds,
    overlap_s: float = AUDIO_SETTINGS.chunk_overlap_seconds,
    progress_rptr: ProgressReporter | None = None,
    keep_files: bool = True,
) -> AudioTranscript:
    """ Transcribe the audio from a YouTube video given its URL, using yt_dlp to download the audio,
        convert it to WAV, and then transcribe it using Whisper.
        Args:
            url: The URL of the YouTube video.
            model_name: The Whisper model name to use.
            chunk_duration_s: Duration of each audio chunk in seconds.
            overlap_s: Overlap between audio chunks in seconds.
            progress_rptr: Optional progress reporter callback.
            keep_files: Whether to keep the downloaded and converted audio files.
        Returns:
            An AudioTranscript object containing the video ID, title, and transcription snippets.
    """
    source_dir: Path = Path(_get_audio_dir() / "source_audio")
    wav_dir: Path = Path(_get_audio_dir() / "wav")
    if progress_rptr:
        await progress_rptr.set_total(100)
        await progress_rptr.increment(5)
        await progress_rptr.set_message("Starting transcription process.")
        await progress_rptr.set_message("Downloading audio")
    source = await asyncio.to_thread(download_audio_source, url, source_dir)
    if progress_rptr:
        await progress_rptr.increment(30)
        await progress_rptr.set_message("Converting audio to 16 kHz mono WAV")
    converted = await convert_to_16k_mono_wav(
        source.source_path,
        wav_dir,
        output_stem=source.video_id,
    )
    if progress_rptr:
        await progress_rptr.increment(25)
        await progress_rptr.set_message("Running Whisper transcription")
    snippets = await transcribe_wav_in_chunks(
        converted.wav_path,
        model_name=model_name,
        chunk_duration=chunk_duration_s,
        overlap=overlap_s,
        progress_rptr=progress_rptr,
    )
    # text = "\n".join(chunk.text for chunk in chunks if chunk.text)

    if progress_rptr:
        await progress_rptr.set_message("Transcription complete")

    if not keep_files:
        source.source_path.unlink(missing_ok=True)
        converted.wav_path.unlink(missing_ok=True)

    return AudioTranscript(
        video_id=source.video_id,
        title=source.title,
        snippets=snippets,
    )
