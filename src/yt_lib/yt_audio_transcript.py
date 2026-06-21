""" Module for transcribing YouTube video audio using yt_dlp to download the audio, ffmpeg to
    convert it to a suitable format, and Whisper to perform the transcription.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from yt_lib.audio.ffmpeg_audio import convert_to_16k_mono_wav
from yt_lib.audio.audio_types import AUDIO_SETTINGS
from yt_lib.audio.whisper_audio import transcribe_wav_in_chunks
from yt_lib.audio.ytdlp_audio import download_audio_source
from yt_lib.yt_types import ProgressReporter, Snippet
from yt_lib.utils.log_utils import get_logger


# -----------------------------
# Logging setup
# -----------------------------
logger = get_logger(__name__)



@dataclass(slots=True, frozen=True)
class AudioPaths:
    """ Provides methods to determine where to cache audio files for a given video ID."""
    audio_dir: Path

    def audio_source_dir(self) -> Path:
        """ Directory for temporary storage of yt_dlp audio cache.
        Returns:
            Path to the audio cache directory.
        """
        pth = self.audio_dir / "source"
        pth.mkdir(parents=True, exist_ok=True)
        return pth

    def audio_source_path(self, audio_file: str) -> Path:
        """ Path for temporary storage of a specific yt_dlp audio file.
            Args:
                audio_file: The name of the audio file.
            Returns:
                Path to the audio file.
        """
        return self.audio_source_dir() / audio_file

    def audio_converted_dir(self) -> Path:
        """ Directory for temporary storage of converted audio files.
            Returns:
                Path to the converted audio cache directory.
        """
        pth = self.audio_dir / "processed"
        pth.mkdir(parents=True, exist_ok=True)
        return pth

    def audio_converted_path(self, audio_file: str) -> Path:
        """ Path for temporary storage of a specific converted audio file.
            Args:
                audio_file: The name of the converted audio file.
            Returns:
                Path to the converted audio file.
        """
        return self.audio_converted_dir() / audio_file


# Global info for cache path provider; must be set by MCP at runtime before use.
_INFO: AudioPaths | None = None


def set_info(info: AudioPaths) -> None:
    """ Set the global info for transcript cache path provision.
        Args:
            info: An object implementing the `AudioPaths` protocol, which provides
                    methods to determine where to cache audio files for a given video ID.
    """
    global _INFO             #pylint: disable=global-statement
    _INFO = info


def _get_audio_source_dir() -> Path:
    """ Folder for temporary storage of yt_dlp audio cache.
        We delete these if they are over a day old in the code below.
        Returns:
            Path to the audio cache directory.
    """
    if _INFO is None:
        msg = "yt_transcript runtime info has not been initialized."
        raise RuntimeError(msg)

    return _INFO.audio_source_dir()

def _get_audio_converted_dir() -> Path:
    """ Folder for temporary storage of yt_dlp audio cache.
        We delete these if they are over a day old in the code below.
        Returns:
            Path to the audio cache directory.
    """
    if _INFO is None:
        msg = "yt_transcript runtime info has not been initialized."
        raise RuntimeError(msg)

    return _INFO.audio_converted_dir()




async def transcribe_youtube_audio_async(
    url: str,
    *,
    model_name: str = AUDIO_SETTINGS.whisper_model,
    chunk_duration_s: float = AUDIO_SETTINGS.chunk_duration_seconds,
    overlap_s: float = AUDIO_SETTINGS.chunk_overlap_seconds,
    progress_rptr: ProgressReporter | None = None,
    keep_files: bool = False,
) -> list[Snippet]:
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
            A list of Snippet objects containing the transcription snippets.
    """
    source_dir: Path = _get_audio_source_dir()
    wav_dir: Path = _get_audio_converted_dir()
    if progress_rptr:
        await progress_rptr.set_total(100)
        await progress_rptr.increment(5)
        await progress_rptr.set_message("Starting transcription process.")
        await progress_rptr.set_message(f"Downloading audio to {source_dir}")
    source = await asyncio.to_thread(download_audio_source, url, source_dir)
    if progress_rptr:
        await progress_rptr.increment(30)
        await progress_rptr.set_message(f"Audio download complete: {source}")
        await progress_rptr.set_message("Converting audio to 16 kHz mono WAV")
    converted = await convert_to_16k_mono_wav(
        source.source_path,
        wav_dir,
        output_stem=source.video_id,
    )
    if progress_rptr:
        await progress_rptr.increment(25)
        await progress_rptr.set_message(f"Audio conversion complete: {converted}")
        await progress_rptr.set_message("Running Whisper transcription")
    snippets: list[Snippet] = await transcribe_wav_in_chunks(
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

    return snippets
