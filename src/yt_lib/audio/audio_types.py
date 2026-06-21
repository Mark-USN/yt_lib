""" Defines data classes for audio processing and transcription in the YouTube library."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

@dataclass(slots=True, frozen=True)
class AudioSettingsBase:
    """ Configuration settings for audio processing and transcription."""
    whisper_model: str = "small"
    # ----------------- Whisper chunking configuration -----------------
    # Duration (in seconds) for each Whisper chunk
    chunk_duration_seconds: float = 30.0
    # Overlap (in seconds) between consecutive chunks
    chunk_overlap_seconds: float = 5.0
    # Preferred sample rate for audio processing (e.g., for Whisper)
    sample_rate: int = 16_000
    preferred_langs: list[str] = field(
        default_factory=lambda: ["en", "en-US", "en-GB", "es", "es-419", "es-ES"]
    )


AUDIO_SETTINGS = AudioSettingsBase()  # Default settings instance

@dataclass(slots=True, frozen=True)
class AudioSource:
    """ Represents the source audio information for a YouTube video."""
    video_id: str
    title: str | None
    source_path: Path


@dataclass(slots=True, frozen=True)
class ConvertedAudio:
    """ Represents the converted audio information for a YouTube video."""
    wav_path: Path
    sample_rate: int = 16_000
    channels: int = 1
