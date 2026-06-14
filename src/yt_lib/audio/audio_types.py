""" Defines data classes for audio processing and transcription in the YouTube library."""
from __future__ import annotations
from dataclasses import dataclass
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
    preferred_langs: list[str] = ["en", "en-US", "en-GB", "es", "es-419", "es-ES"]

    @property
    def whisper_model(self) -> str:
        """Returns the name of the Whisper model."""
        return self.whisper_model

    @property
    def chunk_duration_seconds(self) -> float:
        """Returns the chunk duration in seconds."""
        return self.chunk_duration_seconds

    @property
    def chunk_overlap_seconds(self) -> float:
        """Returns the chunk overlap in seconds."""
        return self.chunk_overlap_seconds

    @property
    def sample_rate(self) -> int:
        """Returns the preferred sample rate for audio processing."""
        return self.sample_rate

    @property
    def preferred_langs(self) -> list[str]:
        """Returns the list of preferred languages for transcription."""
        return self.preferred_langs

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


@dataclass(slots=True, frozen=True)
class AudioTranscriptChunk:
    """ Represents a chunk of transcribed audio."""
    index: int
    start: float
    end: float
    text: str


@dataclass(slots=True, frozen=True)
class AudioTranscript:
    """ Represents the complete transcript for a YouTube video's audio."""
    video_id: str
    title: str | None
    snippets: list[AudioTranscriptChunk]
