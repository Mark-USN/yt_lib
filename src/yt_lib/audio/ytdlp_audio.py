""" Audio source downloader using yt-dlp."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yt_dlp
from yt_lib.audio.audio_types import AudioSource


def download_audio_source(url: str, output_dir: Path) -> AudioSource:
    """ Download audio from the given URL using yt-dlp and return the path to the downloaded file.
        Args:
            url: The URL of the audio source to download.
            output_dir: Directory where the downloaded file will be saved.
        Returns:
            An AudioSource object containing the video ID, title, and path to the
            downloaded audio file.
        Raises:
            FileNotFoundError: If the downloaded audio file is not found.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded_path: Path | None = None

    def progress_hook(status: dict[str, Any]) -> None:
        """ Progress hook to capture the downloaded file path when the download is finished."""
        nonlocal downloaded_path

        if status.get("status") == "finished":
            filename = status.get("filename")
            if isinstance(filename, str):
                downloaded_path = Path(filename)

    ydl_opts: dict[str, Any] = {
        "format": "bestaudio/best",
        "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "progress_hooks": [progress_hook],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

        if downloaded_path is None:
            # Fallback: okay for simple downloads without postprocessing.
            downloaded_path = Path(ydl.prepare_filename(info))

    if not downloaded_path.exists():
        raise FileNotFoundError(f"Downloaded audio file not found: {downloaded_path}")

    return AudioSource(
                video_id=info.get("id"),
                title=info.get("title"),
                source_path=downloaded_path
            )
