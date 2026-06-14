""" Audio conversion to 16kHz mono WAV using ffmpeg."""

from __future__ import annotations

import asyncio
from pathlib import Path

from yt_lib.audio.audio_types import ConvertedAudio


async def convert_to_16k_mono_wav(
    source_path: Path,
    output_dir: Path,
    *,
    output_stem: str | None = None,
    ffmpeg_path: str = "ffmpeg",
) -> ConvertedAudio:
    """ Convert the given audio file to 16kHz mono WAV format using ffmpeg and return the path
        to the converted file.
        Args:
            source_path: Path to the source audio file.
            output_dir: Directory where the converted file will be saved.
            output_stem: Optional stem for the output file name.
            ffmpeg_path: Path to the ffmpeg executable.
        Returns:
            ConvertedAudio: An object containing the path to the converted WAV file.
        Raises:
            FileNotFoundError: If the converted WAV file is not found.
            RuntimeError: If the ffmpeg conversion fails.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = output_stem or source_path.stem
    wav_path = output_dir / f"{stem}.16k.mono.wav"

    cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        str(source_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(wav_path),
    ]

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        raise RuntimeError(
            "ffmpeg conversion failed\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout: {stdout.decode(errors='replace')}\n"
            f"stderr: {stderr.decode(errors='replace')}"
        )

    if not wav_path.exists():
        raise FileNotFoundError(f"Converted WAV file not found: {wav_path}")

    return ConvertedAudio(wav_path=wav_path)
