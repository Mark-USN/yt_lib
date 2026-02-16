"""
ffmpeg_bootstrap.py

Install or locate ffmpeg so that both yt_dlp and whisper can use it.

Behavior:
  * If a local ffmpeg exists under PROJECT_ROOT/.bin, use that.
  * Else, if 'ffmpeg' is already on PATH, just use that.
  * On Windows, if not found, download a static build from BtbN,
    extract ffmpeg.exe (and related tools) into a local directory,
    extract the license file as 'ffmpeg-license.txt',
    and prepend that directory to PATH.
  * On non-Windows, print instructions to install ffmpeg manually.

Intended usage:
    from ffmpeg_bootstrap import ensure_ffmpeg_on_path

    ffmpeg_dir = ensure_ffmpeg_on_path()
    # Now yt_dlp and whisper can both call `ffmpeg` via subprocess.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import tempfile
import zipfile
# import logging
from pathlib import Path
from typing import Optional
from urllib.request import urlopen
from yt_lib.utils.log_utils import get_logger # , log_tree


# -----------------------------
# Logging setup
# -----------------------------
logger = get_logger(__name__)



# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Where to place a locally downloaded static build (Windows)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOCAL_FFMPEG_DIR = PROJECT_ROOT / ".bin"
LOCAL_LICENSE_DIR = PROJECT_ROOT / "License"
# Name of the license file we’ll write next to ffmpeg.exe if we can find one
LOCAL_LICENSE_NAME = "ffmpeg-license.txt"

# BtbN latest static Win64 GPL build
BTBN_FFMPEG_ZIP_URL = (
    "https://github.com/BtbN/FFmpeg-Builds/"
    "releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_windows() -> bool:
    return platform.system().lower().startswith("win")


def find_ffmpeg_in_path() -> Optional[str]:
    """
    Return directory containing an existing ffmpeg binary found via PATH,
    or None if not found.
    """
    exe_name = "ffmpeg.exe" if _is_windows() else "ffmpeg"
    ffmpeg_path = shutil.which(exe_name)
    if ffmpeg_path:
        return str(Path(ffmpeg_path).resolve().parent)
    return None


def _download_file(url: str, dest: Path) -> None:
    """Download a file from URL to dest."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as resp, open(dest, "wb") as f:
        shutil.copyfileobj(resp, f)


def _extract_zip(src_zip: Path, dest_dir: Path) -> None:
    """Extract all contents of src_zip into dest_dir."""
    with zipfile.ZipFile(src_zip, "r") as zf:
        zf.extractall(dest_dir)


def _make_executable(path: Path) -> None:
    """Ensure the file is executable on POSIX."""
    mode = path.stat().st_mode
    # add user execute bit
    path.chmod(mode | 0o100)


def _install_ffmpeg_windows() -> str:
    """
    Download and install a static ffmpeg build into LOCAL_FFMPEG_DIR.

    Returns:
        str: The directory path where ffmpeg.exe was installed.

    Raises:
        RuntimeError if installation fails.
    """
    bin_dir = LOCAL_FFMPEG_DIR
    bin_dir.mkdir(parents=True, exist_ok=True)
    lic_dir = LOCAL_LICENSE_DIR
    lic_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ffmpeg-bootstrap] Installing static ffmpeg into {bin_dir} ...")

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        zip_path = tmpdir / "ffmpeg.zip"

        print(f"[ffmpeg-bootstrap] Downloading: {BTBN_FFMPEG_ZIP_URL}")
        _download_file(BTBN_FFMPEG_ZIP_URL, zip_path)

        print("[ffmpeg-bootstrap] Extracting archive...")
        extract_dir = tmpdir / "extract"
        extract_dir.mkdir(parents=True, exist_ok=True)
        _extract_zip(zip_path, extract_dir)

        # Search recursively for ffmpeg.exe and a license text file
        ffmpeg_exe: Optional[Path] = None
        license_source: Optional[Path] = None

        for p in extract_dir.rglob("*"):
            if p.is_file():
                name_lower = p.name.lower()
                if name_lower == "ffmpeg.exe":
                    ffmpeg_exe = p

                if (
                    "license" in name_lower and name_lower.endswith(".txt")
                ) or name_lower in {"copying", "license"}:
                    if license_source is None:
                        license_source = p

        if ffmpeg_exe is None:
            raise RuntimeError(
                "[ffmpeg-bootstrap] Could not find ffmpeg.exe in the downloaded zip."
            )

        # Copy ffmpeg.exe (and optionally ffprobe.exe / ffplay.exe) to bin_dir
        def _copy_binary(name: str) -> None:
            src = next(
                (f for f in extract_dir.rglob(name) if f.is_file()),
                None,
            )
            if src:
                dest = bin_dir / name
                shutil.copy2(src, dest)
                _make_executable(dest)
                print(f"[ffmpeg-bootstrap] Installed {name} -> {dest}")

        _copy_binary("ffmpeg.exe")
        _copy_binary("ffprobe.exe")
        _copy_binary("ffplay.exe")

        # Copy license file as ffmpeg-license.txt if we found one
        if license_source:
            license_dest = lic_dir / LOCAL_LICENSE_NAME
            shutil.copy2(license_source, license_dest)
            print(f"[ffmpeg-bootstrap] Saved license as {license_dest}")
        else:
            license_dest = lic_dir / LOCAL_LICENSE_NAME
            license_dest.write_text(
                "FFmpeg is licensed mainly under LGPL/GPL.\n"
                "See https://ffmpeg.org/legal.html and https://github.com/FFmpeg/FFmpeg "
                "for full license text and details.\n",
                encoding="utf-8",
            )
            print(
                f"[ffmpeg-bootstrap] No LICENSE.txt found in archive; "
                f"wrote notice to {license_dest}"
            )

    ffmpeg_install_dir = str(bin_dir)
    print(f"[ffmpeg-bootstrap] Static ffmpeg ready at {ffmpeg_install_dir}")
    return ffmpeg_install_dir


def _print_unix_instructions() -> None:
    """
    For non-Windows platforms where ffmpeg is not found, print install hints.
    """
    print(
        "[ffmpeg-bootstrap] ffmpeg not found on PATH.\n"
        "Please install ffmpeg using your package manager, e.g.:\n\n"
        "  # Debian/Ubuntu\n"
        "  sudo apt update && sudo apt install ffmpeg\n\n"
        "  # macOS (Homebrew)\n"
        '  brew install ffmpeg\n\n'
        "After installation, make sure `ffmpeg` is on your PATH."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_ffmpeg_binary_path() -> Optional[str]:
    """
    Ensure an ffmpeg binary exists, and return the directory containing it.

    Order of preference:
      1) PROJECT_ROOT/.bin/ffmpeg(.exe) if present
      2) Any ffmpeg found on PATH
      3) On Windows, install static build into PROJECT_ROOT/.bin
      4) On non-Windows, print instructions and return None.
    """
    exe_name = "ffmpeg.exe" if _is_windows() else "ffmpeg"

    # 1) Prefer local project .bin if present
    local_ffmpeg = LOCAL_FFMPEG_DIR / exe_name
    if local_ffmpeg.exists():
        return str(LOCAL_FFMPEG_DIR)

    # 2) Try existing ffmpeg on PATH
    path_dir = find_ffmpeg_in_path()
    if path_dir:
        return path_dir

    # 3) On Windows, attempt static installation into .bin
    if _is_windows():
        return _install_ffmpeg_windows()

    # 4) On non-Windows, give install instructions
    _print_unix_instructions()
    return None


def ensure_ffmpeg_on_path() -> str:
    """
    Ensure ffmpeg exists *and* is on PATH so that subprocess-based tools
    like whisper and yt_dlp can find it.

    Returns:
        str: directory containing the ffmpeg binary.

    Raises:
        SystemExit if ffmpeg cannot be found or installed.
    """
    ffmpeg_dir = get_ffmpeg_binary_path()
    if not ffmpeg_dir:
        raise SystemExit(
            "[ffmpeg-bootstrap] ffmpeg is required but could not be found or installed.\n"
            "Please install ffmpeg and ensure it is on PATH."
        )

    current_path = os.environ.get("PATH", "")
    parts = current_path.split(os.pathsep) if current_path else []

    if ffmpeg_dir not in parts:
        # Prepend our ffmpeg directory to PATH
        new_path = ffmpeg_dir if not current_path else ffmpeg_dir + os.pathsep + current_path
        os.environ["PATH"] = new_path

    return ffmpeg_dir


def show_ffmpeg_license() -> None:
    """
    Print the contents or path of the saved ffmpeg license file, if present.
    """
    licence_dir = PROJECT_ROOT / "License"
    license_path = licence_dir / LOCAL_LICENSE_NAME
    if license_path.exists():
        print(f"[ffmpeg-bootstrap] License file: {license_path}")
        print(license_path.read_text(encoding="utf-8"))
    else:
        print("[ffmpeg-bootstrap] No ffmpeg-license.txt found.")


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    exe_dir = ensure_ffmpeg_on_path()
    print(f"[ffmpeg-bootstrap] ffmpeg directory: {exe_dir}")
    result = subprocess.run(["ffmpeg", "-version"], check=True)
    if result.returncode != 0:
        print("[ffmpeg-bootstrap] Warning: 'ffmpeg -version' failed!")
    show_ffmpeg_license()
