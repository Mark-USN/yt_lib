from __future__ import annotations

import logging
import json
from pathlib import Path
from dataclasses import dataclass, field
from copy import deepcopy
from typing import Any
from tempfile import NamedTemporaryFile

from yt_dlp import YoutubeDL

from yt_lib.utils.log_utils import get_logger


logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Small typed coercion helpers
# -----------------------------------------------------------------------------

def _as_str(value: Any, default: str | None = None) -> str | None:
    """Return a non-empty string, else default."""
    return value if isinstance(value, str) and value else default


def _as_int(value: Any, default: int | None = None) -> int | None:
    """Return an int, excluding bool, else default."""
    if isinstance(value, bool):
        return default
    return value if isinstance(value, int) else default


def _as_float(value: Any, default: float | None = None) -> float | None:
    """Return a float from int/float, else default."""
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        return float(value)
    return default


def _best_filesize_from_mapping(fmt: dict[str, Any] | None) -> int | None:
    """Prefer exact filesize, then approximate filesize."""
    if not fmt:
        return None
    return _as_int(fmt.get("filesize")) or _as_int(fmt.get("filesize_approx"))


def _resolution_from_mapping(fmt: dict[str, Any] | None) -> str | None:
    """
    Resolve a human-ish resolution string.

    Preference:
    1) yt-dlp's explicit "resolution" field, unless it says "audio only"
    2) width x height
    """
    if not fmt:
        return None

    resolution = _as_str(fmt.get("resolution"))
    if resolution and resolution != "audio only":
        return resolution

    width = _as_int(fmt.get("width"))
    height = _as_int(fmt.get("height"))
    if width and height:
        return f"{width}x{height}"

    return None


def _bytes_to_mbps(filesize_bytes: int, duration_s: int) -> float:
    """Convert total bytes over duration to Mi-bit/s style Mbps estimate."""
    return (filesize_bytes * 8.0) / duration_s / 1_048_576.0


def _kbps_to_mbps(kbps: float) -> float:
    """Convert kbps to Mbps using 1024 kbps == 1 Mbps convention."""
    return kbps / 1024.0


# -----------------------------------------------------------------------------
# yt-dlp option helpers
# -----------------------------------------------------------------------------

def build_ytdlp_options(
    *,
    format_selector: str = "bestvideo+bestaudio/best",
    quiet: bool = True,
    no_warnings: bool = True,
    skip_download: bool = True,
    no_progress: bool = True,
    include_logger: bool = True,
) -> dict[str, Any]:
    """
    Build YoutubeDL options for metadata-only extraction.

    Notes:
    - The earlier `ytdlp_opts` dataclass version was not suitable because
      `YoutubeDL(...)` expects a mapping, not an iterable dataclass.
    - `format_selector` controls yt-dlp's preferred selection logic.
    """
    options: dict[str, Any] = {
        "quiet": quiet,
        "no_warnings": no_warnings,
        "skip_download": skip_download,
        "noprogress": no_progress,
        "format": format_selector,
    }

    if include_logger:
        options["logger"] = logger

    return options


def fetch_yt_dlp_info(
    url: str,
    *,
    format_selector: str = "bestvideo+bestaudio/best",
    extra_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Fetch raw yt-dlp info for a URL.

    Returns the raw info mapping from yt-dlp.
    """
    options = build_ytdlp_options(format_selector=format_selector)
    if extra_options:
        options.update(extra_options)

    with YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=False)

    if not isinstance(info, dict):
        raise TypeError(f"yt-dlp returned {type(info).__name__}, expected dict")

    return info


# -----------------------------------------------------------------------------
# Typed format model
# -----------------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class YtdlpFormat:
    """Typed view of one yt-dlp format entry."""

    format_id: str | None = None
    format_note: str | None = None
    format_name: str | None = None

    ext: str | None = None
    container: str | None = None

    protocol: str | None = None

    vcodec: str | None = None
    acodec: str | None = None

    width: int | None = None
    height: int | None = None
    fps: float | None = None
    resolution: str | None = None

    asr: int | None = None
    audio_channels: int | None = None

    tbr_kbps: float | None = None
    filesize: int | None = None
    filesize_approx: int | None = None

    @property
    def best_filesize(self) -> int | None:
        """Prefer exact filesize, then approximate filesize."""
        return self.filesize if self.filesize is not None else self.filesize_approx

    @property
    def computed_resolution(self) -> str | None:
        """Return explicit resolution or compute width x height."""
        if self.resolution and self.resolution != "audio only":
            return self.resolution
        if self.width and self.height:
            return f"{self.width}x{self.height}"
        return None

    @property
    def is_video(self) -> bool:
        return self.vcodec not in (None, "none")

    @property
    def is_audio(self) -> bool:
        return self.acodec not in (None, "none")

    @property
    def is_video_only(self) -> bool:
        return self.is_video and not self.is_audio

    @property
    def is_audio_only(self) -> bool:
        return self.is_audio and not self.is_video

    @property
    def is_muxed(self) -> bool:
        return self.is_video and self.is_audio


def format_from_dict(data: dict[str, Any]) -> YtdlpFormat:
    """Convert one raw yt-dlp format mapping to a typed format object."""
    return YtdlpFormat(
        format_id=_as_str(data.get("format_id")),
        format_note=_as_str(data.get("format_note")),
        format_name=_as_str(data.get("format")),
        ext=_as_str(data.get("ext")),
        container=_as_str(data.get("container")),
        protocol=_as_str(data.get("protocol")),
        vcodec=_as_str(data.get("vcodec")),
        acodec=_as_str(data.get("acodec")),
        width=_as_int(data.get("width")),
        height=_as_int(data.get("height")),
        fps=_as_float(data.get("fps")),
        resolution=_as_str(data.get("resolution")),
        asr=_as_int(data.get("asr")),
        audio_channels=_as_int(data.get("audio_channels")),
        tbr_kbps=_as_float(data.get("tbr")),
        filesize=_as_int(data.get("filesize")),
        filesize_approx=_as_int(data.get("filesize_approx")),
    )


def parse_formats(formats: Any) -> list[YtdlpFormat]:
    """Parse a raw yt-dlp 'formats' style list into typed format objects."""
    if not isinstance(formats, list):
        return []
    return [format_from_dict(item) for item in formats if isinstance(item, dict)]


# -----------------------------------------------------------------------------
# Selection helpers
# -----------------------------------------------------------------------------

def pick_selected_format_dicts(info: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Return the formats yt-dlp appears to have selected.

    Strategy:
    1) Prefer 'requested_formats' if present. This is common for
       bestvideo+bestaudio style selection.
    2) Otherwise fall back to matching 'format_id' inside 'formats'.
    """
    requested = info.get("requested_formats")
    if isinstance(requested, list) and requested:
        return [item for item in requested if isinstance(item, dict)]

    format_id = _as_str(info.get("format_id"))
    formats = info.get("formats")
    if format_id and isinstance(formats, list):
        for item in formats:
            if isinstance(item, dict) and _as_str(item.get("format_id")) == format_id:
                return [item]

    return []


def pick_selected_formats(info: dict[str, Any]) -> list[YtdlpFormat]:
    """Typed wrapper around pick_selected_format_dicts()."""
    return [format_from_dict(item) for item in pick_selected_format_dicts(info)]


def split_selected_streams(
    selected: list[YtdlpFormat],
) -> tuple[YtdlpFormat | None, YtdlpFormat | None, YtdlpFormat | None]:
    """
    Split selected formats into video-only, audio-only, and muxed.

    Returns:
        (video_only, audio_only, muxed)
    """
    video_only: YtdlpFormat | None = None
    audio_only: YtdlpFormat | None = None
    muxed: YtdlpFormat | None = None

    for fmt in selected:
        if fmt.is_video_only and video_only is None:
            video_only = fmt
        elif fmt.is_audio_only and audio_only is None:
            audio_only = fmt
        elif fmt.is_muxed and muxed is None:
            muxed = fmt

    return video_only, audio_only, muxed


def pick_best_format(info: dict[str, Any]) -> YtdlpFormat | None:
    """
    Heuristic: choose the 'best' format entry from the full formats list.

    This is especially useful in info-only mode when there is no explicit
    selected download result to inspect.

    Score priority:
    1) Higher height
    2) Higher fps
    3) Larger filesize
    """
    candidates = parse_formats(info.get("formats"))
    if not candidates:
        return None

    def score(fmt: YtdlpFormat) -> tuple[int, float, int]:
        return (
            fmt.height or 0,
            fmt.fps or 0.0,
            fmt.best_filesize or 0,
        )

    return max(candidates, key=score)


# -----------------------------------------------------------------------------
# Stream estimation / selection summary
# -----------------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class StreamEstimate:
    format_id: str | None
    ext: str | None
    codec: str | None
    kind: str  # "video" | "audio" | "muxed" | "unknown"

    duration_s: int | None
    filesize_bytes: int | None
    tbr_kbps: float | None

    mbps_from_tbr: float | None
    mbps_from_filesize: float | None
    kbps_delta: float | None


@dataclass(slots=True, frozen=True)
class SelectionSummary:
    duration_s: int | None

    overall_format_id: str | None
    overall_format: str | None
    overall_ext: str | None

    selected_formats: list[YtdlpFormat]

    video: StreamEstimate | None
    audio: StreamEstimate | None
    muxed: StreamEstimate | None

    total_filesize_bytes: int | None
    total_mbps_from_filesize: float | None


def estimate_stream(
    fmt: YtdlpFormat,
    duration_s: int | None,
) -> StreamEstimate:
    """Estimate bitrate/size relationships for one selected stream."""
    filesize = fmt.best_filesize
    tbr_kbps = fmt.tbr_kbps

    if fmt.is_video_only:
        kind = "video"
        codec = fmt.vcodec
    elif fmt.is_audio_only:
        kind = "audio"
        codec = fmt.acodec
    elif fmt.is_muxed:
        kind = "muxed"
        codec = fmt.vcodec or fmt.acodec
    else:
        kind = "unknown"
        codec = None

    mbps_from_tbr = _kbps_to_mbps(tbr_kbps) if tbr_kbps is not None else None
    mbps_from_filesize = (
        _bytes_to_mbps(filesize, duration_s)
        if filesize is not None and duration_s not in (None, 0)
        else None
    )

    kbps_from_filesize = (
        (filesize * 8.0 / duration_s) / 1024.0
        if filesize is not None and duration_s not in (None, 0)
        else None
    )

    kbps_delta = (
        kbps_from_filesize - tbr_kbps
        if kbps_from_filesize is not None and tbr_kbps is not None
        else None
    )

    return StreamEstimate(
        format_id=fmt.format_id,
        ext=fmt.ext,
        codec=codec,
        kind=kind,
        duration_s=duration_s,
        filesize_bytes=filesize,
        tbr_kbps=tbr_kbps,
        mbps_from_tbr=mbps_from_tbr,
        mbps_from_filesize=mbps_from_filesize,
        kbps_delta=kbps_delta,
    )


def summarize_selection(info: dict[str, Any]) -> SelectionSummary:
    """
    Summarize the formats yt-dlp appears to have selected.

    Works for:
    - separate video/audio selection
    - muxed single-file selection
    """
    duration_s = _as_int(info.get("duration"))
    selected = pick_selected_formats(info)

    video_fmt, audio_fmt, muxed_fmt = split_selected_streams(selected)

    video = estimate_stream(video_fmt, duration_s) if video_fmt else None
    audio = estimate_stream(audio_fmt, duration_s) if audio_fmt else None
    muxed = estimate_stream(muxed_fmt, duration_s) if muxed_fmt else None

    total_filesize: int | None = None
    if muxed and muxed.filesize_bytes is not None:
        total_filesize = muxed.filesize_bytes
    else:
        parts = [
            stream.filesize_bytes
            for stream in (video, audio)
            if stream and stream.filesize_bytes is not None
        ]
        if parts:
            total_filesize = sum(parts)

    total_mbps = (
        _bytes_to_mbps(total_filesize, duration_s)
        if total_filesize is not None and duration_s not in (None, 0)
        else None
    )

    return SelectionSummary(
        duration_s=duration_s,
        overall_format_id=_as_str(info.get("format_id")),
        overall_format=_as_str(info.get("format")),
        overall_ext=_as_str(info.get("ext")),
        selected_formats=selected,
        video=video,
        audio=audio,
        muxed=muxed,
        total_filesize_bytes=total_filesize,
        total_mbps_from_filesize=total_mbps,
    )


# -----------------------------------------------------------------------------
# Full typed info view
# -----------------------------------------------------------------------------

@dataclass(slots=True)
class YtdlpInfo:
    """
    Full typed view of the yt-dlp info dict.

    This is useful when you want:
    - commonly used top-level metadata
    - parsed selected formats
    - parsed full format list
    - optional raw retention for debugging
    """

    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    extractor: str | None = None
    extractor_key: str | None = None
    webpage_url: str | None = None
    original_url: str | None = None

    id: str | None = None
    title: str | None = None
    description: str | None = None
    channel: str | None = None
    uploader: str | None = None
    upload_date: str | None = None

    duration: int | None = None

    format_id: str | None = None
    format_name: str | None = None
    ext: str | None = None

    requested_formats: list[YtdlpFormat] = field(default_factory=list)
    formats: list[YtdlpFormat] = field(default_factory=list)

    @classmethod
    def from_dict(
        cls,
        info: dict[str, Any],
        *,
        include_raw: bool = True,
        copy_raw: bool = False,
        include_formats: bool = True,
    ) -> YtdlpInfo:
        """
        Build a typed info object from raw yt-dlp info.

        Args:
            include_raw:
                Whether to keep the full raw info mapping.
            copy_raw:
                Whether to deep-copy the raw info before storing it.
                Useful if callers may mutate the original dict later.
            include_formats:
                Whether to parse and retain the full format lists.
                Turn off if you want a lighter object.
        """
        if not isinstance(info, dict):
            raise TypeError(f"info must be dict[str, Any], got {type(info).__name__}")

        raw: dict[str, Any]
        if include_raw:
            raw = deepcopy(info) if copy_raw else info
        else:
            raw = {}

        requested_formats = (
            parse_formats(info.get("requested_formats"))
            if include_formats
            else []
        )
        formats = parse_formats(info.get("formats")) if include_formats else []

        return cls(
            raw=raw,
            extractor=_as_str(info.get("extractor")),
            extractor_key=_as_str(info.get("extractor_key")),
            webpage_url=_as_str(info.get("webpage_url")),
            original_url=_as_str(info.get("original_url")),
            id=_as_str(info.get("id")),
            title=_as_str(info.get("title")),
            description=_as_str(info.get("description")),
            channel=_as_str(info.get("channel")),
            uploader=_as_str(info.get("uploader")),
            upload_date=_as_str(info.get("upload_date")),
            duration=_as_int(info.get("duration")),
            format_id=_as_str(info.get("format_id")),
            format_name=_as_str(info.get("format")),
            ext=_as_str(info.get("ext")),
            requested_formats=requested_formats,
            formats=formats,
        )

    @property
    def best_format(self) -> YtdlpFormat | None:
        """Heuristic best format from the full 'formats' list."""
        if self.formats:
            return max(
                self.formats,
                key=lambda fmt: (
                    fmt.height or 0,
                    fmt.fps or 0.0,
                    fmt.best_filesize or 0,
                ),
            )
        return None

    @property
    def selection_summary(self) -> SelectionSummary:
        """Convenience wrapper around summarize_selection(raw)."""
        return summarize_selection(self.raw)


# -----------------------------------------------------------------------------
# High-level convenience APIs
# -----------------------------------------------------------------------------

def fetch_YtdlpInfo_object(
    url: str,
    *,
    format_selector: str = "bestvideo+bestaudio/best",
    include_raw: bool = True,
    copy_raw: bool = False,
    include_formats: bool = True,
    extra_options: dict[str, Any] | None = None,
) -> YtdlpInfo:
    """Fetch yt-dlp info and return a typed YtdlpInfo object."""
    info = fetch_yt_dlp_info(
        url,
        format_selector=format_selector,
        extra_options=extra_options,
    )
    return YtdlpInfo.from_dict(
        info,
        include_raw=include_raw,
        copy_raw=copy_raw,
        include_formats=include_formats,
    )


def _atomic_write_json(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write text atomically by replace()'ing a temp file in the same directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        delete=False,
        dir=path.parent,
        encoding=encoding,
        newline="\n",
    ) as tf:
        tmp = Path(tf.name)
        # tf.write(text)
        json.dump(text, tf, indent=2, ensure_ascii=False)
        tf.flush()
    tmp.replace(path)

def write_info(path: Path, info: dict) -> None:
    _atomic_write_json(path, info)

def read_info(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def write_YtdlpInfo(path: Path, info: YtdlpInfo) -> None:
    if not isinstance(info, YtdlpInfo):
        raise TypeError(f"info must be YtdlpInfo, got {type(info).__name__}")
    if info.raw is None:
        raise ValueError("YtdlpInfo.raw must be present to write to file")
    _atomic_write_json(path, info)

def read_YtdlpInfo(path: Path) -> YtdlpInfo:
    try:
        with path.open("r", encoding="utf-8") as f:
            info = json.load(f)
        return YtdlpInfo.from_dict(info, include_raw=True, copy_raw=False, include_formats=True)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning("Error reading YtdlpInfo from %s: %s", latest_file, e)
        return None
