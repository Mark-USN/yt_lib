from __future__ import annotations

import re
# from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum, auto
from typing import Final
from urllib.parse import parse_qs, urlparse

_VIDEO_ID_RE: Final = re.compile(r"^[A-Za-z0-9_-]{11}$")
_PLAYLIST_ID_RE: Final = re.compile(r"^(PL|UU|LL|FL|OL|RD|WL)[A-Za-z0-9_-]{10,200}$")
_CHANNEL_ID_RE: Final = re.compile(r"^UC[A-Za-z0-9_-]{22}$")


class YoutubeIdKind(Enum):
    VIDEO = auto()
    PLAYLIST = auto()
    CHANNEL = auto()
    UNKNOWN = auto()


@dataclass(slots=True, frozen=True)
class YoutubeIdentifier:
    kind: YoutubeIdKind
    value: str


def is_video_id(value: str) -> bool:
    return _VIDEO_ID_RE.fullmatch(value) is not None


def is_playlist_id(value: str) -> bool:
    return _PLAYLIST_ID_RE.fullmatch(value) is not None


def classify_youtube_id(value: str) -> YoutubeIdKind:
    if is_video_id(value):
        return YoutubeIdKind.VIDEO
    if is_playlist_id(value):
        return YoutubeIdKind.PLAYLIST
    if _CHANNEL_ID_RE.fullmatch(value):
        return YoutubeIdKind.CHANNEL
    return YoutubeIdKind.UNKNOWN


def extract_video_id(text: str) -> str | None:
    """Extract a video id from a URL or return the input if it is already a video id."""
    if is_video_id(text):
        return text

    parsed = urlparse(text)
    host = (parsed.netloc or "").lower()
    path = parsed.path or ""

    # youtu.be/<id>
    if host.endswith("youtu.be"):
        vid = path.lstrip("/").split("/", 1)[0]
        return vid if is_video_id(vid) else None

    # youtube.com/watch?v=<id>
    if path == "/watch":
        vid = (parse_qs(parsed.query).get("v") or [None])[0]
        return vid if isinstance(vid, str) and is_video_id(vid) else None

    # youtube.com/shorts/<id>
    if path.startswith("/shorts/"):
        vid = path.removeprefix("/shorts/")
        if "/" in vid:
            return None
        return vid if is_video_id(vid) else None

    # youtube.com/embed/<id>
    if path.startswith("/embed/"):
        vid = path.removeprefix("/embed/").split("/", 1)[0]
        return vid if is_video_id(vid) else None

    return None


def extract_playlist_id(text: str) -> str | None:
    """Extract a playlist id from a URL or return the input if it is already a playlist id."""
    if is_playlist_id(text):
        return text

    parsed = urlparse(text)
    qs = parse_qs(parsed.query)
    pid = (qs.get("list") or [None])[0]
    return pid if isinstance(pid, str) and is_playlist_id(pid) else None


def extract_any_identifier(text: str) -> YoutubeIdentifier | None:
    """Return the first recognized YouTube identifier (video or playlist) from text."""
    if vid := extract_video_id(text):
        return YoutubeIdentifier(YoutubeIdKind.VIDEO, vid)
    if pid := extract_playlist_id(text):
        return YoutubeIdentifier(YoutubeIdKind.PLAYLIST, pid)
    kind = classify_youtube_id(text)
    if kind is YoutubeIdKind.UNKNOWN:
        return None
    return YoutubeIdentifier(kind, text)

@dataclass(slots=True)
class YtdlpMetadata:
    url: str
    video_id: str
    title: str
    description: str
    channel: str | None
    upload_date: str | None
    duration: int | None
    view_count: int | None
    like_count: int | None
    webpage_url: str | None = None

    @classmethod
    def from_yt_dlp(cls, *, url: str, info: dict[str, object]) -> YtdlpMetadata:
        return cls(
            url=url,
            video_id=info["id"],
            title=info["title"],
            description=info.get("description", ""),
            channel=info.get("channel"),
            upload_date=info.get("upload_date"),
            duration=info.get("duration"),
            webpage_url=info.get("webpage_url"),
            view_count=info.get("view_count"),
            like_count=info.get("like_count"),
        )
