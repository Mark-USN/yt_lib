""" Utilities for parsing YouTube video and playlist identifiers from URLs and text."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Final, Any, TypedDict, Protocol
from urllib.parse import parse_qs, urlparse


class Snippet(TypedDict):
    """ Raw snippet shape returned by `FetchedTranscript.to_raw_data()`."""
    text: str
    start: float
    duration: float


class ProgressReporter(Protocol):
    """ Protocol for reporting progress of an asynchronous operation."""

    @property
    def current(self) -> int | None:
        """ Returns the current progress value, or None if not available."""

    @property
    def total(self) -> int:
        """ Returns the total progress value."""

    async def set_total(self, total: int) -> None:
        """ Sets the total progress value."""

    async def set_message(self, message: str | None) -> None:
        """ Sets the progress status message."""

    async def increment(self, amount: int = 1) -> None:
        """ Atomically increment the current progress value."""


@dataclass(slots=True)
class IntegerProgressAllocator:
    """ Utility class to allocate integer progress increments based on a total range and number
        of steps.
    """
    start: int
    total_range: int
    steps: int
    last_reported: int

    @classmethod
    def from_progress(cls, *, current: int, total: int, steps: int) -> "IntegerProgressAllocator":
        """ Factory method to create an IntegerProgressAllocator from current and total
            progress values
            Args:
                current: The current progress value.
                total: The total progress value.
                steps: The number of steps to divide the remaining progress into.
            Returns:
                An instance of IntegerProgressAllocator.
            Raises:
                ValueError: If steps is not greater than zero.
        """
        if steps <= 0:
            raise ValueError("steps must be greater than zero")

        return cls(
            start=current,
            total_range=total - current,
            steps=steps,
            last_reported=current,
        )

    def delta_for_completed_steps(self, completed_steps: int) -> int:
        """ Calculate the progress delta for a given number of completed steps.
            Args:
                completed_steps: The number of steps that have been completed.
            Returns:
                The progress delta for the completed steps.
        """
        completed_steps = min(max(completed_steps, 0), self.steps)

        target = self.start + round(
            self.total_range * completed_steps / self.steps
        )

        delta = target - self.last_reported
        self.last_reported = target

        return max(delta, 0)

    def final_delta(self) -> int:
        """ Calculate the final progress delta to reach the total, regardless of steps completed.
            Returns:
                The final progress delta to reach the total.
        """
        target = self.start + self.total_range
        delta = target - self.last_reported
        self.last_reported = target
        return max(delta, 0)




_VIDEO_ID_RE: Final = re.compile(r"^[A-Za-z0-9_-]{11}$")
_PLAYLIST_ID_RE: Final = re.compile(r"^(PL|UU|LL|FL|OL|RD|WL)[A-Za-z0-9_-]{10,200}$")
_CHANNEL_ID_RE: Final = re.compile(r"^UC[A-Za-z0-9_-]{22}$")


class YoutubeIdKind(Enum):
    """ The type of YouTube identifier."""
    VIDEO = auto()
    PLAYLIST = auto()
    CHANNEL = auto()
    UNKNOWN = auto()


@dataclass(slots=True, frozen=True)
class YoutubeIdentifier:
    """ A YouTube identifier with its classified type."""
    kind: YoutubeIdKind
    value: str


def is_video_id(value: str) -> bool:
    """ Check if the value is a valid YouTube video ID.
        Args:
            value: The string to check.
        Returns:
            True if the value is a valid YouTube video ID, False otherwise.
    """
    return _VIDEO_ID_RE.fullmatch(value) is not None


def is_playlist_id(value: str) -> bool:
    """ Check if the value is a valid YouTube playlist ID.
        Args:
            value: The string to check.
        Returns:
            True if the value is a valid YouTube playlist ID, False otherwise.
    """
    return _PLAYLIST_ID_RE.fullmatch(value) is not None


def classify_youtube_id(value: str) -> YoutubeIdKind:
    """ Classify the type of YouTube identifier based on its format.
        Args:
            value: The string to classify.
        Returns:
            The kind of YouTube identifier (VIDEO, PLAYLIST, CHANNEL, or UNKNOWN).
    """
    if is_video_id(value):
        return YoutubeIdKind.VIDEO
    if is_playlist_id(value):
        return YoutubeIdKind.PLAYLIST
    if _CHANNEL_ID_RE.fullmatch(value):
        return YoutubeIdKind.CHANNEL
    return YoutubeIdKind.UNKNOWN


def extract_video_id(text: str) -> str | None:
    """ Extract a video id from a URL or return the input if it is already a video id.
        Args:
            text: The input string, which can be a YouTube video URL or a video ID
        Returns:
            The extracted video ID if found, otherwise None.
    """
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
    """ Extract a playlist id from a URL or return the input if it is already a playlist id.
        Args:
            text: The input string, which can be a YouTube playlist URL or a playlist ID
        Returns:
            The extracted playlist ID if found, otherwise None.
    """
    if is_playlist_id(text):
        return text

    parsed = urlparse(text)
    qs = parse_qs(parsed.query)
    pid = (qs.get("list") or [None])[0]
    return pid if isinstance(pid, str) and is_playlist_id(pid) else None


def extract_any_identifier(text: str) -> YoutubeIdentifier | None:
    """ Return the first recognized YouTube identifier (video or playlist) from text.
        Args:
            text: The input string, which can be a YouTube video or playlist URL or an ID
        Returns:
            A YoutubeIdentifier object if a valid identifier is found, otherwise None.
    """
    if vid := extract_video_id(text):
        return YoutubeIdentifier(YoutubeIdKind.VIDEO, vid)
    if pid := extract_playlist_id(text):
        return YoutubeIdentifier(YoutubeIdKind.PLAYLIST, pid)
    kind = classify_youtube_id(text)
    if kind is YoutubeIdKind.UNKNOWN:
        return None
    return YoutubeIdentifier(kind, text)

@dataclass(slots=True)
class VideoMetadata:
    """ Metadata extracted from yt-dlp info dict for a YouTube video."""
    url: str
    video_id: str
    title: str
    description: str
    channel: str | None
    upload_date: str | None
    duration: float | None
    view_count: int | None
    like_count: int | None
    webpage_url: str | None = None
    ext:str | None = None
    video_format:str | None = None
    filesize:int | None = None
    fps: float | None = None
    resolution: str | None = None


    @classmethod
    def extract_video_metadata(cls, info: dict[str, object]) -> dict[str, Any]:
        """ Extract relevant video metadata from a yt-dlp info dict, handling both
            'requested_formats' and top-level format fields.
            Args:
                info: The yt-dlp info dictionary for a video.
            Returns:
                A dictionary containing extracted metadata fields.
        """
        fmt = info["requested_formats"][0] if "requested_formats" in info else info
        return {
            "ext": fmt.get("ext"),
            "video_format": fmt.get("format"),
            "filesize": fmt.get("filesize") or fmt.get("filesize_approx"),
            "fps": fmt.get("fps"),
            "resolution": fmt.get("resolution")
                or f"{fmt.get('width')}x{fmt.get('height')}",
            "duration": fmt.get("duration"),
        }

    @classmethod
    def from_yt_dlp(cls, *, url: str, info: dict[str, object]) -> VideoMetadata:
        """ Create a VideoMetadata instance from a yt-dlp info dict for a video.
            Args:
                url: The original URL of the video.
                info: The yt-dlp info dictionary for the video.
            Returns:
                A VideoMetadata instance populated with the extracted metadata.
        """
        format_data = VideoMetadata.extract_video_metadata(info)
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
            ext=format_data["ext"],
            video_format=format_data["video_format"],
            filesize=format_data["filesize"],
            fps=format_data["fps"],
            resolution=format_data["resolution"],
        )
