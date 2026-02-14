"""
YouTube search + metadata tools (playlist-aware; playlist expansion is opt-in).

CHANGES (per your 3 requests)
1) Playlist expansion is now a separate tool:
   - youtube_playlist_info(): playlist metadata ONLY (cheap-ish)
   - youtube_playlist_video_list(): playlistItems + videos.list enrichment (potentially expensive)
   The restored helper you added has been cleaned up and kept as:
     youtube_get_playlist_videos()

2) Python 3.12+ recommendations:
   - Built-in generics: list[str], dict[str, Any]
   - `|` unions
   - collections.abc Mapping / Iterable
   - Minimize `typing` imports (only Any + Annotated)

3) youtube_playlist_info() signature:
   - playlist: str | list[str]
   (Removed Mapping[str, Any] options as requested.)

Source baseline: /mnt/data/youtube_search.py :contentReference[oaicite:0]{index=0}
"""

from __future__ import annotations

# import json
import logging
import re
import threading
import os
from collections.abc import Iterable
from enum import Enum
from typing import Any, Annotated
from pydantic import Field
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from lib.utils.youtube_ids import (
        extract_video_id,
        is_playlist_id,
        extract_playlist_id
    )
from lib.utils.api_keys import api_vault
from lib.utils.log_utils import get_logger, log_tree
logger = get_logger(__name__)

# A small, process-wide throttle to avoid overwhelming upstream services when a
# workflow engine fans out work in parallel.
_MAX_CONCURRENT_FETCHES = int(os.environ.get("MCP_SEARCH_MAX_CONCURRENCY", "6"))
_SEARCH_SEM = threading.BoundedSemaphore(value=max(1, _MAX_CONCURRENT_FETCHES))


def yt_execute(req, *, label: str = "") -> dict[str, Any]:
    """Execute a googleapiclient request with debug logging."""
    if logger.isEnabledFor(logging.DEBUG):
        log_tree(
            logger,
            logging.INFO,
            "Server Request",
            req,
            collapse_keys={"env"},  # env can be huge/noisy
            redact_keys={"token", "api_key"},
        )

        logger.debug("[YT %s] %s %s", label, getattr(req, "method", ""), getattr(req, "uri", ""))
    try:
        with _SEARCH_SEM:
            resp = req.execute()
            if logger.isEnabledFor(logging.DEBUG):
                log_tree(
                    logger,
                    logging.INFO,
                    "Server Response",
                    resp,
                    collapse_keys={"env"},  # env can be huge/noisy
                    redact_keys={"token", "api_key"},
                )
            return resp
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning("⚠️ %s failed with %s", getattr(req, "method", ""), exc)
        return {}




# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class YtOrder(str, Enum):
    date = "date"
    rating = "rating"
    relevance = "relevance"
    title = "title"
    videocount = "videoCount"
    viewcount = "viewCount"

    @property
    def help(self) -> str:
        return {
            YtOrder.date: "Reverse chronological by publish date.",
            YtOrder.rating: "Highest to lowest rating.",
            YtOrder.relevance: "Most relevant to the query (default).",
            YtOrder.title: "Alphabetical by title.",
            YtOrder.videocount: "Channels by uploaded video count; live by concurrent viewers.",
            YtOrder.viewcount: "Highest to lowest view count.",
        }[self]

    @classmethod
    def coerce(cls, value: "YtOrder | str") -> "YtOrder":
        if isinstance(value, cls):
            return value
        v = str(value).strip()
        try:
            return cls(v)
        except Exception as exc:
            raise ValueError(f"Invalid YtOrder {value!r}. Valid: {[e.value for e in cls]}") from exc

    @classmethod
    def help_text(cls) -> str:
        return " | ".join(f"{e.value}: {e.help}" for e in cls)


class SearchKind(str, Enum):
    video = "video"
    playlist = "playlist"
    both = "video,playlist"

    @classmethod
    def coerce(cls, value: "SearchKind | str") -> "SearchKind":
        if isinstance(value, cls):
            return value
        raw = str(value).strip().lower()
        aliases = {
            "both": cls.both,
            "all": cls.both,
            "any": cls.both,
            "video,playlist": cls.both,
            "videos": cls.video,
            "playlists": cls.playlist,
        }
        if raw in aliases:
            return aliases[raw]
        try:
            return cls(raw)
        except Exception as exc:
            raise ValueError(f"Invalid SearchKind {value!r}. Valid: {[e.value for e in cls]}") from exc


# ---------------------------------------------------------------------------
# ID extraction + duration parsing
# ---------------------------------------------------------------------------

_ISO8601_DUR_RE = re.compile(
    r"^P"
    r"(?:(?P<days>\d+)D)?"
    r"(?:T"
    r"(?:(?P<hours>\d+)H)?"
    r"(?:(?P<minutes>\d+)M)?"
    r"(?:(?P<seconds>\d+)S)?"
    r")?$"
)


def parse_iso8601_duration_to_seconds(dur: str) -> int:
    m = _ISO8601_DUR_RE.match(dur or "")
    if not m:
        return 0
    days = int(m.group("days") or 0)
    hours = int(m.group("hours") or 0)
    minutes = int(m.group("minutes") or 0)
    seconds = int(m.group("seconds") or 0)
    return (((days * 24 + hours) * 60 + minutes) * 60) + seconds


def _as_int(v: Any) -> int:
    try:
        return int(v)
    except Exception:    #pylint: disable=broad-exception-caught   
        return 0


def _chunked(seq: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def dedupe_preserve_order(items: Iterable[Any]) -> list[Any]:
    seen: set[Any] = set()
    out: list[Any] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def merge_outer(target: dict[str, dict[str, Any]], source: dict[str, dict[str, Any]]) -> None:
    """Merge nested dicts: target[outer].update(source[outer])."""
    for outer_key, inner_updates in source.items():
        if outer_key not in target:
            target[outer_key] = dict(inner_updates)
        else:
            target[outer_key].update(inner_updates)


def _coerce_to_list_str(x: str | list[str]) -> list[str]:
    return x if isinstance(x, list) else [x]



def normalize_video_inputs(inputs: Iterable[str]) -> tuple[list[str], list[dict[str, Any]]]:
    video_ids: list[str] = []
    errors: list[dict[str, Any]] = []

    for raw in inputs:
        vid = extract_video_id(raw)
        if not vid:
            errors.append({"input": raw, "error": "Could not extract video_id"})
        else:
            video_ids.append(vid)

    return list(dedupe_preserve_order(video_ids)), errors


def normalize_playlist_inputs(inputs: Iterable[str]) -> tuple[list[str], list[dict[str, Any]]]:
    playlist_ids: list[str] = []
    errors: list[dict[str, Any]] = []

    for raw in inputs:
        pl = extract_playlist_id(raw)
        if not pl:
            errors.append({"input": raw, "error": "Could not extract playlist_id"})
        else:
            playlist_ids.append(pl)

    return list(dedupe_preserve_order(playlist_ids)), errors



# ---------------------------------------------------------------------------
# YouTube API client helpers
# ---------------------------------------------------------------------------

def _get_youtube_client():
    vault = api_vault()
    google_key = vault.get_value(key="GOOGLE_KEY")
    if not google_key:
        raise RuntimeError("Missing GOOGLE_KEY from api_vault()")
    return build("youtube", "v3", developerKey=google_key)


def _get_video_details(youtube, video_ids: str | list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    out: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    if not video_ids:
        return out, errors

    try:
        if isinstance(video_ids, str):
            vid = extract_video_id(video_ids) or video_ids
            req = youtube.videos().list(part="snippet,contentDetails,statistics", id=vid, maxResults=1)
            resp = yt_execute(req, label="videos.list single")
            out.extend(resp.get("items", []) or [])
            return out, errors

        for chunk in _chunked(video_ids, 50):
            req = youtube.videos().list(
                part="snippet,contentDetails,statistics",
                id=",".join(chunk),
                maxResults=len(chunk),
            )
            resp = yt_execute(req, label="videos.list batch")
            out.extend(resp.get("items", []) or [])

    except HttpError as e:
        errors.append({"error": "YouTube API error", "details": str(e)})

    return out, errors


def _get_playlist_details(youtube, playlist_ids: str | list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    out: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    if not playlist_ids:
        return out, errors

    try:
        if isinstance(playlist_ids, str):
            pid = extract_playlist_id(playlist_ids) or playlist_ids
            req = youtube.playlists().list(part="snippet,contentDetails,status", id=pid, maxResults=1)
            resp = yt_execute(req, label="playlists.list single")
            out.extend(resp.get("items", []) or [])
            return out, errors

        for chunk in _chunked(playlist_ids, 50):
            req = youtube.playlists().list(part="snippet,contentDetails,status", id=",".join(chunk), maxResults=len(chunk))
            resp = yt_execute(req, label="playlists.list batch")
            out.extend(resp.get("items", []) or [])

    except HttpError as e:
        errors.append({"error": "YouTube API error", "details": str(e)})

    return out, errors


# CHANGE #1 (cleaned helper you restored):
def youtube_get_playlist_videos(
    youtube,
    playlist_id: str,
    *,
    max_videos: int = 50,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fetch playlistItems for a single playlistId (paged).

    Returns (items, errors).
    """
    # Assume playlist_id is already validated/extracted.
    errors: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    page_token: str | None = None

    if not playlist_id:
        return items, [{"error": "missing_playlist_id"}]

    try:
        while len(items) < max_videos:
            req = youtube.playlistItems().list(
                part="id,snippet,contentDetails,status",
                playlistId=playlist_id,
                maxResults=min(50, max_videos - len(items)),
                pageToken=page_token,
            )
            resp = yt_execute(req, label="playlistItems.list")
            items.extend(resp.get("items", []) or [])

            page_token = resp.get("nextPageToken")
            if not page_token:
                break

    except HttpError as e:
        errors.append({"error": "YouTube API error", "details": str(e)})

    return items, errors


# ---------------------------------------------------------------------------
# Shapers
# ---------------------------------------------------------------------------

def _shape_video_info(video_id: str, video_item: dict[str, Any] | None) -> dict[str, Any]:
    if not video_item:
        return {
            "kind": "video",
            "video_id": video_id,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "title": "",
            "description": "",
            "publishedAt": "",
            "duration": {"iso8601": "PT0S", "seconds": 0},
            "statistics": {"views": 0, "likes": 0, "comments": 0},
            "available": False,
        }

    snippet = video_item.get("snippet") or {}
    content = video_item.get("contentDetails") or {}
    stats = video_item.get("statistics") or {}

    dur_iso = content.get("duration") or "PT0S"
    dur_s = parse_iso8601_duration_to_seconds(dur_iso)

    return {
        "kind": "video",
        "video_id": video_id,
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "title": snippet.get("title", ""),
        "description": snippet.get("description", ""),
        "publishedAt": snippet.get("publishedAt", ""),
        "duration": {"iso8601": dur_iso, "seconds": dur_s},
        "statistics": {
            "views": _as_int(stats.get("viewCount")),
            "likes": _as_int(stats.get("likeCount")),
            "comments": _as_int(stats.get("commentCount")),
        },
        "available": True,
    }


def _shape_playlist_info(playlist_id: str, playlist_item: dict[str, Any] | None) -> dict[str, Any]:
    if not playlist_item:
        return {
            "kind": "playlist",
            "playlist_id": playlist_id,
            "url": f"https://www.youtube.com/playlist?list={playlist_id}",
            "title": "",
            "description": "",
            "publishedAt": "",
            "channelTitle": "",
            "privacyStatus": "",
            "itemCount": 0,
            "available": False,
        }

    snippet = playlist_item.get("snippet") or {}
    content = playlist_item.get("contentDetails") or {}
    status = playlist_item.get("status") or {}

    pid = playlist_item.get("id") or playlist_id

    return {
        "kind": "playlist",
        "playlist_id": pid,
        "url": f"https://www.youtube.com/playlist?list={pid}",
        "title": snippet.get("title", ""),
        "description": snippet.get("description", ""),
        "publishedAt": snippet.get("publishedAt", ""),
        "channelTitle": snippet.get("channelTitle", ""),
        "privacyStatus": status.get("privacyStatus", ""),
        "itemCount": _as_int(content.get("itemCount")),
        "available": True,
    }


def _shape_playlist_video_entry(playlist_id: str, playlist_item: dict[str, Any]) -> dict[str, Any]:
    snippet = playlist_item.get("snippet") or {}
    content = playlist_item.get("contentDetails") or {}
    status = playlist_item.get("status") or {}

    return {
        "kind": "playlist#video",
        "playlistId": playlist_id,
        "videoId": content.get("videoId", ""),
        "title": snippet.get("title", ""),
        "description": snippet.get("description", ""),
        "publishedAt": snippet.get("publishedAt", ""),
        "startAt": content.get("startAt", ""),
        "endAt": content.get("endAt", ""),
        "note": content.get("note", ""),
        "privacyStatus": status.get("privacyStatus", ""),
        "position": _as_int(snippet.get("position")),
    }


# ---------------------------------------------------------------------------
# Search enrichment
# ---------------------------------------------------------------------------

def enrich_search_items(youtube, search_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Preserve original order and enrich search items.

    - Videos -> videos.list -> _shape_video_info
    - Playlists -> playlists.list -> _shape_playlist_info
    Does NOT expand playlists via playlistItems.list.
    """
    spine: list[dict[str, Any]] = []
    video_ids: list[str] = []
    playlist_ids: list[str] = []

    for it in search_items:
        id_obj = it.get("id") or {}
        kind = (id_obj.get("kind") or "").lower()

        if kind.endswith("#video"):
            vid = id_obj.get("videoId") or ""
            spine.append({"kind": "video", "id": vid})
            if vid:
                video_ids.append(vid)
        elif kind.endswith("#playlist"):
            pid = id_obj.get("playlistId") or ""
            spine.append({"kind": "playlist", "id": pid})
            if pid:
                playlist_ids.append(pid)
        else:
            spine.append({"kind": "unknown", "raw": it})

    video_ids = list(dedupe_preserve_order(video_ids))
    playlist_ids = list(dedupe_preserve_order(playlist_ids))

    video_items, _video_errors = _get_video_details(youtube, video_ids)
    playlist_items, _playlist_errors = _get_playlist_details(youtube, playlist_ids)

    video_by_id = {it.get("id"): it for it in (video_items or []) if it.get("id")}
    playlist_by_id = {it.get("id"): it for it in (playlist_items or []) if it.get("id")}

    out: list[dict[str, Any]] = []
    for node in spine:
        if node["kind"] == "video":
            vid = node["id"]
            out.append(_shape_video_info(vid, video_by_id.get(vid)))
        elif node["kind"] == "playlist":
            pid = node["id"]
            out.append(_shape_playlist_info(pid, playlist_by_id.get(pid)))
        else:
            out.append({"kind": "unknown", "raw": node.get("raw")})

    return out


def youtube_search(
    query: Annotated[
        str,
        Field(
            description=(
                "Search query using standard web-search syntax when possible. "
                "Supports quoted phrases and exclusion with '-'."
            )
        ),
    ],
    order: Annotated[
        YtOrder | str,
        Field(default=YtOrder.relevance, description="Sort order. " + YtOrder.help_text()),
    ] = YtOrder.relevance,
    max_results: Annotated[int, Field(description="Max search items (1-50).", ge=1, le=50)] = 10,
    kinds: Annotated[
        SearchKind | str,
        Field(default=SearchKind.both, description="Return videos only, playlists only, or both."),
    ] = SearchKind.both,
) -> dict[str, Any]:
    """Search YouTube and return enriched MCP-friendly JSON (no playlist expansion)."""
    sk_kinds = SearchKind.coerce(kinds)
    yt_order = YtOrder.coerce(order)
    youtube = _get_youtube_client()

    req = youtube.search().list(        # pylint: disable=no-member
        part="id,snippet",
        q=query,
        type=sk_kinds.value,
        maxResults=max_results,
        order=yt_order.value,
    )

    try:
        resp = yt_execute(req, label="search.list")
    except HttpError as e:
        result = {
            "query": query,
            "order": yt_order.value,
            "maxResults": max_results,
            "kinds": sk_kinds.value,
            "items": [],
            "errors": [{"error": "YouTube API error", "details": str(e)}],
        }
        log_tree(
            logger,
            logging.DEBUG,
            "result",
            result,
            collapse_keys={"env"},  # env can be huge/noisy
            redact_keys={"token", "api_key"},
        )
        # log_tool_result("youtube_search", result, level=logging.DEBUG)

        return result

    search_items: list[dict[str, Any]] = resp.get("items") or []
    out_items = enrich_search_items(youtube, search_items)

    result = {
        "query": query,
        "order": yt_order.value,
        "maxResults": max_results,
        "kinds": sk_kinds.value,
        "items": out_items,
        "errors": [],
    }

    # log_tool_result("youtube_search", result, level=logging.DEBUG)
    return result


def youtube_video_info(
    inputs: Annotated[list[str], Field(description="List of YouTube video URLs or video IDs.")],
) -> dict[str, Any]:
    """Return full metadata for one or many videos."""
    youtube = _get_youtube_client()

    video_ids, errors = normalize_video_inputs(inputs)
    video_items, api_errors = _get_video_details(youtube, video_ids)
    errors.extend(api_errors)
    # Make a dict for easy lookup
    video_by_id = {it.get("id"): it for it in (video_items or []) if it.get("id")}

    items: list[dict[str, Any]] = []
    for vid in video_ids:
        items.append(_shape_video_info(vid, video_by_id.get(vid)))

    result = {
        "inputs_count": len(inputs),
        "video_ids_count": len(video_ids),
        "items": items,
        "errors": errors,
    }

    # log_tool_result("youtube_video_info", result, level=logging.DEBUG)
    return result


# CHANGE (request #3): playlist arg is ONLY str | list[str]
def youtube_playlist_info(
    playlist: Annotated[
        str | list[str],
        Field(description="Playlist URL/ID or list of playlist URLs/IDs."),
    ],
) -> dict[str, Any] | list[dict[str, Any]]:
    """Return general playlist metadata only.

    This does NOT fetch playlistItems or video metadata.
    """
    youtube = _get_youtube_client()
    inputs = _coerce_to_list_str(playlist)

    # Preserve input order.
    playlist_ids: list[str] = []
    parse_errors: list[dict[str, Any]] = []
    for raw in inputs:
        pid = extract_playlist_id(raw) or raw.strip()
        if not pid or not is_playlist_id(pid):
            parse_errors.append({"input": raw, "error": "Could not extract playlist_id"})
        playlist_ids.append(pid)

    unique_ids = [pid for pid in dedupe_preserve_order(playlist_ids) if pid]
    meta_items, meta_errors = _get_playlist_details(youtube, unique_ids)
    meta_by_id = {it.get("id"): it for it in (meta_items or []) if it.get("id")}

    out: list[dict[str, Any]] = []
    for pid, raw in zip(playlist_ids, inputs, strict=False):
        if not pid or pid not in meta_by_id:
            shaped = _shape_playlist_info(pid or "", None)
            shaped["errors"] = [{"input": raw, "error": "Playlist not found or invalid"}]
        else:
            shaped = _shape_playlist_info(pid, meta_by_id.get(pid))
            shaped["errors"] = []
        out.append(shaped)

    # Batch-level problems apply to all results:
    if parse_errors or meta_errors:
        for item in out:
            item.setdefault("errors", []).extend(parse_errors)
            item.setdefault("errors", []).extend(meta_errors)

    if isinstance(playlist, list):
        result: dict[str, Any] | list[dict[str, Any]] = out
        # log_tool_result("youtube_playlist_info", {"playlist_ids_count": len(out), "items": out}, level=logging.DEBUG)
    else:
        result = out[0] if out else {}
        # log_tool_result("youtube_playlist_info", {"playlist_ids_count": 1, "items": [result]}, level=logging.DEBUG)

    return result


# CHANGE (request #1): separate opt-in expansion tool
def youtube_playlist_video_list(
    playlist: Annotated[
        str | list[str],
        Field(description="Playlist URL/ID or list of playlist URLs/IDs."),
    ],
    max_videos: Annotated[int, Field(description="Max videos per playlist.", ge=1, le=500)] = 50,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Return playlist videos enriched with video_info-style metadata.

    Output shape per playlist:
      {
        "kind": "playlist_videos",
        "playlist_id": "...",
        "url": "https://www.youtube.com/playlist?list=...",
        "max_videos": 50,
        "items": { "<videoId>": { ...playlist_fields..., ...video_fields... } },
        "errors": [...]
      }
    """
    youtube = _get_youtube_client()
    inputs = _coerce_to_list_str(playlist)

    playlist_ids: list[str] = []
    parse_errors: list[dict[str, Any]] = []
    for raw in inputs:
        pid = extract_playlist_id(raw) or raw.strip()
        if not pid or not is_playlist_id(pid):
            parse_errors.append({"input": raw, "error": "Could not extract playlist_id"})
        playlist_ids.append(pid)

    # Optional: fetch playlist metadata for title/channel/itemCount (batched)
    unique_ids = [pid for pid in dedupe_preserve_order(playlist_ids) if pid]
    meta_items, meta_errors = _get_playlist_details(youtube, unique_ids)
    meta_by_id = {it.get("id"): it for it in (meta_items or []) if it.get("id")}

    out: list[dict[str, Any]] = []
    for pid, raw in zip(playlist_ids, inputs, strict=False):
        errors: list[dict[str, Any]] = []
        errors.extend(meta_errors)

        if not pid or not is_playlist_id(pid):
            out.append({
                "kind": "playlist_videos",
                "playlist_id": "",
                "url": "",
                "max_videos": max_videos,
                "items": {},
                "errors": errors + [{"input": raw, "error": "invalid playlist"}],
            })
            continue

        pl_items, pl_errs = youtube_get_playlist_videos(youtube, pid, max_videos=max_videos)
        errors.extend(pl_errs)

        # Shape playlistItems rows
        pl_video: dict[str, dict[str, Any]] = {}
        for itm in pl_items:
            vid = (itm.get("contentDetails") or {}).get("videoId")
            if not vid:
                continue
            pl_video[vid] = _shape_playlist_video_entry(pid, itm)

        # Enrich using videos.list (same fields as youtube_video_info)
        video_ids = list(pl_video.keys())
        if video_ids:
            v_items, v_errs = _get_video_details(youtube, video_ids)
            errors.extend(v_errs)
            v_by_id = {it.get("id"): it for it in (v_items or []) if it.get("id")}
            v_shaped = {vid: _shape_video_info(vid, v_by_id.get(vid)) for vid in video_ids}
            merge_outer(pl_video, v_shaped)

        pl_meta = _shape_playlist_info(pid, meta_by_id.get(pid))
        out.append({
            "kind": "playlist_videos",
            "playlist_id": pid,
            "url": pl_meta.get("url", f"https://www.youtube.com/playlist?list={pid}"),
            "title": pl_meta.get("title", ""),
            "channelTitle": pl_meta.get("channelTitle", ""),
            "itemCount": pl_meta.get("itemCount", 0),
            "max_videos": max_videos,
            "items": pl_video,
            "errors": errors,
        })

    # Parsing errors apply globally to the call:
    if parse_errors:
        for item in out:
            item.setdefault("errors", []).extend(parse_errors)

    if isinstance(playlist, list):
        result: dict[str, Any] | list[dict[str, Any]] = out
    else:
        result = out[0] if out else {}

    return result

# ---------------------------------------------------------------------------
# Test / CLI entry point (not a formal unit test, just a quick way to run and see results)q
# ---------------------------------------------------------------------------

def test() -> None:
    """Simple CLI entry point."""
    yt_search = "Python tutorials about list comprehension -shorts"

    logger.info("Executing youtube_search(query=%s)", yt_search)
    sr = youtube_search(query=yt_search, order="date", max_results=5, kinds="video,playlist")
    # log_tool_result("youtube_search", sr, level=logging.INFO)

    playlist_ids = [it.get("playlist_id") for it in sr.get("items", []) if it.get("kind") == "playlist"]
    playlist_ids = [pid for pid in playlist_ids if pid]

    if playlist_ids:
        logger.info("Fetching playlist info only (no expansion)")
        pi = youtube_playlist_info(playlist=playlist_ids[:2])
        # log_tool_result("youtube_playlist_info", {"items": pi if isinstance(pi, list) else [pi]}, level=logging.INFO)

        logger.info("Expanding playlist videos (opt-in)")
        pv = youtube_playlist_video_list(playlist=playlist_ids[0], max_videos=10)
        # if isinstance(pv, dict):
        #     log_tool_result("youtube_playlist_video_list", {"items": pv.get("items", {})}, level=logging.INFO, items_key="items")

# -------------------------------------------------------------------------------
# Search for Playlists only
# -------------------------------------------------------------------------------


    # logger.info("Executing youtube_search(query=%s)", yt_search)
    sr = youtube_search(query=yt_search, order="date", max_results=5, kinds="playlist")
    # log_tool_result("youtube_search", sr, level=logging.INFO)

    playlist_ids = [it.get("playlist_id") for it in sr.get("items", []) if it.get("kind") == "playlist"]
    playlist_ids = [pid for pid in playlist_ids if pid]

    if playlist_ids:
        logger.info("Expanding playlist videos (opt-in)")
        pv = youtube_playlist_video_list(playlist=playlist_ids[0], max_videos=3)
        # if isinstance(pv, dict):
        #     log_tool_result("youtube_playlist_video_list", {"items": pv.get("items", {})}, level=logging.INFO, items_key="items")


if __name__ == "__main__":
    test()
