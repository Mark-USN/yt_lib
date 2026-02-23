"""Utilities for consistent logging across the project."""

# src/lib/utils/log_utils.py
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, asdict, is_dataclass
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import sys
from typing import Any

from lib.utils.paths import project_name, resolve_cache_paths


# -----------------------------------------------------------------------------
# Public: Logger factory + configuration
# -----------------------------------------------------------------------------

_DEFAULT_ROOT = os.environ.get("MCP_LOG_ROOT", "yt_mcp")
_DEFAULT_LEVEL = os.environ.get("MCP_LOG_LEVEL", "INFO").upper()


@dataclass(frozen=True, slots=True)
class LogConfig:
    """Central logging policy for the whole app/library."""

    root: str = _DEFAULT_ROOT
    level: str = _DEFAULT_LEVEL

    # One place to define the default formatter for the entire project.
    fmt: str = "%(asctime)s %(levelname)s %(name)s%(context)s %(message)s"
    datefmt: str = "%H:%M:%S"

    # Destinations
    tee_console: bool = False
    console_stream: str = "stderr"  # "stderr" is safer with prompt-toolkit
    log_file: Path | None = None

    # Rotation (only applies when log_file is set)
    rotate_max_bytes: int = 5_000_000  # 5 MB
    rotate_backup_count: int = 5

    # Tree rendering defaults (used by log_tree)
    tree_indent: int = 2
    tree_max_depth: int = 10
    tree_max_items: int = 50
    tree_max_str: int = 220


class ContextAdapter(logging.LoggerAdapter):
    """Inject context values into LogRecord."""

    def process(self, msg: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        extra = kwargs.setdefault("extra", {})
        if not isinstance(extra, dict):
            extra = {}
            kwargs["extra"] = extra
        extra |= self.extra
        return msg, kwargs


def default_log_path(
    *,
    app_name: str,
    start: Path,
    env_vars: Sequence[str] | None = None,
    env_var: str = "MCP_CACHE_DIR",
) -> Path:
    """Compute the default rotating log file path:
    <app_cache_dir>/logs/<project_name>.log
    """
    cache = resolve_cache_paths(app_name=app_name, start=start, env_var=env_var, env_vars=env_vars)
    pname = project_name(start)
    return cache.app_cache_dir / "logs" / f"{pname}.log"


def configure_logging(cfg: LogConfig | None = None, *, force: bool = False) -> None:
    """Configure root logging with optional rotating file + optional console tee.

    Notes for prompt-toolkit:
      - Prefer cfg.tee_console=False (file-only), OR tee to stderr.
    """
    cfg = cfg or LogConfig()
    root_logger = logging.getLogger()

    level = _parse_level(cfg.level)
    root_logger.setLevel(level)

    if force:
        root_logger.handlers.clear()

    formatter = _ContextFormatter(cfg.fmt, datefmt=cfg.datefmt)

    # Avoid duplicate handlers by name.
    have_console = any(getattr(h, "name", "") == "mcp_console" for h in root_logger.handlers)
    have_file = any(getattr(h, "name", "") == "mcp_file" for h in root_logger.handlers)

    if cfg.tee_console and (force or not have_console):
        stream = sys.stderr if cfg.console_stream.lower() == "stderr" else sys.stdout
        ch = logging.StreamHandler(stream=stream)
        ch.name = "mcp_console"
        ch.setLevel(level)
        ch.setFormatter(formatter)
        root_logger.addHandler(ch)

    if cfg.log_file and (force or not have_file):
        cfg.log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            cfg.log_file,
            maxBytes=cfg.rotate_max_bytes,
            backupCount=cfg.rotate_backup_count,
            encoding="utf-8",
        )
        fh.name = "mcp_file"
        fh.setLevel(level)
        fh.setFormatter(formatter)
        root_logger.addHandler(fh)

    # Optional: quiet noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(
    name: str,
    *,
    cfg: LogConfig | None = None,
    child: str | None = None,
    **context: object,
) -> logging.Logger:
    """Factory: return a logger with project policy applied."""
    cfg = cfg or LogConfig()

    full_name = _normalize_name(name, root=cfg.root)
    base = logging.getLogger(full_name)
    if child:
        base = base.getChild(child)

    return ContextAdapter(base, context) if context else base


def bind(logger: logging.Logger, **context: object) -> logging.Logger:
    """Add/override context on an existing logger."""
    if isinstance(logger, ContextAdapter):
        merged = dict(logger.extra)
        merged |= context
        return ContextAdapter(logger.logger, merged)
    return ContextAdapter(logger, context)


# -----------------------------------------------------------------------------
# Public: Tree logging for nested dict/list payloads
# -----------------------------------------------------------------------------

def log_tree(
    logger: logging.Logger,
    level: int,
    label: str,
    obj: object,
    *,
    cfg: LogConfig | None = None,
    indent: int | None = None,
    max_depth: int | None = None,
    max_items: int | None = None,
    max_str: int | None = None,
    collapse_keys: set[str] | None = None,
    redact_keys: set[str] | None = None,
) -> None:
    if not logger.isEnabledFor(level):
        return

    cfg = cfg or LogConfig()
    rendered = format_tree(
        obj,
        indent=indent if indent is not None else cfg.tree_indent,
        max_depth=max_depth if max_depth is not None else cfg.tree_max_depth,
        max_items=max_items if max_items is not None else cfg.tree_max_items,
        max_str=max_str if max_str is not None else cfg.tree_max_str,
        collapse_keys=collapse_keys or {"raw"},
        redact_keys=redact_keys or set(),
    )
    logger.log(level, "%s\n%s", label, rendered)


def format_tree(
    obj: object,
    *,
    indent: int = 2,
    max_depth: int = 10,
    max_items: int = 50,
    max_str: int = 500,
    sort_dict_keys: bool = False,
    collapse_keys: set[str] | None = None,
    redact_keys: set[str] | None = None,
) -> str:
    collapse_keys = collapse_keys or set()
    redact_keys = redact_keys or set()

    seen: set[int] = set()
    lines: list[str] = []

    def _short(v: object) -> str:
        if isinstance(v, str):
            s = v.replace("\n", "\\n")
            return s if len(s) <= max_str else f"{s[: max_str - 1]} "
        try:
            r = repr(v)
        except Exception as e:  # pylint: disable=broad-exception-caught
            r = f"<repr failed: {type(e).__name__}: {e}>"
        return r if len(r) <= max_str else f"{r[: max_str - 1]} "

    def _is_seq(v: object) -> bool:
        return isinstance(v, Sequence) and not isinstance(v, (str, bytes, bytearray))

    def _collapsed_hint(v: object) -> str:
        if isinstance(v, Mapping):
            return f"<collapsed dict keys={len(v)}>"
        if _is_seq(v):
            return f"<collapsed list items={len(v)}>"
        return "<collapsed>"

    def _kind_summary(d: Mapping[object, object]) -> str | None:
        kind = d.get("kind")

        if kind == "video":
            vid = d.get("video_id", "")
            title = d.get("title", "")
            pub = d.get("publishedAt", "")
            stats = d.get("statistics")
            views = stats.get("views", "") if isinstance(stats, Mapping) else ""
            vpart = f" views={views}" if views != "" else ""
            return f"video {vid!s}{vpart} title={_short(title)} publishedAt={_short(pub)}"

        if kind == "playlist":
            pid = d.get("playlist_id", "")
            title = d.get("title", "")
            pub = d.get("publishedAt", "")
            cnt = d.get("itemCount", "")
            return (
                f"playlist {pid!s} itemCount={cnt!s} title={_short(title)}"
                f" publishedAt={_short(pub)}"
            )

        if kind == "playlist#video":
            pid = d.get("playlistId", "")
            vid = d.get("videoId", "")
            pos = d.get("position", "")
            title = d.get("title", "")
            pub = d.get("publishedAt", "")
            return (
                f"playlist#video playlistId={pid!s} videoId={vid!s} "
                f"position={pos!s} title={_short(title)} publishedAt={_short(pub)}"
            )

        return None

    def _coerce_to_walkable(v: object) -> object:
        if isinstance(v, Mapping):
            return v
        if isinstance(v, Sequence) and not isinstance(v, (str, bytes, bytearray)):
            return v

        model_dump = getattr(v, "model_dump", None)
        if callable(model_dump):
            try:
                return model_dump()
            except Exception:  # pylint: disable=broad-exception-caught
                pass

        as_dict = getattr(v, "dict", None)
        if callable(as_dict):
            try:
                return as_dict()
            except Exception:  # pylint: disable=broad-exception-caught
                pass

        if is_dataclass(v):
            try:
                return asdict(v)
            except Exception:  # pylint: disable=broad-exception-caught
                pass

        _asdict = getattr(v, "_asdict", None)
        if callable(_asdict):
            try:
                return _asdict()
            except Exception:  # pylint: disable=broad-exception-caught
                pass

        try:
            return vars(v)
        except TypeError:
            return v

    def _walk(v: object, prefix: str, depth: int) -> None:
        v = _coerce_to_walkable(v)

        if depth >= max_depth:
            lines.append(f"{prefix}<max_depth {max_depth} reached>")
            return

        if isinstance(v, Mapping):
            vid = id(v)
            if vid in seen:
                lines.append(f"{prefix}<cycle dict id={vid}>")
                return
            seen.add(vid)

            header = _kind_summary(v)
            child_prefix = prefix + " " * indent if header is not None else prefix
            if header is not None:
                lines.append(f"{prefix}{header}")

            keys = list(v.keys())
            if sort_dict_keys:
                try:
                    keys.sort()
                except Exception:  # pylint: disable=broad-exception-caught
                    pass

            shown = 0
            for k in keys:
                if shown >= max_items:
                    remaining = max(0, len(keys) - shown)
                    lines.append(f"{child_prefix}  <{remaining} more keys>")
                    break

                key = str(k)

                if key in redact_keys:
                    lines.append(f"{child_prefix}{key}: <redacted>")
                    shown += 1
                    continue

                val = v.get(k)

                if key in collapse_keys and (isinstance(val, Mapping) or _is_seq(val)):
                    lines.append(f"{child_prefix}{key}: {_collapsed_hint(val)}")
                    shown += 1
                    continue

                if isinstance(val, Mapping) or _is_seq(val):
                    lines.append(f"{child_prefix}{key}:")
                    _walk(val, child_prefix + " " * indent, depth + 1)
                else:
                    lines.append(f"{child_prefix}{key}: {_short(val)}")
                shown += 1
            return

        if _is_seq(v):
            vid = id(v)
            if vid in seen:
                lines.append(f"{prefix}<cycle seq id={vid}>")
                return
            seen.add(vid)

            n = len(v)
            limit = min(n, max_items)
            for i in range(limit):
                item = v[i]
                if isinstance(item, Mapping) or _is_seq(item):
                    lines.append(f"{prefix}[{i}]:")
                    _walk(item, prefix + " " * indent, depth + 1)
                else:
                    lines.append(f"{prefix}[{i}]: {_short(item)}")

            if n > limit:
                lines.append(f"{prefix}  <{n - limit} more items>")
            return

        lines.append(f"{prefix}{_short(v)}")

    _walk(obj, "", 0)
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Internals
# -----------------------------------------------------------------------------

class _ContextFormatter(logging.Formatter):
    """Adds %(context)s to the record based on any extra keys in the record."""

    _KNOWN_STD = {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "asctime",
    }

    def format(self, record: logging.LogRecord) -> str:
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k not in self._KNOWN_STD and not k.startswith("_")
        }
        if extras:
            parts = " ".join(f"{k}={_safe_value(v)}" for k, v in sorted(extras.items()))
            record.context = f" [{parts}]"
        else:
            record.context = ""
        return super().format(record)


def _safe_value(v: object) -> str:
    try:
        s = str(v)
    except Exception as e:  # pylint: disable=broad-exception-caught
        return f"<str failed: {type(e).__name__}: {e}>"
    return s.replace("\n", "\\n")


def _normalize_name(name: str, *, root: str) -> str:
    if name == "__main__":
        base = root
    else:
        base = name

    base = base.removeprefix("src.").removeprefix("lib.")

    if base.startswith(root + ".") or base == root:
        return base
    return f"{root}.{base}"


def _parse_level(level: str) -> int:
    value = logging.getLevelName(level.upper().strip())
    if not isinstance(value, int):
        raise ValueError(f"Invalid logging level: {level!r}")
    return value