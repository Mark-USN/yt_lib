""" Utilities for consistent logging across the project."""

# src/lib/utils/log_utils.py
from __future__ import annotations

from dataclasses import dataclass
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import os
import sys
from typing import Any
from yt_lib.utils.tree_view import TreeView, TreeViewConfig

Logger = logging.Logger

# Define here so that they can be used in the LogConfig defaults without circular imports.
CRITICAL = logging.CRITICAL
FATAL = logging.FATAL
ERROR = logging.ERROR
WARNING = logging.WARNING
WARN = logging.WARN
INFO = logging.INFO
DEBUG = logging.DEBUG
NOTSET = logging.NOTSET

# -----------------------------------------------------------------------------
# Public: Logger factory + configuration
# -----------------------------------------------------------------------------

_DEFAULT_ROOT = os.environ.get("MCP_LOG_ROOT", "yt_mcp")
_DEFAULT_LEVEL = os.environ.get("MCP_LOG_LEVEL", "INFO").upper()


@dataclass(frozen=True, slots=True)
class LogConfig:
    """ Central logging policy for the whole app/library."""

    log_root: str = _DEFAULT_ROOT
    log_level: str = _DEFAULT_LEVEL
    # One place to define the default formatter for the entire project.
    fmt: str = (
        "%(asctime)s %(levelname)s %(name)s"
        "%(context)s"
        "%(message)s"
    )
    datefmt: str = "%H:%M:%S"

@dataclass(frozen=True, slots=True)
class FileLogConfig:
    """ Configuration for file-based logging. """
    log_file: Path = None
    max_bytes: int = 5_000_000
    backup_count: int = 5
    encoding: str = "utf-8"
    @property
    def log_file(self) -> Path:
        """ Returns the log file path, ensuring its parent directory exists."""
        path = self.log_file if isinstance(self.log_file, Path) else Path(self.log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


class ContextAdapter(logging.LoggerAdapter):
    """ Inject context values into LogRecord.

        Use get_logger(..., job_id=..., tool=...) so you don't have to pass `extra=`
        in every log call.

        The formatter can reference these with %(job_id)s, %(tool)s, etc.
    """

    def process(self, msg: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """ Merge adapter context with any extra provided in the log call.
            Args:
                msg: The log message format string.
                kwargs: The keyword arguments passed to the logging call.
            Returns:
                A tuple of the message and the updated kwargs with merged context.
        """
        extra = kwargs.setdefault("extra", {})
        if not isinstance(extra, dict):
            # If user passed something unexpected, don't crash logging.
            extra = {}
            kwargs["extra"] = extra
        extra |= self.extra
        return msg, kwargs


def configure_logging(
                        cfg: LogConfig | None = None,
                        *,
                        file_log_conf: FileLogConfig | None = None,
                        force: bool = False,
                        tee_console: bool = True,
                    ) -> None:

    """ Idempotent-ish basic configuration for console apps.
        Args:
            cfg: Optional LogConfig to override defaults.
            force: If True, force reconfiguration even if handlers exist.
            file: Optional FileLogConfig for file-based logging.
            tee_console: If True, also log to console.

        - Safe to call early in your entrypoint (server/client).
        - If handlers already exist on the root logger, we won't override them,
          unless force=True.
    """
    cfg = cfg or LogConfig()

    root_logger = logging.getLogger()

    if root_logger.handlers and not force:
        return

    level = _parse_level(cfg.log_level)
    root_logger.setLevel(level)

    if force:
        root_logger.handlers.clear()

    formatter = _ContextFormatter(cfg.fmt, datefmt=cfg.datefmt)

    if tee_console:
        sh = logging.StreamHandler(stream=sys.stdout)
        sh.setLevel(level)
        sh.setFormatter(formatter)
        root_logger.addHandler(sh)

    if file_log_conf is not None:
        fh = RotatingFileHandler(
            file_log_conf.log_file,
            maxBytes=file_log_conf.max_bytes,
            backupCount=file_log_conf.backup_count,
            encoding=file_log_conf.encoding,
        )
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
) -> Logger:
    """ Factory: return a logger with project policy applied.
        Args:
            name: The name of the logger.
            cfg: Optional LogConfig to override defaults.
            child: Optional child logger name.
            **context: Optional context values to attach to the logger.
        Returns:
            A logger instance with the specified configuration and context.

        Benefits over logging.getLogger(__name__):
          - Ensures consistent root namespace (e.g. yt_mcp.*)
          - Optionally attaches structured context (job_id, tool, session_id, ...)
          - Optionally returns a child logger (name.child)
    """
    cfg = cfg or LogConfig()

    full_name = _normalize_name(name, root=cfg.log_root)
    base = logging.getLogger(full_name)
    if child:
        base = base.getChild(child)

    return ContextAdapter(base, context) if context else base


def bind(logger: logging.Logger, **context: object) -> Logger:
    """ Add/override context on an existing logger.
        Args:
            logger: The logger instance to bind context to.
            **context: Key-value pairs of context to add or override.
        Returns:
            A new logger instance with the merged context.

        Useful when you create a module logger once, then later learn job/session ids.
    """
    if isinstance(logger, ContextAdapter):
        merged = dict(logger.extra)
        merged |= context
        return ContextAdapter(logger.logger, merged)

    return ContextAdapter(logger, context)


# -----------------------------------------------------------------------------
# Public: Tree logging for nested dict/list payloads (YT results, raw blobs, etc.)
# -----------------------------------------------------------------------------

def log_tree(
    logger: logging.Logger,
    level: int,
    label: str,
    obj: object,
    *,
    cfg: TreeViewConfig | None = None,
    **kwargs,       # Optional extra kwargs to pass to TreeViewConfig, like collapse_keys
                    # or redact_keys.
) -> None:
    """ Log nested structures in a stable, readable, indented format.
        Args:
            logger: The logger instance to use for logging.
            level: The logging level.
            label: A label for the logged object.
            obj: The object to log.
            **kwargs: Additional keyword arguments passed to TreeView.render_tree().
    Typical usage:
        log_tree(logger, logging.DEBUG, "playlist_info", payload, cfg=cfg)

    Notes:
      - If you're adding a big "raw" subtree, use collapse_keys={"raw"}.
      - For secrets/tokens, use redact_keys={"accessToken", "api_key"}.
    """
    if not logger.isEnabledFor(level):
        return

    logger.log(level, "%s", TreeView().render_tree(obj=obj, title=label, cfg=cfg, **kwargs))

# -----------------------------------------------------------------------------
# Internals
# -----------------------------------------------------------------------------

class _ContextFormatter(logging.Formatter):
    """ Adds %(context)s to the record based on any extra keys in the record. """

    _KNOWN_STD = {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "asctime",
    }

    def format(self, record: logging.LogRecord) -> str:
        """ Build a " [key=value key=value]" suffix for any extra fields in the record. 
            Args:
                record: The LogRecord to format.
            Returns:
                The formatted log record as a string.
        """
        # Build a compact " key=value" suffix from extra fields (job_id, tool, etc.)
        extras = {
            k: v for k, v in record.__dict__.items()
            if k not in self._KNOWN_STD and not k.startswith("_")
        }
        if extras:
            # Stable ordering makes logs easy to diff.
            parts = " ".join(f"{k}={_safe_value(v)}" for k, v in sorted(extras.items()))
            record.context = f" [{parts}]"
        else:
            record.context = ""
        return super().format(record)


def _safe_value(v: object) -> str:
    """ Convert a value to a string, safely handling exceptions and escaping newlines.
        Args:
            v: The value to convert to a string.
        Returns:
            A string representation of the value, with newlines escaped. If conversion fails,
            returns a placeholder string indicating the failure.
    """
    try:
        s = str(v)
    except Exception as err:  # pylint: disable=broad-exception-caught
        return f"<str failed: {type(err).__name__}: {err}>"
    return s.replace("\n", "\\n")


def _normalize_name(name: str, *, root: str) -> str:
    """ Normalize a logger name to ensure it starts with the root namespace.
        Args:
            name: The original logger name (e.g., __name__).
            root: The root namespace to enforce (e.g., "yt_mcp").
        Returns:
            A normalized logger name that starts with the root namespace.
    """
    # Common case: name is __name__ like "src.module.youtube.search"
    # You may want to map "src.module." away; keep it simple and predictable.
    if name == "__main__":
        base = root
    else:
        base = name

    # Optional cleanup: if you run from src/, you might get "src.<...>"
    base = base.removeprefix("src.").removeprefix("lib.")

    if base.startswith(root + ".") or base == root:
        return base
    return f"{root}.{base}"


def _parse_level(level: str) -> int:
    """ Parse a logging level from a string, case-insensitively.
        Args:
            level: The logging level as a string (e.g., "DEBUG", "INFO", "WARNING").
        Returns:
            The corresponding logging level as an integer.
        Raises:
            ValueError: If the provided level string is not a valid logging level.
    """
    value = logging.getLevelName(level.upper().strip())
    if not isinstance(value, int):
        raise ValueError(f"Invalid logging level: {level!r}")
    return value
