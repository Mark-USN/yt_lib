"""
app_info.py

Runtime info helpers for application and service paths.

User applications use platformdirs user directories.

Services use machine/system-level directories:

Windows:
    C:\\ProgramData\\<app_name>\\cache
    C:\\ProgramData\\<app_name>\\config
    C:\\ProgramData\\<app_name>\\data
    C:\\ProgramData\\<app_name>\\state
    C:\\ProgramData\\<app_name>\\logs

Linux:
    /var/cache/<app_name>
    /etc/<app_name>
    /var/lib/<app_name>
    /var/lib/<app_name>/state
    /var/log/<app_name>
"""

from __future__ import annotations

import ctypes
import locale as py_locale
import os
import sys
from ctypes import wintypes
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

from babel.numbers import format_decimal
from platformdirs import PlatformDirs


_LOCALE_NAME_MAX_LENGTH = 85


def _round_half_up(value: float | Decimal) -> int:
    """ Round a number to the nearest integer using round-half-up.
        Args:
            value: The number to round.
        Returns:
            The rounded integer.
    """
    return int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


@dataclass(slots=True, frozen=True)
class Baseinfo:
    """ Resolved runtime info for application paths and locale.
        Args:
            app_name: The name of the application.
            app_author: The author of the application.
            app_dir: The base directory of the application.
            locale: The locale of the application.
            cache_dir: The directory for cache files.
            config_dir: The directory for configuration files.
            data_dir: The directory for data files.
            state_dir: The directory for state files.
            log_dir: The directory for log files.
    """

    app_name: str
    app_author: str
    app_dir:Path
    locale: str
    cache_dir: Path
    config_dir: Path
    data_dir: Path
    state_dir: Path
    log_dir: Path
    documents_dir: Path | None = None


def get_windows_user_posix_locale() -> str:
    """ Return the Windows user's locale, such as 'en-US'.

        This uses the Windows API because Python's locale helpers may not return
        a usable POSIX/Babel-style locale name on Windows.
    """
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    func = kernel32.GetUserDefaultLocaleName
    func.argtypes = [wintypes.LPWSTR, ctypes.c_int]
    func.restype = ctypes.c_int

    buf = ctypes.create_unicode_buffer(_LOCALE_NAME_MAX_LENGTH)

    if func(buf, len(buf)) == 0:
        raise OSError(ctypes.get_last_error(), "GetUserDefaultLocaleName failed")

    return buf.value


def get_system_posix_locale() -> str:
    """ Return a POSIX-style locale name for non-Windows platforms.
        Returns:
            A locale string in the format language_COUNTRY, suitable for Babel.
    """
    loc = py_locale.getlocale()[0]
    if loc:
        return loc

    loc = py_locale.getlocale()[0]
    if loc:
        return loc

    return "en_US"


def detect_locale(default: str = "en_US") -> str:
    """ Detect a Babel-compatible locale.
        Args:
            default: The default locale to return if detection fails.
        Returns:
            A locale string in the format language_COUNTRY, suitable for Babel.

        Returns values like:
            en_US
            de_DE
            fr_FR
    """
    try:
        if sys.platform == "win32":
            loc = get_windows_user_posix_locale()
            return loc.replace("-", "_") if loc else default

        loc = get_system_posix_locale()
        return loc.replace("-", "_") if loc else default

    except Exception:  # pylint: disable=broad-exception-caught
        return default


def _env_path(name: str) -> Path | None:
    """ Return a Path from an environment variable, if it is set.
        Args:
            name: The name of the environment variable.
        Returns:
            A Path object if the environment variable is set, or None if it is not set.
    """
    value = os.environ.get(name)
    return Path(value) if value else None


def _path_from_env_or_default(env_name: str, default: Path) -> Path:
    """ Return path from env var, falling back to default.
        Args:
            env_name: The name of the environment variable.
            default: The default path to use if the environment variable is not set.
        Returns:
            A Path object from the environment variable if set, otherwise the default path.
    """
    return _env_path(env_name) or default


def create_user_info(app_name: str, app_author: str, app_dir: Path | str) -> Baseinfo:
    """ Create a Baseinfo for a normal user application.
        Args:
            app_name: The name of the application, used for directory paths.
            app_author: The author of the application, used for directory paths.
            app_dir: The base directory of the application.
        Returns:
            A Baseinfo object for the user application.
    """
    dirs = PlatformDirs(appname=app_name, appauthor=app_author)

    env_prefix = app_name.upper().replace("-", "_")

    return Baseinfo(
        app_name=app_name,
        app_author=app_author,
        app_dir=Path(app_dir).resolve(),
        locale=detect_locale(),
        cache_dir=_path_from_env_or_default(
            f"{env_prefix}_CACHE_DIR",
            Path(dirs.user_cache_dir)
        ),
        config_dir=_path_from_env_or_default(
            f"{env_prefix}_CONFIG_DIR",
            Path(dirs.user_config_dir)
        ),
        data_dir=_path_from_env_or_default(
            f"{env_prefix}_DATA_DIR",
            Path(dirs.user_data_dir)
        ),
        state_dir=_path_from_env_or_default(
            f"{env_prefix}_STATE_DIR",
            Path(dirs.user_state_dir),
        ),
        log_dir=_path_from_env_or_default(
            f"{env_prefix}_LOG_DIR",
            Path(dirs.user_log_dir)
        ),
        documents_dir=_path_from_env_or_default(
            f"{env_prefix}_DOCUMENTS_DIR",
            Path(dirs.user_documents_dir)
        ),
    )


def create_service_info(app_name: str, app_author: str, app_dir) -> Baseinfo:
    """ Create a Baseinfo for a Windows or Linux service.
        Args:
            app_name: The name of the application, used for directory paths.
            app_author: The author of the application, used for directory paths.
            app_dir: The base directory of the application.
        Returns:
            A Baseinfo object for the service application.
    """
    env_prefix = app_name.upper().replace("-", "_")

    if sys.platform == "win32":
        base = _path_from_env_or_default(
            f"{env_prefix}_BASE_DIR",
            Path(os.environ.get("PROGRAMDATA", r"C:\ProgramData")) / app_name,
        )

        return Baseinfo(
            app_name=app_name,
            app_author=app_author,
            app_dir=Path(app_dir).resolve(),
            locale=detect_locale(),
            cache_dir=_path_from_env_or_default(f"{env_prefix}_CACHE_DIR", base / "cache"),
            config_dir=_path_from_env_or_default(f"{env_prefix}_CONFIG_DIR", base / "config"),
            data_dir=_path_from_env_or_default(f"{env_prefix}_DATA_DIR", base / "data"),
            state_dir=_path_from_env_or_default(f"{env_prefix}_STATE_DIR", base / "state"),
            log_dir=_path_from_env_or_default(f"{env_prefix}_LOG_DIR", base / "logs"),
            documents_dir=None,
        )

    return Baseinfo(
        app_name=app_name,
        app_author=app_author,
        app_dir=Path(app_dir).resolve(),
        locale=detect_locale(),
        cache_dir=_path_from_env_or_default(
            f"{env_prefix}_CACHE_DIR",
            Path("/var/cache") / app_name,
        ),
        config_dir=_path_from_env_or_default(
            f"{env_prefix}_CONFIG_DIR",
            Path("/etc") / app_name,
        ),
        data_dir=_path_from_env_or_default(
            f"{env_prefix}_DATA_DIR",
            Path("/var/lib") / app_name,
        ),
        state_dir=_path_from_env_or_default(
            f"{env_prefix}_STATE_DIR",
            Path("/var/lib") / app_name / "state",
        ),
        log_dir=_path_from_env_or_default(
            f"{env_prefix}_LOG_DIR",
            Path("/var/log") / app_name,
        ),
        documents_dir=None,
    )


@dataclass(slots=True)
class RuntimeInfo:
    """ Convenient shared access to Baseinfo information.
        Args:
            info: The Baseinfo object to wrap.
    """

    info: Baseinfo

    @property
    def app_name(self) -> str:
        """ Returns the name of the application."""
        return self.info.app_name

    @property
    def app_author(self) -> str:
        """ Returns the author of the application."""
        return self.info.app_author

    @property
    def locale(self) -> str:
        """ Returns the locale of the application."""
        return self.info.locale

    @property
    def app_dir(self) -> Path:
        """ Returns the application root directory path, ensuring it exists."""
        path = self.info.app_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def cache_dir(self) -> Path:
        """ Returns the cache directory path, ensuring it exists."""
        path = self.info.cache_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def config_dir(self) -> Path:
        """ Returns the configuration directory path, ensuring it exists."""
        path = self.info.config_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def data_dir(self) -> Path:
        """ Returns the data directory path, ensuring it exists."""
        path = self.info.data_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def state_dir(self) -> Path:
        """ Returns the state directory path, ensuring it exists."""
        path = self.info.state_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def log_dir(self) -> Path:
        """ Returns the log directory path, ensuring it exists."""
        path = self.info.log_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def log_path(self) -> Path:
        """ Returns the path to a file in the log directory, ensuring the
            directory exists.
        
        """
        return self.log_dir / f"{self.app_name}.log"


    @property
    def documents_dir(self) -> Path:
        """ Returns the documents directory path, ensuring it exists."""
        if self.info.documents_dir is None:
            msg = "This runtime info does not define a documents directory."
            raise RuntimeError(msg)

        path = self.info.documents_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    def documents_path(self, file_path: str) -> Path:
        """ Returns the path to a file in the documents directory, ensuring the
            directory exists.
                Args:
                    file_path: The relative path to the file within the documents directory.
                Returns:
                    A Path object representing the full path to the file in the documents directory.
                Raises:
                    RuntimeError: If this runtime info does not define a documents directory.
        
        """
        return self.documents_dir / file_path

    @property
    def transcript_dir(self) -> Path:
        """ Returns the transcript directory path, ensuring it exists."""
        path = self.data_dir / "transcripts"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def transcript_path(self, video_id: str) -> Path:
        """ Returns the path to a transcript file for a given video ID, ensuring the
            directory exists.
            Args:
                video_id: The ID of the video for which to get the transcript path.
            Returns:
                A Path object representing the full path to the transcript file.
        """
        return self.transcript_dir / f"{video_id}.json"

    @property
    def vid_info_dir(self) -> Path:
        """ Returns the video info directory path, ensuring it exists."""
        path = self.cache_dir / "vid_info"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def vid_info_path(self, video_id: str) -> Path:
        """ Returns the path to a video info file for a given video ID, ensuring the
            directory exists.
            Args:
                video_id: The ID of the video for which to get the info path.
            Returns:
                A Path object representing the full path to the video info file.
        """
        return self.vid_info_dir / f"{video_id}.json"

    @property
    def audio_dir(self) -> Path:
        """ Returns the audio directory path, ensuring it exists."""
        path = self.cache_dir / "audio"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def audio_path(self, audio_file: str) -> Path:
        """ Returns the path to an audio file for a given video ID, ensuring the
            directory exists.
            Args:
                audio_file: The name of the audio file to get the path for.
            Returns:
                A Path object representing the full path to the audio file.
        """
        return self.audio_dir / audio_file

    def format_number(
        self,
        value: int | float | Decimal | None,
        *,
        decimals: int = 2,
        as_int: bool = False,
    ) -> str:
        """ Formats a number according to the locale and specified options.
            Args:
                value: The number to format.
                decimals: The number of decimal places to include.
                as_int: Whether to format the number as an integer.
            Returns:
                A string representing the formatted number.
        """
        if value is None:
            return ""

        if as_int:
            if isinstance(value, float | Decimal):
                value = _round_half_up(value)

            return format_decimal(value, format="#,##0", locale=self.locale)

        pattern = "#,##0"
        if decimals > 0:
            pattern = f"#,##0.{'0' * decimals}"

        return format_decimal(value, format=pattern, locale=self.locale)
