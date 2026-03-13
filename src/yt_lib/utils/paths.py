from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from platformdirs import PlatformDirs


@dataclass(frozen=True, slots=True)
class RuntimeDirs:
    """OS-appropriate directories for runtime use."""

    app_name: str
    app_author: str
    base_cache_dir: Path
    base_log_dir: Path
    # base_data_dir: Path
    # base_config_dir: Path

    # Optional per-component subdir (often same as app_name; useful if app_name=""
    # or you want a second-level component like "yt_mcp/universal_client").
    # component_cache_dir: Path
    # component_log_dir: Path
    # component_data_dir: Path
    # component_config_dir: Path


def _first_env_value(env_vars: Iterable[str]) -> str | None:
    for k in env_vars:
        if not isinstance(k, str):
            continue
        k = k.strip()
        if not k:
            continue
        v = os.environ.get(k)
        if v:
            return v
    return None


def _resolve_override(
    *,
    primary: str | None,
    fallbacks: Iterable[str] | None,
) -> str | None:
    if primary:
        v = os.environ.get(primary)
        if v:
            return v
    if fallbacks is not None:
        v = _first_env_value(fallbacks)
        if v:
            return v
    return None


def resolve_runtime_dirs(
    *,
    app_name: str,
    app_author: str = "HenCode",
    # component: str = "",
    cache_env_var: str = "MCP_CACHE_DIR",
    cache_env_vars: Iterable[str] | None = None,
    log_env_var: str = "MCP_LOG_DIR",
    log_env_vars: Iterable[str] | None = None,
    # data_env_var: str = "MCP_DATA_DIR",
    # data_env_vars: Iterable[str] | None = None,
    # config_env_var: str = "MCP_CONFIG_DIR",
    # config_env_vars: Iterable[str] | None = None,
    ensure_exists: bool = True,
) -> RuntimeDirs:
    """
    Resolve platform-appropriate runtime directories with optional env overrides.

    - Uses platformdirs for OS-native locations.
    - Allows separate overrides for cache/log/data/config.
    - Optionally creates directories.

    Directory behavior:
    - "base_*" are the platformdirs locations for the app.
    - "component_*" are either base_* or base_*/<component> if component is non-empty.
    """
    app_name_clean = app_name.strip()
    if not app_name_clean:
        raise ValueError("app_name must be a non-empty string")

    app_author_clean = app_author.strip()
    if not app_author_clean:
        raise ValueError("app_author must be a non-empty string")

    # component_clean = component.strip()

    dirs = PlatformDirs(app_name_clean, app_author_clean)

    cache_override = _resolve_override(primary=cache_env_var, fallbacks=cache_env_vars)
    log_override = _resolve_override(primary=log_env_var, fallbacks=log_env_vars)
    # data_override = _resolve_override(primary=data_env_var, fallbacks=data_env_vars)
    # config_override = _resolve_override(primary=config_env_var, fallbacks=config_env_vars)

    base_cache = Path(cache_override).expanduser().resolve() if cache_override else Path(dirs.user_cache_dir).resolve()
    base_log = Path(log_override).expanduser().resolve() if log_override else Path(dirs.user_log_dir).resolve()
    # base_data = Path(data_override).expanduser().resolve() if data_override else Path(dirs.user_data_dir).resolve()
    # base_config = (
    #     Path(config_override).expanduser().resolve()
    #     if config_override
    #     else Path(dirs.user_config_dir).resolve()
    # )
 
    if ensure_exists:
        for p in (base_cache, base_log):
            p.mkdir(parents=True, exist_ok=True)

    # comp_cache = base_cache if component_clean == "" else (base_cache / component_clean).resolve()
    # comp_log = base_log if component_clean == "" else (base_log / component_clean).resolve()
    # comp_data = base_data if component_clean == "" else (base_data / component_clean).resolve()
    # comp_config = base_config if component_clean == "" else (base_config / component_clean).resolve()

    # if ensure_exists:
    #     for p in (comp_cache, comp_log, comp_data, comp_config):
    #         p.mkdir(parents=True, exist_ok=True)

    return RuntimeDirs(
        app_name=app_name_clean,
        app_author=app_author_clean,
        base_cache_dir=base_cache,
        base_log_dir=base_log,
        # base_data_dir=base_data,
        # base_config_dir=base_config,
        # component_cache_dir=comp_cache,
        # component_log_dir=comp_log,
        # component_data_dir=comp_data,
        # component_config_dir=comp_config,
    )