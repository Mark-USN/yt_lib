"""
src/lib/utils/paths.py

Project path helpers for a repo layout like:

  PARENT/
    src/
      ...
    cache/
      ...

Invariant:
  - project root == parent directory of the nearest 'src' directory.

Why not Path.cwd()?
  - The working directory can vary (uv, VS, tasks, tests). These helpers anchor
    on a known file path (typically Path(__file__)) instead.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tomllib
from collections.abc import Iterable


class ProjectLayoutError(RuntimeError):
    """Raised when we cannot determine the project root from a file path."""


def project_root(start: Path) -> Path:
    """Return PARENT for a layout of PARENT/src/... by locating 'src' in parents.

    Args:
        start: A file or directory path somewhere under PARENT/src.

    Returns:
        The project root path (PARENT).

    Raises:
        ProjectLayoutError: If no 'src' ancestor directory is found.
    """
    here = start.resolve()
    for p in (here, *here.parents):
        if p.name == "src" or p.name == ".venv":
            return p.parent
    raise ProjectLayoutError(f"Could not locate 'src' in parents of: {start}")


def project_name(start: Path, *, fallback: str = "app") -> str:
    """Best-effort project name.

    Preference order:
      1) [project].name in pyproject.toml at project root
      2) project root directory name
      3) fallback
    """
    root = project_root(start)
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            name = data.get("project", {}).get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()
        except Exception:
            # If parsing fails, fall back below.
            pass

    # fallback to directory name (e.g., repo folder)
    if root.name.strip():
        return root.name.strip()

    return fallback


@dataclass(slots=True, frozen=True)
class CachePaths:
    """Resolved cache directory paths for a given component/app."""

    base_cache_dir: Path
    app_cache_dir: Path


def resolve_cache_paths(
    *,
    app_name: str,
    start: Path,
    env_var: str = "MCP_CACHE_DIR",
    env_vars: Iterable[str] | None = None,
) -> CachePaths:
    """Resolve cache directories for a given app/component.

    Resolution order:
      1) If any env var in env_vars is set, use the first one found as base cache dir.
      2) Else if env_var is set (legacy single-var), use it as base cache dir.
      3) Otherwise derive project root by finding 'src' above `start`, then use PARENT/cache.

    The app cache directory is:
      <base_cache_dir>/<app_name> if app_name is non-empty, else just <base_cache_dir>.

    Args:
        app_name: Component/app name (e.g., "universal_client").
        start: A path under the project's src tree (typically Path(__file__)).
        env_var: Legacy single environment variable override (default: MCP_CACHE_DIR).
        env_vars: Optional list/tuple of env var names to check in priority order.

    Returns:
        CachePaths with both base and app cache directories created/returned.
    """
    override: str | None = None

    if env_vars is not None:
        for k in env_vars:
            if not isinstance(k, str) or not k.strip():
                continue
            v = os.environ.get(k)
            if v:
                override = v
                break

    if override is None:
        override = os.environ.get(env_var)

    if override:
        base_cache = Path(override).expanduser().resolve()
    else:
        base_cache = project_root(start) / "cache"

    app_name_clean = app_name.strip()
    app_cache = base_cache if app_name_clean == "" else (base_cache / app_name_clean).resolve()
    app_cache.mkdir(parents=True, exist_ok=True)

    return CachePaths(base_cache_dir=base_cache, app_cache_dir=app_cache)


def resolve_project_path(*, start: Path) -> Path:
    """Resolve Projects base directory."""
    return project_root(start)


def get_module_path(*, start: Path) -> Path:
    """Resolve Module directory: <project_root>/src/lib."""
    return project_root(start) / "src" / "lib"