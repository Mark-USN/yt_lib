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
from pathlib import Path
import os


class ProjectLayoutError(RuntimeError):
    """Raised when we cannot determine the project root from a file path."""


def project_root_from_src(start: Path) -> Path:
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
        if p.name == "src":
            return p.parent
    raise ProjectLayoutError(f"Could not locate 'src' in parents of: {start}")


@dataclass(slots=True, frozen=True)
class CachePaths:
    """Resolved cache directory paths for a given component/app."""
    base_cache_dir: Path
    app_cache_dir: Path


def resolve_cache_paths(*, app_name: str, start: Path, env_var: str = "MCP_CACHE_DIR") -> CachePaths:
    """Resolve cache directories for a given app/component.

    Resolution order:
      1) If env_var is set (default: MCP_CACHE_DIR), use it as the base cache directory.
      2) Otherwise, derive project root by finding 'src' above `start`, then use PARENT/cache.

    The app cache directory is:
      <base_cache_dir>/<app_name> if app_name is non-empty, else just <base_cache_dir>.

    Args:
        app_name: Component/app name (e.g., "universal_client").
        start: A path under the project's src tree (typically Path(__file__)).
        env_var: Environment variable that overrides the cache base directory.

    Returns:
        CachePaths with both base and app cache directories created/returned.
    """
    if override := os.environ.get(env_var):
        base_cache = Path(override).expanduser().resolve()
    else:
        base_cache = project_root_from_src(start) / "cache"
    if app_name.strip() == "":
        app_cache = base_cache
    else:
        app_cache = (base_cache / app_name).resolve()
    app_cache.mkdir(parents=True, exist_ok=True)
    return CachePaths(base_cache_dir=base_cache, app_cache_dir=app_cache)

def resolve_project_path(*, start: Path) -> Path:
    """Resolve Projects base directory.

    Resolution order:
      2) Recurse the start path to 'src' and take its parent.

    Args:
        start: A path under the project's src tree (typically Path(__file__)).

    Returns:
        A Path object pointing at the Project's base path.
    """
    return project_root_from_src(start) 

def get_module_path(*, start: Path) -> CachePaths:
    """Resolve Module directory.

    Resolution order:
      Find the projects base path and add src/lib to it..

    Args:
        start: A path under the project's src tree (typically Path(__file__)).

    Returns:
        A Path object pointing to the project's module path.
    """
    return project_root_from_src(start) / "src" / "lib"
