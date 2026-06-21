""" A class to transverse nested data structures generating an indented text output of the 
    structure.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, is_dataclass, field
from collections.abc import Mapping, Sequence
from typing import Any

@dataclass(slots=True)
class TreeViewConfig:
    """ Configuration for TreeView rendering. """
    indent: int = 2
    max_depth: int = 10
    max_items: int = 50
    max_str: int = 500
    sort_dict_keys: bool = True
    collapse_keys: set[str] = field(default_factory=set)
    redact_keys: set[str] = field(default_factory=set)


    def add_collapse_keys(self, *keys: str) -> None:
        """ Add keys to the set of keys to collapse. """
        self.collapse_keys.update(keys)

    def remove_collapse_key(self, key: str) -> None:
        """ Remove a key from the set of keys to collapse. """
        self.collapse_keys.discard(key)

    def clear_collapse_keys(self) -> None:
        """ Clear all keys from the set of keys to collapse. """
        self.collapse_keys.clear()

    def add_redact_keys(self, *keys: str) -> None:
        """ Add keys to the set of keys to redact. """
        self.redact_keys.update(keys)

    def remove_redact_key(self, key: str) -> None:
        """ Remove a key from the set of keys to redact. """
        self.redact_keys.discard(key)

    def clear_redact_keys(self) -> None:
        """ Clear all keys from the set of keys to redact. """
        self.redact_keys.clear()

    def deep_copy(self) -> TreeViewConfig:
        """ Return a deep copy of this configuration. """
        return TreeViewConfig(
            indent=self.indent,
            max_depth=self.max_depth,
            max_items=self.max_items,
            max_str=self.max_str,
            sort_dict_keys=self.sort_dict_keys,
            collapse_keys=set(self.collapse_keys),
            redact_keys=set(self.redact_keys),
        )


@dataclass(slots=True)
class TreeView:
    """ A class to traverse nested data structures generating an indented text output of the
        structure.
    """
    cfg: TreeViewConfig = field(default_factory=TreeViewConfig)

    @property
    def config_settings(self) -> dict[str, Any]:
        """ Get the current configuration settings as a dictionary.
            Returns:
                A dictionary containing the current configuration settings.
        """
        return {
            "indent": self.cfg.indent,
            "max_depth": self.cfg.max_depth,
            "max_items": self.cfg.max_items,
            "max_str": self.cfg.max_str,
            "sort_dict_keys": self.cfg.sort_dict_keys,
            "collapse_keys": set(self.cfg.collapse_keys),
            "redact_keys": set(self.cfg.redact_keys),
        }

    @config_settings.setter
    def config_settings(self, settings: Mapping[str, Any]) -> None:
        """ Sets the configuration settings from a dictionary.
            Args:
                settings: A dictionary containing the configuration settings to apply.
        """
        for key, value in settings.items():
            if hasattr(self.cfg, key):
                setattr(self.cfg, key, value)

    def add_collapse_keys(self, *keys: str) -> None:
        """ Add keys to the set of keys to collapse.
            Args:
                *keys: One or more keys to add to the collapse set.
        """
        self.cfg.add_collapse_keys(*keys)

    def remove_collapse_key(self, key: str) -> None:
        """ Remove a key from the set of keys to collapse.
            Args:
                key: The key to remove from the collapse set.
        """
        self.cfg.remove_collapse_key(key)

    def clear_collapse_keys(self) -> None:
        """ Clear all keys from the set of keys to collapse. """
        self.cfg.clear_collapse_keys()

    def add_redact_keys(self, *keys: str) -> None:
        """ Add keys to the set of keys to redact.
            Args:
                *keys: One or more keys to add to the redact set.
        """
        self.cfg.add_redact_keys(*keys)

    def remove_redact_key(self, key: str) -> None:
        """ Remove a key from the set of keys to redact.
            Args:
                key: The key to remove from the redact set.
        """
        self.cfg.remove_redact_key(key)

    def clear_redact_keys(self) -> None:
        """ Clear all keys from the set of keys to redact. """
        self.cfg.clear_redact_keys()

    def render_tree(
                self,
                obj: object,
                *,
                title: str | None = None,
                cfg: TreeViewConfig | None = None,
                **kwargs
            ) -> str:
        """ Return a string containing an indented tree view of nested dict/list structures.
            Args:
                obj: The object to render.
                title: An optional title to include at the top of the tree view.
                cfg: An optional TreeViewConfig to override the default configuration.
                **kwargs: Additional keyword arguments.
            Returns:
                A string containing an indented tree view of the object.
        """
        active_cfg = cfg.deep_copy() if cfg is not None else self.cfg.deep_copy()

        if kwargs:
            for key, value in kwargs.items():
                if not hasattr(active_cfg, key):
                    raise AttributeError(f"Unknown TreeViewConfig option: {key}")
                setattr(active_cfg, key, value)

        active: set[int] = set()
        lines: list[str] = []

        if title is not None:
            lines.append(f"{title}:\\n")

        def _short(v: object) -> str:
            """ Return a short string representation of a value, with newlines escaped and truncated 
                if necessary. 
                Args:
                    v: The value to represent as a string.
                Returns:
                    A short string representation of the value.
            """
            if isinstance(v, str):
                s = v.replace("\n", "\\n")
                return s if len(s) <= active_cfg.max_str else f"{s[: active_cfg.max_str - 1]} ..."
            try:
                r = repr(v)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                r = f"<repr failed: {type(exc).__name__}: {exc}>"
            return r if len(r) <= active_cfg.max_str else f"{r[: active_cfg.max_str - 1]} ..."

        def _is_seq(v: object) -> bool:
            """ Return True if v is a sequence type (like list or tuple) but not a string/bytes. 
                Args:
                    v: The value to check.
                Returns:
                    True if v is a sequence type, False otherwise.
            """
            return isinstance(v, Sequence) and not isinstance(v, (str, bytes, bytearray))

        def _collapsed_hint(v: object) -> str:
            """ Return a hint string for a collapsed value, indicating its type and size if
                possible.
                Args:
                    v: The value to generate a hint for.
                Returns:
                    A hint string for the collapsed value.
            """
            if isinstance(v, Mapping):
                return f"<collapsed dict keys={len(v)}>"
            if _is_seq(v):
                return f"<collapsed list items={len(v)}>"
            return "<collapsed>"

        def _coerce_to_walkable(v: object) -> object:
            """ Coerce various object types to something we can walk (Mapping or Sequence).
                Args:
                    v: The value to coerce.
                Returns:
                    The coerced value, which is either a Mapping, a Sequence, or the original
                    value if it cannot be coerced.
            """
            # Already walkable
            if isinstance(v, Mapping):
                return v
            if isinstance(v, Sequence) and not isinstance(v, (str, bytes, bytearray)):
                return v

            # Pydantic v2 models
            model_dump = getattr(v, "model_dump", None)
            if callable(model_dump):
                try:
                    return model_dump()
                except Exception:       # pylint: disable=broad-exception-caught
                    pass

            # Pydantic v1 models (or other .dict()-style)
            as_dict = getattr(v, "dict", None)
            if callable(as_dict):
                try:
                    return as_dict()
                except Exception:       # pylint: disable=broad-exception-caught
                    pass

            # dataclasses
            if is_dataclass(v):
                try:
                    return asdict(v)
                except Exception:       # pylint: disable=broad-exception-caught
                    pass

            # namedtuple-ish
            _asdict = getattr(v, "_asdict", None)
            if callable(_asdict):
                try:
                    return _asdict()
                except Exception:       # pylint: disable=broad-exception-caught
                    pass

            # plain objects with attributes (may fail for slots-only objects)
            try:
                return vars(v)
            except TypeError:
                return v


        def _walk(v: object, prefix: str, depth: int) -> None:
            """ Recursively walk the object and build lines for the tree representation.
                Args:
                    v: The value to walk.
                    prefix: The prefix for each line.
                    depth: The current depth of the recursion.
            """
            v = _coerce_to_walkable(v)

            if depth >= active_cfg.max_depth:
                lines.append(f"{prefix}<max_depth {active_cfg.max_depth} reached>")
                return

            if isinstance(v, Mapping):
                container_id = id(v)
                if container_id in active:
                    lines.append(f"{prefix}<cycle dict id={container_id}>")
                    return
                active.add(container_id)
                try:
                    child_prefix = prefix

                    keys = list(v.keys())
                    if active_cfg.sort_dict_keys:
                        try:
                            keys.sort()
                        except Exception:  # pylint: disable=broad-exception-caught
                            pass

                    shown = 0
                    for k in keys:
                        if shown >= active_cfg.max_items:
                            remaining = max(0, len(keys) - shown)
                            lines.append(f"{child_prefix}  <{remaining} more keys>")
                            break

                        key = str(k)

                        if key in active_cfg.redact_keys:
                            lines.append(f"{child_prefix}{key}: <redacted>")
                            shown += 1
                            continue

                        val = v[k]

                        if key in active_cfg.collapse_keys and (isinstance(val, Mapping) or
                                                                _is_seq(val)):
                            lines.append(f"{child_prefix}{key}: {_collapsed_hint(val)}")
                            shown += 1
                            continue

                        if isinstance(val, Mapping) or _is_seq(val):
                            lines.append(f"{child_prefix}{key}:")
                            _walk(val, child_prefix + " " * active_cfg.indent, depth + 1)
                        else:
                            lines.append(f"{child_prefix}{key}: {_short(val)}")
                        shown += 1
                    return
                finally:
                    active.remove(container_id)

            if _is_seq(v):
                container_id = id(v)
                if container_id in active:
                    lines.append(f"{prefix}<cycle seq id={container_id}>")
                    return
                active.add(container_id)
                try:
                    n = len(v)
                    limit = min(n, active_cfg.max_items)
                    for i in range(limit):
                        item = v[i]
                        if isinstance(item, Mapping) or _is_seq(item):
                            lines.append(f"{prefix}[{i}]:")
                            _walk(item, prefix + " " * active_cfg.indent, depth + 1)
                        else:
                            lines.append(f"{prefix}[{i}]: {_short(item)}")

                    if n > limit:
                        lines.append(f"{prefix}  <{n - limit} more items>")
                    return
                finally:
                    active.remove(container_id)

            lines.append(f"{prefix}{_short(v)}")
        # Get the tree structure info
        _walk(obj, "", 0)
        # Create the final string output
        return "\n".join(lines)
