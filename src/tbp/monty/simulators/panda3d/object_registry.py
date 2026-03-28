# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Panda3D object registry for mapping logical names to object specifications.

Bridges the gap between EnvironmentInterfacePerObject (which cycles through
object names like "sphere", "cube") and Panda3DSimulator (which needs to know
whether to generate a primitive or load a glTF file).

Usage::

    registry = Panda3DObjectRegistry({
        "sphere": {"primitive": "sphere"},
        "red_cube": {"primitive": "cube", "color": (0.9, 0.2, 0.2, 1.0)},
        "fox": {"model_path": "/path/to/Fox.glb", "animated": True},
    })

    spec = registry.get_spec("sphere")
    # -> {"name": "sphere", "animated": False}

    spec = registry.get_spec("fox")
    # -> {"name": "/path/to/Fox.glb", "animated": True}
"""

from __future__ import annotations

from typing import Any


class Panda3DObjectRegistry:
    """Maps logical object names to Panda3D add_object() specifications.

    Each entry translates a human-readable name (used by
    EnvironmentInterfacePerObject) into the kwargs that
    Panda3DSimulator.add_object() expects.

    Parameters
    ----------
    entries : dict[str, dict] or None
        Mapping of logical name to spec dict. Each spec can contain:
        - ``primitive``: str — name of a primitive shape ("sphere", "cube",
          "cone", "cylinder"). Passed as the ``name`` arg to add_object.
        - ``model_path``: str — path to a glTF/GLB file. Used as ``name``.
        - ``animated``: bool — whether to load as animated Actor (default False).
        - ``color``: tuple — RGBA color for primitives (not used by add_object
          directly, but stored for reference).
        - ``scale``: tuple — default scale override.

        Either ``primitive`` or ``model_path`` must be provided, not both.
    """

    def __init__(self, entries: dict[str, dict] | None = None):
        self._entries: dict[str, dict] = {}
        if entries:
            for name, spec in entries.items():
                self.register(name, spec)

    def register(self, name: str, spec: dict) -> None:
        """Register a logical object name with its specification."""
        if "primitive" not in spec and "model_path" not in spec:
            raise ValueError(
                f"Object spec for '{name}' must contain 'primitive' or "
                f"'model_path'. Got: {spec}"
            )
        self._entries[name] = dict(spec)

    def get_spec(self, name: str) -> dict[str, Any]:
        """Return add_object() kwargs for a logical name.

        Returns a dict with at least ``name`` (the string to pass to
        Panda3DSimulator.add_object) and ``animated`` (bool).

        Raises KeyError if the name is not registered.
        """
        if name not in self._entries:
            raise KeyError(
                f"Object '{name}' not in registry. "
                f"Available: {list(self._entries.keys())}"
            )
        spec = self._entries[name]

        result = {"animated": spec.get("animated", False)}

        if "primitive" in spec:
            result["name"] = spec["primitive"]
        else:
            result["name"] = spec["model_path"]

        if "scale" in spec:
            result["scale"] = spec["scale"]

        return result

    def has(self, name: str) -> bool:
        """Check if a name is registered."""
        return name in self._entries

    def list_objects(self) -> list[str]:
        """Return all registered object names."""
        return list(self._entries.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)
