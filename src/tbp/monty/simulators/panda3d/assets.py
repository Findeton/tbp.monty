# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""glTF/GLB asset loading for Panda3D simulator.

Loads 3D models from glTF 2.0 (.gltf) and GLB (.glb) files using
the panda3d-gltf plugin. Supports caching to avoid redundant parsing.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional

from panda3d.core import NodePath

logger = logging.getLogger(__name__)


class AssetRegistry:
    """Loads and caches glTF/GLB 3D models.

    Models are loaded via ``panda3d-gltf`` and cached by resolved path
    so repeated ``add_object`` calls with the same model reuse the parsed
    scene graph (each instance gets its own ``copyTo`` copy).

    Parameters
    ----------
    search_paths : list[str | Path] or None
        Directories to search when a relative path is given.
        If None, only absolute paths and cwd-relative paths work.
    skip_animations : bool
        If True, skip loading animations from glTF files (faster load).
    """

    SUPPORTED_EXTENSIONS = {".gltf", ".glb"}

    def __init__(
        self,
        search_paths: list | None = None,
        skip_animations: bool = False,
    ):
        self._search_paths = [Path(p) for p in (search_paths or [])]
        self._skip_animations = skip_animations
        self._cache: Dict[str, NodePath] = {}

    def can_load(self, name: str) -> bool:
        """Return True if ``name`` looks like a glTF/GLB file path.

        Handles compound extensions like ``.glb.orig`` (YCB dataset stores
        original GLB files with standard PNG textures under this name).
        """
        suffixes = Path(name).suffixes
        return any(s.lower() in self.SUPPORTED_EXTENSIONS for s in suffixes)

    def resolve_path(self, name: str) -> Optional[Path]:
        """Find the file on disk, searching search_paths if needed.

        Returns None if the file is not found.
        """
        p = Path(name)
        if p.is_absolute() and p.exists():
            return p

        # Try relative to cwd
        if p.exists():
            return p.resolve()

        # Try search paths
        for sp in self._search_paths:
            candidate = sp / name
            if candidate.exists():
                return candidate.resolve()

        return None

    def load(self, name: str, instance_name: Optional[str] = None) -> NodePath:
        """Load a glTF/GLB model and return a new NodePath instance.

        The parsed model is cached by resolved path. Each call returns
        an independent copy via ``copyTo`` so transforms on one instance
        don't affect others.

        Parameters
        ----------
        name : str
            Path to .gltf or .glb file (absolute, relative, or searched).
        instance_name : str or None
            Name for the returned NodePath. Defaults to the file stem.

        Returns
        -------
        NodePath
            A new node containing the loaded model geometry.

        Raises
        ------
        FileNotFoundError
            If the file cannot be found.
        ValueError
            If panda3d-gltf is not installed or loading fails.
        """
        resolved = self.resolve_path(name)
        if resolved is None:
            raise FileNotFoundError(
                f"Model file not found: '{name}'. "
                f"Search paths: {[str(p) for p in self._search_paths]}"
            )

        cache_key = str(resolved)

        if cache_key not in self._cache:
            self._cache[cache_key] = self._load_gltf(resolved)
            logger.info("Loaded glTF model: %s", resolved)

        # Create independent copy
        template = self._cache[cache_key]
        if instance_name is None:
            instance_name = resolved.stem
        copy = NodePath(instance_name)
        template.copyTo(copy)
        return copy

    def _load_gltf(self, path: Path) -> NodePath:
        """Parse a glTF/GLB file via panda3d-gltf."""
        try:
            from gltf import GltfSettings, load_model
        except ImportError:
            raise ValueError(
                "panda3d-gltf is required for glTF model loading. "
                "Install with: pip install panda3d-gltf"
            )

        settings = GltfSettings(
            skip_animations=self._skip_animations,
            no_srgb=True,  # We handle color space ourselves
        )

        try:
            model_root = load_model(str(path), gltf_settings=settings)
        except Exception as e:
            raise ValueError(f"Failed to load glTF model '{path}': {e}") from e

        return NodePath(model_root)

    def clear_cache(self) -> None:
        """Remove all cached models."""
        self._cache.clear()

    @property
    def cached_paths(self) -> list[str]:
        """Return list of currently cached model paths."""
        return list(self._cache.keys())
