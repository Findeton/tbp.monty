# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""YCB object resolver for Panda3D evaluation.

Maps YCB object names (e.g. ``"011_banana"``) to their ``.glb`` mesh paths
on disk. The 79 YCB objects are stored as textured glTF binary files that
Panda3D's ``AssetRegistry`` can load directly.

The YCB dataset ships two versions of each mesh:

- ``textured.glb`` — Basis Universal compressed textures via the
  ``GOOGLE_texture_basis`` glTF extension (used by Habitat).
- ``textured.glb.orig`` — standard PNG textures with a standard
  ``source`` field on the texture object.

``panda3d-gltf`` does not support the Basis extension, so this resolver
prefers the ``.orig`` variant when available.

Usage::

    from tbp.monty.simulators.panda3d.ycb import ycb_glb_path, YCB_EVAL_OBJECTS

    path = ycb_glb_path("011_banana")
    # -> ~/tbp/data/.../011_banana/google_16k/textured.glb.orig

    for name in YCB_EVAL_OBJECTS:
        print(name, ycb_glb_path(name))
"""

from __future__ import annotations

import os
from pathlib import Path

YCB_MESH_ROOT = (
    Path(os.environ.get("MONTY_DATA", "~/tbp/data")).expanduser()
    / "habitat"
    / "objects"
    / "ycb"
    / "meshes"
)

# 15 geometrically diverse objects for evaluation benchmarks.
YCB_EVAL_OBJECTS: list[str] = [
    # Kitchen
    "025_mug",
    "024_bowl",
    "029_plate",
    "030_fork",
    "032_knife",
    # Food
    "011_banana",
    "013_apple",
    "016_pear",
    # Tools
    "035_power_drill",
    "042_adjustable_wrench",
    "037_scissors",
    # Containers
    "003_cracker_box",
    "002_master_chef_can",
    # Other
    "056_tennis_ball",
    "061_foam_brick",
]


def ycb_glb_path(object_name: str) -> Path:
    """Resolve a YCB object name to its ``.glb`` mesh path.

    Prefers ``textured.glb.orig`` (standard PNG textures) over
    ``textured.glb`` (Basis Universal, unsupported by panda3d-gltf).

    Parameters
    ----------
    object_name : str
        YCB directory name, e.g. ``"011_banana"`` or ``"025_mug"``.

    Returns
    -------
    Path
        Absolute path to the mesh file.

    Raises
    ------
    FileNotFoundError
        If neither variant exists on disk.
    """
    mesh_dir = YCB_MESH_ROOT / object_name / "google_16k"

    # Prefer .orig (standard PNG textures compatible with panda3d-gltf)
    orig = mesh_dir / "textured.glb.orig"
    if orig.exists():
        return orig

    basis = mesh_dir / "textured.glb"
    if basis.exists():
        return basis

    raise FileNotFoundError(
        f"YCB mesh not found for '{object_name}' in {mesh_dir}\n"
        f"Ensure YCB data is at {YCB_MESH_ROOT} or set MONTY_DATA env var."
    )


def list_available_ycb() -> list[str]:
    """Return sorted names of all YCB objects whose .glb exists on disk."""
    if not YCB_MESH_ROOT.exists():
        return []
    return sorted(
        d.name
        for d in YCB_MESH_ROOT.iterdir()
        if d.is_dir()
        and (
            (d / "google_16k" / "textured.glb.orig").exists()
            or (d / "google_16k" / "textured.glb").exists()
        )
    )
