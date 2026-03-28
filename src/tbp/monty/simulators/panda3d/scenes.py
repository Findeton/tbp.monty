# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Multi-object scene definitions for Panda3D evaluation.

Defines structured scenes with multiple real YCB objects at realistic
positions, used by the temporal validation infrastructure (V2) to create
episodes where the agent navigates between objects.

Each scene is a :class:`SceneSpec` containing :class:`SceneObject` entries.
The scene builder loads all objects into a :class:`Panda3DSimulator` and
provides waypoints for the :class:`WaypointMotorPolicy` to navigate.

Usage::

    from tbp.monty.simulators.panda3d.scenes import KITCHEN_SCENE, load_scene

    sim = Panda3DSimulator(agents=[agent])
    load_scene(sim, KITCHEN_SCENE)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from tbp.monty.math import QuaternionWXYZ, VectorXYZ


@dataclass(frozen=True)
class SceneObject:
    """A single object placed in a scene.

    Attributes
    ----------
    ycb_name : str
        YCB object directory name (e.g. ``"011_banana"``).
    position : VectorXYZ
        World-space position ``(x, y, z)``.
    rotation : QuaternionWXYZ
        Orientation as ``(w, x, y, z)`` quaternion.
    """

    ycb_name: str
    position: VectorXYZ
    rotation: QuaternionWXYZ = (1.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class SceneSpec:
    """Specification for a multi-object scene.

    Attributes
    ----------
    name : str
        Human-readable scene name (e.g. ``"kitchen_counter"``).
    objects : tuple[SceneObject, ...]
        Objects in the scene.
    """

    name: str
    objects: tuple[SceneObject, ...]

    @property
    def object_names(self) -> list[str]:
        """Return list of YCB object names in this scene."""
        return [obj.ycb_name for obj in self.objects]

    @property
    def object_positions(self) -> list[VectorXYZ]:
        """Return list of object positions (useful as waypoints)."""
        return [obj.position for obj in self.objects]


# ---------------------------------------------------------------------------
# Predefined scenes using real YCB objects
# ---------------------------------------------------------------------------

KITCHEN_SCENE = SceneSpec(
    name="kitchen_counter",
    objects=(
        SceneObject("025_mug", (0.0, 0.0, 0.0)),
        SceneObject("030_fork", (0.3, 0.0, 0.0)),
        SceneObject("029_plate", (-0.3, 0.0, 0.0)),
        SceneObject("003_cracker_box", (0.0, 0.0, 0.3)),
    ),
)

WORKSHOP_SCENE = SceneSpec(
    name="workshop_bench",
    objects=(
        SceneObject("035_power_drill", (0.0, 0.0, 0.0)),
        SceneObject("042_adjustable_wrench", (0.3, 0.0, 0.0)),
        SceneObject("037_scissors", (-0.3, 0.0, 0.0)),
        SceneObject("048_hammer", (0.0, 0.0, 0.3)),
    ),
)

FRUIT_BOWL_SCENE = SceneSpec(
    name="fruit_bowl",
    objects=(
        SceneObject("011_banana", (0.0, 0.0, 0.0)),
        SceneObject("013_apple", (0.15, 0.0, 0.1)),
        SceneObject("016_pear", (-0.15, 0.0, 0.1)),
        SceneObject("024_bowl", (0.0, -0.05, 0.0)),
    ),
)

ALL_SCENES = {
    "kitchen_counter": KITCHEN_SCENE,
    "workshop_bench": WORKSHOP_SCENE,
    "fruit_bowl": FRUIT_BOWL_SCENE,
}


def load_scene(sim, scene: SceneSpec) -> list:
    """Load all objects from a SceneSpec into a Panda3D simulator.

    Parameters
    ----------
    sim : Panda3DSimulator
        The simulator to add objects to. Existing objects are removed first.
    scene : SceneSpec
        Scene specification with objects to load.

    Returns
    -------
    list[ObjectInfo]
        ObjectInfo for each loaded object, in scene order.
    """
    from tbp.monty.simulators.panda3d.ycb import ycb_glb_path

    sim.remove_all_objects()

    infos = []
    for obj in scene.objects:
        glb_path = str(ycb_glb_path(obj.ycb_name))
        info = sim.add_object(
            name=glb_path,
            position=obj.position,
            rotation=obj.rotation,
        )
        infos.append(info)

    return infos


def swap_object(sim, scene: SceneSpec, index: int, new_ycb_name: str) -> SceneSpec:
    """Create a modified scene with one object swapped out.

    Returns a new SceneSpec with the object at ``index`` replaced by
    ``new_ycb_name`` at the same position and rotation. Does NOT reload
    the simulator — call :func:`load_scene` with the returned spec.

    Parameters
    ----------
    sim : Panda3DSimulator
        Unused (kept for API symmetry); the caller reloads.
    scene : SceneSpec
        Original scene.
    index : int
        Index of the object to replace.
    new_ycb_name : str
        YCB name for the replacement object.

    Returns
    -------
    SceneSpec
        New scene with the swapped object.
    """
    objects = list(scene.objects)
    old = objects[index]
    objects[index] = SceneObject(
        ycb_name=new_ycb_name,
        position=old.position,
        rotation=old.rotation,
    )
    return SceneSpec(
        name=f"{scene.name}_swap_{index}_{new_ycb_name}",
        objects=tuple(objects),
    )
