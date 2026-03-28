# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Panda3D environment implementing SimulatedObjectEnvironment protocol.

Wraps Panda3DSimulator to provide the standard environment interface
that EnvironmentInterface expects. Supports an optional
:class:`Panda3DObjectRegistry` for resolving logical object names
(e.g. "sphere", "red_cube") to Panda3D primitives or glTF models.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from tbp.monty.frameworks.actions.actions import Action
from tbp.monty.frameworks.environments.environment import (
    ObjectID,
    SemanticID,
    SimulatedObjectEnvironment,
)
from tbp.monty.frameworks.models.abstract_monty_classes import Observations
from tbp.monty.frameworks.models.motor_system_state import ProprioceptiveState
from tbp.monty.math import QuaternionWXYZ, VectorXYZ
from tbp.monty.simulators.panda3d.agents import Panda3DAgent
from tbp.monty.simulators.panda3d.object_registry import Panda3DObjectRegistry
from tbp.monty.simulators.panda3d.simulator import Panda3DSimulator

logger = logging.getLogger(__name__)


class Panda3DEnvironment(SimulatedObjectEnvironment):
    """Environment backed by Panda3D for GPU-accelerated 3D rendering.

    Implements the SimulatedObjectEnvironment protocol so it can be used
    with EnvironmentInterface and the full Monty experiment pipeline.

    Parameters
    ----------
    agents : list[Panda3DAgent] or Panda3DAgent
        Agent configuration(s).
    objects : list[dict] or None
        Initial objects to add (each dict passed to add_object).
    near : float
        Near clip plane distance.
    far : float
        Far clip plane distance.
    object_registry : Panda3DObjectRegistry or dict or None
        Optional registry mapping logical object names to Panda3D specs.
        When provided, ``add_object(name=...)`` will resolve the name
        through the registry before passing to the simulator. A plain
        dict is automatically wrapped.
    """

    def __init__(
        self,
        agents,
        objects=None,
        near: float = 0.01,
        far: float = 10.0,
        asset_search_paths: list | None = None,
        object_registry: Panda3DObjectRegistry | dict | None = None,
    ):
        if isinstance(agents, Panda3DAgent):
            agents = [agents]
        elif isinstance(agents, dict):
            agents = [Panda3DAgent(**agents)]
        elif isinstance(agents, list):
            agents = [
                a if isinstance(a, Panda3DAgent) else Panda3DAgent(**a)
                for a in agents
            ]

        self._sim = Panda3DSimulator(
            agents=agents, near=near, far=far,
            asset_search_paths=asset_search_paths,
        )

        # Object registry for name resolution
        if isinstance(object_registry, dict):
            self._registry = Panda3DObjectRegistry(object_registry)
        else:
            self._registry = object_registry

        if objects:
            for obj in objects:
                if isinstance(obj, dict):
                    self._sim.add_object(**obj)
                else:
                    self._sim.add_object(name=obj)

    @property
    def _agents(self):
        """Expose simulator agents for EnvironmentInterface compatibility."""
        return self._sim._agents

    def add_object(
        self,
        name: str,
        position: VectorXYZ = (0.0, 0.0, 0.0),
        rotation: QuaternionWXYZ = (1.0, 0.0, 0.0, 0.0),
        scale: VectorXYZ = (1.0, 1.0, 1.0),
        semantic_id: SemanticID | None = None,
        primary_target_object: ObjectID | None = None,
        animated: bool = False,
    ) -> ObjectID:
        # Resolve logical name through registry if available
        if self._registry is not None and self._registry.has(name):
            spec = self._registry.get_spec(name)
            name = spec["name"]
            animated = spec.get("animated", animated)
            if "scale" in spec:
                scale = spec["scale"]

        # Convert numpy quaternion to plain tuple (from object_init_sampler)
        rotation = self._normalize_quaternion(rotation)
        position = tuple(float(x) for x in position)
        scale = tuple(float(x) for x in scale)

        info = self._sim.add_object(
            name=name,
            position=position,
            rotation=rotation,
            scale=scale,
            semantic_id=semantic_id,
            primary_target_object=primary_target_object,
            animated=animated,
        )
        return info.object_id

    @staticmethod
    def _normalize_quaternion(q) -> tuple:
        """Convert various quaternion formats to a plain (w,x,y,z) tuple."""
        if isinstance(q, np.ndarray):
            return tuple(float(x) for x in q)
        if hasattr(q, "w") and hasattr(q, "x"):
            # numpy-quaternion object
            return (float(q.w), float(q.x), float(q.y), float(q.z))
        return tuple(float(x) for x in q)

    def get_animated_object(self, obj_id: ObjectID):
        """Get the AnimatedObject for a given object ID."""
        return self._sim.get_animated_object(obj_id)

    def pose_object(self, obj_id: ObjectID, frame: int, anim_name=None):
        """Pose an animated object at a specific frame."""
        self._sim.pose_object(obj_id, frame, anim_name)

    def remove_all_objects(self) -> None:
        self._sim.remove_all_objects()

    def step(
        self, actions: Sequence[Action]
    ) -> tuple[Observations, ProprioceptiveState]:
        return self._sim.step(actions)

    def reset(self) -> tuple[Observations, ProprioceptiveState]:
        return self._sim.reset()

    def close(self) -> None:
        if self._sim is not None:
            self._sim.close()
            self._sim = None
