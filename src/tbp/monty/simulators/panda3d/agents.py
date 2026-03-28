# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Panda3D agent configuration.

Defines agent types for the Panda3D simulator. Each agent wraps a camera
with RGBD sensor capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from tbp.monty.frameworks.agents import AgentID
from tbp.monty.math import QuaternionWXYZ, VectorXYZ


@dataclass
class Panda3DAgent:
    """Agent with a single RGBD camera sensor.

    Parameters
    ----------
    agent_id : AgentID
        Unique identifier for this agent.
    sensor_id : str
        Identifier for the camera sensor.
    position : VectorXYZ
        Initial camera position in world coordinates.
    rotation : QuaternionWXYZ
        Initial camera rotation as (w, x, y, z) quaternion.
    resolution : tuple
        Render resolution (width, height).
    fov : float
        Field of view in degrees.
    """

    agent_id: AgentID
    sensor_id: str = "sensor_id_0"
    position: VectorXYZ = (0.0, 1.5, 0.0)
    rotation: QuaternionWXYZ = (1.0, 0.0, 0.0, 0.0)
    resolution: Tuple[int, int] = (64, 64)
    fov: float = 90.0
    action_space_type: str = "distant_agent"
