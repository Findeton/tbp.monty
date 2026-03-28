# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Waypoint navigation motor policy for multi-object scene exploration.

Produces a camera pose sequence that visits multiple waypoints (typically
object positions in a scene) with a structured explore-transit-explore
rhythm:

1. Orbit around waypoint[0] for ``dwell_steps`` (exploring one object)
2. Interpolate camera to waypoint[1] over ``transit_steps``
3. Orbit around waypoint[1] for ``dwell_steps``
4. ...

This gives the agent temporal structure that TM can learn: surprise spikes
at object transitions, drops during steady exploration of a single object.

Usage::

    from tbp.monty.simulators.panda3d.waypoint_policy import WaypointMotorPolicy
    from tbp.monty.simulators.panda3d.scenes import KITCHEN_SCENE

    policy = WaypointMotorPolicy(
        waypoints=KITCHEN_SCENE.object_positions,
        dwell_steps=20,
        transit_steps=5,
    )

    for step in range(policy.total_steps):
        position, rotation = policy.get_camera_pose(step)
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from tbp.monty.frameworks.agents import AgentID
from tbp.monty.math import VectorXYZ


def _look_at(eye: tuple, target: tuple) -> tuple:
    """Compute a quaternion rotation looking from ``eye`` toward ``target``.

    Returns (w, x, y, z) quaternion compatible with Panda3D.
    """
    from scipy.spatial.transform import Rotation

    eye = np.array(eye, dtype=np.float64)
    target = np.array(target, dtype=np.float64)
    forward = target - eye
    norm = np.linalg.norm(forward)
    if norm < 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    forward /= norm

    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(forward, world_up)) > 0.99:
        world_up = np.array([0.0, 1.0, 0.0])

    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right) + 1e-12
    up = np.cross(right, forward)

    # Panda3D convention: Y is forward, Z is up, X is right
    mat = np.eye(3)
    mat[:, 0] = right
    mat[:, 1] = forward
    mat[:, 2] = up

    r = Rotation.from_matrix(mat)
    q = r.as_quat()  # scipy: (x, y, z, w)
    return (float(q[3]), float(q[0]), float(q[1]), float(q[2]))


class WaypointMotorPolicy:
    """Navigates the camera through a sequence of waypoints with dwell orbits.

    Each waypoint has a ``dwell_steps``-long orbital exploration phase where
    the camera circles the waypoint at ``orbit_radius``. Between waypoints,
    the camera smoothly interpolates over ``transit_steps``.

    Parameters
    ----------
    waypoints : list[VectorXYZ]
        Positions to visit in order. Typically object positions from a scene.
    agent_id : AgentID
        Agent to control.
    dwell_steps : int
        Steps to orbit around each waypoint.
    transit_steps : int
        Steps to interpolate between consecutive waypoints.
    orbit_radius : float
        Camera distance from the waypoint during orbital exploration.
    elevation_deg : float
        Camera elevation angle during orbits (degrees above horizontal).
    """

    def __init__(
        self,
        waypoints: Sequence[VectorXYZ],
        agent_id: AgentID = AgentID("eval_agent"),
        dwell_steps: int = 20,
        transit_steps: int = 5,
        orbit_radius: float = 0.3,
        elevation_deg: float = 15.0,
    ):
        if len(waypoints) < 1:
            raise ValueError("WaypointMotorPolicy requires at least one waypoint")

        self._waypoints = [tuple(w) for w in waypoints]
        self.agent_id = agent_id
        self._dwell_steps = dwell_steps
        self._transit_steps = transit_steps
        self._orbit_radius = orbit_radius
        self._elevation_rad = math.radians(elevation_deg)
        self._step = 0

        # Precompute segment boundaries
        self._segments = self._build_segments()

    def _build_segments(self):
        """Build a list of (type, start_step, end_step, data) segments."""
        segments = []
        step = 0
        n = len(self._waypoints)

        for i in range(n):
            # Dwell (orbit) at waypoint i
            segments.append(
                ("dwell", step, step + self._dwell_steps, i)
            )
            step += self._dwell_steps

            # Transit to next waypoint (if not last)
            if i < n - 1:
                segments.append(
                    ("transit", step, step + self._transit_steps, (i, i + 1))
                )
                step += self._transit_steps

        return segments

    @property
    def total_steps(self) -> int:
        """Total number of steps for one complete traversal."""
        if not self._segments:
            return 0
        return self._segments[-1][2]  # end of last segment

    @property
    def n_waypoints(self) -> int:
        return len(self._waypoints)

    @property
    def waypoint_indices(self) -> list[int]:
        """Return the step index where each waypoint's dwell phase begins."""
        return [
            seg[1] for seg in self._segments if seg[0] == "dwell"
        ]

    def get_segment_info(self, step: int) -> dict:
        """Return info about what segment the given step falls into.

        Returns dict with:
        - ``type``: ``"dwell"`` or ``"transit"``
        - ``waypoint_index``: current waypoint index (for dwell) or
          ``(from, to)`` tuple (for transit)
        - ``local_step``: step within this segment
        """
        for seg_type, start, end, data in self._segments:
            if start <= step < end:
                return {
                    "type": seg_type,
                    "waypoint_index": data,
                    "local_step": step - start,
                }
        # Past end — clamp to last waypoint
        return {
            "type": "dwell",
            "waypoint_index": len(self._waypoints) - 1,
            "local_step": 0,
        }

    def get_camera_pose(self, step: int = None):
        """Compute camera position and rotation for the given step.

        During dwell phases, the camera orbits the waypoint.
        During transit phases, the camera interpolates linearly between
        the end position of the previous dwell and the start position
        of the next dwell.

        Returns
        -------
        tuple[VectorXYZ, QuaternionWXYZ]
            Camera position and look-at-waypoint rotation.
        """
        if step is None:
            step = self._step
            self._step += 1

        info = self.get_segment_info(step)

        if info["type"] == "dwell":
            wp_idx = info["waypoint_index"]
            local = info["local_step"]
            return self._dwell_pose(wp_idx, local)
        else:
            from_idx, to_idx = info["waypoint_index"]
            local = info["local_step"]
            t = local / max(self._transit_steps - 1, 1)
            return self._transit_pose(from_idx, to_idx, t)

    def _orbit_position(self, center: tuple, azimuth_step: int) -> tuple:
        """Compute an orbital camera position around a center point."""
        azimuth_rad = 2 * math.pi * azimuth_step / max(self._dwell_steps, 1)
        r = self._orbit_radius
        el = self._elevation_rad

        dx = r * math.cos(el) * math.sin(azimuth_rad)
        dy = -r * math.cos(el) * math.cos(azimuth_rad)
        dz = r * math.sin(el)

        return (
            center[0] + dx,
            center[1] + dy,
            center[2] + dz,
        )

    def _dwell_pose(self, wp_idx: int, local_step: int):
        """Camera pose during dwell (orbit) around a waypoint."""
        center = self._waypoints[wp_idx]
        pos = self._orbit_position(center, local_step)
        rot = _look_at(pos, center)
        return pos, rot

    def _transit_pose(self, from_idx: int, to_idx: int, t: float):
        """Camera pose during transit between waypoints.

        Interpolates position linearly from the last dwell orbit position
        to the first orbit position at the next waypoint.
        """
        # End of dwell at from_idx = last orbit position
        from_center = self._waypoints[from_idx]
        from_pos = self._orbit_position(from_center, self._dwell_steps - 1)

        # Start of dwell at to_idx = first orbit position
        to_center = self._waypoints[to_idx]
        to_pos = self._orbit_position(to_center, 0)

        # Linear interpolation
        pos = tuple(
            from_pos[i] + t * (to_pos[i] - from_pos[i]) for i in range(3)
        )

        # Look toward the target waypoint
        look_target = tuple(
            from_center[i] + t * (to_center[i] - from_center[i]) for i in range(3)
        )
        rot = _look_at(pos, look_target)
        return pos, rot

    def reset(self):
        """Reset the internal step counter."""
        self._step = 0
