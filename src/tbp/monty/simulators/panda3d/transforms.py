# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Panda3D-specific observation transforms.

Bridges the gap between Panda3D's depth output conventions and what the
existing ``DepthTo3DLocations`` transform expects.

Panda3D produces depth as linearized metric distance with background pixels
at the far-plane value (e.g., 10.0). The standard pipeline expects background
at ``max_depth=1.0`` (set by ``MissingToMaxDepth`` for Habitat, which returns
0 for missing depth). This transform normalizes Panda3D depth so the
downstream ``DepthTo3DLocations`` semantic fallback (``depth >= 1.0`` → off-object)
works correctly.
"""

from __future__ import annotations

import numpy as np

from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.environment_utils.transforms import TransformContext
from tbp.monty.frameworks.models.abstract_monty_classes import Observations


class Panda3DDepthNormalize:
    """Normalize Panda3D depth for the DepthTo3DLocations pipeline.

    1. Squeezes depth from (H, W, 1) to (H, W) if needed.
    2. Scales depth so that objects within ``object_range`` map to [near, 1.0)
       and background (at far plane) maps to ``max_depth`` (≥ 1.0).

    This ensures ``DepthTo3DLocations``'s fallback semantic mask
    (``depth >= 1.0`` → off-object) works correctly regardless of the
    scene's actual metric depth range.

    Parameters
    ----------
    agent_id : AgentID
        Agent whose observations to transform.
    near : float
        Simulator near clip plane.
    far : float
        Simulator far clip plane.
    max_depth : float
        Sentinel value for background pixels (must be >= 1.0).
    far_threshold : float
        Fraction of ``far``; pixels with depth >= ``far * far_threshold``
        are treated as background.
    """

    def __init__(
        self,
        agent_id: AgentID,
        near: float = 0.01,
        far: float = 10.0,
        max_depth: float = 1.0,
        far_threshold: float = 0.99,
    ):
        self.agent_id = agent_id
        self.near = near
        self.far = far
        self.max_depth = max_depth
        self.far_threshold = far_threshold

    def __call__(
        self, observations: Observations, _ctx: TransformContext
    ) -> Observations:
        far_cutoff = self.far * self.far_threshold

        for sensor_id in observations[self.agent_id]:
            obs = observations[self.agent_id][sensor_id]
            depth = obs["depth"]

            # Squeeze (H, W, 1) → (H, W)
            if depth.ndim == 3 and depth.shape[2] == 1:
                depth = depth[:, :, 0]

            # Scale: map [near, far_cutoff] → [near/far_cutoff, 1.0)
            # Background (>= far_cutoff) → max_depth
            normalized = depth / far_cutoff
            normalized[depth >= far_cutoff] = self.max_depth

            obs["depth"] = normalized.astype(np.float32)

        return observations
