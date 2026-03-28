# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Depth estimation from motion parallax.

Estimates depth from consecutive RGBA frames and known ego-motion,
without requiring a hardware depth sensor. Implements the same Transform
protocol as Panda3DDepthNormalize so it can be swapped in the pipeline.

Algorithm:
  1. Compute dense optical flow between consecutive RGBA frames using
     phase correlation (FFT-based, no external dependencies).
  2. Use known ego-motion (camera translation between frames) to
     convert flow magnitude to metric depth via motion parallax:
     depth ∝ (focal_length × baseline) / flow_magnitude
  3. Normalize to [0, 1) for objects, 1.0 for background.

When stereo camera data is available (two RGBA images per step),
triangulates depth from binocular disparity instead of motion.

Biological analogue:
  - Motion parallax: as the head moves, nearby objects shift more in the
    retinal image than distant ones. The brain uses this to perceive depth
    monocularly. This is exactly what our algorithm computes.
  - Stereo disparity: the brain fuses left/right retinal images to compute
    depth from binocular disparity. Our stereo mode mirrors this.
"""

from __future__ import annotations

import logging

import numpy as np

from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.environment_utils.transforms import TransformContext
from tbp.monty.frameworks.models.abstract_monty_classes import Observations

logger = logging.getLogger(__name__)


class DepthFromMotion:
    """Estimate depth from motion parallax or stereo disparity.

    Replaces or augments ground-truth depth in the observation pipeline.
    On the first frame, falls back to uniform depth (max_depth) since
    there is no previous frame for flow computation.

    Parameters
    ----------
    agent_id : AgentID
        Agent whose observations to transform.
    hfov : float
        Horizontal field of view in degrees.
    resolution : tuple
        (height, width) of the rendered image.
    max_depth : float
        Sentinel value for background / indeterminate pixels.
    min_flow : float
        Minimum flow magnitude (pixels) below which depth is set to
        max_depth (avoids division by near-zero).
    far_plane : float
        Far clip plane distance (meters). Estimated metric depths are
        normalized by this value so the output matches the [0, 1)
        convention used by Panda3DDepthNormalize / DepthTo3DLocations.
    baseline_scale : float
        Scale factor for the ego-motion baseline. Larger values increase
        depth sensitivity. Default 1.0 for metric units.
    use_ground_truth_blend : float
        If > 0, blend estimated depth with ground truth by this factor
        (for validation). 0.0 = pure estimation, 1.0 = pure ground truth.
    stereo_sensor_id : str or None
        If provided, use stereo disparity from this sensor instead of
        motion parallax. The second sensor's RGBA is the right eye.
    stereo_baseline : float
        Physical baseline (meters) between stereo cameras.
    """

    def __init__(
        self,
        agent_id: AgentID,
        hfov: float = 90.0,
        resolution: tuple = (64, 64),
        max_depth: float = 1.0,
        min_flow: float = 0.1,
        far_plane: float = 10.0,
        baseline_scale: float = 1.0,
        use_ground_truth_blend: float = 0.0,
        stereo_sensor_id: str = None,
        stereo_baseline: float = 0.065,  # ~human IPD
    ):
        self.agent_id = agent_id
        self.max_depth = max_depth
        self.min_flow = min_flow
        self.far_plane = far_plane
        self.baseline_scale = baseline_scale
        self.use_ground_truth_blend = use_ground_truth_blend
        self.stereo_sensor_id = stereo_sensor_id
        self.stereo_baseline = stereo_baseline

        h, w = resolution
        self._h = h
        self._w = w
        # Focal length in pixels from HFOV
        self._focal_px = (w / 2.0) / np.tan(np.radians(hfov) / 2.0)

        # Previous frame state (per sensor)
        self._prev_gray = {}  # sensor_id → (H, W) float32 grayscale
        self._prev_position = None  # (x, y, z) tuple
        self._prev_rotation = None  # numpy.quaternion
        self._step_count = 0

    def reset(self):
        """Reset between episodes."""
        self._prev_gray = {}
        self._prev_position = None
        self._prev_rotation = None
        self._step_count = 0

    def __call__(
        self, observations: Observations, ctx: TransformContext
    ) -> Observations:
        # Get ego-motion from proprioceptive state
        position = None
        rotation = None
        if ctx.state is not None and self.agent_id in ctx.state:
            agent_state = ctx.state[self.agent_id]
            position = agent_state.position
            rotation = agent_state.rotation

        for sensor_id in observations[self.agent_id]:
            if sensor_id == "view_finder":
                continue  # skip alias
            obs = observations[self.agent_id][sensor_id]
            if "rgba" not in obs:
                continue

            rgba = obs["rgba"]
            gray = self._to_grayscale(rgba)

            if self.stereo_sensor_id is not None:
                # Stereo mode: compute depth from disparity
                depth = self._stereo_depth(observations, sensor_id)
            elif sensor_id in self._prev_gray and self._prev_rotation is not None:
                # Motion parallax mode: use both translation and rotation
                motion = self._compute_ego_motion(position, rotation)
                flow_mag = self._compute_flow(self._prev_gray[sensor_id], gray)
                depth = self._flow_to_depth(flow_mag, motion)
            else:
                # First frame — no previous data for flow. Use ground-truth
                # depth if available (bootstraps the pipeline by providing
                # at least one valid observation), otherwise max_depth.
                if "depth" in obs:
                    depth = obs["depth"].copy()
                    if depth.ndim == 3 and depth.shape[2] == 1:
                        depth = depth[:, :, 0]
                    # Normalize raw Panda3D depth the same way
                    # Panda3DDepthNormalize would: scale by far plane
                    far_cutoff = 10.0 * 0.99  # far * threshold
                    depth = depth / far_cutoff
                    depth[depth >= 1.0] = self.max_depth
                    depth = depth.astype(np.float32)
                else:
                    depth = np.full(
                        (self._h, self._w), self.max_depth, dtype=np.float32
                    )

            # Optionally blend with ground truth for validation
            if self.use_ground_truth_blend > 0 and "depth" in obs:
                gt = obs["depth"]
                if gt.ndim == 3 and gt.shape[2] == 1:
                    gt = gt[:, :, 0]
                alpha = self.use_ground_truth_blend
                depth = alpha * gt.astype(np.float32) + (1 - alpha) * depth

            obs["depth"] = depth
            self._prev_gray[sensor_id] = gray

        self._prev_position = position
        self._prev_rotation = rotation
        self._step_count += 1
        return observations

    def _to_grayscale(self, rgba: np.ndarray) -> np.ndarray:
        """Convert RGBA (H, W, 4) uint8 to grayscale (H, W) float32."""
        # Standard luminance weights
        r = rgba[:, :, 0].astype(np.float32)
        g = rgba[:, :, 1].astype(np.float32)
        b = rgba[:, :, 2].astype(np.float32)
        return 0.299 * r + 0.587 * g + 0.114 * b

    def _compute_ego_motion(self, position, rotation) -> float:
        """Compute effective ego-motion magnitude between frames.

        Combines translational baseline and angular rotation into a single
        motion magnitude. For pure rotation (which is what the motor policy
        typically produces), this uses the rotation angle directly.

        Returns
        -------
        float
            Effective motion in image-space units (pixels of expected flow
            at unit depth).
        """
        motion = 0.0

        # Translational component
        if position is not None and self._prev_position is not None:
            dx = position[0] - self._prev_position[0]
            dy = position[1] - self._prev_position[1]
            dz = position[2] - self._prev_position[2]
            trans = np.sqrt(dx * dx + dy * dy + dz * dz)
            # Convert to pixel motion at unit depth
            motion += self._focal_px * trans * self.baseline_scale

        # Rotational component: pixel displacement = f × tan(θ)
        if rotation is not None and self._prev_rotation is not None:
            # Compute rotation angle between frames via quaternion
            q_curr = rotation
            q_prev = self._prev_rotation
            # Relative rotation: q_rel = q_curr * conj(q_prev)
            import quaternion as quat_lib
            q_rel = q_curr * q_prev.conjugate()
            # Rotation angle from quaternion: angle = 2 × arccos(|w|)
            w = float(q_rel.w)
            w = min(1.0, max(-1.0, w))  # clamp for numerical stability
            angle = 2.0 * np.arccos(abs(w))  # radians
            # Expected pixel displacement at unit depth
            if angle > 1e-6:
                motion += self._focal_px * np.tan(min(angle, np.pi / 4))

        return motion

    def _compute_flow(
        self, prev_gray: np.ndarray, curr_gray: np.ndarray
    ) -> np.ndarray:
        """Compute dense optical flow magnitude via phase correlation.

        Uses FFT-based phase correlation on image patches to estimate
        per-pixel displacement without external dependencies (no OpenCV).
        """
        h, w = prev_gray.shape
        # Block-based phase correlation for local flow estimation.
        # Smaller blocks give better spatial resolution for depth.
        block_size = max(4, min(h, w) // 8)
        flow_mag = np.zeros((h, w), dtype=np.float32)

        for y0 in range(0, h, block_size):
            for x0 in range(0, w, block_size):
                y1 = min(y0 + block_size, h)
                x1 = min(x0 + block_size, w)

                patch_prev = prev_gray[y0:y1, x0:x1]
                patch_curr = curr_gray[y0:y1, x0:x1]

                dy, dx = self._phase_correlate(patch_prev, patch_curr)
                mag = np.sqrt(dx * dx + dy * dy)
                flow_mag[y0:y1, x0:x1] = mag

        return flow_mag

    def _phase_correlate(
        self, patch_a: np.ndarray, patch_b: np.ndarray
    ) -> tuple:
        """Phase correlation between two patches.

        Returns (dy, dx) displacement in pixels.
        """
        if patch_a.shape != patch_b.shape or patch_a.size == 0:
            return (0.0, 0.0)

        # Apply Hanning window to reduce edge effects
        h, w = patch_a.shape
        if h < 2 or w < 2:
            return (0.0, 0.0)

        win_y = np.hanning(h)
        win_x = np.hanning(w)
        window = np.outer(win_y, win_x).astype(np.float32)

        a = patch_a * window
        b = patch_b * window

        # Cross-power spectrum
        fa = np.fft.fft2(a)
        fb = np.fft.fft2(b)
        cross = fa * np.conj(fb)
        denom = np.abs(cross)
        denom[denom < 1e-10] = 1e-10
        cross /= denom

        # Inverse FFT → correlation surface
        corr = np.fft.ifft2(cross).real

        # Find peak
        peak = np.unravel_index(np.argmax(corr), corr.shape)
        dy = float(peak[0])
        dx = float(peak[1])

        # Wrap around for negative displacements
        if dy > h / 2:
            dy -= h
        if dx > w / 2:
            dx -= w

        return (dy, dx)

    def _flow_to_depth(
        self, flow_mag: np.ndarray, motion: float
    ) -> np.ndarray:
        """Convert optical flow magnitude to depth via motion parallax.

        For a camera with focal length f, motion m (expected flow at unit
        depth), and observed flow F:
            metric_depth = m / F

        The metric depth is then normalized to [0, 1) by dividing by the
        far plane, matching the convention of Panda3DDepthNormalize where
        depth < 1.0 = on-object and depth >= 1.0 = background.

        Pixels with flow below min_flow are marked as background (max_depth).
        """
        depth = np.full_like(flow_mag, self.max_depth)

        if motion < 1e-8:
            return depth

        valid = flow_mag > self.min_flow
        # metric_depth = motion / flow, then normalize to [0, 1) range
        far_cutoff = self.far_plane * 0.99  # match Panda3DDepthNormalize
        depth[valid] = (motion / flow_mag[valid]) / far_cutoff

        # Anything beyond far plane is background
        depth[depth >= 1.0] = self.max_depth

        return depth

    def _stereo_depth(
        self, observations: Observations, primary_sensor_id: str
    ) -> np.ndarray:
        """Compute depth from stereo disparity.

        Uses the primary sensor as left eye and stereo_sensor_id as right eye.
        Disparity is computed via block-based phase correlation.
        """
        left = observations[self.agent_id][primary_sensor_id]
        right_obs = observations[self.agent_id].get(self.stereo_sensor_id)

        if right_obs is None or "rgba" not in right_obs:
            return np.full((self._h, self._w), self.max_depth, dtype=np.float32)

        left_gray = self._to_grayscale(left["rgba"])
        right_gray = self._to_grayscale(right_obs["rgba"])

        # Horizontal disparity via phase correlation
        disparity = self._compute_flow(right_gray, left_gray)

        depth = np.full_like(disparity, self.max_depth)
        valid = disparity > self.min_flow
        # metric depth = (focal_length × baseline) / disparity
        far_cutoff = self.far_plane * 0.99
        depth[valid] = (
            (self._focal_px * self.stereo_baseline) / disparity[valid]
        ) / far_cutoff
        depth[depth >= 1.0] = self.max_depth

        return depth
