# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Tests for ChangeDetectingSM (Phase 5 of object behaviors)."""

import unittest

import numpy as np

from tbp.monty.frameworks.models.change_detecting_sm import ChangeDetectingSM


def _make_observation(points_3d, rgba=None):
    """Create a minimal observation dict with semantic_3d.

    Args:
        points_3d: (N, 3) array of 3D points.
        rgba: Optional (H, W, 4) RGBA image.
    """
    semantic_ids = np.ones(len(points_3d))
    semantic_3d = np.column_stack([points_3d, semantic_ids])
    obs = {"semantic_3d": semantic_3d}
    if rgba is not None:
        obs["rgba"] = rgba
    return obs


def _make_points_around(center, n=20, spread=0.02, seed=42):
    """Generate random 3D points around a center."""
    rng = np.random.RandomState(seed)
    return center + rng.randn(n, 3) * spread


def _make_curved_patch(size=8):
    """Generate a square on-object patch with mild curvature."""
    xs = np.linspace(-0.05, 0.05, size)
    ys = np.linspace(-0.05, 0.05, size)
    points = []
    for y in ys:
        for x in xs:
            z = 0.5 + 0.2 * (x**2 + 0.5 * y**2)
            points.append([x, y, z])
    return np.asarray(points, dtype=float)


class TestChangeDetectingSMStatic(unittest.TestCase):
    """Test with static (non-changing) observations."""

    def test_first_observation_no_change(self):
        """First observation should always return use_state=False."""
        sm = ChangeDetectingSM(sensor_module_id="change_SM_0")
        points = _make_points_around(np.array([0.0, 0.0, 0.5]))
        obs = _make_observation(points)
        state = sm.step(None, obs)
        self.assertFalse(state.use_state)

    def test_identical_observations_no_change(self):
        """Quiet frames should emit a low-confidence context state."""
        sm = ChangeDetectingSM(
            sensor_module_id="change_SM_0",
            flow_threshold=0.01,
        )
        points = _make_points_around(np.array([0.0, 0.0, 0.5]))
        obs = _make_observation(points)
        sm.step(None, obs)
        state = sm.step(None, obs)
        self.assertTrue(state.use_state)
        self.assertLess(state.confidence, 0.3)
        self.assertIn("flow_magnitude", state.non_morphological_features)

    def test_very_small_movement_no_change(self):
        """Sub-threshold movement should still emit usable context."""
        sm = ChangeDetectingSM(
            sensor_module_id="change_SM_0",
            flow_threshold=0.01,
        )
        center1 = np.array([0.0, 0.0, 0.5])
        center2 = np.array([0.001, 0.0, 0.5])  # 1mm movement
        obs1 = _make_observation(_make_points_around(center1, seed=1))
        obs2 = _make_observation(_make_points_around(center2, seed=1))
        sm.step(None, obs1)
        state = sm.step(None, obs2)
        self.assertTrue(state.use_state)
        self.assertLess(state.confidence, 0.3)


class TestChangeDetectingSMMotion(unittest.TestCase):
    """Test with moving objects."""

    def test_object_moves_detects_change(self):
        """Object moves significantly → change detected."""
        sm = ChangeDetectingSM(
            sensor_module_id="change_SM_0",
            flow_threshold=0.01,
            global_flow_suppression=False,
        )
        center1 = np.array([0.0, 0.0, 0.5])
        center2 = np.array([0.05, 0.0, 0.5])  # 5cm movement
        obs1 = _make_observation(_make_points_around(center1, seed=1))
        obs2 = _make_observation(_make_points_around(center2, seed=1))
        sm.step(None, obs1)
        state = sm.step(None, obs2)
        self.assertTrue(state.use_state)

    def test_flow_direction_correct(self):
        """Flow direction should match the movement direction."""
        sm = ChangeDetectingSM(
            sensor_module_id="change_SM_0",
            flow_threshold=0.005,
            global_flow_suppression=False,
        )
        center1 = np.array([0.0, 0.0, 0.5])
        center2 = np.array([0.1, 0.0, 0.5])  # Move along +X
        obs1 = _make_observation(_make_points_around(center1, seed=1))
        obs2 = _make_observation(_make_points_around(center2, seed=1))
        sm.step(None, obs1)
        state = sm.step(None, obs2)
        self.assertTrue(state.use_state)

        flow_dir = state.non_morphological_features["flow_direction"]
        # Flow should be primarily in +X direction
        self.assertGreater(flow_dir[0], 0.5)
        self.assertAlmostEqual(abs(flow_dir[1]), 0.0, places=1)
        self.assertAlmostEqual(abs(flow_dir[2]), 0.0, places=1)

    def test_flow_magnitude_reported(self):
        """Flow magnitude should be in the output features."""
        sm = ChangeDetectingSM(
            sensor_module_id="change_SM_0",
            flow_threshold=0.005,
            global_flow_suppression=False,
        )
        center1 = np.array([0.0, 0.0, 0.5])
        center2 = np.array([0.0, 0.05, 0.5])  # 5cm in Y
        obs1 = _make_observation(_make_points_around(center1, seed=1))
        obs2 = _make_observation(_make_points_around(center2, seed=1))
        sm.step(None, obs1)
        state = sm.step(None, obs2)

        mag = state.non_morphological_features["flow_magnitude"]
        self.assertGreater(mag[0], 0.01)

    def test_pose_vectors_from_flow(self):
        """Pose vectors should be orthonormal based on flow direction."""
        sm = ChangeDetectingSM(
            sensor_module_id="change_SM_0",
            flow_threshold=0.005,
            global_flow_suppression=False,
        )
        center1 = np.array([0.0, 0.0, 0.5])
        center2 = np.array([0.1, 0.0, 0.5])
        obs1 = _make_observation(_make_points_around(center1, seed=1))
        obs2 = _make_observation(_make_points_around(center2, seed=1))
        sm.step(None, obs1)
        state = sm.step(None, obs2)

        pv = state.morphological_features["pose_vectors"]
        self.assertEqual(pv.shape, (3, 3))
        # Check orthonormality
        product = pv @ pv.T
        np.testing.assert_allclose(product, np.eye(3), atol=1e-6)


class TestChangeDetectingSMFeatureChange(unittest.TestCase):
    """Test feature-based change detection (color changes without motion)."""

    def test_color_change_detected(self):
        """Color change above threshold → change detected."""
        sm = ChangeDetectingSM(
            sensor_module_id="change_SM_0",
            flow_threshold=0.01,
            feature_change_thresholds={"rgba": 20.0},
            global_flow_suppression=False,
        )
        points = _make_points_around(np.array([0.0, 0.0, 0.5]), seed=1)

        # Red image
        rgba1 = np.full((64, 64, 4), [255, 0, 0, 255], dtype=np.uint8)
        obs1 = _make_observation(points, rgba=rgba1)

        # Green image (large color change)
        rgba2 = np.full((64, 64, 4), [0, 255, 0, 255], dtype=np.uint8)
        obs2 = _make_observation(points, rgba=rgba2)

        sm.step(None, obs1)
        state = sm.step(None, obs2)
        self.assertTrue(state.use_state)

    def test_color_change_below_threshold(self):
        """Small color change below threshold still emits context features."""
        sm = ChangeDetectingSM(
            sensor_module_id="change_SM_0",
            flow_threshold=0.01,
            feature_change_thresholds={"rgba": 100.0},
            global_flow_suppression=False,
        )
        points = _make_points_around(np.array([0.0, 0.0, 0.5]), seed=1)
        rgba1 = np.full((64, 64, 4), [128, 128, 128, 255], dtype=np.uint8)
        rgba2 = np.full((64, 64, 4), [130, 130, 130, 255], dtype=np.uint8)
        obs1 = _make_observation(points, rgba=rgba1)
        obs2 = _make_observation(points, rgba=rgba2)

        sm.step(None, obs1)
        state = sm.step(None, obs2)
        self.assertTrue(state.use_state)
        self.assertLess(state.confidence, 0.3)
        self.assertIn("hsv", state.non_morphological_features)
        self.assertIn("delta_rgba", state.non_morphological_features)
        self.assertIn("delta_hsv", state.non_morphological_features)


class TestChangeDetectingSMReset(unittest.TestCase):
    """Test episode reset behavior."""

    def test_pre_episode_resets(self):
        """pre_episode should clear internal state."""
        sm = ChangeDetectingSM(sensor_module_id="change_SM_0")
        points = _make_points_around(np.array([0.0, 0.0, 0.5]))
        obs = _make_observation(points)
        sm.step(None, obs)
        sm.pre_episode()
        # After reset, first observation should return no change
        state = sm.step(None, obs)
        self.assertFalse(state.use_state)

    def test_motor_only_step(self):
        """motor_only_step should return no-change state."""
        sm = ChangeDetectingSM(sensor_module_id="change_SM_0")
        points = _make_points_around(np.array([0.0, 0.0, 0.5]))
        obs = _make_observation(points)
        state = sm.step(None, obs, motor_only_step=True)
        self.assertFalse(state.use_state)


class TestChangeDetectingSMStateFormat(unittest.TestCase):
    """Test that output States conform to CMP format."""

    def test_no_change_state_format(self):
        """No-change state should have correct CMP fields."""
        sm = ChangeDetectingSM(sensor_module_id="change_SM_0")
        state = sm.step(None, {"semantic_3d": np.zeros((0, 4))})
        self.assertFalse(state.use_state)
        self.assertEqual(state.sender_id, "change_SM_0")
        self.assertEqual(state.sender_type, "SM")

    def test_change_state_format(self):
        """Change state should have correct CMP fields."""
        sm = ChangeDetectingSM(
            sensor_module_id="change_SM_0",
            flow_threshold=0.001,
            global_flow_suppression=False,
        )
        points1 = _make_points_around(np.array([0.0, 0.0, 0.5]), seed=1)
        points2 = _make_points_around(np.array([0.1, 0.0, 0.5]), seed=1)
        sm.step(None, _make_observation(points1))
        state = sm.step(None, _make_observation(points2))

        self.assertTrue(state.use_state)
        self.assertEqual(state.sender_type, "SM")
        self.assertEqual(state.location.shape, (3,))
        self.assertIn("pose_vectors", state.morphological_features)
        self.assertIn("flow_direction", state.non_morphological_features)
        self.assertIn("flow_magnitude", state.non_morphological_features)
        self.assertGreaterEqual(state.confidence, 0.0)
        self.assertLessEqual(state.confidence, 1.0)

    def test_quiet_state_carries_absolute_hsv_context(self):
        """Quiet frames should still carry absolute appearance context."""
        sm = ChangeDetectingSM(
            sensor_module_id="change_SM_0",
            flow_threshold=0.01,
        )
        points = _make_points_around(np.array([0.0, 0.0, 0.5]), seed=1)
        rgba = np.full((16, 16, 4), [255, 0, 0, 255], dtype=np.uint8)
        obs = _make_observation(points, rgba=rgba)

        sm.step(None, obs)
        state = sm.step(None, obs)

        self.assertTrue(state.use_state)
        self.assertEqual(state.sender_type, "SM")
        self.assertEqual(state.location.shape, (3,))
        self.assertIn("hsv", state.non_morphological_features)
        self.assertIn("flow_direction", state.non_morphological_features)
        self.assertIn("flow_magnitude", state.non_morphological_features)
        self.assertLess(state.confidence, 0.3)

    def test_quiet_state_carries_curvature_context_when_patch_available(self):
        """Square semantic patches should contribute absolute curvature context."""
        sm = ChangeDetectingSM(
            sensor_module_id="change_SM_0",
            flow_threshold=0.01,
        )
        rgba = np.full((8, 8, 4), [180, 180, 180, 255], dtype=np.uint8)
        obs = _make_observation(_make_curved_patch(), rgba=rgba)

        sm.step(None, obs)
        state = sm.step(None, obs)

        curvature = state.non_morphological_features.get("principal_curvatures_log")
        self.assertIsNotNone(curvature)
        self.assertEqual(curvature.shape, (2,))
        self.assertTrue(np.isfinite(curvature).all())


if __name__ == "__main__":
    unittest.main()
