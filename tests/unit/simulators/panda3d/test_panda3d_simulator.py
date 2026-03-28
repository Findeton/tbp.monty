# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Tests for the Panda3D simulator.

Validates:
1. Offscreen rendering produces valid RGBA + depth buffers
2. Primitive shapes render correctly
3. Depth linearization produces correct metric distances
4. Actions move the camera
5. Observations match the format expected by DepthTo3DLocations
6. Objects can be added and removed
"""

import unittest

import numpy as np

from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.sensors import SensorID

try:
    import panda3d  # noqa: F401
except ImportError:
    raise unittest.SkipTest("Panda3D not installed")

from tbp.monty.simulators.panda3d.agents import Panda3DAgent
from tbp.monty.simulators.panda3d.simulator import Panda3DSimulator


AGENT_ID = AgentID("test_cam")
SENSOR_ID = "sensor_0"


def _make_simulator(
    position=(0.0, 0.0, 0.0),
    rotation=(1.0, 0.0, 0.0, 0.0),
    resolution=(64, 64),
    fov=90.0,
    near=0.01,
    far=10.0,
):
    """Create a minimal Panda3D simulator with one agent."""
    agent = Panda3DAgent(
        agent_id=AGENT_ID,
        sensor_id=SENSOR_ID,
        position=position,
        rotation=rotation,
        resolution=resolution,
        fov=fov,
    )
    return Panda3DSimulator(agents=[agent], near=near, far=far)


class TestPanda3DInitialization(unittest.TestCase):
    """Simulator creation and teardown."""

    def test_create_and_close(self):
        """Simulator can be created and closed without errors."""
        sim = _make_simulator()
        sim.close()

    def test_empty_scene_observations(self):
        """Empty scene produces valid observation shapes."""
        sim = _make_simulator()
        try:
            obs, states = sim.step([])

            self.assertIn(AGENT_ID, obs)
            self.assertIn(SensorID(SENSOR_ID), obs[AGENT_ID])

            sensor_obs = obs[AGENT_ID][SensorID(SENSOR_ID)]
            self.assertIn("rgba", sensor_obs)
            self.assertIn("depth", sensor_obs)

            self.assertEqual(sensor_obs["rgba"].shape, (64, 64, 4))
            self.assertEqual(sensor_obs["depth"].shape, (64, 64, 1))
            self.assertEqual(sensor_obs["rgba"].dtype, np.uint8)
            self.assertEqual(sensor_obs["depth"].dtype, np.float32)
        finally:
            sim.close()

    def test_proprioceptive_state(self):
        """Proprioceptive state contains agent position and rotation."""
        sim = _make_simulator(position=(1.0, 2.0, 3.0))
        try:
            _, states = sim.step([])

            self.assertIn(AGENT_ID, states)
            agent_state = states[AGENT_ID]

            # Position should match initialization
            pos = agent_state.position
            self.assertAlmostEqual(pos[0], 1.0, places=2)
            self.assertAlmostEqual(pos[1], 2.0, places=2)
            self.assertAlmostEqual(pos[2], 3.0, places=2)

            # Should have sensor state
            self.assertIn(SensorID(SENSOR_ID), agent_state.sensors)
        finally:
            sim.close()


class TestPanda3DObjectManagement(unittest.TestCase):
    """Adding and removing objects."""

    def test_add_sphere(self):
        """Can add a sphere primitive."""
        sim = _make_simulator()
        try:
            info = sim.add_object("sphere", position=(0, 5, 0))
            self.assertIsNotNone(info.object_id)
            self.assertIsNotNone(info.semantic_id)
        finally:
            sim.close()

    def test_add_multiple_primitives(self):
        """Can add multiple different primitive shapes."""
        sim = _make_simulator()
        try:
            info1 = sim.add_object("sphere", position=(0, 5, 0))
            info2 = sim.add_object("cube", position=(2, 5, 0))
            info3 = sim.add_object("cone", position=(-2, 5, 0))

            # Each object gets a unique ID
            self.assertNotEqual(info1.object_id, info2.object_id)
            self.assertNotEqual(info2.object_id, info3.object_id)
        finally:
            sim.close()

    def test_remove_all_objects(self):
        """remove_all_objects clears the scene."""
        sim = _make_simulator()
        try:
            sim.add_object("sphere", position=(0, 3, 0))
            sim.add_object("cube", position=(2, 3, 0))
            sim.remove_all_objects()

            # After removal, scene should be empty (no objects)
            self.assertEqual(len(sim._objects), 0)
        finally:
            sim.close()

    def test_unknown_object_raises(self):
        """Adding an unknown object name raises ValueError."""
        sim = _make_simulator()
        try:
            with self.assertRaises((ValueError, Exception)):
                sim.add_object("nonexistent_shape_xyz")
        finally:
            sim.close()


class TestPanda3DRendering(unittest.TestCase):
    """Rendering produces correct RGBA and depth."""

    def test_sphere_renders_nonzero_rgba(self):
        """A sphere in front of the camera produces non-zero RGBA pixels."""
        sim = _make_simulator()
        try:
            sim.add_object("sphere", position=(0, 3, 0), scale=(0.5, 0.5, 0.5))
            obs, _ = sim.step([])

            rgba = obs[AGENT_ID][SensorID(SENSOR_ID)]["rgba"]
            # At least some pixels should be non-black
            nonzero = np.any(rgba[:, :, :3] > 0, axis=-1).sum()
            self.assertGreater(nonzero, 10, "Sphere should render visible pixels")
        finally:
            sim.close()

    def test_sphere_depth_is_valid(self):
        """Sphere produces depth values less than the far plane."""
        sim = _make_simulator(near=0.01, far=10.0)
        try:
            sim.add_object("sphere", position=(0, 3, 0), scale=(0.5, 0.5, 0.5))
            obs, _ = sim.step([])

            depth = obs[AGENT_ID][SensorID(SENSOR_ID)]["depth"]
            center_depth = depth[32, 32, 0]

            # Center pixel should see the sphere (depth < far)
            self.assertLess(
                center_depth, 10.0,
                f"Center depth {center_depth:.2f} should be < far plane"
            )
            # Depth should be approximately 3.0 - 0.5 = 2.5 (distance - radius)
            self.assertGreater(center_depth, 1.0, "Sphere should be in front")
            self.assertLess(center_depth, 5.0, "Sphere shouldn't be too far")
        finally:
            sim.close()

    def test_depth_increases_with_distance(self):
        """Objects further away produce larger depth values."""
        sim = _make_simulator(near=0.01, far=20.0)
        try:
            # Near sphere
            sim.add_object("sphere", position=(0, 2, 0), scale=(0.3, 0.3, 0.3))
            obs_near, _ = sim.step([])
            depth_near = obs_near[AGENT_ID][SensorID(SENSOR_ID)]["depth"][32, 32, 0]

            sim.remove_all_objects()

            # Far sphere
            sim.add_object("sphere", position=(0, 8, 0), scale=(0.3, 0.3, 0.3))
            obs_far, _ = sim.step([])
            depth_far = obs_far[AGENT_ID][SensorID(SENSOR_ID)]["depth"][32, 32, 0]

            self.assertGreater(
                depth_far, depth_near,
                f"Far depth {depth_far:.2f} should > near depth {depth_near:.2f}"
            )
        finally:
            sim.close()

    def test_empty_scene_depth_is_far(self):
        """Empty scene depth should be at or near the far plane."""
        sim = _make_simulator(near=0.01, far=10.0)
        try:
            obs, _ = sim.step([])
            depth = obs[AGENT_ID][SensorID(SENSOR_ID)]["depth"]
            # All pixels should be at the far plane
            self.assertTrue(
                np.all(depth >= 9.0),
                f"Empty scene depth range [{depth.min():.2f}, {depth.max():.2f}] "
                f"should be near far plane (10.0)"
            )
        finally:
            sim.close()

    def test_cube_renders(self):
        """Cube primitive renders visible pixels."""
        sim = _make_simulator()
        try:
            sim.add_object("cube", position=(0, 3, 0), scale=(0.5, 0.5, 0.5))
            obs, _ = sim.step([])

            rgba = obs[AGENT_ID][SensorID(SENSOR_ID)]["rgba"]
            nonzero = np.any(rgba[:, :, :3] > 0, axis=-1).sum()
            self.assertGreater(nonzero, 10)
        finally:
            sim.close()


class TestPanda3DActions(unittest.TestCase):
    """Camera movement via actions."""

    def test_move_forward_changes_depth(self):
        """Moving forward reduces depth to the object."""
        sim = _make_simulator(position=(0, 0, 0))
        try:
            sim.add_object("sphere", position=(0, 10, 0), scale=(1, 1, 1))

            obs1, _ = sim.step([])
            depth1 = obs1[AGENT_ID][SensorID(SENSOR_ID)]["depth"][32, 32, 0]

            # Move forward
            from tbp.monty.frameworks.actions.actions import MoveForward
            action = MoveForward(agent_id=AGENT_ID, distance=3.0)
            obs2, _ = sim.step([action])
            depth2 = obs2[AGENT_ID][SensorID(SENSOR_ID)]["depth"][32, 32, 0]

            self.assertLess(
                depth2, depth1,
                f"After moving forward, depth {depth2:.2f} should < {depth1:.2f}"
            )
        finally:
            sim.close()


class TestPanda3DReset(unittest.TestCase):
    """Reset functionality."""

    def test_reset_returns_observations(self):
        """reset() returns valid observations."""
        sim = _make_simulator()
        try:
            sim.add_object("sphere", position=(0, 3, 0))
            obs, states = sim.reset()

            self.assertIn(AGENT_ID, obs)
            self.assertIn(SensorID(SENSOR_ID), obs[AGENT_ID])
            self.assertIn(AGENT_ID, states)
        finally:
            sim.close()


class TestPanda3DObservationFormat(unittest.TestCase):
    """Verify observation format matches what DepthTo3DLocations expects."""

    def test_rgba_dtype_and_range(self):
        """RGBA is uint8 in [0, 255]."""
        sim = _make_simulator()
        try:
            sim.add_object("sphere", position=(0, 3, 0))
            obs, _ = sim.step([])
            rgba = obs[AGENT_ID][SensorID(SENSOR_ID)]["rgba"]
            self.assertEqual(rgba.dtype, np.uint8)
            self.assertGreaterEqual(rgba.min(), 0)
            self.assertLessEqual(rgba.max(), 255)
        finally:
            sim.close()

    def test_depth_dtype_and_shape(self):
        """Depth is float32 with shape (H, W, 1)."""
        sim = _make_simulator(resolution=(32, 32))
        try:
            sim.add_object("sphere", position=(0, 3, 0))
            obs, _ = sim.step([])
            depth = obs[AGENT_ID][SensorID(SENSOR_ID)]["depth"]
            self.assertEqual(depth.dtype, np.float32)
            self.assertEqual(depth.shape, (32, 32, 1))
        finally:
            sim.close()

    def test_depth_positive(self):
        """All depth values should be positive."""
        sim = _make_simulator()
        try:
            sim.add_object("sphere", position=(0, 3, 0))
            obs, _ = sim.step([])
            depth = obs[AGENT_ID][SensorID(SENSOR_ID)]["depth"]
            self.assertTrue(np.all(depth > 0), "All depth values should be > 0")
        finally:
            sim.close()


if __name__ == "__main__":
    unittest.main()
