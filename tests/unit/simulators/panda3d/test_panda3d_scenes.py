# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Tests for V2 multi-object scene infrastructure.

Validates:
1. Scene definitions (SceneSpec, SceneObject) and predefined scenes
2. WaypointMotorPolicy geometry and segment structure
3. Temporal scene evaluation (Panda3D integration tests)
"""

import math
import unittest

import numpy as np
import pytest


pytestmark = pytest.mark.xdist_group(name="panda3d")


# ---------------------------------------------------------------------------
# V2.1: Scene Builder tests (no Panda3D dependency)
# ---------------------------------------------------------------------------

class TestSceneSpec(unittest.TestCase):
    """Tests for scenes.py — SceneSpec and SceneObject dataclasses."""

    def test_scene_object_defaults(self):
        """SceneObject has identity rotation by default."""
        from tbp.monty.simulators.panda3d.scenes import SceneObject

        obj = SceneObject("011_banana", (0.0, 0.0, 0.0))
        self.assertEqual(obj.rotation, (1.0, 0.0, 0.0, 0.0))

    def test_scene_object_frozen(self):
        """SceneObject is immutable."""
        from tbp.monty.simulators.panda3d.scenes import SceneObject

        obj = SceneObject("011_banana", (0.0, 0.0, 0.0))
        with self.assertRaises(AttributeError):
            obj.ycb_name = "025_mug"

    def test_scene_spec_object_names(self):
        """SceneSpec.object_names returns YCB names in order."""
        from tbp.monty.simulators.panda3d.scenes import KITCHEN_SCENE

        names = KITCHEN_SCENE.object_names
        self.assertEqual(names, ["025_mug", "030_fork", "029_plate", "003_cracker_box"])

    def test_scene_spec_object_positions(self):
        """SceneSpec.object_positions returns positions in order."""
        from tbp.monty.simulators.panda3d.scenes import KITCHEN_SCENE

        positions = KITCHEN_SCENE.object_positions
        self.assertEqual(len(positions), 4)
        self.assertEqual(positions[0], (0.0, 0.0, 0.0))

    def test_kitchen_scene_has_4_objects(self):
        from tbp.monty.simulators.panda3d.scenes import KITCHEN_SCENE

        self.assertEqual(len(KITCHEN_SCENE.objects), 4)

    def test_workshop_scene_has_4_objects(self):
        from tbp.monty.simulators.panda3d.scenes import WORKSHOP_SCENE

        self.assertEqual(len(WORKSHOP_SCENE.objects), 4)

    def test_fruit_bowl_scene_has_4_objects(self):
        from tbp.monty.simulators.panda3d.scenes import FRUIT_BOWL_SCENE

        self.assertEqual(len(FRUIT_BOWL_SCENE.objects), 4)

    def test_all_scenes_dict(self):
        from tbp.monty.simulators.panda3d.scenes import ALL_SCENES

        self.assertIn("kitchen_counter", ALL_SCENES)
        self.assertIn("workshop_bench", ALL_SCENES)
        self.assertIn("fruit_bowl", ALL_SCENES)
        self.assertEqual(len(ALL_SCENES), 3)

    def test_scene_objects_are_valid_ycb(self):
        """All scene objects should be resolvable YCB names."""
        from tbp.monty.simulators.panda3d.scenes import ALL_SCENES
        from tbp.monty.simulators.panda3d.ycb import ycb_glb_path

        for scene_name, scene in ALL_SCENES.items():
            for obj in scene.objects:
                path = ycb_glb_path(obj.ycb_name)
                self.assertTrue(
                    path.exists(),
                    f"Scene '{scene_name}' object '{obj.ycb_name}' not found at {path}",
                )

    def test_swap_object_creates_new_spec(self):
        """swap_object returns a new SceneSpec with the replaced object."""
        from tbp.monty.simulators.panda3d.scenes import KITCHEN_SCENE, swap_object

        modified = swap_object(None, KITCHEN_SCENE, 0, "011_banana")
        self.assertNotEqual(modified.name, KITCHEN_SCENE.name)
        self.assertEqual(modified.objects[0].ycb_name, "011_banana")
        # Other objects unchanged
        self.assertEqual(modified.objects[1].ycb_name, "030_fork")
        # Same position
        self.assertEqual(modified.objects[0].position, KITCHEN_SCENE.objects[0].position)

    def test_swap_preserves_length(self):
        from tbp.monty.simulators.panda3d.scenes import KITCHEN_SCENE, swap_object

        modified = swap_object(None, KITCHEN_SCENE, 2, "056_tennis_ball")
        self.assertEqual(len(modified.objects), len(KITCHEN_SCENE.objects))


# ---------------------------------------------------------------------------
# V2.2: WaypointMotorPolicy tests (no Panda3D dependency)
# ---------------------------------------------------------------------------

class TestWaypointMotorPolicy(unittest.TestCase):
    """Tests for waypoint_policy.py — camera navigation through waypoints."""

    def _make_policy(self, n_waypoints=3, dwell=10, transit=5):
        from tbp.monty.simulators.panda3d.waypoint_policy import WaypointMotorPolicy

        waypoints = [(i * 0.3, 0.0, 0.0) for i in range(n_waypoints)]
        return WaypointMotorPolicy(
            waypoints=waypoints,
            dwell_steps=dwell,
            transit_steps=transit,
            orbit_radius=0.3,
        )

    def test_total_steps(self):
        """Total steps = n * dwell + (n-1) * transit."""
        policy = self._make_policy(n_waypoints=3, dwell=10, transit=5)
        # 3*10 + 2*5 = 40
        self.assertEqual(policy.total_steps, 40)

    def test_total_steps_single_waypoint(self):
        """Single waypoint = dwell steps only, no transit."""
        policy = self._make_policy(n_waypoints=1, dwell=20, transit=5)
        self.assertEqual(policy.total_steps, 20)

    def test_n_waypoints(self):
        policy = self._make_policy(n_waypoints=4)
        self.assertEqual(policy.n_waypoints, 4)

    def test_waypoint_indices(self):
        """waypoint_indices returns step where each dwell begins."""
        policy = self._make_policy(n_waypoints=3, dwell=10, transit=5)
        indices = policy.waypoint_indices
        self.assertEqual(indices, [0, 15, 30])  # 0, 10+5, 10+5+10+5

    def test_segment_info_dwell(self):
        """Steps within dwell range return dwell segment info."""
        policy = self._make_policy(n_waypoints=2, dwell=10, transit=5)
        info = policy.get_segment_info(3)
        self.assertEqual(info["type"], "dwell")
        self.assertEqual(info["waypoint_index"], 0)
        self.assertEqual(info["local_step"], 3)

    def test_segment_info_transit(self):
        """Steps within transit range return transit segment info."""
        policy = self._make_policy(n_waypoints=2, dwell=10, transit=5)
        info = policy.get_segment_info(12)  # step 12 is in transit (10-14)
        self.assertEqual(info["type"], "transit")
        self.assertEqual(info["waypoint_index"], (0, 1))
        self.assertEqual(info["local_step"], 2)

    def test_segment_info_second_dwell(self):
        """Steps in the second dwell phase are attributed correctly."""
        policy = self._make_policy(n_waypoints=2, dwell=10, transit=5)
        info = policy.get_segment_info(17)  # step 17 = second dwell step 2
        self.assertEqual(info["type"], "dwell")
        self.assertEqual(info["waypoint_index"], 1)
        self.assertEqual(info["local_step"], 2)

    def test_camera_pose_returns_tuple(self):
        """get_camera_pose returns (position, rotation) tuples."""
        policy = self._make_policy()
        pos, rot = policy.get_camera_pose(0)
        self.assertEqual(len(pos), 3)
        self.assertEqual(len(rot), 4)

    def test_dwell_orbits_around_waypoint(self):
        """During dwell, camera orbits around the waypoint center."""
        policy = self._make_policy(n_waypoints=1, dwell=20)
        center = (0.0, 0.0, 0.0)
        for step in range(20):
            pos, _ = policy.get_camera_pose(step)
            dist = math.sqrt(sum((p - c) ** 2 for p, c in zip(pos, center)))
            self.assertAlmostEqual(dist, 0.3, places=2,
                                   msg=f"Step {step}: dist={dist}")

    def test_different_waypoints_produce_different_positions(self):
        """Camera positions at different waypoints are in different regions."""
        policy = self._make_policy(n_waypoints=2, dwell=10, transit=5)
        # First step of dwell 0
        pos0, _ = policy.get_camera_pose(0)
        # First step of dwell 1 (step 15)
        pos1, _ = policy.get_camera_pose(15)
        # Waypoints are 0.3 apart in x, so camera positions should differ
        dx = abs(pos1[0] - pos0[0])
        self.assertGreater(dx, 0.1, "Camera should be near different waypoints")

    def test_transit_interpolates(self):
        """During transit, camera position is between the two waypoints."""
        policy = self._make_policy(n_waypoints=2, dwell=10, transit=5)
        # Midpoint of transit (step 12)
        pos_mid, _ = policy.get_camera_pose(12)
        # Should be roughly between the two orbit regions
        # Waypoint 0 at x=0, waypoint 1 at x=0.3
        self.assertGreater(pos_mid[0], -0.1)
        self.assertLess(pos_mid[0], 0.6)

    def test_reset_resets_step_counter(self):
        """After reset, get_camera_pose starts from step 0."""
        policy = self._make_policy()
        policy.get_camera_pose()  # auto-increment to 1
        policy.get_camera_pose()  # auto-increment to 2
        policy.reset()
        pos_after_reset, _ = policy.get_camera_pose()  # should be step 0
        pos_step_0, _ = policy.get_camera_pose(0)
        # They should be the same (both step 0)
        np.testing.assert_allclose(pos_after_reset, pos_step_0, atol=1e-6)

    def test_empty_waypoints_raises(self):
        """WaypointMotorPolicy requires at least one waypoint."""
        from tbp.monty.simulators.panda3d.waypoint_policy import WaypointMotorPolicy

        with self.assertRaises(ValueError):
            WaypointMotorPolicy(waypoints=[])

    def test_rotation_is_valid_quaternion(self):
        """Returned rotation quaternion has unit norm."""
        policy = self._make_policy()
        for step in range(policy.total_steps):
            _, rot = policy.get_camera_pose(step)
            norm = math.sqrt(sum(r ** 2 for r in rot))
            self.assertAlmostEqual(norm, 1.0, places=3,
                                   msg=f"Step {step}: quaternion norm={norm}")


# ---------------------------------------------------------------------------
# V2.3: Temporal evaluation integration tests (require Panda3D + YCB data)
# ---------------------------------------------------------------------------

def _panda3d_available():
    try:
        import panda3d  # noqa: F401
        import gltf  # noqa: F401
        return True
    except ImportError:
        return False


def _ycb_data_available():
    from tbp.monty.simulators.panda3d.ycb import YCB_MESH_ROOT
    return (YCB_MESH_ROOT / "011_banana" / "google_16k" / "textured.glb").exists()


@unittest.skipUnless(_panda3d_available(), "Panda3D not installed")
@unittest.skipUnless(_ycb_data_available(), "YCB data not available")
class TestSceneLoading(unittest.TestCase):
    """Integration tests for loading multi-object scenes into Panda3D."""

    _sim = None

    @classmethod
    def setUpClass(cls):
        from tbp.monty.frameworks.agents import AgentID
        from tbp.monty.simulators.panda3d.agents import Panda3DAgent
        from tbp.monty.simulators.panda3d.simulator import Panda3DSimulator

        agent = Panda3DAgent(
            agent_id=AgentID("test_agent"),
            sensor_id="patch",
            resolution=(64, 64),
            fov=90.0,
        )
        cls._sim = Panda3DSimulator(agents=[agent])

    @classmethod
    def tearDownClass(cls):
        if cls._sim is not None:
            cls._sim.close()

    def test_load_kitchen_scene(self):
        """KITCHEN_SCENE loads all 4 objects without error."""
        from tbp.monty.simulators.panda3d.scenes import KITCHEN_SCENE, load_scene

        infos = load_scene(self._sim, KITCHEN_SCENE)
        self.assertEqual(len(infos), 4)

    def test_load_workshop_scene(self):
        """WORKSHOP_SCENE loads all 4 objects without error."""
        from tbp.monty.simulators.panda3d.scenes import WORKSHOP_SCENE, load_scene

        infos = load_scene(self._sim, WORKSHOP_SCENE)
        self.assertEqual(len(infos), 4)

    def test_load_fruit_bowl_scene(self):
        """FRUIT_BOWL_SCENE loads all 4 objects without error."""
        from tbp.monty.simulators.panda3d.scenes import FRUIT_BOWL_SCENE, load_scene

        infos = load_scene(self._sim, FRUIT_BOWL_SCENE)
        self.assertEqual(len(infos), 4)

    def test_load_scene_clears_previous(self):
        """Loading a scene clears previously loaded objects."""
        from tbp.monty.simulators.panda3d.scenes import (
            KITCHEN_SCENE,
            FRUIT_BOWL_SCENE,
            load_scene,
        )

        load_scene(self._sim, KITCHEN_SCENE)
        load_scene(self._sim, FRUIT_BOWL_SCENE)
        # Should only have fruit bowl objects now (4)
        self.assertEqual(len(self._sim._objects), 4)

    def test_scene_renders_non_empty(self):
        """A loaded scene produces non-empty RGBA observations."""
        from tbp.monty.simulators.panda3d.scenes import FRUIT_BOWL_SCENE, load_scene

        load_scene(self._sim, FRUIT_BOWL_SCENE)
        obs, _ = self._sim.step([])
        agent_id = list(obs.keys())[0]
        sensor_id = list(obs[agent_id].keys())[0]
        rgba = obs[agent_id][sensor_id]["rgba"]
        # At least some non-zero pixels (objects visible)
        self.assertGreater(np.count_nonzero(rgba[:, :, :3]), 0)


@unittest.skipUnless(_panda3d_available(), "Panda3D not installed")
@unittest.skipUnless(_ycb_data_available(), "YCB data not available")
class TestWaypointNavigation(unittest.TestCase):
    """Integration: WaypointMotorPolicy navigating a loaded scene."""

    _sim = None

    @classmethod
    def setUpClass(cls):
        from tbp.monty.frameworks.agents import AgentID
        from tbp.monty.simulators.panda3d.agents import Panda3DAgent
        from tbp.monty.simulators.panda3d.scenes import FRUIT_BOWL_SCENE, load_scene
        from tbp.monty.simulators.panda3d.simulator import Panda3DSimulator
        from tbp.monty.simulators.panda3d.waypoint_policy import WaypointMotorPolicy

        agent_id = AgentID("nav_agent")
        agent = Panda3DAgent(
            agent_id=agent_id,
            sensor_id="patch",
            resolution=(64, 64),
            fov=90.0,
        )
        cls._sim = Panda3DSimulator(agents=[agent])
        cls._agent_id = agent_id
        load_scene(cls._sim, FRUIT_BOWL_SCENE)

        cls._policy = WaypointMotorPolicy(
            waypoints=FRUIT_BOWL_SCENE.object_positions,
            agent_id=agent_id,
            dwell_steps=10,
            transit_steps=3,
            orbit_radius=0.3,
        )

    @classmethod
    def tearDownClass(cls):
        if cls._sim is not None:
            cls._sim.close()

    def test_navigation_renders_at_each_step(self):
        """Each navigation step produces renderable observations."""
        self._policy.reset()
        rendered_steps = 0
        for step in range(min(self._policy.total_steps, 20)):
            pos, rot = self._policy.get_camera_pose(step)
            cam_info = self._sim._agent_buffers[self._agent_id]
            self._sim._set_node_pose(cam_info["camera_np"], pos, rot)
            obs, _ = self._sim.step([])
            agent_obs = obs[self._agent_id]
            sensor_obs = list(agent_obs.values())[0]
            if np.count_nonzero(sensor_obs["rgba"][:, :, :3]) > 0:
                rendered_steps += 1
        self.assertGreater(rendered_steps, 0, "Expected at least some steps to render objects")

    def test_different_waypoints_see_different_content(self):
        """Observations at different waypoints should differ."""
        self._policy.reset()

        # First step at waypoint 0
        pos0, rot0 = self._policy.get_camera_pose(0)
        cam_info = self._sim._agent_buffers[self._agent_id]
        self._sim._set_node_pose(cam_info["camera_np"], pos0, rot0)
        obs0, _ = self._sim.step([])
        rgba0 = list(obs0[self._agent_id].values())[0]["rgba"].copy()

        # First step at waypoint 2 (step = 10 + 3 + 10 + 3 = 26)
        step_wp2 = self._policy.waypoint_indices[2]
        pos2, rot2 = self._policy.get_camera_pose(step_wp2)
        self._sim._set_node_pose(cam_info["camera_np"], pos2, rot2)
        obs2, _ = self._sim.step([])
        rgba2 = list(obs2[self._agent_id].values())[0]["rgba"].copy()

        # Images should differ (different camera positions, different objects in view)
        self.assertFalse(
            np.array_equal(rgba0, rgba2),
            "Expected different observations at different waypoints",
        )


@unittest.skipUnless(_panda3d_available(), "Panda3D not installed")
@unittest.skipUnless(_ycb_data_available(), "YCB data not available")
class TestTemporalSceneEvaluator(unittest.TestCase):
    """Integration tests for the full temporal scene evaluation pipeline."""

    _evaluator = None

    @classmethod
    def setUpClass(cls):
        from tbp.monty.simulators.panda3d.temporal_evaluation import (
            TemporalSceneEvaluator,
        )

        cls._evaluator = TemporalSceneEvaluator(
            resolution=(64, 64),
            fov=90.0,
            orbit_radius=0.3,
            dwell_steps=10,
            transit_steps=3,
        )

    @classmethod
    def tearDownClass(cls):
        if cls._evaluator is not None:
            cls._evaluator.close()

    def test_run_scene_produces_result(self):
        """run_scene returns a SceneEvalResult with traversals."""
        from tbp.monty.simulators.panda3d.scenes import FRUIT_BOWL_SCENE

        result = self._evaluator.run_scene(FRUIT_BOWL_SCENE, n_traversals=2)
        self.assertEqual(len(result.traversals), 2)
        self.assertEqual(result.scene_name, "fruit_bowl")

    def test_traversal_has_surprise_values(self):
        """Each traversal should have non-empty surprise values."""
        from tbp.monty.simulators.panda3d.scenes import FRUIT_BOWL_SCENE

        result = self._evaluator.run_scene(FRUIT_BOWL_SCENE, n_traversals=1)
        t = result.traversals[0]
        self.assertGreater(t.usable_steps, 0)
        self.assertGreater(len(t.surprise_per_step), 0)

    def test_surprise_values_in_range(self):
        """Surprise values should be in [0, 1]."""
        from tbp.monty.simulators.panda3d.scenes import FRUIT_BOWL_SCENE

        result = self._evaluator.run_scene(FRUIT_BOWL_SCENE, n_traversals=1)
        for s in result.traversals[0].surprise_per_step:
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)

    def test_segment_types_are_valid(self):
        """Segment types should be 'dwell' or 'transit'."""
        from tbp.monty.simulators.panda3d.scenes import FRUIT_BOWL_SCENE

        result = self._evaluator.run_scene(FRUIT_BOWL_SCENE, n_traversals=1)
        for seg_type in result.traversals[0].segment_types:
            self.assertIn(seg_type, ("dwell", "transit"))

    def test_habituation_ratio_is_positive(self):
        """Habituation ratio should be a positive number."""
        from tbp.monty.simulators.panda3d.scenes import FRUIT_BOWL_SCENE

        result = self._evaluator.run_scene(FRUIT_BOWL_SCENE, n_traversals=2)
        self.assertGreater(result.habituation_ratio, 0.0)

    def test_summary_is_nonempty(self):
        """Summary string should contain key metrics."""
        from tbp.monty.simulators.panda3d.scenes import FRUIT_BOWL_SCENE

        result = self._evaluator.run_scene(FRUIT_BOWL_SCENE, n_traversals=1)
        summary = result.summary()
        self.assertIn("fruit_bowl", summary)
        self.assertIn("Surprise", summary)

    def test_wall_clock_positive(self):
        """Wall clock time should be positive."""
        from tbp.monty.simulators.panda3d.scenes import FRUIT_BOWL_SCENE

        result = self._evaluator.run_scene(FRUIT_BOWL_SCENE, n_traversals=1)
        self.assertGreater(result.wall_clock_seconds, 0.0)

    def test_config_is_populated(self):
        """Config dict should contain evaluation parameters."""
        from tbp.monty.simulators.panda3d.scenes import FRUIT_BOWL_SCENE

        result = self._evaluator.run_scene(FRUIT_BOWL_SCENE, n_traversals=1)
        self.assertIn("dwell_steps", result.config)
        self.assertEqual(result.config["dwell_steps"], 10)


@unittest.skipUnless(_panda3d_available(), "Panda3D not installed")
@unittest.skipUnless(_ycb_data_available(), "YCB data not available")
class TestSwapDetection(unittest.TestCase):
    """Integration tests for object swap detection."""

    _evaluator = None

    @classmethod
    def setUpClass(cls):
        from tbp.monty.simulators.panda3d.temporal_evaluation import (
            TemporalSceneEvaluator,
        )

        cls._evaluator = TemporalSceneEvaluator(
            resolution=(64, 64),
            orbit_radius=0.3,
            dwell_steps=10,
            transit_steps=3,
        )

    @classmethod
    def tearDownClass(cls):
        if cls._evaluator is not None:
            cls._evaluator.close()

    def test_swap_detection_returns_result(self):
        """run_swap_detection returns a SwapDetectionResult."""
        from tbp.monty.simulators.panda3d.scenes import FRUIT_BOWL_SCENE
        from tbp.monty.simulators.panda3d.temporal_evaluation import (
            SwapDetectionResult,
        )

        result = self._evaluator.run_swap_detection(
            scene=FRUIT_BOWL_SCENE,
            swap_index=0,
            replacement_ycb="025_mug",
            n_learning_traversals=2,
        )
        self.assertIsInstance(result, SwapDetectionResult)
        self.assertEqual(result.original_object, "011_banana")
        self.assertEqual(result.replacement_object, "025_mug")

    def test_swap_detection_fields_populated(self):
        """SwapDetectionResult has all expected fields."""
        from tbp.monty.simulators.panda3d.scenes import FRUIT_BOWL_SCENE

        result = self._evaluator.run_swap_detection(
            scene=FRUIT_BOWL_SCENE,
            swap_index=0,
            replacement_ycb="025_mug",
            n_learning_traversals=2,
        )
        self.assertIsInstance(result.surprise_at_swap_original, float)
        self.assertIsInstance(result.surprise_at_swap_modified, float)
        self.assertIsInstance(result.surprise_delta, float)
        self.assertIsInstance(result.detected, bool)


if __name__ == "__main__":
    unittest.main()
