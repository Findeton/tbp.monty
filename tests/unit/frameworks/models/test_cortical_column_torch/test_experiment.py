import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from tbp.monty.frameworks.actions.actions import (
    LookUp,
    MoveForward,
    MoveTangentially,
    TurnLeft,
)
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.models.cortical_column_torch.experiment import (
    AxisAwareConstantSampler,
    Panda3DTorchExperiment,
)
from tbp.monty.frameworks.models.change_detecting_sm import ChangeDetectingSM
from tbp.monty.frameworks.models.sensor_modules import CameraSM
from tbp.monty.frameworks.models.predictive_hypothesis_torch import (
    DetailAwareCameraSM,
    DetailAwareChangeDetectingSM,
)


class TestPanda3DTorchExperimentInitialCamera(unittest.TestCase):
    def test_position_camera_initial_renders_after_pose(self):
        exp = Panda3DTorchExperiment.__new__(Panda3DTorchExperiment)

        call_order = []
        camera_np = Mock()
        anim_obj = Mock()
        anim_obj.pose.side_effect = lambda frame, anim: call_order.append(
            ("pose", frame, anim)
        )

        sim = Mock()
        sim._agent_buffers = {"agent": {"camera_np": camera_np}}
        sim._objects = {"obj": Mock()}
        sim._objects["obj"].getTightBounds.return_value = (
            (0.0, 1.0, 2.0),
            (2.0, 3.0, 4.0),
        )
        sim._render.side_effect = lambda: call_order.append(("render",))

        exp._sim = sim
        exp._agent_id = "agent"
        exp._anim_obj = anim_obj
        exp._obj_id = "obj"
        exp._object_position = (10.0, 20.0, 30.0)
        exp._initial_distance = 3.0
        exp._sync_motor_state = Mock()

        exp._position_camera_initial(anim_name="Walk", frame=7)

        render_count = Panda3DTorchExperiment._ACTOR_SYNC_RENDER_PASSES
        self.assertEqual(call_order[0], ("pose", 7, "Walk"))
        self.assertEqual(call_order[1 : 1 + render_count], [("render",)] * render_count)
        camera_np.setPos.assert_called_once_with(1.0, -1.0, 3.0)
        camera_np.lookAt.assert_called_once()
        exp._sync_motor_state.assert_called_once_with()

    def test_resolve_camera_target_center_retries_until_bounds_are_sane(self):
        exp = Panda3DTorchExperiment.__new__(Panda3DTorchExperiment)

        sim = Mock()
        sim._objects = {"obj": Mock()}
        sim._objects["obj"].getTightBounds.side_effect = [
            ((0.0, -3200.0, -3200.0), (10.0, -3100.0, -3100.0)),
            ((0.0, 1.0, 2.0), (2.0, 3.0, 4.0)),
        ]
        sim._render = Mock()

        exp._sim = sim
        exp._anim_obj = Mock()
        exp._obj_id = "obj"
        exp._object_position = (10.0, 20.0, 30.0)
        exp._initial_distance = 3.0
        exp._last_camera_init_debug = None

        center = exp._resolve_camera_target_center(anim_name="Walk", frame=7)

        self.assertEqual(center, (1.0, 2.0, 3.0))
        self.assertEqual(exp._anim_obj.pose.call_count, 2)
        self.assertEqual(
            sim._render.call_count,
            2 * Panda3DTorchExperiment._ACTOR_SYNC_RENDER_PASSES,
        )
        self.assertEqual(exp._last_camera_init_debug["selected_source"], "tight_bounds")
        self.assertEqual(exp._last_camera_init_debug["selected_attempt"], 2)

    def test_resolve_camera_target_center_falls_back_after_invalid_bounds(self):
        exp = Panda3DTorchExperiment.__new__(Panda3DTorchExperiment)

        sim = Mock()
        sim._objects = {"obj": Mock()}
        sim._objects["obj"].getTightBounds.side_effect = [
            ((0.0, -3200.0, -3200.0), (10.0, -3100.0, -3100.0)),
        ] * Panda3DTorchExperiment._CAMERA_BOUNDS_RETRY_ATTEMPTS
        sim._render = Mock()

        exp._sim = sim
        exp._anim_obj = Mock()
        exp._obj_id = "obj"
        exp._object_position = (10.0, 20.0, 30.0)
        exp._initial_distance = 3.0
        exp._last_camera_init_debug = None

        center = exp._resolve_camera_target_center(anim_name="Walk", frame=7)

        self.assertEqual(center, (10.0, 20.0, 30.0))
        self.assertEqual(
            exp._last_camera_init_debug["selected_source"],
            "object_position_fallback",
        )
        self.assertIsNone(exp._last_camera_init_debug["selected_attempt"])
        self.assertEqual(
            len(exp._last_camera_init_debug["attempts"]),
            Panda3DTorchExperiment._CAMERA_BOUNDS_RETRY_ATTEMPTS,
        )

    def test_swap_model_renders_after_loading_animated_object(self):
        exp = Panda3DTorchExperiment.__new__(Panda3DTorchExperiment)

        sim = Mock()
        info = Mock()
        info.object_id = "obj"
        sim.add_object.return_value = info
        sim.get_animated_object.return_value = "animated"

        exp._sim = sim
        exp._model_path = Path("before.glb")
        exp._object_position = (0.0, 0.0, 0.0)
        exp._object_scale = (1.0, 1.0, 1.0)
        exp._initial_distance = 2.0

        exp.swap_model("after.glb", object_scale=(0.3, 0.3, 0.3), initial_distance=3.0)

        sim.remove_all_objects.assert_called_once_with()
        sim.add_object.assert_called_once_with(
            name="after.glb",
            position=(0.0, 0.0, 0.0),
            scale=(0.3, 0.3, 0.3),
            animated=True,
        )
        self.assertEqual(
            sim._render.call_count,
            Panda3DTorchExperiment._ACTOR_SYNC_RENDER_PASSES,
        )
        self.assertEqual(exp._anim_obj, "animated")
        self.assertEqual(exp._obj_id, "obj")
        self.assertEqual(exp._initial_distance, 3.0)

    def test_encode_action_context_summarizes_motor_commands(self):
        agent_id = AgentID("agent_id_0")
        action_context = Panda3DTorchExperiment._encode_action_context(
            [
                MoveForward(agent_id=agent_id, distance=0.4),
                TurnLeft(agent_id=agent_id, rotation_degrees=5.0),
                LookUp(agent_id=agent_id, rotation_degrees=2.0),
            ]
        )

        np.testing.assert_allclose(
            action_context,
            np.array([0.4, -5.0, 2.0, 0.0, 0.0, 3.0, 1.0, 2.0], dtype=np.float32),
        )

    def test_warmup_episode_observation_discards_cold_start_frame(self):
        exp = Panda3DTorchExperiment.__new__(Panda3DTorchExperiment)
        exp._sim = object()
        exp._step_environment = Mock()

        exp._warmup_episode_observation(anim_name="Walk", frame=7)

        exp._step_environment.assert_called_once_with(
            [],
            frame=7,
            anim_name="Walk",
            step=-1,
        )

    def test_axis_aware_sampler_includes_backward_and_tangential_axes(self):
        sampler = AxisAwareConstantSampler(
            actions=[MoveForward, MoveTangentially],
            translation_distance=0.4,
            allow_backward_actions=True,
            tangential_directions=[
                (-1.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.0, 0.0, -1.0),
            ],
        )

        rng = np.random.RandomState(7)
        agent_id = AgentID("agent_id_0")

        forward_distances = {
            sampler.sample_move_forward(agent_id, rng).distance for _ in range(32)
        }
        tangential_directions = {
            tuple(sampler.sample_move_tangentially(agent_id, rng).direction)
            for _ in range(64)
        }

        self.assertEqual(forward_distances, {-0.4, 0.4})
        self.assertEqual(
            tangential_directions,
            {
                (-1.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.0, 0.0, -1.0),
            },
        )

    def test_track12_uses_legacy_sensor_modules_by_default(self):
        exp = Panda3DTorchExperiment.__new__(Panda3DTorchExperiment)
        exp._camera_features = ["pose_vectors", "on_object", "hsv"]
        exp._flow_threshold = 1e-6

        camera_sm = exp._build_camera_sensor_module()
        change_sm = exp._build_change_sensor_module()

        self.assertIsInstance(camera_sm, CameraSM)
        self.assertIsInstance(change_sm, ChangeDetectingSM)

    def test_track12_can_opt_into_detail_aware_sensor_modules(self):
        exp = Panda3DTorchExperiment.__new__(Panda3DTorchExperiment)
        exp._camera_features = ["pose_vectors", "on_object", "hsv"]
        exp._flow_threshold = 1e-6
        exp._use_detail_aware_sensors = True

        camera_sm = exp._build_camera_sensor_module()
        change_sm = exp._build_change_sensor_module()

        self.assertIsInstance(camera_sm, DetailAwareCameraSM)
        self.assertIsInstance(change_sm, DetailAwareChangeDetectingSM)
        self.assertEqual(camera_sm._detail_grid_shape, (5, 5))
        self.assertEqual(change_sm._detail_grid_shape, (5, 5))
        self.assertFalse(camera_sm._include_world_pose)
        self.assertFalse(change_sm._include_world_pose)

    def test_track12_detail_aware_encoder_uses_sensor_frame_location(self):
        exp = Panda3DTorchExperiment.__new__(Panda3DTorchExperiment)
        exp._use_detail_aware_sensors = True

        lm = exp._build_learning_module(
            learning_module_id="lm_0",
            column_kwargs={"n_minicolumns": 64, "sparsity": 0.1, "seed": 42},
            lm_kwargs={},
        )

        self.assertEqual(
            lm._column._encoder._observation_packet_location_mode,
            "sensor_frame_centroid",
        )
        self.assertTrue(
            lm._column._encoder._ignore_pose_vectors_for_observation_packets
        )

    def test_setup_hierarchical_applies_child_specific_column_kwargs(self):
        exp = Panda3DTorchExperiment.__new__(Panda3DTorchExperiment)
        exp._camera_features = ["pose_vectors", "on_object", "hsv"]
        exp._flow_threshold = 1e-6
        exp._seed = 42
        exp._column_kwargs = {"n_minicolumns": 64, "sparsity": 0.1}
        exp._morphology_column_kwargs = {}
        exp._behavior_column_kwargs = {"defer_sensor_auto_label": True}
        exp._parent_column_kwargs = {}
        exp._morphology_lm_kwargs = {}
        exp._behavior_lm_kwargs = {}
        exp._parent_lm_kwargs = {}
        exp._hopfield_voting = False
        exp._hopfield_surprise_threshold = 0.3
        exp._conditional_voting = False
        exp._vote_confident_threshold = 2
        exp._predictive_voting = False
        exp._temporal_confusion_threshold = 0.5
        exp._vote_after_steps = 2
        exp._vote_cooldown_steps = 0
        exp._authoritative_goal_sender_ids = []
        exp._agent_id = AgentID("agent_id_0")
        exp._build_motor_system = Mock(return_value=Mock())

        exp._setup_hierarchical()

        lm_morphology, lm_behavior, lm_parent = exp._monty.learning_modules
        self.assertFalse(lm_morphology.column._defer_sensor_auto_label)
        self.assertTrue(lm_behavior.column._defer_sensor_auto_label)
        self.assertTrue(lm_parent.column._defer_context_auto_label)
        exp._build_motor_system.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()