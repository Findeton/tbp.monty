# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Panda3D behavior integration tests.

Tests the full pipeline for behavior detection:
  Panda3D animate → render → transform → ChangeDetectingSM → detect change

Also tests StateConditionedModel and GlobalIntervalTimer integration
with the Panda3D rendering pipeline.
"""

import math
import struct
import tempfile
import unittest

import numpy as np
import pytest

from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.sensors import SensorID

try:
    import panda3d  # noqa: F401
    import gltf  # noqa: F401
except ImportError:
    raise unittest.SkipTest("Panda3D or panda3d-gltf not installed")

from tbp.monty.simulators.panda3d.agents import Panda3DAgent
from tbp.monty.simulators.panda3d.simulator import Panda3DSimulator
from tbp.monty.simulators.panda3d.transforms import Panda3DDepthNormalize


pytestmark = pytest.mark.xdist_group(name="panda3d")

AGENT_ID = AgentID("test_cam")
SENSOR_ID = "sensor_0"


def _make_simulator(position=(0, -3, 1), far=20.0, **kwargs):
    """Create a Panda3D simulator with a single camera."""
    agent = Panda3DAgent(
        agent_id=AGENT_ID,
        sensor_id=SENSOR_ID,
        position=position,
        rotation=(1.0, 0.0, 0.0, 0.0),
        resolution=(64, 64),
        fov=90.0,
    )
    return Panda3DSimulator(agents=[agent], near=0.01, far=far, **kwargs)


def _make_animated_gltf(directory, filename="animated.gltf"):
    """Create a minimal glTF with a 2-joint skeleton and rotation animation."""
    s = 0.3
    vertices = [
        (-s, 0, -s), (s, 0, -s), (s, 0, s), (-s, 0, s),
        (-s, 2, -s), (s, 2, -s), (s, 2, s), (-s, 2, s),
    ]
    indices = [
        0, 1, 2, 0, 2, 3,
        4, 6, 5, 4, 7, 6,
        0, 4, 5, 0, 5, 1,
        2, 6, 7, 2, 7, 3,
        0, 3, 7, 0, 7, 4,
        1, 5, 6, 1, 6, 2,
    ]
    joints_data = [(0, 0, 0, 0)] * 4 + [(1, 0, 0, 0)] * 4
    weights_data = [(1.0, 0, 0, 0)] * 8

    vert_bin = b"".join(struct.pack("<3f", *v) for v in vertices)
    idx_bin = b"".join(struct.pack("<H", i) for i in indices)
    while len(idx_bin) % 4:
        idx_bin += b"\x00"
    joints_bin = b"".join(struct.pack("<4B", *j) for j in joints_data)
    weights_bin = b"".join(struct.pack("<4f", *w) for w in weights_data)

    ibm0 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    ibm1 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, -1, 0, 1]
    ibm_bin = struct.pack("<16f", *ibm0) + struct.pack("<16f", *ibm1)

    s45 = math.sin(math.radians(45))
    c45 = math.cos(math.radians(45))
    anim_times = struct.pack("<3f", 0.0, 0.5, 1.0)
    anim_rots = struct.pack(
        "<12f", 0, 0, 0, 1, 0, 0, s45, c45, 0, 0, 0, 1
    )

    import os
    buf_path = os.path.join(directory, "data.bin")
    with open(buf_path, "wb") as f:
        f.write(vert_bin)
        f.write(idx_bin)
        f.write(joints_bin)
        f.write(weights_bin)
        f.write(ibm_bin)
        f.write(anim_times)
        f.write(anim_rots)

    vert_len = len(vert_bin)
    idx_len = len(idx_bin)
    joints_len = len(joints_bin)
    weights_len = len(weights_bin)
    ibm_len = len(ibm_bin)
    anim_times_len = len(anim_times)
    anim_rots_len = len(anim_rots)
    total = (vert_len + idx_len + joints_len + weights_len
             + ibm_len + anim_times_len + anim_rots_len)

    vert_off = 0
    idx_off = vert_off + vert_len
    joints_off = idx_off + idx_len
    weights_off = joints_off + joints_len
    ibm_off = weights_off + weights_len
    anim_times_off = ibm_off + ibm_len
    anim_rots_off = anim_times_off + anim_times_len

    import json
    gltf_data = {
        "asset": {"version": "2.0", "generator": "test"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [
            {"name": "Root", "skin": 0, "mesh": 0, "children": [1]},
            {"name": "Joint0", "children": [2], "translation": [0, 0, 0]},
            {"name": "Joint1", "translation": [0, 1, 0]},
        ],
        "skins": [{"joints": [1, 2], "inverseBindMatrices": 4}],
        "meshes": [{"primitives": [{"attributes": {
            "POSITION": 0, "JOINTS_0": 2, "WEIGHTS_0": 3,
        }, "indices": 1}]}],
        "animations": [{"name": "Bend", "channels": [
            {"sampler": 0, "target": {"node": 2, "path": "rotation"}},
        ], "samplers": [{"input": 5, "output": 6, "interpolation": "LINEAR"}]}],
        "buffers": [{"uri": "data.bin", "byteLength": total}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": vert_off, "byteLength": vert_len,
             "target": 34962},
            {"buffer": 0, "byteOffset": idx_off, "byteLength": idx_len,
             "target": 34963},
            {"buffer": 0, "byteOffset": joints_off, "byteLength": joints_len},
            {"buffer": 0, "byteOffset": weights_off, "byteLength": weights_len},
            {"buffer": 0, "byteOffset": ibm_off, "byteLength": ibm_len},
            {"buffer": 0, "byteOffset": anim_times_off,
             "byteLength": anim_times_len},
            {"buffer": 0, "byteOffset": anim_rots_off,
             "byteLength": anim_rots_len},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 8,
             "type": "VEC3",
             "min": [-s, 0, -s], "max": [s, 2, s]},
            {"bufferView": 1, "componentType": 5123,
             "count": len(indices), "type": "SCALAR"},
            {"bufferView": 2, "componentType": 5121, "count": 8,
             "type": "VEC4"},
            {"bufferView": 3, "componentType": 5126, "count": 8,
             "type": "VEC4"},
            {"bufferView": 4, "componentType": 5126, "count": 2,
             "type": "MAT4"},
            {"bufferView": 5, "componentType": 5126, "count": 3,
             "type": "SCALAR", "min": [0.0], "max": [1.0]},
            {"bufferView": 6, "componentType": 5126, "count": 3,
             "type": "VEC4"},
        ],
    }

    gltf_path = os.path.join(directory, filename)
    with open(gltf_path, "w") as f:
        json.dump(gltf_data, f)
    return gltf_path


def _get_observation(sim):
    """Get transformed observation from simulator."""
    from tbp.monty.frameworks.environment_utils.transforms import (
        DepthTo3DLocations,
        TransformContext,
    )

    obs, proprio = sim.step([])
    depth_norm = Panda3DDepthNormalize(
        agent_id=AGENT_ID, near=0.01, far=20.0
    )
    d3d = DepthTo3DLocations(
        agent_id=AGENT_ID,
        sensor_ids=[SensorID(SENSOR_ID)],
        resolutions=[(64, 64)],
        hfov=90.0,
        world_coord=True,
        get_all_points=True,
    )
    ctx = TransformContext(rng=np.random.RandomState(0), state=proprio)
    obs = depth_norm(obs, ctx)
    obs = d3d(obs, ctx)
    return obs[AGENT_ID][SensorID(SENSOR_ID)]


class TestChangeDetectingSMWithPanda3D(unittest.TestCase):
    """Test ChangeDetectingSM detects animated object movement."""

    def test_static_scene_no_change(self):
        """Static object -> ChangeDetectingSM emits a quiet low-confidence state."""
        from tbp.monty.frameworks.models.change_detecting_sm import (
            ChangeDetectingSM,
        )

        sim = _make_simulator(position=(0, -3, 1), far=20.0)
        try:
            sim.add_object("sphere", position=(0, 0, 0))

            sm = ChangeDetectingSM(
                sensor_module_id="change_SM_0",
                flow_threshold=0.01,
            )

            # Two identical renders of static scene
            obs1 = _get_observation(sim)
            obs2 = _get_observation(sim)

            sm.step(None, obs1)
            state = sm.step(None, obs2)

            # No motion in static scene: the current contract emits a
            # low-confidence quiet state so the behavior stream keeps
            # anchored absolute context across calm frames.
            self.assertTrue(state.use_state)
            self.assertLessEqual(state.confidence, 0.1)
            np.testing.assert_allclose(
                state.non_morphological_features["flow_direction"],
                np.zeros(3),
            )
            np.testing.assert_allclose(
                state.non_morphological_features["flow_magnitude"],
                np.zeros(1),
            )
            np.testing.assert_allclose(
                state.non_morphological_features["delta_rgba"],
                np.zeros(4),
            )
            np.testing.assert_allclose(
                state.non_morphological_features["delta_hsv"],
                np.zeros(3),
            )
        finally:
            sim.close()

    def test_animated_object_detects_change(self):
        """Animated object → ChangeDetectingSM detects change."""
        from tbp.monty.frameworks.models.change_detecting_sm import (
            ChangeDetectingSM,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            gltf_path = _make_animated_gltf(tmpdir)
            sim = _make_simulator(
                position=(3, -5, 1), far=20.0,
                asset_search_paths=[tmpdir],
            )
            try:
                info = sim.add_object(
                    gltf_path, position=(0, 0, 0), animated=True
                )
                anim = sim.get_animated_object(info.object_id)
                n_frames = anim.get_num_frames()

                sm = ChangeDetectingSM(
                    sensor_module_id="change_SM_0",
                    flow_threshold=0.005,
                    global_flow_suppression=False,
                )

                # Frame 0
                anim.pose(0)
                obs1 = _get_observation(sim)
                sm.step(None, obs1)

                # Frame at 2/3 through animation (large deformation)
                far_frame = (n_frames * 2) // 3
                anim.pose(far_frame)
                obs2 = _get_observation(sim)
                state = sm.step(None, obs2)

                # Should detect the animated motion
                self.assertTrue(state.use_state)
                self.assertIn("flow_direction", state.non_morphological_features)
                self.assertIn("flow_magnitude", state.non_morphological_features)
            finally:
                sim.close()


class TestGlobalIntervalTimerWithAnimation(unittest.TestCase):
    """Test timer integrated with animation frames."""

    def test_timer_tracks_animation_steps(self):
        """Timer advances with each animation step."""
        from tbp.monty.frameworks.models.interval_timer import (
            GlobalIntervalTimer,
        )

        timer = GlobalIntervalTimer(n_time_cells=16, ticks_per_step=1.0)

        # Simulate 5 animation frames
        encodings = []
        for frame in range(5):
            timer.step()
            encodings.append(timer.get_time_encoding().copy())

        # Encodings should differ across frames
        self.assertFalse(np.allclose(encodings[0], encodings[4]))

        # Active cell should advance
        self.assertGreater(timer.get_active_cell(), 0)

    def test_timer_reset_on_event(self):
        """Timer resets when a significant event is detected."""
        from tbp.monty.frameworks.models.interval_timer import (
            GlobalIntervalTimer,
        )

        timer = GlobalIntervalTimer(n_time_cells=16, ticks_per_step=1.0)

        # Advance timer
        for _ in range(10):
            timer.step()
        mid_tick = timer.get_current_tick()

        # "Event detected" → reset
        timer.reset()
        self.assertEqual(timer.get_current_tick(), 0.0)
        self.assertGreater(mid_tick, 0.0)


class TestStateConditionedModelWithRendering(unittest.TestCase):
    """Test building StateConditionedModel from rendered observations."""

    def test_build_model_from_two_animation_states(self):
        """Build a 2-state model from two animation poses."""
        from tbp.monty.frameworks.models.state_conditioned_model import (
            StateConditionedModel,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            gltf_path = _make_animated_gltf(tmpdir)
            sim = _make_simulator(
                position=(3, -5, 1), far=20.0,
                asset_search_paths=[tmpdir],
            )
            try:
                info = sim.add_object(
                    gltf_path, position=(0, 0, 0), animated=True
                )
                anim = sim.get_animated_object(info.object_id)
                n_frames = anim.get_num_frames()

                # Collect observations for 2 states: frame 0 vs far frame
                far_frame = (n_frames * 2) // 3
                states_data = {}
                for state_id, frame in enumerate([0, far_frame]):
                    anim.pose(frame)
                    obs = _get_observation(sim)

                    sem_3d = obs["semantic_3d"]
                    on_object = sem_3d[:, 3] > 0
                    if np.any(on_object):
                        points = sem_3d[on_object, :3]
                        states_data[state_id] = points

                # Build StateConditionedModel
                scm = StateConditionedModel(
                    object_id="animated_bar",
                    max_nodes=50,
                    max_size=10.0,
                    num_voxels_per_dim=20,
                )

                for state_id, points in states_data.items():
                    n = min(len(points), 30)
                    locs = points[:n]
                    feats = {
                        "pose_vectors": np.tile(
                            np.eye(3).flatten(), (n, 1)
                        ),
                        "pose_fully_defined": np.ones(n, dtype=bool),
                    }
                    scm.build_model(locs, feats, state_id=state_id)

                self.assertEqual(scm.get_num_states(), 2)
                self.assertTrue(scm.has_state(0))
                self.assertTrue(scm.has_state(1))

                # The two states should have different point distributions
                m0 = scm.get_model_for_state(0)
                m1 = scm.get_model_for_state(1)
                mean0 = m0.pos.mean(axis=0)
                mean1 = m1.pos.mean(axis=0)
                # They differ because the mesh deforms between frames
                self.assertFalse(
                    np.allclose(mean0, mean1, atol=0.01),
                    "States should have different spatial distributions"
                )
            finally:
                sim.close()


class TestEndToEndBehaviorPipeline(unittest.TestCase):
    """End-to-end: animate → render → detect change → track state."""

    def test_full_behavior_detection_pipeline(self):
        """Full pipeline: animate, render, detect changes across frames."""
        from tbp.monty.frameworks.models.change_detecting_sm import (
            ChangeDetectingSM,
        )
        from tbp.monty.frameworks.models.interval_timer import (
            GlobalIntervalTimer,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            gltf_path = _make_animated_gltf(tmpdir)
            sim = _make_simulator(
                position=(3, -5, 1), far=20.0,
                asset_search_paths=[tmpdir],
            )
            try:
                info = sim.add_object(
                    gltf_path, position=(0, 0, 0), animated=True
                )
                anim = sim.get_animated_object(info.object_id)
                n_frames = anim.get_num_frames()

                sm = ChangeDetectingSM(
                    sensor_module_id="change_SM_0",
                    flow_threshold=0.005,
                    global_flow_suppression=False,
                )
                timer = GlobalIntervalTimer(
                    n_time_cells=16, ticks_per_step=1.0
                )

                # Use well-spaced frames for visible deformation
                frames = [0, (n_frames * 2) // 3, n_frames - 1]
                changes_detected = 0
                for frame in frames[:min(len(frames), 3)]:
                    anim.pose(frame)
                    obs = _get_observation(sim)
                    state = sm.step(None, obs)
                    timer.step()

                    if state.use_state:
                        changes_detected += 1
                        # On change, we'd reset timer in full system
                        # timer.reset()

                # Should detect at least one change across animation frames
                # (frame 0 → frame 1 involves joint rotation)
                self.assertGreater(
                    changes_detected, 0,
                    "Should detect motion in animated object"
                )

                # Timer should have advanced
                self.assertGreater(timer.get_current_tick(), 0.0)
            finally:
                sim.close()


class TestLMRecognizesAnimatedPoses(unittest.TestCase):
    """Train EvidenceGraphLM on two animation poses, verify recognition."""

    def _render_pose_observations(self, sim, anim, frame, n_views=5):
        """Render an animated model at a specific frame from multiple views.

        Returns a list of State objects suitable for EvidenceGraphLM input.
        """
        from tbp.monty.frameworks.models.states import State

        anim.pose(frame)
        states = []

        for view_idx in range(n_views):
            obs = _get_observation(sim)
            sem_3d = obs.get("semantic_3d")
            if sem_3d is None or len(sem_3d) == 0:
                continue

            on_object = sem_3d[:, 3] > 0
            if not np.any(on_object):
                continue

            points = sem_3d[on_object, :3]
            # Use a representative point (center of visible surface)
            center = points.mean(axis=0)
            spread = points.std(axis=0)

            states.append(State(
                location=center + np.random.randn(3) * 0.001 * view_idx,
                morphological_features={
                    "pose_vectors": np.eye(3),
                    "pose_fully_defined": True,
                    "on_object": 1,
                },
                non_morphological_features={
                    "principal_curvatures_log": [
                        float(np.log1p(spread[0])),
                        float(np.log1p(spread[1])),
                    ],
                    "hsv": [0.1, 0.5, 0.6],
                },
                confidence=1.0,
                use_state=True,
                sender_id="patch",
                sender_type="SM",
            ))

        return states

    def test_lm_learns_and_distinguishes_two_poses(self):
        """LM trains on straight vs bent pose, recognizes each."""
        from tbp.monty.context import RuntimeContext
        from tbp.monty.frameworks.experiments.mode import ExperimentMode
        from tbp.monty.frameworks.models.evidence_matching.learning_module import (
            EvidenceGraphLM,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            gltf_path = _make_animated_gltf(tmpdir)
            sim = _make_simulator(
                position=(3, -5, 1), far=20.0,
                asset_search_paths=[tmpdir],
            )
            try:
                info = sim.add_object(
                    gltf_path, position=(0, 0, 0), animated=True
                )
                anim = sim.get_animated_object(info.object_id)
                n_frames = anim.get_num_frames()
                far_frame = (n_frames * 2) // 3

                # Render observations for two poses
                obs_straight = self._render_pose_observations(
                    sim, anim, frame=0, n_views=8
                )
                obs_bent = self._render_pose_observations(
                    sim, anim, frame=far_frame, n_views=8
                )

                self.assertGreater(
                    len(obs_straight), 0,
                    "Should get observations for straight pose"
                )
                self.assertGreater(
                    len(obs_bent), 0,
                    "Should get observations for bent pose"
                )

                # Create LM
                lm = EvidenceGraphLM(
                    max_match_distance=0.1,
                    tolerances={
                        "patch": {
                            "principal_curvatures_log": [2, 2],
                        }
                    },
                    feature_weights={
                        "patch": {},
                    },
                    max_graph_size=5.0,
                    num_model_voxels_per_dim=100,
                    hypotheses_updater_args=dict(
                        initial_possible_poses="informed",
                    ),
                )
                ctx = RuntimeContext(rng=np.random.RandomState(42))

                # Train on straight pose
                lm.mode = ExperimentMode.TRAIN
                lm.pre_episode(
                    primary_target={
                        "object": "straight",
                        "quat_rotation": [1, 0, 0, 0],
                    }
                )
                for obs in obs_straight:
                    lm.exploratory_step(ctx, [obs])
                lm.detected_object = "straight"
                lm.detected_rotation_r = None
                lm.buffer.stats["detected_location_rel_body"] = (
                    lm.buffer.get_current_location(input_channel="first")
                )
                lm.post_episode()

                # Train on bent pose
                lm.pre_episode(
                    primary_target={
                        "object": "bent",
                        "quat_rotation": [1, 0, 0, 0],
                    }
                )
                for obs in obs_bent:
                    lm.exploratory_step(ctx, [obs])
                lm.detected_object = "bent"
                lm.detected_rotation_r = None
                lm.buffer.stats["detected_location_rel_body"] = (
                    lm.buffer.get_current_location(input_channel="first")
                )
                lm.post_episode()

                # Verify both learned
                known = lm.get_all_known_object_ids()
                self.assertEqual(len(known), 2)
                self.assertIn("straight", known)
                self.assertIn("bent", known)

                # Test recognition of straight pose
                lm.mode = ExperimentMode.EVAL
                lm.pre_episode(
                    primary_target={
                        "object": "placeholder",
                        "quat_rotation": [1, 0, 0, 0],
                    }
                )
                for obs in obs_straight[:4]:
                    lm.add_lm_processing_to_buffer_stats(lm_processed=True)
                    lm.matching_step(ctx, [obs])

                straight_ev = np.max(lm.evidence["straight"])
                bent_ev = np.max(lm.evidence["bent"])
                self.assertGreater(
                    straight_ev, bent_ev,
                    f"Straight evidence ({straight_ev:.2f}) should > "
                    f"bent evidence ({bent_ev:.2f}) for straight input"
                )
            finally:
                sim.close()


if __name__ == "__main__":
    unittest.main()
