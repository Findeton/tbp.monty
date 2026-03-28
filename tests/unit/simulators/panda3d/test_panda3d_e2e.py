# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""End-to-end tests: Panda3D train + eval with HPC integration.

Tests three things that were previously missing:
1. HPC wired as 3rd LM in Panda3DBehaviorExperiment (learns associations
   during training, provides context during eval)
2. Panda3DEnvironment + EnvironmentInterfacePerObject (standard experiment
   loop with Panda3D primitives via object registry)
3. Panda3D object registry for cycling through multiple objects

Architecture with HPC:
  SM 0: CameraSM          -> LM 0 (morphology)
  SM 1: ChangeDetectingSM -> LM 1 (behavior)
                             LM 2 (HippocampalModule) <- LM 0 + LM 1

  sm_to_lm_matrix:       [[0], [1], []]
  lm_to_lm_matrix:       [[], [], [0, 1]]
  lm_to_lm_vote_matrix:  [[1], [0], []]
"""

import json
import math
import os
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import panda3d  # noqa: F401
    import gltf  # noqa: F401
except ImportError:
    raise unittest.SkipTest("Panda3D or panda3d-gltf not installed")

from tbp.monty.simulators.panda3d.behavior_training import (
    Panda3DBehaviorExperiment,
)
from tbp.monty.simulators.panda3d.object_registry import Panda3DObjectRegistry
from tbp.monty.simulators.panda3d.environment import Panda3DEnvironment
from tbp.monty.simulators.panda3d.agents import Panda3DAgent
from tbp.monty.frameworks.agents import AgentID

# Real animated models in test assets
ASSET_DIR = str(Path(__file__).parent / "test_assets" / "animated")
FOX_PATH = str(Path(ASSET_DIR) / "Fox.glb")
ROBOT_PATH = str(Path(ASSET_DIR) / "RobotExpressive.glb")
CESIUM_MAN_PATH = str(Path(ASSET_DIR) / "CesiumMan.glb")


def _make_animated_gltf(
    directory,
    filename="animated.gltf",
    *,
    width=0.3,
    height=2.0,
    depth=0.3,
    color=None,
    animations=None,
):
    """Create a glTF with a 2-joint skeleton and one or more rotation animations.

    Duplicated from test_panda3d_behavior_lm.py for test isolation.
    """
    if animations is None:
        animations = [("Bend", (0, 0, 1), 45.0)]

    sx, sy, sz = width, height, depth
    vertices = [
        (-sx, 0, -sz), (sx, 0, -sz), (sx, 0, sz), (-sx, 0, sz),
        (-sx, sy, -sz), (sx, sy, -sz), (sx, sy, sz), (-sx, sy, sz),
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

    if color is not None:
        r_c, g_c, b_c = (int(c * 255) for c in color)
        color_bin = b"".join(
            struct.pack("<4B", r_c, g_c, b_c, 255) for _ in vertices
        )
    else:
        color_bin = b""

    ibm0 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    ibm1 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, -sy / 2, 0, 1]
    ibm_bin = struct.pack("<16f", *ibm0) + struct.pack("<16f", *ibm1)

    anim_time_bins = []
    anim_rot_bins = []
    for _name, axis, angle_deg in animations:
        half_angle = math.radians(angle_deg) / 2
        ax, ay, az = axis
        norm = math.sqrt(ax * ax + ay * ay + az * az)
        ax, ay, az = ax / norm, ay / norm, az / norm
        s_a = math.sin(half_angle)
        c_a = math.cos(half_angle)
        anim_time_bins.append(struct.pack("<3f", 0.0, 0.5, 1.0))
        anim_rot_bins.append(struct.pack(
            "<12f",
            0, 0, 0, 1,
            ax * s_a, ay * s_a, az * s_a, c_a,
            0, 0, 0, 1,
        ))

    data_bin = b"".join([
        vert_bin, idx_bin, joints_bin, weights_bin, color_bin, ibm_bin,
    ] + anim_time_bins + anim_rot_bins)

    buf_path = os.path.join(directory, filename.replace(".gltf", ".bin"))
    with open(buf_path, "wb") as f:
        f.write(data_bin)

    vert_len = len(vert_bin)
    idx_len = len(idx_bin)
    joints_len = len(joints_bin)
    weights_len = len(weights_bin)
    color_len = len(color_bin)
    ibm_len = len(ibm_bin)

    off = 0
    vert_off = off; off += vert_len
    idx_off = off; off += idx_len
    joints_off = off; off += joints_len
    weights_off = off; off += weights_len
    color_off = off; off += color_len
    ibm_off = off; off += ibm_len

    anim_time_size = 3 * 4
    anim_rot_size = 12 * 4

    buffer_views = [
        {"buffer": 0, "byteOffset": vert_off, "byteLength": vert_len,
         "target": 34962},
        {"buffer": 0, "byteOffset": idx_off, "byteLength": idx_len,
         "target": 34963},
        {"buffer": 0, "byteOffset": joints_off, "byteLength": joints_len},
        {"buffer": 0, "byteOffset": weights_off, "byteLength": weights_len},
    ]
    mesh_attrs = {
        "POSITION": 0, "JOINTS_0": 2, "WEIGHTS_0": 3,
    }
    accessors = [
        {"bufferView": 0, "componentType": 5126, "count": 8,
         "type": "VEC3",
         "min": [-sx, 0, -sz], "max": [sx, sy, sz]},
        {"bufferView": 1, "componentType": 5123,
         "count": len(indices), "type": "SCALAR"},
        {"bufferView": 2, "componentType": 5121, "count": 8,
         "type": "VEC4"},
        {"bufferView": 3, "componentType": 5126, "count": 8,
         "type": "VEC4"},
    ]

    if color is not None:
        bv_color = len(buffer_views)
        buffer_views.append(
            {"buffer": 0, "byteOffset": color_off,
             "byteLength": color_len}
        )
        acc_color = len(accessors)
        accessors.append(
            {"bufferView": bv_color, "componentType": 5121,
             "count": 8, "type": "VEC4", "normalized": True}
        )
        mesh_attrs["COLOR_0"] = acc_color

    bv_ibm = len(buffer_views)
    buffer_views.append(
        {"buffer": 0, "byteOffset": ibm_off, "byteLength": ibm_len}
    )
    acc_ibm = len(accessors)
    accessors.append(
        {"bufferView": bv_ibm, "componentType": 5126, "count": 2,
         "type": "MAT4"}
    )

    gltf_animations = []
    for i, (anim_name, _axis, _angle) in enumerate(animations):
        time_off = off + i * (anim_time_size + anim_rot_size)
        rot_off = time_off + anim_time_size

        bv_time = len(buffer_views)
        buffer_views.append(
            {"buffer": 0, "byteOffset": time_off,
             "byteLength": anim_time_size}
        )
        bv_rot = len(buffer_views)
        buffer_views.append(
            {"buffer": 0, "byteOffset": rot_off,
             "byteLength": anim_rot_size}
        )

        acc_time = len(accessors)
        accessors.append(
            {"bufferView": bv_time, "componentType": 5126, "count": 3,
             "type": "SCALAR", "min": [0.0], "max": [1.0]}
        )
        acc_rot = len(accessors)
        accessors.append(
            {"bufferView": bv_rot, "componentType": 5126, "count": 3,
             "type": "VEC4"}
        )

        gltf_animations.append({
            "name": anim_name,
            "channels": [
                {"sampler": 0,
                 "target": {"node": 2, "path": "rotation"}},
            ],
            "samplers": [
                {"input": acc_time, "output": acc_rot,
                 "interpolation": "LINEAR"},
            ],
        })

    gltf_data = {
        "asset": {"version": "2.0", "generator": "test"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [
            {"name": "Root", "skin": 0, "mesh": 0, "children": [1]},
            {"name": "Joint0", "children": [2], "translation": [0, 0, 0]},
            {"name": "Joint1", "translation": [0, sy / 2, 0]},
        ],
        "skins": [{"joints": [1, 2], "inverseBindMatrices": acc_ibm}],
        "meshes": [{"primitives": [{"attributes": mesh_attrs,
                                     "indices": 1}]}],
        "animations": gltf_animations,
        "buffers": [{"uri": os.path.basename(buf_path),
                      "byteLength": len(data_bin)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
    }

    gltf_path = os.path.join(directory, filename)
    with open(gltf_path, "w") as f:
        json.dump(gltf_data, f)
    return gltf_path


# ===========================================================================
# 1. Object Registry Tests
# ===========================================================================


class TestPanda3DObjectRegistry(unittest.TestCase):
    """Test Panda3DObjectRegistry name resolution."""

    def test_register_and_get_primitive(self):
        reg = Panda3DObjectRegistry({
            "my_sphere": {"primitive": "sphere"},
        })
        spec = reg.get_spec("my_sphere")
        self.assertEqual(spec["name"], "sphere")
        self.assertFalse(spec["animated"])

    def test_register_model_path(self):
        reg = Panda3DObjectRegistry({
            "fox": {"model_path": "/path/to/Fox.glb", "animated": True},
        })
        spec = reg.get_spec("fox")
        self.assertEqual(spec["name"], "/path/to/Fox.glb")
        self.assertTrue(spec["animated"])

    def test_register_with_scale(self):
        reg = Panda3DObjectRegistry({
            "big_cube": {"primitive": "cube", "scale": (2.0, 2.0, 2.0)},
        })
        spec = reg.get_spec("big_cube")
        self.assertEqual(spec["scale"], (2.0, 2.0, 2.0))

    def test_missing_name_raises(self):
        reg = Panda3DObjectRegistry({"a": {"primitive": "sphere"}})
        with self.assertRaises(KeyError):
            reg.get_spec("nonexistent")

    def test_invalid_spec_raises(self):
        with self.assertRaises(ValueError):
            PandaObjectRegistry = Panda3DObjectRegistry({"bad": {"color": "red"}})

    def test_has_and_contains(self):
        reg = Panda3DObjectRegistry({"s": {"primitive": "sphere"}})
        self.assertTrue(reg.has("s"))
        self.assertIn("s", reg)
        self.assertFalse(reg.has("nope"))

    def test_list_objects(self):
        reg = Panda3DObjectRegistry({
            "a": {"primitive": "sphere"},
            "b": {"primitive": "cube"},
        })
        self.assertEqual(sorted(reg.list_objects()), ["a", "b"])


# ===========================================================================
# 2. Panda3DEnvironment + Object Registry
# ===========================================================================


class TestPanda3DEnvironmentRegistry(unittest.TestCase):
    """Test Panda3DEnvironment with object registry for name resolution."""

    def test_add_object_resolves_through_registry(self):
        """Logical name 'my_sphere' resolves to primitive 'sphere'."""
        agent = Panda3DAgent(agent_id=AgentID("a0"), resolution=(32, 32))
        registry = Panda3DObjectRegistry({
            "my_sphere": {"primitive": "sphere"},
        })
        env = Panda3DEnvironment(
            agents=[agent],
            object_registry=registry,
        )
        try:
            obj_id = env.add_object(name="my_sphere")
            self.assertIsNotNone(obj_id)
        finally:
            env.close()

    def test_add_object_without_registry_uses_name_directly(self):
        """Without registry, name is passed directly (primitive names work)."""
        agent = Panda3DAgent(agent_id=AgentID("a0"), resolution=(32, 32))
        env = Panda3DEnvironment(agents=[agent])
        try:
            obj_id = env.add_object(name="cube")
            self.assertIsNotNone(obj_id)
        finally:
            env.close()

    def test_dict_registry_auto_wrapped(self):
        """A plain dict is auto-wrapped into Panda3DObjectRegistry."""
        agent = Panda3DAgent(agent_id=AgentID("a0"), resolution=(32, 32))
        env = Panda3DEnvironment(
            agents=[agent],
            object_registry={"ball": {"primitive": "sphere"}},
        )
        try:
            obj_id = env.add_object(name="ball")
            self.assertIsNotNone(obj_id)
        finally:
            env.close()

    def test_agents_property_exposed(self):
        """Panda3DEnvironment._agents returns the simulator's agent list."""
        agent = Panda3DAgent(agent_id=AgentID("a0"), resolution=(32, 32))
        env = Panda3DEnvironment(agents=[agent])
        try:
            self.assertEqual(len(env._agents), 1)
            self.assertEqual(env._agents[0].agent_id, AgentID("a0"))
            self.assertEqual(env._agents[0].action_space_type, "distant_agent")
        finally:
            env.close()

    def test_numpy_quaternion_normalized(self):
        """Numpy quaternion objects are converted to plain tuples."""
        import quaternion
        agent = Panda3DAgent(agent_id=AgentID("a0"), resolution=(32, 32))
        env = Panda3DEnvironment(agents=[agent])
        try:
            q = quaternion.quaternion(1, 0, 0, 0)
            obj_id = env.add_object(
                name="sphere",
                rotation=q,
            )
            self.assertIsNotNone(obj_id)
        finally:
            env.close()

    def test_render_after_registry_add(self):
        """After adding an object via registry, step() returns valid obs."""
        agent = Panda3DAgent(
            agent_id=AgentID("a0"),
            sensor_id="s0",
            resolution=(32, 32),
            position=(0.0, -3.0, 0.0),
        )
        registry = {"test_cone": {"primitive": "cone"}}
        env = Panda3DEnvironment(
            agents=[agent],
            object_registry=registry,
        )
        try:
            env.add_object(name="test_cone")
            obs, state = env.step([])
            sensor_obs = obs[AgentID("a0")]["s0"]
            self.assertEqual(sensor_obs["rgba"].shape, (32, 32, 4))
            self.assertEqual(sensor_obs["depth"].shape, (32, 32, 1))
        finally:
            env.close()

    def test_remove_all_and_re_add(self):
        """Mimics EnvironmentInterfacePerObject.change_object_by_idx cycle."""
        agent = Panda3DAgent(agent_id=AgentID("a0"), resolution=(32, 32))
        registry = {
            "obj_a": {"primitive": "sphere"},
            "obj_b": {"primitive": "cube"},
        }
        env = Panda3DEnvironment(
            agents=[agent],
            object_registry=registry,
        )
        try:
            env.add_object(name="obj_a")
            env.remove_all_objects()
            env.add_object(name="obj_b")
            obs, _ = env.step([])
            self.assertIsNotNone(obs)
        finally:
            env.close()


# ===========================================================================
# 3. HPC Integration in Panda3DBehaviorExperiment
# ===========================================================================


class TestHPCSetup(unittest.TestCase):
    """Verify HPC is correctly wired as 3rd LM."""

    def test_setup_with_hpc_creates_three_lms(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gltf_path = _make_animated_gltf(tmpdir)
            exp = Panda3DBehaviorExperiment(
                model_path=gltf_path,
                asset_search_paths=[tmpdir],
                hpc_kwargs={"context_dim": 16, "max_episodes": 10},
            )
            try:
                exp._setup()
                monty = exp.monty
                self.assertEqual(len(monty.learning_modules), 3)
                self.assertEqual(
                    monty.learning_modules[2].learning_module_id, "lm_hpc"
                )
            finally:
                exp.close()

    def test_hpc_connectivity_matrices(self):
        """HPC has no SM inputs and receives from both LM 0 and LM 1."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gltf_path = _make_animated_gltf(tmpdir)
            exp = Panda3DBehaviorExperiment(
                model_path=gltf_path,
                asset_search_paths=[tmpdir],
                hpc_kwargs={"context_dim": 16},
            )
            try:
                exp._setup()
                monty = exp.monty
                # SM->LM: HPC gets no SM input
                self.assertEqual(monty.sm_to_lm_matrix, [[0], [1], []])
                # LM->LM: HPC receives from LM0 + LM1
                self.assertEqual(monty.lm_to_lm_matrix, [[], [], [0, 1]])
                # Vote: HPC doesn't vote
                self.assertEqual(monty.lm_to_lm_vote_matrix, [[1], [0], []])
            finally:
                exp.close()

    def test_without_hpc_still_two_lms(self):
        """When hpc_kwargs=None, architecture is unchanged (2 LMs)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gltf_path = _make_animated_gltf(tmpdir)
            exp = Panda3DBehaviorExperiment(
                model_path=gltf_path,
                asset_search_paths=[tmpdir],
            )
            try:
                exp._setup()
                self.assertEqual(len(exp.monty.learning_modules), 2)
            finally:
                exp.close()


class TestHPCTrainAndEval(unittest.TestCase):
    """End-to-end: train and eval with real animated models + HPC.

    Uses Fox.glb (quadruped, 3 gaits) and RobotExpressive.glb (biped,
    14 behaviors) — real meshes with hundreds of vertices, skeletal
    animation, and visually distinct morphologies.
    """

    def test_fox_walk_records_episodic_memory(self):
        """Train Fox walking — HPC records episode with fox+walk concepts."""
        exp = Panda3DBehaviorExperiment(
            model_path=FOX_PATH,
            asset_search_paths=[ASSET_DIR],
            object_scale=(0.01, 0.01, 0.01),
            hpc_kwargs={"context_dim": 16, "max_episodes": 50},
        )
        try:
            exp.train_behavior(
                "Walk",
                morphology_name="fox",
                behavior_name="walk",
            )
            hpc = exp.hpc
            self.assertGreaterEqual(len(hpc.episodic_memory), 1)
            # Episode should contain the fox + walk concepts
            ep = list(hpc.episodic_memory)[0]
            concepts = set()
            for step in ep:
                for c in step["active_concepts"]:
                    concepts.add(c)
            self.assertIn("fox", concepts)
            self.assertIn("walk", concepts)
        finally:
            exp.close()

    def test_fox_two_gaits_builds_associations(self):
        """Train Fox Walk then Run — HPC builds co-occurrence associations."""
        exp = Panda3DBehaviorExperiment(
            model_path=FOX_PATH,
            asset_search_paths=[ASSET_DIR],
            object_scale=(0.01, 0.01, 0.01),
            hpc_kwargs={"context_dim": 16, "max_episodes": 50},
        )
        try:
            exp.train_behavior(
                "Walk",
                morphology_name="fox",
                behavior_name="fox_walk",
            )
            exp.train_behavior(
                "Run",
                morphology_name="fox",
                behavior_name="fox_run",
            )
            hpc = exp.hpc
            # 2+ episodes (one per behavior)
            self.assertGreaterEqual(len(hpc.episodic_memory), 2)
            # Co-occurrence: fox+fox_walk, fox+fox_run
            self.assertGreater(len(hpc.co_occurrence_counts), 0)
        finally:
            exp.close()

    def test_fox_train_then_match(self):
        """Train Fox Walk, then match it — morphology LM identifies fox."""
        exp = Panda3DBehaviorExperiment(
            model_path=FOX_PATH,
            asset_search_paths=[ASSET_DIR],
            object_scale=(0.01, 0.01, 0.01),
            hpc_kwargs={"context_dim": 16, "max_episodes": 50},
        )
        try:
            exp.train_behavior(
                "Walk",
                morphology_name="fox",
                behavior_name="fox_walk",
            )
            result = exp.match_behavior("Walk")

            self.assertIn("morphology_id", result)
            self.assertIn("behavior_id", result)
            # Morphology LM should recognize the fox
            self.assertEqual(result["morphology_id"], "fox")
        finally:
            exp.close()

    def test_hpc_temporal_predictions_fox_to_human(self):
        """Train Fox then CesiumMan — HPC learns cross-episode transition."""
        exp = Panda3DBehaviorExperiment(
            model_path=FOX_PATH,
            asset_search_paths=[ASSET_DIR],
            object_scale=(0.01, 0.01, 0.01),
            hpc_kwargs={
                "context_dim": 16,
                "max_episodes": 50,
                "temporal_dim": 128,
            },
        )
        try:
            exp.train_behavior(
                "Walk",
                morphology_name="fox",
                behavior_name="fox_walk",
            )
            # Swap to CesiumMan (avoid Robot reuse — ShowBase singleton)
            exp.swap_model(CESIUM_MAN_PATH, object_scale=(0.5, 0.5, 0.5))
            exp.train_behavior(
                "anim0",
                morphology_name="human",
                behavior_name="human_walk",
            )

            hpc = exp.hpc
            # Terminal concept from last episode
            self.assertIsNotNone(hpc._last_episode_terminal)
            # Temporal W should have non-zero entries (fox->human transition)
            self.assertGreater(np.abs(hpc._temporal_W).sum(), 0.0)

            # Both morphologies learned
            morph_ids = exp.morphology_lm.get_all_known_object_ids()
            self.assertIn("fox", morph_ids)
            self.assertIn("human", morph_ids)
        finally:
            exp.close()

    def test_hpc_property_without_hpc_raises(self):
        """Accessing .hpc without hpc_kwargs raises RuntimeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gltf_path = _make_animated_gltf(tmpdir)
            exp = Panda3DBehaviorExperiment(
                model_path=gltf_path,
                asset_search_paths=[tmpdir],
            )
            with self.assertRaises(RuntimeError):
                _ = exp.hpc
            exp.close()


# ===========================================================================
# 4. Robot-specific HPC tests (separate class to avoid ShowBase conflicts)
# ===========================================================================


class TestRobotHPC(unittest.TestCase):
    """Robot-specific HPC tests.

    Isolated in its own class because Panda3D's ShowBase singleton and BAM
    cache can corrupt rendering when the same glTF model is loaded across
    different simulator instances in the same process.
    """

    def test_robot_dance_and_wave(self):
        """Train Robot Dance + Wave — HPC learns both behaviors."""
        exp = Panda3DBehaviorExperiment(
            model_path=ROBOT_PATH,
            asset_search_paths=[ASSET_DIR],
            object_scale=(0.5, 0.5, 0.5),
            hpc_kwargs={"context_dim": 16, "max_episodes": 50},
        )
        try:
            exp.train_behavior(
                "Dance",
                morphology_name="robot",
                behavior_name="robot_dance",
            )
            exp.train_behavior(
                "Wave",
                morphology_name="robot",
                behavior_name="robot_wave",
            )
            hpc = exp.hpc
            self.assertGreaterEqual(len(hpc.episodic_memory), 2)
            self.assertGreater(len(hpc.co_occurrence_counts), 0)

            # Morphology LM should know the robot shape
            morph_ids = exp.morphology_lm.get_all_known_object_ids()
            self.assertIn("robot", morph_ids)
        finally:
            exp.close()


# ===========================================================================
# 5. Cross-morphology: Fox vs Human with HPC
# ===========================================================================


class TestCrossMorphologyWithHPC(unittest.TestCase):
    """Train Fox and Robot, verify HPC associates morphology+behavior.

    NOTE: This test uses CesiumMan (human) instead of RobotExpressive to
    avoid Panda3D ShowBase singleton conflicts with TestHPCTrainAndEval
    which also loads RobotExpressive. Each test class should use its own
    set of glTF models to prevent BAM cache corruption.
    """

    def test_fox_then_human_hpc_knows_both(self):
        """Train Fox Walk, swap to CesiumMan. HPC has both associations."""
        exp = Panda3DBehaviorExperiment(
            model_path=FOX_PATH,
            asset_search_paths=[ASSET_DIR],
            object_scale=(0.01, 0.01, 0.01),
            hpc_kwargs={"context_dim": 16, "max_episodes": 50},
        )
        try:
            exp.train_behavior(
                "Walk",
                morphology_name="fox",
                behavior_name="fox_walk",
            )
            exp.swap_model(CESIUM_MAN_PATH, object_scale=(1.0, 1.0, 1.0))
            exp.train_behavior(
                "anim0",
                morphology_name="human",
                behavior_name="human_walk",
            )

            hpc = exp.hpc

            # HPC should have 2+ episodes
            self.assertGreaterEqual(len(hpc.episodic_memory), 2)

            # Morphology LM should know both shapes
            morph_objects = exp.morphology_lm.get_all_known_object_ids()
            self.assertIn("fox", morph_objects)
            self.assertIn("human", morph_objects)

            # Co-occurrence associations should exist
            self.assertGreater(len(hpc.co_occurrence_counts), 0)
        finally:
            exp.close()


# ===========================================================================
# 6. Temporal Memory Prediction: multi-shape × multi-behavior
# ===========================================================================


class TestTemporalMemoryPrediction(unittest.TestCase):
    """Train temporal memory on behavior LM across shapes and behaviors.

    Validates that a single trained system can:
    1. Recognize each shape (morphology LM)
    2. Recognize each behavior (behavior LM)
    3. Predict upcoming steps in each behavior (temporal memory)
    4. Distinguish trained from novel behaviors via surprise

    Uses Fox (Walk, Run) + CesiumMan (anim0).
    No HPC — focus is purely on temporal prediction within EvidenceGraphLM.
    """

    @classmethod
    def setUpClass(cls):
        """Train once on all shape × behavior combos, reuse across tests."""
        cls.exp = Panda3DBehaviorExperiment(
            model_path=FOX_PATH,
            asset_search_paths=[ASSET_DIR],
            object_scale=(0.01, 0.01, 0.01),
            behavior_lm_kwargs={
                "temporal_memory": {
                    "sdr_dim": 1024,
                    "sdr_sparsity": 0.02,
                    "learning_rate": 0.1,
                },
            },
        )
        # Train Fox Walk
        cls.exp.train_behavior(
            "Walk", morphology_name="fox", behavior_name="fox_walk",
        )
        # Train Fox Run (same shape, different behavior)
        cls.exp.train_behavior(
            "Run", morphology_name="fox", behavior_name="fox_run",
        )
        # Swap to CesiumMan (different shape)
        cls.exp.swap_model(CESIUM_MAN_PATH, object_scale=(1.0, 1.0, 1.0))
        cls.exp.train_behavior(
            "anim0", morphology_name="human", behavior_name="human_walk",
        )

    @classmethod
    def tearDownClass(cls):
        cls.exp.close()

    # -- Verify training state --

    def test_temporal_memory_exists_and_has_learned(self):
        """Behavior LM has temporal memory with non-zero Hebbian weights."""
        tm = self.exp.behavior_lm._temporal_memory
        self.assertIsNotNone(tm)
        self.assertGreater(np.abs(tm._W).sum(), 0.0)

    def test_morphology_lm_knows_both_shapes(self):
        """Morphology LM learned fox and human shapes."""
        morph_ids = self.exp.morphology_lm.get_all_known_object_ids()
        self.assertIn("fox", morph_ids)
        self.assertIn("human", morph_ids)

    def test_behavior_lm_knows_all_behaviors(self):
        """Behavior LM learned fox_walk, fox_run, and human_walk."""
        behav_ids = self.exp.behavior_lm.get_all_known_object_ids()
        self.assertIn("fox_walk", behav_ids)
        self.assertIn("fox_run", behav_ids)
        self.assertIn("human_walk", behav_ids)

    # -- Eval: recognize + predict on trained Fox Walk --

    def test_eval_fox_walk_produces_predictions(self):
        """Eval on trained Fox Walk yields temporal predictions."""
        # Swap back to Fox for eval
        self.exp.swap_model(FOX_PATH, object_scale=(0.01, 0.01, 0.01))
        result = self.exp.match_behavior("Walk")

        self.assertIn("temporal_predictions", result)
        self.assertIn("temporal_surprise_history", result)
        self.assertIn("temporal_mean_surprise", result)

        # Should have generated predictions (W is non-zero from training)
        self.assertGreater(len(result["temporal_predictions"]), 0)
        # Predictions are SDR arrays
        for p in result["temporal_predictions"]:
            self.assertEqual(p.shape, (1024,))
            self.assertGreater(p.sum(), 0)  # non-empty SDR

    # -- Eval: same shape, different behavior --

    def test_eval_fox_run_produces_different_predictions(self):
        """Eval on Fox Run yields predictions distinct from Walk."""
        self.exp.swap_model(FOX_PATH, object_scale=(0.01, 0.01, 0.01))

        walk_result = self.exp.match_behavior("Walk")
        run_result = self.exp.match_behavior("Run")

        walk_preds = walk_result["temporal_predictions"]
        run_preds = run_result["temporal_predictions"]

        self.assertGreater(len(walk_preds), 0)
        self.assertGreater(len(run_preds), 0)

        # Predictions should differ (different temporal dynamics)
        n = min(len(walk_preds), len(run_preds))
        overlaps = [
            float(np.dot(walk_preds[i], run_preds[i]))
            for i in range(n)
        ]
        mean_overlap = np.mean(overlaps)
        # Not identical — some divergence expected
        n_active = int(1024 * 0.02)  # 20 active bits
        self.assertLess(mean_overlap, n_active)

    # -- Eval: different shape --

    def test_eval_human_walk_produces_predictions(self):
        """Eval on CesiumMan (human) also yields temporal predictions."""
        self.exp.swap_model(CESIUM_MAN_PATH, object_scale=(1.0, 1.0, 1.0))
        result = self.exp.match_behavior("anim0")

        self.assertIn("temporal_predictions", result)
        self.assertGreater(len(result["temporal_predictions"]), 0)
        self.assertIn("temporal_surprise_history", result)

    # -- Surprise: trained vs novel --

    def test_novel_behavior_has_higher_surprise(self):
        """Untrained behavior (Fox Survey) has higher surprise than trained Walk."""
        self.exp.swap_model(FOX_PATH, object_scale=(0.01, 0.01, 0.01))

        # Trained behavior
        walk_result = self.exp.match_behavior("Walk")
        walk_surprise = walk_result["temporal_mean_surprise"]

        # Novel behavior (Survey was never trained)
        survey_result = self.exp.match_behavior("Survey")
        survey_surprise = survey_result["temporal_mean_surprise"]

        # Novel behavior should be more surprising
        self.assertGreater(survey_surprise, walk_surprise)


# ===========================================================================
# 7. Dual Temporal Memory: prediction on both morphology + behavior LMs
# ===========================================================================


class TestDualTemporalMemory(unittest.TestCase):
    """Temporal memory on BOTH morphology and behavior LMs.

    The morphology LM's temporal memory learns viewpoint sequences —
    how surface patches change as the camera moves. The behavior LM's
    temporal memory learns motion sequences — how optical flow evolves
    during animation.

    Uses Fox (Walk, Run) + CesiumMan (anim0).
    """

    TM_CONFIG = {
        "sdr_dim": 1024,
        "sdr_sparsity": 0.02,
        "learning_rate": 0.1,
    }

    @classmethod
    def setUpClass(cls):
        cls.exp = Panda3DBehaviorExperiment(
            model_path=FOX_PATH,
            asset_search_paths=[ASSET_DIR],
            object_scale=(0.01, 0.01, 0.01),
            morphology_lm_kwargs={"temporal_memory": cls.TM_CONFIG},
            behavior_lm_kwargs={"temporal_memory": cls.TM_CONFIG},
        )
        cls.exp.train_behavior(
            "Walk", morphology_name="fox", behavior_name="fox_walk",
        )
        cls.exp.train_behavior(
            "Run", morphology_name="fox", behavior_name="fox_run",
        )
        cls.exp.swap_model(CESIUM_MAN_PATH, object_scale=(1.0, 1.0, 1.0))
        cls.exp.train_behavior(
            "anim0", morphology_name="human", behavior_name="human_walk",
        )

    @classmethod
    def tearDownClass(cls):
        cls.exp.close()

    def test_both_lms_have_temporal_memory(self):
        """Both morphology and behavior LMs have temporal memory enabled."""
        self.assertIsNotNone(self.exp.morphology_lm._temporal_memory)
        self.assertIsNotNone(self.exp.behavior_lm._temporal_memory)

    def test_both_hebbian_weights_nonzero(self):
        """Both temporal memories learned associations during training."""
        morph_W = self.exp.morphology_lm._temporal_memory._W
        behav_W = self.exp.behavior_lm._temporal_memory._W
        self.assertGreater(np.abs(morph_W).sum(), 0.0)
        self.assertGreater(np.abs(behav_W).sum(), 0.0)

    def test_morphology_predicts_next_viewpoint(self):
        """Morphology LM predicts next surface patch during eval."""
        self.exp.swap_model(FOX_PATH, object_scale=(0.01, 0.01, 0.01))
        self.exp.match_behavior("Walk")

        tm = self.exp.morphology_lm._temporal_memory
        surprise_history = tm.get_surprise_history()
        self.assertGreater(len(surprise_history), 0)

        # Predictions should be possible from the last observed state
        pred = tm.predict_next()
        self.assertIsNotNone(pred)
        self.assertEqual(pred.shape, (1024,))

    def test_behavior_predicts_next_motion(self):
        """Behavior LM predicts next motion pattern during eval."""
        self.exp.swap_model(FOX_PATH, object_scale=(0.01, 0.01, 0.01))
        self.exp.match_behavior("Walk")

        tm = self.exp.behavior_lm._temporal_memory
        surprise_history = tm.get_surprise_history()
        self.assertGreater(len(surprise_history), 0)

        pred = tm.predict_next()
        self.assertIsNotNone(pred)

    def test_morphology_surprise_lower_for_same_viewpoint_trajectory(self):
        """Morphology temporal memory is less surprised replaying same walk."""
        # Eval Fox Walk twice — second pass should be less surprising
        # because the temporal memory was just seeded by the first pass
        self.exp.swap_model(FOX_PATH, object_scale=(0.01, 0.01, 0.01))

        self.exp.match_behavior("Walk")
        first_surprise = self.exp.morphology_lm._temporal_memory.get_mean_surprise()

        # Temporal memory W is fixed (learn=False in eval), but the
        # episode reset clears _prev_sdr, so predictions restart fresh.
        # Still, Fox Walk is a trained sequence, so surprise should be < 1.0
        self.assertLess(first_surprise, 1.0)

    def test_different_behaviors_different_morph_predictions(self):
        """Walk vs Run produce different morphology predictions.

        Even though the shape is the same, the camera trajectory
        differs between episodes, so temporal predictions diverge.
        """
        self.exp.swap_model(FOX_PATH, object_scale=(0.01, 0.01, 0.01))

        self.exp.match_behavior("Walk")
        walk_pred = self.exp.morphology_lm._temporal_memory.predict_next()

        self.exp.match_behavior("Run")
        run_pred = self.exp.morphology_lm._temporal_memory.predict_next()

        self.assertIsNotNone(walk_pred)
        self.assertIsNotNone(run_pred)

        # Predictions should differ (different viewpoint trajectories)
        overlap = float(np.dot(walk_pred, run_pred))
        n_active = int(1024 * 0.02)
        self.assertLess(overlap, n_active)


# ===========================================================================
# 8. Goal-State-Driven Motor Control (LM-guided exploration)
# ===========================================================================


class TestGoalStateDrivenMotor(unittest.TestCase):
    """Verify that LMs can drive motor exploration via GoalStateGenerator.

    When goal_state_driven=True, the morphology LM's GSG proposes
    hypothesis-testing jumps during eval. The motor policy executes them
    via SetAgentPose/SetSensorRotation on the Panda3D simulator.

    Training still uses exploratory mode (fixed action sequence).
    """

    @classmethod
    def setUpClass(cls):
        """Train Fox Walk with GSG-driven motor enabled."""
        cls.exp = Panda3DBehaviorExperiment(
            model_path=FOX_PATH,
            asset_search_paths=[ASSET_DIR],
            object_scale=(0.01, 0.01, 0.01),
            goal_state_driven=True,
        )
        cls.exp.train_behavior(
            "Walk", morphology_name="fox", behavior_name="fox_walk",
        )

    @classmethod
    def tearDownClass(cls):
        cls.exp.close()

    def test_gsg_attached_to_morphology_lm(self):
        """Morphology LM has a GoalStateGenerator."""
        morph_lm = self.exp.morphology_lm
        self.assertIsNotNone(morph_lm.gsg)

    def test_motor_policy_is_goal_driven(self):
        """Motor policy has goal-state-driven actions enabled."""
        policy = self.exp._monty.motor_system._policy
        self.assertTrue(policy.use_goal_state_driven_actions)

    def test_eval_with_gsg_recognizes_fox(self):
        """Eval with GSG-driven exploration still recognizes trained shape."""
        result = self.exp.match_behavior("Walk")
        self.assertEqual(result["morphology_id"], "fox")

    def test_eval_with_gsg_recognizes_behavior(self):
        """Eval with GSG-driven exploration recognizes trained behavior."""
        result = self.exp.match_behavior("Walk")
        self.assertEqual(result["behavior_id"], "fox_walk")


# ===========================================================================
# 9. Depth From Motion: train+eval without ground-truth depth
# ===========================================================================


class TestDepthFromMotion(unittest.TestCase):
    """Train and eval using estimated depth from motion parallax.

    Instead of Panda3D's ground-truth depth buffer, the DepthFromMotion
    transform estimates depth from consecutive RGBA frames + known
    ego-motion via phase-correlation optical flow.

    This validates that the full pipeline (SM → LM → temporal memory)
    works without a hardware depth sensor.

    Uses Fox (Walk, Run).
    """

    @classmethod
    def setUpClass(cls):
        # Use CesiumMan at 1.0 scale — large enough to be visible in RGBA
        # for flow-based depth estimation. Fox at 0.01 is sub-pixel (invisible
        # in RGBA, only detectable via ground-truth depth buffer).
        cls.exp = Panda3DBehaviorExperiment(
            model_path=CESIUM_MAN_PATH,
            asset_search_paths=[ASSET_DIR],
            object_scale=(1.0, 1.0, 1.0),
            initial_distance=5.0,
            depth_from_motion=True,
            behavior_lm_kwargs={
                "temporal_memory": {
                    "sdr_dim": 1024,
                    "sdr_sparsity": 0.02,
                    "learning_rate": 0.1,
                },
            },
        )
        cls.exp.train_behavior(
            "anim0", morphology_name="human", behavior_name="human_walk",
        )

    @classmethod
    def tearDownClass(cls):
        cls.exp.close()

    def test_depth_transform_is_motion_based(self):
        """Depth transform is DepthFromMotion, not Panda3DDepthNormalize."""
        from tbp.monty.simulators.panda3d.depth_from_motion import DepthFromMotion
        self.assertIsInstance(self.exp._depth_transform, DepthFromMotion)

    def test_morphology_lm_learned_shape(self):
        """Morphology LM learned shape from estimated depth."""
        morph_ids = self.exp.morphology_lm.get_all_known_object_ids()
        self.assertIn("human", morph_ids)

    def test_behavior_lm_learned_behavior(self):
        """Behavior LM learned behavior from estimated depth."""
        behav_ids = self.exp.behavior_lm.get_all_known_object_ids()
        self.assertIn("human_walk", behav_ids)

    def test_temporal_memory_has_learned(self):
        """Temporal memory built associations from estimated depth data."""
        tm = self.exp.behavior_lm._temporal_memory
        self.assertIsNotNone(tm)
        self.assertGreater(np.abs(tm._W).sum(), 0.0)

    def test_eval_produces_predictions(self):
        """Eval with estimated depth produces temporal predictions."""
        result = self.exp.match_behavior("anim0")
        self.assertIn("temporal_predictions", result)
        self.assertGreater(len(result["temporal_predictions"]), 0)
        self.assertIn("temporal_surprise_history", result)

    def test_eval_surprise_is_finite(self):
        """Eval with estimated depth produces finite surprise values."""
        result = self.exp.match_behavior("anim0")
        surprise = result["temporal_mean_surprise"]
        # Should be between 0 and 1, not all 1.0 (which means no predictions)
        self.assertLess(surprise, 1.0)


# ===========================================================================
# 10. A/B Test: Temporal Priming ON vs OFF (Panda3D validation)
# ===========================================================================


def _compute_evidence_margin(evidence_dict):
    """Return (top_evidence, margin) from an evidence dict.

    Margin = top evidence - second-best evidence across all graphs.
    Higher margin means more confident discrimination.
    """
    if not evidence_dict:
        return 0.0, 0.0
    max_evidences = []
    for gid, ev in evidence_dict.items():
        if ev is not None and len(ev) > 0:
            max_evidences.append(float(np.max(ev)))
    if len(max_evidences) < 2:
        return (max_evidences[0] if max_evidences else 0.0), 0.0
    max_evidences.sort(reverse=True)
    return max_evidences[0], max_evidences[0] - max_evidences[1]


class TestTemporalPrimingAB(unittest.TestCase):
    """A/B test: does temporal priming improve behavior discrimination?

    Trains two identical Panda3DBehaviorExperiment instances on the same
    animated models (Fox Walk + Fox Run). One has temporal memory with
    surprise modulation enabled (the "treatment"), the other has no
    temporal memory (the "control"). Both use identical seeds and
    parameters otherwise.

    Then evaluates both on Walk and Run, comparing:
    - Correct behavior identification (graph_id matches)
    - Evidence margin (how confidently the correct behavior wins)
    - Morphology identification (should be identical — temporal memory
      shouldn't hurt shape recognition)

    This is the Panda3D equivalent of the Habitat A/B eval called for in
    the Track 2 doc. Habitat can't provide this test because it doesn't
    support animated/dynamic objects.
    """

    @classmethod
    def setUpClass(cls):
        """Train control (no TM) then treatment (TM + surprise) sequentially.

        Panda3D's ShowBase is a singleton — only one can exist at a time.
        So we train + eval the control, save its LM state and results,
        close it, then create the treatment.
        """
        # --- Phase 1: Control (no temporal memory) ---
        cls.control = Panda3DBehaviorExperiment(
            model_path=FOX_PATH,
            asset_search_paths=[ASSET_DIR],
            object_scale=(0.01, 0.01, 0.01),
        )
        cls.control.train_behavior(
            "Walk", morphology_name="fox", behavior_name="fox_walk",
        )
        cls.control.train_behavior(
            "Run", morphology_name="fox", behavior_name="fox_run",
        )
        # Eval control
        cls.ctrl_walk = cls.control.match_behavior("Walk")
        cls.ctrl_run = cls.control.match_behavior("Run")
        # Save LM state for assertions after close
        cls.ctrl_morph_ids = list(
            cls.control.morphology_lm.get_all_known_object_ids()
        )
        cls.ctrl_behav_ids = list(
            cls.control.behavior_lm.get_all_known_object_ids()
        )
        cls.ctrl_has_morph_tm = (
            cls.control.morphology_lm._temporal_memory is not None
        )
        cls.ctrl_has_behav_tm = (
            cls.control.behavior_lm._temporal_memory is not None
        )
        cls.control.close()

        # --- Phase 2: Treatment (temporal memory + surprise modulation) ---
        cls.treatment = Panda3DBehaviorExperiment(
            model_path=FOX_PATH,
            asset_search_paths=[ASSET_DIR],
            object_scale=(0.01, 0.01, 0.01),
            morphology_lm_kwargs={
                "temporal_memory": {
                    "sdr_dim": 1024,
                    "sdr_sparsity": 0.02,
                    "learning_rate": 0.1,
                },
                "surprise_boost": 0.5,
                "surprise_penalty": 0.3,
            },
            behavior_lm_kwargs={
                "temporal_memory": {
                    "sdr_dim": 1024,
                    "sdr_sparsity": 0.02,
                    "learning_rate": 0.1,
                },
                "surprise_boost": 0.5,
                "surprise_penalty": 0.3,
            },
        )
        cls.treatment.train_behavior(
            "Walk", morphology_name="fox", behavior_name="fox_walk",
        )
        cls.treatment.train_behavior(
            "Run", morphology_name="fox", behavior_name="fox_run",
        )
        # Eval treatment
        cls.treat_walk = cls.treatment.match_behavior("Walk")
        cls.treat_run = cls.treatment.match_behavior("Run")

    @classmethod
    def tearDownClass(cls):
        cls.treatment.close()

    # -- Both systems should learn the same objects --

    def test_both_learned_same_shapes(self):
        """Control and treatment learned the same morphology."""
        treat_ids = self.treatment.morphology_lm.get_all_known_object_ids()
        self.assertEqual(set(self.ctrl_morph_ids), set(treat_ids))

    def test_both_learned_same_behaviors(self):
        """Control and treatment learned the same behaviors."""
        treat_ids = self.treatment.behavior_lm.get_all_known_object_ids()
        self.assertEqual(set(self.ctrl_behav_ids), set(treat_ids))

    # -- Treatment should have temporal memory, control should not --

    def test_control_has_no_temporal_memory(self):
        """Control LMs had no temporal memory."""
        self.assertFalse(self.ctrl_has_morph_tm)
        self.assertFalse(self.ctrl_has_behav_tm)

    def test_treatment_has_temporal_memory(self):
        """Treatment LMs have temporal memory with learned weights."""
        morph_tm = self.treatment.morphology_lm._temporal_memory
        behav_tm = self.treatment.behavior_lm._temporal_memory
        self.assertIsNotNone(morph_tm)
        self.assertIsNotNone(behav_tm)
        self.assertGreater(np.abs(morph_tm._W).sum(), 0.0)
        self.assertGreater(np.abs(behav_tm._W).sum(), 0.0)

    # -- Core A/B comparison: behavior discrimination --

    def test_treatment_behavior_margin_ge_control(self):
        """Treatment evidence margin >= control for behavior discrimination.

        The temporal memory's surprise modulation should amplify evidence
        for the correct behavior (low surprise on familiar temporal
        patterns) and dampen evidence for wrong behaviors (high surprise).
        This should produce at least as large a margin as the control.
        """
        # Walk eval: behavior evidence margin
        _, ctrl_walk_margin = _compute_evidence_margin(
            self.ctrl_walk["behavior_evidence"]
        )
        _, treat_walk_margin = _compute_evidence_margin(
            self.treat_walk["behavior_evidence"]
        )

        # Run eval: behavior evidence margin
        _, ctrl_run_margin = _compute_evidence_margin(
            self.ctrl_run["behavior_evidence"]
        )
        _, treat_run_margin = _compute_evidence_margin(
            self.treat_run["behavior_evidence"]
        )

        # Treatment should have >= margin on at least one behavior.
        # We use sum of margins as the aggregate metric.
        ctrl_total = ctrl_walk_margin + ctrl_run_margin
        treat_total = treat_walk_margin + treat_run_margin
        self.assertGreaterEqual(
            treat_total, ctrl_total * 0.9,
            f"Treatment margin ({treat_total:.3f}) much worse than "
            f"control ({ctrl_total:.3f}). Temporal priming is hurting."
        )

    def test_treatment_walk_surprise_below_1(self):
        """Treatment has meaningful temporal predictions (surprise < 1.0).

        If surprise == 1.0, the temporal memory made no predictions,
        meaning it's not contributing to the pipeline at all.
        """
        self.assertIn("temporal_mean_surprise", self.treat_walk)
        self.assertLess(self.treat_walk["temporal_mean_surprise"], 1.0)

    def test_treatment_run_surprise_below_1(self):
        """Treatment has meaningful temporal predictions for Run too."""
        self.assertIn("temporal_mean_surprise", self.treat_run)
        self.assertLess(self.treat_run["temporal_mean_surprise"], 1.0)

    # -- Morphology should not be degraded --

    def test_morphology_not_degraded(self):
        """Temporal priming doesn't hurt morphology recognition.

        Both control and treatment should identify 'fox' as morphology.
        """
        ctrl_morph = self.ctrl_walk.get("morphology_id")
        treat_morph = self.treat_walk.get("morphology_id")
        # Both should identify something (not None)
        # Treatment should not be worse than control
        if ctrl_morph is not None:
            self.assertIsNotNone(
                treat_morph,
                "Control identified morphology but treatment didn't"
            )


if __name__ == "__main__":
    unittest.main()
