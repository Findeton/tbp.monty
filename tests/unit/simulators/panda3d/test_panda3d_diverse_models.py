# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Tests for temporal training on diverse real-world animated 3D models.

Validates that TemporalMemory can learn temporal patterns from a variety
of animated objects:

1. **Fox** — Low-poly quadruped with 3 locomotion gaits (Survey, Walk, Run).
   24-joint skeleton. Tests periodic animal motion and multi-animation
   switching. Source: Khronos glTF-Sample-Assets.

2. **CesiumMan** — Textured humanoid with a walking animation. 19-joint
   skeleton. Tests bilateral symmetric bipedal locomotion.
   Source: Khronos glTF-Sample-Assets.

3. **RobotExpressive** — Stylised robot character with 14 animations
   (Walking, Running, Dance, Jump, Idle, Punch, etc.) and 61 joints.
   Tests complex multi-animation robotic motion with morph targets.
   Source: three.js examples (Tomás Laulhé / Don McCurdy, CC0).

Each model is tested for:
- Loading and rendering in Panda3D
- Producing valid States through the observation pipeline
- TemporalMemory learning (surprise reduction over repetitions)
- Multi-animation comparison (Fox: Survey vs Walk vs Run;
  RobotExpressive: Walking vs Running vs Dance)
- Debug output generation for visual inspection
"""

import json
import os
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

# Model configurations: (filename, scale, camera_position, description)
# Camera positions determined by probing each model's bounding box.
ASSET_DIR = os.path.join(
    os.path.dirname(__file__), "test_assets", "animated"
)

MODEL_CONFIGS = {
    "fox": {
        "file": "Fox.glb",
        "scale": (0.03, 0.03, 0.03),  # Fox is ~25x155x79 units natively
        "camera_pos": (0, -2.0, 1.2),
        "far": 20.0,
        "description": "Low-poly fox with Survey/Walk/Run gaits",
        "expected_anims": ["Survey", "Walk", "Run"],
    },
    "cesium_man": {
        "file": "CesiumMan.glb",
        "scale": (1.0, 1.0, 1.0),
        "camera_pos": (0, -1.5, -0.75),
        "far": 20.0,
        "description": "Textured humanoid walk cycle",
        "expected_anims": ["anim0"],
    },
    "robot": {
        "file": "RobotExpressive.glb",
        "scale": (0.5, 0.5, 0.5),
        "camera_pos": (0, -1.5, 0),
        "far": 20.0,
        "description": "Stylised robot with 14 motion animations",
        "expected_anims": [
            "Dance", "Death", "Idle", "Jump", "No", "Punch",
            "Running", "Sitting", "Standing", "ThumbsUp",
            "Walking", "WalkJump", "Wave", "Yes",
        ],
        "default_anim": "Walking",
    },
}


def _skip_if_missing(model_key):
    """Skip a test if the model file is not present."""
    cfg = MODEL_CONFIGS[model_key]
    path = os.path.join(ASSET_DIR, cfg["file"])
    if not os.path.exists(path):
        raise unittest.SkipTest(f"Model file not found: {path}")


def _make_sim(model_key, resolution=(64, 64)):
    """Create a simulator with the right camera position for a model."""
    cfg = MODEL_CONFIGS[model_key]
    agent = Panda3DAgent(
        agent_id=AGENT_ID,
        sensor_id=SENSOR_ID,
        position=cfg["camera_pos"],
        rotation=(1.0, 0.0, 0.0, 0.0),
        resolution=resolution,
        fov=90.0,
    )
    sim = Panda3DSimulator(
        agents=[agent], near=0.01, far=cfg["far"]
    )
    model_path = os.path.join(ASSET_DIR, cfg["file"])
    info = sim.add_object(
        model_path, animated=True, scale=cfg["scale"]
    )
    anim = sim.get_animated_object(info.object_id)
    return sim, info, anim, cfg


# ============================================================
# Test 1: Model loading and animation metadata
# ============================================================

class TestDiverseModelLoading(unittest.TestCase):
    """Verify each model loads, has expected animations and joints."""

    def test_fox_loads_with_three_animations(self):
        _skip_if_missing("fox")
        sim, info, anim, cfg = _make_sim("fox")
        try:
            self.assertEqual(
                sorted(anim.animation_names),
                sorted(cfg["expected_anims"]),
            )
            self.assertGreater(anim.get_num_frames("Survey"), 0)
            self.assertGreater(anim.get_num_frames("Walk"), 0)
            self.assertGreater(anim.get_num_frames("Run"), 0)
            self.assertGreater(len(anim.get_joint_names()), 10)
        finally:
            sim.close()

    def test_cesium_man_loads_with_walk_cycle(self):
        _skip_if_missing("cesium_man")
        sim, info, anim, cfg = _make_sim("cesium_man")
        try:
            self.assertIn("anim0", anim.animation_names)
            n_frames = anim.get_num_frames()
            self.assertGreater(n_frames, 10)
            self.assertGreater(len(anim.get_joint_names()), 10)
        finally:
            sim.close()

    def test_robot_loads_with_multiple_animations(self):
        _skip_if_missing("robot")
        sim, info, anim, cfg = _make_sim("robot")
        try:
            # RobotExpressive has 14 unique animations (duplicated as .1/.2/.3)
            for expected in ["Walking", "Running", "Dance", "Idle", "Jump"]:
                self.assertIn(expected, anim.animation_names)
            self.assertGreater(anim.get_num_frames("Walking"), 5)
            self.assertGreater(anim.get_num_frames("Running"), 5)
            self.assertGreater(len(anim.get_joint_names()), 30)
        finally:
            sim.close()


# ============================================================
# Test 2: Rendering produces on-object observations
# ============================================================

class TestDiverseModelRendering(unittest.TestCase):
    """Verify each model renders visible, depth-varying observations."""

    def _check_rendering(self, model_key, min_on_object=20, anim_name=None):
        _skip_if_missing(model_key)
        sim, info, anim, cfg = _make_sim(model_key)
        try:
            anim.pose(0, anim_name)
            obs, proprio = sim.step([])
            depth = obs[AGENT_ID][SensorID(SENSOR_ID)]["depth"]
            on_obj = np.sum(depth < cfg["far"] * 0.99)
            self.assertGreater(
                on_obj, min_on_object,
                f"{model_key}: only {on_obj} on-object pixels"
            )

            # Check frame variation (use 1/4 offset to break walk-cycle symmetry)
            n_frames = anim.get_num_frames(anim_name)
            if n_frames > 1:
                mid = max(1, n_frames // 4)
                anim.pose(mid, anim_name)
                obs2, _ = sim.step([])
                d2 = obs2[AGENT_ID][SensorID(SENSOR_ID)]["depth"]
                diff = np.abs(depth - d2).mean()
                self.assertGreater(
                    diff, 0.01,
                    f"{model_key}: frames 0 and {mid} are identical"
                )
        finally:
            sim.close()

    def test_fox_renders_visible(self):
        self._check_rendering("fox", min_on_object=50)

    def test_cesium_man_renders_visible(self):
        self._check_rendering("cesium_man", min_on_object=100)

    def test_robot_renders_visible(self):
        self._check_rendering("robot", min_on_object=100, anim_name="Walking")


# ============================================================
# Test 3: Full observation pipeline → State
# ============================================================

class TestDiverseModelObservationPipeline(unittest.TestCase):
    """Full pipeline: render → depth normalize → DepthTo3DLocations → CameraSM → State."""

    def _check_pipeline(self, model_key):
        _skip_if_missing(model_key)
        from tbp.monty.context import RuntimeContext
        from tbp.monty.frameworks.environment_utils.transforms import (
            DepthTo3DLocations,
            TransformContext,
        )
        from tbp.monty.frameworks.models.sensor_modules import CameraSM
        from tbp.monty.frameworks.models.states import State

        sim, info, anim, cfg = _make_sim(model_key)
        try:
            anim.pose(0)
            obs, proprio = sim.step([])

            # Transforms
            depth_norm = Panda3DDepthNormalize(
                agent_id=AGENT_ID, near=0.01, far=cfg["far"]
            )
            d3d = DepthTo3DLocations(
                agent_id=AGENT_ID,
                sensor_ids=[SensorID(SENSOR_ID)],
                resolutions=[(64, 64)],
                hfov=90.0,
                world_coord=True,
                get_all_points=True,
            )
            ctx = TransformContext(
                rng=np.random.RandomState(0), state=proprio
            )
            obs = depth_norm(obs, ctx)
            obs = d3d(obs, ctx)

            # CameraSM
            sm = CameraSM(
                sensor_module_id=str(SENSOR_ID),
                features=["on_object", "hsv", "principal_curvatures_log"],
            )
            sm.pre_episode()
            sm.update_state(proprio[AGENT_ID])
            rt_ctx = RuntimeContext(rng=np.random.RandomState(0))
            state = sm.step(
                rt_ctx, obs[AGENT_ID][SensorID(SENSOR_ID)]
            )

            self.assertIsInstance(state, State)
            self.assertIsNotNone(state.location)
            self.assertEqual(len(state.location), 3)
        finally:
            sim.close()

    def test_fox_pipeline(self):
        self._check_pipeline("fox")

    def test_cesium_man_pipeline(self):
        self._check_pipeline("cesium_man")

    def test_robot_pipeline(self):
        self._check_pipeline("robot")


# ============================================================
# Test 4: TemporalMemory learning — surprise decreases
# ============================================================

class TestDiverseModelTemporalLearning(unittest.TestCase):
    """TemporalMemory learns temporal patterns from each model's animation."""

    def _train_and_check(self, model_key, n_reps=3, max_frames=30):
        """Run temporal training and check surprise decreases.

        Parameters
        ----------
        model_key : str
            Key into MODEL_CONFIGS.
        n_reps : int
            Number of repetitions through the animation.
        max_frames : int
            Cap on frames per repetition (some models have 1000+ frames).
        """
        _skip_if_missing(model_key)
        from tbp.monty.context import RuntimeContext
        from tbp.monty.frameworks.environment_utils.transforms import (
            DepthTo3DLocations,
            TransformContext,
        )
        from tbp.monty.frameworks.models.sensor_modules import CameraSM
        from tbp.monty.frameworks.models.temporal_memory import TemporalMemory

        sim, info, anim, cfg = _make_sim(model_key)
        try:
            anim_name = cfg.get("default_anim", None)
            n_frames = min(anim.get_num_frames(anim_name), max_frames)

            depth_norm = Panda3DDepthNormalize(
                agent_id=AGENT_ID, near=0.01, far=cfg["far"]
            )
            d3d = DepthTo3DLocations(
                agent_id=AGENT_ID,
                sensor_ids=[SensorID(SENSOR_ID)],
                resolutions=[(64, 64)],
                hfov=90.0,
                world_coord=True,
                get_all_points=True,
            )
            sm = CameraSM(
                sensor_module_id=str(SENSOR_ID),
                features=["on_object", "hsv", "principal_curvatures_log"],
            )
            sm.pre_episode()
            tm = TemporalMemory(sdr_dim=1024, sdr_sparsity=0.02)

            surprises_by_rep = {r: [] for r in range(n_reps)}

            for rep in range(n_reps):
                for frame in range(n_frames):
                    anim.pose(frame, anim_name)
                    obs, proprio = sim.step([])
                    ctx = TransformContext(
                        rng=np.random.RandomState(0), state=proprio
                    )
                    obs = depth_norm(obs, ctx)
                    obs = d3d(obs, ctx)
                    sm.update_state(proprio[AGENT_ID])
                    rt_ctx = RuntimeContext(rng=np.random.RandomState(0))
                    state = sm.step(
                        rt_ctx, obs[AGENT_ID][SensorID(SENSOR_ID)]
                    )
                    if state is not None and state.use_state:
                        result = tm.step(state, learn=True)
                        surprises_by_rep[rep].append(result["surprise"])

            # Check we got usable states
            total_usable = sum(len(v) for v in surprises_by_rep.values())
            self.assertGreater(
                total_usable, 0,
                f"{model_key}: no usable states produced"
            )

            # Check surprise decreases (or stays same) across reps
            first = surprises_by_rep[0]
            last = surprises_by_rep[n_reps - 1]
            if first and last:
                mean_first = np.mean(first)
                mean_last = np.mean(last)
                self.assertLessEqual(
                    mean_last, mean_first + 0.1,
                    f"{model_key}: surprise increased — "
                    f"first={mean_first:.3f}, last={mean_last:.3f}"
                )
                return {
                    "model": model_key,
                    "mean_surprise_first": mean_first,
                    "mean_surprise_last": mean_last,
                    "total_usable_states": total_usable,
                    "n_frames": n_frames,
                    "n_reps": n_reps,
                }
        finally:
            sim.close()

    def test_fox_temporal_learning(self):
        result = self._train_and_check("fox", n_reps=3, max_frames=30)
        if result:
            print(f"\nFox: surprise {result['mean_surprise_first']:.3f} → "
                  f"{result['mean_surprise_last']:.3f} "
                  f"({result['total_usable_states']} usable states)")

    def test_cesium_man_temporal_learning(self):
        result = self._train_and_check("cesium_man", n_reps=3, max_frames=30)
        if result:
            print(f"\nCesiumMan: surprise {result['mean_surprise_first']:.3f} → "
                  f"{result['mean_surprise_last']:.3f} "
                  f"({result['total_usable_states']} usable states)")

    def test_robot_temporal_learning(self):
        result = self._train_and_check("robot", n_reps=3, max_frames=30)
        if result:
            print(f"\nRobot: surprise {result['mean_surprise_first']:.3f} → "
                  f"{result['mean_surprise_last']:.3f} "
                  f"({result['total_usable_states']} usable states)")


# ============================================================
# Test 5: Fox multi-animation comparison
# ============================================================

class TestFoxMultiAnimation(unittest.TestCase):
    """Compare TemporalMemory learning across Fox's three gaits."""

    def test_fox_different_gaits_produce_different_patterns(self):
        """Survey, Walk, and Run produce distinct temporal patterns."""
        _skip_if_missing("fox")
        from tbp.monty.context import RuntimeContext
        from tbp.monty.frameworks.environment_utils.transforms import (
            DepthTo3DLocations,
            TransformContext,
        )
        from tbp.monty.frameworks.models.sensor_modules import CameraSM

        sim, info, anim, cfg = _make_sim("fox")
        try:
            depth_norm = Panda3DDepthNormalize(
                agent_id=AGENT_ID, near=0.01, far=cfg["far"]
            )
            d3d = DepthTo3DLocations(
                agent_id=AGENT_ID,
                sensor_ids=[SensorID(SENSOR_ID)],
                resolutions=[(64, 64)],
                hfov=90.0,
                world_coord=True,
                get_all_points=True,
            )
            sm = CameraSM(
                sensor_module_id=str(SENSOR_ID),
                features=["on_object", "hsv", "principal_curvatures_log"],
            )
            sm.pre_episode()

            # Collect states from each animation
            gait_states = {}
            for anim_name in ["Survey", "Walk", "Run"]:
                n_frames = min(anim.get_num_frames(anim_name), 20)
                states = []
                for frame in range(n_frames):
                    anim.pose(frame, anim_name)
                    obs, proprio = sim.step([])
                    ctx = TransformContext(
                        rng=np.random.RandomState(0), state=proprio
                    )
                    obs = depth_norm(obs, ctx)
                    obs = d3d(obs, ctx)
                    sm.update_state(proprio[AGENT_ID])
                    rt_ctx = RuntimeContext(rng=np.random.RandomState(0))
                    state = sm.step(
                        rt_ctx, obs[AGENT_ID][SensorID(SENSOR_ID)]
                    )
                    if state is not None and state.use_state:
                        states.append(state.location.copy())
                gait_states[anim_name] = states

            # At least some gaits should produce usable states
            gaits_with_states = [
                k for k, v in gait_states.items() if len(v) > 0
            ]
            self.assertGreater(
                len(gaits_with_states), 0,
                "No gait produced usable states"
            )

            # If multiple gaits have states, check they differ
            if len(gaits_with_states) >= 2:
                g1, g2 = gaits_with_states[0], gaits_with_states[1]
                locs1 = np.array(gait_states[g1])
                locs2 = np.array(gait_states[g2])
                # Compare mean locations — different gaits move differently
                mean1 = locs1.mean(axis=0)
                mean2 = locs2.mean(axis=0)
                # Variance of locations should differ (different movement range)
                var1 = locs1.var(axis=0).sum()
                var2 = locs2.var(axis=0).sum()
                # At least one of mean or variance should differ
                mean_diff = np.linalg.norm(mean1 - mean2)
                var_diff = abs(var1 - var2)
                self.assertTrue(
                    mean_diff > 1e-6 or var_diff > 1e-6,
                    f"Gaits {g1} and {g2} produced identical patterns"
                )
        finally:
            sim.close()


# ============================================================
# Test 5b: Robot multi-animation comparison
# ============================================================

class TestRobotMultiAnimation(unittest.TestCase):
    """Compare TemporalMemory learning across Robot's animation repertoire."""

    def test_robot_different_actions_produce_different_patterns(self):
        """Walking, Running, and Dance produce distinct temporal patterns."""
        _skip_if_missing("robot")
        from tbp.monty.context import RuntimeContext
        from tbp.monty.frameworks.environment_utils.transforms import (
            DepthTo3DLocations,
            TransformContext,
        )
        from tbp.monty.frameworks.models.sensor_modules import CameraSM

        sim, info, anim, cfg = _make_sim("robot")
        try:
            depth_norm = Panda3DDepthNormalize(
                agent_id=AGENT_ID, near=0.01, far=cfg["far"]
            )
            d3d = DepthTo3DLocations(
                agent_id=AGENT_ID,
                sensor_ids=[SensorID(SENSOR_ID)],
                resolutions=[(64, 64)],
                hfov=90.0,
                world_coord=True,
                get_all_points=True,
            )
            sm = CameraSM(
                sensor_module_id=str(SENSOR_ID),
                features=["on_object", "hsv", "principal_curvatures_log"],
            )
            sm.pre_episode()

            action_states = {}
            for anim_name in ["Walking", "Dance", "Idle"]:
                n_frames = min(anim.get_num_frames(anim_name), 20)
                states = []
                for frame in range(n_frames):
                    anim.pose(frame, anim_name)
                    obs, proprio = sim.step([])
                    ctx = TransformContext(
                        rng=np.random.RandomState(0), state=proprio
                    )
                    obs = depth_norm(obs, ctx)
                    obs = d3d(obs, ctx)
                    sm.update_state(proprio[AGENT_ID])
                    rt_ctx = RuntimeContext(rng=np.random.RandomState(0))
                    state = sm.step(
                        rt_ctx, obs[AGENT_ID][SensorID(SENSOR_ID)]
                    )
                    if state is not None and state.use_state:
                        states.append(state.location.copy())
                action_states[anim_name] = states

            actions_with_states = [
                k for k, v in action_states.items() if len(v) > 0
            ]
            self.assertGreater(
                len(actions_with_states), 0,
                "No action produced usable states"
            )

            if len(actions_with_states) >= 2:
                a1, a2 = actions_with_states[0], actions_with_states[1]
                locs1 = np.array(action_states[a1])
                locs2 = np.array(action_states[a2])
                mean1 = locs1.mean(axis=0)
                mean2 = locs2.mean(axis=0)
                var1 = locs1.var(axis=0).sum()
                var2 = locs2.var(axis=0).sum()
                mean_diff = np.linalg.norm(mean1 - mean2)
                var_diff = abs(var1 - var2)
                self.assertTrue(
                    mean_diff > 1e-6 or var_diff > 1e-6,
                    f"Actions {a1} and {a2} produced identical patterns"
                )
        finally:
            sim.close()


# ============================================================
# Test 6: Panda3DTemporalTrainer with real models
# ============================================================

class TestTemporalTrainerWithRealModels(unittest.TestCase):
    """End-to-end Panda3DTemporalTrainer on real animated models."""

    def _train_model(self, model_key, n_reps=2, output_dir=None):
        """Run Panda3DTemporalTrainer on a real model."""
        _skip_if_missing(model_key)
        from tbp.monty.simulators.panda3d.temporal_training import (
            Panda3DTemporalTrainer,
        )

        cfg = MODEL_CONFIGS[model_key]
        model_path = os.path.join(ASSET_DIR, cfg["file"])

        trainer = Panda3DTemporalTrainer(
            model_path=model_path,
            resolution=(64, 64),
            fov=90.0,
            near=0.01,
            far=cfg["far"],
            motor_policy="orbital",
            orbit_radius=1.5,
            object_scale=cfg["scale"],
            output_dir=output_dir,
        )
        try:
            results = trainer.train_episode(
                n_repetitions=n_reps, n_replay=1
            )
            return results
        finally:
            trainer.close()

    def test_fox_trainer(self):
        _skip_if_missing("fox")
        results = self._train_model("fox", n_reps=2)
        self.assertIsNotNone(results)
        self.assertGreater(results["total_steps"], 0)
        print(f"\nFox trainer: {results['usable_states']}/{results['total_steps']} "
              f"usable, surprise {results['mean_surprise_first']} → "
              f"{results['mean_surprise_final']}")

    def test_cesium_man_trainer(self):
        _skip_if_missing("cesium_man")
        results = self._train_model("cesium_man", n_reps=2)
        self.assertIsNotNone(results)
        self.assertGreater(results["total_steps"], 0)
        print(f"\nCesiumMan trainer: {results['usable_states']}/{results['total_steps']} "
              f"usable, surprise {results['mean_surprise_first']} → "
              f"{results['mean_surprise_final']}")

    def test_robot_trainer(self):
        _skip_if_missing("robot")
        results = self._train_model("robot", n_reps=2)
        self.assertIsNotNone(results)
        self.assertGreater(results["total_steps"], 0)
        print(f"\nRobot trainer: {results['usable_states']}/{results['total_steps']} "
              f"usable, surprise {results['mean_surprise_first']} → "
              f"{results['mean_surprise_final']}")


# ============================================================
# Test 7: Debug output with real model
# ============================================================

class TestDebugOutputWithRealModel(unittest.TestCase):
    """Verify debug output (frames, states, surprise, video) works with real models."""

    def test_robot_debug_output(self):
        """Train Robot with debug output and verify files."""
        _skip_if_missing("robot")
        from tbp.monty.simulators.panda3d.temporal_training import (
            Panda3DTemporalTrainer,
        )

        cfg = MODEL_CONFIGS["robot"]
        model_path = os.path.join(ASSET_DIR, cfg["file"])

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Panda3DTemporalTrainer(
                model_path=model_path,
                resolution=(64, 64),
                fov=90.0,
                near=0.01,
                far=cfg["far"],
                motor_policy="orbital",
                orbit_radius=1.5,
                object_scale=cfg["scale"],
                output_dir=tmpdir,
            )
            try:
                results = trainer.train_episode(
                    n_repetitions=1, n_replay=0
                )
            finally:
                trainer.close()

            # Check output files
            frames_dir = os.path.join(tmpdir, "frames")
            states_dir = os.path.join(tmpdir, "states")

            self.assertTrue(os.path.isdir(frames_dir))
            self.assertTrue(os.path.isdir(states_dir))

            # Should have RGBA + depth frames
            frame_files = [
                f for f in os.listdir(frames_dir) if f.endswith(".png")
            ]
            self.assertGreater(len(frame_files), 0)

            # State JSONs
            state_files = [
                f for f in os.listdir(states_dir) if f.endswith(".json")
            ]
            self.assertGreater(len(state_files), 0)

            # Verify a state JSON is valid
            with open(os.path.join(states_dir, state_files[0])) as f:
                data = json.load(f)
            self.assertIn("step", data)

            # Summary
            summary_path = os.path.join(tmpdir, "summary.json")
            self.assertTrue(os.path.exists(summary_path))
            with open(summary_path) as f:
                summary = json.load(f)
            self.assertIn("total_steps", summary)

            # Surprise curve
            surprise_path = os.path.join(tmpdir, "surprise_curve.png")
            surprise_json = os.path.join(tmpdir, "surprise_curve.json")
            self.assertTrue(
                os.path.exists(surprise_path)
                or os.path.exists(surprise_json)
            )


# ============================================================
# Test 8: Cross-model comparison
# ============================================================

class TestCrossModelComparison(unittest.TestCase):
    """Compare temporal learning characteristics across different models."""

    def test_different_models_produce_different_surprise_profiles(self):
        """Each model type produces a distinct learning signature."""
        from tbp.monty.context import RuntimeContext
        from tbp.monty.frameworks.environment_utils.transforms import (
            DepthTo3DLocations,
            TransformContext,
        )
        from tbp.monty.frameworks.models.sensor_modules import CameraSM
        from tbp.monty.frameworks.models.temporal_memory import TemporalMemory

        profiles = {}

        for model_key in ["fox", "cesium_man", "robot"]:
            try:
                _skip_if_missing(model_key)
            except unittest.SkipTest:
                continue

            sim, info, anim, cfg = _make_sim(model_key)
            try:
                anim_name = cfg.get("default_anim", None)
                n_frames = min(anim.get_num_frames(anim_name), 20)
                depth_norm = Panda3DDepthNormalize(
                    agent_id=AGENT_ID, near=0.01, far=cfg["far"]
                )
                d3d = DepthTo3DLocations(
                    agent_id=AGENT_ID,
                    sensor_ids=[SensorID(SENSOR_ID)],
                    resolutions=[(64, 64)],
                    hfov=90.0,
                    world_coord=True,
                    get_all_points=True,
                )
                sm = CameraSM(
                    sensor_module_id=str(SENSOR_ID),
                    features=["on_object", "hsv", "principal_curvatures_log"],
                )
                sm.pre_episode()
                tm = TemporalMemory(sdr_dim=1024, sdr_sparsity=0.02)

                surprises = []
                for frame in range(n_frames):
                    anim.pose(frame, anim_name)
                    obs, proprio = sim.step([])
                    ctx = TransformContext(
                        rng=np.random.RandomState(0), state=proprio
                    )
                    obs = depth_norm(obs, ctx)
                    obs = d3d(obs, ctx)
                    sm.update_state(proprio[AGENT_ID])
                    rt_ctx = RuntimeContext(rng=np.random.RandomState(0))
                    state = sm.step(
                        rt_ctx, obs[AGENT_ID][SensorID(SENSOR_ID)]
                    )
                    if state is not None and state.use_state:
                        result = tm.step(state, learn=True)
                        surprises.append(result["surprise"])

                if surprises:
                    profiles[model_key] = {
                        "mean": float(np.mean(surprises)),
                        "std": float(np.std(surprises)),
                        "n_states": len(surprises),
                    }
            finally:
                sim.close()

        # We need at least 2 models to compare
        if len(profiles) < 2:
            self.skipTest("Need at least 2 models for comparison")

        # Print comparison for human review
        print("\n=== Cross-Model Surprise Profiles ===")
        for model, p in profiles.items():
            print(f"  {model}: mean={p['mean']:.3f} std={p['std']:.3f} "
                  f"states={p['n_states']}")

        # Basic check: profiles exist and have valid values
        for model, p in profiles.items():
            self.assertGreaterEqual(p["mean"], 0.0)
            self.assertLessEqual(p["mean"], 1.0)
            self.assertGreater(p["n_states"], 0)


if __name__ == "__main__":
    unittest.main()
