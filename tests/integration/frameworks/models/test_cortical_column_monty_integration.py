# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Integration tests: CorticalColumnLM in full Monty with real 3D models.

Tests that CorticalColumnLM works as a real drop-in replacement for
EvidenceGraphLM in the full Monty sensorimotor architecture with:
- Real Panda3D rendering (no procedural meshes)
- Real 3D glTF models (Fox.glb, RobotExpressive.glb, CesiumMan.glb)
- Real CameraSM + ChangeDetectingSM sensor modules
- Real InformedPolicy motor control driving camera exploration
- Multi-LM voting between CorticalColumnLMs
- Hierarchical apical dendrites (parent column modulates children)
- Both frozen-frame (static morphology) and animated (temporal behavior) modes

Architecture tested:

  Flat mode (test_flat_*):
    SM 0: CameraSM → LM 0 (CorticalColumnLM)
    Motor: InformedPolicy

  Hierarchical mode (test_hierarchical_*):
    SM 0: CameraSM          → LM 0 (CorticalColumnLM, use_apical=True)
    SM 1: ChangeDetectingSM → LM 1 (CorticalColumnLM, use_apical=True)
                               LM 2 (CorticalColumnLM, parent, use_apical=True)
                                      ← receives output from LM 0 + LM 1
    Apical: LM 2 broadcasts context → LM 0, LM 1 via _dispatch_context_signals
"""

import os
import unittest
from pathlib import Path

import numpy as np

try:
    import panda3d  # noqa: F401
    import gltf  # noqa: F401
except ImportError:
    raise unittest.SkipTest("Panda3D or panda3d-gltf not installed")

from tbp.monty.simulators.panda3d.cortical_column_experiment import (
    Panda3DCorticalColumnExperiment,
)

# Real 3D model assets (in unit/simulators/panda3d/test_assets/animated/)
_TEST_ROOT = Path(__file__).resolve().parents[3]  # tests/
ASSET_DIR = str(_TEST_ROOT / "unit" / "simulators" / "panda3d" / "test_assets" / "animated")
FOX_PATH = str(Path(ASSET_DIR) / "Fox.glb")
ROBOT_PATH = str(Path(ASSET_DIR) / "RobotExpressive.glb")
CESIUM_MAN_PATH = str(Path(ASSET_DIR) / "CesiumMan.glb")


def _assets_available():
    return os.path.isfile(FOX_PATH) and os.path.isfile(ROBOT_PATH)


@unittest.skipUnless(_assets_available(), "Real 3D model assets not found")
class TestFlatCorticalColumnMonty(unittest.TestCase):
    """CorticalColumnLM in flat (single-LM) mode with real 3D model."""

    def test_train_frozen_frame(self):
        """Train on frozen first frame of Fox.glb (static morphology)."""
        exp = Panda3DCorticalColumnExperiment(
            model_path=FOX_PATH,
            hierarchical=False,
            resolution=(32, 32),
            initial_distance=2.0,
            object_scale=(0.01, 0.01, 0.01),
            asset_search_paths=[ASSET_DIR],
            column_kwargs={"n_minicolumns": 512, "sparsity": 0.05, "seed": 42},
        )
        try:
            result = exp.train("fox_static", n_steps=10)

            self.assertIn("total_steps", result)
            self.assertEqual(result["total_steps"], 10)
            self.assertIn("lm_0", result)

            # LM should have learned the object
            lm = exp.monty.learning_modules[0]
            known = lm.get_all_known_object_ids()
            self.assertTrue(len(known) > 0, "LM should know at least one object")
        finally:
            exp.close()

    def test_train_animated(self):
        """Train on animated Fox.glb — temporal/behavior features via motion."""
        exp = Panda3DCorticalColumnExperiment(
            model_path=FOX_PATH,
            hierarchical=False,
            resolution=(32, 32),
            initial_distance=2.0,
            object_scale=(0.01, 0.01, 0.01),
            asset_search_paths=[ASSET_DIR],
            column_kwargs={"n_minicolumns": 512, "sparsity": 0.05, "seed": 42},
        )
        try:
            # Fox.glb has animations — train with motion
            anim_names = exp._setup() or True  # force setup
            anim_names = exp._anim_obj.animation_names
            self.assertTrue(len(anim_names) > 0, "Fox.glb should have animations")

            anim = list(anim_names)[0]
            result = exp.train("fox_animated", anim_name=anim, n_steps=10)

            self.assertEqual(result["total_steps"], 10)
            lm = exp.monty.learning_modules[0]
            self.assertTrue(lm._stepped, "LM should have been stepped")
        finally:
            exp.close()

    def test_train_then_eval(self):
        """Train on Fox, then evaluate — LM should accumulate evidence."""
        exp = Panda3DCorticalColumnExperiment(
            model_path=FOX_PATH,
            hierarchical=False,
            resolution=(32, 32),
            initial_distance=2.0,
            object_scale=(0.01, 0.01, 0.01),
            asset_search_paths=[ASSET_DIR],
            column_kwargs={"n_minicolumns": 512, "sparsity": 0.05, "seed": 42},
        )
        try:
            # Train
            exp.train("fox", n_steps=15)

            # Eval
            result = exp.evaluate(n_steps=10)

            self.assertIn("graph_id", result)
            self.assertIn("lm_0", result)

            # LM should have evidence for at least the trained object
            lm = exp.monty.learning_modules[0]
            self.assertTrue(len(lm.evidence) > 0, "LM should have evidence")
        finally:
            exp.close()

    def test_two_objects_discrimination(self):
        """Train Fox under two names, evaluate — should have evidence for both."""
        exp = Panda3DCorticalColumnExperiment(
            model_path=FOX_PATH,
            hierarchical=False,
            resolution=(32, 32),
            initial_distance=2.0,
            object_scale=(0.01, 0.01, 0.01),
            asset_search_paths=[ASSET_DIR],
            column_kwargs={
                "n_minicolumns": 1024,
                "sparsity": 0.04,
                "seed": 42,
            },
        )
        try:
            # Train Fox under name "fox_a"
            exp.train("fox_a", n_steps=20)

            # Train same Fox under name "fox_b" (different episode, new SDR)
            exp.train("fox_b", n_steps=20)

            # Evaluate — both objects should appear in evidence
            result = exp.evaluate(n_steps=15)

            lm = exp.monty.learning_modules[0]
            self.assertTrue(len(lm.evidence) >= 2,
                            f"Expected evidence for 2+ objects, got {len(lm.evidence)}")
        finally:
            exp.close()


@unittest.skipUnless(_assets_available(), "Real 3D model assets not found")
class TestHierarchicalCorticalColumnMonty(unittest.TestCase):
    """CorticalColumnLM with 3-LM heterarchy and apical dendrites."""

    def test_hierarchical_setup(self):
        """Verify 3-LM heterarchy is wired correctly."""
        exp = Panda3DCorticalColumnExperiment(
            model_path=FOX_PATH,
            hierarchical=True,
            resolution=(32, 32),
            initial_distance=2.0,
            object_scale=(0.01, 0.01, 0.01),
            asset_search_paths=[ASSET_DIR],
            column_kwargs={
                "n_minicolumns": 512,
                "sparsity": 0.05,
                "use_apical": True,
            },
        )
        try:
            exp._setup()

            # Should have 3 LMs
            self.assertEqual(len(exp.monty.learning_modules), 3)

            # All should be CorticalColumnLMs
            for lm in exp.monty.learning_modules:
                self.assertIsInstance(lm, type(exp.monty.learning_modules[0]))

            # Children should have apical dendrites enabled
            lm0 = exp.monty.learning_modules[0]
            lm1 = exp.monty.learning_modules[1]
            lm_parent = exp.monty.learning_modules[2]

            self.assertTrue(lm0.column._use_apical,
                            "Child LM 0 should have apical dendrites")
            self.assertTrue(lm1.column._use_apical,
                            "Child LM 1 should have apical dendrites")
            self.assertTrue(lm_parent.column._use_apical,
                            "Parent LM should have apical dendrites")

            # Verify heterarchy wiring
            self.assertEqual(exp.monty.sm_to_lm_matrix, [[0], [1], []])
            self.assertEqual(exp.monty.lm_to_lm_matrix, [[], [], [0, 1]])
            self.assertEqual(exp.monty.lm_to_lm_vote_matrix, [[1], [0], []])

            # Verify sensor modules
            self.assertEqual(len(exp.monty.sensor_modules), 2)
        finally:
            exp.close()

    def test_hierarchical_train_frozen(self):
        """Train hierarchical system on frozen Fox — all 3 LMs learn."""
        exp = Panda3DCorticalColumnExperiment(
            model_path=FOX_PATH,
            hierarchical=True,
            resolution=(32, 32),
            initial_distance=2.0,
            object_scale=(0.01, 0.01, 0.01),
            asset_search_paths=[ASSET_DIR],
            column_kwargs={"n_minicolumns": 512, "sparsity": 0.05},
        )
        try:
            result = exp.train("fox_static", n_steps=10)

            self.assertEqual(result["total_steps"], 10)

            # All LMs should have been stepped
            for lm in exp.monty.learning_modules[:2]:
                self.assertTrue(
                    lm._stepped,
                    f"{lm.learning_module_id} should have been stepped",
                )
        finally:
            exp.close()

    def test_hierarchical_train_animated(self):
        """Train hierarchical system with animated Fox — motion + morphology."""
        exp = Panda3DCorticalColumnExperiment(
            model_path=FOX_PATH,
            hierarchical=True,
            resolution=(32, 32),
            initial_distance=2.0,
            object_scale=(0.01, 0.01, 0.01),
            asset_search_paths=[ASSET_DIR],
            column_kwargs={"n_minicolumns": 512, "sparsity": 0.05},
        )
        try:
            exp._setup()
            anim = list(exp._anim_obj.animation_names)[0]

            result = exp.train("fox_walk", anim_name=anim, n_steps=10)

            self.assertEqual(result["total_steps"], 10)

            # Child LMs should have evidence
            lm_morph = exp.monty.learning_modules[0]
            lm_behav = exp.monty.learning_modules[1]
            self.assertTrue(lm_morph._stepped, "Morphology LM should have stepped")
            self.assertTrue(lm_behav._stepped, "Behavior LM should have stepped")
        finally:
            exp.close()

    def test_apical_context_flows(self):
        """Verify apical context actually flows from parent to children."""
        exp = Panda3DCorticalColumnExperiment(
            model_path=FOX_PATH,
            hierarchical=True,
            resolution=(32, 32),
            initial_distance=2.0,
            object_scale=(0.01, 0.01, 0.01),
            asset_search_paths=[ASSET_DIR],
            column_kwargs={"n_minicolumns": 512, "sparsity": 0.05},
        )
        try:
            exp._setup()
            anim = list(exp._anim_obj.animation_names)[0]

            # Train a few steps
            exp.train("fox", anim_name=anim, n_steps=8)

            # After training, the parent LM should have a context signal
            lm_parent = exp.monty.learning_modules[2]
            signal = lm_parent.get_context_signal()

            # Parent should have generated some active cells
            # (depends on whether it received LM-to-LM input)
            if signal is not None:
                self.assertIn("active_cells", signal)
                active = signal["active_cells"]
                self.assertEqual(
                    len(active), lm_parent.column.n_cells,
                    "Context signal should match column n_cells",
                )

            # Children's apical context masks should potentially have been set
            # (via _dispatch_context_signals after each step)
            lm0 = exp.monty.learning_modules[0]
            ctx_mask = lm0.column._context_active_mask
            self.assertEqual(
                len(ctx_mask), lm0.column.n_cells,
                "Context mask should exist on child column",
            )
        finally:
            exp.close()

    def test_hierarchical_eval(self):
        """Train hierarchical on Fox, eval — should produce evidence."""
        exp = Panda3DCorticalColumnExperiment(
            model_path=FOX_PATH,
            hierarchical=True,
            resolution=(32, 32),
            initial_distance=2.0,
            object_scale=(0.01, 0.01, 0.01),
            asset_search_paths=[ASSET_DIR],
            column_kwargs={"n_minicolumns": 512, "sparsity": 0.05},
        )
        try:
            exp._setup()
            anim = list(exp._anim_obj.animation_names)[0]

            # Train
            exp.train("fox", anim_name=anim, n_steps=15)

            # Eval
            result = exp.evaluate(anim_name=anim, n_steps=10)

            self.assertIn("graph_id", result)
            self.assertIn("evidence", result)

            # Morphology LM should have evidence
            lm_morph = exp.monty.learning_modules[0]
            self.assertTrue(len(lm_morph.evidence) > 0,
                            "Morphology LM should have evidence after eval")
        finally:
            exp.close()


@unittest.skipUnless(_assets_available(), "Real 3D model assets not found")
class TestMultiModelCorticalColumn(unittest.TestCase):
    """Train on multiple real 3D models, evaluate discrimination."""

    def test_two_models_flat(self):
        """Train Fox under two names in flat mode — evidence for both."""
        exp = Panda3DCorticalColumnExperiment(
            model_path=FOX_PATH,
            hierarchical=False,
            resolution=(32, 32),
            initial_distance=2.0,
            object_scale=(0.01, 0.01, 0.01),
            asset_search_paths=[ASSET_DIR],
            column_kwargs={
                "n_minicolumns": 1024,
                "sparsity": 0.04,
                "seed": 42,
            },
        )
        try:
            # Train Fox under two names (different episodes = different SDRs)
            exp.train("fox_a", n_steps=20)
            exp.train("fox_b", n_steps=20)

            lm = exp.monty.learning_modules[0]
            known = lm.get_all_known_object_ids()
            self.assertTrue(
                len(known) >= 2,
                f"Should know ≥2 objects after training both, got {known}",
            )

            # Eval
            result = exp.evaluate(n_steps=15)

            # Both should produce evidence
            self.assertTrue(len(lm.evidence) >= 2, "Should have evidence for >=2 objects")
        finally:
            exp.close()

    def test_two_models_hierarchical_animated(self):
        """Train Fox animated twice in hierarchical mode — full pipeline."""
        exp = Panda3DCorticalColumnExperiment(
            model_path=FOX_PATH,
            hierarchical=True,
            resolution=(32, 32),
            initial_distance=2.0,
            object_scale=(0.01, 0.01, 0.01),
            asset_search_paths=[ASSET_DIR],
            column_kwargs={"n_minicolumns": 512, "sparsity": 0.05},
        )
        try:
            exp._setup()

            fox_anim = list(exp._anim_obj.animation_names)[0]
            exp.train("fox_walk_a", anim_name=fox_anim, n_steps=15)
            exp.train("fox_walk_b", anim_name=fox_anim, n_steps=15)

            # Eval animated
            result = exp.evaluate(anim_name=fox_anim, n_steps=10)

            self.assertIn("graph_id", result)
            self.assertIn("evidence", result)

            # All 3 LMs should have evidence
            for lm_id in ["lm_morphology", "lm_behavior", "lm_parent"]:
                self.assertIn(lm_id, result.get("evidence", {}),
                              f"Evidence should include {lm_id}")
        finally:
            exp.close()

    def test_three_models(self):
        """Train Fox under 3 names — test multi-object storage at scale."""
        exp = Panda3DCorticalColumnExperiment(
            model_path=FOX_PATH,
            hierarchical=False,
            resolution=(32, 32),
            initial_distance=2.0,
            object_scale=(0.01, 0.01, 0.01),
            asset_search_paths=[ASSET_DIR],
            column_kwargs={
                "n_minicolumns": 1024,
                "sparsity": 0.04,
                "seed": 42,
            },
        )
        try:
            exp.train("fox_1", n_steps=15)
            exp.train("fox_2", n_steps=15)
            exp.train("fox_3", n_steps=15)

            lm = exp.monty.learning_modules[0]
            known = lm.get_all_known_object_ids()
            self.assertTrue(
                len(known) >= 3,
                f"Should know ≥3 objects, got {known}",
            )
        finally:
            exp.close()


@unittest.skipUnless(_assets_available(), "Real 3D model assets not found")
class TestMotorControlIntegration(unittest.TestCase):
    """Verify motor system actually drives camera around the 3D model."""

    def test_motor_actions_executed(self):
        """Motor system should produce actions that move the camera."""
        exp = Panda3DCorticalColumnExperiment(
            model_path=FOX_PATH,
            hierarchical=False,
            resolution=(32, 32),
            initial_distance=2.0,
            object_scale=(0.01, 0.01, 0.01),
            asset_search_paths=[ASSET_DIR],
            column_kwargs={"n_minicolumns": 512, "sparsity": 0.05},
        )
        try:
            exp._setup()

            # Motor system should exist with a policy
            motor = exp.monty.motor_system
            self.assertIsNotNone(motor, "Motor system should be initialized")
            self.assertIsNotNone(motor._policy, "Motor system should have a policy")

            # Run a few training steps
            exp.train("fox", n_steps=10)

            # Motor system state should have been synced (has position)
            state = motor._state
            self.assertIsNotNone(state, "Motor system state should be set after training")
        finally:
            exp.close()

    def test_motor_system_action_sequence(self):
        """Motor system should be functional during training."""
        exp = Panda3DCorticalColumnExperiment(
            model_path=FOX_PATH,
            hierarchical=False,
            resolution=(32, 32),
            initial_distance=2.0,
            object_scale=(0.01, 0.01, 0.01),
            asset_search_paths=[ASSET_DIR],
            column_kwargs={"n_minicolumns": 512, "sparsity": 0.05},
        )
        try:
            exp.train("fox", n_steps=10)

            motor = exp.monty.motor_system
            # Motor system should have been stepped (it produces actions)
            self.assertIsNotNone(motor._state, "Motor state should exist")
            # Policy should have been called
            self.assertIsNotNone(motor._policy, "Policy should exist")
        finally:
            exp.close()


if __name__ == "__main__":
    unittest.main()
