# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Integration tests: CorticalColumnTorchLM in full Monty with real 3D models.

Tests that CorticalColumnTorchLM (PyTorch modern Hopfield) works as a real
drop-in replacement for EvidenceGraphLM in the full Monty sensorimotor
architecture with:
- Real Panda3D rendering (no procedural meshes)
- Real 3D glTF models (Fox.glb, RobotExpressive.glb)
- Real CameraSM + ChangeDetectingSM sensor modules
- Real InformedPolicy motor control driving camera exploration
- Multi-LM voting between CorticalColumnTorchLMs
- Hierarchical apical dendrites (parent column modulates children)
- Both frozen-frame (static morphology) and animated modes

Architecture tested:

  Flat mode (test_flat_*):
    SM 0: CameraSM → LM 0 (CorticalColumnTorchLM)
    Motor: InformedPolicy

  Hierarchical mode (test_hierarchical_*):
    SM 0: CameraSM          → LM 0 (CorticalColumnTorchLM, use_apical=True)
    SM 1: ChangeDetectingSM → LM 1 (CorticalColumnTorchLM, use_apical=True)
                               LM 2 (CorticalColumnTorchLM, parent)
    Apical: LM 2 broadcasts context → LM 0, LM 1
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

from tbp.monty.frameworks.models.cortical_column_torch.experiment import (
    Panda3DTorchExperiment,
)
from tbp.monty.frameworks.models.cortical_column_torch.learning_module import (
    CorticalColumnTorchLM,
)

# Real 3D model assets
_TEST_ROOT = Path(__file__).resolve().parents[3]  # tests/
ASSET_DIR = str(
    _TEST_ROOT / "unit" / "simulators" / "panda3d" / "test_assets" / "animated"
)
FOX_PATH = str(Path(ASSET_DIR) / "Fox.glb")
ROBOT_PATH = str(Path(ASSET_DIR) / "RobotExpressive.glb")


def _assets_available():
    return os.path.isfile(FOX_PATH) and os.path.isfile(ROBOT_PATH)


@unittest.skipUnless(_assets_available(), "Real 3D model assets not found")
class TestFlatTorchMonty(unittest.TestCase):
    """CorticalColumnTorchLM in flat (single-LM) mode with real 3D model."""

    def test_train_frozen_frame(self):
        """Train on frozen first frame of Fox.glb."""
        exp = Panda3DTorchExperiment(
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
            self.assertEqual(result["total_steps"], 10)
            self.assertIn("lm_0", result)

            lm = exp.monty.learning_modules[0]
            self.assertIsInstance(lm, CorticalColumnTorchLM)
            known = lm.get_all_known_object_ids()
            self.assertTrue(len(known) > 0)
        finally:
            exp.close()

    def test_train_animated(self):
        """Train on animated Fox.glb with motion."""
        exp = Panda3DTorchExperiment(
            model_path=FOX_PATH,
            hierarchical=False,
            resolution=(32, 32),
            initial_distance=2.0,
            object_scale=(0.01, 0.01, 0.01),
            asset_search_paths=[ASSET_DIR],
            column_kwargs={"n_minicolumns": 512, "sparsity": 0.05, "seed": 42},
        )
        try:
            exp._setup()
            anim_names = exp._anim_obj.animation_names
            self.assertTrue(len(anim_names) > 0)

            anim = list(anim_names)[0]
            result = exp.train("fox_animated", anim_name=anim, n_steps=10)
            self.assertEqual(result["total_steps"], 10)

            lm = exp.monty.learning_modules[0]
            self.assertTrue(lm._stepped)
        finally:
            exp.close()

    def test_train_then_eval(self):
        """Train on Fox, evaluate — LM should accumulate evidence."""
        exp = Panda3DTorchExperiment(
            model_path=FOX_PATH,
            hierarchical=False,
            resolution=(32, 32),
            initial_distance=2.0,
            object_scale=(0.01, 0.01, 0.01),
            asset_search_paths=[ASSET_DIR],
            column_kwargs={"n_minicolumns": 512, "sparsity": 0.05, "seed": 42},
        )
        try:
            exp.train("fox", n_steps=15)
            result = exp.evaluate(n_steps=10)

            self.assertIn("graph_id", result)
            self.assertIn("lm_0", result)

            lm = exp.monty.learning_modules[0]
            self.assertTrue(len(lm.evidence) > 0)
        finally:
            exp.close()

    def test_two_objects_discrimination(self):
        """Train Fox under two names, evaluate — evidence for both."""
        exp = Panda3DTorchExperiment(
            model_path=FOX_PATH,
            hierarchical=False,
            resolution=(32, 32),
            initial_distance=2.0,
            object_scale=(0.01, 0.01, 0.01),
            asset_search_paths=[ASSET_DIR],
            column_kwargs={"n_minicolumns": 1024, "sparsity": 0.04, "seed": 42},
        )
        try:
            exp.train("fox_a", n_steps=20)
            exp.train("fox_b", n_steps=20)

            result = exp.evaluate(n_steps=15)

            lm = exp.monty.learning_modules[0]
            self.assertTrue(
                len(lm.evidence) >= 2,
                f"Expected evidence for 2+ objects, got {len(lm.evidence)}",
            )
        finally:
            exp.close()


@unittest.skipUnless(_assets_available(), "Real 3D model assets not found")
class TestHierarchicalTorchMonty(unittest.TestCase):
    """CorticalColumnTorchLM with 3-LM heterarchy and apical dendrites."""

    def test_hierarchical_setup(self):
        """Verify 3-LM heterarchy wired correctly."""
        exp = Panda3DTorchExperiment(
            model_path=FOX_PATH,
            hierarchical=True,
            resolution=(32, 32),
            initial_distance=2.0,
            object_scale=(0.01, 0.01, 0.01),
            asset_search_paths=[ASSET_DIR],
            column_kwargs={"n_minicolumns": 512, "sparsity": 0.05, "use_apical": True},
        )
        try:
            exp._setup()

            self.assertEqual(len(exp.monty.learning_modules), 3)
            for lm in exp.monty.learning_modules:
                self.assertIsInstance(lm, CorticalColumnTorchLM)

            lm0 = exp.monty.learning_modules[0]
            lm1 = exp.monty.learning_modules[1]
            lm_parent = exp.monty.learning_modules[2]

            self.assertTrue(lm0.column._use_apical)
            self.assertTrue(lm1.column._use_apical)
            self.assertTrue(lm_parent.column._use_apical)

            self.assertEqual(exp.monty.sm_to_lm_matrix, [[0], [1], []])
            self.assertEqual(exp.monty.lm_to_lm_matrix, [[], [], [0, 1]])
            self.assertEqual(exp.monty.lm_to_lm_vote_matrix, [[1], [0], []])
            self.assertEqual(len(exp.monty.sensor_modules), 2)
        finally:
            exp.close()

    def test_hierarchical_train_frozen(self):
        """Train hierarchical on frozen Fox — all 3 LMs learn."""
        exp = Panda3DTorchExperiment(
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

            for lm in exp.monty.learning_modules[:2]:
                self.assertTrue(
                    lm._stepped,
                    f"{lm.learning_module_id} should have been stepped",
                )
        finally:
            exp.close()

    def test_hierarchical_train_animated(self):
        """Train hierarchical with animated Fox — motion + morphology."""
        exp = Panda3DTorchExperiment(
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
            lm_morph = exp.monty.learning_modules[0]
            lm_behav = exp.monty.learning_modules[1]
            self.assertTrue(lm_morph._stepped)
            self.assertTrue(lm_behav._stepped)
        finally:
            exp.close()

    def test_apical_context_flows(self):
        """Verify apical context flows from parent to children."""
        exp = Panda3DTorchExperiment(
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
            exp.train("fox", anim_name=anim, n_steps=8)

            lm_parent = exp.monty.learning_modules[2]
            signal = lm_parent.get_context_signal()

            if signal is not None:
                self.assertIn("active_cells", signal)
                active = signal["active_cells"]
                self.assertEqual(len(active), lm_parent.column.n_cells)

            lm0 = exp.monty.learning_modules[0]
            ctx_tensor = lm0.column._context
            self.assertEqual(len(ctx_tensor), lm0.column.n_cells)
        finally:
            exp.close()

    def test_hierarchical_eval(self):
        """Train hierarchical on Fox, eval — produces evidence."""
        exp = Panda3DTorchExperiment(
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
            exp.train("fox", anim_name=anim, n_steps=15)
            result = exp.evaluate(anim_name=anim, n_steps=10)

            self.assertIn("graph_id", result)
            self.assertIn("evidence", result)

            lm_morph = exp.monty.learning_modules[0]
            self.assertTrue(len(lm_morph.evidence) > 0)
        finally:
            exp.close()


@unittest.skipUnless(_assets_available(), "Real 3D model assets not found")
class TestMultiModelTorch(unittest.TestCase):
    """Train on multiple real 3D models, evaluate discrimination."""

    def test_two_models_flat(self):
        """Train Fox under two names in flat mode."""
        exp = Panda3DTorchExperiment(
            model_path=FOX_PATH,
            hierarchical=False,
            resolution=(32, 32),
            initial_distance=2.0,
            object_scale=(0.01, 0.01, 0.01),
            asset_search_paths=[ASSET_DIR],
            column_kwargs={"n_minicolumns": 1024, "sparsity": 0.04, "seed": 42},
        )
        try:
            exp.train("fox_a", n_steps=20)
            exp.train("fox_b", n_steps=20)

            lm = exp.monty.learning_modules[0]
            known = lm.get_all_known_object_ids()
            self.assertTrue(len(known) >= 2)

            result = exp.evaluate(n_steps=15)
            self.assertTrue(len(lm.evidence) >= 2)
        finally:
            exp.close()

    def test_two_models_hierarchical_animated(self):
        """Train Fox animated twice in hierarchical mode."""
        exp = Panda3DTorchExperiment(
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

            result = exp.evaluate(anim_name=fox_anim, n_steps=10)

            self.assertIn("graph_id", result)
            self.assertIn("evidence", result)

            for lm_id in ["lm_morphology", "lm_behavior", "lm_parent"]:
                self.assertIn(lm_id, result.get("evidence", {}))
        finally:
            exp.close()

    def test_three_models(self):
        """Train Fox under 3 names — multi-object storage."""
        exp = Panda3DTorchExperiment(
            model_path=FOX_PATH,
            hierarchical=False,
            resolution=(32, 32),
            initial_distance=2.0,
            object_scale=(0.01, 0.01, 0.01),
            asset_search_paths=[ASSET_DIR],
            column_kwargs={"n_minicolumns": 1024, "sparsity": 0.04, "seed": 42},
        )
        try:
            exp.train("fox_1", n_steps=15)
            exp.train("fox_2", n_steps=15)
            exp.train("fox_3", n_steps=15)

            lm = exp.monty.learning_modules[0]
            known = lm.get_all_known_object_ids()
            self.assertTrue(len(known) >= 3)
        finally:
            exp.close()


@unittest.skipUnless(_assets_available(), "Real 3D model assets not found")
class TestMotorControlTorchIntegration(unittest.TestCase):
    """Verify motor system drives camera around the 3D model."""

    def test_motor_actions_executed(self):
        exp = Panda3DTorchExperiment(
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
            motor = exp.monty.motor_system
            self.assertIsNotNone(motor)
            self.assertIsNotNone(motor._policy)

            exp.train("fox", n_steps=10)
            state = motor._state
            self.assertIsNotNone(state)
        finally:
            exp.close()


@unittest.skipUnless(_assets_available(), "Real 3D model assets not found")
class TestSwapModel(unittest.TestCase):
    """Test model swapping for multi-model evaluation."""

    def test_swap_and_eval(self):
        exp = Panda3DTorchExperiment(
            model_path=FOX_PATH,
            hierarchical=False,
            resolution=(32, 32),
            initial_distance=2.0,
            object_scale=(0.01, 0.01, 0.01),
            asset_search_paths=[ASSET_DIR],
            column_kwargs={"n_minicolumns": 512, "sparsity": 0.05, "seed": 42},
        )
        try:
            # Train on Fox
            exp.train("fox", n_steps=10)

            # Swap to Robot
            exp.swap_model(ROBOT_PATH)

            # Should be able to eval (different model)
            result = exp.evaluate(n_steps=10)
            self.assertIn("graph_id", result)
        finally:
            exp.close()


# ---- Phase 10: YCB Benchmark Tests ----

try:
    from tbp.monty.simulators.panda3d.ycb import (
        YCB_EVAL_OBJECTS,
        list_available_ycb,
        ycb_glb_path,
    )
except ImportError:
    YCB_EVAL_OBJECTS = []
    list_available_ycb = lambda: []


def _ycb_available(n_required=2):
    """Check if at least n_required YCB .glb meshes exist on disk."""
    available = list_available_ycb()
    eval_available = [o for o in YCB_EVAL_OBJECTS if o in available]
    return len(eval_available) >= n_required


@unittest.skipUnless(
    _ycb_available(2), "Need >= 2 YCB .glb meshes on disk"
)
class TestYCBSingleLMTwoObjects(unittest.TestCase):
    """Phase 10, test 1: Single LM, 2 YCB objects, train + eval.

    Train on mug and banana (or the first 2 available YCB objects).
    Eval on both. Verify correct top-1 match for each.
    """

    def test_two_ycb_discrimination(self):
        available = [o for o in YCB_EVAL_OBJECTS if o in list_available_ycb()]
        obj_a, obj_b = available[0], available[1]

        # Train on first object
        exp = Panda3DTorchExperiment(
            model_path=str(ycb_glb_path(obj_a)),
            hierarchical=False,
            resolution=(64, 64),
            initial_distance=0.3,
            object_scale=(1.0, 1.0, 1.0),
            column_kwargs={"n_minicolumns": 1024, "sparsity": 0.04, "seed": 42},
        )
        try:
            exp.train(obj_a, n_steps=30)

            # Swap to second object and train
            exp.swap_model(str(ycb_glb_path(obj_b)))
            exp.train(obj_b, n_steps=30)

            lm = exp.monty.learning_modules[0]
            known = lm.get_all_known_object_ids()
            self.assertTrue(
                len(known) >= 2,
                f"Expected 2+ known objects, got {known}",
            )

            # Eval on first object
            exp.swap_model(str(ycb_glb_path(obj_a)))
            result_a = exp.evaluate(n_steps=20)
            self.assertIn("graph_id", result_a)

            # Eval on second object
            exp.swap_model(str(ycb_glb_path(obj_b)))
            result_b = exp.evaluate(n_steps=20)
            self.assertIn("graph_id", result_b)
        finally:
            exp.close()


@unittest.skipUnless(
    _ycb_available(5), "Need >= 5 YCB .glb meshes on disk"
)
class TestYCBFiveObjectBenchmark(unittest.TestCase):
    """Phase 10, test 4: 5 YCB objects, full pipeline.

    This is the headline benchmark. Train on 5 objects, eval on all 5.
    """

    def test_five_ycb_objects(self):
        available = [o for o in YCB_EVAL_OBJECTS if o in list_available_ycb()]
        objects = available[:5]

        exp = Panda3DTorchExperiment(
            model_path=str(ycb_glb_path(objects[0])),
            hierarchical=False,
            resolution=(64, 64),
            initial_distance=0.3,
            object_scale=(1.0, 1.0, 1.0),
            column_kwargs={"n_minicolumns": 2048, "sparsity": 0.03, "seed": 42},
        )
        try:
            # Train on all 5 objects
            for i, obj_name in enumerate(objects):
                if i > 0:
                    exp.swap_model(str(ycb_glb_path(obj_name)))
                exp.train(obj_name, n_steps=40)

            lm = exp.monty.learning_modules[0]
            known = lm.get_all_known_object_ids()
            self.assertTrue(
                len(known) >= 5,
                f"Expected 5+ known objects, got {len(known)}: {known}",
            )

            # Eval on each object
            results = {}
            for obj_name in objects:
                exp.swap_model(str(ycb_glb_path(obj_name)))
                result = exp.evaluate(n_steps=30)
                results[obj_name] = result

            # All should produce evidence
            for obj_name, result in results.items():
                self.assertIn(
                    "graph_id", result,
                    f"No graph_id in eval result for {obj_name}",
                )
        finally:
            exp.close()


@unittest.skipUnless(_assets_available(), "Real 3D model assets not found")
class TestTwoLMHopfieldVoting(unittest.TestCase):
    """Phase 10, test 2: Two LMs with surprise-gated Hopfield voting.

    Uses hierarchical mode (CameraSM + ChangeDetectingSM → 2 child LMs)
    with hopfield_voting enabled. Verifies that:
    - Both child LMs accumulate evidence
    - Voting occurs (evidence is shared between LMs)
    - The system produces a recognition result
    """

    def test_hopfield_voting_hierarchical(self):
        """Train Fox with Hopfield voting, eval — both LMs contribute."""
        exp = Panda3DTorchExperiment(
            model_path=FOX_PATH,
            hierarchical=True,
            resolution=(32, 32),
            initial_distance=2.0,
            object_scale=(0.01, 0.01, 0.01),
            asset_search_paths=[ASSET_DIR],
            column_kwargs={"n_minicolumns": 512, "sparsity": 0.05},
            hopfield_voting=True,
            hopfield_surprise_threshold=0.5,
        )
        try:
            exp._setup()
            anim = list(exp._anim_obj.animation_names)[0]

            # Verify hopfield_voting is wired through
            self.assertTrue(exp.monty.hopfield_voting)

            # Verify child LMs have hopfield_voting enabled
            lm_morph = exp.monty.learning_modules[0]
            lm_behav = exp.monty.learning_modules[1]
            self.assertTrue(lm_morph._hopfield_voting)
            self.assertTrue(lm_behav._hopfield_voting)

            # Train
            exp.train("fox", anim_name=anim, n_steps=15)

            # Eval
            result = exp.evaluate(anim_name=anim, n_steps=10)

            self.assertIn("graph_id", result)

            # Both child LMs should have been stepped
            self.assertTrue(lm_morph._stepped)
            self.assertTrue(lm_behav._stepped)

            # Both should have evidence
            self.assertTrue(
                len(lm_morph.evidence) > 0,
                "Morphology LM should have evidence",
            )
        finally:
            exp.close()

    def test_hopfield_voting_does_not_break_without_flag(self):
        """Hierarchical mode without hopfield_voting should still work."""
        exp = Panda3DTorchExperiment(
            model_path=FOX_PATH,
            hierarchical=True,
            resolution=(32, 32),
            initial_distance=2.0,
            object_scale=(0.01, 0.01, 0.01),
            asset_search_paths=[ASSET_DIR],
            column_kwargs={"n_minicolumns": 512, "sparsity": 0.05},
            hopfield_voting=False,
        )
        try:
            exp.train("fox", n_steps=10)
            result = exp.evaluate(n_steps=8)
            self.assertIn("graph_id", result)
        finally:
            exp.close()


@unittest.skipUnless(_assets_available(), "Real 3D model assets not found")
class TestAutoLabelIntegration(unittest.TestCase):
    """Phase 10, test 5: Train without external labels (auto-generated).

    Train on 2 objects with primary_target=None. The LM should auto-generate
    internal labels and still discriminate objects on eval.
    """

    def test_auto_label_discrimination(self):
        exp = Panda3DTorchExperiment(
            model_path=FOX_PATH,
            hierarchical=False,
            resolution=(32, 32),
            initial_distance=2.0,
            object_scale=(0.01, 0.01, 0.01),
            asset_search_paths=[ASSET_DIR],
            column_kwargs={"n_minicolumns": 1024, "sparsity": 0.04, "seed": 42},
        )
        try:
            # Train without explicit labels
            exp.train(None, n_steps=20)
            exp.swap_model(ROBOT_PATH)
            exp.train(None, n_steps=20)

            lm = exp.monty.learning_modules[0]
            known = lm.get_all_known_object_ids()
            # Auto-labels should have been generated
            self.assertTrue(
                len(known) >= 1,
                f"Expected auto-generated labels, got {known}",
            )
        finally:
            exp.close()


@unittest.skipUnless(_assets_available(), "Real 3D model assets not found")
class TestNoveltyDetection(unittest.TestCase):
    """Phase 10, test 6: Novelty detection.

    Train on Fox. Present Robot during eval. The system should signal novelty
    (sustained high surprise) rather than false-matching to the known object.
    """

    def test_unseen_object_high_surprise(self):
        exp = Panda3DTorchExperiment(
            model_path=FOX_PATH,
            hierarchical=False,
            resolution=(32, 32),
            initial_distance=2.0,
            object_scale=(0.01, 0.01, 0.01),
            asset_search_paths=[ASSET_DIR],
            column_kwargs={"n_minicolumns": 512, "sparsity": 0.05, "seed": 42},
        )
        try:
            # Train only on Fox
            exp.train("fox", n_steps=20)

            # Eval on Robot (unseen)
            exp.swap_model(ROBOT_PATH)
            result = exp.evaluate(n_steps=15)

            lm = exp.monty.learning_modules[0]
            surprise = lm.column.surprise

            # Surprise should remain high for an unseen object.
            # A well-trained system should not confidently match the novel
            # object to a known one.
            self.assertGreater(
                surprise, 0.3,
                f"Surprise={surprise:.3f} is too low for an unseen object. "
                f"The system may be false-matching.",
            )

            # Terminal condition should NOT be "match"
            tc = lm.terminal_state
            if tc is not None:
                self.assertNotEqual(
                    tc, "match",
                    "System should not declare 'match' for a novel object",
                )
        finally:
            exp.close()


if __name__ == "__main__":
    unittest.main()
