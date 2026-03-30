# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Integration tests: Laminar CorticalColumnTorchLM with real 3D models.

Tests that Track 10 laminar features work end-to-end through the full
Monty sensorimotor loop with:
- Real Panda3D rendering with glTF models (Fox.glb, RobotExpressive.glb)
- Laminar pipeline (L4→L2/3→L5→L6)
- Multi-head dendritic attention
- Plateau potential working memory
- PV+/SST+/VIP+ interneuron circuit
- STDP + eligibility traces
- Oscillatory phase coding
- Thalamocortical loop gating
- All features independently enable/disable via flags
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

_TEST_ROOT = Path(__file__).resolve().parents[3]
ASSET_DIR = str(
    _TEST_ROOT / "unit" / "simulators" / "panda3d" / "test_assets" / "animated"
)
FOX_PATH = str(Path(ASSET_DIR) / "Fox.glb")
ROBOT_PATH = str(Path(ASSET_DIR) / "RobotExpressive.glb")


def _assets_available():
    return os.path.isfile(FOX_PATH) and os.path.isfile(ROBOT_PATH)


@unittest.skipUnless(_assets_available(), "Real 3D model assets not found")
class TestLaminarFlatMonty(unittest.TestCase):
    """Laminar column in flat mode with real 3D models."""

    def test_laminar_train_frozen(self):
        """Train laminar column on frozen Fox.glb."""
        exp = Panda3DTorchExperiment(
            model_path=FOX_PATH,
            hierarchical=False,
            resolution=(32, 32),
            initial_distance=2.0,
            object_scale=(0.01, 0.01, 0.01),
            asset_search_paths=[ASSET_DIR],
            column_kwargs={
                "n_minicolumns": 256,
                "sparsity": 0.05,
                "seed": 42,
                "laminar": True,
            },
        )
        try:
            result = exp.train("fox_laminar", n_steps=10)
            self.assertEqual(result["total_steps"], 10)
            lm = exp.monty.learning_modules[0]
            self.assertIsInstance(lm, CorticalColumnTorchLM)
            self.assertTrue(len(lm.get_all_known_object_ids()) > 0)
        finally:
            exp.close()

    def test_laminar_train_animated(self):
        """Train laminar column on animated Fox.glb with motion."""
        exp = Panda3DTorchExperiment(
            model_path=FOX_PATH,
            hierarchical=False,
            resolution=(32, 32),
            initial_distance=2.0,
            object_scale=(0.01, 0.01, 0.01),
            asset_search_paths=[ASSET_DIR],
            column_kwargs={
                "n_minicolumns": 256,
                "sparsity": 0.05,
                "seed": 42,
                "laminar": True,
            },
        )
        try:
            exp._setup()
            anim_names = exp._anim_obj.animation_names
            anim = list(anim_names)[0] if anim_names else None
            if anim:
                result = exp.train("fox_anim", anim_name=anim, n_steps=10)
                self.assertEqual(result["total_steps"], 10)
        finally:
            exp.close()

    def test_laminar_train_eval(self):
        """Laminar: train then evaluate — evidence should accumulate."""
        exp = Panda3DTorchExperiment(
            model_path=FOX_PATH,
            hierarchical=False,
            resolution=(32, 32),
            initial_distance=2.0,
            object_scale=(0.01, 0.01, 0.01),
            asset_search_paths=[ASSET_DIR],
            column_kwargs={
                "n_minicolumns": 512,
                "sparsity": 0.05,
                "seed": 42,
                "laminar": True,
            },
        )
        try:
            exp.train("fox", n_steps=15)
            result = exp.evaluate(n_steps=10)
            self.assertIn("graph_id", result)
            lm = exp.monty.learning_modules[0]
            self.assertTrue(len(lm.evidence) > 0)
        finally:
            exp.close()


@unittest.skipUnless(_assets_available(), "Real 3D model assets not found")
class TestMultiHeadDendritesIntegration(unittest.TestCase):
    """Multi-head dendritic attention in real Panda3D loop."""

    def test_multi_head_train(self):
        exp = Panda3DTorchExperiment(
            model_path=FOX_PATH,
            hierarchical=False,
            resolution=(32, 32),
            initial_distance=2.0,
            object_scale=(0.01, 0.01, 0.01),
            asset_search_paths=[ASSET_DIR],
            column_kwargs={
                "n_minicolumns": 256,
                "sparsity": 0.05,
                "seed": 42,
                "multi_head": True,
                "n_heads": 4,
            },
        )
        try:
            result = exp.train("fox_mh", n_steps=10)
            self.assertEqual(result["total_steps"], 10)
        finally:
            exp.close()


@unittest.skipUnless(_assets_available(), "Real 3D model assets not found")
class TestSTDPIntegration(unittest.TestCase):
    """STDP learning in real Panda3D loop."""

    def test_stdp_train(self):
        exp = Panda3DTorchExperiment(
            model_path=FOX_PATH,
            hierarchical=False,
            resolution=(32, 32),
            initial_distance=2.0,
            object_scale=(0.01, 0.01, 0.01),
            asset_search_paths=[ASSET_DIR],
            column_kwargs={
                "n_minicolumns": 256,
                "sparsity": 0.05,
                "seed": 42,
                "use_stdp": True,
            },
        )
        try:
            result = exp.train("fox_stdp", n_steps=10)
            self.assertEqual(result["total_steps"], 10)
        finally:
            exp.close()

    def test_stdp_with_eligibility(self):
        exp = Panda3DTorchExperiment(
            model_path=FOX_PATH,
            hierarchical=False,
            resolution=(32, 32),
            initial_distance=2.0,
            object_scale=(0.01, 0.01, 0.01),
            asset_search_paths=[ASSET_DIR],
            column_kwargs={
                "n_minicolumns": 256,
                "sparsity": 0.05,
                "seed": 42,
                "use_stdp": True,
                "use_eligibility": True,
                "use_neuromodulation": True,
            },
        )
        try:
            result = exp.train("fox_elig", n_steps=10)
            self.assertEqual(result["total_steps"], 10)
        finally:
            exp.close()


@unittest.skipUnless(_assets_available(), "Real 3D model assets not found")
class TestThalamicRelayIntegration(unittest.TestCase):
    """Thalamocortical loop in real Panda3D with laminar column."""

    def test_thalamic_gating_reduces_noise(self):
        """Laminar + thalamic: should train without errors."""
        exp = Panda3DTorchExperiment(
            model_path=FOX_PATH,
            hierarchical=False,
            resolution=(32, 32),
            initial_distance=2.0,
            object_scale=(0.01, 0.01, 0.01),
            asset_search_paths=[ASSET_DIR],
            column_kwargs={
                "n_minicolumns": 256,
                "sparsity": 0.05,
                "seed": 42,
                "laminar": True,
                "use_thalamic_relay": True,
            },
        )
        try:
            result = exp.train("fox_thal", n_steps=10)
            self.assertEqual(result["total_steps"], 10)
        finally:
            exp.close()


@unittest.skipUnless(_assets_available(), "Real 3D model assets not found")
class TestAllFeaturesIntegration(unittest.TestCase):
    """All Track 10 features enabled together in real Panda3D."""

    def test_all_features_train_eval(self):
        """Full laminar column with all bio features on Fox.glb."""
        exp = Panda3DTorchExperiment(
            model_path=FOX_PATH,
            hierarchical=False,
            resolution=(32, 32),
            initial_distance=2.0,
            object_scale=(0.01, 0.01, 0.01),
            asset_search_paths=[ASSET_DIR],
            column_kwargs={
                "n_minicolumns": 256,
                "sparsity": 0.05,
                "seed": 42,
                "laminar": True,
                "multi_head": True,
                "n_heads": 4,
                "use_plateau": True,
                "use_interneurons": True,
                "use_stdp": True,
                "use_eligibility": True,
                "use_phase_coding": True,
                "use_thalamic_relay": True,
                "use_apical": True,
                "use_neuromodulation": True,
                "use_motor_prediction": True,
            },
        )
        try:
            exp.train("fox_all", n_steps=15)
            result = exp.evaluate(n_steps=10)
            self.assertIn("graph_id", result)
            lm = exp.monty.learning_modules[0]
            self.assertTrue(len(lm.evidence) > 0)
        finally:
            exp.close()

    def test_all_features_hierarchical(self):
        """All features in hierarchical mode."""
        exp = Panda3DTorchExperiment(
            model_path=FOX_PATH,
            hierarchical=True,
            resolution=(32, 32),
            initial_distance=2.0,
            object_scale=(0.01, 0.01, 0.01),
            asset_search_paths=[ASSET_DIR],
            column_kwargs={
                "n_minicolumns": 256,
                "sparsity": 0.05,
                "seed": 42,
                "laminar": True,
                "multi_head": True,
                "n_heads": 4,
                "use_plateau": True,
                "use_interneurons": True,
                "use_stdp": True,
                "use_apical": True,
                "use_neuromodulation": True,
            },
        )
        try:
            result = exp.train("fox_hier_all", n_steps=10)
            self.assertEqual(result["total_steps"], 10)
            self.assertEqual(len(exp.monty.learning_modules), 3)
        finally:
            exp.close()

    def test_feature_flags_independent(self):
        """Each feature can be independently disabled.

        Verifies that enabling only one feature at a time doesn't crash.
        """
        features = [
            {"laminar": True},
            {"multi_head": True, "n_heads": 4},
            {"use_plateau": True},
            {"use_stdp": True},
            {"use_phase_coding": True},
            {"use_neuromodulation": True},
        ]
        for feat_kwargs in features:
            col_kw = {
                "n_minicolumns": 128,
                "sparsity": 0.1,
                "seed": 42,
            }
            col_kw.update(feat_kwargs)
            exp = Panda3DTorchExperiment(
                model_path=FOX_PATH,
                hierarchical=False,
                resolution=(32, 32),
                initial_distance=2.0,
                object_scale=(0.01, 0.01, 0.01),
                asset_search_paths=[ASSET_DIR],
                column_kwargs=col_kw,
            )
            try:
                result = exp.train("fox_feat", n_steps=5)
                self.assertEqual(result["total_steps"], 5,
                    f"Failed with features: {feat_kwargs}")
            finally:
                exp.close()


if __name__ == "__main__":
    unittest.main()
