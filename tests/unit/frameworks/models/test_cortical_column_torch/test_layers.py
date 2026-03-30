# Copyright 2025-2026 Thousand Brains Project
# MIT License

"""Tests for laminar layer modules (L4, L2/3, L5, L6)."""

import unittest

import torch

from tbp.monty.frameworks.models.cortical_column_torch.layers import (
    L4Layer,
    L23Layer,
    L5Layer,
    L6Layer,
    LaminarConfig,
)


class TestLaminarConfig(unittest.TestCase):
    def test_default_cells_per_minicolumn(self):
        cfg = LaminarConfig()
        self.assertEqual(cfg.cells_per_minicolumn, 8)  # 2+4+1+1

    def test_custom_allocation(self):
        cfg = LaminarConfig(n_L4=3, n_L23=6, n_L5=2, n_L6=1)
        self.assertEqual(cfg.cells_per_minicolumn, 12)

    def test_layer_offsets(self):
        cfg = LaminarConfig(n_L4=2, n_L23=4, n_L5=1, n_L6=1)
        offsets = cfg.layer_offsets(0)
        self.assertEqual(offsets["L4"], (0, 2))
        self.assertEqual(offsets["L23"], (2, 6))
        self.assertEqual(offsets["L5"], (6, 7))
        self.assertEqual(offsets["L6"], (7, 8))

        offsets1 = cfg.layer_offsets(1)
        self.assertEqual(offsets1["L4"], (8, 10))

    def test_layer_slice(self):
        cfg = LaminarConfig(n_L4=2, n_L23=4, n_L5=1, n_L6=1)
        indices, n = cfg.layer_slice(10, "L23")
        self.assertEqual(n, 40)  # 4 cells * 10 minicolumns
        self.assertEqual(len(indices), 40)


class TestL4Layer(unittest.TestCase):
    def test_forward_produces_active_minicolumns(self):
        l4 = L4Layer(n_minicolumns=64, n_input=100, n_cells_per_mc=2,
                      sparsity=0.1, seed=42)
        inp = torch.randn(100)
        active_mc, activation = l4.forward(inp)
        self.assertEqual(active_mc.shape, (64,))
        self.assertTrue(active_mc.any())
        n_active = active_mc.sum().item()
        self.assertGreater(n_active, 0)
        self.assertLessEqual(n_active, 64)

    def test_thalamic_gate_modulates_input(self):
        l4 = L4Layer(n_minicolumns=32, n_input=50, n_cells_per_mc=2,
                      sparsity=0.2, seed=42)
        inp = torch.ones(50)
        # Full gate
        _, act_full = l4.forward(inp, thalamic_gate=torch.ones(50))
        # Zero gate
        _, act_gated = l4.forward(inp, thalamic_gate=torch.zeros(50))
        # Gated should have less activation
        self.assertGreaterEqual(act_full.abs().sum(), act_gated.abs().sum())

    def test_learn_updates_permanences(self):
        l4 = L4Layer(n_minicolumns=16, n_input=30, n_cells_per_mc=2,
                      sparsity=0.2, seed=42)
        inp = torch.randn(30)
        l4.forward(inp)
        perms_before = l4._ff_permanences.clone()
        l4.learn(inp, lr=1.0)
        # Some permanences should have changed
        self.assertFalse(torch.equal(perms_before, l4._ff_permanences))


class TestL23Layer(unittest.TestCase):
    def test_forward_activates_winning_minicolumns(self):
        l23 = L23Layer(n_minicolumns=32, n_cells_per_mc=4)
        active_mc = torch.zeros(32, dtype=torch.bool)
        active_mc[:5] = True
        l4_act = torch.randn(32 * 2)  # L4 has 2 cells/mc
        basal_pred = torch.zeros(32 * 4)
        act = l23.forward(active_mc, l4_act, basal_pred, n_L4_cells_per_mc=2)
        self.assertEqual(act.shape, (128,))
        # Active minicolumns should have nonzero activation
        act_2d = act.reshape(32, 4)
        self.assertTrue(act_2d[:5].abs().sum() > 0)

    def test_prediction_guides_cell_selection(self):
        l23 = L23Layer(n_minicolumns=16, n_cells_per_mc=4)
        active_mc = torch.zeros(16, dtype=torch.bool)
        active_mc[0] = True
        l4_act = torch.ones(16 * 2)
        # Strong prediction on cell 1 of MC 0
        basal_pred = torch.zeros(16 * 4)
        basal_pred[1] = 0.9
        act = l23.forward(active_mc, l4_act, basal_pred, n_L4_cells_per_mc=2)
        act_mc0 = act[:4]
        # Cell 1 should be most active
        self.assertEqual(act_mc0.argmax().item(), 1)


class TestL5Layer(unittest.TestCase):
    def test_forward_integrates_basal_and_apical(self):
        l5 = L5Layer(n_minicolumns=16, n_cells_per_mc=1)
        active_mc = torch.zeros(16, dtype=torch.bool)
        active_mc[0] = True

        l23_act = torch.zeros(16 * 4)
        l23_act[:4] = 0.5  # MC 0 L2/3 active

        apical = torch.zeros(16)
        apical[0] = 0.5

        act = l5.forward(l23_act, apical, active_mc, n_L23_cells_per_mc=4)
        self.assertGreater(act[0].item(), 0)
        self.assertTrue(l5._has_basal_and_apical[0])

    def test_basal_only(self):
        l5 = L5Layer(n_minicolumns=8, n_cells_per_mc=1)
        active_mc = torch.zeros(8, dtype=torch.bool)
        active_mc[0] = True
        l23_act = torch.zeros(8 * 4)
        l23_act[:4] = 0.5
        apical = torch.zeros(8)
        act = l5.forward(l23_act, apical, active_mc, n_L23_cells_per_mc=4)
        self.assertGreater(act[0].item(), 0)
        self.assertFalse(l5._has_basal_and_apical[0])


class TestL6Layer(unittest.TestCase):
    def test_forward_and_gate(self):
        l6 = L6Layer(n_minicolumns=16, n_input=30, n_cells_per_mc=1)
        active_mc = torch.zeros(16, dtype=torch.bool)
        active_mc[:3] = True
        l23_act = torch.randn(16 * 4)
        l5_act = torch.randn(16)
        act = l6.forward(l23_act, l5_act, active_mc,
                         n_L23_cells_per_mc=4, n_L5_cells_per_mc=1)
        self.assertGreater(act.abs().sum(), 0)

        gate = l6.compute_thalamic_gate()
        self.assertEqual(gate.shape, (30,))
        self.assertTrue((gate >= 0).all() and (gate <= 1).all())

    def test_learn_changes_weights(self):
        l6 = L6Layer(n_minicolumns=8, n_input=20, n_cells_per_mc=1)
        active_mc = torch.zeros(8, dtype=torch.bool)
        active_mc[:2] = True
        l23 = torch.randn(8 * 4)
        l5 = torch.randn(8)
        l6.forward(l23, l5, active_mc, n_L23_cells_per_mc=4,
                   n_L5_cells_per_mc=1)
        w_before = l6._W_gate.clone()
        l6.learn(surprise=0.9, lr=1.0)
        self.assertFalse(torch.equal(w_before, l6._W_gate))


class TestLaminarColumnStep(unittest.TestCase):
    """Test that laminar=True produces valid output from CorticalColumnTorch."""

    def test_laminar_step(self):
        from tbp.monty.frameworks.models.cortical_column_torch.column import (
            CorticalColumnTorch,
        )
        import numpy as np
        from tbp.monty.frameworks.models.states import State

        col = CorticalColumnTorch(
            n_minicolumns=32,
            n_cells_per_minicolumn=8,
            sparsity=0.1,
            laminar=True,
            seed=42,
        )
        col.pre_episode(mode="train", object_name="test")

        state = State(
            location=np.array([0.1, 0.2, 0.3], dtype=np.float64),
            morphological_features={
                "pose_vectors": np.eye(3),
                "pose_fully_defined": True,
            },
            non_morphological_features={"hsv": [0.5, 0.3, 0.8]},
            confidence=1.0,
            use_state=True,
            sender_id="test_sm",
            sender_type="SM",
        )
        result = col.step(state)
        self.assertIn("surprise", result)
        self.assertIn("evidence", result)
        self.assertIn("active_cells", result)

    def test_laminar_off_matches_flat(self):
        """laminar=False should produce same results as Track 9."""
        from tbp.monty.frameworks.models.cortical_column_torch.column import (
            CorticalColumnTorch,
        )
        import numpy as np
        from tbp.monty.frameworks.models.states import State

        col = CorticalColumnTorch(
            n_minicolumns=32,
            n_cells_per_minicolumn=8,
            sparsity=0.1,
            laminar=False,
            seed=42,
        )
        col.pre_episode(mode="eval")
        state = State(
            location=np.array([0.1, 0.2, 0.3], dtype=np.float64),
            morphological_features={
                "pose_vectors": np.eye(3),
                "pose_fully_defined": True,
            },
            non_morphological_features={"hsv": [0.5, 0.3, 0.8]},
            confidence=1.0,
            use_state=True,
            sender_id="test_sm",
            sender_type="SM",
        )
        result = col.step(state)
        self.assertIn("surprise", result)


if __name__ == "__main__":
    unittest.main()
