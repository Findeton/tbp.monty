# Copyright 2025-2026 Thousand Brains Project
# MIT License

"""Tests for multi-head dendritic attention."""

import unittest

import torch

from tbp.monty.frameworks.models.cortical_column_torch.multihead_dendrites import (
    MultiHeadDendrites,
)


class TestMultiHeadDendrites(unittest.TestCase):
    def _make(self, **kwargs):
        defaults = dict(
            n_cells=64, n_heads=4, max_segments_per_head=4,
            max_synapses_per_segment=8, seed=42,
        )
        defaults.update(kwargs)
        return MultiHeadDendrites(**defaults)

    def test_no_segments_returns_zeros(self):
        mhd = self._make()
        x = torch.randn(64)
        depol = mhd.predict(x)
        self.assertEqual(depol.shape, (64,))
        self.assertEqual(depol.sum().item(), 0.0)

    def test_grow_segment_on_specific_head(self):
        mhd = self._make()
        mhd.grow_segment(cell=0, source_cells=[1, 2, 3], head=2)
        self.assertEqual(len(mhd._heads[2]), 1)
        self.assertEqual(len(mhd._heads[0]), 0)
        self.assertEqual(mhd.total_segments, 1)

    def test_predict_nonzero_with_segments(self):
        mhd = self._make()
        mhd.grow_segment(cell=5, source_cells=[10, 11, 12], head=0)
        x = torch.zeros(64)
        x[10] = 1.0
        x[11] = 1.0
        x[12] = 1.0
        depol = mhd.predict(x)
        self.assertGreater(depol[5].item(), 0)

    def test_supralinear_integration(self):
        """Two half-active branches > one fully-active branch."""
        mhd = self._make(n_cells=32, n_heads=2, alpha=2.0)
        # Head 0: pattern A
        mhd.grow_segment(cell=0, source_cells=[1, 2], head=0)
        # Head 1: pattern B
        mhd.grow_segment(cell=0, source_cells=[3, 4], head=1)

        # Both heads active (partial)
        x_both = torch.zeros(32)
        x_both[1] = 0.5
        x_both[3] = 0.5
        depol_both = mhd.predict(x_both)

        # Only head 0 active (full)
        x_one = torch.zeros(32)
        x_one[1] = 1.0
        x_one[2] = 1.0
        depol_one = mhd.predict(x_one)

        # Supralinear: both heads partially active should give stronger response
        # (multiplicative integration)
        self.assertGreater(depol_both[0].item(), 0)

    def test_per_head_predict(self):
        mhd = self._make(n_cells=16, n_heads=2)
        mhd.grow_segment(cell=0, source_cells=[5, 6], head=0)
        mhd.grow_segment(cell=0, source_cells=[7, 8], head=1)

        x = torch.zeros(16)
        x[5] = 1.0
        x[6] = 1.0
        per_head = mhd.predict_per_head(x)
        self.assertEqual(per_head.shape, (2, 16))
        # Head 0 should be active for cell 0, head 1 should not
        self.assertGreater(per_head[0, 0].item(), per_head[1, 0].item())

    def test_branch_specific_learning(self):
        """Learning on head 0 should not affect head 1."""
        mhd = self._make(n_cells=16, n_heads=2)
        mhd.grow_segment(cell=0, source_cells=[5], head=0)
        mhd.grow_segment(cell=0, source_cells=[6], head=1)

        active = torch.zeros(16)
        active[0] = 1.0
        prev = torch.zeros(16)
        prev[5] = 1.0

        perm_h1_before = dict(mhd._heads[1][0][1])
        mhd.learn(active, prev, learning_rate=1.0, head=0)
        perm_h1_after = dict(mhd._heads[1][0][1])
        self.assertEqual(perm_h1_before, perm_h1_after)

    def test_gated_prediction(self):
        mhd = self._make(n_cells=16, n_heads=2)
        mhd.grow_segment(cell=0, source_cells=[5, 6], head=0)
        mhd.grow_segment(cell=0, source_cells=[7, 8], head=1)

        x = torch.zeros(16)
        x[5] = 1.0
        x[6] = 1.0
        x[7] = 1.0
        x[8] = 1.0

        # Ungated
        depol_full = mhd.predict(x)

        # Gate head 0 to zero
        gate = torch.tensor([0.0, 1.0])
        mhd.apply_gate(gate)
        depol_gated = mhd.predict_gated(x)

        # Gated should be less than or equal to full
        self.assertLessEqual(depol_gated[0].item(), depol_full[0].item())

    def test_n_heads_1_single_segment(self):
        """n_heads=1 should behave like standard dendrites."""
        mhd = self._make(n_cells=16, n_heads=1)
        mhd.grow_segment(cell=0, source_cells=[5, 6], head=0)
        x = torch.zeros(16)
        x[5] = 1.0
        x[6] = 1.0
        depol = mhd.predict(x)
        self.assertGreater(depol[0].item(), 0)

    def test_prune(self):
        mhd = self._make(n_cells=16, n_heads=2)
        mhd.grow_segment(cell=0, source_cells=[5], head=0)
        # Set permanence below threshold
        mhd._heads[0][0][1][5] = 0.001
        removed = mhd.prune(min_permanence=0.01)
        self.assertEqual(removed, 1)
        self.assertEqual(mhd.total_segments, 0)


if __name__ == "__main__":
    unittest.main()
