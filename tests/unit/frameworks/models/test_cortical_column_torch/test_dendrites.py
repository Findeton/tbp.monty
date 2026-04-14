# Copyright 2025-2026 Thousand Brains Project
# MIT License

"""Tests for sparse dendrites and their connected-synapse cache."""

import unittest

import torch

from tbp.monty.frameworks.models.cortical_column_torch.dendrites import (
    SparseDendrites,
)

HAS_CUDA = torch.cuda.is_available()


class TestSparseDendrites(unittest.TestCase):
    def test_empty_predict(self):
        dend = SparseDendrites(n_cells=32)
        x = torch.randn(32)
        depol = dend.predict(x)
        self.assertEqual(depol.shape, (32,))
        self.assertTrue((depol == 0).all())

    def test_grow_and_predict(self):
        dend = SparseDendrites(
            n_cells=16, activation_threshold=0.2, connected_threshold=0.3,
        )
        # Grow a segment on cell 0 connected to cells [1, 2, 3]
        dend.grow_segment(0, [1, 2, 3])
        self.assertEqual(dend.total_segments, 1)

        # Activate cells 1, 2, 3
        x = torch.zeros(16)
        x[1] = 1.0
        x[2] = 1.0
        x[3] = 1.0
        depol = dend.predict(x)
        self.assertGreater(depol[0].item(), 0.5, "Cell 0 should be depolarized")

    def test_graded_depolarization(self):
        """Depolarization should be graded, not binary.

        More active pre-synaptic sources → higher depolarization.
        """
        dend = SparseDendrites(
            n_cells=16, activation_threshold=0.5, connected_threshold=0.3,
            sigmoid_temp=0.2,
        )
        dend.grow_segment(0, [1, 2, 3, 4])

        # Partial activation (1 of 4 sources)
        x_low = torch.zeros(16)
        x_low[1] = 1.0
        depol_low = dend.predict(x_low)[0].item()

        # Full activation (all 4 sources)
        x_high = torch.zeros(16)
        x_high[1] = 1.0
        x_high[2] = 1.0
        x_high[3] = 1.0
        x_high[4] = 1.0
        depol_high = dend.predict(x_high)[0].item()

        self.assertGreater(
            depol_high, depol_low,
            "More source activation should yield higher depolarization",
        )

    def test_max_segments_per_cell(self):
        dend = SparseDendrites(n_cells=8, max_segments_per_cell=2)
        dend.grow_segment(0, [1, 2])
        dend.grow_segment(0, [3, 4])
        dend.grow_segment(0, [5, 6])  # Should be rejected
        self.assertEqual(dend.total_segments, 2)

    def test_learn_strengthens(self):
        dend = SparseDendrites(
            n_cells=8, initial_permanence=0.4, connected_threshold=0.3,
        )
        dend.grow_segment(0, [1])

        # Active: cell 0. Prev active: cell 1.
        active = torch.zeros(8)
        active[0] = 1.0
        prev = torch.zeros(8)
        prev[1] = 1.0

        old_perm = dend._segments[0][1][1]
        dend.learn(active, prev, learning_rate=1.0)
        new_perm = dend._segments[0][1][1]
        self.assertGreater(new_perm, old_perm)

    def test_grow_for_unpredicted(self):
        dend = SparseDendrites(n_cells=8, seed=42)
        active = torch.zeros(8)
        active[0] = 1.0
        active[2] = 1.0
        prev = torch.zeros(8)
        prev[4] = 1.0
        predicted = torch.zeros(8)
        predicted[0] = 1.0  # Cell 0 was predicted, cell 2 was not

        dend.grow_for_unpredicted(active, prev, predicted)
        # Should grow segment for cell 2 (unpredicted)
        self.assertGreater(dend.total_segments, 0)
        parent_cells = [seg[0] for seg in dend._segments]
        self.assertIn(2, parent_cells)

    def test_prune(self):
        dend = SparseDendrites(n_cells=8, initial_permanence=0.001)
        dend.grow_segment(0, [1])
        removed = dend.prune(min_permanence=0.01)
        self.assertEqual(removed, 1)
        self.assertEqual(dend.total_segments, 0)

    def test_state_dict_roundtrip(self):
        dend = SparseDendrites(n_cells=8)
        dend.grow_segment(0, [1, 2])
        sd = dend.state_dict()

        dend2 = SparseDendrites(n_cells=8)
        dend2.load_state_dict(sd)
        self.assertEqual(dend2.total_segments, 1)

        # Predictions should match after reload
        x = torch.zeros(8)
        x[1] = 1.0
        x[2] = 1.0
        depol1 = dend.predict(x)
        depol2 = dend2.predict(x)
        self.assertTrue(torch.allclose(depol1, depol2))

    def test_connected_synapse_cache_architecture(self):
        """Verify the internal connected-synapse cache is actually built."""
        dend = SparseDendrites(
            n_cells=16, connected_threshold=0.3, initial_permanence=0.5,
        )
        dend.grow_segment(0, [1, 2])
        dend.grow_segment(3, [4, 5])

        # Force rebuild
        dend._rebuild_sparse()

        self.assertIsNotNone(dend._segment_sources)
        self.assertIsNotNone(dend._segment_permanences)
        self.assertEqual(dend._segment_sources.shape, (2, dend.max_synapses_per_segment))
        self.assertEqual(
            dend._segment_permanences.shape,
            (2, dend.max_synapses_per_segment),
        )
        self.assertEqual(dend._segment_sources[0, 0].item(), 1)
        self.assertEqual(dend._segment_sources[0, 1].item(), 2)
        self.assertEqual(dend._segment_sources[1, 0].item(), 4)
        self.assertEqual(dend._segment_sources[1, 1].item(), 5)

        # seg_to_cell should map segment 0→cell 0, segment 1→cell 3
        self.assertEqual(dend._seg_to_cell[0].item(), 0)
        self.assertEqual(dend._seg_to_cell[1].item(), 3)

    def test_cache_invalidation(self):
        """Mutations invalidate the connected-synapse cache."""
        dend = SparseDendrites(n_cells=8)
        dend.grow_segment(0, [1])
        dend._rebuild_sparse()
        self.assertFalse(dend._dirty)

        # Growing a new segment should invalidate
        dend.grow_segment(2, [3])
        self.assertTrue(dend._dirty)

        # Predict should auto-rebuild
        x = torch.zeros(8)
        x[1] = 1.0
        dend.predict(x)
        self.assertFalse(dend._dirty)

    def test_scatter_max_picks_best_segment(self):
        """When a cell has multiple segments, the max activation wins."""
        dend = SparseDendrites(
            n_cells=16, max_segments_per_cell=4,
            activation_threshold=0.3, connected_threshold=0.3,
        )
        # Two segments on cell 0, one strong and one weak
        dend.grow_segment(0, [1, 2, 3])  # 3 sources
        dend.grow_segment(0, [4])          # 1 source

        x = torch.zeros(16)
        x[1] = 1.0
        x[2] = 1.0
        x[3] = 1.0
        # x[4] is NOT active — second segment gets low overlap

        depol = dend.predict(x)
        # Cell 0 depolarization should reflect the strong segment
        self.assertGreater(depol[0].item(), 0.5)

    def test_prediction_matches_manual_computation(self):
        """Cross-validate sparse matmul against manual numpy-style computation."""
        dend = SparseDendrites(
            n_cells=8, activation_threshold=0.3, sigmoid_temp=0.1,
            connected_threshold=0.3, initial_permanence=0.5,
        )
        dend.grow_segment(0, [1, 2])
        dend.grow_segment(3, [4, 5, 6])

        x = torch.tensor([0.0, 1.0, 0.5, 0.0, 0.8, 0.0, 1.0, 0.0])

        # Manual computation
        # Segment 0 (cell 0): sources [1, 2], perms [0.5, 0.5] >= 0.3
        #   overlap = 0.5*1.0 + 0.5*0.5 = 0.75
        #   activation = sigmoid((0.75 - 0.3) / 0.1)
        import math
        overlap_0 = 0.5 * 1.0 + 0.5 * 0.5
        act_0 = 1.0 / (1.0 + math.exp(-(overlap_0 - 0.3) / 0.1))

        # Segment 1 (cell 3): sources [4, 5, 6], perms [0.5, 0.5, 0.5]
        #   overlap = 0.5*0.8 + 0.5*0.0 + 0.5*1.0 = 0.9
        overlap_1 = 0.5 * 0.8 + 0.5 * 0.0 + 0.5 * 1.0
        act_1 = 1.0 / (1.0 + math.exp(-(overlap_1 - 0.3) / 0.1))

        depol = dend.predict(x)

        self.assertAlmostEqual(depol[0].item(), act_0, places=4)
        self.assertAlmostEqual(depol[3].item(), act_1, places=4)
        # Other cells should be zero
        for c in [1, 2, 4, 5, 6, 7]:
            self.assertAlmostEqual(depol[c].item(), 0.0, places=4)

    @unittest.skipUnless(HAS_CUDA, "CUDA not available")
    def test_gpu_parity(self):
        """CPU and CUDA produce identical predictions."""
        cpu_dend = SparseDendrites(n_cells=16, device="cpu")
        cpu_dend.grow_segment(0, [1, 2, 3])
        cpu_dend.grow_segment(4, [5, 6])

        gpu_dend = SparseDendrites(n_cells=16, device="cuda")
        gpu_dend.grow_segment(0, [1, 2, 3])
        gpu_dend.grow_segment(4, [5, 6])

        x = torch.zeros(16)
        x[1] = 1.0
        x[2] = 1.0
        x[5] = 0.5

        cpu_result = cpu_dend.predict(x)
        gpu_result = gpu_dend.predict(x.cuda()).cpu()
        self.assertTrue(torch.allclose(cpu_result, gpu_result, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
