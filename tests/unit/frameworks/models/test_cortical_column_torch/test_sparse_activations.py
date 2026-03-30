# Copyright 2025-2026 Thousand Brains Project
# MIT License

"""Tests for sparse activation utilities."""

import unittest

import torch

from tbp.monty.frameworks.models.cortical_column_torch.sparse_activations import (
    active_minicolumn_mask,
    enforce_sparsity,
    sparse_overlap,
)

HAS_CUDA = torch.cuda.is_available()


class TestEnforceSparsity(unittest.TestCase):
    def test_topk_preserves_values(self):
        """Top-k keeps original values, zeros the rest."""
        x = torch.randn(32)  # 4 MCs × 8 cells
        result = enforce_sparsity(x, n_minicolumns=4, cells_per_mc=8, k_per_mc=1)
        # Should have exactly 4 nonzeros (1 per MC)
        self.assertEqual((result != 0).sum().item(), 4)
        # Values should be from the original tensor
        for i in range(4):
            mc = result[i * 8:(i + 1) * 8]
            nonzero = mc[mc != 0]
            self.assertEqual(len(nonzero), 1)
            self.assertIn(nonzero[0].item(), x[i * 8:(i + 1) * 8].tolist())

    def test_topk_with_k2(self):
        """Top-2 per minicolumn."""
        x = torch.randn(16)  # 2 MCs × 8 cells
        result = enforce_sparsity(x, n_minicolumns=2, cells_per_mc=8, k_per_mc=2)
        for i in range(2):
            mc = result[i * 8:(i + 1) * 8]
            self.assertEqual((mc != 0).sum().item(), 2)

    def test_batch_dim(self):
        """Works with batch dimension."""
        x = torch.randn(3, 32)
        result = enforce_sparsity(x, n_minicolumns=4, cells_per_mc=8, k_per_mc=1)
        self.assertEqual(result.shape, (3, 32))
        for b in range(3):
            self.assertEqual((result[b] != 0).sum().item(), 4)

    def test_1d_squeeze(self):
        """1D input returns 1D output."""
        x = torch.randn(16)
        result = enforce_sparsity(x, n_minicolumns=2, cells_per_mc=8)
        self.assertEqual(result.dim(), 1)

    def test_k_exceeds_cells(self):
        """When k >= cells_per_mc, all values are preserved."""
        x = torch.randn(16)
        result = enforce_sparsity(x, n_minicolumns=2, cells_per_mc=8, k_per_mc=10)
        self.assertTrue(torch.allclose(result, x))

    @unittest.skipUnless(HAS_CUDA, "CUDA not available")
    def test_gpu_parity(self):
        """CPU and CUDA produce identical results."""
        x = torch.randn(32)
        cpu_result = enforce_sparsity(x, n_minicolumns=4, cells_per_mc=8)
        gpu_result = enforce_sparsity(
            x.cuda(), n_minicolumns=4, cells_per_mc=8,
        ).cpu()
        self.assertTrue(torch.allclose(cpu_result, gpu_result))


class TestSparseOverlap(unittest.TestCase):
    def test_identical_patterns(self):
        a = torch.tensor([1.0, 0.0, 0.5, 0.0])
        overlap = sparse_overlap(a, a)
        self.assertAlmostEqual(overlap.item(), 1.0, places=5)

    def test_disjoint_patterns(self):
        a = torch.tensor([1.0, 0.0, 0.0, 0.0])
        b = torch.tensor([0.0, 0.0, 1.0, 0.0])
        overlap = sparse_overlap(a, b)
        self.assertAlmostEqual(overlap.item(), 0.0, places=5)

    def test_partial_overlap(self):
        a = torch.tensor([1.0, 0.5, 0.0, 0.0])
        b = torch.tensor([1.0, 0.0, 0.5, 0.0])
        overlap = sparse_overlap(a, b)
        self.assertTrue(0.0 < overlap.item() < 1.0)

    def test_both_zero(self):
        """Two zero patterns should yield zero overlap, not NaN."""
        a = torch.zeros(8)
        b = torch.zeros(8)
        overlap = sparse_overlap(a, b)
        self.assertEqual(overlap.item(), 0.0)


class TestActiveMinicolumnMask(unittest.TestCase):
    def test_basic(self):
        x = torch.zeros(16)
        x[0] = 1.0  # MC 0 active
        x[15] = 0.5  # MC 1 active
        mask = active_minicolumn_mask(x, n_minicolumns=2, cells_per_mc=8)
        self.assertTrue(mask[0].item())
        self.assertTrue(mask[1].item())

    def test_inactive(self):
        x = torch.zeros(16)
        mask = active_minicolumn_mask(x, n_minicolumns=2, cells_per_mc=8)
        self.assertFalse(mask.any().item())


if __name__ == "__main__":
    unittest.main()
