# Copyright 2025-2026 Thousand Brains Project
# MIT License

"""Tests for Modern Hopfield memory."""

import unittest

import torch

from tbp.monty.frameworks.models.cortical_column_torch.hopfield import (
    ModernHopfieldMemory,
)

HAS_CUDA = torch.cuda.is_available()


class TestModernHopfieldMemory(unittest.TestCase):
    def test_store_and_count(self):
        hop = ModernHopfieldMemory(n_cells=64, max_stored=100)
        self.assertEqual(hop.n_stored, 0)
        hop.store(torch.randn(64))
        self.assertEqual(hop.n_stored, 1)

    def test_retrieve_single_pattern(self):
        """Store one pattern, retrieve it from noisy version."""
        hop = ModernHopfieldMemory(n_cells=64, beta=10.0)
        pattern = torch.randn(64)
        hop.store(pattern)

        # Query with noise
        noisy = pattern + torch.randn(64) * 0.1
        result = hop.retrieve(noisy)
        # Should be close to stored pattern (normalized)
        norm_pattern = pattern / pattern.norm()
        similarity = torch.dot(result / result.norm(), norm_pattern)
        self.assertGreater(similarity.item(), 0.95)

    def test_energy_decreases_during_settling(self):
        """Energy should be non-increasing during settling iterations."""
        hop = ModernHopfieldMemory(n_cells=64, beta=5.0, max_settle_iters=20)
        for _ in range(5):
            hop.store(torch.randn(64))

        x = torch.randn(64)
        initial_energy = hop.energy(x).item()

        # Settle
        settled, _ = hop.settle(x)
        final_energy = hop.energy(settled).item()
        self.assertLessEqual(final_energy, initial_energy + 1e-5)

    def test_energy_monotonic_per_iteration(self):
        """Track energy at every iteration to verify monotonic decrease."""
        hop = ModernHopfieldMemory(
            n_cells=64, beta=5.0, max_settle_iters=50,
            convergence_threshold=1e-8,  # force many iterations
        )
        for _ in range(5):
            hop.store(torch.randn(64))

        x = torch.randn(64)
        energies = [hop.energy(x).item()]

        # Manual settling loop to track per-iteration energy
        x_cur = x.clone()
        xi = hop._patterns[:min(hop._n_stored, hop.max_stored)]
        for _ in range(20):
            sims = hop.beta * torch.mv(xi, x_cur)
            sims = sims - sims.max()
            w = torch.softmax(sims, dim=0)
            x_cur = torch.mv(xi.t(), w)
            energies.append(hop.energy(x_cur).item())

        for i in range(1, len(energies)):
            self.assertLessEqual(
                energies[i], energies[i - 1] + 1e-5,
                f"Energy increased at iteration {i}: "
                f"{energies[i-1]:.6f} → {energies[i]:.6f}",
            )

    def test_convergence(self):
        """Settling should terminate within max_iters."""
        hop = ModernHopfieldMemory(n_cells=32, max_settle_iters=10)
        for _ in range(3):
            hop.store(torch.randn(32))

        x = torch.randn(32)
        _, n_iters = hop.settle(x)
        self.assertLessEqual(n_iters, 10)

    def test_multiple_patterns_retrieval(self):
        """Store N patterns, each should be retrievable with >95% similarity."""
        # Use higher dimensionality for reliable retrieval at 5 patterns
        hop = ModernHopfieldMemory(n_cells=256, beta=20.0, max_settle_iters=20)
        torch.manual_seed(42)
        patterns = []
        for _ in range(5):
            p = torch.randn(256)
            p = p / p.norm()
            patterns.append(p)
            hop.store(p)

        for i, p in enumerate(patterns):
            noisy = p + torch.randn(256) * 0.05
            retrieved = hop.retrieve(noisy)
            retrieved = retrieved / retrieved.norm()
            similarity = torch.dot(retrieved, p)
            self.assertGreater(
                similarity.item(), 0.95,
                f"Pattern {i} not retrieved: similarity={similarity.item():.3f}",
            )

    def test_capacity_scaling(self):
        """Capacity degrades gracefully as N grows; exceeds classical O(N).

        Classical Hopfield: capacity ~ 0.14 * n_cells.
        Modern Hopfield: capacity ~ exponential in n_cells.
        With n_cells=256 and beta=20, we should retrieve at least 50 patterns
        reliably (classical limit would be ~36).
        """
        n_cells = 256
        beta = 20.0
        hop = ModernHopfieldMemory(
            n_cells=n_cells, beta=beta, max_stored=200, max_settle_iters=20,
        )
        torch.manual_seed(0)

        n_patterns = 50
        patterns = []
        for _ in range(n_patterns):
            p = torch.randn(n_cells)
            p = p / p.norm()
            patterns.append(p)
            hop.store(p)

        correct = 0
        for i, p in enumerate(patterns):
            noisy = p + torch.randn(n_cells) * 0.1
            retrieved = hop.retrieve(noisy)
            retrieved = retrieved / retrieved.norm()
            # Find best-matching stored pattern
            best_sim = max(
                torch.dot(retrieved, q).item() for q in patterns
            )
            actual_sim = torch.dot(retrieved, p).item()
            if abs(actual_sim - best_sim) < 1e-4:
                correct += 1

        accuracy = correct / n_patterns
        self.assertGreater(
            accuracy, 0.9,
            f"Capacity test: {correct}/{n_patterns} correct ({accuracy:.0%}). "
            f"Exceeds classical Hopfield limit of ~{int(0.14 * n_cells)} patterns.",
        )

    def test_ring_buffer(self):
        """Exceed max_stored: oldest patterns overwritten."""
        hop = ModernHopfieldMemory(n_cells=32, max_stored=3)
        for _ in range(5):
            hop.store(torch.randn(32))
        self.assertEqual(hop._patterns.shape[0], 3)

    def test_clear(self):
        hop = ModernHopfieldMemory(n_cells=32)
        hop.store(torch.randn(32))
        hop.clear()
        self.assertEqual(hop.n_stored, 0)

    def test_state_dict_roundtrip(self):
        hop = ModernHopfieldMemory(n_cells=32)
        hop.store(torch.randn(32))
        sd = hop.state_dict()

        hop2 = ModernHopfieldMemory(n_cells=32)
        hop2.load_state_dict(sd)
        self.assertEqual(hop2.n_stored, 1)

    def test_sparsity_fn(self):
        """Sparsity function applied during settling."""
        hop = ModernHopfieldMemory(n_cells=16, max_settle_iters=5)
        hop.store(torch.randn(16))

        def sparsity_fn(x):
            # Keep only top-2
            topk_vals, topk_idx = torch.topk(x.abs(), 2)
            out = torch.zeros_like(x)
            out[topk_idx] = x[topk_idx]
            return out

        x = torch.randn(16)
        settled, _ = hop.settle(x, sparsity_fn=sparsity_fn)
        self.assertLessEqual((settled != 0).sum().item(), 2)

    def test_empty_memory_settle(self):
        """Settling with no stored patterns returns input."""
        hop = ModernHopfieldMemory(n_cells=32)
        x = torch.randn(32)
        settled, n_iters = hop.settle(x)
        self.assertTrue(torch.equal(settled, x))
        self.assertEqual(n_iters, 0)

    @unittest.skipUnless(HAS_CUDA, "CUDA not available")
    def test_gpu_parity(self):
        """CPU and CUDA produce identical retrieval results."""
        torch.manual_seed(42)
        pattern = torch.randn(64)

        cpu_hop = ModernHopfieldMemory(n_cells=64, beta=10.0, device="cpu")
        cpu_hop.store(pattern)
        cpu_result = cpu_hop.retrieve(pattern + torch.randn(64) * 0.1)

        gpu_hop = ModernHopfieldMemory(n_cells=64, beta=10.0, device="cuda")
        gpu_hop.store(pattern.cuda())
        gpu_result = gpu_hop.retrieve(
            (pattern + torch.randn(64) * 0.1).cuda()
        ).cpu()

        self.assertTrue(torch.allclose(cpu_result, gpu_result, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
