# Copyright 2025-2026 Thousand Brains Project
# MIT License

"""Tests for EpisodicMemory (HPC-like one-shot storage)."""

import unittest

import torch

from tbp.monty.frameworks.models.cortical_column_torch.episodic_memory import (
    EpisodicMemory,
)


class TestEpisodicMemory(unittest.TestCase):
    def test_store_and_count(self):
        mem = EpisodicMemory(n_cells=64, max_episodes=100)
        self.assertEqual(mem.n_stored, 0)
        mem.store(torch.randn(64), label="obj_A")
        self.assertEqual(mem.n_stored, 1)

    def test_store_with_label(self):
        mem = EpisodicMemory(n_cells=64)
        mem.store(torch.randn(64), label="mug")
        mem.store(torch.randn(64), label="banana")
        mem.store(torch.randn(64), label="mug")
        self.assertEqual(mem.known_labels, {"mug", "banana"})

    def test_novelty_gating(self):
        """Near-identical patterns should be rejected."""
        mem = EpisodicMemory(n_cells=64, novelty_threshold=0.95)
        p = torch.randn(64)
        self.assertTrue(mem.store(p, label="a"))
        # Near-identical (tiny noise)
        self.assertFalse(mem.store(p + torch.randn(64) * 0.001, label="a"))
        # Sufficiently different
        self.assertTrue(mem.store(torch.randn(64), label="b"))
        self.assertEqual(mem.n_stored, 2)

    def test_novelty_gating_disabled(self):
        """With threshold=None, all patterns stored."""
        mem = EpisodicMemory(n_cells=64, novelty_threshold=None)
        p = torch.randn(64)
        self.assertTrue(mem.store(p, label="a"))
        self.assertTrue(mem.store(p.clone(), label="a"))
        self.assertEqual(mem.n_stored, 2)

    def test_ring_buffer(self):
        """Oldest patterns overwritten when capacity exceeded."""
        mem = EpisodicMemory(n_cells=32, max_episodes=3, novelty_threshold=None)
        for i in range(5):
            mem.store(torch.randn(32), label=f"obj_{i}")
        self.assertEqual(mem._patterns.shape[0], 3)

    def test_retrieve_by_label(self):
        mem = EpisodicMemory(n_cells=32, novelty_threshold=None)
        torch.manual_seed(42)
        for _ in range(3):
            mem.store(torch.randn(32), label="mug")
        for _ in range(2):
            mem.store(torch.randn(32), label="banana")

        mug_pats = mem.retrieve_by_label("mug")
        self.assertIsNotNone(mug_pats)
        self.assertEqual(mug_pats.shape[0], 3)

        banana_pats = mem.retrieve_by_label("banana")
        self.assertIsNotNone(banana_pats)
        self.assertEqual(banana_pats.shape[0], 2)

        self.assertIsNone(mem.retrieve_by_label("nonexistent"))

    def test_replay_batch(self):
        mem = EpisodicMemory(n_cells=32, novelty_threshold=None)
        for i in range(10):
            mem.store(torch.randn(32), label=f"obj_{i % 3}")

        batch = mem.replay_batch(batch_size=5)
        self.assertEqual(len(batch), 5)
        for pattern, label in batch:
            self.assertEqual(pattern.shape, (32,))
            self.assertIsNotNone(label)

    def test_replay_batch_empty(self):
        mem = EpisodicMemory(n_cells=32)
        self.assertEqual(mem.replay_batch(), [])

    def test_clear(self):
        mem = EpisodicMemory(n_cells=32)
        mem.store(torch.randn(32), label="a")
        mem.clear()
        self.assertEqual(mem.n_stored, 0)
        self.assertEqual(mem.labels, [])

    def test_state_dict_roundtrip(self):
        mem = EpisodicMemory(n_cells=32, novelty_threshold=None)
        mem.store(torch.randn(32), label="mug")
        mem.store(torch.randn(32), label="banana")

        sd = mem.state_dict()
        mem2 = EpisodicMemory(n_cells=32)
        mem2.load_state_dict(sd)

        self.assertEqual(mem2.n_stored, 2)
        self.assertEqual(mem2.known_labels, {"mug", "banana"})


if __name__ == "__main__":
    unittest.main()
