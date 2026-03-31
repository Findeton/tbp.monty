# Copyright 2025-2026 Thousand Brains Project
# MIT License

"""Tests for Hopfield associative memory."""

import unittest

import torch

from tbp.monty.frameworks.models.cortical_column_torch.associative_memory import (
    HopfieldAssociativeMemory,
)


class TestHopfieldAssociativeMemory(unittest.TestCase):
    def test_learn_and_recall(self):
        mem = HopfieldAssociativeMemory(n_cells=64, beta=10.0)
        pattern = torch.randn(64)
        mem.learn(pattern, "mug")

        scores = mem.recall(pattern)
        self.assertIn("mug", scores)
        self.assertGreater(scores["mug"], 0)

    def test_multiple_objects(self):
        mem = HopfieldAssociativeMemory(n_cells=128, beta=10.0)

        patterns = {}
        for name in ["mug", "banana", "apple"]:
            p = torch.randn(128)
            patterns[name] = p
            for _ in range(5):
                mem.learn(p + torch.randn(128) * 0.1, name)

        self.assertEqual(len(mem.known_objects), 3)

        # Each pattern should retrieve its own object with highest score
        for name, p in patterns.items():
            scores = mem.recall(p)
            best = max(scores, key=scores.get)
            self.assertEqual(
                best, name,
                f"Expected '{name}' to be top-1, got '{best}'. Scores: {scores}",
            )

    def test_multiple_objects_discrimination(self):
        """With sufficient observations, each object should be discriminated."""
        torch.manual_seed(42)
        mem = HopfieldAssociativeMemory(n_cells=256, beta=15.0)

        # Train with many observations per object (biologically realistic)
        patterns = {}
        for name in ["mug", "banana", "apple", "drill", "box"]:
            prototype = torch.randn(256)
            prototype = prototype / prototype.norm()
            patterns[name] = prototype
            for _ in range(20):
                noisy = prototype + torch.randn(256) * 0.1
                mem.learn(noisy, name)

        # Each prototype should recall its own object as top-1
        correct = 0
        for name, proto in patterns.items():
            scores = mem.recall(proto)
            best = max(scores, key=scores.get)
            if best == name:
                correct += 1

        self.assertEqual(
            correct, len(patterns),
            f"Only {correct}/{len(patterns)} objects correctly discriminated",
        )

    def test_capacity_at_scale(self):
        """Store 100+ objects and verify >90% top-1 accuracy.

        The softmax attention readout provides exponential storage capacity,
        so 100 objects with n_cells=512 should be retrievable.
        """
        torch.manual_seed(0)
        n_objects = 100
        n_cells = 512
        mem = HopfieldAssociativeMemory(n_cells=n_cells, beta=20.0)

        prototypes = {}
        for i in range(n_objects):
            name = f"object_{i:03d}"
            proto = torch.randn(n_cells)
            proto = proto / proto.norm()
            prototypes[name] = proto
            # 10 observations per object
            for _ in range(10):
                noisy = proto + torch.randn(n_cells) * 0.1
                mem.learn(noisy, name)

        correct = 0
        for name, proto in prototypes.items():
            scores = mem.recall(proto)
            best = max(scores, key=scores.get)
            if best == name:
                correct += 1

        accuracy = correct / n_objects
        self.assertGreater(
            accuracy, 0.9,
            f"Capacity test: {correct}/{n_objects} correct ({accuracy:.0%}). "
            f"Expected >90% for {n_objects} objects with {n_cells} cells.",
        )

    def test_auto_label(self):
        mem = HopfieldAssociativeMemory(n_cells=32)
        history = [torch.randn(32) for _ in range(5)]
        label = mem.auto_label(history)
        self.assertTrue(label.startswith("auto_"))

        # Deterministic
        label2 = mem.auto_label(history)
        self.assertEqual(label, label2)

    def test_auto_label_distinct_histories(self):
        """Different activation histories should produce different labels."""
        mem = HopfieldAssociativeMemory(n_cells=32)
        torch.manual_seed(0)
        h1 = [torch.randn(32) for _ in range(5)]
        torch.manual_seed(99)
        h2 = [torch.randn(32) for _ in range(5)]
        self.assertNotEqual(mem.auto_label(h1), mem.auto_label(h2))

    def test_auto_label_stable_under_noise(self):
        """Auto-labels should be stable when activation patterns have small noise.

        R6: The structural fingerprint should produce the same label even
        when individual sparse patterns vary slightly.  Uses sparse patterns
        matching real column output (~5% active) rather than dense Gaussians.
        """
        torch.manual_seed(42)
        mem = HopfieldAssociativeMemory(n_cells=256)

        # Create a sparse prototype (like real column output: ~5% active)
        prototype = torch.zeros(256)
        active_idx = [3, 17, 42, 88, 120, 155, 199, 210, 230, 250]
        for idx in active_idx:
            prototype[idx] = 1.0

        # Generate two histories with small noise around the same prototype
        h1 = [prototype + torch.randn(256) * 0.05 for _ in range(10)]
        h2 = [prototype + torch.randn(256) * 0.05 for _ in range(10)]

        label1 = mem.auto_label(h1)
        label2 = mem.auto_label(h2)

        self.assertEqual(
            label1, label2,
            f"Same sparse prototype with small noise should produce same "
            f"label: {label1} != {label2}",
        )

    def test_auto_label_different_objects_differ(self):
        """Auto-labels for genuinely different objects should differ.

        R6: Two orthogonal prototypes should produce different fingerprints.
        """
        torch.manual_seed(42)
        mem = HopfieldAssociativeMemory(n_cells=256)

        p1 = torch.zeros(256)
        p1[:32] = 1.0  # Active in first 32 dims

        p2 = torch.zeros(256)
        p2[128:160] = 1.0  # Active in dims 128-160

        h1 = [p1 + torch.randn(256) * 0.01 for _ in range(5)]
        h2 = [p2 + torch.randn(256) * 0.01 for _ in range(5)]

        self.assertNotEqual(
            mem.auto_label(h1), mem.auto_label(h2),
            "Different objects should produce different auto-labels",
        )

    def test_state_dict_roundtrip(self):
        mem = HopfieldAssociativeMemory(n_cells=32)
        mem.learn(torch.randn(32), "test_obj")

        sd = mem.state_dict()
        mem2 = HopfieldAssociativeMemory(n_cells=32)
        mem2.load_state_dict(sd)

        self.assertIn("test_obj", mem2.known_objects)

        # Verify recall produces same scores after reload
        query = torch.randn(32)
        scores1 = mem.recall(query)
        scores2 = mem2.recall(query)
        self.assertAlmostEqual(
            scores1["test_obj"], scores2["test_obj"], places=5,
        )

    def test_recall_empty(self):
        mem = HopfieldAssociativeMemory(n_cells=32)
        scores = mem.recall(torch.randn(32))
        self.assertEqual(len(scores), 0)


if __name__ == "__main__":
    unittest.main()
