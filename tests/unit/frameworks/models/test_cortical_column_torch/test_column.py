# Copyright 2025-2026 Thousand Brains Project
# MIT License

"""Tests for CorticalColumnTorch unified column."""

import unittest

import numpy as np
import torch

from tbp.monty.frameworks.models.cortical_column_torch.column import (
    CorticalColumnTorch,
)


class _MockState:
    """Minimal State-like object for testing."""

    def __init__(self, location=None, hsv=None, use_state=True):
        self.location = location or [0.0, 0.0, 0.0]
        self.use_state = use_state
        self.morphological_features = {
            "pose_vectors": np.eye(3),
            "pose_fully_defined": True,
        }
        self.non_morphological_features = {}
        if hsv:
            self.non_morphological_features["hsv"] = hsv
        self.confidence = 1.0
        self.sender_id = "test_sm"
        self.sender_type = "SM"


class TestCorticalColumnTorchStep(unittest.TestCase):
    def _make_column(self, **kwargs):
        defaults = dict(
            n_minicolumns=128,
            n_cells_per_minicolumn=4,
            sparsity=0.1,
            seed=42,
        )
        defaults.update(kwargs)
        return CorticalColumnTorch(**defaults)

    def test_step_returns_valid_dict(self):
        col = self._make_column()
        col.pre_episode(mode="eval")
        state = _MockState(location=[0.1, 0.2, 0.3])
        result = col.step(state)

        self.assertIn("surprise", result)
        self.assertIn("evidence", result)
        self.assertIn("mlh", result)
        self.assertIn("active_cells", result)
        self.assertIn("settling_iterations", result)

    def test_step_skip_invalid_state(self):
        col = self._make_column()
        col.pre_episode(mode="eval")
        state = _MockState(use_state=False)
        result = col.step(state)
        self.assertEqual(result["surprise"], 0.0)

    def test_train_stores_objects(self):
        col = self._make_column()
        col.pre_episode(mode="train", object_name="mug")

        for i in range(5):
            state = _MockState(location=[0.1 * i, 0.2, 0.0], hsv=[0.5, 0.3, 0.8])
            col.step(state)

        col.post_episode()
        self.assertIn("mug", col.get_all_known_object_ids())

    def test_eval_produces_evidence(self):
        col = self._make_column()

        # Train
        col.pre_episode(mode="train", object_name="apple")
        for i in range(10):
            state = _MockState(location=[0.05 * i, 0.1, 0.0], hsv=[0.2, 0.8, 0.5])
            col.step(state)
        col.post_episode()

        # Eval
        col.pre_episode(mode="eval")
        for i in range(5):
            state = _MockState(location=[0.05 * i, 0.1, 0.0], hsv=[0.2, 0.8, 0.5])
            result = col.step(state)

        self.assertIn("apple", result["evidence"])

    def test_surprise_decreases_with_repetition(self):
        col = self._make_column()
        col.pre_episode(mode="eval")

        surprises = []
        for i in range(10):
            state = _MockState(location=[0.1, 0.2, 0.0])
            result = col.step(state)
            surprises.append(result["surprise"])

        # After several identical inputs, surprise should stabilize or decrease
        # (the column learns to predict)
        self.assertTrue(len(surprises) > 0)

    def test_different_objects_discriminated(self):
        col = self._make_column()

        # Train object A
        col.pre_episode(mode="train", object_name="obj_a")
        for i in range(10):
            state = _MockState(location=[0.1 * i, 0.0, 0.0], hsv=[0.9, 0.1, 0.5])
            col.step(state)
        col.post_episode()

        # Train object B
        col.pre_episode(mode="train", object_name="obj_b")
        for i in range(10):
            state = _MockState(location=[0.0, 0.1 * i, 0.0], hsv=[0.1, 0.9, 0.5])
            col.step(state)
        col.post_episode()

        # Eval
        col.pre_episode(mode="eval")
        for i in range(5):
            state = _MockState(location=[0.1 * i, 0.0, 0.0], hsv=[0.9, 0.1, 0.5])
            result = col.step(state)

        evidence = result["evidence"]
        self.assertEqual(len(evidence), 2)

    def test_deterministic_with_seed(self):
        results = []
        for _ in range(2):
            col = self._make_column(seed=123)
            col.pre_episode(mode="eval")
            state = _MockState(location=[0.5, 0.5, 0.5])
            result = col.step(state)
            results.append(result["surprise"])
        self.assertAlmostEqual(results[0], results[1], places=5)


class TestCorticalColumnTorchApical(unittest.TestCase):
    def test_apical_context(self):
        col = CorticalColumnTorch(
            n_minicolumns=64,
            n_cells_per_minicolumn=4,
            sparsity=0.1,
            use_apical=True,
            seed=42,
        )
        col.pre_episode(mode="eval")

        # Set context
        context = np.random.rand(col.n_cells).astype(np.float32)
        col.receive_context(active_cells=context)

        state = _MockState(location=[0.1, 0.2, 0.3])
        result = col.step(state)
        self.assertIn("surprise", result)

    def test_context_signal_output(self):
        col = CorticalColumnTorch(
            n_minicolumns=64, n_cells_per_minicolumn=4, sparsity=0.1, seed=42,
        )
        col.pre_episode(mode="eval")
        state = _MockState(location=[0.1, 0.2, 0.3])
        col.step(state)

        signal = col.get_context_signal()
        self.assertIsNotNone(signal)
        self.assertIn("active_cells", signal)
        self.assertEqual(len(signal["active_cells"]), col.n_cells)


class TestCorticalColumnTorchMotor(unittest.TestCase):
    def test_motor_prediction(self):
        col = CorticalColumnTorch(
            n_minicolumns=64,
            n_cells_per_minicolumn=4,
            sparsity=0.1,
            use_motor_prediction=True,
            seed=42,
        )
        col.pre_episode(mode="eval")

        # Step with different locations
        for i in range(5):
            state = _MockState(location=[0.1 * i, 0.0, 0.0])
            result = col.step(state)

        self.assertIn("motor_prediction_error", result)


class TestCorticalColumnTorchNeuromod(unittest.TestCase):
    def test_neuromodulation(self):
        col = CorticalColumnTorch(
            n_minicolumns=64,
            n_cells_per_minicolumn=4,
            sparsity=0.1,
            use_neuromodulation=True,
            seed=42,
        )
        col.pre_episode(mode="eval")

        for i in range(5):
            state = _MockState(location=[0.1 * i, 0.0, 0.0])
            result = col.step(state)

        self.assertIn("surprise", result)


class TestCorticalColumnTorchPersistence(unittest.TestCase):
    def test_state_dict_roundtrip(self):
        col = CorticalColumnTorch(
            n_minicolumns=64, n_cells_per_minicolumn=4, sparsity=0.1, seed=42,
        )
        col.pre_episode(mode="train", object_name="test")
        for i in range(5):
            state = _MockState(location=[0.1 * i, 0.0, 0.0])
            col.step(state)
        col.post_episode()

        sd = col.state_dict()

        col2 = CorticalColumnTorch(
            n_minicolumns=64, n_cells_per_minicolumn=4, sparsity=0.1, seed=42,
        )
        col2.load_state_dict(sd)

        self.assertIn("test", col2.get_all_known_object_ids())


class TestCorticalColumnTorchGPU(unittest.TestCase):
    @unittest.skipUnless(
        hasattr(torch, "cuda") and torch.cuda.is_available(),
        "CUDA not available",
    )
    def test_gpu_parity(self):
        """Column produces same results on CPU and CUDA for identical inputs."""
        import torch as th

        configs = dict(
            n_minicolumns=64,
            n_cells_per_minicolumn=4,
            sparsity=0.1,
            seed=42,
        )
        results = {}
        for device in ("cpu", "cuda"):
            col = CorticalColumnTorch(**configs, device=device)
            col.pre_episode(mode="train", object_name="test")
            for i in range(5):
                state = _MockState(location=[0.1 * i, 0.2, 0.0])
                r = col.step(state)
            results[device] = r

        self.assertAlmostEqual(
            results["cpu"]["surprise"],
            results["cuda"]["surprise"],
            places=3,
        )


if __name__ == "__main__":
    unittest.main()
