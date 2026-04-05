# Copyright 2025-2026 Thousand Brains Project
# MIT License

"""Tests for PyTorch encoders."""

import unittest

import numpy as np
import torch

from tbp.monty.frameworks.models.cortical_column_torch.encoders import (
    TorchFeatureEncoder,
    TorchGridCellEncoder,
    TorchScalarEncoder,
)

HAS_CUDA = torch.cuda.is_available()


class TestTorchGridCellEncoder(unittest.TestCase):
    def test_output_shape(self):
        enc = TorchGridCellEncoder(n_modules=10, n_phases=8)
        result = enc.encode(np.array([1.0, 2.0, 3.0]))
        self.assertEqual(result.shape, (10 * 2 * 8,))

    def test_deterministic(self):
        enc = TorchGridCellEncoder(seed=42)
        a = enc.encode(np.array([1.0, 0.0, 0.0]))
        b = enc.encode(np.array([1.0, 0.0, 0.0]))
        self.assertTrue(torch.allclose(a, b))

    def test_different_locations(self):
        enc = TorchGridCellEncoder(seed=42)
        a = enc.encode(np.array([0.0, 0.0, 0.0]))
        b = enc.encode(np.array([1.0, 1.0, 1.0]))
        self.assertFalse(torch.allclose(a, b))

    def test_values_bounded(self):
        enc = TorchGridCellEncoder()
        result = enc.encode(np.array([5.0, -3.0, 2.0]))
        self.assertTrue((result >= -1.0).all())
        self.assertTrue((result <= 1.0).all())

    def test_numpy_cross_validation(self):
        """Verify torch encoder matches numpy reimplementation of same math."""
        n_modules, n_phases, seed = 10, 8, 42
        enc = TorchGridCellEncoder(
            n_modules=n_modules, n_phases=n_phases, seed=seed,
        )
        loc = np.array([1.5, -0.3, 2.7], dtype=np.float32)

        # Reimplement the same math in numpy
        rng = np.random.RandomState(seed)
        dirs = rng.randn(n_modules, n_phases, 3).astype(np.float32)
        norms = np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-8
        dirs /= norms
        scales = np.array([2.0 ** i for i in range(n_modules)], dtype=np.float32)

        proj = dirs @ loc                            # (n_modules, n_phases)
        scaled = proj * scales[:, np.newaxis]         # (n_modules, n_phases)
        cos_part = np.cos(scaled)
        sin_part = np.sin(scaled)
        expected = np.stack([cos_part, sin_part], axis=-1).reshape(-1)

        result = enc.encode(loc).cpu().numpy()
        np.testing.assert_allclose(result, expected, atol=1e-6)

    @unittest.skipUnless(HAS_CUDA, "CUDA not available")
    def test_gpu_parity(self):
        """CPU and CUDA produce identical results."""
        loc = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        cpu_enc = TorchGridCellEncoder(seed=42, device="cpu")
        gpu_enc = TorchGridCellEncoder(seed=42, device="cuda")
        cpu_out = cpu_enc.encode(loc).cpu()
        gpu_out = gpu_enc.encode(loc).cpu()
        self.assertTrue(torch.allclose(cpu_out, gpu_out, atol=1e-6))


class TestTorchScalarEncoder(unittest.TestCase):
    def test_output_shape(self):
        enc = TorchScalarEncoder(n_bits=32)
        result = enc.encode(0.5)
        self.assertEqual(result.shape, (32,))

    def test_values_positive(self):
        enc = TorchScalarEncoder(n_bits=32)
        result = enc.encode(0.5)
        self.assertTrue((result >= 0).all())

    def test_distinct_values(self):
        enc = TorchScalarEncoder(n_bits=32)
        a = enc.encode(0.1)
        b = enc.encode(0.9)
        self.assertFalse(torch.allclose(a, b))

    def test_numpy_cross_validation(self):
        """Verify torch scalar encoder matches numpy Gaussian bump math."""
        n_bits, seed = 32, 42
        enc = TorchScalarEncoder(
            n_bits=n_bits, min_val=0.0, max_val=1.0, seed=seed,
        )
        value = 0.6

        rng = np.random.RandomState(seed)
        centers = np.sort(rng.uniform(0.0, 1.0, n_bits)).astype(np.float32)
        width = (1.0 - 0.0) / n_bits * 2.0
        expected = np.exp(-0.5 * ((centers - value) / width) ** 2)

        result = enc.encode(value).cpu().numpy()
        np.testing.assert_allclose(result, expected, atol=1e-6)

    @unittest.skipUnless(HAS_CUDA, "CUDA not available")
    def test_gpu_parity(self):
        cpu_enc = TorchScalarEncoder(n_bits=32, seed=42, device="cpu")
        gpu_enc = TorchScalarEncoder(n_bits=32, seed=42, device="cuda")
        cpu_out = cpu_enc.encode(0.5).cpu()
        gpu_out = gpu_enc.encode(0.5).cpu()
        self.assertTrue(torch.allclose(cpu_out, gpu_out, atol=1e-6))


class TestTorchFeatureEncoder(unittest.TestCase):
    def test_output_shape(self):
        enc = TorchFeatureEncoder(
            location_bits=160, feature_bits_per_dim=16, n_feature_dims=5,
        )
        # Create a minimal state-like object
        class MockState:
            location = [1.0, 2.0, 3.0]
            use_state = True
            morphological_features = {"pose_vectors": np.eye(3)}
            non_morphological_features = {"hsv": [0.5, 0.3, 0.8]}

        result = enc.encode(MockState())
        self.assertEqual(result.shape, (enc.total_bits,))

    def test_encode_location(self):
        enc = TorchFeatureEncoder()
        result = enc.encode_location(np.array([1.0, 0.0, 0.0]))
        self.assertEqual(result.shape, (enc._location_bits,))

    def test_extract_features_includes_hsv_flow_and_pose(self):
        class MockState:
            location = [1.0, 2.0, 3.0]
            use_state = True
            sender_type = "SM"
            morphological_features = {"pose_vectors": np.eye(3)}
            non_morphological_features = {
                "hsv": [0.5, 0.3, 0.8],
                "flow_direction": [0.25, -0.5, 0.75],
                "flow_magnitude": [0.4],
            }

        result = TorchFeatureEncoder._extract_features(MockState())

        np.testing.assert_allclose(
            result,
            [0.5, 0.3, 0.8, 0.25, -0.5, 0.75, 0.4, 1.0, 0.0, 0.0],
            atol=1e-6,
        )

    def test_extract_features_includes_lm_identity_evidence_and_pose(self):
        class MockState:
            location = [1.0, 2.0, 3.0]
            use_state = True
            sender_type = "LM"
            sender_id = "lm_behavior"
            morphological_features = {"pose_vectors": np.eye(3)}
            non_morphological_features = {
                "graph_id": "fox",
                "evidence": 2.5,
                "surprise": 0.2,
            }

        result = TorchFeatureEncoder._extract_features(MockState())
        expected = [
            *TorchFeatureEncoder._hash_text_features("fox", 4),
            *TorchFeatureEncoder._hash_text_features("lm_behavior", 1),
            float(np.tanh(2.5)),
            0.2,
            1.0,
            0.0,
            0.0,
        ]

        np.testing.assert_allclose(result[: len(expected)], expected, atol=1e-6)

    def test_hash_label_deterministic(self):
        a = TorchFeatureEncoder.hash_label("test_object")
        b = TorchFeatureEncoder.hash_label("test_object")
        self.assertTrue(torch.equal(a, b))

    def test_hash_label_distinct(self):
        a = TorchFeatureEncoder.hash_label("object_a")
        b = TorchFeatureEncoder.hash_label("object_b")
        self.assertFalse(torch.equal(a, b))

    @unittest.skipUnless(HAS_CUDA, "CUDA not available")
    def test_gpu_parity(self):
        class MockState:
            location = [1.0, 2.0, 3.0]
            use_state = True
            morphological_features = {"pose_vectors": np.eye(3)}
            non_morphological_features = {"hsv": [0.5, 0.3, 0.8]}

        cpu_enc = TorchFeatureEncoder(device="cpu")
        gpu_enc = TorchFeatureEncoder(device="cuda")
        cpu_out = cpu_enc.encode(MockState()).cpu()
        gpu_out = gpu_enc.encode(MockState()).cpu()
        self.assertTrue(torch.allclose(cpu_out, gpu_out, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
