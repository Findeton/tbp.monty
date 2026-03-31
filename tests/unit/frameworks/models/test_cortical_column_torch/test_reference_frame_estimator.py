# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Unit tests for ReferenceFrameEstimator."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tbp.monty.frameworks.models.cortical_column_torch.reference_frame_estimator import (
    ReferenceFrameEstimator,
)


@pytest.fixture
def rfe():
    return ReferenceFrameEstimator(
        min_pairs=3,
        evidence_threshold=0.1,
        confidence_threshold=0.5,
        max_pairs=30,
    )


def _apply_rotation(R, points, t=None):
    """Apply rotation (and optional translation) to points."""
    result = (R @ points.T).T
    if t is not None:
        result += t
    return result


class TestBasicRotationRecovery:
    def test_identity_rotation(self, rfe):
        """No rotation should give identity transform."""
        rng = np.random.RandomState(42)
        for _ in range(5):
            loc = rng.randn(3)
            rfe.add_observation("mug", loc, loc, evidence=0.5)

        assert rfe.has_rotation("mug")
        # Transform should be ~identity
        test_loc = np.array([1.0, 2.0, 3.0])
        transformed = rfe.transform("mug", test_loc)
        np.testing.assert_allclose(transformed, test_loc, atol=1e-6)

    def test_90_degree_rotation(self, rfe):
        """Recover a known 90-degree rotation around Z axis."""
        R = Rotation.from_euler("z", 90, degrees=True).as_matrix()

        rng = np.random.RandomState(42)
        for _ in range(10):
            world_loc = rng.randn(3) * 0.5
            obj_loc = R @ world_loc
            rfe.add_observation("mug", world_loc, obj_loc, evidence=0.5)

        assert rfe.has_rotation("mug")
        test_loc = np.array([1.0, 0.0, 0.0])
        transformed = rfe.transform("mug", test_loc)
        expected = R @ test_loc
        np.testing.assert_allclose(transformed, expected, atol=1e-4)

    def test_arbitrary_rotation(self, rfe):
        """Recover an arbitrary rotation."""
        R = Rotation.from_euler("xyz", [30, 45, 60], degrees=True).as_matrix()

        rng = np.random.RandomState(7)
        for _ in range(15):
            world_loc = rng.randn(3) * 0.3
            obj_loc = R @ world_loc
            rfe.add_observation("drill", world_loc, obj_loc, evidence=0.8)

        assert rfe.has_rotation("drill")
        test_loc = np.array([0.5, -0.3, 0.1])
        transformed = rfe.transform("drill", test_loc)
        expected = R @ test_loc
        np.testing.assert_allclose(transformed, expected, atol=1e-3)

    def test_rotation_plus_translation(self, rfe):
        """Recover rotation + translation."""
        R = Rotation.from_euler("y", 45, degrees=True).as_matrix()
        t = np.array([1.0, -0.5, 0.3])

        rng = np.random.RandomState(0)
        for _ in range(10):
            world_loc = rng.randn(3) * 0.5
            obj_loc = R @ world_loc + t
            rfe.add_observation("box", world_loc, obj_loc, evidence=0.6)

        assert rfe.has_rotation("box")
        test_loc = np.array([0.2, 0.7, -0.1])
        transformed = rfe.transform("box", test_loc)
        expected = R @ test_loc + t
        np.testing.assert_allclose(transformed, expected, atol=1e-3)


class TestMinPairs:
    def test_no_rotation_before_min_pairs(self, rfe):
        """Should not have a rotation before collecting min_pairs."""
        rfe.add_observation("mug", np.zeros(3), np.zeros(3), evidence=0.5)
        rfe.add_observation("mug", np.ones(3), np.ones(3), evidence=0.5)
        assert not rfe.has_rotation("mug")

    def test_rotation_after_min_pairs(self, rfe):
        """Should have a rotation after collecting min_pairs."""
        rng = np.random.RandomState(42)
        for _ in range(3):
            loc = rng.randn(3)
            rfe.add_observation("mug", loc, loc, evidence=0.5)
        assert rfe.has_rotation("mug")


class TestEvidenceGating:
    def test_low_evidence_ignored(self, rfe):
        """Observations below evidence threshold should be ignored."""
        rng = np.random.RandomState(42)
        for _ in range(10):
            loc = rng.randn(3)
            rfe.add_observation("mug", loc, loc, evidence=0.05)
        assert not rfe.has_rotation("mug")

    def test_mixed_evidence(self, rfe):
        """Only high-evidence observations should count."""
        rng = np.random.RandomState(42)
        # Low evidence — ignored
        for _ in range(10):
            loc = rng.randn(3)
            rfe.add_observation("mug", loc, loc * -1, evidence=0.01)
        # High evidence — correct
        for _ in range(5):
            loc = rng.randn(3)
            rfe.add_observation("mug", loc, loc, evidence=0.5)

        assert rfe.has_rotation("mug")
        test_loc = np.array([1.0, 2.0, 3.0])
        transformed = rfe.transform("mug", test_loc)
        np.testing.assert_allclose(transformed, test_loc, atol=1e-3)


class TestMultiObject:
    def test_independent_objects(self, rfe):
        """Each object gets its own rotation estimate."""
        R_mug = Rotation.from_euler("z", 90, degrees=True).as_matrix()
        R_drill = Rotation.from_euler("x", 45, degrees=True).as_matrix()

        rng = np.random.RandomState(42)
        for _ in range(10):
            loc = rng.randn(3) * 0.5
            rfe.add_observation("mug", loc, R_mug @ loc, evidence=0.5)
            rfe.add_observation("drill", loc, R_drill @ loc, evidence=0.5)

        assert rfe.has_rotation("mug")
        assert rfe.has_rotation("drill")

        test_loc = np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(
            rfe.transform("mug", test_loc), R_mug @ test_loc, atol=1e-3
        )
        np.testing.assert_allclose(
            rfe.transform("drill", test_loc), R_drill @ test_loc, atol=1e-3
        )


class TestConfidence:
    def test_noisy_observations_low_confidence(self):
        """Very noisy observation pairs should yield low confidence."""
        rfe = ReferenceFrameEstimator(
            min_pairs=3, evidence_threshold=0.0, confidence_threshold=0.9,
        )
        rng = np.random.RandomState(42)
        for _ in range(10):
            world = rng.randn(3)
            # Random "predicted" location — no consistent transform
            obj = rng.randn(3)
            rfe.add_observation("mug", world, obj, evidence=1.0)

        # Confidence should be low because there's no consistent rotation
        assert not rfe.has_rotation("mug")
        assert rfe.get_confidence("mug") < 0.9

    def test_perfect_observations_high_confidence(self, rfe):
        """Exact rotation pairs should give high confidence."""
        R = Rotation.from_euler("z", 45, degrees=True).as_matrix()
        rng = np.random.RandomState(42)
        for _ in range(10):
            loc = rng.randn(3) * 0.5
            rfe.add_observation("mug", loc, R @ loc, evidence=0.5)

        assert rfe.get_confidence("mug") > 0.95


class TestReset:
    def test_reset_clears_all(self, rfe):
        rng = np.random.RandomState(42)
        for _ in range(5):
            loc = rng.randn(3)
            rfe.add_observation("mug", loc, loc, evidence=0.5)
        assert rfe.has_rotation("mug")

        rfe.reset()
        assert not rfe.has_rotation("mug")
        assert rfe.estimated_objects == []


class TestRingBuffer:
    def test_max_pairs_enforced(self):
        rfe = ReferenceFrameEstimator(min_pairs=3, max_pairs=5)
        rng = np.random.RandomState(42)
        for i in range(20):
            loc = rng.randn(3)
            rfe.add_observation("mug", loc, loc, evidence=0.5)
        # Internal list should be capped
        assert len(rfe._world_locs["mug"]) == 5


class TestNoisyRecovery:
    def test_noisy_pairs_still_recoverable(self, rfe):
        """With moderate noise, rotation should still be approximately correct."""
        R = Rotation.from_euler("xyz", [20, 30, 40], degrees=True).as_matrix()
        rng = np.random.RandomState(42)
        for _ in range(20):
            world = rng.randn(3) * 0.5
            # Add noise to predicted location (simulates imperfect feature matching)
            obj = R @ world + rng.randn(3) * 0.02
            rfe.add_observation("mug", world, obj, evidence=0.5)

        assert rfe.has_rotation("mug")
        test_loc = np.array([0.5, -0.3, 0.2])
        transformed = rfe.transform("mug", test_loc)
        expected = R @ test_loc
        # Allow larger tolerance due to noise
        np.testing.assert_allclose(transformed, expected, atol=0.15)
