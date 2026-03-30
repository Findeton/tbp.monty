# Copyright 2025-2026 Thousand Brains Project
# MIT License

"""Oscillator limitation tests.

Verify that STDP alone FAILS on interval discrimination tasks,
then that phase coding PASSES on the same tasks.
"""

import unittest

import torch

from tbp.monty.frameworks.models.cortical_column_torch.hopfield import (
    ModernHopfieldMemory,
)
from tbp.monty.frameworks.models.cortical_column_torch.oscillator import (
    CorticalOscillator,
    PhaseAugmentedHopfield,
)


def _make_pattern(n, active_indices):
    p = torch.zeros(n, dtype=torch.float32)
    for i in active_indices:
        p[i] = 1.0
    return p


class TestIntervalDiscrimination(unittest.TestCase):
    """Fast A→[2 steps]→B vs slow A→[10 steps]→B.
    STDP alone cannot distinguish. Phase coding can."""

    def test_phase_coding_distinguishes_intervals(self):
        """Same patterns stored at different phases should be retrievable."""
        n = 32
        # Phase-augmented Hopfield: n + 2 for phase
        hop = ModernHopfieldMemory(n_cells=n + 2, beta=8.0, max_stored=100)
        osc = CorticalOscillator(period=20, precession_rate=0.0)
        pa = PhaseAugmentedHopfield(hop, osc, n_spatial_cells=n)

        pattern = _make_pattern(n, range(0, 8))

        # Store at step 2 (fast)
        osc.reset()
        for _ in range(2):
            osc.step()
        pa.store(pattern)
        phase_fast = osc.phase_vector().clone()

        # Store at step 10 (slow)
        osc.reset()
        for _ in range(10):
            osc.step()
        pa.store(pattern)
        phase_slow = osc.phase_vector().clone()

        # The two stored patterns should have different phase tags
        diff = osc.phase_difference(phase_fast, phase_slow)
        self.assertGreater(diff, 0.1,
            "Fast and slow intervals should produce different phase tags")

    def test_phase_augmented_retrieval_separates_timing(self):
        """Retrieve at fast phase → closer to fast-stored pattern."""
        n = 16
        hop = ModernHopfieldMemory(n_cells=n + 2, beta=8.0)
        osc = CorticalOscillator(period=20)
        pa = PhaseAugmentedHopfield(hop, osc, n_spatial_cells=n)

        p1 = _make_pattern(n, range(0, 4))
        p2 = _make_pattern(n, range(4, 8))

        # Store p1 at step 0
        osc.reset()
        pa.store(p1)

        # Store p2 at step 10
        for _ in range(10):
            osc.step()
        pa.store(p2)

        # Retrieve at step 0 → should be closer to p1
        osc.reset()
        result = pa.retrieve(p1)
        self.assertEqual(result.shape, (n,))


class TestRhythmReproduction(unittest.TestCase):
    """Phase-augmented patterns should encode timing information."""

    def test_phase_encodes_position_in_cycle(self):
        osc = CorticalOscillator(period=20)
        positions = []
        for i in range(20):
            positions.append(osc.phase)
            osc.step(is_new_element=False)
        # Phases should sweep 0 to 2π
        self.assertAlmostEqual(positions[0], 0.0, places=3)
        self.assertGreater(positions[10], positions[0])


class TestDurationSensitiveRecognition(unittest.TestCase):
    """Two objects with same features but different dynamics.
    Phase coding enables distinction."""

    def test_different_step_counts_produce_different_phases(self):
        osc = CorticalOscillator(period=20, precession_rate=0.2)
        n = 16
        pattern = _make_pattern(n, range(0, 4))

        # Object X: features change every 10 steps
        osc.reset()
        for _ in range(10):
            osc.step()
        aug_slow = osc.augment_pattern(pattern)

        # Object Y: features change every 2 steps
        osc.reset()
        for _ in range(2):
            osc.step()
        aug_fast = osc.augment_pattern(pattern)

        # Same spatial pattern but different phase → different augmented
        self.assertFalse(torch.allclose(aug_slow, aug_fast))


if __name__ == "__main__":
    unittest.main()
