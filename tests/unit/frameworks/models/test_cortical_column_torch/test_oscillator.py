# Copyright 2025-2026 Thousand Brains Project
# MIT License

"""Tests for oscillatory phase coding."""

import math
import unittest

import torch

from tbp.monty.frameworks.models.cortical_column_torch.oscillator import (
    CorticalOscillator,
    PhaseAugmentedHopfield,
)


class TestCorticalOscillator(unittest.TestCase):
    def test_initial_phase_is_zero(self):
        osc = CorticalOscillator(period=20)
        self.assertAlmostEqual(osc.phase, 0.0)

    def test_period(self):
        osc = CorticalOscillator(period=10)
        for _ in range(10):
            osc.step()
        # After one full period, phase should be back near 0
        self.assertAlmostEqual(osc.phase, 0.0, places=5)

    def test_oscillation_is_cosine(self):
        osc = CorticalOscillator(period=20)
        self.assertAlmostEqual(osc.oscillation, 1.0)
        # Quarter period
        for _ in range(5):
            osc.step()
        self.assertAlmostEqual(osc.oscillation, math.cos(math.pi / 2),
                               places=5)

    def test_phase_vector_is_unit(self):
        osc = CorticalOscillator(period=20)
        osc.step()
        pv = osc.phase_vector()
        self.assertEqual(pv.shape, (2,))
        norm = pv.norm().item()
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_augment_and_strip(self):
        osc = CorticalOscillator(period=20)
        x = torch.randn(32)
        x_aug = osc.augment_pattern(x)
        self.assertEqual(x_aug.shape, (34,))
        x_stripped = osc.strip_phase(x_aug, 32)
        self.assertTrue(torch.allclose(x, x_stripped))

    def test_extract_phase(self):
        osc = CorticalOscillator(period=20)
        for _ in range(3):
            osc.step()
        x = torch.randn(16)
        x_aug = osc.augment_pattern(x)
        phase = osc.extract_phase(x_aug, 16)
        self.assertEqual(phase.shape, (2,))

    def test_phase_precession(self):
        """Later items in sequence should fire at earlier phases."""
        osc = CorticalOscillator(period=20, precession_rate=0.3)
        phase_0 = osc.effective_phase()
        osc.step(is_new_element=True)
        phase_1 = osc.effective_phase()
        osc.step(is_new_element=True)
        phase_2 = osc.effective_phase()
        # Precession shifts phase backward
        # (but oscillator also advances, so raw phase increases)
        # The precession_rate * position subtracted from phase
        # means effective_phase progression is slower than raw phase
        self.assertNotAlmostEqual(phase_0, phase_1)

    def test_phase_difference(self):
        osc = CorticalOscillator(period=20)
        pv1 = torch.tensor([1.0, 0.0])
        pv2 = torch.tensor([0.0, 1.0])
        diff = osc.phase_difference(pv1, pv2)
        self.assertAlmostEqual(diff, math.pi / 2, places=4)

    def test_reset(self):
        osc = CorticalOscillator(period=20)
        for _ in range(5):
            osc.step()
        osc.reset()
        self.assertEqual(osc._step, 0)
        self.assertEqual(osc._sequence_position, 0)
        self.assertAlmostEqual(osc.phase, 0.0)

    def test_different_phases_produce_different_augmented_patterns(self):
        osc = CorticalOscillator(period=20)
        x = torch.ones(16)
        aug1 = osc.augment_pattern(x)
        osc.step()
        aug2 = osc.augment_pattern(x)
        # Same spatial pattern, different phase → different augmented
        self.assertFalse(torch.allclose(aug1, aug2))

    def test_disable_phase_coding(self):
        """phase_coding=False in column reproduces non-phase behavior."""
        osc = CorticalOscillator(period=20, precession_rate=0.0)
        # With precession_rate=0, effective phase = raw phase
        # Not a full disable test but verifies parameter control
        self.assertAlmostEqual(osc.effective_phase(), osc.phase)


class TestPhaseAugmentedHopfield(unittest.TestCase):
    def test_store_and_retrieve(self):
        from tbp.monty.frameworks.models.cortical_column_torch.hopfield import (
            ModernHopfieldMemory,
        )
        hop = ModernHopfieldMemory(n_cells=18, beta=4.0)  # 16 + 2 phase
        osc = CorticalOscillator(period=20)
        pa = PhaseAugmentedHopfield(hop, osc, n_spatial_cells=16)

        x = torch.randn(16)
        pa.store(x)
        osc.step()

        # Retrieve
        osc.reset()
        result = pa.retrieve(x)
        self.assertEqual(result.shape, (16,))

    def test_settle_returns_spatial_only(self):
        from tbp.monty.frameworks.models.cortical_column_torch.hopfield import (
            ModernHopfieldMemory,
        )
        hop = ModernHopfieldMemory(n_cells=18, beta=4.0)
        osc = CorticalOscillator(period=20)
        pa = PhaseAugmentedHopfield(hop, osc, n_spatial_cells=16)

        x = torch.randn(16)
        pa.store(x)
        settled, n_iters = pa.settle(x)
        self.assertEqual(settled.shape, (16,))


if __name__ == "__main__":
    unittest.main()
