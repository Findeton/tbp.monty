# Copyright 2025-2026 Thousand Brains Project
# MIT License

"""Tests for neuromodulatory gating."""

import unittest

from tbp.monty.frameworks.models.cortical_column_torch.neuromodulators import (
    NeuromodulatoryGating,
)


class TestNeuromodulatoryGating(unittest.TestCase):
    def test_default_scales(self):
        nm = NeuromodulatoryGating()
        # Default state should give near-unity scales
        self.assertGreater(nm.beta_scale(), 0)
        self.assertGreater(nm.learning_rate_scale(), 0)
        self.assertGreater(nm.sparsity_scale(), 0)

    def test_high_surprise_boosts_learning(self):
        nm = NeuromodulatoryGating()
        for _ in range(10):
            nm.update(surprise=0.9)
        lr_high = nm.learning_rate_scale()

        nm.reset()
        for _ in range(10):
            nm.update(surprise=0.1)
        lr_low = nm.learning_rate_scale()

        self.assertGreater(lr_high, lr_low)

    def test_high_arousal_broadens_retrieval(self):
        nm = NeuromodulatoryGating()
        # High variance in surprise → high arousal → lower beta scale
        for s in [0.1, 0.9, 0.2, 0.8, 0.1, 0.9]:
            nm.update(surprise=s)
        high_var_beta = nm.beta_scale()

        nm.reset()
        for s in [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]:
            nm.update(surprise=s)
        low_var_beta = nm.beta_scale()

        # High variance should give lower beta (broader)
        self.assertLess(high_var_beta, low_var_beta)

    def test_reward_modulates_consolidation(self):
        nm = NeuromodulatoryGating()
        nm.update(surprise=0.5, reward=1.0)
        high_reward = nm.consolidation_scale()

        nm.reset()
        nm.update(surprise=0.5, reward=0.0)
        no_reward = nm.consolidation_scale()

        self.assertGreater(high_reward, no_reward)

    def test_reset(self):
        nm = NeuromodulatoryGating()
        for _ in range(10):
            nm.update(surprise=0.9)
        nm.reset()
        # After reset, should be back to default
        self.assertAlmostEqual(nm._novelty, 0.5)
        self.assertAlmostEqual(nm._arousal, 0.5)
        self.assertAlmostEqual(nm._reward, 0.0)
        self.assertAlmostEqual(nm._temporal, 0.5)
        self.assertEqual(len(nm._surprise_history), 0)

    def test_sparsity_scale_positive(self):
        nm = NeuromodulatoryGating()
        nm.update(surprise=0.8)
        self.assertGreater(nm.sparsity_scale(), 0)

    def test_dendrite_threshold_offset(self):
        nm = NeuromodulatoryGating()
        # Low surprise → high temporal → negative offset (lower threshold)
        for _ in range(10):
            nm.update(surprise=0.1)
        offset = nm.dendrite_threshold_offset()
        self.assertLess(offset, 0)

    def test_state_dict_roundtrip(self):
        """State dict should preserve all internal neuromodulatory state."""
        nm = NeuromodulatoryGating()
        for s in [0.9, 0.3, 0.7, 0.1]:
            nm.update(surprise=s, reward=0.5)

        sd = nm.state_dict()

        nm2 = NeuromodulatoryGating()
        nm2.load_state_dict(sd)

        self.assertAlmostEqual(nm._novelty, nm2._novelty)
        self.assertAlmostEqual(nm._arousal, nm2._arousal)
        self.assertAlmostEqual(nm._reward, nm2._reward)
        self.assertAlmostEqual(nm._temporal, nm2._temporal)
        self.assertEqual(nm._surprise_history, nm2._surprise_history)

        # Scales should match after reload
        self.assertAlmostEqual(nm.beta_scale(), nm2.beta_scale())
        self.assertAlmostEqual(nm.learning_rate_scale(), nm2.learning_rate_scale())
        self.assertAlmostEqual(nm.consolidation_scale(), nm2.consolidation_scale())

    def test_all_scales_bounded(self):
        """All scale factors should remain positive across a range of inputs."""
        nm = NeuromodulatoryGating()
        for surprise in [0.0, 0.1, 0.5, 0.9, 1.0]:
            for reward in [-1.0, 0.0, 0.5, 1.0]:
                nm.update(surprise=surprise, reward=reward)
                self.assertGreater(nm.beta_scale(), 0)
                self.assertGreater(nm.learning_rate_scale(), 0)
                self.assertGreater(nm.sparsity_scale(), 0)
                self.assertGreaterEqual(nm.consolidation_scale(), 1.0)

    def test_surprise_history_window(self):
        """Surprise history should not exceed window size (20)."""
        nm = NeuromodulatoryGating()
        for i in range(30):
            nm.update(surprise=float(i) / 30.0)
        self.assertEqual(len(nm._surprise_history), 20)

    def test_novelty_learning_rate_end_to_end(self):
        """High novelty (high surprise) should produce measurably higher
        learning rate scaling than low novelty.

        This is an end-to-end test: after a sequence of high-surprise steps,
        the learning_rate_scale should be at least 1.5× the low-surprise case.
        """
        nm_novel = NeuromodulatoryGating(ach_sensitivity=2.0)
        nm_familiar = NeuromodulatoryGating(ach_sensitivity=2.0)

        # Novel: 20 steps of high surprise
        for _ in range(20):
            nm_novel.update(surprise=0.95)

        # Familiar: 20 steps of low surprise
        for _ in range(20):
            nm_familiar.update(surprise=0.05)

        ratio = nm_novel.learning_rate_scale() / nm_familiar.learning_rate_scale()
        self.assertGreater(
            ratio, 1.5,
            f"Novel/familiar learning rate ratio={ratio:.2f}, expected >1.5. "
            f"Neuromodulation has insufficient effect on learning rate.",
        )

    def test_arousal_beta_effect(self):
        """Highly aroused state (variable surprise) should produce measurably
        lower beta_scale than calm state (steady surprise)."""
        nm_aroused = NeuromodulatoryGating()
        nm_calm = NeuromodulatoryGating()

        # Aroused: alternating surprise
        for _ in range(20):
            nm_aroused.update(surprise=0.9)
            nm_aroused.update(surprise=0.1)

        # Calm: steady surprise
        for _ in range(40):
            nm_calm.update(surprise=0.5)

        ratio = nm_aroused.beta_scale() / nm_calm.beta_scale()
        self.assertLess(
            ratio, 0.9,
            f"Aroused/calm beta ratio={ratio:.2f}, expected <0.9. "
            f"Arousal is not sufficiently lowering temperature.",
        )


if __name__ == "__main__":
    unittest.main()
