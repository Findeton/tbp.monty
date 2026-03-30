# Copyright 2025-2026 Thousand Brains Project
# MIT License

"""Tests for plateau potential working memory."""

import unittest

import torch

from tbp.monty.frameworks.models.cortical_column_torch.plateau import (
    PlateauPotentialMemory,
)


class TestPlateauPotentialMemory(unittest.TestCase):
    def test_initial_state_is_zero(self):
        p = PlateauPotentialMemory(n_cells=32)
        self.assertEqual(p.state.sum().item(), 0.0)

    def test_plateau_triggered_by_coincidence(self):
        p = PlateauPotentialMemory(n_cells=16, plateau_threshold=0.2)
        activation = torch.zeros(16)
        activation[0] = 0.8
        has_ba = torch.zeros(16, dtype=torch.bool)
        has_ba[0] = True
        p.update(activation, has_ba)
        self.assertGreater(p.state[0].item(), 0)

    def test_no_plateau_without_apical(self):
        p = PlateauPotentialMemory(n_cells=16, plateau_threshold=0.2)
        activation = torch.zeros(16)
        activation[0] = 0.8
        has_ba = torch.zeros(16, dtype=torch.bool)  # no coincidence
        p.update(activation, has_ba)
        self.assertEqual(p.state[0].item(), 0.0)

    def test_exponential_decay(self):
        p = PlateauPotentialMemory(n_cells=8, tau=5.0, plateau_threshold=0.1)
        act = torch.ones(8) * 0.5
        ba = torch.ones(8, dtype=torch.bool)
        p.update(act, ba)
        val_after_1 = p.state[0].item()

        # Decay without new trigger
        empty_act = torch.zeros(8)
        empty_ba = torch.zeros(8, dtype=torch.bool)
        p.update(empty_act, empty_ba)
        val_after_2 = p.state[0].item()

        self.assertGreater(val_after_1, val_after_2)
        self.assertGreater(val_after_2, 0)

    def test_context_window_scales_with_tau(self):
        # High tau → slow decay → longer persistence
        p_long = PlateauPotentialMemory(n_cells=4, tau=50.0,
                                         plateau_threshold=0.1)
        p_short = PlateauPotentialMemory(n_cells=4, tau=2.0,
                                          plateau_threshold=0.1)
        act = torch.ones(4) * 0.5
        ba = torch.ones(4, dtype=torch.bool)
        p_long.update(act, ba)
        p_short.update(act, ba)

        empty_act = torch.zeros(4)
        empty_ba = torch.zeros(4, dtype=torch.bool)
        for _ in range(10):
            p_long.update(empty_act, empty_ba)
            p_short.update(empty_act, empty_ba)

        self.assertGreater(p_long.state[0].item(), p_short.state[0].item())

    def test_enrich_query(self):
        p = PlateauPotentialMemory(n_cells=8, gamma=0.5,
                                    plateau_threshold=0.1)
        act = torch.ones(8) * 0.5
        ba = torch.ones(8, dtype=torch.bool)
        p.update(act, ba)

        query = torch.ones(8) * 0.3
        enriched = p.enrich_query(query)
        self.assertTrue((enriched > query).any())

    def test_neuromodulator_controls_tau(self):
        p = PlateauPotentialMemory(n_cells=4, tau=10.0)
        self.assertAlmostEqual(p.tau, 10.0)
        p.tau = 50.0
        self.assertAlmostEqual(p.tau, 50.0)

    def test_reset(self):
        p = PlateauPotentialMemory(n_cells=8, plateau_threshold=0.1)
        act = torch.ones(8)
        ba = torch.ones(8, dtype=torch.bool)
        p.update(act, ba)
        self.assertGreater(p.state.sum().item(), 0)
        p.reset()
        self.assertEqual(p.state.sum().item(), 0.0)

    def test_state_dict_roundtrip(self):
        p = PlateauPotentialMemory(n_cells=8, tau=15.0,
                                    plateau_threshold=0.1)
        act = torch.ones(8) * 0.5
        ba = torch.ones(8, dtype=torch.bool)
        p.update(act, ba)
        sd = p.state_dict()

        p2 = PlateauPotentialMemory(n_cells=8)
        p2.load_state_dict(sd)
        self.assertTrue(torch.equal(p.state, p2.state))
        self.assertAlmostEqual(p2.tau, 15.0)


if __name__ == "__main__":
    unittest.main()
