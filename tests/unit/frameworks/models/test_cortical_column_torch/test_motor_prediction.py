# Copyright 2025-2026 Thousand Brains Project
# MIT License

"""Tests for contrastive Hebbian motor prediction."""

import unittest

import torch

from tbp.monty.frameworks.models.cortical_column_torch.motor_prediction import (
    ContrastiveMotorPrediction,
)


class TestContrastiveMotorPrediction(unittest.TestCase):
    def test_first_step_no_error(self):
        mp = ContrastiveMotorPrediction(location_dim=32, motor_dim=3)
        loc = torch.randn(32)
        disp = torch.zeros(3)
        result = mp.step(loc, disp)
        self.assertEqual(result["prediction_error"], 0.0)

    def test_prediction_error_exists(self):
        mp = ContrastiveMotorPrediction(location_dim=32, motor_dim=3)
        loc1 = torch.randn(32)
        loc2 = torch.randn(32)
        disp = torch.ones(3) * 0.1

        mp.step(loc1, disp)
        result = mp.step(loc2, disp)
        self.assertIsInstance(result["prediction_error"], float)

    def test_no_autograd(self):
        """Verify no gradient tracking — purely Hebbian, no backprop."""
        mp = ContrastiveMotorPrediction(location_dim=16, motor_dim=3)
        loc = torch.randn(16, requires_grad=False)
        disp = torch.zeros(3)

        mp.step(loc, disp)
        self.assertFalse(mp._W_ih.requires_grad)
        self.assertFalse(mp._W_ho.requires_grad)

    def test_weights_update_during_learning(self):
        """W_ih and W_ho should actually change when learn=True."""
        mp = ContrastiveMotorPrediction(
            location_dim=16, motor_dim=3, learning_rate=0.01,
        )
        loc1 = torch.randn(16)
        loc2 = torch.randn(16)
        disp = torch.ones(3) * 0.5

        mp.step(loc1, disp, learn=True)

        W_ih_before = mp._W_ih.clone()
        W_ho_before = mp._W_ho.clone()

        mp.step(loc2, disp, learn=True)

        # Both weight matrices should have changed
        self.assertFalse(
            torch.equal(W_ih_before, mp._W_ih),
            "W_ih did not change during learning — contrastive Hebbian is broken",
        )
        self.assertFalse(
            torch.equal(W_ho_before, mp._W_ho),
            "W_ho did not change during learning",
        )

    def test_no_weight_update_without_learning(self):
        """Weights should NOT change when learn=False."""
        mp = ContrastiveMotorPrediction(location_dim=16, motor_dim=3)
        loc1 = torch.randn(16)
        loc2 = torch.randn(16)
        disp = torch.ones(3)

        mp.step(loc1, disp, learn=False)

        W_ih_before = mp._W_ih.clone()
        W_ho_before = mp._W_ho.clone()

        mp.step(loc2, disp, learn=False)

        self.assertTrue(torch.equal(W_ih_before, mp._W_ih))
        self.assertTrue(torch.equal(W_ho_before, mp._W_ho))

    def test_prediction_error_decreases_over_training(self):
        """Repeated exposure to the same sensorimotor pattern should improve.

        This validates that contrastive Hebbian learning actually reduces
        prediction error — the fundamental learning signal.
        """
        torch.manual_seed(42)
        mp = ContrastiveMotorPrediction(
            location_dim=16, motor_dim=3,
            learning_rate=0.005, n_settle_iters=3,
        )

        # Fixed sensorimotor mapping: loc2 = loc1 + fixed_displacement_encoding
        loc1 = torch.randn(16)
        loc2 = loc1 + torch.randn(16) * 0.3  # deterministic target
        disp = torch.ones(3) * 0.1

        errors = []
        for _ in range(50):
            mp.step(loc1, disp, learn=True)
            result = mp.step(loc2, disp, learn=True)
            errors.append(result["prediction_error"])
            mp.reset()
            # Re-present the same pair
            mp.step(loc1, disp, learn=True)

        # Error in second half should be lower than first half
        first_half = sum(errors[:25]) / 25
        second_half = sum(errors[25:]) / 25
        self.assertLess(
            second_half, first_half,
            f"Error did not decrease: first_half={first_half:.4f}, "
            f"second_half={second_half:.4f}",
        )

    def test_hidden_clamped_differs_from_free(self):
        """The clamped phase must produce different hidden activations
        than the free phase — otherwise W_ih never updates."""
        mp = ContrastiveMotorPrediction(
            location_dim=16, motor_dim=3, n_settle_iters=3,
        )
        loc1 = torch.randn(16)
        loc2 = torch.randn(16)  # different from prediction
        disp = torch.ones(3) * 0.1

        # First step initializes prev_location
        mp.step(loc1, disp, learn=False)

        # Manually compute free and clamped hidden to verify they differ
        inp = torch.cat([loc1, disp])
        hidden_free = torch.relu(mp._W_ih @ inp)

        target = loc2[:mp.location_dim]
        hidden_clamped = hidden_free.clone()
        for _ in range(mp.n_settle_iters):
            feedback = mp._W_ho.t() @ target
            hidden_clamped = torch.relu(mp._W_ih @ inp + feedback)

        # They should differ because feedback from target changes hidden state
        self.assertFalse(
            torch.allclose(hidden_free, hidden_clamped, atol=1e-8),
            "Clamped hidden == free hidden — contrastive learning is degenerate",
        )

    def test_reset(self):
        mp = ContrastiveMotorPrediction(location_dim=16, motor_dim=3)
        mp.step(torch.randn(16), torch.zeros(3))
        mp.reset()
        self.assertIsNone(mp._prev_location)
        self.assertEqual(mp._prediction_error, 0.0)

    def test_state_dict_roundtrip(self):
        mp = ContrastiveMotorPrediction(location_dim=16, motor_dim=3)
        mp.step(torch.randn(16), torch.zeros(3))
        sd = mp.state_dict()

        mp2 = ContrastiveMotorPrediction(location_dim=16, motor_dim=3)
        mp2.load_state_dict(sd)
        self.assertTrue(torch.equal(mp._W_ih.cpu(), mp2._W_ih.cpu()))
        self.assertTrue(torch.equal(mp._W_ho.cpu(), mp2._W_ho.cpu()))


if __name__ == "__main__":
    unittest.main()
