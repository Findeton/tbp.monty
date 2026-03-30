# Copyright 2025-2026 Thousand Brains Project
# MIT License

"""Tests for thalamocortical loop gating."""

import unittest

import torch

from tbp.monty.frameworks.models.cortical_column_torch.thalamic_relay import (
    ThalamicRelay,
)


class TestThalamicRelay(unittest.TestCase):
    def test_no_feedback_gate_open(self):
        relay = ThalamicRelay(n_input=32, n_l6_cells=16)
        inp = torch.ones(32)
        out = relay.forward(inp, l6_activation=None)
        self.assertTrue(torch.allclose(inp, out))

    def test_gate_modulates_input(self):
        relay = ThalamicRelay(n_input=32, n_l6_cells=16,
                               initial_gate_bias=2.0)
        inp = torch.ones(32)
        l6 = torch.randn(16)
        out = relay.forward(inp, l6_activation=l6)
        self.assertEqual(out.shape, (32,))
        # Gate is sigmoid → values between 0 and 1 → output ≤ input
        self.assertTrue((out.abs() <= inp.abs() + 0.01).all())

    def test_gate_values_in_range(self):
        relay = ThalamicRelay(n_input=20, n_l6_cells=10)
        l6 = torch.randn(10)
        relay.forward(torch.ones(20), l6_activation=l6)
        self.assertTrue((relay.gate >= 0).all() and (relay.gate <= 1).all())

    def test_learn_changes_weights(self):
        relay = ThalamicRelay(n_input=16, n_l6_cells=8,
                               learning_rate=0.1)
        l6 = torch.randn(8)
        relay.forward(torch.ones(16), l6_activation=l6)
        w_before = relay._W_gate.clone()
        relay.learn(surprise=0.9, l6_activation=l6)
        self.assertFalse(torch.equal(w_before, relay._W_gate))

    def test_high_surprise_opens_gate(self):
        """After learning from high surprise, gate should open more."""
        relay = ThalamicRelay(n_input=16, n_l6_cells=8,
                               learning_rate=0.5)
        l6 = torch.ones(8) * 0.5
        inp = torch.ones(16)

        # Baseline gate
        relay.forward(inp, l6)
        gate_before = relay.gate.mean().item()

        # Learn from high surprise repeatedly
        for _ in range(20):
            relay.forward(inp, l6)
            relay.learn(surprise=0.95, l6_activation=l6)

        relay.forward(inp, l6)
        gate_after = relay.gate.mean().item()
        # Gate should be more open after high surprise learning
        self.assertGreater(gate_after, gate_before)

    def test_low_surprise_closes_gate(self):
        """After learning from low surprise, gate should close."""
        relay = ThalamicRelay(n_input=16, n_l6_cells=8,
                               learning_rate=0.5)
        l6 = torch.ones(8) * 0.5
        inp = torch.ones(16)

        relay.forward(inp, l6)
        gate_before = relay.gate.mean().item()

        for _ in range(20):
            relay.forward(inp, l6)
            relay.learn(surprise=0.05, l6_activation=l6)

        relay.forward(inp, l6)
        gate_after = relay.gate.mean().item()
        self.assertLess(gate_after, gate_before)

    def test_reset(self):
        relay = ThalamicRelay(n_input=16, n_l6_cells=8)
        l6 = torch.randn(8)
        relay.forward(torch.ones(16), l6)
        relay.reset()
        self.assertTrue(torch.allclose(relay.gate, torch.ones(16)))

    def test_disable_reproduces_passthrough(self):
        """Without L6 feedback, relay is a passthrough."""
        relay = ThalamicRelay(n_input=32, n_l6_cells=16)
        inp = torch.randn(32)
        out = relay.forward(inp)
        self.assertTrue(torch.allclose(inp, out))


if __name__ == "__main__":
    unittest.main()
