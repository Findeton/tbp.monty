# Copyright 2025-2026 Thousand Brains Project
# MIT License

"""Tests for inhibitory interneuron populations (PV+, SST+, VIP+)."""

import unittest

import torch

from tbp.monty.frameworks.models.cortical_column_torch.interneurons import (
    InterneuronCircuit,
    PVInhibition,
    SSTInhibition,
    VIPDisinhibition,
)


class TestPVInhibition(unittest.TestCase):
    def test_layer_specific_sparsity(self):
        pv = PVInhibition(
            n_minicolumns=100,
            layer_sparsities={"L4": 0.05, "L23": 0.1},
        )
        overlap = torch.randn(100)
        winners_l4 = pv.inhibit(overlap, "L4")
        winners_l23 = pv.inhibit(overlap, "L23")
        # L23 should have more winners (higher sparsity)
        self.assertGreater(winners_l23.sum().item(), winners_l4.sum().item())

    def test_override_n_active(self):
        pv = PVInhibition(n_minicolumns=50)
        overlap = torch.randn(50)
        winners = pv.inhibit(overlap, "L4", n_active_override=10)
        self.assertEqual(winners.sum().item(), 10)


class TestSSTInhibition(unittest.TestCase):
    def test_gate_returns_valid_range(self):
        sst = SSTInhibition(n_cells=64, n_heads=4)
        act = torch.randn(64)
        gate = sst.compute_gate(act)
        self.assertEqual(gate.shape, (4,))
        self.assertTrue((gate >= 0).all() and (gate <= 1).all())

    def test_gating_with_vip_disinhibition(self):
        sst = SSTInhibition(n_cells=32, n_heads=4)
        act = torch.randn(32)
        # Without VIP
        gate_no_vip = sst.compute_gate(act)
        # With strong VIP (disinhibition → gate opens)
        vip_signal = torch.ones(4)
        gate_with_vip = sst.compute_gate(act, vip_disinhibition=vip_signal)
        # VIP disinhibition should make gate more open (higher values)
        self.assertGreater(gate_with_vip.mean().item(),
                           gate_no_vip.mean().item())

    def test_learn_changes_weights(self):
        sst = SSTInhibition(n_cells=16, n_heads=2, learning_rate=0.1)
        act = torch.randn(16)
        sst.compute_gate(act)
        w_before = sst._W_gate.clone()
        sst.learn(act, surprise=0.1, lr=1.0)
        self.assertFalse(torch.equal(w_before, sst._W_gate))


class TestVIPDisinhibition(unittest.TestCase):
    def test_high_novelty_activates_vip(self):
        vip = VIPDisinhibition(n_heads=4, novelty_weight=2.0)
        act_high = vip.compute(novelty=0.9)
        vip.reset()
        act_low = vip.compute(novelty=0.1)
        self.assertGreater(act_high.mean().item(), act_low.mean().item())

    def test_attention_adds_to_activation(self):
        vip = VIPDisinhibition(n_heads=4, attention_weight=1.0)
        attn = torch.ones(4)
        act_with = vip.compute(novelty=0.5, top_down_attention=attn)
        vip.reset()
        act_without = vip.compute(novelty=0.5)
        self.assertGreater(act_with.mean().item(), act_without.mean().item())

    def test_reset(self):
        vip = VIPDisinhibition(n_heads=4)
        vip.compute(novelty=0.9)
        self.assertGreater(vip._activation.sum().item(), 0)
        vip.reset()
        self.assertEqual(vip._activation.sum().item(), 0.0)


class TestInterneuronCircuit(unittest.TestCase):
    def test_full_circuit(self):
        circuit = InterneuronCircuit(
            n_minicolumns=32, n_cells=256, n_heads=4,
        )
        act = torch.randn(256)
        gate = circuit.compute_branch_gate(act, novelty=0.5)
        self.assertEqual(gate.shape, (4,))
        self.assertTrue((gate >= 0).all() and (gate <= 1).all())

    def test_high_novelty_opens_branches(self):
        circuit = InterneuronCircuit(
            n_minicolumns=16, n_cells=128, n_heads=4,
        )
        act = torch.randn(128)
        gate_novel = circuit.compute_branch_gate(act, novelty=0.95)
        circuit.reset()
        gate_familiar = circuit.compute_branch_gate(act, novelty=0.05)
        # High novelty → VIP → SST suppressed → branches more open
        self.assertGreater(gate_novel.mean().item(),
                           gate_familiar.mean().item())

    def test_learn(self):
        circuit = InterneuronCircuit(
            n_minicolumns=8, n_cells=64, n_heads=2,
        )
        act = torch.randn(64)
        circuit.compute_branch_gate(act, novelty=0.5)
        circuit.learn(act, surprise=0.3, lr=1.0)
        # Should not crash


if __name__ == "__main__":
    unittest.main()
