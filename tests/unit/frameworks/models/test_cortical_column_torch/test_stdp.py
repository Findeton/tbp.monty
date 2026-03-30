# Copyright 2025-2026 Thousand Brains Project
# MIT License

"""Tests for STDP and eligibility traces."""

import unittest

import torch

from tbp.monty.frameworks.models.cortical_column_torch.stdp import (
    EligibilityTrace,
    STDPRule,
)


class TestSTDPRule(unittest.TestCase):
    def test_pre_before_post_strengthens(self):
        """Pre fires, then post fires → LTP (synapse strengthened)."""
        stdp = STDPRule(n_cells=16, lr_ltp=0.1, lr_ltd=0.05,
                        trace_decay=0.8)
        pre = torch.zeros(16)
        pre[0] = 1.0  # Pre fires first
        post = torch.zeros(16)

        # Step 1: pre fires
        stdp.update_traces(pre, post)

        # Step 2: post fires
        post[5] = 1.0
        pre2 = torch.zeros(16)
        stdp.update_traces(pre2, post)

        # Check: trace_pre[0] should still be nonzero (decayed)
        self.assertGreater(stdp._trace_pre[0].item(), 0)

        # Compute update: post=5, trace_pre[0] active → LTP for (5,0)
        dw = stdp.compute_weight_update(pre2, post)
        self.assertGreater(dw[5, 0].item(), 0)  # LTP

    def test_post_before_pre_weakens(self):
        """Post fires, then pre fires → LTD (synapse weakened)."""
        stdp = STDPRule(n_cells=16, lr_ltp=0.1, lr_ltd=0.1,
                        trace_decay=0.8)
        pre = torch.zeros(16)
        post = torch.zeros(16)
        post[5] = 1.0  # Post fires first

        stdp.update_traces(pre, post)

        # Step 2: pre fires
        pre2 = torch.zeros(16)
        pre2[0] = 1.0
        post2 = torch.zeros(16)
        stdp.update_traces(pre2, post2)

        dw = stdp.compute_weight_update(pre2, post2)
        # LTD: pre=0 active, trace_post[5] active → weaken (5,0) direction
        # In matrix: dw[5,0] should be negative (or the transpose convention)
        # The LTD term contributes negatively
        self.assertLess(dw[5, 0].item(), 0)

    def test_symmetric_coactivation_cancels(self):
        """Simultaneous pre and post → LTP and LTD roughly cancel."""
        stdp = STDPRule(n_cells=8, lr_ltp=0.1, lr_ltd=0.1,
                        trace_decay=0.8)
        both = torch.zeros(8)
        both[0] = 1.0
        both[1] = 1.0

        stdp.update_traces(both, both)
        dw = stdp.compute_weight_update(both, both)
        # Net change should be small for symmetric case
        self.assertLess(abs(dw[0, 1].item() + dw[1, 0].item()), 0.5)

    def test_trace_decay(self):
        stdp = STDPRule(n_cells=8, trace_decay=0.5)
        pre = torch.zeros(8)
        pre[0] = 1.0
        stdp.update_traces(pre, torch.zeros(8))
        val1 = stdp._trace_pre[0].item()
        stdp.update_traces(torch.zeros(8), torch.zeros(8))
        val2 = stdp._trace_pre[0].item()
        self.assertAlmostEqual(val2, val1 * 0.5, places=5)

    def test_sparse_update(self):
        stdp = STDPRule(n_cells=16, lr_ltp=0.1, lr_ltd=0.05)
        pre = torch.zeros(16)
        pre[0] = 1.0
        post = torch.zeros(16)
        stdp.update_traces(pre, post)

        post[5] = 1.0
        stdp.update_traces(torch.zeros(16), post)

        segments = [(5, {0: 0.5, 1: 0.5})]
        updates = stdp.compute_sparse_update(
            torch.zeros(16), post, segments
        )
        # Should have an update for synapse (5, 0)
        self.assertIn((5, 0), updates)

    def test_apply_sparse_update(self):
        stdp = STDPRule(n_cells=8)
        segments = [(0, {1: 0.5})]
        updates = {(0, 1): 0.1}
        stdp.apply_sparse_update(segments, updates)
        self.assertAlmostEqual(segments[0][1][1], 0.6)

    def test_reset(self):
        stdp = STDPRule(n_cells=8)
        stdp.update_traces(torch.ones(8), torch.ones(8))
        stdp.reset()
        self.assertEqual(stdp._trace_pre.sum().item(), 0.0)
        self.assertEqual(stdp._trace_post.sum().item(), 0.0)


class TestEligibilityTrace(unittest.TestCase):
    def test_accumulate_and_consolidate(self):
        et = EligibilityTrace(decay=0.99)
        updates = {(0, 1): 0.1, (2, 3): -0.05}
        et.accumulate(updates)
        self.assertEqual(et.n_eligible, 2)

        segments = [(0, {1: 0.5}), (2, {3: 0.5})]
        n_mod = et.consolidate(neuromod_signal=1.0, connectivity=segments)
        self.assertGreater(n_mod, 0)
        # Synapse (0,1) should be strengthened
        self.assertGreater(segments[0][1][1], 0.5)

    def test_no_consolidation_without_signal(self):
        et = EligibilityTrace(decay=0.99)
        et.accumulate({(0, 1): 0.1})
        segments = [(0, {1: 0.5})]
        n_mod = et.consolidate(neuromod_signal=0.0, connectivity=segments)
        self.assertEqual(n_mod, 0)
        self.assertAlmostEqual(segments[0][1][1], 0.5)

    def test_eligibility_decays(self):
        et = EligibilityTrace(decay=0.5)
        et.accumulate({(0, 1): 1.0})
        val1 = et._traces[(0, 1)]
        et.accumulate({})  # Just decay
        val2 = et._traces.get((0, 1), 0.0)
        self.assertLess(abs(val2), abs(val1))

    def test_delayed_reward(self):
        """Reward N steps after co-activation still consolidates."""
        et = EligibilityTrace(decay=0.95)
        et.accumulate({(0, 1): 0.5})

        # Wait 10 steps
        for _ in range(10):
            et.accumulate({})

        # Eligibility should still exist (decayed but nonzero)
        self.assertGreater(et.n_eligible, 0)

        segments = [(0, {1: 0.5})]
        n = et.consolidate(neuromod_signal=2.0, connectivity=segments)
        self.assertGreater(n, 0)
        # Should still modify (though less than immediate)
        self.assertNotAlmostEqual(segments[0][1][1], 0.5, places=3)

    def test_negative_reward_weakens(self):
        et = EligibilityTrace(decay=0.99)
        et.accumulate({(0, 1): 0.1})
        segments = [(0, {1: 0.5})]
        et.consolidate(neuromod_signal=-1.0, connectivity=segments)
        self.assertLess(segments[0][1][1], 0.5)

    def test_clear(self):
        et = EligibilityTrace()
        et.accumulate({(0, 1): 0.5})
        et.clear()
        self.assertEqual(et.n_eligible, 0)


if __name__ == "__main__":
    unittest.main()
