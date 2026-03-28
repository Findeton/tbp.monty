# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Tests for Phase 7: Event Detection and Timer Speed Adjustment.

Tests that:
- LMs detect state transition events (behavioral state changes)
- Timer receives and applies reset/speed signals from multiple LMs
- Multi-LM consensus thresholds work correctly
- Speed adjustment clamps to safe ranges
"""

import unittest

import numpy as np

from tbp.monty.frameworks.models.interval_timer import GlobalIntervalTimer


class TestTimerResetSignals(unittest.TestCase):
    """Test receive_reset_signal and apply_pending_signals."""

    def test_single_lm_reset_default_threshold(self):
        """One LM signaling resets timer (default threshold=1)."""
        timer = GlobalIntervalTimer(n_time_cells=16, ticks_per_step=1.0)
        for _ in range(5):
            timer.step()
        self.assertGreater(timer.get_current_tick(), 0)

        timer.receive_reset_signal("LM_0")
        reset = timer.apply_pending_signals()

        self.assertTrue(reset)
        self.assertEqual(timer.get_current_tick(), 0.0)

    def test_signals_cleared_after_apply(self):
        """Pending signals are cleared after apply_pending_signals."""
        timer = GlobalIntervalTimer(n_time_cells=16, ticks_per_step=1.0)
        for _ in range(3):
            timer.step()

        timer.receive_reset_signal("LM_0")
        timer.apply_pending_signals()

        # Second apply without new signals should not reset
        for _ in range(3):
            timer.step()
        reset = timer.apply_pending_signals()
        self.assertFalse(reset)
        self.assertGreater(timer.get_current_tick(), 0)

    def test_threshold_requires_multiple_lms(self):
        """Reset only when enough LMs signal (threshold=2)."""
        timer = GlobalIntervalTimer(
            n_time_cells=16, ticks_per_step=1.0, reset_threshold=2
        )
        for _ in range(5):
            timer.step()
        tick_before = timer.get_current_tick()

        # One LM signals → not enough
        timer.receive_reset_signal("LM_0")
        reset = timer.apply_pending_signals()
        self.assertFalse(reset)
        # Timer should have advanced (no reset)
        self.assertGreater(timer.get_current_tick(), 0)

    def test_threshold_met_with_two_lms(self):
        """Two LMs signaling meets threshold=2 → reset."""
        timer = GlobalIntervalTimer(
            n_time_cells=16, ticks_per_step=1.0, reset_threshold=2
        )
        for _ in range(5):
            timer.step()

        timer.receive_reset_signal("LM_0")
        timer.receive_reset_signal("LM_1")
        reset = timer.apply_pending_signals()

        self.assertTrue(reset)
        self.assertEqual(timer.get_current_tick(), 0.0)

    def test_duplicate_signals_from_same_lm(self):
        """Same LM signaling twice counts as one signal."""
        timer = GlobalIntervalTimer(
            n_time_cells=16, ticks_per_step=1.0, reset_threshold=2
        )
        for _ in range(5):
            timer.step()

        timer.receive_reset_signal("LM_0")
        timer.receive_reset_signal("LM_0")  # duplicate
        reset = timer.apply_pending_signals()

        self.assertFalse(reset)  # Only 1 unique LM


class TestTimerSpeedAdjustment(unittest.TestCase):
    """Test receive_speed_adjustment and apply_pending_signals."""

    def test_single_speed_adjustment(self):
        """Single LM speed correction is applied."""
        timer = GlobalIntervalTimer(n_time_cells=16, ticks_per_step=1.0)
        self.assertEqual(timer.speed_multiplier, 1.0)

        timer.receive_speed_adjustment("LM_0", 1.5)
        timer.apply_pending_signals()

        self.assertAlmostEqual(timer.speed_multiplier, 1.5)

    def test_median_of_multiple_adjustments(self):
        """Multiple LM corrections are combined via median."""
        timer = GlobalIntervalTimer(n_time_cells=16, ticks_per_step=1.0)

        timer.receive_speed_adjustment("LM_0", 1.2)
        timer.receive_speed_adjustment("LM_1", 1.8)
        timer.receive_speed_adjustment("LM_2", 1.5)
        timer.apply_pending_signals()

        # Median of [1.2, 1.5, 1.8] = 1.5
        self.assertAlmostEqual(timer.speed_multiplier, 1.5)

    def test_speed_clamped_high(self):
        """Speed correction clamped to prevent runaway."""
        timer = GlobalIntervalTimer(n_time_cells=16, ticks_per_step=1.0)

        timer.receive_speed_adjustment("LM_0", 100.0)
        timer.apply_pending_signals()

        # Clamped to 10.0 max
        self.assertAlmostEqual(timer.speed_multiplier, 10.0)

    def test_speed_clamped_low(self):
        """Speed correction clamped to prevent stall."""
        timer = GlobalIntervalTimer(n_time_cells=16, ticks_per_step=1.0)

        timer.receive_speed_adjustment("LM_0", 0.001)
        timer.apply_pending_signals()

        # Clamped to 0.1 min
        self.assertAlmostEqual(timer.speed_multiplier, 0.1)

    def test_speed_adjustments_cleared_after_apply(self):
        """Speed adjustments don't persist across steps."""
        timer = GlobalIntervalTimer(n_time_cells=16, ticks_per_step=1.0)

        timer.receive_speed_adjustment("LM_0", 2.0)
        timer.apply_pending_signals()
        speed_after_first = timer.speed_multiplier

        # No new adjustments → speed unchanged
        timer.apply_pending_signals()
        self.assertAlmostEqual(timer.speed_multiplier, speed_after_first)


class TestTimerResetAndSpeedCombined(unittest.TestCase):
    """Test reset and speed signals in the same step."""

    def test_reset_and_speed_in_same_step(self):
        """Both reset and speed adjustment apply in same step."""
        timer = GlobalIntervalTimer(n_time_cells=16, ticks_per_step=1.0)
        for _ in range(5):
            timer.step()

        timer.receive_reset_signal("LM_0")
        timer.receive_speed_adjustment("LM_0", 2.0)
        reset = timer.apply_pending_signals()

        self.assertTrue(reset)
        self.assertEqual(timer.get_current_tick(), 0.0)
        self.assertAlmostEqual(timer.speed_multiplier, 2.0)

    def test_timer_advances_at_new_speed_after_reset(self):
        """After reset + speed change, timer advances at new speed."""
        timer = GlobalIntervalTimer(n_time_cells=16, ticks_per_step=1.0)

        timer.receive_speed_adjustment("LM_0", 2.0)
        timer.apply_pending_signals()

        # Step at 2x speed
        timer.step()
        self.assertAlmostEqual(timer.get_current_tick(), 2.0)


class TestGetExpectedCellForTick(unittest.TestCase):
    """Test get_expected_cell_for_tick helper."""

    def test_tick_zero_is_cell_zero(self):
        timer = GlobalIntervalTimer(n_time_cells=16)
        self.assertEqual(timer.get_expected_cell_for_tick(0.0), 0)

    def test_large_tick_is_last_cell(self):
        timer = GlobalIntervalTimer(n_time_cells=16)
        max_center = timer.cell_centers[-1]
        self.assertEqual(
            timer.get_expected_cell_for_tick(max_center * 2),
            15,
        )

    def test_cell_centers_property(self):
        timer = GlobalIntervalTimer(n_time_cells=8)
        centers = timer.cell_centers
        self.assertEqual(len(centers), 8)
        # Should be monotonically increasing
        self.assertTrue(np.all(np.diff(centers) >= 0))


class TestEventDetectionInLM(unittest.TestCase):
    """Test _detect_event in EvidenceGraphLM."""

    def _make_lm(self):
        from tbp.monty.frameworks.models.evidence_matching.learning_module import (
            EvidenceGraphLM,
        )

        return EvidenceGraphLM(
            max_match_distance=0.01,
            tolerances={
                "patch": {
                    "hsv": [0.2, 1, 1],
                    "principal_curvatures_log": [2, 2],
                }
            },
            feature_weights={"patch": {"hsv": np.array([1, 0, 0])}},
            max_graph_size=1.0,
            hypotheses_updater_args=dict(
                initial_possible_poses="informed",
            ),
        )

    def test_no_event_when_no_state(self):
        """No event when MLH has no state tracking."""
        lm = self._make_lm()
        lm.current_mlh = {"state": None, "evidence": 5.0}
        lm.previous_mlh = {"state": None, "evidence": 4.0}
        self.assertFalse(lm._detect_event())

    def test_no_event_when_state_unchanged(self):
        """No event when state stays the same."""
        lm = self._make_lm()
        lm.current_mlh = {"state": 1, "evidence": 5.0}
        lm.previous_mlh = {"state": 1, "evidence": 4.0}
        self.assertFalse(lm._detect_event())

    def test_event_when_state_changes(self):
        """Event detected when state transitions."""
        lm = self._make_lm()
        lm.current_mlh = {"state": 2, "evidence": 3.0}
        lm.previous_mlh = {"state": 1, "evidence": 4.0}
        self.assertTrue(lm._detect_event())

    def test_no_event_when_no_previous_mlh(self):
        """No event on first step (no previous MLH)."""
        lm = self._make_lm()
        lm.current_mlh = {"state": 1, "evidence": 3.0}
        lm.previous_mlh = None
        self.assertFalse(lm._detect_event())

    def test_event_from_none_to_state(self):
        """Event detected when transitioning from no state to a state."""
        lm = self._make_lm()
        lm.current_mlh = {"state": 0, "evidence": 2.0}
        lm.previous_mlh = {"state": None, "evidence": 1.0}
        self.assertTrue(lm._detect_event())

    def test_get_event_signal_reflects_detection(self):
        """get_event_signal returns the result of _detect_event."""
        lm = self._make_lm()
        lm._event_signal = True
        self.assertTrue(lm.get_event_signal())
        lm._event_signal = False
        self.assertFalse(lm.get_event_signal())

    def test_get_speed_signal_default_none(self):
        """get_speed_signal returns None when no speed signal computed."""
        lm = self._make_lm()
        self.assertIsNone(lm.get_speed_signal())


class TestMontyTimerDispatch(unittest.TestCase):
    """Test _dispatch_timer_signals in MontyBase."""

    def test_dispatch_calls_timer_signals(self):
        """Monty dispatches LM signals to timer."""
        from tbp.monty.context import RuntimeContext

        timer = GlobalIntervalTimer(n_time_cells=16, ticks_per_step=1.0)
        for _ in range(5):
            timer.step()

        ctx = RuntimeContext(rng=np.random.RandomState(42), timer=timer)

        # Create a minimal mock LM with event signal
        class MockLM:
            learning_module_id = "LM_mock"

            def get_event_signal(self):
                return True

            def get_speed_signal(self):
                return 1.5

        # Create a minimal Monty-like object
        from tbp.monty.frameworks.models.monty_base import MontyBase

        class TestMonty(MontyBase):
            def __init__(self):
                self.learning_modules = [MockLM()]

        monty = TestMonty()
        monty._dispatch_timer_signals(ctx)

        # Timer should have been reset (event signal)
        self.assertEqual(timer.get_current_tick(), 0.0)
        # Speed should have been adjusted
        self.assertAlmostEqual(timer.speed_multiplier, 1.5)

    def test_dispatch_no_timer(self):
        """Dispatch is a no-op when no timer in context."""
        from tbp.monty.context import RuntimeContext
        from tbp.monty.frameworks.models.monty_base import MontyBase

        ctx = RuntimeContext(rng=np.random.RandomState(42))

        class TestMonty(MontyBase):
            def __init__(self):
                self.learning_modules = []

        monty = TestMonty()
        # Should not raise
        monty._dispatch_timer_signals(ctx)

    def test_dispatch_lm_without_signals(self):
        """LMs without signal methods are gracefully skipped."""
        from tbp.monty.context import RuntimeContext
        from tbp.monty.frameworks.models.monty_base import MontyBase

        timer = GlobalIntervalTimer(n_time_cells=16, ticks_per_step=1.0)
        for _ in range(3):
            timer.step()
        tick_before = timer.get_current_tick()

        ctx = RuntimeContext(rng=np.random.RandomState(42), timer=timer)

        class PlainLM:
            learning_module_id = "LM_plain"

        class TestMonty(MontyBase):
            def __init__(self):
                self.learning_modules = [PlainLM()]

        monty = TestMonty()
        monty._dispatch_timer_signals(ctx)

        # Timer should not have been reset
        self.assertEqual(timer.get_current_tick(), tick_before)


if __name__ == "__main__":
    unittest.main()
