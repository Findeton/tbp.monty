# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Tests for GlobalIntervalTimer (Phase 4 of object behaviors)."""

import unittest

import numpy as np

from tbp.monty.frameworks.models.interval_timer import GlobalIntervalTimer


class TestTimerBasic(unittest.TestCase):
    """Basic timer construction and step/reset."""

    def test_initial_state(self):
        timer = GlobalIntervalTimer(n_time_cells=32)
        self.assertEqual(timer.get_current_tick(), 0.0)
        self.assertEqual(timer.get_active_cell(), 0)
        self.assertEqual(timer.n_time_cells, 32)
        self.assertEqual(timer.speed_multiplier, 1.0)

    def test_step_advances_tick(self):
        timer = GlobalIntervalTimer(n_time_cells=16, ticks_per_step=1.0)
        timer.step()
        self.assertGreater(timer.get_current_tick(), 0.0)

    def test_multiple_steps(self):
        timer = GlobalIntervalTimer(n_time_cells=16, ticks_per_step=1.0)
        for _ in range(5):
            timer.step()
        self.assertAlmostEqual(timer.get_current_tick(), 5.0)

    def test_reset(self):
        timer = GlobalIntervalTimer(n_time_cells=16)
        timer.step()
        timer.step()
        timer.reset()
        self.assertEqual(timer.get_current_tick(), 0.0)
        self.assertEqual(timer.get_active_cell(), 0)

    def test_step_then_reset_then_step(self):
        timer = GlobalIntervalTimer(n_time_cells=16)
        timer.step()
        timer.step()
        timer.reset()
        timer.step()
        self.assertAlmostEqual(timer.get_current_tick(), 1.0)


class TestTimerEncoding(unittest.TestCase):
    """Test time encoding vector."""

    def test_encoding_shape(self):
        timer = GlobalIntervalTimer(n_time_cells=32)
        encoding = timer.get_time_encoding()
        self.assertEqual(encoding.shape, (32,))

    def test_encoding_at_zero(self):
        """At tick=0, first cell should be most active."""
        timer = GlobalIntervalTimer(n_time_cells=16)
        encoding = timer.get_time_encoding()
        self.assertEqual(np.argmax(encoding), 0)
        self.assertAlmostEqual(encoding[0], 1.0)

    def test_encoding_changes_with_steps(self):
        """Encoding should change as timer advances."""
        timer = GlobalIntervalTimer(n_time_cells=16, ticks_per_step=1.0)
        enc0 = timer.get_time_encoding().copy()
        timer.step()
        timer.step()
        timer.step()
        enc3 = timer.get_time_encoding()
        # They should differ
        self.assertFalse(np.allclose(enc0, enc3))

    def test_active_cell_advances(self):
        """Active cell should increase as timer advances."""
        timer = GlobalIntervalTimer(n_time_cells=16, ticks_per_step=1.0)
        cells = []
        for _ in range(15):
            cells.append(timer.get_active_cell())
            timer.step()
        # Active cell should generally increase (monotonic with possible ties
        # due to log spacing)
        self.assertGreater(cells[-1], cells[0])

    def test_encoding_is_normalized(self):
        """Max activation should be 1.0."""
        timer = GlobalIntervalTimer(n_time_cells=16)
        for _ in range(5):
            timer.step()
        encoding = timer.get_time_encoding()
        self.assertAlmostEqual(encoding.max(), 1.0)

    def test_encoding_non_negative(self):
        """All activations should be non-negative."""
        timer = GlobalIntervalTimer(n_time_cells=16)
        for _ in range(10):
            encoding = timer.get_time_encoding()
            self.assertTrue(np.all(encoding >= 0))
            timer.step()


class TestTimerSpeed(unittest.TestCase):
    """Test speed adjustment."""

    def test_adjust_speed(self):
        timer = GlobalIntervalTimer(n_time_cells=16, ticks_per_step=1.0)
        timer.adjust_speed(2.0)
        self.assertAlmostEqual(timer.speed_multiplier, 2.0)
        timer.step()
        # With 2x speed, one step = 2 ticks
        self.assertAlmostEqual(timer.get_current_tick(), 2.0)

    def test_set_speed(self):
        timer = GlobalIntervalTimer(n_time_cells=16, ticks_per_step=1.0)
        timer.set_speed(0.5)
        self.assertAlmostEqual(timer.speed_multiplier, 0.5)
        timer.step()
        self.assertAlmostEqual(timer.get_current_tick(), 0.5)

    def test_speed_double_same_encoding_at_half_steps(self):
        """At 2x speed, step N should match 1x speed at step 2N."""
        timer_normal = GlobalIntervalTimer(
            n_time_cells=16, ticks_per_step=1.0, sigma=1.5,
        )
        timer_fast = GlobalIntervalTimer(
            n_time_cells=16, ticks_per_step=1.0, sigma=1.5,
        )
        timer_fast.set_speed(2.0)

        # Advance normal by 4 steps, fast by 2 steps
        for _ in range(4):
            timer_normal.step()
        for _ in range(2):
            timer_fast.step()

        # Both should be at tick=4
        self.assertAlmostEqual(
            timer_normal.get_current_tick(),
            timer_fast.get_current_tick(),
        )
        np.testing.assert_allclose(
            timer_normal.get_time_encoding(),
            timer_fast.get_time_encoding(),
        )


class TestTimerClamping(unittest.TestCase):
    """Test that timer clamps at maximum."""

    def test_clamp_at_max(self):
        timer = GlobalIntervalTimer(n_time_cells=8, ticks_per_step=1.0)
        # Step way beyond the range
        for _ in range(100):
            timer.step()
        # Should be clamped, not infinite
        self.assertTrue(np.isfinite(timer.get_current_tick()))
        # Active cell should be the last one
        self.assertEqual(timer.get_active_cell(), 7)


class TestTimerLogSpacing(unittest.TestCase):
    """Test logarithmic cell spacing."""

    def test_early_cells_closer_together(self):
        """Early cells should be closer together than late cells."""
        timer = GlobalIntervalTimer(n_time_cells=16)
        centers = timer._cell_centers
        # Gap between cell 0 and 1
        early_gap = centers[1] - centers[0]
        # Gap between cell 14 and 15
        late_gap = centers[-1] - centers[-2]
        self.assertLess(early_gap, late_gap)

    def test_single_cell(self):
        timer = GlobalIntervalTimer(n_time_cells=1)
        self.assertEqual(timer.get_active_cell(), 0)
        encoding = timer.get_time_encoding()
        self.assertEqual(encoding.shape, (1,))


if __name__ == "__main__":
    unittest.main()
