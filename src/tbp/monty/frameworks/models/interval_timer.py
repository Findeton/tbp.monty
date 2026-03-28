# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Global interval timer for behavioral sequence timing.

Inspired by time cells in the thalamus (matrix cells) that fire at different
temporal delays from the start of an interval. The timer broadcasts elapsed
time since the last significant event to all learning modules via L1 input.

The timer is discrete: different "time cells" become active at different
delays, tiling the temporal space. Resolution is higher for short intervals
(logarithmic tiling). The speed can be adjusted during inference to recognize
behaviors at different tempos.

Phase 7 additions: multi-LM coordination via receive_reset_signal() and
receive_speed_adjustment(). Multiple LMs can independently signal events
and speed corrections; the timer applies them using consensus thresholds.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class GlobalIntervalTimer:
    """Discrete timer counting elapsed time since the last event.

    Parameters
    ----------
    n_time_cells : int
        Number of discrete time cells tiling the interval.
    ticks_per_step : float
        How many ticks to advance per Monty step at default speed.
    sigma : float
        Gaussian blur sigma for the time encoding (in time-cell units).
        Larger sigma = smoother, more overlapping time cell activations.
    """

    def __init__(
        self,
        n_time_cells: int = 32,
        ticks_per_step: float = 1.0,
        sigma: float = 1.0,
        reset_threshold: int = 1,
    ):
        self._n_time_cells = n_time_cells
        self._ticks_per_step = ticks_per_step
        self._sigma = sigma
        self._speed_multiplier = 1.0
        self._current_tick = 0.0
        self._reset_threshold = reset_threshold

        # Per-step signal accumulators (cleared each step)
        self._pending_resets: Dict[str, bool] = {}
        self._pending_speed_adjustments: Dict[str, float] = {}

        # Pre-compute cell centers for encoding
        # Logarithmic tiling: cells cover exponentially growing intervals
        # Cell 0 = tick 0, cell N-1 = max tick. Higher resolution at start.
        self._cell_centers = self._compute_cell_centers()

    def step(self) -> None:
        """Advance the timer by one step (scaled by speed multiplier)."""
        self._current_tick += self._ticks_per_step * self._speed_multiplier
        # Clamp at the last cell center + margin
        max_tick = self._cell_centers[-1] * 1.1
        self._current_tick = min(self._current_tick, max_tick)

    def reset(self) -> None:
        """Reset the timer to the beginning of the interval."""
        self._current_tick = 0.0

    def get_active_cell(self) -> int:
        """Return the index of the currently most-active time cell."""
        distances = np.abs(self._cell_centers - self._current_tick)
        return int(np.argmin(distances))

    def get_time_encoding(self) -> np.ndarray:
        """Return a soft encoding of the current time as activations.

        Returns a vector of shape (n_time_cells,) where each element
        represents how strongly that time cell is activated. The encoding
        is a Gaussian-blurred version of the current position.
        """
        distances = self._cell_centers - self._current_tick
        encoding = np.exp(-0.5 * (distances / self._sigma) ** 2)
        # Normalize so max activation is 1.0
        max_val = encoding.max()
        if max_val > 0:
            encoding = encoding / max_val
        return encoding

    def get_current_tick(self) -> float:
        """Return the current tick value."""
        return self._current_tick

    @property
    def n_time_cells(self) -> int:
        """Number of time cells."""
        return self._n_time_cells

    @property
    def speed_multiplier(self) -> float:
        """Current speed multiplier."""
        return self._speed_multiplier

    def adjust_speed(self, factor: float) -> None:
        """Multiply the current speed by a factor.

        Args:
            factor: Multiplier for speed. >1 = faster, <1 = slower.
        """
        self._speed_multiplier *= factor

    def set_speed(self, speed: float) -> None:
        """Set the speed multiplier directly.

        Args:
            speed: New speed multiplier value.
        """
        self._speed_multiplier = speed

    # --- Multi-LM coordination (Phase 7) ---

    def receive_reset_signal(self, sender_id: str) -> None:
        """Accumulate a reset signal from an LM that detected a state event.

        Multiple LMs can independently detect events. The timer resets
        only when at least ``reset_threshold`` LMs signal within the
        same Monty step. Call ``apply_pending_signals()`` at the end
        of the step to apply.

        Args:
            sender_id: Identifier of the signaling LM.
        """
        self._pending_resets[sender_id] = True

    def receive_speed_adjustment(self, sender_id: str, factor: float) -> None:
        """Accumulate a speed correction signal from an LM.

        Each LM compares the expected time cell (from its model) with
        the actual time cell and produces a correction factor.
        The timer applies the median of all corrections when
        ``apply_pending_signals()`` is called.

        Args:
            sender_id: Identifier of the signaling LM.
            factor: Multiplicative correction (>1 = speed up, <1 = slow down).
        """
        self._pending_speed_adjustments[sender_id] = factor

    def apply_pending_signals(self) -> bool:
        """Apply accumulated reset and speed signals.

        Called once per Monty step after all LMs have been stepped.

        Returns:
            True if a reset was applied.
        """
        reset_applied = False

        # Apply reset if enough LMs signaled
        n_resets = len(self._pending_resets)
        if n_resets >= self._reset_threshold:
            logger.debug(
                f"Timer reset: {n_resets} LM(s) signaled "
                f"(threshold={self._reset_threshold})"
            )
            self.reset()
            reset_applied = True

        # Apply speed adjustment (median of LM corrections)
        if self._pending_speed_adjustments:
            factors = list(self._pending_speed_adjustments.values())
            median_factor = float(np.median(factors))
            # Clamp to prevent runaway speed
            median_factor = max(0.1, min(10.0, median_factor))
            self.adjust_speed(median_factor)
            logger.debug(
                f"Timer speed adjusted by {median_factor:.3f} "
                f"(from {len(factors)} LM(s))"
            )

        # Clear pending signals for next step
        self._pending_resets.clear()
        self._pending_speed_adjustments.clear()

        return reset_applied

    def get_expected_cell_for_tick(self, tick: float) -> int:
        """Return which time cell should be active at a given tick.

        Useful for LMs to compare model-expected timing with actual timing.
        """
        distances = np.abs(self._cell_centers - tick)
        return int(np.argmin(distances))

    @property
    def cell_centers(self) -> np.ndarray:
        """Tick positions of each time cell center."""
        return self._cell_centers.copy()

    def _compute_cell_centers(self) -> np.ndarray:
        """Compute the tick positions at which each time cell is centered.

        Uses logarithmic spacing so early time cells have higher temporal
        resolution (closer together) and later cells cover longer durations.
        This matches biological observations that temporal discrimination
        is finer for shorter intervals.
        """
        n = self._n_time_cells
        if n == 1:
            return np.array([0.0])

        # Exponential spacing: cells close together near t=0 (high temporal
        # resolution for short intervals), farther apart for long intervals.
        # Maps cell index i ∈ [0, n-1] → tick position via expm1.
        indices = np.arange(n, dtype=np.float64)
        normalized = indices / (n - 1)  # [0, 1]
        # expm1 grows slowly near 0, fast near 1 → denser at start
        centers = np.expm1(normalized * np.log(n)) / (n - 1) * (n - 1)
        return centers
