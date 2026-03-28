# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Neuromodulatory gating for cortical column plasticity.

Implements the computational role of four neuromodulatory systems as scalar
signals that dynamically modulate existing column parameters. Each modulator
is updated based on internal column state (surprise, burst ratio, etc.)
and optionally external signals (reward).

Modulators:

- **Novelty (ACh analog)**: High burst ratio → trust feedforward more,
  learn more aggressively. Low burst ratio → trust predictions/recurrent.
- **Arousal (NE analog)**: Sustained surprise → broaden competition,
  grow more segments. Low surprise → narrow, maintenance only.
- **Reward (DA analog)**: External reward signal → consolidate recent
  learning. Negative reward → decay recent connections.
- **Temporal horizon (5-HT analog)**: Controls prediction depth via
  dendritic activation threshold.

All modulators are scalar floats in [0, 1] with exponential moving
average dynamics. Default values reproduce pre-Phase-5 behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class NeuromodulatoryState:
    """Current neuromodulatory state of a cortical column.

    All values in [0, 1]. Default 0.5 = neutral (no modulation).

    Parameters
    ----------
    novelty : float
        ACh analog. High → trust feedforward, increase SP learning rate.
    arousal : float
        NE analog. High → broaden competition (increase k in top-k).
    reward : float
        DA analog. High → consolidate, low → decay recent weights.
    temporal_horizon : float
        5-HT analog. High → lower activation threshold (more predictions).
    ema_rate : float
        Exponential moving average rate for updates.
    """

    novelty: float = 0.5
    arousal: float = 0.5
    reward: float = 0.5
    temporal_horizon: float = 0.5
    ema_rate: float = 0.1

    def update(
        self,
        burst_ratio: float = 0.5,
        surprise_ema: float = 0.5,
        reward_signal: float = 0.0,
    ) -> None:
        """Update modulator levels from column state.

        Parameters
        ----------
        burst_ratio : float
            Fraction of active minicolumns that are bursting [0, 1].
        surprise_ema : float
            Exponential moving average of surprise [0, 1].
        reward_signal : float
            External reward signal [-1, 1]. 0 = neutral.
        """
        a = self.ema_rate

        # Novelty tracks burst ratio directly
        self.novelty = (1 - a) * self.novelty + a * burst_ratio

        # Arousal tracks sustained surprise
        self.arousal = (1 - a) * self.arousal + a * surprise_ema

        # Reward: map [-1, 1] → [0, 1]
        reward_01 = np.clip(0.5 + 0.5 * reward_signal, 0.0, 1.0)
        self.reward = (1 - a) * self.reward + a * reward_01

        # Temporal horizon: inversely related to novelty
        # High novelty → focus on immediate → low horizon
        # Low novelty → extend predictions → high horizon
        self.temporal_horizon = (
            (1 - a) * self.temporal_horizon + a * (1.0 - burst_ratio)
        )

    def sp_learning_rate_scale(self) -> float:
        """Scale factor for spatial pooler learning rate.

        High novelty → aggressive learning (up to 2x).
        Low novelty → maintenance learning (down to 0.2x).
        """
        return 0.2 + 1.8 * self.novelty

    def settling_iterations_scale(self) -> float:
        """Scale factor for max settling iterations.

        High novelty → fewer iterations (trust feedforward).
        Low novelty → more iterations (trust recurrent).
        """
        return 0.5 + 1.0 * (1.0 - self.novelty)

    def competition_width_scale(self) -> float:
        """Scale factor for number of active minicolumns (k in top-k).

        High arousal → broader competition (up to 1.5x).
        Low arousal → narrower competition (down to 0.8x).
        """
        return 0.8 + 0.7 * self.arousal

    def segment_growth_scale(self) -> float:
        """Scale factor for dendritic segment growth probability.

        High arousal → grow more segments.
        """
        return 0.5 + 1.0 * self.arousal

    def consolidation_scale(self) -> float:
        """Scale factor for weight consolidation vs decay.

        High reward → strengthen (up to 2x permanence increment).
        Low reward → weaken (reduce permanence).
        """
        return 0.5 + 1.5 * self.reward

    def activation_threshold_offset(self) -> int:
        """Offset to dendritic activation threshold.

        High temporal_horizon → lower threshold (more predictions).
        Low temporal_horizon → higher threshold (fewer, confident).
        Returns value in [-3, +3].
        """
        return int(round(3.0 * (0.5 - self.temporal_horizon)))

    def reset(self) -> None:
        """Reset all modulators to neutral."""
        self.novelty = 0.5
        self.arousal = 0.5
        self.reward = 0.5
        self.temporal_horizon = 0.5
