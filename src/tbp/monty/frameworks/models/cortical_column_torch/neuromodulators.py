# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Neuromodulatory gating for Hopfield dynamics.

Four modulators wired to continuous dynamics:
- ACh (novelty) → learning rate scaling
- NE (arousal) → β (Hopfield temperature), sparsity level
- DA (reward) → consolidation
- 5-HT (temporal) → dendritic sigmoid center
"""

from __future__ import annotations

import math


class NeuromodulatoryGating:
    """Maps surprise/reward signals to modulation of Hopfield parameters.

    Parameters
    ----------
    ach_sensitivity : float
        How strongly novelty (surprise) modulates learning rate.
    ne_sensitivity : float
        How strongly arousal modulates temperature/sparsity.
    da_sensitivity : float
        How strongly reward modulates consolidation.
    ht_sensitivity : float
        How strongly temporal horizon modulates dendrite threshold.
    ema_alpha : float
        Smoothing factor for running averages.
    """

    def __init__(
        self,
        ach_sensitivity: float = 2.0,
        ne_sensitivity: float = 1.0,
        da_sensitivity: float = 1.0,
        ht_sensitivity: float = 0.5,
        ema_alpha: float = 0.1,
    ):
        self.ach_sensitivity = ach_sensitivity
        self.ne_sensitivity = ne_sensitivity
        self.da_sensitivity = da_sensitivity
        self.ht_sensitivity = ht_sensitivity
        self.ema_alpha = ema_alpha

        # Internal state
        self._novelty = 0.5     # ACh: running average of surprise
        self._arousal = 0.5     # NE: running average of surprise variance
        self._reward = 0.0      # DA: cumulative reward signal
        self._temporal = 0.5    # 5-HT: temporal horizon estimate

        self._surprise_history: list[float] = []
        self._surprise_ema = 0.5

    def update(self, surprise: float, reward: float = 0.0) -> None:
        """Update neuromodulator levels from current surprise and reward."""
        alpha = self.ema_alpha

        # ACh: novelty = running surprise
        self._novelty = (1 - alpha) * self._novelty + alpha * surprise

        # NE: arousal = surprise variance (high variance = uncertain)
        self._surprise_history.append(surprise)
        if len(self._surprise_history) > 20:
            self._surprise_history.pop(0)
        if len(self._surprise_history) >= 2:
            mean_s = sum(self._surprise_history) / len(self._surprise_history)
            var_s = sum((s - mean_s) ** 2 for s in self._surprise_history) / len(
                self._surprise_history
            )
            self._arousal = (1 - alpha) * self._arousal + alpha * min(var_s * 4, 1.0)

        # DA: reward
        self._reward = (1 - alpha) * self._reward + alpha * reward

        # 5-HT: temporal = inverse of surprise trend (stable = high horizon)
        self._temporal = 1.0 - self._novelty

    def beta_scale(self) -> float:
        """Scale factor for Hopfield β.  High arousal → low β → broader retrieval."""
        # Arousal reduces temperature: β_eff = β * (1 / (1 + ne * arousal))
        return 1.0 / (1.0 + self.ne_sensitivity * self._arousal)

    def learning_rate_scale(self) -> float:
        """Scale factor for Hebbian learning rate.  High novelty → fast learning."""
        return 1.0 + self.ach_sensitivity * self._novelty

    def consolidation_scale(self) -> float:
        """Scale factor for weight consolidation.  Positive reward strengthens."""
        return 1.0 + self.da_sensitivity * max(self._reward, 0.0)

    def dendrite_threshold_offset(self) -> float:
        """Offset for dendritic activation threshold.  High temporal → lower threshold."""
        return -self.ht_sensitivity * self._temporal

    def sparsity_scale(self) -> float:
        """Scale for top-k sparsity. High arousal → broader (more active cells)."""
        return 1.0 + 0.5 * self.ne_sensitivity * self._arousal

    def reset(self) -> None:
        self._novelty = 0.5
        self._arousal = 0.5
        self._reward = 0.0
        self._temporal = 0.5
        self._surprise_history.clear()
        self._surprise_ema = 0.5

    def state_dict(self) -> dict:
        """Serialize internal state for persistence across episodes."""
        return {
            "novelty": self._novelty,
            "arousal": self._arousal,
            "reward": self._reward,
            "temporal": self._temporal,
            "surprise_history": list(self._surprise_history),
            "surprise_ema": self._surprise_ema,
        }

    def load_state_dict(self, sd: dict) -> None:
        """Restore internal state from a saved dict."""
        self._novelty = sd["novelty"]
        self._arousal = sd["arousal"]
        self._reward = sd["reward"]
        self._temporal = sd["temporal"]
        self._surprise_history = list(sd["surprise_history"])
        self._surprise_ema = sd["surprise_ema"]
