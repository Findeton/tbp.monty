"""Lightweight semi-Markov transition memory over settled LM states.

This memory operates on discrete settled labels such as ``mug:0`` or
``walk_phase_3``. It learns:

- which label tends to follow which other label
- how long each label tends to persist before changing

The goal is not to replace the local current-state attractor. It complements
it with an explicit dwell-time-aware transition model, matching the Track 9b
requirement that temporal structure should not be forced into a static memory.
"""

from __future__ import annotations

from collections import Counter, defaultdict


class SemiMarkovTransitionMemory:
    """Learn discrete transitions plus per-state dwell durations.

    Parameters
    ----------
    min_stable_steps : int
        Minimum dwell length required before a state's duration contributes to
        the learned duration statistics.
    default_duration : float
        Fallback expected duration for states that have not yet been observed
        long enough to estimate a dwell time.
    """

    def __init__(
        self,
        min_stable_steps: int = 2,
        default_duration: float = 2.0,
    ):
        self.min_stable_steps = max(1, int(min_stable_steps))
        self.default_duration = float(default_duration)

        self._transition_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self._duration_totals: dict[str, float] = defaultdict(float)
        self._duration_counts: dict[str, int] = defaultdict(int)

        self.reset_episode()

    def reset_episode(self) -> None:
        """Reset episode-local run tracking while preserving learned memory."""
        self._current_label: str | None = None
        self._current_dwell = 0
        self._predicted_label: str | None = None
        self._prediction_status: str | None = None
        self._last_event_detected = False

    def finalize_episode(self, learn: bool = True) -> None:
        """Commit the final state's dwell duration at episode end."""
        if not learn:
            return
        if self._current_label is None:
            return
        if self._current_dwell < self.min_stable_steps:
            return

        self._duration_totals[self._current_label] += self._current_dwell
        self._duration_counts[self._current_label] += 1

    def observe(self, label: str | None, learn: bool = True) -> dict:
        """Update the transition memory with the current settled label.

        Returns a small status dictionary containing the updated prediction,
        prediction status, and whether a label transition event was detected.
        """
        canonical = None if label is None else str(label)

        previous_prediction = self._predicted_label
        prediction_status = None
        if previous_prediction is not None and canonical is not None:
            prediction_status = (
                "confident" if canonical == previous_prediction else "confused"
            )

        event_detected = False

        if canonical is None:
            self._predicted_label = None
            self._prediction_status = prediction_status
            self._last_event_detected = False
            return {
                "current_label": self._current_label,
                "current_dwell": self._current_dwell,
                "predicted_label": self._predicted_label,
                "prediction_status": self._prediction_status,
                "event_detected": self._last_event_detected,
            }

        if self._current_label is None:
            self._current_label = canonical
            self._current_dwell = 1
        elif canonical == self._current_label:
            self._current_dwell += 1
        else:
            event_detected = True
            if learn and self._current_dwell >= self.min_stable_steps:
                self._transition_counts[self._current_label][canonical] += 1
                self._duration_totals[self._current_label] += self._current_dwell
                self._duration_counts[self._current_label] += 1
            self._current_label = canonical
            self._current_dwell = 1

        self._predicted_label = self._predict_label(
            self._current_label, self._current_dwell
        )
        self._prediction_status = prediction_status
        self._last_event_detected = event_detected

        return {
            "current_label": self._current_label,
            "current_dwell": self._current_dwell,
            "predicted_label": self._predicted_label,
            "prediction_status": self._prediction_status,
            "event_detected": self._last_event_detected,
        }

    def get_expected_duration(self, label: str | None) -> float:
        """Return the learned mean dwell duration for a label."""
        if label is None:
            return self.default_duration
        key = str(label)
        count = self._duration_counts.get(key, 0)
        if count <= 0:
            return self.default_duration
        return self._duration_totals[key] / count

    def get_most_likely_next(self, label: str | None) -> str | None:
        """Return the most frequently observed next label."""
        if label is None:
            return None
        key = str(label)
        counts = self._transition_counts.get(key)
        if not counts:
            return None
        return max(counts.items(), key=lambda item: (item[1], item[0]))[0]

    def get_transition_distribution(self, label: str | None) -> dict[str, int]:
        """Return the outgoing transition counts for a label."""
        if label is None:
            return {}
        return dict(self._transition_counts.get(str(label), {}))

    def get_prediction(self) -> str | None:
        """Return the currently predicted label."""
        return self._predicted_label

    def get_prediction_status(self) -> str | None:
        """Return the last prediction status."""
        return self._prediction_status

    def get_event_signal(self) -> bool:
        """Return whether the last update detected a transition event."""
        return self._last_event_detected

    def get_current_dwell(self) -> int:
        """Return the current dwell count for the active label."""
        return self._current_dwell

    def state_dict(self) -> dict:
        """Serialize learned transitions and durations."""
        return {
            "min_stable_steps": self.min_stable_steps,
            "default_duration": self.default_duration,
            "transition_counts": {
                label: dict(counter)
                for label, counter in self._transition_counts.items()
            },
            "duration_totals": dict(self._duration_totals),
            "duration_counts": dict(self._duration_counts),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore learned transitions and durations."""
        self.min_stable_steps = int(state.get("min_stable_steps", 2))
        self.default_duration = float(state.get("default_duration", 2.0))

        self._transition_counts = defaultdict(Counter)
        for label, counter_dict in state.get("transition_counts", {}).items():
            self._transition_counts[str(label)] = Counter(counter_dict)

        self._duration_totals = defaultdict(float)
        for label, total in state.get("duration_totals", {}).items():
            self._duration_totals[str(label)] = float(total)

        self._duration_counts = defaultdict(int)
        for label, count in state.get("duration_counts", {}).items():
            self._duration_counts[str(label)] = int(count)

        self.reset_episode()

    def _predict_label(self, label: str | None, dwell: int) -> str | None:
        if label is None:
            return None

        next_label = self.get_most_likely_next(label)
        if next_label is None:
            return label

        expected_duration = self.get_expected_duration(label)
        threshold = max(1, int(round(expected_duration)))
        if dwell >= threshold:
            return next_label
        return label