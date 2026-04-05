# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Forward-only self-supervised semi-Markov memory over Torch embeddings.

This module approximates an online HSMM-style filter without backpropagation.
It discovers anonymous latent states directly from settled embedding patterns,
learns per-state dwell statistics, and accumulates transition counts between
discovered states.

The design is intentionally forward-only:

- state discovery uses cosine similarity against learned prototypes
- prototype learning uses exponential moving averages
- transition learning uses simple counts and running duration estimates
- prediction is based on the current state's dwell time and outgoing counts

This makes the temporal semantics emerge from predictive sensorimotor structure
rather than hand-authored labels such as ``walk_phase_0``.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import torch


class SelfSupervisedSemiMarkovMemory:
    """Discover latent states and learn dwell-aware transitions online.

    Parameters
    ----------
    match_threshold : float
        Minimum cosine similarity required before a matched state's prototype
        is updated.
    new_state_threshold : float
        If the best emission similarity falls below this threshold during
        training, a new latent state may be created.
    inference_threshold : float
        Minimum similarity required to trust a matched state during evaluation.
    prototype_ema_alpha : float
        Base EMA update rate for latent-state prototypes.
    min_segment_steps : int
        Minimum dwell required before a newly different emission is allowed to
        split into a new segment without a strong surprise signal.
    default_duration : float
        Fallback expected duration for states with insufficient dwell history.
    persistence_weight : float
        Bonus applied to the current state while it remains within its expected
        dwell duration.
    transition_weight : float
        Bonus applied according to the learned transition distribution from the
        current state.
    surprise_event_threshold : float
        Surprise level above which segment changes are allowed even before the
        minimum dwell requirement is met.
    switch_margin : float
        Minimum emission improvement required to switch away from the current
        state without an explicit surprise-driven boundary.
    label_prefix : str
        Prefix used for discovered latent-state labels.
    device : str
        PyTorch device used for prototype tensors.
    """

    def __init__(
        self,
        match_threshold: float = 0.75,
        new_state_threshold: float = 0.55,
        inference_threshold: float = 0.45,
        prototype_ema_alpha: float = 0.05,
        min_segment_steps: int = 2,
        default_duration: float = 2.0,
        persistence_weight: float = 0.2,
        transition_weight: float = 0.15,
        surprise_event_threshold: float = 0.6,
        switch_margin: float = 0.05,
        label_prefix: str = "latent",
        device: str = "cpu",
    ):
        self.match_threshold = float(match_threshold)
        self.new_state_threshold = float(new_state_threshold)
        self.inference_threshold = float(inference_threshold)
        self.prototype_ema_alpha = float(prototype_ema_alpha)
        self.min_segment_steps = max(1, int(min_segment_steps))
        self.default_duration = float(default_duration)
        self.persistence_weight = float(persistence_weight)
        self.transition_weight = float(transition_weight)
        self.surprise_event_threshold = float(surprise_event_threshold)
        self.switch_margin = float(switch_margin)
        self.label_prefix = str(label_prefix)
        self.device = torch.device(device)

        self._prototypes: dict[str, torch.Tensor] = {}
        self._prototype_counts: dict[str, int] = {}
        self._transition_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self._duration_totals: dict[str, float] = defaultdict(float)
        self._duration_counts: dict[str, int] = defaultdict(int)
        self._next_state_id = 0

        self.reset_episode()

    def reset_episode(self) -> None:
        """Reset episode-local tracking while keeping learned memory."""
        self._current_label: str | None = None
        self._current_dwell = 0
        self._predicted_label: str | None = None
        self._prediction_status: str | None = None
        self._last_event_detected = False
        self._last_match_score = 0.0
        self._last_surprise = 1.0
        self._surprise_history: list[float] = []

    def finalize_episode(self, learn: bool = True) -> None:
        """Commit the final state's dwell duration at episode end."""
        if not learn:
            return
        if self._current_label is None:
            return
        if self._current_dwell < self.min_segment_steps:
            return

        self._duration_totals[self._current_label] += self._current_dwell
        self._duration_counts[self._current_label] += 1

    def observe(
        self,
        embedding,
        learn: bool = True,
        surprise: float | None = None,
    ) -> dict:
        """Update the online semi-Markov state from the current embedding."""
        previous_label = self._current_label
        previous_prediction = self._predicted_label

        if embedding is None:
            self._prediction_status = None
            self._last_event_detected = False
            return self._make_info()

        with torch.no_grad():
            x = self._prepare_embedding(embedding)
            emission_scores = self._score_existing_states(x)
            best_emission_label, best_emission_score = self._best_scoring_label(
                emission_scores
            )
            surprise_value = self._blend_surprise(best_emission_score, surprise)

            if self._should_create_state(
                best_emission_score=best_emission_score,
                learn=learn,
                surprise_value=surprise_value,
            ):
                selected_label = self._create_state(x)
                selected_emission = 1.0
            else:
                candidate_scores = dict(emission_scores)
                self._apply_state_priors(candidate_scores, previous_label)
                selected_label, _ = self._best_scoring_label(candidate_scores)
                selected_label = self._stabilize_state_switch(
                    selected_label=selected_label,
                    emission_scores=emission_scores,
                    previous_label=previous_label,
                    previous_prediction=previous_prediction,
                    surprise_value=surprise_value,
                )
                if (
                    not learn
                    and previous_label is None
                    and selected_label is not None
                    and emission_scores.get(selected_label, 0.0)
                    < self.inference_threshold
                ):
                    selected_label = None
                selected_emission = (
                    emission_scores.get(selected_label, 0.0)
                    if selected_label is not None
                    else 0.0
                )

            event_detected = self._commit_state(selected_label, learn=learn)

            if (
                learn
                and selected_label is not None
                and selected_label in self._prototypes
                and selected_emission >= self.match_threshold
            ):
                self._update_prototype(selected_label, x)

            prediction_status = None
            if previous_prediction is not None and selected_label is not None:
                prediction_status = (
                    "confident"
                    if selected_label == previous_prediction
                    else "confused"
                )

            self._predicted_label = self._predict_label(
                self._current_label,
                self._current_dwell,
            )
            self._prediction_status = prediction_status
            self._last_event_detected = event_detected
            self._last_match_score = float(selected_emission)
            self._last_surprise = float(surprise_value)
            self._surprise_history.append(self._last_surprise)

        return self._make_info()

    def get_known_states(self) -> list[str]:
        """Return the discovered latent-state labels."""
        return list(self._prototypes.keys())

    def get_current_label(self) -> str | None:
        """Return the active latent-state label."""
        return self._current_label

    def get_current_dwell(self) -> int:
        """Return the dwell count for the active latent state."""
        return self._current_dwell

    def get_prediction(self) -> str | None:
        """Return the currently predicted next latent state."""
        return self._predicted_label

    def get_prediction_status(self) -> str | None:
        """Return whether the last prediction matched the new observation."""
        return self._prediction_status

    def get_event_signal(self) -> bool:
        """Return whether the last update crossed a latent-state boundary."""
        return self._last_event_detected

    def get_surprise_history(self) -> list[float]:
        """Return the episode-local surprise history."""
        return list(self._surprise_history)

    def get_mean_surprise(self, last_n: int | None = None) -> float:
        """Return mean surprise over the full or trailing history."""
        surprises = self._surprise_history
        if not surprises:
            return 1.0
        if last_n is not None:
            surprises = surprises[-last_n:]
        return float(sum(surprises) / len(surprises))

    def get_expected_duration(self, label: str | None) -> float:
        """Return the learned mean dwell duration for a latent state."""
        if label is None:
            return self.default_duration
        count = self._duration_counts.get(str(label), 0)
        if count <= 0:
            return self.default_duration
        return self._duration_totals[str(label)] / count

    def get_most_likely_next(self, label: str | None) -> str | None:
        """Return the most frequently observed successor state."""
        if label is None:
            return None
        counts = self._transition_counts.get(str(label))
        if not counts:
            return None
        return max(counts.items(), key=lambda item: (item[1], item[0]))[0]

    def get_transition_distribution(self, label: str | None) -> dict[str, int]:
        """Return outgoing transition counts for a latent state."""
        if label is None:
            return {}
        return dict(self._transition_counts.get(str(label), {}))

    def get_temporal_context(self) -> dict | None:
        """Return compact temporal context for downstream consumers."""
        if self._current_label is None and not self._prototypes:
            return None

        return {
            "current_label": self._current_label,
            "predicted_label": self._predicted_label,
            "current_dwell": self._current_dwell,
            "event_detected": self._last_event_detected,
            "match_score": self._last_match_score,
            "mean_surprise": self.get_mean_surprise(last_n=5),
            "known_states": len(self._prototypes),
        }

    def state_dict(self) -> dict:
        """Serialize learned prototypes, transitions, and durations."""
        return {
            "match_threshold": self.match_threshold,
            "new_state_threshold": self.new_state_threshold,
            "inference_threshold": self.inference_threshold,
            "prototype_ema_alpha": self.prototype_ema_alpha,
            "min_segment_steps": self.min_segment_steps,
            "default_duration": self.default_duration,
            "persistence_weight": self.persistence_weight,
            "transition_weight": self.transition_weight,
            "surprise_event_threshold": self.surprise_event_threshold,
            "switch_margin": self.switch_margin,
            "label_prefix": self.label_prefix,
            "next_state_id": self._next_state_id,
            "prototypes": {k: v.cpu() for k, v in self._prototypes.items()},
            "prototype_counts": dict(self._prototype_counts),
            "transition_counts": {
                label: dict(counter)
                for label, counter in self._transition_counts.items()
            },
            "duration_totals": dict(self._duration_totals),
            "duration_counts": dict(self._duration_counts),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore learned prototypes, transitions, and durations."""
        self.match_threshold = float(state.get("match_threshold", 0.75))
        self.new_state_threshold = float(state.get("new_state_threshold", 0.55))
        self.inference_threshold = float(state.get("inference_threshold", 0.45))
        self.prototype_ema_alpha = float(state.get("prototype_ema_alpha", 0.05))
        self.min_segment_steps = max(1, int(state.get("min_segment_steps", 2)))
        self.default_duration = float(state.get("default_duration", 2.0))
        self.persistence_weight = float(state.get("persistence_weight", 0.2))
        self.transition_weight = float(state.get("transition_weight", 0.15))
        self.surprise_event_threshold = float(
            state.get("surprise_event_threshold", 0.6)
        )
        self.switch_margin = float(state.get("switch_margin", 0.05))
        self.label_prefix = str(state.get("label_prefix", "latent"))
        self._next_state_id = int(state.get("next_state_id", 0))

        self._prototypes = {
            str(k): v.to(self.device).float()
            for k, v in state.get("prototypes", {}).items()
        }
        self._prototype_counts = {
            str(k): int(v)
            for k, v in state.get("prototype_counts", {}).items()
        }

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

    def _prepare_embedding(self, embedding) -> torch.Tensor:
        tensor = torch.as_tensor(embedding, dtype=torch.float32, device=self.device)
        return tensor.detach().flatten()

    def _score_existing_states(self, embedding: torch.Tensor) -> dict[str, float]:
        if not self._prototypes:
            return {}

        if float(embedding.norm()) <= 1e-8:
            return {label: 0.0 for label in self._prototypes}

        labels = list(self._prototypes.keys())
        protos = torch.stack([self._prototypes[label] for label in labels])
        proto_norms = protos / (protos.norm(dim=1, keepdim=True) + 1e-8)
        emb_norm = embedding / (embedding.norm() + 1e-8)
        sims = torch.mv(proto_norms, emb_norm).clamp(min=0.0, max=1.0)
        return {
            label: float(score)
            for label, score in zip(labels, sims.tolist())
        }

    @staticmethod
    def _best_scoring_label(scores: dict[str, float]) -> tuple[str | None, float]:
        if not scores:
            return None, 0.0
        label, score = max(scores.items(), key=lambda item: (item[1], item[0]))
        return label, float(score)

    def _blend_surprise(
        self,
        best_emission_score: float,
        surprise: float | None,
    ) -> float:
        base_surprise = 1.0 - float(best_emission_score)
        if surprise is None:
            return base_surprise
        return max(base_surprise, float(surprise))

    def _should_create_state(
        self,
        best_emission_score: float,
        learn: bool,
        surprise_value: float,
    ) -> bool:
        if not learn:
            return False
        if not self._prototypes:
            return True
        if best_emission_score >= self.new_state_threshold:
            return False
        if self._current_label is None:
            return True
        if self._current_dwell >= self.min_segment_steps:
            return True
        return surprise_value >= self.surprise_event_threshold

    def _apply_state_priors(
        self,
        candidate_scores: dict[str, float],
        previous_label: str | None,
    ) -> None:
        if previous_label is None:
            return

        expected_duration = max(self.get_expected_duration(previous_label), 1.0)
        remaining_ratio = max(expected_duration - self._current_dwell, 0.0) / expected_duration
        candidate_scores[previous_label] = candidate_scores.get(previous_label, 0.0) + (
            self.persistence_weight * remaining_ratio
        )

        transition_counts = self._transition_counts.get(previous_label)
        if not transition_counts:
            return

        total = sum(transition_counts.values())
        if total <= 0:
            return

        for label, count in transition_counts.items():
            probability = float(count) / total
            candidate_scores[label] = candidate_scores.get(label, 0.0) + (
                self.transition_weight * probability
            )

    def _stabilize_state_switch(
        self,
        selected_label: str | None,
        emission_scores: dict[str, float],
        previous_label: str | None,
        previous_prediction: str | None,
        surprise_value: float,
    ) -> str | None:
        if previous_label is None:
            return selected_label
        if selected_label is None or selected_label == previous_label:
            return selected_label

        if self._current_dwell >= self.min_segment_steps:
            return selected_label

        current_score = emission_scores.get(previous_label, 0.0)
        selected_score = emission_scores.get(selected_label, 0.0)
        expected_duration = max(1, int(round(self.get_expected_duration(previous_label))))
        boundary_pressure = (
            surprise_value >= self.surprise_event_threshold
            or selected_label == previous_prediction
            or (selected_score - current_score) >= self.switch_margin
            or self._current_dwell >= expected_duration
        )
        if boundary_pressure:
            return selected_label
        return previous_label

    def _create_state(self, embedding: torch.Tensor) -> str:
        label = f"{self.label_prefix}_{self._next_state_id:04d}"
        self._next_state_id += 1
        self._prototypes[label] = embedding.clone()
        self._prototype_counts[label] = 1
        return label

    def _update_prototype(self, label: str, embedding: torch.Tensor) -> None:
        count = self._prototype_counts.get(label, 0) + 1
        self._prototype_counts[label] = count
        alpha = max(self.prototype_ema_alpha, 1.0 / count)
        self._prototypes[label] = (
            (1.0 - alpha) * self._prototypes[label]
            + alpha * embedding
        )

    def _commit_state(self, selected_label: str | None, learn: bool) -> bool:
        if selected_label is None:
            self._last_event_detected = False
            return False

        if self._current_label is None:
            self._current_label = selected_label
            self._current_dwell = 1
            self._last_event_detected = False
            return False

        if selected_label == self._current_label:
            self._current_dwell += 1
            self._last_event_detected = False
            return False

        if learn and self._current_dwell >= self.min_segment_steps:
            self._transition_counts[self._current_label][selected_label] += 1
            self._duration_totals[self._current_label] += self._current_dwell
            self._duration_counts[self._current_label] += 1

        self._current_label = selected_label
        self._current_dwell = 1
        self._last_event_detected = True
        return True

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

    def _make_info(self) -> dict:
        return {
            "current_label": self._current_label,
            "current_dwell": self._current_dwell,
            "predicted_label": self._predicted_label,
            "prediction_status": self._prediction_status,
            "event_detected": self._last_event_detected,
            "match_score": self._last_match_score,
            "surprise": self._last_surprise,
            "known_states": len(self._prototypes),
        }