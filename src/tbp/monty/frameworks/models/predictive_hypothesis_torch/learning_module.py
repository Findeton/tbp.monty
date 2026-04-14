from __future__ import annotations

from collections import defaultdict
import copy
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.models.abstract_monty_classes import LearningModule
from tbp.monty.frameworks.models.buffer import FeatureAtLocationBuffer
from tbp.monty.frameworks.models.predictive_hypothesis_torch.core import (
    PredictiveHypothesisCore,
)
from tbp.monty.frameworks.models.predictive_hypothesis_torch.tensor_objects import (
    PredictiveContextSignal,
    PredictiveVoteMessage,
    RankedHypothesisVote,
)
from tbp.monty.frameworks.models.states import GoalState, State


def _normalize_graph_id(graph_id: Any) -> str | None:
    if graph_id is None:
        return None

    normalized = str(graph_id)
    if normalized in {"", "unknown", "no_observations_yet", "None"}:
        return None
    return normalized


class PredictiveHypothesisTorchLM(LearningModule):
    @staticmethod
    def _empty_target_registration_stats() -> dict[str, Any]:
        return {
            "total_steps": 0,
            "agreement_steps": 0,
            "mismatch_steps": 0,
            "learning_only_steps": 0,
            "output_only_steps": 0,
            "per_target": {},
        }

    def __init__(
        self,
        core_kwargs: dict | None = None,
        learning_module_id: str = "PredictiveHypothesisTorchLM_0",
        vote_top_k: int | None = 3,
        output_evidence_threshold: float = 0.55,
        evidence_match_threshold: float = 0.75,
        evidence_separation_ratio: float = 1.15,
        goal_state_min_steps: int = 0,
        goal_state_min_separation_ratio: float | None = None,
        named_target_min_registration_support: float = 0.55,
        named_target_shared_override_ratio: float = 10.0,
        named_target_shared_min_learning_posterior: float = 0.8,
        named_target_shared_min_mean_support: float = 0.75,
    ):
        self.learning_module_id = learning_module_id
        self._core = PredictiveHypothesisCore(**(core_kwargs or {}))
        self._vote_top_k = None if vote_top_k is None else max(int(vote_top_k), 1)
        self._output_evidence_threshold = float(max(output_evidence_threshold, 1e-6))
        self._evidence_match_threshold = float(max(evidence_match_threshold, 1e-6))
        self._evidence_separation_ratio = float(max(evidence_separation_ratio, 1.0))
        self._goal_state_min_steps = max(int(goal_state_min_steps), 0)
        self._goal_state_min_separation_ratio = (
            None
            if goal_state_min_separation_ratio is None
            else float(max(goal_state_min_separation_ratio, 0.0))
        )
        self._named_target_min_registration_support = float(
            max(named_target_min_registration_support, 0.0)
        )
        self._named_target_shared_override_ratio = float(
            max(named_target_shared_override_ratio, 1.0)
        )
        self._named_target_shared_min_learning_posterior = float(
            min(
                max(named_target_shared_min_learning_posterior, 0.0),
                1.0,
            )
        )
        self._named_target_shared_min_mean_support = float(
            min(
                max(named_target_shared_min_mean_support, 0.0),
                1.0,
            )
        )

        self.buffer = FeatureAtLocationBuffer()
        self.terminal_state = None
        self.primary_target = None
        self.primary_target_rotation_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.stepwise_target_object = None
        self.stepwise_targets_list = []
        self.detected_object = None
        self.detected_pose = [None for _ in range(7)]
        self.detected_rotation_r = None
        self.symmetry_evidence = 0
        self.has_detailed_logger = False
        self.latent_id_to_target = defaultdict(set)
        self.target_to_latent_id = defaultdict(set)
        self.mode = None
        self.possible_matches = {}
        self.possible_poses = {}
        self.possible_paths = {}
        self.pose_similarity_threshold = 0.35
        self.evidence = {}

        self._mode: ExperimentMode | None = None
        self._stepped = False
        self._step_count = 0
        self._last_result: dict[str, Any] = {}
        self._last_observed_state: State | None = None
        self._last_input_state: State | None = None
        self._last_output_state: State | None = None
        self._last_skip_reason: str | None = None
        self._primary_target_object: str | None = None
        self._primary_target_state: int | None = None
        self.stepwise_target_state: int | None = None
        self._last_vote_message: PredictiveVoteMessage | None = None
        self._last_received_vote_messages: list[PredictiveVoteMessage] = []
        self._last_context_message: PredictiveContextSignal | None = None
        self._last_received_context_message: PredictiveContextSignal | None = None
        self._episode_index = 0
        self._last_target_registration_debug: dict[str, Any] | None = None
        self._last_target_registration_commit: dict[str, Any] | None = None
        self._target_registration_history: list[dict[str, Any]] = []
        self._target_registration_stats = self._empty_target_registration_stats()

    @property
    def latent_id_to_target(self):
        return self._latent_id_to_target

    @latent_id_to_target.setter
    def latent_id_to_target(self, value) -> None:
        mapping = defaultdict(set)
        for key, values in (value or {}).items():
            mapping[str(key)] = {str(item) for item in values}
        self._latent_id_to_target = mapping

    @property
    def graph_id_to_target(self):
        return self._latent_id_to_target

    @graph_id_to_target.setter
    def graph_id_to_target(self, value) -> None:
        self.latent_id_to_target = value

    @property
    def target_to_latent_id(self):
        return self._target_to_latent_id

    @target_to_latent_id.setter
    def target_to_latent_id(self, value) -> None:
        mapping = defaultdict(set)
        for key, values in (value or {}).items():
            mapping[str(key)] = {str(item) for item in values}
        self._target_to_latent_id = mapping

    @property
    def target_to_graph_id(self):
        return self._target_to_latent_id

    @target_to_graph_id.setter
    def target_to_graph_id(self, value) -> None:
        self.target_to_latent_id = value

    def reset(self):
        self._stepped = False
        self._step_count = 0
        self._last_result = {}
        self._last_observed_state = None
        self._last_input_state = None
        self._last_output_state = None
        self._last_skip_reason = None
        self._last_vote_message = None
        self._last_received_vote_messages = []
        self._last_context_message = None
        self._last_received_context_message = None
        self.buffer = FeatureAtLocationBuffer()
        self.terminal_state = None
        self.detected_object = None
        self.detected_pose = [None for _ in range(7)]
        self.detected_rotation_r = None
        self.possible_matches = {}
        self.evidence = {}
        self.stepwise_target_state = None
        self._last_target_registration_debug = None
        self._last_target_registration_commit = None
        self._core.reset_episode()

    def pre_episode(self, primary_target=None, **kwargs) -> None:
        self.reset()
        self._episode_index += 1
        self.stepwise_target_object = None
        self.stepwise_targets_list = []
        self.symmetry_evidence = 0

        if primary_target is not None:
            if isinstance(primary_target, dict):
                self.primary_target = primary_target.get("object")
                self.primary_target_rotation_quat = np.asarray(
                    primary_target.get("quat_rotation", [1.0, 0.0, 0.0, 0.0]),
                    dtype=np.float64,
                )
                state_id = primary_target.get("state")
            else:
                self.primary_target = str(primary_target)
                self.primary_target_rotation_quat = np.array([1.0, 0.0, 0.0, 0.0])
                state_id = None
        else:
            self.primary_target = None
            self.primary_target_rotation_quat = np.array([1.0, 0.0, 0.0, 0.0])
            state_id = None

        self._primary_target_object = self.primary_target
        self._primary_target_state = state_id
        self._core.pre_episode(
            self._mode,
            object_name=("episode_anchor" if self._primary_target_object is not None else None),
        )

    def post_episode(self):
        self._finalize_episode_target_registrations()
        self._core.post_episode()

    def set_experiment_mode(self, mode: ExperimentMode) -> None:
        self._mode = mode
        self.mode = mode

    def matching_step(self, ctx, observations=None):
        self._step_impl(observations, learn=False)

    def exploratory_step(self, ctx, observations=None):
        self._step_impl(observations, learn=True)

    def _step_impl(self, observations, learn: bool) -> None:
        state = self._extract_usable_state(observations)
        if state is None:
            self._stepped = False
            self._last_result = {}
            self.possible_matches = {}
            self.evidence = {}
            self._last_skip_reason = "no_usable_state"
            return

        self._last_skip_reason = None
        self._last_observed_state = state
        if getattr(state, "sender_type", None) == "SM":
            self._last_input_state = state

        result = self._core.step(state, learn=learn)
        if learn:
            self._register_reporting_target(result)
        self._last_result = result
        self._stepped = True
        self._step_count += 1

        self.evidence = dict(result.get("evidence", {}))
        self.possible_matches = {
            object_id: score for object_id, score in self.evidence.items()
        }
        self._last_output_state = None

    def _extract_usable_state(self, observations) -> State | None:
        states = []
        if observations is None:
            return None
        if isinstance(observations, list):
            for item in observations:
                if isinstance(item, State):
                    states.append(item)
                elif isinstance(item, list):
                    states.extend(s for s in item if isinstance(s, State))
        elif isinstance(observations, State):
            states = [observations]

        usable = [state for state in states if bool(getattr(state, "use_state", False))]
        if not usable:
            return None

        sensory = [state for state in usable if getattr(state, "sender_type", None) == "SM"]
        if sensory:
            return sensory[0]

        return self._build_fused_lm_state(usable)

    def _build_fused_lm_state(self, states: list[State]) -> State:
        locations = [np.asarray(state.location, dtype=np.float64) for state in states]
        location = np.mean(np.stack(locations), axis=0)

        pose_vectors = np.eye(3, dtype=np.float64)
        for state in states:
            morph = getattr(state, "morphological_features", None) or {}
            if "pose_vectors" in morph:
                pose_vectors = np.asarray(morph["pose_vectors"], dtype=np.float64)
                break

        child_sender_ids = [str(getattr(state, "sender_id", "unknown")) for state in states]
        child_latent_ids = []
        child_confidences = []
        child_active_cells = []
        child_appearance_signatures = []
        child_predicted_appearance_signatures = []
        child_predicted_change_signatures = []
        child_behavior_labels = []
        child_behavior_signatures = []
        child_appearance_residuals = []
        child_change_residuals = []
        child_appearance_prediction_errors = []
        child_change_prediction_errors = []
        detail_shards = []

        for index, state in enumerate(states):
            non_morph = getattr(state, "non_morphological_features", None) or {}
            child_latent_id = _normalize_graph_id(
                non_morph.get("latent_id", non_morph.get("graph_id"))
            )
            child_latent_ids.append(child_latent_id)
            child_confidences.append(float(getattr(state, "confidence", 0.0)))
            active_cells = non_morph.get("active_cells")
            if isinstance(active_cells, np.ndarray):
                child_active_cells.append(active_cells.astype(np.float32))
            appearance_signature = non_morph.get("appearance_signature")
            if appearance_signature is not None:
                signature_array = np.asarray(
                    appearance_signature,
                    dtype=np.float32,
                ).reshape(-1)
                if signature_array.size > 0:
                    child_appearance_signatures.append(signature_array)
            predicted_appearance_signature = non_morph.get(
                "predicted_appearance_signature"
            )
            if predicted_appearance_signature is not None:
                signature_array = np.asarray(
                    predicted_appearance_signature,
                    dtype=np.float32,
                ).reshape(-1)
                if signature_array.size > 0:
                    child_predicted_appearance_signatures.append(signature_array)
            predicted_change_signature = non_morph.get("predicted_change_signature")
            if predicted_change_signature is not None:
                signature_array = np.asarray(
                    predicted_change_signature,
                    dtype=np.float32,
                ).reshape(-1)
                if signature_array.size > 0:
                    child_predicted_change_signatures.append(signature_array)
            behavior_label = non_morph.get("behavior_label")
            if behavior_label is not None:
                child_behavior_labels.append(str(behavior_label))
            behavior_signature = non_morph.get("behavior_signature")
            if behavior_signature is not None:
                signature_array = np.asarray(
                    behavior_signature,
                    dtype=np.float32,
                ).reshape(-1)
                if signature_array.size > 0:
                    child_behavior_signatures.append(signature_array)
            appearance_residual = non_morph.get("appearance_residual")
            if appearance_residual is not None:
                residual_array = np.asarray(
                    appearance_residual,
                    dtype=np.float32,
                ).reshape(-1)
                if residual_array.size > 0:
                    child_appearance_residuals.append(residual_array)
            change_residual = non_morph.get("change_residual")
            if change_residual is not None:
                residual_array = np.asarray(
                    change_residual,
                    dtype=np.float32,
                ).reshape(-1)
                if residual_array.size > 0:
                    child_change_residuals.append(residual_array)
            if non_morph.get("appearance_prediction_error") is not None:
                child_appearance_prediction_errors.append(
                    float(non_morph.get("appearance_prediction_error", 0.0))
                )
            if non_morph.get("change_prediction_error") is not None:
                child_change_prediction_errors.append(
                    float(non_morph.get("change_prediction_error", 0.0))
                )
            detail_shards.append(
                {
                    "child_index": float(index),
                    "child_sender_id": str(getattr(state, "sender_id", "unknown")),
                    "child_graph_id": child_latent_id,
                    "child_latent_id": child_latent_id,
                    "child_confidence": float(getattr(state, "confidence", 0.0)),
                    "child_location": np.asarray(state.location, dtype=np.float32),
                    "child_behavior_label": behavior_label,
                    "child_residual": float(non_morph.get("residual", 0.0)),
                }
            )

        fused_non_morph = {
            "child_sender_ids": child_sender_ids,
            "child_graph_ids": list(child_latent_ids),
            "child_latent_ids": list(child_latent_ids),
            "child_confidences": child_confidences,
            "detail_packet": {
                "packet_type": "lm_fusion_packet",
                "shard_count": len(detail_shards),
                "shards": detail_shards,
            },
        }
        if child_active_cells:
            fused_non_morph["active_cells"] = np.mean(np.stack(child_active_cells), axis=0)
        if child_appearance_signatures:
            signature_dim = max(int(signature.size) for signature in child_appearance_signatures)
            padded_signatures = [
                np.pad(signature, (0, signature_dim - int(signature.size)), mode="constant")
                for signature in child_appearance_signatures
            ]
            fused_non_morph["appearance_signature"] = np.mean(
                np.stack(padded_signatures),
                axis=0,
            ).astype(np.float32)
        if child_predicted_appearance_signatures:
            signature_dim = max(
                int(signature.size) for signature in child_predicted_appearance_signatures
            )
            padded_signatures = [
                np.pad(signature, (0, signature_dim - int(signature.size)), mode="constant")
                for signature in child_predicted_appearance_signatures
            ]
            fused_non_morph["predicted_appearance_signature"] = np.mean(
                np.stack(padded_signatures),
                axis=0,
            ).astype(np.float32)
        if child_predicted_change_signatures:
            signature_dim = max(
                int(signature.size) for signature in child_predicted_change_signatures
            )
            padded_signatures = [
                np.pad(signature, (0, signature_dim - int(signature.size)), mode="constant")
                for signature in child_predicted_change_signatures
            ]
            fused_non_morph["predicted_change_signature"] = np.mean(
                np.stack(padded_signatures),
                axis=0,
            ).astype(np.float32)
        if child_behavior_labels:
            fused_non_morph["child_behavior_labels"] = list(child_behavior_labels)
            label_counts = {}
            for label in child_behavior_labels:
                label_counts[label] = label_counts.get(label, 0) + 1
            fused_non_morph["behavior_label"] = max(
                label_counts.items(),
                key=lambda item: (int(item[1]), str(item[0])),
            )[0]
        if child_behavior_signatures:
            signature_dim = max(int(signature.size) for signature in child_behavior_signatures)
            padded_signatures = [
                np.pad(signature, (0, signature_dim - int(signature.size)), mode="constant")
                for signature in child_behavior_signatures
            ]
            fused_non_morph["behavior_signature"] = np.mean(
                np.stack(padded_signatures),
                axis=0,
            ).astype(np.float32)
        if child_appearance_residuals:
            residual_dim = max(int(residual.size) for residual in child_appearance_residuals)
            padded_residuals = [
                np.pad(residual, (0, residual_dim - int(residual.size)), mode="constant")
                for residual in child_appearance_residuals
            ]
            fused_non_morph["appearance_residual"] = np.mean(
                np.stack(padded_residuals),
                axis=0,
            ).astype(np.float32)
        if child_change_residuals:
            residual_dim = max(int(residual.size) for residual in child_change_residuals)
            padded_residuals = [
                np.pad(residual, (0, residual_dim - int(residual.size)), mode="constant")
                for residual in child_change_residuals
            ]
            fused_non_morph["change_residual"] = np.mean(
                np.stack(padded_residuals),
                axis=0,
            ).astype(np.float32)
        if child_appearance_prediction_errors:
            fused_non_morph["appearance_prediction_error"] = float(
                np.mean(np.asarray(child_appearance_prediction_errors, dtype=np.float32))
            )
        if child_change_prediction_errors:
            fused_non_morph["change_prediction_error"] = float(
                np.mean(np.asarray(child_change_prediction_errors, dtype=np.float32))
            )

        return State(
            location=location,
            morphological_features={
                "pose_vectors": pose_vectors,
                "pose_fully_defined": True,
                "on_object": True,
            },
            non_morphological_features=fused_non_morph,
            confidence=float(max(child_confidences) if child_confidences else 0.0),
            use_state=True,
            sender_id=f"fused:{self.learning_module_id}",
            sender_type="LM",
        )

    def _resolve_reporting_target_object(self) -> str | None:
        object_name = _normalize_graph_id(self.stepwise_target_object)
        if object_name == "no_label":
            object_name = None
        if object_name is None:
            object_name = _normalize_graph_id(self._primary_target_object)
        return None if object_name is None else str(object_name)

    @staticmethod
    def _increment_debug_count(counts: dict[str, int], key: str | None) -> None:
        if key is None:
            return
        normalized_key = str(key)
        counts[normalized_key] = int(counts.get(normalized_key, 0)) + 1

    def _record_target_registration_debug(
        self,
        *,
        target_object: str,
        learning_support: dict[str, Any],
        result: dict[str, Any],
        registered_latent_ids: list[str],
    ) -> dict[str, Any]:
        ranked_hypotheses = list(result.get("ranked_hypotheses", []))
        learning_latent_id = _normalize_graph_id(
            learning_support.get("learning_latent_id")
            or learning_support.get("learning_object_id")
        )
        learning_chart_id = _normalize_graph_id(learning_support.get("learning_chart_id"))
        output_mlh = result.get("mlh") or {}
        output_latent_id = _normalize_graph_id(
            output_mlh.get("latent_id", output_mlh.get("graph_id"))
        )
        output_chart_id = _normalize_graph_id(output_mlh.get("chart_id"))
        learning_rank = None
        learning_probability = None
        learning_evidence = None
        output_probability = None
        output_evidence = None
        top_hypotheses = []
        for rank, hypothesis in enumerate(ranked_hypotheses[:4], start=1):
            latent_id = _normalize_graph_id(
                hypothesis.get("latent_id", hypothesis.get("object_id"))
            )
            chart_id = _normalize_graph_id(hypothesis.get("chart_id"))
            probability = float(hypothesis.get("probability", 0.0))
            evidence = float(
                hypothesis.get("evidence", hypothesis.get("scaled_evidence", 0.0))
            )
            top_hypotheses.append(
                {
                    "rank": int(rank),
                    "latent_id": latent_id,
                    "chart_id": chart_id,
                    "probability": probability,
                    "evidence": evidence,
                    "source": hypothesis.get("source"),
                    "age": int(hypothesis.get("age", 0)),
                }
            )
            if latent_id == learning_latent_id and learning_rank is None:
                learning_rank = int(rank)
                learning_probability = probability
                learning_evidence = evidence
            if latent_id == output_latent_id and output_probability is None:
                output_probability = probability
                output_evidence = evidence
        if output_evidence is None:
            output_evidence = float(output_mlh.get("evidence", 0.0))

        agreement = (
            learning_latent_id is not None
            and output_latent_id is not None
            and learning_latent_id == output_latent_id
        )
        event = {
            "episode_index": int(self._episode_index),
            "step_count": int(self._step_count),
            "target_object": str(target_object),
            "learning_latent_id": learning_latent_id,
            "learning_chart_id": learning_chart_id,
            "learning_selected_source": learning_support.get("selected_source"),
            "learning_selection_reason": learning_support.get(
                "selection_reason",
                learning_support.get("allocation_reason"),
            ),
            "learning_selected_posterior_probability": float(
                learning_support.get("selected_posterior_probability", 0.0)
            ),
            "learning_rank": learning_rank,
            "learning_probability": learning_probability,
            "learning_evidence": learning_evidence,
            "output_latent_id": output_latent_id,
            "output_chart_id": output_chart_id,
            "output_probability": output_probability,
            "output_evidence": output_evidence,
            "agreement": bool(agreement),
            "registered_latent_ids": list(registered_latent_ids),
            "top_hypotheses": top_hypotheses,
        }
        if (
            learning_evidence is not None
            and output_evidence is not None
            and learning_latent_id is not None
            and output_latent_id is not None
            and learning_latent_id != output_latent_id
        ):
            event["output_minus_learning_evidence"] = float(
                output_evidence - learning_evidence
            )

        self._last_target_registration_debug = event
        self._target_registration_history.append(copy.deepcopy(event))
        if len(self._target_registration_history) > 256:
            self._target_registration_history = self._target_registration_history[-256:]

        stats = self._target_registration_stats
        stats["total_steps"] = int(stats.get("total_steps", 0)) + 1
        if agreement:
            stats["agreement_steps"] = int(stats.get("agreement_steps", 0)) + 1
        else:
            stats["mismatch_steps"] = int(stats.get("mismatch_steps", 0)) + 1
        if learning_latent_id is not None and output_latent_id is None:
            stats["learning_only_steps"] = int(stats.get("learning_only_steps", 0)) + 1
        if output_latent_id is not None and learning_latent_id is None:
            stats["output_only_steps"] = int(stats.get("output_only_steps", 0)) + 1

        per_target = stats.setdefault("per_target", {})
        target_stats = per_target.setdefault(
            str(target_object),
            {
                "steps": 0,
                "agreement_steps": 0,
                "mismatch_steps": 0,
                "learning_only_steps": 0,
                "output_only_steps": 0,
                "learning_latent_counts": {},
                "output_latent_counts": {},
                "registered_latent_counts": {},
                "learning_output_pairs": {},
            },
        )
        target_stats["steps"] = int(target_stats.get("steps", 0)) + 1
        if agreement:
            target_stats["agreement_steps"] = int(target_stats.get("agreement_steps", 0)) + 1
        else:
            target_stats["mismatch_steps"] = int(target_stats.get("mismatch_steps", 0)) + 1
        if learning_latent_id is not None and output_latent_id is None:
            target_stats["learning_only_steps"] = int(
                target_stats.get("learning_only_steps", 0)
            ) + 1
        if output_latent_id is not None and learning_latent_id is None:
            target_stats["output_only_steps"] = int(
                target_stats.get("output_only_steps", 0)
            ) + 1
        self._increment_debug_count(
            target_stats.setdefault("learning_latent_counts", {}),
            learning_latent_id,
        )
        self._increment_debug_count(
            target_stats.setdefault("output_latent_counts", {}),
            output_latent_id,
        )
        for latent_id in registered_latent_ids:
            self._increment_debug_count(
                target_stats.setdefault("registered_latent_counts", {}),
                latent_id,
            )
        pair_key = f"{learning_latent_id or 'none'}->{output_latent_id or 'none'}"
        pair_counts = target_stats.setdefault("learning_output_pairs", {})
        pair_counts[pair_key] = int(pair_counts.get(pair_key, 0)) + 1
        return event

    def _is_child_reporting_lm(self) -> bool:
        observed_state = self._last_observed_state
        return getattr(observed_state, "sender_type", None) == "SM"

    def _episode_registration_events_for_current_episode(
        self,
    ) -> dict[str, list[dict[str, Any]]]:
        by_target: dict[str, list[dict[str, Any]]] = {}
        for event in self._target_registration_history:
            if int(event.get("episode_index", -1)) != int(self._episode_index):
                continue
            target_object = _normalize_graph_id(event.get("target_object"))
            if target_object is None:
                continue
            by_target.setdefault(target_object, []).append(copy.deepcopy(event))
        for events in by_target.values():
            events.sort(
                key=lambda event: (
                    int(event.get("step_count", 0)),
                    str(event.get("learning_latent_id") or ""),
                    str(event.get("output_latent_id") or ""),
                )
            )
        return by_target

    def _summarize_episode_registration_candidates(
        self,
        *,
        target_object: str,
        events: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        candidates: dict[str, dict[str, Any]] = {}
        last_event = events[-1] if events else {}
        final_learning_latent_id = _normalize_graph_id(last_event.get("learning_latent_id"))
        final_output_latent_id = _normalize_graph_id(last_event.get("output_latent_id"))

        for event in events:
            learning_latent_id = _normalize_graph_id(event.get("learning_latent_id"))
            output_latent_id = _normalize_graph_id(event.get("output_latent_id"))
            learning_support = float(
                max(
                    event.get("learning_selected_posterior_probability", 0.0),
                    event.get("learning_probability") or 0.0,
                )
            )
            output_support = float(max(event.get("output_probability") or 0.0, 0.0))
            agreement = bool(event.get("agreement", False))
            step_count = int(event.get("step_count", 0))

            for latent_id, role_support, is_learning_role in (
                (learning_latent_id, learning_support, True),
                (output_latent_id, output_support, False),
            ):
                if latent_id is None:
                    continue
                candidate = candidates.setdefault(
                    latent_id,
                    {
                        "latent_id": latent_id,
                        "target_object": str(target_object),
                        "support_score": 0.0,
                        "learning_support_sum": 0.0,
                        "output_support_sum": 0.0,
                        "learning_steps": 0,
                        "output_steps": 0,
                        "agreement_steps": 0,
                        "max_learning_posterior": 0.0,
                        "max_output_probability": 0.0,
                        "last_step_count": 0,
                        "final_learning": False,
                        "final_output": False,
                        "existing_targets": [],
                        "shared_with_other_targets": False,
                    },
                )
                candidate["last_step_count"] = max(
                    int(candidate.get("last_step_count", 0)),
                    step_count,
                )
                if is_learning_role:
                    candidate["learning_steps"] = int(candidate.get("learning_steps", 0)) + 1
                    candidate["learning_support_sum"] = float(
                        candidate.get("learning_support_sum", 0.0)
                    ) + float(role_support)
                    candidate["max_learning_posterior"] = max(
                        float(candidate.get("max_learning_posterior", 0.0)),
                        float(role_support),
                    )
                    candidate["support_score"] = float(candidate.get("support_score", 0.0)) + float(
                        role_support
                    )
                else:
                    candidate["output_steps"] = int(candidate.get("output_steps", 0)) + 1
                    candidate["output_support_sum"] = float(
                        candidate.get("output_support_sum", 0.0)
                    ) + float(role_support)
                    candidate["max_output_probability"] = max(
                        float(candidate.get("max_output_probability", 0.0)),
                        float(role_support),
                    )
                    candidate["support_score"] = float(candidate.get("support_score", 0.0)) + (
                        0.35 * float(role_support)
                    )

            if agreement and learning_latent_id is not None and learning_latent_id == output_latent_id:
                candidate = candidates.get(learning_latent_id)
                if candidate is not None:
                    candidate["agreement_steps"] = int(candidate.get("agreement_steps", 0)) + 1
                    candidate["support_score"] = float(candidate.get("support_score", 0.0)) + 0.15

        summaries = []
        for latent_id, candidate in candidates.items():
            candidate["final_learning"] = latent_id == final_learning_latent_id
            candidate["final_output"] = latent_id == final_output_latent_id
            existing_targets = sorted(
                str(existing_target)
                for existing_target in self.latent_id_to_target.get(latent_id, set())
                if str(existing_target) != str(target_object)
            )
            candidate["existing_targets"] = existing_targets
            candidate["shared_with_other_targets"] = bool(existing_targets)
            seen_steps = max(
                int(candidate.get("learning_steps", 0)),
                int(candidate.get("output_steps", 0)),
                1,
            )
            candidate["mean_support"] = float(candidate.get("support_score", 0.0)) / float(
                seen_steps
            )
            candidate["plausible"] = bool(
                int(candidate.get("learning_steps", 0)) > 0
                and (
                    float(candidate.get("support_score", 0.0))
                    >= self._named_target_min_registration_support
                    or float(candidate.get("max_learning_posterior", 0.0))
                    >= self._named_target_min_registration_support
                )
            )
            summaries.append(candidate)

        summaries.sort(
            key=self._episode_registration_candidate_sort_key,
            reverse=True,
        )
        return summaries

    @staticmethod
    def _episode_registration_candidate_sort_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
        return (
            float(candidate.get("support_score", 0.0))
            + (0.25 if candidate.get("final_learning", False) else 0.0)
            + (0.1 if candidate.get("final_output", False) else 0.0),
            float(candidate.get("max_learning_posterior", 0.0)),
            int(candidate.get("agreement_steps", 0)),
            float(candidate.get("max_output_probability", 0.0)),
            int(candidate.get("last_step_count", 0)),
            str(candidate.get("latent_id", "")),
        )

    def _shared_registration_is_overwhelming(
        self,
        *,
        shared_candidate: dict[str, Any],
        exclusive_candidate: dict[str, Any] | None,
    ) -> bool:
        if not shared_candidate.get("shared_with_other_targets", False):
            return True
        if float(shared_candidate.get("max_learning_posterior", 0.0)) < (
            self._named_target_shared_min_learning_posterior
        ):
            return False
        if float(shared_candidate.get("mean_support", 0.0)) < (
            self._named_target_shared_min_mean_support
        ):
            return False
        if exclusive_candidate is None:
            return float(shared_candidate.get("support_score", 0.0)) >= max(
                self._named_target_min_registration_support,
                self._named_target_shared_min_mean_support,
            )
        exclusive_score = float(exclusive_candidate.get("support_score", 0.0))
        if exclusive_score <= 1e-6:
            return True
        return float(shared_candidate.get("support_score", 0.0)) >= (
            exclusive_score * self._named_target_shared_override_ratio
        )

    def _select_episode_registration_candidate(
        self,
        *,
        target_object: str,
        candidates: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        child_lm = self._is_child_reporting_lm()
        plausible_candidates = [candidate for candidate in candidates if candidate.get("plausible", False)]
        exclusive_candidates = [
            candidate
            for candidate in plausible_candidates
            if not candidate.get("shared_with_other_targets", False)
        ]
        shared_candidates = [
            candidate
            for candidate in plausible_candidates
            if candidate.get("shared_with_other_targets", False)
        ]
        best_exclusive = (
            max(exclusive_candidates, key=self._episode_registration_candidate_sort_key)
            if exclusive_candidates
            else None
        )
        best_shared = (
            max(shared_candidates, key=self._episode_registration_candidate_sort_key)
            if shared_candidates
            else None
        )
        best_overall = (
            max(plausible_candidates, key=self._episode_registration_candidate_sort_key)
            if plausible_candidates
            else (candidates[0] if candidates else None)
        )

        decision = {
            "episode_index": int(self._episode_index),
            "target_object": str(target_object),
            "is_child_lm": bool(child_lm),
            "selected_latent_id": None,
            "selection_reason": "no_candidates",
            "candidate_summaries": [
                {
                    "latent_id": str(candidate.get("latent_id")),
                    "support_score": float(candidate.get("support_score", 0.0)),
                    "mean_support": float(candidate.get("mean_support", 0.0)),
                    "learning_steps": int(candidate.get("learning_steps", 0)),
                    "output_steps": int(candidate.get("output_steps", 0)),
                    "agreement_steps": int(candidate.get("agreement_steps", 0)),
                    "max_learning_posterior": float(
                        candidate.get("max_learning_posterior", 0.0)
                    ),
                    "max_output_probability": float(
                        candidate.get("max_output_probability", 0.0)
                    ),
                    "shared_with_other_targets": bool(
                        candidate.get("shared_with_other_targets", False)
                    ),
                    "existing_targets": list(candidate.get("existing_targets", [])),
                    "final_learning": bool(candidate.get("final_learning", False)),
                    "final_output": bool(candidate.get("final_output", False)),
                    "plausible": bool(candidate.get("plausible", False)),
                }
                for candidate in candidates
            ],
        }

        if best_overall is None:
            return None, decision

        if not child_lm:
            decision["selection_reason"] = "best_overall_non_child"
            decision["selected_latent_id"] = str(best_overall.get("latent_id"))
            return best_overall, decision

        if best_exclusive is not None:
            if best_shared is not None and self._shared_registration_is_overwhelming(
                shared_candidate=best_shared,
                exclusive_candidate=best_exclusive,
            ):
                decision["selection_reason"] = "shared_override_overwhelming"
                decision["selected_latent_id"] = str(best_shared.get("latent_id"))
                return best_shared, decision
            decision["selection_reason"] = "best_exclusive_named_anchor"
            decision["selected_latent_id"] = str(best_exclusive.get("latent_id"))
            return best_exclusive, decision

        if best_shared is not None and self._shared_registration_is_overwhelming(
            shared_candidate=best_shared,
            exclusive_candidate=None,
        ):
            decision["selection_reason"] = "shared_only_overwhelming"
            decision["selected_latent_id"] = str(best_shared.get("latent_id"))
            return best_shared, decision

        decision["selection_reason"] = "best_overall_fallback"
        decision["selected_latent_id"] = str(best_overall.get("latent_id"))
        return best_overall, decision

    def _finalize_episode_target_registrations(self) -> None:
        self._last_target_registration_commit = None
        for target_object, events in self._episode_registration_events_for_current_episode().items():
            candidates = self._summarize_episode_registration_candidates(
                target_object=target_object,
                events=events,
            )
            selected_candidate, decision = self._select_episode_registration_candidate(
                target_object=target_object,
                candidates=candidates,
            )
            if selected_candidate is None:
                self._last_target_registration_commit = decision
                continue
            latent_id = str(selected_candidate.get("latent_id"))
            self.latent_id_to_target[latent_id].add(target_object)
            self.target_to_latent_id[target_object].add(latent_id)
            decision["selected_latent_id"] = latent_id
            self._last_target_registration_commit = decision

    def _register_reporting_target(self, result: dict[str, Any]) -> None:
        target_object = self._resolve_reporting_target_object()
        if target_object is None:
            return

        learning_support = ((result.get("self_supervised_support") or {}).get("learning") or {})
        raw_latent_ids = []
        learning_latent_id = _normalize_graph_id(
            learning_support.get("learning_latent_id")
            or learning_support.get("learning_object_id")
        )
        if learning_latent_id is not None:
            raw_latent_ids.append(learning_latent_id)

        output_latent_id = _normalize_graph_id(
            (result.get("mlh") or {}).get(
                "latent_id",
                (result.get("mlh") or {}).get("graph_id"),
            )
        )
        if output_latent_id is not None:
            raw_latent_ids.append(output_latent_id)

        raw_latent_ids = list(dict.fromkeys(raw_latent_ids))
        if not raw_latent_ids:
            return

        self._record_target_registration_debug(
            target_object=target_object,
            learning_support=learning_support,
            result=result,
            registered_latent_ids=raw_latent_ids,
        )

    def send_out_vote(self):
        if not self._stepped:
            self._last_vote_message = None
            return None

        emission_state = self._last_observed_state
        if emission_state is None:
            self._last_vote_message = None
            return None

        ranked_hypotheses = list(self._last_result.get("ranked_hypotheses", []))
        if not ranked_hypotheses:
            self._last_vote_message = None
            return None

        if self._vote_top_k is not None:
            ranked_hypotheses = ranked_hypotheses[: self._vote_top_k]

        location = np.asarray(emission_state.location, dtype=np.float64)
        pose_vectors = np.asarray(
            emission_state.morphological_features.get("pose_vectors", np.eye(3)),
            dtype=np.float64,
        )
        top_hypothesis = ranked_hypotheses[0]
        top_behavior_signature = np.asarray(
            top_hypothesis.get("behavior_signature", np.zeros(0, dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1)
        top_evidence = float(top_hypothesis.get("evidence", 0.0))
        runner_up_evidence = (
            float(ranked_hypotheses[1].get("evidence", 0.0))
            if len(ranked_hypotheses) > 1
            else 0.0
        )
        appearance_summary_is_reliable = (
            top_evidence >= self._evidence_match_threshold
            and (
                len(ranked_hypotheses) == 1
                or top_evidence
                > (runner_up_evidence * self._evidence_separation_ratio)
            )
        )
        vote_appearance_signatures = []
        vote_appearance_weights = []
        if appearance_summary_is_reliable:
            for hypothesis in ranked_hypotheses:
                appearance_signature = np.asarray(
                    hypothesis.get("appearance_signature", np.zeros(0, dtype=np.float32)),
                    dtype=np.float32,
                ).reshape(-1)
                if appearance_signature.size == 0:
                    continue
                vote_appearance_signatures.append(appearance_signature)
                vote_appearance_weights.append(
                    max(float(hypothesis.get("probability", 0.0)), 0.0)
                )
        if vote_appearance_signatures:
            signature_dim = max(
                int(signature.size) for signature in vote_appearance_signatures
            )
            padded_signatures = [
                np.pad(signature, (0, signature_dim - int(signature.size)), mode="constant")
                for signature in vote_appearance_signatures
            ]
            appearance_weights = np.asarray(vote_appearance_weights, dtype=np.float32)
            if float(appearance_weights.sum()) <= 1e-6:
                appearance_weights = np.ones_like(appearance_weights, dtype=np.float32)
            top_appearance_signature = np.average(
                np.stack(padded_signatures),
                axis=0,
                weights=appearance_weights,
            ).astype(np.float32)
        else:
            top_appearance_signature = np.zeros(0, dtype=np.float32)

        possible_states = {}
        for hypothesis in ranked_hypotheses:
            object_id = str(hypothesis.get("object_id"))
            hypothesis_pose = np.asarray(
                hypothesis.get("pose_vectors", pose_vectors),
                dtype=np.float64,
            )
            hypothesis_location = np.asarray(
                hypothesis.get("location", location),
                dtype=np.float64,
            )
            appearance_signature = np.asarray(
                hypothesis.get("appearance_signature", np.zeros(0, dtype=np.float32)),
                dtype=np.float32,
            ).reshape(-1)
            behavior_signature = np.asarray(
                hypothesis.get("behavior_signature", np.zeros(0, dtype=np.float32)),
                dtype=np.float32,
            ).reshape(-1)
            state_non_morph = {
                "vote_probability": float(hypothesis.get("probability", 0.0)),
                "vote_evidence": float(hypothesis.get("evidence", 0.0)),
                "vote_rank": int(hypothesis.get("rank", 1)),
                "graph_id": object_id,
                "latent_id": object_id,
            }
            if hypothesis.get("chart_id") is not None:
                state_non_morph["chart_id"] = hypothesis.get("chart_id")
            if hypothesis.get("behavior_label") is not None:
                state_non_morph["behavior_label"] = hypothesis.get("behavior_label")
            if appearance_signature.size > 0:
                state_non_morph["appearance_signature"] = appearance_signature.copy()
            if behavior_signature.size > 0:
                state_non_morph["behavior_signature"] = behavior_signature.copy()
            possible_states.setdefault(object_id, []).append(
                State(
                    location=hypothesis_location.copy(),
                    morphological_features={
                        "pose_vectors": hypothesis_pose.copy(),
                        "pose_fully_defined": True,
                    },
                    non_morphological_features=state_non_morph,
                    confidence=float(hypothesis.get("probability", 0.0)),
                    use_state=True,
                    sender_id=self.learning_module_id,
                    sender_type="LM",
                )
            )

        self._last_vote_message = PredictiveVoteMessage(
            sender_id=self.learning_module_id,
            sensed_pose_rel_body=np.vstack([location.reshape(1, 3), pose_vectors]),
            active_cells=self._core.get_last_message_state().active_cells.detach().clone(),
            appearance_signature=torch.as_tensor(
                top_appearance_signature,
                dtype=torch.float32,
            ),
            behavior_label=top_hypothesis.get("behavior_label"),
            behavior_signature=torch.as_tensor(
                top_behavior_signature,
                dtype=torch.float32,
            ),
            ranked_hypotheses=[
                RankedHypothesisVote.from_dict(
                    {
                        **copy.deepcopy(hypothesis),
                        "sender_id": self.learning_module_id,
                    }
                )
                for hypothesis in ranked_hypotheses
            ],
            possible_states=possible_states,
        )
        return self._last_vote_message.to_dict()

    def receive_votes(self, votes):
        self._last_received_vote_messages = []
        if votes is None:
            return

        if isinstance(votes, dict):
            votes = [votes]

        for vote in votes:
            if not isinstance(vote, dict):
                continue
            vote_message = PredictiveVoteMessage.from_dict(vote)
            self._last_received_vote_messages.append(vote_message)
            self._core.receive_vote_message(vote_message)

    def propose_goal_states(self) -> list[GoalState]:
        if not self._stepped or self._step_count < self._goal_state_min_steps:
            return []

        mlh = self.get_current_mlh()
        confidence = float(mlh.get("evidence", 0.0))
        if confidence < self._output_evidence_threshold:
            return []

        values = sorted(self.evidence.values(), reverse=True)
        if (
            self._goal_state_min_separation_ratio is not None
            and len(values) >= 2
            and values[0] <= values[1] * self._goal_state_min_separation_ratio
        ):
            return []

        state = self._last_observed_state
        if state is None:
            return []

        return [
            GoalState(
                location=np.asarray(state.location, dtype=np.float64).copy(),
                morphological_features={
                    "pose_vectors": np.asarray(
                        state.morphological_features.get("pose_vectors", np.eye(3)),
                        dtype=np.float64,
                    ).copy(),
                    "pose_fully_defined": True,
                },
                non_morphological_features={
                    "graph_id": mlh.get("latent_id", mlh.get("graph_id")),
                    "latent_id": mlh.get("latent_id", mlh.get("graph_id")),
                },
                confidence=confidence,
                use_state=True,
                sender_id=self.learning_module_id,
                sender_type="GSG",
                goal_tolerances={"location": 0.02},
            )
        ]

    def get_output(self):
        if not self._stepped:
            return None

        state = self._last_observed_state
        if state is None:
            return None

        mlh = self.get_current_mlh()
        evidence = float(mlh.get("evidence", 0.0))
        use_state = evidence >= self._output_evidence_threshold
        context_message = self._core.get_context_signal_message()
        message_state = self._core.get_last_message_state()
        temporal_context = self._core.get_temporal_context()
        raw_latent_id = _normalize_graph_id(
            mlh.get("latent_id", mlh.get("graph_id"))
        )
        non_morphological_features = {
            "evidence": evidence,
            "active_cells": None
            if context_message is None
            else context_message.active_cells.detach().cpu().numpy().astype(np.float32),
            "residual": float(message_state.residual),
        }
        if raw_latent_id is not None:
            non_morphological_features["graph_id"] = raw_latent_id
            non_morphological_features["latent_id"] = raw_latent_id
        if mlh.get("chart_id") is not None:
            non_morphological_features["chart_id"] = mlh.get("chart_id")
        if mlh.get("behavior_label") is not None:
            non_morphological_features["behavior_label"] = mlh.get("behavior_label")
        appearance_signature = np.asarray(
            mlh.get("appearance_signature", np.zeros(0, dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1)
        if appearance_signature.size > 0:
            non_morphological_features["appearance_signature"] = appearance_signature.copy()
        predicted_appearance_signature = (
            message_state.predicted_appearance_signature.detach()
            .cpu()
            .numpy()
            .astype(np.float32)
            .reshape(-1)
        )
        if predicted_appearance_signature.size > 0:
            non_morphological_features["predicted_appearance_signature"] = (
                predicted_appearance_signature.copy()
            )
        behavior_signature = np.asarray(
            mlh.get("behavior_signature", np.zeros(0, dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1)
        if behavior_signature.size > 0:
            non_morphological_features["behavior_signature"] = behavior_signature.copy()
        predicted_change_signature = (
            message_state.predicted_change_signature.detach()
            .cpu()
            .numpy()
            .astype(np.float32)
            .reshape(-1)
        )
        if predicted_change_signature.size > 0:
            non_morphological_features["predicted_change_signature"] = (
                predicted_change_signature.copy()
            )
        appearance_residual = (
            message_state.appearance_residual.detach()
            .cpu()
            .numpy()
            .astype(np.float32)
            .reshape(-1)
        )
        if appearance_residual.size > 0:
            non_morphological_features["appearance_residual"] = appearance_residual.copy()
        change_residual = (
            message_state.change_residual.detach().cpu().numpy().astype(np.float32).reshape(-1)
        )
        if change_residual.size > 0:
            non_morphological_features["change_residual"] = change_residual.copy()
        if temporal_context is not None:
            if temporal_context.get("appearance_prediction_error") is not None:
                non_morphological_features["appearance_prediction_error"] = float(
                    temporal_context["appearance_prediction_error"]
                )
            if temporal_context.get("change_prediction_error") is not None:
                non_morphological_features["change_prediction_error"] = float(
                    temporal_context["change_prediction_error"]
                )

        if state.sender_type == "LM":
            child_sender_ids = (state.non_morphological_features or {}).get("child_sender_ids")
            if child_sender_ids is not None:
                non_morphological_features["child_sender_ids"] = list(child_sender_ids)
            child_latent_ids = (state.non_morphological_features or {}).get(
                "child_latent_ids",
                (state.non_morphological_features or {}).get("child_graph_ids"),
            )
            if child_latent_ids is not None:
                non_morphological_features["child_graph_ids"] = list(child_latent_ids)
                non_morphological_features["child_latent_ids"] = list(child_latent_ids)

        output_state = State(
            location=np.asarray(state.location, dtype=np.float64).copy(),
            morphological_features={
                "pose_vectors": np.asarray(
                    state.morphological_features.get("pose_vectors", np.eye(3)),
                    dtype=np.float64,
                ).copy(),
                "pose_fully_defined": True,
                "on_object": True,
            },
            non_morphological_features=non_morphological_features,
            confidence=min(1.0, evidence),
            use_state=use_state,
            sender_id=self.learning_module_id,
            sender_type="LM",
            inferred_state=getattr(state, "inferred_state", None),
        )
        self._last_output_state = output_state
        return output_state

    def receive_context(self, **context_signal):
        self.receive_context_message(
            PredictiveContextSignal.from_dict(
                context_signal,
                device=self._core.device,
            )
        )

    def receive_context_message(self, context_message: PredictiveContextSignal):
        self._last_received_context_message = context_message
        self._core.receive_context_message(context_message)

    def _extract_child_latent_ids(self) -> list[str]:
        state = self._last_observed_state
        if state is None or state.sender_type != "LM":
            return []

        non_morph = getattr(state, "non_morphological_features", None) or {}
        child_latent_ids = []
        for item in non_morph.get("child_latent_ids", non_morph.get("child_graph_ids")) or []:
            normalized = _normalize_graph_id(item)
            if normalized is not None:
                child_latent_ids.append(normalized)
        return child_latent_ids

    def _extract_child_graph_ids(self) -> list[str]:
        return self._extract_child_latent_ids()

    def _extract_child_sender_ids(self) -> list[str]:
        state = self._last_observed_state
        if state is None or state.sender_type != "LM":
            return []

        non_morph = getattr(state, "non_morphological_features", None) or {}
        child_sender_ids = []
        for item in non_morph.get("child_sender_ids") or []:
            if item is None:
                continue
            normalized = str(item).strip()
            if not normalized or normalized == "unknown":
                continue
            if normalized not in child_sender_ids:
                child_sender_ids.append(normalized)
        return child_sender_ids

    def get_context_message(self) -> PredictiveContextSignal | None:
        signal = self._core.get_context_signal_message()
        if signal is None:
            self._last_context_message = None
            return None

        signal = copy.deepcopy(signal)
        target_sender_ids = self._extract_child_sender_ids()
        signal.latent_id = None
        signal.child_latent_ids = []
        signal.sender_id = self.learning_module_id
        signal.sender_step_count = self._step_count
        signal.routing_scope = (
            "targeted" if target_sender_ids else "graph_neighbors"
        )
        signal.target_sender_ids = target_sender_ids
        self._last_context_message = signal
        return signal

    def get_context_signal(self) -> dict | None:
        signal = self.get_context_message()
        return None if signal is None else signal.to_dict()

    def state_dict(self):
        latent_id_to_target = {k: sorted(v) for k, v in self.latent_id_to_target.items()}
        target_to_latent_id = {k: sorted(v) for k, v in self.target_to_latent_id.items()}
        return {
            "core": self._core.state_dict(),
            "latent_id_to_target": latent_id_to_target,
            "graph_id_to_target": dict(latent_id_to_target),
            "target_to_latent_id": target_to_latent_id,
            "target_to_graph_id": dict(target_to_latent_id),
        }

    def load_state_dict(self, state_dict):
        self._core.load_state_dict(state_dict.get("core", {}))
        self.latent_id_to_target = state_dict.get(
            "latent_id_to_target",
            state_dict.get("graph_id_to_target", {}),
        )
        self.target_to_latent_id = state_dict.get(
            "target_to_latent_id",
            state_dict.get("target_to_graph_id", {}),
        )

    def get_possible_matches(self) -> list:
        return list(self.possible_matches.keys())

    def collect_stats_to_save(self) -> dict:
        return {"possible_matches": self.get_possible_matches()}

    def set_individual_ts(self, terminal_state) -> None:
        self.terminal_state = terminal_state
        if terminal_state == "match":
            self.detected_object = self.get_current_mlh().get(
                "latent_id",
                self.get_current_mlh().get("graph_id"),
            )
            self.buffer.stats["detected_rotation_quat"] = Rotation.identity().as_quat()
        elif terminal_state == "no_match":
            self.detected_object = None
        else:
            self.detected_object = terminal_state
        self.buffer.stats["individual_ts_reached_at_step"] = self._step_count
        self.buffer.stats["individual_ts_object"] = self.detected_object
        self.buffer.stats["individual_ts_rot"] = self.buffer.stats.get(
            "detected_rotation_quat",
            Rotation.identity().as_quat(),
        )

    def update_terminal_condition(self) -> str | None:
        matches = self.get_possible_matches()
        if not matches and self._stepped:
            self.set_individual_ts("no_match")
            return self.terminal_state

        if len(matches) == 1:
            self.set_individual_ts("match")
            return self.terminal_state

        values = sorted((float(value) for value in self.evidence.values()), reverse=True)
        if len(values) >= 2 and values[0] > self._evidence_match_threshold:
            if values[0] > values[1] * self._evidence_separation_ratio:
                self.set_individual_ts("match")
        return self.terminal_state

    def add_lm_processing_to_buffer_stats(self, lm_processed: bool) -> None:
        self.buffer.update_stats(dict(lm_processed_steps=lm_processed), update_time=False)
        self.buffer.on_object.append(lm_processed)

    def get_all_known_object_ids(self) -> list[str]:
        return self._core.get_all_known_object_ids()

    def get_all_known_latent_ids(self) -> list[str]:
        return self._core.get_all_known_latent_ids()

    def set_action_context(self, action_context) -> None:
        self._core.set_action_context(action_context)

    def get_current_mlh(self) -> dict:
        return self._core.get_current_mlh()

    def get_evidence_debug(self) -> dict[str, Any]:
        if not self._stepped:
            return {
                "step_skipped": True,
                "skip_reason": self._last_skip_reason or "not_stepped",
            }

        debug = self._core.get_evidence_debug()
        debug["boundary_pressure"] = float(self._core.get_temporal_surprise())
        debug["ranked_hypotheses"] = (
            self._core.get_last_message_state().ranked_hypotheses_as_dicts()
        )
        debug["target_registration"] = (
            None
            if self._last_target_registration_debug is None
            else copy.deepcopy(self._last_target_registration_debug)
        )
        debug["target_registration_commit"] = (
            None
            if self._last_target_registration_commit is None
            else copy.deepcopy(self._last_target_registration_commit)
        )
        return debug

    def get_reporting_alias_diagnostics(self) -> dict[str, Any]:
        per_target = {}
        for target_object, stats in sorted(
            self._target_registration_stats.get("per_target", {}).items()
        ):
            per_target[str(target_object)] = {
                "steps": int(stats.get("steps", 0)),
                "agreement_steps": int(stats.get("agreement_steps", 0)),
                "mismatch_steps": int(stats.get("mismatch_steps", 0)),
                "learning_only_steps": int(stats.get("learning_only_steps", 0)),
                "output_only_steps": int(stats.get("output_only_steps", 0)),
                "learning_latent_counts": {
                    str(latent_id): int(count)
                    for latent_id, count in sorted(
                        (stats.get("learning_latent_counts") or {}).items()
                    )
                },
                "output_latent_counts": {
                    str(latent_id): int(count)
                    for latent_id, count in sorted(
                        (stats.get("output_latent_counts") or {}).items()
                    )
                },
                "registered_latent_counts": {
                    str(latent_id): int(count)
                    for latent_id, count in sorted(
                        (stats.get("registered_latent_counts") or {}).items()
                    )
                },
                "learning_output_pairs": {
                    str(pair): int(count)
                    for pair, count in sorted(
                        (stats.get("learning_output_pairs") or {}).items()
                    )
                },
            }
        return {
            "total_steps": int(self._target_registration_stats.get("total_steps", 0)),
            "agreement_steps": int(
                self._target_registration_stats.get("agreement_steps", 0)
            ),
            "mismatch_steps": int(
                self._target_registration_stats.get("mismatch_steps", 0)
            ),
            "learning_only_steps": int(
                self._target_registration_stats.get("learning_only_steps", 0)
            ),
            "output_only_steps": int(
                self._target_registration_stats.get("output_only_steps", 0)
            ),
            "per_target": per_target,
            "last_event": (
                None
                if self._last_target_registration_debug is None
                else copy.deepcopy(self._last_target_registration_debug)
            ),
            "last_commit": (
                None
                if self._last_target_registration_commit is None
                else copy.deepcopy(self._last_target_registration_commit)
            ),
            "recent_events": copy.deepcopy(self._target_registration_history[-16:]),
        }

    def get_temporal_prediction_status(self) -> str | None:
        return self._core.get_temporal_prediction_status()

    def get_temporal_surprise(self) -> float:
        return self._core.get_temporal_surprise()

    def get_temporal_context(self) -> dict[str, Any]:
        return self._core.get_temporal_context()

    def get_event_signal(self) -> bool:
        return bool(self._core.get_temporal_state().event_detected_bool)

    def get_speed_signal(self):
        return None

    def get_last_detail_packet(self) -> dict[str, Any] | None:
        return self._core.get_last_detail_packet()

    def get_last_observation_field(self):
        return self._core.get_last_observation_field()

    def get_last_embeddings(self):
        return self._core.get_last_embeddings()

    def get_memory_slots(self):
        return self._core.get_memory_slots()

    def get_change_memory_slots(self):
        return self._core.get_change_memory_slots()

    def get_last_memory_retrieval(self):
        return self._core.get_last_memory_retrieval()

    def get_last_change_memory_retrieval(self):
        return self._core.get_last_change_memory_retrieval()

    def get_last_stream_memory_retrievals(self):
        return self._core.get_last_stream_memory_retrievals()

    def get_hypothesis_state(self):
        return self._core.get_hypothesis_state()

    def get_temporal_state(self):
        return self._core.get_temporal_state()

    def get_last_vote_message(self):
        return self._last_vote_message

    def get_last_received_vote_messages(self):
        return list(self._last_received_vote_messages)

    def get_last_received_context_message(self):
        return self._last_received_context_message

    def get_hypothesis_bank(self) -> list[dict[str, Any]]:
        return list(self._core.hypotheses.as_ranked_hypotheses())
