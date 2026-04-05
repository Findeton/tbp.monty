#!/usr/bin/env python

"""Synthetic Track 9b benchmark for temporal and behavior modeling.

This benchmark focuses on the first viable self-supervised Track 9b prototype:

- same LM family for morphology and behavior
- temporal head is opt-in inside CorticalColumnTorchLM
- self-supervised latent-state discovery over settled Torch embeddings
- transient latent-state-to-object evidence bias during matching

The tasks are deliberately synthetic so they can be run quickly and isolate the
temporal mechanism from rendering, motor control, and environment confounds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.models.cortical_column_torch.learning_module import (
    CorticalColumnTorchLM,
)
from tbp.monty.frameworks.models.states import State


def _make_state(
    phase_id: int,
    noise: float = 0.0,
    include_inferred_state: bool = True,
) -> State:
    hsv_bank = {
        0: np.array([0.15, 0.30, 0.45]),
        1: np.array([0.55, 0.30, 0.45]),
        2: np.array([0.85, 0.30, 0.45]),
    }
    curv_bank = {
        0: np.array([0.10, 0.30]),
        1: np.array([0.35, 0.55]),
        2: np.array([0.60, 0.80]),
    }

    rng = np.random.RandomState(phase_id + 17)
    hsv = hsv_bank[phase_id].copy()
    curv = curv_bank[phase_id].copy()
    if noise > 0:
        hsv += rng.normal(scale=noise, size=hsv.shape)
        curv += rng.normal(scale=noise, size=curv.shape)

    return State(
        location=np.zeros(3, dtype=np.float64),
        morphological_features={
            "pose_vectors": np.eye(3),
            "pose_fully_defined": True,
        },
        non_morphological_features={
            "hsv": hsv,
            "principal_curvatures_log": curv,
        },
        confidence=1.0,
        use_state=True,
        sender_id="synthetic_sm",
        sender_type="SM",
        inferred_state=phase_id if include_inferred_state else None,
    )


def _behavior_sequence(
    phases: list[int],
    tempo: int = 1,
    noise: float = 0.0,
    include_inferred_state: bool = True,
):
    seq = []
    for phase in phases:
        for _ in range(tempo):
            seq.append(
                _make_state(
                    phase,
                    noise=noise,
                    include_inferred_state=include_inferred_state,
                )
            )
    return seq


def _train_lm(lm: CorticalColumnTorchLM, label: str, seq, use_stepwise_state=False):
    lm.set_experiment_mode(ExperimentMode.TRAIN)
    lm.pre_episode(primary_target={"object": label, "quat_rotation": [1, 0, 0, 0]})
    for obs in seq:
        lm.stepwise_target_state = obs.inferred_state if use_stepwise_state else None
        lm.exploratory_step(None, [obs])
    lm.post_episode()


def _eval_lm(lm: CorticalColumnTorchLM, seq):
    lm.set_experiment_mode(ExperimentMode.EVAL)
    lm.pre_episode(primary_target={"object": "placeholder", "quat_rotation": [1, 0, 0, 0]})
    outputs = []
    for obs in seq:
        lm.matching_step(None, [obs])
        temporal_context = lm.get_temporal_context() or {}
        outputs.append(
            {
                "graph_id": lm.get_current_mlh().get("graph_id"),
                "evidence": float(lm.get_current_mlh().get("evidence", 0.0)),
                "temporal_status": lm.get_temporal_prediction_status(),
                "temporal_surprise": float(lm.get_temporal_surprise()),
                "current_label": temporal_context.get("current_label"),
                "predicted_label": temporal_context.get("predicted_label"),
            }
        )
    return outputs


def _fit_latent_decoder(outputs, target_phases, label_key="current_label"):
    counts = {}

    for output, target_phase in zip(outputs, target_phases):
        label = output.get(label_key)
        if label is None:
            continue
        counts.setdefault(label, {})
        counts[label][target_phase] = counts[label].get(target_phase, 0) + 1

    decoder = {}
    for label, label_counts in counts.items():
        decoder[label] = max(
            label_counts.items(),
            key=lambda item: (item[1], -item[0]),
        )[0]

    return decoder


def _decode_accuracy(outputs, target_phases, decoder, label_key="current_label"):
    total = 0
    correct = 0

    for output, target_phase in zip(outputs, target_phases):
        label = output.get(label_key)
        decoded = decoder.get(label)
        if decoded is None:
            continue
        total += 1
        correct += int(decoded == target_phase)

    return (correct / total) if total > 0 else None, total


def _make_behavior_lm(use_temporal: bool) -> CorticalColumnTorchLM:
    kwargs = {}
    if use_temporal:
        kwargs.update(
            self_supervised_temporal_config={
                "match_threshold": 0.8,
                "new_state_threshold": 0.55,
                "min_segment_steps": 1,
                "default_duration": 2.0,
                "switch_margin": 0.05,
            },
            self_supervised_temporal_transition_weight=4.0,
            self_supervised_temporal_prediction_weight=1.5,
            self_supervised_temporal_evidence_weight=0.25,
        )

    return CorticalColumnTorchLM(
        column_kwargs={
            "n_minicolumns": 128,
            "n_cells_per_minicolumn": 4,
            "sparsity": 0.08,
            "seed": 42,
        },
        output_evidence_threshold=0.5,
        evidence_match_threshold=0.75,
        evidence_separation_ratio=1.1,
        **kwargs,
    )


def _make_state_lm(use_temporal: bool) -> CorticalColumnTorchLM:
    kwargs = {}
    if use_temporal:
        kwargs.update(
            self_supervised_temporal_config={
                "match_threshold": 0.8,
                "new_state_threshold": 0.55,
                "min_segment_steps": 1,
                "default_duration": 2.0,
                "switch_margin": 0.05,
            },
        )

    return CorticalColumnTorchLM(
        column_kwargs={
            "n_minicolumns": 128,
            "n_cells_per_minicolumn": 4,
            "sparsity": 0.08,
            "seed": 84,
        },
        output_evidence_threshold=0.5,
        evidence_match_threshold=0.75,
        evidence_separation_ratio=1.1,
        **kwargs,
    )


def run_ambiguous_order_benchmark() -> dict:
    phases_a = [0, 1, 2, 1, 0]
    phases_b = [0, 2, 1, 2, 0]

    baseline = _make_behavior_lm(use_temporal=False)
    temporal = _make_behavior_lm(use_temporal=True)

    for _ in range(5):
        _train_lm(
            baseline,
            "behavior_a",
            _behavior_sequence(
                phases_a, tempo=1, noise=0.01, include_inferred_state=False
            ),
        )
        _train_lm(
            baseline,
            "behavior_b",
            _behavior_sequence(
                phases_b, tempo=1, noise=0.01, include_inferred_state=False
            ),
        )
        _train_lm(
            temporal,
            "behavior_a",
            _behavior_sequence(
                phases_a, tempo=1, noise=0.01, include_inferred_state=False
            ),
        )
        _train_lm(
            temporal,
            "behavior_b",
            _behavior_sequence(
                phases_b, tempo=1, noise=0.01, include_inferred_state=False
            ),
        )

    eval_a = _behavior_sequence(
        phases_a, tempo=1, noise=0.02, include_inferred_state=False
    )
    eval_b = _behavior_sequence(
        phases_b, tempo=1, noise=0.02, include_inferred_state=False
    )

    base_out_a = _eval_lm(baseline, eval_a)
    base_out_b = _eval_lm(baseline, eval_b)
    temporal_train_out_a = _eval_lm(
        temporal,
        _behavior_sequence(phases_a, tempo=1, noise=0.01, include_inferred_state=False),
    )
    temporal_train_out_b = _eval_lm(
        temporal,
        _behavior_sequence(phases_b, tempo=1, noise=0.01, include_inferred_state=False),
    )
    temp_out_a = _eval_lm(temporal, eval_a)
    temp_out_b = _eval_lm(temporal, eval_b)
    decoder = _fit_latent_decoder(
        temporal_train_out_a + temporal_train_out_b,
        phases_a + phases_b,
    )
    next_acc_a, next_total_a = _decode_accuracy(
        temp_out_a[:-1],
        phases_a[1:],
        decoder,
        label_key="predicted_label",
    )
    next_acc_b, next_total_b = _decode_accuracy(
        temp_out_b[:-1],
        phases_b[1:],
        decoder,
        label_key="predicted_label",
    )

    return {
        "baseline_final": [base_out_a[-1]["graph_id"], base_out_b[-1]["graph_id"]],
        "temporal_final": [temp_out_a[-1]["graph_id"], temp_out_b[-1]["graph_id"]],
        "baseline_correct": int(base_out_a[-1]["graph_id"] == "behavior_a")
        + int(base_out_b[-1]["graph_id"] == "behavior_b"),
        "temporal_correct": int(temp_out_a[-1]["graph_id"] == "behavior_a")
        + int(temp_out_b[-1]["graph_id"] == "behavior_b"),
        "discovered_states": sorted(label for label in decoder if label is not None),
        "latent_next_phase_accuracy_a": next_acc_a,
        "latent_next_phase_accuracy_b": next_acc_b,
        "latent_next_phase_evaluable_steps": next_total_a + next_total_b,
    }


def run_tempo_invariance_benchmark() -> dict:
    phases_a = [0, 1, 2, 1, 0]
    phases_b = [0, 2, 1, 2, 0]

    baseline = _make_behavior_lm(use_temporal=False)
    temporal = _make_behavior_lm(use_temporal=True)

    for _ in range(5):
        _train_lm(
            baseline,
            "behavior_a",
            _behavior_sequence(
                phases_a, tempo=1, noise=0.01, include_inferred_state=False
            ),
        )
        _train_lm(
            baseline,
            "behavior_b",
            _behavior_sequence(
                phases_b, tempo=1, noise=0.01, include_inferred_state=False
            ),
        )
        _train_lm(
            temporal,
            "behavior_a",
            _behavior_sequence(
                phases_a, tempo=1, noise=0.01, include_inferred_state=False
            ),
        )
        _train_lm(
            temporal,
            "behavior_b",
            _behavior_sequence(
                phases_b, tempo=1, noise=0.01, include_inferred_state=False
            ),
        )

    stretched_a = _behavior_sequence(
        phases_a, tempo=2, noise=0.02, include_inferred_state=False
    )
    stretched_b = _behavior_sequence(
        phases_b, tempo=3, noise=0.02, include_inferred_state=False
    )

    base_out_a = _eval_lm(baseline, stretched_a)
    base_out_b = _eval_lm(baseline, stretched_b)
    temporal_train_out_a = _eval_lm(
        temporal,
        _behavior_sequence(phases_a, tempo=1, noise=0.01, include_inferred_state=False),
    )
    temporal_train_out_b = _eval_lm(
        temporal,
        _behavior_sequence(phases_b, tempo=1, noise=0.01, include_inferred_state=False),
    )
    temp_out_a = _eval_lm(temporal, stretched_a)
    temp_out_b = _eval_lm(temporal, stretched_b)
    decoder = _fit_latent_decoder(
        temporal_train_out_a + temporal_train_out_b,
        phases_a + phases_b,
    )
    target_next_a = [phase for idx, phase in enumerate(phases_a) for _ in range(2)][1:]
    target_next_b = [phase for idx, phase in enumerate(phases_b) for _ in range(3)][1:]
    next_acc_a, next_total_a = _decode_accuracy(
        temp_out_a[:-1],
        target_next_a,
        decoder,
        label_key="predicted_label",
    )
    next_acc_b, next_total_b = _decode_accuracy(
        temp_out_b[:-1],
        target_next_b,
        decoder,
        label_key="predicted_label",
    )

    return {
        "baseline_final": [base_out_a[-1]["graph_id"], base_out_b[-1]["graph_id"]],
        "temporal_final": [temp_out_a[-1]["graph_id"], temp_out_b[-1]["graph_id"]],
        "baseline_correct": int(base_out_a[-1]["graph_id"] == "behavior_a")
        + int(base_out_b[-1]["graph_id"] == "behavior_b"),
        "temporal_correct": int(temp_out_a[-1]["graph_id"] == "behavior_a")
        + int(temp_out_b[-1]["graph_id"] == "behavior_b"),
        "discovered_states": sorted(label for label in decoder if label is not None),
        "latent_next_phase_accuracy_a": next_acc_a,
        "latent_next_phase_accuracy_b": next_acc_b,
        "latent_next_phase_evaluable_steps": next_total_a + next_total_b,
    }


def run_state_transition_benchmark() -> dict:
    train_seq = _behavior_sequence([0, 0, 1, 1, 0, 0], tempo=1, noise=0.01)
    eval_seq = _behavior_sequence([0, 0, 1, 1, 0, 0], tempo=1, noise=0.02)

    baseline = _make_state_lm(use_temporal=False)
    temporal = _make_state_lm(use_temporal=True)

    for _ in range(6):
        _train_lm(baseline, "hinge", train_seq, use_stepwise_state=True)
        _train_lm(temporal, "hinge", train_seq, use_stepwise_state=False)

    base_out = _eval_lm(baseline, eval_seq)
    temporal_train_out = _eval_lm(temporal, train_seq)
    temp_out = _eval_lm(temporal, eval_seq)

    expected_labels = [f"hinge:{obs.inferred_state}" for obs in eval_seq]
    baseline_state_correct = sum(
        int(step["graph_id"] == expected)
        for step, expected in zip(base_out, expected_labels)
    )
    temporal_state_correct = sum(
        int(step["graph_id"] == expected)
        for step, expected in zip(temp_out, expected_labels)
    )
    temporal_prediction_confident = sum(
        int(step["temporal_status"] == "confident") for step in temp_out
    )
    train_phases = [obs.inferred_state for obs in train_seq]
    eval_phases = [obs.inferred_state for obs in eval_seq]
    decoder = _fit_latent_decoder(temporal_train_out, train_phases)
    temporal_state_accuracy, temporal_state_evaluable = _decode_accuracy(
        temp_out,
        eval_phases,
        decoder,
        label_key="current_label",
    )
    temporal_next_state_accuracy, temporal_next_state_evaluable = _decode_accuracy(
        temp_out[:-1],
        eval_phases[1:],
        decoder,
        label_key="predicted_label",
    )

    return {
        "baseline_state_accuracy": baseline_state_correct / len(expected_labels),
        "supervised_temporal_state_accuracy": (
            temporal_state_correct / len(expected_labels)
        ),
        "latent_temporal_state_accuracy": temporal_state_accuracy,
        "latent_temporal_state_evaluable_steps": temporal_state_evaluable,
        "latent_next_state_accuracy": temporal_next_state_accuracy,
        "latent_next_state_evaluable_steps": temporal_next_state_evaluable,
        "temporal_confident_fraction": temporal_prediction_confident / len(temp_out),
        "final_temporal_label": temp_out[-1]["graph_id"],
        "discovered_states": sorted(label for label in decoder if label is not None),
    }


@dataclass
class BenchmarkReport:
    ambiguous_order: dict
    tempo_invariance: dict
    state_transition: dict

    def to_dict(self):
        return {
            "ambiguous_order": self.ambiguous_order,
            "tempo_invariance": self.tempo_invariance,
            "state_transition": self.state_transition,
        }


def main():
    report = BenchmarkReport(
        ambiguous_order=run_ambiguous_order_benchmark(),
        tempo_invariance=run_tempo_invariance_benchmark(),
        state_transition=run_state_transition_benchmark(),
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()