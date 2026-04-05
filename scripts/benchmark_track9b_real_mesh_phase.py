#!/usr/bin/env python

"""Track 9b benchmark on a real animated mesh with motion-grounded phases.

This uses the existing Fox.glb animation in the test assets. The mesh is real,
the motion is real, and phase labels are derived from animated joint
kinematics so the behavior LM can be evaluated on local phase recognition and
simple next-phase anticipation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.models.cortical_column_torch.experiment import (
    Panda3DTorchExperiment,
)
from tbp.monty.simulators.panda3d.kinematic_phase import (
    extract_foot_cycle_phase_info,
    extract_joint_phase_info,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = (
    REPO_ROOT
    / "tests"
    / "unit"
    / "simulators"
    / "panda3d"
    / "test_assets"
    / "animated"
)
FOX_PATH = ASSET_DIR / "Fox.glb"


def _default_self_supervised_behavior_lm_kwargs():
    return {
        "self_supervised_temporal_config": {
            "match_threshold": 0.8,
            "new_state_threshold": 0.55,
            "min_segment_steps": 1,
            "default_duration": 2.0,
            "switch_margin": 0.05,
        },
        "self_supervised_temporal_evidence_weight": 0.15,
        "self_supervised_temporal_transition_weight": 1.5,
        "self_supervised_temporal_prediction_weight": 0.75,
    }


def _select_animation_name(exp, preferred=None):
    names = list(exp._anim_obj.animation_names)
    if not names:
        raise RuntimeError("No animations found on the loaded mesh")
    if preferred and preferred in names:
        return preferred

    for name in names:
        if "walk" in str(name).lower():
            return name

    return names[0]


def _build_schedule(n_frames, cycles, stretch=1):
    schedule = []
    for _ in range(cycles):
        for frame in range(n_frames):
            schedule.extend([frame] * max(1, int(stretch)))
    return schedule


def _phase_targets_for_schedule(phase_by_frame, frame_schedule):
    phase_by_frame = list(int(phase) for phase in phase_by_frame)
    n_frames = len(phase_by_frame)
    if n_frames <= 0:
        return [0 for _ in frame_schedule]
    return [phase_by_frame[int(frame) % n_frames] for frame in frame_schedule]


def _fit_latent_phase_decoder(trace, phase_targets, target_lm_id="lm_behavior"):
    counts = {}

    for step_trace, target_phase in zip(trace, phase_targets):
        temporal_context = (
            step_trace.get("learning_modules", {})
            .get(target_lm_id, {})
            .get("temporal_context", {})
        )
        label = temporal_context.get("current_label")
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


def _compute_metrics(trace, phase_targets, latent_decoder, target_lm_id="lm_behavior"):
    current_total = 0
    current_correct = 0
    next_total = 0
    next_correct = 0
    confident_total = 0
    confident_count = 0

    for idx, (step_trace, target_phase) in enumerate(zip(trace, phase_targets)):
        lm_trace = step_trace.get("learning_modules", {}).get(target_lm_id, {})
        temporal_context = lm_trace.get("temporal_context") or {}

        current_phase = latent_decoder.get(temporal_context.get("current_label"))
        if current_phase is not None:
            current_total += 1
            current_correct += int(current_phase == target_phase)

        status = lm_trace.get("temporal_status")
        if status in {"confident", "confused"}:
            confident_total += 1
            confident_count += int(status == "confident")

        predicted_phase = latent_decoder.get(temporal_context.get("predicted_label"))
        if predicted_phase is None or idx + 1 >= len(trace):
            continue

        next_total += 1
        next_correct += int(predicted_phase == phase_targets[idx + 1])

    return {
        "current_phase_accuracy": (
            current_correct / current_total if current_total > 0 else None
        ),
        "current_phase_evaluable_steps": current_total,
        "next_phase_accuracy": next_correct / next_total if next_total > 0 else None,
        "next_phase_evaluable_steps": next_total,
        "temporal_confident_fraction": (
            confident_count / confident_total if confident_total > 0 else None
        ),
        "temporal_status_steps": confident_total,
    }


def run_benchmark(
    phase_bins=4,
    train_cycles=4,
    eval_cycles=2,
    stretch=2,
    preferred_animation=None,
    phase_method="foot_cycle",
):
    if not FOX_PATH.is_file():
        raise FileNotFoundError(f"Fox asset not found: {FOX_PATH}")

    exp = Panda3DTorchExperiment(
        model_path=FOX_PATH,
        hierarchical=True,
        resolution=(32, 32),
        initial_distance=2.0,
        object_scale=(0.01, 0.01, 0.01),
        flow_threshold=1e-6,
        asset_search_paths=[str(ASSET_DIR)],
        column_kwargs={"n_minicolumns": 512, "sparsity": 0.05},
        behavior_lm_kwargs=_default_self_supervised_behavior_lm_kwargs(),
    )

    try:
        exp._setup()
        anim_name = _select_animation_name(exp, preferred=preferred_animation)
        n_frames = exp._anim_obj.get_num_frames(anim_name)
        if phase_method == "foot_cycle":
            try:
                phase_info = extract_foot_cycle_phase_info(
                    exp._anim_obj,
                    anim_name=anim_name,
                    n_phase_bins=phase_bins,
                )
            except ValueError:
                phase_info = extract_joint_phase_info(
                    exp._anim_obj,
                    anim_name=anim_name,
                    n_phase_bins=phase_bins,
                )
        elif phase_method == "joint_configuration_pca":
            phase_info = extract_joint_phase_info(
                exp._anim_obj,
                anim_name=anim_name,
                n_phase_bins=phase_bins,
            )
        else:
            raise ValueError(f"Unknown phase_method: {phase_method}")
        matched_train_schedule = _build_schedule(n_frames, cycles=1)
        matched_eval_schedule = _build_schedule(n_frames, cycles=eval_cycles)
        stretched_eval_schedule = _build_schedule(
            n_frames,
            cycles=eval_cycles,
            stretch=stretch,
        )
        training_trace = []

        for _ in range(train_cycles):
            result = exp.run_episode(
                mode=ExperimentMode.TRAIN,
                object_name="fox_phase",
                anim_name=anim_name,
                frame_schedule=matched_train_schedule,
                collect_trace=True,
            )
            training_trace.extend(result.get("trace", []))

        train_phase_targets = _phase_targets_for_schedule(
            phase_info["phase_by_frame"],
            matched_train_schedule * train_cycles,
        )
        latent_decoder = _fit_latent_phase_decoder(training_trace, train_phase_targets)

        matched = exp.run_episode(
            mode=ExperimentMode.EVAL,
            anim_name=anim_name,
            frame_schedule=matched_eval_schedule,
            collect_trace=True,
        )
        stretched = exp.run_episode(
            mode=ExperimentMode.EVAL,
            anim_name=anim_name,
            frame_schedule=stretched_eval_schedule,
            collect_trace=True,
        )

        behavior_lm = exp.monty.learning_modules[1]
        known_labels = sorted(
            behavior_lm.get_known_temporal_states()
        )
        matched_phase_targets = _phase_targets_for_schedule(
            phase_info["phase_by_frame"],
            matched_eval_schedule,
        )
        stretched_phase_targets = _phase_targets_for_schedule(
            phase_info["phase_by_frame"],
            stretched_eval_schedule,
        )

        return {
            "asset": str(FOX_PATH),
            "animation": anim_name,
            "n_frames": n_frames,
            "phase_bins": phase_bins,
            "train_cycles": train_cycles,
            "eval_cycles": eval_cycles,
            "stretch": stretch,
            "phase_method": phase_info.get("phase_method", phase_method),
            "phase_joint_names": phase_info["joint_names"],
            "phase_root_joint": phase_info["root_joint_name"],
            "phase_span": phase_info["phase_span"],
            "phase_primary_axis": phase_info.get("primary_axis"),
            "phase_secondary_axis": phase_info.get("secondary_axis"),
            "discovered_latent_states": known_labels,
            "latent_phase_decoder": latent_decoder,
            "matched": _compute_metrics(
                matched.get("trace", []),
                matched_phase_targets,
                latent_decoder,
            ),
            "stretched": _compute_metrics(
                stretched.get("trace", []),
                stretched_phase_targets,
                latent_decoder,
            ),
        }
    finally:
        exp.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-bins", type=int, default=4)
    parser.add_argument("--train-cycles", type=int, default=4)
    parser.add_argument("--eval-cycles", type=int, default=2)
    parser.add_argument("--stretch", type=int, default=2)
    parser.add_argument("--animation", default=None)
    parser.add_argument(
        "--phase-method",
        choices=["foot_cycle", "joint_configuration_pca"],
        default="foot_cycle",
    )
    args = parser.parse_args()

    report = run_benchmark(
        phase_bins=args.phase_bins,
        train_cycles=args.train_cycles,
        eval_cycles=args.eval_cycles,
        stretch=args.stretch,
        preferred_animation=args.animation,
        phase_method=args.phase_method,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()