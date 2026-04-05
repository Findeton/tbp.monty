#!/usr/bin/env python

"""Track 9b benchmark across multiple real animated Panda3D models.

This benchmark trains one hierarchical ``Panda3DTorchExperiment`` across
multiple real animated meshes and evaluates two things on each model:

- morphology discrimination via the morphology LM
- local motion-phase decoding via the behavior LM

It uses the real animated assets already shipped with the repo:

- Fox.glb
- CesiumMan.glb
- RobotExpressive.glb
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

MODEL_SPECS = {
    "fox": {
        "path": ASSET_DIR / "Fox.glb",
        "object_scale": (0.01, 0.01, 0.01),
        "initial_distance": 3.0,
        "preferred_animation": "Walk",
        "preferred_phase_method": "foot_cycle",
    },
    "cesiumman": {
        "path": ASSET_DIR / "CesiumMan.glb",
        "object_scale": (1.0, 1.0, 1.0),
        "initial_distance": 3.0,
        "preferred_animation": "anim0",
        "preferred_phase_method": "auto",
    },
    "robot": {
        "path": ASSET_DIR / "RobotExpressive.glb",
        "object_scale": (0.3, 0.3, 0.3),
        "initial_distance": 3.0,
        "preferred_animation": "Walking",
        "preferred_phase_method": "auto",
    },
}


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

    walk_names = [
        name for name in names if "walk" in str(name).lower()
    ]
    if walk_names:
        return walk_names[0]

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


def _compute_behavior_metrics(trace, target_lm_id="lm_behavior"):
    return _compute_behavior_metrics_with_decoder(
        trace,
        [None for _ in trace],
        {},
        target_lm_id=target_lm_id,
    )


def _compute_behavior_metrics_with_decoder(
    trace,
    phase_targets,
    latent_decoder,
    target_lm_id="lm_behavior",
):
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
        "next_phase_accuracy": (
            next_correct / next_total if next_total > 0 else None
        ),
        "next_phase_evaluable_steps": next_total,
        "temporal_confident_fraction": (
            confident_count / confident_total if confident_total > 0 else None
        ),
        "temporal_status_steps": confident_total,
    }


def _compute_morphology_metrics(trace, target_label, target_lm_id="lm_morphology"):
    evaluable_steps = 0
    correct_steps = 0
    predicted_labels = []

    for step_trace in trace:
        graph_id = (
            step_trace.get("learning_modules", {})
            .get(target_lm_id, {})
            .get("graph_id")
        )
        if graph_id in (None, "no_observations_yet"):
            continue
        evaluable_steps += 1
        correct_steps += int(graph_id == target_label)
        predicted_labels.append(graph_id)

    return {
        "current_object_accuracy": (
            correct_steps / evaluable_steps if evaluable_steps > 0 else None
        ),
        "evaluable_steps": evaluable_steps,
        "predicted_labels": sorted(set(predicted_labels)),
    }


def _extract_phase_info(animated_object, anim_name, n_phase_bins, preferred_method):
    if preferred_method in {"auto", "foot_cycle"}:
        try:
            return extract_foot_cycle_phase_info(
                animated_object,
                anim_name=anim_name,
                n_phase_bins=n_phase_bins,
            )
        except ValueError:
            if preferred_method == "foot_cycle":
                raise

    return extract_joint_phase_info(
        animated_object,
        anim_name=anim_name,
        n_phase_bins=n_phase_bins,
    )


def _mean(values):
    usable = [value for value in values if value is not None]
    if not usable:
        return None
    return sum(usable) / len(usable)


def run_benchmark(
    models=None,
    phase_bins=4,
    train_cycles=1,
    eval_cycles=1,
    stretch=2,
):
    model_names = list(models or ("fox", "cesiumman", "robot"))
    specs = []
    for name in model_names:
        if name not in MODEL_SPECS:
            raise ValueError(f"Unknown model spec: {name}")
        spec = dict(MODEL_SPECS[name])
        spec["name"] = name
        specs.append(spec)

    missing = [str(spec["path"]) for spec in specs if not spec["path"].is_file()]
    if missing:
        raise FileNotFoundError(f"Missing model assets: {missing}")

    initial_spec = specs[0]
    exp = Panda3DTorchExperiment(
        model_path=initial_spec["path"],
        hierarchical=True,
        resolution=(32, 32),
        far=20.0,
        initial_distance=initial_spec["initial_distance"],
        object_scale=initial_spec["object_scale"],
        flow_threshold=1e-6,
        asset_search_paths=[str(ASSET_DIR)],
        column_kwargs={"n_minicolumns": 512, "sparsity": 0.05},
        behavior_lm_kwargs=_default_self_supervised_behavior_lm_kwargs(),
    )

    try:
        exp._setup()
        # Keep the benchmark focused on shared-memory multi-model learning
        # rather than child-to-child vote arbitration, which can dominate the
        # result before the behavior LM emits stable phase labels.
        exp.monty.lm_to_lm_vote_matrix = [[], [], []]

        training_specs = []
        for spec in specs:
            exp.swap_model(
                spec["path"],
                object_scale=spec["object_scale"],
                initial_distance=spec["initial_distance"],
            )
            anim_name = _select_animation_name(
                exp,
                preferred=spec["preferred_animation"],
            )
            n_frames = exp._anim_obj.get_num_frames(anim_name)
            phase_info = _extract_phase_info(
                exp._anim_obj,
                anim_name=anim_name,
                n_phase_bins=phase_bins,
                preferred_method=spec["preferred_phase_method"],
            )
            train_schedule = _build_schedule(n_frames, cycles=1)
            training_trace = []

            for _ in range(train_cycles):
                result = exp.run_episode(
                    mode=ExperimentMode.TRAIN,
                    object_name=spec["name"],
                    anim_name=anim_name,
                    frame_schedule=train_schedule,
                    collect_trace=True,
                )
                training_trace.extend(result.get("trace", []))

            train_phase_targets = _phase_targets_for_schedule(
                phase_info["phase_by_frame"],
                train_schedule * train_cycles,
            )
            latent_decoder = _fit_latent_phase_decoder(
                training_trace,
                train_phase_targets,
            )

            training_specs.append(
                {
                    **spec,
                    "animation": anim_name,
                    "n_frames": n_frames,
                    "phase_info": phase_info,
                    "latent_decoder": latent_decoder,
                }
            )

        per_model = {}
        morph_lm = exp.monty.learning_modules[0]
        behavior_lm = exp.monty.learning_modules[1]

        for spec in training_specs:
            exp.swap_model(
                spec["path"],
                object_scale=spec["object_scale"],
                initial_distance=spec["initial_distance"],
            )
            matched_schedule = _build_schedule(spec["n_frames"], cycles=eval_cycles)
            stretched_schedule = _build_schedule(
                spec["n_frames"],
                cycles=eval_cycles,
                stretch=stretch,
            )

            matched = exp.run_episode(
                mode=ExperimentMode.EVAL,
                anim_name=spec["animation"],
                frame_schedule=matched_schedule,
                collect_trace=True,
            )
            stretched = exp.run_episode(
                mode=ExperimentMode.EVAL,
                anim_name=spec["animation"],
                frame_schedule=stretched_schedule,
                collect_trace=True,
            )

            known_phase_labels = sorted(
                behavior_lm.get_known_temporal_states()
            )
            matched_phase_targets = _phase_targets_for_schedule(
                spec["phase_info"]["phase_by_frame"],
                matched_schedule,
            )
            stretched_phase_targets = _phase_targets_for_schedule(
                spec["phase_info"]["phase_by_frame"],
                stretched_schedule,
            )

            per_model[spec["name"]] = {
                "asset": str(spec["path"]),
                "animation": spec["animation"],
                "object_scale": list(spec["object_scale"]),
                "initial_distance": spec["initial_distance"],
                "phase_method": spec["phase_info"].get("phase_method"),
                "phase_joint_names": spec["phase_info"]["joint_names"],
                "phase_root_joint": spec["phase_info"]["root_joint_name"],
                "phase_span": spec["phase_info"]["phase_span"],
                "phase_primary_axis": spec["phase_info"].get("primary_axis"),
                "phase_secondary_axis": spec["phase_info"].get("secondary_axis"),
                "discovered_latent_states": known_phase_labels,
                "latent_phase_decoder": spec["latent_decoder"],
                "matched": {
                    "morphology": {
                        **_compute_morphology_metrics(
                            matched.get("trace", []),
                            spec["name"],
                        ),
                        "final_prediction": matched.get("graph_id"),
                        "final_prediction_correct": (
                            matched.get("graph_id") == spec["name"]
                        ),
                    },
                    "behavior": _compute_behavior_metrics_with_decoder(
                        matched.get("trace", []),
                        matched_phase_targets,
                        spec["latent_decoder"],
                    ),
                },
                "stretched": {
                    "morphology": {
                        **_compute_morphology_metrics(
                            stretched.get("trace", []),
                            spec["name"],
                        ),
                        "final_prediction": stretched.get("graph_id"),
                        "final_prediction_correct": (
                            stretched.get("graph_id") == spec["name"]
                        ),
                    },
                    "behavior": _compute_behavior_metrics_with_decoder(
                        stretched.get("trace", []),
                        stretched_phase_targets,
                        spec["latent_decoder"],
                    ),
                },
            }

        aggregate = {
            "n_models": len(per_model),
            "morphology_top1_accuracy_matched": _mean(
                int(report["matched"]["morphology"]["final_prediction_correct"])
                for report in per_model.values()
            ),
            "morphology_top1_accuracy_stretched": _mean(
                int(report["stretched"]["morphology"]["final_prediction_correct"])
                for report in per_model.values()
            ),
            "morphology_step_accuracy_matched_mean": _mean(
                report["matched"]["morphology"]["current_object_accuracy"]
                for report in per_model.values()
            ),
            "morphology_step_accuracy_stretched_mean": _mean(
                report["stretched"]["morphology"]["current_object_accuracy"]
                for report in per_model.values()
            ),
            "behavior_current_phase_accuracy_matched_mean": _mean(
                report["matched"]["behavior"]["current_phase_accuracy"]
                for report in per_model.values()
            ),
            "behavior_current_phase_accuracy_stretched_mean": _mean(
                report["stretched"]["behavior"]["current_phase_accuracy"]
                for report in per_model.values()
            ),
            "behavior_next_phase_accuracy_matched_mean": _mean(
                report["matched"]["behavior"]["next_phase_accuracy"]
                for report in per_model.values()
            ),
            "behavior_next_phase_accuracy_stretched_mean": _mean(
                report["stretched"]["behavior"]["next_phase_accuracy"]
                for report in per_model.values()
            ),
            "behavior_temporal_confident_fraction_matched_mean": _mean(
                report["matched"]["behavior"]["temporal_confident_fraction"]
                for report in per_model.values()
            ),
            "behavior_temporal_confident_fraction_stretched_mean": _mean(
                report["stretched"]["behavior"]["temporal_confident_fraction"]
                for report in per_model.values()
            ),
        }

        return {
            "models": model_names,
            "phase_bins": phase_bins,
            "train_cycles": train_cycles,
            "eval_cycles": eval_cycles,
            "stretch": stretch,
            "morphology_known_objects": sorted(
                morph_lm.get_all_known_object_ids()
            ),
            "behavior_known_objects": sorted(
                behavior_lm.get_all_known_object_ids()
            ),
            "per_model": per_model,
            "aggregate": aggregate,
        }
    finally:
        exp.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="*",
        default=["fox", "cesiumman", "robot"],
        choices=sorted(MODEL_SPECS.keys()),
    )
    parser.add_argument("--phase-bins", type=int, default=4)
    parser.add_argument("--train-cycles", type=int, default=1)
    parser.add_argument("--eval-cycles", type=int, default=1)
    parser.add_argument("--stretch", type=int, default=2)
    args = parser.parse_args()

    report = run_benchmark(
        models=args.models,
        phase_bins=args.phase_bins,
        train_cycles=args.train_cycles,
        eval_cycles=args.eval_cycles,
        stretch=args.stretch,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()