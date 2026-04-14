#!/usr/bin/env python

"""Track 12 Stage 1 falsifier benchmark on real animated assets only.

This benchmark is intentionally constrained to repo-shipped real animated
meshes. It does not use spheres, boxes, bars, or any programmatically created
3D models.

Temporal learning is self-supervised:

- no temporal `state_provider` labels are used during training
- no phase labels are used as a training signal
- motion-grounded phase information is used only post hoc for diagnostics
- semi-Markov latent-state modules are not part of the default inference path

Motor remains closed-loop and LM-controlled through Monty's own policy. The
camera path is not scripted externally as the primary evaluation mode.

It evaluates the narrowed Track 12 temporal theory on real-model schedule
perturbations that are feasible with current Panda3D scaffolding:

- matched replay
- tempo dilation by frame repetition
- tempo compression by frame striding
- omission of a real motion segment
- transient out-of-order perturbation using frames from the same real animation

Primary metrics are predictive-timing oriented:

- temporal surprise around the disturbance
- temporal boundary pressure and boundary-active fraction
- temporal confidence fraction
- recovery latency to confident temporal state

The summary-head temporal event fraction is retained only as a secondary
comparator.

Motion-grounded phase decoding is retained only as a secondary report.

The benchmark can also run the newer Track 14 predictive-hypothesis LM family
through the same hierarchical Panda3D full-Monty path.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from itertools import combinations
import json
import os
import random
from pathlib import Path
import subprocess
import sys


# The Track 14 benchmark showed real cross-run drift under multithreaded CPU
# reductions even with a fixed seed. Default to single-threaded numeric kernels
# so the benchmark is reproducible unless the caller explicitly opts out.
for _thread_var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_var, "1")

import numpy as np
import torch

from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.actions.actions import (
    LookDown,
    LookUp,
    MoveForward,
    MoveTangentially,
    TurnLeft,
    TurnRight,
)
from tbp.monty.frameworks.models.cortical_column_torch.learning_module import (
    CorticalColumnTorchLM,
)
from tbp.monty.frameworks.models.cortical_column_torch.experiment import (
    Panda3DTorchExperiment,
)
from tbp.monty.frameworks.models.predictive_hypothesis_torch.learning_module import (
    PredictiveHypothesisTorchLM,
)
from tbp.monty.frameworks.models.states import State
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
        "preferred_phase_method": "frame_index",
    },
}


# Tuned to the current direct-query Track 12 pressure range so the thresholded
# metric is informative instead of staying pinned at zero.
BOUNDARY_PRESSURE_ACTIVE_THRESHOLD = 0.37
VALID_LM_FAMILIES = (
    "cortical_column_torch",
    "predictive_hypothesis_torch",
)
TRACK14_TRAINING_MAX_ATTEMPTS = 8
_BENCHMARK_THREADING_CONFIGURED = False
BEHAVIOR_OBJECT_DECODER_AMBIGUITY_RATIO = 1.25
BEHAVIOR_OBJECT_DECODER_AMBIGUITY_MIN_SUPPORT = 2


def _configure_benchmark_threading() -> None:
    global _BENCHMARK_THREADING_CONFIGURED

    if _BENCHMARK_THREADING_CONFIGURED:
        return

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch only allows setting interop threads once per process.
        pass
    _BENCHMARK_THREADING_CONFIGURED = True


def _default_predictive_child_lm_kwargs():
    return {
        "vote_top_k": 3,
        "vote_probability_temperature": 1.0,
        "inferred_state_config": {
            "location_correction_weight": 0.35,
            "flow_update_rate": 0.35,
            "use_tracker_location_prior": True,
        },
        "temporal_trace_config": {
            "n_scales": 4,
            "min_decay": 0.2,
            "max_decay": 0.95,
            "trace_weight": 0.35,
            "input_gain": 1.0,
            "boundary_surprise_weight": 1.0,
            "boundary_discontinuity_weight": 1.0,
            "boundary_threshold": 1.0,
        },
        "action_predictive_config": {
            "action_dim": 8,
            "hidden_dim": 128,
            "learning_rate": 0.001,
            "n_settle_iters": 3,
            "query_bias_weight": 0.2,
            "trace_gain": 0.2,
            "boundary_weight": 0.75,
        },
    }


def _default_predictive_hypothesis_lm_kwargs():
    return {
        "output_evidence_threshold": 0.0,
        "core_kwargs": {
            "max_hypotheses": 4,
        },
    }


def _normalize_detail_grid_shape(detail_grid_shape):
    if len(detail_grid_shape) != 2:
        raise ValueError("detail_grid_shape must have exactly two integers")

    rows = max(int(detail_grid_shape[0]), 1)
    cols = max(int(detail_grid_shape[1]), 1)
    return (rows, cols)


def _is_predictive_hypothesis_family(lm_family):
    return str(lm_family) == "predictive_hypothesis_torch"


def _apply_family_lm_defaults(lm_family, lm_kwargs, *, is_parent=False):
    merged = deepcopy(lm_kwargs or {})
    if not _is_predictive_hypothesis_family(lm_family):
        return merged

    defaults = _default_predictive_hypothesis_lm_kwargs()
    merged.setdefault(
        "output_evidence_threshold",
        defaults["output_evidence_threshold"],
    )
    core_kwargs = dict(merged.get("core_kwargs", {}))
    core_kwargs.setdefault(
        "max_hypotheses",
        6 if is_parent else defaults["core_kwargs"]["max_hypotheses"],
    )
    merged["core_kwargs"] = core_kwargs
    return merged


def _apply_family_resolution_defaults(resolution, lm_family):
    rows = max(int(resolution[0]), 1)
    cols = max(int(resolution[1]), 1)
    if not _is_predictive_hypothesis_family(lm_family):
        return (rows, cols)

    # Track 14 depends on center-pixel on-object states plus richer packet fields.
    # The Track 12 smoke resolution is too sparse for small meshes like Fox.glb.
    return (max(rows, 24), max(cols, 24))


def _apply_family_model_geometry_defaults(spec, lm_family):
    adjusted = dict(spec)
    if not _is_predictive_hypothesis_family(lm_family):
        return adjusted

    object_scale = np.asarray(adjusted.get("object_scale", (1.0, 1.0, 1.0)), dtype=np.float32)
    if object_scale.size > 0 and float(np.max(np.abs(object_scale))) <= 0.05:
        adjusted["initial_distance"] = min(
            float(adjusted.get("initial_distance", 3.0)),
            2.0,
        )
    return adjusted


def _learning_modules_missing_target_object(exp, target_name, lm_family):
    if not _is_predictive_hypothesis_family(lm_family):
        return []

    normalized_target = _normalize_graph_id(target_name)
    if normalized_target is None or exp is None:
        return []

    monty = getattr(exp, "monty", None)
    if monty is None:
        return []

    missing_learning_modules = []
    saw_predictive_learning_module = False
    for lm in getattr(monty, "learning_modules", []):
        lm_exposes_target_catalog = False
        graph_id_to_target = getattr(lm, "graph_id_to_target", None)
        mapped_targets = set()
        if isinstance(graph_id_to_target, Mapping):
            lm_exposes_target_catalog = True
            saw_predictive_learning_module = True
            mapped_targets = {
                normalized
                for normalized in (
                    _normalize_graph_id(target_object)
                    for target_objects in graph_id_to_target.values()
                    for target_object in (target_objects or [])
                )
                if normalized is not None
            }
        else:
            get_all_known_object_ids = getattr(lm, "get_all_known_object_ids", None)
            if callable(get_all_known_object_ids):
                lm_exposes_target_catalog = True
                saw_predictive_learning_module = True
                mapped_targets = {
                    normalized
                    for normalized in (
                        _normalize_graph_id(object_id)
                        for object_id in (get_all_known_object_ids() or [])
                    )
                    if normalized is not None
                }

        if not lm_exposes_target_catalog:
            continue
        if normalized_target not in mapped_targets:
            missing_learning_modules.append(
                getattr(lm, "learning_module_id", lm.__class__.__name__)
            )

    if not saw_predictive_learning_module:
        return []
    return missing_learning_modules


def _snapshot_graph_id_target_mappings(exp):
    monty = getattr(exp, "monty", None)
    if monty is None:
        return {}

    per_lm = {}
    for lm in getattr(monty, "learning_modules", []):
        graph_id_to_target = getattr(lm, "graph_id_to_target", None)
        if not graph_id_to_target:
            continue

        lm_mapping = {}
        for raw_graph_id, target_objects in graph_id_to_target.items():
            normalized_graph_id = _normalize_graph_id(raw_graph_id)
            if normalized_graph_id is None:
                continue

            normalized_targets = sorted(
                {
                    normalized_target
                    for normalized_target in (
                        _normalize_graph_id(target_object)
                        for target_object in (target_objects or [])
                    )
                    if normalized_target is not None
                }
            )
            if normalized_targets:
                lm_mapping[normalized_graph_id] = normalized_targets

        if lm_mapping:
            per_lm[str(getattr(lm, "learning_module_id", "unknown_lm"))] = lm_mapping

    return per_lm


def _snapshot_reporting_alias_diagnostics(exp):
    monty = getattr(exp, "monty", None)
    if monty is None:
        return {}

    per_lm = {}
    for lm in getattr(monty, "learning_modules", []):
        getter = getattr(lm, "get_reporting_alias_diagnostics", None)
        if not callable(getter):
            continue
        diagnostics = getter() or {}
        if not isinstance(diagnostics, Mapping):
            continue
        if (
            int(diagnostics.get("total_steps", 0)) <= 0
            and not diagnostics.get("per_target")
        ):
            continue
        per_lm[str(getattr(lm, "learning_module_id", "unknown_lm"))] = deepcopy(
            diagnostics
        )

    return per_lm


def _result_has_processed_learning_step(result):
    for step_trace in result.get("trace", []):
        for lm_trace in (step_trace.get("learning_modules") or {}).values():
            evidence_debug = lm_trace.get("evidence_debug")
            if not isinstance(evidence_debug, dict):
                continue
            if evidence_debug.get("step_skipped"):
                continue
            return True
    return False


def _resolve_child_lm_kwargs(
    morphology_lm_kwargs=None,
    behavior_lm_kwargs=None,
):
    if morphology_lm_kwargs is None and behavior_lm_kwargs is None:
        base = _default_predictive_child_lm_kwargs()
        return deepcopy(base), deepcopy(base)

    if morphology_lm_kwargs is None:
        mirrored = deepcopy(behavior_lm_kwargs or {})
        return deepcopy(mirrored), mirrored

    if behavior_lm_kwargs is None:
        mirrored = deepcopy(morphology_lm_kwargs or {})
        return mirrored, deepcopy(mirrored)

    return deepcopy(morphology_lm_kwargs), deepcopy(behavior_lm_kwargs)


def _track12_child_lm_summary(lm_kwargs):
    lm_kwargs = dict(lm_kwargs or {})
    return {
        "vote_top_k": lm_kwargs.get("vote_top_k"),
        "vote_probability_temperature": float(
            lm_kwargs.get("vote_probability_temperature", 1.0)
        ),
        "inferred_state_enabled": bool(lm_kwargs.get("inferred_state_config")),
        "inferred_state_config": dict(
            lm_kwargs.get("inferred_state_config", {})
        ),
        "temporal_trace_enabled": bool(lm_kwargs.get("temporal_trace_config")),
        "temporal_trace_config": dict(lm_kwargs.get("temporal_trace_config", {})),
        "action_predictive_enabled": bool(
            lm_kwargs.get("action_predictive_config")
        ),
        "action_predictive_config": dict(
            lm_kwargs.get("action_predictive_config", {})
        ),
        "self_supervised_temporal_enabled": (
            lm_kwargs.get("self_supervised_temporal_config") is not None
        ),
        "self_supervised_temporal_evidence_weight": float(
            lm_kwargs.get("self_supervised_temporal_evidence_weight", 0.0)
        ),
        "self_supervised_temporal_transition_weight": float(
            lm_kwargs.get("self_supervised_temporal_transition_weight", 0.0)
        ),
        "self_supervised_temporal_prediction_weight": float(
            lm_kwargs.get("self_supervised_temporal_prediction_weight", 0.0)
        ),
    }


def _build_child_lm_track12_matrix(
    morphology_lm_kwargs,
    behavior_lm_kwargs,
):
    morphology_summary = _track12_child_lm_summary(morphology_lm_kwargs)
    behavior_summary = _track12_child_lm_summary(behavior_lm_kwargs)
    return {
        "lm_morphology": morphology_summary,
        "lm_behavior": behavior_summary,
        "in_parity": morphology_summary == behavior_summary,
    }


def _select_animation_name(exp, preferred=None):
    names = list(exp._anim_obj.animation_names)
    if not names:
        raise RuntimeError("No animations found on the loaded mesh")
    if preferred and preferred in names:
        return preferred

    walk_names = [name for name in names if "walk" in str(name).lower()]
    if walk_names:
        return walk_names[0]

    return names[0]


def _phase_info_is_usable(phase_info, n_phase_bins):
    phase_by_frame = phase_info.get("phase_by_frame") or []
    if not phase_by_frame:
        return False

    unique_bins = len({int(value) for value in phase_by_frame})
    required_unique_bins = 1
    if len(phase_by_frame) > 1 and int(n_phase_bins) > 1:
        required_unique_bins = 2
    if unique_bins < required_unique_bins:
        return False

    phase_span = phase_info.get("phase_span")
    if phase_span is None or not np.isfinite(float(phase_span)):
        return False
    if len(phase_by_frame) > 1 and float(phase_span) <= 1e-6:
        return False

    return True


def _build_frame_index_phase_info(
    animated_object,
    anim_name,
    n_phase_bins,
    phase_method="frame_index_bins_fallback",
):
    n_frames = int(animated_object.get_num_frames(anim_name))
    if n_frames <= 0:
        raise ValueError("Animation must have at least one frame")

    frame_positions = np.arange(n_frames, dtype=np.float32)
    phase_by_frame = (
        np.floor((frame_positions * float(n_phase_bins)) / max(n_frames, 1))
        .astype(int)
        % max(1, int(n_phase_bins))
    )
    phase_angles = (2.0 * np.pi * frame_positions) / max(n_frames, 1)
    return {
        "joint_names": [],
        "root_joint_name": None,
        "phase_by_frame": phase_by_frame.tolist(),
        "phase_angles": phase_angles.tolist(),
        "phase_span": float(phase_angles[-1] - phase_angles[0]) if n_frames > 1 else 0.0,
        "phase_method": phase_method,
    }


def _extract_phase_info(animated_object, anim_name, n_phase_bins, preferred_method):
    if preferred_method == "frame_index":
        return _build_frame_index_phase_info(
            animated_object,
            anim_name,
            n_phase_bins,
            phase_method="frame_index_bins_requested",
        )

    if preferred_method in {"auto", "foot_cycle"}:
        try:
            phase_info = extract_foot_cycle_phase_info(
                animated_object,
                anim_name=anim_name,
                n_phase_bins=n_phase_bins,
            )
        except (AssertionError, ValueError):
            if preferred_method == "foot_cycle":
                # Panda3D occasionally reports invalid joint transforms during
                # foot-cycle extraction. Fall back to generic joint phase
                # extraction so the benchmark can still run on real assets.
                preferred_method = "auto"
        else:
            if _phase_info_is_usable(phase_info, n_phase_bins):
                return phase_info
            preferred_method = "auto"

    try:
        phase_info = extract_joint_phase_info(
            animated_object,
            anim_name=anim_name,
            n_phase_bins=n_phase_bins,
        )
    except (AssertionError, ValueError):
        return _build_frame_index_phase_info(animated_object, anim_name, n_phase_bins)

    if _phase_info_is_usable(phase_info, n_phase_bins):
        return phase_info
    return _build_frame_index_phase_info(animated_object, anim_name, n_phase_bins)


def _build_schedule(n_frames, cycles, stretch=1):
    schedule = []
    for _ in range(cycles):
        for frame in range(n_frames):
            schedule.extend([frame] * max(1, int(stretch)))
    return schedule


def _build_strided_schedule(n_frames, cycles, stride=2):
    stride = max(1, int(stride))
    base_frames = list(range(0, n_frames, stride))
    if base_frames and base_frames[-1] != n_frames - 1:
        base_frames.append(n_frames - 1)

    schedule = []
    for _ in range(cycles):
        schedule.extend(base_frames)
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


def _mean(values):
    usable = [value for value in values if value is not None]
    if not usable:
        return None
    return sum(usable) / len(usable)


def _prediction_is_resolved(prediction):
    if not prediction:
        return False
    if "final_prediction_resolved" in prediction:
        return bool(prediction.get("final_prediction_resolved"))
    return prediction.get("final_prediction_correct") is not None


def _prediction_accuracy_summary(predictions):
    predictions = list(predictions)
    return {
        "accuracy_mean": _mean(
            prediction.get("final_prediction_correct")
            for prediction in predictions
        ),
        "resolved_fraction_mean": _mean(
            int(_prediction_is_resolved(prediction))
            for prediction in predictions
        ),
        "strict_accuracy_mean": _mean(
            int(bool(prediction.get("final_prediction_correct")))
            for prediction in predictions
        ),
    }


def _extract_boundary_pressure(lm_trace):
    temporal_context = lm_trace.get("temporal_context") or {}
    value = temporal_context.get("boundary_pressure")
    if value is None:
        return None
    return float(value)


def _find_phase_segment(n_frames, n_phase_bins, phase_by_frame=None):
    n_frames = int(n_frames)
    if n_frames <= 0:
        raise ValueError("Cannot choose omission segment from empty animation")

    segment_length = max(3, int(round(n_frames / max(1, int(n_phase_bins)))))
    segment_length = min(segment_length, n_frames)
    start = max(0, (n_frames - segment_length) // 2)
    end = min(n_frames, start + segment_length)
    phase = None
    phases = list(int(value) for value in (phase_by_frame or []))
    if phases:
        center_idx = min(len(phases) - 1, max(0, (start + end - 1) // 2))
        phase = phases[center_idx]

    return {
        "phase": phase,
        "start": start,
        "end": end,
        "length": end - start,
        "selection_mode": "deterministic_center_window",
    }


def _build_omission_schedule(n_frames, cycles, segment):
    schedule = []
    windows = []
    omitted_frames = list(range(segment["start"], segment["end"]))

    for cycle in range(cycles):
        condition_cycle_start = len(schedule)
        baseline_cycle_start = cycle * n_frames
        cycle_frames = [
            frame for frame in range(n_frames) if frame not in omitted_frames
        ]
        schedule.extend(cycle_frames)
        windows.append(
            {
                "cycle": cycle,
                "phase": segment["phase"],
                "omitted_frames": omitted_frames,
                "condition_start": condition_cycle_start + segment["start"],
                "baseline_start": baseline_cycle_start + segment["start"],
                "length": max(3, segment["length"]),
                "recovery_anchor": condition_cycle_start + segment["start"],
                "baseline_recovery_anchor": baseline_cycle_start + segment["end"],
            }
        )

    return schedule, windows


def _build_perturbed_schedule(n_frames, cycles, segment):
    schedule = []
    windows = []
    shift = max(1, n_frames // 2)
    replacement = [
        (frame + shift) % n_frames
        for frame in range(segment["start"], segment["end"])
    ]
    if replacement == list(range(segment["start"], segment["end"])):
        replacement = [
            (frame + max(1, n_frames // 3)) % n_frames
            for frame in range(segment["start"], segment["end"])
        ]

    for cycle in range(cycles):
        cycle_frames = list(range(n_frames))
        cycle_frames[segment["start"]:segment["end"]] = replacement
        cycle_start = len(schedule)
        schedule.extend(cycle_frames)
        windows.append(
            {
                "cycle": cycle,
                "replaced_frames": replacement,
                "condition_start": cycle_start + segment["start"],
                "baseline_start": cycle * n_frames + segment["start"],
                "length": max(3, segment["length"]),
                "recovery_anchor": cycle_start + segment["end"],
                "baseline_recovery_anchor": cycle * n_frames + segment["end"],
            }
        )

    return schedule, windows


def _compute_primary_temporal_metrics(
    trace,
    target_lm_id="lm_behavior",
    boundary_active_threshold=BOUNDARY_PRESSURE_ACTIVE_THRESHOLD,
):
    surprises = []
    statuses = []
    boundary_pressures = []
    event_count = 0

    for step_trace in trace:
        lm_trace = step_trace.get("learning_modules", {}).get(target_lm_id, {})
        if "temporal_surprise" in lm_trace:
            surprises.append(float(lm_trace["temporal_surprise"]))

        status = lm_trace.get("temporal_status")
        if status in {"confident", "confused"}:
            statuses.append(status)

        boundary_pressure = _extract_boundary_pressure(lm_trace)
        if boundary_pressure is not None:
            boundary_pressures.append(boundary_pressure)

        temporal_context = lm_trace.get("temporal_context") or {}
        event_count += int(bool(temporal_context.get("event_detected", False)))

    return {
        "trace_steps": len(trace),
        "temporal_surprise_mean": _mean(surprises),
        "temporal_surprise_max": max(surprises) if surprises else None,
        "temporal_boundary_pressure_mean": _mean(boundary_pressures),
        "temporal_boundary_pressure_max": (
            max(boundary_pressures) if boundary_pressures else None
        ),
        "temporal_boundary_active_fraction": (
            sum(
                value >= boundary_active_threshold
                for value in boundary_pressures
            )
            / len(boundary_pressures)
            if boundary_pressures
            else None
        ),
        "temporal_boundary_pressure_steps": len(boundary_pressures),
        "temporal_event_fraction": (
            event_count / len(trace) if trace else None
        ),
        "temporal_confident_fraction": (
            sum(status == "confident" for status in statuses) / len(statuses)
            if statuses
            else None
        ),
        "temporal_status_steps": len(statuses),
    }


def _window_stats(
    trace,
    start_step,
    length,
    target_lm_id="lm_behavior",
    boundary_active_threshold=BOUNDARY_PRESSURE_ACTIVE_THRESHOLD,
):
    start_step = max(0, int(start_step))
    end_step = min(len(trace), start_step + max(1, int(length)))
    if end_step <= start_step:
        return {
            "steps": 0,
            "mean_surprise": None,
            "max_surprise": None,
            "mean_boundary_pressure": None,
            "max_boundary_pressure": None,
            "boundary_active_fraction": None,
            "boundary_pressure_steps": 0,
            "event_fraction": None,
            "confident_fraction": None,
        }

    surprises = []
    statuses = []
    boundary_pressures = []
    event_count = 0

    for step_trace in trace[start_step:end_step]:
        lm_trace = step_trace.get("learning_modules", {}).get(target_lm_id, {})
        if "temporal_surprise" in lm_trace:
            surprises.append(float(lm_trace["temporal_surprise"]))
        status = lm_trace.get("temporal_status")
        if status in {"confident", "confused"}:
            statuses.append(status)
        boundary_pressure = _extract_boundary_pressure(lm_trace)
        if boundary_pressure is not None:
            boundary_pressures.append(boundary_pressure)
        temporal_context = lm_trace.get("temporal_context") or {}
        event_count += int(bool(temporal_context.get("event_detected", False)))

    return {
        "steps": end_step - start_step,
        "mean_surprise": _mean(surprises),
        "max_surprise": max(surprises) if surprises else None,
        "mean_boundary_pressure": _mean(boundary_pressures),
        "max_boundary_pressure": (
            max(boundary_pressures) if boundary_pressures else None
        ),
        "boundary_active_fraction": (
            sum(
                value >= boundary_active_threshold
                for value in boundary_pressures
            )
            / len(boundary_pressures)
            if boundary_pressures
            else None
        ),
        "boundary_pressure_steps": len(boundary_pressures),
        "event_fraction": event_count / (end_step - start_step),
        "confident_fraction": (
            sum(status == "confident" for status in statuses) / len(statuses)
            if statuses
            else None
        ),
    }


def _recovery_steps_to_confident(
    trace,
    start_step,
    required_consecutive=3,
    target_lm_id="lm_behavior",
):
    statuses = []
    for step_trace in trace:
        lm_trace = step_trace.get("learning_modules", {}).get(target_lm_id, {})
        statuses.append(lm_trace.get("temporal_status"))

    start_step = max(0, int(start_step))
    required_consecutive = max(1, int(required_consecutive))

    for idx in range(start_step, len(statuses) - required_consecutive + 1):
        if all(
            statuses[idx + offset] == "confident"
            for offset in range(required_consecutive)
        ):
            return idx - start_step

    return None


def _compute_windowed_deltas(
    condition_trace,
    baseline_trace,
    windows,
    target_lm_id="lm_behavior",
    boundary_active_threshold=BOUNDARY_PRESSURE_ACTIVE_THRESHOLD,
):
    per_window = []

    for window in windows:
        pre_condition_stats = _window_stats(
            condition_trace,
            window["condition_start"] - window["length"],
            window["length"],
            target_lm_id=target_lm_id,
            boundary_active_threshold=boundary_active_threshold,
        )
        pre_baseline_stats = _window_stats(
            baseline_trace,
            window["baseline_start"] - window["length"],
            window["length"],
            target_lm_id=target_lm_id,
            boundary_active_threshold=boundary_active_threshold,
        )
        condition_stats = _window_stats(
            condition_trace,
            window["condition_start"],
            window["length"],
            target_lm_id=target_lm_id,
            boundary_active_threshold=boundary_active_threshold,
        )
        baseline_stats = _window_stats(
            baseline_trace,
            window["baseline_start"],
            window["length"],
            target_lm_id=target_lm_id,
            boundary_active_threshold=boundary_active_threshold,
        )
        recovery_condition_stats = _window_stats(
            condition_trace,
            window["recovery_anchor"],
            window["length"],
            target_lm_id=target_lm_id,
            boundary_active_threshold=boundary_active_threshold,
        )
        recovery_baseline_stats = _window_stats(
            baseline_trace,
            window.get("baseline_recovery_anchor", window["baseline_start"]),
            window["length"],
            target_lm_id=target_lm_id,
            boundary_active_threshold=boundary_active_threshold,
        )
        recovery_steps = _recovery_steps_to_confident(
            condition_trace,
            window["recovery_anchor"],
            target_lm_id=target_lm_id,
        )

        per_window.append(
            {
                **window,
                "pre_condition": pre_condition_stats,
                "pre_baseline": pre_baseline_stats,
                "condition": condition_stats,
                "matched_baseline": baseline_stats,
                "recovery_condition": recovery_condition_stats,
                "recovery_baseline": recovery_baseline_stats,
                "pre_surprise_delta": (
                    None
                    if pre_condition_stats["mean_surprise"] is None
                    or pre_baseline_stats["mean_surprise"] is None
                    else pre_condition_stats["mean_surprise"]
                    - pre_baseline_stats["mean_surprise"]
                ),
                "surprise_delta": (
                    None
                    if condition_stats["mean_surprise"] is None
                    or baseline_stats["mean_surprise"] is None
                    else condition_stats["mean_surprise"]
                    - baseline_stats["mean_surprise"]
                ),
                "recovery_surprise_delta": (
                    None
                    if recovery_condition_stats["mean_surprise"] is None
                    or recovery_baseline_stats["mean_surprise"] is None
                    else recovery_condition_stats["mean_surprise"]
                    - recovery_baseline_stats["mean_surprise"]
                ),
                "pre_boundary_pressure_delta": (
                    None
                    if pre_condition_stats["mean_boundary_pressure"] is None
                    or pre_baseline_stats["mean_boundary_pressure"] is None
                    else pre_condition_stats["mean_boundary_pressure"]
                    - pre_baseline_stats["mean_boundary_pressure"]
                ),
                "boundary_pressure_delta": (
                    None
                    if condition_stats["mean_boundary_pressure"] is None
                    or baseline_stats["mean_boundary_pressure"] is None
                    else condition_stats["mean_boundary_pressure"]
                    - baseline_stats["mean_boundary_pressure"]
                ),
                "recovery_boundary_pressure_delta": (
                    None
                    if recovery_condition_stats["mean_boundary_pressure"] is None
                    or recovery_baseline_stats["mean_boundary_pressure"] is None
                    else recovery_condition_stats["mean_boundary_pressure"]
                    - recovery_baseline_stats["mean_boundary_pressure"]
                ),
                "pre_boundary_active_fraction_delta": (
                    None
                    if pre_condition_stats["boundary_active_fraction"] is None
                    or pre_baseline_stats["boundary_active_fraction"] is None
                    else pre_condition_stats["boundary_active_fraction"]
                    - pre_baseline_stats["boundary_active_fraction"]
                ),
                "boundary_active_fraction_delta": (
                    None
                    if condition_stats["boundary_active_fraction"] is None
                    or baseline_stats["boundary_active_fraction"] is None
                    else condition_stats["boundary_active_fraction"]
                    - baseline_stats["boundary_active_fraction"]
                ),
                "recovery_boundary_active_fraction_delta": (
                    None
                    if recovery_condition_stats["boundary_active_fraction"] is None
                    or recovery_baseline_stats["boundary_active_fraction"] is None
                    else recovery_condition_stats["boundary_active_fraction"]
                    - recovery_baseline_stats["boundary_active_fraction"]
                ),
                "pre_event_fraction_delta": (
                    None
                    if pre_condition_stats["event_fraction"] is None
                    or pre_baseline_stats["event_fraction"] is None
                    else pre_condition_stats["event_fraction"]
                    - pre_baseline_stats["event_fraction"]
                ),
                "event_fraction_delta": (
                    None
                    if condition_stats["event_fraction"] is None
                    or baseline_stats["event_fraction"] is None
                    else condition_stats["event_fraction"]
                    - baseline_stats["event_fraction"]
                ),
                "recovery_event_fraction_delta": (
                    None
                    if recovery_condition_stats["event_fraction"] is None
                    or recovery_baseline_stats["event_fraction"] is None
                    else recovery_condition_stats["event_fraction"]
                    - recovery_baseline_stats["event_fraction"]
                ),
                "recovery_steps_to_confident": recovery_steps,
            }
        )

    return {
        "per_window": per_window,
        "mean_pre_surprise_delta": _mean(
            window["pre_surprise_delta"] for window in per_window
        ),
        "mean_surprise_delta": _mean(
            window["surprise_delta"] for window in per_window
        ),
        "mean_recovery_surprise_delta": _mean(
            window["recovery_surprise_delta"] for window in per_window
        ),
        "mean_pre_boundary_pressure_delta": _mean(
            window["pre_boundary_pressure_delta"] for window in per_window
        ),
        "mean_boundary_pressure_delta": _mean(
            window["boundary_pressure_delta"] for window in per_window
        ),
        "mean_recovery_boundary_pressure_delta": _mean(
            window["recovery_boundary_pressure_delta"] for window in per_window
        ),
        "mean_pre_boundary_active_fraction_delta": _mean(
            window["pre_boundary_active_fraction_delta"] for window in per_window
        ),
        "mean_boundary_active_fraction_delta": _mean(
            window["boundary_active_fraction_delta"] for window in per_window
        ),
        "mean_recovery_boundary_active_fraction_delta": _mean(
            window["recovery_boundary_active_fraction_delta"]
            for window in per_window
        ),
        "mean_pre_event_fraction_delta": _mean(
            window["pre_event_fraction_delta"] for window in per_window
        ),
        "mean_event_fraction_delta": _mean(
            window["event_fraction_delta"] for window in per_window
        ),
        "mean_recovery_event_fraction_delta": _mean(
            window["recovery_event_fraction_delta"] for window in per_window
        ),
        "mean_recovery_steps_to_confident": _mean(
            window["recovery_steps_to_confident"] for window in per_window
        ),
    }


def _condition_report(
    result,
    phase_targets,
    latent_decoder,
    target_name,
    target_lm_id="lm_behavior",
    boundary_active_threshold=BOUNDARY_PRESSURE_ACTIVE_THRESHOLD,
    object_decoders=None,
    object_decoder_diagnostics=None,
    joint_object_decoder=None,
    joint_object_decoder_diagnostics=None,
    joint_object_decoder_lm_ids=("lm_morphology", "lm_behavior"),
):
    trace = result.get("trace", [])
    lm_object_predictions = _collect_lm_object_predictions(
        result,
        target_name,
        object_decoders=object_decoders,
        object_decoder_diagnostics=object_decoder_diagnostics,
    )
    parent_prediction_info = lm_object_predictions.get("lm_parent", {})
    final_prediction_info = dict(
        lm_object_predictions.get(
            target_lm_id,
            {
                "final_prediction_latent_id": None,
                "final_prediction_graph_id": None,
                "final_prediction": None,
                "final_prediction_correct": False,
                "final_prediction_decoded": False,
                "has_latent_id": False,
                "has_graph_id": False,
            },
        )
    )
    if not final_prediction_info.get("has_latent_id"):
        raw_graph_id, has_identity = _extract_raw_identity(result)
        decoded_object = _decode_object_name(
            raw_graph_id,
            (object_decoders or {}).get(target_lm_id),
        )
        final_prediction_info = {
            "final_prediction_latent_id": raw_graph_id,
            "final_prediction_graph_id": raw_graph_id,
            "final_prediction": (
                decoded_object if decoded_object is not None else raw_graph_id
            ),
            "final_prediction_correct": (
                (decoded_object if decoded_object is not None else raw_graph_id)
                == target_name
            ),
            "final_prediction_decoded": (
                (object_decoders or {}).get(target_lm_id) is not None
                and _normalize_graph_id(raw_graph_id) is not None
                and decoded_object is not None
            ),
            "final_prediction_ambiguous": False,
            "final_prediction_candidate_objects": [],
            "final_prediction_resolved": raw_graph_id is not None,
            "has_latent_id": has_identity,
            "has_graph_id": has_identity,
        }
        final_prediction_info = _annotate_decoder_prediction(
            final_prediction_info,
            target_name,
            (object_decoder_diagnostics or {}).get(target_lm_id),
        )
    final_prediction_info = _resolve_prediction_info_with_ambiguity_fallback(
        result,
        final_prediction_info,
        prediction_lm_id=target_lm_id,
        target_name=target_name,
        object_decoder_diagnostics=object_decoder_diagnostics,
        joint_object_decoder=joint_object_decoder,
        joint_object_decoder_diagnostics=joint_object_decoder_diagnostics,
        joint_object_decoder_lm_ids=joint_object_decoder_lm_ids,
        parent_prediction_info=parent_prediction_info,
        allow_parent_fallback=True,
    )
    final_prediction_lm_id = final_prediction_info.get("final_prediction_lm_id")

    child_lm_primary = {}
    for lm_id in ("lm_morphology", "lm_behavior"):
        child_prediction_info = dict(
            lm_object_predictions.get(
                lm_id,
                {
                    "final_prediction_latent_id": None,
                    "final_prediction_graph_id": None,
                    "final_prediction": None,
                    "final_prediction_correct": False,
                    "final_prediction_decoded": False,
                    "has_latent_id": False,
                    "has_graph_id": False,
                },
            )
        )
        if lm_id == "lm_behavior":
            child_prediction_info = _resolve_prediction_info_with_ambiguity_fallback(
                result,
                child_prediction_info,
                prediction_lm_id=lm_id,
                target_name=target_name,
                object_decoder_diagnostics=object_decoder_diagnostics,
                joint_object_decoder=joint_object_decoder,
                joint_object_decoder_diagnostics=joint_object_decoder_diagnostics,
                joint_object_decoder_lm_ids=joint_object_decoder_lm_ids,
                parent_prediction_info=parent_prediction_info,
                allow_parent_fallback=False,
            )
        child_lm_primary[lm_id] = {
            **_compute_primary_temporal_metrics(
                trace,
                target_lm_id=lm_id,
                boundary_active_threshold=boundary_active_threshold,
            ),
            **child_prediction_info,
        }

    return {
        "primary": {
            **_compute_primary_temporal_metrics(
                trace,
                target_lm_id=target_lm_id,
                boundary_active_threshold=boundary_active_threshold,
            ),
            **final_prediction_info,
            "final_prediction_lm_id": final_prediction_lm_id,
        },
        "child_lm_primary": child_lm_primary,
        "lm_object_predictions": lm_object_predictions,
        "secondary": _compute_behavior_metrics_with_decoder(
            trace,
            phase_targets,
            latent_decoder,
            target_lm_id=target_lm_id,
        ),
    }


def _compute_primary_metric_deltas(condition_primary, baseline_primary):
    keys = (
        "temporal_surprise_mean",
        "temporal_boundary_pressure_mean",
        "temporal_boundary_active_fraction",
        "temporal_confident_fraction",
        "temporal_event_fraction",
    )
    deltas = {}
    for key in keys:
        condition_value = condition_primary.get(key)
        baseline_value = baseline_primary.get(key)
        delta_key = key.replace("temporal_", "") + "_delta"
        if condition_value is None or baseline_value is None:
            deltas[delta_key] = None
        else:
            deltas[delta_key] = condition_value - baseline_value

    deltas["final_prediction_changed"] = (
        condition_primary.get("final_prediction")
        != baseline_primary.get("final_prediction")
    )
    return deltas


def _compute_action_history_stats(action_history, action_context_history):
    total_steps = max(len(action_history), len(action_context_history))
    if total_steps <= 0:
        return {
            "steps": 0,
            "nonempty_action_fraction": None,
            "nonzero_action_context_fraction": None,
            "mean_action_context_norm": None,
        }

    nonempty_action_steps = sum(1 for actions in action_history if actions)
    context_norms = [
        float(np.abs(np.asarray(ctx, dtype=np.float32)).sum())
        for ctx in action_context_history
    ]
    nonzero_context_steps = sum(norm > 1e-8 for norm in context_norms)

    return {
        "steps": total_steps,
        "nonempty_action_fraction": nonempty_action_steps / total_steps,
        "nonzero_action_context_fraction": nonzero_context_steps / total_steps,
        "mean_action_context_norm": _mean(context_norms),
    }


def _compute_temporal_trace_disagreement(
    condition_trace,
    baseline_trace,
    target_lm_id="lm_behavior",
):
    step_count = min(len(condition_trace), len(baseline_trace))
    if step_count <= 0:
        return {
            "comparable_steps": 0,
            "current_label_disagreement_fraction": None,
            "predicted_label_disagreement_fraction": None,
            "temporal_status_disagreement_fraction": None,
        }

    current_total = 0
    current_disagreements = 0
    predicted_total = 0
    predicted_disagreements = 0
    status_total = 0
    status_disagreements = 0

    for idx in range(step_count):
        condition_context = (
            condition_trace[idx].get("learning_modules", {})
            .get(target_lm_id, {})
            .get("temporal_context", {})
        )
        baseline_context = (
            baseline_trace[idx].get("learning_modules", {})
            .get(target_lm_id, {})
            .get("temporal_context", {})
        )

        condition_current = condition_context.get("current_label")
        baseline_current = baseline_context.get("current_label")
        if condition_current is not None or baseline_current is not None:
            current_total += 1
            current_disagreements += int(condition_current != baseline_current)

        condition_predicted = condition_context.get("predicted_label")
        baseline_predicted = baseline_context.get("predicted_label")
        if condition_predicted is not None or baseline_predicted is not None:
            predicted_total += 1
            predicted_disagreements += int(
                condition_predicted != baseline_predicted
            )

        condition_status = (
            condition_trace[idx].get("learning_modules", {})
            .get(target_lm_id, {})
            .get("temporal_status")
        )
        baseline_status = (
            baseline_trace[idx].get("learning_modules", {})
            .get(target_lm_id, {})
            .get("temporal_status")
        )
        if condition_status is not None or baseline_status is not None:
            status_total += 1
            status_disagreements += int(condition_status != baseline_status)

    return {
        "comparable_steps": step_count,
        "current_label_disagreement_fraction": (
            current_disagreements / current_total if current_total > 0 else None
        ),
        "predicted_label_disagreement_fraction": (
            predicted_disagreements / predicted_total
            if predicted_total > 0
            else None
        ),
        "temporal_status_disagreement_fraction": (
            status_disagreements / status_total if status_total > 0 else None
        ),
    }


def _normalize_graph_id(graph_id):
    if graph_id is None:
        return None

    normalized = str(graph_id)
    if not normalized or normalized == "no_observations_yet":
        return None
    return normalized


def _extract_raw_identity(payload):
    if not isinstance(payload, Mapping):
        return None, False

    has_identity = ("latent_id" in payload) or ("graph_id" in payload)
    raw_identity = payload.get("latent_id", payload.get("graph_id"))
    return None if raw_identity is None else str(raw_identity), has_identity


def _decode_object_name(identity_id, object_decoder=None):
    raw_identity = _normalize_graph_id(identity_id)
    if raw_identity is None:
        return None

    if object_decoder is None:
        return raw_identity
    return object_decoder.get(raw_identity)


def _extract_lm_object_prediction(result, lm_id, object_decoder=None):
    has_identity = False
    raw_identity = None
    lm_result = result.get(lm_id)
    if isinstance(lm_result, Mapping):
        raw_identity, has_identity = _extract_raw_identity(lm_result)

    decoded_object = _decode_object_name(raw_identity, object_decoder)
    final_prediction = decoded_object if decoded_object is not None else raw_identity
    return {
        "final_prediction_latent_id": raw_identity,
        "final_prediction_graph_id": raw_identity,
        "final_prediction": final_prediction,
        "final_prediction_decoded": (
            object_decoder is not None
            and _normalize_graph_id(raw_identity) is not None
            and decoded_object is not None
        ),
        "final_prediction_ambiguous": False,
        "final_prediction_candidate_objects": [],
        "final_prediction_resolved": (
            has_identity and final_prediction is not None
        ),
        "has_latent_id": has_identity,
        "has_graph_id": has_identity,
    }


def _finalize_prediction_info(
    prediction_info,
    target_name,
    *,
    ambiguous=False,
    candidate_objects=None,
):
    resolved_prediction_info = dict(prediction_info or {})
    raw_identity = resolved_prediction_info.get(
        "final_prediction_latent_id",
        resolved_prediction_info.get("final_prediction_graph_id"),
    )
    raw_identity = None if raw_identity is None else str(raw_identity)
    resolved_prediction_info["final_prediction_latent_id"] = raw_identity
    resolved_prediction_info["final_prediction_graph_id"] = raw_identity
    has_identity = bool(
        resolved_prediction_info.get(
            "has_latent_id",
            resolved_prediction_info.get("has_graph_id", False),
        )
    )
    resolved_prediction_info["has_latent_id"] = has_identity
    resolved_prediction_info["has_graph_id"] = has_identity
    signature_ids = resolved_prediction_info.get(
        "final_prediction_signature_latent_ids",
        resolved_prediction_info.get("final_prediction_signature_graph_ids"),
    )
    if signature_ids is not None:
        signature_ids = {
            str(key): None if value is None else str(value)
            for key, value in dict(signature_ids).items()
        }
        resolved_prediction_info["final_prediction_signature_latent_ids"] = dict(
            signature_ids
        )
        resolved_prediction_info["final_prediction_signature_graph_ids"] = dict(
            signature_ids
        )
    resolved_prediction_info["final_prediction_ambiguous"] = bool(ambiguous)
    resolved_prediction_info["final_prediction_candidate_objects"] = [
        str(candidate)
        for candidate in (candidate_objects or [])
        if candidate is not None
    ]

    if ambiguous:
        resolved_prediction_info["final_prediction"] = None
        resolved_prediction_info["final_prediction_correct"] = None
        resolved_prediction_info["final_prediction_decoded"] = False
    else:
        resolved_prediction_info["final_prediction_correct"] = (
            resolved_prediction_info.get("final_prediction") == target_name
        )

    resolved_prediction_info["final_prediction_resolved"] = bool(
        resolved_prediction_info.get("has_latent_id")
        and resolved_prediction_info.get("final_prediction") is not None
        and not resolved_prediction_info["final_prediction_ambiguous"]
    )
    return resolved_prediction_info


def _annotate_decoder_prediction(
    prediction_info,
    target_name,
    object_decoder_diagnostics=None,
):
    candidates = _decoder_graph_id_candidates(
        prediction_info.get(
            "final_prediction_latent_id",
            prediction_info.get("final_prediction_graph_id"),
        ),
        object_decoder_diagnostics,
    )
    return _finalize_prediction_info(
        prediction_info,
        target_name,
        ambiguous=len(candidates) > 1,
        candidate_objects=candidates,
    )


def _collect_lm_object_predictions(
    result,
    target_name,
    object_decoders=None,
    object_decoder_diagnostics=None,
    lm_ids=("lm_morphology", "lm_behavior", "lm_parent"),
):
    predictions = {}
    object_decoders = dict(object_decoders or {})
    object_decoder_diagnostics = dict(object_decoder_diagnostics or {})

    for lm_id in lm_ids:
        prediction = _extract_lm_object_prediction(
            result,
            lm_id,
            object_decoder=object_decoders.get(lm_id),
        )
        prediction = _annotate_decoder_prediction(
            prediction,
            target_name,
            object_decoder_diagnostics.get(lm_id),
        )
        predictions[lm_id] = prediction

    return predictions


def _resolve_prediction_info_with_ambiguity_fallback(
    result,
    prediction_info,
    *,
    prediction_lm_id,
    target_name,
    object_decoder_diagnostics=None,
    joint_object_decoder=None,
    joint_object_decoder_diagnostics=None,
    joint_object_decoder_lm_ids=("lm_morphology", "lm_behavior"),
    parent_prediction_info=None,
    allow_parent_fallback=True,
):
    resolved_prediction_info = dict(prediction_info or {})
    prediction_diagnostics = (object_decoder_diagnostics or {}).get(prediction_lm_id)
    if (
        "final_prediction_ambiguous" not in resolved_prediction_info
        or "final_prediction_resolved" not in resolved_prediction_info
    ):
        resolved_prediction_info = _annotate_decoder_prediction(
            resolved_prediction_info,
            target_name,
            prediction_diagnostics,
        )
    resolved_prediction_lm_id = (
        prediction_lm_id if resolved_prediction_info.get("has_latent_id") else None
    )
    prediction_ambiguous = bool(
        resolved_prediction_info.get("final_prediction_ambiguous")
    )
    parent_prediction_info = dict(parent_prediction_info or {})
    if parent_prediction_info and (
        "final_prediction_ambiguous" not in parent_prediction_info
        or "final_prediction_resolved" not in parent_prediction_info
    ):
        parent_prediction_info = _annotate_decoder_prediction(
            parent_prediction_info,
            target_name,
            (object_decoder_diagnostics or {}).get("lm_parent"),
        )
    parent_prediction_unambiguous = (
        parent_prediction_info.get("has_latent_id")
        and not parent_prediction_info.get("final_prediction_ambiguous", False)
        and parent_prediction_info.get("final_prediction") is not None
    )
    parent_prediction_object = (
        parent_prediction_info.get("final_prediction")
        if parent_prediction_unambiguous
        else None
    )

    if (
        joint_object_decoder
        and prediction_lm_id in set(joint_object_decoder_lm_ids)
    ):
        joint_signature_key, joint_signature_parts = _extract_joint_object_signature_key(
            result,
            lm_ids=joint_object_decoder_lm_ids,
        )
        joint_prediction = None
        joint_prediction_ambiguous = False
        if joint_signature_key is not None:
            joint_prediction = joint_object_decoder.get(joint_signature_key)
            joint_prediction_ambiguous = _is_joint_object_signature_ambiguous(
                joint_signature_key,
                joint_object_decoder_diagnostics,
            )
        if joint_prediction is not None and not joint_prediction_ambiguous and (
            prediction_ambiguous
            or (
                joint_prediction != resolved_prediction_info.get("final_prediction")
                and parent_prediction_object == joint_prediction
            )
        ):
            resolved_prediction_info = _finalize_prediction_info(
                {
                    "final_prediction_latent_id": joint_signature_key,
                    "final_prediction_graph_id": joint_signature_key,
                    "final_prediction": joint_prediction,
                    "final_prediction_decoded": True,
                    "has_latent_id": True,
                    "has_graph_id": True,
                    "final_prediction_signature_latent_ids": dict(
                        joint_signature_parts or {}
                    ),
                    "final_prediction_signature_graph_ids": dict(
                        joint_signature_parts or {}
                    ),
                },
                target_name,
                ambiguous=False,
                candidate_objects=[joint_prediction],
            )
            resolved_prediction_lm_id = "lm_child_joint_decoder"

    if (
        allow_parent_fallback
        and prediction_ambiguous
        and resolved_prediction_lm_id != "lm_child_joint_decoder"
    ):
        if parent_prediction_unambiguous:
            resolved_prediction_info = parent_prediction_info
            resolved_prediction_lm_id = "lm_parent_decoder_fallback"

    resolved_prediction_info = _finalize_prediction_info(
        resolved_prediction_info,
        target_name,
        ambiguous=bool(resolved_prediction_info.get("final_prediction_ambiguous")),
        candidate_objects=resolved_prediction_info.get(
            "final_prediction_candidate_objects",
            [],
        ),
    )
    resolved_prediction_info["final_prediction_lm_id"] = resolved_prediction_lm_id
    return resolved_prediction_info


def _extract_joint_object_signature_key_from_graph_ids(graph_ids_by_lm, lm_ids):
    lm_ids = tuple(lm_ids)
    signature_parts = {}
    for lm_id in lm_ids:
        raw_graph_id = _normalize_graph_id(graph_ids_by_lm.get(lm_id))
        if raw_graph_id is None:
            return None, None
        signature_parts[str(lm_id)] = raw_graph_id

    signature_key = "|".join(
        f"{lm_id}={signature_parts[str(lm_id)]}"
        for lm_id in lm_ids
    )
    return signature_key, signature_parts


def _extract_joint_object_signature_key(
    result,
    lm_ids=("lm_morphology", "lm_behavior"),
):
    graph_ids_by_lm = {}
    for lm_id in lm_ids:
        lm_result = result.get(lm_id)
        graph_ids_by_lm[lm_id] = _extract_raw_identity(lm_result)[0]

    return _extract_joint_object_signature_key_from_graph_ids(
        graph_ids_by_lm,
        lm_ids=lm_ids,
    )


def _extract_trace_joint_object_signature_key(
    step_trace,
    lm_ids=("lm_morphology", "lm_behavior"),
):
    learning_modules = step_trace.get("learning_modules", {})
    graph_ids_by_lm = {
        lm_id: _extract_raw_identity(learning_modules.get(lm_id, {}))[0]
        for lm_id in lm_ids
    }
    return _extract_joint_object_signature_key_from_graph_ids(
        graph_ids_by_lm,
        lm_ids=lm_ids,
    )


def _iter_joint_decoder_signature_keys(
    condition_result,
    lm_ids=("lm_morphology", "lm_behavior"),
    tail_steps=5,
):
    signature_key, signature_parts = _extract_joint_object_signature_key(
        condition_result,
        lm_ids=lm_ids,
    )
    if signature_key is not None:
        yield signature_key, signature_parts

    trace = list(condition_result.get("trace", []))
    if tail_steps is not None and tail_steps > 0:
        trace = trace[-int(tail_steps):]

    for step_trace in trace:
        signature_key, signature_parts = _extract_trace_joint_object_signature_key(
            step_trace,
            lm_ids=lm_ids,
        )
        if signature_key is not None:
            yield signature_key, signature_parts


def _iter_decoder_graph_ids(condition_result, lm_id, tail_steps=5):
    prediction = _extract_lm_object_prediction(condition_result, lm_id)
    raw_graph_id = _normalize_graph_id(
        prediction.get(
            "final_prediction_latent_id",
            prediction.get("final_prediction_graph_id"),
        )
    )
    if raw_graph_id is not None:
        yield raw_graph_id

    trace = list(condition_result.get("trace", []))
    if tail_steps is not None and tail_steps > 0:
        trace = trace[-int(tail_steps):]

    for step_trace in trace:
        raw_graph_id = _normalize_graph_id(
            _extract_raw_identity(
                step_trace.get("learning_modules", {}).get(lm_id, {})
            )[0]
        )
        if raw_graph_id is not None:
            yield raw_graph_id


def _fit_self_supervised_object_decoders(
    per_model_condition_results,
    lm_ids=("lm_morphology", "lm_behavior", "lm_parent"),
    tail_steps=5,
    training_graph_id_to_target_by_model=None,
):
    support = {lm_id: {} for lm_id in lm_ids}

    for object_name, condition_results in per_model_condition_results.items():
        matched_result = condition_results.get("matched", {})
        for lm_id in lm_ids:
            for raw_graph_id in _iter_decoder_graph_ids(
                matched_result,
                lm_id,
                tail_steps=tail_steps,
            ):
                support[lm_id].setdefault(raw_graph_id, Counter())
                support[lm_id][raw_graph_id][object_name] += 1

        training_graph_id_to_target = (
            (training_graph_id_to_target_by_model or {}).get(object_name, {})
        )
        for lm_id in lm_ids:
            lm_mapping = training_graph_id_to_target.get(lm_id, {}) or {}
            for raw_graph_id, target_objects in lm_mapping.items():
                normalized_targets = {
                    normalized_target
                    for normalized_target in (
                        _normalize_graph_id(target_object)
                        for target_object in (target_objects or [])
                    )
                    if normalized_target is not None
                }
                if object_name not in normalized_targets:
                    continue

                support[lm_id].setdefault(raw_graph_id, Counter())
                support[lm_id][raw_graph_id][object_name] += 1

    decoders = {}
    diagnostics = {}
    for lm_id, lm_support in support.items():
        latent_id_to_object = {}
        latent_id_support = {}
        latent_id_candidate_objects = {}
        for raw_graph_id, counts in lm_support.items():
            latent_id_support[raw_graph_id] = _sorted_counter(counts)
            candidate_objects = _decoder_candidate_objects_from_counts(
                counts,
                lm_id=lm_id,
            )
            latent_id_candidate_objects[raw_graph_id] = list(candidate_objects)
            if len(candidate_objects) == 1:
                latent_id_to_object[raw_graph_id] = candidate_objects[0]

        decoders[lm_id] = latent_id_to_object
        diagnostics[lm_id] = {
            "latent_id_to_object": dict(latent_id_to_object),
            "graph_id_to_object": dict(latent_id_to_object),
            "latent_id_support": latent_id_support,
            "graph_id_support": latent_id_support,
            "latent_id_candidate_objects": {
                raw_graph_id: list(candidate_objects)
                for raw_graph_id, candidate_objects in sorted(
                    latent_id_candidate_objects.items()
                )
            },
            "graph_id_candidate_objects": {
                raw_graph_id: list(candidate_objects)
                for raw_graph_id, candidate_objects in sorted(
                    latent_id_candidate_objects.items()
                )
            },
        }

    return decoders, diagnostics


def _fit_self_supervised_joint_object_decoder(
    per_model_condition_results,
    lm_ids=("lm_morphology", "lm_behavior"),
    tail_steps=5,
):
    support = {}
    signature_parts = {}

    for object_name, condition_results in per_model_condition_results.items():
        matched_result = condition_results.get("matched", {})
        for signature_key, parts in _iter_joint_decoder_signature_keys(
            matched_result,
            lm_ids=lm_ids,
            tail_steps=tail_steps,
        ):
            support.setdefault(signature_key, Counter())
            support[signature_key][object_name] += 1
            signature_parts[signature_key] = dict(parts)

    decoder = {}
    diagnostics = {
        "lm_ids": list(lm_ids),
        "signature_to_object": {},
        "signature_support": {},
        "signature_candidates": {},
        "signature_parts": {},
    }

    for signature_key, counts in support.items():
        decoder[signature_key] = max(
            counts.items(),
            key=lambda item: (item[1], item[0]),
        )[0]
        diagnostics["signature_to_object"][signature_key] = decoder[signature_key]
        diagnostics["signature_support"][signature_key] = _sorted_counter(counts)
        diagnostics["signature_candidates"][signature_key] = (
            _top_support_candidates(counts)
        )
        diagnostics["signature_parts"][signature_key] = dict(
            signature_parts.get(signature_key, {})
        )

    return decoder, diagnostics


def _top_debug_label(evidence):
    if not evidence:
        return None

    return max(
        ((str(label), float(value)) for label, value in evidence.items()),
        key=lambda item: (item[1], item[0]),
    )[0]


def _sorted_counter(counter):
    return {
        label: int(count)
        for label, count in sorted(
            counter.items(),
            key=lambda item: (-item[1], item[0]),
        )
    }


def _top_support_candidates(counter):
    if not counter:
        return []

    max_count = max(int(count) for count in counter.values())
    return [
        str(label)
        for label, count in sorted(
            counter.items(),
            key=lambda item: (-item[1], item[0]),
        )
        if int(count) == max_count
    ]


def _decoder_candidate_objects_from_counts(counter, *, lm_id=None):
    if not counter:
        return []

    if str(lm_id) != "lm_behavior":
        return _top_support_candidates(counter)

    max_count = max(int(count) for count in counter.values())
    if max_count <= 0:
        return []

    ratio_threshold = float(max(BEHAVIOR_OBJECT_DECODER_AMBIGUITY_RATIO, 1.0))
    min_support = max(int(BEHAVIOR_OBJECT_DECODER_AMBIGUITY_MIN_SUPPORT), 1)
    candidates = [
        str(label)
        for label, count in sorted(
            counter.items(),
            key=lambda item: (-item[1], item[0]),
        )
        if int(count) >= min_support
        and (float(int(count)) * ratio_threshold) >= float(max_count)
    ]
    return candidates if candidates else _top_support_candidates(counter)


def _decoder_graph_id_support_counts(raw_graph_id, object_decoder_diagnostics):
    raw_graph_id = _normalize_graph_id(raw_graph_id)
    if raw_graph_id is None or not object_decoder_diagnostics:
        return {}

    return (
        (
            object_decoder_diagnostics.get("latent_id_support")
            or object_decoder_diagnostics.get("graph_id_support")
            or {}
        ).get(
            raw_graph_id,
            {},
        )
        or {}
    )


def _decoder_graph_id_candidates(raw_graph_id, object_decoder_diagnostics):
    if object_decoder_diagnostics:
        raw_graph_id = _normalize_graph_id(raw_graph_id)
        candidate_objects = (
            (
                object_decoder_diagnostics.get("latent_id_candidate_objects")
                or object_decoder_diagnostics.get("graph_id_candidate_objects")
                or {}
            ).get(
                raw_graph_id,
                [],
            )
            or []
        )
        if candidate_objects:
            return [str(candidate) for candidate in candidate_objects]
    return _top_support_candidates(
        _decoder_graph_id_support_counts(
            raw_graph_id,
            object_decoder_diagnostics,
        )
    )


def _is_decoder_graph_id_ambiguous(raw_graph_id, object_decoder_diagnostics):
    return (
        len(
            _decoder_graph_id_candidates(
                raw_graph_id,
                object_decoder_diagnostics,
            )
        )
        > 1
    )


def _is_joint_object_signature_ambiguous(
    signature_key,
    joint_object_decoder_diagnostics,
):
    if signature_key is None or not joint_object_decoder_diagnostics:
        return False

    counts = (
        (joint_object_decoder_diagnostics.get("signature_support") or {}).get(
            signature_key,
            {},
        )
        or {}
    )
    return len(_top_support_candidates(counts)) > 1


def _summarize_lm_alias_source_for_model(
    object_name,
    matched_result,
    *,
    training_graph_id_to_target=None,
    training_reporting_alias_diagnostics=None,
    object_decoder_diagnostics=None,
    lm_ids=("lm_morphology", "lm_behavior", "lm_parent"),
):
    normalized_target = _normalize_graph_id(object_name)
    if normalized_target is None:
        return {}

    training_graph_id_to_target = dict(training_graph_id_to_target or {})
    training_reporting_alias_diagnostics = dict(training_reporting_alias_diagnostics or {})
    object_decoder_diagnostics = dict(object_decoder_diagnostics or {})
    per_lm = {}

    for lm_id in lm_ids:
        lm_training_mapping = {
            _normalize_graph_id(raw_graph_id): sorted(
                {
                    normalized
                    for normalized in (
                        _normalize_graph_id(target_object)
                        for target_object in (target_objects or [])
                    )
                    if normalized is not None
                }
            )
            for raw_graph_id, target_objects in (
                training_graph_id_to_target.get(lm_id, {}) or {}
            ).items()
            if _normalize_graph_id(raw_graph_id) is not None
        }
        lm_training_mapping = {
            raw_graph_id: targets
            for raw_graph_id, targets in lm_training_mapping.items()
            if targets
        }

        target_training_latents = {
            raw_graph_id: list(targets)
            for raw_graph_id, targets in lm_training_mapping.items()
            if normalized_target in targets
        }
        shared_target_training_latents = {
            raw_graph_id: list(targets)
            for raw_graph_id, targets in target_training_latents.items()
            if len(targets) > 1
        }
        alias_stats = (
            (
                training_reporting_alias_diagnostics.get(lm_id, {}) or {}
            ).get("per_target", {})
            or {}
        ).get(normalized_target, {}) or {}
        training_mismatch_steps = int(alias_stats.get("mismatch_steps", 0))
        shared_training_expected = str(lm_id) == "lm_behavior"
        training_registration_issue = bool(
            training_mismatch_steps > 0
            or (
                shared_target_training_latents
                and not shared_training_expected
            )
        )

        matched_output_counts = Counter(
            _iter_decoder_graph_ids(
                matched_result or {},
                lm_id,
                tail_steps=None,
            )
        )
        correct_output_graph_ids = {}
        shared_output_graph_ids = {}
        foreign_output_graph_ids = {}
        unmapped_output_graph_ids = {}
        output_graph_id_details = {}
        for raw_graph_id, count in matched_output_counts.items():
            training_targets = list(lm_training_mapping.get(raw_graph_id, []))
            decoder_candidates = _decoder_graph_id_candidates(
                raw_graph_id,
                object_decoder_diagnostics.get(lm_id),
            )
            output_graph_id_details[raw_graph_id] = {
                "count": int(count),
                "training_targets": list(training_targets),
                "decoder_candidates": list(decoder_candidates),
            }
            if not training_targets and not decoder_candidates:
                unmapped_output_graph_ids[raw_graph_id] = int(count)
            elif (
                normalized_target not in training_targets
                and normalized_target not in decoder_candidates
            ):
                foreign_output_graph_ids[raw_graph_id] = int(count)
            elif len(training_targets) > 1 or len(decoder_candidates) > 1:
                shared_output_graph_ids[raw_graph_id] = int(count)
            else:
                correct_output_graph_ids[raw_graph_id] = int(count)

        matched_eval_output_issue = bool(
            foreign_output_graph_ids or unmapped_output_graph_ids
        )
        if training_registration_issue and matched_eval_output_issue:
            confusion_source = "both"
        elif training_registration_issue:
            confusion_source = "training_registration"
        elif matched_eval_output_issue:
            confusion_source = "matched_eval_output"
        else:
            confusion_source = "none"

        per_lm[lm_id] = {
            "confusion_source": confusion_source,
            "training_registration_issue": bool(training_registration_issue),
            "matched_eval_output_issue": bool(matched_eval_output_issue),
            "training": {
                "target_training_latents": {
                    raw_graph_id: list(targets)
                    for raw_graph_id, targets in sorted(target_training_latents.items())
                },
                "shared_target_training_latents": {
                    raw_graph_id: list(targets)
                    for raw_graph_id, targets in sorted(
                        shared_target_training_latents.items()
                    )
                },
                "shared_training_expected": bool(shared_training_expected),
                "agreement_steps": int(alias_stats.get("agreement_steps", 0)),
                "mismatch_steps": training_mismatch_steps,
                "learning_only_steps": int(alias_stats.get("learning_only_steps", 0)),
                "output_only_steps": int(alias_stats.get("output_only_steps", 0)),
                "learning_latent_counts": {
                    str(raw_graph_id): int(count)
                    for raw_graph_id, count in sorted(
                        (alias_stats.get("learning_latent_counts") or {}).items()
                    )
                },
                "output_latent_counts": {
                    str(raw_graph_id): int(count)
                    for raw_graph_id, count in sorted(
                        (alias_stats.get("output_latent_counts") or {}).items()
                    )
                },
                "registered_latent_counts": {
                    str(raw_graph_id): int(count)
                    for raw_graph_id, count in sorted(
                        (alias_stats.get("registered_latent_counts") or {}).items()
                    )
                },
                "learning_output_pairs": {
                    str(pair): int(count)
                    for pair, count in sorted(
                        (alias_stats.get("learning_output_pairs") or {}).items()
                    )
                },
            },
            "matched_eval": {
                "output_graph_id_counts": _sorted_counter(matched_output_counts),
                "correct_target_graph_ids": {
                    str(raw_graph_id): int(count)
                    for raw_graph_id, count in sorted(correct_output_graph_ids.items())
                },
                "shared_target_graph_ids": {
                    str(raw_graph_id): int(count)
                    for raw_graph_id, count in sorted(shared_output_graph_ids.items())
                },
                "foreign_graph_ids": {
                    str(raw_graph_id): int(count)
                    for raw_graph_id, count in sorted(foreign_output_graph_ids.items())
                },
                "unmapped_graph_ids": {
                    str(raw_graph_id): int(count)
                    for raw_graph_id, count in sorted(unmapped_output_graph_ids.items())
                },
                "graph_id_details": {
                    str(raw_graph_id): dict(details)
                    for raw_graph_id, details in sorted(output_graph_id_details.items())
                },
            },
        }

    return per_lm


def _summarize_learning_module_trace_coverage(trace):
    coverage = {}

    for step_trace in trace:
        for lm_id, lm_trace in step_trace.get("learning_modules", {}).items():
            stats = coverage.setdefault(
                lm_id,
                {
                    "trace_steps": 0,
                    "temporal_status_steps": 0,
                    "temporal_context_steps": 0,
                    "evidence_debug_steps": 0,
                    "evidence_debug_evidence_steps": 0,
                    "skipped_debug_steps": 0,
                },
            )
            stats["trace_steps"] += 1

            if lm_trace.get("temporal_status") is not None:
                stats["temporal_status_steps"] += 1
            if lm_trace.get("temporal_context") is not None:
                stats["temporal_context_steps"] += 1

            evidence_debug = lm_trace.get("evidence_debug")
            if evidence_debug is None:
                continue

            stats["evidence_debug_steps"] += 1
            if evidence_debug.get("step_skipped"):
                stats["skipped_debug_steps"] += 1
            else:
                stats["evidence_debug_evidence_steps"] += 1

    return {
        lm_id: dict(stats)
        for lm_id, stats in sorted(coverage.items())
    }


def _select_best_available_trace_lm(trace, preferred_lm_id="lm_behavior"):
    coverage = _summarize_learning_module_trace_coverage(trace)
    if not coverage:
        return preferred_lm_id, coverage

    if coverage.get(preferred_lm_id, {}).get("evidence_debug_evidence_steps", 0) > 0:
        return preferred_lm_id, coverage

    best_lm_id = max(
        coverage.items(),
        key=lambda item: (
            int(item[1].get("evidence_debug_evidence_steps", 0)),
            int(item[1].get("temporal_status_steps", 0)),
            int(item[1].get("temporal_context_steps", 0)),
            str(item[0]),
        ),
    )[0]
    return best_lm_id, coverage


def _summarize_training_diagnostics(
    exp,
    trace,
    latent_decoder,
    target_lm_id="lm_behavior",
):
    lm = next(
        (
            candidate
            for candidate in exp.monty.learning_modules
            if candidate.learning_module_id == target_lm_id
        ),
        None,
    )
    trace_coverage = _summarize_learning_module_trace_coverage(trace)
    current_label_steps = 0
    predicted_label_steps = 0
    for step_trace in trace:
        temporal_context = (
            step_trace.get("learning_modules", {})
            .get(target_lm_id, {})
            .get("temporal_context", {})
        )
        if temporal_context.get("current_label") is not None:
            current_label_steps += 1
        if temporal_context.get("predicted_label") is not None:
            predicted_label_steps += 1

    known_object_ids = []
    known_state_count = 0
    state_object_totals = Counter()
    reporting_alias_diagnostics = _snapshot_reporting_alias_diagnostics(exp)
    if lm is not None:
        if hasattr(lm, "get_all_known_object_ids"):
            known_object_ids = sorted(lm.get_all_known_object_ids())
        temporal_memory = getattr(lm, "_self_supervised_temporal_memory", None)
        if temporal_memory is not None:
            known_state_count = len(temporal_memory.get_known_states())
        for counter in getattr(
            lm,
            "_self_supervised_state_object_counts",
            {},
        ).values():
            state_object_totals.update(counter)

    return {
        "target_lm_id": target_lm_id,
        "trace_coverage": trace_coverage,
        "current_label_steps": current_label_steps,
        "predicted_label_steps": predicted_label_steps,
        "latent_decoder_size": len(latent_decoder),
        "known_object_ids": known_object_ids,
        "known_state_count": known_state_count,
        "state_object_totals": _sorted_counter(state_object_totals),
        "reporting_alias_diagnostics": reporting_alias_diagnostics,
    }


def _compute_interference_debug(
    trace,
    target_name,
    target_lm_id="lm_behavior",
    object_decoder=None,
):
    base_top_label_counts = Counter()
    after_temporal_behavior_top_label_counts = Counter()
    final_top_label_counts = Counter()
    final_non_target_top_label_counts = Counter()
    base_top_object_counts = Counter()
    after_temporal_behavior_top_object_counts = Counter()
    final_top_object_counts = Counter()
    final_non_target_top_object_counts = Counter()
    skip_reason_counts = Counter()
    steps_with_debug = 0
    steps_with_evidence_debug = 0
    base_to_final_flips = 0
    behavior_to_final_flips = 0
    target_lost_after_temporal_behavior = 0
    target_lost_after_self_supervised = 0
    steps_with_joint_hypothesis_candidates = 0
    steps_with_context_ranked_chart_prior = 0
    steps_with_vote_ranked_chart_prior = 0
    steps_with_joint_top_chart = 0
    steps_with_joint_top_latent_multi_chart = 0
    steps_with_joint_top_context_ranked_chart_prior = 0
    steps_with_joint_top_vote_ranked_chart_prior = 0
    joint_top_chart_counts = Counter()
    joint_chart_switches = 0
    joint_chart_switch_opportunities = 0
    joint_same_latent_chart_switches = 0
    joint_same_latent_chart_switch_opportunities = 0
    previous_joint_top_label = None
    previous_joint_top_chart_id = None
    per_step = []

    for step_trace in trace:
        lm_trace = step_trace.get("learning_modules", {}).get(target_lm_id, {})
        evidence_debug = lm_trace.get("evidence_debug") or {}
        if not evidence_debug:
            continue

        steps_with_debug += 1
        if evidence_debug.get("step_skipped"):
            skip_reason = str(evidence_debug.get("skip_reason", "unknown"))
            skip_reason_counts[skip_reason] += 1
            per_step.append(
                {
                    "step": int(step_trace.get("step", len(per_step))),
                    "frame": int(step_trace.get("frame", 0)),
                    "step_skipped": True,
                    "skip_reason": skip_reason,
                    "observation_summary": dict(
                        evidence_debug.get("observation_summary") or {}
                    ),
                }
            )
            continue

        steps_with_evidence_debug += 1
        base_evidence = dict(evidence_debug.get("base_evidence") or {})
        after_temporal_behavior_evidence = dict(
            evidence_debug.get("after_temporal_behavior_evidence") or base_evidence
        )
        final_evidence = dict(
            evidence_debug.get("final_evidence") or after_temporal_behavior_evidence
        )

        winner_path = evidence_debug.get("winner_path") or {}
        base_top_label = winner_path.get("base") or _top_debug_label(base_evidence)
        after_temporal_behavior_top_label = (
            winner_path.get("after_temporal_behavior")
            or _top_debug_label(after_temporal_behavior_evidence)
        )
        final_top_label = winner_path.get("final") or _top_debug_label(final_evidence)

        base_top_object = _decode_object_name(base_top_label, object_decoder)
        after_temporal_behavior_top_object = _decode_object_name(
            after_temporal_behavior_top_label,
            object_decoder,
        )
        final_top_object = _decode_object_name(final_top_label, object_decoder)
        base_top_object_display = (
            base_top_object if base_top_object is not None else base_top_label
        )
        after_temporal_behavior_top_object_display = (
            after_temporal_behavior_top_object
            if after_temporal_behavior_top_object is not None
            else after_temporal_behavior_top_label
        )
        final_top_object_display = (
            final_top_object if final_top_object is not None else final_top_label
        )

        joint_candidates = [
            dict(candidate)
            for candidate in (evidence_debug.get("joint_hypothesis_candidates") or [])
            if isinstance(candidate, Mapping)
        ]
        joint_candidates = sorted(
            joint_candidates,
            key=lambda candidate: (
                -float(candidate.get("score", 0.0)),
                str(candidate.get("latent_id", candidate.get("object_id", ""))),
                ""
                if candidate.get("chart_id") is None
                else str(candidate.get("chart_id")),
            ),
        )
        joint_top_candidate = joint_candidates[0] if joint_candidates else None
        joint_top_label = None
        joint_top_object = None
        joint_top_chart_id = None
        joint_top_latent_candidates = []
        joint_top_context_ranked_chart_prior = 0.0
        joint_top_vote_ranked_chart_prior = 0.0
        joint_context_ranked_chart_prior_active = False
        joint_vote_ranked_chart_prior_active = False
        if joint_candidates:
            steps_with_joint_hypothesis_candidates += 1
            joint_context_ranked_chart_prior_active = any(
                float(candidate.get("context_ranked_chart_prior", 0.0)) > 0.0
                for candidate in joint_candidates
            )
            joint_vote_ranked_chart_prior_active = any(
                float(candidate.get("vote_ranked_chart_prior", 0.0)) > 0.0
                for candidate in joint_candidates
            )
            if joint_context_ranked_chart_prior_active:
                steps_with_context_ranked_chart_prior += 1
            if joint_vote_ranked_chart_prior_active:
                steps_with_vote_ranked_chart_prior += 1

            joint_top_label = str(
                joint_top_candidate.get(
                    "latent_id",
                    joint_top_candidate.get("object_id"),
                )
            )
            joint_top_object = _decode_object_name(joint_top_label, object_decoder)
            joint_top_chart_id = joint_top_candidate.get("chart_id")
            joint_top_context_ranked_chart_prior = float(
                joint_top_candidate.get("context_ranked_chart_prior", 0.0)
            )
            joint_top_vote_ranked_chart_prior = float(
                joint_top_candidate.get("vote_ranked_chart_prior", 0.0)
            )
            if joint_top_chart_id is not None:
                steps_with_joint_top_chart += 1
                joint_top_chart_counts[str(joint_top_chart_id)] += 1
            joint_top_latent_candidates = [
                dict(candidate)
                for candidate in joint_candidates
                if str(candidate.get("latent_id", candidate.get("object_id")))
                == joint_top_label
            ]
            if len(joint_top_latent_candidates) > 1:
                steps_with_joint_top_latent_multi_chart += 1
            if joint_top_context_ranked_chart_prior > 0.0:
                steps_with_joint_top_context_ranked_chart_prior += 1
            if joint_top_vote_ranked_chart_prior > 0.0:
                steps_with_joint_top_vote_ranked_chart_prior += 1

        if previous_joint_top_chart_id is not None and joint_top_chart_id is not None:
            joint_chart_switch_opportunities += 1
            if previous_joint_top_chart_id != joint_top_chart_id:
                joint_chart_switches += 1
        if (
            previous_joint_top_label is not None
            and previous_joint_top_label == joint_top_label
            and previous_joint_top_chart_id is not None
            and joint_top_chart_id is not None
        ):
            joint_same_latent_chart_switch_opportunities += 1
            if previous_joint_top_chart_id != joint_top_chart_id:
                joint_same_latent_chart_switches += 1

        previous_joint_top_label = joint_top_label
        previous_joint_top_chart_id = joint_top_chart_id

        if base_top_label is not None:
            base_top_label_counts[base_top_label] += 1
        if base_top_object_display is not None:
            base_top_object_counts[base_top_object_display] += 1
        if after_temporal_behavior_top_label is not None:
            after_temporal_behavior_top_label_counts[
                after_temporal_behavior_top_label
            ] += 1
        if after_temporal_behavior_top_object_display is not None:
            after_temporal_behavior_top_object_counts[
                after_temporal_behavior_top_object_display
            ] += 1
        if final_top_label is not None:
            final_top_label_counts[final_top_label] += 1
            if final_top_label != target_name:
                final_non_target_top_label_counts[final_top_label] += 1
        if final_top_object_display is not None:
            final_top_object_counts[final_top_object_display] += 1
            if final_top_object_display != target_name:
                final_non_target_top_object_counts[final_top_object_display] += 1

        if (base_top_label is not None or final_top_label is not None) and (
            base_top_label != final_top_label
        ):
            base_to_final_flips += 1
        if (
            after_temporal_behavior_top_label is not None
            or final_top_label is not None
        ) and (after_temporal_behavior_top_label != final_top_label):
            behavior_to_final_flips += 1
        if (
            base_top_object_display == target_name
            and after_temporal_behavior_top_object_display != target_name
        ):
            target_lost_after_temporal_behavior += 1
        if (
            after_temporal_behavior_top_object_display == target_name
            and final_top_object_display != target_name
        ):
            target_lost_after_self_supervised += 1

        per_step.append(
            {
                "step": int(step_trace.get("step", len(per_step))),
                "frame": int(step_trace.get("frame", 0)),
                "base_top_label": base_top_label,
                "base_top_object": base_top_object_display,
                "after_temporal_behavior_top_label": after_temporal_behavior_top_label,
                "after_temporal_behavior_top_object": (
                    after_temporal_behavior_top_object_display
                ),
                "final_top_label": final_top_label,
                "final_top_object": final_top_object_display,
                "target_base_evidence": base_evidence.get(target_name),
                "target_after_temporal_behavior_evidence": (
                    after_temporal_behavior_evidence.get(target_name)
                ),
                "target_final_evidence": final_evidence.get(target_name),
                "base_evidence": base_evidence,
                "after_temporal_behavior_evidence": after_temporal_behavior_evidence,
                "final_evidence": final_evidence,
                "temporal_behavior_adjustments": dict(
                    evidence_debug.get("temporal_behavior_adjustments") or {}
                ),
                "temporal_behavior_scores": dict(
                    evidence_debug.get("temporal_behavior_scores") or {}
                ),
                "self_supervised_adjustments": dict(
                    evidence_debug.get("self_supervised_adjustments") or {}
                ),
                "self_supervised_support": {
                    key: dict(value)
                    for key, value in (
                        evidence_debug.get("self_supervised_support") or {}
                    ).items()
                },
                "temporal_labels": dict(evidence_debug.get("temporal_labels") or {}),
                "joint_top_label": joint_top_label,
                "joint_top_object": (
                    joint_top_object if joint_top_object is not None else joint_top_label
                ),
                "joint_top_chart_id": joint_top_chart_id,
                "joint_hypothesis_candidate_count": len(joint_candidates),
                "joint_top_latent_candidate_count": len(joint_top_latent_candidates),
                "joint_top_context_ranked_chart_prior": (
                    joint_top_context_ranked_chart_prior
                ),
                "joint_top_vote_ranked_chart_prior": (
                    joint_top_vote_ranked_chart_prior
                ),
                "joint_context_ranked_chart_prior_active": (
                    joint_context_ranked_chart_prior_active
                ),
                "joint_vote_ranked_chart_prior_active": (
                    joint_vote_ranked_chart_prior_active
                ),
                "joint_top_latent_candidates": joint_top_latent_candidates,
                "query_bias_norms": dict(evidence_debug.get("query_bias_norms") or {}),
                "prediction_mismatch": evidence_debug.get("prediction_mismatch"),
                "action_prediction_error": evidence_debug.get(
                    "action_prediction_error"
                ),
            }
        )

    if steps_with_debug <= 0:
        return {
            "steps_with_debug": 0,
            "steps_with_evidence_debug": 0,
            "skipped_step_count": 0,
            "skip_reason_counts": {},
            "base_top_label_counts": {},
            "after_temporal_behavior_top_label_counts": {},
            "final_top_label_counts": {},
            "final_non_target_top_label_counts": {},
            "base_top_object_counts": {},
            "after_temporal_behavior_top_object_counts": {},
            "final_top_object_counts": {},
            "final_non_target_top_object_counts": {},
            "base_target_win_fraction": None,
            "after_temporal_behavior_target_win_fraction": None,
            "final_target_win_fraction": None,
            "base_to_final_flip_fraction": None,
            "behavior_to_final_flip_fraction": None,
            "target_lost_after_temporal_behavior_fraction": None,
            "target_lost_after_self_supervised_fraction": None,
            "steps_with_joint_hypothesis_candidates": 0,
            "joint_hypothesis_candidate_fraction": None,
            "joint_top_chart_counts": {},
            "joint_top_chart_resolved_fraction": None,
            "joint_top_latent_multi_chart_fraction": None,
            "joint_context_ranked_chart_prior_active_fraction": None,
            "joint_vote_ranked_chart_prior_active_fraction": None,
            "joint_top_context_ranked_chart_prior_fraction": None,
            "joint_top_vote_ranked_chart_prior_fraction": None,
            "joint_chart_switch_fraction": None,
            "joint_same_latent_chart_switch_fraction": None,
            "joint_dominant_chart_fraction": None,
            "per_step": [],
        }

    if steps_with_evidence_debug <= 0:
        return {
            "steps_with_debug": steps_with_debug,
            "steps_with_evidence_debug": 0,
            "skipped_step_count": steps_with_debug,
            "skip_reason_counts": _sorted_counter(skip_reason_counts),
            "base_top_label_counts": {},
            "after_temporal_behavior_top_label_counts": {},
            "final_top_label_counts": {},
            "final_non_target_top_label_counts": {},
            "base_top_object_counts": {},
            "after_temporal_behavior_top_object_counts": {},
            "final_top_object_counts": {},
            "final_non_target_top_object_counts": {},
            "base_target_win_fraction": None,
            "after_temporal_behavior_target_win_fraction": None,
            "final_target_win_fraction": None,
            "base_to_final_flip_fraction": None,
            "behavior_to_final_flip_fraction": None,
            "target_lost_after_temporal_behavior_fraction": None,
            "target_lost_after_self_supervised_fraction": None,
            "steps_with_joint_hypothesis_candidates": 0,
            "joint_hypothesis_candidate_fraction": None,
            "joint_top_chart_counts": {},
            "joint_top_chart_resolved_fraction": None,
            "joint_top_latent_multi_chart_fraction": None,
            "joint_context_ranked_chart_prior_active_fraction": None,
            "joint_vote_ranked_chart_prior_active_fraction": None,
            "joint_top_context_ranked_chart_prior_fraction": None,
            "joint_top_vote_ranked_chart_prior_fraction": None,
            "joint_chart_switch_fraction": None,
            "joint_same_latent_chart_switch_fraction": None,
            "joint_dominant_chart_fraction": None,
            "per_step": per_step,
        }

    dominant_joint_chart_count = (
        max(joint_top_chart_counts.values()) if joint_top_chart_counts else 0
    )

    return {
        "steps_with_debug": steps_with_debug,
        "steps_with_evidence_debug": steps_with_evidence_debug,
        "skipped_step_count": steps_with_debug - steps_with_evidence_debug,
        "skip_reason_counts": _sorted_counter(skip_reason_counts),
        "base_top_label_counts": _sorted_counter(base_top_label_counts),
        "after_temporal_behavior_top_label_counts": _sorted_counter(
            after_temporal_behavior_top_label_counts
        ),
        "final_top_label_counts": _sorted_counter(final_top_label_counts),
        "final_non_target_top_label_counts": _sorted_counter(
            final_non_target_top_label_counts
        ),
        "base_top_object_counts": _sorted_counter(base_top_object_counts),
        "after_temporal_behavior_top_object_counts": _sorted_counter(
            after_temporal_behavior_top_object_counts
        ),
        "final_top_object_counts": _sorted_counter(final_top_object_counts),
        "final_non_target_top_object_counts": _sorted_counter(
            final_non_target_top_object_counts
        ),
        "base_target_win_fraction": (
            base_top_object_counts.get(target_name, 0) / steps_with_evidence_debug
        ),
        "after_temporal_behavior_target_win_fraction": (
            after_temporal_behavior_top_object_counts.get(target_name, 0)
            / steps_with_evidence_debug
        ),
        "final_target_win_fraction": (
            final_top_object_counts.get(target_name, 0) / steps_with_evidence_debug
        ),
        "base_to_final_flip_fraction": (
            base_to_final_flips / steps_with_evidence_debug
        ),
        "behavior_to_final_flip_fraction": (
            behavior_to_final_flips / steps_with_evidence_debug
        ),
        "target_lost_after_temporal_behavior_fraction": (
            target_lost_after_temporal_behavior / steps_with_evidence_debug
        ),
        "target_lost_after_self_supervised_fraction": (
            target_lost_after_self_supervised / steps_with_evidence_debug
        ),
        "steps_with_joint_hypothesis_candidates": (
            steps_with_joint_hypothesis_candidates
        ),
        "joint_hypothesis_candidate_fraction": (
            steps_with_joint_hypothesis_candidates / steps_with_evidence_debug
        ),
        "joint_top_chart_counts": _sorted_counter(joint_top_chart_counts),
        "joint_top_chart_resolved_fraction": (
            steps_with_joint_top_chart / steps_with_joint_hypothesis_candidates
            if steps_with_joint_hypothesis_candidates > 0
            else None
        ),
        "joint_top_latent_multi_chart_fraction": (
            steps_with_joint_top_latent_multi_chart
            / steps_with_joint_hypothesis_candidates
            if steps_with_joint_hypothesis_candidates > 0
            else None
        ),
        "joint_context_ranked_chart_prior_active_fraction": (
            steps_with_context_ranked_chart_prior / steps_with_evidence_debug
        ),
        "joint_vote_ranked_chart_prior_active_fraction": (
            steps_with_vote_ranked_chart_prior / steps_with_evidence_debug
        ),
        "joint_top_context_ranked_chart_prior_fraction": (
            steps_with_joint_top_context_ranked_chart_prior
            / steps_with_joint_hypothesis_candidates
            if steps_with_joint_hypothesis_candidates > 0
            else None
        ),
        "joint_top_vote_ranked_chart_prior_fraction": (
            steps_with_joint_top_vote_ranked_chart_prior
            / steps_with_joint_hypothesis_candidates
            if steps_with_joint_hypothesis_candidates > 0
            else None
        ),
        "joint_chart_switch_fraction": (
            joint_chart_switches / joint_chart_switch_opportunities
            if joint_chart_switch_opportunities > 0
            else None
        ),
        "joint_same_latent_chart_switch_fraction": (
            joint_same_latent_chart_switches
            / joint_same_latent_chart_switch_opportunities
            if joint_same_latent_chart_switch_opportunities > 0
            else None
        ),
        "joint_dominant_chart_fraction": (
            dominant_joint_chart_count / steps_with_joint_hypothesis_candidates
            if steps_with_joint_hypothesis_candidates > 0
            else None
        ),
        "per_step": per_step,
    }


def _extract_trace_context_packet(step_trace, sender_id):
    lm_trace = (
        (step_trace.get("learning_modules", {}) or {}).get(sender_id, {}) or {}
    )
    output_state = dict(lm_trace.get("output_state") or {})
    packet = (
        lm_trace.get("context_signal")
    )
    if not isinstance(packet, dict):
        return None

    active_cells = packet.get("active_cells")
    if active_cells is None:
        return None

    if isinstance(active_cells, np.ndarray):
        context = active_cells.astype(np.float32, copy=True)
    else:
        context = np.asarray(active_cells, dtype=np.float32)

    latent_id = packet.get("latent_id")
    graph_id = packet.get("graph_id")
    if latent_id is None:
        latent_id = graph_id
    if graph_id is None:
        graph_id = latent_id
    if graph_id is None:
        latent_id = output_state.get("latent_id", output_state.get("graph_id"))
        graph_id = latent_id
    if graph_id is None:
        latent_id = lm_trace.get("latent_id", lm_trace.get("graph_id"))
        graph_id = latent_id

    confidence = packet.get("confidence")
    if confidence is None:
        confidence = output_state.get("confidence")
    if confidence is None:
        confidence = lm_trace.get("evidence", 0.0)

    location = output_state.get("location")
    if isinstance(location, np.ndarray):
        location = location.astype(np.float32, copy=True)
    elif location is not None:
        location = np.asarray(location, dtype=np.float32)

    pose_vectors = output_state.get("pose_vectors")
    if isinstance(pose_vectors, np.ndarray):
        pose_vectors = pose_vectors.astype(np.float32, copy=True)
    elif pose_vectors is not None:
        pose_vectors = np.asarray(pose_vectors, dtype=np.float32)

    return {
        "active_cells": context,
        "sender_id": str(packet.get("sender_id") or sender_id),
        "latent_id": latent_id,
        "graph_id": graph_id,
        "confidence": float(confidence),
        "sender_step_count": int(packet.get("sender_step_count", 0)),
        "location": location,
        "pose_vectors": pose_vectors,
    }


def _collect_replayed_lm_step_trace(lm, step, frame):
    mlh = lm.get_current_mlh() if hasattr(lm, "get_current_mlh") else {}
    raw_identity = mlh.get("latent_id", mlh.get("graph_id"))
    lm_trace = {
        "latent_id": raw_identity,
        "graph_id": raw_identity,
        "evidence": float(mlh.get("evidence", 0.0)),
    }
    if hasattr(lm, "get_temporal_prediction_status"):
        lm_trace["temporal_status"] = lm.get_temporal_prediction_status()
    if hasattr(lm, "get_temporal_surprise"):
        lm_trace["temporal_surprise"] = float(lm.get_temporal_surprise())
    if hasattr(lm, "get_temporal_context"):
        context = Panda3DTorchExperiment._simplify_temporal_context(
            lm.get_temporal_context()
        )
        if context is not None:
            lm_trace["temporal_context"] = context
    if hasattr(lm, "get_evidence_debug"):
        evidence_debug = lm.get_evidence_debug()
        if evidence_debug is not None:
            lm_trace["evidence_debug"] = evidence_debug

    return {
        "step": int(step),
        "frame": int(frame),
        "learning_modules": {lm.learning_module_id: lm_trace},
    }


def _build_predictive_replay_child_state(packet, child_index):
    graph_id = _normalize_graph_id(packet.get("latent_id", packet.get("graph_id")))
    if graph_id is None:
        return None

    active_cells = packet.get("active_cells")
    if active_cells is None:
        active_cells = np.zeros(0, dtype=np.float32)
    else:
        active_cells = np.asarray(active_cells, dtype=np.float32)

    location = packet.get("location")
    if location is None:
        location = np.zeros(3, dtype=np.float64)
    else:
        location = np.asarray(location, dtype=np.float64)

    pose_vectors = packet.get("pose_vectors")
    if pose_vectors is None:
        pose_vectors = np.eye(3, dtype=np.float64)
    else:
        pose_vectors = np.asarray(pose_vectors, dtype=np.float64)

    return State(
        location=location,
        morphological_features={
            "pose_vectors": pose_vectors,
            "pose_fully_defined": True,
            "on_object": True,
        },
        non_morphological_features={
            "latent_id": graph_id,
            "graph_id": graph_id,
            "active_cells": active_cells,
        },
        confidence=float(np.clip(packet.get("confidence", 0.0), 0.0, 1.0)),
        use_state=True,
        sender_id=str(packet.get("sender_id") or f"replay_child_{child_index}"),
        sender_type="LM",
    )


def _instantiate_parent_replay_lm(
    parent_state_dict,
    parent_column_kwargs,
    parent_lm_kwargs,
    *,
    lm_family,
):
    if _is_predictive_hypothesis_family(lm_family):
        effective_lm_kwargs = deepcopy(parent_lm_kwargs)
        core_kwargs = dict(effective_lm_kwargs.get("core_kwargs", {}))
        core_state = (parent_state_dict or {}).get("core", {})
        replay_context_dim = core_state.get("context_dim")
        if replay_context_dim is None:
            replay_context_dim = (
                core_state.get("memory", {}) or {}
            ).get("embedding_dim")
        if replay_context_dim is not None:
            core_kwargs["context_dim"] = int(replay_context_dim)
        effective_lm_kwargs["core_kwargs"] = core_kwargs

        return PredictiveHypothesisTorchLM(
            learning_module_id="lm_parent",
            **effective_lm_kwargs,
        )

    return CorticalColumnTorchLM(
        column_kwargs=deepcopy(parent_column_kwargs),
        learning_module_id="lm_parent",
        **deepcopy(parent_lm_kwargs),
    )


def _build_replayed_primary_summary(
    replay_lm,
    replay_trace,
    target_name,
    object_decoder=None,
    boundary_active_threshold=BOUNDARY_PRESSURE_ACTIVE_THRESHOLD,
):
    mlh = replay_lm.get_current_mlh() if hasattr(replay_lm, "get_current_mlh") else {}
    raw_graph_id = _normalize_graph_id(mlh.get("latent_id", mlh.get("graph_id")))
    decoded_object = _decode_object_name(raw_graph_id, object_decoder)
    primary = _compute_primary_temporal_metrics(
        replay_trace,
        target_lm_id=replay_lm.learning_module_id,
        boundary_active_threshold=boundary_active_threshold,
    )
    primary.update(
        {
            "final_prediction_latent_id": raw_graph_id,
            "final_prediction_graph_id": raw_graph_id,
            "final_prediction": (
                decoded_object if decoded_object is not None else raw_graph_id
            ),
            "final_prediction_correct": (
                (decoded_object if decoded_object is not None else raw_graph_id)
                == target_name
            ),
            "final_prediction_decoded": (
                object_decoder is not None
                and raw_graph_id is not None
                and decoded_object is not None
            ),
        }
    )
    return primary


def _compare_parent_replay_to_live(live_parent_report, replay_summary):
    live_primary = dict(live_parent_report.get("primary") or {})
    live_debug = dict(live_parent_report.get("interference_debug") or {})
    replay_primary = dict(replay_summary.get("primary") or {})
    replay_debug = dict(replay_summary.get("interference_debug") or {})

    def _delta(lhs, rhs):
        if lhs is None or rhs is None:
            return None
        return lhs - rhs

    return {
        "final_prediction_matches_live": (
            replay_primary.get("final_prediction")
            == live_primary.get("final_prediction")
        ),
        "final_prediction_latent_id_matches_live": (
            replay_primary.get("final_prediction_latent_id")
            == live_primary.get("final_prediction_latent_id")
        ),
        "final_prediction_graph_id_matches_live": (
            replay_primary.get("final_prediction_graph_id")
            == live_primary.get("final_prediction_graph_id")
        ),
        "final_prediction_correct_matches_live": (
            replay_primary.get("final_prediction_correct")
            == live_primary.get("final_prediction_correct")
        ),
        "base_target_win_fraction_delta_vs_live": _delta(
            replay_debug.get("base_target_win_fraction"),
            live_debug.get("base_target_win_fraction"),
        ),
        "final_target_win_fraction_delta_vs_live": _delta(
            replay_debug.get("final_target_win_fraction"),
            live_debug.get("final_target_win_fraction"),
        ),
        "joint_hypothesis_candidate_fraction_delta_vs_live": _delta(
            replay_debug.get("joint_hypothesis_candidate_fraction"),
            live_debug.get("joint_hypothesis_candidate_fraction"),
        ),
        "joint_top_latent_multi_chart_fraction_delta_vs_live": _delta(
            replay_debug.get("joint_top_latent_multi_chart_fraction"),
            live_debug.get("joint_top_latent_multi_chart_fraction"),
        ),
        "joint_same_latent_chart_switch_fraction_delta_vs_live": _delta(
            replay_debug.get("joint_same_latent_chart_switch_fraction"),
            live_debug.get("joint_same_latent_chart_switch_fraction"),
        ),
        "joint_top_context_ranked_chart_prior_fraction_delta_vs_live": _delta(
            replay_debug.get("joint_top_context_ranked_chart_prior_fraction"),
            live_debug.get("joint_top_context_ranked_chart_prior_fraction"),
        ),
        "joint_top_vote_ranked_chart_prior_fraction_delta_vs_live": _delta(
            replay_debug.get("joint_top_vote_ranked_chart_prior_fraction"),
            live_debug.get("joint_top_vote_ranked_chart_prior_fraction"),
        ),
    }


def _replay_parent_from_context_trace(
    parent_state_dict,
    parent_column_kwargs,
    parent_lm_kwargs,
    matched_trace,
    target_name,
    lm_family="cortical_column_torch",
    object_decoder=None,
    boundary_active_threshold=BOUNDARY_PRESSURE_ACTIVE_THRESHOLD,
):
    replay_lm = _instantiate_parent_replay_lm(
        parent_state_dict,
        parent_column_kwargs,
        parent_lm_kwargs,
        lm_family=lm_family,
    )
    replay_lm.load_state_dict(deepcopy(parent_state_dict))
    replay_lm.set_experiment_mode(ExperimentMode.EVAL)
    replay_lm.pre_episode(
        primary_target={"object": "placeholder", "quat_rotation": [1, 0, 0, 0]}
    )

    replay_trace = []
    replayable_steps = 0
    missing_context_steps = 0
    sender_ids = ("lm_morphology", "lm_behavior")

    for step_trace in matched_trace:
        packets = []
        for sender_id in sender_ids:
            packet = _extract_trace_context_packet(step_trace, sender_id)
            if packet is None:
                packets = []
                break
            packets.append(packet)

        if not packets:
            missing_context_steps += 1
            continue

        previous_step_count = getattr(replay_lm, "_step_count", 0)
        if _is_predictive_hypothesis_family(lm_family):
            child_states = []
            for index, packet in enumerate(packets):
                child_state = _build_predictive_replay_child_state(packet, index)
                if child_state is None:
                    child_states = []
                    break
                child_states.append(child_state)

            if not child_states:
                missing_context_steps += 1
                continue

            if hasattr(replay_lm, "set_action_context"):
                replay_lm.set_action_context(np.zeros(8, dtype=np.float32))
            replay_lm.matching_step(None, child_states)
        else:
            for packet in packets:
                replay_lm.receive_context(**packet)

        if getattr(replay_lm, "_step_count", 0) <= previous_step_count:
            missing_context_steps += 1
            continue

        replayable_steps += 1
        replay_trace.append(
            _collect_replayed_lm_step_trace(
                replay_lm,
                step_trace.get("step", replayable_steps - 1),
                step_trace.get("frame", 0),
            )
        )

    replay_summary = {
        "replayable_steps": replayable_steps,
        "missing_context_steps": missing_context_steps,
        "primary": _build_replayed_primary_summary(
            replay_lm,
            replay_trace,
            target_name,
            object_decoder=object_decoder,
            boundary_active_threshold=boundary_active_threshold,
        ),
        "interference_debug": _compute_interference_debug(
            replay_trace,
            target_name,
            target_lm_id="lm_parent",
            object_decoder=object_decoder,
        ),
    }
    replay_lm.post_episode()
    return replay_summary


def _build_parent_context_replay_diagnostics(
    per_model,
    raw_eval_results,
    parent_state_dict,
    parent_column_kwargs,
    parent_lm_kwargs,
    object_decoders,
    lm_family="cortical_column_torch",
    boundary_active_threshold=BOUNDARY_PRESSURE_ACTIVE_THRESHOLD,
):
    per_model_diag = {}

    for name, model_report in per_model.items():
        matched_trace = raw_eval_results.get(name, {}).get("matched", {}).get("trace", [])
        replay_summary = _replay_parent_from_context_trace(
            parent_state_dict=parent_state_dict,
            parent_column_kwargs=parent_column_kwargs,
            parent_lm_kwargs=parent_lm_kwargs,
            matched_trace=matched_trace,
            target_name=name,
            lm_family=lm_family,
            object_decoder=(object_decoders or {}).get("lm_parent"),
            boundary_active_threshold=boundary_active_threshold,
        )
        live_parent_report = {
            "primary": model_report["matched"]["lm_object_predictions"]["lm_parent"],
            "interference_debug": model_report["matched"][
                "child_interference_debug"
            ].get("lm_parent", {}),
        }
        per_model_diag[name] = {
            "live_parent": live_parent_report,
            "replayed_parent": replay_summary,
            "vs_live": _compare_parent_replay_to_live(
                live_parent_report,
                replay_summary,
            ),
        }

    return {
        "supported": True,
        "lm_family": str(lm_family),
        "per_model": per_model_diag,
        "aggregate": {
            "supported": True,
            "final_prediction_matches_live_mean": _mean(
                int(report["vs_live"]["final_prediction_matches_live"])
                for report in per_model_diag.values()
            ),
            "final_prediction_latent_id_matches_live_mean": _mean(
                int(report["vs_live"]["final_prediction_latent_id_matches_live"])
                for report in per_model_diag.values()
            ),
            "final_prediction_graph_id_matches_live_mean": _mean(
                int(report["vs_live"]["final_prediction_graph_id_matches_live"])
                for report in per_model_diag.values()
            ),
            "base_target_win_fraction_delta_vs_live_mean": _mean(
                report["vs_live"]["base_target_win_fraction_delta_vs_live"]
                for report in per_model_diag.values()
            ),
            "final_target_win_fraction_delta_vs_live_mean": _mean(
                report["vs_live"]["final_target_win_fraction_delta_vs_live"]
                for report in per_model_diag.values()
            ),
            "joint_hypothesis_candidate_fraction_delta_vs_live_mean": _mean(
                report["vs_live"][
                    "joint_hypothesis_candidate_fraction_delta_vs_live"
                ]
                for report in per_model_diag.values()
            ),
            "joint_top_latent_multi_chart_fraction_delta_vs_live_mean": _mean(
                report["vs_live"][
                    "joint_top_latent_multi_chart_fraction_delta_vs_live"
                ]
                for report in per_model_diag.values()
            ),
            "joint_same_latent_chart_switch_fraction_delta_vs_live_mean": _mean(
                report["vs_live"][
                    "joint_same_latent_chart_switch_fraction_delta_vs_live"
                ]
                for report in per_model_diag.values()
            ),
            "joint_top_context_ranked_chart_prior_fraction_delta_vs_live_mean": _mean(
                report["vs_live"][
                    "joint_top_context_ranked_chart_prior_fraction_delta_vs_live"
                ]
                for report in per_model_diag.values()
            ),
            "joint_top_vote_ranked_chart_prior_fraction_delta_vs_live_mean": _mean(
                report["vs_live"][
                    "joint_top_vote_ranked_chart_prior_fraction_delta_vs_live"
                ]
                for report in per_model_diag.values()
            ),
        },
    }


def _pairwise_model_key(model_pair):
    return "__".join(model_pair)


def _summarize_pairwise_benchmark(pair_report):
    summary = {
        "models": list(pair_report.get("models", [])),
        "aggregate": {
            "matched_top1_accuracy_mean": pair_report.get("aggregate", {}).get(
                "matched_top1_accuracy_mean"
            ),
            "matched_top1_resolved_fraction_mean": pair_report.get(
                "aggregate", {}
            ).get("matched_top1_resolved_fraction_mean"),
            "matched_top1_strict_accuracy_mean": pair_report.get(
                "aggregate", {}
            ).get("matched_top1_strict_accuracy_mean"),
            "morphology_matched_top1_accuracy_mean": pair_report.get(
                "aggregate", {}
            ).get("morphology_matched_top1_accuracy_mean"),
            "morphology_matched_top1_resolved_fraction_mean": pair_report.get(
                "aggregate", {}
            ).get("morphology_matched_top1_resolved_fraction_mean"),
            "morphology_matched_top1_strict_accuracy_mean": pair_report.get(
                "aggregate", {}
            ).get("morphology_matched_top1_strict_accuracy_mean"),
            "behavior_matched_top1_accuracy_mean": pair_report.get(
                "aggregate", {}
            ).get("behavior_matched_top1_accuracy_mean"),
            "behavior_matched_top1_resolved_fraction_mean": pair_report.get(
                "aggregate", {}
            ).get("behavior_matched_top1_resolved_fraction_mean"),
            "behavior_matched_top1_strict_accuracy_mean": pair_report.get(
                "aggregate", {}
            ).get("behavior_matched_top1_strict_accuracy_mean"),
            "parent_matched_top1_accuracy_mean": pair_report.get(
                "aggregate", {}
            ).get("parent_matched_top1_accuracy_mean"),
            "parent_matched_top1_resolved_fraction_mean": pair_report.get(
                "aggregate", {}
            ).get("parent_matched_top1_resolved_fraction_mean"),
            "parent_matched_top1_strict_accuracy_mean": pair_report.get(
                "aggregate", {}
            ).get("parent_matched_top1_strict_accuracy_mean"),
            "matched_joint_hypothesis_candidate_fraction_mean": pair_report.get(
                "aggregate", {}
            ).get("matched_joint_hypothesis_candidate_fraction_mean"),
            "matched_joint_top_chart_resolved_fraction_mean": pair_report.get(
                "aggregate", {}
            ).get("matched_joint_top_chart_resolved_fraction_mean"),
            "matched_joint_top_latent_multi_chart_fraction_mean": pair_report.get(
                "aggregate", {}
            ).get("matched_joint_top_latent_multi_chart_fraction_mean"),
            "matched_joint_chart_switch_fraction_mean": pair_report.get(
                "aggregate", {}
            ).get("matched_joint_chart_switch_fraction_mean"),
            "matched_joint_same_latent_chart_switch_fraction_mean": pair_report.get(
                "aggregate", {}
            ).get("matched_joint_same_latent_chart_switch_fraction_mean"),
            "matched_joint_top_context_ranked_chart_prior_fraction_mean": pair_report.get(
                "aggregate", {}
            ).get("matched_joint_top_context_ranked_chart_prior_fraction_mean"),
            "matched_joint_top_vote_ranked_chart_prior_fraction_mean": pair_report.get(
                "aggregate", {}
            ).get("matched_joint_top_vote_ranked_chart_prior_fraction_mean"),
            "matched_joint_dominant_chart_fraction_mean": pair_report.get(
                "aggregate", {}
            ).get("matched_joint_dominant_chart_fraction_mean"),
        },
        "per_model": {},
    }

    for name in pair_report.get("models", []):
        model_report = pair_report.get("per_model", {}).get(name, {})
        matched = model_report.get("matched", {})
        child_predictions = matched.get("child_lm_primary", {})
        child_debug = matched.get("child_interference_debug", {})
        parent_prediction = matched.get("lm_object_predictions", {}).get(
            "lm_parent", {}
        )
        summary["per_model"][name] = {
            "matched": {
                "primary": {
                    key: matched.get("primary", {}).get(key)
                    for key in (
                        "final_prediction",
                        "final_prediction_latent_id",
                        "final_prediction_graph_id",
                        "final_prediction_correct",
                    )
                },
                "lm_morphology": {
                    "final_prediction": child_predictions.get(
                        "lm_morphology", {}
                    ).get("final_prediction"),
                    "final_prediction_latent_id": child_predictions.get(
                        "lm_morphology", {}
                    ).get("final_prediction_latent_id"),
                    "final_prediction_graph_id": child_predictions.get(
                        "lm_morphology", {}
                    ).get("final_prediction_graph_id"),
                    "final_prediction_correct": child_predictions.get(
                        "lm_morphology", {}
                    ).get("final_prediction_correct"),
                    "base_target_win_fraction": child_debug.get(
                        "lm_morphology", {}
                    ).get("base_target_win_fraction"),
                    "final_target_win_fraction": child_debug.get(
                        "lm_morphology", {}
                    ).get("final_target_win_fraction"),
                    "joint_hypothesis_candidate_fraction": child_debug.get(
                        "lm_morphology", {}
                    ).get("joint_hypothesis_candidate_fraction"),
                    "joint_top_chart_resolved_fraction": child_debug.get(
                        "lm_morphology", {}
                    ).get("joint_top_chart_resolved_fraction"),
                    "joint_top_latent_multi_chart_fraction": child_debug.get(
                        "lm_morphology", {}
                    ).get("joint_top_latent_multi_chart_fraction"),
                    "joint_chart_switch_fraction": child_debug.get(
                        "lm_morphology", {}
                    ).get("joint_chart_switch_fraction"),
                    "joint_same_latent_chart_switch_fraction": child_debug.get(
                        "lm_morphology", {}
                    ).get("joint_same_latent_chart_switch_fraction"),
                    "joint_top_context_ranked_chart_prior_fraction": child_debug.get(
                        "lm_morphology", {}
                    ).get("joint_top_context_ranked_chart_prior_fraction"),
                    "joint_top_vote_ranked_chart_prior_fraction": child_debug.get(
                        "lm_morphology", {}
                    ).get("joint_top_vote_ranked_chart_prior_fraction"),
                    "joint_dominant_chart_fraction": child_debug.get(
                        "lm_morphology", {}
                    ).get("joint_dominant_chart_fraction"),
                    "joint_top_chart_counts": child_debug.get(
                        "lm_morphology", {}
                    ).get("joint_top_chart_counts", {}),
                    "final_top_label_counts": child_debug.get(
                        "lm_morphology", {}
                    ).get("final_top_label_counts", {}),
                },
                "lm_behavior": {
                    "final_prediction": child_predictions.get(
                        "lm_behavior", {}
                    ).get("final_prediction"),
                    "final_prediction_latent_id": child_predictions.get(
                        "lm_behavior", {}
                    ).get("final_prediction_latent_id"),
                    "final_prediction_graph_id": child_predictions.get(
                        "lm_behavior", {}
                    ).get("final_prediction_graph_id"),
                    "final_prediction_correct": child_predictions.get(
                        "lm_behavior", {}
                    ).get("final_prediction_correct"),
                    "base_target_win_fraction": child_debug.get(
                        "lm_behavior", {}
                    ).get("base_target_win_fraction"),
                    "final_target_win_fraction": child_debug.get(
                        "lm_behavior", {}
                    ).get("final_target_win_fraction"),
                    "joint_hypothesis_candidate_fraction": child_debug.get(
                        "lm_behavior", {}
                    ).get("joint_hypothesis_candidate_fraction"),
                    "joint_top_chart_resolved_fraction": child_debug.get(
                        "lm_behavior", {}
                    ).get("joint_top_chart_resolved_fraction"),
                    "joint_top_latent_multi_chart_fraction": child_debug.get(
                        "lm_behavior", {}
                    ).get("joint_top_latent_multi_chart_fraction"),
                    "joint_chart_switch_fraction": child_debug.get(
                        "lm_behavior", {}
                    ).get("joint_chart_switch_fraction"),
                    "joint_same_latent_chart_switch_fraction": child_debug.get(
                        "lm_behavior", {}
                    ).get("joint_same_latent_chart_switch_fraction"),
                    "joint_top_context_ranked_chart_prior_fraction": child_debug.get(
                        "lm_behavior", {}
                    ).get("joint_top_context_ranked_chart_prior_fraction"),
                    "joint_top_vote_ranked_chart_prior_fraction": child_debug.get(
                        "lm_behavior", {}
                    ).get("joint_top_vote_ranked_chart_prior_fraction"),
                    "joint_dominant_chart_fraction": child_debug.get(
                        "lm_behavior", {}
                    ).get("joint_dominant_chart_fraction"),
                    "joint_top_chart_counts": child_debug.get(
                        "lm_behavior", {}
                    ).get("joint_top_chart_counts", {}),
                    "final_top_label_counts": child_debug.get(
                        "lm_behavior", {}
                    ).get("final_top_label_counts", {}),
                },
                "lm_parent": {
                    key: parent_prediction.get(key)
                    for key in (
                        "final_prediction",
                        "final_prediction_latent_id",
                        "final_prediction_graph_id",
                        "final_prediction_correct",
                    )
                },
            }
        }

    return summary


def _run_pairwise_benchmark_subprocess(
    model_pair,
    phase_bins,
    train_cycles,
    eval_cycles,
    stretch,
    compression_stride,
    resolution,
    column_kwargs,
    morphology_lm_kwargs,
    behavior_lm_kwargs,
    parent_column_kwargs,
    parent_lm_kwargs,
    boundary_active_threshold,
    fully_self_supervised_lms,
    object_decoder_tail_steps,
    lm_family,
    detail_grid_shape,
    use_detail_aware_sensors,
):
    if any(
        cfg
        for cfg in (
            morphology_lm_kwargs,
            behavior_lm_kwargs,
            parent_column_kwargs,
            parent_lm_kwargs,
        )
    ):
        return None

    column_kwargs = dict(column_kwargs or {})
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--models",
        *list(model_pair),
        "--phase-bins",
        str(int(phase_bins)),
        "--train-cycles",
        str(int(train_cycles)),
        "--eval-cycles",
        str(int(eval_cycles)),
        "--stretch",
        str(int(stretch)),
        "--compression-stride",
        str(int(compression_stride)),
        "--resolution",
        str(int(resolution[0])),
        str(int(resolution[1])),
        "--n-minicolumns",
        str(int(column_kwargs.get("n_minicolumns", 512))),
        "--sparsity",
        str(float(column_kwargs.get("sparsity", 0.05))),
        "--seed",
        str(int(column_kwargs.get("seed", 42))),
        "--boundary-active-threshold",
        str(float(boundary_active_threshold)),
        "--object-decoder-tail-steps",
        str(int(object_decoder_tail_steps)),
        "--lm-family",
        str(lm_family),
        "--detail-grid-shape",
        str(int(detail_grid_shape[0])),
        str(int(detail_grid_shape[1])),
    ]
    if use_detail_aware_sensors:
        cmd.append("--use-detail-aware-sensors")
    if fully_self_supervised_lms:
        cmd.append("--fully-self-supervised-lms")

    completed = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _run_pairwise_child_diagnostics(
    models,
    phase_bins,
    train_cycles,
    eval_cycles,
    stretch,
    compression_stride,
    resolution,
    column_kwargs,
    morphology_lm_kwargs,
    behavior_lm_kwargs,
    parent_column_kwargs,
    parent_lm_kwargs,
    boundary_active_threshold,
    fully_self_supervised_lms,
    object_decoder_tail_steps,
    lm_family,
    detail_grid_shape,
    use_detail_aware_sensors,
):
    pairs = list(combinations(list(models), 2))
    if not pairs:
        return {"pair_count": 0, "pairs": {}, "focus_pairs": {}}

    pair_reports = {}
    for model_pair in pairs:
        pair_report = None
        if not _is_predictive_hypothesis_family(lm_family):
            try:
                pair_report = _run_pairwise_benchmark_subprocess(
                    model_pair=model_pair,
                    phase_bins=phase_bins,
                    train_cycles=train_cycles,
                    eval_cycles=eval_cycles,
                    stretch=stretch,
                    compression_stride=compression_stride,
                    resolution=resolution,
                    column_kwargs=deepcopy(column_kwargs),
                    morphology_lm_kwargs=deepcopy(morphology_lm_kwargs),
                    behavior_lm_kwargs=deepcopy(behavior_lm_kwargs),
                    parent_column_kwargs=deepcopy(parent_column_kwargs),
                    parent_lm_kwargs=deepcopy(parent_lm_kwargs),
                    boundary_active_threshold=boundary_active_threshold,
                    fully_self_supervised_lms=fully_self_supervised_lms,
                    object_decoder_tail_steps=object_decoder_tail_steps,
                    lm_family=lm_family,
                    detail_grid_shape=detail_grid_shape,
                    use_detail_aware_sensors=use_detail_aware_sensors,
                )
            except Exception:
                pair_report = None

        if pair_report is None:
            pair_report = run_benchmark(
                models=list(model_pair),
                phase_bins=phase_bins,
                train_cycles=train_cycles,
                eval_cycles=eval_cycles,
                stretch=stretch,
                compression_stride=compression_stride,
                resolution=resolution,
                column_kwargs=deepcopy(column_kwargs),
                morphology_lm_kwargs=deepcopy(morphology_lm_kwargs),
                behavior_lm_kwargs=deepcopy(behavior_lm_kwargs),
                parent_column_kwargs=deepcopy(parent_column_kwargs),
                parent_lm_kwargs=deepcopy(parent_lm_kwargs),
                boundary_active_threshold=boundary_active_threshold,
                fully_self_supervised_lms=fully_self_supervised_lms,
                object_decoder_tail_steps=object_decoder_tail_steps,
                include_parent_replay_diagnostics=False,
                lm_family=lm_family,
                detail_grid_shape=detail_grid_shape,
                use_detail_aware_sensors=use_detail_aware_sensors,
            )
        pair_reports[_pairwise_model_key(model_pair)] = _summarize_pairwise_benchmark(
            pair_report
        )

    focus_pairs = {}
    focus_map = {
        "behavior": ("fox", "robot"),
        "morphology": ("fox", "cesiumman"),
    }
    for focus_name, focus_pair in focus_map.items():
        focus_key = _pairwise_model_key(focus_pair)
        if focus_key in pair_reports:
            focus_pairs[focus_name] = pair_reports[focus_key]

    return {
        "pair_count": len(pair_reports),
        "pairs": pair_reports,
        "focus_pairs": focus_pairs,
    }


def run_benchmark(
    models=None,
    phase_bins=4,
    train_cycles=1,
    eval_cycles=1,
    stretch=2,
    compression_stride=2,
    resolution=(32, 32),
    column_kwargs=None,
    morphology_lm_kwargs=None,
    behavior_lm_kwargs=None,
    parent_column_kwargs=None,
    parent_lm_kwargs=None,
    boundary_active_threshold=BOUNDARY_PRESSURE_ACTIVE_THRESHOLD,
    fully_self_supervised_lms=False,
    object_decoder_tail_steps=5,
    include_parent_replay_diagnostics=False,
    lm_family="cortical_column_torch",
    detail_grid_shape=(5, 5),
    use_detail_aware_sensors=False,
):
    _configure_benchmark_threading()

    lm_family = str(lm_family)
    if lm_family not in VALID_LM_FAMILIES:
        raise ValueError(f"Unknown lm_family: {lm_family}")
    requested_resolution = tuple(int(value) for value in resolution)
    resolution = _apply_family_resolution_defaults(requested_resolution, lm_family)
    detail_grid_shape = _normalize_detail_grid_shape(detail_grid_shape)
    use_detail_aware_sensors = bool(use_detail_aware_sensors)

    model_names = list(models or ("fox", "cesiumman", "robot"))
    column_kwargs = dict(column_kwargs or {})
    benchmark_seed = int(column_kwargs.get("seed", 42))
    column_kwargs.setdefault("seed", benchmark_seed)
    morphology_column_kwargs = {}
    behavior_column_kwargs = {}
    if fully_self_supervised_lms and "defer_sensor_auto_label" not in column_kwargs:
        morphology_column_kwargs["defer_sensor_auto_label"] = True
        behavior_column_kwargs["defer_sensor_auto_label"] = True

    random.seed(benchmark_seed)
    np.random.seed(benchmark_seed)
    torch.manual_seed(benchmark_seed)

    specs = []
    for name in model_names:
        if name not in MODEL_SPECS:
            raise ValueError(f"Unknown model spec: {name}")
        spec = dict(MODEL_SPECS[name])
        spec = _apply_family_model_geometry_defaults(spec, lm_family)
        spec["name"] = name
        specs.append(spec)

    missing = [str(spec["path"]) for spec in specs if not spec["path"].is_file()]
    if missing:
        raise FileNotFoundError(f"Missing model assets: {missing}")

    initial_spec = specs[0]
    if "n_minicolumns" not in column_kwargs:
        column_kwargs["n_minicolumns"] = 512
    if "sparsity" not in column_kwargs:
        column_kwargs["sparsity"] = 0.05
    parent_column_kwargs = dict(parent_column_kwargs or {})
    parent_column_kwargs.setdefault("use_location_feature_memory", True)
    parent_lm_kwargs = dict(parent_lm_kwargs or {})
    parent_lm_kwargs.setdefault("evidence_match_threshold", 1.0)
    parent_lm_kwargs.setdefault("goal_state_min_steps", 3)
    parent_lm_kwargs.setdefault(
        "goal_state_min_separation_ratio",
        parent_lm_kwargs.get("evidence_separation_ratio", 1.5),
    )
    if _is_predictive_hypothesis_family(lm_family):
        parent_column_kwargs.setdefault(
            "n_minicolumns",
            int(column_kwargs.get("n_minicolumns", 512)),
        )
        if morphology_lm_kwargs is None:
            morphology_lm_kwargs = {}
        if behavior_lm_kwargs is None:
            behavior_lm_kwargs = {}

    morphology_lm_kwargs, behavior_lm_kwargs = _resolve_child_lm_kwargs(
        morphology_lm_kwargs=morphology_lm_kwargs,
        behavior_lm_kwargs=behavior_lm_kwargs,
    )
    morphology_lm_kwargs = _apply_family_lm_defaults(
        lm_family,
        morphology_lm_kwargs,
    )
    behavior_lm_kwargs = _apply_family_lm_defaults(
        lm_family,
        behavior_lm_kwargs,
    )
    parent_lm_kwargs = _apply_family_lm_defaults(
        lm_family,
        parent_lm_kwargs,
        is_parent=True,
    )

    if _is_predictive_hypothesis_family(lm_family):
        child_lm_track12_matrix = {
            "family": lm_family,
            "in_parity": None,
            "reason": "track12_child_temporal_configs_do_not_apply_to_track14_core",
            "morphology": {},
            "behavior": {},
        }
    else:
        child_lm_track12_matrix = _build_child_lm_track12_matrix(
            morphology_lm_kwargs=morphology_lm_kwargs,
            behavior_lm_kwargs=behavior_lm_kwargs,
        )

    def _make_experiment(spec, state_dict=None):
        exp = Panda3DTorchExperiment(
            model_path=spec["path"],
            hierarchical=True,
            resolution=resolution,
            far=20.0,
            initial_distance=spec["initial_distance"],
            object_scale=spec["object_scale"],
            flow_threshold=1e-6,
            asset_search_paths=[str(ASSET_DIR)],
            column_kwargs=deepcopy(column_kwargs),
            morphology_column_kwargs=deepcopy(morphology_column_kwargs),
            behavior_column_kwargs=deepcopy(behavior_column_kwargs),
            parent_column_kwargs=deepcopy(parent_column_kwargs),
            morphology_lm_kwargs=deepcopy(morphology_lm_kwargs),
            behavior_lm_kwargs=deepcopy(behavior_lm_kwargs),
            parent_lm_kwargs=deepcopy(parent_lm_kwargs),
            lm_family=lm_family,
            detail_grid_shape=detail_grid_shape,
            use_detail_aware_sensors=use_detail_aware_sensors,
            motor_actions=[
                MoveForward,
                MoveTangentially,
                TurnLeft,
                TurnRight,
                LookUp,
                LookDown,
            ],
            allow_backward_actions=True,
            tangential_directions=[
                (-1.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.0, 0.0, -1.0),
            ],
            goal_state_driven_actions=True,
            authoritative_goal_sender_ids=["lm_parent"],
            conditional_voting=True,
            vote_confident_threshold=2,
            vote_cooldown_steps=10000,
            allow_action_sampler_fallback=True,
        )
        exp._setup()
        if state_dict is not None:
            exp.monty.load_state_dict(deepcopy(state_dict))
        return exp

    training_specs = []
    raw_eval_results = {}
    saved_state = None
    final_parent_state_dict = None

    for spec in specs:
        phase_exp = _make_experiment(spec, state_dict=saved_state)
        try:
            anim_name = _select_animation_name(
                phase_exp,
                preferred=spec["preferred_animation"],
            )
            n_frames = phase_exp._anim_obj.get_num_frames(anim_name)
            phase_info = _extract_phase_info(
                phase_exp._anim_obj,
                anim_name=anim_name,
                n_phase_bins=phase_bins,
                preferred_method=spec["preferred_phase_method"],
            )
        finally:
            phase_exp.close()

        train_schedule = _build_schedule(n_frames, cycles=1)
        train_object_name = None if fully_self_supervised_lms else spec["name"]
        max_training_attempts = (
            TRACK14_TRAINING_MAX_ATTEMPTS
            if (
                _is_predictive_hypothesis_family(lm_family)
                and train_object_name is not None
            )
            else 1
        )

        for training_attempt in range(1, max_training_attempts + 1):
            exp = _make_experiment(spec, state_dict=saved_state)
            try:
                anim_name = _select_animation_name(
                    exp,
                    preferred=spec["preferred_animation"],
                )
                training_trace = []

                for _ in range(train_cycles):
                    result = exp.run_episode(
                        mode=ExperimentMode.TRAIN,
                        object_name=train_object_name,
                        anim_name=anim_name,
                        frame_schedule=train_schedule,
                        collect_trace=True,
                    )
                    training_trace.extend(result.get("trace", []))

                missing_learning_modules = _learning_modules_missing_target_object(
                    exp,
                    train_object_name,
                    lm_family,
                )
                if missing_learning_modules:
                    if training_attempt < max_training_attempts:
                        continue
                    missing_str = ", ".join(sorted(missing_learning_modules))
                    raise RuntimeError(
                        "Track 14 training did not learn target "
                        f"{train_object_name} after {max_training_attempts} attempts "
                        f"(missing: {missing_str})"
                    )

                train_phase_targets = _phase_targets_for_schedule(
                    phase_info["phase_by_frame"],
                    train_schedule * train_cycles,
                )
                latent_decoder = _fit_latent_phase_decoder(
                    training_trace,
                    train_phase_targets,
                )
                training_diagnostics = _summarize_training_diagnostics(
                    exp,
                    training_trace,
                    latent_decoder,
                )
                training_diagnostics["attempts_used"] = training_attempt
                training_diagnostics["camera_initialization"] = dict(
                    getattr(exp, "_last_camera_init_debug", {}) or {}
                )
                training_graph_id_to_target = _snapshot_graph_id_target_mappings(exp)
                omission_segment = _find_phase_segment(
                    n_frames,
                    phase_bins,
                    phase_info.get("phase_by_frame"),
                )

                training_specs.append(
                    {
                        **spec,
                        "animation": anim_name,
                        "n_frames": n_frames,
                        "phase_info": phase_info,
                        "latent_decoder": latent_decoder,
                        "training_diagnostics": training_diagnostics,
                        "training_graph_id_to_target": training_graph_id_to_target,
                        "omission_segment": omission_segment,
                    }
                )
                saved_state = deepcopy(exp.monty.state_dict())
                final_parent_state_dict = deepcopy(
                    exp.monty.learning_modules[2].state_dict()
                )
                break
            finally:
                exp.close()

    for spec in training_specs:
        matched_schedule = _build_schedule(spec["n_frames"], cycles=eval_cycles)
        stretched_schedule = _build_schedule(
            spec["n_frames"],
            cycles=eval_cycles,
            stretch=stretch,
        )
        compressed_schedule = _build_strided_schedule(
            spec["n_frames"],
            cycles=eval_cycles,
            stride=compression_stride,
        )
        omission_schedule, omission_windows = _build_omission_schedule(
            spec["n_frames"],
            cycles=eval_cycles,
            segment=spec["omission_segment"],
        )
        perturb_schedule, perturb_windows = _build_perturbed_schedule(
            spec["n_frames"],
            cycles=eval_cycles,
            segment=spec["omission_segment"],
        )
        max_eval_attempts = (
            TRACK14_TRAINING_MAX_ATTEMPTS
            if _is_predictive_hypothesis_family(lm_family)
            else 1
        )

        for eval_attempt in range(1, max_eval_attempts + 1):
            exp = _make_experiment(spec, state_dict=saved_state)
            try:
                matched = exp.run_episode(
                    mode=ExperimentMode.EVAL,
                    anim_name=spec["animation"],
                    frame_schedule=matched_schedule,
                    collect_trace=True,
                    collect_context_signals=include_parent_replay_diagnostics,
                    collect_action_history=True,
                )
                if (
                    _is_predictive_hypothesis_family(lm_family)
                    and not _result_has_processed_learning_step(matched)
                ):
                    if eval_attempt < max_eval_attempts:
                        continue
                    raise RuntimeError(
                        "Track 14 evaluation did not produce a processed LM step "
                        f"for {spec['name']} after {max_eval_attempts} attempts"
                    )

                matched_action_history = matched.get("action_history", [])
                matched_action_stats = _compute_action_history_stats(
                    matched_action_history,
                    matched.get("action_context_history", []),
                )
                matched_camera_initialization = dict(
                    getattr(exp, "_last_camera_init_debug", {}) or {}
                )
                active_replay = exp.run_episode(
                    mode=ExperimentMode.EVAL,
                    anim_name=spec["animation"],
                    frame_schedule=matched_schedule,
                    collect_trace=True,
                    forced_action_sequences=matched_action_history,
                    action_context_mode="executed",
                )
                action_blind_replay = exp.run_episode(
                    mode=ExperimentMode.EVAL,
                    anim_name=spec["animation"],
                    frame_schedule=matched_schedule,
                    collect_trace=True,
                    forced_action_sequences=matched_action_history,
                    action_context_mode="none",
                )
                stretched = exp.run_episode(
                    mode=ExperimentMode.EVAL,
                    anim_name=spec["animation"],
                    frame_schedule=stretched_schedule,
                    collect_trace=True,
                )
                compressed = exp.run_episode(
                    mode=ExperimentMode.EVAL,
                    anim_name=spec["animation"],
                    frame_schedule=compressed_schedule,
                    collect_trace=True,
                )
                omitted = exp.run_episode(
                    mode=ExperimentMode.EVAL,
                    anim_name=spec["animation"],
                    frame_schedule=omission_schedule,
                    collect_trace=True,
                )
                perturbed = exp.run_episode(
                    mode=ExperimentMode.EVAL,
                    anim_name=spec["animation"],
                    frame_schedule=perturb_schedule,
                    collect_trace=True,
                )

                raw_eval_results[spec["name"]] = {
                    "matched": matched,
                    "matched_schedule": matched_schedule,
                    "matched_eval_attempts_used": eval_attempt,
                    "matched_camera_initialization": matched_camera_initialization,
                    "matched_action_stats": matched_action_stats,
                    "active_replay": active_replay,
                    "action_blind_replay": action_blind_replay,
                    "stretched": stretched,
                    "stretched_schedule": stretched_schedule,
                    "compressed": compressed,
                    "compressed_schedule": compressed_schedule,
                    "omitted": omitted,
                    "omission_schedule": omission_schedule,
                    "omission_windows": omission_windows,
                    "perturbed": perturbed,
                    "perturb_schedule": perturb_schedule,
                    "perturb_windows": perturb_windows,
                }
                break
            finally:
                exp.close()

        # Wait until all per-model eval results are collected before fitting
        # decoders and assembling the benchmark report.
        if spec != training_specs[-1]:
            continue

        joint_object_decoder = {}
        joint_object_decoder_diagnostics = {}
        use_latent_object_decoders = (
            fully_self_supervised_lms
            or _is_predictive_hypothesis_family(lm_family)
        )
        if use_latent_object_decoders:
            training_graph_id_to_target_by_model = {
                spec["name"]: spec.get("training_graph_id_to_target", {})
                for spec in training_specs
            }
            object_decoders, object_decoder_diagnostics = (
                _fit_self_supervised_object_decoders(
                    raw_eval_results,
                    tail_steps=object_decoder_tail_steps,
                    training_graph_id_to_target_by_model=training_graph_id_to_target_by_model,
                )
            )
            joint_object_decoder, joint_object_decoder_diagnostics = (
                _fit_self_supervised_joint_object_decoder(
                    raw_eval_results,
                    tail_steps=object_decoder_tail_steps,
                )
            )
        else:
            object_decoders = {}
            object_decoder_diagnostics = {}

        per_model = {}

        for spec in training_specs:
            raw_eval = raw_eval_results[spec["name"]]
            matched = raw_eval["matched"]
            matched_schedule = raw_eval["matched_schedule"]
            matched_eval_attempts_used = raw_eval["matched_eval_attempts_used"]
            matched_camera_initialization = raw_eval[
                "matched_camera_initialization"
            ]
            matched_action_stats = raw_eval["matched_action_stats"]
            active_replay = raw_eval["active_replay"]
            action_blind_replay = raw_eval["action_blind_replay"]
            stretched = raw_eval["stretched"]
            stretched_schedule = raw_eval["stretched_schedule"]
            compressed = raw_eval["compressed"]
            compressed_schedule = raw_eval["compressed_schedule"]
            omitted = raw_eval["omitted"]
            omission_schedule = raw_eval["omission_schedule"]
            omission_windows = raw_eval["omission_windows"]
            perturbed = raw_eval["perturbed"]
            perturb_schedule = raw_eval["perturb_schedule"]
            perturb_windows = raw_eval["perturb_windows"]

            matched_phase_targets = _phase_targets_for_schedule(
                spec["phase_info"]["phase_by_frame"],
                matched_schedule,
            )
            stretched_phase_targets = _phase_targets_for_schedule(
                spec["phase_info"]["phase_by_frame"],
                stretched_schedule,
            )
            compressed_phase_targets = _phase_targets_for_schedule(
                spec["phase_info"]["phase_by_frame"],
                compressed_schedule,
            )
            omission_phase_targets = _phase_targets_for_schedule(
                spec["phase_info"]["phase_by_frame"],
                omission_schedule,
            )
            perturb_phase_targets = _phase_targets_for_schedule(
                spec["phase_info"]["phase_by_frame"],
                perturb_schedule,
            )

            matched_report = _condition_report(
                matched,
                matched_phase_targets,
                spec["latent_decoder"],
                spec["name"],
                boundary_active_threshold=boundary_active_threshold,
                object_decoders=object_decoders,
                object_decoder_diagnostics=object_decoder_diagnostics,
                joint_object_decoder=joint_object_decoder,
                joint_object_decoder_diagnostics=joint_object_decoder_diagnostics,
            )
            matched_best_lm_id, matched_lm_trace_coverage = (
                _select_best_available_trace_lm(matched.get("trace", []))
            )
            matched_report["interference_debug"] = _compute_interference_debug(
                matched.get("trace", []),
                spec["name"],
                object_decoder=object_decoders.get("lm_behavior"),
            )
            matched_report["child_interference_debug"] = {
                lm_id: _compute_interference_debug(
                    matched.get("trace", []),
                    spec["name"],
                    target_lm_id=lm_id,
                    object_decoder=object_decoders.get(lm_id),
                )
                for lm_id in ("lm_morphology", "lm_behavior", "lm_parent")
            }
            matched_report["lm_trace_coverage"] = matched_lm_trace_coverage
            matched_report["interference_debug_best_lm_id"] = matched_best_lm_id
            matched_report["interference_debug_best_available"] = (
                _compute_interference_debug(
                    matched.get("trace", []),
                    spec["name"],
                    target_lm_id=matched_best_lm_id,
                    object_decoder=object_decoders.get(matched_best_lm_id),
                )
            )
            active_replay_report = _condition_report(
                active_replay,
                matched_phase_targets,
                spec["latent_decoder"],
                spec["name"],
                boundary_active_threshold=boundary_active_threshold,
                object_decoders=object_decoders,
                object_decoder_diagnostics=object_decoder_diagnostics,
                joint_object_decoder=joint_object_decoder,
                joint_object_decoder_diagnostics=joint_object_decoder_diagnostics,
            )
            action_blind_replay_report = _condition_report(
                action_blind_replay,
                matched_phase_targets,
                spec["latent_decoder"],
                spec["name"],
                boundary_active_threshold=boundary_active_threshold,
                object_decoders=object_decoders,
                object_decoder_diagnostics=object_decoder_diagnostics,
                joint_object_decoder=joint_object_decoder,
                joint_object_decoder_diagnostics=joint_object_decoder_diagnostics,
            )
            stretched_report = _condition_report(
                stretched,
                stretched_phase_targets,
                spec["latent_decoder"],
                spec["name"],
                boundary_active_threshold=boundary_active_threshold,
                object_decoders=object_decoders,
                object_decoder_diagnostics=object_decoder_diagnostics,
                joint_object_decoder=joint_object_decoder,
                joint_object_decoder_diagnostics=joint_object_decoder_diagnostics,
            )
            compressed_report = _condition_report(
                compressed,
                compressed_phase_targets,
                spec["latent_decoder"],
                spec["name"],
                boundary_active_threshold=boundary_active_threshold,
                object_decoders=object_decoders,
                object_decoder_diagnostics=object_decoder_diagnostics,
                joint_object_decoder=joint_object_decoder,
                joint_object_decoder_diagnostics=joint_object_decoder_diagnostics,
            )
            omission_report = _condition_report(
                omitted,
                omission_phase_targets,
                spec["latent_decoder"],
                spec["name"],
                boundary_active_threshold=boundary_active_threshold,
                object_decoders=object_decoders,
                object_decoder_diagnostics=object_decoder_diagnostics,
                joint_object_decoder=joint_object_decoder,
                joint_object_decoder_diagnostics=joint_object_decoder_diagnostics,
            )
            perturbation_report = _condition_report(
                perturbed,
                perturb_phase_targets,
                spec["latent_decoder"],
                spec["name"],
                boundary_active_threshold=boundary_active_threshold,
                object_decoders=object_decoders,
                object_decoder_diagnostics=object_decoder_diagnostics,
                joint_object_decoder=joint_object_decoder,
                joint_object_decoder_diagnostics=joint_object_decoder_diagnostics,
            )

            per_model[spec["name"]] = {
                "asset": str(spec["path"]),
                "animation": spec["animation"],
                "object_scale": list(spec["object_scale"]),
                "initial_distance": spec["initial_distance"],
                "phase_method": spec["phase_info"].get("phase_method"),
                "phase_joint_names": spec["phase_info"]["joint_names"],
                "phase_root_joint": spec["phase_info"]["root_joint_name"],
                "phase_segment_used_for_omission": spec["omission_segment"],
                "omission_segment_selection_mode": spec["omission_segment"].get(
                    "selection_mode"
                ),
                "matched_eval_attempts_used": matched_eval_attempts_used,
                "matched_camera_initialization": matched_camera_initialization,
                "matched_action_stats": matched_action_stats,
                "latent_phase_decoder": spec["latent_decoder"],
                "object_identity_decoder": dict(object_decoder_diagnostics),
                "alias_source_diagnostics": _summarize_lm_alias_source_for_model(
                    spec["name"],
                    matched,
                    training_graph_id_to_target=spec.get(
                        "training_graph_id_to_target",
                        {},
                    ),
                    training_reporting_alias_diagnostics=(
                        (spec.get("training_diagnostics") or {}).get(
                            "reporting_alias_diagnostics",
                            {},
                        )
                    ),
                    object_decoder_diagnostics=object_decoder_diagnostics,
                ),
                "training": spec["training_diagnostics"],
                "matched": matched_report,
                "active_replay": {
                    **active_replay_report,
                    "vs_matched": {
                        "primary_deltas": _compute_primary_metric_deltas(
                            active_replay_report["primary"],
                            matched_report["primary"],
                        ),
                        "trace_disagreement": _compute_temporal_trace_disagreement(
                            active_replay.get("trace", []),
                            matched.get("trace", []),
                        ),
                    },
                },
                "action_blind_replay": {
                    **action_blind_replay_report,
                    "vs_active_replay": {
                        "primary_deltas": _compute_primary_metric_deltas(
                            action_blind_replay_report["primary"],
                            active_replay_report["primary"],
                        ),
                        "trace_disagreement": _compute_temporal_trace_disagreement(
                            action_blind_replay.get("trace", []),
                            active_replay.get("trace", []),
                        ),
                    },
                },
                "stretched": stretched_report,
                "compressed": compressed_report,
                "omission": {
                    **omission_report,
                    "windowed_primary": _compute_windowed_deltas(
                        omitted.get("trace", []),
                        matched.get("trace", []),
                        omission_windows,
                        boundary_active_threshold=boundary_active_threshold,
                    ),
                },
                "perturbation": {
                    **perturbation_report,
                    "windowed_primary": _compute_windowed_deltas(
                        perturbed.get("trace", []),
                        matched.get("trace", []),
                        perturb_windows,
                        boundary_active_threshold=boundary_active_threshold,
                    ),
                },
            }

            per_model[spec["name"]]["matched"].pop("action_history", None)

        matched_primary_summary = _prediction_accuracy_summary(
            report["matched"]["primary"] for report in per_model.values()
        )
        matched_morphology_summary = _prediction_accuracy_summary(
            report["matched"]["child_lm_primary"]["lm_morphology"]
            for report in per_model.values()
        )
        matched_behavior_summary = _prediction_accuracy_summary(
            report["matched"]["child_lm_primary"]["lm_behavior"]
            for report in per_model.values()
        )
        matched_parent_summary = _prediction_accuracy_summary(
            report["matched"]["lm_object_predictions"]["lm_parent"]
            for report in per_model.values()
        )
        stretched_primary_summary = _prediction_accuracy_summary(
            report["stretched"]["primary"] for report in per_model.values()
        )
        compressed_primary_summary = _prediction_accuracy_summary(
            report["compressed"]["primary"] for report in per_model.values()
        )

        aggregate = {
            "n_models": len(per_model),
            "training_attempts_used_max": max(
                report["training"].get("attempts_used", 1)
                for report in per_model.values()
            ),
            "matched_eval_attempts_used_max": max(
                report.get("matched_eval_attempts_used", 1)
                for report in per_model.values()
            ),
            "matched_top1_accuracy_mean": matched_primary_summary[
                "accuracy_mean"
            ],
            "matched_top1_resolved_fraction_mean": matched_primary_summary[
                "resolved_fraction_mean"
            ],
            "matched_top1_strict_accuracy_mean": matched_primary_summary[
                "strict_accuracy_mean"
            ],
            "morphology_matched_top1_accuracy_mean": matched_morphology_summary[
                "accuracy_mean"
            ],
            "morphology_matched_top1_resolved_fraction_mean": (
                matched_morphology_summary["resolved_fraction_mean"]
            ),
            "morphology_matched_top1_strict_accuracy_mean": (
                matched_morphology_summary["strict_accuracy_mean"]
            ),
            "behavior_matched_top1_accuracy_mean": matched_behavior_summary[
                "accuracy_mean"
            ],
            "behavior_matched_top1_resolved_fraction_mean": (
                matched_behavior_summary["resolved_fraction_mean"]
            ),
            "behavior_matched_top1_strict_accuracy_mean": (
                matched_behavior_summary["strict_accuracy_mean"]
            ),
            "parent_matched_top1_accuracy_mean": matched_parent_summary[
                "accuracy_mean"
            ],
            "parent_matched_top1_resolved_fraction_mean": (
                matched_parent_summary["resolved_fraction_mean"]
            ),
            "parent_matched_top1_strict_accuracy_mean": matched_parent_summary[
                "strict_accuracy_mean"
            ],
            "stretched_top1_accuracy_mean": stretched_primary_summary[
                "accuracy_mean"
            ],
            "stretched_top1_resolved_fraction_mean": stretched_primary_summary[
                "resolved_fraction_mean"
            ],
            "stretched_top1_strict_accuracy_mean": stretched_primary_summary[
                "strict_accuracy_mean"
            ],
            "compressed_top1_accuracy_mean": compressed_primary_summary[
                "accuracy_mean"
            ],
            "compressed_top1_resolved_fraction_mean": compressed_primary_summary[
                "resolved_fraction_mean"
            ],
            "compressed_top1_strict_accuracy_mean": compressed_primary_summary[
                "strict_accuracy_mean"
            ],
            "omission_surprise_delta_mean": _mean(
                report["omission"]["windowed_primary"]["mean_surprise_delta"]
                for report in per_model.values()
            ),
            "perturbation_surprise_delta_mean": _mean(
                report["perturbation"]["windowed_primary"]["mean_surprise_delta"]
                for report in per_model.values()
            ),
            "omission_boundary_pressure_delta_mean": _mean(
                report["omission"]["windowed_primary"][
                    "mean_boundary_pressure_delta"
                ]
                for report in per_model.values()
            ),
            "perturbation_boundary_pressure_delta_mean": _mean(
                report["perturbation"]["windowed_primary"][
                    "mean_boundary_pressure_delta"
                ]
                for report in per_model.values()
            ),
            "omission_boundary_active_fraction_delta_mean": _mean(
                report["omission"]["windowed_primary"][
                    "mean_boundary_active_fraction_delta"
                ]
                for report in per_model.values()
            ),
            "perturbation_boundary_active_fraction_delta_mean": _mean(
                report["perturbation"]["windowed_primary"][
                    "mean_boundary_active_fraction_delta"
                ]
                for report in per_model.values()
            ),
            "omission_recovery_steps_mean": _mean(
                report["omission"]["windowed_primary"][
                    "mean_recovery_steps_to_confident"
                ]
                for report in per_model.values()
            ),
            "perturbation_recovery_steps_mean": _mean(
                report["perturbation"]["windowed_primary"][
                    "mean_recovery_steps_to_confident"
                ]
                for report in per_model.values()
            ),
            "behavior_current_phase_accuracy_matched_mean": _mean(
                report["matched"]["secondary"]["current_phase_accuracy"]
                for report in per_model.values()
            ),
            "behavior_current_phase_accuracy_stretched_mean": _mean(
                report["stretched"]["secondary"]["current_phase_accuracy"]
                for report in per_model.values()
            ),
            "behavior_current_phase_accuracy_compressed_mean": _mean(
                report["compressed"]["secondary"]["current_phase_accuracy"]
                for report in per_model.values()
            ),
            "matched_base_target_win_fraction_mean": _mean(
                report["matched"]["interference_debug"]["base_target_win_fraction"]
                for report in per_model.values()
            ),
            "matched_after_temporal_behavior_target_win_fraction_mean": _mean(
                report["matched"]["interference_debug"][
                    "after_temporal_behavior_target_win_fraction"
                ]
                for report in per_model.values()
            ),
            "matched_final_target_win_fraction_mean": _mean(
                report["matched"]["interference_debug"]["final_target_win_fraction"]
                for report in per_model.values()
            ),
            "matched_base_to_final_flip_fraction_mean": _mean(
                report["matched"]["interference_debug"]["base_to_final_flip_fraction"]
                for report in per_model.values()
            ),
            "matched_behavior_to_final_flip_fraction_mean": _mean(
                report["matched"]["interference_debug"][
                    "behavior_to_final_flip_fraction"
                ]
                for report in per_model.values()
            ),
            "matched_target_lost_after_temporal_behavior_fraction_mean": _mean(
                report["matched"]["interference_debug"][
                    "target_lost_after_temporal_behavior_fraction"
                ]
                for report in per_model.values()
            ),
            "matched_target_lost_after_self_supervised_fraction_mean": _mean(
                report["matched"]["interference_debug"][
                    "target_lost_after_self_supervised_fraction"
                ]
                for report in per_model.values()
            ),
            "matched_joint_hypothesis_candidate_fraction_mean": _mean(
                report["matched"]["interference_debug"][
                    "joint_hypothesis_candidate_fraction"
                ]
                for report in per_model.values()
            ),
            "matched_joint_top_chart_resolved_fraction_mean": _mean(
                report["matched"]["interference_debug"][
                    "joint_top_chart_resolved_fraction"
                ]
                for report in per_model.values()
            ),
            "matched_joint_top_latent_multi_chart_fraction_mean": _mean(
                report["matched"]["interference_debug"][
                    "joint_top_latent_multi_chart_fraction"
                ]
                for report in per_model.values()
            ),
            "matched_joint_chart_switch_fraction_mean": _mean(
                report["matched"]["interference_debug"][
                    "joint_chart_switch_fraction"
                ]
                for report in per_model.values()
            ),
            "matched_joint_same_latent_chart_switch_fraction_mean": _mean(
                report["matched"]["interference_debug"][
                    "joint_same_latent_chart_switch_fraction"
                ]
                for report in per_model.values()
            ),
            "matched_joint_top_context_ranked_chart_prior_fraction_mean": _mean(
                report["matched"]["interference_debug"][
                    "joint_top_context_ranked_chart_prior_fraction"
                ]
                for report in per_model.values()
            ),
            "matched_joint_top_vote_ranked_chart_prior_fraction_mean": _mean(
                report["matched"]["interference_debug"][
                    "joint_top_vote_ranked_chart_prior_fraction"
                ]
                for report in per_model.values()
            ),
            "matched_joint_dominant_chart_fraction_mean": _mean(
                report["matched"]["interference_debug"][
                    "joint_dominant_chart_fraction"
                ]
                for report in per_model.values()
            ),
            "matched_action_context_nonzero_fraction_mean": _mean(
                report["matched_action_stats"]["nonzero_action_context_fraction"]
                for report in per_model.values()
            ),
            "action_blind_surprise_delta_mean": _mean(
                report["action_blind_replay"]["vs_active_replay"][
                    "primary_deltas"
                ]["surprise_mean_delta"]
                for report in per_model.values()
            ),
            "action_blind_boundary_pressure_delta_mean": _mean(
                report["action_blind_replay"]["vs_active_replay"][
                    "primary_deltas"
                ]["boundary_pressure_mean_delta"]
                for report in per_model.values()
            ),
            "action_blind_boundary_active_fraction_delta_mean": _mean(
                report["action_blind_replay"]["vs_active_replay"][
                    "primary_deltas"
                ]["boundary_active_fraction_delta"]
                for report in per_model.values()
            ),
            "action_blind_confident_fraction_delta_mean": _mean(
                report["action_blind_replay"]["vs_active_replay"][
                    "primary_deltas"
                ]["confident_fraction_delta"]
                for report in per_model.values()
            ),
            "action_blind_current_label_disagreement_fraction_mean": _mean(
                report["action_blind_replay"]["vs_active_replay"][
                    "trace_disagreement"
                ]["current_label_disagreement_fraction"]
                for report in per_model.values()
            ),
            "action_blind_predicted_label_disagreement_fraction_mean": _mean(
                report["action_blind_replay"]["vs_active_replay"][
                    "trace_disagreement"
                ]["predicted_label_disagreement_fraction"]
                for report in per_model.values()
            ),
        }

        report = {
            "models": model_names,
            "phase_bins": phase_bins,
            "train_cycles": train_cycles,
            "eval_cycles": eval_cycles,
            "stretch": stretch,
            "compression_stride": compression_stride,
            "resolution": list(resolution),
            "requested_resolution": list(requested_resolution),
            "lm_family": lm_family,
            "detail_grid_shape": list(detail_grid_shape),
            "use_detail_aware_sensors": use_detail_aware_sensors,
            "column_kwargs": dict(column_kwargs),
            "morphology_column_kwargs": dict(morphology_column_kwargs),
            "behavior_column_kwargs": dict(behavior_column_kwargs),
            "parent_column_kwargs": dict(parent_column_kwargs),
            "benchmark_seed": benchmark_seed,
            "boundary_active_threshold": float(boundary_active_threshold),
            "real_assets_only": True,
            "core_temporal_state_mode": (
                "predictive_hypothesis_message_state"
                if _is_predictive_hypothesis_family(lm_family)
                else "trace_bank_d_t"
            ),
            "input_geometry_mode": (
                "detail_packet_plus_internal_object_state"
                if _is_predictive_hypothesis_family(lm_family)
                else "inferred_relative_hidden_state"
            ),
            "trace_biased_inference": (
                False if _is_predictive_hypothesis_family(lm_family) else True
            ),
            "child_lm_temporal_parity": child_lm_track12_matrix["in_parity"],
            "child_lm_track12_matrix": child_lm_track12_matrix,
            "temporal_trace_config": dict(
                behavior_lm_kwargs.get("temporal_trace_config", {})
            ),
            "action_context_falsifier_mode": (
                "forced_action_replay_with_efference_ablation"
            ),
            "full_active_passive_test": False,
            "temporal_learning_mode": (
                "local_predictive_hypothesis_online_update"
                if _is_predictive_hypothesis_family(lm_family)
                else "self_supervised_predictive_trace_state"
            ),
            "temporal_state_provider_used": False,
            "phase_labels_used_in_training": False,
            "motor_control_mode": (
                "lm_parent_goal_state_driven_with_exploratory_fallback"
            ),
            "motor_action_space": [
                "move_forward",
                "move_backward",
                "move_left",
                "move_right",
                "move_up",
                "move_down",
                "turn_left",
                "turn_right",
                "look_up",
                "look_down",
            ],
            "action_sampler_fallback_enabled": True,
            "goal_state_driven_motor": True,
            "goal_state_authority": ["lm_parent"],
            "parent_goal_location_memory_enabled": bool(
                parent_column_kwargs.get("use_location_feature_memory", False)
            ),
            "lateral_voting_enabled": True,
            "lateral_voting_mode": "conditional_settle_gated_top_k",
            "vote_after_steps": int(exp.monty.vote_after_steps),
            "vote_confident_threshold": int(exp.monty.vote_confident_threshold),
            "vote_cooldown_steps": int(exp.monty.vote_cooldown_steps),
            "fully_self_supervised_lms": bool(fully_self_supervised_lms),
            "object_identity_training_mode": (
                "self_supervised_auto_label_posthoc_decoded"
                if fully_self_supervised_lms
                else "named_object_graph_training"
            ),
            "object_identity_disambiguation_mode": (
                "joint_child_signature_then_parent_fallback_on_ambiguous_graph_id"
                if fully_self_supervised_lms
                else "none"
            ),
            "object_identity_ambiguity_reporting_mode": (
                "ambiguous_graph_ids_remain_unresolved_and_are_excluded_from_accuracy"
                if fully_self_supervised_lms
                else "none"
            ),
            "object_identity_decoder_tail_steps": int(object_decoder_tail_steps),
            "object_identity_decoder": dict(object_decoder_diagnostics),
            "joint_object_identity_decoder": dict(
                joint_object_decoder_diagnostics
            ),
            "per_model": per_model,
            "aggregate": aggregate,
        }

        if include_parent_replay_diagnostics and final_parent_state_dict is not None:
            parent_state_dict = deepcopy(final_parent_state_dict)
            report.setdefault("diagnostics", {})[
                "parent_context_replay"
            ] = _build_parent_context_replay_diagnostics(
                per_model=per_model,
                raw_eval_results=raw_eval_results,
                parent_state_dict=parent_state_dict,
                parent_column_kwargs=parent_column_kwargs,
                parent_lm_kwargs=parent_lm_kwargs,
                object_decoders=object_decoders,
                lm_family=lm_family,
                boundary_active_threshold=boundary_active_threshold,
            )

        return report


def run_diagnostic_harness(
    models=None,
    phase_bins=4,
    train_cycles=1,
    eval_cycles=1,
    stretch=2,
    compression_stride=2,
    resolution=(32, 32),
    column_kwargs=None,
    morphology_lm_kwargs=None,
    behavior_lm_kwargs=None,
    parent_column_kwargs=None,
    parent_lm_kwargs=None,
    boundary_active_threshold=BOUNDARY_PRESSURE_ACTIVE_THRESHOLD,
    fully_self_supervised_lms=False,
    object_decoder_tail_steps=5,
    lm_family="cortical_column_torch",
    detail_grid_shape=(5, 5),
    use_detail_aware_sensors=False,
):
    report = run_benchmark(
        models=models,
        phase_bins=phase_bins,
        train_cycles=train_cycles,
        eval_cycles=eval_cycles,
        stretch=stretch,
        compression_stride=compression_stride,
        resolution=resolution,
        column_kwargs=deepcopy(column_kwargs),
        morphology_lm_kwargs=deepcopy(morphology_lm_kwargs),
        behavior_lm_kwargs=deepcopy(behavior_lm_kwargs),
        parent_column_kwargs=deepcopy(parent_column_kwargs),
        parent_lm_kwargs=deepcopy(parent_lm_kwargs),
        boundary_active_threshold=boundary_active_threshold,
        fully_self_supervised_lms=fully_self_supervised_lms,
        object_decoder_tail_steps=object_decoder_tail_steps,
        include_parent_replay_diagnostics=True,
        lm_family=lm_family,
        detail_grid_shape=detail_grid_shape,
        use_detail_aware_sensors=use_detail_aware_sensors,
    )

    report.setdefault("diagnostics", {})["pairwise_child_only"] = (
        _run_pairwise_child_diagnostics(
            models=report["models"],
            phase_bins=phase_bins,
            train_cycles=train_cycles,
            eval_cycles=eval_cycles,
            stretch=stretch,
            compression_stride=compression_stride,
            resolution=resolution,
            column_kwargs=column_kwargs,
            morphology_lm_kwargs=morphology_lm_kwargs,
            behavior_lm_kwargs=behavior_lm_kwargs,
            parent_column_kwargs=parent_column_kwargs,
            parent_lm_kwargs=parent_lm_kwargs,
            boundary_active_threshold=boundary_active_threshold,
            fully_self_supervised_lms=fully_self_supervised_lms,
            object_decoder_tail_steps=object_decoder_tail_steps,
            lm_family=lm_family,
            detail_grid_shape=detail_grid_shape,
            use_detail_aware_sensors=use_detail_aware_sensors,
        )
    )
    return report


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
    parser.add_argument("--compression-stride", type=int, default=2)
    parser.add_argument("--resolution", nargs=2, type=int, default=[32, 32])
    parser.add_argument("--n-minicolumns", type=int, default=512)
    parser.add_argument("--sparsity", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--boundary-active-threshold",
        type=float,
        default=BOUNDARY_PRESSURE_ACTIVE_THRESHOLD,
    )
    parser.add_argument(
        "--fully-self-supervised-lms",
        action="store_true",
    )
    parser.add_argument(
        "--object-decoder-tail-steps",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--diagnostic-harness",
        action="store_true",
    )
    parser.add_argument(
        "--lm-family",
        default="cortical_column_torch",
        choices=VALID_LM_FAMILIES,
    )
    parser.add_argument(
        "--detail-grid-shape",
        nargs=2,
        type=int,
        default=[5, 5],
    )
    parser.add_argument(
        "--use-detail-aware-sensors",
        action="store_true",
    )
    args = parser.parse_args()

    runner = run_diagnostic_harness if args.diagnostic_harness else run_benchmark
    report = runner(
        models=args.models,
        phase_bins=args.phase_bins,
        train_cycles=args.train_cycles,
        eval_cycles=args.eval_cycles,
        stretch=args.stretch,
        compression_stride=args.compression_stride,
        resolution=tuple(args.resolution),
        column_kwargs={
            "n_minicolumns": args.n_minicolumns,
            "sparsity": args.sparsity,
            "seed": args.seed,
        },
        boundary_active_threshold=args.boundary_active_threshold,
        fully_self_supervised_lms=args.fully_self_supervised_lms,
        object_decoder_tail_steps=args.object_decoder_tail_steps,
        lm_family=args.lm_family,
        detail_grid_shape=tuple(args.detail_grid_shape),
        use_detail_aware_sensors=args.use_detail_aware_sensors,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
