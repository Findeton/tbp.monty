"""Utilities for extracting motion-grounded phase labels from animated meshes.

The goal is to derive per-frame phase labels from actual joint kinematics,
not from uniform frame buckets. This keeps Track 9b supervision tied to the
real mesh motion in the animation.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


_PRIMARY_JOINT_GROUPS = (
    (("left", "foot"), ("left", "toe"), ("left", "paw"), ("left", "ankle")),
    (("right", "foot"), ("right", "toe"), ("right", "paw"), ("right", "ankle")),
    (("left", "hand"), ("left", "wrist"), ("left", "arm")),
    (("right", "hand"), ("right", "wrist"), ("right", "arm")),
)
_GENERIC_DISTAL_TOKENS = ("foot", "toe", "paw", "hand", "wrist", "ankle")
_GENERIC_LIMB_TOKENS = ("leg", "arm")
_FOOT_TOKENS = ("foot", "toe", "paw", "ankle")
_AXIS_NAMES = ("x", "y", "z")
_LEFT_SIDE_MARKERS = ("left", ".l", "_l", "-l", " l")
_RIGHT_SIDE_MARKERS = ("right", ".r", "_r", "-r", " r")


def _normalize_name(name: str) -> str:
    return str(name).lower()


def _is_root_joint(name: str) -> bool:
    lower = _normalize_name(name)
    return (
        "root" in lower
        or lower.startswith("hips")
        or lower.endswith("hips")
        or "pelvis" in lower
        or "torso" in lower
        or lower == "body"
    )


def _has_side_token(name: str, side: str) -> bool:
    lower = _normalize_name(name)
    markers = _LEFT_SIDE_MARKERS if side == "left" else _RIGHT_SIDE_MARKERS
    return any(marker in lower for marker in markers)


def _joint_specificity_score(name: str) -> tuple[int, int, int]:
    lower = _normalize_name(name)
    distal_score = sum(token in lower for token in _GENERIC_DISTAL_TOKENS)
    limb_score = sum(token in lower for token in _GENERIC_LIMB_TOKENS)

    digits = []
    current = ""
    for ch in str(name):
        if ch.isdigit():
            current += ch
        elif current:
            digits.append(int(current))
            current = ""
    if current:
        digits.append(int(current))

    numeric_score = max(digits) if digits else 0
    return distal_score, limb_score, numeric_score


def infer_root_joint(joint_names: Sequence[str]) -> str:
    """Pick the most likely skeleton root joint name."""
    candidates = [name for name in joint_names if _is_root_joint(name)]
    if not candidates:
        if not joint_names:
            raise ValueError("Could not infer a root joint from the animation")
        return str(joint_names[0])

    for name in candidates:
        if not str(name).startswith("_"):
            return name
    return candidates[0]


def select_phase_joints(
    joint_names: Sequence[str],
    max_joints: int = 4,
) -> list[str]:
    """Choose distal limb joints that best capture cyclic gait motion."""
    if max_joints <= 0:
        return []

    normalized = {name: _normalize_name(name) for name in joint_names}
    selected: list[str] = []

    grouped_candidates = [
        ("left", ("foot", "toe", "paw", "ankle")),
        ("right", ("foot", "toe", "paw", "ankle")),
        ("left", ("hand", "wrist", "arm", "leg")),
        ("right", ("hand", "wrist", "arm", "leg")),
    ]

    for side, tokens in grouped_candidates:
        candidates = []
        for name, lower in normalized.items():
            if name in selected:
                continue
            if not _has_side_token(lower, side):
                continue
            if any(token in lower for token in tokens):
                candidates.append(name)
        if candidates:
            selected.append(max(candidates, key=lambda name: (_joint_specificity_score(name), name)))
        if len(selected) >= max_joints:
            return selected[:max_joints]

    generic = []
    for name, lower in normalized.items():
        if name in selected or _is_root_joint(name):
            continue
        if any(token in lower for token in _GENERIC_DISTAL_TOKENS):
            generic.append(name)
    for name in generic:
        if len(selected) >= max_joints:
            break
        selected.append(name)

    if len(selected) < max_joints:
        fallback = []
        for name, lower in normalized.items():
            if name in selected or _is_root_joint(name):
                continue
            if any(token in lower for token in _GENERIC_LIMB_TOKENS):
                fallback.append(name)
        for name in fallback:
            if len(selected) >= max_joints:
                break
            selected.append(name)

    if not selected:
        non_root = [name for name in joint_names if not _is_root_joint(name)]
        selected = list(non_root[:max_joints])

    return selected[:max_joints]


def select_foot_joints(joint_names: Sequence[str]) -> tuple[str, str]:
    """Choose one left and one right distal foot joint when available."""
    normalized = {name: _normalize_name(name) for name in joint_names}

    def _pick(side: str) -> str | None:
        candidates = [
            name
            for name, lower in normalized.items()
            if _has_side_token(lower, side)
            and any(token in lower for token in _FOOT_TOKENS)
        ]
        if not candidates:
            candidates = [
                name
                for name, lower in normalized.items()
                if _has_side_token(lower, side)
                and "leg" in lower
            ]
        if not candidates:
            return None
        return max(candidates, key=lambda name: (_joint_specificity_score(name), name))

    left_joint = _pick("left")
    right_joint = _pick("right")
    if left_joint is None or right_joint is None:
        raise ValueError("Could not identify left/right distal foot joints")
    return left_joint, right_joint


def _collect_joint_positions(
    animated_obj,
    anim_name: str,
    joint_names: Sequence[str],
    root_joint_name: str,
    part_name: str = "modelRoot",
) -> np.ndarray:
    root_joint = animated_obj.expose_joint(root_joint_name, part_name=part_name)
    exposed_joints = [
        animated_obj.expose_joint(name, part_name=part_name)
        for name in joint_names
    ]

    n_frames = int(animated_obj.get_num_frames(anim_name))
    trajectory = []
    for frame in range(n_frames):
        animated_obj.pose(frame, anim_name)
        if getattr(animated_obj, "_base", None) is not None:
            animated_obj._base.taskMgr.step()

        features = []
        for joint in exposed_joints:
            pos = joint.getPos(root_joint)
            features.append((float(pos.x), float(pos.y), float(pos.z)))
        trajectory.append(features)

    return np.asarray(trajectory, dtype=np.float64)


def _phase_from_coordinates(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    phase_angles = np.mod(np.arctan2(coords[:, 1], coords[:, 0]), 2 * np.pi)
    phase_angles = np.mod(phase_angles - phase_angles[0], 2 * np.pi)
    unwrapped = np.unwrap(phase_angles)
    if np.mean(np.diff(unwrapped)) < 0:
        phase_angles = np.mod(-phase_angles, 2 * np.pi)
        unwrapped = np.unwrap(phase_angles)
    return phase_angles, unwrapped


def _quantize_phase(phase_angles: np.ndarray, n_phase_bins: int) -> np.ndarray:
    return np.floor(phase_angles * n_phase_bins / (2 * np.pi)).astype(int) % n_phase_bins


def extract_foot_cycle_phase_info(
    animated_obj,
    anim_name: str,
    n_phase_bins: int = 4,
    left_joint_name: str | None = None,
    right_joint_name: str | None = None,
    root_joint_name: str | None = None,
    part_name: str = "modelRoot",
) -> dict:
    """Extract phase bins from left/right distal foot trajectories.

    This is intended for gait-like animations where the most meaningful local
    phase variable comes from the relative motion of the distal feet.
    """
    n_phase_bins = max(1, int(n_phase_bins))

    all_joint_names = list(animated_obj.get_joint_names(part_name=part_name))
    if not all_joint_names:
        raise ValueError("Animated object has no joints to derive a motion phase")

    root_joint_name = root_joint_name or infer_root_joint(all_joint_names)
    if left_joint_name is None or right_joint_name is None:
        inferred_left, inferred_right = select_foot_joints(all_joint_names)
        left_joint_name = left_joint_name or inferred_left
        right_joint_name = right_joint_name or inferred_right

    positions = _collect_joint_positions(
        animated_obj,
        anim_name=anim_name,
        joint_names=[left_joint_name, right_joint_name],
        root_joint_name=root_joint_name,
        part_name=part_name,
    )
    left_positions = positions[:, 0, :]
    right_positions = positions[:, 1, :]

    axis_energy = np.var(left_positions, axis=0) + np.var(right_positions, axis=0)
    axis_order = np.argsort(axis_energy)[::-1]
    primary_axis = int(axis_order[0])
    secondary_axis = int(axis_order[1]) if len(axis_order) > 1 else int(axis_order[0])

    primary_signal = left_positions[:, primary_axis] - right_positions[:, primary_axis]
    secondary_signal = left_positions[:, secondary_axis] - right_positions[:, secondary_axis]

    primary_std = max(float(primary_signal.std()), 1e-8)
    secondary_std = max(float(secondary_signal.std()), 1e-8)
    coords = np.column_stack(
        [
            (primary_signal - primary_signal.mean()) / primary_std,
            (secondary_signal - secondary_signal.mean()) / secondary_std,
        ]
    )

    phase_angles, unwrapped = _phase_from_coordinates(coords)
    phase_by_frame = _quantize_phase(phase_angles, n_phase_bins)

    return {
        "joint_names": [left_joint_name, right_joint_name],
        "root_joint_name": root_joint_name,
        "phase_by_frame": phase_by_frame.tolist(),
        "phase_angles": phase_angles.tolist(),
        "phase_span": float(unwrapped[-1] - unwrapped[0]) if len(unwrapped) > 0 else 0.0,
        "primary_axis": _AXIS_NAMES[primary_axis],
        "secondary_axis": _AXIS_NAMES[secondary_axis],
        "phase_method": "foot_cycle_angle",
    }


def extract_joint_phase_info(
    animated_obj,
    anim_name: str,
    n_phase_bins: int = 4,
    joint_names: Sequence[str] | None = None,
    root_joint_name: str | None = None,
    part_name: str = "modelRoot",
) -> dict:
    """Extract cyclic motion phase bins from real joint trajectories.

    The method treats the animation as a loop in joint-configuration space,
    projects the loop into its top two principal components, then uses the
    polar angle around that loop as a continuous phase variable.
    """
    n_phase_bins = max(1, int(n_phase_bins))

    all_joint_names = list(animated_obj.get_joint_names(part_name=part_name))
    if not all_joint_names:
        raise ValueError("Animated object has no joints to derive a motion phase")

    root_joint_name = root_joint_name or infer_root_joint(all_joint_names)
    joint_names = list(joint_names) if joint_names is not None else select_phase_joints(all_joint_names)
    if not joint_names:
        raise ValueError("Could not select any joints for motion-phase extraction")

    data = _collect_joint_positions(
        animated_obj,
        anim_name=anim_name,
        joint_names=joint_names,
        root_joint_name=root_joint_name,
        part_name=part_name,
    ).reshape(int(animated_obj.get_num_frames(anim_name)), -1)
    centered = data - data.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)

    if vt.shape[0] >= 2 and singular_values[1] > 1e-8:
        coords = centered @ vt[:2].T
    else:
        primary = centered[:, :1]
        secondary = np.gradient(primary[:, 0])[:, None]
        coords = np.concatenate([primary, secondary], axis=1)

    phase_angles, unwrapped = _phase_from_coordinates(coords)
    phase_by_frame = _quantize_phase(phase_angles, n_phase_bins)

    return {
        "joint_names": joint_names,
        "root_joint_name": root_joint_name,
        "phase_by_frame": phase_by_frame.tolist(),
        "phase_angles": phase_angles.tolist(),
        "phase_span": float(unwrapped[-1] - unwrapped[0]) if len(unwrapped) > 0 else 0.0,
        "phase_method": "joint_configuration_pca",
    }