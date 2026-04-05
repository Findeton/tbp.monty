#!/usr/bin/env python

"""Benchmark a valid two-view multi-LM Panda3D setup on YCB objects.

This script addresses the main flaw in the earlier two-LM Panda3D runner:
it does not duplicate one physical patch stream into multiple logical sensor
modules. Instead, each LM receives observations from its own real Panda3D
camera with a distinct initial azimuth around the object.

The benchmark compares three conditions under the same LM-driven motor policy:

- ``single_view``: one camera, one LM
- ``two_view_no_voting``: two real cameras, two LMs, no lateral votes
- ``two_view_hopfield``: same two real cameras with surprise-gated lateral votes

This is meant to be the first clean control for the hypothesis that multi-LM
only helps when the columns receive complementary, alignable evidence.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

import run_full_monty_torch_panda3d as base

from tbp.monty.frameworks.actions.action_samplers import ConstantSampler
from tbp.monty.frameworks.actions.actions import (
    LookDown,
    LookUp,
    MoveForward,
    TurnLeft,
    TurnRight,
)
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.environment_utils.transforms import DepthTo3DLocations
from tbp.monty.frameworks.environments.embodied_data import (
    EnvironmentInterfacePerObject,
)
from tbp.monty.frameworks.environments.object_init_samplers import Predefined
from tbp.monty.frameworks.experiments.object_recognition_experiments import (
    MontyObjectRecognitionExperiment,
)
from tbp.monty.frameworks.loggers.monty_handlers import BasicCSVStatsHandler
from tbp.monty.frameworks.models.abstract_monty_classes import Observations
from tbp.monty.frameworks.models.cortical_column_torch.learning_module import (
    CorticalColumnTorchLM,
)
from tbp.monty.frameworks.models.evidence_matching.model import (
    MontyForEvidenceGraphMatching,
)
from tbp.monty.frameworks.models.motor_policies import MultiAgentInformedPolicy
from tbp.monty.frameworks.models.motor_system import MotorSystem
from tbp.monty.frameworks.models.sensor_modules import CameraSM
from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.simulators.panda3d.agents import Panda3DAgent
from tbp.monty.simulators.panda3d.transforms import Panda3DDepthNormalize


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


AGENT_IDS = [AgentID("agent_id_0"), AgentID("agent_id_1")]
SENSOR_IDS = [SensorID("patch_0"), SensorID("patch_1")]
SM_IDS = ["patch_0", "patch_1"]

PERFORMANCE_PRECEDENCE = [
    "patch_off_object",
    "no_label",
    "pose_time_out",
    "time_out",
    "consistent_child_obj",
    "confused_mlh",
    "correct_mlh",
    "no_match",
    "confused",
    "correct",
]
PERFORMANCE_RANK = {
    performance: idx for idx, performance in enumerate(PERFORMANCE_PRECEDENCE)
}


def _limit_sequence(values, limit):
    seq = list(values)
    if limit is None or limit <= 0:
        return seq
    return seq[:limit]


def _look_at(eye: tuple[float, float, float], target: tuple[float, float, float]):
    eye = np.array(eye, dtype=np.float64)
    target = np.array(target, dtype=np.float64)
    forward = target - eye
    norm = np.linalg.norm(forward)
    if norm < 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    forward /= norm

    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(np.dot(forward, world_up)) > 0.99:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right) + 1e-12
    up = np.cross(right, forward)

    mat = np.eye(3)
    mat[:, 0] = right
    mat[:, 1] = forward
    mat[:, 2] = up

    q = Rotation.from_matrix(mat).as_quat()
    return (float(q[3]), float(q[0]), float(q[1]), float(q[2]))


def _orbit_pose(
    azimuth_deg: float,
    orbit_radius: float,
    elevation_deg: float,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    azimuth_rad = math.radians(azimuth_deg)
    elevation_rad = math.radians(elevation_deg)

    dx = orbit_radius * math.cos(elevation_rad) * math.sin(azimuth_rad)
    dy = -orbit_radius * math.cos(elevation_rad) * math.cos(azimuth_rad)
    dz = orbit_radius * math.sin(elevation_rad)

    position = (
        center[0] + dx,
        center[1] + dy,
        center[2] + dz,
    )
    rotation = _look_at(position, center)
    return position, rotation


def build_transform(agent_sensor_specs, resolution, fov, near, far):
    transform = []
    for agent_id, sensor_id in agent_sensor_specs:
        transform.append(
            Panda3DDepthNormalize(agent_id=agent_id, near=near, far=far)
        )
        transform.append(
            DepthTo3DLocations(
                agent_id=agent_id,
                sensor_ids=[sensor_id],
                resolutions=[resolution],
                hfov=fov,
                world_coord=True,
                get_all_points=True,
            )
        )
    return transform


def aggregate_episode_stats(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "episode_seed",
        "primary_target_object",
        "primary_target_rotation_euler",
    ]

    def pick_outcome(series: pd.Series) -> str:
        return max(series.tolist(), key=lambda x: PERFORMANCE_RANK.get(x, -1))

    return (
        df.groupby(group_cols, dropna=False)
        .agg(
            system_outcome=("primary_performance", pick_outcome),
            monty_matching_steps=("monty_matching_steps", "max"),
        )
        .reset_index()
    )


def get_objects_in_memory(exp: MontyObjectRecognitionExperiment) -> dict:
    diagnostics = {}

    for idx, lm in enumerate(exp.model.learning_modules):
        column = lm.column
        lfm = column._lfm

        if lfm is not None:
            pattern_counts = {
                object_id: int(count)
                for object_id, count in pd.Series(lfm._object_ids[: lfm.n_stored])
                .value_counts()
                .sort_index()
                .items()
            }
            lfm_known_objects = sorted(lfm.known_objects)
            lfm_n_stored = int(lfm.n_stored)
        else:
            pattern_counts = {}
            lfm_known_objects = []
            lfm_n_stored = 0

        diagnostics[f"LM_{idx}"] = {
            "learning_module_id": lm.learning_module_id,
            "known_objects": sorted(lm.get_all_known_object_ids()),
            "lfm_known_objects": lfm_known_objects,
            "associative_memory_objects": sorted(
                column._associative_memory.known_objects
            ),
            "lfm_pattern_counts": pattern_counts,
            "lfm_n_stored": lfm_n_stored,
            "hopfield_n_stored": int(column._hopfield.n_stored),
        }

    return diagnostics


def summarize_per_lm_eval(df: pd.DataFrame) -> dict:
    per_lm = {}

    for lm_id, lm_df in df.groupby("lm_id"):
        outcome_counts = lm_df["primary_performance"].value_counts().to_dict()
        outcomes_by_target = (
            lm_df.groupby(["primary_target_object", "primary_performance"])
            .size()
            .unstack(fill_value=0)
            .to_dict(orient="index")
        )
        predictions_by_target = (
            lm_df.groupby(["primary_target_object", "most_likely_object"])
            .size()
            .unstack(fill_value=0)
            .to_dict(orient="index")
        )
        mean_steps_by_target = {
            target: float(value)
            for target, value in lm_df.groupby("primary_target_object")[
                "num_steps"
            ].mean().items()
        }

        per_lm[lm_id] = {
            "outcome_counts": outcome_counts,
            "outcomes_by_target": outcomes_by_target,
            "predictions_by_target": predictions_by_target,
            "mean_num_steps_by_target": mean_steps_by_target,
        }

    return per_lm


def summarize_lm_agreement(df: pd.DataFrame) -> dict:
    group_cols = [
        "episode_seed",
        "primary_target_object",
        "primary_target_rotation_euler",
    ]
    prediction_df = (
        df.groupby(group_cols + ["lm_id"])["most_likely_object"]
        .first()
        .unstack("lm_id")
        .reset_index()
    )

    lm_cols = [col for col in prediction_df.columns if col.startswith("LM_")]
    if len(lm_cols) < 2:
        return {}

    disagreement_mask = prediction_df[lm_cols].nunique(axis=1) > 1
    disagreement_df = prediction_df.loc[disagreement_mask]

    return {
        "episode_disagreement_count": int(disagreement_mask.sum()),
        "episode_count": int(len(prediction_df)),
        "disagreement_by_target": disagreement_df.groupby(
            "primary_target_object"
        ).size().to_dict(),
    }


def summarize_eval(output_dir: Path, exp: MontyObjectRecognitionExperiment) -> dict:
    csv_path = output_dir / "eval_stats.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"eval_stats.csv not found at {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise RuntimeError(f"eval_stats.csv is empty at {csv_path}")

    episode_df = aggregate_episode_stats(df)
    total = len(episode_df)
    correct = (episode_df["system_outcome"] == "correct").sum()
    correct_mlh = (episode_df["system_outcome"] == "correct_mlh").sum()
    confused = (episode_df["system_outcome"] == "confused").sum()
    converged = episode_df["system_outcome"].isin(["correct", "confused"]).sum()
    converged_steps = episode_df.loc[
        episode_df["system_outcome"].isin(["correct", "confused"]),
        "monty_matching_steps",
    ]

    return {
        "total_eval_episodes": int(total),
        "correct": int(correct),
        "correct_mlh": int(correct_mlh),
        "confused": int(confused),
        "converged": int(converged),
        "accuracy": float(correct / total) if total else None,
        "converged_fraction": float(converged / total) if total else None,
        "mean_converged_steps": (
            float(converged_steps.mean()) if not converged_steps.empty else None
        ),
        "objects_in_memory": {
            lm.learning_module_id: sorted(lm.get_all_known_object_ids())
            for lm in exp.model.learning_modules
        },
        "motor_control": "lm_driven_multi_agent_informed_policy",
        "objects_in_memory_by_lm_id": get_objects_in_memory(exp),
        "per_lm_eval": summarize_per_lm_eval(df),
        "lm_agreement": summarize_lm_agreement(df),
        "output_dir": str(output_dir),
    }


def build_config(
    condition: str,
    object_names: list[str],
    output_dir: Path,
    train_rotations,
    eval_rotations,
    max_train_steps: int,
    max_eval_steps: int,
    orbit_radius: float,
    elevation_deg: float,
    azimuth_step_deg: float,
    view_offset_deg: float,
    hopfield_surprise_threshold: float,
    hopfield_vote_after_steps: int,
    vote_cooldown_steps: int,
    min_lms_match: int,
    lm_novelty_thresholds: dict[int, float | None] | None = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    if condition == "single_view":
        active_agent_ids = [AGENT_IDS[0]]
        active_sensor_ids = [SENSOR_IDS[0]]
        active_sm_ids = [SM_IDS[0]]
        offsets = {AGENT_IDS[0]: 0.0}
        voting_matrix = [[]]
        hopfield_voting = False
    elif condition in {"two_view_no_voting", "two_view_hopfield"}:
        active_agent_ids = AGENT_IDS
        active_sensor_ids = SENSOR_IDS
        active_sm_ids = SM_IDS
        offsets = {
            AGENT_IDS[0]: 0.0,
            AGENT_IDS[1]: float(view_offset_deg),
        }
        voting_matrix = [[1], [0]] if condition == "two_view_hopfield" else [[], []]
        hopfield_voting = condition == "two_view_hopfield"
    else:
        raise ValueError(f"Unknown condition: {condition}")

    agent_sensor_specs = list(zip(active_agent_ids, active_sensor_ids))
    transform = build_transform(
        agent_sensor_specs,
        resolution=base.RESOLUTION,
        fov=base.FOV,
        near=base.NEAR,
        far=base.FAR,
    )

    agents = []
    for agent_id, sensor_id in agent_sensor_specs:
        position, rotation = _orbit_pose(
            azimuth_deg=offsets[agent_id],
            orbit_radius=orbit_radius,
            elevation_deg=elevation_deg,
        )
        agents.append(
            Panda3DAgent(
                agent_id=agent_id,
                sensor_id=str(sensor_id),
                resolution=base.RESOLUTION,
                fov=base.FOV,
                position=position,
                rotation=rotation,
            )
        )

    env_interface_config = {
        "env_init_func": base.Panda3DEnvWithLookAt,
        "env_init_args": {
            "agents": agents,
            "near": base.NEAR,
            "far": base.FAR,
            "object_registry": base.build_ycb_registry(object_names),
        },
        "transform": transform,
    }

    train_env_interface_args = {
        "object_names": object_names,
        "object_init_sampler": Predefined(
            positions=[(0.0, 0.0, 0.0)],
            rotations=train_rotations,
            scales=[(1.0, 1.0, 1.0)],
            change_every_episode=False,
        ),
    }

    eval_env_interface_args = {
        "object_names": object_names,
        "object_init_sampler": Predefined(
            positions=[(0.0, 0.0, 0.0)],
            rotations=eval_rotations,
            scales=[(1.0, 1.0, 1.0)],
            change_every_episode=False,
        ),
    }

    sensor_module_configs = {}
    learning_module_configs = {}
    sm_to_lm_matrix = []
    sm_to_agent_dict = {}
    sender_to_agent_dict = {}

    for idx, (agent_id, sensor_id, sm_id) in enumerate(
        zip(active_agent_ids, active_sensor_ids, active_sm_ids)
    ):
        sensor_module_configs[f"sensor_module_{idx}"] = {
            "sensor_module_class": CameraSM,
            "sensor_module_args": {
                "sensor_module_id": sm_id,
                "features": [
                    "pose_vectors",
                    "pose_fully_defined",
                    "on_object",
                    "hsv",
                    "principal_curvatures_log",
                ],
                "save_raw_obs": False,
            },
        }

        column_kwargs = copy.deepcopy(base.COLUMN_KWARGS)
        column_kwargs["seed"] = 42 + idx
        novelty_override = (lm_novelty_thresholds or {}).get(idx)
        if novelty_override is not None:
            column_kwargs.setdefault("lfm_kwargs", {})[
                "novelty_threshold"
            ] = novelty_override

        learning_module_configs[f"learning_module_{idx}"] = {
            "learning_module_class": CorticalColumnTorchLM,
            "learning_module_args": {
                "column_kwargs": column_kwargs,
                "hopfield_voting": hopfield_voting,
                "surprise_vote_threshold": hopfield_surprise_threshold,
            },
        }

        sm_to_lm_matrix.append([idx])
        sm_to_agent_dict[sm_id] = agent_id
        sender_to_agent_dict[sm_id] = agent_id
        sender_to_agent_dict[f"learning_module_{idx}"] = agent_id

    monty_config = {
        "monty_class": MontyForEvidenceGraphMatching,
        "monty_args": {
            "min_eval_steps": 5,
            "min_train_steps": 5,
            "num_exploratory_steps": max_train_steps,
            "max_total_steps": max(max_train_steps, max_eval_steps) * 25,
        },
        "sensor_module_configs": sensor_module_configs,
        "learning_module_configs": learning_module_configs,
        "motor_system_config": {
            "motor_system_class": MotorSystem,
            "motor_system_args": {
                "policy": MultiAgentInformedPolicy(
                    action_sampler=ConstantSampler(
                        actions=[MoveForward, TurnLeft, TurnRight, LookUp, LookDown],
                        rotation_degrees=azimuth_step_deg,
                        translation_distance=0.004,
                    ),
                    agent_ids=active_agent_ids,
                    view_finder_ids={
                        agent_id: sm_id
                        for agent_id, sm_id in zip(active_agent_ids, active_sm_ids)
                    },
                    sender_to_agent_dict=sender_to_agent_dict,
                    use_goal_state_driven_actions=True,
                ),
            },
        },
        "sm_to_lm_matrix": sm_to_lm_matrix,
        "lm_to_lm_matrix": [[] for _ in active_sm_ids],
        "lm_to_lm_vote_matrix": voting_matrix,
        "sm_to_agent_dict": sm_to_agent_dict,
        "hopfield_voting": hopfield_voting,
        "hopfield_surprise_threshold": hopfield_surprise_threshold,
        "hopfield_vote_after_steps": hopfield_vote_after_steps,
        "vote_cooldown_steps": vote_cooldown_steps,
    }

    logging_config = {
        "python_log_level": "INFO",
        "python_log_to_file": True,
        "python_log_to_stderr": True,
        "output_dir": str(output_dir),
        "run_name": condition,
        "monty_log_level": "BASIC",
        "monty_handlers": [BasicCSVStatsHandler],
        "wandb_handlers": [],
    }

    return {
        "seed": 42,
        "do_train": True,
        "do_eval": True,
        "max_train_steps": max_train_steps,
        "max_eval_steps": max_eval_steps,
        "max_total_steps": max(max_train_steps, max_eval_steps) * 25,
        "n_train_epochs": len(train_rotations),
        "n_eval_epochs": len(eval_rotations),
        "model_name_or_path": None,
        "min_lms_match": min_lms_match,
        "show_sensor_output": False,
        "supervised_lm_ids": [],
        "logging": logging_config,
        "monty_config": monty_config,
        "env_interface_config": env_interface_config,
        "train_env_interface_class": EnvironmentInterfacePerObject,
        "train_env_interface_args": train_env_interface_args,
        "eval_env_interface_class": EnvironmentInterfacePerObject,
        "eval_env_interface_args": eval_env_interface_args,
    }


def run_condition(condition: str, args) -> dict:
    object_names = _limit_sequence(base.YCB_OBJECTS, args.object_limit)
    train_rotations = _limit_sequence(base.TRAIN_ROTATIONS, args.train_rotation_limit)
    eval_rotations = _limit_sequence(base.EVAL_ROTATIONS, args.eval_rotation_limit)
    output_dir = args.output_root / condition

    logger.info("Condition %s: %d objects", condition, len(object_names))
    config = build_config(
        condition=condition,
        object_names=object_names,
        output_dir=output_dir,
        train_rotations=train_rotations,
        eval_rotations=eval_rotations,
        max_train_steps=args.max_train_steps,
        max_eval_steps=args.max_eval_steps,
        orbit_radius=args.orbit_radius,
        elevation_deg=args.elevation_deg,
        azimuth_step_deg=args.azimuth_step_deg,
        view_offset_deg=args.view_offset_deg,
        hopfield_surprise_threshold=args.hopfield_surprise_threshold,
        hopfield_vote_after_steps=args.hopfield_vote_after_steps,
        vote_cooldown_steps=args.vote_cooldown_steps,
        min_lms_match=args.min_lms_match,
        lm_novelty_thresholds={
            0: args.lm0_novelty_threshold,
            1: args.lm1_novelty_threshold,
        },
    )

    with MontyObjectRecognitionExperiment(config) as exp:
        exp.train()
        objects_in_memory_post_train = get_objects_in_memory(exp)
        exp.evaluate()
        objects_in_memory_post_eval = get_objects_in_memory(exp)

        summary = summarize_eval(output_dir, exp)
        summary.update(
            {
                "condition": condition,
                "objects": object_names,
                "train_rotations": len(train_rotations),
                "eval_rotations": len(eval_rotations),
                "max_train_steps": args.max_train_steps,
                "max_eval_steps": args.max_eval_steps,
                "lm0_novelty_threshold": args.lm0_novelty_threshold,
                "lm1_novelty_threshold": args.lm1_novelty_threshold,
                "objects_in_memory_post_train": objects_in_memory_post_train,
                "objects_in_memory_post_eval": objects_in_memory_post_eval,
            }
        )
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark a valid two-view multi-LM Panda3D YCB setup."
    )
    parser.add_argument(
        "--condition",
        choices=["single_view", "two_view_no_voting", "two_view_hopfield", "all"],
        default="all",
    )
    parser.add_argument("--object-limit", type=int, default=0)
    parser.add_argument("--train-rotation-limit", type=int, default=0)
    parser.add_argument("--eval-rotation-limit", type=int, default=0)
    parser.add_argument("--max-train-steps", type=int, default=120)
    parser.add_argument("--max-eval-steps", type=int, default=120)
    parser.add_argument("--min-lms-match", type=int, default=1)
    parser.add_argument("--orbit-radius", type=float, default=0.30)
    parser.add_argument("--elevation-deg", type=float, default=10.0)
    parser.add_argument("--azimuth-step-deg", type=float, default=5.0)
    parser.add_argument("--view-offset-deg", type=float, default=90.0)
    parser.add_argument("--hopfield-surprise-threshold", type=float, default=0.15)
    parser.add_argument("--hopfield-vote-after-steps", type=int, default=10)
    parser.add_argument("--vote-cooldown-steps", type=int, default=8)
    parser.add_argument("--lm0-novelty-threshold", type=float, default=None)
    parser.add_argument("--lm1-novelty-threshold", type=float, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            Path(os.environ.get("MONTY_RESULTS", "~/tbp/results/monty")).expanduser()
            / "runs"
            / "ycb_torch_panda3d_two_view_benchmark"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    conditions = (
        ["single_view", "two_view_no_voting", "two_view_hopfield"]
        if args.condition == "all"
        else [args.condition]
    )

    all_results = []
    for condition in conditions:
        summary = run_condition(condition, args)
        all_results.append(summary)
        logger.info("%s summary: %s", condition, json.dumps(summary, indent=2))

    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()