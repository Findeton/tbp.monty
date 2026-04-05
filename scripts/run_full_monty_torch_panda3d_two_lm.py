#!/usr/bin/env python
# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT license that can be found in
# the LICENSE file or at https://opensource.org/licenses/MIT.

"""Two-LM Full Monty Panda3D/YCB run with sparse surprise-gated voting.

This keeps the standard MontyObjectRecognitionExperiment pipeline and the same
real Panda3D YCB meshes as the single-LM runner, but fans out one physical
surface sensor stream into two logical CameraSMs:

- geometry LM: pose + curvature driven
- appearance LM: pose + color driven

The two LMs exchange Hopfield votes only when needed:

- sender must be settled (low surprise)
- receiver must still be stuck (high surprise)
- voting starts only after an initial independent-processing window
- repeated votes to the same receiver are rate-limited by a cooldown
"""

from __future__ import annotations

import copy
import logging
import os
from pathlib import Path

import pandas as pd

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
from tbp.monty.frameworks.environments.embodied_data import (
    EnvironmentInterfacePerObject,
)
from tbp.monty.frameworks.environments.object_init_samplers import Predefined
from tbp.monty.frameworks.experiments.object_recognition_experiments import (
    MontyObjectRecognitionExperiment,
)
from tbp.monty.frameworks.loggers.monty_handlers import BasicCSVStatsHandler
from tbp.monty.frameworks.models.cortical_column_torch.learning_module import (
    CorticalColumnTorchLM,
)
from tbp.monty.frameworks.models.evidence_matching.model import (
    MontyForEvidenceGraphMatching,
)
from tbp.monty.frameworks.models.motor_policies import InformedPolicy
from tbp.monty.frameworks.models.motor_system import MotorSystem
from tbp.monty.frameworks.models.sensor_modules import CameraSM
from tbp.monty.frameworks.sensors import SensorID

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

AGENT_ID = AgentID("agent_id_0")
SOURCE_SENSOR_ID = base.SENSOR_ID
SOURCE_SM_ID = base.SM_ID
GEOM_SM_ID = "patch_geometry"
APPEAR_SM_ID = "patch_appearance"

HOPFIELD_SURPRISE_THRESHOLD = 0.3
HOPFIELD_VOTE_AFTER_STEPS = 5
VOTE_COOLDOWN_STEPS = 4
HOPFIELD_VOTING_ENABLED = os.environ.get("MONTY_HOPFIELD_VOTING", "1") != "0"
LM_VOTING_ENABLED = os.environ.get("MONTY_LM_VOTING", "1") != "0"

OUTPUT_DIR = (
    Path(os.environ.get("MONTY_RESULTS", "~/tbp/results/monty")).expanduser()
    / "runs"
    / "ycb_torch_panda3d_two_lm"
)


def _limit_sequence(values, env_name: str):
    limit = int(os.environ.get(env_name, "0"))
    seq = list(values)
    if limit <= 0:
        return seq
    return seq[:limit]


ACTIVE_OBJECTS = _limit_sequence(base.YCB_OBJECTS, "MONTY_OBJECT_LIMIT")
ACTIVE_TRAIN_ROTATIONS = _limit_sequence(
    base.TRAIN_ROTATIONS, "MONTY_TRAIN_ROTATION_LIMIT"
)
ACTIVE_EVAL_ROTATIONS = _limit_sequence(
    base.EVAL_ROTATIONS, "MONTY_EVAL_ROTATION_LIMIT"
)
MAX_TRAIN_STEPS = int(os.environ.get("MONTY_MAX_TRAIN_STEPS", "200"))
MAX_EVAL_STEPS = int(os.environ.get("MONTY_MAX_EVAL_STEPS", "200"))
DO_TRAIN = os.environ.get("MONTY_DO_TRAIN", "1") != "0"
DO_EVAL = os.environ.get("MONTY_DO_EVAL", "1") != "0"
MODEL_NAME_OR_PATH = os.environ.get("MONTY_MODEL_PATH") or None

if "MONTY_HOPFIELD_SURPRISE_THRESHOLD" in os.environ:
    HOPFIELD_SURPRISE_THRESHOLD = float(
        os.environ["MONTY_HOPFIELD_SURPRISE_THRESHOLD"]
    )
if "MONTY_HOPFIELD_VOTE_AFTER_STEPS" in os.environ:
    HOPFIELD_VOTE_AFTER_STEPS = int(
        os.environ["MONTY_HOPFIELD_VOTE_AFTER_STEPS"]
    )
if "MONTY_VOTE_COOLDOWN_STEPS" in os.environ:
    VOTE_COOLDOWN_STEPS = int(os.environ["MONTY_VOTE_COOLDOWN_STEPS"])

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


class Panda3DEnvWithSensorFanout(base.Panda3DEnvWithLookAt):
    """Duplicate the real patch stream under two logical sensor IDs."""

    @staticmethod
    def _fanout_sensor_stream(observations):
        for agent_obs in observations.values():
            if SOURCE_SENSOR_ID not in agent_obs:
                continue
            source_obs = agent_obs[SOURCE_SENSOR_ID]
            agent_obs[SensorID(GEOM_SM_ID)] = source_obs
            agent_obs[SensorID(APPEAR_SM_ID)] = source_obs
        return observations

    @staticmethod
    def _fanout_sensor_state(state):
        for agent_state in state.values():
            if SOURCE_SENSOR_ID not in agent_state.sensors:
                continue
            source_state = agent_state.sensors[SOURCE_SENSOR_ID]
            agent_state.sensors[SensorID(GEOM_SM_ID)] = source_state
            agent_state.sensors[SensorID(APPEAR_SM_ID)] = source_state
        return state

    def reset(self):
        observations, state = super().reset()
        return self._fanout_sensor_stream(observations), self._fanout_sensor_state(state)

    def step(self, actions):
        observations, state = super().step(actions)
        return self._fanout_sensor_stream(observations), self._fanout_sensor_state(state)


def build_config(object_names: list[str]) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    registry = base.build_ycb_registry(object_names)
    transform = base.build_transform()

    env_interface_config = {
        "env_init_func": Panda3DEnvWithSensorFanout,
        "env_init_args": {
            "agents": [
                base.Panda3DAgent(
                    agent_id=AGENT_ID,
                    sensor_id=str(SOURCE_SENSOR_ID),
                    resolution=base.RESOLUTION,
                    fov=base.FOV,
                    position=(0.0, -0.3, 0.0),
                    rotation=(1.0, 0.0, 0.0, 0.0),
                )
            ],
            "near": base.NEAR,
            "far": base.FAR,
            "object_registry": registry,
        },
        "transform": transform,
    }

    train_env_interface_args = {
        "object_names": object_names,
        "object_init_sampler": Predefined(
            positions=[(0.0, 0.0, 0.0)],
            rotations=ACTIVE_TRAIN_ROTATIONS,
            scales=[(1.0, 1.0, 1.0)],
            change_every_episode=False,
        ),
    }

    eval_env_interface_args = {
        "object_names": object_names,
        "object_init_sampler": Predefined(
            positions=[(0.0, 0.0, 0.0)],
            rotations=ACTIVE_EVAL_ROTATIONS,
            scales=[(1.0, 1.0, 1.0)],
            change_every_episode=False,
        ),
    }

    geom_column_kwargs = copy.deepcopy(base.COLUMN_KWARGS)
    geom_column_kwargs["seed"] = 42

    appear_column_kwargs = copy.deepcopy(base.COLUMN_KWARGS)
    appear_column_kwargs["seed"] = 43

    monty_config = {
        "monty_class": MontyForEvidenceGraphMatching,
        "monty_args": {
            "min_eval_steps": 5,
            "min_train_steps": 5,
            "num_exploratory_steps": 200,
            "max_total_steps": 5000,
        },
        "hopfield_voting": HOPFIELD_VOTING_ENABLED,
        "hopfield_surprise_threshold": HOPFIELD_SURPRISE_THRESHOLD,
        "hopfield_vote_after_steps": HOPFIELD_VOTE_AFTER_STEPS,
        "vote_cooldown_steps": VOTE_COOLDOWN_STEPS,
        "sensor_module_configs": {
            "sensor_module_0": {
                "sensor_module_class": CameraSM,
                "sensor_module_args": {
                    "sensor_module_id": GEOM_SM_ID,
                    "features": [
                        "pose_vectors",
                        "pose_fully_defined",
                        "on_object",
                        "principal_curvatures_log",
                    ],
                    "save_raw_obs": False,
                },
            },
            "sensor_module_1": {
                "sensor_module_class": CameraSM,
                "sensor_module_args": {
                    "sensor_module_id": APPEAR_SM_ID,
                    "features": [
                        "pose_vectors",
                        "pose_fully_defined",
                        "on_object",
                        "hsv",
                    ],
                    "save_raw_obs": False,
                },
            },
        },
        "learning_module_configs": {
            "learning_module_0": {
                "learning_module_class": CorticalColumnTorchLM,
                "learning_module_args": {
                    "column_kwargs": geom_column_kwargs,
                    "hopfield_voting": HOPFIELD_VOTING_ENABLED,
                    "surprise_vote_threshold": HOPFIELD_SURPRISE_THRESHOLD,
                },
            },
            "learning_module_1": {
                "learning_module_class": CorticalColumnTorchLM,
                "learning_module_args": {
                    "column_kwargs": appear_column_kwargs,
                    "hopfield_voting": HOPFIELD_VOTING_ENABLED,
                    "surprise_vote_threshold": HOPFIELD_SURPRISE_THRESHOLD,
                },
            },
        },
        "motor_system_config": {
            "motor_system_class": MotorSystem,
            "motor_system_args": {
                "policy": InformedPolicy(
                    action_sampler=ConstantSampler(
                        actions=[MoveForward, TurnLeft, TurnRight, LookUp, LookDown],
                        rotation_degrees=5.0,
                        translation_distance=0.004,
                    ),
                    agent_id=AGENT_ID,
                    use_goal_state_driven_actions=True,
                    view_finder_id=SOURCE_SM_ID,
                ),
            },
        },
        "sm_to_lm_matrix": [[0], [1]],
        "lm_to_lm_matrix": [[], []],
        "lm_to_lm_vote_matrix": [[1], [0]] if LM_VOTING_ENABLED else None,
        "sm_to_agent_dict": {
            GEOM_SM_ID: AGENT_ID,
            APPEAR_SM_ID: AGENT_ID,
        },
    }

    logging_config = {
        "python_log_level": "INFO",
        "python_log_to_file": True,
        "python_log_to_stderr": True,
        "output_dir": str(OUTPUT_DIR),
        "run_name": "ycb_torch_panda3d_two_lm",
        "monty_log_level": "BASIC",
        "monty_handlers": [BasicCSVStatsHandler],
        "wandb_handlers": [],
    }

    return {
        "seed": 42,
        "do_train": DO_TRAIN,
        "do_eval": DO_EVAL,
        "max_train_steps": MAX_TRAIN_STEPS,
        "max_eval_steps": MAX_EVAL_STEPS,
        "max_total_steps": 5000,
        "n_train_epochs": len(ACTIVE_TRAIN_ROTATIONS),
        "n_eval_epochs": len(ACTIVE_EVAL_ROTATIONS),
        "model_name_or_path": MODEL_NAME_OR_PATH,
        "min_lms_match": 1,
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


def aggregate_episode_stats(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["episode_seed", "primary_target_object", "primary_target_rotation_euler"]

    def pick_outcome(series: pd.Series) -> str:
        return max(series.tolist(), key=lambda x: PERFORMANCE_RANK.get(x, -1))

    episode_df = (
        df.groupby(group_cols, dropna=False)
        .agg(
            system_outcome=("primary_performance", pick_outcome),
            monty_matching_steps=("monty_matching_steps", "max"),
        )
        .reset_index()
    )
    return episode_df


def print_summary(exp: MontyObjectRecognitionExperiment, object_names: list[str]):
    logger.info("\n" + "=" * 60)
    logger.info("Two-LM Full Monty Panda3D Results")
    logger.info("=" * 60)
    for lm in exp.model.learning_modules:
        logger.info("%s objects in memory: %s", lm.learning_module_id, lm.get_all_known_object_ids())
    logger.info("Output dir: %s", OUTPUT_DIR)

    logger.info("\n--- Full Monty Checklist ---")
    logger.info("  [OK] MontyObjectRecognitionExperiment pipeline")
    logger.info("  [OK] MontyForEvidenceGraphMatching")
    logger.info("  [OK] Two CameraSM modules on one real Panda3D patch stream")
    logger.info("  [OK] Two CorticalColumnTorchLM columns with shared real YCB input")
    logger.info("  [OK] use_goal_state_driven_actions=True")
    logger.info(
        "  [%s] lateral LM voting enabled",
        "OK" if LM_VOTING_ENABLED else "--",
    )
    logger.info(
        "  [%s] surprise-gated voting with delayed start (%d steps)",
        "OK" if HOPFIELD_VOTING_ENABLED and LM_VOTING_ENABLED else "--",
        HOPFIELD_VOTE_AFTER_STEPS,
    )
    logger.info(
        "  [%s] vote cooldown enabled (%d steps per receiver)",
        "OK" if HOPFIELD_VOTING_ENABLED and LM_VOTING_ENABLED else "--",
        VOTE_COOLDOWN_STEPS,
    )
    logger.info("  [OK] Panda3D environment with real YCB GLB meshes")

    csv_path = OUTPUT_DIR / "eval_stats.csv"
    if not csv_path.exists():
        logger.warning("eval_stats.csv not found at %s", csv_path)
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        logger.warning("eval_stats.csv is empty")
        return

    episode_df = aggregate_episode_stats(df)
    total = len(episode_df)
    converged = episode_df["system_outcome"].isin(["correct", "confused"]).sum()
    correct = (episode_df["system_outcome"] == "correct").sum()
    correct_mlh = (episode_df["system_outcome"] == "correct_mlh").sum()
    confused = (episode_df["system_outcome"] == "confused").sum()
    confused_mlh = (episode_df["system_outcome"] == "confused_mlh").sum()
    patch_off_object = (episode_df["system_outcome"] == "patch_off_object").sum()
    converged_steps = episode_df.loc[
        episode_df["system_outcome"].isin(["correct", "confused"]),
        "monty_matching_steps",
    ]

    logger.info("\n--- Accuracy Results (%d eval episodes) ---", total)
    logger.info(
        "  Converged: %d / %d = %.1f%%",
        converged,
        total,
        100.0 * converged / total,
    )
    logger.info(
        "  Correct:   %d / %d = %.1f%%",
        correct,
        total,
        100.0 * correct / total,
    )
    logger.info(
        "  Best guess accuracy: %d / %d = %.1f%%",
        correct + correct_mlh,
        total,
        100.0 * (correct + correct_mlh) / total,
    )
    if not converged_steps.empty:
        logger.info("  Avg steps in converged episodes: %.2f", converged_steps.mean())
    logger.info(
        "  Outcome breakdown: %s",
        episode_df["system_outcome"].value_counts().to_dict(),
    )
    logger.info(
        "  Per-object breakdown: %s",
        episode_df.groupby("primary_target_object")["system_outcome"]
        .value_counts()
        .unstack(fill_value=0)
        .to_dict(orient="index"),
    )
    logger.info(
        "  Failure detail: confused=%d confused_mlh=%d patch_off_object=%d",
        confused,
        confused_mlh,
        patch_off_object,
    )


def main():
    logger.info("Building 2-LM config for %d YCB objects ...", len(ACTIVE_OBJECTS))
    config = build_config(ACTIVE_OBJECTS)

    logger.info("Instantiating MontyObjectRecognitionExperiment ...")
    exp = MontyObjectRecognitionExperiment(config)
    exp.setup_experiment(config)

    if DO_TRAIN:
        logger.info("Training on %d objects ...", len(ACTIVE_OBJECTS))
        exp.train()

    if DO_EVAL:
        logger.info(
            "Evaluating (%d rotations x %d objects) ...",
            len(ACTIVE_EVAL_ROTATIONS),
            len(ACTIVE_OBJECTS),
        )
        exp.evaluate()

    if DO_EVAL:
        print_summary(exp, ACTIVE_OBJECTS)


if __name__ == "__main__":
    main()