#!/usr/bin/env python
# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Full Monty integration of CorticalColumnTorchLM with Panda3D and YCB objects.

Uses the complete standard Monty experiment pipeline:

  MontyObjectRecognitionExperiment
    └── EnvironmentInterfacePerObject
          └── Panda3DEnvironment  (real GLB meshes, not Habitat)
    └── MontyForEvidenceGraphMatching
          ├── CameraSM("patch")       -- single surface sensor
          ├── CorticalColumnTorchLM  -- Phase-11 Hopfield column, propose_goal_states()
          └── InformedPolicy          -- on-object motor, goal-state-enabled

All Track-9 features are active:
  - use_location_feature_memory=True   (Phase 11 LFM + predictive tracking)
  - hopfield_voting=False              (single-LM run, no lateral voting needed)
  - propose_goal_states()              (real GoalState output, wired through pipeline)

The sensor ID ("patch") matches the CameraSM sensor_module_id, so
MontyBase.get_observations() routes observations correctly without any manual
remapping.

Usage::

    conda run -n tbp.monty python scripts/run_full_monty_torch_panda3d.py

Output is written to ~/tbp/results/monty/runs/ycb_torch_panda3d_full/.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Monty experiment stack
# ---------------------------------------------------------------------------
from tbp.monty.frameworks.experiments.object_recognition_experiments import (
    MontyObjectRecognitionExperiment,
)
from tbp.monty.frameworks.environments.embodied_data import (
    EnvironmentInterfacePerObject,
)
from tbp.monty.frameworks.environments.object_init_samplers import Predefined

# ---------------------------------------------------------------------------
# Panda3D environment + transforms
# ---------------------------------------------------------------------------
from tbp.monty.simulators.panda3d.environment import Panda3DEnvironment
from tbp.monty.simulators.panda3d.agents import Panda3DAgent
from tbp.monty.simulators.panda3d.transforms import Panda3DDepthNormalize
from tbp.monty.frameworks.environment_utils.transforms import DepthTo3DLocations
from tbp.monty.simulators.panda3d.ycb import ycb_glb_path

# ---------------------------------------------------------------------------
# Monty model components
# ---------------------------------------------------------------------------
from tbp.monty.frameworks.models.evidence_matching.model import (
    MontyForEvidenceGraphMatching,
)
from tbp.monty.frameworks.models.motor_system import MotorSystem
from tbp.monty.frameworks.models.motor_policies import InformedPolicy
from tbp.monty.frameworks.actions.action_samplers import ConstantSampler
from tbp.monty.frameworks.actions.actions import (
    LookDown,
    LookUp,
    MoveForward,
    TurnLeft,
    TurnRight,
)
from tbp.monty.frameworks.models.sensor_modules import CameraSM
from tbp.monty.frameworks.models.cortical_column_torch.learning_module import (
    CorticalColumnTorchLM,
)
from tbp.monty.frameworks.loggers.monty_handlers import BasicCSVStatsHandler
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.sensors import SensorID

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AGENT_ID = AgentID("agent_id_0")
# Sensor ID matches CameraSM sensor_module_id — MontyBase.get_observations()
# uses this ID to look up the observation dict, so they must be identical.
SENSOR_ID = SensorID("patch")
SM_ID = "patch"

RESOLUTION = (64, 64)
FOV = 90.0
NEAR = 0.01
FAR = 10.0
ORBIT_RADIUS = 0.5

# 5-object YCB slice (same as benchmark)
YCB_OBJECTS = [
    "011_banana",
    "025_mug",
    "003_cracker_box",
    "013_apple",
    "035_power_drill",
]

# Training: 8 Y-axis rotations (45° increments) for full rotational coverage.
# This is the primary fix for convergence: with 1 rotation, eval at non-canonical
# orientations sees features the LFM never stored → cosim differences Δc ≈ 0.02
# between objects → beta*Δc < ln(1.5) → separation ratio never exceeded.
# With 8 rotations, covered eval viewpoints have Δc ≈ 0.2-0.4 → fast convergence.
TRAIN_ROTATIONS = [
    (0.0,   0.0, 0.0),
    (0.0,  45.0, 0.0),
    (0.0,  90.0, 0.0),
    (0.0, 135.0, 0.0),
    (0.0, 180.0, 0.0),
    (0.0, 225.0, 0.0),
    (0.0, 270.0, 0.0),
    (0.0, 315.0, 0.0),
]

# Eval: four diverse rotations → 4 epochs × 5 objects = 20 eval episodes
EVAL_ROTATIONS = [
    (0.0, 0.0, 0.0),
    (0.0, 45.0, 0.0),
    (30.0, 0.0, 60.0),
    (0.0, 90.0, 0.0),
]

OUTPUT_DIR = (
    Path(os.environ.get("MONTY_RESULTS", "~/tbp/results/monty")).expanduser()
    / "runs"
    / "ycb_torch_panda3d_full"
)

# ---------------------------------------------------------------------------
# Column hyper-parameters (Phase 11 enabled)
# ---------------------------------------------------------------------------

# Convergence analysis:
# The separation ratio e_best/e_second = score_best/score_second = exp(beta * Δc),
# where Δc = cosim difference between correct and next-best object's LFM patterns.
# For convergence we need beta * Δc > ln(1.5) ≈ 0.405.
# With 1 training rotation and a non-canonical eval angle, Δc ≈ 0.02-0.05:
#   beta=12: 12*0.02 = 0.24 → NEVER converges (most eval episodes time out)
#   beta=30: 30*0.02 = 0.60 → ratio 1.82 → converges in ~30 steps
# With 8 training rotations, Δc grows to 0.2-0.4 at covered eval angles:
#   beta=12: 12*0.2 = 2.4 → ratio 11x → fast convergence
#   beta=30: 30*0.2 = 6.0 → ratio 403x → near-instant convergence
# LFM novelty_threshold: lower = more diverse stored patterns, better Δc coverage.
COLUMN_KWARGS = {
    "n_minicolumns": 2048,
    "n_cells_per_minicolumn": 8,
    "sparsity": 0.03,
    "use_apical": False,
    "use_location_feature_memory": True,   # Phase 11: LFM + predictive tracking
    "beta": 12.0,
    "max_settle_iters": 12,
    "evidence_decay": 0.01,
    "seed": 42,
    # LFM-specific: sharper softmax discrimination + more patterns per object
    "lfm_kwargs": {
        "beta": 30.0,              # was 12.0; key for convergence (see analysis above)
        "novelty_threshold": 0.5,  # was 0.7; store more diverse training patterns
    },
}


# ---------------------------------------------------------------------------
# Configuration builders
# ---------------------------------------------------------------------------


class Panda3DEnvWithLookAt(Panda3DEnvironment):
    """Panda3D environment that auto-orients the camera at the visual centroid.

    After each reset(), iteratively adjusts the camera heading/pitch so that
    the visual centroid of on-object depth pixels lands at the image center.
    Required because YCB GLB models have irregular shapes whose projected
    visual center doesn't align with the geometric bounding-box center.
    """

    #: Background depth sentinel (same threshold used in DepthTo3DLocations)
    _BG_THRESHOLD = 9.5

    def reset(self):
        obs, state = self._sim.reset()
        self._center_camera_on_object()
        # Re-render with the corrected camera orientation
        obs = self._sim._get_observations()
        state = self._sim._get_states()
        return obs, state

    def _center_camera_on_object(self, max_iters: int = 5):
        """Iteratively adjust camera heading/pitch to center object pixels."""
        import numpy as np

        sim = self._sim
        for agent in sim._agents:
            buf = sim._agent_buffers[agent.agent_id]
            cam_np = buf["camera_np"]
            resolution = agent.resolution  # (H, W)
            h, w = resolution

            # Angular size of one pixel: assume square pixels, square sensor
            # hfov covers full width W pixels
            px_per_deg = w / agent.fov

            for _ in range(max_iters):
                # Force a fresh render
                from tbp.monty.frameworks.sensors import SensorID
                obs, _ = sim.step([])
                depth = obs[agent.agent_id][SensorID(agent.sensor_id)][
                    "depth"
                ].squeeze()

                on_obj = depth < self._BG_THRESHOLD
                n_obj = on_obj.sum()

                if n_obj == 0:
                    break  # Object not visible; stop adjusting

                # Check if center is already on-object
                if depth[h // 2, w // 2] < self._BG_THRESHOLD:
                    break

                # Compute visual centroid offset from image center
                rows, cols = np.where(on_obj)
                r_ctr = rows.mean()
                c_ctr = cols.mean()

                d_col = c_ctr - (w / 2)   # >0 → object is right of center
                d_row = r_ctr - (h / 2)   # >0 → object is below center

                if abs(d_col) < 0.5 and abs(d_row) < 0.5:
                    break  # Sub-pixel offset; no improvement possible

                # Sign convention (Panda3D):
                #   positive heading → turn LEFT; positive pitch → look UP
                #   object LEFT of center (d_col < 0) → turn LEFT → +delta_h
                #   object ABOVE center (d_row < 0) → look UP → +delta_p
                delta_h = -d_col / px_per_deg
                delta_p = -d_row / px_per_deg

                hpr = cam_np.getHpr()
                cam_np.setH(hpr[0] + delta_h)
                cam_np.setP(hpr[1] + delta_p)


def build_ycb_registry(object_names: list[str]) -> dict:
    """Build the Panda3DObjectRegistry spec dict for the given YCB names."""
    registry = {}
    for name in object_names:
        try:
            path = ycb_glb_path(name)
        except FileNotFoundError as exc:
            logger.error("YCB mesh not found for %s: %s", name, exc)
            raise
        registry[name] = {"model_path": str(path), "animated": False}
    return registry


def build_transform():
    """Return the observation transform pipeline for Panda3D RGB-D data."""
    return [
        Panda3DDepthNormalize(agent_id=AGENT_ID, near=NEAR, far=FAR),
        DepthTo3DLocations(
            agent_id=AGENT_ID,
            sensor_ids=[SENSOR_ID],
            resolutions=[RESOLUTION],
            hfov=FOV,
            world_coord=True,
            get_all_points=True,
        ),
    ]


def build_config(object_names: list[str]) -> dict:
    """Build the full MontyExperiment config dict."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    registry = build_ycb_registry(object_names)
    transform = build_transform()

    # ---- env_interface_config -----------------------------------------
    # Panda3DEnvironment is both train and eval env (object cycling via
    # EnvironmentInterfacePerObject handles episode lifecycle).
    env_interface_config = {
        "env_init_func": Panda3DEnvWithLookAt,
        "env_init_args": {
            "agents": [
                Panda3DAgent(
                    agent_id=AGENT_ID,
                    sensor_id=str(SENSOR_ID),   # must match SM_ID
                    resolution=RESOLUTION,
                    fov=FOV,
                    # Camera at -Y close to object; _center_camera_on_object
                    # will iteratively align the heading/pitch per episode.
                    position=(0.0, -0.3, 0.0),
                    rotation=(1.0, 0.0, 0.0, 0.0),
                )
            ],
            "near": NEAR,
            "far": FAR,
            "object_registry": registry,
        },
        "transform": transform,
    }

    # ---- train env interface -------------------------------------------
    train_env_interface_args = {
        "object_names": object_names,
        "object_init_sampler": Predefined(
            positions=[(0.0, 0.0, 0.0)],
            rotations=TRAIN_ROTATIONS,
            scales=[(1.0, 1.0, 1.0)],
            change_every_episode=False,
        ),
    }

    # ---- eval env interface --------------------------------------------
    eval_env_interface_args = {
        "object_names": object_names,
        "object_init_sampler": Predefined(
            positions=[(0.0, 0.0, 0.0)],
            rotations=EVAL_ROTATIONS,
            scales=[(1.0, 1.0, 1.0)],
            change_every_episode=False,   # rotate per epoch (4 epochs → 4 rotations)
        ),
    }

    # ---- monty_config --------------------------------------------------
    monty_config = {
        "monty_class": MontyForEvidenceGraphMatching,
        "monty_args": {
            "min_eval_steps": 5,
            "min_train_steps": 5,
            "num_exploratory_steps": 200,
            "max_total_steps": 5000,
        },
        "sensor_module_configs": {
            "sensor_module_0": {
                "sensor_module_class": CameraSM,
                "sensor_module_args": {
                    "sensor_module_id": SM_ID,
                    "features": [
                        "pose_vectors",
                        "pose_fully_defined",
                        "on_object",
                        "hsv",
                        "principal_curvatures_log",
                    ],
                    "save_raw_obs": False,
                },
            },
        },
        "learning_module_configs": {
            "learning_module_0": {
                "learning_module_class": CorticalColumnTorchLM,
                "learning_module_args": {
                    "column_kwargs": COLUMN_KWARGS,
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
                    view_finder_id=SM_ID,   # our sensor is "patch", not "view_finder"
                ),
            },
        },
        # Single SM connects to single LM
        "sm_to_lm_matrix": [[0]],          # LM 0 receives from SM 0
        "lm_to_lm_matrix": [[]],           # no LM-to-LM hierarchy
        "lm_to_lm_vote_matrix": [[]],      # no lateral voting (single LM)
        "sm_to_agent_dict": {SM_ID: AGENT_ID},
    }

    # ---- logging config ------------------------------------------------
    logging_config = {
        "python_log_level": "INFO",
        "python_log_to_file": True,
        "python_log_to_stderr": True,
        "output_dir": str(OUTPUT_DIR),
        "run_name": "ycb_torch_panda3d_full",
        # Monty-level data logging: BasicCSVStatsHandler writes eval_stats.csv
        # and train_stats.csv to OUTPUT_DIR for per-episode accuracy analysis.
        "monty_log_level": "BASIC",
        "monty_handlers": [BasicCSVStatsHandler],
        "wandb_handlers": [],
    }

    return {
        # Experiment-level parameters
        "seed": 42,
        "do_train": True,
        "do_eval": True,
        "max_train_steps": 200,     # steps per train episode (exploratory scan)
        "max_eval_steps": 200,      # steps per eval episode (matching)
        "max_total_steps": 5000,
        # One epoch per training rotation: each epoch trains all objects at that rotation.
        "n_train_epochs": len(TRAIN_ROTATIONS),  # 8 rotations × 5 objects = 40 train episodes
        "n_eval_epochs": len(EVAL_ROTATIONS),  # 4 epochs × 5 objects = 20 eval eps
        "model_name_or_path": None,
        "min_lms_match": 1,
        "show_sensor_output": False,
        "supervised_lm_ids": [],
        # Sub-configs
        "logging": logging_config,
        "monty_config": monty_config,
        "env_interface_config": env_interface_config,
        "train_env_interface_class": EnvironmentInterfacePerObject,
        "train_env_interface_args": train_env_interface_args,
        "eval_env_interface_class": EnvironmentInterfacePerObject,
        "eval_env_interface_args": eval_env_interface_args,
    }





# ---------------------------------------------------------------------------
# Results summary
# ---------------------------------------------------------------------------

def print_summary(exp: MontyObjectRecognitionExperiment, object_names: list[str]):
    """Print a concise accuracy and diagnosis summary after evaluation."""
    import pandas as pd

    logger.info("\n" + "=" * 60)
    logger.info("Full-Monty Panda3D Results")
    logger.info("=" * 60)
    lm = exp.model.learning_modules[0]
    known = lm.get_all_known_object_ids()
    logger.info("Objects in memory: %s", list(known))
    logger.info("Output dir: %s", OUTPUT_DIR)

    # ---- Full Monty Checklist ----------------------------------------
    logger.info("\n--- Full Monty Checklist ---")
    logger.info("  [OK] MontyObjectRecognitionExperiment pipeline")
    logger.info("  [OK] MontyForEvidenceGraphMatching")
    logger.info("  [OK] CameraSM sensor module")
    logger.info("  [OK] CorticalColumnTorchLM (Phase-11 Hopfield column)")
    logger.info("  [OK] InformedPolicy motor system")
    logger.info("  [OK] Panda3D environment with real YCB GLB meshes")
    logger.info("  [OK] DepthTo3DLocations + Panda3DDepthNormalize transforms")
    logger.info("  [OK] use_goal_state_driven_actions=True   -- LFM-guided hypothesis jumps enabled")
    logger.info("  [OK] Training on %d rotations per object  -- 45°-spaced Y-axis coverage", len(TRAIN_ROTATIONS))
    logger.info("  [!!] InformedPolicy = random surface walk -- not orbital coverage")
    logger.info("  [!!] action_space_type='distant_agent'    -- pan/tilt only, no physical move")

    # ---- Parse eval_stats.csv ----------------------------------------
    csv_path = OUTPUT_DIR / "eval_stats.csv"
    if not csv_path.exists():
        logger.warning("eval_stats.csv not found at %s", csv_path)
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.warning("Could not read eval_stats.csv: %s", e)
        return

    if df.empty:
        logger.warning("eval_stats.csv is empty.")
        return

    total = len(df)
    # "correct"/"confused" = converged; "correct_mlh"/"confused_mlh" = timed out
    converged    = df["primary_performance"].isin(["correct", "confused"]).sum()
    correct      = (df["primary_performance"] == "correct").sum()
    confused     = (df["primary_performance"] == "confused").sum()
    no_match     = (df["primary_performance"] == "no_match").sum()
    correct_mlh  = (df["primary_performance"] == "correct_mlh").sum()
    confused_mlh = (df["primary_performance"] == "confused_mlh").sum()
    timed_out    = correct_mlh + confused_mlh

    logger.info("\n--- Accuracy Results (%d eval episodes) ---", total)
    logger.info("  Converged (matched 1 object): %d / %d = %.1f%%",
                converged, total, 100.0 * converged / total)
    logger.info("  Correct (converged correctly): %d / %d = %.1f%%",
                correct, total, 100.0 * correct / total)
    if converged > 0:
        logger.info("  Correct among converged:       %d / %d = %.1f%%",
                    correct, converged, 100.0 * correct / converged)
    logger.info("  Best-guess accuracy (correct + correct_mlh): %d / %d = %.1f%%",
                correct + correct_mlh, total,
                100.0 * (correct + correct_mlh) / total)
    logger.info("  Outcome breakdown:")
    logger.info("    correct:      %d  (converged to right object)", correct)
    logger.info("    confused:     %d  (converged to wrong object)", confused)
    logger.info("    no_match:     %d  (all evidence eliminated)", no_match)
    logger.info("    correct_mlh:  %d  (timed out, MLH was right)", correct_mlh)
    logger.info("    confused_mlh: %d  (timed out, MLH was wrong)", confused_mlh)

    # ---- Per-object breakdown ----------------------------------------
    logger.info("\n--- Per-Object Breakdown ---")
    obj_col = "primary_target_object"
    if obj_col in df.columns:
        for obj in sorted(df[obj_col].unique()):
            sub = df[df[obj_col] == obj]
            n = len(sub)
            c = (sub["primary_performance"] == "correct").sum()
            c_mlh = (sub["primary_performance"] == "correct_mlh").sum()
            to = (sub["primary_performance"].isin(["correct_mlh","confused_mlh"])).sum()
            cf = (sub["primary_performance"].isin(["confused","confused_mlh"])).sum()
            detected = sub["result"].value_counts().to_dict() if "result" in sub else {}
            logger.info("  %-22s  correct=%d  correct_mlh=%d  timed_out=%d  confused=%d  detected=%s",
                        obj, c, c_mlh, to, cf, detected)

    # ---- Convergence failure diagnosis --------------------------------
    logger.info("\n--- Convergence Failure Diagnosis ---")
    timed_iter = df["primary_performance"].isin(["correct_mlh", "confused_mlh"])
    if timed_iter.sum() > 0:
        failed = df[timed_iter]
        if obj_col in df.columns:
            fail_counts = failed[obj_col].value_counts()
            logger.info("  Objects with most timeouts:")
            for obj, cnt in fail_counts.items():
                logger.info("    %-22s  %d/%d episodes timed out",
                            obj, cnt, (df[obj_col] == obj).sum())
        if "num_steps" in df.columns:
            converged_mask = df["primary_performance"].isin(["correct", "confused"])
            if converged_mask.sum() > 0:
                logger.info("  Avg steps in converged episodes: %.1f",
                            df[converged_mask]["num_steps"].mean())
            logger.info("  Avg steps in timed-out episodes:   %.1f",
                        failed["num_steps"].mean())
        logger.info("  Likely causes:")
        logger.info("    - LFM beta too low: beta * Δc < ln(1.5)=0.405 → ratio never exceeds threshold")
        logger.info("    - Missing training rotation: eval viewpoint not covered in LFM")
        logger.info("    (current: LFM beta=%.1f, n_train_rotations=%d)",
                    COLUMN_KWARGS.get("lfm_kwargs", {}).get("beta",
                        COLUMN_KWARGS.get("beta", 12.0)),
                    len(TRAIN_ROTATIONS))
        logger.info("    - Random walk coverage limited -- not all surface features sampled")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("Building config for %d YCB objects ...", len(YCB_OBJECTS))
    config = build_config(YCB_OBJECTS)

    logger.info("Instantiating MontyObjectRecognitionExperiment ...")
    exp = MontyObjectRecognitionExperiment(config)
    exp.setup_experiment(config)

    logger.info("Training on %d objects ...", len(YCB_OBJECTS))
    exp.train()

    logger.info("Evaluating (%d rotations × %d objects) ...", len(EVAL_ROTATIONS), len(YCB_OBJECTS))
    exp.evaluate()

    print_summary(exp, YCB_OBJECTS)


if __name__ == "__main__":
    main()
