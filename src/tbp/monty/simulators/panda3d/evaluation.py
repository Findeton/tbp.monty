# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Panda3D object recognition evaluation harness.

Runs structured training and evaluation episodes through real
``EvidenceGraphLM`` instances using Panda3D with real YCB meshes.
No Hydra config, no Habitat dependency — directly programmable.

The pipeline mirrors what ``MontyObjectRecognitionExperiment.run_epoch()``
does internally, but wired to Panda3D:

1. **Training**: For each object, orbit the camera around the YCB mesh,
   feed observations through the transform pipeline → CameraSM → State →
   ``EvidenceGraphLM.exploratory_step()``, then persist the learned graph.

2. **Evaluation**: For each (object, rotation) pair, orbit the camera,
   feed observations through ``EvidenceGraphLM.matching_step()``, record
   whether the LM correctly identifies the object and its pose.

Usage::

    from tbp.monty.simulators.panda3d.evaluation import Panda3DEvalHarness
    from tbp.monty.simulators.panda3d.ycb import YCB_EVAL_OBJECTS

    harness = Panda3DEvalHarness(object_names=YCB_EVAL_OBJECTS[:5])
    result = harness.run()
    print(result.summary())
    harness.close()
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.environment_utils.transforms import (
    DepthTo3DLocations,
    TransformContext,
)
from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.models.evidence_matching.learning_module import (
    EvidenceGraphLM,
)
from tbp.monty.frameworks.models.sensor_modules import CameraSM
from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.simulators.panda3d.agents import Panda3DAgent
from tbp.monty.simulators.panda3d.metrics import (
    EvalEpisodeResult,
    EvalRunResult,
)
from tbp.monty.simulators.panda3d.simulator import Panda3DSimulator
from tbp.monty.simulators.panda3d.temporal_training import OrbitalMotorPolicy
from tbp.monty.simulators.panda3d.transforms import Panda3DDepthNormalize
from tbp.monty.simulators.panda3d.ycb import ycb_glb_path

logger = logging.getLogger(__name__)


class Panda3DEvalHarness:
    """Object recognition evaluation using Panda3D with real YCB meshes.

    Trains an ``EvidenceGraphLM`` on a set of YCB objects, then evaluates
    recognition accuracy across rotated presentations.

    Parameters
    ----------
    object_names : list[str]
        YCB object names to train and evaluate on (e.g. ``["011_banana"]``).
    eval_rotations : list[tuple[float, float, float]] or None
        Euler rotations (rx, ry, rz) in degrees to evaluate each object at.
        Defaults to 4 canonical rotations if None.
    train_steps : int
        Number of orbital observation steps per training episode.
    eval_steps : int
        Number of orbital observation steps per evaluation episode.
    resolution : tuple[int, int]
        Render resolution (width, height).
    fov : float
        Camera horizontal field of view in degrees.
    orbit_radius : float
        Camera distance from object center during orbital exploration.
    lm_kwargs : dict or None
        Extra keyword arguments passed to ``EvidenceGraphLM``. Overrides
        defaults for tolerances, feature_weights, etc.
    sm_features : list[str] or None
        Features for CameraSM. Defaults to on_object + hsv + curvatures.
    seed : int
        Random seed for reproducibility.
    """

    # Defaults for the LM — intentionally permissive for initial benchmarks.
    _DEFAULT_LM_KWARGS = dict(
        max_match_distance=0.05,
        tolerances={
            "patch": {
                "hsv": [0.1, 1, 1],
                "principal_curvatures_log": [2, 2],
            },
        },
        feature_weights={
            "patch": {"hsv": np.array([1.0, 2.0, 0.5])},
        },
        max_graph_size=10.0,
    )

    _DEFAULT_ROTATIONS = [
        (0.0, 0.0, 0.0),
        (0.0, 45.0, 0.0),
        (30.0, 0.0, 60.0),
        (0.0, 90.0, 0.0),
    ]

    def __init__(
        self,
        object_names: list[str],
        eval_rotations: list[tuple[float, float, float]] | None = None,
        train_steps: int = 40,
        eval_steps: int = 40,
        resolution: tuple[int, int] = (64, 64),
        fov: float = 90.0,
        orbit_radius: float = 0.5,
        lm_kwargs: dict | None = None,
        sm_features: list[str] | None = None,
        seed: int = 42,
    ):
        self._object_names = list(object_names)
        self._eval_rotations = eval_rotations or self._DEFAULT_ROTATIONS
        self._train_steps = train_steps
        self._eval_steps = eval_steps
        self._resolution = resolution
        self._fov = fov
        self._orbit_radius = orbit_radius
        self._seed = seed

        if sm_features is None:
            sm_features = ["on_object", "hsv", "principal_curvatures_log"]
        self._sm_features = sm_features

        # Merge user LM kwargs over defaults
        merged_lm = dict(self._DEFAULT_LM_KWARGS)
        if lm_kwargs:
            merged_lm.update(lm_kwargs)
        self._lm_kwargs = merged_lm

        # IDs
        self._agent_id = AgentID("eval_agent")
        self._sensor_id = SensorID("patch")

        # Lazy init
        self._sim: Panda3DSimulator | None = None
        self._depth_norm: Panda3DDepthNormalize | None = None
        self._d3d: DepthTo3DLocations | None = None
        self._sm: CameraSM | None = None
        self._motor: OrbitalMotorPolicy | None = None
        self._lm: EvidenceGraphLM | None = None

    def _setup(self) -> None:
        """Initialize simulator, transforms, sensor module, and LM."""
        near, far = 0.01, 10.0

        agent = Panda3DAgent(
            agent_id=self._agent_id,
            sensor_id=str(self._sensor_id),
            resolution=self._resolution,
            fov=self._fov,
        )
        self._sim = Panda3DSimulator(
            agents=[agent], near=near, far=far,
        )
        self._depth_norm = Panda3DDepthNormalize(
            agent_id=self._agent_id, near=near, far=far,
        )
        self._d3d = DepthTo3DLocations(
            agent_id=self._agent_id,
            sensor_ids=[self._sensor_id],
            resolutions=[self._resolution],
            hfov=self._fov,
            world_coord=True,
            get_all_points=True,
        )
        self._sm = CameraSM(
            sensor_module_id=str(self._sensor_id),
            features=self._sm_features,
        )
        self._motor = OrbitalMotorPolicy(
            agent_id=self._agent_id,
            orbit_radius=self._orbit_radius,
            azimuth_step_deg=360.0 / max(self._train_steps, 1),
        )
        self._lm = EvidenceGraphLM(**self._lm_kwargs)

    def _render_observations(
        self,
        glb_path: str,
        rotation_euler_deg: tuple[float, float, float],
        n_steps: int,
    ) -> list:
        """Place an object and collect States from an orbital pass.

        Returns a list of usable State objects.
        """
        # Convert Euler to quaternion (w, x, y, z)
        r = Rotation.from_euler("xyz", rotation_euler_deg, degrees=True)
        quat_xyzw = r.as_quat()  # scipy: (x, y, z, w)
        quat_wxyz = (
            float(quat_xyzw[3]),
            float(quat_xyzw[0]),
            float(quat_xyzw[1]),
            float(quat_xyzw[2]),
        )

        self._sim.remove_all_objects()
        self._sim.add_object(
            name=glb_path,
            position=(0.0, 0.0, 0.0),
            rotation=quat_wxyz,
        )

        # Adjust azimuth step for the requested number of steps
        self._motor = OrbitalMotorPolicy(
            agent_id=self._agent_id,
            orbit_radius=self._orbit_radius,
            azimuth_step_deg=360.0 / max(n_steps, 1),
        )
        self._motor.reset()
        self._sm.pre_episode()

        states = []
        for step in range(n_steps):
            position, rotation = self._motor.get_camera_pose(step)
            cam_info = self._sim._agent_buffers[self._agent_id]
            self._sim._set_node_pose(cam_info["camera_np"], position, rotation)

            obs, proprio = self._sim.step([])

            ctx = TransformContext(
                rng=np.random.RandomState(self._seed + step),
                state=proprio,
            )
            obs = self._depth_norm(obs, ctx)
            obs = self._d3d(obs, ctx)

            agent_state = proprio[self._agent_id]
            self._sm.update_state(agent_state)

            rt_ctx = RuntimeContext(
                rng=np.random.RandomState(self._seed + step),
            )
            state = self._sm.step(rt_ctx, obs[self._agent_id][self._sensor_id])

            if state is not None and state.use_state:
                states.append(state)

        return states

    def _train_object(self, object_name: str) -> int:
        """Train the LM on one object. Returns number of usable states."""
        glb_path = str(ycb_glb_path(object_name))
        states = self._render_observations(
            glb_path,
            rotation_euler_deg=(0.0, 0.0, 0.0),
            n_steps=self._train_steps,
        )
        if not states:
            logger.warning("No usable states for training %s", object_name)
            return 0

        ctx = RuntimeContext(rng=np.random.RandomState(self._seed))
        target = {"object": object_name, "quat_rotation": [1, 0, 0, 0]}

        self._lm.mode = ExperimentMode.TRAIN
        self._lm.pre_episode(primary_target=target)

        for s in states:
            self._lm.exploratory_step(ctx, [s])

        # Provide ground-truth convergence (same as experiment infrastructure)
        self._lm.detected_object = object_name
        self._lm.detected_rotation_r = Rotation.identity()
        current_loc = self._lm.buffer.get_current_location(
            input_channel="first"
        )
        self._lm.buffer.stats["detected_location_rel_body"] = current_loc
        self._lm.buffer.stats["detected_location_on_model"] = current_loc
        self._lm.post_episode()

        logger.info("Trained %s: %d usable states", object_name, len(states))
        return len(states)

    def _eval_object(
        self,
        object_name: str,
        rotation_euler_deg: tuple[float, float, float],
    ) -> EvalEpisodeResult:
        """Evaluate the LM on one object at one rotation."""
        t0 = time.monotonic()

        glb_path = str(ycb_glb_path(object_name))
        states = self._render_observations(
            glb_path,
            rotation_euler_deg=rotation_euler_deg,
            n_steps=self._eval_steps,
        )

        ctx = RuntimeContext(rng=np.random.RandomState(self._seed))
        placeholder = {"object": "placeholder", "quat_rotation": [1, 0, 0, 0]}
        self._lm.mode = ExperimentMode.EVAL
        self._lm.pre_episode(primary_target=placeholder)

        steps_to_converge = None
        for i, s in enumerate(states):
            self._lm.matching_step(ctx, [s])
            # Check if LM has converged
            if len(self._lm.get_possible_matches()) == 1:
                mlh = self._lm.get_current_mlh()
                if (
                    "evidence" in mlh
                    and mlh["evidence"] > self._lm.object_evidence_threshold
                ):
                    steps_to_converge = i + 1
                    break

        # Read LM's best guess
        mlh = self._lm.get_current_mlh()
        detected = mlh.get("graph_id") if mlh else None
        correct = detected == object_name

        # Rotation error
        rotation_error_deg = None
        if correct and "rotation" in mlh and mlh["rotation"] is not None:
            gt_rot = Rotation.from_euler("xyz", rotation_euler_deg, degrees=True)
            det_rot = mlh["rotation"].inv()
            diff = (gt_rot * det_rot.inv()).magnitude()
            rotation_error_deg = float(np.degrees(diff))

        max_evidence = 0.0
        for ev_array in self._lm.evidence.values():
            if ev_array is not None and len(ev_array) > 0:
                max_evidence = max(max_evidence, float(np.max(ev_array)))

        elapsed = time.monotonic() - t0

        result = EvalEpisodeResult(
            object_name=object_name,
            rotation=rotation_euler_deg,
            detected_object=detected,
            correct=correct,
            steps_to_converge=steps_to_converge,
            rotation_error_deg=rotation_error_deg,
            max_evidence=max_evidence,
            wall_clock_seconds=elapsed,
        )

        logger.info(
            "Eval %s @ rot=%s: detected=%s correct=%s steps=%s (%.1fs)",
            object_name,
            rotation_euler_deg,
            detected,
            correct,
            steps_to_converge,
            elapsed,
        )
        return result

    def run(self) -> EvalRunResult:
        """Run full training + evaluation pipeline.

        Returns
        -------
        EvalRunResult
            Aggregate metrics across all evaluation episodes.
        """
        if self._sim is None:
            self._setup()

        t_start = time.monotonic()

        # --- Training phase ---
        logger.info(
            "Training on %d objects (%d steps each)",
            len(self._object_names),
            self._train_steps,
        )
        for name in self._object_names:
            self._train_object(name)

        known = self._lm.get_all_known_object_ids()
        logger.info("Training complete. Known objects: %s", known)

        # --- Evaluation phase ---
        episodes: list[EvalEpisodeResult] = []
        total_eval = len(self._object_names) * len(self._eval_rotations)
        logger.info(
            "Evaluating %d episodes (%d objects x %d rotations)",
            total_eval,
            len(self._object_names),
            len(self._eval_rotations),
        )

        for name in self._object_names:
            for rot in self._eval_rotations:
                result = self._eval_object(name, rot)
                episodes.append(result)

        wall_total = time.monotonic() - t_start

        config = {
            "object_names": self._object_names,
            "eval_rotations": self._eval_rotations,
            "train_steps": self._train_steps,
            "eval_steps": self._eval_steps,
            "resolution": self._resolution,
            "fov": self._fov,
            "orbit_radius": self._orbit_radius,
            "seed": self._seed,
            "sm_features": self._sm_features,
        }

        run_result = EvalRunResult.from_episodes(
            episodes=episodes,
            wall_clock_total=wall_total,
            config=config,
        )

        logger.info("Evaluation complete.\n%s", run_result.summary())
        return run_result

    def get_lm(self) -> EvidenceGraphLM:
        """Return the underlying LM (e.g. for inspection after training)."""
        if self._lm is None:
            raise RuntimeError("Harness not yet initialized. Call run() first.")
        return self._lm

    def close(self) -> None:
        """Release simulator resources."""
        if self._sim is not None:
            self._sim.close()
            self._sim = None
