# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Temporal validation evaluation harness for multi-object scenes.

Runs structured scene navigation episodes through ``TemporalMemory`` using
Panda3D with real YCB meshes. Measures surprise signals at object transitions,
object swaps, and repeated traversals — validating that TM produces meaningful
temporal predictions for the Track 6 predictive coding features.

The key temporal signal quality metrics:

- **Transition surprise**: TM surprise should spike when the agent moves
  from one object to another (transit phases) and drop during steady
  exploration of a single object (dwell phases).
- **Object swap detection**: Replacing one object in a scene should produce
  a surprise spike at the swapped location compared to an unmodified scene.
- **Habituation**: Repeated traversals of the same scene should decrease
  mean TM surprise as temporal associations are learned.

Usage::

    from tbp.monty.simulators.panda3d.temporal_evaluation import (
        TemporalSceneEvaluator,
    )
    from tbp.monty.simulators.panda3d.scenes import KITCHEN_SCENE

    evaluator = TemporalSceneEvaluator()
    result = evaluator.run_scene(KITCHEN_SCENE, n_traversals=3)
    print(result.summary())
    evaluator.close()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.environment_utils.transforms import (
    DepthTo3DLocations,
    TransformContext,
)
from tbp.monty.frameworks.models.sensor_modules import CameraSM
from tbp.monty.frameworks.models.temporal_memory import TemporalMemory
from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.simulators.panda3d.agents import Panda3DAgent
from tbp.monty.simulators.panda3d.scenes import SceneSpec, load_scene, swap_object
from tbp.monty.simulators.panda3d.simulator import Panda3DSimulator
from tbp.monty.simulators.panda3d.transforms import Panda3DDepthNormalize
from tbp.monty.simulators.panda3d.waypoint_policy import WaypointMotorPolicy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TraversalResult:
    """Result of one traversal through a scene.

    Attributes
    ----------
    surprise_per_step : list[float]
        TM surprise at each usable step.
    segment_types : list[str]
        ``"dwell"`` or ``"transit"`` for each usable step.
    waypoint_indices : list[int]
        Waypoint index (for dwell) or -1 (for transit) at each step.
    mean_surprise : float
        Mean surprise across all usable steps.
    mean_dwell_surprise : float
        Mean surprise during dwell (orbit) phases.
    mean_transit_surprise : float or None
        Mean surprise during transit phases (None if no transit steps).
    usable_steps : int
        Number of steps with valid State objects.
    """

    surprise_per_step: list
    segment_types: list
    waypoint_indices: list
    mean_surprise: float
    mean_dwell_surprise: float
    mean_transit_surprise: float | None
    usable_steps: int


@dataclass
class SceneEvalResult:
    """Aggregate result from evaluating one scene across multiple traversals.

    Attributes
    ----------
    scene_name : str
        Name of the evaluated scene.
    traversals : list[TraversalResult]
        Per-traversal results.
    surprise_first : float
        Mean surprise on the first traversal.
    surprise_last : float
        Mean surprise on the last traversal.
    habituation_ratio : float
        ``surprise_last / surprise_first``. Values < 1.0 indicate learning.
    dwell_transit_ratio : float or None
        ``mean_transit_surprise / mean_dwell_surprise`` across all traversals.
        Values > 1.0 indicate higher surprise at object transitions (good).
    wall_clock_seconds : float
        Total wall-clock time.
    config : dict
        Evaluation parameters for reproducibility.
    """

    scene_name: str
    traversals: list
    surprise_first: float
    surprise_last: float
    habituation_ratio: float
    dwell_transit_ratio: float | None
    wall_clock_seconds: float
    config: dict = field(default_factory=dict)

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"Scene: {self.scene_name}",
            f"Traversals: {len(self.traversals)}",
            f"Surprise (first): {self.surprise_first:.3f}",
            f"Surprise (last):  {self.surprise_last:.3f}",
            f"Habituation ratio: {self.habituation_ratio:.3f}",
        ]
        if self.dwell_transit_ratio is not None:
            lines.append(
                f"Dwell/transit ratio: {self.dwell_transit_ratio:.2f}"
            )
        lines.append(f"Wall clock: {self.wall_clock_seconds:.1f}s")
        return "\n".join(lines)


@dataclass
class SwapDetectionResult:
    """Result from an object-swap detection trial.

    Attributes
    ----------
    scene_name : str
        Original scene name.
    swap_index : int
        Index of the swapped object.
    original_object : str
        YCB name of the original object.
    replacement_object : str
        YCB name of the replacement object.
    surprise_at_swap_original : float
        Mean surprise at the swapped waypoint in the original scene.
    surprise_at_swap_modified : float
        Mean surprise at the swapped waypoint in the modified scene.
    surprise_delta : float
        ``surprise_at_swap_modified - surprise_at_swap_original``.
        Positive values indicate TM detected the change.
    detected : bool
        Whether surprise_delta > detection_threshold.
    """

    scene_name: str
    swap_index: int
    original_object: str
    replacement_object: str
    surprise_at_swap_original: float
    surprise_at_swap_modified: float
    surprise_delta: float
    detected: bool


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

class TemporalSceneEvaluator:
    """Evaluates TM temporal signal quality on multi-object YCB scenes.

    Parameters
    ----------
    resolution : tuple[int, int]
        Render resolution (width, height).
    fov : float
        Camera horizontal field of view in degrees.
    orbit_radius : float
        Camera distance during orbital exploration of each object.
    dwell_steps : int
        Steps to orbit around each waypoint.
    transit_steps : int
        Steps to interpolate between waypoints.
    sm_features : list[str] or None
        Features for CameraSM.
    tm_kwargs : dict or None
        Extra keyword arguments for TemporalMemory.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        resolution: tuple[int, int] = (64, 64),
        fov: float = 90.0,
        orbit_radius: float = 0.3,
        dwell_steps: int = 20,
        transit_steps: int = 5,
        sm_features: list[str] | None = None,
        tm_kwargs: dict | None = None,
        seed: int = 42,
    ):
        self._resolution = resolution
        self._fov = fov
        self._orbit_radius = orbit_radius
        self._dwell_steps = dwell_steps
        self._transit_steps = transit_steps
        self._seed = seed

        if sm_features is None:
            sm_features = ["on_object", "hsv", "principal_curvatures_log"]
        self._sm_features = sm_features
        self._tm_kwargs = tm_kwargs or {}

        self._agent_id = AgentID("eval_agent")
        self._sensor_id = SensorID("patch")

        # Lazy init
        self._sim: Panda3DSimulator | None = None
        self._depth_norm: Panda3DDepthNormalize | None = None
        self._d3d: DepthTo3DLocations | None = None
        self._sm: CameraSM | None = None
        self._tm: TemporalMemory | None = None

    def _setup(self) -> None:
        """Initialize simulator, transforms, sensor module, and TM."""
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
        self._tm = TemporalMemory(**self._tm_kwargs)

    def _run_traversal(
        self, scene: SceneSpec, policy: WaypointMotorPolicy, learn: bool = True
    ) -> TraversalResult:
        """Run one traversal through a loaded scene, collecting surprise."""
        self._tm.reset_episode()
        self._sm.pre_episode()
        policy.reset()

        surprises = []
        seg_types = []
        wp_indices = []

        for step in range(policy.total_steps):
            position, rotation = policy.get_camera_pose(step)

            # Set camera pose
            cam_info = self._sim._agent_buffers[self._agent_id]
            self._sim._set_node_pose(cam_info["camera_np"], position, rotation)

            # Render
            obs, proprio = self._sim.step([])

            # Transform
            ctx = TransformContext(
                rng=np.random.RandomState(self._seed + step),
                state=proprio,
            )
            obs = self._depth_norm(obs, ctx)
            obs = self._d3d(obs, ctx)

            # CameraSM → State
            agent_state = proprio[self._agent_id]
            self._sm.update_state(agent_state)
            rt_ctx = RuntimeContext(
                rng=np.random.RandomState(self._seed + step),
            )
            state = self._sm.step(
                rt_ctx, obs[self._agent_id][self._sensor_id]
            )

            if state is not None and state.use_state:
                # TM step
                tm_result = self._tm.step(state, learn=learn)
                surprise = tm_result["surprise"]

                surprises.append(surprise)

                seg_info = policy.get_segment_info(step)
                seg_types.append(seg_info["type"])
                if seg_info["type"] == "dwell":
                    wp_indices.append(seg_info["waypoint_index"])
                else:
                    wp_indices.append(-1)

        # Compute aggregates
        dwell_surprises = [
            s for s, t in zip(surprises, seg_types) if t == "dwell"
        ]
        transit_surprises = [
            s for s, t in zip(surprises, seg_types) if t == "transit"
        ]

        return TraversalResult(
            surprise_per_step=surprises,
            segment_types=seg_types,
            waypoint_indices=wp_indices,
            mean_surprise=float(np.mean(surprises)) if surprises else 0.0,
            mean_dwell_surprise=(
                float(np.mean(dwell_surprises)) if dwell_surprises else 0.0
            ),
            mean_transit_surprise=(
                float(np.mean(transit_surprises))
                if transit_surprises
                else None
            ),
            usable_steps=len(surprises),
        )

    def run_scene(
        self,
        scene: SceneSpec,
        n_traversals: int = 3,
    ) -> SceneEvalResult:
        """Run multiple traversals of a scene and measure temporal signals.

        Parameters
        ----------
        scene : SceneSpec
            Scene to evaluate.
        n_traversals : int
            Number of times to traverse the scene.

        Returns
        -------
        SceneEvalResult
            Aggregate temporal metrics across traversals.
        """
        if self._sim is None:
            self._setup()

        t0 = time.monotonic()

        # Load scene
        load_scene(self._sim, scene)

        # Build waypoint policy
        policy = WaypointMotorPolicy(
            waypoints=scene.object_positions,
            agent_id=self._agent_id,
            dwell_steps=self._dwell_steps,
            transit_steps=self._transit_steps,
            orbit_radius=self._orbit_radius,
        )

        # Run traversals
        traversals = []
        for i in range(n_traversals):
            result = self._run_traversal(scene, policy, learn=True)
            traversals.append(result)
            logger.info(
                "Traversal %d/%d: mean_surprise=%.3f (dwell=%.3f, transit=%s)",
                i + 1,
                n_traversals,
                result.mean_surprise,
                result.mean_dwell_surprise,
                f"{result.mean_transit_surprise:.3f}"
                if result.mean_transit_surprise is not None
                else "N/A",
            )

        elapsed = time.monotonic() - t0

        # Compute aggregate metrics
        surprise_first = traversals[0].mean_surprise
        surprise_last = traversals[-1].mean_surprise
        habituation_ratio = (
            surprise_last / surprise_first if surprise_first > 0 else 1.0
        )

        # Dwell vs transit ratio (across all traversals)
        all_dwell = []
        all_transit = []
        for t in traversals:
            all_dwell.append(t.mean_dwell_surprise)
            if t.mean_transit_surprise is not None:
                all_transit.append(t.mean_transit_surprise)

        mean_dwell = float(np.mean(all_dwell)) if all_dwell else 0.0
        mean_transit = float(np.mean(all_transit)) if all_transit else None
        dwell_transit_ratio = (
            mean_transit / mean_dwell
            if mean_transit is not None and mean_dwell > 0
            else None
        )

        config = {
            "scene_name": scene.name,
            "n_traversals": n_traversals,
            "dwell_steps": self._dwell_steps,
            "transit_steps": self._transit_steps,
            "orbit_radius": self._orbit_radius,
            "resolution": self._resolution,
            "fov": self._fov,
            "seed": self._seed,
        }

        return SceneEvalResult(
            scene_name=scene.name,
            traversals=traversals,
            surprise_first=surprise_first,
            surprise_last=surprise_last,
            habituation_ratio=habituation_ratio,
            dwell_transit_ratio=dwell_transit_ratio,
            wall_clock_seconds=elapsed,
            config=config,
        )

    def run_swap_detection(
        self,
        scene: SceneSpec,
        swap_index: int,
        replacement_ycb: str,
        n_learning_traversals: int = 3,
        detection_threshold: float = 0.1,
    ) -> SwapDetectionResult:
        """Test whether TM detects an object swap in a familiar scene.

        1. Learn the original scene over ``n_learning_traversals``.
        2. Run one traversal of the original scene (baseline surprise).
        3. Swap the object at ``swap_index`` and run one traversal.
        4. Compare surprise at the swapped waypoint.

        Parameters
        ----------
        scene : SceneSpec
            Original scene.
        swap_index : int
            Index of the object to replace.
        replacement_ycb : str
            YCB name for the replacement object.
        n_learning_traversals : int
            Traversals to learn the original scene before testing.
        detection_threshold : float
            Minimum surprise_delta to count as "detected".

        Returns
        -------
        SwapDetectionResult
        """
        if self._sim is None:
            self._setup()

        # Reset TM for a fresh experiment
        self._tm = TemporalMemory(**self._tm_kwargs)

        policy = WaypointMotorPolicy(
            waypoints=scene.object_positions,
            agent_id=self._agent_id,
            dwell_steps=self._dwell_steps,
            transit_steps=self._transit_steps,
            orbit_radius=self._orbit_radius,
        )

        # Phase 1: Learn the original scene
        load_scene(self._sim, scene)
        for i in range(n_learning_traversals):
            self._run_traversal(scene, policy, learn=True)

        # Phase 2: Baseline traversal (no learning, measure surprise)
        baseline = self._run_traversal(scene, policy, learn=False)

        # Phase 3: Swap and measure
        modified_scene = swap_object(self._sim, scene, swap_index, replacement_ycb)
        load_scene(self._sim, modified_scene)

        # Use same waypoints (same positions)
        swapped = self._run_traversal(modified_scene, policy, learn=False)

        # Extract surprise at the swapped waypoint
        def mean_surprise_at_waypoint(result: TraversalResult, wp_idx: int) -> float:
            vals = [
                s
                for s, w in zip(result.surprise_per_step, result.waypoint_indices)
                if w == wp_idx
            ]
            return float(np.mean(vals)) if vals else 0.0

        surprise_orig = mean_surprise_at_waypoint(baseline, swap_index)
        surprise_swap = mean_surprise_at_waypoint(swapped, swap_index)
        delta = surprise_swap - surprise_orig

        original_object = scene.objects[swap_index].ycb_name

        logger.info(
            "Swap detection: %s→%s at index %d: "
            "surprise orig=%.3f, swap=%.3f, delta=%.3f, detected=%s",
            original_object,
            replacement_ycb,
            swap_index,
            surprise_orig,
            surprise_swap,
            delta,
            delta > detection_threshold,
        )

        return SwapDetectionResult(
            scene_name=scene.name,
            swap_index=swap_index,
            original_object=original_object,
            replacement_object=replacement_ycb,
            surprise_at_swap_original=surprise_orig,
            surprise_at_swap_modified=surprise_swap,
            surprise_delta=delta,
            detected=delta > detection_threshold,
        )

    def get_tm(self) -> TemporalMemory:
        """Return the underlying TemporalMemory for inspection."""
        if self._tm is None:
            raise RuntimeError("Not yet initialized. Call run_scene() first.")
        return self._tm

    def close(self) -> None:
        """Release simulator resources."""
        if self._sim is not None:
            self._sim.close()
            self._sim = None
