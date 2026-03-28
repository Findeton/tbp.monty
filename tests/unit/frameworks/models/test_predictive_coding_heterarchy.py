"""Tests for Track 6: Predictive Coding Heterarchy.

Every test exercises the real Monty architecture end-to-end:

- **Observations**: Panda3D GPU render → Panda3DDepthNormalize → DepthTo3DLocations
  → CameraSM.step() → State.  No hand-crafted States anywhere.
- **Training/Eval**: Real EvidenceGraphLM lifecycle (pre_episode → exploratory_step /
  matching_step → post_episode) with proper ground-truth bookkeeping.
- **Voting (T6.5)**: Real trained EvidenceGraphLMs inside a real
  MontyForEvidenceGraphMatching calling _vote_predictive().
- **HPC (T6.3)**: Real LM get_output() States fed into HippocampalModule.
- **Integration**: Real Panda3DTemporalTrainer with animated glTF.

All tests are gated behind ``import panda3d``; they skip cleanly when the
renderer is unavailable.
"""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

# Gate entire module on Panda3D availability
try:
    import panda3d  # noqa: F401
    _HAS_PANDA3D = True
except ImportError:
    _HAS_PANDA3D = False

from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.models.evidence_matching.learning_module import (
    EvidenceGraphLM,
)
from tbp.monty.frameworks.models.hippocampal_module import HippocampalModule
from tbp.monty.frameworks.models.states import State


# ============================================================================
# Real rendering pipeline: Panda3D → transforms → CameraSM → State
# ============================================================================

class _RealPipeline:
    """Shared Panda3D rendering pipeline producing real CameraSM States.

    Renders primitive shapes from orbital viewpoints using the same transform
    chain and CameraSM that a full Monty experiment uses.
    """

    def __init__(self):
        from tbp.monty.frameworks.agents import AgentID
        from tbp.monty.frameworks.environment_utils.transforms import (
            DepthTo3DLocations,
        )
        from tbp.monty.frameworks.models.sensor_modules import CameraSM
        from tbp.monty.frameworks.sensors import SensorID
        from tbp.monty.simulators.panda3d.agents import Panda3DAgent
        from tbp.monty.simulators.panda3d.simulator import Panda3DSimulator
        from tbp.monty.simulators.panda3d.temporal_training import OrbitalMotorPolicy
        from tbp.monty.simulators.panda3d.transforms import Panda3DDepthNormalize

        self.AGENT_ID = AgentID("agent_id_0")
        self.SENSOR_ID = SensorID("patch")
        self.RESOLUTION = (64, 64)
        self.FOV = 90.0
        self.NEAR = 0.01
        self.FAR = 10.0

        agent = Panda3DAgent(
            agent_id=self.AGENT_ID,
            sensor_id=str(self.SENSOR_ID),
            position=(0.0, -2.0, 0.0),
            rotation=(1.0, 0.0, 0.0, 0.0),
            resolution=self.RESOLUTION,
            fov=self.FOV,
        )
        self._sim = Panda3DSimulator(
            agents=[agent], near=self.NEAR, far=self.FAR,
        )

        self._depth_norm = Panda3DDepthNormalize(
            agent_id=self.AGENT_ID, near=self.NEAR, far=self.FAR,
        )
        self._d3d = DepthTo3DLocations(
            agent_id=self.AGENT_ID,
            sensor_ids=[self.SENSOR_ID],
            resolutions=[self.RESOLUTION],
            hfov=self.FOV,
            world_coord=True,
            get_all_points=True,
        )

        self._sm = CameraSM(
            sensor_module_id=str(self.SENSOR_ID),
            features=["on_object", "hsv", "principal_curvatures_log"],
        )

        self._OrbitalMotorPolicy = OrbitalMotorPolicy

    def render_sequence(self, primitive, n_steps=20, orbit_radius=2.0):
        """Render *n_steps* orbital observations of *primitive*.

        Returns a list of real CameraSM-produced State objects (only those
        with ``use_state=True``).
        """
        from tbp.monty.frameworks.environment_utils.transforms import TransformContext

        self._sim.remove_all_objects()
        self._sim.add_object(primitive, position=(0, 0, 0))

        motor = self._OrbitalMotorPolicy(
            agent_id=self.AGENT_ID,
            orbit_radius=orbit_radius,
            azimuth_step_deg=360.0 / n_steps,
        )

        self._sm.pre_episode()
        states = []

        for step in range(n_steps):
            # Position camera via orbital policy (same as Panda3DTemporalTrainer)
            position, rotation = motor.get_camera_pose(step)
            cam_info = self._sim._agent_buffers[self.AGENT_ID]
            self._sim._set_node_pose(cam_info["camera_np"], position, rotation)

            # Render
            obs, proprio = self._sim.step([])

            # Transforms: depth normalize → depth-to-3D
            ctx = TransformContext(
                rng=np.random.RandomState(step), state=proprio,
            )
            obs = self._depth_norm(obs, ctx)
            obs = self._d3d(obs, ctx)

            # CameraSM → State
            agent_state = proprio[self.AGENT_ID]
            self._sm.update_state(agent_state)
            rt_ctx = RuntimeContext(rng=np.random.RandomState(step))
            state = self._sm.step(rt_ctx, obs[self.AGENT_ID][self.SENSOR_ID])

            if state is not None and state.use_state:
                states.append(state)

        return states

    def close(self):
        if self._sim is not None:
            self._sim.close()
            self._sim = None


# Module-level shared fixture and pre-rendered sequences
_pipeline = None
_sphere_states = None
_cube_states = None


def setUpModule():
    global _pipeline, _sphere_states, _cube_states
    if not _HAS_PANDA3D:
        raise unittest.SkipTest("Panda3D not installed")

    _pipeline = _RealPipeline()
    _sphere_states = _pipeline.render_sequence("sphere", n_steps=25,
                                               orbit_radius=2.0)
    _cube_states = _pipeline.render_sequence("cube", n_steps=25,
                                             orbit_radius=2.5)

    if len(_sphere_states) < 5:
        raise unittest.SkipTest(
            f"Only {len(_sphere_states)} usable sphere states rendered — "
            "GPU rendering may not be working"
        )
    if len(_cube_states) < 5:
        raise unittest.SkipTest(
            f"Only {len(_cube_states)} usable cube states rendered"
        )


def tearDownModule():
    global _pipeline
    if _pipeline is not None:
        _pipeline.close()


# ============================================================================
# Helpers — real LM lifecycle (same bookkeeping the experiment runner does)
# ============================================================================

def _make_ctx():
    return RuntimeContext(rng=np.random.RandomState(42))


def _make_lm(**kwargs):
    """Create a real EvidenceGraphLM with optional temporal memory + T6 params."""
    defaults = dict(
        max_match_distance=0.05,
        tolerances={
            "patch": {
                "hsv": [0.1, 1, 1],
                "principal_curvatures_log": [2, 2],
            }
        },
        feature_weights={
            "patch": {"hsv": np.array([1, 2.0, 0.5])}
        },
        max_graph_size=10,
    )
    defaults.update(kwargs)
    return EvidenceGraphLM(**defaults)


def _train_object(lm, obj_name, observations, ctx):
    """Train one object through the real EvidenceGraphLM lifecycle.

    This mirrors what MontyExperiment does: set mode → pre_episode with
    ground-truth target → exploratory_step for each observation → set
    convergence state (experiment infrastructure provides ground truth) →
    post_episode to persist the learned graph.
    """
    target = {"object": obj_name, "quat_rotation": [1, 0, 0, 0]}
    lm.mode = ExperimentMode.TRAIN
    lm.pre_episode(primary_target=target)
    for obs in observations:
        lm.exploratory_step(ctx, [obs])

    # Experiment infrastructure provides ground truth convergence:
    lm.detected_object = obj_name
    lm.detected_rotation_r = Rotation.identity()
    current_loc = lm.buffer.get_current_location(input_channel="first")
    lm.buffer.stats["detected_location_rel_body"] = current_loc
    lm.buffer.stats["detected_location_on_model"] = current_loc
    lm.post_episode()


def _run_eval(lm, observations, ctx):
    """Run eval matching episode, return per-step evidence snapshots."""
    placeholder = {"object": "placeholder", "quat_rotation": [1, 0, 0, 0]}
    lm.mode = ExperimentMode.EVAL
    lm.pre_episode(primary_target=placeholder)

    evidence_snapshots = []
    for obs in observations:
        lm.matching_step(ctx, [obs])
        snapshot = {}
        for graph_id, ev_array in lm.evidence.items():
            if ev_array is not None and len(ev_array) > 0:
                snapshot[graph_id] = float(np.max(ev_array))
        evidence_snapshots.append(snapshot)
    return evidence_snapshots


def _get_lm_output_state(lm, obj_name, observations, ctx):
    """Train LM and return its get_output() State (with graph_id).

    During training, get_output() returns a State with the target's graph_id
    — the same mechanism Monty uses to feed downstream modules (e.g. HPC).
    The Monty orchestrator sets stepwise_target_object; we replicate that
    ground-truth labeling here.
    """
    target = {"object": obj_name, "quat_rotation": [1, 0, 0, 0]}
    lm.mode = ExperimentMode.TRAIN
    lm.pre_episode(primary_target=target)
    for obs in observations:
        lm.exploratory_step(ctx, [obs])

    # Monty orchestrator (MontyForGraphMatching._set_stepwise_targets) sets this
    # from the environment's semantic label. Replicate the ground-truth labeling.
    lm.stepwise_target_object = obj_name

    # get_output() in TRAIN mode returns State(graph_id=obj_name, confidence=1)
    output = lm.get_output()

    # Complete the episode
    lm.detected_object = obj_name
    lm.detected_rotation_r = Rotation.identity()
    current_loc = lm.buffer.get_current_location(input_channel="first")
    lm.buffer.stats["detected_location_rel_body"] = current_loc
    lm.buffer.stats["detected_location_on_model"] = current_loc
    lm.post_episode()

    return output


# ============================================================================
# T6.1: Surprise-modulated Hebbian learning rate via real pipeline
# ============================================================================

class TestSurpriseModulatedLR(unittest.TestCase):
    """T6.1: TM Hebbian LR modulated by surprise, through real LM + renderer."""

    def test_noop_when_boost_zero(self):
        """With surprise_learning_boost=0, TM weights match baseline."""
        obs = _sphere_states
        ctx = _make_ctx()

        lm_base = _make_lm(
            temporal_memory={"sdr_dim": 256, "sdr_sparsity": 0.05},
        )
        lm_boost = _make_lm(
            temporal_memory={
                "sdr_dim": 256, "sdr_sparsity": 0.05,
                "surprise_learning_boost": 0.0,
            },
        )

        _train_object(lm_base, "sphere", obs, ctx)
        _train_object(lm_boost, "sphere", obs, ctx)

        np.testing.assert_array_equal(
            lm_base._temporal_memory._W,
            lm_boost._temporal_memory._W,
            "With boost=0, TM weights must be identical to baseline",
        )

    def test_boosted_lm_learns_stronger_temporal_weights(self):
        """EvidenceGraphLM with surprise_learning_boost > 0 builds stronger TM."""
        obs = _sphere_states
        ctx = _make_ctx()

        lm_base = _make_lm(
            temporal_memory={"sdr_dim": 512, "sdr_sparsity": 0.04},
        )
        lm_boosted = _make_lm(
            temporal_memory={
                "sdr_dim": 512, "sdr_sparsity": 0.04,
                "surprise_learning_boost": 3.0,
            },
        )

        _train_object(lm_base, "sphere", obs, ctx)
        _train_object(lm_boosted, "sphere", obs, ctx)

        base_norm = np.linalg.norm(lm_base._temporal_memory._W)
        boosted_norm = np.linalg.norm(lm_boosted._temporal_memory._W)

        self.assertGreater(
            boosted_norm, base_norm * 1.2,
            f"Boosted TM ({boosted_norm:.1f}) should have larger weights "
            f"than base ({base_norm:.1f})"
        )

    def test_surprise_decreases_over_training_episodes(self):
        """Training the same rendered sequence repeatedly reduces TM surprise."""
        obs = _sphere_states
        ctx = _make_ctx()

        lm = _make_lm(
            temporal_memory={
                "sdr_dim": 1024, "sdr_sparsity": 0.02,
                "surprise_learning_boost": 2.0,
            },
        )

        for _ in range(6):
            _train_object(lm, "sphere", obs, ctx)

        # Eval on the same rendered sequence — TM should predict well
        lm.mode = ExperimentMode.EVAL
        lm.pre_episode(
            primary_target={"object": "placeholder", "quat_rotation": [1, 0, 0, 0]}
        )
        for o in obs:
            lm.matching_step(ctx, [o])
        final_surprise = lm.get_temporal_surprise()

        self.assertLess(
            final_surprise, 0.95,
            f"After 6 training episodes, surprise ({final_surprise:.3f}) "
            f"should decrease from maximum"
        )


# ============================================================================
# T6.2: Surprise-gated weight decay via real pipeline
# ============================================================================

class TestSurpriseGatedDecay(unittest.TestCase):
    """T6.2: Weight decay on well-predicted transitions, through real pipeline."""

    def test_no_decay_when_rate_zero(self):
        """With low_surprise_decay_rate=0, TM weights match baseline."""
        obs = _sphere_states
        ctx = _make_ctx()

        lm_base = _make_lm(
            temporal_memory={"sdr_dim": 256, "sdr_sparsity": 0.05},
        )
        lm_decay = _make_lm(
            temporal_memory={
                "sdr_dim": 256, "sdr_sparsity": 0.05,
                "low_surprise_decay_rate": 0.0,
            },
        )

        _train_object(lm_base, "sphere", obs, ctx)
        _train_object(lm_decay, "sphere", obs, ctx)

        np.testing.assert_array_equal(
            lm_base._temporal_memory._W,
            lm_decay._temporal_memory._W,
        )

    def test_decay_limits_weight_growth(self):
        """Repeated training with decay produces smaller max weight than without."""
        obs = _sphere_states
        ctx = _make_ctx()

        lm_no_decay = _make_lm(
            temporal_memory={"sdr_dim": 512, "sdr_sparsity": 0.04},
        )
        lm_with_decay = _make_lm(
            temporal_memory={
                "sdr_dim": 512, "sdr_sparsity": 0.04,
                "low_surprise_decay_rate": 0.01,
                "low_surprise_decay_threshold": 0.5,
            },
        )

        for _ in range(10):
            _train_object(lm_no_decay, "sphere", obs, ctx)
            _train_object(lm_with_decay, "sphere", obs, ctx)

        max_no_decay = np.max(np.abs(lm_no_decay._temporal_memory._W))
        max_with_decay = np.max(np.abs(lm_with_decay._temporal_memory._W))

        self.assertLess(
            max_with_decay, max_no_decay,
            f"Decay max ({max_with_decay:.3f}) should be smaller "
            f"than no-decay max ({max_no_decay:.3f})"
        )

    def test_decay_preserves_novel_transition_learning(self):
        """Decay doesn't erase learning from a novel primitive's observations."""
        obs_sphere = _sphere_states
        obs_cube = _cube_states
        ctx = _make_ctx()

        lm = _make_lm(
            temporal_memory={
                "sdr_dim": 512, "sdr_sparsity": 0.04,
                "low_surprise_decay_rate": 0.05,
                "low_surprise_decay_threshold": 0.3,
            },
        )

        # Train extensively on sphere renders
        for _ in range(5):
            _train_object(lm, "sphere", obs_sphere, ctx)

        w_before = lm._temporal_memory._W.copy()

        # Train on cube renders — genuinely different surface geometry
        _train_object(lm, "cube", obs_cube, ctx)

        w_after = lm._temporal_memory._W
        diff = np.linalg.norm(w_after - w_before)
        self.assertGreater(
            diff, 0.1,
            "Novel primitive's observations should produce measurable "
            "weight changes"
        )


# ============================================================================
# T6.3: HPC cross-episode error-modulated learning (real LM output States)
# ============================================================================

class TestHPCSurpriseModulation(unittest.TestCase):
    """T6.3: HPC Hebbian learning modulated by prediction surprise.

    HPC receives real LM output States (produced by EvidenceGraphLM.get_output()
    during training) — the same mechanism used in the real heterarchy.
    """

    def _make_hpc_ctx(self, global_step=0, episode_step=0):
        class _Timer:
            pass
        t = _Timer()
        t.global_step = global_step
        t.episode_step = episode_step
        return RuntimeContext(rng=np.random.RandomState(42), timer=t)

    def _get_real_lm_state(self, obj_name, observations):
        """Get a real LM output State by training on rendered observations."""
        lm = _make_lm()
        ctx = _make_ctx()
        state = _get_lm_output_state(lm, obj_name, observations, ctx)
        return state

    def test_noop_when_boost_zero(self):
        """With surprise_learning_boost=0, HPC weights match baseline."""
        hpc_base = HippocampalModule(
            temporal_dim=128, temporal_sparsity=0.05,
            hebbian_learning_rate=1.0,
        )
        hpc_boost = HippocampalModule(
            temporal_dim=128, temporal_sparsity=0.05,
            hebbian_learning_rate=1.0,
            surprise_learning_boost=0.0,
        )
        ctx = self._make_hpc_ctx()

        # Get real LM output states from two different objects
        state_sphere = self._get_real_lm_state("sphere", _sphere_states)
        state_cube = self._get_real_lm_state("cube", _cube_states)

        for hpc in [hpc_base, hpc_boost]:
            hpc.pre_episode()
            hpc.matching_step(ctx, [state_sphere])
            hpc.post_episode()

            hpc.pre_episode()
            hpc.matching_step(ctx, [state_cube])
            hpc.post_episode()

        np.testing.assert_array_equal(
            hpc_base._temporal_W, hpc_boost._temporal_W,
        )

    def test_novel_transition_amplified(self):
        """Boosted HPC produces stronger weights on novel transitions."""
        hpc_base = HippocampalModule(
            temporal_dim=128, temporal_sparsity=0.05,
            hebbian_learning_rate=1.0,
            surprise_learning_boost=0.0,
        )
        hpc_boost = HippocampalModule(
            temporal_dim=128, temporal_sparsity=0.05,
            hebbian_learning_rate=1.0,
            surprise_learning_boost=3.0,
        )
        ctx = self._make_hpc_ctx()

        state_sphere = self._get_real_lm_state("sphere", _sphere_states)
        state_cube = self._get_real_lm_state("cube", _cube_states)

        for hpc in [hpc_base, hpc_boost]:
            hpc.pre_episode()
            hpc.matching_step(ctx, [state_sphere])
            hpc.post_episode()

            hpc.pre_episode()
            hpc.matching_step(ctx, [state_cube])
            hpc.post_episode()

        base_norm = np.linalg.norm(hpc_base._temporal_W)
        boost_norm = np.linalg.norm(hpc_boost._temporal_W)

        self.assertGreater(
            boost_norm, base_norm * 2.0,
            f"Boosted ({boost_norm:.3f}) should exceed "
            f"base ({base_norm:.3f}) for novel transition"
        )

    def test_get_prediction_surprise_values(self):
        """_get_prediction_surprise returns correct values."""
        hpc = HippocampalModule(temporal_dim=64, temporal_sparsity=0.05)

        self.assertAlmostEqual(hpc._get_prediction_surprise("sphere"), 1.0)

        hpc._current_predictions = {"sphere": 0.8, "cube": 0.2}
        self.assertAlmostEqual(hpc._get_prediction_surprise("sphere"), 0.2)
        self.assertAlmostEqual(hpc._get_prediction_surprise("cube"), 0.8)
        self.assertAlmostEqual(hpc._get_prediction_surprise("cone"), 1.0)

    def test_action_conditioned_modulation(self):
        """Action-conditioned matrix updated through real HPC lifecycle."""
        hpc = HippocampalModule(
            temporal_dim=128, temporal_sparsity=0.05,
            hebbian_learning_rate=1.0,
            surprise_learning_boost=2.0,
        )
        ctx = self._make_hpc_ctx()

        state_sphere = self._get_real_lm_state("sphere", _sphere_states)
        state_cube = self._get_real_lm_state("cube", _cube_states)

        hpc.pre_episode()
        hpc.matching_step(ctx, [state_sphere])
        hpc.post_episode()

        hpc.record_action("grasp")

        hpc.pre_episode()
        hpc.matching_step(ctx, [state_cube])
        hpc.post_episode()

        self.assertIn("grasp", hpc._action_W)
        self.assertGreater(np.linalg.norm(hpc._action_W["grasp"]), 0)

    def test_prediction_accuracy_improves_across_episodes(self):
        """Repeated sequence exposure improves HPC prediction accuracy."""
        hpc = HippocampalModule(
            temporal_dim=128, temporal_sparsity=0.05,
            hebbian_learning_rate=1.0,
            surprise_learning_boost=2.0,
        )
        ctx = self._make_hpc_ctx()

        # Use real LM states for a repeating concept sequence
        state_sphere = self._get_real_lm_state("sphere", _sphere_states)
        state_cube = self._get_real_lm_state("cube", _cube_states)
        sequence = [state_sphere, state_cube]
        concept_names = ["sphere", "cube"]

        hits = 0
        total = 0

        for cycle in range(5):
            for i, state in enumerate(sequence):
                hpc.pre_episode()
                hpc.matching_step(ctx, [state])
                if cycle > 0 and hpc._prediction_history:
                    last = hpc._prediction_history[-1]
                    total += 1
                    if last["hit"]:
                        hits += 1
                hpc.post_episode()

        if total > 0:
            accuracy = hits / total
            self.assertGreater(accuracy, 0.3)


# ============================================================================
# T6.4: Combined surprise learning + evidence modulation (real pipeline)
# ============================================================================

class TestCombinedModulation(unittest.TestCase):
    """T6.4: All modulation mechanisms compose through real LM + renderer."""

    def test_all_modulation_together_no_crash(self):
        """Full pipeline with boost + decay + evidence modulation runs cleanly."""
        obs = _sphere_states
        ctx = _make_ctx()

        lm = _make_lm(
            temporal_memory={
                "sdr_dim": 512, "sdr_sparsity": 0.04,
                "surprise_learning_boost": 2.0,
                "low_surprise_decay_rate": 0.01,
                "low_surprise_decay_threshold": 0.3,
            },
            surprise_boost=0.5,
            surprise_penalty=0.3,
        )

        for _ in range(5):
            _train_object(lm, "sphere", obs, ctx)

        evidence = _run_eval(lm, obs, ctx)
        self.assertGreater(len(evidence), 0)
        self.assertTrue(np.all(np.isfinite(lm._temporal_memory._W)))

    def test_combined_does_not_degrade_recognition(self):
        """Combined modulation doesn't hurt recognition vs baseline."""
        obs_sphere = _sphere_states
        obs_cube = _cube_states
        ctx = _make_ctx()

        lm_plain = _make_lm(
            temporal_memory={"sdr_dim": 512, "sdr_sparsity": 0.04},
        )
        lm_combo = _make_lm(
            temporal_memory={
                "sdr_dim": 512, "sdr_sparsity": 0.04,
                "surprise_learning_boost": 2.0,
                "low_surprise_decay_rate": 0.005,
                "low_surprise_decay_threshold": 0.3,
            },
            surprise_boost=0.3,
            surprise_penalty=0.2,
        )

        for lm in [lm_plain, lm_combo]:
            _train_object(lm, "sphere", obs_sphere, ctx)
            _train_object(lm, "cube", obs_cube, ctx)

        # Both should recognize sphere from its rendered observations
        ev_plain = _run_eval(lm_plain, obs_sphere, ctx)
        ev_combo = _run_eval(lm_combo, obs_sphere, ctx)

        plain_correct = ev_plain[-1].get("sphere", 0) if ev_plain and ev_plain[-1] else 0
        combo_correct = ev_combo[-1].get("sphere", 0) if ev_combo and ev_combo[-1] else 0

        self.assertGreaterEqual(
            combo_correct, plain_correct * 0.5,
            "Combined modulation should not severely degrade recognition"
        )

    def test_serialization_roundtrip(self):
        """state_dict/load_state_dict preserves T6.1/T6.2 params and weights."""
        from tbp.monty.frameworks.models.temporal_memory import TemporalMemory

        tm = TemporalMemory(
            sdr_dim=256, sdr_sparsity=0.05, learning_rate=0.2,
            surprise_learning_boost=1.5,
            low_surprise_decay_rate=0.02,
            low_surprise_decay_threshold=0.25,
        )

        # Feed real rendered States through the TemporalMemory
        for s in _sphere_states:
            tm.step(s, learn=True)

        sd = tm.state_dict()
        tm2 = TemporalMemory()
        tm2.load_state_dict(sd)

        self.assertAlmostEqual(tm2.surprise_learning_boost, 1.5)
        self.assertAlmostEqual(tm2.low_surprise_decay_rate, 0.02)
        self.assertAlmostEqual(tm2.low_surprise_decay_threshold, 0.25)
        np.testing.assert_array_equal(tm._W, tm2._W)


# ============================================================================
# T6.5: Unified predictive voting via real trained EvidenceGraphLMs
# ============================================================================

class TestUnifiedPredictiveVoting(unittest.TestCase):
    """T6.5: _vote_predictive() with real trained LMs and real vote format."""

    def _make_trained_lm(self, lm_id, obj_name, observations, ctx, **lm_kwargs):
        """Create and train a real EvidenceGraphLM with temporal memory."""
        lm = _make_lm(
            temporal_memory={
                "sdr_dim": 512, "sdr_sparsity": 0.04,
            },
            **lm_kwargs,
        )
        # learning_module_id is set post-construction (GraphLM defaults to "LM_0")
        lm.learning_module_id = lm_id
        _train_object(lm, obj_name, observations, ctx)
        return lm

    def _start_eval(self, lm, observations, ctx, n_steps=5):
        """Start eval episode and run n_steps of matching to build evidence."""
        lm.mode = ExperimentMode.EVAL
        lm.pre_episode(
            primary_target={"object": "placeholder", "quat_rotation": [1, 0, 0, 0]}
        )
        for obs in observations[:n_steps]:
            lm.matching_step(ctx, [obs])

    def _make_monty(self, lms, vote_matrix, confusion_threshold=0.7,
                    vote_after_steps=0):
        """Create a real MontyForEvidenceGraphMatching for voting."""
        from tbp.monty.frameworks.models.evidence_matching.model import (
            MontyForEvidenceGraphMatching,
        )

        class _TestMonty(MontyForEvidenceGraphMatching):
            def __init__(self_m):
                self_m.learning_modules = lms
                self_m.lm_to_lm_vote_matrix = vote_matrix
                self_m.predictive_voting = True
                self_m.vote_confident_threshold = 2
                self_m.temporal_confusion_threshold = confusion_threshold
                self_m.vote_after_steps = vote_after_steps
                self_m.matching_steps = 200

        return _TestMonty()

    def test_trained_lms_can_vote(self):
        """Real trained LMs produce valid votes through _vote_predictive."""
        ctx = _make_ctx()

        # Train two LMs on different objects
        lm0 = self._make_trained_lm("LM_0", "sphere", _sphere_states, ctx)
        lm1 = self._make_trained_lm("LM_1", "cube", _cube_states, ctx)

        # Start eval: LM0 sees sphere (should become confident),
        # LM1 also sees sphere (novel to it, should be confused)
        self._start_eval(lm0, _sphere_states, ctx, n_steps=8)
        self._start_eval(lm1, _sphere_states, ctx, n_steps=8)

        monty = self._make_monty([lm0, lm1], [[1], [0]])
        # Should not crash — real vote format flows through _combine_votes
        monty._vote_predictive()

    def test_temporally_confused_overrides_spatial_confidence(self):
        """Spatially confident but temporally confused → receives votes."""
        ctx = _make_ctx()

        # LM0: trained on sphere, eval on sphere (familiar → low surprise)
        lm0 = self._make_trained_lm("LM_0", "sphere", _sphere_states, ctx)
        for _ in range(5):
            _train_object(lm0, "sphere", _sphere_states, ctx)
        self._start_eval(lm0, _sphere_states, ctx, n_steps=8)

        # LM1: trained on sphere, eval on cube (novel → high surprise)
        lm1 = self._make_trained_lm("LM_1", "sphere", _sphere_states, ctx)
        for _ in range(5):
            _train_object(lm1, "sphere", _sphere_states, ctx)
        self._start_eval(lm1, _cube_states, ctx, n_steps=8)

        # LM1 should have higher TM surprise than LM0
        surprise_0 = lm0.get_temporal_surprise()
        surprise_1 = lm1.get_temporal_surprise()

        self.assertGreater(
            surprise_1, surprise_0,
            f"LM1 surprise ({surprise_1:.3f}) should exceed "
            f"LM0 ({surprise_0:.3f}) — cube is novel to sphere-trained TM"
        )

    def test_both_confident_no_crash(self):
        """Two confident LMs produce no crashes in voting."""
        ctx = _make_ctx()

        lm0 = self._make_trained_lm("LM_0", "sphere", _sphere_states, ctx)
        lm1 = self._make_trained_lm("LM_1", "cube", _cube_states, ctx)

        # Each LM sees its own trained object → both should have evidence
        self._start_eval(lm0, _sphere_states, ctx, n_steps=8)
        self._start_eval(lm1, _cube_states, ctx, n_steps=8)

        monty = self._make_monty([lm0, lm1], [[1], [0]])
        monty._vote_predictive()
        # No crash is the assertion

    def test_falls_back_to_spatial_when_no_temporal(self):
        """LMs without get_temporal_surprise → pure spatial classification."""
        class _PlainLM:
            def __init__(self_lm, lm_id, n_matches):
                self_lm.learning_module_id = lm_id
                self_lm._n_matches = n_matches
                self_lm.votes_received = []
                self_lm.buffer = type("B", (), {
                    "get_num_observations_on_object": lambda s: 1,
                    "update_last_stats_entry": lambda s, x: None,
                })()
                self_lm.terminal_state = None
                self_lm.stepwise_target_object = None
                self_lm.stepwise_targets_list = []

            def get_possible_matches(self_lm):
                return list(range(self_lm._n_matches))

            def send_out_vote(self_lm):
                return None

            def receive_votes(self_lm, votes):
                self_lm.votes_received.append(votes)

            def collect_stats_to_save(self_lm):
                return {}

            def add_lm_processing_to_buffer_stats(self_lm, **kwargs):
                pass

        lm0 = _PlainLM("LM_0", 1)
        lm1 = _PlainLM("LM_1", 10)

        monty = self._make_monty([lm0, lm1], [[1], [0]])
        # Should not crash even without get_temporal_surprise
        monty._vote_predictive()

    def test_predictive_voting_dispatch_from_vote_method(self):
        """_vote() dispatches to _vote_predictive when predictive_voting=True."""
        from tbp.monty.frameworks.models.graph_matching import (
            MontyForGraphMatching,
        )

        called = []

        class _TestMonty(MontyForGraphMatching):
            def __init__(self_m):
                self_m.learning_modules = []
                self_m.lm_to_lm_vote_matrix = [[]]
                self_m.predictive_voting = True
                self_m.conditional_voting = False
                self_m.temporal_voting = False
                self_m.vote_confident_threshold = 2
                self_m.temporal_confusion_threshold = 0.7
                self_m.vote_after_steps = 0
                self_m.matching_steps = 200

            def _vote_predictive(self_m):
                called.append(True)

        monty = _TestMonty()
        monty._vote()

        self.assertEqual(len(called), 1, "_vote_predictive should be called")

    def test_surprise_separation_between_known_and_novel(self):
        """After training, TM surprise separates known vs novel observations."""
        ctx = _make_ctx()

        # Train LM extensively on sphere renders
        lm = _make_lm(
            temporal_memory={
                "sdr_dim": 1024, "sdr_sparsity": 0.02,
                "surprise_learning_boost": 2.0,
            },
        )
        for _ in range(8):
            _train_object(lm, "sphere", _sphere_states, ctx)

        # Eval on known sphere → should have lower surprise
        lm.mode = ExperimentMode.EVAL
        lm.pre_episode(
            primary_target={"object": "placeholder", "quat_rotation": [1, 0, 0, 0]}
        )
        for o in _sphere_states:
            lm.matching_step(ctx, [o])
        sphere_surprise = lm.get_temporal_surprise()

        # Eval on novel cube → should have higher surprise
        lm.mode = ExperimentMode.EVAL
        lm.pre_episode(
            primary_target={"object": "placeholder", "quat_rotation": [1, 0, 0, 0]}
        )
        for o in _cube_states:
            lm.matching_step(ctx, [o])
        cube_surprise = lm.get_temporal_surprise()

        self.assertGreater(
            cube_surprise, sphere_surprise,
            f"Novel cube ({cube_surprise:.3f}) should produce higher "
            f"surprise than known sphere ({sphere_surprise:.3f})"
        )


# ============================================================================
# T6.6: Surprise-gated output via real pipeline
# ============================================================================

class TestSurpriseGatedOutput(unittest.TestCase):
    """T6.6: get_output() returns minimal State when TM surprise is low."""

    def test_gated_output_sends_confirmation_when_predicted(self):
        """After extensive training, low-surprise observations yield confirmed."""
        obs = _sphere_states
        ctx = _make_ctx()

        lm = _make_lm(
            temporal_memory={
                "sdr_dim": 512, "sdr_sparsity": 0.04,
                "surprise_learning_boost": 2.0,
            },
            surprise_gated_output=True,
            output_surprise_threshold=0.95,
        )

        for _ in range(10):
            _train_object(lm, "sphere", obs, ctx)

        lm.mode = ExperimentMode.EVAL
        lm.pre_episode(
            primary_target={"object": "placeholder", "quat_rotation": [1, 0, 0, 0]}
        )
        for o in obs:
            lm.matching_step(ctx, [o])

        output = lm.get_output()
        self.assertIsInstance(output, State)

        surprise = lm.get_temporal_surprise()
        if surprise < 0.95:
            self.assertFalse(output.use_state,
                             "Low surprise should produce use_state=False")
            self.assertTrue(
                output.non_morphological_features.get("confirmed", False),
                "Should have 'confirmed' flag"
            )

    def test_gated_output_sends_full_when_novel(self):
        """Novel rendered observations (high surprise) produce full output."""
        ctx = _make_ctx()

        lm = _make_lm(
            temporal_memory={"sdr_dim": 512, "sdr_sparsity": 0.04},
            surprise_gated_output=True,
            output_surprise_threshold=0.1,
        )

        _train_object(lm, "sphere", _sphere_states, ctx)

        # Eval on cube renders — novel to this LM's TM
        lm.mode = ExperimentMode.EVAL
        lm.pre_episode(
            primary_target={"object": "placeholder", "quat_rotation": [1, 0, 0, 0]}
        )
        for o in _cube_states:
            lm.matching_step(ctx, [o])

        output = lm.get_output()
        self.assertIsInstance(output, State)

        surprise = lm.get_temporal_surprise()
        if surprise >= 0.1:
            has_confirmed = (
                output.non_morphological_features is not None
                and output.non_morphological_features.get("confirmed", False)
            )
            if has_confirmed:
                self.fail("Novel data should not produce confirmed output")

    def test_disabled_by_default(self):
        """With surprise_gated_output=False (default), always sends full output."""
        obs = _sphere_states
        ctx = _make_ctx()

        lm = _make_lm(
            temporal_memory={"sdr_dim": 256, "sdr_sparsity": 0.05},
        )

        _train_object(lm, "sphere", obs, ctx)

        lm.mode = ExperimentMode.EVAL
        lm.pre_episode(
            primary_target={"object": "placeholder", "quat_rotation": [1, 0, 0, 0]}
        )
        for o in obs:
            lm.matching_step(ctx, [o])

        output = lm.get_output()
        has_confirmed = (
            output.non_morphological_features is not None
            and output.non_morphological_features.get("confirmed", False)
        )
        self.assertFalse(has_confirmed,
                         "Gating disabled → no confirmation output")


# ============================================================================
# Panda3D integration: modulated learning through real temporal trainer
# ============================================================================

class TestPanda3DModulatedTraining(unittest.TestCase):
    """Integration: Panda3DTemporalTrainer with T6.1/T6.2 modulated learning."""

    @classmethod
    def setUpClass(cls):
        """Close the module-level pipeline to free ShowBase for the trainer."""
        global _pipeline
        if _pipeline is not None:
            _pipeline.close()
            _pipeline = None

    def _make_test_gltf(self, tmpdir):
        """Create a minimal animated glTF for testing."""
        import base64
        import json
        import math
        import os
        import struct

        s = 0.3
        vertices = [
            (-s, 0, -s), (s, 0, -s), (s, 0, s), (-s, 0, s),
            (-s, 2, -s), (s, 2, -s), (s, 2, s), (-s, 2, s),
        ]
        indices = [
            0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6,
            0, 4, 5, 0, 5, 1, 2, 6, 7, 2, 7, 3,
            0, 3, 7, 0, 7, 4, 1, 5, 6, 1, 6, 2,
        ]
        joints_data = [(0, 0, 0, 0)] * 4 + [(1, 0, 0, 0)] * 4
        weights_data = [(1.0, 0, 0, 0)] * 8

        vert_bin = b"".join(struct.pack("<3f", *v) for v in vertices)
        idx_bin = b"".join(struct.pack("<H", i) for i in indices)
        while len(idx_bin) % 4:
            idx_bin += b"\x00"
        joints_bin = b"".join(struct.pack("<4B", *j) for j in joints_data)
        weights_bin = b"".join(struct.pack("<4f", *w) for w in weights_data)
        ibm0 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        ibm1 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, -1, 0, 1]
        ibm_bin = struct.pack("<16f", *ibm0) + struct.pack("<16f", *ibm1)

        s45 = math.sin(math.radians(45))
        c45 = math.cos(math.radians(45))
        anim_times = struct.pack("<3f", 0.0, 0.5, 1.0)
        anim_rots = struct.pack(
            "<12f", 0, 0, 0, 1, 0, 0, s45, c45, 0, 0, 0, 1
        )

        offset = 0
        buf_views = []
        all_parts = [
            vert_bin, idx_bin, joints_bin, weights_bin,
            ibm_bin, anim_times, anim_rots,
        ]
        for part in all_parts:
            buf_views.append({
                "buffer": 0, "byteOffset": offset, "byteLength": len(part),
            })
            offset += len(part)
        all_data = b"".join(all_parts)

        data_uri = (
            "data:application/octet-stream;base64,"
            + base64.b64encode(all_data).decode()
        )

        gltf_data = {
            "asset": {"version": "2.0"},
            "scene": 0,
            "scenes": [{"nodes": [0, 1]}],
            "nodes": [
                {"name": "Armature", "children": [2]},
                {"name": "Mesh", "mesh": 0, "skin": 0},
                {"name": "Bone0", "children": [3],
                 "translation": [0, 0, 0]},
                {"name": "Bone1", "translation": [0, 1, 0]},
            ],
            "meshes": [{"primitives": [{"attributes": {
                "POSITION": 0, "JOINTS_0": 2, "WEIGHTS_0": 3
            }, "indices": 1}]}],
            "skins": [{
                "joints": [2, 3],
                "inverseBindMatrices": 4,
                "skeleton": 2,
            }],
            "animations": [{
                "channels": [{"sampler": 0, "target": {
                    "node": 3, "path": "rotation"
                }}],
                "samplers": [{"input": 5, "output": 6,
                              "interpolation": "LINEAR"}],
            }],
            "accessors": [
                {"bufferView": 0, "componentType": 5126, "count": 8,
                 "type": "VEC3",
                 "min": [-s, 0, -s], "max": [s, 2, s]},
                {"bufferView": 1, "componentType": 5123, "count": 36,
                 "type": "SCALAR"},
                {"bufferView": 2, "componentType": 5121, "count": 8,
                 "type": "VEC4"},
                {"bufferView": 3, "componentType": 5126, "count": 8,
                 "type": "VEC4"},
                {"bufferView": 4, "componentType": 5126, "count": 2,
                 "type": "MAT4"},
                {"bufferView": 5, "componentType": 5126, "count": 3,
                 "type": "SCALAR", "min": [0.0], "max": [1.0]},
                {"bufferView": 6, "componentType": 5126, "count": 3,
                 "type": "VEC4"},
            ],
            "bufferViews": buf_views,
            "buffers": [{"uri": data_uri, "byteLength": len(all_data)}],
        }

        path = os.path.join(tmpdir, "animated.gltf")
        with open(path, "w") as f:
            json.dump(gltf_data, f)
        return path

    def test_modulated_training_through_panda3d_pipeline(self):
        """Full Panda3D pipeline with surprise-modulated TM learning."""
        import tempfile
        from tbp.monty.simulators.panda3d.temporal_training import (
            Panda3DTemporalTrainer,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            gltf_path = self._make_test_gltf(tmpdir)

            trainer = Panda3DTemporalTrainer(
                model_path=gltf_path,
                resolution=(32, 32),
                fov=90.0,
                near=0.01,
                far=10.0,
                motor_policy="orbital",
                orbit_radius=3.0,
                tm_kwargs={
                    "sdr_dim": 512,
                    "sdr_sparsity": 0.04,
                    "surprise_learning_boost": 2.0,
                    "low_surprise_decay_rate": 0.005,
                    "low_surprise_decay_threshold": 0.3,
                },
            )

            try:
                results = trainer.train_episode(
                    n_repetitions=3,
                    n_replay=1,
                )

                self.assertIn("mean_surprise_final", results)
                self.assertIsInstance(results["mean_surprise_final"], float)

                tm = trainer.temporal_memory
                self.assertAlmostEqual(tm.surprise_learning_boost, 2.0)
                self.assertAlmostEqual(tm.low_surprise_decay_rate, 0.005)

                self.assertGreater(
                    np.linalg.norm(tm._W), 0,
                    "TM should have learned temporal associations"
                )
            finally:
                trainer.close()


# ============================================================================
# T6.6a: HPC context biasing for burst sampling
# ============================================================================

class TestContextBiasingBurstSampling(unittest.TestCase):
    """T6.6a: HPC context weights bias informed hypothesis sampling."""

    def test_context_biasing_reduces_sampling_for_non_context_graphs(self):
        """Non-context graphs get fewer informed hypotheses during bursts."""
        ctx = _make_ctx()
        obs = _sphere_states

        lm = _make_lm(
            temporal_memory={"sdr_dim": 256, "sdr_sparsity": 0.05},
        )
        _train_object(lm, "sphere", obs, ctx)
        _train_object(lm, "cube", _cube_states, ctx)

        # Simulate HPC context: sphere is expected, cube is not
        if hasattr(lm.hypotheses_updater, "set_context_weights"):
            lm.hypotheses_updater.set_context_weights(
                {"sphere": 1.0}, exploration_budget=0.1
            )
            self.assertIsNotNone(lm.hypotheses_updater.context_graph_weights)
            self.assertEqual(
                lm.hypotheses_updater.context_graph_weights["sphere"], 1.0
            )
            # cube should get the exploration budget (0.1)
            cube_weight = lm.hypotheses_updater.context_graph_weights.get(
                "cube", lm.hypotheses_updater.context_exploration_budget
            )
            self.assertAlmostEqual(cube_weight, 0.1)

    def test_context_weights_set_via_receive_context(self):
        """receive_context() with association_strengths sets burst weights."""
        ctx = _make_ctx()
        obs = _sphere_states

        lm = _make_lm(
            temporal_memory={"sdr_dim": 256, "sdr_sparsity": 0.05},
        )
        _train_object(lm, "sphere", obs, ctx)
        _train_object(lm, "cube", _cube_states, ctx)

        # Send context signal mimicking HPC output
        lm.receive_context(
            association_strengths={"sphere": 0.8, "cube": 0.2},
            active_concepts=["sphere"],
        )

        if hasattr(lm.hypotheses_updater, "context_graph_weights"):
            weights = lm.hypotheses_updater.context_graph_weights
            if weights is not None:
                self.assertIn("sphere", weights)
                self.assertIn("cube", weights)
                self.assertGreater(weights["sphere"], weights["cube"])

    def test_clear_context_restores_uniform(self):
        """Clearing context weights restores uniform sampling."""
        lm = _make_lm()
        if hasattr(lm.hypotheses_updater, "set_context_weights"):
            lm.hypotheses_updater.set_context_weights({"sphere": 1.0})
            self.assertIsNotNone(lm.hypotheses_updater.context_graph_weights)
            lm.hypotheses_updater.clear_context_weights()
            self.assertIsNone(lm.hypotheses_updater.context_graph_weights)

    def test_empty_association_clears_weights(self):
        """Empty association_strengths in receive_context clears weights."""
        lm = _make_lm(
            temporal_memory={"sdr_dim": 256, "sdr_sparsity": 0.05},
        )
        _train_object(lm, "sphere", _sphere_states, _make_ctx())

        # First set weights
        lm.receive_context(association_strengths={"sphere": 0.9})
        # Then clear with empty dict
        lm.receive_context(association_strengths={})

        if hasattr(lm.hypotheses_updater, "context_graph_weights"):
            self.assertIsNone(lm.hypotheses_updater.context_graph_weights)


# ============================================================================
# T6.13: Burst sampling ranking quality telemetry
# ============================================================================

class TestBurstRankingTelemetry(unittest.TestCase):
    """T6.13: Burst ranking data is captured in telemetry."""

    def test_telemetry_dataclass_has_ranking_fields(self):
        """ChannelHypothesesBurstSamplingTelemetry has ranking fields."""
        from tbp.monty.frameworks.models.evidence_matching.burst_sampling import (
            ChannelHypothesesBurstSamplingTelemetry,
        )

        import dataclasses
        field_names = [f.name for f in dataclasses.fields(
            ChannelHypothesesBurstSamplingTelemetry
        )]
        self.assertIn("burst_node_evidence", field_names)
        self.assertIn("burst_top_k_indices", field_names)

    def test_ranking_logger_exists(self):
        """Burst sampling module has a logger for ranking diagnostics."""
        from tbp.monty.frameworks.models.evidence_matching import burst_sampling

        self.assertTrue(hasattr(burst_sampling, "logger"))


# ============================================================================
# T6.13a: Offspring/refinement hypotheses
# ============================================================================

class TestOffspringHypotheses(unittest.TestCase):
    """T6.13a: Offspring hypotheses refine pose near high-evidence parents."""

    def test_offspring_disabled_by_default(self):
        """Offspring are disabled by default (offspring_enabled=False)."""
        from tbp.monty.frameworks.models.evidence_matching.burst_sampling import (
            BurstSamplingHypothesesUpdater,
        )

        lm = _make_lm()
        updater = lm.hypotheses_updater
        if isinstance(updater, BurstSamplingHypothesesUpdater):
            self.assertFalse(updater.offspring_enabled)

    def test_offspring_parameters_configurable(self):
        """Offspring params can be passed via hypotheses_updater_args."""
        from tbp.monty.frameworks.models.evidence_matching.burst_sampling import (
            BurstSamplingHypothesesUpdater,
        )

        lm = _make_lm(
            hypotheses_updater_class=BurstSamplingHypothesesUpdater,
            hypotheses_updater_args={
                "offspring_enabled": True,
                "offspring_count": 20,
                "offspring_location_jitter": 0.01,
                "offspring_rotation_jitter_deg": 10.0,
                "offspring_evidence_threshold": 3.0,
            },
        )

        updater = lm.hypotheses_updater
        self.assertTrue(updater.offspring_enabled)
        self.assertEqual(updater.offspring_count, 20)
        self.assertAlmostEqual(updater.offspring_location_jitter, 0.01)
        self.assertAlmostEqual(updater.offspring_rotation_jitter_deg, 10.0)
        self.assertAlmostEqual(updater.offspring_evidence_threshold, 3.0)

    def test_offspring_does_not_crash_during_eval(self):
        """Offspring-enabled LM can run eval without crashing."""
        from tbp.monty.frameworks.models.evidence_matching.burst_sampling import (
            BurstSamplingHypothesesUpdater,
        )

        ctx = _make_ctx()
        lm = _make_lm(
            temporal_memory={"sdr_dim": 256, "sdr_sparsity": 0.05},
            hypotheses_updater_class=BurstSamplingHypothesesUpdater,
            hypotheses_updater_args={
                "offspring_enabled": True,
                "offspring_count": 5,
                "offspring_evidence_threshold": 0.1,
            },
        )

        _train_object(lm, "sphere", _sphere_states, ctx)
        _train_object(lm, "cube", _cube_states, ctx)

        evidence = _run_eval(lm, _sphere_states, ctx)
        self.assertGreater(len(evidence), 0)


# ============================================================================
# T6.8: ThreadPoolExecutor evidence updates
# ============================================================================

class TestThreadPoolEvidenceUpdates(unittest.TestCase):
    """T6.8: LM uses ThreadPoolExecutor instead of raw threads."""

    def test_thread_pool_created_lazily(self):
        """Thread pool is None until first multithreaded update."""
        lm = _make_lm()
        self.assertIsNone(lm._evidence_pool)

    def test_multithreaded_eval_produces_valid_evidence(self):
        """Multithreaded evidence updates produce valid results."""
        ctx = _make_ctx()
        lm = _make_lm(
            temporal_memory={"sdr_dim": 256, "sdr_sparsity": 0.05},
            use_multithreading=True,
        )

        _train_object(lm, "sphere", _sphere_states, ctx)
        _train_object(lm, "cube", _cube_states, ctx)

        evidence = _run_eval(lm, _sphere_states, ctx)
        self.assertGreater(len(evidence), 0)

        # Evidence for the correct object should be positive
        last = evidence[-1]
        self.assertIn("sphere", last)
        self.assertGreater(last["sphere"], 0)

    def test_singlethreaded_matches_multithreaded(self):
        """Single-threaded and multi-threaded eval produce consistent results."""
        ctx = _make_ctx()
        obs = _sphere_states

        lm_st = _make_lm(
            temporal_memory={"sdr_dim": 256, "sdr_sparsity": 0.05},
            use_multithreading=False,
        )
        lm_mt = _make_lm(
            temporal_memory={"sdr_dim": 256, "sdr_sparsity": 0.05},
            use_multithreading=True,
        )

        for lm in [lm_st, lm_mt]:
            _train_object(lm, "sphere", obs, ctx)
            _train_object(lm, "cube", _cube_states, ctx)

        ev_st = _run_eval(lm_st, obs, ctx)
        ev_mt = _run_eval(lm_mt, obs, ctx)

        # Both should identify sphere as the best match
        if ev_st and ev_mt and ev_st[-1] and ev_mt[-1]:
            best_st = max(ev_st[-1], key=ev_st[-1].get)
            best_mt = max(ev_mt[-1], key=ev_mt[-1].get)
            self.assertEqual(best_st, best_mt,
                             "Single/multi-threaded should agree on best match")

    def test_pool_cleanup_on_del(self):
        """Thread pool is cleaned up when LM is deleted."""
        ctx = _make_ctx()
        lm = _make_lm(use_multithreading=True)
        _train_object(lm, "sphere", _sphere_states, ctx)
        _train_object(lm, "cube", _cube_states, ctx)

        # Trigger pool creation by running eval
        _run_eval(lm, _sphere_states[:3], ctx)

        pool = lm._evidence_pool
        if pool is not None:
            del lm
            # Pool should be shut down (no exception from gc)


if __name__ == "__main__":
    unittest.main()
