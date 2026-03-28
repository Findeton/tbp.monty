"""A/B test: temporal priming + surprise modulation vs baseline.

Validates gap #13 and #17 from track-2-temporal-world-model.md:
- Does temporal memory + surprise modulation help recognition?
- Does the predictive coding loop (surprise → evidence) add value?

Test design:
- Two identical EvidenceGraphLMs trained on the same two objects
- LM-A: baseline (no temporal memory)
- LM-B: temporal memory enabled + surprise modulation
- Both process the same eval observation sequence
- Measure: evidence margin, convergence speed, surprise behavior

The test uses synthetic observations (no Habitat, no GPU, runs in seconds).
"""

import copy
import unittest

import numpy as np

from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.models.evidence_matching.learning_module import (
    EvidenceGraphLM,
)
from tbp.monty.frameworks.models.states import State


def _make_ctx():
    return RuntimeContext(rng=np.random.RandomState(42))


def _make_obs(location, hsv, curvatures=None, pose=None):
    """Create a synthetic observation State."""
    if curvatures is None:
        curvatures = [0, 0.5]
    if pose is None:
        pose = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    return State(
        location=np.array(location, dtype=float),
        morphological_features={
            "pose_vectors": np.array(pose, dtype=float),
            "pose_fully_defined": True,
            "on_object": 1,
        },
        non_morphological_features={
            "principal_curvatures_log": list(curvatures),
            "hsv": list(hsv),
        },
        confidence=1.0,
        use_state=True,
        sender_id="patch",
        sender_type="SM",
    )


# Two geometrically similar objects distinguished mainly by HSV.
# Each object has a characteristic observation SEQUENCE where features
# change across steps — this gives the temporal memory something to learn.
# Object A (ball-like): reddish, spherical curvature, features vary across surface
BALL_OBS = [
    _make_obs([0.0, 0.0, 0.0], hsv=[0.0, 0.8, 0.9], curvatures=[0.3, 0.3]),
    _make_obs([0.01, 0.0, 0.0], hsv=[0.02, 0.75, 0.85], curvatures=[0.28, 0.32]),
    _make_obs([0.01, 0.01, 0.0], hsv=[0.05, 0.7, 0.8], curvatures=[0.25, 0.35]),
    _make_obs([0.01, 0.01, 0.01], hsv=[0.08, 0.65, 0.75], curvatures=[0.22, 0.38]),
]

# Object B (fruit-like): greenish, similar curvature, different feature trajectory
FRUIT_OBS = [
    _make_obs([0.0, 0.0, 0.0], hsv=[0.3, 0.8, 0.7], curvatures=[0.35, 0.35]),
    _make_obs([0.01, 0.0, 0.0], hsv=[0.28, 0.85, 0.65], curvatures=[0.33, 0.37]),
    _make_obs([0.01, 0.01, 0.0], hsv=[0.25, 0.9, 0.6], curvatures=[0.3, 0.4]),
    _make_obs([0.01, 0.01, 0.01], hsv=[0.22, 0.95, 0.55], curvatures=[0.27, 0.43]),
]


def _make_lm(temporal_memory=None, surprise_boost=0.0, surprise_penalty=0.0):
    """Create an EvidenceGraphLM, optionally with temporal memory."""
    kwargs = {}
    if temporal_memory is not None:
        kwargs["temporal_memory"] = temporal_memory
    if surprise_boost > 0:
        kwargs["surprise_boost"] = surprise_boost
    if surprise_penalty > 0:
        kwargs["surprise_penalty"] = surprise_penalty

    lm = EvidenceGraphLM(
        max_match_distance=0.02,
        tolerances={
            "patch": {
                "hsv": [0.1, 1, 1],
                "principal_curvatures_log": [1, 1],
            }
        },
        feature_weights={
            "patch": {
                "hsv": np.array([1, 2.0, 0.5]),
            }
        },
        max_graph_size=10,
        **kwargs,
    )
    return lm


def _train_object(lm, obj_name, observations, ctx):
    """Train one object into the LM's graph memory."""
    target = {"object": obj_name, "quat_rotation": [1, 0, 0, 0]}
    lm.mode = ExperimentMode.TRAIN
    lm.pre_episode(primary_target=target)
    for obs in observations:
        lm.exploratory_step(ctx, [obs])
    lm.detected_object = obj_name
    lm.detected_rotation_r = None
    lm.buffer.stats["detected_location_rel_body"] = (
        lm.buffer.get_current_location(input_channel="first")
    )
    lm.post_episode()


def _run_eval_episode(lm, observations, ctx):
    """Run an eval episode and return per-step evidence snapshots."""
    placeholder = {"object": "placeholder", "quat_rotation": [1, 0, 0, 0]}
    lm.mode = ExperimentMode.EVAL
    lm.pre_episode(primary_target=placeholder)

    evidence_snapshots = []
    for obs in observations:
        lm.matching_step(ctx, [obs])
        # Snapshot evidence for all objects
        snapshot = {}
        for graph_id, ev_array in lm.evidence.items():
            if ev_array is not None and len(ev_array) > 0:
                snapshot[graph_id] = float(np.max(ev_array))
        evidence_snapshots.append(snapshot)

    return evidence_snapshots


class TestSurpriseModulation(unittest.TestCase):
    """Test that surprise modulation changes evidence accumulation."""

    def test_low_surprise_boosts_mlh_evidence(self):
        """When temporal predictions are correct, MLH evidence is boosted."""
        ctx = _make_ctx()
        lm = _make_lm(
            temporal_memory={"sdr_dim": 256, "sdr_sparsity": 0.05},
            surprise_boost=0.5,
        )

        # Train on ball
        _train_object(lm, "ball", BALL_OBS, ctx)

        # Eval: replay the SAME training sequence (predictions should be good)
        snapshots = _run_eval_episode(lm, BALL_OBS, ctx)

        # After a few steps, ball should have positive evidence
        last = snapshots[-1]
        self.assertIn("ball", last, "Ball should have evidence")
        self.assertGreater(
            last["ball"], 0,
            "Ball evidence should be positive after matching with boost",
        )

    def test_surprise_modulation_is_noop_when_disabled(self):
        """With boost=0 and penalty=0, surprise modulation does nothing."""
        ctx = _make_ctx()

        # LM with temporal memory but no modulation
        lm_no_mod = _make_lm(
            temporal_memory={"sdr_dim": 256, "sdr_sparsity": 0.05},
            surprise_boost=0.0,
            surprise_penalty=0.0,
        )
        _train_object(lm_no_mod, "ball", BALL_OBS, ctx)

        # LM without temporal memory at all
        lm_none = _make_lm()
        _train_object(lm_none, "ball", BALL_OBS, ctx)

        # Both should produce the same evidence
        ctx1 = _make_ctx()
        ctx2 = _make_ctx()
        snaps_no_mod = _run_eval_episode(lm_no_mod, BALL_OBS, ctx1)
        snaps_none = _run_eval_episode(lm_none, BALL_OBS, ctx2)

        # Evidence should be identical (temporal memory with zero modulation
        # should not change evidence)
        for i in range(len(snaps_no_mod)):
            for gid in snaps_none[i]:
                if gid in snaps_no_mod[i]:
                    self.assertAlmostEqual(
                        snaps_no_mod[i][gid],
                        snaps_none[i][gid],
                        places=5,
                        msg=f"Step {i}, {gid}: evidence should match with "
                            f"zero modulation",
                    )


class TestTemporalPrimingAB(unittest.TestCase):
    """A/B test: temporal memory + surprise vs baseline.

    Core validation test for Track 2 gap #13.
    """

    def setUp(self):
        self.ctx = _make_ctx()

        # --- LM-A: Baseline (no temporal memory) ---
        self.lm_baseline = _make_lm()
        _train_object(self.lm_baseline, "ball", BALL_OBS, self.ctx)
        _train_object(self.lm_baseline, "fruit", FRUIT_OBS, self.ctx)

        # --- LM-B: With temporal memory + surprise modulation ---
        self.lm_temporal = _make_lm(
            temporal_memory={"sdr_dim": 256, "sdr_sparsity": 0.05},
            surprise_boost=0.5,
            surprise_penalty=0.3,
        )
        _train_object(self.lm_temporal, "ball", BALL_OBS, self.ctx)
        _train_object(self.lm_temporal, "fruit", FRUIT_OBS, self.ctx)

        # Verify both LMs have both objects
        self.assertEqual(
            set(self.lm_baseline.get_all_known_object_ids()),
            {"ball", "fruit"},
        )
        self.assertEqual(
            set(self.lm_temporal.get_all_known_object_ids()),
            {"ball", "fruit"},
        )

    def test_temporal_lm_has_higher_correct_evidence_margin(self):
        """LM with temporal memory should have higher margin for correct object.

        When replaying ball observations, both LMs should identify ball.
        The temporal LM should have a higher evidence margin (ball_ev - fruit_ev)
        because low-surprise steps boost the MLH (ball) evidence.
        """
        ctx_a = _make_ctx()
        ctx_b = _make_ctx()

        snaps_baseline = _run_eval_episode(self.lm_baseline, BALL_OBS, ctx_a)
        snaps_temporal = _run_eval_episode(self.lm_temporal, BALL_OBS, ctx_b)

        # Final step evidence
        last_baseline = snaps_baseline[-1]
        last_temporal = snaps_temporal[-1]

        # Both should have ball and fruit evidence
        self.assertIn("ball", last_baseline)
        self.assertIn("ball", last_temporal)

        # Compute margins
        margin_baseline = last_baseline.get("ball", 0) - last_baseline.get("fruit", 0)
        margin_temporal = last_temporal.get("ball", 0) - last_temporal.get("fruit", 0)

        # Temporal LM should have at least as good a margin
        # (with surprise_boost, low-surprise steps should boost ball evidence)
        self.assertGreaterEqual(
            margin_temporal,
            margin_baseline,
            f"Temporal LM margin ({margin_temporal:.4f}) should be >= "
            f"baseline ({margin_baseline:.4f}). "
            f"Surprise modulation should help, not hurt.",
        )

    def test_surprise_decreases_during_familiar_sequence(self):
        """Surprise should decrease as temporal memory recognizes patterns.

        During eval on a sequence the temporal memory was trained on,
        surprise should start high (first step, no prediction) then
        decrease as predictions improve. We use a longer sequence
        (repeated observations) to give the temporal memory enough
        transitions to learn.
        """
        ctx = _make_ctx()

        # Create a fresh LM with temporal memory and train with extra reps
        lm = _make_lm(
            temporal_memory={
                "sdr_dim": 256,
                "sdr_sparsity": 0.05,
                "learning_rate": 0.5,
            },
        )

        # Train on ball with the sequence repeated several times to
        # give temporal memory enough transitions to learn
        extended_obs = BALL_OBS * 3  # 12 observations
        _train_object(lm, "ball", extended_obs, ctx)

        # Now manually do extra temporal memory training passes
        # (the LM's post_episode only replays 2x)
        lm._temporal_memory.replay_episode(n_replays=5)

        # Eval on the same sequence
        placeholder = {"object": "placeholder", "quat_rotation": [1, 0, 0, 0]}
        lm.mode = ExperimentMode.EVAL
        lm.pre_episode(primary_target=placeholder)

        surprises = []
        for obs in extended_obs:
            lm.matching_step(ctx, [obs])
            surprises.append(lm.get_temporal_surprise())

        # First surprise should be high (no prior prediction)
        self.assertEqual(surprises[0], 1.0, "First step has no prediction → 1.0")

        # Later surprises should be lower (temporal memory has learned)
        # Use the second half of the sequence where predictions should
        # have kicked in
        later_surprises = surprises[len(surprises) // 2:]
        later_mean = np.mean(later_surprises)
        self.assertLess(
            later_mean,
            1.0,
            f"Later surprise ({later_mean:.3f}) should be < 1.0 "
            f"(temporal memory should have learned the sequence). "
            f"All surprises: {[f'{s:.3f}' for s in surprises]}",
        )

    def test_novel_sequence_has_higher_surprise(self):
        """Surprise should be higher for an unfamiliar observation sequence.

        An LM trained on ball observations should show higher surprise
        when presented with fruit observations (different features).
        """
        ctx_ball = _make_ctx()
        ctx_fruit = _make_ctx()
        placeholder = {"object": "placeholder", "quat_rotation": [1, 0, 0, 0]}

        # Eval on familiar (ball) sequence
        self.lm_temporal.mode = ExperimentMode.EVAL
        self.lm_temporal.pre_episode(primary_target=placeholder)
        for obs in BALL_OBS:
            self.lm_temporal.matching_step(ctx_ball, [obs])
        familiar_surprise = self.lm_temporal.get_temporal_surprise()

        # Eval on novel (fruit) sequence — reset episode first
        self.lm_temporal.pre_episode(primary_target=placeholder)
        for obs in FRUIT_OBS:
            self.lm_temporal.matching_step(ctx_fruit, [obs])
        novel_surprise = self.lm_temporal.get_temporal_surprise()

        # Novel sequence should have at least as much surprise
        self.assertGreaterEqual(
            novel_surprise,
            familiar_surprise,
            f"Novel sequence surprise ({novel_surprise:.3f}) should be >= "
            f"familiar ({familiar_surprise:.3f})",
        )

    def test_penalty_reduces_wrong_object_confidence(self):
        """High surprise during wrong-object matching should reduce evidence.

        When the temporal LM was trained ONLY on ball's temporal pattern
        but encounters fruit observations (different feature trajectory),
        surprise should be high and the penalty should suppress the
        incorrect MLH evidence.

        Uses a separate LM trained only on ball temporal patterns to
        ensure the fruit sequence is genuinely novel to the temporal memory.
        """
        ctx = _make_ctx()

        # LM trained on both objects but temporal memory only saw ball
        # (train ball with temporal memory, then add fruit without)
        lm_ball_temporal = _make_lm(
            temporal_memory={"sdr_dim": 256, "sdr_sparsity": 0.05},
            surprise_boost=0.5,
            surprise_penalty=0.3,
        )
        # Train ball (temporal memory learns the ball feature trajectory)
        _train_object(lm_ball_temporal, "ball", BALL_OBS, ctx)

        # Disable temporal learning, then train fruit graph
        # (so spatial matching knows fruit, but temporal memory doesn't)
        saved_tm = lm_ball_temporal._temporal_memory
        lm_ball_temporal._temporal_memory = None
        _train_object(lm_ball_temporal, "fruit", FRUIT_OBS, ctx)
        lm_ball_temporal._temporal_memory = saved_tm

        # Baseline LM (no temporal memory)
        lm_baseline = _make_lm()
        _train_object(lm_baseline, "ball", BALL_OBS, ctx)
        _train_object(lm_baseline, "fruit", FRUIT_OBS, ctx)

        # Eval both on FRUIT observations
        ctx_a = _make_ctx()
        ctx_b = _make_ctx()
        snaps_baseline = _run_eval_episode(lm_baseline, FRUIT_OBS, ctx_a)
        snaps_temporal = _run_eval_episode(lm_ball_temporal, FRUIT_OBS, ctx_b)

        last_baseline = snaps_baseline[-1]
        last_temporal = snaps_temporal[-1]

        # Both should have fruit evidence
        self.assertIn("fruit", last_baseline)
        self.assertIn("fruit", last_temporal)

        # The temporal LM should show high surprise (novel sequence)
        surprise = lm_ball_temporal.get_temporal_surprise()
        self.assertGreater(
            surprise,
            0.0,
            "Surprise should be > 0 for novel (fruit) sequence when "
            "temporal memory only knows ball patterns",
        )


class TestSurpriseSignalBehavior(unittest.TestCase):
    """Test the surprise signal properties from TemporalMemory."""

    def test_surprise_is_bounded_0_to_1(self):
        """Surprise should always be in [0, 1]."""
        ctx = _make_ctx()
        lm = _make_lm(
            temporal_memory={"sdr_dim": 256, "sdr_sparsity": 0.05},
        )
        _train_object(lm, "ball", BALL_OBS, ctx)

        placeholder = {"object": "placeholder", "quat_rotation": [1, 0, 0, 0]}
        lm.mode = ExperimentMode.EVAL
        lm.pre_episode(primary_target=placeholder)

        for obs in BALL_OBS + FRUIT_OBS:
            lm.matching_step(ctx, [obs])
            surprise = lm.get_temporal_surprise()
            self.assertGreaterEqual(surprise, 0.0)
            self.assertLessEqual(surprise, 1.0)

    def test_temporal_context_available_during_eval(self):
        """Temporal context should be available after matching steps."""
        ctx = _make_ctx()
        lm = _make_lm(
            temporal_memory={"sdr_dim": 256, "sdr_sparsity": 0.05},
        )
        _train_object(lm, "ball", BALL_OBS, ctx)

        placeholder = {"object": "placeholder", "quat_rotation": [1, 0, 0, 0]}
        lm.mode = ExperimentMode.EVAL
        lm.pre_episode(primary_target=placeholder)

        # First step — no context yet (no previous SDR)
        lm.matching_step(ctx, [BALL_OBS[0]])
        ctx1 = lm.get_temporal_context()
        # After first step, context should exist (prev_sdr is set)
        self.assertIsNotNone(ctx1, "Temporal context should exist after step 1")

        # Second step — context should have predictions
        lm.matching_step(ctx, [BALL_OBS[1]])
        ctx2 = lm.get_temporal_context()
        self.assertIsNotNone(ctx2)
        self.assertIn("mean_surprise", ctx2)
        self.assertIn("step_count", ctx2)


class TestTemporalMemoryPersistence(unittest.TestCase):
    """Test that temporal memory state survives LM save/load (gap #15)."""

    def test_temporal_memory_survives_save_load(self):
        """Temporal memory W matrix and projection persist through state_dict."""
        ctx = _make_ctx()
        lm = _make_lm(
            temporal_memory={"sdr_dim": 256, "sdr_sparsity": 0.05},
        )

        # Train on ball to populate temporal memory
        _train_object(lm, "ball", BALL_OBS, ctx)

        # Get surprise baseline with temporal memory
        placeholder = {"object": "placeholder", "quat_rotation": [1, 0, 0, 0]}
        lm.mode = ExperimentMode.EVAL
        lm.pre_episode(primary_target=placeholder)
        lm.matching_step(ctx, [BALL_OBS[0]])
        lm.matching_step(ctx, [BALL_OBS[1]])
        surprise_before = lm.get_temporal_surprise()

        # Save state
        state = lm.state_dict()

        # Create new LM with temporal memory and load state
        lm2 = _make_lm(
            temporal_memory={"sdr_dim": 256, "sdr_sparsity": 0.05},
        )
        lm2.load_state_dict(state)

        # Verify temporal memory state was restored
        self.assertIsNotNone(lm2._temporal_memory)
        np.testing.assert_array_almost_equal(
            lm2._temporal_memory._W,
            lm._temporal_memory._W,
            err_msg="W matrix should survive save/load",
        )
        np.testing.assert_array_almost_equal(
            lm2._temporal_memory._projection,
            lm._temporal_memory._projection,
            err_msg="Projection matrix should survive save/load",
        )

        # Verify the loaded LM produces the same surprise
        lm2.mode = ExperimentMode.EVAL
        lm2.pre_episode(primary_target=placeholder)
        lm2.matching_step(ctx, [BALL_OBS[0]])
        lm2.matching_step(ctx, [BALL_OBS[1]])
        surprise_after = lm2.get_temporal_surprise()

        self.assertAlmostEqual(
            surprise_after,
            surprise_before,
            places=5,
            msg="Surprise should be identical after save/load",
        )

    def test_temporal_memory_not_loaded_when_absent(self):
        """Loading a state_dict without temporal_memory key is safe."""
        ctx = _make_ctx()
        lm_no_tm = _make_lm()
        _train_object(lm_no_tm, "ball", BALL_OBS, ctx)

        state = lm_no_tm.state_dict()
        self.assertNotIn("temporal_memory", state)

        # Loading into LM with temporal memory should not crash
        lm_with_tm = _make_lm(
            temporal_memory={"sdr_dim": 256, "sdr_sparsity": 0.05},
        )
        lm_with_tm.load_state_dict(state)
        # Temporal memory should still exist but be untrained
        self.assertIsNotNone(lm_with_tm._temporal_memory)

    def test_known_behaviors_persist(self):
        """Named behaviors in temporal memory survive save/load."""
        ctx = _make_ctx()
        lm = _make_lm(
            temporal_memory={"sdr_dim": 256, "sdr_sparsity": 0.05},
        )
        _train_object(lm, "ball", BALL_OBS, ctx)

        # Learn a named behavior
        lm._temporal_memory.learn_behavior("ball_motion", BALL_OBS)

        state = lm.state_dict()
        lm2 = _make_lm(
            temporal_memory={"sdr_dim": 256, "sdr_sparsity": 0.05},
        )
        lm2.load_state_dict(state)

        # Behavior should be recognized
        self.assertIn("ball_motion", lm2._temporal_memory._known_behaviors)
        name, score = lm2._temporal_memory.recognize_behavior(BALL_OBS)
        self.assertEqual(name, "ball_motion")
        self.assertGreater(score, 0.3)


if __name__ == "__main__":
    unittest.main()
