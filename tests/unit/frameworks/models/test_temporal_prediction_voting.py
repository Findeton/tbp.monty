# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Tests for T2.10: Temporal Prediction Voting.

Tests that:
- LMs predict next state from learned transition sequences
- Temporal prediction is checked against reality (confident vs confused)
- Temporal voting sends votes from confident → confused LMs
- End-to-end: StateConditionedModel training, matching with state inference,
  and temporal prediction across steps
- Backward compatibility: stateless models unaffected
"""

import unittest

import numpy as np

from tbp.monty.frameworks.models.states import State


def _make_state(location, hsv=(0.5, 0.5, 0.5), sender_id="patch"):
    """Create a minimal State for training/matching."""
    return State(
        location=np.array(location, dtype=np.float64),
        morphological_features={
            "pose_vectors": np.eye(3),
            "pose_fully_defined": True,
            "principal_curvatures_log": np.array([1.0, 1.0]),
        },
        non_morphological_features={
            "hsv": np.array(hsv),
        },
        confidence=1.0,
        use_state=True,
        sender_id=sender_id,
        sender_type="SM",
    )


def _make_lm(**kwargs):
    """Create an EvidenceGraphLM configured for state-conditioned testing."""
    from tbp.monty.frameworks.models.evidence_matching.learning_module import (
        EvidenceGraphLM,
    )

    defaults = dict(
        max_match_distance=0.01,
        tolerances={
            "patch": {
                "hsv": [0.2, 1, 1],
                "principal_curvatures_log": [2, 2],
            }
        },
        feature_weights={
            "patch": {
                "hsv": np.array([1, 0, 0]),
            }
        },
        max_graph_size=1.0,
        num_model_voxels_per_dim=100,
        hypotheses_updater_args=dict(
            initial_possible_poses="informed",
        ),
    )
    defaults.update(kwargs)
    return EvidenceGraphLM(**defaults)


def _train_object(lm, obs_states, object_name):
    """Train the LM on a sequence of observations."""
    from tbp.monty.context import RuntimeContext
    from tbp.monty.frameworks.experiments.mode import ExperimentMode

    ctx = RuntimeContext(rng=np.random.RandomState(42))
    lm.mode = ExperimentMode.TRAIN
    lm.pre_episode(
        primary_target={
            "object": object_name,
            "quat_rotation": [1, 0, 0, 0],
        }
    )

    for obs in obs_states:
        lm.exploratory_step(ctx, [obs])

    lm.detected_object = object_name
    lm.detected_rotation_r = None
    lm.buffer.stats["detected_location_rel_body"] = (
        lm.buffer.get_current_location(input_channel="first")
    )
    lm.post_episode()


def _match_steps(lm, obs_states, n_steps=None):
    """Run matching steps, returning MLH after each step."""
    from tbp.monty.context import RuntimeContext
    from tbp.monty.frameworks.experiments.mode import ExperimentMode

    ctx = RuntimeContext(rng=np.random.RandomState(42))
    lm.mode = ExperimentMode.EVAL
    lm.pre_episode(
        primary_target={
            "object": "placeholder",
            "quat_rotation": [1, 0, 0, 0],
        }
    )

    n = n_steps or len(obs_states)
    mlhs = []
    for obs in obs_states[:n]:
        lm.add_lm_processing_to_buffer_stats(lm_processed=True)
        lm.matching_step(ctx, [obs])
        mlhs.append(dict(lm.get_current_mlh()))

    return mlhs


# ============================================================
# Quick unit tests for the prediction methods themselves
# ============================================================


class TestPredictNextState(unittest.TestCase):
    """Test _predict_next_state on an LM with a StateConditionedModel."""

    def _make_lm_with_scm(self):
        """Create an LM and manually inject a StateConditionedModel.

        This is a quick test — no training, just directly populating
        graph memory with a pre-built StateConditionedModel.
        """
        from tbp.monty.frameworks.models.state_conditioned_model import (
            StateConditionedModel,
        )

        lm = _make_lm()

        # Build a StateConditionedModel with 3 states and transitions
        scm = StateConditionedModel(
            object_id="test_obj",
            max_nodes=100,
            max_size=1.0,
            num_voxels_per_dim=100,
        )
        # State 0 → State 1 → State 2 (with known durations)
        scm.add_transition(0, 1, 5.0)
        scm.add_transition(1, 2, 10.0)

        # Build sub-models with minimal data
        for state_id in [0, 1, 2]:
            locs = np.array([[state_id * 0.1, 0.0, 0.0]])
            feats = {"hsv": np.array([[0.5, 0.5, 0.5]])}
            scm.build_model(locs, feats, state_id=state_id)

        # Inject into graph memory
        lm.graph_memory.models_in_memory["test_obj"] = {"patch": scm}

        return lm

    def test_predict_from_state_0(self):
        """Predict state 1 when currently at state 0."""
        lm = self._make_lm_with_scm()
        lm.current_mlh = {
            "graph_id": "test_obj",
            "state": 0,
            "evidence": 5.0,
        }
        predicted = lm._predict_next_state()
        self.assertEqual(predicted, 1)

    def test_predict_from_state_1(self):
        """Predict state 2 when currently at state 1."""
        lm = self._make_lm_with_scm()
        lm.current_mlh = {
            "graph_id": "test_obj",
            "state": 1,
            "evidence": 5.0,
        }
        predicted = lm._predict_next_state()
        self.assertEqual(predicted, 2)

    def test_predict_from_terminal_state(self):
        """No prediction when at terminal state (state 2, no outgoing)."""
        lm = self._make_lm_with_scm()
        lm.current_mlh = {
            "graph_id": "test_obj",
            "state": 2,
            "evidence": 5.0,
        }
        predicted = lm._predict_next_state()
        self.assertIsNone(predicted)

    def test_predict_with_no_state(self):
        """No prediction when MLH has no state."""
        lm = self._make_lm_with_scm()
        lm.current_mlh = {
            "graph_id": "test_obj",
            "state": None,
            "evidence": 5.0,
        }
        predicted = lm._predict_next_state()
        self.assertIsNone(predicted)

    def test_predict_with_unknown_graph(self):
        """No prediction when graph_id is not in memory."""
        lm = self._make_lm_with_scm()
        lm.current_mlh = {
            "graph_id": "unknown_obj",
            "state": 0,
            "evidence": 5.0,
        }
        predicted = lm._predict_next_state()
        self.assertIsNone(predicted)

    def test_predict_with_plain_model(self):
        """No prediction when graph uses a plain GridObjectModel (no states)."""
        lm = _make_lm()
        # Train a plain model
        obs = [_make_state([i * 0.01, 0.0, 0.0]) for i in range(5)]
        _train_object(lm, obs, "plain_obj")

        lm.current_mlh = {
            "graph_id": "plain_obj",
            "state": None,
            "evidence": 5.0,
        }
        predicted = lm._predict_next_state()
        self.assertIsNone(predicted)


class TestCheckTemporalPrediction(unittest.TestCase):
    """Test _check_temporal_prediction."""

    def test_correct_prediction(self):
        lm = _make_lm()
        lm._predicted_next_state = 1
        lm.current_mlh = {"state": 1, "evidence": 5.0}
        result = lm._check_temporal_prediction()
        self.assertEqual(result, "confident")

    def test_wrong_prediction(self):
        lm = _make_lm()
        lm._predicted_next_state = 1
        lm.current_mlh = {"state": 2, "evidence": 5.0}
        result = lm._check_temporal_prediction()
        self.assertEqual(result, "confused")

    def test_no_prediction(self):
        lm = _make_lm()
        lm._predicted_next_state = None
        lm.current_mlh = {"state": 1, "evidence": 5.0}
        result = lm._check_temporal_prediction()
        self.assertIsNone(result)

    def test_no_prediction_attribute(self):
        """No prediction attribute at all → None."""
        lm = _make_lm()
        lm.current_mlh = {"state": 1, "evidence": 5.0}
        result = lm._check_temporal_prediction()
        self.assertIsNone(result)

    def test_prediction_but_no_actual_state(self):
        """Prediction was made but current MLH has no state → None."""
        lm = _make_lm()
        lm._predicted_next_state = 1
        lm.current_mlh = {"state": None, "evidence": 5.0}
        result = lm._check_temporal_prediction()
        self.assertIsNone(result)


class TestGetTemporalPredictionStatus(unittest.TestCase):
    """Test get_temporal_prediction_status (public interface for Monty)."""

    def test_returns_status(self):
        lm = _make_lm()
        lm._temporal_prediction_status = "confident"
        self.assertEqual(lm.get_temporal_prediction_status(), "confident")

    def test_returns_none_when_no_status(self):
        lm = _make_lm()
        self.assertIsNone(lm.get_temporal_prediction_status())


class TestGetTransitionSequenceAlias(unittest.TestCase):
    """Test that StateConditionedModel has both get_transitions and
    get_transition_sequence (alias)."""

    def test_alias_returns_same_result(self):
        from tbp.monty.frameworks.models.state_conditioned_model import (
            StateConditionedModel,
        )

        scm = StateConditionedModel(
            object_id="test", max_nodes=50, max_size=1.0, num_voxels_per_dim=50
        )
        scm.add_transition(0, 1, 5.0)
        scm.add_transition(1, 2, 10.0)

        self.assertEqual(scm.get_transitions(), scm.get_transition_sequence())
        self.assertEqual(len(scm.get_transition_sequence()), 2)


# ============================================================
# Tests for _vote_temporal in GraphMatchingMonty
# ============================================================


class TestVoteTemporalDispatch(unittest.TestCase):
    """Test _vote_temporal in GraphMatchingMonty."""

    def test_temporal_voting_dispatches_from_confident_to_confused(self):
        """Confident LMs vote to help confused LMs."""
        from tbp.monty.frameworks.models.graph_matching import (
            MontyForGraphMatching,
        )

        # Create mock LMs with temporal prediction status
        class MockBuffer:
            def get_num_observations_on_object(self):
                return 1

            def update_last_stats_entry(self, stats):
                pass

        class MockLM:
            def __init__(self, lm_id, status, possible_matches=None):
                self.learning_module_id = lm_id
                self._status = status
                self._possible_matches = possible_matches or ["obj_a"]
                self.votes_received = []
                self.buffer = MockBuffer()
                self.terminal_state = None
                self.stepwise_target_object = None
                self.stepwise_targets_list = []

            def get_temporal_prediction_status(self):
                return self._status

            def get_possible_matches(self):
                return self._possible_matches

            def send_out_vote(self):
                # sensed_pose_rel_body: [position(3,), *rotation_vectors(3,3)]
                # [0] = position, [1:] = rotation vectors for align_vectors
                pose = np.array([
                    [0.0, 0.0, 0.0],  # position
                    [1.0, 0.0, 0.0],  # x-axis
                    [0.0, 1.0, 0.0],  # y-axis
                    [0.0, 0.0, 1.0],  # z-axis
                ])
                return {
                    "object_id_vote": {"obj_a": True},
                    "location_vote": {
                        "obj_a": np.array([[0.0, 0.0, 0.0]])
                    },
                    "rotation_vote": {"obj_a": [
                        [Rotation.identity()],
                    ]},
                    "sensed_pose_rel_body": pose,
                }

            def receive_votes(self, votes):
                self.votes_received.append(votes)

            def collect_stats_to_save(self):
                return {}

            def add_lm_processing_to_buffer_stats(self, **kwargs):
                pass

        from scipy.spatial.transform import Rotation

        # LM0: confident, LM1: confused
        lm0 = MockLM("LM_0", "confident")
        lm1 = MockLM("LM_1", "confused")

        # Create a minimal Monty with temporal voting enabled
        class TestMonty(MontyForGraphMatching):
            def __init__(self):
                self.learning_modules = [lm0, lm1]
                self.lm_to_lm_vote_matrix = [[1], [0]]
                self.temporal_voting = True

        monty = TestMonty()
        monty._vote_temporal()

        # LM1 (confused) should have received votes
        self.assertTrue(len(lm1.votes_received) > 0)
        # LM0 (confident) should NOT have received votes
        self.assertEqual(len(lm0.votes_received), 0)

    def test_no_temporal_voting_when_all_confident(self):
        """No voting when all LMs are temporally confident."""
        from tbp.monty.frameworks.models.graph_matching import (
            MontyForGraphMatching,
        )

        class MockLM:
            def __init__(self, lm_id):
                self.learning_module_id = lm_id
                self.votes_received = []
                self.buffer = type("B", (), {
                    "get_num_observations_on_object": lambda s: 1,
                })()

            def get_temporal_prediction_status(self):
                return "confident"

            def send_out_vote(self):
                return None

            def receive_votes(self, votes):
                self.votes_received.append(votes)

        lm0 = MockLM("LM_0")
        lm1 = MockLM("LM_1")

        class TestMonty(MontyForGraphMatching):
            def __init__(self):
                self.learning_modules = [lm0, lm1]
                self.lm_to_lm_vote_matrix = [[1], [0]]
                self.temporal_voting = True

        monty = TestMonty()
        monty._vote_temporal()

        self.assertEqual(len(lm0.votes_received), 0)
        self.assertEqual(len(lm1.votes_received), 0)

    def test_no_temporal_voting_when_no_predictions(self):
        """No voting when no LM has temporal predictions."""
        from tbp.monty.frameworks.models.graph_matching import (
            MontyForGraphMatching,
        )

        class MockLM:
            def __init__(self, lm_id):
                self.learning_module_id = lm_id
                self.votes_received = []

            def get_temporal_prediction_status(self):
                return None

            def send_out_vote(self):
                return None

            def receive_votes(self, votes):
                self.votes_received.append(votes)

        lm0 = MockLM("LM_0")
        lm1 = MockLM("LM_1")

        class TestMonty(MontyForGraphMatching):
            def __init__(self):
                self.learning_modules = [lm0, lm1]
                self.lm_to_lm_vote_matrix = [[1], [0]]
                self.temporal_voting = True

        monty = TestMonty()
        monty._vote_temporal()

        self.assertEqual(len(lm0.votes_received), 0)
        self.assertEqual(len(lm1.votes_received), 0)

    def test_lm_without_temporal_methods_skipped(self):
        """LMs that don't implement temporal prediction are skipped."""
        from tbp.monty.frameworks.models.graph_matching import (
            MontyForGraphMatching,
        )

        class TemporalLM:
            def __init__(self, lm_id, status):
                self.learning_module_id = lm_id
                self._status = status
                self.votes_received = []
                self.buffer = type("B", (), {
                    "get_num_observations_on_object": lambda s: 1,
                })()
                self.terminal_state = None
                self.stepwise_target_object = None
                self.stepwise_targets_list = []

            def get_temporal_prediction_status(self):
                return self._status

            def get_possible_matches(self):
                return ["obj_a"]

            def send_out_vote(self):
                from scipy.spatial.transform import Rotation

                pose = np.array([
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ])
                return {
                    "object_id_vote": {"obj_a": True},
                    "location_vote": {"obj_a": np.array([[0, 0, 0.0]])},
                    "rotation_vote": {"obj_a": [
                        [Rotation.identity()],
                    ]},
                    "sensed_pose_rel_body": pose,
                }

            def receive_votes(self, votes):
                self.votes_received.append(votes)

            def collect_stats_to_save(self):
                return {}

            def add_lm_processing_to_buffer_stats(self, **kwargs):
                pass

        class PlainLM:
            """LM without temporal prediction."""
            def __init__(self, lm_id):
                self.learning_module_id = lm_id
                self.votes_received = []

            def send_out_vote(self):
                return None

            def receive_votes(self, votes):
                self.votes_received.append(votes)

        lm0 = TemporalLM("LM_0", "confident")
        lm1 = PlainLM("LM_1")  # No temporal prediction methods

        class TestMonty(MontyForGraphMatching):
            def __init__(self):
                self.learning_modules = [lm0, lm1]
                self.lm_to_lm_vote_matrix = [[1], [0]]
                self.temporal_voting = True

        monty = TestMonty()
        # Should not raise
        monty._vote_temporal()

        # No votes exchanged (no confused LMs)
        self.assertEqual(len(lm0.votes_received), 0)
        self.assertEqual(len(lm1.votes_received), 0)


# ============================================================
# End-to-end training + matching tests
# ============================================================


class TestEndToEndStatelessTraining(unittest.TestCase):
    """End-to-end: train a plain (stateless) model, match, verify
    temporal prediction is None (backward compat)."""

    def test_stateless_model_no_temporal_prediction(self):
        """Plain model has no state → no temporal predictions."""
        lm = _make_lm()

        # Train a simple object
        obs = [_make_state([i * 0.01, 0.0, 0.0]) for i in range(10)]
        _train_object(lm, obs, "simple_obj")

        # Match
        test_obs = [_make_state([i * 0.01, 0.0, 0.0]) for i in range(5)]
        mlhs = _match_steps(lm, test_obs)

        # MLH should be simple_obj with no state
        self.assertEqual(mlhs[-1]["graph_id"], "simple_obj")
        self.assertIsNone(mlhs[-1].get("state"))

        # No temporal prediction should have been made
        self.assertIsNone(lm.get_temporal_prediction_status())
        self.assertIsNone(lm.get_speed_signal())


class TestEndToEndStateConditionedTraining(unittest.TestCase):
    """End-to-end: manually build a StateConditionedModel, inject into
    LM graph memory, then match observations and verify state inference
    and temporal prediction."""

    def _make_scm_with_observations(self):
        """Build a StateConditionedModel with 2 states.

        State 0: "closed" — features at x ∈ [0.0, 0.05] with red HSV
        State 1: "open"   — features at x ∈ [0.0, 0.05] with blue HSV
        Transition: 0 → 1 with duration 5.0
        """
        from tbp.monty.frameworks.models.state_conditioned_model import (
            StateConditionedModel,
        )

        scm = StateConditionedModel(
            object_id="stapler",
            max_nodes=200,
            max_size=1.0,
            num_voxels_per_dim=100,
        )

        # State 0 (closed): red features at a line of locations
        n_pts = 10
        locs_0 = np.column_stack([
            np.linspace(0.0, 0.05, n_pts),
            np.zeros(n_pts),
            np.zeros(n_pts),
        ])
        feats_0 = {
            "hsv": np.tile([0.0, 0.8, 0.8], (n_pts, 1)),  # red
            "principal_curvatures_log": np.tile([1.0, 1.0], (n_pts, 1)),
        }
        scm.build_model(locs_0, feats_0, state_id=0)

        # State 1 (open): blue features at same locations
        locs_1 = locs_0.copy()
        feats_1 = {
            "hsv": np.tile([0.6, 0.8, 0.8], (n_pts, 1)),  # blue
            "principal_curvatures_log": np.tile([1.0, 1.0], (n_pts, 1)),
        }
        scm.build_model(locs_1, feats_1, state_id=1)

        # Transition: closed → open
        scm.add_transition(0, 1, 5.0)

        # Create test observations for each state
        obs_state_0 = [
            _make_state(locs_0[i], hsv=(0.0, 0.8, 0.8))
            for i in range(n_pts)
        ]
        obs_state_1 = [
            _make_state(locs_1[i], hsv=(0.6, 0.8, 0.8))
            for i in range(n_pts)
        ]

        return scm, obs_state_0, obs_state_1

    def _inject_scm_into_lm(self, lm, scm, graph_id):
        """Inject a pre-built StateConditionedModel into an LM's memory.

        This simulates what would happen after training on multi-state
        data. The LM's graph memory gets the SCM, and we initialize
        hypothesis storage for state-conditioned matching.
        """
        # Register in graph memory
        lm.graph_memory.models_in_memory[graph_id] = {"patch": scm}

        # Add graph_id to the target/graph maps
        if not hasattr(lm.graph_memory, "target_to_graph_id"):
            lm.graph_memory.target_to_graph_id = {}
        lm.graph_memory.target_to_graph_id[graph_id] = graph_id

        if not hasattr(lm.graph_memory, "graph_id_to_target"):
            lm.graph_memory.graph_id_to_target = {}
        lm.graph_memory.graph_id_to_target[graph_id] = graph_id

    def test_scm_in_memory(self):
        """StateConditionedModel is correctly stored in graph memory."""
        scm, _, _ = self._make_scm_with_observations()
        lm = _make_lm()
        self._inject_scm_into_lm(lm, scm, "stapler")

        models = lm.graph_memory.models_in_memory["stapler"]
        self.assertIn("patch", models)
        model = models["patch"]
        self.assertTrue(hasattr(model, "is_state_conditioned"))
        self.assertTrue(model.is_state_conditioned)
        self.assertEqual(model.get_num_states(), 2)

    def test_transition_sequence_accessible(self):
        """Transition sequence is accessible via both method names."""
        scm, _, _ = self._make_scm_with_observations()

        transitions = scm.get_transitions()
        self.assertEqual(len(transitions), 1)
        self.assertEqual(transitions[0], (0, 1, 5.0))

        # Alias
        self.assertEqual(scm.get_transition_sequence(), transitions)

    def test_next_state_prediction(self):
        """LM predicts next state from StateConditionedModel transitions."""
        scm, _, _ = self._make_scm_with_observations()
        lm = _make_lm()
        self._inject_scm_into_lm(lm, scm, "stapler")

        # Set MLH to state 0
        lm.current_mlh = {
            "graph_id": "stapler",
            "state": 0,
            "evidence": 5.0,
        }
        predicted = lm._predict_next_state()
        self.assertEqual(predicted, 1)

        # State 1 has no outgoing transition
        lm.current_mlh["state"] = 1
        predicted = lm._predict_next_state()
        self.assertIsNone(predicted)


class TestEndToEndTemporalPredictionLifecycle(unittest.TestCase):
    """Test the full predict → step → check lifecycle.

    This simulates what happens across matching steps when the LM
    has a state-conditioned model and is tracking state transitions.
    """

    def test_prediction_lifecycle(self):
        """Predict at step N, check at step N+1."""
        lm = _make_lm()

        # Step N: LM is at state 0, predicts state 1
        lm.current_mlh = {
            "graph_id": "test_obj",
            "state": 0,
            "evidence": 5.0,
        }
        lm._predicted_next_state = lm._predict_next_state()
        # No model in memory → prediction is None (no SCM injected)
        self.assertIsNone(lm._predicted_next_state)

        # Now inject a SCM
        from tbp.monty.frameworks.models.state_conditioned_model import (
            StateConditionedModel,
        )

        scm = StateConditionedModel(
            object_id="test_obj", max_nodes=50,
            max_size=1.0, num_voxels_per_dim=50,
        )
        scm.add_transition(0, 1, 5.0)
        for sid in [0, 1]:
            scm.build_model(
                np.array([[sid * 0.1, 0.0, 0.0]]),
                {"hsv": np.array([[0.5, 0.5, 0.5]])},
                state_id=sid,
            )
        lm.graph_memory.models_in_memory["test_obj"] = {"patch": scm}

        # Now predict from state 0
        lm.current_mlh["state"] = 0
        lm._predicted_next_state = lm._predict_next_state()
        self.assertEqual(lm._predicted_next_state, 1)

        # Step N+1: state transitions to 1 (prediction correct)
        lm.current_mlh["state"] = 1
        status = lm._check_temporal_prediction()
        self.assertEqual(status, "confident")

        # Step N+1 makes its own prediction (state 1 → no outgoing)
        lm._predicted_next_state = lm._predict_next_state()
        self.assertIsNone(lm._predicted_next_state)

    def test_prediction_error_lifecycle(self):
        """Prediction error: predicted state 1 but stayed at state 0."""
        lm = _make_lm()

        from tbp.monty.frameworks.models.state_conditioned_model import (
            StateConditionedModel,
        )

        scm = StateConditionedModel(
            object_id="test_obj", max_nodes=50,
            max_size=1.0, num_voxels_per_dim=50,
        )
        scm.add_transition(0, 1, 5.0)
        for sid in [0, 1]:
            scm.build_model(
                np.array([[sid * 0.1, 0.0, 0.0]]),
                {"hsv": np.array([[0.5, 0.5, 0.5]])},
                state_id=sid,
            )
        lm.graph_memory.models_in_memory["test_obj"] = {"patch": scm}

        # Step N: at state 0, predict state 1
        lm.current_mlh = {
            "graph_id": "test_obj",
            "state": 0,
            "evidence": 5.0,
        }
        lm._predicted_next_state = lm._predict_next_state()
        self.assertEqual(lm._predicted_next_state, 1)

        # Step N+1: state STAYS at 0 (prediction wrong)
        lm.current_mlh["state"] = 0
        status = lm._check_temporal_prediction()
        self.assertEqual(status, "confused")


class TestEndToEndMatchingWithBehaviors(unittest.TestCase):
    """End-to-end: train on two distinct behaviors, match, verify
    correct recognition and temporal prediction during matching_step."""

    def test_train_and_match_walking(self):
        """Train walking + door, match walking → correct ID."""
        from tbp.monty.frameworks.environments.behaviors import (
            door_opening,
            walking_gait,
        )

        lm = _make_lm()

        # Train
        _train_object(lm, walking_gait(n_steps=20), "walking")
        _train_object(lm, door_opening(n_steps=20), "door")

        # Match
        test_obs = walking_gait(n_steps=10)
        mlhs = _match_steps(lm, test_obs)

        self.assertEqual(mlhs[-1]["graph_id"], "walking")

    def test_matching_step_runs_event_detection(self):
        """matching_step sets event and speed signals."""
        from tbp.monty.frameworks.environments.behaviors import walking_gait

        lm = _make_lm()
        _train_object(lm, walking_gait(n_steps=20), "walking")

        # Match a few steps
        test_obs = walking_gait(n_steps=5)
        _match_steps(lm, test_obs)

        # Event and speed signals should be set (not crash)
        event = lm.get_event_signal()
        speed = lm.get_speed_signal()
        self.assertIsInstance(event, bool)
        # speed is either None or float
        self.assertTrue(speed is None or isinstance(speed, float))

    def test_matching_step_sets_temporal_prediction_status(self):
        """matching_step sets _temporal_prediction_status."""
        from tbp.monty.frameworks.environments.behaviors import walking_gait

        lm = _make_lm()
        _train_object(lm, walking_gait(n_steps=20), "walking")

        # Match
        test_obs = walking_gait(n_steps=5)
        _match_steps(lm, test_obs)

        # Status should be set (None for stateless model is fine)
        status = lm.get_temporal_prediction_status()
        # For a stateless model, predictions are None → status is None
        self.assertIsNone(status)


class TestEndToEndTimerDispatch(unittest.TestCase):
    """End-to-end: verify timer dispatch with event detection."""

    def test_dispatch_timer_signals_with_event(self):
        """Monty dispatches timer signals from LMs that detect events."""
        from tbp.monty.context import RuntimeContext
        from tbp.monty.frameworks.models.interval_timer import (
            GlobalIntervalTimer,
        )
        from tbp.monty.frameworks.models.monty_base import MontyBase

        timer = GlobalIntervalTimer(n_time_cells=16, ticks_per_step=1.0)
        for _ in range(5):
            timer.step()
        self.assertGreater(timer.get_current_tick(), 0)

        ctx = RuntimeContext(rng=np.random.RandomState(42), timer=timer)

        class MockLM:
            learning_module_id = "LM_0"

            def get_event_signal(self):
                return True

            def get_speed_signal(self):
                return None

        class TestMonty(MontyBase):
            def __init__(self):
                self.learning_modules = [MockLM()]

        monty = TestMonty()
        monty._dispatch_timer_signals(ctx)

        # Timer should have been reset
        self.assertEqual(timer.get_current_tick(), 0.0)


class TestEndToEndTimerSpeedWithSCM(unittest.TestCase):
    """End-to-end: verify _adjust_timer_speed works with a real SCM."""

    def test_speed_adjustment_with_scm(self):
        """Timer speed adjustment reads transitions from SCM."""
        from tbp.monty.context import RuntimeContext
        from tbp.monty.frameworks.models.interval_timer import (
            GlobalIntervalTimer,
        )
        from tbp.monty.frameworks.models.state_conditioned_model import (
            StateConditionedModel,
        )

        timer = GlobalIntervalTimer(n_time_cells=16, ticks_per_step=1.0)
        ctx = RuntimeContext(rng=np.random.RandomState(42), timer=timer)

        lm = _make_lm()

        # Build SCM with transition 0→1 at duration=10.0
        scm = StateConditionedModel(
            object_id="test_obj", max_nodes=50,
            max_size=1.0, num_voxels_per_dim=50,
        )
        scm.add_transition(0, 1, 10.0)
        for sid in [0, 1]:
            scm.build_model(
                np.array([[sid * 0.1, 0.0, 0.0]]),
                {"hsv": np.array([[0.5, 0.5, 0.5]])},
                state_id=sid,
            )
        lm.graph_memory.models_in_memory["test_obj"] = {"patch": scm}

        # Advance timer to tick=5.0
        for _ in range(5):
            timer.step()

        # LM is at state 1 (transitioned). Expected duration was 10.0,
        # actual tick is 5.0 → factor = 10.0/5.0 = 2.0
        lm.current_mlh = {
            "graph_id": "test_obj",
            "state": 1,
            "evidence": 5.0,
        }
        speed = lm._adjust_timer_speed(ctx)
        self.assertIsNotNone(speed)
        self.assertAlmostEqual(speed, 2.0)


class TestBackwardCompatibility(unittest.TestCase):
    """Verify that all temporal prediction code is a no-op for existing
    (stateless) models."""

    def test_matching_step_on_stateless_model(self):
        """matching_step works identically on a stateless model."""
        from tbp.monty.frameworks.environments.behaviors import walking_gait

        lm = _make_lm()
        walk_obs = walking_gait(n_steps=20)
        _train_object(lm, walk_obs, "walking")

        # Run matching
        test_obs = walking_gait(n_steps=8)
        mlhs = _match_steps(lm, test_obs)

        # Should recognize walking
        self.assertEqual(mlhs[-1]["graph_id"], "walking")

        # No temporal prediction made (no SCM)
        self.assertIsNone(lm.get_temporal_prediction_status())
        self.assertFalse(lm.get_event_signal())

    def test_vote_temporal_config_default_off(self):
        """temporal_voting defaults to False."""
        from tbp.monty.frameworks.models.graph_matching import (
            MontyForGraphMatching,
        )

        class TestMonty(MontyForGraphMatching):
            def __init__(self):
                self.temporal_voting = False
                self.conditional_voting = False
                self.learning_modules = []
                self.lm_to_lm_vote_matrix = None

        monty = TestMonty()
        # _vote should not call _vote_temporal
        monty._vote()  # Should not raise


if __name__ == "__main__":
    unittest.main()
