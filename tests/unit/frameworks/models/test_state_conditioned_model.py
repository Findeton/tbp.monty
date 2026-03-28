# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Tests for StateConditionedModel (Phase 2 of object behaviors)."""

import unittest

import numpy as np

from tbp.monty.frameworks.models.state_conditioned_model import (
    StateConditionedModel,
)


def _make_features(n_points, seed=42):
    """Create minimal valid features dict for GridObjectModel.build_model."""
    rng = np.random.RandomState(seed)
    # Minimal features: pose_vectors (N, 9), pose_fully_defined (N,)
    pose_vecs = np.tile(np.eye(3).flatten(), (n_points, 1))
    # Add small noise to avoid degenerate grids
    pose_vecs += rng.randn(*pose_vecs.shape) * 0.01
    return {
        "pose_vectors": pose_vecs,
        "pose_fully_defined": np.ones(n_points, dtype=bool),
    }


def _make_locations(n_points, center=(0.0, 0.0, 0.0), spread=0.05, seed=42):
    """Create random 3D locations around a center."""
    rng = np.random.RandomState(seed)
    center = np.array(center)
    return center + rng.randn(n_points, 3) * spread


class TestStateConditionedModelBasic(unittest.TestCase):
    """Basic construction and state management tests."""

    def test_create_empty_model(self):
        model = StateConditionedModel(
            object_id="test_obj",
            max_nodes=50,
            max_size=0.3,
            num_voxels_per_dim=20,
        )
        self.assertEqual(model.get_num_states(), 0)
        self.assertEqual(model.get_states(), [])
        self.assertTrue(model.is_state_conditioned)

    def test_build_single_state(self):
        model = StateConditionedModel(
            object_id="stapler",
            max_nodes=50,
            max_size=0.3,
            num_voxels_per_dim=20,
        )
        locs = _make_locations(30)
        feats = _make_features(30)
        model.build_model(locs, feats, state_id=0)

        self.assertEqual(model.get_num_states(), 1)
        self.assertEqual(model.get_states(), [0])
        self.assertTrue(model.has_state(0))
        self.assertFalse(model.has_state(1))

    def test_build_multiple_states(self):
        model = StateConditionedModel(
            object_id="stapler",
            max_nodes=50,
            max_size=0.5,
            num_voxels_per_dim=20,
        )
        for sid in [0, 1, 2]:
            locs = _make_locations(20, center=(sid * 0.05, 0, 0), seed=sid)
            feats = _make_features(20, seed=sid)
            model.build_model(locs, feats, state_id=sid)

        self.assertEqual(model.get_num_states(), 3)
        self.assertEqual(model.get_states(), [0, 1, 2])

    def test_get_model_for_state(self):
        model = StateConditionedModel(
            object_id="obj",
            max_nodes=50,
            max_size=0.3,
            num_voxels_per_dim=20,
        )
        locs = _make_locations(20)
        feats = _make_features(20)
        model.build_model(locs, feats, state_id=0)

        sub_model = model.get_model_for_state(0)
        self.assertIsNotNone(sub_model._graph)

    def test_get_model_for_missing_state_raises(self):
        model = StateConditionedModel(
            object_id="obj",
            max_nodes=50,
            max_size=0.3,
            num_voxels_per_dim=20,
        )
        with self.assertRaises(KeyError):
            model.get_model_for_state(99)


class TestStateConditionedModelTransitions(unittest.TestCase):
    """Test state transition sequence storage."""

    def test_add_and_get_transitions(self):
        model = StateConditionedModel(
            object_id="obj",
            max_nodes=50,
            max_size=0.3,
            num_voxels_per_dim=20,
        )
        model.add_transition(0, 1, interval_duration=5.0)
        model.add_transition(1, 2, interval_duration=3.0)
        model.add_transition(2, 0, interval_duration=4.0)

        transitions = model.get_transitions()
        self.assertEqual(len(transitions), 3)
        self.assertEqual(transitions[0], (0, 1, 5.0))
        self.assertEqual(transitions[1], (1, 2, 3.0))
        self.assertEqual(transitions[2], (2, 0, 4.0))

    def test_get_next_state(self):
        model = StateConditionedModel(
            object_id="obj",
            max_nodes=50,
            max_size=0.3,
            num_voxels_per_dim=20,
        )
        model.add_transition(0, 1, 1.0)
        model.add_transition(1, 0, 1.0)

        self.assertEqual(model.get_next_state(0), 1)
        self.assertEqual(model.get_next_state(1), 0)
        self.assertIsNone(model.get_next_state(99))

    def test_empty_transitions(self):
        model = StateConditionedModel(
            object_id="obj",
            max_nodes=50,
            max_size=0.3,
            num_voxels_per_dim=20,
        )
        self.assertEqual(model.get_transitions(), [])
        self.assertIsNone(model.get_next_state(0))


class TestStateConditionedModelNearestNeighbors(unittest.TestCase):
    """Test nearest-neighbor search scoped to state."""

    def _build_two_state_model(self):
        """Build a model with 2 states at different locations."""
        model = StateConditionedModel(
            object_id="obj",
            max_nodes=50,
            max_size=0.5,
            num_voxels_per_dim=20,
        )
        # State 0: points around origin
        locs0 = _make_locations(25, center=(0.0, 0.0, 0.0), seed=0)
        feats0 = _make_features(25, seed=0)
        model.build_model(locs0, feats0, state_id=0)

        # State 1: points around (0.2, 0, 0)
        locs1 = _make_locations(25, center=(0.2, 0.0, 0.0), seed=1)
        feats1 = _make_features(25, seed=1)
        model.build_model(locs1, feats1, state_id=1)

        return model

    def test_nn_search_state_0(self):
        model = self._build_two_state_model()
        query = np.array([[0.0, 0.0, 0.0]])
        result = model.find_nearest_neighbors(
            query, num_neighbors=3, state_id=0, return_distance=True,
        )
        # KDTree returns 1D for single query point
        distances = np.atleast_2d(result[0])
        self.assertEqual(distances.shape[1], 3)
        # Nearest neighbor in state 0 should be close to origin
        self.assertLess(distances[0, 0], 0.1)

    def test_nn_search_state_1(self):
        model = self._build_two_state_model()
        query = np.array([[0.2, 0.0, 0.0]])
        result = model.find_nearest_neighbors(
            query, num_neighbors=3, state_id=1, return_distance=True,
        )
        distances = np.atleast_2d(result[0])
        self.assertLess(distances[0, 0], 0.1)

    def test_nn_search_wrong_state(self):
        """Searching near state 0's points in state 1's graph returns farther."""
        model = self._build_two_state_model()
        query = np.array([[0.0, 0.0, 0.0]])
        # Search in state 1 (centered at 0.2)
        result = model.find_nearest_neighbors(
            query, num_neighbors=1, state_id=1, return_distance=True,
        )
        distances = np.atleast_1d(result[0])
        # Should be farther than if we searched state 0
        self.assertGreater(distances.flat[0], 0.1)


class TestStateConditionedModelRepr(unittest.TestCase):
    """Test string representation."""

    def test_repr(self):
        model = StateConditionedModel(
            object_id="test",
            max_nodes=50,
            max_size=0.3,
            num_voxels_per_dim=20,
        )
        locs = _make_locations(20)
        feats = _make_features(20)
        model.build_model(locs, feats, state_id=0)

        repr_str = repr(model)
        self.assertIn("StateConditionedModel", repr_str)
        self.assertIn("test", repr_str)
        self.assertIn("states=[0]", repr_str)


class TestStateConditionedModelGraphProperty(unittest.TestCase):
    """Test backward-compat _graph property."""

    def test_graph_property_returns_first_state(self):
        model = StateConditionedModel(
            object_id="obj",
            max_nodes=50,
            max_size=0.3,
            num_voxels_per_dim=20,
        )
        locs = _make_locations(20)
        feats = _make_features(20)
        model.build_model(locs, feats, state_id=0)

        self.assertIsNotNone(model._graph)
        self.assertEqual(model._graph, model.get_model_for_state(0)._graph)

    def test_graph_property_none_when_empty(self):
        model = StateConditionedModel(
            object_id="obj",
            max_nodes=50,
            max_size=0.3,
            num_voxels_per_dim=20,
        )
        self.assertIsNone(model._graph)


if __name__ == "__main__":
    unittest.main()
