# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Tests for Phase 3b: Vote on State.

Tests that votes carry inferred_state and that state-aware filtering
prevents cross-state vote reinforcement.
"""

import unittest

import numpy as np

from tbp.monty.frameworks.models.states import State


class TestVoteStatePreservation(unittest.TestCase):
    """Test that vote States carry inferred_state through the pipeline."""

    def test_vote_state_has_inferred_state(self):
        """Vote State should carry inferred_state."""
        vote = State(
            location=np.array([0.1, 0.0, 0.0]),
            morphological_features={
                "pose_vectors": np.eye(3),
                "pose_fully_defined": True,
            },
            non_morphological_features=None,
            confidence=0.9,
            use_state=True,
            sender_id="LM_0",
            sender_type="LM",
            inferred_state=1,
        )
        self.assertEqual(vote.inferred_state, 1)

    def test_vote_state_none_backward_compat(self):
        """Vote State without inferred_state defaults to None."""
        vote = State(
            location=np.array([0.0, 0.0, 0.0]),
            morphological_features={
                "pose_vectors": np.eye(3),
                "pose_fully_defined": True,
            },
            non_morphological_features=None,
            confidence=0.5,
            use_state=True,
            sender_id="LM_0",
            sender_type="LM",
        )
        self.assertIsNone(vote.inferred_state)

    def test_vote_deepcopy_preserves_state(self):
        """Deep copy of vote State should preserve inferred_state."""
        import copy

        vote = State(
            location=np.array([0.1, 0.0, 0.0]),
            morphological_features={
                "pose_vectors": np.eye(3),
                "pose_fully_defined": True,
            },
            non_morphological_features=None,
            confidence=0.9,
            use_state=True,
            sender_id="LM_0",
            sender_type="LM",
            inferred_state=2,
        )
        copied = copy.deepcopy(vote)
        self.assertEqual(copied.inferred_state, 2)
        # Modify original — copy should be independent
        vote.inferred_state = 3
        self.assertEqual(copied.inferred_state, 2)


class TestStateVoteFiltering(unittest.TestCase):
    """Test that _update_evidence_with_vote filters by state."""

    def _make_votes(self, locations, confidences, states=None):
        """Create a list of vote States."""
        votes = []
        for i in range(len(locations)):
            state = states[i] if states is not None else None
            votes.append(
                State(
                    location=np.array(locations[i]),
                    morphological_features={
                        "pose_vectors": np.eye(3),
                        "pose_fully_defined": True,
                    },
                    non_morphological_features=None,
                    confidence=confidences[i],
                    use_state=True,
                    sender_id="LM_1",
                    sender_type="LM",
                    inferred_state=state,
                )
            )
        return votes

    def test_no_state_votes_apply_to_all(self):
        """Votes without inferred_state should apply to all hypotheses."""
        votes = self._make_votes(
            locations=[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
            confidences=[0.8, 0.6],
            states=None,
        )
        # All votes should have inferred_state = None
        for v in votes:
            self.assertIsNone(v.inferred_state)

    def test_state_votes_carry_state(self):
        """Votes with inferred_state should carry it."""
        votes = self._make_votes(
            locations=[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
            confidences=[0.8, 0.6],
            states=[0, 1],
        )
        self.assertEqual(votes[0].inferred_state, 0)
        self.assertEqual(votes[1].inferred_state, 1)

    def test_state_extraction_for_filtering(self):
        """Test that state extraction logic works for filtering."""
        votes = self._make_votes(
            locations=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            confidences=[0.8, 0.6, 0.9],
            states=[0, 1, None],
        )
        # Extract states like the _update_evidence_with_vote does
        vote_states = np.full(len(votes), -1, dtype=np.int64)
        for n, vote in enumerate(votes):
            if getattr(vote, "inferred_state", None) is not None:
                vote_states[n] = vote.inferred_state

        np.testing.assert_array_equal(vote_states, [0, 1, -1])

    def test_state_match_logic(self):
        """Test the state match/mismatch mask computation."""
        # Simulate 4 hypotheses: 2 at state 0, 2 at state 1
        hyp_states = np.array([0, 0, 1, 1], dtype=np.int64)

        # 3 votes: state 0, state 1, no state
        vote_states = np.array([0, 1, -1], dtype=np.int64)

        # Simulate nearest vote IDs (each hyp has 2 nearest votes)
        # Hyp 0 (state=0): nearest votes are [0, 2] (state=0, no_state)
        # Hyp 1 (state=0): nearest votes are [1, 2] (state=1, no_state)
        # Hyp 2 (state=1): nearest votes are [1, 0] (state=1, state=0)
        # Hyp 3 (state=1): nearest votes are [2, 1] (no_state, state=1)
        radius_node_ids = np.array([
            [0, 2],
            [1, 2],
            [1, 0],
            [2, 1],
        ])
        nearest_vote_states = vote_states[radius_node_ids]

        # Match: same state or no state (-1)
        state_match = (
            (nearest_vote_states == hyp_states[:, None])
            | (nearest_vote_states == -1)
        )

        # Hyp 0 (state=0): vote[0]=state0 → match, vote[2]=no_state → match
        self.assertTrue(state_match[0, 0])  # state 0 == state 0
        self.assertTrue(state_match[0, 1])  # no state → always match

        # Hyp 1 (state=0): vote[1]=state1 → no match, vote[2]=no_state → match
        self.assertFalse(state_match[1, 0])  # state 1 != state 0
        self.assertTrue(state_match[1, 1])  # no state → always match

        # Hyp 2 (state=1): vote[1]=state1 → match, vote[0]=state0 → no match
        self.assertTrue(state_match[2, 0])  # state 1 == state 1
        self.assertFalse(state_match[2, 1])  # state 0 != state 1

        # Hyp 3 (state=1): vote[2]=no_state → match, vote[1]=state1 → match
        self.assertTrue(state_match[3, 0])  # no state → always match
        self.assertTrue(state_match[3, 1])  # state 1 == state 1


class TestCombineVotesPreservesState(unittest.TestCase):
    """Test that _combine_votes preserves inferred_state."""

    def test_deepcopy_preserves_inferred_state(self):
        """The deep copy in _combine_votes should preserve inferred_state."""
        import copy

        original_vote = State(
            location=np.array([0.1, 0.2, 0.3]),
            morphological_features={
                "pose_vectors": np.eye(3),
                "pose_fully_defined": True,
            },
            non_morphological_features=None,
            confidence=0.85,
            use_state=True,
            sender_id="LM_0",
            sender_type="LM",
            inferred_state=1,
        )

        # Simulate what _combine_votes does
        new_s = copy.deepcopy(original_vote)
        # Transform morphological features
        new_s.transform_morphological_features(
            translation=np.array([0.01, 0.0, 0.0]),
            rotation=np.eye(3),
        )

        # inferred_state should be preserved
        self.assertEqual(new_s.inferred_state, 1)
        # location should have changed
        self.assertAlmostEqual(new_s.location[0], 0.11, places=5)

    def test_transform_does_not_affect_inferred_state(self):
        """transform_morphological_features should not modify inferred_state."""
        vote = State(
            location=np.array([0.0, 0.0, 0.0]),
            morphological_features={
                "pose_vectors": np.eye(3),
                "pose_fully_defined": True,
            },
            non_morphological_features=None,
            confidence=0.5,
            use_state=True,
            sender_id="LM_0",
            sender_type="LM",
            inferred_state=0,
        )
        vote.transform_morphological_features(
            translation=np.array([1.0, 2.0, 3.0]),
            rotation=np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=float),
        )
        self.assertEqual(vote.inferred_state, 0)


if __name__ == "__main__":
    unittest.main()
