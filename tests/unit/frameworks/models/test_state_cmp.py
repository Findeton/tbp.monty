# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Tests for inferred_state in CMP State class (Phase 1 of object behaviors)."""

import unittest

import numpy as np

from tbp.monty.frameworks.models.states import State


class TestStateInferredState(unittest.TestCase):
    """Test that inferred_state is correctly handled in State."""

    def _make_state(self, inferred_state=None, use_state=True):
        """Helper to create a valid State with optional inferred_state."""
        return State(
            location=np.array([0.1, 0.2, 0.3]),
            morphological_features={
                "pose_vectors": np.eye(3),
                "pose_fully_defined": True,
            },
            non_morphological_features={"hsv": np.array([0.5, 0.5, 0.5])},
            confidence=0.8,
            use_state=use_state,
            sender_id="SM_0",
            sender_type="SM",
            inferred_state=inferred_state,
        )

    def test_state_without_inferred_state(self):
        """State without inferred_state defaults to None (backward compat)."""
        state = State(
            location=np.array([0.0, 0.0, 0.0]),
            morphological_features={
                "pose_vectors": np.eye(3),
                "pose_fully_defined": True,
            },
            non_morphological_features=None,
            confidence=1.0,
            use_state=True,
            sender_id="SM_0",
            sender_type="SM",
        )
        self.assertIsNone(state.inferred_state)

    def test_state_with_inferred_state(self):
        """State with inferred_state stores and retrieves it."""
        state = self._make_state(inferred_state=3)
        self.assertEqual(state.inferred_state, 3)

    def test_state_with_inferred_state_zero(self):
        """State with inferred_state=0 is valid (not confused with None)."""
        state = self._make_state(inferred_state=0)
        self.assertEqual(state.inferred_state, 0)
        self.assertIsNotNone(state.inferred_state)

    def test_repr_without_inferred_state(self):
        """repr does not mention inferred_state when None."""
        state = self._make_state(inferred_state=None)
        repr_str = repr(state)
        self.assertNotIn("Inferred State", repr_str)

    def test_repr_with_inferred_state(self):
        """repr includes inferred_state when set."""
        state = self._make_state(inferred_state=5)
        repr_str = repr(state)
        self.assertIn("Inferred State: 5", repr_str)

    def test_check_attributes_passes_with_inferred_state(self):
        """_check_all_attributes still passes when inferred_state is set."""
        state = self._make_state(inferred_state=2)
        # If we get here without AssertionError, the check passed
        self.assertEqual(state.inferred_state, 2)

    def test_check_attributes_passes_without_inferred_state(self):
        """_check_all_attributes still passes when inferred_state is None."""
        state = self._make_state(inferred_state=None)
        self.assertIsNone(state.inferred_state)

    def test_use_state_false_with_inferred_state(self):
        """use_state=False with inferred_state works (no attribute check)."""
        state = self._make_state(inferred_state=1, use_state=False)
        self.assertEqual(state.inferred_state, 1)
        self.assertFalse(state.use_state)

    def test_inferred_state_preserved_after_transform(self):
        """transform_morphological_features preserves inferred_state."""
        state = self._make_state(inferred_state=4)
        state.transform_morphological_features(
            translation=np.array([0.1, 0.0, 0.0]),
            rotation=np.eye(3),
        )
        self.assertEqual(state.inferred_state, 4)

    def test_inferred_state_preserved_after_set_displacement(self):
        """set_displacement preserves inferred_state."""
        state = self._make_state(inferred_state=7)
        state.set_displacement(np.array([0.01, 0.02, 0.03]))
        self.assertEqual(state.inferred_state, 7)

    def test_lm_state_with_inferred_state(self):
        """LM-sent State with inferred_state works correctly."""
        state = State(
            location=np.array([0.1, 0.2, 0.3]),
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
        self.assertEqual(state.inferred_state, 2)
        self.assertEqual(state.sender_type, "LM")


if __name__ == "__main__":
    unittest.main()
