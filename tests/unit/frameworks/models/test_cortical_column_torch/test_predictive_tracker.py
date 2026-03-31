# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Unit tests for PredictiveTracker."""

import numpy as np
import pytest

from tbp.monty.frameworks.models.cortical_column_torch.predictive_tracker import (
    PredictiveTracker,
)


@pytest.fixture
def tracker():
    return PredictiveTracker(
        surprise_threshold=0.3,
        anchor_evidence_threshold=0.1,
        prediction_bonus_weight=0.5,
        max_consecutive_drops=3,
    )


class TestAnchor:
    def test_anchor_sets_location(self, tracker):
        tracker.anchor("mug", np.array([1.0, 2.0, 3.0]))
        assert tracker.is_anchored("mug")
        np.testing.assert_allclose(
            tracker.get_location("mug"), [1.0, 2.0, 3.0]
        )

    def test_not_anchored_by_default(self, tracker):
        assert not tracker.is_anchored("mug")

    def test_anchor_overwrites(self, tracker):
        tracker.anchor("mug", np.array([1.0, 0.0, 0.0]))
        tracker.anchor("mug", np.array([0.0, 5.0, 0.0]))
        np.testing.assert_allclose(
            tracker.get_location("mug"), [0.0, 5.0, 0.0]
        )


class TestTrack:
    def test_displacement_updates_location(self, tracker):
        tracker.anchor("mug", np.array([1.0, 0.0, 0.0]))
        tracker.track("mug", np.array([0.1, 0.0, 0.0]))
        np.testing.assert_allclose(
            tracker.get_location("mug"), [1.1, 0.0, 0.0]
        )

    def test_multiple_displacements_accumulate(self, tracker):
        tracker.anchor("mug", np.array([0.0, 0.0, 0.0]))
        for _ in range(10):
            tracker.track("mug", np.array([0.1, 0.0, 0.0]))
        np.testing.assert_allclose(
            tracker.get_location("mug"), [1.0, 0.0, 0.0], atol=1e-10
        )

    def test_track_unanchored_is_noop(self, tracker):
        # Should not raise
        tracker.track("mug", np.array([1.0, 0.0, 0.0]))
        assert not tracker.is_anchored("mug")


class TestConfirmAndDrop:
    def test_confirm_increments(self, tracker):
        tracker.anchor("mug", np.array([0.0, 0.0, 0.0]))
        assert tracker.get_n_confirmed("mug") == 0
        tracker.confirm("mug")
        assert tracker.get_n_confirmed("mug") == 1
        tracker.confirm("mug")
        assert tracker.get_n_confirmed("mug") == 2

    def test_drop_removes_anchor(self, tracker):
        tracker.anchor("mug", np.array([0.0, 0.0, 0.0]))
        tracker.drop("mug")
        assert not tracker.is_anchored("mug")

    def test_drop_resets_confirmation_count(self, tracker):
        tracker.anchor("mug", np.array([0.0, 0.0, 0.0]))
        tracker.confirm("mug")
        tracker.confirm("mug")
        tracker.drop("mug")
        assert tracker.get_n_confirmed("mug") == 0

    def test_confirm_resets_consecutive_drops(self, tracker):
        tracker.anchor("mug", np.array([0.0, 0.0, 0.0]))
        tracker.drop("mug")
        tracker.anchor("mug", np.array([0.0, 0.0, 0.0]))
        tracker.confirm("mug")
        # Consecutive drops should be reset
        assert not tracker.is_retired("mug")


class TestRetirement:
    def test_retire_after_max_drops(self, tracker):
        """Object retires after max_consecutive_drops."""
        for _ in range(3):  # max_consecutive_drops=3
            tracker.anchor("mug", np.array([0.0, 0.0, 0.0]))
            tracker.drop("mug")
        assert tracker.is_retired("mug")

    def test_retired_cannot_anchor(self, tracker):
        for _ in range(3):
            tracker.anchor("mug", np.array([0.0, 0.0, 0.0]))
            tracker.drop("mug")
        assert tracker.is_retired("mug")
        tracker.anchor("mug", np.array([1.0, 1.0, 1.0]))
        assert not tracker.is_anchored("mug")

    def test_confirm_prevents_retirement(self, tracker):
        """Successful confirmation resets the drop counter."""
        tracker.anchor("mug", np.array([0.0, 0.0, 0.0]))
        tracker.drop("mug")
        tracker.anchor("mug", np.array([0.0, 0.0, 0.0]))
        tracker.drop("mug")
        # 2 consecutive drops, now confirm
        tracker.anchor("mug", np.array([0.0, 0.0, 0.0]))
        tracker.confirm("mug")
        # Drop count should be reset, can drop more before retiring
        tracker.drop("mug")
        tracker.anchor("mug", np.array([0.0, 0.0, 0.0]))
        tracker.drop("mug")
        assert not tracker.is_retired("mug")  # only 2 consecutive


class TestMultiObject:
    def test_independent_tracking(self, tracker):
        tracker.anchor("mug", np.array([0.0, 0.0, 0.0]))
        tracker.anchor("drill", np.array([5.0, 0.0, 0.0]))
        tracker.track("mug", np.array([0.1, 0.0, 0.0]))
        tracker.track("drill", np.array([0.1, 0.0, 0.0]))

        np.testing.assert_allclose(
            tracker.get_location("mug"), [0.1, 0.0, 0.0]
        )
        np.testing.assert_allclose(
            tracker.get_location("drill"), [5.1, 0.0, 0.0]
        )

    def test_drop_one_keeps_other(self, tracker):
        tracker.anchor("mug", np.array([0.0, 0.0, 0.0]))
        tracker.anchor("drill", np.array([5.0, 0.0, 0.0]))
        tracker.drop("mug")
        assert not tracker.is_anchored("mug")
        assert tracker.is_anchored("drill")


class TestReset:
    def test_reset_clears_everything(self, tracker):
        tracker.anchor("mug", np.array([0.0, 0.0, 0.0]))
        tracker.confirm("mug")
        for _ in range(3):
            tracker.anchor("drill", np.array([0.0, 0.0, 0.0]))
            tracker.drop("drill")

        tracker.reset()
        assert not tracker.is_anchored("mug")
        assert not tracker.is_retired("drill")
        assert tracker.anchored_objects == []


class TestAnchoredObjects:
    def test_lists_anchored(self, tracker):
        tracker.anchor("mug", np.array([0.0, 0.0, 0.0]))
        tracker.anchor("drill", np.array([1.0, 0.0, 0.0]))
        assert set(tracker.anchored_objects) == {"mug", "drill"}

    def test_dropped_not_listed(self, tracker):
        tracker.anchor("mug", np.array([0.0, 0.0, 0.0]))
        tracker.anchor("drill", np.array([1.0, 0.0, 0.0]))
        tracker.drop("mug")
        assert tracker.anchored_objects == ["drill"]
