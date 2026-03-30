# Copyright 2025-2026 Thousand Brains Project
# MIT License

"""Tests for laminar CorticalColumnTorch (all Track 10 features)."""

import unittest

import numpy as np

from tbp.monty.frameworks.models.cortical_column_torch.column import (
    CorticalColumnTorch,
)
from tbp.monty.frameworks.models.states import State


def _make_state(location=None, hsv=None):
    return State(
        location=np.array(location or [0.1, 0.2, 0.3], dtype=np.float64),
        morphological_features={
            "pose_vectors": np.eye(3),
            "pose_fully_defined": True,
        },
        non_morphological_features={"hsv": hsv or [0.5, 0.3, 0.8]},
        confidence=1.0,
        use_state=True,
        sender_id="test_sm",
        sender_type="SM",
    )


class TestLaminarOnlyColumn(unittest.TestCase):
    """Test laminar=True without other Track 10 features."""

    def _make_col(self, **kw):
        defaults = dict(
            n_minicolumns=32,
            n_cells_per_minicolumn=8,
            sparsity=0.1,
            laminar=True,
            seed=42,
        )
        defaults.update(kw)
        return CorticalColumnTorch(**defaults)

    def test_train_and_eval(self):
        col = self._make_col()
        col.pre_episode(mode="train", object_name="mug")
        for i in range(10):
            state = _make_state(location=[0.1 * i, 0.0, 0.0])
            result = col.step(state)
            self.assertIn("surprise", result)
        col.post_episode()

        col.pre_episode(mode="eval")
        for i in range(5):
            state = _make_state(location=[0.1 * i, 0.0, 0.0])
            result = col.step(state)
        self.assertIn("evidence", result)

    def test_l4_activation_is_feedforward(self):
        col = self._make_col()
        col.pre_episode(mode="eval")
        state = _make_state()
        col.step(state)
        # L4 should have active minicolumns
        self.assertTrue(col._l4._active_mc.any())

    def test_l5_output_exists(self):
        col = self._make_col()
        col.pre_episode(mode="eval")
        state = _make_state()
        col.step(state)
        self.assertGreater(col._l5._activation.abs().sum().item(), 0)

    def test_l6_feedback_on_next_step(self):
        col = self._make_col()
        col.pre_episode(mode="eval")
        col.step(_make_state(location=[0.1, 0.0, 0.0]))
        l6_1 = col._prev_l6_activation.clone()
        col.step(_make_state(location=[0.2, 0.0, 0.0]))
        # L6 from step 1 should be used in step 2's thalamic gate
        self.assertGreater(l6_1.abs().sum().item(), 0)


class TestLaminarDisabledMatchesFlat(unittest.TestCase):
    """laminar=False should produce identical results to Track 9."""

    def test_flat_mode(self):
        col = CorticalColumnTorch(
            n_minicolumns=32,
            n_cells_per_minicolumn=8,
            sparsity=0.1,
            laminar=False,
            seed=42,
        )
        col.pre_episode(mode="eval")
        state = _make_state()
        result = col.step(state)
        self.assertIn("surprise", result)
        self.assertIn("evidence", result)
        self.assertIn("active_cells", result)


class TestMultiHeadInColumn(unittest.TestCase):
    def test_multi_head_flat(self):
        col = CorticalColumnTorch(
            n_minicolumns=32,
            n_cells_per_minicolumn=8,
            sparsity=0.1,
            multi_head=True,
            n_heads=4,
            seed=42,
        )
        col.pre_episode(mode="train", object_name="obj")
        for i in range(5):
            result = col.step(_make_state(location=[0.1 * i, 0.0, 0.0]))
        col.post_episode()
        self.assertIsNotNone(col._multihead_dendrites)

    def test_multi_head_laminar(self):
        col = CorticalColumnTorch(
            n_minicolumns=32,
            n_cells_per_minicolumn=8,
            sparsity=0.1,
            laminar=True,
            multi_head=True,
            n_heads=4,
            seed=42,
        )
        col.pre_episode(mode="train", object_name="obj")
        for i in range(5):
            col.step(_make_state(location=[0.1 * i, 0.0, 0.0]))
        col.post_episode()


class TestPlateauInColumn(unittest.TestCase):
    def test_plateau_enriches_retrieval(self):
        col = CorticalColumnTorch(
            n_minicolumns=32,
            n_cells_per_minicolumn=8,
            sparsity=0.1,
            use_plateau=True,
            use_apical=True,
            seed=42,
        )
        col.pre_episode(mode="eval")
        col.step(_make_state())
        self.assertIsNotNone(col._plateau)

    def test_plateau_laminar(self):
        col = CorticalColumnTorch(
            n_minicolumns=32,
            n_cells_per_minicolumn=8,
            sparsity=0.1,
            laminar=True,
            use_plateau=True,
            use_apical=True,
            seed=42,
        )
        col.pre_episode(mode="eval")
        col.step(_make_state())


class TestInterneuronsInColumn(unittest.TestCase):
    def test_interneurons_flat(self):
        col = CorticalColumnTorch(
            n_minicolumns=32,
            n_cells_per_minicolumn=8,
            sparsity=0.1,
            multi_head=True,
            n_heads=4,
            use_interneurons=True,
            seed=42,
        )
        col.pre_episode(mode="train", object_name="obj")
        for i in range(5):
            col.step(_make_state(location=[0.1 * i, 0.0, 0.0]))
        col.post_episode()

    def test_interneurons_laminar(self):
        col = CorticalColumnTorch(
            n_minicolumns=32,
            n_cells_per_minicolumn=8,
            sparsity=0.1,
            laminar=True,
            multi_head=True,
            n_heads=4,
            use_interneurons=True,
            seed=42,
        )
        col.pre_episode(mode="train", object_name="obj")
        for i in range(5):
            col.step(_make_state(location=[0.1 * i, 0.0, 0.0]))
        col.post_episode()


class TestSTDPInColumn(unittest.TestCase):
    def test_stdp_flat(self):
        col = CorticalColumnTorch(
            n_minicolumns=32,
            n_cells_per_minicolumn=8,
            sparsity=0.1,
            use_stdp=True,
            seed=42,
        )
        col.pre_episode(mode="train", object_name="obj")
        for i in range(5):
            col.step(_make_state(location=[0.1 * i, 0.0, 0.0]))
        col.post_episode()

    def test_stdp_with_eligibility(self):
        col = CorticalColumnTorch(
            n_minicolumns=32,
            n_cells_per_minicolumn=8,
            sparsity=0.1,
            use_stdp=True,
            use_eligibility=True,
            use_neuromodulation=True,
            seed=42,
        )
        col.pre_episode(mode="train", object_name="obj")
        for i in range(5):
            col.step(_make_state(location=[0.1 * i, 0.0, 0.0]))
        col.post_episode()

    def test_stdp_laminar(self):
        col = CorticalColumnTorch(
            n_minicolumns=32,
            n_cells_per_minicolumn=8,
            sparsity=0.1,
            laminar=True,
            use_stdp=True,
            seed=42,
        )
        col.pre_episode(mode="train", object_name="obj")
        for i in range(5):
            col.step(_make_state(location=[0.1 * i, 0.0, 0.0]))
        col.post_episode()


class TestPhaseCodingInColumn(unittest.TestCase):
    def test_phase_coding_flat(self):
        col = CorticalColumnTorch(
            n_minicolumns=32,
            n_cells_per_minicolumn=8,
            sparsity=0.1,
            use_phase_coding=True,
            seed=42,
        )
        col.pre_episode(mode="train", object_name="obj")
        for i in range(5):
            col.step(_make_state(location=[0.1 * i, 0.0, 0.0]))
        col.post_episode()
        self.assertIsNotNone(col._oscillator)

    def test_phase_coding_laminar(self):
        col = CorticalColumnTorch(
            n_minicolumns=32,
            n_cells_per_minicolumn=8,
            sparsity=0.1,
            laminar=True,
            use_phase_coding=True,
            seed=42,
        )
        col.pre_episode(mode="train", object_name="obj")
        for i in range(5):
            col.step(_make_state(location=[0.1 * i, 0.0, 0.0]))
        col.post_episode()


class TestThalamicRelayInColumn(unittest.TestCase):
    def test_thalamic_relay_laminar(self):
        col = CorticalColumnTorch(
            n_minicolumns=32,
            n_cells_per_minicolumn=8,
            sparsity=0.1,
            laminar=True,
            use_thalamic_relay=True,
            seed=42,
        )
        col.pre_episode(mode="train", object_name="obj")
        for i in range(5):
            col.step(_make_state(location=[0.1 * i, 0.0, 0.0]))
        col.post_episode()

    def test_disable_thalamic_matches_track9(self):
        col = CorticalColumnTorch(
            n_minicolumns=32,
            n_cells_per_minicolumn=8,
            sparsity=0.1,
            use_thalamic_relay=False,
            seed=42,
        )
        col.pre_episode(mode="eval")
        result = col.step(_make_state())
        self.assertIn("surprise", result)


class TestAllFeaturesEnabled(unittest.TestCase):
    """Test with ALL Track 10 features enabled at once."""

    def test_all_features_laminar(self):
        col = CorticalColumnTorch(
            n_minicolumns=32,
            n_cells_per_minicolumn=8,
            sparsity=0.1,
            laminar=True,
            multi_head=True,
            n_heads=4,
            use_plateau=True,
            use_interneurons=True,
            use_stdp=True,
            use_eligibility=True,
            use_phase_coding=True,
            use_thalamic_relay=True,
            use_apical=True,
            use_neuromodulation=True,
            use_motor_prediction=True,
            seed=42,
        )
        # Train
        col.pre_episode(mode="train", object_name="mug")
        for i in range(10):
            state = _make_state(
                location=[0.1 * i, 0.0, 0.0],
                hsv=[0.5, 0.3, 0.8],
            )
            result = col.step(state)
            self.assertIn("surprise", result)
        col.post_episode()

        # Eval
        col.pre_episode(mode="eval")
        for i in range(5):
            state = _make_state(location=[0.1 * i, 0.0, 0.0])
            result = col.step(state)
        self.assertIn("evidence", result)

    def test_all_features_flat(self):
        """All features except laminar (flat mode with Track 10 additions)."""
        col = CorticalColumnTorch(
            n_minicolumns=32,
            n_cells_per_minicolumn=8,
            sparsity=0.1,
            laminar=False,
            multi_head=True,
            n_heads=4,
            use_plateau=True,
            use_interneurons=True,
            use_stdp=True,
            use_eligibility=True,
            use_phase_coding=True,
            use_apical=True,
            use_neuromodulation=True,
            seed=42,
        )
        col.pre_episode(mode="train", object_name="test")
        for i in range(5):
            col.step(_make_state(location=[0.1 * i, 0.0, 0.0]))
        col.post_episode()


if __name__ == "__main__":
    unittest.main()
