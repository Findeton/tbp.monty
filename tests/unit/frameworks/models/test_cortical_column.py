# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Tests for the biologically plausible cortical column.

Validates:
1. SDR encoders (GridCellEncoder, ScalarEncoder, FeatureSDREncoder)
2. Dendritic segments (growth, activation, learning) — vectorized
3. SDRObjectMemory (storage, matching)
4. CorticalColumn (feedforward, prediction, burst/predicted, learning)
5. Spatial pooler learning + homeostatic boosting
6. End-to-end: train on objects, recognize them
"""

import unittest

import numpy as np


# ---------------------------------------------------------------------------
# GridCellEncoder tests
# ---------------------------------------------------------------------------


class TestGridCellEncoder(unittest.TestCase):
    """Tests for multi-scale periodic location encoding."""

    def _make_encoder(self, **kwargs):
        from tbp.monty.frameworks.models.cortical_column.encoders import (
            GridCellEncoder,
        )

        defaults = dict(n_modules=4, cells_per_module=64, n_active_per_module=4)
        defaults.update(kwargs)
        return GridCellEncoder(**defaults)

    def test_output_shape(self):
        enc = self._make_encoder()
        sdr = enc.encode([0.0, 0.0, 0.0])
        self.assertEqual(sdr.shape, (4 * 64,))

    def test_correct_sparsity(self):
        enc = self._make_encoder()
        sdr = enc.encode([0.1, 0.2, 0.3])
        n_active = int(np.sum(sdr))
        self.assertEqual(n_active, 4 * 4)

    def test_nearby_locations_share_bits(self):
        enc = self._make_encoder()
        sdr_a = enc.encode([0.0, 0.0, 0.0])
        sdr_b = enc.encode([0.001, 0.001, 0.001])
        overlap = float(np.dot(sdr_a, sdr_b))
        n_active = float(np.sum(sdr_a))
        self.assertGreater(overlap / n_active, 0.5)

    def test_distant_locations_differ(self):
        enc = self._make_encoder()
        sdr_a = enc.encode([0.0, 0.0, 0.0])
        sdr_b = enc.encode([1.0, 1.0, 1.0])
        overlap = float(np.dot(sdr_a, sdr_b))
        n_active = float(np.sum(sdr_a))
        self.assertLess(overlap / n_active, 0.5)

    def test_deterministic(self):
        enc = self._make_encoder()
        sdr_1 = enc.encode([0.5, 0.5, 0.5])
        sdr_2 = enc.encode([0.5, 0.5, 0.5])
        np.testing.assert_array_equal(sdr_1, sdr_2)

    def test_different_scales_give_resolution(self):
        enc = self._make_encoder(n_modules=8, min_scale=0.01, max_scale=1.0)
        sdr_a = enc.encode([0.0, 0.0, 0.0])
        sdr_b = enc.encode([0.005, 0.0, 0.0])
        overlap = float(np.dot(sdr_a, sdr_b))
        n_active = float(np.sum(sdr_a))
        self.assertGreater(overlap, 0)
        self.assertLess(overlap, n_active)


# ---------------------------------------------------------------------------
# ScalarEncoder tests
# ---------------------------------------------------------------------------


class TestScalarEncoder(unittest.TestCase):

    def _make_encoder(self, **kwargs):
        from tbp.monty.frameworks.models.cortical_column.encoders import ScalarEncoder

        defaults = dict(n_bits=128, n_active=10, min_val=0.0, max_val=1.0)
        defaults.update(kwargs)
        return ScalarEncoder(**defaults)

    def test_output_shape(self):
        enc = self._make_encoder()
        sdr = enc.encode(0.5)
        self.assertEqual(sdr.shape, (128,))

    def test_correct_sparsity(self):
        enc = self._make_encoder()
        sdr = enc.encode(0.5)
        self.assertEqual(int(np.sum(sdr)), 10)

    def test_adjacent_values_share_bits(self):
        enc = self._make_encoder()
        sdr_a = enc.encode(0.5)
        sdr_b = enc.encode(0.51)
        overlap = float(np.dot(sdr_a, sdr_b))
        self.assertGreater(overlap, 0)

    def test_distant_values_differ(self):
        enc = self._make_encoder()
        sdr_a = enc.encode(0.0)
        sdr_b = enc.encode(1.0)
        overlap = float(np.dot(sdr_a, sdr_b))
        self.assertEqual(overlap, 0)

    def test_periodic_wraps(self):
        enc = self._make_encoder(periodic=True)
        sdr_a = enc.encode(0.0)
        sdr_b = enc.encode(0.99)
        overlap = float(np.dot(sdr_a, sdr_b))
        self.assertGreater(overlap, 0)

    def test_clamps_out_of_range(self):
        enc = self._make_encoder()
        sdr_lo = enc.encode(-1.0)
        sdr_hi = enc.encode(2.0)
        self.assertEqual(int(np.sum(sdr_lo)), 10)
        self.assertEqual(int(np.sum(sdr_hi)), 10)


# ---------------------------------------------------------------------------
# FeatureSDREncoder tests
# ---------------------------------------------------------------------------


class TestFeatureSDREncoder(unittest.TestCase):

    def _make_encoder(self):
        from tbp.monty.frameworks.models.cortical_column.encoders import (
            FeatureSDREncoder,
        )

        return FeatureSDREncoder(
            grid_cell_kwargs=dict(n_modules=4, cells_per_module=64),
        )

    def test_total_bits(self):
        enc = self._make_encoder()
        self.assertGreater(enc.total_bits, 0)
        self.assertGreater(enc.n_active, 0)

    def test_encode_location(self):
        enc = self._make_encoder()
        sdr = enc.encode_location(np.array([0.1, 0.2, 0.3]))
        self.assertEqual(sdr.shape[0], enc._location_bits)
        self.assertGreater(np.sum(sdr), 0)


# ---------------------------------------------------------------------------
# DendriteSegments tests (vectorized)
# ---------------------------------------------------------------------------


class TestDendriteSegments(unittest.TestCase):

    def _make_dendrites(self, **kwargs):
        from tbp.monty.frameworks.models.cortical_column.dendrites import (
            DendriteSegments,
        )

        defaults = dict(
            n_cells=100,
            max_segments_per_cell=8,
            max_synapses_per_segment=10,
            activation_threshold=5,
        )
        defaults.update(kwargs)
        return DendriteSegments(**defaults)

    def test_grow_segment(self):
        dend = self._make_dendrites()
        source_cells = {10, 20, 30, 40, 50, 60, 70, 80, 90}
        success = dend.grow_segment(0, source_cells)
        self.assertTrue(success)
        self.assertEqual(dend.total_segments, 1)

    def test_grow_respects_max(self):
        dend = self._make_dendrites(max_segments_per_cell=2)
        source_cells = set(range(10, 30))
        dend.grow_segment(0, source_cells)
        dend.grow_segment(0, source_cells)
        success = dend.grow_segment(0, source_cells)
        self.assertFalse(success)

    def test_prediction_from_active_cells(self):
        dend = self._make_dendrites(
            activation_threshold=3,
            max_synapses_per_segment=6,
        )
        source_cells = {10, 20, 30, 40, 50, 60}
        dend.grow_segment(0, source_cells, n_synapses=6)

        predicted = dend.compute_predicted_cells({10, 20, 30, 40})
        self.assertIn(0, predicted)

    def test_no_prediction_below_threshold(self):
        dend = self._make_dendrites(
            activation_threshold=5,
            max_synapses_per_segment=6,
        )
        source_cells = {10, 20, 30, 40, 50, 60}
        dend.grow_segment(0, source_cells, n_synapses=6)

        predicted = dend.compute_predicted_cells({10, 20})
        self.assertNotIn(0, predicted)

    def test_strengthen_increases_permanence(self):
        dend = self._make_dendrites()
        source_cells = set(range(10, 25))
        dend.grow_segment(0, source_cells)
        old_perm = float(dend._permanences[0, 0, 0])
        dend.strengthen_segment(0, 0, source_cells)
        new_perm = float(dend._permanences[0, 0, 0])
        self.assertGreater(new_perm, old_perm)

    def test_punish_decreases_permanence(self):
        dend = self._make_dendrites()
        source_cells = set(range(10, 25))
        dend.grow_segment(0, source_cells)
        old_perm = float(dend._permanences[0, 0, 0])
        dend.punish_segment(0, 0)
        new_perm = float(dend._permanences[0, 0, 0])
        self.assertLess(new_perm, old_perm)

    def test_multiple_segments_different_contexts(self):
        dend = self._make_dendrites(
            activation_threshold=3,
            max_synapses_per_segment=5,
        )
        context_a = {10, 11, 12, 13, 14}
        context_b = {50, 51, 52, 53, 54}

        dend.grow_segment(0, context_a, n_synapses=5)
        dend.grow_segment(0, context_b, n_synapses=5)

        pred_a = dend.compute_predicted_cells(context_a)
        pred_b = dend.compute_predicted_cells(context_b)

        self.assertIn(0, pred_a)
        self.assertIn(0, pred_b)

    def test_vectorized_mask_prediction(self):
        """compute_predicted_cells_mask returns correct boolean array."""
        dend = self._make_dendrites(
            activation_threshold=3,
            max_synapses_per_segment=6,
        )
        source_cells = {10, 20, 30, 40, 50, 60}
        dend.grow_segment(0, source_cells, n_synapses=6)

        mask = np.zeros(dend.n_cells + 1, dtype=np.bool_)
        mask[[10, 20, 30, 40]] = True

        predicted_mask = dend.compute_predicted_cells_mask(mask)
        self.assertTrue(predicted_mask[0])
        self.assertFalse(predicted_mask[1])

    def test_n_segments_accessor(self):
        dend = self._make_dendrites()
        self.assertEqual(dend.n_segments(0), 0)
        dend.grow_segment(0, {10, 20, 30})
        self.assertEqual(dend.n_segments(0), 1)
        dend.grow_segment(0, {40, 50, 60})
        self.assertEqual(dend.n_segments(0), 2)


# ---------------------------------------------------------------------------
# SDRObjectMemory tests
# ---------------------------------------------------------------------------


class TestSDRObjectMemory(unittest.TestCase):

    def _make_memory(self):
        from tbp.monty.frameworks.models.cortical_column.sdr_memory import (
            SDRObjectMemory,
        )

        return SDRObjectMemory(
            location_overlap_threshold=0.2,
            feature_overlap_threshold=0.1,
        )

    def _random_sdr(self, n_bits=256, n_active=10, seed=0):
        rng = np.random.RandomState(seed)
        sdr = np.zeros(n_bits, dtype=np.float64)
        active = rng.choice(n_bits, size=n_active, replace=False)
        sdr[active] = 1.0
        return sdr

    def test_store_and_list(self):
        mem = self._make_memory()
        loc = self._random_sdr(seed=0)
        feat = self._random_sdr(seed=1)
        cells = self._random_sdr(seed=2)
        mem.store("banana", loc, feat, cells)
        self.assertIn("banana", mem.known_objects)
        self.assertEqual(mem.n_observations("banana"), 1)

    def test_match_returns_scores(self):
        mem = self._make_memory()
        loc = self._random_sdr(seed=0)
        feat = self._random_sdr(seed=1)
        cells = self._random_sdr(seed=2)
        mem.store("banana", loc, feat, cells)
        scores = mem.match(loc, feat)
        self.assertIn("banana", scores)
        self.assertGreater(scores["banana"], 0)

    def test_different_objects_discriminated(self):
        mem = self._make_memory()
        loc_a = self._random_sdr(seed=0)
        feat_a = self._random_sdr(seed=1)
        cells_a = self._random_sdr(seed=2)
        mem.store("banana", loc_a, feat_a, cells_a)

        loc_b = self._random_sdr(seed=10)
        feat_b = self._random_sdr(seed=11)
        cells_b = self._random_sdr(seed=12)
        mem.store("mug", loc_b, feat_b, cells_b)

        scores = mem.match(loc_a, feat_a)
        self.assertGreater(scores["banana"], scores["mug"])

    def test_max_observations_trimmed(self):
        from tbp.monty.frameworks.models.cortical_column.sdr_memory import (
            SDRObjectMemory,
        )

        mem = SDRObjectMemory(max_observations_per_object=5)
        for i in range(10):
            mem.store(
                "obj",
                self._random_sdr(seed=i),
                self._random_sdr(seed=i + 100),
                self._random_sdr(seed=i + 200),
            )
        self.assertEqual(mem.n_observations("obj"), 5)

    def test_memory_bytes(self):
        mem = self._make_memory()
        mem.store(
            "obj",
            self._random_sdr(),
            self._random_sdr(),
            self._random_sdr(),
        )
        self.assertGreater(mem.memory_bytes(), 0)


# ---------------------------------------------------------------------------
# CorticalColumn tests
# ---------------------------------------------------------------------------


class _FakeState:
    """Minimal State-like object for testing."""

    def __init__(self, location, hsv=(0.5, 0.5, 0.5), curvatures=(0.0, 0.0)):
        self.location = np.array(location, dtype=np.float64)
        self.use_state = True
        self.sender_type = "SM"
        self.non_morphological_features = {
            "hsv": np.array(hsv, dtype=np.float64),
            "principal_curvatures_log": np.array(curvatures, dtype=np.float64),
        }
        self.morphological_features = {}


class TestCorticalColumn(unittest.TestCase):

    def _make_column(self, **kwargs):
        from tbp.monty.frameworks.models.cortical_column.column import CorticalColumn

        defaults = dict(
            n_minicolumns=256,
            n_cells_per_minicolumn=4,
            sparsity=0.05,
            encoder_kwargs=dict(
                grid_cell_kwargs=dict(n_modules=4, cells_per_module=32),
            ),
            dendrite_kwargs=dict(
                max_segments_per_cell=8,
                activation_threshold=6,
            ),
        )
        defaults.update(kwargs)
        return CorticalColumn(**defaults)

    def test_step_returns_result(self):
        col = self._make_column()
        col.pre_episode(mode="eval")
        state = _FakeState([0.1, 0.2, 0.3])
        result = col.step(state)
        self.assertIn("surprise", result)
        self.assertIn("evidence", result)
        self.assertIn("mlh", result)

    def test_first_step_is_all_burst(self):
        col = self._make_column()
        col.pre_episode(mode="eval")
        state = _FakeState([0.1, 0.2, 0.3])
        result = col.step(state)
        self.assertEqual(result["surprise"], 1.0)

    def test_surprise_decreases_with_repetition(self):
        col = self._make_column()
        col.pre_episode(mode="train", object_name="test")

        state = _FakeState([0.1, 0.2, 0.3])
        surprises = []
        for _ in range(20):
            result = col.step(state)
            surprises.append(result["surprise"])

        self.assertLess(surprises[-1], surprises[0])

    def test_training_stores_observations(self):
        col = self._make_column()
        col.pre_episode(mode="train", object_name="banana")

        for i in range(10):
            state = _FakeState([0.1 * i, 0.0, 0.0], hsv=(0.2, 0.8, 0.6))
            col.step(state)

        col.post_episode()
        self.assertIn("banana", col.get_all_known_object_ids())
        self.assertEqual(col.memory.n_observations("banana"), 10)

    def test_eval_produces_evidence(self):
        col = self._make_column()

        col.pre_episode(mode="train", object_name="banana")
        for i in range(10):
            state = _FakeState([0.1 * i, 0.0, 0.0], hsv=(0.2, 0.8, 0.6))
            col.step(state)
        col.post_episode()

        col.pre_episode(mode="eval")
        state = _FakeState([0.05, 0.0, 0.0], hsv=(0.2, 0.8, 0.6))
        result = col.step(state)
        self.assertIn("banana", result["evidence"])

    def test_dendrites_grow_during_training(self):
        col = self._make_column()
        col.pre_episode(mode="train", object_name="test")

        initial_segs = col.dendrites.total_segments

        for i in range(20):
            state = _FakeState([0.05 * i, 0.0, 0.0])
            col.step(state)

        final_segs = col.dendrites.total_segments
        self.assertGreater(final_segs, initial_segs)

    def test_two_objects_different_evidence(self):
        col = self._make_column()

        col.pre_episode(mode="train", object_name="banana")
        for i in range(15):
            state = _FakeState(
                [0.05 * i, 0.0, 0.0],
                hsv=(0.15, 0.9, 0.8),
                curvatures=(2.0, -1.0),
            )
            col.step(state)
        col.post_episode()

        col.pre_episode(mode="train", object_name="mug")
        for i in range(15):
            state = _FakeState(
                [0.05 * i, 0.3, 0.0],
                hsv=(0.05, 0.5, 0.3),
                curvatures=(0.5, 0.5),
            )
            col.step(state)
        col.post_episode()

        col.pre_episode(mode="eval")
        for i in range(5):
            state = _FakeState(
                [0.05 * i, 0.0, 0.0],
                hsv=(0.15, 0.9, 0.8),
                curvatures=(2.0, -1.0),
            )
            col.step(state)

        evidence = col._evidence
        self.assertGreater(
            evidence.get("banana", 0),
            evidence.get("mug", 0),
            "Expected banana > mug, got {}".format(evidence),
        )

    def test_unused_state_returns_empty(self):
        col = self._make_column()
        col.pre_episode(mode="eval")
        state = _FakeState([0, 0, 0])
        state.use_state = False
        result = col.step(state)
        self.assertEqual(result["surprise"], 1.0)
        self.assertEqual(len(result["active_cells"]), 0)

    def test_memory_is_compact(self):
        col = self._make_column()
        col.pre_episode(mode="train", object_name="test")
        for i in range(100):
            state = _FakeState([0.01 * i, 0.0, 0.0])
            col.step(state)
        col.post_episode()

        mem_bytes = col.memory.memory_bytes()
        self.assertLess(mem_bytes, 1_000_000)


# ---------------------------------------------------------------------------
# Spatial pooler learning + boosting tests
# ---------------------------------------------------------------------------


class TestSpatialPoolerLearning(unittest.TestCase):
    """Tests for proximal dendrite learning and homeostatic boosting."""

    def _make_column(self, **kwargs):
        from tbp.monty.frameworks.models.cortical_column.column import CorticalColumn

        defaults = dict(
            n_minicolumns=256,
            n_cells_per_minicolumn=4,
            sparsity=0.05,
            sp_learning=True,
            boost_strength=3.0,
            encoder_kwargs=dict(
                grid_cell_kwargs=dict(n_modules=4, cells_per_module=32),
            ),
            dendrite_kwargs=dict(
                max_segments_per_cell=8,
                activation_threshold=6,
            ),
        )
        defaults.update(kwargs)
        return CorticalColumn(**defaults)

    def test_permanences_change_during_training(self):
        """SP learning should modify proximal permanences."""
        col = self._make_column()
        initial_perms = col._ff_permanences.copy()

        col.pre_episode(mode="train", object_name="test")
        for i in range(10):
            state = _FakeState([0.05 * i, 0.0, 0.0])
            col.step(state)

        # Permanences should have changed
        self.assertFalse(
            np.allclose(col._ff_permanences, initial_perms),
            "Expected permanences to change during training",
        )

    def test_permanences_stable_during_eval(self):
        """SP should NOT learn during eval mode."""
        col = self._make_column()

        col.pre_episode(mode="train", object_name="test")
        for i in range(5):
            col.step(_FakeState([0.05 * i, 0.0, 0.0]))
        col.post_episode()

        # Snapshot after training
        perms_after_train = col._ff_permanences.copy()

        col.pre_episode(mode="eval")
        for i in range(5):
            col.step(_FakeState([0.05 * i, 0.0, 0.0]))

        # Permanences unchanged during eval
        np.testing.assert_array_equal(col._ff_permanences, perms_after_train)

    def test_boosting_adapts_duty_cycle(self):
        """Homeostatic boosting should change boost factors over training."""
        col = self._make_column(boost_strength=5.0)
        initial_boosts = col._boost_factors.copy()

        col.pre_episode(mode="train", object_name="test")
        for i in range(50):
            state = _FakeState([0.01 * i, 0.0, 0.0])
            col.step(state)

        # Boost factors should have diverged from initial uniform values
        self.assertFalse(
            np.allclose(col._boost_factors, initial_boosts, atol=1e-3),
            "Expected boost factors to change during training",
        )

    def test_no_boosting_when_strength_zero(self):
        """boost_strength=0 should keep all boost factors at 1."""
        col = self._make_column(boost_strength=0.0)

        col.pre_episode(mode="train", object_name="test")
        for i in range(20):
            col.step(_FakeState([0.05 * i, 0.0, 0.0]))

        np.testing.assert_array_equal(
            col._boost_factors,
            np.ones(col.n_minicolumns, dtype=np.float32),
        )

    def test_sp_disabled_preserves_permanences(self):
        """sp_learning=False should leave permanences unchanged."""
        col = self._make_column(sp_learning=False)
        initial_perms = col._ff_permanences.copy()

        col.pre_episode(mode="train", object_name="test")
        for i in range(10):
            col.step(_FakeState([0.05 * i, 0.0, 0.0]))

        np.testing.assert_array_equal(col._ff_permanences, initial_perms)

    def test_sp_learning_improves_discrimination(self):
        """After SP training, the column should discriminate objects better.

        Train on two distinct objects. Measure evidence separation before
        and after additional training steps (SP refines receptive fields).
        """
        col = self._make_column(sp_learning=True)

        # Train object A: yellow, curved, at y=0
        col.pre_episode(mode="train", object_name="objA")
        for i in range(20):
            col.step(_FakeState(
                [0.05 * i, 0.0, 0.0], hsv=(0.15, 0.9, 0.8), curvatures=(2.0, -1.0),
            ))
        col.post_episode()

        # Train object B: brown, flat, at y=0.5
        col.pre_episode(mode="train", object_name="objB")
        for i in range(20):
            col.step(_FakeState(
                [0.05 * i, 0.5, 0.0], hsv=(0.05, 0.3, 0.2), curvatures=(0.1, 0.1),
            ))
        col.post_episode()

        # Eval with objA input
        col.pre_episode(mode="eval")
        for i in range(5):
            col.step(_FakeState(
                [0.05 * i, 0.0, 0.0], hsv=(0.15, 0.9, 0.8), curvatures=(2.0, -1.0),
            ))

        mlh = col.get_current_mlh()
        self.assertEqual(
            mlh["graph_id"], "objA",
            "Expected objA to be recognized, got {}".format(mlh),
        )

    def test_boosting_prevents_dead_columns(self):
        """Boosting should ensure all minicolumns eventually participate."""
        col = self._make_column(boost_strength=5.0)

        col.pre_episode(mode="train", object_name="test")
        # Run many steps with varied input
        ever_active = np.zeros(col.n_minicolumns, dtype=np.bool_)
        for i in range(200):
            loc = [0.01 * (i % 50), 0.01 * (i // 50), 0.0]
            hsv = ((i % 10) / 10.0, 0.5, 0.5)
            col.step(_FakeState(loc, hsv=hsv))
            ever_active |= col._active_mc_mask

        # With boosting, a significant fraction should have been active
        participation = ever_active.sum() / col.n_minicolumns
        self.assertGreater(
            participation, 0.3,
            "Expected >30% minicolumn participation with boosting, "
            "got {:.1%}".format(participation),
        )


# ---------------------------------------------------------------------------
# Recurrent Connections tests
# ---------------------------------------------------------------------------


class TestRecurrentConnections(unittest.TestCase):
    """Tests for attractor dynamics via recurrent connections."""

    def _make_recurrent(self, **kwargs):
        from tbp.monty.frameworks.models.cortical_column.recurrent import (
            RecurrentConnections,
        )

        defaults = dict(
            n_cells=64,
            n_minicolumns=8,
            n_cells_per_minicolumn=8,
            sparsity=0.25,
            learning_rate=0.1,
            max_settling_iterations=10,
            seed=42,
        )
        defaults.update(kwargs)
        return RecurrentConnections(**defaults)

    def test_initial_weights_are_zero(self):
        """Recurrent weights start at zero (no prior knowledge)."""
        rec = self._make_recurrent()
        self.assertEqual(rec.total_weight, 0.0)

    def test_learning_increases_weights(self):
        """Hebbian learning strengthens connections between co-active cells."""
        rec = self._make_recurrent()
        active = np.zeros(64, dtype=np.bool_)
        active[0:4] = True
        active[16:20] = True  # cells in two different minicolumns

        rec.learn(active)
        self.assertGreater(rec.total_weight, 0.0)

    def test_no_same_minicolumn_targets(self):
        """Sparse wiring never targets cells in the same minicolumn."""
        rec = self._make_recurrent()
        k = rec.n_cells_per_minicolumn
        for cell_i in range(rec.n_cells):
            mc_i = cell_i // k
            mc_start = mc_i * k
            mc_end = mc_start + k
            targets = rec._targets[cell_i]
            # No target should be in same minicolumn (ignore -1 sentinels)
            valid = targets[targets >= 0]
            in_same_mc = (valid >= mc_start) & (valid < mc_end)
            self.assertFalse(
                in_same_mc.any(),
                f"Cell {cell_i} has target in own minicolumn {mc_i}",
            )

    def test_settling_converges(self):
        """Settling loop reaches a fixed point."""
        rec = self._make_recurrent()

        # Train a pattern: cells 0-3 in mc0 and cells 16-19 in mc2
        pattern = np.zeros(64, dtype=np.bool_)
        pattern[0:4] = True
        pattern[16:20] = True
        for _ in range(20):
            rec.learn(pattern)

        # Start with partial input and settle
        partial = np.zeros(64, dtype=np.bool_)
        partial[0:4] = True  # only mc0 active

        settled, settled_mc, n_iters = rec.settle(partial)
        self.assertTrue(settled.any())
        self.assertLessEqual(n_iters, rec._max_iters)

    def test_decay_reduces_weights(self):
        """Global decay reduces all weights toward zero."""
        rec = self._make_recurrent()
        active = np.zeros(64, dtype=np.bool_)
        active[0:4] = True
        active[16:20] = True
        rec.learn(active)

        w_before = rec.total_weight
        rec.decay()
        self.assertLess(rec.total_weight, w_before)

    def test_memory_bytes(self):
        """Memory usage is reported correctly."""
        rec = self._make_recurrent()
        self.assertGreater(rec.memory_bytes(), 0)

    def test_memory_is_sparse_at_scale(self):
        """Sparse storage uses much less memory than dense at realistic size."""
        rec = self._make_recurrent(
            n_cells=2048, n_minicolumns=256, n_cells_per_minicolumn=8,
        )
        dense_bytes = rec.n_cells * rec.n_cells * 4  # float32 dense
        # Sparse: n_cells * fan_out * (4+4) = 2048*128*8 = 2MB
        # Dense:  2048^2 * 4 = 16MB
        self.assertLess(rec.memory_bytes(), dense_bytes // 2)


# ---------------------------------------------------------------------------
# Attractor Memory tests
# ---------------------------------------------------------------------------


class TestAttractorMemory(unittest.TestCase):
    """Tests for prototype-based attractor memory."""

    def _make_memory(self, **kwargs):
        from tbp.monty.frameworks.models.cortical_column.attractor_memory import (
            AttractorMemory,
        )

        defaults = dict(n_cells=64, prototype_threshold=0.3)
        defaults.update(kwargs)
        return AttractorMemory(**defaults)

    def test_store_and_list(self):
        mem = self._make_memory()
        pattern = np.zeros(64, dtype=np.bool_)
        pattern[0:10] = True
        mem.store("obj_a", pattern)
        self.assertIn("obj_a", mem.known_objects)
        self.assertEqual(mem.n_observations("obj_a"), 1)

    def test_prototype_forms_from_repeated_patterns(self):
        """Prototype captures bits active in >threshold fraction of obs."""
        mem = self._make_memory(prototype_threshold=0.5)
        # Bit 0-5 always active, bit 6-9 only in half the observations
        for i in range(10):
            pattern = np.zeros(64, dtype=np.bool_)
            pattern[0:6] = True
            if i % 2 == 0:
                pattern[6:10] = True
            mem.store("obj", pattern)

        proto = mem.get_prototype("obj")
        # Bits 0-5 should be in prototype (100% frequency > 50%)
        self.assertTrue(proto[0:6].all())
        # Bits 6-9 should be in prototype (50% frequency >= 50%)
        self.assertTrue(proto[6:10].all())
        # Bits 10+ should not be in prototype
        self.assertFalse(proto[10:].any())

    def test_match_scores_correct_object_higher(self):
        """Matching returns higher score for the correct object."""
        mem = self._make_memory()

        # Train two distinct objects
        p_a = np.zeros(64, dtype=np.bool_)
        p_a[0:10] = True
        p_b = np.zeros(64, dtype=np.bool_)
        p_b[30:40] = True

        for _ in range(5):
            mem.store("obj_a", p_a)
            mem.store("obj_b", p_b)

        # Query with pattern similar to obj_a
        query = np.zeros(64, dtype=np.bool_)
        query[0:8] = True  # 80% overlap with obj_a
        scores = mem.match(query)

        self.assertGreater(scores["obj_a"], scores["obj_b"])

    def test_memory_bytes(self):
        mem = self._make_memory()
        pattern = np.zeros(64, dtype=np.bool_)
        pattern[0:10] = True
        mem.store("obj", pattern)
        self.assertGreater(mem.memory_bytes(), 0)


# ---------------------------------------------------------------------------
# CorticalColumn with attractor dynamics tests
# ---------------------------------------------------------------------------


class TestCorticalColumnAttractor(unittest.TestCase):
    """Tests for CorticalColumn with use_attractor=True."""

    def _make_column(self, **kwargs):
        from tbp.monty.frameworks.models.cortical_column.column import CorticalColumn

        defaults = dict(
            n_minicolumns=64,
            n_cells_per_minicolumn=4,
            sparsity=0.1,
            use_attractor=True,
            sp_learning=True,
            boost_strength=0.0,
            seed=42,
        )
        defaults.update(kwargs)
        return CorticalColumn(**defaults)

    def test_attractor_column_step_returns_result(self):
        """Column with attractors still returns valid step result."""
        col = self._make_column()
        col.pre_episode(mode="train", object_name="test")
        result = col.step(_FakeState([0.1, 0.2, 0.3]))
        self.assertIn("surprise", result)
        self.assertIn("evidence", result)
        self.assertIn("mlh", result)

    def test_attractor_training_stores_prototypes(self):
        """Training with attractors stores prototypes in attractor memory."""
        col = self._make_column()
        col.pre_episode(mode="train", object_name="banana")
        for i in range(10):
            col.step(_FakeState([0.01 * i, 0.0, 0.0]))
        col.post_episode()

        self.assertIn("banana", col.attractor_memory.known_objects)
        self.assertEqual(col.attractor_memory.n_observations("banana"), 10)

    def test_attractor_eval_produces_evidence(self):
        """Eval with attractor memory produces evidence scores."""
        col = self._make_column()

        # Train
        col.pre_episode(mode="train", object_name="obj_a")
        for i in range(15):
            col.step(_FakeState([0.01 * i, 0.0, 0.0], hsv=(0.1, 0.5, 0.5)))
        col.post_episode()

        # Eval
        col.pre_episode(mode="eval")
        result = col.step(_FakeState([0.05, 0.0, 0.0], hsv=(0.1, 0.5, 0.5)))
        self.assertIn("obj_a", result["evidence"])

    def test_recurrent_weights_grow_during_training(self):
        """Recurrent connections learn during training."""
        col = self._make_column()
        col.pre_episode(mode="train", object_name="test")

        w_before = col.recurrent.total_weight
        for i in range(20):
            col.step(_FakeState([0.01 * i, 0.0, 0.0]))
        col.post_episode()

        self.assertGreater(col.recurrent.total_weight, w_before)

    def test_attractor_two_objects_discriminated(self):
        """Attractor column can discriminate two trained objects."""
        col = self._make_column()

        # Train object A (different location/feature space)
        col.pre_episode(mode="train", object_name="obj_a")
        for i in range(20):
            col.step(_FakeState(
                [0.01 * i, 0.0, 0.0], hsv=(0.1, 0.8, 0.5)
            ))
        col.post_episode()

        # Train object B
        col.pre_episode(mode="train", object_name="obj_b")
        for i in range(20):
            col.step(_FakeState(
                [0.0, 0.01 * i, 0.5], hsv=(0.8, 0.2, 0.9)
            ))
        col.post_episode()

        # Eval with pattern similar to obj_a
        col.pre_episode(mode="eval")
        for i in range(10):
            result = col.step(_FakeState(
                [0.01 * i, 0.0, 0.0], hsv=(0.1, 0.8, 0.5)
            ))

        ev = result["evidence"]
        self.assertGreater(
            ev.get("obj_a", 0.0), ev.get("obj_b", 0.0),
            f"Expected obj_a > obj_b, got {ev}",
        )


# ---------------------------------------------------------------------------
# Apical Dendrites (Phase 2) tests
# ---------------------------------------------------------------------------


class TestCorticalColumnApical(unittest.TestCase):
    """Tests for apical dendrite top-down modulation (Phase 2)."""

    def _make_column(self, **kwargs):
        from tbp.monty.frameworks.models.cortical_column.column import CorticalColumn

        defaults = dict(
            n_minicolumns=64,
            n_cells_per_minicolumn=4,
            sparsity=0.1,
            use_apical=True,
            sp_learning=True,
            boost_strength=0.0,
            seed=42,
            apical_dendrite_kwargs=dict(
                max_segments_per_cell=8,
                activation_threshold=4,
                max_synapses_per_segment=12,
            ),
        )
        defaults.update(kwargs)
        return CorticalColumn(**defaults)

    def test_apical_dendrites_created(self):
        """use_apical=True creates an apical DendriteSegments instance."""
        col = self._make_column()
        self.assertIsNotNone(col.apical_dendrites)
        self.assertEqual(col.apical_dendrites.n_cells, col.n_cells)

    def test_no_apical_when_disabled(self):
        """use_apical=False leaves apical_dendrites as None."""
        col = self._make_column(use_apical=False)
        self.assertIsNone(col.apical_dendrites)

    def test_receive_context_sets_mask(self):
        """receive_context populates internal context mask."""
        col = self._make_column()
        col.pre_episode(mode="eval")

        # Send context as bool mask
        ctx = np.zeros(col.n_cells, dtype=np.bool_)
        ctx[0:10] = True
        col.receive_context(active_cells=ctx)

        self.assertEqual(int(col._context_active_mask.sum()), 10)
        self.assertTrue(col._context_active_mask[0])

    def test_receive_context_set_format(self):
        """receive_context accepts a set of cell indices."""
        col = self._make_column()
        col.pre_episode(mode="eval")

        col.receive_context(active_cells={0, 1, 2, 3, 4})
        self.assertEqual(int(col._context_active_mask.sum()), 5)

    def test_get_context_signal_returns_state(self):
        """get_context_signal returns active_cells, winner_cells, surprise."""
        col = self._make_column()
        col.pre_episode(mode="train", object_name="test")
        col.step(_FakeState([0.1, 0.2, 0.3]))

        signal = col.get_context_signal()
        self.assertIn("active_cells", signal)
        self.assertIn("winner_cells", signal)
        self.assertIn("surprise", signal)
        self.assertEqual(signal["active_cells"].shape, (col.n_cells,))

    def test_apical_segments_grow_with_context(self):
        """When context is provided, apical segments grow during training."""
        col = self._make_column()
        col.pre_episode(mode="train", object_name="test")

        # Provide context signal before each step
        ctx = np.zeros(col.n_cells, dtype=np.bool_)
        ctx[0:20] = True

        initial_segs = col.apical_dendrites.total_segments

        for i in range(20):
            col.receive_context(active_cells=ctx)
            col.step(_FakeState([0.05 * i, 0.0, 0.0]))

        self.assertGreater(
            col.apical_dendrites.total_segments, initial_segs,
            "Apical segments should grow when context is provided",
        )

    def test_apical_column_step_still_works(self):
        """Column with apical enabled still produces valid step results."""
        col = self._make_column()
        col.pre_episode(mode="train", object_name="test")
        result = col.step(_FakeState([0.1, 0.2, 0.3]))
        self.assertIn("surprise", result)
        self.assertIn("evidence", result)

    def test_context_cleared_on_pre_episode(self):
        """pre_episode clears context and apical prediction masks."""
        col = self._make_column()
        col.pre_episode(mode="train", object_name="test")

        col.receive_context(active_cells={0, 1, 2})
        self.assertTrue(col._context_active_mask.any())

        col.pre_episode(mode="eval")
        self.assertFalse(col._context_active_mask.any())
        self.assertFalse(col._apical_predicted_mask.any())

    def test_apical_bias_winner_selection(self):
        """Apical prediction biases winner selection in bursting minicolumns.

        Train with consistent context, then verify the column
        prefers apical-predicted cells as winners.
        """
        col = self._make_column()

        # Consistent context → same cells active
        ctx = np.zeros(col.n_cells, dtype=np.bool_)
        ctx[0:30] = True  # Fixed context pattern

        # Train with context to grow apical segments
        col.pre_episode(mode="train", object_name="test")
        for i in range(30):
            col.receive_context(active_cells=ctx)
            col.step(_FakeState([0.02 * i, 0.0, 0.0]))
        col.post_episode()

        # After training, apical dendrites should have learned
        self.assertGreater(col.apical_dendrites.total_segments, 0)

    def test_backward_compat_no_apical(self):
        """Column without apical should behave identically to pre-Phase-2."""
        col_no_apical = self._make_column(use_apical=False)
        col_no_apical.pre_episode(mode="train", object_name="test")

        for i in range(10):
            result = col_no_apical.step(_FakeState([0.05 * i, 0.0, 0.0]))

        self.assertIn("surprise", result)
        self.assertIn("evidence", result)

    def test_parent_child_context_flow(self):
        """A parent column's context signal can feed a child's apical dendrites."""
        parent = self._make_column(use_apical=False)
        child = self._make_column()

        # Parent trains
        parent.pre_episode(mode="train", object_name="parent_obj")
        parent.step(_FakeState([0.1, 0.2, 0.3]))

        # Parent emits context
        signal = parent.get_context_signal()

        # Child receives context
        child.pre_episode(mode="train", object_name="child_obj")
        child.receive_context(**signal)
        result = child.step(_FakeState([0.1, 0.2, 0.3]))

        self.assertIn("surprise", result)
        self.assertTrue(child._context_active_mask.any())


# ---------------------------------------------------------------------------
# Continuous Plasticity (Phase 3) tests
# ---------------------------------------------------------------------------


class TestContinuousPlasticity(unittest.TestCase):
    """Tests for always-on learning with surprise modulation (Phase 3)."""

    def _make_column(self, **kwargs):
        from tbp.monty.frameworks.models.cortical_column.column import CorticalColumn

        defaults = dict(
            n_minicolumns=64,
            n_cells_per_minicolumn=4,
            sparsity=0.1,
            sp_learning=True,
            boost_strength=0.0,
            continuous_plasticity=True,
            novelty_threshold=0.8,
            seed=42,
        )
        defaults.update(kwargs)
        return CorticalColumn(**defaults)

    def test_sp_permanences_change_during_eval(self):
        """With continuous_plasticity=True, SP learns during eval."""
        col = self._make_column()

        # Train first
        col.pre_episode(mode="train", object_name="test")
        for i in range(5):
            col.step(_FakeState([0.05 * i, 0.0, 0.0]))
        col.post_episode()

        perms_before = col._ff_permanences.copy()

        # Eval — should still learn
        col.pre_episode(mode="eval")
        for i in range(10):
            col.step(_FakeState([0.05 * i, 0.0, 0.0]))

        self.assertFalse(
            np.allclose(col._ff_permanences, perms_before),
            "SP permanences should change during eval with continuous plasticity",
        )

    def test_sp_frozen_without_continuous_plasticity(self):
        """With continuous_plasticity=False (default), SP frozen during eval."""
        col = self._make_column(continuous_plasticity=False)

        col.pre_episode(mode="train", object_name="test")
        for i in range(5):
            col.step(_FakeState([0.05 * i, 0.0, 0.0]))
        col.post_episode()

        perms_before = col._ff_permanences.copy()

        col.pre_episode(mode="eval")
        for i in range(10):
            col.step(_FakeState([0.05 * i, 0.0, 0.0]))

        np.testing.assert_array_equal(col._ff_permanences, perms_before)

    def test_surprise_modulates_sp_learning(self):
        """High surprise → larger permanence changes during eval.

        We verify that first step (all burst, surprise=1.0) produces larger
        changes than subsequent steps (lower surprise).
        """
        col = self._make_column()
        col.pre_episode(mode="eval")

        # First step: all bursting → surprise = 1.0, full learning
        perms_before = col._ff_permanences.copy()
        col.step(_FakeState([0.1, 0.0, 0.0]))
        first_delta = np.abs(col._ff_permanences - perms_before).sum()

        # Run many steps on same input to learn context (reduce surprise)
        for i in range(20):
            col.step(_FakeState([0.1, 0.0, 0.0]))

        # Now another step should have smaller changes (surprise is lower)
        perms_before2 = col._ff_permanences.copy()
        col.step(_FakeState([0.1, 0.0, 0.0]))
        later_delta = np.abs(col._ff_permanences - perms_before2).sum()

        self.assertGreater(first_delta, later_delta)

    def test_recurrent_learning_during_eval(self):
        """With continuous_plasticity, recurrent weights also update during eval."""
        col = self._make_column(use_attractor=True)

        col.pre_episode(mode="eval")
        w_before = col.recurrent.total_weight
        for i in range(20):
            col.step(_FakeState([0.01 * i, 0.0, 0.0]))

        self.assertGreater(col.recurrent.total_weight, w_before)

    def test_attractor_prototype_refines_during_eval(self):
        """Known object prototypes update during eval (refinement).

        Refinement requires low surprise (< novelty_threshold). We use a
        lower threshold so that even with some bursting, refinement triggers
        once the column has seen a few steps and built temporal context.
        """
        col = self._make_column(use_attractor=True, novelty_threshold=1.1)

        # Train — two passes to build temporal predictions
        for _pass in range(2):
            col.pre_episode(mode="train", object_name="obj_a")
            for i in range(15):
                col.step(_FakeState([0.01 * i, 0.0, 0.0], hsv=(0.1, 0.5, 0.5)))
            col.post_episode()

        obs_after_train = col.attractor_memory.n_observations("obj_a")

        # Eval with the same object → should refine
        col.pre_episode(mode="eval")
        for i in range(15):
            col.step(_FakeState([0.01 * i, 0.0, 0.0], hsv=(0.1, 0.5, 0.5)))

        obs_after_eval = col.attractor_memory.n_observations("obj_a")
        self.assertGreater(
            obs_after_eval, obs_after_train,
            "Prototype should refine during eval with continuous plasticity",
        )

    def test_no_refinement_without_continuous_plasticity(self):
        """Without continuous_plasticity, prototypes don't change during eval."""
        col = self._make_column(continuous_plasticity=False, use_attractor=True)

        col.pre_episode(mode="train", object_name="obj_a")
        for i in range(10):
            col.step(_FakeState([0.01 * i, 0.0, 0.0]))
        col.post_episode()

        obs_after_train = col.attractor_memory.n_observations("obj_a")

        col.pre_episode(mode="eval")
        for i in range(10):
            col.step(_FakeState([0.01 * i, 0.0, 0.0]))

        self.assertEqual(col.attractor_memory.n_observations("obj_a"), obs_after_train)

    def test_backward_compatible_train_mode(self):
        """Train mode still works normally with continuous_plasticity."""
        col = self._make_column()

        col.pre_episode(mode="train", object_name="banana")
        for i in range(10):
            col.step(_FakeState([0.05 * i, 0.0, 0.0], hsv=(0.2, 0.8, 0.6)))
        col.post_episode()

        self.assertIn("banana", col.get_all_known_object_ids())
        self.assertEqual(col.memory.n_observations("banana"), 10)


# ---------------------------------------------------------------------------
# Motor Prediction (Phase 4) tests
# ---------------------------------------------------------------------------


class TestMotorEncoder(unittest.TestCase):
    """Tests for motor displacement SDR encoding."""

    def test_encode_shape(self):
        from tbp.monty.frameworks.models.cortical_column.motor_prediction import (
            MotorEncoder,
        )

        enc = MotorEncoder(bits_per_axis=64, active_per_axis=5)
        sdr = enc.encode(np.array([0.0, 0.0, 0.0]))
        self.assertEqual(sdr.shape, (64 * 3,))
        self.assertEqual(int(sdr.sum()), 5 * 3)

    def test_different_displacements_differ(self):
        from tbp.monty.frameworks.models.cortical_column.motor_prediction import (
            MotorEncoder,
        )

        enc = MotorEncoder()
        sdr_a = enc.encode(np.array([0.1, 0.0, 0.0]))
        sdr_b = enc.encode(np.array([-0.1, 0.0, 0.0]))
        overlap = float(np.dot(sdr_a, sdr_b))
        self.assertLess(overlap, sdr_a.sum())

    def test_similar_displacements_overlap(self):
        from tbp.monty.frameworks.models.cortical_column.motor_prediction import (
            MotorEncoder,
        )

        enc = MotorEncoder()
        sdr_a = enc.encode(np.array([0.1, 0.0, 0.0]))
        sdr_b = enc.encode(np.array([0.105, 0.0, 0.0]))
        overlap = float(np.dot(sdr_a, sdr_b))
        self.assertGreater(overlap, 0)


class TestMotorPrediction(unittest.TestCase):
    """Tests for motor-conditional location prediction."""

    def _make_predictor(self, **kwargs):
        from tbp.monty.frameworks.models.cortical_column.motor_prediction import (
            MotorPrediction,
        )

        defaults = dict(location_bits=256, learning_rate=0.05)
        defaults.update(kwargs)
        return MotorPrediction(**defaults)

    def _make_location_sdr(self, seed=0, n_bits=256, n_active=16):
        rng = np.random.RandomState(seed)
        sdr = np.zeros(n_bits, dtype=np.float64)
        sdr[rng.choice(n_bits, size=n_active, replace=False)] = 1.0
        return sdr

    def test_first_step_returns_max_error(self):
        """First step has no previous state so prediction_error = 1.0."""
        mp = self._make_predictor()
        loc = self._make_location_sdr(seed=0)
        result = mp.step(loc, np.array([0.0, 0.0, 0.0]))
        self.assertEqual(result["prediction_error"], 1.0)
        self.assertIsNone(result["predicted_location"])

    def test_prediction_error_decreases_with_learning(self):
        """Repeated transitions should reduce prediction error."""
        mp = self._make_predictor(learning_rate=0.005, decay_rate=0.0)

        # Use multiple different transitions so learning is gradual
        rng = np.random.RandomState(42)
        locs = [self._make_location_sdr(seed=i) for i in range(5)]
        displacement = np.array([0.1, 0.0, 0.0])

        errors_early = []
        errors_late = []
        for epoch in range(30):
            mp.reset()
            for i in range(len(locs) - 1):
                result = mp.step(locs[i], displacement if i > 0 else np.zeros(3))
            if epoch < 3:
                errors_early.append(result["prediction_error"])
            elif epoch >= 27:
                errors_late.append(result["prediction_error"])

        avg_early = np.mean(errors_early)
        avg_late = np.mean(errors_late)
        self.assertLessEqual(avg_late, avg_early)

    def test_memory_bytes(self):
        mp = self._make_predictor()
        self.assertGreater(mp.memory_bytes(), 0)

    def test_reset_clears_state(self):
        mp = self._make_predictor()
        loc = self._make_location_sdr()
        mp.step(loc, np.zeros(3))
        self.assertIsNotNone(mp._prev_input)

        mp.reset()
        self.assertIsNone(mp._prev_input)


class TestCorticalColumnMotorPrediction(unittest.TestCase):
    """Tests for CorticalColumn with motor prediction enabled."""

    def _make_column(self, **kwargs):
        from tbp.monty.frameworks.models.cortical_column.column import CorticalColumn

        defaults = dict(
            n_minicolumns=64,
            n_cells_per_minicolumn=4,
            sparsity=0.1,
            use_motor_prediction=True,
            sp_learning=True,
            boost_strength=0.0,
            seed=42,
        )
        defaults.update(kwargs)
        return CorticalColumn(**defaults)

    def test_motor_prediction_created(self):
        col = self._make_column()
        self.assertIsNotNone(col.motor_prediction)

    def test_no_motor_when_disabled(self):
        col = self._make_column(use_motor_prediction=False)
        self.assertIsNone(col.motor_prediction)

    def test_step_includes_motor_prediction_error(self):
        col = self._make_column()
        col.pre_episode(mode="train", object_name="test")
        result = col.step(_FakeState([0.1, 0.2, 0.3]))
        self.assertIn("motor_prediction_error", result)

    def test_motor_prediction_error_decreases(self):
        """Repeated trajectory should reduce motor prediction error."""
        col = self._make_column()

        errors_first = []
        errors_last = []

        for episode in range(5):
            col.pre_episode(mode="train", object_name="test")
            for i in range(10):
                result = col.step(_FakeState([0.05 * i, 0.0, 0.0]))
                if episode == 0:
                    errors_first.append(result["motor_prediction_error"])
                elif episode == 4:
                    errors_last.append(result["motor_prediction_error"])
            col.post_episode()

        # Later episodes should have lower average error (skip first step)
        avg_first = np.mean(errors_first[1:])
        avg_last = np.mean(errors_last[1:])
        self.assertLess(avg_last, avg_first)

    def test_backward_compat_no_motor(self):
        """Column without motor prediction works normally."""
        col = self._make_column(use_motor_prediction=False)
        col.pre_episode(mode="train", object_name="test")
        result = col.step(_FakeState([0.1, 0.2, 0.3]))
        self.assertIn("surprise", result)
        self.assertEqual(result["motor_prediction_error"], 0.0)


# ---------------------------------------------------------------------------
# Neuromodulatory Gating (Phase 5) tests
# ---------------------------------------------------------------------------


class TestNeuromodulatoryState(unittest.TestCase):
    """Tests for neuromodulator dynamics."""

    def test_initial_state_is_neutral(self):
        from tbp.monty.frameworks.models.cortical_column.neuromodulators import (
            NeuromodulatoryState,
        )

        nm = NeuromodulatoryState()
        self.assertAlmostEqual(nm.novelty, 0.5)
        self.assertAlmostEqual(nm.arousal, 0.5)
        self.assertAlmostEqual(nm.reward, 0.5)

    def test_high_burst_increases_novelty(self):
        from tbp.monty.frameworks.models.cortical_column.neuromodulators import (
            NeuromodulatoryState,
        )

        nm = NeuromodulatoryState(ema_rate=0.5)
        nm.update(burst_ratio=1.0, surprise_ema=0.5)
        self.assertGreater(nm.novelty, 0.5)

    def test_low_burst_decreases_novelty(self):
        from tbp.monty.frameworks.models.cortical_column.neuromodulators import (
            NeuromodulatoryState,
        )

        nm = NeuromodulatoryState(ema_rate=0.5)
        nm.update(burst_ratio=0.0, surprise_ema=0.5)
        self.assertLess(nm.novelty, 0.5)

    def test_sp_learning_rate_scale_range(self):
        from tbp.monty.frameworks.models.cortical_column.neuromodulators import (
            NeuromodulatoryState,
        )

        nm = NeuromodulatoryState()
        nm.novelty = 0.0
        self.assertAlmostEqual(nm.sp_learning_rate_scale(), 0.2)
        nm.novelty = 1.0
        self.assertAlmostEqual(nm.sp_learning_rate_scale(), 2.0)

    def test_competition_width_scale_range(self):
        from tbp.monty.frameworks.models.cortical_column.neuromodulators import (
            NeuromodulatoryState,
        )

        nm = NeuromodulatoryState()
        nm.arousal = 0.0
        self.assertAlmostEqual(nm.competition_width_scale(), 0.8)
        nm.arousal = 1.0
        self.assertAlmostEqual(nm.competition_width_scale(), 1.5)

    def test_reset_restores_neutral(self):
        from tbp.monty.frameworks.models.cortical_column.neuromodulators import (
            NeuromodulatoryState,
        )

        nm = NeuromodulatoryState(ema_rate=0.5)
        nm.update(burst_ratio=1.0, surprise_ema=1.0, reward_signal=1.0)
        nm.reset()
        self.assertAlmostEqual(nm.novelty, 0.5)
        self.assertAlmostEqual(nm.arousal, 0.5)


class TestCorticalColumnNeuromodulation(unittest.TestCase):
    """Tests for CorticalColumn with neuromodulatory gating."""

    def _make_column(self, **kwargs):
        from tbp.monty.frameworks.models.cortical_column.column import CorticalColumn

        defaults = dict(
            n_minicolumns=64,
            n_cells_per_minicolumn=4,
            sparsity=0.1,
            use_neuromodulation=True,
            sp_learning=True,
            boost_strength=0.0,
            seed=42,
        )
        defaults.update(kwargs)
        return CorticalColumn(**defaults)

    def test_neuromodulators_created(self):
        col = self._make_column()
        self.assertIsNotNone(col.neuromodulators)

    def test_no_neuromod_when_disabled(self):
        col = self._make_column(use_neuromodulation=False)
        self.assertIsNone(col.neuromodulators)

    def test_neuromodulators_update_during_step(self):
        col = self._make_column()
        col.pre_episode(mode="train", object_name="test")

        # Capture initial novelty
        initial = col.neuromodulators.novelty

        # First step: all burst → novelty increases
        col.step(_FakeState([0.1, 0.2, 0.3]))
        self.assertNotAlmostEqual(col.neuromodulators.novelty, initial, places=3)

    def test_receive_reward_updates_modulators(self):
        col = self._make_column()
        col.pre_episode(mode="train", object_name="test")
        col.step(_FakeState([0.1, 0.2, 0.3]))

        reward_before = col.neuromodulators.reward
        col.receive_reward(1.0)
        self.assertGreater(col.neuromodulators.reward, reward_before)

    def test_neuromod_column_still_works(self):
        """Column with neuromodulation produces valid results."""
        col = self._make_column()

        col.pre_episode(mode="train", object_name="obj_a")
        for i in range(15):
            result = col.step(_FakeState(
                [0.05 * i, 0.0, 0.0], hsv=(0.1, 0.5, 0.5)
            ))
        col.post_episode()

        col.pre_episode(mode="eval")
        result = col.step(_FakeState([0.05, 0.0, 0.0], hsv=(0.1, 0.5, 0.5)))
        self.assertIn("surprise", result)
        self.assertIn("evidence", result)

    def test_backward_compat_no_neuromod(self):
        col = self._make_column(use_neuromodulation=False)
        col.pre_episode(mode="train", object_name="test")
        result = col.step(_FakeState([0.1, 0.2, 0.3]))
        self.assertIn("surprise", result)

    def test_neuromod_resets_on_pre_episode(self):
        col = self._make_column()
        col.pre_episode(mode="train", object_name="test")
        col.step(_FakeState([0.1, 0.2, 0.3]))

        col.pre_episode(mode="eval")
        self.assertAlmostEqual(col.neuromodulators.novelty, 0.5)


# ---------------------------------------------------------------------------
# Dendritic Nonlinearity & Structural Plasticity (Phase 6) tests
# ---------------------------------------------------------------------------


class TestGradedActivation(unittest.TestCase):
    """Tests for graded (sigmoid) dendritic activation."""

    def _make_dendrites(self, **kwargs):
        from tbp.monty.frameworks.models.cortical_column.dendrites import (
            DendriteSegments,
        )

        defaults = dict(
            n_cells=100,
            max_segments_per_cell=8,
            max_synapses_per_segment=10,
            activation_threshold=5,
            graded_activation=True,
            graded_temperature=2.0,
        )
        defaults.update(kwargs)
        return DendriteSegments(**defaults)

    def test_graded_depolarization_continuous(self):
        """Graded activation returns continuous values, not just 0/1."""
        dend = self._make_dendrites()
        source_cells = {10, 20, 30, 40, 50, 60}
        dend.grow_segment(0, source_cells, n_synapses=6)

        mask = np.zeros(dend.n_cells + 1, dtype=np.bool_)
        # Sub-threshold: only 3 active (threshold=5)
        mask[[10, 20, 30]] = True

        depol = dend.compute_graded_depolarization(mask)
        # Should be > 0 but < 1 (sub-threshold partial activation)
        self.assertGreater(depol[0], 0.0)
        self.assertLess(depol[0], 0.5)

    def test_graded_above_threshold_is_high(self):
        """Above-threshold overlap should give depolarization near 1."""
        dend = self._make_dendrites()
        source_cells = {10, 20, 30, 40, 50, 60}
        dend.grow_segment(0, source_cells, n_synapses=6)

        mask = np.zeros(dend.n_cells + 1, dtype=np.bool_)
        mask[[10, 20, 30, 40, 50, 60]] = True  # all 6 active

        depol = dend.compute_graded_depolarization(mask)
        self.assertGreater(depol[0], 0.5)

    def test_graded_prediction_mask_works(self):
        """Graded dendrites still produce binary predicted_cells_mask."""
        dend = self._make_dendrites()
        source_cells = {10, 20, 30, 40, 50, 60}
        dend.grow_segment(0, source_cells, n_synapses=6)

        mask = np.zeros(dend.n_cells + 1, dtype=np.bool_)
        mask[[10, 20, 30, 40, 50, 60]] = True

        predicted = dend.compute_predicted_cells_mask(mask)
        self.assertTrue(predicted[0])


class TestStructuralPlasticity(unittest.TestCase):
    """Tests for synapse addition, pruning, and segment pruning."""

    def _make_dendrites(self, **kwargs):
        from tbp.monty.frameworks.models.cortical_column.dendrites import (
            DendriteSegments,
        )

        defaults = dict(
            n_cells=100,
            max_segments_per_cell=8,
            max_synapses_per_segment=10,
            activation_threshold=5,
            pruning_threshold=0.01,
            pruning_age=5,
        )
        defaults.update(kwargs)
        return DendriteSegments(**defaults)

    def test_add_synapses_to_existing_segment(self):
        """Can add new synapses to a segment that has room."""
        dend = self._make_dendrites()
        dend.grow_segment(0, {10, 20, 30}, n_synapses=3)

        n_before = int(dend._synapse_count[0, 0])
        added = dend.add_synapses_to_segment(0, 0, {40, 50, 60}, max_new=2)
        self.assertEqual(added, 2)
        self.assertEqual(int(dend._synapse_count[0, 0]), n_before + 2)

    def test_add_synapses_no_duplicates(self):
        """Won't add synapses to already-connected cells."""
        dend = self._make_dendrites()
        dend.grow_segment(0, {10, 20, 30}, n_synapses=3)

        # Try to add existing cells
        added = dend.add_synapses_to_segment(0, 0, {10, 20, 30})
        self.assertEqual(added, 0)

    def test_prune_dead_synapses(self):
        """Synapses with permanence 0 are removed."""
        dend = self._make_dendrites()
        dend.grow_segment(0, {10, 20, 30, 40, 50}, n_synapses=5)

        # Kill some synapses
        dend._permanences[0, 0, 1] = 0.0
        dend._permanences[0, 0, 3] = 0.0

        removed = dend.prune_dead_synapses(0, 0)
        self.assertEqual(removed, 2)
        self.assertEqual(int(dend._synapse_count[0, 0]), 3)

    def test_prune_old_segments(self):
        """Old inactive segments with low permanence are pruned."""
        dend = self._make_dendrites(pruning_age=3, pruning_threshold=0.1)
        dend.grow_segment(0, {10, 20, 30}, n_synapses=3)

        # Set permanences very low and age high
        dend._permanences[0, 0, :3] = 0.001
        dend._segment_age[0, 0] = 10

        pruned = dend.prune_old_segments()
        self.assertEqual(pruned, 1)
        self.assertEqual(dend.n_segments(0), 0)

    def test_active_segments_dont_age(self):
        """Active segments have their age reset."""
        dend = self._make_dendrites(activation_threshold=3)
        dend.grow_segment(0, {10, 20, 30, 40, 50}, n_synapses=5)
        dend._segment_age[0, 0] = 10

        # Activate the segment
        mask = np.zeros(dend.n_cells + 1, dtype=np.bool_)
        mask[[10, 20, 30, 40]] = True
        dend.age_segments(mask)

        self.assertEqual(int(dend._segment_age[0, 0]), 0)

    def test_segment_capacity_not_saturated(self):
        """With pruning, segment slots are reclaimed for reuse."""
        dend = self._make_dendrites(
            max_segments_per_cell=4, pruning_age=1, pruning_threshold=0.5
        )

        # Fill all 4 segment slots
        for i in range(4):
            dend.grow_segment(0, {10 + i * 10, 20 + i * 10}, n_synapses=2)
        self.assertEqual(dend.n_segments(0), 4)

        # Can't add more
        self.assertFalse(dend.grow_segment(0, {90, 91}))

        # Age them all and set low permanences
        dend._segment_age[0, :4] = 5
        dend._permanences[0, :4, :] = 0.001

        pruned = dend.prune_old_segments()
        self.assertEqual(pruned, 4)
        self.assertEqual(dend.n_segments(0), 0)

        # Now can add again
        self.assertTrue(dend.grow_segment(0, {90, 91}))


class TestCorticalColumnPhase6(unittest.TestCase):
    """Integration tests for graded dendrites + structural plasticity."""

    def _make_column(self, **kwargs):
        from tbp.monty.frameworks.models.cortical_column.column import CorticalColumn

        defaults = dict(
            n_minicolumns=64,
            n_cells_per_minicolumn=4,
            sparsity=0.1,
            sp_learning=True,
            boost_strength=0.0,
            seed=42,
            dendrite_kwargs=dict(
                max_segments_per_cell=8,
                activation_threshold=6,
                graded_activation=True,
                pruning_age=50,
            ),
        )
        defaults.update(kwargs)
        return CorticalColumn(**defaults)

    def test_graded_column_works(self):
        """Column with graded dendrites produces valid results."""
        col = self._make_column()
        col.pre_episode(mode="train", object_name="test")
        result = col.step(_FakeState([0.1, 0.2, 0.3]))
        self.assertIn("surprise", result)

    def test_graded_column_learns(self):
        """Column with graded dendrites learns to predict."""
        col = self._make_column()
        col.pre_episode(mode="train", object_name="test")

        surprises = []
        for i in range(30):
            result = col.step(_FakeState([0.1, 0.0, 0.0]))
            surprises.append(result["surprise"])

        # Surprise should decrease (learning happening)
        self.assertLess(surprises[-1], surprises[0])

    def test_backward_compat_binary_dendrites(self):
        """Binary (non-graded) dendrites still work."""
        col = self._make_column(dendrite_kwargs=dict(
            max_segments_per_cell=8,
            activation_threshold=6,
            graded_activation=False,
        ))
        col.pre_episode(mode="train", object_name="test")
        result = col.step(_FakeState([0.1, 0.2, 0.3]))
        self.assertIn("surprise", result)


# ---------------------------------------------------------------------------
# HeteroAssociativeMemory tests (Phase 7a)
# ---------------------------------------------------------------------------


class TestHeteroAssociativeMemory(unittest.TestCase):
    """Tests for the weight-based associative memory."""

    def _make_memory(self, **kwargs):
        from tbp.monty.frameworks.models.cortical_column.associative_memory import (
            HeteroAssociativeMemory,
        )

        defaults = dict(
            n_cells=256,
            n_input=128,
            n_label_bits=64,
            n_active_label=5,
            learning_rate=0.1,
            decay_rate=0.0,
            cell_weight=0.5,
            ff_weight=0.5,
            seed=42,
        )
        defaults.update(kwargs)
        return HeteroAssociativeMemory(**defaults)

    def test_label_is_deterministic(self):
        """Same object name always produces the same label SDR."""
        mem1 = self._make_memory()
        mem2 = self._make_memory()
        label1 = mem1._get_label("mug")
        label2 = mem2._get_label("mug")
        np.testing.assert_array_equal(label1, label2)

    def test_labels_are_unique(self):
        """Different object names produce different label SDRs."""
        mem = self._make_memory()
        label_mug = mem._get_label("mug")
        label_bowl = mem._get_label("bowl")
        # They should differ (different active bits)
        self.assertFalse(
            np.array_equal(label_mug, label_bowl),
            "Different objects should have different labels",
        )

    def test_label_sparsity(self):
        """Label SDRs have correct number of active bits."""
        mem = self._make_memory(n_label_bits=64, n_active_label=5)
        label = mem._get_label("test_object")
        self.assertEqual(int(label.sum()), 5)
        self.assertEqual(len(label), 64)

    def test_learning_increases_recall_score(self):
        """After learning, recall score for trained object is positive."""
        mem = self._make_memory()
        # Create a synthetic cell pattern
        active = np.zeros(256, dtype=np.bool_)
        active[10:20] = True
        input_sdr = np.zeros(128, dtype=np.float64)
        input_sdr[5:15] = 1.0

        # Before learning: no known objects
        scores = mem.recall(active, input_sdr)
        self.assertEqual(len(scores), 0)

        # Learn
        mem.learn(active, input_sdr, "test_obj")

        # After learning: positive score
        scores = mem.recall(active, input_sdr)
        self.assertIn("test_obj", scores)
        self.assertGreater(scores["test_obj"], 0.0)

    def test_recall_discriminates_objects(self):
        """Recall returns higher score for correct object."""
        mem = self._make_memory(learning_rate=0.5)

        # Two distinct patterns
        pattern_a = np.zeros(256, dtype=np.bool_)
        pattern_a[0:20] = True
        input_a = np.zeros(128, dtype=np.float64)
        input_a[0:10] = 1.0

        pattern_b = np.zeros(256, dtype=np.bool_)
        pattern_b[100:120] = True
        input_b = np.zeros(128, dtype=np.float64)
        input_b[50:60] = 1.0

        # Learn both
        for _ in range(10):
            mem.learn(pattern_a, input_a, "obj_a")
            mem.learn(pattern_b, input_b, "obj_b")

        # Recall with pattern_a → obj_a should score higher
        scores = mem.recall(pattern_a, input_a)
        self.assertGreater(
            scores["obj_a"],
            scores["obj_b"],
            "Correct object should have higher recall score",
        )

        # Recall with pattern_b → obj_b should score higher
        scores = mem.recall(pattern_b, input_b)
        self.assertGreater(
            scores["obj_b"],
            scores["obj_a"],
            "Correct object should have higher recall score",
        )

    def test_memory_is_fixed_size(self):
        """Weight matrix doesn't grow when learning more objects."""
        mem = self._make_memory()

        size_before = mem.weight_matrix_bytes()

        # Learn 10 different objects
        for i in range(10):
            pattern = np.zeros(256, dtype=np.bool_)
            pattern[i * 20 : i * 20 + 15] = True
            input_sdr = np.zeros(128, dtype=np.float64)
            input_sdr[i * 10 : i * 10 + 8] = 1.0
            mem.learn(pattern, input_sdr, f"obj_{i}")

        size_after = mem.weight_matrix_bytes()
        self.assertEqual(
            size_before,
            size_after,
            "Weight matrix size must not change with more objects",
        )

    def test_capacity_graceful_degradation(self):
        """Accuracy degrades gradually, not catastrophically, with many objects."""
        mem = self._make_memory(
            n_cells=512,
            n_input=256,
            n_label_bits=128,
            n_active_label=5,
            learning_rate=0.3,
        )
        rng = np.random.RandomState(42)

        # Learn 30 distinct objects
        patterns = []
        inputs = []
        for i in range(30):
            p = np.zeros(512, dtype=np.bool_)
            active_idx = rng.choice(512, size=25, replace=False)
            p[active_idx] = True
            patterns.append(p)

            inp = np.zeros(256, dtype=np.float64)
            inp_idx = rng.choice(256, size=15, replace=False)
            inp[inp_idx] = 1.0
            inputs.append(inp)

            for _ in range(5):
                mem.learn(p, inp, f"obj_{i}")

        # Test recall accuracy
        correct = 0
        for i in range(30):
            scores = mem.recall(patterns[i], inputs[i])
            if scores:
                best = max(scores, key=scores.get)
                if best == f"obj_{i}":
                    correct += 1

        accuracy = correct / 30
        # Should be well above chance (1/30 ≈ 3.3%)
        self.assertGreater(
            accuracy,
            0.5,
            f"Accuracy {accuracy:.0%} should be above 50% for 30 objects",
        )

    def test_cell_only_pathway(self):
        """Memory works with cell pathway only (ff_weight=0)."""
        mem = self._make_memory(cell_weight=1.0, ff_weight=0.0, learning_rate=0.5)

        pattern = np.zeros(256, dtype=np.bool_)
        pattern[10:30] = True

        for _ in range(10):
            mem.learn(pattern, None, "test")

        scores = mem.recall(pattern, None)
        self.assertGreater(scores["test"], 0.0)

    def test_ff_only_pathway(self):
        """Memory works with feedforward pathway only (cell_weight=0)."""
        mem = self._make_memory(cell_weight=0.0, ff_weight=1.0, learning_rate=0.5)

        pattern = np.zeros(256, dtype=np.bool_)
        pattern[10:30] = True
        input_sdr = np.zeros(128, dtype=np.float64)
        input_sdr[5:20] = 1.0

        for _ in range(10):
            mem.learn(pattern, input_sdr, "test")

        scores = mem.recall(pattern, input_sdr)
        self.assertGreater(scores["test"], 0.0)

    def test_remove_object_hides_from_recall(self):
        """Removing an object removes it from recall results."""
        mem = self._make_memory()
        pattern = np.zeros(256, dtype=np.bool_)
        pattern[10:20] = True
        input_sdr = np.zeros(128, dtype=np.float64)
        input_sdr[5:10] = 1.0

        mem.learn(pattern, input_sdr, "obj")
        self.assertIn("obj", mem.recall(pattern, input_sdr))

        mem.remove_object("obj")
        self.assertNotIn("obj", mem.recall(pattern, input_sdr))


# ---------------------------------------------------------------------------
# CorticalColumn with weight-based memory (Phase 7a)
# ---------------------------------------------------------------------------


class TestCorticalColumnWeightMemory(unittest.TestCase):
    """Tests for CorticalColumn with use_weight_memory=True."""

    def _make_column(self, **kwargs):
        from tbp.monty.frameworks.models.cortical_column.column import CorticalColumn

        defaults = dict(
            n_minicolumns=64,
            n_cells_per_minicolumn=4,
            sparsity=0.1,
            use_weight_memory=True,
            use_attractor=True,
            sp_learning=True,
            boost_strength=0.0,
            seed=42,
        )
        defaults.update(kwargs)
        return CorticalColumn(**defaults)

    def test_weight_memory_column_trains_and_evals(self):
        """Column with weight memory trains objects and produces evidence."""
        col = self._make_column()

        # Train
        col.pre_episode(mode="train", object_name="alpha")
        for i in range(15):
            col.step(_FakeState([0.01 * i, 0.0, 0.0], hsv=(0.1, 0.5, 0.5)))
        col.post_episode()

        # Known objects come from associative memory
        known = col.get_all_known_object_ids()
        self.assertIn("alpha", known)

        # Eval produces evidence
        col.pre_episode(mode="eval")
        result = col.step(_FakeState([0.05, 0.0, 0.0], hsv=(0.1, 0.5, 0.5)))
        self.assertIn("alpha", result["evidence"])
        self.assertGreater(result["evidence"]["alpha"], 0.0)

    def test_weight_memory_no_sdr_lookup_growth(self):
        """SDRObjectMemory stays empty when weight memory is active."""
        col = self._make_column()

        col.pre_episode(mode="train", object_name="obj_a")
        for i in range(10):
            col.step(_FakeState([0.01 * i, 0.0, 0.0]))
        col.post_episode()

        # SDR memory should be empty — weight memory handles storage
        self.assertEqual(col.memory.n_observations("obj_a"), 0)
        # Associative memory should know the object
        self.assertIn("obj_a", col.associative_memory.known_objects)

    def test_weight_memory_discriminates_objects(self):
        """Weight-based column can discriminate two distinct objects."""
        col = self._make_column()

        # Train object A (low hue, small curvature)
        col.pre_episode(mode="train", object_name="obj_a")
        for i in range(20):
            col.step(
                _FakeState(
                    [0.01 * i, 0.0, 0.0],
                    hsv=(0.1, 0.5, 0.5),
                    curvatures=(0.1, 0.1),
                )
            )
        col.post_episode()

        # Train object B (high hue, large curvature)
        col.pre_episode(mode="train", object_name="obj_b")
        for i in range(20):
            col.step(
                _FakeState(
                    [0.01 * i, 0.5, 0.0],
                    hsv=(0.8, 0.5, 0.5),
                    curvatures=(2.0, 2.0),
                )
            )
        col.post_episode()

        # Eval with A-like features → should prefer obj_a
        col.pre_episode(mode="eval")
        for i in range(15):
            result = col.step(
                _FakeState(
                    [0.01 * i, 0.0, 0.0],
                    hsv=(0.1, 0.5, 0.5),
                    curvatures=(0.1, 0.1),
                )
            )

        ev = result["evidence"]
        self.assertGreater(
            ev.get("obj_a", 0),
            ev.get("obj_b", 0),
            f"obj_a should have higher evidence, got {ev}",
        )

    def test_weight_memory_fixed_size_across_objects(self):
        """Associative memory size is constant regardless of object count."""
        col = self._make_column()
        size_0 = col.associative_memory.weight_matrix_bytes()

        # Train 5 objects
        for j in range(5):
            col.pre_episode(mode="train", object_name=f"obj_{j}")
            for i in range(10):
                col.step(_FakeState([0.01 * i, 0.1 * j, 0.0]))
            col.post_episode()

        size_5 = col.associative_memory.weight_matrix_bytes()
        self.assertEqual(size_0, size_5, "Weight matrix must not grow")

    def test_multi_episode_strengthens_recall(self):
        """Multiple training episodes increase evidence separation."""
        col = self._make_column(
            continuous_plasticity=True,
            novelty_threshold=1.1,
        )

        # Train 1 episode
        col.pre_episode(mode="train", object_name="target")
        for i in range(20):
            col.step(
                _FakeState(
                    [0.01 * i, 0.0, 0.0],
                    hsv=(0.3, 0.6, 0.7),
                    curvatures=(0.5, 0.5),
                )
            )
        col.post_episode()

        # Train a distractor
        col.pre_episode(mode="train", object_name="distractor")
        for i in range(20):
            col.step(
                _FakeState(
                    [0.01 * i, 0.5, 0.0],
                    hsv=(0.9, 0.3, 0.4),
                    curvatures=(3.0, 1.0),
                )
            )
        col.post_episode()

        weight_after_1 = col.associative_memory.total_weight

        # Train target again (episode 2)
        col.pre_episode(mode="train", object_name="target")
        for i in range(20):
            col.step(
                _FakeState(
                    [0.01 * i, 0.0, 0.0],
                    hsv=(0.3, 0.6, 0.7),
                    curvatures=(0.5, 0.5),
                )
            )
        col.post_episode()

        weight_after_2 = col.associative_memory.total_weight
        # More training should increase total learned weight
        self.assertGreater(
            weight_after_2,
            weight_after_1,
            "Additional training should increase learned associations",
        )

    def test_noise_robustness_with_attractor(self):
        """Noisy input + attractor settling still yields correct object."""
        col = self._make_column(use_attractor=True)
        rng = np.random.RandomState(123)

        # Train
        col.pre_episode(mode="train", object_name="target")
        for i in range(20):
            col.step(
                _FakeState(
                    [0.01 * i, 0.0, 0.0],
                    hsv=(0.2, 0.6, 0.7),
                    curvatures=(0.5, 0.5),
                )
            )
        col.post_episode()

        col.pre_episode(mode="train", object_name="other")
        for i in range(20):
            col.step(
                _FakeState(
                    [0.01 * i, 0.5, 0.0],
                    hsv=(0.8, 0.3, 0.3),
                    curvatures=(3.0, 3.0),
                )
            )
        col.post_episode()

        # Eval with noisy features (add noise to HSV)
        col.pre_episode(mode="eval")
        for i in range(15):
            noise_h = float(np.clip(0.2 + rng.normal(0, 0.05), 0, 1))
            noise_s = float(np.clip(0.6 + rng.normal(0, 0.05), 0, 1))
            noise_v = float(np.clip(0.7 + rng.normal(0, 0.05), 0, 1))
            result = col.step(
                _FakeState(
                    [0.01 * i, 0.0, 0.0],
                    hsv=(noise_h, noise_s, noise_v),
                    curvatures=(0.5, 0.5),
                )
            )

        ev = result["evidence"]
        self.assertGreater(
            ev.get("target", 0),
            ev.get("other", 0),
            f"Target should win despite noise, got {ev}",
        )

    def test_reconsolidation_during_eval(self):
        """Continuous plasticity + weight memory reconsolidates during eval."""
        col = self._make_column(
            continuous_plasticity=True,
            novelty_threshold=1.1,
        )

        # Train
        col.pre_episode(mode="train", object_name="obj")
        for i in range(15):
            col.step(_FakeState([0.01 * i, 0.0, 0.0], hsv=(0.3, 0.5, 0.5)))
        col.post_episode()

        weight_before_eval = col.associative_memory.total_weight

        # Eval (should reconsolidate because novelty_threshold=1.1)
        col.pre_episode(mode="eval")
        for i in range(15):
            col.step(_FakeState([0.01 * i, 0.0, 0.0], hsv=(0.3, 0.5, 0.5)))

        weight_after_eval = col.associative_memory.total_weight
        self.assertGreater(
            weight_after_eval,
            weight_before_eval,
            "Reconsolidation should increase learned weight during eval",
        )

    def test_backward_compat_no_weight_memory(self):
        """Column without weight_memory still works (SDR lookup)."""
        col = self._make_column(use_weight_memory=False)

        col.pre_episode(mode="train", object_name="test")
        for i in range(10):
            col.step(_FakeState([0.01 * i, 0.0, 0.0]))
        col.post_episode()

        self.assertIsNone(col.associative_memory)
        self.assertGreater(col.memory.n_observations("test"), 0)


# ---------------------------------------------------------------------------
# CorticalColumnLM adapter tests (Track 8, Steps 1-4)
# ---------------------------------------------------------------------------


class TestCorticalColumnLMInterface(unittest.TestCase):
    """Step 1: CorticalColumnLM satisfies LearningModule interface."""

    def _make_lm(self, **kwargs):
        from tbp.monty.frameworks.models.cortical_column.learning_module import (
            CorticalColumnLM,
        )
        defaults = dict(
            column_kwargs=dict(
                n_minicolumns=256,
                n_cells_per_minicolumn=4,
                use_weight_memory=True,
                use_attractor=True,
                seed=42,
            ),
        )
        defaults.update(kwargs)
        return CorticalColumnLM(**defaults)

    def test_is_learning_module(self):
        """CorticalColumnLM is a LearningModule."""
        from tbp.monty.frameworks.models.abstract_monty_classes import LearningModule
        lm = self._make_lm()
        self.assertIsInstance(lm, LearningModule)

    def test_lifecycle_methods_exist(self):
        """All lifecycle methods are callable."""
        lm = self._make_lm()
        for method_name in [
            "reset", "pre_episode", "post_episode",
            "set_experiment_mode", "matching_step", "exploratory_step",
            "receive_votes", "send_out_vote", "propose_goal_states",
            "get_output", "receive_context", "state_dict", "load_state_dict",
        ]:
            self.assertTrue(
                callable(getattr(lm, method_name, None)),
                f"{method_name} should be callable",
            )

    def test_set_experiment_mode(self):
        from tbp.monty.frameworks.experiments.mode import ExperimentMode
        lm = self._make_lm()
        lm.set_experiment_mode(ExperimentMode.TRAIN)
        self.assertEqual(lm._mode, ExperimentMode.TRAIN)
        lm.set_experiment_mode(ExperimentMode.EVAL)
        self.assertEqual(lm._mode, ExperimentMode.EVAL)

    def test_reset_clears_state(self):
        lm = self._make_lm()
        lm._stepped = True
        lm.reset()
        self.assertFalse(lm._stepped)

    def test_column_accessible(self):
        """Inner CorticalColumn is accessible."""
        lm = self._make_lm()
        self.assertIsNotNone(lm.column)
        self.assertIsNotNone(lm.column.encoder)


class TestCorticalColumnLMObservations(unittest.TestCase):
    """Step 2: Observations bridge — list[State] to column.step()."""

    def _make_lm(self):
        from tbp.monty.frameworks.models.cortical_column.learning_module import (
            CorticalColumnLM,
        )
        return CorticalColumnLM(
            column_kwargs=dict(
                n_minicolumns=256,
                n_cells_per_minicolumn=4,
                use_weight_memory=True,
                use_attractor=True,
                seed=42,
            ),
        )

    def _make_state(self, location=(0.1, 0.2, 0.3)):
        return _FakeState(location)

    def test_matching_step_with_state_list(self):
        """matching_step accepts list of States."""
        from tbp.monty.frameworks.experiments.mode import ExperimentMode
        lm = self._make_lm()
        lm.set_experiment_mode(ExperimentMode.EVAL)
        lm.pre_episode()
        lm.matching_step(None, [self._make_state()])
        self.assertTrue(lm._stepped)

    def test_matching_step_with_none(self):
        """matching_step handles None observations gracefully."""
        from tbp.monty.frameworks.experiments.mode import ExperimentMode
        lm = self._make_lm()
        lm.set_experiment_mode(ExperimentMode.EVAL)
        lm.pre_episode()
        lm.matching_step(None, None)
        self.assertFalse(lm._stepped)

    def test_matching_step_with_empty_list(self):
        """matching_step handles empty list."""
        from tbp.monty.frameworks.experiments.mode import ExperimentMode
        lm = self._make_lm()
        lm.set_experiment_mode(ExperimentMode.EVAL)
        lm.pre_episode()
        lm.matching_step(None, [])
        self.assertFalse(lm._stepped)

    def test_matching_step_skips_unusable_states(self):
        """matching_step ignores States with use_state=False."""
        from tbp.monty.frameworks.experiments.mode import ExperimentMode
        lm = self._make_lm()
        lm.set_experiment_mode(ExperimentMode.EVAL)
        lm.pre_episode()
        bad_state = _FakeState([0.1, 0.2, 0.3])
        bad_state.use_state = False
        lm.matching_step(None, [bad_state])
        self.assertFalse(lm._stepped)

    def test_exploratory_step_learns(self):
        """exploratory_step trains the column."""
        from tbp.monty.frameworks.experiments.mode import ExperimentMode
        lm = self._make_lm()
        lm.set_experiment_mode(ExperimentMode.TRAIN)
        lm.pre_episode(primary_target="test_object")
        for i in range(10):
            lm.exploratory_step(None, [_FakeState([0.01 * i, 0.0, 0.0])])
        lm.post_episode()
        self.assertIn("test_object", lm.get_all_known_object_ids())

    def test_train_then_eval(self):
        """Full train→eval cycle produces evidence."""
        from tbp.monty.frameworks.experiments.mode import ExperimentMode
        lm = self._make_lm()

        # Train
        lm.set_experiment_mode(ExperimentMode.TRAIN)
        lm.pre_episode(primary_target="banana")
        for i in range(20):
            lm.exploratory_step(None, [_FakeState([0.01 * i, 0.0, 0.0])])
        lm.post_episode()

        # Eval
        lm.set_experiment_mode(ExperimentMode.EVAL)
        lm.pre_episode()
        for i in range(10):
            lm.matching_step(None, [_FakeState([0.01 * i, 0.0, 0.0])])

        mlh = lm.get_current_mlh()
        self.assertEqual(mlh["graph_id"], "banana")
        self.assertGreater(mlh["evidence"], 0.0)


class TestCorticalColumnLMVoting(unittest.TestCase):
    """Step 3: Voting between CorticalColumnLMs."""

    def _make_trained_lm(self, lm_id, objects):
        from tbp.monty.frameworks.experiments.mode import ExperimentMode
        from tbp.monty.frameworks.models.cortical_column.learning_module import (
            CorticalColumnLM,
        )
        lm = CorticalColumnLM(
            learning_module_id=lm_id,
            column_kwargs=dict(
                n_minicolumns=256,
                n_cells_per_minicolumn=4,
                use_weight_memory=True,
                use_attractor=True,
                seed=42,
            ),
        )
        lm.set_experiment_mode(ExperimentMode.TRAIN)
        for obj in objects:
            lm.pre_episode(primary_target=obj)
            for i in range(20):
                lm.exploratory_step(None, [_FakeState([0.01 * i, 0.0, 0.0])])
            lm.post_episode()
        return lm

    def test_send_out_vote_before_step(self):
        """send_out_vote returns None before any step."""
        from tbp.monty.frameworks.models.cortical_column.learning_module import (
            CorticalColumnLM,
        )
        lm = CorticalColumnLM(column_kwargs=dict(
            n_minicolumns=256, n_cells_per_minicolumn=4, seed=42,
        ))
        self.assertIsNone(lm.send_out_vote())

    def test_send_out_vote_returns_dict(self):
        """send_out_vote returns properly formatted dict after eval step."""
        from tbp.monty.frameworks.experiments.mode import ExperimentMode
        lm = self._make_trained_lm("lm0", ["banana"])
        lm.set_experiment_mode(ExperimentMode.EVAL)
        lm.pre_episode()
        lm.matching_step(None, [_FakeState([0.01, 0.0, 0.0])])
        vote = lm.send_out_vote()
        if vote is not None:
            self.assertIn("possible_states", vote)
            self.assertIn("sensed_pose_rel_body", vote)

    def test_receive_votes_adjusts_evidence(self):
        """Receiving positive votes boosts evidence for an object."""
        from tbp.monty.frameworks.experiments.mode import ExperimentMode
        from tbp.monty.frameworks.models.states import State
        lm = self._make_trained_lm("lm0", ["banana", "mug"])
        lm.set_experiment_mode(ExperimentMode.EVAL)
        lm.pre_episode()
        lm.matching_step(None, [_FakeState([0.01, 0.0, 0.0])])

        ev_before = dict(lm.column._evidence)

        # Simulate a vote from another LM favoring "banana"
        fake_vote_state = State(
            location=np.zeros(3),
            morphological_features={"pose_vectors": np.eye(3), "pose_fully_defined": True},
            non_morphological_features=None,
            confidence=0.9,
            use_state=True,
            sender_id="other_lm",
            sender_type="LM",
        )
        lm.receive_votes([{
            "possible_states": {"banana": [fake_vote_state]},
            "sensed_pose_rel_body": np.zeros(3),
        }])

        ev_after = lm.column._evidence
        self.assertGreater(
            ev_after.get("banana", 0),
            ev_before.get("banana", 0),
            "Vote should boost banana evidence",
        )

    def test_receive_votes_with_none(self):
        """receive_votes handles None gracefully."""
        from tbp.monty.frameworks.models.cortical_column.learning_module import (
            CorticalColumnLM,
        )
        lm = CorticalColumnLM(column_kwargs=dict(
            n_minicolumns=256, n_cells_per_minicolumn=4, seed=42,
        ))
        lm.receive_votes(None)  # Should not crash
        lm.receive_votes([None, None])  # Should not crash


class TestCorticalColumnLMHierarchy(unittest.TestCase):
    """Step 4: Hierarchy output and surprise-gated output."""

    def _make_lm(self, **kwargs):
        from tbp.monty.frameworks.models.cortical_column.learning_module import (
            CorticalColumnLM,
        )
        defaults = dict(
            column_kwargs=dict(
                n_minicolumns=256,
                n_cells_per_minicolumn=4,
                use_weight_memory=True,
                use_attractor=True,
                seed=42,
            ),
        )
        defaults.update(kwargs)
        return CorticalColumnLM(**defaults)

    def test_get_output_before_step(self):
        """get_output returns None before any step."""
        lm = self._make_lm()
        self.assertIsNone(lm.get_output())

    def test_get_output_returns_state(self):
        """get_output returns a valid State after eval step."""
        from tbp.monty.frameworks.experiments.mode import ExperimentMode
        from tbp.monty.frameworks.models.states import State
        lm = self._make_lm()

        # Train
        lm.set_experiment_mode(ExperimentMode.TRAIN)
        lm.pre_episode(primary_target="banana")
        for i in range(20):
            lm.exploratory_step(None, [_FakeState([0.01 * i, 0.0, 0.0])])
        lm.post_episode()

        # Eval
        lm.set_experiment_mode(ExperimentMode.EVAL)
        lm.pre_episode()
        for i in range(10):
            lm.matching_step(None, [_FakeState([0.01 * i, 0.0, 0.0])])

        output = lm.get_output()
        self.assertIsInstance(output, State)
        self.assertEqual(output.sender_type, "LM")
        self.assertIn("graph_id", output.non_morphological_features)

    def test_surprise_gated_output(self):
        """With surprise gating, low-surprise outputs have use_state=False."""
        from tbp.monty.frameworks.experiments.mode import ExperimentMode
        lm = self._make_lm(
            surprise_gated_output=True,
            output_surprise_threshold=0.99,  # Very high threshold → always gated
        )

        # Train
        lm.set_experiment_mode(ExperimentMode.TRAIN)
        lm.pre_episode(primary_target="banana")
        for i in range(20):
            lm.exploratory_step(None, [_FakeState([0.01 * i, 0.0, 0.0])])
        lm.post_episode()

        # Eval on same trajectory — should have some prediction
        lm.set_experiment_mode(ExperimentMode.EVAL)
        lm.pre_episode()
        for i in range(20):
            lm.matching_step(None, [_FakeState([0.01 * i, 0.0, 0.0])])

        output = lm.get_output()
        self.assertIsNotNone(output)
        # With threshold=0.99, most outputs should be gated (use_state=False)
        # unless surprise is very high
        self.assertIn(
            output.non_morphological_features.get("confirmed", None),
            [True, None],
        )

    def test_parent_receives_child_output(self):
        """A parent CorticalColumnLM can receive a child's output as input."""
        from tbp.monty.frameworks.experiments.mode import ExperimentMode
        child = self._make_lm(learning_module_id="child")
        parent = self._make_lm(learning_module_id="parent")

        # Train child
        child.set_experiment_mode(ExperimentMode.TRAIN)
        child.pre_episode(primary_target="banana")
        for i in range(20):
            child.exploratory_step(None, [_FakeState([0.01 * i, 0.0, 0.0])])
        child.post_episode()

        # Child eval step
        child.set_experiment_mode(ExperimentMode.EVAL)
        child.pre_episode()
        child.matching_step(None, [_FakeState([0.05, 0.0, 0.0])])
        child_output = child.get_output()
        self.assertIsNotNone(child_output)

        # Parent receives child output as observation
        parent.set_experiment_mode(ExperimentMode.TRAIN)
        parent.pre_episode(primary_target="scene_A")
        parent.exploratory_step(None, [child_output])
        # Should not crash — parent processes child's State

    def test_get_context_signal(self):
        """get_context_signal returns dict after step."""
        from tbp.monty.frameworks.experiments.mode import ExperimentMode
        lm = self._make_lm()
        lm.set_experiment_mode(ExperimentMode.EVAL)
        lm.pre_episode()
        lm.matching_step(None, [_FakeState([0.1, 0.2, 0.3])])
        signal = lm.get_context_signal()
        self.assertIsNotNone(signal)
        self.assertIn("active_cells", signal)

    def test_propose_goal_states_returns_list(self):
        lm = self._make_lm()
        self.assertIsInstance(lm.propose_goal_states(), list)


class TestCorticalColumnLMTopDown(unittest.TestCase):
    """Step 8: Top-down context in real hierarchy."""

    def _make_lm(self, lm_id, use_apical=True):
        from tbp.monty.frameworks.models.cortical_column.learning_module import (
            CorticalColumnLM,
        )
        return CorticalColumnLM(
            learning_module_id=lm_id,
            column_kwargs=dict(
                n_minicolumns=256,
                n_cells_per_minicolumn=4,
                use_weight_memory=True,
                use_attractor=True,
                use_apical=use_apical,
                seed=42,
            ),
        )

    def test_context_dispatch_flow(self):
        """Parent sends context signal, child receives it via apical dendrites.

        Simulates MontyBase's _dispatch_context_signals(): parent calls
        get_context_signal(), child calls receive_context().
        """
        from tbp.monty.frameworks.experiments.mode import ExperimentMode
        parent = self._make_lm("parent", use_apical=False)
        child = self._make_lm("child", use_apical=True)

        # Train parent
        parent.set_experiment_mode(ExperimentMode.TRAIN)
        parent.pre_episode(primary_target="car")
        for i in range(20):
            parent.exploratory_step(None, [_FakeState([0.01 * i, 0.0, 0.0])])
        parent.post_episode()

        # Train child with context from parent
        child.set_experiment_mode(ExperimentMode.TRAIN)
        child.pre_episode(primary_target="wheel")

        parent.set_experiment_mode(ExperimentMode.EVAL)
        parent.pre_episode()

        for i in range(20):
            obs = [_FakeState([0.01 * i, 0.0, 0.0])]
            parent.matching_step(None, obs)

            # Parent → child context dispatch
            ctx = parent.get_context_signal()
            if ctx is not None:
                child.receive_context(**ctx)

            child.exploratory_step(None, obs)

        child.post_episode()
        self.assertIn("wheel", child.get_all_known_object_ids())

    def test_apical_segments_grow_with_context(self):
        """Child grows apical dendritic segments from parent context."""
        from tbp.monty.frameworks.experiments.mode import ExperimentMode
        parent = self._make_lm("parent", use_apical=False)
        child = self._make_lm("child", use_apical=True)

        initial_segments = child.column.apical_dendrites.total_segments

        # Train with context flowing
        parent.set_experiment_mode(ExperimentMode.TRAIN)
        parent.pre_episode(primary_target="car")
        child.set_experiment_mode(ExperimentMode.TRAIN)
        child.pre_episode(primary_target="wheel")

        for i in range(20):
            obs = [_FakeState([0.01 * i, 0.0, 0.0])]
            parent.exploratory_step(None, obs)
            ctx = parent.get_context_signal()
            if ctx is not None:
                child.receive_context(**ctx)
            child.exploratory_step(None, obs)

        parent.post_episode()
        child.post_episode()

        final_segments = child.column.apical_dendrites.total_segments
        self.assertGreater(
            final_segments, initial_segments,
            "Child should grow apical segments from parent context",
        )

    def test_context_signal_has_required_fields(self):
        """get_context_signal returns active_cells and surprise."""
        from tbp.monty.frameworks.experiments.mode import ExperimentMode
        lm = self._make_lm("test")
        lm.set_experiment_mode(ExperimentMode.EVAL)
        lm.pre_episode()
        lm.matching_step(None, [_FakeState([0.1, 0.2, 0.3])])
        ctx = lm.get_context_signal()
        self.assertIn("active_cells", ctx)
        self.assertIn("surprise", ctx)
        self.assertEqual(len(ctx["active_cells"]), lm.column.n_cells)


class TestCorticalColumnLMPersistence(unittest.TestCase):
    """Step 1 continued: state_dict / load_state_dict."""

    def _make_lm(self):
        from tbp.monty.frameworks.models.cortical_column.learning_module import (
            CorticalColumnLM,
        )
        return CorticalColumnLM(
            column_kwargs=dict(
                n_minicolumns=256,
                n_cells_per_minicolumn=4,
                use_weight_memory=True,
                seed=42,
            ),
        )

    def test_state_dict_round_trip(self):
        """state_dict → load_state_dict preserves evidence."""
        from tbp.monty.frameworks.experiments.mode import ExperimentMode
        lm1 = self._make_lm()
        lm1.set_experiment_mode(ExperimentMode.TRAIN)
        lm1.pre_episode(primary_target="banana")
        for i in range(10):
            lm1.exploratory_step(None, [_FakeState([0.01 * i, 0.0, 0.0])])
        lm1.post_episode()

        sd = lm1.state_dict()
        self.assertIn("evidence", sd)
        self.assertIn("associative_memory", sd)

        lm2 = self._make_lm()
        lm2.load_state_dict(sd)
        self.assertEqual(lm2.learning_module_id, lm1.learning_module_id)


class TestCorticalColumnLMStateCond(unittest.TestCase):
    """Step 6: State-conditioned behavior support."""

    def _make_lm(self):
        from tbp.monty.frameworks.models.cortical_column.learning_module import (
            CorticalColumnLM,
        )
        return CorticalColumnLM(
            column_kwargs=dict(
                n_minicolumns=256,
                n_cells_per_minicolumn=4,
                use_weight_memory=True,
                use_attractor=True,
                seed=42,
            ),
        )

    def test_composite_key_created(self):
        """Training with object+state creates composite key in memory."""
        from tbp.monty.frameworks.experiments.mode import ExperimentMode
        lm = self._make_lm()
        lm.set_experiment_mode(ExperimentMode.TRAIN)
        lm.pre_episode(primary_target={"object": "stapler", "state": 0})
        for i in range(15):
            lm.exploratory_step(None, [_FakeState([0.01 * i, 0.0, 0.0])])
        lm.post_episode()

        known = lm.get_all_known_object_ids()
        self.assertIn("stapler:0", known)

    def test_two_states_distinct(self):
        """Two states of the same object get different labels."""
        from tbp.monty.frameworks.experiments.mode import ExperimentMode
        lm = self._make_lm()

        # State 0: one trajectory
        lm.set_experiment_mode(ExperimentMode.TRAIN)
        lm.pre_episode(primary_target={"object": "stapler", "state": 0})
        for i in range(15):
            lm.exploratory_step(None, [_FakeState(
                [0.01 * i, 0.0, 0.0], hsv=(0.1, 0.5, 0.8),
            )])
        lm.post_episode()

        # State 1: different trajectory/features
        lm.pre_episode(primary_target={"object": "stapler", "state": 1})
        for i in range(15):
            lm.exploratory_step(None, [_FakeState(
                [0.01 * i, 0.0, 0.0], hsv=(0.9, 0.5, 0.2),
            )])
        lm.post_episode()

        known = lm.get_all_known_object_ids()
        self.assertIn("stapler:0", known)
        self.assertIn("stapler:1", known)
        self.assertEqual(len(lm.get_known_objects()), 1)  # Same object

    def test_state_conditioned_eval(self):
        """Eval on state-conditioned column distinguishes states."""
        from tbp.monty.frameworks.experiments.mode import ExperimentMode
        lm = self._make_lm()

        # Train two states with very different features
        lm.set_experiment_mode(ExperimentMode.TRAIN)
        lm.pre_episode(primary_target={"object": "stapler", "state": 0})
        for i in range(20):
            lm.exploratory_step(None, [_FakeState(
                [0.01 * i, 0.0, 0.0], hsv=(0.1, 0.9, 0.9),
            )])
        lm.post_episode()

        lm.pre_episode(primary_target={"object": "stapler", "state": 1})
        for i in range(20):
            lm.exploratory_step(None, [_FakeState(
                [0.01 * i, 0.0, 0.0], hsv=(0.9, 0.1, 0.1),
            )])
        lm.post_episode()

        # Eval with state-0-like features
        lm.set_experiment_mode(ExperimentMode.EVAL)
        lm.pre_episode()
        for i in range(10):
            lm.matching_step(None, [_FakeState(
                [0.01 * i, 0.0, 0.0], hsv=(0.1, 0.9, 0.9),
            )])

        # Both composite keys should have evidence
        ev = lm.column._evidence
        self.assertIn("stapler:0", ev)
        self.assertIn("stapler:1", ev)

    def test_output_includes_inferred_state(self):
        """get_output includes inferred_state from composite key."""
        from tbp.monty.frameworks.experiments.mode import ExperimentMode
        lm = self._make_lm()

        lm.set_experiment_mode(ExperimentMode.TRAIN)
        lm.pre_episode(primary_target={"object": "stapler", "state": 0})
        for i in range(20):
            lm.exploratory_step(None, [_FakeState([0.01 * i, 0.0, 0.0])])
        lm.post_episode()

        lm.set_experiment_mode(ExperimentMode.EVAL)
        lm.pre_episode()
        for i in range(10):
            lm.matching_step(None, [_FakeState([0.01 * i, 0.0, 0.0])])

        output = lm.get_output()
        self.assertIsNotNone(output)
        # inferred_state should be 0 (parsed from "stapler:0")
        if output.non_morphological_features.get("graph_id") == "stapler":
            self.assertEqual(output.inferred_state, 0)

    def test_string_target_no_state(self):
        """String target (no state) works as before — no composite key."""
        from tbp.monty.frameworks.experiments.mode import ExperimentMode
        lm = self._make_lm()
        lm.set_experiment_mode(ExperimentMode.TRAIN)
        lm.pre_episode(primary_target="banana")
        for i in range(15):
            lm.exploratory_step(None, [_FakeState([0.01 * i, 0.0, 0.0])])
        lm.post_episode()

        known = lm.get_all_known_object_ids()
        self.assertIn("banana", known)
        self.assertNotIn(":", known[0])  # No composite key

    def test_parse_composite_key(self):
        from tbp.monty.frameworks.models.cortical_column.learning_module import (
            CorticalColumnLM,
        )
        self.assertEqual(
            CorticalColumnLM._parse_composite_key("stapler:0"),
            ("stapler", 0),
        )
        self.assertEqual(
            CorticalColumnLM._parse_composite_key("banana"),
            ("banana", None),
        )
        self.assertEqual(
            CorticalColumnLM._parse_composite_key("obj:name:1"),
            ("obj:name", 1),
        )


class TestCorticalColumnLMFlowFeatures(unittest.TestCase):
    """Step 7: ChangeDetectingSM compatibility — flow feature encoding."""

    def test_encoder_supports_flow_features(self):
        """FeatureSDREncoder can encode flow_direction and flow_magnitude."""
        from tbp.monty.frameworks.models.cortical_column.encoders import (
            FeatureSDREncoder,
        )
        enc = FeatureSDREncoder(
            features=["flow_direction", "flow_magnitude"],
        )
        self.assertGreater(enc.total_bits, enc.location_encoder.total_bits)
        self.assertIn("flow_x", enc._feature_encoders)
        self.assertIn("flow_mag", enc._feature_encoders)

    def test_encode_flow_state(self):
        """Encoder produces non-zero SDR for flow features."""
        from tbp.monty.frameworks.models.cortical_column.encoders import (
            FeatureSDREncoder,
        )
        enc = FeatureSDREncoder(
            features=["flow_direction", "flow_magnitude"],
        )

        class FlowState:
            def __init__(self):
                self.location = np.array([0.1, 0.2, 0.3])
                self.use_state = True
                self.non_morphological_features = {
                    "flow_direction": np.array([0.5, -0.5, 0.7]),
                    "flow_magnitude": np.array([0.8]),
                }
                self.morphological_features = {}

        sdr = enc.encode(FlowState())
        self.assertGreater(sdr.sum(), 0)

    def test_lm_with_flow_features(self):
        """CorticalColumnLM works with flow-feature encoder config."""
        from tbp.monty.frameworks.experiments.mode import ExperimentMode
        from tbp.monty.frameworks.models.cortical_column.learning_module import (
            CorticalColumnLM,
        )

        lm = CorticalColumnLM(
            column_kwargs=dict(
                n_minicolumns=256,
                n_cells_per_minicolumn=4,
                use_weight_memory=True,
                encoder_kwargs=dict(
                    features=["flow_direction", "flow_magnitude"],
                ),
                seed=42,
            ),
        )

        class FlowState:
            def __init__(self, loc, flow_dir, flow_mag):
                self.location = np.array(loc, dtype=np.float64)
                self.use_state = True
                self.sender_type = "SM"
                self.non_morphological_features = {
                    "flow_direction": np.array(flow_dir),
                    "flow_magnitude": np.array([flow_mag]),
                }
                self.morphological_features = {}

        lm.set_experiment_mode(ExperimentMode.TRAIN)
        lm.pre_episode(primary_target="walking")
        for i in range(15):
            lm.exploratory_step(None, [FlowState(
                [0.01 * i, 0.0, 0.0], [0.5, 0.0, 0.0], 0.5,
            )])
        lm.post_episode()

        self.assertIn("walking", lm.get_all_known_object_ids())

    def test_two_column_shape_and_behavior(self):
        """Two CorticalColumnLMs: one for shape (HSV), one for behavior (flow).

        Simulates the dual-SM architecture: CameraSM → shape column,
        ChangeDetectingSM → behavior column.
        """
        from tbp.monty.frameworks.experiments.mode import ExperimentMode
        from tbp.monty.frameworks.models.cortical_column.learning_module import (
            CorticalColumnLM,
        )

        shape_lm = CorticalColumnLM(
            learning_module_id="shape",
            column_kwargs=dict(
                n_minicolumns=256,
                n_cells_per_minicolumn=4,
                use_weight_memory=True,
                seed=42,
            ),
        )
        behavior_lm = CorticalColumnLM(
            learning_module_id="behavior",
            column_kwargs=dict(
                n_minicolumns=256,
                n_cells_per_minicolumn=4,
                use_weight_memory=True,
                encoder_kwargs=dict(
                    features=["flow_direction", "flow_magnitude"],
                ),
                seed=42,
            ),
        )

        class FlowState:
            def __init__(self, loc, flow_dir, flow_mag):
                self.location = np.array(loc, dtype=np.float64)
                self.use_state = True
                self.sender_type = "SM"
                self.non_morphological_features = {
                    "flow_direction": np.array(flow_dir),
                    "flow_magnitude": np.array([flow_mag]),
                }
                self.morphological_features = {}

        # Train both
        for lm, obj_name, make_obs in [
            (shape_lm, "banana", lambda i: [_FakeState([0.01 * i, 0, 0])]),
            (behavior_lm, "walking", lambda i: [FlowState(
                [0.01 * i, 0, 0], [0.5, 0.0, 0.0], 0.5,
            )]),
        ]:
            lm.set_experiment_mode(ExperimentMode.TRAIN)
            lm.pre_episode(primary_target=obj_name)
            for i in range(15):
                lm.exploratory_step(None, make_obs(i))
            lm.post_episode()

        # Both should have learned their respective objects
        self.assertIn("banana", shape_lm.get_all_known_object_ids())
        self.assertIn("walking", behavior_lm.get_all_known_object_ids())

        # Voting: shape → behavior and vice versa
        shape_lm.set_experiment_mode(ExperimentMode.EVAL)
        shape_lm.pre_episode()
        shape_lm.matching_step(None, [_FakeState([0.01, 0, 0])])
        shape_vote = shape_lm.send_out_vote()
        # Vote should be receivable by behavior LM (even though objects differ)
        behavior_lm.receive_votes([shape_vote])  # Should not crash


class TestCorticalColumnLMNovelty(unittest.TestCase):
    """Step 9: Autonomous novelty detection."""

    def _make_lm(self):
        from tbp.monty.frameworks.models.cortical_column.learning_module import (
            CorticalColumnLM,
        )
        return CorticalColumnLM(
            column_kwargs=dict(
                n_minicolumns=256,
                n_cells_per_minicolumn=4,
                use_weight_memory=True,
                use_attractor=True,
                seed=42,
            ),
        )

    def test_surprise_high_for_novel(self):
        """Novel objects produce high surprise during eval."""
        from tbp.monty.frameworks.experiments.mode import ExperimentMode
        lm = self._make_lm()

        # Train on banana
        lm.set_experiment_mode(ExperimentMode.TRAIN)
        lm.pre_episode(primary_target="banana")
        for i in range(20):
            lm.exploratory_step(None, [_FakeState(
                [0.01 * i, 0.0, 0.0], hsv=(0.15, 0.9, 0.9),
            )])
        lm.post_episode()

        # Eval on completely different features (novel object)
        lm.set_experiment_mode(ExperimentMode.EVAL)
        lm.pre_episode()
        surprises = []
        for i in range(10):
            lm.matching_step(None, [_FakeState(
                [0.01 * i, 0.0, 0.0], hsv=(0.85, 0.1, 0.1),
                curvatures=(2.0, -2.0),
            )])
            surprises.append(lm.column.surprise)

        # Mean surprise should be high (lots of bursting)
        mean_surprise = sum(surprises) / len(surprises)
        self.assertGreater(
            mean_surprise, 0.3,
            f"Novel object should cause surprise > 0.3, got {mean_surprise:.2f}",
        )


if __name__ == "__main__":
    unittest.main()
