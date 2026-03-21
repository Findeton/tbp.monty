# Copyright 2025-2026 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.models.evidence_matching.feature_evidence.sdr_calculator import (
    SDRFeatureEvidenceCalculator,
)
from tbp.monty.frameworks.models.evidence_sdr_matching import (
    EncoderSDR,
    EvidenceSDRLMMixin,
    EvidenceSDRTargetOverlaps,
)
from tbp.monty.frameworks.models.graph_matching import MontyForGraphMatching
from tbp.monty.frameworks.models.states import State


def set_seed(seed):
    """Set seed for reproducibility."""
    np.random.seed(seed)


class EvidenceSDRUnitTest(unittest.TestCase):
    def setUp(self):
        """Setup function at the beginning of each experiment."""
        # set seed for reproducibility
        set_seed(42)

    def test_unit_sdr_conf(self):
        """This tests edge condition of SDR configuration with invalid sparsity.

        For example, an SDR size of 100 with 100 on bits or 0 on bits.

        Expected behaviour:
            - Create warning
            - Adjust sparsity to 2%

        """
        encoder_sdr = EncoderSDR(
            sdr_length=100, sdr_on_bits=100, lr=1e-2, n_epochs=100, log_flag=False
        )

        # test that the sparsity is adjusted to 2%
        self.assertEqual(encoder_sdr.sdr_on_bits, int(100 * 0.02))

        encoder_sdr = EncoderSDR(
            sdr_length=100, sdr_on_bits=0, lr=1e-2, n_epochs=100, log_flag=False
        )

        # test that the sparsity is adjusted to 2%
        self.assertEqual(encoder_sdr.sdr_on_bits, int(100 * 0.02))

    def test_overlap_running_average(self):
        """This tests the model's ability to calculate the running average.

        For example, after giving the `EvidenceSDRTargetOverlaps` class
        three overlap tensors, it should store equally weighted average.

        Expected behaviour:
            - Expand to the specified number of objects
            - Track the correct running average after each overlap tensor

        """
        # define data
        first_overlaps = np.array([[1.0, 3.0], [2.0, 4.0]])
        second_overlaps = np.array([[3.0, 4.0], [8.0, 1.0]])
        third_overlaps = np.array([[6.0, 2.0], [10.0, 2.0]])

        first_second_avg = np.round((first_overlaps + second_overlaps) / 2)
        first_second_third_avg = np.round(
            (first_overlaps + second_overlaps + third_overlaps) / 3
        )

        # Instantiate target overlap class to keep track of running average
        target_overlaps = EvidenceSDRTargetOverlaps()

        # test expanding overlaps to the right size
        target_overlaps.add_objects(2)
        self.assertEqual(target_overlaps.overlaps.shape, ((2, 2)))

        # test adding first overlap to the array
        target_overlaps.add_overlaps(first_overlaps)
        self.assertTrue(np.all(first_overlaps == target_overlaps.overlaps))

        # test running average after adding second overlaps tensor
        target_overlaps.add_overlaps(second_overlaps)
        self.assertTrue(np.all(first_second_avg == target_overlaps.overlaps))

        # test running average after adding third overlaps tensor
        target_overlaps.add_overlaps(third_overlaps)
        self.assertTrue(np.all(first_second_third_avg == target_overlaps.overlaps))

    def test_map_to_overlaps_handles_constant_evidence(self):
        """Constant relative evidence should map to a finite overlap target."""
        target_overlaps = EvidenceSDRTargetOverlaps()
        evidence = np.array([[0.0, np.nan], [np.nan, 0.0]])

        mapped = target_overlaps.map_to_overlaps(evidence.copy(), [0, 5])

        self.assertTrue(np.all(np.isfinite(mapped[~np.isnan(mapped)])))
        self.assertEqual(mapped[0, 0], 5)
        self.assertEqual(mapped[1, 1], 5)

    def test_unit_train_zero_objects(self):
        """This tests edge condition of not sending any data to the training function.

        For example, we add 5 objects but attempt to train with no overlap targets.

        Expected behaviour:
            - Create warning
            - Return without training

        """
        encoder = EncoderSDR(
            sdr_length=100, sdr_on_bits=5, lr=1e-2, n_epochs=100, log_flag=True
        )
        encoder.add_objects(5)

        training_data = np.full((0, 0), np.nan)
        stats = encoder.train_sdrs(training_data)

        # test that it should return without training
        self.assertTrue("training" not in stats)

    def test_unit_can_train_subset_objects(self):
        """Test training on a subset of objects.

        This tests sending overlap targets for a subset of the objects and training
        the SDRs for them.

        For example adding 5 objects and training overlaps for 3 only.

        Expected behaviour:
            - Train normally without warning
            - Representations of the first 3 objects should change
            - Representations of the last 2 objects shouldn't change
        """
        encoder = EncoderSDR(
            sdr_length=100, sdr_on_bits=5, lr=1e-2, n_epochs=100, log_flag=True
        )
        encoder.add_objects(5)
        training_data = np.full((5, 5), np.nan)
        training_data[0, 1] = 12
        training_data[0, 2] = 33
        training_data[1, 2] = 17

        stats = encoder.train_sdrs(training_data)

        # test that it should train normally
        self.assertTrue("training" in stats)

        representations_before = stats["training"][0]["obj_dense"]
        representations_after = stats["training"][90]["obj_dense"]

        # test that the first 3 representations changed
        self.assertFalse(
            np.all(representations_before[:3] == representations_after[:3])
        )

        # test that the last 2 representations did not change
        self.assertTrue(
            np.all(representations_before[-2:] == representations_after[-2:])
        )

    def test_unit_can_train_superset_objects(self):
        """Test training on a superset of objects.

        This tests sending overlap targets for a objects not added to
        the encoder and training the SDRs for them.

        For example adding 3 objects and training overlaps for 5 objects.

        Expected behaviour when some points are out of bounds:
            - Create warning
            - Remove the out of bounds keys
            - Continue to train normally for valid points

        """
        # test for some points out of bound
        encoder = EncoderSDR(
            sdr_length=100, sdr_on_bits=5, lr=1e-2, n_epochs=100, log_flag=True
        )
        encoder.add_objects(3)
        training_data = np.full((5, 5), np.nan)
        training_data[0, 1] = 12
        training_data[0, 4] = 33
        training_data[1, 4] = 17

        # some points out of bounds
        stats = encoder.train_sdrs(training_data)

        # check that target overlap only has 3 objects, not 5
        self.assertEqual(stats["target_overlap"].shape, ((3, 3)))

        # check that the value at key 0,1 is 12
        self.assertEqual(stats["target_overlap"][0, 1], 12)

        # test for all points out of bounds
        encoder = EncoderSDR(
            sdr_length=100, sdr_on_bits=5, lr=1e-2, n_epochs=100, log_flag=True
        )
        encoder.add_objects(3)
        training_data = np.full((5, 5), np.nan)
        training_data[0, 3] = 12
        training_data[0, 4] = 33
        training_data[1, 4] = 17

        stats = encoder.train_sdrs(training_data)

        # check that the value of all target overlaps are 0.0
        self.assertTrue(np.all(stats["target_overlap"] == 0.0))

    def test_unit_can_minimize_train_error(self):
        """This tests the model's ability to follow target overlap.

        For example, the error in training 3 objects should be small.

        Expected behaviour:
            - Train normally without warning
            - Overlap error should be decreasing over time during training
            - SDR overlaps should be close to target overlaps
        """
        encoder = EncoderSDR(
            sdr_length=2048, sdr_on_bits=41, lr=1e-2, n_epochs=1000, log_flag=True
        )
        encoder.add_objects(3)
        training_data = np.full((3, 3), np.nan)
        training_data[0, 1] = 12
        training_data[0, 2] = 33
        training_data[1, 2] = 17

        stats = encoder.train_sdrs(training_data)

        mask = stats["mask"]
        overlap_error_before = stats["training"][0]["overlap_error"][mask == 1].mean()
        overlap_error_after = stats["training"][990]["overlap_error"][mask == 1].mean()

        condition_a = overlap_error_after < overlap_error_before  # error decreasing
        condition_b = overlap_error_after <= 2.0  # small average overalp error
        self.assertTrue(condition_a and condition_b)

    def test_can_stabilize_sdrs(self):
        """Test ability to stabilize old object SDRs when new objects are added.

        This behavior is controlled through the `stability` argument passed to the
        `sdr_args` in the LM args.

        In this unit test, we train 3 SDRs, then train a 4th SDR. Therefore,
        the stability will be applied on the 3 trained SDRs. We save a copy
        of those 3 SDRs before training the 4th SDR (before_sdrs), and compare
        it to the 3 SDRs after training the 4th SDR (after_sdrs). Depending on
        the amount of stability applied, they should be very similar
        (high stability) or not similar (low stability).

        We matrix multiply the SDRs (before and after) such that the overlaps
        will be on the diagonal. If the SDRs didn't change, we get the highest
        overlap (i.e., 41). But if they changed we get less that 41 overlap
        bits.

        Here we will test multiple stability values and test if the SDRs
        are more stable with higher stability values.

        Expected behaviour:
            - Train normally without warning
            - Stability value more than 0 stabilizes the SDRs
            - Higher stability values forces SDRs to be more stable
            - Stability value of 1 fixes SDRs to original value
        """
        sdr_persistence = []
        for stability in [0.0, 0.33, 0.66, 1.0]:
            set_seed(42)

            encoder = EncoderSDR(
                sdr_length=2048,
                sdr_on_bits=41,
                lr=1e-2,
                n_epochs=1000,
                stability=stability,
                log_flag=True,
            )

            encoder.add_objects(3)
            training_data = np.full((3, 3), np.nan)
            training_data[0, 1] = 12
            training_data[0, 2] = 33
            training_data[1, 2] = 17
            encoder.train_sdrs(training_data)
            before_sdrs = encoder.sdrs.copy()

            encoder.add_objects(1)
            training_data = np.full((4, 4), np.nan)
            training_data[0, 1] = 12
            training_data[0, 2] = 33
            training_data[1, 2] = 17
            training_data[0, 3] = 38
            training_data[1, 3] = 13
            training_data[2, 3] = 30
            encoder.train_sdrs(training_data)
            after_sdrs = encoder.sdrs.copy()[:3]

            # compare the older sdrs before and after training with new objects
            sdr_persistence.append(np.mean(np.diag(before_sdrs @ after_sdrs.T)))

        # test increasing stability
        for i in range(len(sdr_persistence) - 1):
            self.assertTrue(sdr_persistence[i + 1] > sdr_persistence[i])

        # test fixed sdrs at stability = 1.0
        self.assertEqual(sdr_persistence[-1], 41.0)

    def test_unit_can_train_identical_sdrs(self):
        """This tests the model's ability to train identical sdrs.

        Expected behaviour:
            - Train normally without warning
            - SDRs should be identical with overlap of exactly `sdr_on_bits`
        """
        encoder = EncoderSDR(
            sdr_length=2048, sdr_on_bits=41, lr=1e-2, n_epochs=1000, log_flag=True
        )
        encoder.add_objects(2)
        training_data = np.full((2, 2), np.nan)
        training_data[0, 1] = 41

        # object representations should not be identical at first
        sdrs = encoder.sdrs.copy()
        self.assertNotEqual((sdrs[0] * sdrs[1]).sum(), 41.0)

        encoder.train_sdrs(training_data)

        # object representations should become identical after training
        sdrs = encoder.sdrs.copy()
        self.assertEqual((sdrs[0] * sdrs[1]).sum(), 41.0)

    def test_collect_evidences_ignores_empty_evidence_arrays(self):
        """Empty evidence buffers should be skipped instead of crashing SDR collection."""
        encoder = EncoderSDR(
            sdr_length=64, sdr_on_bits=5, lr=1e-2, n_epochs=10, log_flag=False
        )
        lm = SimpleNamespace(
            obj2id={},
            id2obj={},
            sdr_encoder=encoder,
            target_overlaps=EvidenceSDRTargetOverlaps(),
            sdr_args={"sdr_on_bits": 5},
            evidence={
                "mug": np.array([4.0, 5.0]),
                "fork": np.array([]),
                "spoon": np.array([1.0, 2.0]),
            },
            get_all_known_object_ids=lambda: ["mug", "fork", "spoon"],
            get_current_mlh=lambda: {"graph_id": "mug"},
        )

        EvidenceSDRLMMixin.collect_evidences(lm)

        self.assertEqual(lm.target_overlaps.overlaps.shape, (3, 3))
        self.assertTrue(np.isnan(lm.target_overlaps.overlaps[0, 1]))
        self.assertTrue(np.isfinite(lm.target_overlaps.overlaps[0, 2]))

    def test_has_pairwise_overlap_targets_requires_off_diagonal_signal(self):
        """SDR training should only run once real pairwise targets exist."""
        empty = np.full((3, 3), np.nan)
        diagonal_only = np.full((3, 3), np.nan)
        np.fill_diagonal(diagonal_only, 41)
        pairwise = diagonal_only.copy()
        pairwise[0, 1] = 12

        self.assertFalse(EvidenceSDRLMMixin._has_pairwise_overlap_targets(empty))
        self.assertFalse(
            EvidenceSDRLMMixin._has_pairwise_overlap_targets(diagonal_only)
        )
        self.assertTrue(EvidenceSDRLMMixin._has_pairwise_overlap_targets(pairwise))

    def test_sdr_training_is_disabled_in_eval_by_default(self):
        """Ordinary evaluation should not mutate SDR state unless explicitly enabled."""
        self.assertFalse(
            EvidenceSDRLMMixin._should_train_sdrs(ExperimentMode.EVAL, False)
        )
        self.assertTrue(
            EvidenceSDRLMMixin._should_train_sdrs(ExperimentMode.EVAL, True)
        )
        self.assertTrue(
            EvidenceSDRLMMixin._should_train_sdrs(ExperimentMode.TRAIN, False)
        )

    def test_upward_object_id_features_union_top_k_hypotheses(self):
        """The upward carrier should preserve ambiguity as a union SDR."""
        lm = SimpleNamespace(
            sdr_args={"sdr_length": 4},
            upward_top_k=2,
            get_current_mlh=lambda: {"graph_id": "mug"},
            get_evidence_for_each_graph=lambda: (
                ["mug", "cup", "fork"],
                np.array([5.0, 5.0, 1.0]),
            ),
            _object_id_to_features=lambda object_id: {
                "mug": np.array([1.0, 0.0, 0.0, 0.0]),
                "cup": np.array([0.0, 1.0, 0.0, 0.0]),
                "fork": np.array([0.0, 0.0, 1.0, 0.0]),
            }[object_id],
        )

        features = EvidenceSDRLMMixin._get_upward_object_id_features(lm)

        np.testing.assert_allclose(features, np.array([1.0, 1.0, 0.0, 0.0]))

    def test_upward_object_support_features_encode_top_k_evidence(self):
        """The upward support feature should retain quantitative child evidence."""
        lm = SimpleNamespace(
            sdr_encoder=SimpleNamespace(n_objects=4),
            id2obj={0: "mug", 1: "cup", 2: "fork", 3: "spoon"},
            obj2id={"mug": 0, "cup": 1, "fork": 2, "spoon": 3},
            upward_support_top_k=2,
            upward_support_temperature=1.0,
            upward_support_evidence_normalization="none",
            get_evidence_for_each_graph=lambda: (
                ["mug", "cup", "fork", "spoon"],
                np.array([5.0, 5.0, 1.0, -2.0]),
            ),
        )

        features = EvidenceSDRLMMixin._get_upward_object_support_features(lm)

        np.testing.assert_allclose(features, np.array([0.5, 0.5, 0.0, 0.0]))

    def test_upward_support_range_normalization_softens_large_absolute_gaps(self):
        """Range normalization should prevent raw evidence scale from collapsing support."""
        lm = SimpleNamespace(
            upward_support_temperature=1.0,
            upward_support_evidence_normalization="range",
        )

        weights = EvidenceSDRLMMixin._calculate_upward_support_weights(
            lm,
            np.array([120.0, 90.0, 88.0]),
        )

        self.assertEqual(weights.shape, (3,))
        self.assertGreater(weights[1], 0.0)
        self.assertGreater(weights[2], 0.0)
        self.assertLess(weights[0], 0.6)
        np.testing.assert_allclose(weights.sum(), 1.0)

    def test_upward_object_support_range_normalization_is_used_by_packet_weights(self):
        """Packet support weights should share the same normalization as object support."""
        lm = SimpleNamespace(
            upward_top_k=2,
            upward_support_top_k=3,
            upward_support_temperature=1.0,
            upward_support_evidence_normalization="range",
            obj2id={"mug": 0, "cup": 1, "fork": 2},
            _object_id_to_features=lambda graph_id: {
                "mug": np.array([1.0, 0.0, 0.0, 0.0]),
                "cup": np.array([0.0, 1.0, 0.0, 0.0]),
                "fork": np.array([0.0, 0.0, 1.0, 0.0]),
            }[graph_id],
            get_current_mlh=lambda: {"graph_id": "mug"},
            get_evidence_for_each_graph=lambda: (
                ["mug", "cup", "fork"],
                np.array([120.0, 90.0, 88.0]),
            ),
        )

        packet = EvidenceSDRLMMixin._get_upward_hypothesis_packet(lm)

        self.assertEqual(packet["graph_ids"], ["mug", "cup", "fork"])
        self.assertGreater(packet["support_weights"][1], 0.0)
        self.assertGreater(packet["support_weights"][2], 0.0)
        np.testing.assert_allclose(sum(packet["support_weights"]), 1.0)

    def test_upward_hypothesis_packet_preserves_ranked_child_hypotheses(self):
        """The upward packet should expose the ranked child hypotheses and weights."""
        lm = SimpleNamespace(
            upward_top_k=2,
            upward_support_top_k=3,
            upward_support_temperature=1.0,
            upward_support_evidence_normalization="none",
            obj2id={"mug": 0, "cup": 1, "fork": 2, "spoon": 3},
            _object_id_to_features=lambda graph_id: {
                "mug": np.array([1.0, 0.0, 0.0, 0.0]),
                "cup": np.array([0.0, 1.0, 0.0, 0.0]),
                "fork": np.array([0.0, 0.0, 1.0, 0.0]),
                "spoon": np.array([0.0, 0.0, 0.0, 1.0]),
            }[graph_id],
            get_current_mlh=lambda: {"graph_id": "mug"},
            get_evidence_for_each_graph=lambda: (
                ["mug", "cup", "fork", "spoon"],
                np.array([5.0, 4.0, 1.0, -2.0]),
            ),
        )

        packet = EvidenceSDRLMMixin._get_upward_hypothesis_packet(lm)

        self.assertEqual(packet["mlh_graph_id"], "mug")
        self.assertEqual(packet["top_k"], 3)
        self.assertEqual(packet["graph_ids"], ["mug", "cup", "fork"])
        self.assertEqual(packet["object_indices"], [0, 1, 2])
        np.testing.assert_allclose(
            packet["object_ids"][0],
            np.array([1.0, 0.0, 0.0, 0.0]),
        )
        self.assertGreater(packet["support_weights"][0], packet["support_weights"][1])
        self.assertGreater(packet["support_weights"][1], packet["support_weights"][2])
        self.assertAlmostEqual(sum(packet["support_weights"]), 1.0)

    def test_safe_get_evidence_for_each_graph_handles_empty_startup_state(self):
        """Upward feature extraction should not crash when no graph ids exist yet."""
        lm = SimpleNamespace(
            get_evidence_for_each_graph=lambda: (_ for _ in ()).throw(IndexError()),
        )

        graph_ids, graph_evidences = EvidenceSDRLMMixin._safe_get_evidence_for_each_graph(
            lm
        )

        self.assertEqual(graph_ids, [])
        np.testing.assert_allclose(graph_evidences, np.array([], dtype=np.float64))

    def test_temporal_support_accumulator_blends_child_support_across_steps(self):
        """Parent-side LM support should accumulate rather than overwrite."""
        lm = SimpleNamespace(
            temporal_support_decay=0.5,
            temporal_support_object_id=True,
            temporal_support_accumulators={},
        )

        first = EvidenceSDRLMMixin._accumulate_temporal_lm_features(
            lm,
            "learning_module_0",
            {
                "object_id": np.array([1.0, 0.0, 0.0, 0.0]),
                "object_support": np.array([1.0, 0.0, 0.0, 0.0]),
            },
        )
        second = EvidenceSDRLMMixin._accumulate_temporal_lm_features(
            lm,
            "learning_module_0",
            {
                "object_id": np.array([0.0, 1.0, 0.0, 0.0]),
                "object_support": np.array([0.0, 1.0, 0.0, 0.0]),
            },
        )

        np.testing.assert_allclose(first["object_support"], np.array([1.0, 0.0, 0.0, 0.0]))
        np.testing.assert_allclose(
            second["object_support"],
            np.array([1.0 / 3.0, 2.0 / 3.0, 0.0, 0.0]),
        )
        np.testing.assert_allclose(
            second["object_id"],
            np.array([0.5, 1.0, 0.0, 0.0]),
        )

    def test_temporal_support_accumulator_preserves_upward_hypothesis_packet(self):
        """Temporal preprocessing should keep the structured child packet intact."""
        lm = SimpleNamespace(
            temporal_support_decay=0.5,
            temporal_support_object_id=True,
            temporal_support_accumulators={},
            _copy_feature_value=EvidenceSDRLMMixin._copy_feature_value,
        )

        packet = {
            "mlh_graph_id": "mug",
            "top_k": 2,
            "graph_ids": ["mug", "cup"],
            "object_ids": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            "evidences": [5.0, 4.0],
            "support_weights": [0.6, 0.4],
            "object_indices": [0, 1],
        }
        features = EvidenceSDRLMMixin._accumulate_temporal_lm_features(
            lm,
            "learning_module_0",
            {
                "object_id": np.array([1.0, 0.0, 0.0, 0.0]),
                "object_support": np.array([1.0, 0.0, 0.0, 0.0]),
                "upward_hypothesis_packet": packet,
            },
        )

        self.assertEqual(features["upward_hypothesis_packet"], packet)
        self.assertIsNot(features["upward_hypothesis_packet"], packet)

    def test_get_output_attaches_upward_hypothesis_packet(self):
        """Outgoing LM states should carry the structured top-k child packet."""

        class FakeParent:
            def get_output(self):
                return State(
                    location=np.zeros(3),
                    morphological_features={
                        "pose_vectors": np.eye(3),
                        "pose_fully_defined": True,
                    },
                    non_morphological_features={},
                    confidence=1.0,
                    use_state=True,
                    sender_id="learning_module_0",
                    sender_type="LM",
                )

        class FakeLM(EvidenceSDRLMMixin, FakeParent):
            pass

        lm = object.__new__(FakeLM)
        lm._get_upward_object_id_features = lambda: np.array([1.0, 1.0, 0.0, 0.0])
        lm._get_upward_object_support_features = lambda: np.array([0.6, 0.4, 0.0, 0.0])
        lm._get_upward_hypothesis_packet = lambda: {
            "mlh_graph_id": "mug",
            "top_k": 2,
            "graph_ids": ["mug", "cup"],
            "object_ids": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            "evidences": [5.0, 4.0],
            "support_weights": [0.6, 0.4],
            "object_indices": [0, 1],
        }

        state = lm.get_output()

        np.testing.assert_allclose(
            state.non_morphological_features["object_id"],
            np.array([1.0, 1.0, 0.0, 0.0]),
        )
        np.testing.assert_allclose(
            state.non_morphological_features["object_support"],
            np.array([0.6, 0.4, 0.0, 0.0]),
        )
        self.assertEqual(
            state.non_morphological_features["upward_hypothesis_packet"]["graph_ids"],
            ["mug", "cup"],
        )
        np.testing.assert_allclose(
            state.non_morphological_features["upward_hypothesis_packet"]["object_ids"][0],
            np.array([1.0, 0.0, 0.0, 0.0]),
        )

    def test_get_output_syncs_object_registry_before_upward_features(self):
        """Outgoing LM features should be meaningful even before post-episode SDR sync."""

        class FakeParent:
            def get_output(self):
                return State(
                    location=np.zeros(3),
                    morphological_features={
                        "pose_vectors": np.eye(3),
                        "pose_fully_defined": True,
                    },
                    non_morphological_features={},
                    confidence=1.0,
                    use_state=True,
                    sender_id="learning_module_0",
                    sender_type="LM",
                )

        class FakeEncoder:
            def __init__(self):
                self.n_objects = 0

            def add_objects(self, n_objects):
                self.n_objects += n_objects

            def get_sdr(self, object_index):
                vectors = {
                    0: np.array([1.0, 0.0, 0.0, 0.0]),
                    1: np.array([0.0, 1.0, 0.0, 0.0]),
                }
                return vectors[object_index]

        class FakeLM(EvidenceSDRLMMixin, FakeParent):
            pass

        lm = object.__new__(FakeLM)
        lm.obj2id = {}
        lm.id2obj = {}
        lm.target_overlaps = SimpleNamespace(add_objects=lambda n: None)
        lm.sdr_encoder = FakeEncoder()
        lm.sdr_args = {"sdr_length": 4}
        lm.get_all_known_object_ids = lambda: ["mug", "cup"]
        lm.get_current_mlh = lambda: {"graph_id": "mug"}
        lm.get_evidence_for_each_graph = lambda: (
            ["mug", "cup"],
            np.array([5.0, 4.0]),
        )
        lm.upward_top_k = 2
        lm.upward_support_top_k = 2
        lm.upward_support_temperature = 1.0
        lm.upward_support_evidence_normalization = "none"

        state = lm.get_output()

        np.testing.assert_allclose(
            state.non_morphological_features["object_id"],
            np.array([1.0, 1.0, 0.0, 0.0]),
        )
        np.testing.assert_allclose(
            state.non_morphological_features["object_support"],
            np.array([0.73105858, 0.26894142]),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            state.non_morphological_features["upward_hypothesis_packet"]["object_ids"][0],
            np.array([1.0, 0.0, 0.0, 0.0]),
        )

    def test_materialize_upward_hypothesis_packet_features_expands_rank_slots(self):
        """Receiver-side packet preprocessing should emit fixed numeric rank slots."""
        lm = SimpleNamespace(
            upward_packet_top_k=3,
            upward_packet_weight_features=False,
            sdr_args={"sdr_length": 4},
        )

        transformed = EvidenceSDRLMMixin._materialize_upward_hypothesis_packet_features(
            lm,
            {
                "object_id": np.array([1.0, 1.0, 0.0, 0.0]),
                "object_support": np.array([0.6, 0.4, 0.0, 0.0]),
                "upward_hypothesis_packet": {
                    "top_k": 2,
                    "object_ids": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                    ],
                },
            },
        )

        self.assertNotIn("upward_hypothesis_packet", transformed)
        np.testing.assert_allclose(
            transformed["object_id_rank_0"],
            np.array([1.0, 0.0, 0.0, 0.0]),
        )
        np.testing.assert_allclose(
            transformed["object_id_rank_1"],
            np.array([0.0, 1.0, 0.0, 0.0]),
        )
        np.testing.assert_allclose(
            transformed["object_id_rank_2"],
            np.zeros(4),
        )

    def test_materialize_upward_hypothesis_packet_features_emits_rank_weights(self):
        """Receiver-side packet preprocessing should expose support weights per rank."""
        lm = SimpleNamespace(
            upward_packet_top_k=3,
            upward_packet_weight_features=True,
            sdr_args={"sdr_length": 4},
        )

        transformed = EvidenceSDRLMMixin._materialize_upward_hypothesis_packet_features(
            lm,
            {
                "upward_hypothesis_packet": {
                    "top_k": 2,
                    "object_ids": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                    ],
                    "support_weights": [0.7, 0.3],
                },
            },
        )

        np.testing.assert_allclose(
            transformed["object_rank_weight_0"],
            np.array([0.7]),
        )
        np.testing.assert_allclose(
            transformed["object_rank_weight_1"],
            np.array([0.3]),
        )
        np.testing.assert_allclose(
            transformed["object_rank_weight_2"],
            np.zeros(1),
        )

    def test_augment_packet_weight_feature_config_registers_rank_weights(self):
        """Optional packet rank-weight features should be auto-registered in LM config."""
        tolerances, feature_weights = EvidenceSDRLMMixin._augment_packet_weight_feature_config(
            tolerances={
                "patch_0": {"hsv": np.array([0.1, 1.0, 1.0])},
                "learning_module_0": {"object_id": 16.0},
            },
            feature_weights={"learning_module_0": {"object_id": 1.0}},
            slot_count=2,
            enabled=True,
            default_tolerance=0.75,
            default_weight=0.1,
        )

        self.assertNotIn("object_rank_weight_0", tolerances["patch_0"])
        self.assertEqual(
            tolerances["learning_module_0"]["object_rank_weight_0"],
            0.75,
        )
        self.assertEqual(
            feature_weights["learning_module_0"]["object_rank_weight_1"],
            0.1,
        )

    def test_preprocess_temporal_lm_observations_materializes_packet_features(self):
        """LM preprocessing should expand packet slots even without temporal support."""
        state = State(
            location=np.zeros(3),
            morphological_features={
                "pose_vectors": np.eye(3),
                "pose_fully_defined": True,
            },
            non_morphological_features={
                "object_id": np.array([1.0, 1.0, 0.0, 0.0]),
                "object_support": np.array([0.6, 0.4, 0.0, 0.0]),
                "upward_hypothesis_packet": {
                    "top_k": 2,
                    "object_ids": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                    ],
                },
            },
            confidence=1.0,
            use_state=True,
            sender_id="learning_module_0",
            sender_type="LM",
        )
        lm = SimpleNamespace(
            temporal_support_accumulator=False,
            upward_packet_top_k=2,
            sdr_args={"sdr_length": 4},
            _materialize_upward_hypothesis_packet_features=(
                EvidenceSDRLMMixin._materialize_upward_hypothesis_packet_features
            ),
            _clone_state_with_features=EvidenceSDRLMMixin._clone_state_with_features,
        )

        processed = EvidenceSDRLMMixin._preprocess_temporal_lm_observations(lm, [state])

        self.assertEqual(len(processed), 1)
        self.assertNotIn(
            "upward_hypothesis_packet",
            processed[0].non_morphological_features,
        )
        np.testing.assert_allclose(
            processed[0].non_morphological_features["object_id_rank_0"],
            np.array([1.0, 0.0, 0.0, 0.0]),
        )
        np.testing.assert_allclose(
            processed[0].non_morphological_features["object_id_rank_1"],
            np.array([0.0, 1.0, 0.0, 0.0]),
        )

    def test_temporal_support_accumulator_resets_each_episode(self):
        """Temporal LM support should be episode-local state."""
        lm = SimpleNamespace(
            temporal_support_accumulators={
                "learning_module_0": {"object_support": np.array([1.0, 0.0])}
            }
        )

        EvidenceSDRLMMixin._reset_temporal_support_accumulators(lm)

        self.assertEqual(lm.temporal_support_accumulators, {})

    def test_graph_support_objective_bonus_rewards_matching_parent_prototype(self):
        """Graph-level support bonus should favor the better matching parent graph."""
        matching_graph = np.array(
            [
                [1.0, 1.0, 0.0, 0.0, 0.8, 0.2, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.7, 0.3, 0.0, 0.0],
            ]
        )
        mismatching_graph = np.array(
            [
                [0.0, 1.0, 0.0, 0.0, 0.1, 0.9, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            ]
        )

        class FakeLM:
            def __init__(self, graph_memory):
                self.graph_memory = graph_memory
                self.graph_support_objective_weight = 2.0
                self.graph_support_objective_tolerance = 0.0
                self.graph_support_num_prototypes = 1
                self.graph_support_competitive_objective = False
                self.graph_support_competitive_margin = 0.0
                self.graph_support_competitive_scope = "all"
                self.graph_support_competitive_top_k = 0

            _get_graph_support_prototype = (
                EvidenceSDRLMMixin._get_graph_support_prototype
            )
            _get_graph_support_prototypes = (
                EvidenceSDRLMMixin._get_graph_support_prototypes
            )
            _get_graph_support_competitor_ids = (
                EvidenceSDRLMMixin._get_graph_support_competitor_ids
            )
            _get_graph_support_similarity_details = (
                EvidenceSDRLMMixin._get_graph_support_similarity_details
            )
            _extract_feature_block_from_array = (
                EvidenceSDRLMMixin._extract_feature_block_from_array
            )
            _get_graph_feature_length = EvidenceSDRLMMixin._get_graph_feature_length
            _calculate_support_similarity = staticmethod(
                EvidenceSDRLMMixin._calculate_support_similarity
            )
            _calculate_competitive_support_similarity = staticmethod(
                EvidenceSDRLMMixin._calculate_competitive_support_similarity
            )
            _select_diverse_support_rows = staticmethod(
                EvidenceSDRLMMixin._select_diverse_support_rows
            )
            _calculate_graph_support_objective_bonus_details = (
                EvidenceSDRLMMixin._calculate_graph_support_objective_bonus_details
            )

        graph_memory = SimpleNamespace(
            get_feature_array=lambda graph_id: {
                "learning_module_0": matching_graph if graph_id == "parent_a" else mismatching_graph
            },
            get_feature_order=lambda graph_id: {
                "learning_module_0": ["object_id", "object_support"]
            },
            get_memory_ids=lambda: ["parent_a", "parent_b"],
            get_features_at_node=lambda graph_id, input_channel, node_id, feature_keys: {
                feature_keys[0]: np.array([1.0, 1.0, 0.0, 0.0])
                if feature_keys[0] == "object_id"
                else np.array([0.8, 0.2, 0.0, 0.0])
            },
        )
        lm = FakeLM(graph_memory)
        features = {
            "learning_module_0": {
                "object_support": np.array([0.75, 0.25, 0.0, 0.0]),
            }
        }

        bonus_a = EvidenceSDRLMMixin._calculate_graph_support_objective_bonus(
            lm,
            features,
            "parent_a",
        )
        bonus_b = EvidenceSDRLMMixin._calculate_graph_support_objective_bonus(
            lm,
            features,
            "parent_b",
        )

        self.assertGreater(bonus_a, bonus_b)
        self.assertGreater(bonus_a, 0.0)

    def test_graph_support_multi_prototype_bonus_preserves_minor_support_modes(self):
        """Multi-prototype scoring should recover stored support modes hidden by the mean."""
        graph = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.1, 0.9, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.1, 0.9, 0.0, 0.0],
            ]
        )

        class FakeLM:
            def __init__(self, num_prototypes):
                self.graph_memory = SimpleNamespace(
                    get_feature_array=lambda graph_id: {"learning_module_0": graph},
                    get_feature_order=lambda graph_id: {
                        "learning_module_0": ["object_id", "object_support"]
                    },
                    get_memory_ids=lambda: ["parent"],
                    get_features_at_node=lambda graph_id, input_channel, node_id, feature_keys: {
                        feature_keys[0]: np.array([1.0, 0.0, 0.0, 0.0])
                        if feature_keys[0] == "object_id"
                        else np.array([0.9, 0.1, 0.0, 0.0])
                    },
                )
                self.graph_support_objective_weight = 1.0
                self.graph_support_objective_tolerance = 0.0
                self.graph_support_num_prototypes = num_prototypes
                self.graph_support_competitive_objective = False
                self.graph_support_competitive_margin = 0.0
                self.graph_support_competitive_scope = "all"
                self.graph_support_competitive_top_k = 0

            _get_graph_support_prototype = (
                EvidenceSDRLMMixin._get_graph_support_prototype
            )
            _get_graph_support_prototypes = (
                EvidenceSDRLMMixin._get_graph_support_prototypes
            )
            _get_graph_support_competitor_ids = (
                EvidenceSDRLMMixin._get_graph_support_competitor_ids
            )
            _get_graph_support_similarity_details = (
                EvidenceSDRLMMixin._get_graph_support_similarity_details
            )
            _extract_feature_block_from_array = (
                EvidenceSDRLMMixin._extract_feature_block_from_array
            )
            _get_graph_feature_length = EvidenceSDRLMMixin._get_graph_feature_length
            _calculate_support_similarity = staticmethod(
                EvidenceSDRLMMixin._calculate_support_similarity
            )
            _calculate_competitive_support_similarity = staticmethod(
                EvidenceSDRLMMixin._calculate_competitive_support_similarity
            )
            _select_diverse_support_rows = staticmethod(
                EvidenceSDRLMMixin._select_diverse_support_rows
            )
            _calculate_graph_support_objective_bonus_details = (
                EvidenceSDRLMMixin._calculate_graph_support_objective_bonus_details
            )

        features = {
            "learning_module_0": {
                "object_support": np.array([0.9, 0.1, 0.0, 0.0]),
            }
        }

        mean_only_lm = FakeLM(num_prototypes=1)
        multi_proto_lm = FakeLM(num_prototypes=2)

        mean_only_bonus = EvidenceSDRLMMixin._calculate_graph_support_objective_bonus(
            mean_only_lm,
            features,
            "parent",
        )
        multi_proto_bonus = EvidenceSDRLMMixin._calculate_graph_support_objective_bonus(
            multi_proto_lm,
            features,
            "parent",
        )

        self.assertGreater(multi_proto_bonus, mean_only_bonus)
        self.assertAlmostEqual(multi_proto_bonus, 1.0, places=6)

    def test_graph_support_object_id_context_breaks_support_ties(self):
        """Object-id-conditioned support should resolve parents with identical support rows."""
        parent_a = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.8, 0.2, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.8, 0.2, 0.0, 0.0],
            ]
        )
        parent_b = np.array(
            [
                [0.0, 1.0, 0.0, 0.0, 0.8, 0.2, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.8, 0.2, 0.0, 0.0],
            ]
        )

        class FakeLM:
            def __init__(self):
                self.graph_memory = SimpleNamespace(
                    get_feature_array=lambda graph_id: {
                        "learning_module_0": parent_a if graph_id == "parent_a" else parent_b
                    },
                    get_feature_order=lambda graph_id: {
                        "learning_module_0": ["object_id", "object_support"]
                    },
                    get_memory_ids=lambda: ["parent_a", "parent_b"],
                    get_features_at_node=lambda graph_id, input_channel, node_id, feature_keys: {
                        feature_keys[0]: np.array([1.0, 0.0, 0.0, 0.0])
                        if feature_keys[0] == "object_id"
                        else np.array([0.8, 0.2, 0.0, 0.0])
                    },
                )
                self.graph_support_objective_weight = 1.0
                self.graph_support_objective_tolerance = 0.0
                self.graph_support_num_prototypes = 1
                self.graph_support_object_id_context = True
                self.graph_support_competitive_objective = False
                self.graph_support_competitive_margin = 0.0
                self.graph_support_competitive_scope = "all"
                self.graph_support_competitive_top_k = 0

            _get_graph_support_row_features = (
                EvidenceSDRLMMixin._get_graph_support_row_features
            )
            _get_graph_support_prototype = (
                EvidenceSDRLMMixin._get_graph_support_prototype
            )
            _get_graph_support_prototypes = (
                EvidenceSDRLMMixin._get_graph_support_prototypes
            )
            _get_graph_support_competitor_ids = (
                EvidenceSDRLMMixin._get_graph_support_competitor_ids
            )
            _get_graph_support_similarity_details = (
                EvidenceSDRLMMixin._get_graph_support_similarity_details
            )
            _extract_feature_block_from_array = (
                EvidenceSDRLMMixin._extract_feature_block_from_array
            )
            _get_graph_feature_length = EvidenceSDRLMMixin._get_graph_feature_length
            _calculate_support_similarity = staticmethod(
                EvidenceSDRLMMixin._calculate_support_similarity
            )
            _calculate_graph_support_object_id_context_similarity = staticmethod(
                EvidenceSDRLMMixin._calculate_graph_support_object_id_context_similarity
            )
            _calculate_competitive_support_similarity = staticmethod(
                EvidenceSDRLMMixin._calculate_competitive_support_similarity
            )
            _select_diverse_support_rows = staticmethod(
                EvidenceSDRLMMixin._select_diverse_support_rows
            )
            _calculate_graph_support_objective_bonus_details = (
                EvidenceSDRLMMixin._calculate_graph_support_objective_bonus_details
            )

        lm = FakeLM()
        features = {
            "learning_module_0": {
                "object_id": np.array([1.0, 0.0, 0.0, 0.0]),
                "object_support": np.array([0.8, 0.2, 0.0, 0.0]),
            }
        }

        bonus_a = EvidenceSDRLMMixin._calculate_graph_support_objective_bonus(
            lm,
            features,
            "parent_a",
        )
        bonus_b = EvidenceSDRLMMixin._calculate_graph_support_objective_bonus(
            lm,
            features,
            "parent_b",
        )

        self.assertGreater(bonus_a, bonus_b)
        self.assertAlmostEqual(bonus_a, 1.0, places=6)
        self.assertEqual(bonus_b, 0.0)

    def test_graph_support_packet_context_breaks_support_ties(self):
        """Packet-conditioned support should resolve parents with identical support rows."""
        parent_a = np.array(
            [
                [1.0, 0.0, 0.9, 0.8, 0.2, 0.0, 0.0],
                [1.0, 0.0, 0.9, 0.8, 0.2, 0.0, 0.0],
            ]
        )
        parent_b = np.array(
            [
                [0.0, 1.0, 0.0, 0.8, 0.2, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.8, 0.2, 0.0, 0.0],
            ]
        )

        class FakeLM:
            def __init__(self):
                self.graph_memory = SimpleNamespace(
                    get_feature_array=lambda graph_id: {
                        "learning_module_0": parent_a if graph_id == "parent_a" else parent_b
                    },
                    get_feature_order=lambda graph_id: {
                        "learning_module_0": [
                            "object_id_rank_0",
                            "object_rank_weight_0",
                            "object_support",
                        ]
                    },
                    get_memory_ids=lambda: ["parent_a", "parent_b"],
                    get_features_at_node=lambda graph_id, input_channel, node_id, feature_keys: {
                        "object_id_rank_0": np.array([1.0, 0.0]),
                        "object_rank_weight_0": np.array([0.9]),
                        "object_support": np.array([0.8, 0.2, 0.0, 0.0]),
                    },
                )
                self.graph_support_objective_weight = 1.0
                self.graph_support_objective_tolerance = 0.0
                self.graph_support_num_prototypes = 1
                self.graph_support_packet_context = True
                self.graph_support_object_id_context = False
                self.graph_support_competitive_objective = False
                self.graph_support_competitive_margin = 0.0
                self.graph_support_competitive_scope = "all"
                self.graph_support_competitive_top_k = 0

            _get_query_graph_support_packet_context = (
                EvidenceSDRLMMixin._get_query_graph_support_packet_context
            )
            _get_graph_support_packet_row_features = (
                EvidenceSDRLMMixin._get_graph_support_packet_row_features
            )
            _get_packet_context_slot_indices = staticmethod(
                EvidenceSDRLMMixin._get_packet_context_slot_indices
            )
            _get_graph_support_row_features = (
                EvidenceSDRLMMixin._get_graph_support_row_features
            )
            _get_graph_support_prototype = (
                EvidenceSDRLMMixin._get_graph_support_prototype
            )
            _get_graph_support_prototypes = (
                EvidenceSDRLMMixin._get_graph_support_prototypes
            )
            _get_graph_support_competitor_ids = (
                EvidenceSDRLMMixin._get_graph_support_competitor_ids
            )
            _get_graph_support_similarity_details = (
                EvidenceSDRLMMixin._get_graph_support_similarity_details
            )
            _extract_feature_block_from_array = (
                EvidenceSDRLMMixin._extract_feature_block_from_array
            )
            _get_graph_feature_length = EvidenceSDRLMMixin._get_graph_feature_length
            _calculate_support_similarity = staticmethod(
                EvidenceSDRLMMixin._calculate_support_similarity
            )
            _calculate_graph_support_packet_context_similarity = staticmethod(
                EvidenceSDRLMMixin._calculate_graph_support_packet_context_similarity
            )
            _calculate_graph_support_object_id_context_similarity = staticmethod(
                EvidenceSDRLMMixin._calculate_graph_support_object_id_context_similarity
            )
            _calculate_competitive_support_similarity = staticmethod(
                EvidenceSDRLMMixin._calculate_competitive_support_similarity
            )
            _select_diverse_support_rows = staticmethod(
                EvidenceSDRLMMixin._select_diverse_support_rows
            )
            _calculate_graph_support_objective_bonus_details = (
                EvidenceSDRLMMixin._calculate_graph_support_objective_bonus_details
            )

        lm = FakeLM()
        features = {
            "learning_module_0": {
                "object_id_rank_0": np.array([1.0, 0.0]),
                "object_rank_weight_0": np.array([0.9]),
                "object_support": np.array([0.8, 0.2, 0.0, 0.0]),
            }
        }

        bonus_a = EvidenceSDRLMMixin._calculate_graph_support_objective_bonus(
            lm,
            features,
            "parent_a",
        )
        bonus_b = EvidenceSDRLMMixin._calculate_graph_support_objective_bonus(
            lm,
            features,
            "parent_b",
        )

        self.assertGreater(bonus_a, bonus_b)
        self.assertAlmostEqual(bonus_a, 1.0, places=6)
        self.assertEqual(bonus_b, 0.0)

    def test_graph_support_competitive_bonus_requires_beating_best_alternative(self):
        """Competitive graph-support should reward only the graph with relative advantage."""
        parent_a = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.8, 0.2, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.78, 0.22, 0.0, 0.0],
            ]
        )
        parent_b = np.array(
            [
                [0.0, 1.0, 0.0, 0.0, 0.72, 0.28, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.7, 0.3, 0.0, 0.0],
            ]
        )

        class FakeLM:
            def __init__(self):
                self.graph_memory = SimpleNamespace(
                    get_feature_array=lambda graph_id: {
                        "learning_module_0": parent_a if graph_id == "parent_a" else parent_b
                    },
                    get_feature_order=lambda graph_id: {
                        "learning_module_0": ["object_id", "object_support"]
                    },
                    get_memory_ids=lambda: ["parent_a", "parent_b"],
                    get_features_at_node=lambda graph_id, input_channel, node_id, feature_keys: {
                        feature_keys[0]: np.array([1.0, 0.0, 0.0, 0.0])
                        if feature_keys[0] == "object_id"
                        else np.array([0.8, 0.2, 0.0, 0.0])
                    },
                )
                self.graph_support_objective_weight = 2.0
                self.graph_support_objective_tolerance = 0.0
                self.graph_support_num_prototypes = 1
                self.graph_support_competitive_objective = True
                self.graph_support_competitive_margin = 0.0
                self.graph_support_competitive_penalty = 1.0
                self.graph_support_competitive_scope = "all"
                self.graph_support_competitive_top_k = 0
                self.evidence = {
                    "parent_a": np.array([0.9]),
                    "parent_b": np.array([0.8]),
                }
                self.get_possible_matches = lambda: ["parent_a", "parent_b"]

            _get_graph_support_prototype = (
                EvidenceSDRLMMixin._get_graph_support_prototype
            )
            _get_graph_support_prototypes = (
                EvidenceSDRLMMixin._get_graph_support_prototypes
            )
            _get_graph_support_competitor_ids = (
                EvidenceSDRLMMixin._get_graph_support_competitor_ids
            )
            _get_graph_support_similarity_details = (
                EvidenceSDRLMMixin._get_graph_support_similarity_details
            )
            _extract_feature_block_from_array = (
                EvidenceSDRLMMixin._extract_feature_block_from_array
            )
            _get_graph_feature_length = EvidenceSDRLMMixin._get_graph_feature_length
            _calculate_support_similarity = staticmethod(
                EvidenceSDRLMMixin._calculate_support_similarity
            )
            _calculate_competitive_support_similarity = staticmethod(
                EvidenceSDRLMMixin._calculate_competitive_support_similarity
            )
            _select_diverse_support_rows = staticmethod(
                EvidenceSDRLMMixin._select_diverse_support_rows
            )
            _calculate_graph_support_objective_bonus_details = (
                EvidenceSDRLMMixin._calculate_graph_support_objective_bonus_details
            )

        lm = FakeLM()
        features = {
            "learning_module_0": {
                "object_support": np.array([0.79, 0.21, 0.0, 0.0]),
            }
        }

        bonus_a, details_a = EvidenceSDRLMMixin._calculate_graph_support_objective_bonus_details(
            lm,
            features,
            "parent_a",
        )
        bonus_b, details_b = EvidenceSDRLMMixin._calculate_graph_support_objective_bonus_details(
            lm,
            features,
            "parent_b",
        )

        self.assertGreater(bonus_a, 0.0)
        self.assertEqual(bonus_b, 0.0)
        self.assertEqual(details_a[0]["best_competitor_graph_id"], "parent_b")
        self.assertEqual(details_b[0]["best_competitor_graph_id"], "parent_a")

    def test_graph_support_competitive_penalty_softens_best_rival_subtraction(self):
        """Competitive penalty should preserve some bonus when the rival subtraction is softened."""
        hard_similarity = EvidenceSDRLMMixin._calculate_competitive_support_similarity(
            0.8,
            0.75,
            0.0,
            penalty=1.0,
        )
        soft_similarity = EvidenceSDRLMMixin._calculate_competitive_support_similarity(
            0.8,
            0.75,
            0.0,
            penalty=0.85,
        )

        self.assertAlmostEqual(hard_similarity, 0.05, places=6)
        self.assertAlmostEqual(soft_similarity, 0.1625, places=6)
        self.assertGreater(soft_similarity, hard_similarity)

    def test_graph_support_same_peak_competitor_filter_prefers_sibling_like_rival(self):
        """Competitive support can optionally compare against only same-peak rivals."""
        target_parent = np.array(
            [[1.0, 0.0, 0.0, 0.0, 0.6, 0.4, 0.0, 0.0]]
        )
        same_peak_parent = np.array(
            [[0.0, 1.0, 0.0, 0.0, 0.95, 0.05, 0.0, 0.0]]
        )
        other_peak_parent = np.array(
            [[0.0, 0.0, 1.0, 0.0, 0.45, 0.55, 0.0, 0.0]]
        )

        class FakeLM:
            def __init__(self, same_peak_only):
                self.graph_memory = SimpleNamespace(
                    get_feature_array=lambda graph_id: {
                        "learning_module_0": {
                            "target": target_parent,
                            "same_peak": same_peak_parent,
                            "other_peak": other_peak_parent,
                        }[graph_id]
                    },
                    get_feature_order=lambda graph_id: {
                        "learning_module_0": ["object_id", "object_support"]
                    },
                    get_memory_ids=lambda: ["target", "same_peak", "other_peak"],
                    get_features_at_node=lambda graph_id, input_channel, node_id, feature_keys: {
                        feature_keys[0]: {
                            "target": target_parent,
                            "same_peak": same_peak_parent,
                            "other_peak": other_peak_parent,
                        }[graph_id][0, :4]
                        if feature_keys[0] == "object_id"
                        else {
                            "target": target_parent,
                            "same_peak": same_peak_parent,
                            "other_peak": other_peak_parent,
                        }[graph_id][0, 4:8]
                    },
                )
                self.graph_support_objective_weight = 2.0
                self.graph_support_objective_tolerance = 0.0
                self.graph_support_num_prototypes = 1
                self.graph_support_competitive_objective = True
                self.graph_support_competitive_margin = 0.0
                self.graph_support_competitive_penalty = 1.0
                self.graph_support_competitive_same_peak_only = same_peak_only
                self.graph_support_competitive_scope = "all"
                self.graph_support_competitive_top_k = 0
                self.evidence = {
                    "target": np.array([0.9]),
                    "same_peak": np.array([0.8]),
                    "other_peak": np.array([0.85]),
                }
                self.get_possible_matches = lambda: [
                    "target",
                    "same_peak",
                    "other_peak",
                ]

            _get_graph_support_prototype = (
                EvidenceSDRLMMixin._get_graph_support_prototype
            )
            _get_graph_support_prototypes = (
                EvidenceSDRLMMixin._get_graph_support_prototypes
            )
            _get_graph_support_competitor_ids = (
                EvidenceSDRLMMixin._get_graph_support_competitor_ids
            )
            _get_graph_support_similarity_details = (
                EvidenceSDRLMMixin._get_graph_support_similarity_details
            )
            _extract_feature_block_from_array = (
                EvidenceSDRLMMixin._extract_feature_block_from_array
            )
            _get_graph_feature_length = EvidenceSDRLMMixin._get_graph_feature_length
            _calculate_support_similarity = staticmethod(
                EvidenceSDRLMMixin._calculate_support_similarity
            )
            _calculate_competitive_support_similarity = staticmethod(
                EvidenceSDRLMMixin._calculate_competitive_support_similarity
            )
            _select_diverse_support_rows = staticmethod(
                EvidenceSDRLMMixin._select_diverse_support_rows
            )
            _calculate_graph_support_objective_bonus_details = (
                EvidenceSDRLMMixin._calculate_graph_support_objective_bonus_details
            )

        features = {
            "learning_module_0": {
                "object_support": np.array([0.55, 0.45, 0.0, 0.0]),
            }
        }

        unrestricted_lm = FakeLM(same_peak_only=False)
        unrestricted_bonus, unrestricted_details = (
            EvidenceSDRLMMixin._calculate_graph_support_objective_bonus_details(
                unrestricted_lm,
                features,
                "target",
            )
        )

        same_peak_lm = FakeLM(same_peak_only=True)
        same_peak_bonus, same_peak_details = (
            EvidenceSDRLMMixin._calculate_graph_support_objective_bonus_details(
                same_peak_lm,
                features,
                "target",
            )
        )

        self.assertEqual(
            unrestricted_details[0]["best_competitor_graph_id"],
            "other_peak",
        )
        self.assertEqual(
            same_peak_details[0]["best_competitor_graph_id"],
            "same_peak",
        )
        self.assertEqual(same_peak_details[0]["support_peak_index"], 0)
        self.assertEqual(
            same_peak_details[0]["best_competitor_support_peak_index"],
            0,
        )
        self.assertGreater(same_peak_bonus, unrestricted_bonus)

    def test_graph_support_packet_context_floor_prefers_context_aligned_rival(self):
        """Competitive support can optionally require a strong packet-context match."""

        class FakeLM:
            def __init__(
                self,
                packet_context_floor,
                packet_context_strict=False,
                packet_context_strict_ceiling=1.0,
            ):
                self.graph_support_objective_weight = 2.0
                self.graph_support_objective_tolerance = 0.0
                self.graph_support_num_prototypes = 1
                self.graph_support_competitive_objective = True
                self.graph_support_competitive_margin = 0.0
                self.graph_support_competitive_penalty = 1.0
                self.graph_support_competitive_same_peak_only = False
                self.graph_support_competitive_packet_context_floor = (
                    packet_context_floor
                )
                self.graph_support_competitive_packet_context_strict = (
                    packet_context_strict
                )
                self.graph_support_competitive_packet_context_strict_ceiling = (
                    packet_context_strict_ceiling
                )
                self.graph_support_competitive_scope = "all"
                self.graph_support_competitive_top_k = 0
                self.evidence = {
                    "target": np.array([0.9]),
                    "context_rival": np.array([0.8]),
                    "far_rival": np.array([0.85]),
                }
                self.graph_memory = SimpleNamespace(
                    get_memory_ids=lambda: ["target", "context_rival", "far_rival"]
                )
                self.get_possible_matches = lambda: [
                    "target",
                    "context_rival",
                    "far_rival",
                ]

            _get_graph_support_competitor_ids = (
                EvidenceSDRLMMixin._get_graph_support_competitor_ids
            )
            _calculate_competitive_support_similarity = staticmethod(
                EvidenceSDRLMMixin._calculate_competitive_support_similarity
            )
            _calculate_graph_support_objective_bonus_details = (
                EvidenceSDRLMMixin._calculate_graph_support_objective_bonus_details
            )

            def _get_graph_support_similarity_details(
                self,
                graph_id,
                input_channel,
                query_support,
                query_packet_context=None,
                query_object_id=None,
            ):
                return {
                    "target": {
                        "support_similarity": 0.9,
                        "best_support_prototype_index": 0,
                        "support_prototype_count": 1,
                        "graph_support_norm": 1.0,
                        "packet_context_similarity": 0.95,
                        "object_id_context_similarity": 1.0,
                        "support_peak_index": 0,
                    },
                    "context_rival": {
                        "support_similarity": 0.82,
                        "best_support_prototype_index": 0,
                        "support_prototype_count": 1,
                        "graph_support_norm": 1.0,
                        "packet_context_similarity": 0.9,
                        "object_id_context_similarity": 1.0,
                        "support_peak_index": 0,
                    },
                    "far_rival": {
                        "support_similarity": 0.88,
                        "best_support_prototype_index": 0,
                        "support_prototype_count": 1,
                        "graph_support_norm": 1.0,
                        "packet_context_similarity": 0.5,
                        "object_id_context_similarity": 1.0,
                        "support_peak_index": 1,
                    },
                }[graph_id]

        features = {
            "learning_module_0": {
                "object_support": np.array([0.55, 0.45, 0.0, 0.0]),
            }
        }

        unrestricted_lm = FakeLM(packet_context_floor=0.0)
        unrestricted_bonus, unrestricted_details = (
            EvidenceSDRLMMixin._calculate_graph_support_objective_bonus_details(
                unrestricted_lm,
                features,
                "target",
            )
        )

        context_floor_lm = FakeLM(packet_context_floor=0.9)
        context_floor_bonus, context_floor_details = (
            EvidenceSDRLMMixin._calculate_graph_support_objective_bonus_details(
                context_floor_lm,
                features,
                "target",
            )
        )

        strict_floor_lm = FakeLM(packet_context_floor=0.95, packet_context_strict=True)
        strict_floor_bonus, strict_floor_details = (
            EvidenceSDRLMMixin._calculate_graph_support_objective_bonus_details(
                strict_floor_lm,
                features,
                "target",
            )
        )

        self.assertEqual(
            unrestricted_details[0]["best_competitor_graph_id"],
            "far_rival",
        )
        self.assertEqual(
            context_floor_details[0]["best_competitor_graph_id"],
            "context_rival",
        )
        self.assertAlmostEqual(
            context_floor_details[0]["best_competitor_packet_context_similarity"],
            0.9,
        )
        self.assertGreater(context_floor_bonus, unrestricted_bonus)
        self.assertIsNone(strict_floor_details[0]["best_competitor_graph_id"])
        self.assertEqual(
            strict_floor_details[0]["best_competitor_support_similarity"],
            0.0,
        )
        self.assertIsNone(
            strict_floor_details[0]["best_competitor_packet_context_similarity"]
        )
        self.assertGreater(strict_floor_bonus, context_floor_bonus)

        ceiling_lm = FakeLM(
            packet_context_floor=0.95,
            packet_context_strict=True,
            packet_context_strict_ceiling=0.9,
        )
        ceiling_bonus, ceiling_details = (
            EvidenceSDRLMMixin._calculate_graph_support_objective_bonus_details(
                ceiling_lm,
                features,
                "target",
            )
        )

        self.assertEqual(
            ceiling_details[0]["best_competitor_graph_id"],
            "far_rival",
        )
        self.assertLess(ceiling_bonus, strict_floor_bonus)

    def test_graph_support_competitive_top_k_limits_competitors_to_current_candidates(self):
        """Competitive graph-support should optionally compare only against top evidence rivals."""
        class FakeLM:
            def __init__(self):
                self.graph_memory = SimpleNamespace(
                    get_memory_ids=lambda: ["target", "near", "far"],
                )
                self.graph_support_competitive_scope = "top_k_evidence"
                self.graph_support_competitive_top_k = 1
                self.evidence = {
                    "target": np.array([0.95, 0.9]),
                    "near": np.array([0.85, 0.8]),
                    "far": np.array([0.99, 0.1]),
                }

            _get_graph_support_competitor_ids = (
                EvidenceSDRLMMixin._get_graph_support_competitor_ids
            )

        lm = FakeLM()

        competitor_ids = EvidenceSDRLMMixin._get_graph_support_competitor_ids(
            lm,
            "target",
        )

        self.assertEqual(competitor_ids, ["far"])

    def test_graph_support_competitive_possible_matches_scope_limits_competitors(self):
        """Competitive graph-support should optionally compare only against current possible matches."""
        class FakeLM:
            def __init__(self):
                self.graph_memory = SimpleNamespace(
                    get_memory_ids=lambda: ["target", "pm_a", "pm_b", "other"],
                )
                self.graph_support_competitive_scope = "possible_matches"
                self.graph_support_competitive_top_k = 0
                self.get_possible_matches = lambda: ["target", "pm_a", "pm_b"]

            _get_graph_support_competitor_ids = (
                EvidenceSDRLMMixin._get_graph_support_competitor_ids
            )

        lm = FakeLM()

        competitor_ids = EvidenceSDRLMMixin._get_graph_support_competitor_ids(
            lm,
            "target",
        )

        self.assertEqual(competitor_ids, ["pm_a", "pm_b"])

    def test_graph_support_competitive_top_k_possible_matches_scope_ranks_within_candidates(self):
        """Competitive graph-support should rank only current possible matches before truncating."""
        class FakeLM:
            def __init__(self):
                self.graph_memory = SimpleNamespace(
                    get_memory_ids=lambda: ["target", "pm_a", "pm_b", "pm_c", "other"],
                )
                self.graph_support_competitive_scope = "top_k_possible_matches"
                self.graph_support_competitive_top_k = 2
                self.evidence = {
                    "target": np.array([0.95, 0.9]),
                    "pm_a": np.array([0.8, 0.81]),
                    "pm_b": np.array([0.86, 0.84]),
                    "pm_c": np.array([0.83, 0.82]),
                    "other": np.array([0.99, 0.98]),
                }
                self.get_possible_matches = lambda: ["target", "pm_a", "pm_b", "pm_c"]

            _get_graph_support_competitor_ids = (
                EvidenceSDRLMMixin._get_graph_support_competitor_ids
            )

        lm = FakeLM()

        competitor_ids = EvidenceSDRLMMixin._get_graph_support_competitor_ids(
            lm,
            "target",
        )

        self.assertEqual(competitor_ids, ["pm_b", "pm_c"])

    def test_graph_support_debug_payload_summarizes_bonus_updates(self):
        """Graph-support debug logging should keep compact per-graph summaries."""
        lm = SimpleNamespace(
            graph_support_debug_top_k=2,
            graph_support_objective_weight=2.0,
            graph_support_objective_tolerance=0.1,
            graph_support_competitive_penalty=0.85,
            graph_support_competitive_same_peak_only=True,
            graph_support_competitive_packet_context_floor=0.9,
            graph_support_competitive_packet_context_strict=True,
            graph_support_competitive_packet_context_strict_ceiling=0.9,
            graph_support_competitive_scope="all",
            graph_support_competitive_top_k=0,
        )

        EvidenceSDRLMMixin._reset_graph_support_debug_stats(lm)
        EvidenceSDRLMMixin._record_graph_support_debug_update(
            lm,
            graph_id="parent_a",
            graph_support_bonus=1.5,
            base_evidence=np.array([1.0, 2.0]),
            updated_evidence=np.array([2.5, 3.5]),
            channel_details=[
                {
                    "input_channel": "learning_module_0",
                    "support_similarity": 0.75,
                    "graph_support_norm": 1.0,
                    "query_support_norm": 1.0,
                }
            ],
        )
        EvidenceSDRLMMixin._record_graph_support_debug_update(
            lm,
            graph_id="parent_a",
            graph_support_bonus=0.0,
            base_evidence=np.array([0.5, 1.0]),
            updated_evidence=np.array([0.5, 1.0]),
            channel_details=[
                {
                    "input_channel": "learning_module_0",
                    "support_similarity": 0.2,
                    "graph_support_norm": 1.0,
                    "query_support_norm": 1.0,
                }
            ],
        )
        EvidenceSDRLMMixin._record_graph_support_debug_update(
            lm,
            graph_id="parent_b",
            graph_support_bonus=0.8,
            base_evidence=np.array([1.0, 1.0]),
            updated_evidence=np.array([1.8, 1.8]),
            channel_details=[
                {
                    "input_channel": "learning_module_0",
                    "support_similarity": 0.4,
                    "graph_support_norm": 1.0,
                    "query_support_norm": 1.0,
                }
            ],
        )

        payload = EvidenceSDRLMMixin._get_graph_support_debug_payload(lm)

        self.assertEqual(payload["update_count"], 3)
        self.assertEqual(payload["positive_bonus_count"], 2)
        self.assertAlmostEqual(payload["graph_support_competitive_penalty"], 0.85)
        self.assertTrue(payload["graph_support_competitive_same_peak_only"])
        self.assertAlmostEqual(
            payload["graph_support_competitive_packet_context_floor"],
            0.9,
        )
        self.assertTrue(payload["graph_support_competitive_packet_context_strict"])
        self.assertAlmostEqual(
            payload["graph_support_competitive_packet_context_strict_ceiling"],
            0.9,
        )
        self.assertEqual(payload["graphs"]["parent_a"]["update_count"], 2)
        self.assertEqual(payload["graphs"]["parent_a"]["positive_bonus_count"], 1)
        self.assertAlmostEqual(payload["graphs"]["parent_a"]["max_bonus"], 1.5)
        self.assertEqual(len(payload["top_updates"]), 2)
        self.assertEqual(payload["top_updates"][0]["graph_id"], "parent_a")

    def test_post_episode_logs_graph_support_debug_during_eval(self):
        """Eval episodes should emit graph-support debug logs even without SDR fitting."""

        class FakeParent:
            def post_episode(self, *args, **kwargs):
                self.parent_post_episode_called = True

        class FakeLM(EvidenceSDRLMMixin, FakeParent):
            pass

        logged_payloads = []
        lm = object.__new__(FakeLM)
        lm.mode = ExperimentMode.EVAL
        lm.train_sdr_on_eval = False
        lm.sdr_args = {"sdr_log_flag": True}
        lm.graph_support_objective = True
        lm.graph_support_objective_weight = 2.0
        lm.graph_support_objective_tolerance = 0.1
        lm.graph_support_debug_top_k = 5
        lm.upward_top_k = 1
        lm.upward_support_top_k = 1
        lm.upward_support_temperature = 1.0
        lm.obj2id = {"parent_a": 0}
        lm.id2obj = {0: "parent_a"}
        lm.sdr_encoder = SimpleNamespace(
            get_sdr=lambda object_index: np.array([1.0, 0.0, 0.0, 0.0])
        )
        lm.get_current_mlh = lambda: {"graph_id": "parent_a"}
        lm.get_evidence_for_each_graph = lambda: (["parent_a"], np.array([3.0]))
        lm.tmp_logger = SimpleNamespace(log_episode=lambda data: logged_payloads.append(data))

        EvidenceSDRLMMixin._reset_graph_support_debug_stats(lm)
        EvidenceSDRLMMixin._record_graph_support_debug_update(
            lm,
            graph_id="parent_a",
            graph_support_bonus=1.0,
            base_evidence=np.array([1.0, 2.0]),
            updated_evidence=np.array([2.0, 3.0]),
            channel_details=[
                {
                    "input_channel": "learning_module_0",
                    "support_similarity": 0.5,
                    "graph_support_norm": 1.0,
                    "query_support_norm": 1.0,
                }
            ],
        )

        lm.post_episode()

        self.assertTrue(lm.parent_post_episode_called)
        self.assertEqual(len(logged_payloads), 1)
        self.assertIn("upward_hypothesis_packet", logged_payloads[0])
        self.assertIn("graph_support_debug", logged_payloads[0])
        self.assertEqual(logged_payloads[0]["mode"], "EVAL")

    def test_sdr_feature_evidence_uses_query_self_overlap_for_mixtures(self):
        """Mixture carriers should still normalize to full evidence against themselves."""
        query = np.array([1.0, 1.0, 0.0, 0.0])
        overlaps = SDRFeatureEvidenceCalculator.calculate_feature_evidence_sdr_for_all_nodes(
            channel_feature_array=np.array([
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ]),
            channel_feature_order=["object_id"],
            channel_feature_weights={"object_id": 1.0},
            channel_query_features={"object_id": query},
            channel_tolerances={"object_id": 0.0},
        )

        np.testing.assert_allclose(overlaps[0], 1.0)
        np.testing.assert_allclose(overlaps[1], 0.5)

    def test_sdr_feature_evidence_treats_ranked_object_ids_as_sdr_features(self):
        """Packet rank slots should use overlap scoring rather than dense similarity."""
        query = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        overlaps = SDRFeatureEvidenceCalculator.calculate_feature_evidence_sdr_for_all_nodes(
            channel_feature_array=np.array([
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            ]),
            channel_feature_order=["object_id_rank_0"],
            channel_feature_weights={"object_id_rank_0": 1.0},
            channel_query_features={"object_id_rank_0": query},
            channel_tolerances={"object_id_rank_0": 2.0},
        )

        np.testing.assert_allclose(overlaps[0], 1.0)
        np.testing.assert_allclose(overlaps[1], 0.0)

    def test_sdr_feature_evidence_uses_object_support_for_parent_scoring(self):
        """Parent evidence should reward matching child-support distributions."""
        query_features = {
            "object_id": np.array([1.0, 1.0, 0.0, 0.0]),
            "object_support": np.array([0.5, 0.5, 0.0, 0.0]),
        }
        overlaps = SDRFeatureEvidenceCalculator.calculate_feature_evidence_sdr_for_all_nodes(
            channel_feature_array=np.array([
                [1.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ]),
            channel_feature_order=["object_id", "object_support"],
            channel_feature_weights={"object_id": 1.0, "object_support": 1.0},
            channel_query_features=query_features,
            channel_tolerances={"object_id": 0.0, "object_support": 0.0},
        )

        self.assertGreater(overlaps[0], overlaps[1])

    def test_sdr_feature_evidence_uses_rank_weights_for_packet_scoring(self):
        """Packet support-weight features should distinguish otherwise identical ranks."""
        query_features = {
            "object_id_rank_0": np.array([1.0, 1.0, 0.0, 0.0]),
            "object_rank_weight_0": np.array([0.8]),
        }
        overlaps = SDRFeatureEvidenceCalculator.calculate_feature_evidence_sdr_for_all_nodes(
            channel_feature_array=np.array([
                [1.0, 1.0, 0.0, 0.0, 0.8],
                [1.0, 1.0, 0.0, 0.0, 0.2],
            ]),
            channel_feature_order=["object_id_rank_0", "object_rank_weight_0"],
            channel_feature_weights={
                "object_id_rank_0": 1.0,
                "object_rank_weight_0": 1.0,
            },
            channel_query_features=query_features,
            channel_tolerances={
                "object_id_rank_0": 0.0,
                "object_rank_weight_0": 0.0,
            },
        )

        self.assertGreater(overlaps[0], overlaps[1])

    def test_single_worker_parallel_load_preserves_full_lm_state(self):
        """Single-worker phase0 consolidation should not strip SDR-only LM fields."""

        class DummyMonty:
            def __init__(self):
                self.loaded_state = None

            def load_state_dict(self, state_dict):
                self.loaded_state = state_dict

        state_dict = {
            "lm_dict": {
                0: {
                    "graph_memory": {"mug": {"nodes": 1}},
                    "target_to_graph_id": {"mug": {"mug"}},
                    "graph_id_to_target": {"mug": {"mug"}},
                    "obj2id": {"mug": 0},
                    "id2obj": {0: "mug"},
                    "target_overlaps": {
                        "overlaps": np.array([[5.0]]),
                        "counts": np.array([[1.0]]),
                    },
                    "sdr_encoder": {
                        "sdr_length": 8,
                        "sdr_on_bits": 2,
                        "lr": 1e-2,
                        "n_epochs": 10,
                        "stability": 0.0,
                        "log_flag": False,
                        "obj_sdrs": np.ones((1, 8)),
                        "stable_ids": np.array([], dtype=int),
                    },
                }
            },
            "sm_dict": {},
            "motor_system_dict": {},
            "lm_to_lm_matrix": [],
            "lm_to_lm_vote_matrix": [],
            "sm_to_lm_matrix": [],
        }

        dummy = DummyMonty()
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "0"
            run_dir.mkdir()
            torch.save(state_dict, run_dir / "model.pt")

            MontyForGraphMatching.load_state_dict_from_parallel(
                dummy, [run_dir], save=True
            )

            self.assertIn("sdr_encoder", dummy.loaded_state["lm_dict"][0])
            self.assertIn("obj2id", dummy.loaded_state["lm_dict"][0])

            saved_state = torch.load(Path(tmpdir) / "model.pt")
            self.assertIn("sdr_encoder", saved_state["lm_dict"][0])
            self.assertIn("obj2id", saved_state["lm_dict"][0])
