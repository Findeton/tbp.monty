# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Integration smoke tests for the Track 14 predictive hypothesis LM family.

These tests exercise the new Track 14 path only through the full Monty
hierarchy with real Panda3D-rendered GLB assets. The goal is to verify that
the experiment wiring, richer sensor packets, and multi-hypothesis learning
stay live in the real runtime rather than in an isolated toy harness.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path


def _configure_track14_test_threading() -> None:
    # Match the Track 14 benchmark CLI's one-thread startup so the fixed
    # sparse recurrent backend avoids the known macOS/torch 1.13.1 crash path.
    for env_name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(env_name, "1")

    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch only allows setting interop threads once per process.
        pass


_configure_track14_test_threading()

try:
    import panda3d  # noqa: F401
    import gltf  # noqa: F401
except ImportError:
    raise unittest.SkipTest("Panda3D or panda3d-gltf not installed")

from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.models.cortical_column_torch.experiment import (
    Panda3DTorchExperiment,
)
from tbp.monty.frameworks.models.predictive_hypothesis_torch import (
    DetailAwareCameraSM,
    DetailAwareChangeDetectingSM,
    PredictiveContextSignal,
    PredictiveVoteMessage,
    PredictiveHypothesisTorchLM,
    TemporalState,
)


_TEST_ROOT = Path(__file__).resolve().parents[3]
ASSET_DIR = str(
    _TEST_ROOT / "unit" / "simulators" / "panda3d" / "test_assets" / "animated"
)
FOX_PATH = str(Path(ASSET_DIR) / "Fox.glb")
ROBOT_PATH = str(Path(ASSET_DIR) / "RobotExpressive.glb")


def _assets_available():
    return os.path.isfile(FOX_PATH) and os.path.isfile(ROBOT_PATH)


def _track14_kwargs():
    shared_lm_kwargs = {
        "output_evidence_threshold": 0.0,
        "core_kwargs": {
            "max_hypotheses": 4,
        },
    }
    return {
        "hierarchical": True,
        "lm_family": "predictive_hypothesis_torch",
        "resolution": (24, 24),
        "initial_distance": 2.0,
        "object_scale": (0.01, 0.01, 0.01),
        "asset_search_paths": [ASSET_DIR],
        "detail_grid_shape": (2, 2),
        "column_kwargs": {"n_minicolumns": 128, "seed": 42},
        "parent_column_kwargs": {"n_minicolumns": 128, "seed": 44},
        "morphology_lm_kwargs": dict(shared_lm_kwargs),
        "behavior_lm_kwargs": dict(shared_lm_kwargs),
        "parent_lm_kwargs": dict(shared_lm_kwargs),
    }


def _state_provider(state_id):
    def provider(**kwargs):
        return {
            "lm_morphology": state_id,
            "lm_behavior": state_id,
            "lm_parent": state_id,
        }

    return provider


@unittest.skipUnless(_assets_available(), "Real 3D model assets not found")
class TestPredictiveHypothesisTorchMonty(unittest.TestCase):
    def test_hierarchical_setup_uses_track14_family(self):
        exp = Panda3DTorchExperiment(
            model_path=FOX_PATH,
            **_track14_kwargs(),
        )
        try:
            exp._setup()

            self.assertEqual(len(exp.monty.sensor_modules), 2)
            self.assertIsInstance(exp.monty.sensor_modules[0], DetailAwareCameraSM)
            self.assertIsInstance(
                exp.monty.sensor_modules[1],
                DetailAwareChangeDetectingSM,
            )

            self.assertEqual(len(exp.monty.learning_modules), 3)
            for lm in exp.monty.learning_modules:
                self.assertIsInstance(lm, PredictiveHypothesisTorchLM)

            self.assertEqual(exp.monty.sm_to_lm_matrix, [[0], [1], []])
            self.assertEqual(exp.monty.lm_to_lm_matrix, [[], [], [0, 1]])
            self.assertEqual(exp.monty.lm_to_lm_vote_matrix, [[1], [0], []])
        finally:
            exp.close()

    def test_hierarchical_real_asset_training_builds_detail_and_hypotheses(self):
        exp = Panda3DTorchExperiment(
            model_path=FOX_PATH,
            **_track14_kwargs(),
        )
        try:
            fox_result = exp.run_episode(
                mode=ExperimentMode.TRAIN,
                object_name="fox",
                n_steps=8,
                state_provider=_state_provider(0),
            )
            self.assertEqual(fox_result["total_steps"], 8)

            exp.swap_model(ROBOT_PATH, object_scale=(0.3, 0.3, 0.3))
            robot_result = exp.run_episode(
                mode=ExperimentMode.TRAIN,
                object_name="robot",
                n_steps=8,
                state_provider=_state_provider(1),
            )
            self.assertEqual(robot_result["total_steps"], 8)

            eval_result = exp.run_episode(
                mode=ExperimentMode.EVAL,
                n_steps=6,
                state_provider=_state_provider(1),
            )
            self.assertEqual(eval_result["total_steps"], 6)
            self.assertIn("evidence", eval_result)

            lm_morph, lm_behav, lm_parent = exp.monty.learning_modules

            for lm in (lm_morph, lm_behav, lm_parent):
                self.assertTrue(lm._stepped, f"{lm.learning_module_id} should step")
                self.assertGreaterEqual(
                    len(lm.get_hypothesis_bank()),
                    2,
                    f"{lm.learning_module_id} should retain multiple hypotheses",
                )

            for lm in (lm_morph, lm_behav, lm_parent):
                known = set(lm.get_all_known_object_ids())
                self.assertTrue(
                    len(known) >= 2
                    and all(str(object_id).startswith("latent_object_") for object_id in known),
                    f"Expected latent tracked ids, got {known}",
                )
                mapped_targets = {
                    target
                    for targets in lm.graph_id_to_target.values()
                    for target in targets
                }
                self.assertTrue(
                    {"fox", "robot"}.issubset(mapped_targets),
                    f"Expected report-time target mapping for fox/robot, got {mapped_targets}",
                )

            morph_packet = lm_morph.get_last_detail_packet()
            behav_packet = lm_behav.get_last_detail_packet()
            parent_packet = lm_parent.get_last_detail_packet()
            morph_field = lm_morph.get_last_observation_field()
            behav_field = lm_behav.get_last_observation_field()
            parent_field = lm_parent.get_last_observation_field()
            morph_embeddings = lm_morph.get_last_embeddings()
            parent_slots = lm_parent.get_memory_slots()
            behavior_change_slots = lm_behav.get_change_memory_slots()
            parent_change_slots = lm_parent.get_change_memory_slots()
            parent_retrieval = lm_parent.get_last_memory_retrieval()
            parent_change_retrieval = lm_parent.get_last_change_memory_retrieval()
            parent_stream_retrievals = lm_parent.get_last_stream_memory_retrievals()
            parent_hypothesis_state = lm_parent.get_hypothesis_state()
            parent_ranked_hypotheses = lm_parent.get_hypothesis_bank()

            self.assertIsNotNone(morph_packet)
            self.assertIsNotNone(behav_packet)
            self.assertIsNotNone(parent_packet)
            self.assertEqual(morph_packet["packet_type"], "visual_observation_packet_v2")
            self.assertEqual(behav_packet["packet_type"], "change_observation_packet_v2")
            self.assertEqual(parent_packet["packet_type"], "lm_fusion_packet")
            self.assertEqual(morph_packet["packet_version"], 2)
            self.assertEqual(behav_packet["packet_version"], 2)
            self.assertEqual(morph_packet["micro_patch_shape"], [4, 4])
            self.assertEqual(behav_packet["micro_patch_shape"], [4, 4])
            self.assertIn("cells", morph_packet)
            self.assertIn("cells", behav_packet)
            self.assertGreater(morph_packet["shard_count"], 0)
            self.assertGreater(behav_packet["shard_count"], 0)
            self.assertGreater(parent_packet["shard_count"], 0)
            self.assertNotIn("semantic_id", morph_packet["cells"][0])
            self.assertNotIn("semantic_id", behav_packet["cells"][0])

            self.assertEqual(morph_field.packet_type, "visual_observation_packet_v2")
            self.assertEqual(behav_field.packet_type, "change_observation_packet_v2")
            self.assertEqual(parent_field.packet_type, "lm_fusion_packet")
            self.assertEqual(morph_field.cell_count, morph_packet["cell_count"])
            self.assertEqual(behav_field.cell_count, behav_packet["cell_count"])
            self.assertGreater(parent_field.child_count, 0)
            self.assertEqual(tuple(morph_field.rgb.shape[1:]), (4, 4, 3))
            self.assertEqual(tuple(behav_field.xyz.shape[1:]), (4, 4, 3))
            self.assertEqual(tuple(morph_embeddings.joint.shape), (128,))
            self.assertGreaterEqual(parent_slots.num_slots, 2)
            self.assertGreaterEqual(behavior_change_slots.num_slots, 2)
            self.assertGreaterEqual(parent_change_slots.num_slots, 2)
            self.assertEqual(len(parent_slots.slot_chart_ids), parent_slots.num_slots)
            self.assertEqual(
                len(parent_change_slots.slot_chart_ids),
                parent_change_slots.num_slots,
            )
            self.assertGreaterEqual(parent_hypothesis_state.size, 2)
            self.assertEqual(
                len(parent_hypothesis_state.chart_ids),
                parent_hypothesis_state.size,
            )
            self.assertEqual(
                len(parent_ranked_hypotheses),
                parent_hypothesis_state.size,
            )
            self.assertLess(
                len(set(parent_hypothesis_state.object_ids)),
                parent_hypothesis_state.size,
            )
            self.assertLess(
                len(
                    {
                        str(hypothesis["latent_id"])
                        for hypothesis in parent_ranked_hypotheses
                    }
                ),
                len(parent_ranked_hypotheses),
            )
            self.assertGreater(
                len(
                    {
                        str(chart_id)
                        for chart_id in parent_hypothesis_state.chart_ids
                        if chart_id is not None
                    }
                ),
                1,
            )
            self.assertIsNotNone(parent_retrieval.top_chart_id)
            self.assertIsNotNone(parent_change_retrieval.top_chart_id)
            self.assertEqual(
                set(parent_stream_retrievals.keys()),
                {"appearance", "change"},
            )
            self.assertGreaterEqual(
                parent_stream_retrievals["appearance"].iteration_count,
                1,
            )
            self.assertGreaterEqual(
                parent_stream_retrievals["change"].iteration_count,
                1,
            )

            morph_debug = lm_morph.get_evidence_debug()
            self.assertIn("base_evidence", morph_debug)
            self.assertIn("after_temporal_behavior_evidence", morph_debug)
            self.assertIn("final_evidence", morph_debug)
            self.assertIn("winner_path", morph_debug)
            self.assertIn("action_prediction_error", morph_debug)
            self.assertIn("prediction_mismatch", morph_debug)
            self.assertIn("memory_stream_scores", morph_debug)
            self.assertIn("memory_stream_weights", morph_debug)
            self.assertIn("appearance", morph_debug["memory_retrieval"])
            self.assertIn("change", morph_debug["memory_retrieval"])

            self.assertGreaterEqual(len(lm_morph.evidence), 2)
            self.assertGreaterEqual(len(lm_behav.evidence), 2)
            self.assertTrue(
                all(str(object_id).startswith("latent_object_") for object_id in lm_morph.evidence)
            )
            self.assertTrue(
                all(str(object_id).startswith("latent_object_") for object_id in lm_behav.evidence)
            )
            self.assertTrue(str(eval_result["graph_id"]).startswith("latent_object_"))
            self.assertIn("robot", lm_morph.graph_id_to_target[eval_result["graph_id"]])

            parent_signal = lm_parent.get_context_signal()
            parent_context_message = lm_parent.get_context_message()
            morph_received_context = lm_morph.get_last_received_context_message()
            behav_received_context = lm_behav.get_last_received_context_message()
            parent_received_context = lm_parent.get_last_received_context_message()
            morph_received_votes = lm_morph.get_last_received_vote_messages()
            parent_temporal_state = lm_parent.get_temporal_state()
            morph_vote_message = lm_morph.get_last_vote_message()
            self.assertIsNotNone(parent_signal)
            self.assertIn("active_cells", parent_signal)
            self.assertGreater(len(parent_signal["active_cells"]), 0)
            self.assertIsNone(parent_signal.get("graph_id"))
            self.assertEqual(parent_signal.get("routing_scope"), "targeted")
            self.assertEqual(
                set(parent_signal.get("target_sender_ids", [])),
                {lm_morph.learning_module_id, lm_behav.learning_module_id},
            )
            self.assertIsInstance(parent_context_message, PredictiveContextSignal)
            self.assertIsInstance(morph_received_context, PredictiveContextSignal)
            self.assertIsInstance(behav_received_context, PredictiveContextSignal)
            self.assertIsNone(morph_received_context.graph_id)
            self.assertEqual(morph_received_context.sender_id, lm_parent.learning_module_id)
            self.assertEqual(behav_received_context.sender_id, lm_parent.learning_module_id)
            self.assertIn(
                parent_received_context.sender_id,
                {lm_morph.learning_module_id, lm_behav.learning_module_id},
            )
            self.assertGreaterEqual(len(morph_received_votes), 1)
            self.assertIsInstance(morph_received_votes[0], PredictiveVoteMessage)
            self.assertGreater(morph_received_votes[0].active_cells.numel(), 0)
            self.assertIsInstance(parent_temporal_state, TemporalState)
            self.assertGreater(parent_temporal_state.trace_norm, 0.0)
            self.assertIsInstance(morph_vote_message, PredictiveVoteMessage)
            self.assertIsNotNone(lm_morph.get_temporal_prediction_status())
            self.assertIsNotNone(lm_behav.get_temporal_prediction_status())
        finally:
            exp.close()


if __name__ == "__main__":
    unittest.main()