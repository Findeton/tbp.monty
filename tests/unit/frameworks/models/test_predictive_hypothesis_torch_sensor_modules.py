import copy
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from tbp.monty.frameworks.models.evidence_matching.model import (
    MontyForEvidenceGraphMatching,
)
from tbp.monty.frameworks.models.graph_matching import MontyForGraphMatching
from tbp.monty.frameworks.models.predictive_hypothesis_torch.core import (
    AtlasMemory,
    DetailPacketEncoder,
    HypothesisBank,
    PredictiveHypothesisCore,
)
from tbp.monty.frameworks.models.predictive_hypothesis_torch.sparse_recurrent_memory import (
    FixedSparseRecurrentMemory,
)
from tbp.monty.frameworks.models.predictive_hypothesis_torch.learning_module import (
    PredictiveHypothesisTorchLM,
)
from tbp.monty.frameworks.models.predictive_hypothesis_torch.tensor_objects import (
    HopfieldRetrievalState,
    ObservationEmbeddings,
    ObservationField,
    PredictiveContextSignal,
    PredictiveVoteMessage,
    RankedHypothesisVote,
    TemporalState,
)
from tbp.monty.frameworks.models.predictive_hypothesis_torch.sensor_modules import (
    build_change_detail_packet,
    build_visual_detail_packet,
)
from tbp.monty.frameworks.models.states import State


def _make_state(
    non_morphological_features=None,
    *,
    location=None,
    pose_vectors=None,
):
    return State(
        location=np.array(
            [0.1, 0.2, 0.3] if location is None else location,
            dtype=np.float32,
        ),
        morphological_features={
            "pose_vectors": np.eye(3, dtype=np.float32)
            if pose_vectors is None
            else np.asarray(pose_vectors, dtype=np.float32),
            "pose_fully_defined": True,
            "on_object": 1.0,
        },
        non_morphological_features=dict(non_morphological_features or {}),
        confidence=0.9,
        use_state=True,
        sender_id="sensor.test",
        sender_type="SM",
    )


def _make_observation(height=10, width=10):
    row_coords, col_coords = np.indices((height, width))
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., 0] = (row_coords * 17 + col_coords * 9) % 255
    rgba[..., 1] = (row_coords * 13 + 40) % 255
    rgba[..., 2] = (col_coords * 19 + 20) % 255
    rgba[..., 3] = 255

    depth = (0.5 + 0.01 * row_coords + 0.02 * col_coords).astype(np.float32)
    support = (
        (row_coords >= 2)
        & (row_coords < height - 1)
        & (col_coords >= 1)
        & (col_coords < width - 2)
    ).astype(np.float32)

    sensor_xyz = np.stack(
        [
            col_coords.astype(np.float32) / float(max(width - 1, 1)),
            row_coords.astype(np.float32) / float(max(height - 1, 1)),
            depth,
        ],
        axis=-1,
    )
    sensor_frame_data = np.concatenate(
        [sensor_xyz, support[..., None]],
        axis=-1,
    ).reshape(height * width, 4)

    semantic_ids = support * 7.0
    semantic_3d = np.concatenate(
        [sensor_xyz + np.array([1.0, 2.0, 3.0], dtype=np.float32), semantic_ids[..., None]],
        axis=-1,
    ).reshape(height * width, 4)

    world_camera = np.eye(4, dtype=np.float32)
    world_camera[:3, 3] = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    return {
        "rgba": rgba,
        "depth": depth.reshape(-1),
        "sensor_frame_data": sensor_frame_data,
        "semantic_3d": semantic_3d,
        "world_camera": world_camera,
    }


def _make_texture_field(kind: str, jitter: float = 0.0) -> ObservationField:
    device = torch.device("cpu")
    field = ObservationField.empty(
        device=device,
        packet_type="visual_observation_packet_v2",
        micro_patch_shape=(4, 4),
        grid_shape=(1, 1),
    )

    if kind == "horizontal":
        pattern = np.tile(np.array([[1.0], [0.0], [1.0], [0.0]], dtype=np.float32), (1, 4))
    elif kind == "vertical":
        pattern = np.tile(np.array([[1.0, 0.0, 1.0, 0.0]], dtype=np.float32), (4, 1))
    else:
        raise ValueError(f"Unsupported texture kind: {kind}")

    if jitter > 0.0:
        row_coords, col_coords = np.indices(pattern.shape)
        noise = (((row_coords * 3) + (col_coords * 5)) % 7).astype(np.float32) / 7.0
        pattern = np.clip(pattern + (jitter * (noise - 0.5)), 0.0, 1.0)

    rgb = np.stack(
        [
            pattern,
            1.0 - pattern,
            0.5 * np.ones_like(pattern, dtype=np.float32),
        ],
        axis=-1,
    )
    field.rgb = torch.as_tensor(rgb[None, ...], dtype=torch.float32, device=device)
    field.depth = torch.zeros((1, 4, 4), dtype=torch.float32, device=device)
    field.support = torch.ones((1, 4, 4), dtype=torch.float32, device=device)
    field.xyz = torch.zeros((1, 4, 4, 3), dtype=torch.float32, device=device)
    field.uv = torch.tensor([[0.5, 0.5]], dtype=torch.float32, device=device)
    field.valid_cells = torch.tensor([1.0], dtype=torch.float32, device=device)
    field.flow_direction = torch.zeros(3, dtype=torch.float32, device=device)
    field.flow_magnitude = torch.zeros(1, dtype=torch.float32, device=device)
    field.temporal_feature_stats = torch.zeros((0, 3), dtype=torch.float32, device=device)
    field.anchor_location = torch.zeros(3, dtype=torch.float32, device=device)
    field.anchor_pose = torch.eye(3, dtype=torch.float32, device=device)
    field.anchor_confidence = torch.ones(1, dtype=torch.float32, device=device)
    field.anchor_on_object = torch.ones(1, dtype=torch.float32, device=device)
    field.camera_forward = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)
    return field


def _packet_has_semantic_key(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if "semantic" in str(key):
                return True
            if _packet_has_semantic_key(item):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(_packet_has_semantic_key(item) for item in value)
    return False


class TestPredictiveHypothesisSensorPackets(unittest.TestCase):
    def test_visual_packet_emits_micro_patches_and_soft_support(self):
        observation = _make_observation()
        state = _make_state({"hsv": np.array([0.1, 0.2, 0.3], dtype=np.float32)})

        packet = build_visual_detail_packet(
            observation,
            sender_id="camera",
            grid_shape=(5, 5),
            state=state,
        )

        self.assertEqual(packet["packet_type"], "visual_observation_packet_v2")
        self.assertEqual(packet["packet_version"], 2)
        self.assertEqual(packet["grid_shape"], [5, 5])
        self.assertEqual(packet["micro_patch_shape"], [4, 4])
        self.assertEqual(packet["cell_count"], 25)
        self.assertFalse(_packet_has_semantic_key(packet))

        first_cell = packet["cells"][0]
        self.assertEqual(first_cell["rgb_patch"].shape, (4, 4, 3))
        self.assertEqual(first_cell["depth_patch"].shape, (4, 4))
        self.assertEqual(first_cell["support_patch"].shape, (4, 4))
        self.assertEqual(first_cell["sensor_frame_patch"].shape, (4, 4, 3))
        self.assertGreaterEqual(float(first_cell["support_patch"].min()), 0.0)
        self.assertLessEqual(float(first_cell["support_patch"].max()), 1.0)
        self.assertIn("camera_pose_world", packet["sensor_context"])
        self.assertIn("location", packet["state_anchor"])

    def test_change_packet_carries_temporal_context_without_semantic_ids(self):
        observation = _make_observation()
        state = _make_state(
            {
                "flow_direction": np.array([0.0, 1.0, 0.0], dtype=np.float32),
                "flow_magnitude": np.array([0.35], dtype=np.float32),
                "delta_hsv": np.array([0.1, -0.2, 0.3], dtype=np.float32),
            }
        )

        packet = build_change_detail_packet(
            observation,
            state,
            sender_id="change",
            grid_shape=(5, 5),
        )

        self.assertEqual(packet["packet_type"], "change_observation_packet_v2")
        self.assertEqual(packet["packet_version"], 2)
        self.assertFalse(_packet_has_semantic_key(packet))
        self.assertIn("temporal_context", packet)
        np.testing.assert_allclose(
            packet["temporal_context"]["flow_direction"],
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )
        self.assertAlmostEqual(packet["temporal_context"]["flow_magnitude"], 0.35, places=6)
        self.assertIn("hsv", packet["temporal_context"]["feature_deltas"])

    def test_visual_packet_can_omit_world_pose_metadata(self):
        observation = _make_observation()
        state = _make_state({"hsv": np.array([0.1, 0.2, 0.3], dtype=np.float32)})

        packet = build_visual_detail_packet(
            observation,
            sender_id="camera",
            grid_shape=(5, 5),
            state=state,
            include_world_pose=False,
        )

        self.assertNotIn("camera_pose_world", packet["sensor_context"])
        self.assertNotIn("location", packet["state_anchor"])
        self.assertNotIn("pose_vectors", packet["state_anchor"])
        self.assertIn("confidence", packet["state_anchor"])
        self.assertIn("on_object", packet["state_anchor"])

    def test_encoder_prefers_observation_packet_v2(self):
        observation = _make_observation()
        state = _make_state()
        packet = build_visual_detail_packet(
            observation,
            sender_id="camera",
            grid_shape=(5, 5),
            state=state,
        )
        lm_state = _make_state({"observation_packet_v2": packet})

        encoder = DetailPacketEncoder(context_dim=32)
        encoded_packet, embedding = encoder.encode_state(lm_state)

        self.assertEqual(encoded_packet["packet_type"], "visual_observation_packet_v2")
        self.assertEqual(encoded_packet["cell_count"], 25)
        self.assertEqual(tuple(embedding.shape), (32,))
        self.assertTrue(torch_all_finite(embedding.detach().cpu().numpy()))

    def test_encoder_builds_tensor_native_observation_field_and_embeddings(self):
        observation = _make_observation()
        state = _make_state()
        packet = build_change_detail_packet(
            observation,
            state,
            sender_id="change",
            grid_shape=(5, 5),
        )
        lm_state = _make_state({"observation_packet_v2": packet})

        encoder = DetailPacketEncoder(context_dim=48)
        encoded_packet, observation_field, embeddings = encoder.encode_state_to_tensors(
            lm_state
        )

        self.assertEqual(encoded_packet["packet_type"], "change_observation_packet_v2")
        self.assertIsInstance(observation_field, ObservationField)
        self.assertIsInstance(embeddings, ObservationEmbeddings)
        self.assertEqual(observation_field.cell_count, 25)
        self.assertEqual(tuple(observation_field.rgb.shape), (25, 4, 4, 3))
        self.assertEqual(tuple(observation_field.depth.shape), (25, 4, 4))
        self.assertEqual(tuple(observation_field.support.shape), (25, 4, 4))
        self.assertEqual(tuple(observation_field.xyz.shape), (25, 4, 4, 3))
        self.assertEqual(tuple(observation_field.flow_direction.shape), (3,))
        self.assertEqual(tuple(observation_field.flow_magnitude.shape), (1,))
        self.assertEqual(tuple(embeddings.appearance.shape), (16,))
        self.assertEqual(tuple(embeddings.geometry.shape), (16,))
        self.assertEqual(tuple(embeddings.temporal.shape), (16,))
        self.assertEqual(tuple(embeddings.joint.shape), (48,))
        self.assertTrue(torch_all_finite(embeddings.joint.detach().cpu().numpy()))

    def test_core_tracks_temporal_trace_bank_and_typed_context_signal(self):
        observation = _make_observation()
        state = _make_state()
        packet = build_visual_detail_packet(
            observation,
            sender_id="camera",
            grid_shape=(5, 5),
            state=state,
        )
        lm_state = _make_state({"observation_packet_v2": packet})

        core = PredictiveHypothesisCore(context_dim=48)
        core.pre_episode(mode=None, object_name="fox:0")
        core.step(lm_state, learn=True)
        result = core.step(lm_state, learn=False)

        temporal_state = core.get_temporal_state()
        context_message = core.get_context_signal_message()
        temporal_context = core.get_temporal_context()
        retrieval_state = core.get_last_memory_retrieval()
        hypothesis_state = core.get_hypothesis_state()

        self.assertIsInstance(temporal_state, TemporalState)
        self.assertIsInstance(retrieval_state, HopfieldRetrievalState)
        self.assertEqual(tuple(temporal_state.trace_bank.shape), (3, 48))
        self.assertEqual(tuple(temporal_state.context_vector.shape), (48,))
        self.assertGreater(temporal_state.trace_norm, 0.0)
        self.assertGreater(temporal_state.predicted_appearance_signature.numel(), 0)
        self.assertGreater(temporal_state.predicted_change_signature.numel(), 0)
        self.assertGreaterEqual(temporal_state.dwell_step_count, 1)
        self.assertEqual(
            temporal_state.current_latent_id,
            temporal_state.current_graph_id,
        )
        self.assertEqual(
            temporal_state.known_latent_ids,
            temporal_state.known_states,
        )
        self.assertGreaterEqual(retrieval_state.iteration_count, 1)
        self.assertIsNotNone(retrieval_state.top_chart_id)
        self.assertEqual(len(hypothesis_state.chart_ids), hypothesis_state.size)
        self.assertIsInstance(context_message, PredictiveContextSignal)
        self.assertEqual(tuple(context_message.active_cells.shape), (48,))
        self.assertTrue(
            torch_all_finite(context_message.active_cells.detach().cpu().numpy())
        )
        self.assertGreater(context_message.predicted_appearance_signature.numel(), 0)
        self.assertGreater(context_message.predicted_change_signature.numel(), 0)
        self.assertIn("boundary_pressure", temporal_context)
        self.assertIn("mean_surprise", temporal_context)
        self.assertIn("trace_norm", temporal_context)
        self.assertIn("event_detected", temporal_context)
        self.assertIn("dwell_steps", temporal_context)
        self.assertIn("timescale_trace_norms", temporal_context)
        self.assertIn("appearance_prediction_error", temporal_context)
        self.assertIn("change_prediction_error", temporal_context)
        self.assertIn("predicted_appearance_norm", temporal_context)
        self.assertIn("predicted_change_norm", temporal_context)
        self.assertIn("current_latent_id", temporal_context)
        self.assertIn("current_graph_id", temporal_context)
        self.assertEqual(
            temporal_context["current_latent_id"],
            temporal_context["current_graph_id"],
        )
        self.assertIn("known_latent_ids", temporal_context)
        self.assertEqual(
            temporal_context["known_latent_ids"],
            temporal_context["known_states"],
        )
        self.assertIn("active_self_supervised_latent_id", temporal_context)
        self.assertEqual(
            temporal_context["active_self_supervised_latent_id"],
            temporal_context["active_self_supervised_object_id"],
        )
        self.assertIn("appearance_prediction_error", result)
        self.assertIn("change_prediction_error", result)
        self.assertGreaterEqual(result["appearance_prediction_error"], 0.0)
        self.assertGreaterEqual(result["change_prediction_error"], 0.0)
        self.assertEqual(len(temporal_context["timescale_trace_norms"]), 3)

    def test_core_exposes_split_appearance_and_change_memories(self):
        observation = _make_observation()
        state = _make_state(
            {
                "flow_direction": np.array([0.0, 1.0, 0.0], dtype=np.float32),
                "flow_magnitude": np.array([0.35], dtype=np.float32),
                "delta_hsv": np.array([0.1, -0.2, 0.3], dtype=np.float32),
            }
        )
        packet = build_change_detail_packet(
            observation,
            state,
            sender_id="change",
            grid_shape=(5, 5),
        )
        lm_state = _make_state(
            {
                "flow_direction": np.array([0.0, 1.0, 0.0], dtype=np.float32),
                "flow_magnitude": np.array([0.35], dtype=np.float32),
                "delta_hsv": np.array([0.1, -0.2, 0.3], dtype=np.float32),
                "observation_packet_v2": packet,
            }
        )

        core = PredictiveHypothesisCore(context_dim=48)
        core.pre_episode(mode=None, object_name="robot:1")
        core.step(lm_state, learn=True)
        result = core.step(lm_state, learn=False)

        appearance_slots = core.get_memory_slots()
        change_slots = core.get_change_memory_slots()
        appearance_retrieval = core.get_last_memory_retrieval()
        change_retrieval = core.get_last_change_memory_retrieval()
        stream_retrievals = core.get_last_stream_memory_retrievals()
        memory_summary = result["memory_retrieval"]
        evidence_debug = core.get_evidence_debug()
        raw_known_ids = core.get_all_known_object_ids()

        self.assertEqual(len(raw_known_ids), 1)
        self.assertTrue(raw_known_ids[0].startswith("latent_object_"))
        self.assertGreaterEqual(appearance_slots.num_slots, 1)
        self.assertGreaterEqual(change_slots.num_slots, 1)
        self.assertIsInstance(appearance_retrieval, HopfieldRetrievalState)
        self.assertIsInstance(change_retrieval, HopfieldRetrievalState)
        self.assertGreaterEqual(appearance_retrieval.iteration_count, 1)
        self.assertGreaterEqual(change_retrieval.iteration_count, 1)
        self.assertEqual(set(stream_retrievals.keys()), {"appearance", "change"})
        self.assertIn("appearance", memory_summary)
        self.assertIn("change", memory_summary)
        self.assertIn("stream_weights", memory_summary)
        self.assertEqual(
            memory_summary["appearance"]["top_object_id"],
            raw_known_ids[0],
        )
        self.assertEqual(
            memory_summary["change"]["top_object_id"],
            raw_known_ids[0],
        )
        self.assertEqual(result["mlh"]["graph_id"], raw_known_ids[0])
        self.assertIn(raw_known_ids[0], result["evidence"])
        self.assertIn("memory_stream_scores", evidence_debug)
        self.assertIn("memory_stream_weights", evidence_debug)
        self.assertIn("combined", evidence_debug["memory_stream_scores"])
        self.assertGreater(
            evidence_debug["memory_stream_weights"]["appearance"],
            evidence_debug["memory_stream_weights"]["change"],
        )

    def test_core_hypothesis_bank_exposes_joint_chart_pose_and_behavior_state(self):
        observation = _make_observation()
        state = _make_state(
            {
                "flow_direction": np.array([0.0, 1.0, 0.0], dtype=np.float32),
                "flow_magnitude": np.array([0.45], dtype=np.float32),
                "delta_hsv": np.array([0.1, -0.2, 0.3], dtype=np.float32),
            }
        )
        packet = build_change_detail_packet(
            observation,
            state,
            sender_id="change",
            grid_shape=(5, 5),
        )
        lm_state = _make_state(
            {
                "flow_direction": np.array([0.0, 1.0, 0.0], dtype=np.float32),
                "flow_magnitude": np.array([0.45], dtype=np.float32),
                "delta_hsv": np.array([0.1, -0.2, 0.3], dtype=np.float32),
                "observation_packet_v2": packet,
            }
        )

        core = PredictiveHypothesisCore(context_dim=48)
        core.pre_episode(mode=None, object_name=None)
        core.step(lm_state, learn=True)

        hypothesis_state = core.get_hypothesis_state()
        ranked = core.hypotheses.as_ranked_hypotheses()

        self.assertGreaterEqual(hypothesis_state.size, 1)
        self.assertEqual(hypothesis_state.latent_ids, hypothesis_state.object_ids)
        self.assertEqual(len(hypothesis_state.chart_ids), hypothesis_state.size)
        self.assertEqual(tuple(hypothesis_state.pose_vectors.shape[1:]), (3, 3))
        self.assertEqual(hypothesis_state.appearance_vectors.shape[0], hypothesis_state.size)
        self.assertGreater(hypothesis_state.appearance_vectors.shape[1], 0)
        self.assertEqual(hypothesis_state.behavior_vectors.shape[0], hypothesis_state.size)
        self.assertGreater(hypothesis_state.behavior_vectors.shape[1], 0)
        self.assertEqual(len(hypothesis_state.behavior_labels), hypothesis_state.size)
        self.assertEqual(hypothesis_state.behavior_labels[0], "temporal_change")

        top = ranked[0]
        self.assertEqual(top["latent_id"], top["object_id"])
        self.assertIn("chart_id", top)
        self.assertIn("location", top)
        self.assertIn("pose_vectors", top)
        self.assertIn("appearance_signature", top)
        self.assertIn("behavior_label", top)
        self.assertIn("behavior_signature", top)
        self.assertEqual(top["behavior_label"], "temporal_change")
        self.assertEqual(core.get_current_mlh()["latent_id"], top["latent_id"])
        self.assertEqual(core.get_current_mlh()["graph_id"], top["latent_id"])
        self.assertEqual(np.asarray(top["location"]).shape, (3,))
        self.assertEqual(np.asarray(top["pose_vectors"]).shape, (3, 3))
        self.assertGreater(np.asarray(top["appearance_signature"]).size, 0)
        self.assertGreater(np.asarray(top["behavior_signature"]).size, 0)

    def test_hypothesis_bank_keeps_multiple_joint_rows_for_same_latent(self):
        latent_id = "latent_object_0"
        other_id = "latent_object_1"
        stable_pose = np.eye(3, dtype=np.float32)
        change_pose = Rotation.from_euler(
            "z",
            35.0,
            degrees=True,
        ).as_matrix().astype(np.float32)
        bank = HypothesisBank(max_hypotheses=3, temperature=0.2)

        bank.update(
            {latent_id: 0.82, other_id: 0.80},
            location=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            pose_vectors=stable_pose,
            appearance_vector=np.array([1.0, 0.0], dtype=np.float32),
            behavior_vector=np.array([1.0, 0.0], dtype=np.float32),
            behavior_label="stable_surface",
            inferred_state=0,
            joint_candidates=[
                {
                    "object_id": latent_id,
                    "chart_id": f"{latent_id}#chart0",
                    "score": 0.82,
                    "location": np.array([0.1, 0.2, 0.3], dtype=np.float32),
                    "pose_vectors": stable_pose,
                    "appearance_vector": np.array([1.0, 0.0], dtype=np.float32),
                    "behavior_vector": np.array([1.0, 0.0], dtype=np.float32),
                    "behavior_label": "stable_surface",
                    "inferred_state": 0,
                },
                {
                    "object_id": latent_id,
                    "chart_id": f"{latent_id}#chart1",
                    "score": 0.81,
                    "location": np.array([0.1, 0.2, 0.3], dtype=np.float32),
                    "pose_vectors": change_pose,
                    "appearance_vector": np.array([0.0, 1.0], dtype=np.float32),
                    "behavior_vector": np.array([0.0, 1.0], dtype=np.float32),
                    "behavior_label": "temporal_change",
                    "inferred_state": 1,
                },
                {
                    "object_id": other_id,
                    "chart_id": f"{other_id}#chart0",
                    "score": 0.80,
                    "location": np.array([0.1, 0.2, 0.3], dtype=np.float32),
                    "pose_vectors": stable_pose,
                    "appearance_vector": np.array([0.2, 0.8], dtype=np.float32),
                    "behavior_vector": np.array([0.7, 0.3], dtype=np.float32),
                    "behavior_label": "stable_surface",
                    "inferred_state": 0,
                },
            ],
        )

        ranked = bank.as_ranked_hypotheses()
        evidence = bank.get_evidence()

        self.assertEqual([ranked[0]["latent_id"], ranked[1]["latent_id"]], [latent_id, latent_id])
        self.assertNotEqual(ranked[0]["chart_id"], ranked[1]["chart_id"])
        self.assertEqual(
            {ranked[0]["behavior_label"], ranked[1]["behavior_label"]},
            {"stable_surface", "temporal_change"},
        )
        self.assertGreater(evidence[latent_id], evidence[other_id])
        self.assertEqual(bank.get_current_mlh()["latent_id"], latent_id)

    def test_core_current_mlh_defaults_to_latent_aliases_before_steps(self):
        core = PredictiveHypothesisCore(context_dim=48)

        current = core.get_current_mlh()

        self.assertEqual(current["latent_id"], "no_observations_yet")
        self.assertEqual(current["graph_id"], current["latent_id"])
        self.assertEqual(current["evidence"], 0.0)
        self.assertEqual(core.get_possible_matches(), [])

    def test_core_joint_bank_scoring_uses_appearance_state(self):
        observation_a = _make_observation()
        observation_b = _make_observation()
        observation_b["rgba"] = observation_b["rgba"].copy()
        observation_b["rgba"][..., 0] = 255 - observation_b["rgba"][..., 0]
        observation_b["rgba"][..., 1] = 255 - observation_b["rgba"][..., 1]

        packet_a = build_visual_detail_packet(
            observation_a,
            sender_id="camera",
            grid_shape=(5, 5),
            state=_make_state(),
        )
        packet_b = build_change_detail_packet(
            observation_b,
            _make_state(
                {
                    "flow_direction": np.array([0.0, 1.0, 0.0], dtype=np.float32),
                    "flow_magnitude": np.array([0.95], dtype=np.float32),
                    "delta_hsv": np.array([0.7, -0.5, 0.6], dtype=np.float32),
                }
            ),
            sender_id="change",
            grid_shape=(5, 5),
        )
        lm_state_a = _make_state({"observation_packet_v2": packet_a})
        lm_state_b = _make_state(
            {
                "flow_direction": np.array([0.0, 1.0, 0.0], dtype=np.float32),
                "flow_magnitude": np.array([0.95], dtype=np.float32),
                "delta_hsv": np.array([0.7, -0.5, 0.6], dtype=np.float32),
                "observation_packet_v2": packet_b,
            }
        )

        trained = PredictiveHypothesisCore(context_dim=48)
        trained.pre_episode(mode=None, object_name=None)
        result_a = trained.step(lm_state_a, learn=True)
        trained.step(lm_state_a, learn=True)
        trained.step(lm_state_b, learn=True)

        object_a = result_a["mlh"]["graph_id"]
        known_ids = trained.get_all_known_object_ids()
        self.assertEqual(len(known_ids), 2)
        object_b = next(object_id for object_id in known_ids if object_id != object_a)
        saved_state = trained.state_dict()

        probe = PredictiveHypothesisCore(context_dim=48)
        probe.load_state_dict(saved_state)
        _, observation_field, embeddings = probe.encoder.encode_state_to_tensors(lm_state_b)
        appearance_vector = probe._summarize_appearance_state(observation_field, embeddings)
        behavior_vector, behavior_label = probe._summarize_behavior_state(
            observation_field,
            embeddings,
            packet_type=str(packet_b["packet_type"]),
        )
        base_scores = {object_a: 0.0, object_b: 0.0}
        chart_ids = {object_a: None, object_b: None}

        without_appearance = PredictiveHypothesisCore(context_dim=48)
        without_appearance.load_state_dict(saved_state)
        without_appearance.joint_appearance_score_weight = 0.0
        without_appearance.joint_pose_score_weight = 0.0
        without_appearance.joint_behavior_score_weight = 0.0
        without_scores, _, _, _, _ = without_appearance._apply_joint_hypothesis_support(
            base_scores,
            chart_ids,
            location=np.asarray(lm_state_b.location, dtype=np.float32),
            pose_vectors=np.asarray(
                lm_state_b.morphological_features["pose_vectors"],
                dtype=np.float32,
            ),
            appearance_vector=appearance_vector,
            behavior_vector=behavior_vector,
            behavior_label=behavior_label,
            inferred_state=None,
        )

        with_appearance = PredictiveHypothesisCore(context_dim=48)
        with_appearance.load_state_dict(saved_state)
        with_appearance.joint_pose_score_weight = 0.0
        with_appearance.joint_behavior_score_weight = 0.0
        with_scores, _, support, _, _ = with_appearance._apply_joint_hypothesis_support(
            base_scores,
            chart_ids,
            location=np.asarray(lm_state_b.location, dtype=np.float32),
            pose_vectors=np.asarray(
                lm_state_b.morphological_features["pose_vectors"],
                dtype=np.float32,
            ),
            appearance_vector=appearance_vector,
            behavior_vector=behavior_vector,
            behavior_label=behavior_label,
            inferred_state=None,
        )

        self.assertAlmostEqual(without_scores[object_a], without_scores[object_b], places=6)
        self.assertGreater(with_scores[object_b], with_scores[object_a])
        self.assertGreater(
            support[object_b]["appearance_support"],
            support[object_a]["appearance_support"],
        )

    def test_core_joint_support_expands_multiple_chart_candidates_for_single_latent(self):
        core = PredictiveHypothesisCore(context_dim=48)
        latent_id = "latent_object_0"
        other_id = "latent_object_1"
        stable_pose = np.eye(3, dtype=np.float32)
        change_pose = Rotation.from_euler(
            "z",
            55.0,
            degrees=True,
        ).as_matrix().astype(np.float32)
        core._chart_state_prototypes = {
            f"{latent_id}#chart0": {
                "observation_count": 2.0,
                "location": np.array([0.1, 0.2, 0.3], dtype=np.float32),
                "pose_vectors": stable_pose,
                "appearance_vector": np.array([1.0, 0.0], dtype=np.float32),
                "behavior_vector": np.array([1.0, 0.0], dtype=np.float32),
                "behavior_label": "stable_surface",
                "behavior_label_counts": {"stable_surface": 2.0},
            },
            f"{latent_id}#chart1": {
                "observation_count": 2.0,
                "location": np.array([0.1, 0.2, 0.3], dtype=np.float32),
                "pose_vectors": change_pose,
                "appearance_vector": np.array([0.0, 1.0], dtype=np.float32),
                "behavior_vector": np.array([0.0, 1.0], dtype=np.float32),
                "behavior_label": "temporal_change",
                "behavior_label_counts": {"temporal_change": 2.0},
            },
            f"{other_id}#chart0": {
                "observation_count": 2.0,
                "location": np.array([0.1, 0.2, 0.3], dtype=np.float32),
                "pose_vectors": stable_pose,
                "appearance_vector": np.array([0.4, 0.6], dtype=np.float32),
                "behavior_vector": np.array([0.6, 0.4], dtype=np.float32),
                "behavior_label": "stable_surface",
                "behavior_label_counts": {"stable_surface": 2.0},
            },
        }

        adjusted_scores, _, support, joint_states, joint_candidates = (
            core._apply_joint_hypothesis_support(
                {latent_id: 0.55, other_id: 0.54},
                {
                    latent_id: f"{latent_id}#chart0",
                    other_id: f"{other_id}#chart0",
                },
                location=np.array([0.1, 0.2, 0.3], dtype=np.float32),
                pose_vectors=change_pose,
                appearance_vector=np.array([0.0, 1.0], dtype=np.float32),
                behavior_vector=np.array([0.0, 1.0], dtype=np.float32),
                behavior_label="temporal_change",
                inferred_state=1,
            )
        )
        latent_candidates = [
            candidate
            for candidate in joint_candidates
            if candidate["object_id"] == latent_id
        ]

        self.assertEqual(len(latent_candidates), 2)
        self.assertEqual(
            {candidate["chart_id"] for candidate in latent_candidates},
            {f"{latent_id}#chart0", f"{latent_id}#chart1"},
        )
        self.assertEqual(joint_states[latent_id]["chart_id"], f"{latent_id}#chart1")
        self.assertIn(f"{latent_id}#chart0", support[latent_id]["candidate_chart_ids"])
        self.assertIn(f"{latent_id}#chart1", support[latent_id]["candidate_chart_ids"])
        self.assertGreater(adjusted_scores[latent_id], 0.55)
        self.assertGreater(
            next(
                candidate["score"]
                for candidate in latent_candidates
                if candidate["chart_id"] == f"{latent_id}#chart1"
            ),
            next(
                candidate["score"]
                for candidate in latent_candidates
                if candidate["chart_id"] == f"{latent_id}#chart0"
            ),
        )

    def test_context_ranked_chart_prior_prefers_matching_chart_for_same_latent(self):
        core = PredictiveHypothesisCore(context_dim=48)
        latent_id = "latent_object_0"
        stable_pose = np.eye(3, dtype=np.float32)
        change_pose = Rotation.from_euler(
            "z",
            40.0,
            degrees=True,
        ).as_matrix().astype(np.float32)
        core.joint_appearance_score_weight = 0.0
        core.joint_pose_score_weight = 0.0
        core.joint_behavior_score_weight = 0.0
        core.context_behavior_score_weight = 0.0
        core.prediction_bias_weight = 0.4
        core._chart_state_prototypes = {
            f"{latent_id}#chart0": {
                "observation_count": 1.0,
                "location": np.array([0.1, 0.2, 0.3], dtype=np.float32),
                "pose_vectors": stable_pose,
                "appearance_vector": np.array([1.0, 0.0], dtype=np.float32),
                "behavior_vector": np.array([1.0, 0.0], dtype=np.float32),
                "behavior_label": "stable_surface",
                "behavior_label_counts": {"stable_surface": 1.0},
            },
            f"{latent_id}#chart1": {
                "observation_count": 1.0,
                "location": np.array([0.1, 0.2, 0.3], dtype=np.float32),
                "pose_vectors": change_pose,
                "appearance_vector": np.array([0.0, 1.0], dtype=np.float32),
                "behavior_vector": np.array([0.0, 1.0], dtype=np.float32),
                "behavior_label": "temporal_change",
                "behavior_label_counts": {"temporal_change": 1.0},
            },
        }
        core.receive_context_message(
            PredictiveContextSignal(
                latent_id=None,
                confidence=0.9,
                residual=0.0,
                active_cells=torch.zeros(0, dtype=torch.float32),
                ranked_hypotheses=[
                    RankedHypothesisVote(
                        object_id=latent_id,
                        chart_id=f"{latent_id}#chart1",
                        probability=1.0,
                        evidence=1.0,
                        rank=1,
                    )
                ],
            )
        )

        _, _, support, joint_states, joint_candidates = core._apply_joint_hypothesis_support(
            {latent_id: 0.55},
            {latent_id: f"{latent_id}#chart0"},
            location=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            pose_vectors=stable_pose,
            appearance_vector=np.array([1.0, 0.0], dtype=np.float32),
            behavior_vector=np.array([1.0, 0.0], dtype=np.float32),
            behavior_label="stable_surface",
            inferred_state=0,
        )
        chart0_score = next(
            candidate["score"]
            for candidate in joint_candidates
            if candidate["chart_id"] == f"{latent_id}#chart0"
        )
        chart1_score = next(
            candidate["score"]
            for candidate in joint_candidates
            if candidate["chart_id"] == f"{latent_id}#chart1"
        )

        self.assertGreater(chart1_score, chart0_score)
        self.assertEqual(joint_states[latent_id]["chart_id"], f"{latent_id}#chart1")
        self.assertGreater(support[latent_id]["context_ranked_chart_prior"], 0.0)

    def test_vote_ranked_chart_prior_prefers_matching_chart_for_same_latent(self):
        core = PredictiveHypothesisCore(context_dim=48)
        latent_id = "latent_object_0"
        stable_pose = np.eye(3, dtype=np.float32)
        change_pose = Rotation.from_euler(
            "z",
            55.0,
            degrees=True,
        ).as_matrix().astype(np.float32)
        core.joint_appearance_score_weight = 0.0
        core.joint_pose_score_weight = 0.0
        core.joint_behavior_score_weight = 0.0
        core.context_behavior_score_weight = 0.0
        core.vote_appearance_score_weight = 0.0
        core.vote_behavior_score_weight = 0.0
        core._chart_state_prototypes = {
            f"{latent_id}#chart0": {
                "observation_count": 1.0,
                "location": np.array([0.1, 0.2, 0.3], dtype=np.float32),
                "pose_vectors": stable_pose,
                "appearance_vector": np.array([1.0, 0.0], dtype=np.float32),
                "behavior_vector": np.array([1.0, 0.0], dtype=np.float32),
                "behavior_label": "stable_surface",
                "behavior_label_counts": {"stable_surface": 1.0},
            },
            f"{latent_id}#chart1": {
                "observation_count": 1.0,
                "location": np.array([0.1, 0.2, 0.3], dtype=np.float32),
                "pose_vectors": change_pose,
                "appearance_vector": np.array([0.0, 1.0], dtype=np.float32),
                "behavior_vector": np.array([0.0, 1.0], dtype=np.float32),
                "behavior_label": "temporal_change",
                "behavior_label_counts": {"temporal_change": 1.0},
            },
        }
        core.appearance_memory.get_all_known_object_ids = lambda: [latent_id]
        core.change_memory.get_all_known_object_ids = lambda: [latent_id]
        core.receive_vote_message(
            PredictiveVoteMessage(
                sender_id="child_lm",
                sensed_pose_rel_body=np.zeros((4, 3), dtype=np.float64),
                active_cells=torch.zeros(0, dtype=torch.float32),
                appearance_signature=torch.zeros(0, dtype=torch.float32),
                behavior_label=None,
                behavior_signature=torch.zeros(0, dtype=torch.float32),
                ranked_hypotheses=[
                    RankedHypothesisVote(
                        object_id=latent_id,
                        chart_id=f"{latent_id}#chart1",
                        probability=1.0,
                        evidence=1.0,
                        rank=1,
                    )
                ],
                possible_states={},
            )
        )

        _, _, support, joint_states, joint_candidates = core._apply_joint_hypothesis_support(
            {latent_id: 0.55},
            {latent_id: f"{latent_id}#chart0"},
            location=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            pose_vectors=stable_pose,
            appearance_vector=np.array([1.0, 0.0], dtype=np.float32),
            behavior_vector=np.array([1.0, 0.0], dtype=np.float32),
            behavior_label="stable_surface",
            inferred_state=0,
        )
        chart0_score = next(
            candidate["score"]
            for candidate in joint_candidates
            if candidate["chart_id"] == f"{latent_id}#chart0"
        )
        chart1_score = next(
            candidate["score"]
            for candidate in joint_candidates
            if candidate["chart_id"] == f"{latent_id}#chart1"
        )

        self.assertGreater(chart1_score, chart0_score)
        self.assertEqual(joint_states[latent_id]["chart_id"], f"{latent_id}#chart1")
        self.assertGreater(support[latent_id]["vote_ranked_chart_prior"], 0.0)

    def test_core_learns_latent_objects_without_external_labels(self):
        observation_a = _make_observation()
        observation_b = _make_observation()
        observation_b["rgba"] = observation_b["rgba"].copy()
        observation_b["rgba"][..., 0] = 255 - observation_b["rgba"][..., 0]
        observation_b["rgba"][..., 1] = 255 - observation_b["rgba"][..., 1]

        packet_a = build_visual_detail_packet(
            observation_a,
            sender_id="camera",
            grid_shape=(5, 5),
            state=_make_state(),
        )
        packet_b = build_change_detail_packet(
            observation_b,
            _make_state(
                {
                    "flow_direction": np.array([0.0, 1.0, 0.0], dtype=np.float32),
                    "flow_magnitude": np.array([0.95], dtype=np.float32),
                    "delta_hsv": np.array([0.7, -0.5, 0.6], dtype=np.float32),
                }
            ),
            sender_id="change",
            grid_shape=(5, 5),
        )

        lm_state_a = _make_state({"observation_packet_v2": packet_a})
        lm_state_b = _make_state(
            {
                "flow_direction": np.array([0.0, 1.0, 0.0], dtype=np.float32),
                "flow_magnitude": np.array([0.95], dtype=np.float32),
                "delta_hsv": np.array([0.7, -0.5, 0.6], dtype=np.float32),
                "observation_packet_v2": packet_b,
            }
        )

        core = PredictiveHypothesisCore(context_dim=48)
        core.pre_episode(mode=None, object_name=None)
        core.step(lm_state_a, learn=True)
        core.step(lm_state_a, learn=True)

        known_ids = core.get_all_known_object_ids()
        self.assertEqual(len(known_ids), 1)
        self.assertTrue(known_ids[0].startswith("latent_object_"))
        support = core.get_evidence_debug()["self_supervised_support"]
        self.assertEqual(support["learning"]["learning_mode"], "self_supervised")
        self.assertEqual(
            support["learning"]["selection_mode"],
            "posterior_bank_competition",
        )
        self.assertEqual(
            support["learning"]["learning_latent_id"],
            support["learning"]["learning_object_id"],
        )
        self.assertEqual(
            support["learning"]["allocation_reason"],
            "posterior_existing",
        )
        self.assertTrue(support["learning"]["posterior_candidates"])
        self.assertTrue(
            any(
                candidate["selected"]
                and candidate["latent_id"] == support["learning"]["learning_latent_id"]
                for candidate in support["learning"]["posterior_candidates"]
            )
        )

        core.step(lm_state_b, learn=True)
        known_ids = core.get_all_known_object_ids()
        self.assertEqual(len(known_ids), 2)
        self.assertTrue(all(object_id.startswith("latent_object_") for object_id in known_ids))

    def test_core_records_chart_transitions_and_exposes_transition_support(self):
        observation_a = _make_observation()
        observation_b = _make_observation()
        observation_b["rgba"] = observation_b["rgba"].copy()
        observation_b["rgba"][..., 2] = 255 - observation_b["rgba"][..., 2]

        packet_a = build_visual_detail_packet(
            observation_a,
            sender_id="camera",
            grid_shape=(5, 5),
            state=_make_state(),
        )
        packet_b = build_change_detail_packet(
            observation_b,
            _make_state(
                {
                    "flow_direction": np.array([0.0, 1.0, 0.0], dtype=np.float32),
                    "flow_magnitude": np.array([0.9], dtype=np.float32),
                    "delta_hsv": np.array([0.6, -0.4, 0.5], dtype=np.float32),
                }
            ),
            sender_id="change",
            grid_shape=(5, 5),
        )

        lm_state_a = _make_state({"observation_packet_v2": packet_a})
        lm_state_b = _make_state(
            {
                "flow_direction": np.array([0.0, 1.0, 0.0], dtype=np.float32),
                "flow_magnitude": np.array([0.9], dtype=np.float32),
                "delta_hsv": np.array([0.6, -0.4, 0.5], dtype=np.float32),
                "observation_packet_v2": packet_b,
            }
        )

        core = PredictiveHypothesisCore(context_dim=48)
        core.pre_episode(mode=None, object_name="fox:0")
        core.step(lm_state_a, learn=True)
        core.step(lm_state_b, learn=True)

        transition_counts = core.get_chart_transition_counts()
        self.assertTrue(transition_counts)

        core.step(lm_state_a, learn=False)
        core.step(lm_state_b, learn=False)
        support = core.get_evidence_debug()["self_supervised_support"]
        self.assertIn("transition", support)
        self.assertEqual(
            support["transition"]["previous_latent_id"],
            support["transition"]["previous_object_id"],
        )
        self.assertTrue(support["transition"]["transition_support"])

    def test_core_persists_chart_transitions_across_episodes(self):
        observation_a = _make_observation()
        observation_b = _make_observation()
        observation_b["rgba"] = observation_b["rgba"].copy()
        observation_b["rgba"][..., 2] = 255 - observation_b["rgba"][..., 2]

        packet_a = build_visual_detail_packet(
            observation_a,
            sender_id="camera",
            grid_shape=(5, 5),
            state=_make_state(),
        )
        packet_b = build_change_detail_packet(
            observation_b,
            _make_state(
                {
                    "flow_direction": np.array([0.0, 1.0, 0.0], dtype=np.float32),
                    "flow_magnitude": np.array([0.9], dtype=np.float32),
                    "delta_hsv": np.array([0.6, -0.4, 0.5], dtype=np.float32),
                }
            ),
            sender_id="change",
            grid_shape=(5, 5),
        )

        lm_state_a = _make_state({"observation_packet_v2": packet_a})
        lm_state_b = _make_state(
            {
                "flow_direction": np.array([0.0, 1.0, 0.0], dtype=np.float32),
                "flow_magnitude": np.array([0.9], dtype=np.float32),
                "delta_hsv": np.array([0.6, -0.4, 0.5], dtype=np.float32),
                "observation_packet_v2": packet_b,
            }
        )

        core = PredictiveHypothesisCore(context_dim=48)
        core.pre_episode(mode=None, object_name="fox:0")
        core.step(lm_state_a, learn=True)
        core.step(lm_state_b, learn=True)

        transition_counts = core.get_chart_transition_counts()
        self.assertTrue(transition_counts)

        core.pre_episode(mode=None, object_name="fox:0")
        self.assertEqual(core.get_chart_transition_counts(), transition_counts)

        core.step(lm_state_a, learn=False)
        core.step(lm_state_b, learn=False)
        support = core.get_evidence_debug()["self_supervised_support"]
        self.assertEqual(
            support["transition"]["previous_latent_id"],
            support["transition"]["previous_object_id"],
        )
        self.assertTrue(support["transition"]["transition_support"])

    def test_chart_transition_support_updates_joint_candidates_per_chart(self):
        core = PredictiveHypothesisCore(context_dim=48)
        core.transition_score_weight = 0.2
        core._last_winning_latent_id = "latent_object_0"
        core._last_winning_chart_id = "latent_object_0#chart0"
        core._chart_transition_counts = {
            "latent_object_0#chart0": {
                "latent_object_0#chart0": 1.0,
                "latent_object_0#chart1": 4.0,
            }
        }

        adjusted_scores, adjustments, support, adjusted_candidates, adjusted_chart_ids = (
            core._apply_chart_transition_support(
                {
                    "latent_object_0": 0.80,
                    "latent_object_1": 0.79,
                },
                {
                    "latent_object_0": "latent_object_0#chart0",
                    "latent_object_1": "latent_object_1#chart0",
                },
                joint_candidates=[
                    {
                        "object_id": "latent_object_0",
                        "chart_id": "latent_object_0#chart0",
                        "score": 0.80,
                    },
                    {
                        "object_id": "latent_object_0",
                        "chart_id": "latent_object_0#chart1",
                        "score": 0.78,
                    },
                    {
                        "object_id": "latent_object_1",
                        "chart_id": "latent_object_1#chart0",
                        "score": 0.79,
                    },
                ],
            )
        )

        latent_candidates = [
            candidate
            for candidate in adjusted_candidates
            if candidate["object_id"] == "latent_object_0"
        ]
        chart0_score = next(
            candidate["score"]
            for candidate in latent_candidates
            if candidate["chart_id"] == "latent_object_0#chart0"
        )
        chart1_score = next(
            candidate["score"]
            for candidate in latent_candidates
            if candidate["chart_id"] == "latent_object_0#chart1"
        )

        self.assertGreater(chart1_score, chart0_score)
        self.assertEqual(adjusted_chart_ids["latent_object_0"], "latent_object_0#chart1")
        self.assertAlmostEqual(adjusted_scores["latent_object_0"], chart1_score, places=6)
        self.assertAlmostEqual(
            adjustments["latent_object_0"],
            chart1_score - 0.80,
            places=6,
        )
        self.assertEqual(
            support["transition_support"]["latent_object_0"]["chart_id"],
            "latent_object_0#chart1",
        )
        self.assertEqual(
            len(support["transition_support"]["latent_object_0"]["candidate_supports"]),
            2,
        )

    def test_core_reuses_latent_identity_across_unlabeled_episodes(self):
        observation = _make_observation()
        packet = build_visual_detail_packet(
            observation,
            sender_id="camera",
            grid_shape=(5, 5),
            state=_make_state(),
        )
        lm_state = _make_state({"observation_packet_v2": packet})

        core = PredictiveHypothesisCore(context_dim=48)
        core.pre_episode(mode=None, object_name=None)
        core.step(lm_state, learn=True)

        known_ids = core.get_all_known_object_ids()
        self.assertEqual(len(known_ids), 1)
        self.assertTrue(known_ids[0].startswith("latent_object_"))

        core.pre_episode(mode=None, object_name=None)
        core.step(lm_state, learn=True)

        self.assertEqual(core.get_all_known_object_ids(), known_ids)
        support = core.get_evidence_debug()["self_supervised_support"]
        self.assertEqual(
            support["learning"]["selection_mode"],
            "posterior_bank_competition",
        )
        self.assertEqual(
            support["learning"]["memory_candidate_latent_id"],
            support["learning"]["memory_candidate_object_id"],
        )
        self.assertEqual(
            support["learning"]["allocation_reason"],
            "posterior_existing",
        )
        self.assertNotEqual(support["learning"]["selected_source"], "novel")
        top_hypothesis = core.hypotheses.as_ranked_hypotheses()[0]
        self.assertEqual(top_hypothesis["object_id"], known_ids[0])
        self.assertEqual(top_hypothesis["behavior_label"], "stable_surface")

    def test_core_reuses_single_latent_identity_across_repeated_unlabeled_pose_views(self):
        observation = _make_observation()
        pose_variants = [
            np.eye(3, dtype=np.float32),
            Rotation.from_euler("z", 8.0, degrees=True).as_matrix().astype(np.float32),
            Rotation.from_euler("z", 14.0, degrees=True).as_matrix().astype(np.float32),
        ]

        core = PredictiveHypothesisCore(context_dim=48)
        selection_reasons = []
        for pose_vectors in pose_variants:
            state = _make_state(pose_vectors=pose_vectors)
            packet = build_visual_detail_packet(
                observation,
                sender_id="camera",
                grid_shape=(5, 5),
                state=state,
            )
            lm_state = _make_state(
                {"observation_packet_v2": packet},
                pose_vectors=pose_vectors,
            )
            core.pre_episode(mode=None, object_name=None)
            core.step(lm_state, learn=True)
            selection_reasons.append(
                core.get_evidence_debug()["self_supervised_support"]["learning"]["selection_reason"]
            )

        known_ids = core.get_all_known_object_ids()
        self.assertEqual(len(known_ids), 1)
        self.assertEqual(selection_reasons[0], "posterior_novel")
        self.assertTrue(
            all(reason == "posterior_existing" for reason in selection_reasons[1:])
        )

        inference_state = _make_state(pose_vectors=pose_variants[-1])
        inference_packet = build_visual_detail_packet(
            observation,
            sender_id="camera",
            grid_shape=(5, 5),
            state=inference_state,
        )
        inference_lm_state = _make_state(
            {"observation_packet_v2": inference_packet},
            pose_vectors=pose_variants[-1],
        )
        core.pre_episode(mode=None, object_name=None)
        result = core.step(inference_lm_state, learn=False)

        self.assertEqual(result["mlh"]["graph_id"], known_ids[0])
        joint_support = core.get_evidence_debug()["joint_hypothesis_support"]
        self.assertIn(known_ids[0], joint_support)
        self.assertGreater(joint_support[known_ids[0]]["pose_support"], 0.8)

    def test_core_anchor_bootstrap_prefers_novel_when_reuse_is_not_overwhelming(self):
        core = PredictiveHypothesisCore(context_dim=48)
        core.pre_episode(mode=None, object_name="anchored_episode")
        core._latent_object_counter = 8

        field = _make_texture_field("horizontal")
        embeddings = core.encoder.encode_observation_field(field)
        embedding = core._compose_embedding(embeddings.joint)
        appearance_vector = core._summarize_appearance_state(field, embeddings)
        behavior_vector, behavior_label = core._summarize_behavior_state(
            field,
            embeddings,
            packet_type=field.packet_type,
        )

        learning_latent_id, _, support = core._select_learning_hypothesis_from_posterior(
            learn=True,
            embedding=embedding,
            observation_field=field,
            action_prediction_error=0.0,
            joint_candidates=[
                {
                    "object_id": "latent_object_7",
                    "latent_id": "latent_object_7",
                    "chart_id": "latent_object_7#chart0",
                    "score": 1.20,
                    "source": "memory",
                    "age": 0,
                }
            ],
            location=np.zeros(3, dtype=np.float32),
            pose_vectors=np.eye(3, dtype=np.float32),
            appearance_vector=appearance_vector,
            behavior_vector=behavior_vector,
            behavior_label=behavior_label,
            inferred_state=None,
        )

        self.assertEqual(learning_latent_id, "latent_object_8")
        self.assertEqual(
            support["selection_reason"],
            "posterior_anchor_bootstrap_novel",
        )
        self.assertEqual(support["selected_source"], "novel")

    def test_core_anchor_continuity_prefers_active_latent_without_clear_switch_evidence(self):
        core = PredictiveHypothesisCore(context_dim=48)
        core.pre_episode(mode=None, object_name="anchored_episode")
        core._active_self_supervised_latent_id = "latent_object_7"
        core._latent_object_counter = 9

        field = _make_texture_field("vertical")
        embeddings = core.encoder.encode_observation_field(field)
        embedding = core._compose_embedding(embeddings.joint)
        appearance_vector = core._summarize_appearance_state(field, embeddings)
        behavior_vector, behavior_label = core._summarize_behavior_state(
            field,
            embeddings,
            packet_type=field.packet_type,
        )

        learning_latent_id, _, support = core._select_learning_hypothesis_from_posterior(
            learn=True,
            embedding=embedding,
            observation_field=field,
            action_prediction_error=0.0,
            joint_candidates=[
                {
                    "object_id": "latent_object_7",
                    "latent_id": "latent_object_7",
                    "chart_id": "latent_object_7#chart0",
                    "score": 1.15,
                    "source": "memory",
                    "age": 1,
                },
                {
                    "object_id": "latent_object_8",
                    "latent_id": "latent_object_8",
                    "chart_id": "latent_object_8#chart0",
                    "score": 1.21,
                    "source": "memory",
                    "age": 0,
                },
            ],
            location=np.zeros(3, dtype=np.float32),
            pose_vectors=np.eye(3, dtype=np.float32),
            appearance_vector=appearance_vector,
            behavior_vector=behavior_vector,
            behavior_label=behavior_label,
            inferred_state=None,
        )

        self.assertEqual(learning_latent_id, "latent_object_7")
        self.assertEqual(
            support["selection_reason"],
            "posterior_anchor_continuity",
        )

    def test_core_anchor_continuity_allows_switch_when_alternative_is_overwhelming(self):
        core = PredictiveHypothesisCore(context_dim=48)
        core.pre_episode(mode=None, object_name="anchored_episode")
        core._active_self_supervised_latent_id = "latent_object_7"
        core._latent_object_counter = 9

        field = _make_texture_field("vertical")
        embeddings = core.encoder.encode_observation_field(field)
        embedding = core._compose_embedding(embeddings.joint)
        appearance_vector = core._summarize_appearance_state(field, embeddings)
        behavior_vector, behavior_label = core._summarize_behavior_state(
            field,
            embeddings,
            packet_type=field.packet_type,
        )

        learning_latent_id, _, support = core._select_learning_hypothesis_from_posterior(
            learn=True,
            embedding=embedding,
            observation_field=field,
            action_prediction_error=0.0,
            joint_candidates=[
                {
                    "object_id": "latent_object_7",
                    "latent_id": "latent_object_7",
                    "chart_id": "latent_object_7#chart0",
                    "score": 0.86,
                    "source": "memory",
                    "age": 1,
                },
                {
                    "object_id": "latent_object_8",
                    "latent_id": "latent_object_8",
                    "chart_id": "latent_object_8#chart0",
                    "score": 1.35,
                    "source": "memory",
                    "age": 0,
                },
            ],
            location=np.zeros(3, dtype=np.float32),
            pose_vectors=np.eye(3, dtype=np.float32),
            appearance_vector=appearance_vector,
            behavior_vector=behavior_vector,
            behavior_label=behavior_label,
            inferred_state=None,
        )

        self.assertEqual(learning_latent_id, "latent_object_8")
        self.assertEqual(support["selection_reason"], "posterior_existing")

    def test_core_anchor_learning_selection_carries_stronger_runtime_score_floor(self):
        core = PredictiveHypothesisCore(context_dim=48)
        learning_support = {
            "selected_candidate_score": 1.12,
            "selected_posterior_probability": 0.40,
            "selection_reason": "posterior_anchor_bootstrap_novel",
            "selected_source": "novel",
        }

        unanchored_floor = core._learning_selection_score_floor(learning_support)
        self.assertEqual(unanchored_floor, 1.0)

        core.pre_episode(mode=None, object_name="anchored_episode")
        scores, chart_ids, carried_score_floor = core._carry_learning_selection_into_scores(
            scores={"latent_object_7": 1.18},
            chart_ids={"latent_object_7": "latent_object_7#chart0"},
            learning_latent_id="latent_object_8",
            learning_chart_id="latent_object_8#chart0",
            learning_support=learning_support,
        )

        self.assertGreater(carried_score_floor, 1.20)
        self.assertEqual(scores["latent_object_8"], carried_score_floor)
        self.assertEqual(chart_ids["latent_object_8"], "latent_object_8#chart0")

    def test_core_anchor_learning_selection_carry_is_visual_only(self):
        self.assertTrue(
            PredictiveHypothesisCore._should_carry_learning_selection(
                "visual_observation_packet_v2"
            )
        )
        self.assertFalse(
            PredictiveHypothesisCore._should_carry_learning_selection(
                "change_observation_packet_v2"
            )
        )
        self.assertFalse(
            PredictiveHypothesisCore._should_carry_learning_selection(
                "lm_fusion_packet"
            )
        )

    def test_core_propagates_previous_bank_hypotheses_into_next_step(self):
        observation = _make_observation()
        packet = build_visual_detail_packet(
            observation,
            sender_id="camera",
            grid_shape=(5, 5),
            state=_make_state(),
        )
        lm_state = _make_state({"observation_packet_v2": packet})

        core = PredictiveHypothesisCore(context_dim=48)
        core.pre_episode(mode=None, object_name=None)
        core.step(lm_state, learn=True)
        first_ranked = core.hypotheses.as_ranked_hypotheses()
        self.assertGreater(len(first_ranked), 0)
        first_latent_id = first_ranked[0]["object_id"]

        core.step(lm_state, learn=False)

        ranked = core.hypotheses.as_ranked_hypotheses()
        self.assertEqual(ranked[0]["object_id"], first_latent_id)
        self.assertEqual(ranked[0]["source"], "propagated")
        self.assertGreaterEqual(ranked[0]["age"], 1)

        propagated_sources = {
            candidate.get("source")
            for candidate in core.get_evidence_debug()["joint_hypothesis_candidates"]
            if candidate.get("object_id") == first_latent_id
        }
        self.assertIn("propagated", propagated_sources)

    def test_parent_context_behavior_completes_partial_cue(self):
        observation = _make_observation()
        stable_state = _make_state()
        change_state = _make_state(
            {
                "flow_direction": np.array([0.0, 1.0, 0.0], dtype=np.float32),
                "flow_magnitude": np.array([0.65], dtype=np.float32),
                "delta_hsv": np.array([0.2, -0.1, 0.3], dtype=np.float32),
            }
        )
        stable_packet = build_visual_detail_packet(
            observation,
            sender_id="camera",
            grid_shape=(5, 5),
            state=stable_state,
        )
        change_packet = build_change_detail_packet(
            observation,
            change_state,
            sender_id="change",
            grid_shape=(5, 5),
        )
        partial_packet = copy.deepcopy(stable_packet)
        partial_packet["cells"] = [copy.deepcopy(stable_packet["cells"][0])]
        partial_packet["cell_count"] = 1
        partial_packet["grid_shape"] = [1, 1]
        stable_lm_state = _make_state({"observation_packet_v2": stable_packet})
        partial_lm_state = _make_state({"observation_packet_v2": partial_packet})
        change_lm_state = _make_state(
            {
                "flow_direction": np.array([0.0, 1.0, 0.0], dtype=np.float32),
                "flow_magnitude": np.array([0.65], dtype=np.float32),
                "delta_hsv": np.array([0.2, -0.1, 0.3], dtype=np.float32),
                "observation_packet_v2": change_packet,
            }
        )

        trained = PredictiveHypothesisCore(context_dim=48)
        trained.pre_episode(mode=None, object_name="stable_anchor")
        trained.step(stable_lm_state, learn=True)
        stable_object_id = trained.get_all_known_object_ids()[0]
        trained.pre_episode(mode=None, object_name="change_anchor")
        change_result = trained.step(change_lm_state, learn=True)
        known_ids = trained.get_all_known_object_ids()
        self.assertEqual(len(known_ids), 2)
        change_object_id = next(object_id for object_id in known_ids if object_id != stable_object_id)
        saved_state = trained.state_dict()

        baseline = PredictiveHypothesisCore(context_dim=48)
        baseline.load_state_dict(saved_state)
        baseline.pre_episode(mode=None, object_name=None)
        baseline_result = baseline.step(partial_lm_state, learn=False)
        self.assertEqual(baseline_result["mlh"]["graph_id"], stable_object_id)

        biased = PredictiveHypothesisCore(context_dim=48)
        biased.load_state_dict(saved_state)
        biased.pre_episode(mode=None, object_name=None)
        biased.receive_context_message(
            PredictiveContextSignal(
                latent_id=None,
                confidence=1.0,
                residual=0.0,
                active_cells=torch.zeros(0, dtype=torch.float32),
                behavior_label="temporal_change",
                behavior_signature=torch.as_tensor(
                    np.asarray(
                        change_result["mlh"].get("behavior_signature", np.zeros(0)),
                        dtype=np.float32,
                    ),
                    dtype=torch.float32,
                ),
            )
        )
        biased_result = biased.step(partial_lm_state, learn=False)

        self.assertEqual(biased_result["mlh"]["graph_id"], change_object_id)
        joint_support = biased.get_evidence_debug()["joint_hypothesis_support"]
        joint_candidates = biased.get_evidence_debug()["joint_hypothesis_candidates"]
        self.assertGreater(
            joint_support[change_object_id]["context_behavior_support"],
            joint_support[stable_object_id]["context_behavior_support"],
        )
        self.assertGreater(len(joint_candidates), 0)
        self.assertIn("chart_id", joint_candidates[0])
        self.assertIn("score", joint_candidates[0])
        self.assertIn("selected", joint_candidates[0])

    def test_parent_context_appearance_biases_partial_cue_toward_matching_object(self):
        observation_a = _make_observation()
        observation_b = _make_observation()
        observation_b["rgba"] = observation_b["rgba"].copy()
        observation_b["rgba"][..., 0] = 255 - observation_b["rgba"][..., 0]
        observation_b["rgba"][..., 1] = 255 - observation_b["rgba"][..., 1]

        packet_a = build_visual_detail_packet(
            observation_a,
            sender_id="camera",
            grid_shape=(5, 5),
            state=_make_state(),
        )
        packet_b = build_change_detail_packet(
            observation_b,
            _make_state(
                {
                    "flow_direction": np.array([0.0, 1.0, 0.0], dtype=np.float32),
                    "flow_magnitude": np.array([0.95], dtype=np.float32),
                    "delta_hsv": np.array([0.7, -0.5, 0.6], dtype=np.float32),
                }
            ),
            sender_id="change",
            grid_shape=(5, 5),
        )
        partial_packet = copy.deepcopy(packet_a)
        partial_packet["cells"] = [copy.deepcopy(packet_a["cells"][0])]
        partial_packet["cell_count"] = 1
        partial_packet["grid_shape"] = [1, 1]
        lm_state_a = _make_state({"observation_packet_v2": packet_a})
        lm_state_b = _make_state(
            {
                "flow_direction": np.array([0.0, 1.0, 0.0], dtype=np.float32),
                "flow_magnitude": np.array([0.95], dtype=np.float32),
                "delta_hsv": np.array([0.7, -0.5, 0.6], dtype=np.float32),
                "observation_packet_v2": packet_b,
            }
        )
        partial_lm_state = _make_state({"observation_packet_v2": partial_packet})

        trained = PredictiveHypothesisCore(context_dim=48)
        trained.pre_episode(mode=None, object_name=None)
        result_a = trained.step(lm_state_a, learn=True)
        trained.step(lm_state_a, learn=True)
        result_b = trained.step(lm_state_b, learn=True)
        stable_object_id = result_a["mlh"]["graph_id"]
        known_ids = trained.get_all_known_object_ids()
        self.assertEqual(len(known_ids), 2)
        appearance_object_id = next(
            object_id for object_id in known_ids if object_id != stable_object_id
        )
        saved_state = trained.state_dict()

        baseline = PredictiveHypothesisCore(context_dim=48)
        baseline.load_state_dict(saved_state)
        baseline.joint_behavior_score_weight = 0.0
        baseline.context_behavior_score_weight = 0.0
        baseline.pre_episode(mode=None, object_name=None)
        baseline_result = baseline.step(partial_lm_state, learn=False)
        baseline_evidence = dict(baseline_result["final_evidence"])

        biased = PredictiveHypothesisCore(context_dim=48)
        biased.load_state_dict(saved_state)
        biased.joint_behavior_score_weight = 0.0
        biased.context_behavior_score_weight = 0.0
        biased.pre_episode(mode=None, object_name=None)
        biased.receive_context_message(
            PredictiveContextSignal(
                latent_id=None,
                confidence=1.0,
                residual=0.0,
                active_cells=torch.zeros(0, dtype=torch.float32),
                appearance_signature=torch.as_tensor(
                    np.asarray(
                        result_b["mlh"].get("appearance_signature", np.zeros(0)),
                        dtype=np.float32,
                    ),
                    dtype=torch.float32,
                ),
                behavior_label=None,
                behavior_signature=torch.zeros(0, dtype=torch.float32),
            )
        )
        biased_result = biased.step(partial_lm_state, learn=False)

        joint_support = biased.get_evidence_debug()["joint_hypothesis_support"]
        self.assertGreater(
            joint_support[appearance_object_id]["context_appearance_support"],
            joint_support[stable_object_id]["context_appearance_support"],
        )
        self.assertGreater(
            biased_result["final_evidence"][appearance_object_id],
            baseline_evidence[appearance_object_id],
        )

    def test_parent_context_predictive_priors_shift_temporal_retrieval_without_changing_base_evidence(self):
        observation_a = _make_observation()
        observation_b = _make_observation()
        observation_b["rgba"] = observation_b["rgba"].copy()
        observation_b["rgba"][..., 0] = 255 - observation_b["rgba"][..., 0]
        observation_b["rgba"][..., 1] = 255 - observation_b["rgba"][..., 1]

        packet_a = build_visual_detail_packet(
            observation_a,
            sender_id="camera",
            grid_shape=(5, 5),
            state=_make_state(),
        )
        packet_b = build_change_detail_packet(
            observation_b,
            _make_state(
                {
                    "flow_direction": np.array([0.0, 1.0, 0.0], dtype=np.float32),
                    "flow_magnitude": np.array([0.95], dtype=np.float32),
                    "delta_hsv": np.array([0.7, -0.5, 0.6], dtype=np.float32),
                }
            ),
            sender_id="change",
            grid_shape=(5, 5),
        )
        partial_packet = copy.deepcopy(packet_a)
        partial_packet["cells"] = [copy.deepcopy(packet_a["cells"][0])]
        partial_packet["cell_count"] = 1
        partial_packet["grid_shape"] = [1, 1]
        lm_state_a = _make_state({"observation_packet_v2": packet_a})
        lm_state_b = _make_state(
            {
                "flow_direction": np.array([0.0, 1.0, 0.0], dtype=np.float32),
                "flow_magnitude": np.array([0.95], dtype=np.float32),
                "delta_hsv": np.array([0.7, -0.5, 0.6], dtype=np.float32),
                "observation_packet_v2": packet_b,
            }
        )
        partial_lm_state = _make_state({"observation_packet_v2": partial_packet})

        trained = PredictiveHypothesisCore(context_dim=48)
        trained.pre_episode(mode=None, object_name=None)
        result_a = trained.step(lm_state_a, learn=True)
        trained.step(lm_state_a, learn=True)
        trained.step(lm_state_b, learn=True)
        predicted_message = trained.get_last_message_state()
        stable_object_id = result_a["mlh"]["graph_id"]
        known_ids = trained.get_all_known_object_ids()
        appearance_object_id = next(
            object_id for object_id in known_ids if object_id != stable_object_id
        )
        saved_state = trained.state_dict()

        baseline = PredictiveHypothesisCore(context_dim=48)
        baseline.load_state_dict(saved_state)
        baseline.joint_behavior_score_weight = 0.0
        baseline.context_behavior_score_weight = 0.0
        baseline.joint_appearance_score_weight = 0.0
        baseline.context_appearance_score_weight = 0.0
        baseline.pre_episode(mode=None, object_name=None)
        baseline_result = baseline.step(partial_lm_state, learn=False)
        baseline_evidence = dict(baseline_result["final_evidence"])
        baseline_debug = baseline.get_evidence_debug()

        biased = PredictiveHypothesisCore(context_dim=48)
        biased.load_state_dict(saved_state)
        biased.joint_behavior_score_weight = 0.0
        biased.context_behavior_score_weight = 0.0
        biased.joint_appearance_score_weight = 0.0
        biased.context_appearance_score_weight = 0.0
        biased.pre_episode(mode=None, object_name=None)
        biased.receive_context_message(
            PredictiveContextSignal(
                latent_id=None,
                confidence=1.0,
                residual=0.0,
                active_cells=torch.zeros(0, dtype=torch.float32),
                predicted_appearance_signature=(
                    predicted_message.predicted_appearance_signature.detach().clone()
                ),
                predicted_change_signature=(
                    predicted_message.predicted_change_signature.detach().clone()
                ),
                appearance_signature=torch.zeros(0, dtype=torch.float32),
                behavior_label=None,
                behavior_signature=torch.zeros(0, dtype=torch.float32),
            )
        )
        biased_result = biased.step(partial_lm_state, learn=False)
        debug = biased.get_evidence_debug()

        self.assertGreater(
            debug["query_bias_norms"]["predicted_appearance_prior_norm"],
            0.0,
        )
        self.assertGreater(
            debug["query_bias_norms"]["predicted_change_prior_norm"],
            0.0,
        )
        self.assertEqual(
            baseline_debug["base_evidence"][appearance_object_id],
            debug["base_evidence"][appearance_object_id],
        )
        self.assertGreater(
            abs(
                debug["after_temporal_behavior_evidence"][appearance_object_id]
                - baseline_debug["after_temporal_behavior_evidence"][appearance_object_id]
            ),
            5e-5,
        )
        self.assertNotEqual(
            biased_result["final_evidence"][appearance_object_id],
            baseline_evidence[appearance_object_id],
        )

    def test_child_vote_behavior_completes_partial_cue(self):
        observation = _make_observation()
        stable_state = _make_state()
        change_state = _make_state(
            {
                "flow_direction": np.array([0.0, 1.0, 0.0], dtype=np.float32),
                "flow_magnitude": np.array([0.65], dtype=np.float32),
                "delta_hsv": np.array([0.2, -0.1, 0.3], dtype=np.float32),
            }
        )
        stable_packet = build_visual_detail_packet(
            observation,
            sender_id="camera",
            grid_shape=(5, 5),
            state=stable_state,
        )
        change_packet = build_change_detail_packet(
            observation,
            change_state,
            sender_id="change",
            grid_shape=(5, 5),
        )
        partial_packet = copy.deepcopy(stable_packet)
        partial_packet["cells"] = [copy.deepcopy(stable_packet["cells"][0])]
        partial_packet["cell_count"] = 1
        partial_packet["grid_shape"] = [1, 1]
        stable_lm_state = _make_state({"observation_packet_v2": stable_packet})
        partial_lm_state = _make_state({"observation_packet_v2": partial_packet})
        change_lm_state = _make_state(
            {
                "flow_direction": np.array([0.0, 1.0, 0.0], dtype=np.float32),
                "flow_magnitude": np.array([0.65], dtype=np.float32),
                "delta_hsv": np.array([0.2, -0.1, 0.3], dtype=np.float32),
                "observation_packet_v2": change_packet,
            }
        )

        trained = PredictiveHypothesisCore(context_dim=48)
        trained.pre_episode(mode=None, object_name="stable_anchor")
        trained.step(stable_lm_state, learn=True)
        stable_object_id = trained.get_all_known_object_ids()[0]
        trained.pre_episode(mode=None, object_name="change_anchor")
        change_result = trained.step(change_lm_state, learn=True)
        known_ids = trained.get_all_known_object_ids()
        change_object_id = next(object_id for object_id in known_ids if object_id != stable_object_id)
        saved_state = trained.state_dict()

        baseline = PredictiveHypothesisCore(context_dim=48)
        baseline.load_state_dict(saved_state)
        baseline.pre_episode(mode=None, object_name=None)
        baseline_result = baseline.step(partial_lm_state, learn=False)
        self.assertEqual(baseline_result["mlh"]["graph_id"], stable_object_id)

        biased = PredictiveHypothesisCore(context_dim=48)
        biased.load_state_dict(saved_state)
        biased.pre_episode(mode=None, object_name=None)
        biased.receive_vote_message(
            PredictiveVoteMessage(
                sender_id="child_lm",
                sensed_pose_rel_body=np.vstack(
                    [
                        np.asarray(partial_lm_state.location, dtype=np.float64).reshape(1, 3),
                        np.asarray(
                            partial_lm_state.morphological_features["pose_vectors"],
                            dtype=np.float64,
                        ),
                    ]
                ),
                active_cells=torch.zeros(0, dtype=torch.float32),
                behavior_label="temporal_change",
                behavior_signature=torch.as_tensor(
                    np.asarray(
                        change_result["mlh"].get("behavior_signature", np.zeros(0)),
                        dtype=np.float32,
                    ),
                    dtype=torch.float32,
                ),
                ranked_hypotheses=[],
                possible_states={},
            )
        )
        biased_result = biased.step(partial_lm_state, learn=False)

        self.assertEqual(biased_result["mlh"]["graph_id"], change_object_id)
        final_evidence = biased_result["final_evidence"]
        self.assertGreater(final_evidence[change_object_id], final_evidence[stable_object_id])

    def test_child_vote_appearance_biases_partial_cue_toward_matching_object(self):
        observation_a = _make_observation()
        observation_b = _make_observation()
        observation_b["rgba"] = observation_b["rgba"].copy()
        observation_b["rgba"][..., 0] = 255 - observation_b["rgba"][..., 0]
        observation_b["rgba"][..., 1] = 255 - observation_b["rgba"][..., 1]

        packet_a = build_visual_detail_packet(
            observation_a,
            sender_id="camera",
            grid_shape=(5, 5),
            state=_make_state(),
        )
        packet_b = build_change_detail_packet(
            observation_b,
            _make_state(
                {
                    "flow_direction": np.array([0.0, 1.0, 0.0], dtype=np.float32),
                    "flow_magnitude": np.array([0.95], dtype=np.float32),
                    "delta_hsv": np.array([0.7, -0.5, 0.6], dtype=np.float32),
                }
            ),
            sender_id="change",
            grid_shape=(5, 5),
        )
        partial_packet = copy.deepcopy(packet_a)
        partial_packet["cells"] = [copy.deepcopy(packet_a["cells"][0])]
        partial_packet["cell_count"] = 1
        partial_packet["grid_shape"] = [1, 1]
        lm_state_a = _make_state({"observation_packet_v2": packet_a})
        lm_state_b = _make_state(
            {
                "flow_direction": np.array([0.0, 1.0, 0.0], dtype=np.float32),
                "flow_magnitude": np.array([0.95], dtype=np.float32),
                "delta_hsv": np.array([0.7, -0.5, 0.6], dtype=np.float32),
                "observation_packet_v2": packet_b,
            }
        )
        partial_lm_state = _make_state({"observation_packet_v2": partial_packet})

        trained = PredictiveHypothesisCore(context_dim=48)
        trained.pre_episode(mode=None, object_name=None)
        result_a = trained.step(lm_state_a, learn=True)
        trained.step(lm_state_a, learn=True)
        result_b = trained.step(lm_state_b, learn=True)
        stable_object_id = result_a["mlh"]["graph_id"]
        known_ids = trained.get_all_known_object_ids()
        appearance_object_id = next(
            object_id for object_id in known_ids if object_id != stable_object_id
        )
        saved_state = trained.state_dict()

        baseline = PredictiveHypothesisCore(context_dim=48)
        baseline.load_state_dict(saved_state)
        baseline.joint_behavior_score_weight = 0.0
        baseline.vote_behavior_score_weight = 0.0
        baseline.pre_episode(mode=None, object_name=None)
        baseline_result = baseline.step(partial_lm_state, learn=False)
        baseline_evidence = dict(baseline_result["final_evidence"])

        biased = PredictiveHypothesisCore(context_dim=48)
        biased.load_state_dict(saved_state)
        biased.joint_behavior_score_weight = 0.0
        biased.vote_behavior_score_weight = 0.0
        biased.pre_episode(mode=None, object_name=None)
        biased.receive_vote_message(
            PredictiveVoteMessage(
                sender_id="child_lm",
                sensed_pose_rel_body=np.vstack(
                    [
                        np.asarray(partial_lm_state.location, dtype=np.float64).reshape(1, 3),
                        np.asarray(
                            partial_lm_state.morphological_features["pose_vectors"],
                            dtype=np.float64,
                        ),
                    ]
                ),
                active_cells=torch.zeros(0, dtype=torch.float32),
                appearance_signature=torch.as_tensor(
                    np.asarray(
                        result_b["mlh"].get("appearance_signature", np.zeros(0)),
                        dtype=np.float32,
                    ),
                    dtype=torch.float32,
                ),
                behavior_label=None,
                behavior_signature=torch.zeros(0, dtype=torch.float32),
                ranked_hypotheses=[],
                possible_states={},
            )
        )
        biased_result = biased.step(partial_lm_state, learn=False)

        self.assertGreater(
            biased_result["final_evidence"][appearance_object_id],
            baseline_evidence[appearance_object_id],
        )

    def test_core_keeps_latent_outputs_even_with_supervised_episode_metadata(self):
        observation = _make_observation()
        packet = build_visual_detail_packet(
            observation,
            sender_id="camera",
            grid_shape=(5, 5),
            state=_make_state(),
        )
        lm_state = _make_state({"observation_packet_v2": packet})

        core = PredictiveHypothesisCore(context_dim=48)
        core.pre_episode(mode=None, object_name="fox:0")
        core.step(lm_state, learn=True)

        raw_known_ids = core.get_all_known_object_ids()
        self.assertEqual(len(raw_known_ids), 1)
        self.assertTrue(raw_known_ids[0].startswith("latent_object_"))
        self.assertEqual(core.get_all_known_latent_ids(), raw_known_ids)

        result = core.step(lm_state, learn=False)
        self.assertEqual(result["mlh"]["graph_id"], raw_known_ids[0])
        self.assertEqual(result["mlh"]["latent_id"], raw_known_ids[0])
        self.assertEqual(core.get_current_mlh()["graph_id"], raw_known_ids[0])
        self.assertEqual(core.get_current_mlh()["latent_id"], raw_known_ids[0])
        self.assertIn(raw_known_ids[0], result["evidence"])
        support = core.get_evidence_debug()["self_supervised_support"]
        self.assertEqual(support["learning"]["learning_mode"], "disabled")
        self.assertIsNone(support["learning"]["learning_latent_id"])

    def test_lm_uses_typed_vote_and_context_messages_at_boundary(self):
        observation = _make_observation()
        state = _make_state()
        packet = build_visual_detail_packet(
            observation,
            sender_id="camera",
            grid_shape=(5, 5),
            state=state,
        )
        lm_state = _make_state({"observation_packet_v2": packet})

        sender = PredictiveHypothesisTorchLM(
            learning_module_id="lm_sender",
            output_evidence_threshold=0.0,
            core_kwargs={"context_dim": 48},
        )
        receiver = PredictiveHypothesisTorchLM(
            learning_module_id="lm_receiver",
            output_evidence_threshold=0.0,
            core_kwargs={"context_dim": 48},
        )

        sender.pre_episode(primary_target={"object": "fox", "state": 0})
        receiver.pre_episode(primary_target={"object": "robot", "state": 1})
        sender.exploratory_step(None, [lm_state])

        vote_payload = sender.send_out_vote()
        vote_message = sender.get_last_vote_message()
        context_payload = sender.get_context_signal()
        context_message = sender.get_context_message()

        self.assertIsNotNone(vote_payload)
        self.assertIsInstance(vote_message, PredictiveVoteMessage)
        self.assertGreater(len(vote_message.ranked_hypotheses), 0)
        self.assertEqual(vote_payload["sender_id"], "lm_sender")
        self.assertGreater(vote_message.active_cells.numel(), 0)
        self.assertGreater(vote_message.appearance_signature.numel(), 0)
        self.assertIsNotNone(vote_message.behavior_label)
        self.assertGreater(vote_message.behavior_signature.numel(), 0)
        self.assertTrue(
            all(
                str(hypothesis.object_id).startswith("latent_object_")
                for hypothesis in vote_message.ranked_hypotheses
            )
        )
        self.assertIsNotNone(vote_message.ranked_hypotheses[0].chart_id)
        self.assertIsNotNone(vote_message.ranked_hypotheses[0].behavior_label)
        self.assertGreater(
            vote_message.ranked_hypotheses[0].appearance_signature.numel(),
            0,
        )
        self.assertGreater(
            vote_message.ranked_hypotheses[0].behavior_signature.numel(),
            0,
        )
        raw_vote_object_id = vote_message.ranked_hypotheses[0].object_id
        self.assertIn(raw_vote_object_id, vote_payload["possible_states"])
        self.assertGreaterEqual(
            len(vote_payload["possible_states"][raw_vote_object_id]),
            1,
        )
        self.assertTrue(
            all(
                state.non_morphological_features.get("chart_id") is not None
                for state in vote_payload["possible_states"][raw_vote_object_id]
            )
        )
        self.assertIsNotNone(context_payload)
        self.assertIsInstance(context_message, PredictiveContextSignal)
        self.assertEqual(context_payload["sender_id"], "lm_sender")
        self.assertEqual(context_payload["routing_scope"], "graph_neighbors")

        self.assertIsNone(context_payload.get("graph_id"))
        self.assertIsNone(context_payload.get("latent_id"))
        self.assertIsNotNone(context_message.chart_id)
        self.assertEqual(context_payload.get("chart_id"), context_message.chart_id)
        self.assertGreater(len(context_message.ranked_hypotheses), 0)
        self.assertGreater(len(context_payload.get("ranked_hypotheses", [])), 0)
        self.assertEqual(
            context_message.ranked_hypotheses[0].chart_id,
            context_message.chart_id,
        )
        self.assertGreater(context_message.appearance_signature.numel(), 0)
        self.assertGreater(context_message.predicted_appearance_signature.numel(), 0)
        self.assertGreater(context_message.predicted_change_signature.numel(), 0)
        self.assertIsNotNone(context_message.behavior_label)
        self.assertGreater(context_message.behavior_signature.numel(), 0)
        message_state = sender._core.get_last_message_state()
        self.assertEqual(message_state.latent_id, raw_vote_object_id)
        self.assertEqual(message_state.graph_id, raw_vote_object_id)
        output_state = sender.get_output()
        self.assertIsNotNone(output_state)
        self.assertGreater(
            np.asarray(
                output_state.non_morphological_features.get(
                    "appearance_signature",
                    np.zeros(0, dtype=np.float32),
                ),
                dtype=np.float32,
            ).size,
            0,
        )
        self.assertGreater(
            np.asarray(
                output_state.non_morphological_features.get(
                    "predicted_appearance_signature",
                    np.zeros(0, dtype=np.float32),
                ),
                dtype=np.float32,
            ).size,
            0,
        )
        self.assertGreater(
            np.asarray(
                output_state.non_morphological_features.get(
                    "predicted_change_signature",
                    np.zeros(0, dtype=np.float32),
                ),
                dtype=np.float32,
            ).size,
            0,
        )
        self.assertGreater(
            np.asarray(
                output_state.non_morphological_features.get(
                    "appearance_residual",
                    np.zeros(0, dtype=np.float32),
                ),
                dtype=np.float32,
            ).size,
            0,
        )
        self.assertGreater(
            np.asarray(
                output_state.non_morphological_features.get(
                    "change_residual",
                    np.zeros(0, dtype=np.float32),
                ),
                dtype=np.float32,
            ).size,
            0,
        )
        self.assertIn(
            "appearance_prediction_error",
            output_state.non_morphological_features,
        )
        self.assertIn(
            "change_prediction_error",
            output_state.non_morphological_features,
        )
        self.assertEqual(
            output_state.non_morphological_features.get("graph_id"),
            raw_vote_object_id,
        )
        self.assertEqual(
            output_state.non_morphological_features.get("latent_id"),
            raw_vote_object_id,
        )

        sender.post_episode()
        self.assertIn("fox", sender.latent_id_to_target[raw_vote_object_id])
        self.assertIn("fox", sender.graph_id_to_target[raw_vote_object_id])
        self.assertIn(raw_vote_object_id, sender.target_to_latent_id["fox"])
        self.assertIn(raw_vote_object_id, sender.target_to_graph_id["fox"])

        saved_state = sender.state_dict()
        self.assertEqual(
            saved_state["latent_id_to_target"],
            saved_state["graph_id_to_target"],
        )
        self.assertEqual(
            saved_state["target_to_latent_id"],
            saved_state["target_to_graph_id"],
        )

        restored_sender = PredictiveHypothesisTorchLM(
            learning_module_id="lm_restored",
            output_evidence_threshold=0.0,
            core_kwargs={"context_dim": 48},
        )
        saved_state = dict(saved_state)
        saved_state.pop("graph_id_to_target", None)
        saved_state.pop("target_to_graph_id", None)
        restored_sender.load_state_dict(saved_state)
        self.assertIn("fox", restored_sender.latent_id_to_target[raw_vote_object_id])
        self.assertIn(raw_vote_object_id, restored_sender.target_to_latent_id["fox"])

        receiver.receive_votes(vote_payload)
        receiver.receive_context(**context_payload)

        received_votes = receiver.get_last_received_vote_messages()
        received_context = receiver.get_last_received_context_message()
        self.assertEqual(len(received_votes), 1)
        self.assertIsInstance(received_votes[0], PredictiveVoteMessage)
        self.assertIsInstance(received_context, PredictiveContextSignal)
        self.assertEqual(received_context.sender_id, "lm_sender")
        self.assertIsNone(received_context.graph_id)
        self.assertIsNone(received_context.latent_id)
        self.assertIsNotNone(received_context.chart_id)
        self.assertGreater(len(received_context.ranked_hypotheses), 0)
        self.assertEqual(tuple(received_context.active_cells.shape), (48,))
        self.assertEqual(
            received_votes[0].ranked_hypotheses[0].latent_id,
            raw_vote_object_id,
        )
        self.assertGreater(received_votes[0].appearance_signature.numel(), 0)
        self.assertGreater(
            received_votes[0].ranked_hypotheses[0].appearance_signature.numel(),
            0,
        )
        self.assertGreater(received_context.appearance_signature.numel(), 0)
        self.assertGreater(received_context.predicted_appearance_signature.numel(), 0)
        self.assertGreater(received_context.predicted_change_signature.numel(), 0)
        self.assertIsNotNone(received_context.behavior_label)
        self.assertGreater(received_context.behavior_signature.numel(), 0)
        self.assertEqual(
            received_context.ranked_hypotheses[0].chart_id,
            context_message.chart_id,
        )
        self.assertEqual(receiver._core._predicted_latent_ids[0], raw_vote_object_id)

    def test_send_out_vote_keeps_multiple_chart_states_per_latent(self):
        lm = PredictiveHypothesisTorchLM(
            learning_module_id="lm_sender",
            output_evidence_threshold=0.0,
            vote_top_k=3,
            core_kwargs={"context_dim": 48},
        )
        lm._stepped = True
        lm._last_observed_state = _make_state()
        lm._last_result = {
            "ranked_hypotheses": [
                {
                    "object_id": "latent_object_0",
                    "latent_id": "latent_object_0",
                    "chart_id": "latent_object_0#chart0",
                    "probability": 0.52,
                    "evidence": 1.20,
                    "rank": 1,
                    "location": np.array([0.1, 0.2, 0.3], dtype=np.float64),
                    "pose_vectors": np.eye(3, dtype=np.float64),
                    "appearance_signature": np.array([1.0, 0.0], dtype=np.float32),
                    "behavior_label": "stable_surface",
                    "behavior_signature": np.array([1.0, 0.0], dtype=np.float32),
                },
                {
                    "object_id": "latent_object_0",
                    "latent_id": "latent_object_0",
                    "chart_id": "latent_object_0#chart1",
                    "probability": 0.48,
                    "evidence": 1.05,
                    "rank": 2,
                    "location": np.array([0.1, 0.2, 0.3], dtype=np.float64),
                    "pose_vectors": Rotation.from_euler(
                        "z",
                        25.0,
                        degrees=True,
                    ).as_matrix().astype(np.float64),
                    "appearance_signature": np.array([0.0, 1.0], dtype=np.float32),
                    "behavior_label": "temporal_change",
                    "behavior_signature": np.array([0.0, 1.0], dtype=np.float32),
                },
                {
                    "object_id": "latent_object_1",
                    "latent_id": "latent_object_1",
                    "chart_id": "latent_object_1#chart0",
                    "probability": 0.25,
                    "evidence": 0.70,
                    "rank": 3,
                    "location": np.array([0.1, 0.2, 0.3], dtype=np.float64),
                    "pose_vectors": np.eye(3, dtype=np.float64),
                    "appearance_signature": np.array([0.4, 0.6], dtype=np.float32),
                    "behavior_label": "stable_surface",
                    "behavior_signature": np.array([0.5, 0.5], dtype=np.float32),
                },
            ]
        }

        vote_payload = lm.send_out_vote()

        self.assertIsNotNone(vote_payload)
        self.assertEqual(
            len(vote_payload["possible_states"]["latent_object_0"]),
            2,
        )
        self.assertEqual(
            {
                state.non_morphological_features.get("chart_id")
                for state in vote_payload["possible_states"]["latent_object_0"]
            },
            {"latent_object_0#chart0", "latent_object_0#chart1"},
        )
        self.assertTrue(
            all(
                state.non_morphological_features.get("latent_id")
                == "latent_object_0"
                for state in vote_payload["possible_states"]["latent_object_0"]
            )
        )

    def test_lm_records_reporting_alias_mismatches_between_learning_and_output(self):
        lm = PredictiveHypothesisTorchLM(
            learning_module_id="lm_alias_debug",
            output_evidence_threshold=0.0,
            core_kwargs={"context_dim": 48},
        )
        lm.pre_episode(primary_target={"object": "fox", "state": 0})
        lm._step_count = 3

        lm._register_reporting_target(
            {
                "self_supervised_support": {
                    "learning": {
                        "learning_latent_id": "latent_object_1",
                        "learning_chart_id": "latent_object_1#chart0",
                        "selected_source": "novel",
                        "selection_reason": "posterior_novel",
                        "selected_posterior_probability": 0.62,
                    }
                },
                "mlh": {
                    "latent_id": "latent_object_0",
                    "graph_id": "latent_object_0",
                    "chart_id": "latent_object_0#chart0",
                    "evidence": 0.91,
                },
                "ranked_hypotheses": [
                    {
                        "object_id": "latent_object_0",
                        "latent_id": "latent_object_0",
                        "chart_id": "latent_object_0#chart0",
                        "probability": 0.73,
                        "evidence": 0.91,
                        "source": "memory",
                        "age": 0,
                    },
                    {
                        "object_id": "latent_object_1",
                        "latent_id": "latent_object_1",
                        "chart_id": "latent_object_1#chart0",
                        "probability": 0.27,
                        "evidence": 0.64,
                        "source": "novel",
                        "age": 0,
                    },
                ],
            }
        )

        diagnostics = lm.get_reporting_alias_diagnostics()
        self.assertEqual(diagnostics["total_steps"], 1)
        self.assertEqual(diagnostics["agreement_steps"], 0)
        self.assertEqual(diagnostics["mismatch_steps"], 1)
        fox_stats = diagnostics["per_target"]["fox"]
        self.assertEqual(fox_stats["learning_latent_counts"]["latent_object_1"], 1)
        self.assertEqual(fox_stats["output_latent_counts"]["latent_object_0"], 1)
        self.assertEqual(fox_stats["registered_latent_counts"]["latent_object_0"], 1)
        self.assertEqual(fox_stats["registered_latent_counts"]["latent_object_1"], 1)
        self.assertEqual(
            fox_stats["learning_output_pairs"]["latent_object_1->latent_object_0"],
            1,
        )
        self.assertEqual(diagnostics["last_event"]["target_object"], "fox")
        self.assertFalse(diagnostics["last_event"]["agreement"])
        self.assertEqual(diagnostics["last_event"]["learning_rank"], 2)

    def test_lm_prefers_exclusive_episode_registration_for_child_lms(self):
        lm = PredictiveHypothesisTorchLM(
            learning_module_id="lm_child_registration",
            output_evidence_threshold=0.0,
            core_kwargs={"context_dim": 48},
        )
        lm.latent_id_to_target["latent_object_0"].add("fox")
        lm.target_to_latent_id["fox"].add("latent_object_0")
        lm.pre_episode(primary_target={"object": "robot", "state": 0})
        lm._last_observed_state = _make_state()

        def register_step(
            *,
            step_count,
            learning_latent_id,
            output_latent_id,
            learning_probability,
            output_probability,
        ):
            lm._step_count = step_count
            lm._register_reporting_target(
                {
                    "self_supervised_support": {
                        "learning": {
                            "learning_latent_id": learning_latent_id,
                            "learning_chart_id": f"{learning_latent_id}#chart0",
                            "selected_source": "memory",
                            "selection_reason": "posterior_existing",
                            "selected_posterior_probability": learning_probability,
                        }
                    },
                    "mlh": {
                        "latent_id": output_latent_id,
                        "graph_id": output_latent_id,
                        "chart_id": f"{output_latent_id}#chart0",
                        "evidence": output_probability,
                    },
                    "ranked_hypotheses": [
                        {
                            "object_id": output_latent_id,
                            "latent_id": output_latent_id,
                            "chart_id": f"{output_latent_id}#chart0",
                            "probability": output_probability,
                            "evidence": output_probability,
                            "source": "memory",
                            "age": 0,
                        },
                        {
                            "object_id": learning_latent_id,
                            "latent_id": learning_latent_id,
                            "chart_id": f"{learning_latent_id}#chart0",
                            "probability": learning_probability,
                            "evidence": learning_probability,
                            "source": "memory",
                            "age": 0,
                        },
                    ],
                }
            )

        register_step(
            step_count=1,
            learning_latent_id="latent_object_0",
            output_latent_id="latent_object_0",
            learning_probability=0.82,
            output_probability=0.80,
        )
        register_step(
            step_count=2,
            learning_latent_id="latent_object_0",
            output_latent_id="latent_object_0",
            learning_probability=0.84,
            output_probability=0.81,
        )
        register_step(
            step_count=3,
            learning_latent_id="latent_object_2",
            output_latent_id="latent_object_2",
            learning_probability=0.72,
            output_probability=0.70,
        )

        lm.post_episode()

        self.assertIn("robot", lm.graph_id_to_target["latent_object_2"])
        self.assertIn("latent_object_2", lm.target_to_graph_id["robot"])
        self.assertNotIn("robot", lm.graph_id_to_target["latent_object_0"])
        diagnostics = lm.get_reporting_alias_diagnostics()
        self.assertEqual(diagnostics["last_commit"]["selected_latent_id"], "latent_object_2")
        self.assertEqual(
            diagnostics["last_commit"]["selection_reason"],
            "best_exclusive_named_anchor",
        )

    def test_predictive_context_signal_accepts_graph_id_aliases(self):
        signal = PredictiveContextSignal(
            graph_id="latent_object_0",
            chart_id="latent_object_0#chart0",
            confidence=0.8,
            residual=0.1,
            active_cells=torch.ones(4, dtype=torch.float32),
            child_graph_ids=["latent_object_1"],
            ranked_hypotheses=[
                RankedHypothesisVote(
                    object_id="latent_object_0",
                    chart_id="latent_object_0#chart0",
                    probability=0.8,
                    evidence=1.2,
                    rank=1,
                )
            ],
        )

        self.assertEqual(signal.latent_id, "latent_object_0")
        self.assertEqual(signal.graph_id, "latent_object_0")
        self.assertEqual(signal.chart_id, "latent_object_0#chart0")
        self.assertEqual(signal.child_latent_ids, ["latent_object_1"])
        self.assertEqual(signal.child_graph_ids, ["latent_object_1"])
        payload = signal.to_dict()
        self.assertEqual(payload["latent_id"], payload["graph_id"])
        self.assertEqual(payload["chart_id"], "latent_object_0#chart0")
        self.assertEqual(payload["child_latent_ids"], payload["child_graph_ids"])
        self.assertEqual(payload["ranked_hypotheses"][0]["chart_id"], "latent_object_0#chart0")
        restored = PredictiveContextSignal.from_dict(
            payload,
            device=torch.device("cpu"),
        )
        self.assertEqual(restored.chart_id, "latent_object_0#chart0")
        self.assertEqual(restored.ranked_hypotheses[0].chart_id, "latent_object_0#chart0")

    def test_core_receive_context_message_uses_ranked_hypotheses_when_latent_id_is_stripped(self):
        core = PredictiveHypothesisCore(context_dim=48)

        core.receive_context_message(
            PredictiveContextSignal(
                latent_id=None,
                chart_id="latent_object_0#chart1",
                confidence=0.9,
                residual=0.05,
                active_cells=torch.zeros(48, dtype=torch.float32),
                appearance_signature=torch.zeros(0, dtype=torch.float32),
                behavior_label=None,
                behavior_signature=torch.zeros(0, dtype=torch.float32),
                ranked_hypotheses=[
                    RankedHypothesisVote(
                        object_id="latent_object_0",
                        chart_id="latent_object_0#chart1",
                        probability=0.7,
                        evidence=1.3,
                        rank=1,
                        behavior_label="temporal_change",
                        behavior_signature=torch.tensor([0.0, 1.0], dtype=torch.float32),
                        appearance_signature=torch.tensor([0.2, 0.8], dtype=torch.float32),
                    ),
                    RankedHypothesisVote(
                        object_id="latent_object_1",
                        chart_id="latent_object_1#chart0",
                        probability=0.3,
                        evidence=0.9,
                        rank=2,
                    ),
                ],
            )
        )

        self.assertEqual(core._predicted_latent_ids, ["latent_object_0", "latent_object_1"])
        self.assertGreater(core._external_appearance_signature.numel(), 0)
        self.assertGreater(core._external_behavior_signature.numel(), 0)
        self.assertEqual(core._external_behavior_label, "temporal_change")

    def test_lm_targets_context_to_fused_child_senders(self):
        lm = PredictiveHypothesisTorchLM(
            learning_module_id="lm_parent",
            output_evidence_threshold=0.0,
            core_kwargs={"context_dim": 48},
        )
        lm._last_observed_state = State(
            location=np.zeros(3, dtype=np.float64),
            morphological_features={
                "pose_vectors": np.eye(3, dtype=np.float64),
                "pose_fully_defined": True,
                "on_object": True,
            },
            non_morphological_features={
                "child_sender_ids": ["lm_morphology", "lm_behavior", "lm_morphology"],
                "child_latent_ids": ["latent_object_0"],
            },
            confidence=0.8,
            use_state=True,
            sender_id="fused:lm_parent",
            sender_type="LM",
        )
        lm._core.get_context_signal_message = lambda: PredictiveContextSignal(
            latent_id="latent_object_0",
            confidence=0.8,
            residual=0.05,
            active_cells=torch.ones(48, dtype=torch.float32),
            predicted_appearance_signature=torch.ones(8, dtype=torch.float32),
            predicted_change_signature=torch.ones(8, dtype=torch.float32),
        )

        context_message = lm.get_context_message()
        context_payload = lm.get_context_signal()

        self.assertIsNotNone(context_message)
        self.assertEqual(context_message.routing_scope, "targeted")
        self.assertEqual(
            context_message.target_sender_ids,
            ["lm_morphology", "lm_behavior"],
        )
        self.assertIsNone(context_message.graph_id)
        self.assertEqual(context_message.sender_id, "lm_parent")
        self.assertEqual(context_payload["routing_scope"], "targeted")
        self.assertEqual(
            context_payload["target_sender_ids"],
            ["lm_morphology", "lm_behavior"],
        )

    def test_dispatch_routes_predictive_context_to_targeted_children(self):
        class MockLM:
            def __init__(self, lm_id, signal=None):
                self.learning_module_id = lm_id
                self._signal = signal
                self.received_contexts = []

            def get_context_signal(self):
                return None if self._signal is None else copy.deepcopy(self._signal)

            def receive_context(self, **context_signal):
                self.received_contexts.append(context_signal)

        child_signal = PredictiveContextSignal(
            latent_id=None,
            confidence=0.7,
            residual=0.1,
            active_cells=torch.ones(8, dtype=torch.float32),
            sender_id="lm_morphology",
            routing_scope="none",
        ).to_dict()
        behavior_signal = PredictiveContextSignal(
            latent_id=None,
            confidence=0.7,
            residual=0.1,
            active_cells=torch.ones(8, dtype=torch.float32),
            sender_id="lm_behavior",
            routing_scope="none",
        ).to_dict()
        parent_signal = PredictiveContextSignal(
            latent_id=None,
            confidence=0.9,
            residual=0.05,
            active_cells=torch.ones(8, dtype=torch.float32),
            sender_id="lm_parent",
            routing_scope="targeted",
            target_sender_ids=["lm_morphology", "lm_behavior"],
        ).to_dict()

        lm_morph = MockLM("lm_morphology", signal=child_signal)
        lm_behavior = MockLM("lm_behavior", signal=behavior_signal)
        lm_parent = MockLM("lm_parent", signal=parent_signal)

        monty = MontyForGraphMatching.__new__(MontyForGraphMatching)
        monty.learning_modules = [lm_morph, lm_behavior, lm_parent]

        monty._dispatch_context_signals()

        self.assertEqual(len(lm_morph.received_contexts), 1)
        self.assertEqual(len(lm_behavior.received_contexts), 1)
        self.assertEqual(len(lm_parent.received_contexts), 0)
        self.assertEqual(lm_morph.received_contexts[0]["sender_id"], "lm_parent")
        self.assertEqual(lm_behavior.received_contexts[0]["sender_id"], "lm_parent")

    def test_dispatch_routes_predictive_context_to_graph_neighbors(self):
        class MockLM:
            def __init__(self, lm_id, signal=None):
                self.learning_module_id = lm_id
                self._signal = signal
                self.received_contexts = []

            def get_context_signal(self):
                return None if self._signal is None else copy.deepcopy(self._signal)

            def receive_context(self, **context_signal):
                self.received_contexts.append(context_signal)

        morph_signal = PredictiveContextSignal(
            latent_id=None,
            confidence=0.7,
            residual=0.1,
            active_cells=torch.ones(8, dtype=torch.float32),
            sender_id="lm_morphology",
            routing_scope="graph_neighbors",
        ).to_dict()
        behavior_signal = PredictiveContextSignal(
            latent_id=None,
            confidence=0.7,
            residual=0.1,
            active_cells=torch.ones(8, dtype=torch.float32),
            sender_id="lm_behavior",
            routing_scope="graph_neighbors",
        ).to_dict()

        lm_morph = MockLM("lm_morphology", signal=morph_signal)
        lm_behavior = MockLM("lm_behavior", signal=behavior_signal)
        lm_parent = MockLM("lm_parent", signal=None)

        monty = MontyForGraphMatching.__new__(MontyForGraphMatching)
        monty.learning_modules = [lm_morph, lm_behavior, lm_parent]
        monty.lm_to_lm_vote_matrix = [[1], [0], []]
        monty.lm_to_lm_matrix = [[], [], [0, 1]]

        monty._dispatch_context_signals()

        self.assertEqual(len(lm_morph.received_contexts), 1)
        self.assertEqual(len(lm_behavior.received_contexts), 1)
        self.assertEqual(len(lm_parent.received_contexts), 2)
        self.assertEqual(lm_morph.received_contexts[0]["sender_id"], "lm_behavior")
        self.assertEqual(lm_behavior.received_contexts[0]["sender_id"], "lm_morphology")
        self.assertEqual(
            {context["sender_id"] for context in lm_parent.received_contexts},
            {"lm_morphology", "lm_behavior"},
        )

    def test_evidence_vote_combiner_preserves_chart_ranked_hypotheses(self):
        pose = np.vstack(
            [
                np.zeros((1, 3), dtype=np.float64),
                np.eye(3, dtype=np.float64),
            ]
        )
        receiver_vote = {
            "sender_id": "lm_receiver",
            "sensed_pose_rel_body": pose,
            "possible_states": {},
            "ranked_hypotheses": [],
        }
        sender_a_vote = {
            "sender_id": "lm_a",
            "sensed_pose_rel_body": pose,
            "possible_states": {},
            "ranked_hypotheses": [
                {
                    "object_id": "latent_object_0",
                    "latent_id": "latent_object_0",
                    "chart_id": "latent_object_0#chart0",
                    "probability": 0.6,
                    "evidence": 1.2,
                    "rank": 1,
                }
            ],
        }
        sender_b_vote = {
            "sender_id": "lm_b",
            "sensed_pose_rel_body": pose,
            "possible_states": {},
            "ranked_hypotheses": [
                {
                    "object_id": "latent_object_0",
                    "latent_id": "latent_object_0",
                    "chart_id": "latent_object_0#chart0",
                    "probability": 0.2,
                    "evidence": 0.9,
                    "rank": 2,
                },
                {
                    "object_id": "latent_object_0",
                    "latent_id": "latent_object_0",
                    "chart_id": "latent_object_0#chart1",
                    "probability": 0.5,
                    "evidence": 1.1,
                    "rank": 1,
                },
            ],
        }

        monty = MontyForEvidenceGraphMatching.__new__(
            MontyForEvidenceGraphMatching
        )
        monty.learning_modules = [object(), object(), object()]
        monty.lm_to_lm_vote_matrix = [[1, 2], [], []]

        combined_votes = monty._combine_votes(
            [receiver_vote, sender_a_vote, sender_b_vote]
        )
        ranked = combined_votes[0]["ranked_hypotheses"]

        self.assertEqual(len(ranked), 2)
        self.assertEqual(
            [hypothesis.get("chart_id") for hypothesis in ranked],
            ["latent_object_0#chart0", "latent_object_0#chart1"],
        )
        self.assertEqual(ranked[0]["object_id"], "latent_object_0")
        self.assertEqual(ranked[0]["latent_id"], "latent_object_0")
        self.assertAlmostEqual(ranked[0]["probability"], 0.8, places=6)
        self.assertAlmostEqual(ranked[0]["evidence"], 1.2, places=6)
        self.assertEqual(ranked[0]["sender_ids"], ["lm_a", "lm_b"])
        self.assertEqual(ranked[0]["rank"], 1)
        self.assertEqual(ranked[1]["sender_ids"], ["lm_b"])
        self.assertEqual(ranked[1]["rank"], 2)


class TestPredictiveHypothesisSharedMemory(unittest.TestCase):
    def test_hypothesis_bank_roundtrip_preserves_source_and_age(self):
        bank = HypothesisBank(max_hypotheses=3, temperature=0.2)
        bank.update(
            {"latent_object_0": 1.25},
            location=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            pose_vectors=np.eye(3, dtype=np.float32),
            appearance_vector=np.array([0.2, 0.4], dtype=np.float32),
            behavior_vector=np.array([0.6, 0.8], dtype=np.float32),
            behavior_label="stable_surface",
            joint_candidates=[
                {
                    "object_id": "latent_object_0",
                    "chart_id": "latent_object_0#chart0",
                    "score": 1.25,
                    "location": np.array([0.1, 0.2, 0.3], dtype=np.float32),
                    "pose_vectors": np.eye(3, dtype=np.float32),
                    "appearance_vector": np.array([0.2, 0.4], dtype=np.float32),
                    "behavior_vector": np.array([0.6, 0.8], dtype=np.float32),
                    "behavior_label": "stable_surface",
                    "source": "propagated",
                    "age": 3,
                }
            ],
        )

        restored = HypothesisBank(max_hypotheses=3, temperature=0.2)
        restored.load_state_dict(bank.state_dict())
        ranked = restored.as_ranked_hypotheses()

        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0]["source"], "propagated")
        self.assertEqual(ranked[0]["age"], 3)

    def test_shared_memory_returns_hopfield_retrieval_state_with_chart_ids(self):
        memory = AtlasMemory(
            embedding_dim=3,
            max_slots_per_object=4,
            insertion_similarity_threshold=0.999,
            topk_score_pool=2,
        )
        memory.observe("fox", torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32))
        memory.observe("fox", torch.tensor([0.8, 0.2, 0.0], dtype=torch.float32))
        memory.observe("robot", torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32))

        retrieval = memory.retrieve(
            torch.tensor([0.9, 0.1, 0.0], dtype=torch.float32)
        )

        self.assertIsInstance(retrieval, HopfieldRetrievalState)
        self.assertEqual(retrieval.latent_ids, retrieval.object_ids)
        self.assertEqual(retrieval.top_object_id, "fox")
        self.assertEqual(retrieval.top_latent_id, "fox")
        self.assertIsNotNone(retrieval.top_chart_id)
        self.assertEqual(memory.get_slot_state().num_slots, 3)
        self.assertEqual(
            memory.get_slot_state().latent_ids,
            memory.get_slot_state().object_ids,
        )
        self.assertEqual(
            len(memory.get_slot_state().slot_chart_ids),
            memory.get_slot_state().num_slots,
        )
        self.assertGreaterEqual(retrieval.iteration_count, 1)
        self.assertEqual(len(retrieval.energy_trace), retrieval.iteration_count)

    def test_shared_memory_scores_drop_when_competing_object_enters_slot_pool(self):
        memory = AtlasMemory(
            embedding_dim=3,
            max_slots_per_object=4,
            insertion_similarity_threshold=0.995,
            topk_score_pool=1,
        )
        query = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)

        memory.observe("fox", query)
        score_before = memory.score(query)

        memory.observe(
            "robot",
            torch.tensor([0.85, 0.15, 0.0], dtype=torch.float32),
        )
        score_after = memory.score(query)

        self.assertGreater(score_before["fox"], score_after["fox"])
        self.assertGreater(score_after["fox"], score_after["robot"])

    def test_shared_memory_accumulates_competition_across_multiple_global_slots(self):
        memory = AtlasMemory(
            embedding_dim=3,
            max_slots_per_object=4,
            insertion_similarity_threshold=0.995,
            topk_score_pool=2,
        )
        query = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)

        memory.observe("fox", query)
        fox_only_score = memory.score(query)["fox"]

        memory.observe(
            "robot",
            torch.tensor([0.85, 0.15, 0.0], dtype=torch.float32),
        )
        first_competition_score = memory.score(query)["fox"]

        memory.observe(
            "robot",
            torch.tensor([0.7, 0.3, 0.0], dtype=torch.float32),
        )
        second_competition = memory.score(query)

        self.assertEqual(memory.get_slot_state().num_slots, 3)
        self.assertGreater(fox_only_score, first_competition_score)
        self.assertNotAlmostEqual(
            first_competition_score,
            second_competition["fox"],
            places=6,
        )
        self.assertGreater(fox_only_score, second_competition["fox"])
        self.assertGreater(second_competition["fox"], second_competition["robot"])

    def test_shared_memory_novelty_gate_spawns_new_slot_before_similarity_collapse(self):
        baseline_memory = AtlasMemory(
            embedding_dim=3,
            max_slots_per_object=4,
            insertion_similarity_threshold=0.92,
        )
        novelty_memory = AtlasMemory(
            embedding_dim=3,
            max_slots_per_object=4,
            insertion_similarity_threshold=0.92,
        )
        base = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
        nearby_novel = torch.tensor([0.94, 0.34, 0.0], dtype=torch.float32)

        baseline_memory.observe("fox", base)
        baseline_memory.observe("fox", nearby_novel, novelty_signal=0.0)
        self.assertEqual(baseline_memory.get_slot_state().num_slots, 1)

        novelty_memory.observe("fox", base)
        novelty_memory.observe("fox", nearby_novel, novelty_signal=1.0)
        self.assertEqual(novelty_memory.get_slot_state().num_slots, 2)


class TestFixedSparseRecurrentMemory(unittest.TestCase):
    def test_texture_detail_shifts_support_and_generalizes(self):
        memory = FixedSparseRecurrentMemory(
            embedding_dim=12,
            substrate_cell_count=96,
            max_slots_per_object=4,
            insertion_similarity_threshold=0.98,
        )
        query = torch.ones(12, dtype=torch.float32)
        query = query / query.norm(p=2)

        horizontal_field = _make_texture_field("horizontal")
        vertical_field = _make_texture_field("vertical")
        horizontal_variant = _make_texture_field("horizontal", jitter=0.10)

        for _ in range(4):
            memory.observe("fox", query, observation_field=horizontal_field)
            memory.observe("robot", query, observation_field=vertical_field)

        memory.reset_episode_state()
        horizontal_retrieval = memory.retrieve(query, observation_field=horizontal_field)
        vertical_retrieval = memory.retrieve(query, observation_field=vertical_field)
        horizontal_variant_retrieval = memory.retrieve(
            query,
            observation_field=horizontal_variant,
        )

        self.assertEqual(horizontal_retrieval.top_object_id, "fox")
        self.assertEqual(horizontal_variant_retrieval.top_object_id, "fox")
        self.assertIsNotNone(horizontal_retrieval.top_chart_id)
        self.assertGreaterEqual(memory.get_slot_state().num_slots, 2)
        horizontal_scores = memory.retrieval_scores(horizontal_retrieval)
        vertical_scores = memory.retrieval_scores(vertical_retrieval)
        self.assertGreater(vertical_scores["robot"], horizontal_scores["robot"])
        self.assertLess(vertical_scores["fox"], horizontal_scores["fox"])

    def test_object_patterns_separate_for_confusable_queries(self):
        memory = FixedSparseRecurrentMemory(
            embedding_dim=12,
            substrate_cell_count=96,
            max_slots_per_object=4,
            insertion_similarity_threshold=0.98,
        )
        query = torch.ones(12, dtype=torch.float32)
        query = query / query.norm(p=2)

        horizontal_field = _make_texture_field("horizontal")
        vertical_field = _make_texture_field("vertical")

        for _ in range(4):
            memory.observe("fox", query, observation_field=horizontal_field)
            memory.observe("robot", query, observation_field=vertical_field)

        object_patterns = memory._object_patterns.detach().cpu()
        cosine = float(torch.matmul(object_patterns[0], object_patterns[1]).item())

        self.assertLess(cosine, 0.95)
        memory.reset_episode_state()
        self.assertEqual(memory.retrieve(query, observation_field=horizontal_field).top_object_id, "fox")
        vertical_scores = memory.retrieval_scores(
            memory.retrieve(query, observation_field=vertical_field)
        )
        self.assertGreater(vertical_scores["robot"], 0.6)

    def test_can_load_legacy_slot_bank_state(self):
        legacy_memory = AtlasMemory(
            embedding_dim=3,
            max_slots_per_object=4,
            insertion_similarity_threshold=0.999,
        )
        fox = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
        robot = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
        legacy_memory.observe("fox", fox)
        legacy_memory.observe("fox", torch.tensor([0.9, 0.1, 0.0], dtype=torch.float32))
        legacy_memory.observe("robot", robot)

        migrated_memory = FixedSparseRecurrentMemory(
            embedding_dim=3,
            substrate_cell_count=48,
            max_slots_per_object=4,
        )
        migrated_memory.load_state_dict(legacy_memory.state_dict())

        retrieval = migrated_memory.retrieve(fox)

        self.assertEqual(retrieval.top_object_id, "fox")
        self.assertEqual(
            migrated_memory.get_slot_state().num_slots,
            legacy_memory.get_slot_state().num_slots,
        )

    def test_reset_episode_state_clears_transient_recurrent_traces(self):
        memory = FixedSparseRecurrentMemory(
            embedding_dim=12,
            substrate_cell_count=96,
            max_slots_per_object=4,
            insertion_similarity_threshold=0.98,
        )
        query = torch.ones(12, dtype=torch.float32)
        query = query / query.norm(p=2)

        memory.observe(
            "fox",
            query,
            observation_field=_make_texture_field("horizontal"),
        )

        self.assertGreater(float(memory._prev_active.abs().sum().item()), 0.0)
        self.assertGreater(float(memory._membrane.abs().sum().item()), 0.0)
        self.assertEqual(memory.get_all_known_object_ids(), ["fox"])
        self.assertEqual(memory.get_slot_state().num_slots, 1)

        memory.reset_episode_state()

        self.assertEqual(memory.get_all_known_object_ids(), ["fox"])
        self.assertEqual(memory.get_slot_state().num_slots, 1)
        self.assertEqual(float(memory._prev_active.abs().sum().item()), 0.0)
        self.assertEqual(float(memory._membrane.abs().sum().item()), 0.0)
        self.assertEqual(float(memory._refractory_trace.abs().sum().item()), 0.0)
        self.assertEqual(float(memory._eligibility_trace.abs().sum().item()), 0.0)

    def test_core_uses_sparse_recurrent_memory_by_default(self):
        core = PredictiveHypothesisCore(context_dim=48)

        self.assertEqual(
            core.appearance_memory.state_dict()["memory_layout"],
            "fixed_sparse_recurrent_v1",
        )
        self.assertEqual(
            core.change_memory.state_dict()["memory_layout"],
            "fixed_sparse_recurrent_v1",
        )

    def test_core_propagates_seed_offsets_to_sparse_memories(self):
        core = PredictiveHypothesisCore(context_dim=48, seed=7)

        self.assertEqual(core.appearance_memory.state_dict()["seed"], 7)
        self.assertEqual(core.change_memory.state_dict()["seed"], 8)

    def test_core_roundtrip_rebuilds_sparse_memory_layout(self):
        observation = _make_observation()
        packet = build_visual_detail_packet(
            observation,
            sender_id="camera",
            grid_shape=(5, 5),
            state=_make_state(),
        )
        lm_state = _make_state({"observation_packet_v2": packet})

        core = PredictiveHypothesisCore(context_dim=48)
        core.pre_episode(mode=None, object_name=None)
        core.step(lm_state, learn=True)
        saved_state = core.state_dict()
        self.assertEqual(
            saved_state["appearance_memory"]["latent_ids"],
            saved_state["appearance_memory"]["object_ids"],
        )
        self.assertEqual(
            saved_state["change_memory"]["latent_ids"],
            saved_state["change_memory"]["object_ids"],
        )
        self.assertEqual(
            saved_state["hypotheses"]["latent_ids"],
            saved_state["hypotheses"]["object_ids"],
        )
        for key in ("appearance_memory", "change_memory", "hypotheses"):
            saved_state[key] = dict(saved_state[key])
            saved_state[key].pop("object_ids", None)

        restored = PredictiveHypothesisCore(
            context_dim=48,
            memory_layout="shared_slot_bank_v2",
        )
        restored.load_state_dict(saved_state)

        self.assertEqual(
            restored.appearance_memory.state_dict()["memory_layout"],
            "fixed_sparse_recurrent_v1",
        )
        self.assertEqual(
            restored.change_memory.state_dict()["memory_layout"],
            "fixed_sparse_recurrent_v1",
        )
        self.assertEqual(restored.appearance_memory.state_dict()["seed"], 42)
        self.assertEqual(restored.change_memory.state_dict()["seed"], 43)
        self.assertEqual(
            restored.get_all_known_object_ids(),
            core.get_all_known_object_ids(),
        )
        self.assertEqual(
            restored.get_all_known_latent_ids(),
            core.get_all_known_latent_ids(),
        )

    def test_core_reset_episode_clears_sparse_memory_temporal_state(self):
        core = PredictiveHypothesisCore(context_dim=48)

        core.appearance_memory._prev_active.fill_(1.0)
        core.appearance_memory._membrane.fill_(1.0)
        core.change_memory._prev_active.fill_(1.0)
        core.change_memory._membrane.fill_(1.0)

        core.reset_episode()

        self.assertEqual(
            float(core.appearance_memory._prev_active.abs().sum().item()),
            0.0,
        )
        self.assertEqual(
            float(core.appearance_memory._membrane.abs().sum().item()),
            0.0,
        )
        self.assertEqual(
            float(core.change_memory._prev_active.abs().sum().item()),
            0.0,
        )
        self.assertEqual(
            float(core.change_memory._membrane.abs().sum().item()),
            0.0,
        )


def torch_all_finite(values):
    return bool(np.isfinite(np.asarray(values)).all())


if __name__ == "__main__":
    unittest.main()
