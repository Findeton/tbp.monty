from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import torch.nn.functional as F

from tbp.monty.frameworks.models.predictive_hypothesis_torch.sparse_recurrent_memory import (
    FixedSparseRecurrentMemory,
)
from tbp.monty.frameworks.models.predictive_hypothesis_torch.tensor_objects import (
    HopfieldRetrievalState,
    HypothesisBankState,
    MemorySlots,
    ObservationEmbeddings,
    ObservationField,
    PredictiveContextSignal,
    PredictiveMessageState,
    PredictiveVoteMessage,
    RankedHypothesisVote,
    TemporalState,
)


def _normalize_graph_id(graph_id: Any) -> str | None:
    if graph_id is None:
        return None

    normalized = str(graph_id)
    if normalized in {"", "unknown", "no_observations_yet", "None"}:
        return None
    return normalized


def _string_to_floats(value: str) -> list[float]:
    digest = hashlib.sha256(value.encode("utf-8")).digest()[:8]
    return [((byte / 255.0) * 2.0) - 1.0 for byte in digest]


def _flatten_numeric_values(value: Any, flat: list[float]) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        flat.append(1.0 if value else 0.0)
        return
    if isinstance(value, (int, float, np.integer, np.floating)):
        flat.append(float(value))
        return
    if isinstance(value, str):
        flat.extend(_string_to_floats(value))
        return
    if isinstance(value, np.ndarray):
        flat.extend(np.asarray(value, dtype=np.float32).reshape(-1).tolist())
        return
    if torch.is_tensor(value):
        flat.extend(value.detach().cpu().float().reshape(-1).tolist())
        return
    if isinstance(value, dict):
        for key in sorted(value.keys()):
            _flatten_numeric_values(key, flat)
            _flatten_numeric_values(value[key], flat)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _flatten_numeric_values(item, flat)
        return


def _resize_vector(vector: torch.Tensor, target_dim: int) -> torch.Tensor:
    if target_dim <= 0:
        raise ValueError("target_dim must be > 0")

    if vector.numel() == 0:
        return torch.zeros(target_dim, dtype=torch.float32, device=vector.device)
    if vector.numel() == target_dim:
        return vector.clone()

    resized = F.interpolate(
        vector.reshape(1, 1, -1),
        size=target_dim,
        mode="linear",
        align_corners=False,
    )
    return resized.reshape(-1)


def _cosine_distance(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() == 0 or right.numel() == 0:
        return 0.0

    left_norm = float(left.norm(p=2).item())
    right_norm = float(right.norm(p=2).item())
    if left_norm <= 1e-8 or right_norm <= 1e-8:
        return 0.0

    similarity = F.cosine_similarity(
        left.unsqueeze(0),
        right.unsqueeze(0),
        dim=1,
    )
    return max(0.0, min(1.0, 0.5 * (1.0 - float(similarity.item()))))


def _is_latent_object_id(object_id: Any) -> bool:
    normalized = _normalize_graph_id(object_id)
    return bool(normalized and normalized.startswith("latent_object_"))


def _chart_owner_id(chart_id: Any) -> str | None:
    normalized = _normalize_graph_id(chart_id)
    if normalized is None:
        return None
    return normalized.split("#chart", 1)[0]


def _as_float32_vector(value: Any) -> np.ndarray:
    if value is None:
        return np.zeros(0, dtype=np.float32)
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _normalize_pose_matrix(pose_vectors: Any) -> np.ndarray:
    pose = np.asarray(pose_vectors, dtype=np.float32).reshape(3, 3)
    row_norms = np.linalg.norm(pose, axis=1, keepdims=True)
    row_norms = np.where(row_norms > 1e-8, row_norms, 1.0)
    return pose / row_norms


def _resized_cosine_similarity(left: Any, right: Any) -> float:
    left_array = _as_float32_vector(left)
    right_array = _as_float32_vector(right)
    if left_array.size == 0 or right_array.size == 0:
        return 0.0

    target_dim = max(int(left_array.size), int(right_array.size))
    left_tensor = _resize_vector(
        torch.as_tensor(left_array, dtype=torch.float32),
        target_dim,
    )
    right_tensor = _resize_vector(
        torch.as_tensor(right_array, dtype=torch.float32),
        target_dim,
    )
    left_norm = float(left_tensor.norm(p=2).item())
    right_norm = float(right_tensor.norm(p=2).item())
    if left_norm <= 1e-8 or right_norm <= 1e-8:
        return 0.0

    similarity = F.cosine_similarity(
        left_tensor.unsqueeze(0),
        right_tensor.unsqueeze(0),
        dim=1,
    )
    return float(max(0.0, min(1.0, 0.5 * (float(similarity.item()) + 1.0))))


@dataclass
class PredictiveHypothesis:
    object_id: str
    chart_id: str | None
    weight: float
    score: float
    location: np.ndarray
    pose_vectors: np.ndarray
    appearance_vector: np.ndarray | None = None
    behavior_vector: np.ndarray | None = None
    behavior_label: str | None = None
    inferred_state: int | None = None
    source: str | None = None
    age: int = 0

    @property
    def latent_id(self) -> str:
        return self.object_id


class DetailPacketEncoder:
    def __init__(self, context_dim: int = 256, device: str = "cpu") -> None:
        self.context_dim = max(int(context_dim), 16)
        self.device = torch.device(device)
        self.stream_dim = max(self.context_dim // 3, 16)

    @staticmethod
    def _weighted_mean_and_spread(
        values: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        array = np.asarray(values, dtype=np.float32)
        if array.size == 0:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

        if array.ndim == 1:
            array = array.reshape(-1, 1)
        else:
            array = array.reshape(-1, array.shape[-1])

        if weights is None:
            mean = array.mean(axis=0)
            spread = array.std(axis=0)
            return mean.astype(np.float32), spread.astype(np.float32)

        weight_array = np.asarray(weights, dtype=np.float32).reshape(-1)
        if weight_array.size != array.shape[0]:
            weight_array = None

        if weight_array is None or float(weight_array.sum()) <= 1e-6:
            mean = array.mean(axis=0)
            spread = array.std(axis=0)
            return mean.astype(np.float32), spread.astype(np.float32)

        normalized = weight_array / float(weight_array.sum())
        mean = np.sum(array * normalized[:, None], axis=0)
        centered = array - mean[None, :]
        spread = np.sqrt(np.sum((centered**2) * normalized[:, None], axis=0))
        return mean.astype(np.float32), spread.astype(np.float32)

    @staticmethod
    def _mean_abs_gradient(values: np.ndarray, axis: int) -> float:
        array = np.asarray(values, dtype=np.float32)
        if array.size == 0 or array.shape[axis] <= 1:
            return 0.0
        return float(np.mean(np.abs(np.diff(array, axis=axis))))

    @staticmethod
    def _extend_scaled(flat: list[float], values: Any, scale: float = 1.0) -> None:
        array = np.asarray(values, dtype=np.float32).reshape(-1)
        if array.size == 0:
            return
        flat.extend((array * float(scale)).tolist())

    def _to_tensor(
        self,
        value: Any,
        shape: tuple[int, ...],
    ) -> torch.Tensor:
        array = np.asarray(value, dtype=np.float32)
        target_size = int(np.prod(shape))
        if array.size == 0:
            return torch.zeros(shape, dtype=torch.float32, device=self.device)

        flat = array.reshape(-1)
        if flat.size < target_size:
            flat = np.pad(flat, (0, target_size - flat.size), mode="constant")
        elif flat.size > target_size:
            flat = flat[:target_size]
        return torch.as_tensor(
            flat.reshape(shape),
            dtype=torch.float32,
            device=self.device,
        )

    def _normalize_and_resize(
        self,
        vector: torch.Tensor,
        target_dim: int,
    ) -> torch.Tensor:
        if vector.numel() == 0:
            return torch.zeros(target_dim, dtype=torch.float32, device=self.device)

        resized = _resize_vector(vector.reshape(-1), target_dim)
        norm = float(resized.norm(p=2).item())
        if norm > 1e-8:
            resized = resized / norm
        return resized

    def _scalar_stats(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if values.numel() == 0:
            zero = torch.zeros(1, dtype=torch.float32, device=self.device)
            return zero, zero
        return values.mean().reshape(1), values.std(unbiased=False).reshape(1)

    def _vector_stats(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if values.numel() == 0:
            return (
                torch.zeros(3, dtype=torch.float32, device=self.device),
                torch.zeros(3, dtype=torch.float32, device=self.device),
            )
        return values.mean(dim=0), values.std(dim=0, unbiased=False)

    def _build_temporal_feature_stats(
        self,
        feature_deltas: dict[str, Any],
    ) -> torch.Tensor:
        rows = []
        for key in sorted(feature_deltas.keys()):
            delta = np.asarray(feature_deltas[key], dtype=np.float32).reshape(-1)
            if delta.size == 0:
                continue
            rows.append(
                [
                    float(np.mean(delta)),
                    float(np.std(delta)),
                    float(np.linalg.norm(delta)),
                ]
            )
        if not rows:
            return torch.zeros((0, 3), dtype=torch.float32, device=self.device)
        return torch.as_tensor(rows, dtype=torch.float32, device=self.device)

    def packet_to_observation_field(
        self,
        packet: dict[str, Any],
        state=None,
    ) -> ObservationField:
        packet_type = str(packet.get("packet_type") or "unknown_packet")
        micro_patch_shape = tuple(packet.get("micro_patch_shape") or (4, 4))
        if len(micro_patch_shape) != 2:
            micro_patch_shape = (4, 4)
        grid_shape = tuple(packet.get("grid_shape") or (0, 0))
        if len(grid_shape) != 2:
            grid_shape = (0, 0)
        field = ObservationField.empty(
            device=self.device,
            packet_type=packet_type,
            sender_id=packet.get("sender_id"),
            micro_patch_shape=(int(micro_patch_shape[0]), int(micro_patch_shape[1])),
            grid_shape=(int(grid_shape[0]), int(grid_shape[1])),
        )

        state_anchor = packet.get("state_anchor") or {}
        morph = getattr(state, "morphological_features", None) or {}
        field.anchor_location = self._to_tensor(
            state_anchor.get("location", getattr(state, "location", np.zeros(3))),
            (3,),
        )
        field.anchor_pose = self._to_tensor(
            state_anchor.get("pose_vectors", morph.get("pose_vectors", np.eye(3))),
            (3, 3),
        )
        field.anchor_confidence = self._to_tensor(
            [state_anchor.get("confidence", getattr(state, "confidence", 0.0))],
            (1,),
        )
        field.anchor_on_object = self._to_tensor(
            [state_anchor.get("on_object", morph.get("on_object", 0.0))],
            (1,),
        )

        sensor_context = packet.get("sensor_context") or {}
        camera_pose_world = sensor_context.get("camera_pose_world")
        if camera_pose_world is not None:
            field.camera_forward = self._to_tensor(
                np.asarray(camera_pose_world, dtype=np.float32)[:3, 2],
                (3,),
            )

        temporal_context = packet.get("temporal_context") or {}
        field.flow_direction = self._to_tensor(
            temporal_context.get("flow_direction", np.zeros(3, dtype=np.float32)),
            (3,),
        )
        field.flow_magnitude = self._to_tensor(
            [temporal_context.get("flow_magnitude", 0.0)],
            (1,),
        )
        field.temporal_feature_stats = self._build_temporal_feature_stats(
            temporal_context.get("feature_deltas", {}) or {}
        )

        non_morph = getattr(state, "non_morphological_features", None) or {}
        active_cells = non_morph.get("active_cells")
        if active_cells is not None:
            tensor = torch.as_tensor(
                np.asarray(active_cells, dtype=np.float32).reshape(-1),
                dtype=torch.float32,
                device=self.device,
            )
            field.raw_active_cells = tensor

        cells = packet.get("cells") or []
        if isinstance(cells, list) and cells:
            rgb_list = []
            depth_list = []
            support_list = []
            xyz_list = []
            uv_list = []
            for cell in cells:
                patch_rows, patch_cols = field.micro_patch_shape
                rgb_list.append(
                    self._to_tensor(
                        cell.get("rgb_patch", np.zeros((patch_rows, patch_cols, 3))),
                        (patch_rows, patch_cols, 3),
                    )
                )
                depth_list.append(
                    self._to_tensor(
                        cell.get("depth_patch", np.zeros((patch_rows, patch_cols))),
                        (patch_rows, patch_cols),
                    )
                )
                support_list.append(
                    self._to_tensor(
                        cell.get("support_patch", np.zeros((patch_rows, patch_cols))),
                        (patch_rows, patch_cols),
                    )
                )
                xyz_list.append(
                    self._to_tensor(
                        cell.get("sensor_frame_patch", np.zeros((patch_rows, patch_cols, 3))),
                        (patch_rows, patch_cols, 3),
                    )
                )
                uv_list.append(self._to_tensor(cell.get("uv_center", [0.0, 0.0]), (2,)))

            field.rgb = torch.stack(rgb_list, dim=0)
            field.depth = torch.stack(depth_list, dim=0)
            field.support = torch.stack(support_list, dim=0)
            field.xyz = torch.stack(xyz_list, dim=0)
            field.uv = torch.stack(uv_list, dim=0)
            field.valid_cells = (field.support.mean(dim=(1, 2)) > 0).float()

        shards = packet.get("shards") or []
        if isinstance(shards, list) and shards:
            child_locations = []
            child_confidences = []
            child_graph_hash = []
            for shard in shards:
                child_locations.append(
                    self._to_tensor(
                        shard.get("child_location", shard.get("location", np.zeros(3))),
                        (3,),
                    )
                )
                child_confidences.append(
                    self._to_tensor(
                        [shard.get("child_confidence", shard.get("confidence", 0.0))],
                        (1,),
                    )
                )
                child_graph_hash.append(
                    self._to_tensor(
                        _string_to_floats(
                            str(
                                shard.get(
                                    "child_latent_id",
                                    shard.get(
                                        "child_graph_id",
                                        shard.get("latent_id", shard.get("graph_id", "unknown")),
                                    ),
                                )
                            )
                        ),
                        (8,),
                    )
                )
            field.child_locations = torch.stack(child_locations, dim=0)
            field.child_confidences = torch.cat(child_confidences, dim=0)
            field.child_graph_hash = torch.stack(child_graph_hash, dim=0)

        return field

    def encode_observation_field(self, field: ObservationField) -> ObservationEmbeddings:
        embeddings = ObservationEmbeddings.empty(
            stream_dim=self.stream_dim,
            context_dim=self.context_dim,
            device=self.device,
        )

        support_weights = field.support.mean(dim=(1, 2)) if field.cell_count else torch.zeros(0, device=self.device)
        if support_weights.numel() > 0 and float(support_weights.sum().item()) > 1e-8:
            support_weights = support_weights / support_weights.sum()
        elif field.valid_cells.numel() > 0 and float(field.valid_cells.sum().item()) > 1e-8:
            support_weights = field.valid_cells / field.valid_cells.sum()
        elif field.cell_count > 0:
            support_weights = torch.full(
                (field.cell_count,),
                1.0 / float(field.cell_count),
                dtype=torch.float32,
                device=self.device,
            )

        if field.cell_count > 0:
            rgb_per_cell = field.rgb.mean(dim=(1, 2))
            xyz_per_cell = field.xyz.mean(dim=(1, 2))
            depth_per_cell = field.depth.mean(dim=(1, 2))
            uv_per_cell = field.uv
            if support_weights.numel() > 0:
                rgb_mean = (rgb_per_cell * support_weights[:, None]).sum(dim=0)
                xyz_mean = (xyz_per_cell * support_weights[:, None]).sum(dim=0)
                depth_mean = (depth_per_cell * support_weights).sum().reshape(1)
                uv_mean = (uv_per_cell * support_weights[:, None]).sum(dim=0)
            else:
                rgb_mean = rgb_per_cell.mean(dim=0)
                xyz_mean = xyz_per_cell.mean(dim=0)
                depth_mean = depth_per_cell.mean().reshape(1)
                uv_mean = uv_per_cell.mean(dim=0)

            rgb_spread = rgb_per_cell.std(dim=0, unbiased=False)
            xyz_spread = xyz_per_cell.std(dim=0, unbiased=False)
            depth_std = depth_per_cell.std(unbiased=False).reshape(1)
            uv_spread = uv_per_cell.std(dim=0, unbiased=False)
            texture_x = field.rgb.diff(dim=1).abs().mean().reshape(1) if field.rgb.shape[1] > 1 else torch.zeros(1, dtype=torch.float32, device=self.device)
            texture_y = field.rgb.diff(dim=2).abs().mean().reshape(1) if field.rgb.shape[2] > 1 else torch.zeros(1, dtype=torch.float32, device=self.device)
            support_mean, support_std = self._scalar_stats(field.support.reshape(-1))
        else:
            rgb_mean = torch.zeros(3, dtype=torch.float32, device=self.device)
            rgb_spread = torch.zeros(3, dtype=torch.float32, device=self.device)
            xyz_mean = torch.zeros(3, dtype=torch.float32, device=self.device)
            xyz_spread = torch.zeros(3, dtype=torch.float32, device=self.device)
            depth_mean = torch.zeros(1, dtype=torch.float32, device=self.device)
            depth_std = torch.zeros(1, dtype=torch.float32, device=self.device)
            uv_mean = torch.zeros(2, dtype=torch.float32, device=self.device)
            uv_spread = torch.zeros(2, dtype=torch.float32, device=self.device)
            texture_x = torch.zeros(1, dtype=torch.float32, device=self.device)
            texture_y = torch.zeros(1, dtype=torch.float32, device=self.device)
            support_mean = torch.zeros(1, dtype=torch.float32, device=self.device)
            support_std = torch.zeros(1, dtype=torch.float32, device=self.device)

        child_loc_mean, child_loc_std = self._vector_stats(field.child_locations)
        child_conf_mean, child_conf_std = self._scalar_stats(field.child_confidences)
        child_hash_mean = (
            field.child_graph_hash.mean(dim=0)
            if field.child_graph_hash.numel() > 0
            else torch.zeros(8, dtype=torch.float32, device=self.device)
        )

        if field.temporal_feature_stats.numel() > 0:
            temporal_mean = field.temporal_feature_stats.mean(dim=0)
            temporal_max = torch.amax(field.temporal_feature_stats, dim=0)
        else:
            temporal_mean = torch.zeros(3, dtype=torch.float32, device=self.device)
            temporal_max = torch.zeros(3, dtype=torch.float32, device=self.device)

        raw_context_norm = (
            field.raw_active_cells.norm(p=2).reshape(1)
            if field.raw_active_cells.numel() > 0
            else torch.zeros(1, dtype=torch.float32, device=self.device)
        )
        raw_context_mean, raw_context_std = self._scalar_stats(field.raw_active_cells)

        appearance_seed = torch.cat(
            [
                rgb_mean,
                rgb_spread,
                texture_x,
                texture_y,
                support_mean,
                support_std,
                field.anchor_confidence,
                field.anchor_on_object,
            ]
        )
        geometry_seed = torch.cat(
            [
                xyz_mean,
                xyz_spread,
                depth_mean,
                depth_std,
                uv_mean,
                uv_spread,
                field.anchor_location,
                field.camera_forward,
                field.anchor_pose.reshape(-1)[:6],
            ]
        )
        temporal_seed = torch.cat(
            [
                field.flow_direction,
                field.flow_magnitude,
                temporal_mean,
                temporal_max,
                child_conf_mean,
                child_conf_std,
                child_loc_mean,
                child_loc_std,
                raw_context_norm,
                raw_context_mean,
                raw_context_std,
                child_hash_mean,
            ]
        )

        embeddings.appearance = self._normalize_and_resize(appearance_seed, self.stream_dim)
        embeddings.geometry = self._normalize_and_resize(geometry_seed, self.stream_dim)
        embeddings.temporal = self._normalize_and_resize(temporal_seed, self.stream_dim)

        joint_seed = torch.cat(
            [
                embeddings.appearance,
                embeddings.geometry,
                embeddings.temporal,
            ]
        )
        if field.raw_active_cells.numel() > 0:
            joint_seed = torch.cat(
                [
                    joint_seed,
                    self._normalize_and_resize(field.raw_active_cells, self.stream_dim),
                ]
            )
        embeddings.joint = self._normalize_and_resize(joint_seed, self.context_dim)
        return embeddings

    def build_packet_from_state(self, state) -> dict[str, Any]:
        non_morph = getattr(state, "non_morphological_features", None) or {}
        for packet_key in ("observation_packet_v2", "detail_packet"):
            detail_packet = non_morph.get(packet_key)
            if not isinstance(detail_packet, dict):
                continue

            cells = detail_packet.get("cells")
            if isinstance(cells, list) and len(cells) > 0:
                packet = dict(detail_packet)
                packet.setdefault("cell_count", len(cells))
                packet.setdefault("shard_count", len(cells))
                return packet

            shards = detail_packet.get("shards")
            if isinstance(shards, list) and len(shards) > 0:
                packet = dict(detail_packet)
                packet.setdefault("shard_count", len(shards))
                return packet

        pose_vectors = np.asarray(
            getattr(state, "morphological_features", {}).get(
                "pose_vectors",
                np.eye(3, dtype=np.float32),
            ),
            dtype=np.float32,
        )
        fallback_packet = {
            "packet_type": "fallback_state_packet",
            "sender_type": getattr(state, "sender_type", None),
            "sender_id": getattr(state, "sender_id", None),
            "shard_count": 1,
            "shards": [
                {
                    "location": np.asarray(
                        getattr(state, "location", np.zeros(3, dtype=np.float32)),
                        dtype=np.float32,
                    ),
                    "pose_vectors": pose_vectors,
                    "confidence": float(getattr(state, "confidence", 0.0)),
                    "graph_id": _normalize_graph_id(
                        non_morph.get("latent_id", non_morph.get("graph_id"))
                    ),
                    "latent_id": _normalize_graph_id(
                        non_morph.get("latent_id", non_morph.get("graph_id"))
                    ),
                    "features": {
                        key: value
                        for key, value in sorted(non_morph.items())
                        if key not in {"detail_packet", "active_cells"}
                    },
                }
            ],
        }
        return fallback_packet

    def encode_state_to_tensors(
        self,
        state,
    ) -> tuple[dict[str, Any], ObservationField, ObservationEmbeddings]:
        packet = self.build_packet_from_state(state)
        observation_field = self.packet_to_observation_field(packet, state=state)
        embeddings = self.encode_observation_field(observation_field)
        return packet, observation_field, embeddings

    def encode_state(self, state) -> tuple[dict[str, Any], torch.Tensor]:
        packet, _, embeddings = self.encode_state_to_tensors(state)
        return packet, embeddings.joint


class AtlasMemory:
    def __init__(
        self,
        learning_rate: float = 0.15,
        device: str = "cpu",
        max_slots_per_object: int = 8,
        insertion_similarity_threshold: float = 0.92,
        topk_score_pool: int = 2,
        embedding_dim: int = 256,
        hopfield_beta: float = 8.0,
        hopfield_max_settle_iters: int = 4,
        hopfield_convergence_threshold: float = 1e-4,
        novelty_threshold_bonus: float = 0.05,
        consolidation_count_exponent: float = 0.5,
        min_learning_rate_scale: float = 0.25,
    ) -> None:
        self.learning_rate = float(max(min(learning_rate, 1.0), 1e-4))
        self.device = torch.device(device)
        self.max_slots_per_object = max(int(max_slots_per_object), 1)
        self.insertion_similarity_threshold = float(
            max(min(insertion_similarity_threshold, 0.999), -0.999)
        )
        self.topk_score_pool = max(int(topk_score_pool), 1)
        self._embedding_dim = max(int(embedding_dim), 1)
        self.hopfield_beta = float(max(hopfield_beta, 1e-3))
        self.hopfield_max_settle_iters = max(int(hopfield_max_settle_iters), 1)
        self.hopfield_convergence_threshold = float(
            max(hopfield_convergence_threshold, 1e-8)
        )
        self.novelty_threshold_bonus = float(max(novelty_threshold_bonus, 0.0))
        self.consolidation_count_exponent = float(
            max(consolidation_count_exponent, 0.0)
        )
        self.min_learning_rate_scale = float(
            max(min(min_learning_rate_scale, 1.0), 0.0)
        )
        self._slots = MemorySlots.empty(
            embedding_dim=self._embedding_dim,
            device=self.device,
        )
        self._last_retrieval = HopfieldRetrievalState.empty(
            embedding_dim=self._embedding_dim,
            device=self.device,
        )

    def _normalize_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        vector = embedding.detach().to(self.device).float().reshape(-1)
        if vector.numel() != self._embedding_dim:
            vector = _resize_vector(vector, self._embedding_dim)
        norm = float(vector.norm(p=2).item())
        if norm > 1e-8:
            vector = vector / norm
        return vector

    def _object_index(self, object_id: str) -> int:
        if object_id in self._slots.object_ids:
            return self._slots.object_ids.index(object_id)
        self._slots.object_ids.append(object_id)
        return len(self._slots.object_ids) - 1

    def _shared_slot_capacity(self) -> int:
        return self.max_slots_per_object * max(len(self._slots.object_ids), 1)

    def _object_slot_indices(self, object_index: int) -> torch.Tensor:
        if self._slots.slot_object_indices is None or self._slots.num_slots == 0:
            return torch.zeros(0, dtype=torch.int64, device=self.device)
        return torch.nonzero(
            self._slots.slot_object_indices == int(object_index),
            as_tuple=False,
        ).reshape(-1)

    def _next_chart_id(self, object_index: int) -> str:
        object_id = self._slots.object_ids[object_index]
        next_index = int(self._object_slot_indices(object_index).numel())
        return f"{object_id}#chart{next_index}"

    def _ensure_slot_chart_ids(self) -> None:
        if len(self._slots.slot_chart_ids) == self._slots.num_slots:
            return

        if self._slots.num_slots == 0 or self._slots.slot_object_indices is None:
            self._slots.slot_chart_ids = []
            return

        counts: dict[int, int] = {}
        slot_chart_ids: list[str] = []
        for slot_index in range(self._slots.num_slots):
            object_index = int(self._slots.slot_object_indices[slot_index].item())
            object_id = (
                self._slots.object_ids[object_index]
                if 0 <= object_index < len(self._slots.object_ids)
                else f"object_{object_index}"
            )
            local_index = counts.get(object_index, 0)
            slot_chart_ids.append(f"{object_id}#chart{local_index}")
            counts[object_index] = local_index + 1
        self._slots.slot_chart_ids = slot_chart_ids

    def _append_slot(self, object_index: int, embedding: torch.Tensor) -> None:
        self._ensure_slot_chart_ids()
        self._slots.slot_embeddings = torch.cat(
            [self._slots.slot_embeddings, embedding.reshape(1, -1)],
            dim=0,
        )
        self._slots.slot_object_indices = torch.cat(
            [
                self._slots.slot_object_indices,
                torch.as_tensor([object_index], dtype=torch.int64, device=self.device),
            ],
            dim=0,
        )
        self._slots.slot_counts = torch.cat(
            [
                self._slots.slot_counts,
                torch.as_tensor([1.0], dtype=torch.float32, device=self.device),
            ],
            dim=0,
        )
        self._slots.slot_chart_ids.append(self._next_chart_id(object_index))

    def _slot_similarities(
        self,
        embedding: torch.Tensor,
        slot_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        if slot_embeddings.numel() == 0:
            return torch.zeros(0, dtype=torch.float32, device=self.device)

        return F.cosine_similarity(
            embedding.unsqueeze(0),
            slot_embeddings,
            dim=1,
        )

    def _best_slot(
        self,
        embedding: torch.Tensor,
        slot_indices: torch.Tensor | None = None,
    ) -> tuple[int | None, float]:
        if self._slots.num_slots == 0:
            return None, -1.0

        if slot_indices is None:
            slot_indices = torch.arange(self._slots.num_slots, device=self.device)
        if int(slot_indices.numel()) == 0:
            return None, -1.0

        slot_embeddings = self._slots.slot_embeddings.index_select(0, slot_indices)
        similarities = self._slot_similarities(embedding, slot_embeddings)
        best_local_idx = int(torch.argmax(similarities).item())
        return int(slot_indices[best_local_idx].item()), float(
            similarities[best_local_idx].item()
        )

    def _effective_insertion_threshold(self, novelty_signal: float) -> float:
        novelty = float(np.clip(novelty_signal, 0.0, 1.0))
        return float(
            max(
                -0.999,
                min(
                    0.999,
                    self.insertion_similarity_threshold
                    + (novelty * self.novelty_threshold_bonus),
                ),
            )
        )

    def _effective_learning_rate(self, slot_count: float) -> float:
        if self.consolidation_count_exponent <= 0.0:
            return self.learning_rate

        stabilized_count = max(float(slot_count), 1.0)
        adapted = self.learning_rate / (
            stabilized_count**self.consolidation_count_exponent
        )
        minimum = self.learning_rate * self.min_learning_rate_scale
        return float(max(minimum, min(self.learning_rate, adapted)))

    def observe(
        self,
        object_id: str | None,
        embedding: torch.Tensor,
        *,
        novelty_signal: float = 0.0,
        observation_field: ObservationField | None = None,
    ) -> None:
        normalized_id = _normalize_graph_id(object_id)
        if normalized_id is None:
            return

        embedding = self._normalize_embedding(embedding)
        object_index = self._object_index(normalized_id)
        same_object_slots = self._object_slot_indices(object_index)
        effective_threshold = self._effective_insertion_threshold(novelty_signal)
        if int(same_object_slots.numel()) == 0:
            self._append_slot(object_index, embedding.clone())
            return

        best_slot_idx, best_similarity = self._best_slot(embedding, same_object_slots)
        if (
            best_slot_idx is None
            or (
                best_similarity < effective_threshold
                and self._slots.num_slots < self._shared_slot_capacity()
            )
        ):
            self._append_slot(object_index, embedding.clone())
            return

        if best_slot_idx is None:
            return

        prototype = self._slots.slot_embeddings[best_slot_idx]
        slot_count = float(self._slots.slot_counts[best_slot_idx].item())
        effective_learning_rate = self._effective_learning_rate(slot_count)
        updated = (1.0 - effective_learning_rate) * prototype + (
            effective_learning_rate * embedding
        )
        norm = float(updated.norm(p=2).item())
        if norm > 1e-8:
            updated = updated / norm
        self._slots.slot_embeddings[best_slot_idx] = updated
        self._slots.slot_counts[best_slot_idx] = self._slots.slot_counts[best_slot_idx] + 1.0

    def _hopfield_step(self, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw_similarities = self._slot_similarities(query, self._slots.slot_embeddings)
        if raw_similarities.numel() == 0:
            return query.clone(), raw_similarities

        similarities = self.hopfield_beta * raw_similarities
        similarities = similarities - similarities.max()
        attention = torch.softmax(similarities, dim=0)
        retrieved = torch.mv(self._slots.slot_embeddings.t(), attention)
        norm = float(retrieved.norm(p=2).item())
        if norm > 1e-8:
            retrieved = retrieved / norm
        return retrieved, attention

    def _hopfield_energy(self, query: torch.Tensor) -> torch.Tensor:
        raw_similarities = self._slot_similarities(query, self._slots.slot_embeddings)
        if raw_similarities.numel() == 0:
            return torch.as_tensor(0.0, dtype=torch.float32, device=self.device)

        similarities = self.hopfield_beta * raw_similarities
        lse = torch.logsumexp(similarities, dim=0)
        regularizer = 0.5 * self.hopfield_beta * query.dot(query)
        return -lse + regularizer

    def _aggregate_object_scores(
        self,
        query: torch.Tensor,
        settled: torch.Tensor,
        attention: torch.Tensor,
    ) -> tuple[dict[str, float], dict[str, str | None]]:
        self._ensure_slot_chart_ids()
        object_indices = self._slots.slot_object_indices
        if object_indices is None or attention.numel() == 0:
            return {}, {}

        query_support = 0.5 * (
            self._slot_similarities(query, self._slots.slot_embeddings) + 1.0
        )
        settled_support = 0.5 * (
            self._slot_similarities(settled, self._slots.slot_embeddings) + 1.0
        )
        query_attention = torch.softmax(self.hopfield_beta * query_support, dim=0)
        scores: dict[str, float] = {}
        best_chart_ids: dict[str, str | None] = {}

        for object_index, object_id in enumerate(self._slots.object_ids):
            mask = object_indices == int(object_index)
            if not bool(mask.any().item()):
                continue

            slot_indices = torch.nonzero(mask, as_tuple=False).reshape(-1)
            object_query_support = query_support[mask]
            object_settled_support = settled_support[mask]
            object_query_attention = query_attention[mask]
            object_settled_attention = attention[mask]

            topk = min(self.topk_score_pool, int(object_query_support.numel()))
            top_values = torch.topk(object_query_support, k=topk).values
            best_score = float(top_values[0].item())
            pooled_score = float(top_values.mean().item())
            attention_mass = float(object_query_attention.sum().item())
            if attention_mass > 1e-8:
                settled_weights = object_settled_attention / object_settled_attention.sum().clamp_min(1e-8)
                settled_bonus = float(
                    (settled_weights * object_settled_support).sum().item()
                )
            else:
                settled_bonus = float(object_settled_support.mean().item())

            scores[object_id] = (0.80 * best_score) + (0.15 * pooled_score) + (
                0.05 * settled_bonus
            )

            best_basis = object_query_support + (0.15 * object_settled_support)
            best_local_index = int(torch.argmax(best_basis).item())
            best_slot_index = int(slot_indices[best_local_index].item())
            best_chart_ids[object_id] = self._slots.slot_chart_ids[best_slot_index]

        return scores, best_chart_ids

    def retrieve(
        self,
        embedding: torch.Tensor,
        *,
        store_as_last: bool = True,
        observation_field: ObservationField | None = None,
    ) -> HopfieldRetrievalState:
        if not self._slots.object_ids or self._slots.num_slots == 0:
            state = HopfieldRetrievalState.empty(
                embedding_dim=self._embedding_dim,
                device=self.device,
            )
            if store_as_last:
                self._last_retrieval = state
            return state

        self._ensure_slot_chart_ids()
        query = self._normalize_embedding(embedding)
        settled = query.clone()
        energies: list[float] = []
        iteration_count = 0

        for step in range(self.hopfield_max_settle_iters):
            updated, _ = self._hopfield_step(settled)
            energies.append(float(self._hopfield_energy(updated).item()))
            iteration_count = step + 1
            delta = float((updated - settled).norm(p=2).item())
            settled = updated
            if delta < self.hopfield_convergence_threshold:
                break

        raw_similarities = self._slot_similarities(settled, self._slots.slot_embeddings)
        attention = torch.softmax(self.hopfield_beta * raw_similarities, dim=0)
        scores, best_chart_ids = self._aggregate_object_scores(query, settled, attention)
        object_scores = torch.as_tensor(
            [float(scores.get(object_id, 0.0)) for object_id in self._slots.object_ids],
            dtype=torch.float32,
            device=self.device,
        )
        topk_slots = min(self.topk_score_pool, int(attention.numel()))
        top_slot_indices = (
            torch.topk(attention, k=topk_slots).indices
            if topk_slots > 0
            else torch.zeros(0, dtype=torch.int64, device=self.device)
        )
        state = HopfieldRetrievalState(
            object_ids=list(self._slots.object_ids),
            slot_chart_ids=list(self._slots.slot_chart_ids),
            best_chart_ids=[best_chart_ids.get(object_id) for object_id in self._slots.object_ids],
            query=query.detach().clone(),
            retrieved=settled.detach().clone(),
            settled=settled.detach().clone(),
            slot_similarities=raw_similarities.detach().clone(),
            slot_attention=attention.detach().clone(),
            slot_object_indices=self._slots.slot_object_indices.detach().clone(),
            top_slot_indices=top_slot_indices.detach().clone(),
            object_scores=object_scores.detach().clone(),
            energies=torch.as_tensor(energies, dtype=torch.float32, device=self.device),
            iterations=torch.as_tensor([iteration_count], dtype=torch.int64, device=self.device),
        )
        if store_as_last:
            self._last_retrieval = state
        return state

    def retrieval_scores(self, retrieval_state: HopfieldRetrievalState) -> dict[str, float]:
        if retrieval_state.object_scores is None:
            return {}
        return {
            object_id: float(score.item())
            for object_id, score in zip(
                retrieval_state.object_ids,
                retrieval_state.object_scores,
            )
        }

    def retrieval_latent_scores(
        self,
        retrieval_state: HopfieldRetrievalState,
    ) -> dict[str, float]:
        return self.retrieval_scores(retrieval_state)

    def retrieval_chart_ids(
        self,
        retrieval_state: HopfieldRetrievalState,
    ) -> dict[str, str | None]:
        return {
            object_id: chart_id
            for object_id, chart_id in zip(
                retrieval_state.object_ids,
                retrieval_state.best_chart_ids,
            )
        }

    def score(self, embedding: torch.Tensor) -> dict[str, float]:
        retrieval_state = self.retrieve(embedding, store_as_last=True)
        return self.retrieval_scores(retrieval_state)

    def latent_score(self, embedding: torch.Tensor) -> dict[str, float]:
        return self.score(embedding)

    def get_all_known_object_ids(self) -> list[str]:
        return list(self._slots.object_ids)

    def get_all_known_latent_ids(self) -> list[str]:
        return self.get_all_known_object_ids()

    def get_slot_state(self) -> MemorySlots:
        self._ensure_slot_chart_ids()
        return self._slots

    def get_last_retrieval_state(self) -> HopfieldRetrievalState:
        return self._last_retrieval

    def reset_episode_state(self) -> None:
        self._last_retrieval = HopfieldRetrievalState.empty(
            embedding_dim=self._embedding_dim,
            device=self.device,
        )

    def state_dict(self) -> dict[str, Any]:
        self._ensure_slot_chart_ids()
        return {
            "learning_rate": self.learning_rate,
            "max_slots_per_object": self.max_slots_per_object,
            "insertion_similarity_threshold": self.insertion_similarity_threshold,
            "topk_score_pool": self.topk_score_pool,
            "embedding_dim": self._embedding_dim,
            "hopfield_beta": self.hopfield_beta,
            "hopfield_max_settle_iters": self.hopfield_max_settle_iters,
            "hopfield_convergence_threshold": self.hopfield_convergence_threshold,
            "novelty_threshold_bonus": self.novelty_threshold_bonus,
            "consolidation_count_exponent": self.consolidation_count_exponent,
            "min_learning_rate_scale": self.min_learning_rate_scale,
            "memory_layout": "shared_slot_bank_v2",
            "object_ids": list(self._slots.object_ids),
            "latent_ids": list(self._slots.object_ids),
            "slot_embeddings": self._slots.slot_embeddings.detach().cpu().numpy(),
            "slot_object_indices": self._slots.slot_object_indices.detach().cpu().numpy(),
            "slot_counts": self._slots.slot_counts.detach().cpu().numpy(),
            "slot_chart_ids": list(self._slots.slot_chart_ids),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.learning_rate = float(state_dict.get("learning_rate", self.learning_rate))
        self.max_slots_per_object = int(
            state_dict.get("max_slots_per_object", self.max_slots_per_object)
        )
        self.insertion_similarity_threshold = float(
            state_dict.get(
                "insertion_similarity_threshold",
                self.insertion_similarity_threshold,
            )
        )
        self.topk_score_pool = int(
            state_dict.get("topk_score_pool", self.topk_score_pool)
        )
        self._embedding_dim = int(state_dict.get("embedding_dim", self._embedding_dim))
        self.hopfield_beta = float(state_dict.get("hopfield_beta", self.hopfield_beta))
        self.hopfield_max_settle_iters = int(
            state_dict.get(
                "hopfield_max_settle_iters",
                self.hopfield_max_settle_iters,
            )
        )
        self.hopfield_convergence_threshold = float(
            state_dict.get(
                "hopfield_convergence_threshold",
                self.hopfield_convergence_threshold,
            )
        )
        self.novelty_threshold_bonus = float(
            state_dict.get("novelty_threshold_bonus", self.novelty_threshold_bonus)
        )
        self.consolidation_count_exponent = float(
            state_dict.get(
                "consolidation_count_exponent",
                self.consolidation_count_exponent,
            )
        )
        self.min_learning_rate_scale = float(
            state_dict.get(
                "min_learning_rate_scale",
                self.min_learning_rate_scale,
            )
        )
        self._slots = MemorySlots.empty(
            embedding_dim=self._embedding_dim,
            device=self.device,
        )
        self._slots.object_ids = [
            str(object_id)
            for object_id in state_dict.get(
                "latent_ids",
                state_dict.get("object_ids", []),
            )
        ]
        slot_embeddings = state_dict.get("slot_embeddings")
        slot_object_indices = state_dict.get("slot_object_indices")
        slot_counts = state_dict.get("slot_counts")
        if slot_embeddings is not None:
            slot_embeddings_tensor = torch.as_tensor(
                slot_embeddings,
                dtype=torch.float32,
                device=self.device,
            )
            if slot_embeddings_tensor.numel() == 0:
                slot_embeddings_tensor = slot_embeddings_tensor.reshape(0, self._embedding_dim)
            elif slot_embeddings_tensor.ndim == 1:
                slot_embeddings_tensor = slot_embeddings_tensor.reshape(1, -1)
            self._slots.slot_embeddings = slot_embeddings_tensor
        if slot_object_indices is not None:
            self._slots.slot_object_indices = torch.as_tensor(
                slot_object_indices,
                dtype=torch.int64,
                device=self.device,
            )
        if slot_counts is not None:
            slot_counts_tensor = torch.as_tensor(
                slot_counts,
                dtype=torch.float32,
                device=self.device,
            )
            if slot_counts_tensor.ndim == 0:
                slot_counts_tensor = slot_counts_tensor.reshape(1)
            self._slots.slot_counts = slot_counts_tensor.reshape(-1)
        self._slots.slot_chart_ids = [
            str(chart_id)
            for chart_id in (state_dict.get("slot_chart_ids") or [])
            if chart_id is not None
        ]
        self._ensure_slot_chart_ids()
        self._last_retrieval = HopfieldRetrievalState.empty(
            embedding_dim=self._embedding_dim,
            device=self.device,
        )


def _make_predictive_memory(
    *,
    memory_layout: str,
    learning_rate: float,
    device: str,
    max_slots_per_object: int,
    insertion_similarity_threshold: float,
    embedding_dim: int,
    substrate_cell_count: int | None,
    active_cell_sparsity: float,
    seed: int,
):
    normalized_layout = str(memory_layout or "fixed_sparse_recurrent_v1")
    if normalized_layout == "shared_slot_bank_v2":
        return AtlasMemory(
            learning_rate=learning_rate,
            device=device,
            max_slots_per_object=max_slots_per_object,
            insertion_similarity_threshold=insertion_similarity_threshold,
            embedding_dim=embedding_dim,
        )
    if normalized_layout in {"fixed_sparse_recurrent", "fixed_sparse_recurrent_v1"}:
        return FixedSparseRecurrentMemory(
            learning_rate=learning_rate,
            device=device,
            max_slots_per_object=max_slots_per_object,
            insertion_similarity_threshold=insertion_similarity_threshold,
            embedding_dim=embedding_dim,
            substrate_cell_count=substrate_cell_count,
            active_cell_sparsity=active_cell_sparsity,
            seed=seed,
        )
    raise ValueError(f"Unsupported predictive memory layout: {normalized_layout}")


class HypothesisBank:
    def __init__(
        self,
        max_hypotheses: int = 5,
        temperature: float = 0.35,
        device: str = "cpu",
    ) -> None:
        self.max_hypotheses = max(int(max_hypotheses), 1)
        self.temperature = float(max(temperature, 1e-3))
        self.device = torch.device(device)
        self.state = HypothesisBankState.empty(device=self.device)

    @property
    def entries(self) -> list[PredictiveHypothesis]:
        entries: list[PredictiveHypothesis] = []
        for index, object_id in enumerate(self.state.object_ids):
            appearance_vector = None
            if (
                self.state.appearance_vectors is not None
                and self.state.appearance_vectors.ndim == 2
                and index < self.state.appearance_vectors.shape[0]
            ):
                appearance_vector = (
                    self.state.appearance_vectors[index]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
            behavior_vector = None
            if (
                self.state.behavior_vectors is not None
                and self.state.behavior_vectors.ndim == 2
                and index < self.state.behavior_vectors.shape[0]
            ):
                behavior_vector = (
                    self.state.behavior_vectors[index]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
            entries.append(
                PredictiveHypothesis(
                    object_id=str(object_id),
                    chart_id=(
                        self.state.chart_ids[index]
                        if index < len(self.state.chart_ids)
                        else None
                    ),
                    weight=float(self.state.weights[index].item()),
                    score=float(self.state.scores[index].item()),
                    location=self.state.locations[index].detach().cpu().numpy().astype(np.float64),
                    pose_vectors=self.state.pose_vectors[index].detach().cpu().numpy().astype(np.float64),
                    appearance_vector=appearance_vector,
                    behavior_vector=behavior_vector,
                    behavior_label=(
                        self.state.behavior_labels[index]
                        if index < len(self.state.behavior_labels)
                        else None
                    ),
                    inferred_state=self.state.inferred_states[index],
                    source=(
                        self.state.sources[index]
                        if index < len(self.state.sources)
                        else None
                    ),
                    age=(
                        int(self.state.ages[index].item())
                        if self.state.ages is not None
                        and self.state.ages.numel() > index
                        else 0
                    ),
                )
            )
        return entries

    def reset(self) -> None:
        self.state = HypothesisBankState.empty(device=self.device)

    def update(
        self,
        scores: dict[str, float],
        location: np.ndarray,
        pose_vectors: np.ndarray,
        appearance_vector: np.ndarray | None = None,
        behavior_vector: np.ndarray | None = None,
        behavior_label: str | None = None,
        inferred_state: int | None = None,
        chart_ids: dict[str, str | None] | None = None,
        joint_states: dict[str, dict[str, Any]] | None = None,
        joint_candidates: list[dict[str, Any]] | None = None,
    ) -> None:
        if not scores and not joint_candidates:
            self.reset()
            return

        default_location = np.asarray(location, dtype=np.float32).reshape(3)
        default_pose = np.asarray(pose_vectors, dtype=np.float32).reshape(3, 3)
        default_appearance = _as_float32_vector(appearance_vector)
        default_behavior = _as_float32_vector(behavior_vector)

        if joint_candidates:
            ranked_candidates = sorted(
                (
                    {
                        "object_id": str(
                            candidate.get(
                                "object_id",
                                candidate.get("latent_id", "unknown"),
                            )
                        ),
                        "chart_id": (
                            None
                            if candidate.get("chart_id") is None
                            else str(candidate.get("chart_id"))
                        ),
                        "score": float(
                            candidate.get(
                                "score",
                                scores.get(
                                    str(
                                        candidate.get(
                                            "object_id",
                                            candidate.get("latent_id", "unknown"),
                                        )
                                    ),
                                    0.0,
                                ),
                            )
                        ),
                        "location": np.asarray(
                            candidate.get("location", default_location),
                            dtype=np.float32,
                        ).reshape(3),
                        "pose_vectors": np.asarray(
                            candidate.get("pose_vectors", default_pose),
                            dtype=np.float32,
                        ).reshape(3, 3),
                        "appearance_vector": _as_float32_vector(
                            candidate.get("appearance_vector", default_appearance)
                        ),
                        "behavior_vector": _as_float32_vector(
                            candidate.get("behavior_vector", default_behavior)
                        ),
                        "behavior_label": candidate.get(
                            "behavior_label",
                            behavior_label,
                        ),
                        "inferred_state": candidate.get(
                            "inferred_state",
                            inferred_state,
                        ),
                        "source": (
                            None
                            if candidate.get("source") is None
                            else str(candidate.get("source"))
                        ),
                        "age": max(int(candidate.get("age", 0)), 0),
                    }
                    for candidate in joint_candidates
                ),
                key=lambda candidate: (
                    -float(candidate["score"]),
                    str(candidate["object_id"]),
                    "" if candidate["chart_id"] is None else str(candidate["chart_id"]),
                    ""
                    if candidate["behavior_label"] is None
                    else str(candidate["behavior_label"]),
                ),
            )[: self.max_hypotheses]
            if not ranked_candidates:
                self.reset()
                return

            score_tensor = torch.as_tensor(
                [candidate["score"] for candidate in ranked_candidates],
                dtype=torch.float32,
                device=self.device,
            )
            weights = torch.softmax(score_tensor / self.temperature, dim=0)

            location_rows = [candidate["location"] for candidate in ranked_candidates]
            pose_rows = [candidate["pose_vectors"] for candidate in ranked_candidates]
            appearance_rows = [candidate["appearance_vector"] for candidate in ranked_candidates]
            behavior_rows = [candidate["behavior_vector"] for candidate in ranked_candidates]
            behavior_labels = [candidate["behavior_label"] for candidate in ranked_candidates]
            inferred_states = [candidate["inferred_state"] for candidate in ranked_candidates]
            resolved_chart_ids = [candidate["chart_id"] for candidate in ranked_candidates]
            sources = [
                "memory"
                if candidate["source"] is None
                else str(candidate["source"])
                for candidate in ranked_candidates
            ]
            age_tensor = torch.as_tensor(
                [max(int(candidate["age"]), 0) for candidate in ranked_candidates],
                dtype=torch.int64,
                device=self.device,
            )
            appearance_dim = max((int(row.size) for row in appearance_rows), default=0)
            behavior_dim = max((int(row.size) for row in behavior_rows), default=0)

            location_tensor = torch.as_tensor(
                np.stack(location_rows, axis=0),
                dtype=torch.float32,
                device=self.device,
            )
            pose_tensor = torch.as_tensor(
                np.stack(pose_rows, axis=0),
                dtype=torch.float32,
                device=self.device,
            )
            if appearance_dim <= 0:
                appearance_tensor = torch.zeros(
                    (len(ranked_candidates), 0),
                    dtype=torch.float32,
                    device=self.device,
                )
            else:
                padded_appearance = [
                    np.pad(row, (0, appearance_dim - int(row.size)), mode="constant")
                    for row in appearance_rows
                ]
                appearance_tensor = torch.as_tensor(
                    np.stack(padded_appearance, axis=0),
                    dtype=torch.float32,
                    device=self.device,
                )
            if behavior_dim <= 0:
                behavior_tensor = torch.zeros(
                    (len(ranked_candidates), 0),
                    dtype=torch.float32,
                    device=self.device,
                )
            else:
                padded_behavior = [
                    np.pad(row, (0, behavior_dim - int(row.size)), mode="constant")
                    for row in behavior_rows
                ]
                behavior_tensor = torch.as_tensor(
                    np.stack(padded_behavior, axis=0),
                    dtype=torch.float32,
                    device=self.device,
                )
            self.state = HypothesisBankState(
                object_ids=[candidate["object_id"] for candidate in ranked_candidates],
                chart_ids=resolved_chart_ids,
                scores=score_tensor,
                weights=weights,
                locations=location_tensor,
                pose_vectors=pose_tensor,
                appearance_vectors=appearance_tensor,
                behavior_vectors=behavior_tensor,
                behavior_labels=behavior_labels,
                inferred_states=inferred_states,
                ages=age_tensor,
                sources=sources,
            )
            return

        ranked = sorted(
            ((str(object_id), float(score)) for object_id, score in scores.items()),
            key=lambda item: (-item[1], item[0]),
        )[: self.max_hypotheses]
        score_tensor = torch.as_tensor(
            [score for _, score in ranked], dtype=torch.float32, device=self.device
        )
        weights = torch.softmax(score_tensor / self.temperature, dim=0)
        joint_states = joint_states or {}
        location_rows: list[np.ndarray] = []
        pose_rows: list[np.ndarray] = []
        appearance_rows: list[np.ndarray] = []
        behavior_rows: list[np.ndarray] = []
        behavior_labels: list[str | None] = []
        inferred_states: list[int | None] = []
        resolved_chart_ids: list[str | None] = []
        sources: list[str] = []
        ages: list[int] = []
        appearance_dim = 0
        behavior_dim = 0

        for object_id, _ in ranked:
            joint_state = joint_states.get(object_id, {})
            location_rows.append(
                np.asarray(
                    joint_state.get("location", default_location),
                    dtype=np.float32,
                ).reshape(3)
            )
            pose_rows.append(
                np.asarray(
                    joint_state.get("pose_vectors", default_pose),
                    dtype=np.float32,
                ).reshape(3, 3)
            )
            appearance_array = _as_float32_vector(
                joint_state.get("appearance_vector", default_appearance)
            )
            appearance_rows.append(appearance_array)
            appearance_dim = max(appearance_dim, int(appearance_array.size))
            behavior_array = _as_float32_vector(
                joint_state.get("behavior_vector", default_behavior)
            )
            behavior_rows.append(behavior_array)
            behavior_dim = max(behavior_dim, int(behavior_array.size))
            behavior_labels.append(joint_state.get("behavior_label", behavior_label))
            inferred_states.append(joint_state.get("inferred_state", inferred_state))
            sources.append(str(joint_state.get("source", "memory")))
            ages.append(max(int(joint_state.get("age", 0)), 0))
            resolved_chart_ids.append(
                joint_state.get(
                    "chart_id",
                    None if chart_ids is None else chart_ids.get(object_id),
                )
            )

        location_tensor = torch.as_tensor(
            np.stack(location_rows, axis=0),
            dtype=torch.float32,
            device=self.device,
        )
        pose_tensor = torch.as_tensor(
            np.stack(pose_rows, axis=0),
            dtype=torch.float32,
            device=self.device,
        )
        if appearance_dim <= 0:
            appearance_tensor = torch.zeros(
                (len(ranked), 0),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            padded_appearance = [
                np.pad(row, (0, appearance_dim - int(row.size)), mode="constant")
                for row in appearance_rows
            ]
            appearance_tensor = torch.as_tensor(
                np.stack(padded_appearance, axis=0),
                dtype=torch.float32,
                device=self.device,
            )
        if behavior_dim <= 0:
            behavior_tensor = torch.zeros(
                (len(ranked), 0),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            padded_behavior = [
                np.pad(row, (0, behavior_dim - int(row.size)), mode="constant")
                for row in behavior_rows
            ]
            behavior_tensor = torch.as_tensor(
                np.stack(padded_behavior, axis=0),
                dtype=torch.float32,
                device=self.device,
            )
        self.state = HypothesisBankState(
            object_ids=[object_id for object_id, _ in ranked],
            chart_ids=resolved_chart_ids,
            scores=score_tensor,
            weights=weights,
            locations=location_tensor,
            pose_vectors=pose_tensor,
            appearance_vectors=appearance_tensor,
            behavior_vectors=behavior_tensor,
            behavior_labels=behavior_labels,
            inferred_states=inferred_states,
            ages=torch.as_tensor(
                ages,
                dtype=torch.int64,
                device=self.device,
            ),
            sources=sources,
        )

    def get_possible_matches(self, min_weight: float = 0.05) -> list[str]:
        return [
            object_id
            for object_id, weight in zip(self.state.object_ids, self.state.weights)
            if float(weight.item()) >= min_weight
        ]

    def get_current_mlh(self) -> dict[str, Any]:
        if not self.state.object_ids:
            return {
                "latent_id": "no_observations_yet",
                "graph_id": "no_observations_yet",
                "evidence": 0.0,
                "rotation": Rotation.identity(),
            }

        top_latent_id = self.state.object_ids[0]
        top_weight = float(self.state.weights[0].item())
        return {
            "latent_id": top_latent_id,
            "graph_id": top_latent_id,
            "chart_id": self.state.chart_ids[0] if self.state.chart_ids else None,
            "location": self.state.locations[0].detach().cpu().numpy().astype(np.float64),
            "pose_vectors": self.state.pose_vectors[0].detach().cpu().numpy().astype(np.float64),
            "appearance_signature": (
                self.state.appearance_vectors[0].detach().cpu().numpy().astype(np.float64)
                if self.state.appearance_vectors is not None
                and self.state.appearance_vectors.ndim == 2
                and self.state.appearance_vectors.shape[0] > 0
                else np.zeros(0, dtype=np.float64)
            ),
            "behavior_label": (
                self.state.behavior_labels[0]
                if self.state.behavior_labels
                else None
            ),
            "source": (
                self.state.sources[0]
                if self.state.sources
                else None
            ),
            "age": (
                int(self.state.ages[0].item())
                if self.state.ages is not None
                and self.state.ages.numel() > 0
                else 0
            ),
            "behavior_signature": (
                self.state.behavior_vectors[0].detach().cpu().numpy().astype(np.float64)
                if self.state.behavior_vectors is not None
                and self.state.behavior_vectors.ndim == 2
                and self.state.behavior_vectors.shape[0] > 0
                else np.zeros(0, dtype=np.float64)
            ),
            "evidence": top_weight,
            "rotation": Rotation.identity(),
        }

    def get_evidence(self) -> dict[str, float]:
        evidence: dict[str, float] = {}
        for object_id, weight in zip(self.state.object_ids, self.state.weights):
            evidence[object_id] = evidence.get(object_id, 0.0) + float(weight.item())
        return evidence

    def get_latent_evidence(self) -> dict[str, float]:
        return self.get_evidence()

    def as_ranked_hypotheses(self) -> list[dict[str, Any]]:
        ranked = []
        for rank, object_id in enumerate(self.state.object_ids, start=1):
            index = rank - 1
            ranked.append(
                {
                    "object_id": object_id,
                    "latent_id": object_id,
                    "chart_id": (
                        self.state.chart_ids[index]
                        if index < len(self.state.chart_ids)
                        else None
                    ),
                    "evidence": float(self.state.scores[index].item()),
                    "probability": float(self.state.weights[index].item()),
                    "scaled_evidence": float(self.state.weights[index].item()),
                    "location": self.state.locations[index].detach().cpu().numpy().astype(np.float64),
                    "pose_vectors": self.state.pose_vectors[index].detach().cpu().numpy().astype(np.float64),
                    "appearance_signature": (
                        self.state.appearance_vectors[index].detach().cpu().numpy().astype(np.float64)
                        if self.state.appearance_vectors is not None
                        and self.state.appearance_vectors.ndim == 2
                        and index < self.state.appearance_vectors.shape[0]
                        else np.zeros(0, dtype=np.float64)
                    ),
                    "behavior_label": (
                        self.state.behavior_labels[index]
                        if index < len(self.state.behavior_labels)
                        else None
                    ),
                    "source": (
                        self.state.sources[index]
                        if index < len(self.state.sources)
                        else None
                    ),
                    "age": (
                        int(self.state.ages[index].item())
                        if self.state.ages is not None
                        and self.state.ages.numel() > index
                        else 0
                    ),
                    "behavior_signature": (
                        self.state.behavior_vectors[index].detach().cpu().numpy().astype(np.float64)
                        if self.state.behavior_vectors is not None
                        and self.state.behavior_vectors.ndim == 2
                        and index < self.state.behavior_vectors.shape[0]
                        else np.zeros(0, dtype=np.float64)
                    ),
                    "inferred_state": (
                        self.state.inferred_states[index]
                        if index < len(self.state.inferred_states)
                        else None
                    ),
                    "rank": rank,
                }
            )
        return ranked

    def get_state(self) -> HypothesisBankState:
        return self.state

    def state_dict(self) -> dict[str, Any]:
        return {
            "max_hypotheses": self.max_hypotheses,
            "temperature": self.temperature,
            "object_ids": list(self.state.object_ids),
            "latent_ids": list(self.state.object_ids),
            "chart_ids": list(self.state.chart_ids),
            "scores": self.state.scores.detach().cpu().numpy(),
            "weights": self.state.weights.detach().cpu().numpy(),
            "locations": self.state.locations.detach().cpu().numpy(),
            "pose_vectors": self.state.pose_vectors.detach().cpu().numpy(),
            "appearance_vectors": self.state.appearance_vectors.detach().cpu().numpy(),
            "behavior_vectors": self.state.behavior_vectors.detach().cpu().numpy(),
            "behavior_labels": list(self.state.behavior_labels),
            "inferred_states": list(self.state.inferred_states),
            "ages": self.state.ages.detach().cpu().numpy(),
            "sources": list(self.state.sources),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.max_hypotheses = int(state_dict.get("max_hypotheses", self.max_hypotheses))
        self.temperature = float(state_dict.get("temperature", self.temperature))
        self.state = HypothesisBankState.empty(device=self.device)
        self.state.object_ids = [
            str(object_id)
            for object_id in state_dict.get(
                "latent_ids",
                state_dict.get("object_ids", []),
            )
        ]
        self.state.chart_ids = [
            None if chart_id is None else str(chart_id)
            for chart_id in state_dict.get("chart_ids", [])
        ]
        self.state.scores = torch.as_tensor(
            state_dict.get("scores", np.zeros(0, dtype=np.float32)),
            dtype=torch.float32,
            device=self.device,
        )
        self.state.weights = torch.as_tensor(
            state_dict.get("weights", np.zeros(0, dtype=np.float32)),
            dtype=torch.float32,
            device=self.device,
        )
        self.state.locations = torch.as_tensor(
            state_dict.get("locations", np.zeros((0, 3), dtype=np.float32)),
            dtype=torch.float32,
            device=self.device,
        )
        self.state.pose_vectors = torch.as_tensor(
            state_dict.get("pose_vectors", np.zeros((0, 3, 3), dtype=np.float32)),
            dtype=torch.float32,
            device=self.device,
        )
        self.state.appearance_vectors = torch.as_tensor(
            state_dict.get("appearance_vectors", np.zeros((0, 0), dtype=np.float32)),
            dtype=torch.float32,
            device=self.device,
        )
        self.state.behavior_vectors = torch.as_tensor(
            state_dict.get("behavior_vectors", np.zeros((0, 0), dtype=np.float32)),
            dtype=torch.float32,
            device=self.device,
        )
        self.state.behavior_labels = [
            None if label is None else str(label)
            for label in state_dict.get("behavior_labels", [])
        ]
        self.state.inferred_states = list(state_dict.get("inferred_states", []))
        self.state.ages = torch.as_tensor(
            state_dict.get("ages", np.zeros(0, dtype=np.int64)),
            dtype=torch.int64,
            device=self.device,
        )
        self.state.sources = [
            str(source)
            for source in state_dict.get("sources", [])
            if source is not None
        ]


class PredictiveHypothesisCore:
    def __init__(
        self,
        context_dim: int = 256,
        max_hypotheses: int = 5,
        hypothesis_temperature: float = 0.35,
        memory_learning_rate: float = 0.15,
        memory_max_slots_per_object: int = 8,
        memory_insertion_similarity_threshold: float = 0.92,
        memory_layout: str = "fixed_sparse_recurrent_v1",
        memory_substrate_cell_count: int | None = None,
        memory_active_cell_sparsity: float = 0.08,
        prediction_bias_weight: float = 0.2,
        vote_bias_weight: float = 0.1,
        action_context_weight: float = 0.15,
        temporal_score_weight: float = 0.35,
        boundary_confidence_target: float = 0.75,
        self_supervised_object_reset_distance: float = 0.24,
        self_supervised_boundary_reset_threshold: float = 0.65,
        self_supervised_memory_reuse_threshold: float = 0.88,
        transition_score_weight: float = 0.12,
        anchor_bootstrap_novel_min_posterior: float = 0.35,
        anchor_bootstrap_existing_overwhelm_margin: float = 0.10,
        anchor_bootstrap_existing_overwhelm_ratio: float = 1.25,
        anchor_continuity_switch_margin: float = 0.12,
        anchor_continuity_switch_ratio: float = 1.35,
        seed: int = 42,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.encoder = DetailPacketEncoder(context_dim=context_dim, device=device)
        self.appearance_memory_dim = max(self.encoder.stream_dim * 2, 16)
        self.change_memory_dim = max(self.encoder.stream_dim, 16)
        self.memory_layout = str(memory_layout or "fixed_sparse_recurrent_v1")
        self.memory_learning_rate = float(memory_learning_rate)
        self.memory_max_slots_per_object = int(memory_max_slots_per_object)
        self.memory_insertion_similarity_threshold = float(
            memory_insertion_similarity_threshold
        )
        self.memory_substrate_cell_count = (
            None
            if memory_substrate_cell_count is None
            else max(int(memory_substrate_cell_count), 32)
        )
        self.memory_active_cell_sparsity = float(
            max(min(memory_active_cell_sparsity, 0.5), 0.01)
        )
        self.memory_seed = int(seed)
        self._rebuild_memories(self.memory_layout)
        self.hypotheses = HypothesisBank(
            max_hypotheses=max_hypotheses,
            temperature=hypothesis_temperature,
            device=device,
        )
        self.prediction_bias_weight = float(max(prediction_bias_weight, 0.0))
        self.vote_bias_weight = float(max(vote_bias_weight, 0.0))
        self.action_context_weight = float(max(action_context_weight, 0.0))
        self.temporal_score_weight = float(max(temporal_score_weight, 0.0))
        self.boundary_confidence_target = float(
            max(boundary_confidence_target, 1e-3)
        )
        self.self_supervised_object_reset_distance = float(
            max(self_supervised_object_reset_distance, 1e-3)
        )
        self.self_supervised_boundary_reset_threshold = float(
            max(self_supervised_boundary_reset_threshold, 1e-3)
        )
        self.self_supervised_memory_reuse_threshold = float(
            max(self_supervised_memory_reuse_threshold, 1e-3)
        )
        self.transition_score_weight = float(max(transition_score_weight, 0.0))
        self.anchor_bootstrap_novel_min_posterior = float(
            min(max(anchor_bootstrap_novel_min_posterior, 0.0), 1.0)
        )
        self.anchor_bootstrap_existing_overwhelm_margin = float(
            max(anchor_bootstrap_existing_overwhelm_margin, 0.0)
        )
        self.anchor_bootstrap_existing_overwhelm_ratio = float(
            max(anchor_bootstrap_existing_overwhelm_ratio, 1.0)
        )
        self.anchor_continuity_switch_margin = float(
            max(anchor_continuity_switch_margin, 0.0)
        )
        self.anchor_continuity_switch_ratio = float(
            max(anchor_continuity_switch_ratio, 1.0)
        )
        self.joint_appearance_score_weight = 0.08
        self.joint_pose_score_weight = 0.10
        self.joint_behavior_score_weight = 0.14
        self.context_behavior_score_weight = 0.35
        self.vote_appearance_score_weight = 0.02
        self.vote_behavior_score_weight = 0.50
        self.context_dim = self.encoder.context_dim
        self.reset_episode()

    def _rebuild_memories(self, memory_layout: str) -> None:
        self.memory_layout = str(memory_layout or self.memory_layout)
        self.appearance_memory = _make_predictive_memory(
            memory_layout=self.memory_layout,
            learning_rate=self.memory_learning_rate,
            device=str(self.device),
            max_slots_per_object=self.memory_max_slots_per_object,
            insertion_similarity_threshold=self.memory_insertion_similarity_threshold,
            embedding_dim=self.appearance_memory_dim,
            substrate_cell_count=self.memory_substrate_cell_count,
            active_cell_sparsity=self.memory_active_cell_sparsity,
            seed=self.memory_seed,
        )
        self.change_memory = _make_predictive_memory(
            memory_layout=self.memory_layout,
            learning_rate=self.memory_learning_rate,
            device=str(self.device),
            max_slots_per_object=self.memory_max_slots_per_object,
            insertion_similarity_threshold=self.memory_insertion_similarity_threshold,
            embedding_dim=self.change_memory_dim,
            substrate_cell_count=self.memory_substrate_cell_count,
            active_cell_sparsity=self.memory_active_cell_sparsity,
            seed=self.memory_seed + 1,
        )
        self.memory = self.appearance_memory

    def reset_episode(self) -> None:
        for memory in (self.appearance_memory, self.change_memory):
            if hasattr(memory, "reset_episode_state"):
                memory.reset_episode_state()
        self._mode = None
        self._predicted_latent_ids: list[str] = []
        self._predicted_chart_hypotheses: list[dict[str, Any]] = []
        self._vote_bias: dict[str, float] = {}
        self._vote_chart_bias: dict[tuple[str, str], float] = {}
        self._action_context = np.zeros(8, dtype=np.float32)
        self._external_context: torch.Tensor | None = None
        self._external_predicted_appearance_signature = torch.zeros(
            0,
            dtype=torch.float32,
            device=self.device,
        )
        self._external_predicted_change_signature = torch.zeros(
            0,
            dtype=torch.float32,
            device=self.device,
        )
        self._external_appearance_signature = torch.zeros(
            0,
            dtype=torch.float32,
            device=self.device,
        )
        self._external_behavior_signature = torch.zeros(
            0,
            dtype=torch.float32,
            device=self.device,
        )
        self._external_behavior_label: str | None = None
        self._last_packet: dict[str, Any] | None = None
        self._last_observation_field = ObservationField.empty(
            device=self.device,
            packet_type="reset_packet",
        )
        self._last_embeddings = ObservationEmbeddings.empty(
            stream_dim=self.encoder.stream_dim,
            context_dim=self.context_dim,
            device=self.device,
        )
        self._last_embedding = torch.zeros(self.context_dim, dtype=torch.float32, device=self.device)
        self._last_predicted_embedding = torch.zeros(
            self.context_dim,
            dtype=torch.float32,
            device=self.device,
        )
        self._last_predicted_appearance_signature = torch.zeros(
            self.encoder.stream_dim,
            dtype=torch.float32,
            device=self.device,
        )
        self._last_predicted_change_signature = torch.zeros(
            self.encoder.stream_dim,
            dtype=torch.float32,
            device=self.device,
        )
        self._last_appearance_residual = torch.zeros(
            self.encoder.stream_dim,
            dtype=torch.float32,
            device=self.device,
        )
        self._last_change_residual = torch.zeros(
            self.encoder.stream_dim,
            dtype=torch.float32,
            device=self.device,
        )
        self._has_last_embedding = False
        self._temporal_state = TemporalState.empty(
            context_dim=self.context_dim,
            device=self.device,
        )
        self._last_active_cells = np.zeros(self.context_dim, dtype=np.float32)
        self._last_residual = 0.0
        self._last_boundary_pressure = 0.0
        self._last_prediction_mismatch = 0.0
        self._last_action_prediction_error = 0.0
        self._last_appearance_prediction_error = 0.0
        self._last_change_prediction_error = 0.0
        self._last_base_scores: dict[str, float] = {}
        self._last_after_temporal_behavior_scores: dict[str, float] = {}
        self._last_final_scores: dict[str, float] = {}
        self._last_temporal_behavior_adjustments: dict[str, float] = {}
        self._last_temporal_behavior_scores: dict[str, float] = {}
        self._last_joint_hypothesis_adjustments: dict[str, float] = {}
        self._last_joint_hypothesis_support: dict[str, Any] = {}
        self._last_joint_hypothesis_candidates: list[dict[str, Any]] = []
        self._last_after_joint_hypothesis_scores: dict[str, float] = {}
        self._last_self_supervised_adjustments: dict[str, float] = {}
        self._last_self_supervised_support: dict[str, Any] = {}
        self._last_memory_retrieval = HopfieldRetrievalState.empty(
            embedding_dim=self.appearance_memory_dim,
            device=self.device,
        )
        self._last_change_memory_retrieval = HopfieldRetrievalState.empty(
            embedding_dim=self.change_memory_dim,
            device=self.device,
        )
        self._last_memory_stream_scores = {
            "appearance": {},
            "change": {},
            "combined": {},
        }
        self._last_memory_stream_weights = {
            "appearance": 1.0,
            "change": 0.0,
        }
        self._last_winner_path = {
            "base": None,
            "after_temporal_behavior": None,
            "after_joint_state": None,
            "final": None,
        }
        self._last_message_state = PredictiveMessageState.empty(
            context_dim=self.context_dim,
            device=self.device,
        )
        if not hasattr(self, "_latent_object_counter"):
            self._latent_object_counter = 0
        self._episode_has_anchor = False
        self._active_self_supervised_latent_id: str | None = None
        self._last_learning_latent_id: str | None = None
        self._last_learning_chart_id: str | None = None
        self._last_winning_latent_id: str | None = None
        self._last_winning_chart_id: str | None = None
        if not hasattr(self, "_chart_transition_counts"):
            self._chart_transition_counts = {}
        if not hasattr(self, "_chart_state_prototypes"):
            self._chart_state_prototypes = {}
        self.hypotheses.reset()

    def pre_episode(self, mode, object_name: str | None = None) -> None:
        self.reset_episode()
        self._mode = mode
        self._episode_has_anchor = object_name is not None

    def post_episode(self) -> None:
        return

    def _fallback_chart_id_for_latent(self, latent_id: str | None) -> str | None:
        normalized = _normalize_graph_id(latent_id)
        if normalized is None:
            return None

        for memory in (self.appearance_memory, self.change_memory):
            if not hasattr(memory, "get_slot_state"):
                continue
            slots = memory.get_slot_state()
            if normalized not in slots.object_ids:
                continue
            if slots.slot_object_indices is None or slots.slot_object_indices.numel() == 0:
                continue
            object_index = slots.object_ids.index(normalized)
            slot_indices = torch.nonzero(
                slots.slot_object_indices == int(object_index),
                as_tuple=False,
            ).reshape(-1)
            if int(slot_indices.numel()) == 0:
                continue
            slot_index = int(slot_indices[-1].item())
            if slot_index < len(slots.slot_chart_ids):
                return str(slots.slot_chart_ids[slot_index])
        return None

    def _chart_state_snapshot(self, chart_id: str | None) -> dict[str, Any] | None:
        if chart_id is None:
            return None
        prototype = self._chart_state_prototypes.get(str(chart_id))
        if prototype is None:
            return None
        return {
            "observation_count": float(prototype.get("observation_count", 0.0)),
            "location": np.asarray(
                prototype.get("location", np.zeros(3, dtype=np.float32)),
                dtype=np.float32,
            ).reshape(3),
            "pose_vectors": _normalize_pose_matrix(
                prototype.get("pose_vectors", np.eye(3, dtype=np.float32))
            ),
            "appearance_vector": _as_float32_vector(
                prototype.get("appearance_vector")
            ),
            "behavior_vector": _as_float32_vector(
                prototype.get("behavior_vector")
            ),
            "behavior_label": prototype.get("behavior_label"),
            "behavior_label_counts": {
                str(label): float(count)
                for label, count in (prototype.get("behavior_label_counts") or {}).items()
            },
        }

    def _candidate_joint_state(
        self,
        object_id: str,
        chart_id: str | None,
        *,
        fallback_location: np.ndarray,
        fallback_pose_vectors: np.ndarray,
        fallback_appearance_vector: np.ndarray,
        fallback_behavior_vector: np.ndarray,
        fallback_behavior_label: str | None,
        inferred_state: int | None,
    ) -> dict[str, Any]:
        resolved_chart_id = chart_id or self._fallback_chart_id_for_latent(object_id)
        prototype = self._chart_state_snapshot(resolved_chart_id)
        joint_state = {
            "chart_id": resolved_chart_id,
            "location": np.asarray(fallback_location, dtype=np.float32).reshape(3),
            "pose_vectors": _normalize_pose_matrix(fallback_pose_vectors),
            "appearance_vector": _as_float32_vector(fallback_appearance_vector),
            "behavior_vector": _as_float32_vector(fallback_behavior_vector),
            "behavior_label": fallback_behavior_label,
            "inferred_state": inferred_state,
        }
        if prototype is None:
            return joint_state

        joint_state["location"] = prototype["location"]
        joint_state["pose_vectors"] = prototype["pose_vectors"]
        if prototype["appearance_vector"].size > 0:
            joint_state["appearance_vector"] = prototype["appearance_vector"]
        if prototype["behavior_vector"].size > 0:
            joint_state["behavior_vector"] = prototype["behavior_vector"]
        if prototype.get("behavior_label") is not None:
            joint_state["behavior_label"] = prototype.get("behavior_label")
        return joint_state

    def _candidate_chart_ids_for_latent(
        self,
        object_id: str,
        preferred_chart_id: str | None,
    ) -> list[str | None]:
        candidate_chart_ids: list[str | None] = []
        resolved_preferred_chart_id = preferred_chart_id or self._fallback_chart_id_for_latent(
            object_id
        )
        if resolved_preferred_chart_id is not None:
            candidate_chart_ids.append(str(resolved_preferred_chart_id))

        for chart_id in sorted(self._chart_state_prototypes.keys()):
            if _chart_owner_id(chart_id) != object_id:
                continue
            normalized_chart_id = str(chart_id)
            if normalized_chart_id not in candidate_chart_ids:
                candidate_chart_ids.append(normalized_chart_id)

        if not candidate_chart_ids:
            candidate_chart_ids.append(None)
        return candidate_chart_ids

    def _build_joint_hypothesis_states(
        self,
        chart_ids: dict[str, str | None],
        *,
        location: np.ndarray,
        pose_vectors: np.ndarray,
        appearance_vector: np.ndarray,
        behavior_vector: np.ndarray,
        behavior_label: str | None,
        inferred_state: int | None,
    ) -> dict[str, list[dict[str, Any]]]:
        return {
            object_id: [
                self._candidate_joint_state(
                    object_id,
                    candidate_chart_id,
                    fallback_location=location,
                    fallback_pose_vectors=pose_vectors,
                    fallback_appearance_vector=appearance_vector,
                    fallback_behavior_vector=behavior_vector,
                    fallback_behavior_label=behavior_label,
                    inferred_state=inferred_state,
                )
                for candidate_chart_id in self._candidate_chart_ids_for_latent(
                    object_id,
                    chart_ids.get(object_id),
                )
            ]
            for object_id in chart_ids.keys()
        }

    def _update_chart_state_prototype(
        self,
        *,
        chart_id: str | None,
        location: np.ndarray,
        pose_vectors: np.ndarray,
        appearance_vector: np.ndarray,
        behavior_vector: np.ndarray,
        behavior_label: str | None,
    ) -> None:
        if chart_id is None:
            return

        normalized_chart_id = str(chart_id)
        pose_array = _normalize_pose_matrix(pose_vectors)
        location_array = np.asarray(location, dtype=np.float32).reshape(3)
        appearance_array = _as_float32_vector(appearance_vector)
        behavior_array = _as_float32_vector(behavior_vector)
        prototype = self._chart_state_snapshot(normalized_chart_id)
        if prototype is None:
            behavior_label_counts = {}
            if behavior_label is not None:
                behavior_label_counts[str(behavior_label)] = 1.0
            self._chart_state_prototypes[normalized_chart_id] = {
                "observation_count": 1.0,
                "location": location_array,
                "pose_vectors": pose_array,
                "appearance_vector": appearance_array,
                "behavior_vector": behavior_array,
                "behavior_label": behavior_label,
                "behavior_label_counts": behavior_label_counts,
            }
            return

        observation_count = float(prototype.get("observation_count", 1.0)) + 1.0
        blend = 1.0 / min(observation_count, 4.0)
        updated_location = ((1.0 - blend) * prototype["location"]) + (blend * location_array)
        updated_pose = _normalize_pose_matrix(
            ((1.0 - blend) * prototype["pose_vectors"]) + (blend * pose_array)
        )

        previous_appearance = prototype["appearance_vector"]
        if previous_appearance.size == 0:
            updated_appearance = appearance_array
        elif appearance_array.size == 0:
            updated_appearance = previous_appearance
        else:
            target_dim = max(int(previous_appearance.size), int(appearance_array.size))
            previous_tensor = _resize_vector(
                torch.as_tensor(previous_appearance, dtype=torch.float32),
                target_dim,
            )
            current_tensor = _resize_vector(
                torch.as_tensor(appearance_array, dtype=torch.float32),
                target_dim,
            )
            updated_appearance = (
                ((1.0 - blend) * previous_tensor) + (blend * current_tensor)
            ).detach().cpu().numpy().astype(np.float32)

        previous_behavior = prototype["behavior_vector"]
        if previous_behavior.size == 0:
            updated_behavior = behavior_array
        elif behavior_array.size == 0:
            updated_behavior = previous_behavior
        else:
            target_dim = max(int(previous_behavior.size), int(behavior_array.size))
            previous_tensor = _resize_vector(
                torch.as_tensor(previous_behavior, dtype=torch.float32),
                target_dim,
            )
            current_tensor = _resize_vector(
                torch.as_tensor(behavior_array, dtype=torch.float32),
                target_dim,
            )
            updated_behavior = (
                ((1.0 - blend) * previous_tensor) + (blend * current_tensor)
            ).detach().cpu().numpy().astype(np.float32)

        label_counts = dict(prototype.get("behavior_label_counts") or {})
        if behavior_label is not None:
            label_counts[str(behavior_label)] = float(
                label_counts.get(str(behavior_label), 0.0)
            ) + 1.0
        dominant_label = (
            max(label_counts.items(), key=lambda item: (float(item[1]), str(item[0])))[0]
            if label_counts
            else behavior_label
        )
        self._chart_state_prototypes[normalized_chart_id] = {
            "observation_count": observation_count,
            "location": updated_location.astype(np.float32),
            "pose_vectors": updated_pose.astype(np.float32),
            "appearance_vector": updated_appearance.astype(np.float32),
            "behavior_vector": updated_behavior.astype(np.float32),
            "behavior_label": dominant_label,
            "behavior_label_counts": {
                str(label): float(count)
                for label, count in label_counts.items()
            },
        }

    @staticmethod
    def _candidate_row_sort_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
        chart_id = candidate.get("chart_id")
        object_id = candidate.get("object_id", candidate.get("latent_id", ""))
        return (
            -float(candidate.get("score", 0.0)),
            str(object_id),
            "" if chart_id is None else str(chart_id),
            str(candidate.get("source", "")),
            int(candidate.get("age", 0)),
        )

    def _merge_joint_candidate_rows(
        self,
        candidates: list[dict[str, Any]],
        *,
        fallback_scores: dict[str, float] | None = None,
        fallback_chart_ids: dict[str, str | None] | None = None,
        fallback_joint_states: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[
        dict[str, float],
        dict[str, str | None],
        dict[str, dict[str, Any]],
        list[dict[str, Any]],
    ]:
        fallback_scores = {
            str(object_id): float(score)
            for object_id, score in (fallback_scores or {}).items()
        }
        fallback_chart_ids = {
            str(object_id): (
                None if chart_id is None else str(chart_id)
            )
            for object_id, chart_id in (fallback_chart_ids or {}).items()
        }
        fallback_joint_states = {
            str(object_id): dict(state)
            for object_id, state in (fallback_joint_states or {}).items()
            if state is not None
        }

        best_by_key: dict[tuple[str, str | None, str | None], dict[str, Any]] = {}
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue

            object_id = _normalize_graph_id(
                candidate.get("object_id", candidate.get("latent_id"))
            )
            if object_id is None:
                continue

            chart_id = _normalize_graph_id(candidate.get("chart_id"))
            behavior_label = (
                None
                if candidate.get("behavior_label") is None
                else str(candidate.get("behavior_label"))
            )
            source = str(candidate.get("source", "memory"))
            age = max(int(candidate.get("age", 0)), 0)
            score = float(
                candidate.get(
                    "score",
                    fallback_scores.get(object_id, 0.0),
                )
            )
            base_score = float(candidate.get("base_score", score))
            normalized_candidate = {
                "object_id": object_id,
                "latent_id": object_id,
                "chart_id": chart_id,
                "score": score,
                "base_score": base_score,
                "joint_delta": float(
                    candidate.get("joint_delta", max(score - base_score, 0.0))
                ),
                "pose_support": float(candidate.get("pose_support", 0.0)),
                "appearance_support": float(candidate.get("appearance_support", 0.0)),
                "behavior_similarity": float(
                    candidate.get("behavior_similarity", 0.0)
                ),
                "label_match": float(candidate.get("label_match", 0.0)),
                "behavior_support": float(candidate.get("behavior_support", 0.0)),
                "context_appearance_support": float(
                    candidate.get("context_appearance_support", 0.0)
                ),
                "context_behavior_support": float(
                    candidate.get("context_behavior_support", 0.0)
                ),
                "context_ranked_chart_prior": float(
                    candidate.get("context_ranked_chart_prior", 0.0)
                ),
                "vote_ranked_chart_prior": float(
                    candidate.get("vote_ranked_chart_prior", 0.0)
                ),
                "selected": bool(candidate.get("selected", False)),
                "location": np.asarray(
                    candidate.get("location", np.zeros(3, dtype=np.float32)),
                    dtype=np.float32,
                ).reshape(3),
                "pose_vectors": np.asarray(
                    candidate.get("pose_vectors", np.eye(3, dtype=np.float32)),
                    dtype=np.float32,
                ).reshape(3, 3),
                "appearance_vector": _as_float32_vector(
                    candidate.get("appearance_vector")
                ),
                "behavior_vector": _as_float32_vector(
                    candidate.get("behavior_vector")
                ),
                "behavior_label": behavior_label,
                "inferred_state": candidate.get("inferred_state"),
                "source": source,
                "age": age,
            }
            candidate_key = (object_id, chart_id, behavior_label)
            current = best_by_key.get(candidate_key)
            if current is None or self._candidate_row_sort_key(
                normalized_candidate
            ) < self._candidate_row_sort_key(current):
                best_by_key[candidate_key] = normalized_candidate

        merged_candidates = sorted(
            best_by_key.values(),
            key=self._candidate_row_sort_key,
        )
        scores: dict[str, float] = {}
        chart_ids: dict[str, str | None] = {}
        joint_states: dict[str, dict[str, Any]] = {}
        for candidate in merged_candidates:
            object_id = str(candidate["object_id"])
            current_score = scores.get(object_id, -np.inf)
            current_chart_id = chart_ids.get(object_id)
            if (
                float(candidate["score"]) > float(current_score)
                or (
                    float(candidate["score"]) == float(current_score)
                    and (
                        "" if candidate["chart_id"] is None else str(candidate["chart_id"])
                    )
                    < (
                        "" if current_chart_id is None else str(current_chart_id)
                    )
                )
            ):
                scores[object_id] = float(candidate["score"])
                chart_ids[object_id] = candidate["chart_id"]
                joint_states[object_id] = {
                    "chart_id": candidate["chart_id"],
                    "location": np.asarray(
                        candidate["location"],
                        dtype=np.float32,
                    ).reshape(3),
                    "pose_vectors": np.asarray(
                        candidate["pose_vectors"],
                        dtype=np.float32,
                    ).reshape(3, 3),
                    "appearance_vector": _as_float32_vector(
                        candidate.get("appearance_vector")
                    ),
                    "behavior_vector": _as_float32_vector(
                        candidate.get("behavior_vector")
                    ),
                    "behavior_label": candidate.get("behavior_label"),
                    "inferred_state": candidate.get("inferred_state"),
                    "source": candidate.get("source", "memory"),
                    "age": max(int(candidate.get("age", 0)), 0),
                }

        for object_id, fallback_score in fallback_scores.items():
            fallback_state = fallback_joint_states.get(object_id)
            current_score = scores.get(object_id, -np.inf)
            if (
                fallback_state is None
                and object_id in scores
            ):
                continue
            if (
                object_id not in scores
                or float(fallback_score) > float(current_score)
            ):
                scores[object_id] = float(fallback_score)
                chart_id = (
                    None
                    if fallback_state is None
                    else fallback_state.get(
                        "chart_id",
                        fallback_chart_ids.get(object_id),
                    )
                )
                chart_ids[object_id] = (
                    None if chart_id is None else str(chart_id)
                )
                if fallback_state is not None:
                    joint_states[object_id] = {
                        "chart_id": chart_ids[object_id],
                        "location": np.asarray(
                            fallback_state.get(
                                "location",
                                np.zeros(3, dtype=np.float32),
                            ),
                            dtype=np.float32,
                        ).reshape(3),
                        "pose_vectors": np.asarray(
                            fallback_state.get(
                                "pose_vectors",
                                np.eye(3, dtype=np.float32),
                            ),
                            dtype=np.float32,
                        ).reshape(3, 3),
                        "appearance_vector": _as_float32_vector(
                            fallback_state.get("appearance_vector")
                        ),
                        "behavior_vector": _as_float32_vector(
                            fallback_state.get("behavior_vector")
                        ),
                        "behavior_label": fallback_state.get("behavior_label"),
                        "inferred_state": fallback_state.get("inferred_state"),
                        "source": str(fallback_state.get("source", "memory")),
                        "age": max(int(fallback_state.get("age", 0)), 0),
                    }

        return scores, chart_ids, joint_states, merged_candidates

    def _propagate_bank_candidates(
        self,
        *,
        location: np.ndarray,
        pose_vectors: np.ndarray,
        appearance_vector: np.ndarray,
        behavior_vector: np.ndarray,
        behavior_label: str | None,
    ) -> list[dict[str, Any]]:
        if not self.hypotheses.state.object_ids:
            return []

        cue_pose = _normalize_pose_matrix(pose_vectors)
        cue_appearance = _as_float32_vector(appearance_vector)
        cue_behavior = _as_float32_vector(behavior_vector)
        context_appearance = (
            self._external_appearance_signature.detach().cpu().numpy().astype(np.float32)
        )
        context_behavior = (
            self._external_behavior_signature.detach().cpu().numpy().astype(np.float32)
        )
        persistence_scale = 1.0
        if behavior_label == "temporal_change":
            persistence_scale = 0.20
        elif behavior_label == "hierarchy_fusion":
            persistence_scale = 0.35
        candidates: list[dict[str, Any]] = []
        for entry in self.hypotheses.entries:
            chart_id = entry.chart_id or self._fallback_chart_id_for_latent(entry.object_id)
            entry_appearance = _as_float32_vector(entry.appearance_vector)
            entry_behavior = _as_float32_vector(entry.behavior_vector)
            observation_components = self._joint_support_components(
                candidate_pose_vectors=np.asarray(
                    entry.pose_vectors,
                    dtype=np.float32,
                ),
                candidate_appearance_vector=entry_appearance,
                candidate_behavior_vector=entry_behavior,
                candidate_behavior_label=entry.behavior_label,
                cue_pose_vectors=cue_pose,
                cue_appearance_vector=cue_appearance,
                cue_behavior_vector=cue_behavior,
                cue_behavior_label=behavior_label,
            )
            context_components = self._joint_support_components(
                candidate_pose_vectors=np.asarray(
                    entry.pose_vectors,
                    dtype=np.float32,
                ),
                candidate_appearance_vector=entry_appearance,
                candidate_behavior_vector=entry_behavior,
                candidate_behavior_label=entry.behavior_label,
                cue_appearance_vector=context_appearance,
                cue_behavior_vector=context_behavior,
                cue_behavior_label=self._external_behavior_label,
            )
            next_age = max(int(getattr(entry, "age", 0)), 0) + 1
            persistence_bonus = (0.18 * float(entry.weight)) + (
                0.06 * float(min(next_age, 3))
            )
            if self._active_self_supervised_latent_id == entry.object_id:
                persistence_bonus += 0.03
            if self._last_winning_latent_id == entry.object_id:
                persistence_bonus += 0.03
            if chart_id is not None and chart_id == self._last_winning_chart_id:
                persistence_bonus += 0.02
            persistence_bonus = persistence_bonus * persistence_scale

            context_ranked_chart_prior = self._context_ranked_chart_prior(
                entry.object_id,
                chart_id,
            )
            vote_ranked_chart_prior = self._vote_ranked_chart_prior(
                entry.object_id,
                chart_id,
            )
            candidate_delta = (
                self.joint_appearance_score_weight
                * float(observation_components["appearance_support"])
            ) + (
                self.joint_pose_score_weight
                * float(observation_components["pose_support"])
            ) + (
                self.joint_behavior_score_weight
                * float(observation_components["behavior_support"])
            ) + (
                self.joint_appearance_score_weight
                * float(context_components["appearance_support"])
            ) + (
                self.context_behavior_score_weight
                * float(context_components["behavior_support"])
            ) + float(context_ranked_chart_prior) + float(vote_ranked_chart_prior) + float(
                persistence_bonus
            )
            base_score = (
                (0.35 * float(entry.score)) + (0.65 * float(entry.weight))
            ) * persistence_scale
            score = float(base_score + max(candidate_delta, 0.0))
            candidates.append(
                {
                    "object_id": entry.object_id,
                    "latent_id": entry.object_id,
                    "chart_id": chart_id,
                    "score": score,
                    "base_score": float(base_score),
                    "joint_delta": float(max(score - base_score, 0.0)),
                    "pose_support": float(observation_components["pose_support"]),
                    "appearance_support": float(
                        observation_components["appearance_support"]
                    ),
                    "behavior_similarity": float(
                        observation_components["behavior_similarity"]
                    ),
                    "label_match": float(observation_components["label_match"]),
                    "behavior_support": float(
                        observation_components["behavior_support"]
                    ),
                    "context_appearance_support": float(
                        context_components["appearance_support"]
                    ),
                    "context_behavior_support": float(
                        context_components["behavior_support"]
                    ),
                    "context_ranked_chart_prior": float(context_ranked_chart_prior),
                    "vote_ranked_chart_prior": float(vote_ranked_chart_prior),
                    "selected": False,
                    "location": np.asarray(
                        entry.location,
                        dtype=np.float32,
                    ).reshape(3),
                    "pose_vectors": np.asarray(
                        entry.pose_vectors,
                        dtype=np.float32,
                    ).reshape(3, 3),
                    "appearance_vector": entry_appearance,
                    "behavior_vector": entry_behavior,
                    "behavior_label": entry.behavior_label,
                    "inferred_state": entry.inferred_state,
                    "source": "propagated",
                    "age": next_age,
                }
            )
        return candidates

    def _joint_support_components(
        self,
        *,
        candidate_pose_vectors: np.ndarray,
        candidate_appearance_vector: np.ndarray,
        candidate_behavior_vector: np.ndarray,
        candidate_behavior_label: str | None,
        cue_pose_vectors: np.ndarray | None = None,
        cue_appearance_vector: np.ndarray | None = None,
        cue_behavior_vector: np.ndarray | None = None,
        cue_behavior_label: str | None = None,
    ) -> dict[str, float]:
        pose_support = (
            _resized_cosine_similarity(candidate_pose_vectors, cue_pose_vectors)
            if cue_pose_vectors is not None
            else 0.0
        )
        appearance_support = (
            _resized_cosine_similarity(candidate_appearance_vector, cue_appearance_vector)
            if cue_appearance_vector is not None and cue_appearance_vector.size > 0
            else 0.0
        )
        behavior_similarity = (
            _resized_cosine_similarity(candidate_behavior_vector, cue_behavior_vector)
            if cue_behavior_vector is not None and cue_behavior_vector.size > 0
            else 0.0
        )
        label_match = 1.0 if (
            cue_behavior_label is not None
            and candidate_behavior_label is not None
            and str(cue_behavior_label) == str(candidate_behavior_label)
        ) else 0.0
        behavior_support = (0.85 * behavior_similarity) + (0.15 * label_match)
        return {
            "pose_support": float(pose_support),
            "appearance_support": float(appearance_support),
            "behavior_similarity": float(behavior_similarity),
            "label_match": float(label_match),
            "behavior_support": float(behavior_support),
        }

    def _score_chart_cue_candidates(
        self,
        *,
        pose_vectors: np.ndarray | None = None,
        appearance_vector: np.ndarray | None = None,
        behavior_vector: np.ndarray | None = None,
        behavior_label: str | None = None,
        pose_weight: float | None = None,
        appearance_weight: float | None = None,
        behavior_weight: float | None = None,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        if not self._chart_state_prototypes:
            return {}, {}

        cue_pose = None if pose_vectors is None else _normalize_pose_matrix(pose_vectors)
        cue_appearance = _as_float32_vector(appearance_vector)
        cue_behavior = _as_float32_vector(behavior_vector)
        if cue_pose is None and cue_appearance.size == 0 and cue_behavior.size == 0 and behavior_label is None:
            return {}, {}

        resolved_pose_weight = (
            0.5 * self.joint_pose_score_weight
            if pose_weight is None
            else float(max(pose_weight, 0.0))
        )
        resolved_appearance_weight = (
            self.joint_appearance_score_weight
            if appearance_weight is None
            else float(max(appearance_weight, 0.0))
        )
        resolved_behavior_weight = (
            self.vote_behavior_score_weight
            if behavior_weight is None
            else float(max(behavior_weight, 0.0))
        )
        scores: dict[str, float] = {}
        support: dict[str, Any] = {}
        known_ids = self._known_internal_latent_ids()
        for chart_id in sorted(self._chart_state_prototypes.keys()):
            object_id = _chart_owner_id(chart_id)
            if object_id is None or object_id not in known_ids:
                continue
            prototype = self._chart_state_snapshot(chart_id)
            if prototype is None:
                continue
            components = self._joint_support_components(
                candidate_pose_vectors=prototype["pose_vectors"],
                candidate_appearance_vector=prototype["appearance_vector"],
                candidate_behavior_vector=prototype["behavior_vector"],
                candidate_behavior_label=prototype.get("behavior_label"),
                cue_pose_vectors=cue_pose,
                cue_appearance_vector=cue_appearance,
                cue_behavior_vector=cue_behavior,
                cue_behavior_label=behavior_label,
            )
            delta = (
                resolved_pose_weight * float(components["pose_support"])
            ) + (
                resolved_appearance_weight * float(components["appearance_support"])
            ) + (
                resolved_behavior_weight * float(components["behavior_support"])
            )
            if delta <= scores.get(object_id, -np.inf):
                continue
            scores[object_id] = float(delta)
            support[object_id] = {
                "chart_id": chart_id,
                **components,
            }
        return scores, support

    def _apply_joint_hypothesis_support(
        self,
        scores: dict[str, float],
        chart_ids: dict[str, str | None],
        *,
        location: np.ndarray,
        pose_vectors: np.ndarray,
        appearance_vector: np.ndarray,
        behavior_vector: np.ndarray,
        behavior_label: str | None,
        inferred_state: int | None,
    ) -> tuple[
        dict[str, float],
        dict[str, float],
        dict[str, Any],
        dict[str, dict[str, Any]],
        list[dict[str, Any]],
    ]:
        adjusted_scores = {
            str(object_id): float(score)
            for object_id, score in scores.items()
        }
        if not adjusted_scores:
            return adjusted_scores, {}, {}, {}, []

        joint_state_candidates = self._build_joint_hypothesis_states(
            {
                object_id: chart_ids.get(object_id)
                for object_id in adjusted_scores.keys()
            },
            location=location,
            pose_vectors=pose_vectors,
            appearance_vector=appearance_vector,
            behavior_vector=behavior_vector,
            behavior_label=behavior_label,
            inferred_state=inferred_state,
        )
        adjustments: dict[str, float] = {}
        support: dict[str, Any] = {}
        resolved_joint_states: dict[str, dict[str, Any]] = {}
        ranked_joint_candidates: list[dict[str, Any]] = []
        cue_pose = _normalize_pose_matrix(pose_vectors)
        cue_appearance = _as_float32_vector(appearance_vector)
        cue_behavior = _as_float32_vector(behavior_vector)
        context_appearance = self._external_appearance_signature.detach().cpu().numpy().astype(np.float32)
        context_behavior = self._external_behavior_signature.detach().cpu().numpy().astype(np.float32)
        for object_id, candidate_states in joint_state_candidates.items():
            base_score = float(scores.get(object_id, 0.0))
            best_score = base_score
            best_state = dict(candidate_states[0]) if candidate_states else {
                "chart_id": chart_ids.get(object_id),
                "location": np.asarray(location, dtype=np.float32).reshape(3),
                "pose_vectors": _normalize_pose_matrix(pose_vectors),
                "appearance_vector": _as_float32_vector(appearance_vector),
                "behavior_vector": _as_float32_vector(behavior_vector),
                "behavior_label": behavior_label,
                "inferred_state": inferred_state,
            }
            candidate_chart_ids: list[str | None] = []
            candidate_rows_for_object: list[dict[str, Any]] = []
            best_support = {
                "chart_id": best_state.get("chart_id"),
                "pose_support": 0.0,
                "appearance_support": 0.0,
                "behavior_similarity": 0.0,
                "label_match": 0.0,
                "behavior_support": 0.0,
                "context_appearance_support": 0.0,
                "context_behavior_support": 0.0,
            }

            for candidate_state in candidate_states:
                chart_id = candidate_state.get("chart_id")
                candidate_chart_ids.append(chart_id)
                candidate_support = {
                    "chart_id": chart_id,
                    "pose_support": 0.0,
                    "appearance_support": 0.0,
                    "behavior_similarity": 0.0,
                    "label_match": 0.0,
                    "behavior_support": 0.0,
                    "context_appearance_support": 0.0,
                    "context_behavior_support": 0.0,
                    "context_ranked_chart_prior": 0.0,
                    "vote_ranked_chart_prior": 0.0,
                }
                candidate_delta = 0.0
                if chart_id is not None and chart_id in self._chart_state_prototypes:
                    observation_components = self._joint_support_components(
                        candidate_pose_vectors=np.asarray(
                            candidate_state["pose_vectors"],
                            dtype=np.float32,
                        ),
                        candidate_appearance_vector=_as_float32_vector(
                            candidate_state.get("appearance_vector")
                        ),
                        candidate_behavior_vector=_as_float32_vector(
                            candidate_state.get("behavior_vector")
                        ),
                        candidate_behavior_label=candidate_state.get("behavior_label"),
                        cue_pose_vectors=cue_pose,
                        cue_appearance_vector=cue_appearance,
                        cue_behavior_vector=cue_behavior,
                        cue_behavior_label=behavior_label,
                    )
                    context_components = self._joint_support_components(
                        candidate_pose_vectors=np.asarray(
                            candidate_state["pose_vectors"],
                            dtype=np.float32,
                        ),
                        candidate_appearance_vector=_as_float32_vector(
                            candidate_state.get("appearance_vector")
                        ),
                        candidate_behavior_vector=_as_float32_vector(
                            candidate_state.get("behavior_vector")
                        ),
                        candidate_behavior_label=candidate_state.get("behavior_label"),
                        cue_appearance_vector=context_appearance,
                        cue_behavior_vector=context_behavior,
                        cue_behavior_label=self._external_behavior_label,
                    )
                    candidate_delta = (
                        self.joint_appearance_score_weight
                        * float(observation_components["appearance_support"])
                    ) + (
                        self.joint_pose_score_weight
                        * float(observation_components["pose_support"])
                    ) + (
                        self.joint_behavior_score_weight
                        * float(observation_components["behavior_support"])
                    ) + (
                        self.joint_appearance_score_weight
                        * float(context_components["appearance_support"])
                    ) + (
                        self.context_behavior_score_weight
                        * float(context_components["behavior_support"])
                    )
                    candidate_support = {
                        "chart_id": chart_id,
                        **observation_components,
                        "context_appearance_support": float(
                            context_components["appearance_support"]
                        ),
                        "context_behavior_support": float(
                            context_components["behavior_support"]
                        ),
                        "context_ranked_chart_prior": 0.0,
                        "vote_ranked_chart_prior": 0.0,
                    }

                context_ranked_chart_prior = self._context_ranked_chart_prior(
                    object_id,
                    chart_id,
                )
                vote_ranked_chart_prior = self._vote_ranked_chart_prior(
                    object_id,
                    chart_id,
                )
                candidate_delta += float(context_ranked_chart_prior)
                candidate_delta += float(vote_ranked_chart_prior)
                candidate_support["context_ranked_chart_prior"] = float(
                    context_ranked_chart_prior
                )
                candidate_support["vote_ranked_chart_prior"] = float(
                    vote_ranked_chart_prior
                )

                candidate_score = base_score + max(float(candidate_delta), 0.0)
                candidate_row = {
                    "object_id": object_id,
                    "latent_id": object_id,
                    "chart_id": chart_id,
                    "score": float(candidate_score),
                    "base_score": float(base_score),
                    "joint_delta": float(max(float(candidate_delta), 0.0)),
                    "pose_support": float(candidate_support.get("pose_support", 0.0)),
                    "appearance_support": float(
                        candidate_support.get("appearance_support", 0.0)
                    ),
                    "behavior_similarity": float(
                        candidate_support.get("behavior_similarity", 0.0)
                    ),
                    "label_match": float(candidate_support.get("label_match", 0.0)),
                    "behavior_support": float(
                        candidate_support.get("behavior_support", 0.0)
                    ),
                    "context_appearance_support": float(
                        candidate_support.get("context_appearance_support", 0.0)
                    ),
                    "context_behavior_support": float(
                        candidate_support.get("context_behavior_support", 0.0)
                    ),
                    "context_ranked_chart_prior": float(
                        candidate_support.get("context_ranked_chart_prior", 0.0)
                    ),
                    "vote_ranked_chart_prior": float(
                        candidate_support.get("vote_ranked_chart_prior", 0.0)
                    ),
                    "selected": False,
                    "location": np.asarray(
                        candidate_state.get("location", location),
                        dtype=np.float32,
                    ).reshape(3),
                    "pose_vectors": np.asarray(
                        candidate_state.get("pose_vectors", pose_vectors),
                        dtype=np.float32,
                    ).reshape(3, 3),
                    "appearance_vector": _as_float32_vector(
                        candidate_state.get("appearance_vector")
                    ),
                    "behavior_vector": _as_float32_vector(
                        candidate_state.get("behavior_vector")
                    ),
                    "behavior_label": candidate_state.get("behavior_label"),
                    "inferred_state": candidate_state.get("inferred_state"),
                    "source": str(candidate_state.get("source", "memory")),
                    "age": max(int(candidate_state.get("age", 0)), 0),
                }
                candidate_rows_for_object.append(candidate_row)
                ranked_joint_candidates.append(candidate_row)
                if candidate_score > best_score:
                    best_score = float(candidate_score)
                    best_state = dict(candidate_state)
                    best_support = candidate_support

            best_chart_id = best_state.get("chart_id")
            selected_assigned = False
            for candidate_row in candidate_rows_for_object:
                if (
                    not selected_assigned
                    and candidate_row.get("chart_id") == best_chart_id
                ):
                    candidate_row["selected"] = True
                    selected_assigned = True
                else:
                    candidate_row["selected"] = False

            adjusted_scores[object_id] = float(best_score)
            if best_score > base_score:
                adjustments[object_id] = float(best_score - base_score)
            best_support = dict(best_support)
            best_support["candidate_chart_ids"] = list(candidate_chart_ids)
            support[object_id] = best_support
            resolved_joint_states[object_id] = best_state
        return (
            adjusted_scores,
            adjustments,
            support,
            resolved_joint_states,
            ranked_joint_candidates,
        )

    def _known_internal_latent_ids(self) -> set[str]:
        return set(self.appearance_memory.get_all_known_object_ids()) | set(
            self.change_memory.get_all_known_object_ids()
        )

    def _best_memory_reuse_latent_candidate(
        self,
        scores: dict[str, float],
    ) -> tuple[str | None, float]:
        if not scores:
            return None, 0.0

        object_id, score = max(
            ((str(object_id), float(score)) for object_id, score in scores.items()),
            key=lambda item: (float(item[1]), str(item[0])),
        )
        if float(score) < self.self_supervised_memory_reuse_threshold:
            return None, float(score)
        return object_id, float(score)

    def _output_summary(
        self,
    ) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, Any]]:
        ranked_raw = self.hypotheses.as_ranked_hypotheses()
        if not ranked_raw:
            return {}, [], {
                "latent_id": "no_observations_yet",
                "graph_id": "no_observations_yet",
                "evidence": 0.0,
                "rotation": Rotation.identity(),
            }
        return (
            self.hypotheses.get_evidence(),
            [dict(hypothesis) for hypothesis in ranked_raw],
            self.hypotheses.get_current_mlh(),
        )

    def set_action_context(self, action_context) -> None:
        if action_context is None:
            self._action_context = np.zeros(8, dtype=np.float32)
            return
        self._action_context = np.asarray(action_context, dtype=np.float32).reshape(-1)

    def receive_context(self, **context_signal) -> None:
        self.receive_context_message(
            PredictiveContextSignal.from_dict(
                context_signal,
                device=self.device,
            )
        )

    def receive_context_message(self, context_signal: PredictiveContextSignal) -> None:
        predicted_latent_ids: list[str] = []
        predicted_latent_id = _normalize_graph_id(context_signal.latent_id)
        if predicted_latent_id is not None:
            predicted_latent_ids.append(predicted_latent_id)
        elif context_signal.ranked_hypotheses:
            for hypothesis in context_signal.ranked_hypotheses:
                normalized = _normalize_graph_id(hypothesis.latent_id)
                if normalized is None or normalized in predicted_latent_ids:
                    continue
                predicted_latent_ids.append(normalized)
        for latent_id in context_signal.child_latent_ids:
            normalized = _normalize_graph_id(latent_id)
            if normalized is None or normalized in predicted_latent_ids:
                continue
            predicted_latent_ids.append(normalized)
        self._predicted_latent_ids = predicted_latent_ids
        self._predicted_chart_hypotheses = []
        seen_chart_targets: set[tuple[str, str]] = set()
        for rank, hypothesis in enumerate(context_signal.ranked_hypotheses, start=1):
            normalized_latent_id = _normalize_graph_id(hypothesis.latent_id)
            normalized_chart_id = _normalize_graph_id(hypothesis.chart_id)
            if normalized_latent_id is None or normalized_chart_id is None:
                continue
            chart_target = (normalized_latent_id, normalized_chart_id)
            if chart_target in seen_chart_targets:
                continue
            seen_chart_targets.add(chart_target)
            self._predicted_chart_hypotheses.append(
                {
                    "latent_id": normalized_latent_id,
                    "chart_id": normalized_chart_id,
                    "rank": int(rank),
                    "probability": max(float(hypothesis.probability), 0.0),
                }
            )

        top_ranked_hypothesis = (
            context_signal.ranked_hypotheses[0]
            if context_signal.ranked_hypotheses
            else None
        )

        self._external_predicted_appearance_signature = (
            context_signal.predicted_appearance_signature.to(
                device=self.device,
                dtype=torch.float32,
            ).reshape(-1)
        )
        self._external_predicted_change_signature = (
            context_signal.predicted_change_signature.to(
                device=self.device,
                dtype=torch.float32,
            ).reshape(-1)
        )
        self._external_appearance_signature = context_signal.appearance_signature.to(
            device=self.device,
            dtype=torch.float32,
        ).reshape(-1)
        if (
            self._external_appearance_signature.numel() == 0
            and top_ranked_hypothesis is not None
            and top_ranked_hypothesis.appearance_signature.numel() > 0
        ):
            self._external_appearance_signature = top_ranked_hypothesis.appearance_signature.to(
                device=self.device,
                dtype=torch.float32,
            ).reshape(-1)
        self._external_behavior_signature = context_signal.behavior_signature.to(
            device=self.device,
            dtype=torch.float32,
        ).reshape(-1)
        if (
            self._external_behavior_signature.numel() == 0
            and top_ranked_hypothesis is not None
            and top_ranked_hypothesis.behavior_signature.numel() > 0
        ):
            self._external_behavior_signature = top_ranked_hypothesis.behavior_signature.to(
                device=self.device,
                dtype=torch.float32,
            ).reshape(-1)
        self._external_behavior_label = context_signal.behavior_label
        if self._external_behavior_label is None and top_ranked_hypothesis is not None:
            self._external_behavior_label = top_ranked_hypothesis.behavior_label

        if context_signal.active_cells.numel() == 0:
            self._external_context = None
            return

        tensor = context_signal.active_cells.to(
            device=self.device,
            dtype=torch.float32,
        ).reshape(-1)
        tensor = _resize_vector(tensor, self.context_dim)
        norm = float(tensor.norm(p=2).item())
        if norm > 1e-8:
            tensor = tensor / norm
        self._external_context = tensor

    def _vote_scores_from_activity(self, active_cells: torch.Tensor) -> dict[str, float]:
        if active_cells.numel() == 0 or not self._known_internal_latent_ids():
            return {}

        cue = active_cells.to(device=self.device, dtype=torch.float32).reshape(-1)
        if float(cue.abs().sum().item()) <= 1e-8:
            return {}

        cue = _resize_vector(cue, self.context_dim)
        cue_norm = float(cue.norm(p=2).item())
        if cue_norm > 1e-8:
            cue = cue / cue_norm

        appearance_query = _resize_vector(cue, self.appearance_memory_dim)
        change_query = _resize_vector(cue, self.change_memory_dim)
        appearance_norm = float(appearance_query.norm(p=2).item())
        change_norm = float(change_query.norm(p=2).item())
        if appearance_norm > 1e-8:
            appearance_query = appearance_query / appearance_norm
        if change_norm > 1e-8:
            change_query = change_query / change_norm

        appearance_retrieval = self.appearance_memory.retrieve(
            appearance_query,
            store_as_last=False,
        )
        change_retrieval = self.change_memory.retrieve(
            change_query,
            store_as_last=False,
        )
        scores, _, _ = self._combine_stream_scores(
            self.appearance_memory.retrieval_scores(appearance_retrieval),
            self.change_memory.retrieval_scores(change_retrieval),
            self.appearance_memory.retrieval_chart_ids(appearance_retrieval),
            self.change_memory.retrieval_chart_ids(change_retrieval),
            packet_type="lm_fusion_packet",
        )
        return scores

    def _context_ranked_chart_prior(
        self,
        object_id: str,
        chart_id: str | None,
    ) -> float:
        if chart_id is None:
            return 0.0

        best_prior = 0.0
        for target in self._predicted_chart_hypotheses:
            if target.get("latent_id") != object_id or target.get("chart_id") != chart_id:
                continue
            rank = max(int(target.get("rank", 1)), 1)
            probability = max(float(target.get("probability", 0.0)), 0.0)
            prior = 0.5 * self.prediction_bias_weight * (probability / float(rank))
            if prior > best_prior:
                best_prior = float(prior)
        return best_prior

    def _vote_ranked_chart_prior(
        self,
        object_id: str,
        chart_id: str | None,
    ) -> float:
        if chart_id is None:
            return 0.0
        return 0.5 * self.vote_bias_weight * float(
            self._vote_chart_bias.get((str(object_id), str(chart_id)), 0.0)
        )

    def receive_vote_message(self, vote_message: PredictiveVoteMessage) -> None:
        activity_scores = self._vote_scores_from_activity(vote_message.active_cells)
        vote_appearance_signature = vote_message.appearance_signature.detach().cpu().numpy().astype(np.float32)
        vote_behavior_signature = vote_message.behavior_signature.detach().cpu().numpy().astype(np.float32)
        vote_behavior_label = vote_message.behavior_label
        if vote_message.ranked_hypotheses:
            top_hypothesis = vote_message.ranked_hypotheses[0]
            if vote_behavior_signature.size == 0:
                vote_behavior_signature = (
                    top_hypothesis.behavior_signature.detach().cpu().numpy().astype(np.float32)
                )
                vote_behavior_label = top_hypothesis.behavior_label
        vote_pose_vectors = None
        if vote_message.sensed_pose_rel_body.shape[0] >= 4:
            vote_pose_vectors = vote_message.sensed_pose_rel_body[1:4]

        cue_scores, _ = self._score_chart_cue_candidates(
            pose_vectors=vote_pose_vectors,
            appearance_vector=vote_appearance_signature,
            behavior_vector=vote_behavior_signature,
            behavior_label=vote_behavior_label,
            appearance_weight=self.vote_appearance_score_weight,
        )
        for rank, hypothesis in enumerate(vote_message.ranked_hypotheses, start=1):
            normalized_latent_id = _normalize_graph_id(hypothesis.latent_id)
            normalized_chart_id = _normalize_graph_id(hypothesis.chart_id)
            if normalized_latent_id is None or normalized_chart_id is None:
                continue
            if normalized_latent_id not in self._known_internal_latent_ids():
                continue
            key = (normalized_latent_id, normalized_chart_id)
            self._vote_chart_bias[key] = self._vote_chart_bias.get(key, 0.0) + (
                max(float(hypothesis.probability), 0.0) / float(max(rank, 1))
            )
        if not activity_scores and not cue_scores:
            return

        strength = 1.0
        if vote_message.ranked_hypotheses:
            strength = max(
                float(hypothesis.probability)
                for hypothesis in vote_message.ranked_hypotheses
            )
        for object_id, score in activity_scores.items():
            self.receive_vote_bias(object_id, float(score) * float(strength))
        cue_bias_scale = 1.0 / max(self.vote_bias_weight, 1e-6)
        for object_id, score in cue_scores.items():
            self.receive_vote_bias(
                object_id,
                float(score) * float(strength) * cue_bias_scale,
            )

    def receive_vote_bias(self, object_id: str, amount: float) -> None:
        normalized = _normalize_graph_id(object_id)
        if normalized is None or normalized not in self._known_internal_latent_ids():
            return
        self._vote_bias[normalized] = self._vote_bias.get(normalized, 0.0) + float(amount)

    def _prediction_target_latent_ids_from_state(self, state) -> list[str]:
        return list(self._predicted_latent_ids)

    def _compose_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        combined = embedding.clone()
        if self._external_context is not None:
            combined = combined + 0.2 * self._external_context

        norm = float(combined.norm(p=2).item())
        if norm > 1e-8:
            combined = combined / norm
        return combined

    def _build_action_embedding(self) -> torch.Tensor:
        action_context = torch.as_tensor(
            self._action_context,
            dtype=torch.float32,
            device=self.device,
        ).reshape(-1)
        if action_context.numel() == 0:
            return torch.zeros(
                self.context_dim,
                dtype=torch.float32,
                device=self.device,
            )

        action_embedding = _resize_vector(action_context, self.context_dim)
        norm = float(action_embedding.norm(p=2).item())
        if norm > 1e-8:
            action_embedding = action_embedding / norm
        return action_embedding

    def _predict_next_embedding(self, fallback_embedding: torch.Tensor) -> torch.Tensor:
        if not self._has_last_embedding:
            return fallback_embedding.detach().clone()

        predicted = self._temporal_state.context_vector.detach().clone()
        if float(predicted.abs().sum().item()) <= 1e-8:
            predicted = self._last_embedding.detach().clone()
        if self._external_context is not None:
            predicted = predicted + 0.15 * self._external_context

        action_embedding = self._build_action_embedding()
        if float(action_embedding.abs().sum().item()) > 1e-8:
            predicted = predicted + (self.action_context_weight * action_embedding)

        norm = float(predicted.norm(p=2).item())
        if norm > 1e-8:
            predicted = predicted / norm
        return predicted

    def _predict_next_stream_embedding(
        self,
        fallback_stream: torch.Tensor,
        *,
        previous_stream: torch.Tensor,
        target_dim: int,
        temporal_weight: float,
        action_weight: float,
        context_weight: float,
        external_prior: torch.Tensor | None = None,
        external_prior_weight: float = 0.0,
    ) -> torch.Tensor:
        if self._has_last_embedding and previous_stream.numel() > 0:
            predicted = _resize_vector(previous_stream.detach().clone(), target_dim)
        else:
            predicted = _resize_vector(fallback_stream.detach().clone(), target_dim)

        if float(self._temporal_state.context_vector.abs().sum().item()) > 1e-8:
            predicted = predicted + (
                float(temporal_weight)
                * _resize_vector(self._temporal_state.context_vector, target_dim)
            )

        if self._external_context is not None:
            predicted = predicted + (
                float(context_weight)
                * _resize_vector(self._external_context, target_dim)
            )

        if external_prior is not None and float(external_prior.abs().sum().item()) > 1e-8:
            predicted = predicted + (
                float(external_prior_weight)
                * _resize_vector(external_prior, target_dim)
            )

        action_embedding = self._build_action_embedding()
        if float(action_embedding.abs().sum().item()) > 1e-8:
            predicted = predicted + (
                float(action_weight)
                * _resize_vector(action_embedding, target_dim)
            )

        norm = float(predicted.norm(p=2).item())
        if norm > 1e-8:
            predicted = predicted / norm
        return predicted

    def _predict_next_stream_embeddings(
        self,
        embeddings: ObservationEmbeddings,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        predicted_appearance = self._predict_next_stream_embedding(
            embeddings.appearance,
            previous_stream=self._last_embeddings.appearance,
            target_dim=self.encoder.stream_dim,
            temporal_weight=0.18,
            action_weight=0.08,
            context_weight=0.05,
            external_prior=self._external_predicted_appearance_signature,
            external_prior_weight=0.22,
        )
        predicted_change = self._predict_next_stream_embedding(
            embeddings.temporal,
            previous_stream=self._last_embeddings.temporal,
            target_dim=self.encoder.stream_dim,
            temporal_weight=0.28,
            action_weight=0.18,
            context_weight=0.08,
            external_prior=self._external_predicted_change_signature,
            external_prior_weight=0.26,
        )
        return predicted_appearance, predicted_change

    def _compose_stream_query(
        self,
        sensory_seed: torch.Tensor,
        *,
        target_dim: int,
        temporal_weight: float,
        action_weight: float,
        context_weight: float,
        extra_bias: torch.Tensor | None = None,
        extra_weight: float = 0.0,
    ) -> torch.Tensor:
        combined = _resize_vector(sensory_seed.reshape(-1), target_dim)
        if extra_bias is not None and extra_weight > 0.0:
            combined = combined + (
                float(extra_weight) * _resize_vector(extra_bias, target_dim)
            )

        if float(self._temporal_state.context_vector.abs().sum().item()) > 1e-8:
            combined = combined + (
                float(temporal_weight)
                * _resize_vector(self._temporal_state.context_vector, target_dim)
            )

        if self._external_context is not None:
            combined = combined + (
                float(context_weight)
                * _resize_vector(self._external_context, target_dim)
            )

        if extra_bias is not None:
            action_embedding = self._build_action_embedding()
            if float(action_embedding.abs().sum().item()) > 1e-8:
                combined = combined + (
                    float(action_weight)
                    * _resize_vector(action_embedding, target_dim)
                )

        norm = float(combined.norm(p=2).item())
        if norm > 1e-8:
            combined = combined / norm
        return combined

    def _build_memory_queries(
        self,
        embeddings: ObservationEmbeddings,
        *,
        predicted_embedding: torch.Tensor | None = None,
        predicted_appearance_embedding: torch.Tensor | None = None,
        predicted_change_embedding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        appearance_seed = torch.cat(
            [embeddings.appearance, embeddings.geometry],
            dim=0,
        )
        change_seed = embeddings.temporal
        appearance_query = self._compose_stream_query(
            appearance_seed,
            target_dim=self.appearance_memory_dim,
            temporal_weight=0.10,
            action_weight=0.04,
            context_weight=0.06,
            extra_bias=(
                predicted_appearance_embedding
                if predicted_appearance_embedding is not None
                else predicted_embedding
            ),
            extra_weight=(
                0.26
                if predicted_appearance_embedding is not None
                else (0.18 if predicted_embedding is not None else 0.0)
            ),
        )
        change_query = self._compose_stream_query(
            change_seed,
            target_dim=self.change_memory_dim,
            temporal_weight=0.24,
            action_weight=0.16,
            context_weight=0.08,
            extra_bias=(
                predicted_change_embedding
                if predicted_change_embedding is not None
                else predicted_embedding
            ),
            extra_weight=(
                0.34
                if predicted_change_embedding is not None
                else (0.30 if predicted_embedding is not None else 0.0)
            ),
        )
        return appearance_query, change_query

    @staticmethod
    def _stream_packet_weights(packet_type: str, has_change_scores: bool) -> tuple[float, float]:
        if not has_change_scores:
            return 1.0, 0.0

        normalized_packet_type = str(packet_type or "unknown_packet")
        if normalized_packet_type == "change_observation_packet_v2":
            return 0.68, 0.32
        if normalized_packet_type == "lm_fusion_packet":
            return 0.82, 0.18
        return 0.92, 0.08

    def _combine_stream_scores(
        self,
        appearance_scores: dict[str, float],
        change_scores: dict[str, float],
        appearance_chart_ids: dict[str, str | None],
        change_chart_ids: dict[str, str | None],
        *,
        packet_type: str,
    ) -> tuple[dict[str, float], dict[str, str | None], dict[str, float]]:
        object_ids = set(appearance_scores.keys()) | set(change_scores.keys())
        if not object_ids:
            return {}, {}, {"appearance": 1.0, "change": 0.0}

        if appearance_scores and not change_scores:
            appearance_weight, change_weight = 1.0, 0.0
        elif change_scores and not appearance_scores:
            appearance_weight, change_weight = 0.0, 1.0
        else:
            appearance_weight, change_weight = self._stream_packet_weights(
                packet_type,
                has_change_scores=bool(change_scores),
            )

        combined_scores: dict[str, float] = {}
        chart_ids: dict[str, str | None] = {}
        for object_id in object_ids:
            appearance_score = float(
                appearance_scores.get(
                    object_id,
                    change_scores.get(object_id, 0.0),
                )
            )
            change_score = float(
                change_scores.get(
                    object_id,
                    appearance_scores.get(object_id, 0.0),
                )
            )
            combined_scores[object_id] = (
                (appearance_weight * appearance_score)
                + (change_weight * change_score)
            )
            chart_ids[object_id] = (
                appearance_chart_ids.get(object_id)
                or change_chart_ids.get(object_id)
            )

        return combined_scores, chart_ids, {
            "appearance": float(appearance_weight),
            "change": float(change_weight),
        }

    def _should_update_change_memory(
        self,
        packet: dict[str, Any],
        observation_field: ObservationField,
    ) -> bool:
        packet_type = str(packet.get("packet_type") or "unknown_packet")
        if packet_type in {
            "change_observation_packet_v2",
            "lm_fusion_packet",
            "fallback_state_packet",
        }:
            return True
        if observation_field.child_count > 0:
            return True
        if observation_field.temporal_feature_stats.numel() > 0:
            return True
        return float(observation_field.flow_magnitude.abs().sum().item()) > 1e-6

    def _compute_self_supervised_novelty(
        self,
        embedding: torch.Tensor,
        observation_field: ObservationField,
        *,
        action_prediction_error: float,
    ) -> tuple[float, float]:
        continuity_distance = (
            _cosine_distance(embedding, self._last_embedding)
            if self._has_last_embedding
            else 0.0
        )
        flow_novelty = (
            float(
                min(
                    1.0,
                    observation_field.flow_magnitude.abs().max().item(),
                )
            )
            if observation_field.flow_magnitude.numel() > 0
            else 0.0
        )
        temporal_novelty = (
            float(
                min(
                    1.0,
                    observation_field.temporal_feature_stats.abs().mean().item(),
                )
            )
            if observation_field.temporal_feature_stats.numel() > 0
            else 0.0
        )
        novelty_signal = float(
            max(
                continuity_distance,
                float(max(action_prediction_error, 0.0)),
                0.50 * flow_novelty,
                0.35 * temporal_novelty,
                min(1.0, self._last_boundary_pressure),
            )
        )
        return novelty_signal, continuity_distance

    def _summarize_behavior_state(
        self,
        observation_field: ObservationField,
        embeddings: ObservationEmbeddings,
        *,
        packet_type: str,
    ) -> tuple[np.ndarray, str]:
        flow_magnitude = (
            float(observation_field.flow_magnitude.abs().mean().item())
            if observation_field.flow_magnitude.numel() > 0
            else 0.0
        )
        temporal_energy = (
            float(embeddings.temporal.abs().mean().item())
            if embeddings.temporal.numel() > 0
            else 0.0
        )
        temporal_stats = np.zeros(3, dtype=np.float32)
        if observation_field.temporal_feature_stats.numel() > 0:
            temporal_stats = (
                observation_field.temporal_feature_stats.abs()
                .mean(dim=0)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

        child_count_norm = min(1.0, float(observation_field.child_count) / 4.0)
        boundary_pressure = float(
            min(1.0, max(0.0, self._temporal_state.boundary_pressure_value))
        )
        is_change_packet = 1.0 if packet_type == "change_observation_packet_v2" else 0.0
        is_fusion_packet = 1.0 if packet_type == "lm_fusion_packet" else 0.0
        behavior_vector = np.array(
            [
                flow_magnitude,
                temporal_energy,
                float(temporal_stats[0]),
                float(temporal_stats[1]),
                float(temporal_stats[2]),
                child_count_norm,
                boundary_pressure,
                is_change_packet,
                is_fusion_packet,
            ],
            dtype=np.float32,
        )

        if is_fusion_packet > 0.0 or observation_field.child_count > 0:
            behavior_label = "hierarchy_fusion"
        elif is_change_packet > 0.0 or flow_magnitude >= 0.12 or temporal_energy >= 0.20:
            behavior_label = "temporal_change"
        else:
            behavior_label = "stable_surface"

        return behavior_vector, behavior_label

    def _summarize_appearance_state(
        self,
        observation_field: ObservationField,
        embeddings: ObservationEmbeddings,
    ) -> np.ndarray:
        del observation_field
        if embeddings.appearance.numel() == 0:
            return np.zeros(0, dtype=np.float32)
        return (
            embeddings.appearance.detach().cpu().numpy().astype(np.float32).reshape(-1)
        )

    def _allocate_self_supervised_latent_id(self) -> str:
        latent_id = f"latent_object_{self._latent_object_counter}"
        self._latent_object_counter += 1
        return latent_id

    def _build_learning_novel_candidate(
        self,
        *,
        novelty_signal: float,
        continuity_distance: float,
        best_existing_score: float,
        has_active_candidate: bool,
        location: np.ndarray,
        pose_vectors: np.ndarray,
        appearance_vector: np.ndarray,
        behavior_vector: np.ndarray,
        behavior_label: str | None,
        inferred_state: int | None,
    ) -> dict[str, Any] | None:
        boundary_pressure = float(
            min(1.0, max(0.0, self._last_boundary_pressure))
        )
        novelty_pressure = float(
            max(
                novelty_signal,
                continuity_distance,
                boundary_pressure,
            )
        )
        novelty_margin = float(
            max(
                0.0,
                novelty_pressure - self.self_supervised_object_reset_distance,
            )
        )
        support_gap = float(
            max(
                0.0,
                self.self_supervised_memory_reuse_threshold - best_existing_score,
            )
        )
        if (
            self._episode_has_anchor
            and self._active_self_supervised_latent_id is not None
            and has_active_candidate
        ):
            return None
        anchor_bootstrap = (
            self._episode_has_anchor
            and self._active_self_supervised_latent_id is None
        )
        cold_start = best_existing_score <= 0.0
        if not (
            cold_start
            or support_gap > 0.0
            or novelty_margin > 0.0
            or anchor_bootstrap
        ):
            return None

        anchor_bonus = 0.08 if anchor_bootstrap else 0.0
        baseline_score = (
            0.52
            + (0.28 * novelty_pressure)
            + (0.30 * support_gap)
            + (0.14 * boundary_pressure)
            + anchor_bonus
        )
        if cold_start:
            relative_score = self.self_supervised_memory_reuse_threshold + 0.05 + anchor_bonus
        else:
            relative_score = (
                best_existing_score
                - 0.03
                + (0.18 * novelty_margin)
                + (0.28 * support_gap)
                + (0.12 * boundary_pressure)
                + anchor_bonus
            )
        if anchor_bootstrap:
            relative_score = max(
                relative_score,
                best_existing_score + 0.25,
                self.self_supervised_memory_reuse_threshold + 0.10,
            )
        novel_score = float(
            np.clip(
                max(baseline_score, relative_score),
                0.0,
                1.35,
            )
        )
        return {
            "object_id": "novel_candidate",
            "latent_id": "novel_candidate",
            "chart_id": None,
            "score": novel_score,
            "base_score": novel_score,
            "joint_delta": 0.0,
            "pose_support": 0.0,
            "appearance_support": 0.0,
            "behavior_similarity": 0.0,
            "label_match": 0.0,
            "behavior_support": 0.0,
            "context_appearance_support": 0.0,
            "context_behavior_support": 0.0,
            "context_ranked_chart_prior": 0.0,
            "vote_ranked_chart_prior": 0.0,
            "selected": False,
            "location": np.asarray(location, dtype=np.float32).reshape(3),
            "pose_vectors": np.asarray(pose_vectors, dtype=np.float32).reshape(3, 3),
            "appearance_vector": _as_float32_vector(appearance_vector),
            "behavior_vector": _as_float32_vector(behavior_vector),
            "behavior_label": behavior_label,
            "inferred_state": inferred_state,
            "source": "novel",
            "age": 0,
        }

    @staticmethod
    def _posterior_candidate_sort_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
        return (
            float(candidate.get("posterior_probability", 0.0)),
            float(candidate.get("score", 0.0)),
            1 if str(candidate.get("source", "")) != "novel" else 0,
            -int(candidate.get("age", 0)),
            str(candidate.get("object_id", "")),
            "" if candidate.get("chart_id") is None else str(candidate.get("chart_id")),
        )

    @staticmethod
    def _candidate_selection_sort_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
        return (
            float(candidate.get("score", 0.0)),
            1 if str(candidate.get("source", "")) != "novel" else 0,
            -int(candidate.get("age", 0)),
            str(candidate.get("object_id", "")),
            "" if candidate.get("chart_id") is None else str(candidate.get("chart_id")),
        )

    @staticmethod
    def _candidate_overwhelms(
        preferred: dict[str, Any] | None,
        alternative: dict[str, Any] | None,
        *,
        score_margin: float,
        posterior_ratio: float,
    ) -> bool:
        if preferred is None or alternative is None:
            return False
        preferred_score = float(preferred.get("score", 0.0))
        alternative_score = float(alternative.get("score", 0.0))
        if preferred_score >= alternative_score + float(score_margin):
            return True
        preferred_posterior = float(preferred.get("posterior_probability", 0.0))
        alternative_posterior = float(alternative.get("posterior_probability", 0.0))
        if alternative_posterior <= 1e-8:
            return preferred_posterior > alternative_posterior
        return preferred_posterior >= (
            alternative_posterior * float(max(posterior_ratio, 1.0))
        )

    def _learning_selection_score_floor(
        self,
        learning_support: dict[str, Any],
    ) -> float:
        score_floor = 1.0
        if not self._episode_has_anchor:
            return score_floor

        selected_score = float(
            max(learning_support.get("selected_candidate_score", 0.0), 0.0)
        )
        posterior = float(
            min(
                max(learning_support.get("selected_posterior_probability", 0.0), 0.0),
                1.0,
            )
        )
        selection_reason = str(
            learning_support.get(
                "selection_reason",
                learning_support.get("allocation_reason", ""),
            )
        )
        selected_source = str(learning_support.get("selected_source", "memory"))

        score_floor = max(score_floor, selected_score)
        anchor_bonus = 0.03 + (0.12 * posterior)
        if selection_reason == "posterior_anchor_bootstrap_novel":
            anchor_bonus += 0.08
        elif selection_reason == "posterior_anchor_continuity":
            anchor_bonus += 0.05
        if selected_source == "novel":
            anchor_bonus += 0.02
        return score_floor + anchor_bonus

    def _carry_learning_selection_into_scores(
        self,
        *,
        scores: dict[str, float],
        chart_ids: dict[str, str | None],
        learning_latent_id: str | None,
        learning_chart_id: str | None,
        learning_support: dict[str, Any],
    ) -> tuple[dict[str, float], dict[str, str | None], float]:
        adjusted_scores = {
            str(object_id): float(score)
            for object_id, score in scores.items()
        }
        adjusted_chart_ids = {
            str(object_id): chart_id
            for object_id, chart_id in chart_ids.items()
        }
        if learning_latent_id is None:
            return adjusted_scores, adjusted_chart_ids, 0.0

        score_floor = self._learning_selection_score_floor(learning_support)
        adjusted_scores[learning_latent_id] = max(
            adjusted_scores.get(learning_latent_id, 0.0),
            float(score_floor),
        )
        if learning_chart_id is not None:
            adjusted_chart_ids[learning_latent_id] = learning_chart_id
        return adjusted_scores, adjusted_chart_ids, float(score_floor)

    @staticmethod
    def _should_carry_learning_selection(packet_type: str) -> bool:
        return str(packet_type or "unknown_packet") == "visual_observation_packet_v2"

    def _anchor_adjust_learning_selection(
        self,
        *,
        posterior_candidates: list[dict[str, Any]],
        selected_candidate: dict[str, Any],
    ) -> tuple[dict[str, Any], str | None]:
        if not self._episode_has_anchor or not posterior_candidates:
            return selected_candidate, None

        active_latent_id = self._active_self_supervised_latent_id
        if active_latent_id is None:
            novel_candidates = [
                candidate
                for candidate in posterior_candidates
                if str(candidate.get("source", "")) == "novel"
            ]
            if novel_candidates:
                best_novel_candidate = max(
                    novel_candidates,
                    key=self._posterior_candidate_sort_key,
                )
                if (
                    selected_candidate is best_novel_candidate
                    and float(
                        best_novel_candidate.get("posterior_probability", 0.0)
                    )
                    >= self.anchor_bootstrap_novel_min_posterior
                ):
                    return selected_candidate, "posterior_anchor_bootstrap_novel"
                if (
                    selected_candidate is not best_novel_candidate
                    and float(
                        best_novel_candidate.get("posterior_probability", 0.0)
                    )
                    >= self.anchor_bootstrap_novel_min_posterior
                    and not self._candidate_overwhelms(
                        selected_candidate,
                        best_novel_candidate,
                        score_margin=self.anchor_bootstrap_existing_overwhelm_margin,
                        posterior_ratio=self.anchor_bootstrap_existing_overwhelm_ratio,
                    )
                ):
                    return best_novel_candidate, "posterior_anchor_bootstrap_novel"
            return selected_candidate, None

        active_candidates = [
            candidate
            for candidate in posterior_candidates
            if str(candidate.get("object_id", candidate.get("latent_id")))
            == str(active_latent_id)
        ]
        if not active_candidates:
            return selected_candidate, None
        best_active_candidate = max(
            active_candidates,
            key=self._posterior_candidate_sort_key,
        )
        if best_active_candidate is selected_candidate:
            return selected_candidate, None
        if self._candidate_overwhelms(
            selected_candidate,
            best_active_candidate,
            score_margin=self.anchor_continuity_switch_margin,
            posterior_ratio=self.anchor_continuity_switch_ratio,
        ):
            return selected_candidate, None
        return best_active_candidate, "posterior_anchor_continuity"

    def _select_learning_hypothesis_from_posterior(
        self,
        *,
        learn: bool,
        embedding: torch.Tensor,
        observation_field: ObservationField,
        action_prediction_error: float,
        joint_candidates: list[dict[str, Any]],
        location: np.ndarray,
        pose_vectors: np.ndarray,
        appearance_vector: np.ndarray,
        behavior_vector: np.ndarray,
        behavior_label: str | None,
        inferred_state: int | None,
    ) -> tuple[str | None, str | None, dict[str, Any]]:
        if not learn:
            return None, None, {
                "learning_mode": "disabled",
                "learning_latent_id": None,
                "learning_object_id": None,
            }

        novelty_signal, continuity_distance = self._compute_self_supervised_novelty(
            embedding,
            observation_field,
            action_prediction_error=action_prediction_error,
        )
        normalized_candidates: list[dict[str, Any]] = []
        for candidate in joint_candidates:
            if not isinstance(candidate, dict):
                continue
            latent_id = _normalize_graph_id(
                candidate.get("object_id", candidate.get("latent_id"))
            )
            if latent_id is None:
                continue
            normalized_candidates.append(
                {
                    "object_id": latent_id,
                    "latent_id": latent_id,
                    "chart_id": _normalize_graph_id(candidate.get("chart_id")),
                    "score": float(candidate.get("score", 0.0)),
                    "base_score": float(
                        candidate.get("base_score", candidate.get("score", 0.0))
                    ),
                    "joint_delta": float(candidate.get("joint_delta", 0.0)),
                    "pose_support": float(candidate.get("pose_support", 0.0)),
                    "appearance_support": float(candidate.get("appearance_support", 0.0)),
                    "behavior_similarity": float(
                        candidate.get("behavior_similarity", 0.0)
                    ),
                    "label_match": float(candidate.get("label_match", 0.0)),
                    "behavior_support": float(candidate.get("behavior_support", 0.0)),
                    "context_appearance_support": float(
                        candidate.get("context_appearance_support", 0.0)
                    ),
                    "context_behavior_support": float(
                        candidate.get("context_behavior_support", 0.0)
                    ),
                    "context_ranked_chart_prior": float(
                        candidate.get("context_ranked_chart_prior", 0.0)
                    ),
                    "vote_ranked_chart_prior": float(
                        candidate.get("vote_ranked_chart_prior", 0.0)
                    ),
                    "selected": False,
                    "location": np.asarray(
                        candidate.get("location", location),
                        dtype=np.float32,
                    ).reshape(3),
                    "pose_vectors": np.asarray(
                        candidate.get("pose_vectors", pose_vectors),
                        dtype=np.float32,
                    ).reshape(3, 3),
                    "appearance_vector": _as_float32_vector(
                        candidate.get("appearance_vector", appearance_vector)
                    ),
                    "behavior_vector": _as_float32_vector(
                        candidate.get("behavior_vector", behavior_vector)
                    ),
                    "behavior_label": candidate.get("behavior_label", behavior_label),
                    "inferred_state": candidate.get("inferred_state", inferred_state),
                    "source": str(candidate.get("source", "memory")),
                    "age": max(int(candidate.get("age", 0)), 0),
                }
            )

        best_existing_candidate = (
            max(
                normalized_candidates,
                key=lambda candidate: (
                    float(candidate.get("score", 0.0)),
                    1 if str(candidate.get("source", "")) != "novel" else 0,
                    -int(candidate.get("age", 0)),
                    str(candidate.get("object_id", "")),
                    "" if candidate.get("chart_id") is None else str(candidate.get("chart_id")),
                ),
            )
            if normalized_candidates
            else None
        )
        best_existing_score = float(
            0.0
            if best_existing_candidate is None
            else best_existing_candidate.get("score", 0.0)
        )
        has_active_candidate = any(
            str(candidate.get("object_id")) == self._active_self_supervised_latent_id
            for candidate in normalized_candidates
        )
        novel_candidate = self._build_learning_novel_candidate(
            novelty_signal=novelty_signal,
            continuity_distance=continuity_distance,
            best_existing_score=best_existing_score,
            has_active_candidate=has_active_candidate,
            location=location,
            pose_vectors=pose_vectors,
            appearance_vector=appearance_vector,
            behavior_vector=behavior_vector,
            behavior_label=behavior_label,
            inferred_state=inferred_state,
        )
        posterior_candidates = [dict(candidate) for candidate in normalized_candidates]
        if novel_candidate is not None:
            posterior_candidates.append(dict(novel_candidate))

        if not posterior_candidates:
            posterior_candidates.append(
                self._build_learning_novel_candidate(
                    novelty_signal=max(
                        novelty_signal,
                        self.self_supervised_object_reset_distance,
                    ),
                    continuity_distance=continuity_distance,
                    best_existing_score=0.0,
                    has_active_candidate=False,
                    location=location,
                    pose_vectors=pose_vectors,
                    appearance_vector=appearance_vector,
                    behavior_vector=behavior_vector,
                    behavior_label=behavior_label,
                    inferred_state=inferred_state,
                )
                or {
                    "object_id": "novel_candidate",
                    "latent_id": "novel_candidate",
                    "chart_id": None,
                    "score": float(self.self_supervised_memory_reuse_threshold + 0.05),
                    "base_score": float(self.self_supervised_memory_reuse_threshold + 0.05),
                    "joint_delta": 0.0,
                    "pose_support": 0.0,
                    "appearance_support": 0.0,
                    "behavior_similarity": 0.0,
                    "label_match": 0.0,
                    "behavior_support": 0.0,
                    "context_appearance_support": 0.0,
                    "context_behavior_support": 0.0,
                    "context_ranked_chart_prior": 0.0,
                    "vote_ranked_chart_prior": 0.0,
                    "selected": False,
                    "location": np.asarray(location, dtype=np.float32).reshape(3),
                    "pose_vectors": np.asarray(pose_vectors, dtype=np.float32).reshape(3, 3),
                    "appearance_vector": _as_float32_vector(appearance_vector),
                    "behavior_vector": _as_float32_vector(behavior_vector),
                    "behavior_label": behavior_label,
                    "inferred_state": inferred_state,
                    "source": "novel",
                    "age": 0,
                }
            )

        score_tensor = torch.as_tensor(
            [float(candidate.get("score", 0.0)) for candidate in posterior_candidates],
            dtype=torch.float32,
            device=self.device,
        )
        posterior_probabilities = torch.softmax(
            score_tensor / self.hypotheses.temperature,
            dim=0,
        )
        posterior_by_latent: dict[str, float] = {}
        for index, candidate in enumerate(posterior_candidates):
            probability = float(posterior_probabilities[index].item())
            candidate["posterior_probability"] = probability
            candidate["selected"] = False
            latent_key = str(candidate.get("object_id", "novel_candidate"))
            posterior_by_latent[latent_key] = (
                posterior_by_latent.get(latent_key, 0.0) + probability
            )

        selected_candidate = max(
            posterior_candidates,
            key=self._posterior_candidate_sort_key,
        )
        anchor_selection_reason = None
        selected_candidate, anchor_selection_reason = self._anchor_adjust_learning_selection(
            posterior_candidates=posterior_candidates,
            selected_candidate=selected_candidate,
        )
        selected_source = str(selected_candidate.get("source", "memory"))
        if selected_source == "novel":
            learning_latent_id = self._allocate_self_supervised_latent_id()
            learning_chart_id = None
            selection_reason = (
                anchor_selection_reason or "posterior_novel"
            )
        else:
            learning_latent_id = str(
                selected_candidate.get("object_id", selected_candidate.get("latent_id"))
            )
            learning_chart_id = _normalize_graph_id(selected_candidate.get("chart_id"))
            selection_reason = (
                anchor_selection_reason or "posterior_existing"
            )

        self._active_self_supervised_latent_id = learning_latent_id
        selected_candidate["selected"] = True
        best_existing_reuse_latent_id = None
        if (
            best_existing_candidate is not None
            and best_existing_score >= self.self_supervised_memory_reuse_threshold
        ):
            best_existing_reuse_latent_id = str(best_existing_candidate.get("object_id"))
        ranked_posterior_candidates = sorted(
            posterior_candidates,
            key=lambda candidate: (
                -float(candidate.get("posterior_probability", 0.0)),
                -float(candidate.get("score", 0.0)),
                0 if str(candidate.get("source", "")) == "novel" else 1,
                str(candidate.get("object_id", "")),
                "" if candidate.get("chart_id") is None else str(candidate.get("chart_id")),
            ),
        )
        return learning_latent_id, learning_chart_id, {
            "learning_mode": "self_supervised",
            "selection_mode": "posterior_bank_competition",
            "learning_latent_id": learning_latent_id,
            "learning_object_id": learning_latent_id,
            "learning_chart_id": learning_chart_id,
            "memory_candidate_latent_id": best_existing_reuse_latent_id,
            "memory_candidate_object_id": best_existing_reuse_latent_id,
            "memory_candidate_score": float(best_existing_score),
            "best_existing_latent_id": (
                None
                if best_existing_candidate is None
                else str(best_existing_candidate.get("object_id"))
            ),
            "best_existing_chart_id": (
                None
                if best_existing_candidate is None
                else best_existing_candidate.get("chart_id")
            ),
            "best_existing_score": float(best_existing_score),
            "allocation_reason": selection_reason,
            "selection_reason": selection_reason,
            "selected_source": selected_source,
            "selected_chart_id": learning_chart_id,
            "selected_posterior_probability": float(
                selected_candidate.get("posterior_probability", 0.0)
            ),
            "selected_candidate_score": float(selected_candidate.get("score", 0.0)),
            "continuity_distance": float(continuity_distance),
            "novelty_signal": float(novelty_signal),
            "previous_boundary_pressure": float(self._last_boundary_pressure),
            "episode_anchor_active": bool(self._episode_has_anchor),
            "novel_candidate_considered": bool(novel_candidate is not None),
            "novel_candidate_score": float(
                0.0 if novel_candidate is None else novel_candidate.get("score", 0.0)
            ),
            "candidate_count": int(len(posterior_candidates)),
            "posterior_latent_probabilities": {
                str(latent_id): float(probability)
                for latent_id, probability in sorted(posterior_by_latent.items())
            },
            "posterior_candidates": [
                {
                    "latent_id": str(
                        learning_latent_id
                        if candidate is selected_candidate and selected_source == "novel"
                        else candidate.get("object_id", candidate.get("latent_id"))
                    ),
                    "object_id": str(
                        learning_latent_id
                        if candidate is selected_candidate and selected_source == "novel"
                        else candidate.get("object_id", candidate.get("latent_id"))
                    ),
                    "chart_id": candidate.get("chart_id"),
                    "score": float(candidate.get("score", 0.0)),
                    "posterior_probability": float(
                        candidate.get("posterior_probability", 0.0)
                    ),
                    "source": candidate.get("source"),
                    "age": max(int(candidate.get("age", 0)), 0),
                    "selected": bool(candidate is selected_candidate),
                }
                for candidate in ranked_posterior_candidates[:5]
            ],
        }

    def _apply_chart_transition_support(
        self,
        scores: dict[str, float],
        chart_ids: dict[str, str | None],
        *,
        joint_candidates: list[dict[str, Any]] | None = None,
    ) -> tuple[
        dict[str, float],
        dict[str, float],
        dict[str, Any],
        list[dict[str, Any]],
        dict[str, str | None],
    ]:
        adjusted_scores = {
            str(object_id): float(score)
            for object_id, score in scores.items()
        }
        adjustments: dict[str, float] = {}
        adjusted_joint_candidates = [dict(candidate) for candidate in (joint_candidates or [])]
        previous_chart_id = self._last_winning_chart_id
        previous_latent_id = self._last_winning_latent_id
        if previous_chart_id is None:
            return (
                adjusted_scores,
                adjustments,
                {
                    "previous_chart_id": None,
                    "previous_latent_id": previous_latent_id,
                    "previous_object_id": previous_latent_id,
                    "transition_support": {},
                    "known_next_charts": {},
                },
                adjusted_joint_candidates,
                dict(chart_ids),
            )

        next_counts = {
            str(chart_id): float(count)
            for chart_id, count in self._chart_transition_counts.get(
                previous_chart_id,
                {},
            ).items()
            if float(count) > 0.0
        }
        total = float(sum(next_counts.values()))
        transition_support: dict[str, Any] = {}
        if adjusted_joint_candidates:
            candidate_supports: dict[str, list[dict[str, Any]]] = {}
            resolved_chart_ids = dict(chart_ids)
            best_candidate_scores = {
                str(object_id): float(score)
                for object_id, score in adjusted_scores.items()
            }
            for candidate in adjusted_joint_candidates:
                latent_id = str(candidate.get("object_id", candidate.get("latent_id", "unknown")))
                chart_id = candidate.get("chart_id")
                support = 0.0
                if chart_id is not None:
                    if total > 0.0:
                        support = float(next_counts.get(str(chart_id), 0.0)) / total
                    elif chart_id == previous_chart_id and latent_id == previous_latent_id:
                        support = 0.25
                    elif chart_id == previous_chart_id:
                        support = 0.10

                if support > 0.0:
                    candidate["score"] = float(candidate.get("score", adjusted_scores.get(latent_id, 0.0))) + (
                        self.transition_score_weight * support
                    )
                    candidate_supports.setdefault(latent_id, []).append(
                        {
                            "chart_id": chart_id,
                            "support": float(support),
                            "score": float(candidate["score"]),
                        }
                    )

                candidate_score = float(
                    candidate.get("score", adjusted_scores.get(latent_id, 0.0))
                )
                current_best = best_candidate_scores.get(latent_id, adjusted_scores.get(latent_id, 0.0))
                current_chart_id = resolved_chart_ids.get(latent_id)
                if (
                    candidate_score > float(current_best)
                    or (
                        candidate_score == float(current_best)
                        and (
                            "" if chart_id is None else str(chart_id)
                        )
                        < (
                            "" if current_chart_id is None else str(current_chart_id)
                        )
                    )
                ):
                    best_candidate_scores[latent_id] = candidate_score
                    resolved_chart_ids[latent_id] = None if chart_id is None else str(chart_id)

            for latent_id, updated_score in best_candidate_scores.items():
                adjusted_scores[latent_id] = float(updated_score)
                delta = float(updated_score) - float(scores.get(latent_id, 0.0))
                if delta > 0.0:
                    adjustments[latent_id] = delta

            for latent_id, entries in candidate_supports.items():
                best_chart_id = resolved_chart_ids.get(latent_id)
                best_entry = None
                for entry in entries:
                    if entry.get("chart_id") == best_chart_id:
                        best_entry = entry
                        break
                if best_entry is None:
                    best_entry = max(
                        entries,
                        key=lambda item: (
                            float(item.get("score", 0.0)),
                            float(item.get("support", 0.0)),
                            "" if item.get("chart_id") is None else str(item.get("chart_id")),
                        ),
                    )
                transition_support[latent_id] = {
                    "chart_id": best_entry.get("chart_id"),
                    "support": float(best_entry.get("support", 0.0)),
                    "candidate_supports": [dict(entry) for entry in entries],
                }

            return (
                adjusted_scores,
                adjustments,
                {
                    "previous_chart_id": previous_chart_id,
                    "previous_latent_id": previous_latent_id,
                    "previous_object_id": previous_latent_id,
                    "transition_support": transition_support,
                    "known_next_charts": next_counts,
                },
                adjusted_joint_candidates,
                resolved_chart_ids,
            )

        for latent_id, chart_id in chart_ids.items():
            if chart_id is None:
                continue

            support = 0.0
            if total > 0.0:
                support = float(next_counts.get(chart_id, 0.0)) / total
            elif chart_id == previous_chart_id and latent_id == previous_latent_id:
                support = 0.25
            elif chart_id == previous_chart_id:
                support = 0.10

            if support <= 0.0:
                continue

            delta = self.transition_score_weight * support
            adjusted_scores[latent_id] = adjusted_scores.get(latent_id, 0.0) + delta
            adjustments[latent_id] = adjustments.get(latent_id, 0.0) + delta
            transition_support[latent_id] = {
                "chart_id": chart_id,
                "support": float(support),
            }

        return (
            adjusted_scores,
            adjustments,
            {
                "previous_chart_id": previous_chart_id,
                "previous_latent_id": previous_latent_id,
                "previous_object_id": previous_latent_id,
                "transition_support": transition_support,
                "known_next_charts": next_counts,
            },
            adjusted_joint_candidates,
            dict(chart_ids),
        )

    def _record_chart_transition(
        self,
        *,
        learning_latent_id: str | None,
        chart_id: str | None,
        confidence: float,
    ) -> None:
        if learning_latent_id is None or chart_id is None:
            self._last_learning_latent_id = learning_latent_id
            self._last_learning_chart_id = chart_id
            return

        if (
            self._last_learning_latent_id == learning_latent_id
            and self._last_learning_chart_id is not None
        ):
            next_counts = self._chart_transition_counts.setdefault(
                self._last_learning_chart_id,
                {},
            )
            increment = max(0.25, min(1.0, float(confidence)))
            next_counts[chart_id] = float(next_counts.get(chart_id, 0.0)) + increment

        self._last_learning_latent_id = learning_latent_id
        self._last_learning_chart_id = chart_id

    def _blend_temporal_behavior_scores(
        self,
        base_scores: dict[str, float],
        embeddings: ObservationEmbeddings,
        predicted_embedding: torch.Tensor,
        predicted_appearance_signature: torch.Tensor,
        predicted_change_signature: torch.Tensor,
        appearance_prediction_error: float,
        change_prediction_error: float,
        *,
        packet_type: str,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        has_external_appearance_prior = (
            float(self._external_predicted_appearance_signature.abs().sum().item())
            > 1e-8
        )
        has_external_change_prior = (
            float(self._external_predicted_change_signature.abs().sum().item())
            > 1e-8
        )
        use_stream_specific_appearance = has_external_appearance_prior
        use_stream_specific_change = has_external_change_prior
        predicted_appearance_query, predicted_change_query = self._build_memory_queries(
            embeddings,
            predicted_embedding=predicted_embedding,
            predicted_appearance_embedding=(
                predicted_appearance_signature
                if use_stream_specific_appearance
                else None
            ),
            predicted_change_embedding=(
                predicted_change_signature if use_stream_specific_change else None
            ),
        )
        predicted_appearance_retrieval = self.appearance_memory.retrieve(
            predicted_appearance_query,
            store_as_last=False,
        )
        predicted_change_retrieval = self.change_memory.retrieve(
            predicted_change_query,
            store_as_last=False,
        )
        predicted_appearance_scores = self.appearance_memory.retrieval_scores(
            predicted_appearance_retrieval
        )
        predicted_change_scores = self.change_memory.retrieval_scores(
            predicted_change_retrieval
        )
        predicted_scores, _, _ = self._combine_stream_scores(
            predicted_appearance_scores,
            predicted_change_scores,
            self.appearance_memory.retrieval_chart_ids(predicted_appearance_retrieval),
            self.change_memory.retrieval_chart_ids(predicted_change_retrieval),
            packet_type=packet_type,
        )
        prediction_quality = max(
            0.0,
            1.0 - max(float(appearance_prediction_error), float(change_prediction_error)),
        )
        if has_external_appearance_prior or has_external_change_prior:
            effective_temporal_weight = self.temporal_score_weight
        else:
            effective_temporal_weight = self.temporal_score_weight * prediction_quality
            normalized_packet_type = str(packet_type or "unknown_packet")
            if normalized_packet_type in {
                "change_observation_packet_v2",
                "lm_fusion_packet",
            } and prediction_quality >= 0.20:
                effective_temporal_weight = max(effective_temporal_weight, 0.10)
        if not predicted_scores or effective_temporal_weight <= 0.0:
            return dict(base_scores), {}, dict(predicted_scores)

        adjusted_scores: dict[str, float] = {}
        adjustments: dict[str, float] = {}
        object_ids = set(base_scores.keys()) | set(predicted_scores.keys())
        for object_id in object_ids:
            base_score = float(base_scores.get(object_id, 0.0))
            predicted_score = float(predicted_scores.get(object_id, base_score))
            updated_score = ((1.0 - effective_temporal_weight) * base_score) + (
                effective_temporal_weight * predicted_score
            )
            adjusted_scores[object_id] = updated_score
            adjustments[object_id] = updated_score - base_score

        return adjusted_scores, adjustments, predicted_scores

    def _apply_biases(
        self,
        scores: dict[str, float],
        predicted_ids: list[str],
    ) -> tuple[dict[str, float], dict[str, float]]:
        adjusted = {str(key): float(value) for key, value in scores.items()}
        adjustments: dict[str, float] = {}

        for rank, object_id in enumerate(predicted_ids, start=1):
            delta = self.prediction_bias_weight / float(rank)
            adjusted[object_id] = adjusted.get(object_id, 0.0) + delta
            adjustments[object_id] = adjustments.get(object_id, 0.0) + delta

        for object_id, bias in list(self._vote_bias.items()):
            delta = self.vote_bias_weight * float(bias)
            adjusted[object_id] = adjusted.get(object_id, 0.0) + delta
            adjustments[object_id] = adjustments.get(object_id, 0.0) + delta

        return adjusted, adjustments

    def _update_temporal_state(
        self,
        embedding: torch.Tensor,
        predicted_embedding: torch.Tensor,
        predicted_appearance_signature: torch.Tensor,
        predicted_change_signature: torch.Tensor,
        *,
        top_latent_id: str | None,
        prediction_mismatch: float,
        action_prediction_error: float,
        appearance_prediction_error: float,
        change_prediction_error: float,
        threshold: float = 0.6,
    ) -> None:
        instant_surprise = float(
            max(
                prediction_mismatch,
                action_prediction_error,
                0.5 * appearance_prediction_error,
                0.5 * change_prediction_error,
            )
        )
        previous_mean_surprise = self._temporal_state.mean_surprise_value
        mean_surprise = (0.8 * previous_mean_surprise) + (0.2 * instant_surprise)

        previous_latent_id = self._temporal_state.current_latent_id
        latent_transition = (
            top_latent_id is not None
            and previous_latent_id is not None
            and top_latent_id != previous_latent_id
        )
        event_detected = bool(latent_transition or instant_surprise >= threshold)

        trace_count = int(self._temporal_state.trace_bank.shape[0])
        if event_detected or trace_count == 0:
            trace_bank = embedding.unsqueeze(0).repeat(max(trace_count, 1), 1)
        else:
            decay = self._temporal_state.trace_decay_rates.reshape(-1, 1)
            trace_bank = (decay * self._temporal_state.trace_bank) + (
                (1.0 - decay) * embedding.unsqueeze(0)
            )

        trace_mean = trace_bank.mean(dim=0)
        context_vector = embedding + (0.5 * trace_mean)
        norm = float(context_vector.norm(p=2).item())
        if norm > 1e-8:
            context_vector = context_vector / norm

        if top_latent_id is None:
            dwell_steps = 0
            current_latent_id = previous_latent_id
        elif not event_detected and top_latent_id == previous_latent_id:
            dwell_steps = self._temporal_state.dwell_step_count + 1
            current_latent_id = top_latent_id
        else:
            dwell_steps = 1
            current_latent_id = top_latent_id

        self._temporal_state = TemporalState(
            trace_decay_rates=self._temporal_state.trace_decay_rates.detach().clone(),
            trace_bank=trace_bank.detach().clone(),
            context_vector=context_vector.detach().clone(),
            latest_embedding=embedding.detach().clone(),
            predicted_embedding=predicted_embedding.detach().clone(),
            predicted_appearance_signature=predicted_appearance_signature.detach().clone(),
            predicted_change_signature=predicted_change_signature.detach().clone(),
            mean_surprise=torch.as_tensor(
                [mean_surprise],
                dtype=torch.float32,
                device=self.device,
            ),
            boundary_pressure=torch.as_tensor(
                [instant_surprise],
                dtype=torch.float32,
                device=self.device,
            ),
            prediction_mismatch=torch.as_tensor(
                [prediction_mismatch],
                dtype=torch.float32,
                device=self.device,
            ),
            action_prediction_error=torch.as_tensor(
                [action_prediction_error],
                dtype=torch.float32,
                device=self.device,
            ),
            appearance_prediction_error=torch.as_tensor(
                [appearance_prediction_error],
                dtype=torch.float32,
                device=self.device,
            ),
            change_prediction_error=torch.as_tensor(
                [change_prediction_error],
                dtype=torch.float32,
                device=self.device,
            ),
            dwell_steps=torch.as_tensor(
                [dwell_steps],
                dtype=torch.int64,
                device=self.device,
            ),
            event_detected=torch.as_tensor(
                [1.0 if event_detected else 0.0],
                dtype=torch.float32,
                device=self.device,
            ),
            current_latent_id=current_latent_id,
            known_latent_ids=list(self.hypotheses.state.latent_ids),
        )

    def step(self, state, learn: bool = False) -> dict[str, Any]:
        packet, observation_field, embeddings = self.encoder.encode_state_to_tensors(state)
        pose_vectors = _normalize_pose_matrix(
            getattr(state, "morphological_features", {}).get(
                "pose_vectors",
                np.eye(3, dtype=np.float64),
            )
        ).astype(np.float64)
        location = np.asarray(getattr(state, "location", np.zeros(3)), dtype=np.float64)
        inferred_state = getattr(state, "inferred_state", None)
        appearance_vector = self._summarize_appearance_state(
            observation_field,
            embeddings,
        )
        behavior_vector, behavior_label = self._summarize_behavior_state(
            observation_field,
            embeddings,
            packet_type=str(packet.get("packet_type") or "unknown_packet"),
        )
        base_embedding = embeddings.joint
        predicted_embedding = self._predict_next_embedding(base_embedding)
        predicted_appearance_signature, predicted_change_signature = (
            self._predict_next_stream_embeddings(embeddings)
        )
        embedding = self._compose_embedding(base_embedding)
        appearance_query, change_query = self._build_memory_queries(embeddings)
        action_prediction_error = _cosine_distance(embedding, predicted_embedding)
        appearance_prediction_error = _cosine_distance(
            embeddings.appearance,
            predicted_appearance_signature,
        )
        change_prediction_error = _cosine_distance(
            embeddings.temporal,
            predicted_change_signature,
        )
        appearance_residual = (
            embeddings.appearance.detach().clone()
            - predicted_appearance_signature.detach().clone()
        )
        change_residual = (
            embeddings.temporal.detach().clone()
            - predicted_change_signature.detach().clone()
        )
        pre_learning_appearance_retrieval = self.appearance_memory.retrieve(
            appearance_query,
            store_as_last=False,
            observation_field=observation_field,
        )
        pre_learning_change_retrieval = self.change_memory.retrieve(
            change_query,
            store_as_last=False,
            observation_field=observation_field,
        )
        pre_learning_scores, pre_learning_chart_ids, _ = self._combine_stream_scores(
            self.appearance_memory.retrieval_scores(pre_learning_appearance_retrieval),
            self.change_memory.retrieval_scores(pre_learning_change_retrieval),
            self.appearance_memory.retrieval_chart_ids(pre_learning_appearance_retrieval),
            self.change_memory.retrieval_chart_ids(pre_learning_change_retrieval),
            packet_type=str(packet.get("packet_type") or "unknown_packet"),
        )
        (
            pre_learning_scores,
            _,
            _,
            pre_learning_joint_states,
            pre_learning_joint_candidates,
        ) = self._apply_joint_hypothesis_support(
            pre_learning_scores,
            pre_learning_chart_ids,
            location=location,
            pose_vectors=pose_vectors,
            appearance_vector=appearance_vector,
            behavior_vector=behavior_vector,
            behavior_label=behavior_label,
            inferred_state=inferred_state,
        )
        propagated_pre_learning_candidates = self._propagate_bank_candidates(
            location=location,
            pose_vectors=pose_vectors,
            appearance_vector=appearance_vector,
            behavior_vector=behavior_vector,
            behavior_label=behavior_label,
        )
        pre_learning_scores, pre_learning_chart_ids, pre_learning_joint_states, pre_learning_joint_candidates = self._merge_joint_candidate_rows(
            pre_learning_joint_candidates + propagated_pre_learning_candidates,
            fallback_scores=pre_learning_scores,
            fallback_chart_ids=pre_learning_chart_ids,
            fallback_joint_states=pre_learning_joint_states,
        )
        learning_latent_id, learning_chart_id, learning_support = (
            self._select_learning_hypothesis_from_posterior(
                learn=learn,
                embedding=embedding,
                observation_field=observation_field,
                action_prediction_error=action_prediction_error,
                joint_candidates=pre_learning_joint_candidates,
                location=location,
                pose_vectors=pose_vectors,
                appearance_vector=appearance_vector,
                behavior_vector=behavior_vector,
                behavior_label=behavior_label,
                inferred_state=inferred_state,
            )
        )

        self._last_packet = packet
        self._last_observation_field = observation_field
        self._last_embeddings = embeddings
        self._last_embedding = embedding.detach().clone()
        self._last_predicted_embedding = predicted_embedding.detach().clone()
        self._last_predicted_appearance_signature = (
            predicted_appearance_signature.detach().clone()
        )
        self._last_predicted_change_signature = predicted_change_signature.detach().clone()
        self._last_appearance_residual = appearance_residual.detach().clone()
        self._last_change_residual = change_residual.detach().clone()
        self._has_last_embedding = True
        self._last_action_prediction_error = action_prediction_error
        self._last_appearance_prediction_error = appearance_prediction_error
        self._last_change_prediction_error = change_prediction_error

        if learn and learning_latent_id is not None:
            novelty_signal = float(learning_support.get("novelty_signal", 0.0))
            self.appearance_memory.observe(
                learning_latent_id,
                appearance_query,
                novelty_signal=novelty_signal,
                observation_field=observation_field,
            )
            if self._should_update_change_memory(packet, observation_field):
                change_novelty_signal = max(
                    novelty_signal,
                    float(
                        0.0
                        if observation_field.flow_magnitude.numel() == 0
                        else min(
                            1.0,
                            observation_field.flow_magnitude.abs().max().item(),
                        )
                    ),
                )
                self.change_memory.observe(
                    learning_latent_id,
                    change_query,
                    novelty_signal=change_novelty_signal,
                    observation_field=observation_field,
                )

        appearance_retrieval = self.appearance_memory.retrieve(
            appearance_query,
            store_as_last=True,
            observation_field=observation_field,
        )
        change_retrieval = self.change_memory.retrieve(
            change_query,
            store_as_last=True,
            observation_field=observation_field,
        )
        self._last_memory_retrieval = appearance_retrieval
        self._last_change_memory_retrieval = change_retrieval
        appearance_scores = self.appearance_memory.retrieval_scores(appearance_retrieval)
        change_scores = self.change_memory.retrieval_scores(change_retrieval)
        scores, chart_ids, stream_weights = self._combine_stream_scores(
            appearance_scores,
            change_scores,
            self.appearance_memory.retrieval_chart_ids(appearance_retrieval),
            self.change_memory.retrieval_chart_ids(change_retrieval),
            packet_type=str(packet.get("packet_type") or "unknown_packet"),
        )
        self._last_memory_stream_scores = {
            "appearance": dict(appearance_scores),
            "change": dict(change_scores),
            "combined": dict(scores),
        }
        self._last_memory_stream_weights = dict(stream_weights)
        if learn and learning_latent_id is not None:
            if self._should_carry_learning_selection(
                str(packet.get("packet_type") or "unknown_packet")
            ):
                scores, chart_ids, carried_score_floor = (
                    self._carry_learning_selection_into_scores(
                        scores=scores,
                        chart_ids=chart_ids,
                        learning_latent_id=learning_latent_id,
                        learning_chart_id=learning_chart_id,
                        learning_support=learning_support,
                    )
                )
            else:
                carried_score_floor = 1.0
                scores[learning_latent_id] = max(scores.get(learning_latent_id, 0.0), 1.0)
                if learning_chart_id is not None:
                    chart_ids.setdefault(learning_latent_id, learning_chart_id)
            learning_support["carried_score_floor"] = float(carried_score_floor)

        if learning_latent_id is not None:
            retrieved_learning_chart_id = chart_ids.get(learning_latent_id)
            if learning_chart_id is None:
                learning_chart_id = retrieved_learning_chart_id
            if learning_chart_id is None:
                learning_chart_id = self._fallback_chart_id_for_latent(learning_latent_id)
            if learning_chart_id is not None:
                chart_ids[learning_latent_id] = learning_chart_id
            self._update_chart_state_prototype(
                chart_id=learning_chart_id,
                location=location,
                pose_vectors=pose_vectors,
                appearance_vector=appearance_vector,
                behavior_vector=behavior_vector,
                behavior_label=behavior_label,
            )

        self._last_base_scores = dict(scores)

        after_temporal_behavior_scores, temporal_behavior_adjustments, temporal_behavior_scores = self._blend_temporal_behavior_scores(
            scores,
            embeddings,
            predicted_embedding,
            predicted_appearance_signature,
            predicted_change_signature,
            appearance_prediction_error,
            change_prediction_error,
            packet_type=str(packet.get("packet_type") or "unknown_packet"),
        )
        self._last_after_temporal_behavior_scores = dict(after_temporal_behavior_scores)
        self._last_temporal_behavior_adjustments = dict(temporal_behavior_adjustments)
        self._last_temporal_behavior_scores = dict(temporal_behavior_scores)

        joint_scores, joint_adjustments, joint_support, joint_states, joint_candidates = self._apply_joint_hypothesis_support(
            after_temporal_behavior_scores,
            chart_ids,
            location=location,
            pose_vectors=pose_vectors,
            appearance_vector=appearance_vector,
            behavior_vector=behavior_vector,
            behavior_label=behavior_label,
            inferred_state=inferred_state,
        )
        resolved_chart_ids = {
            object_id: candidate_state.get("chart_id")
            for object_id, candidate_state in joint_states.items()
        }
        propagated_joint_candidates = self._propagate_bank_candidates(
            location=location,
            pose_vectors=pose_vectors,
            appearance_vector=appearance_vector,
            behavior_vector=behavior_vector,
            behavior_label=behavior_label,
        )
        joint_scores, resolved_chart_ids, joint_states, joint_candidates = self._merge_joint_candidate_rows(
            joint_candidates + propagated_joint_candidates,
            fallback_scores=joint_scores,
            fallback_chart_ids=resolved_chart_ids,
            fallback_joint_states=joint_states,
        )
        self._last_after_joint_hypothesis_scores = dict(joint_scores)
        self._last_joint_hypothesis_adjustments = dict(joint_adjustments)
        self._last_joint_hypothesis_support = dict(joint_support)
        joint_candidate_debug_rows = [
            {
                "latent_id": str(candidate.get("latent_id", candidate.get("object_id"))),
                "object_id": str(candidate.get("object_id", candidate.get("latent_id"))),
                "chart_id": candidate.get("chart_id"),
                "score": float(candidate.get("score", 0.0)),
                "base_score": float(candidate.get("base_score", 0.0)),
                "joint_delta": float(candidate.get("joint_delta", 0.0)),
                "pose_support": float(candidate.get("pose_support", 0.0)),
                "appearance_support": float(candidate.get("appearance_support", 0.0)),
                "behavior_similarity": float(candidate.get("behavior_similarity", 0.0)),
                "label_match": float(candidate.get("label_match", 0.0)),
                "behavior_support": float(candidate.get("behavior_support", 0.0)),
                "context_appearance_support": float(
                    candidate.get("context_appearance_support", 0.0)
                ),
                "context_behavior_support": float(
                    candidate.get("context_behavior_support", 0.0)
                ),
                "context_ranked_chart_prior": float(
                    candidate.get("context_ranked_chart_prior", 0.0)
                ),
                "vote_ranked_chart_prior": float(
                    candidate.get("vote_ranked_chart_prior", 0.0)
                ),
                "selected": bool(candidate.get("selected", False)),
                "behavior_label": candidate.get("behavior_label"),
                "inferred_state": candidate.get("inferred_state"),
                "source": candidate.get("source"),
                "age": int(candidate.get("age", 0)),
            }
            for candidate in joint_candidates
        ]
        joint_candidate_debug_rows = sorted(
            joint_candidate_debug_rows,
            key=lambda candidate: (
                -float(candidate.get("score", 0.0)),
                str(candidate.get("latent_id", "")),
                "" if candidate.get("chart_id") is None else str(candidate.get("chart_id")),
            ),
        )
        for rank, candidate in enumerate(joint_candidate_debug_rows, start=1):
            candidate["rank"] = int(rank)
        self._last_joint_hypothesis_candidates = joint_candidate_debug_rows

        transition_scores, transition_adjustments, transition_support, transitioned_joint_candidates, transition_chart_ids = self._apply_chart_transition_support(
            joint_scores,
            resolved_chart_ids,
            joint_candidates=joint_candidates,
        )

        predicted_latent_ids = self._prediction_target_latent_ids_from_state(state)
        adjusted_scores, bias_adjustments = self._apply_biases(
            transition_scores,
            predicted_latent_ids,
        )
        self._last_final_scores = dict(adjusted_scores)
        self._last_self_supervised_adjustments = dict(transition_adjustments)
        for object_id, delta in bias_adjustments.items():
            self._last_self_supervised_adjustments[object_id] = (
                self._last_self_supervised_adjustments.get(object_id, 0.0)
                + float(delta)
            )
        self._last_self_supervised_support = {
            "learning": dict(learning_support),
            "transition": dict(transition_support),
        }
        final_joint_candidates = []
        for candidate in transitioned_joint_candidates:
            object_id = str(candidate.get("object_id", candidate.get("latent_id", "unknown")))
            latent_delta = float(adjusted_scores.get(object_id, 0.0)) - float(
                transition_scores.get(object_id, 0.0)
            )
            final_joint_candidate = dict(candidate)
            final_joint_candidate["score"] = float(
                final_joint_candidate.get("score", 0.0)
            ) + latent_delta
            final_joint_candidates.append(final_joint_candidate)
        self.hypotheses.update(
            adjusted_scores,
            location=location,
            pose_vectors=pose_vectors,
            appearance_vector=appearance_vector,
            behavior_vector=behavior_vector,
            behavior_label=behavior_label,
            inferred_state=inferred_state,
            chart_ids=transition_chart_ids,
            joint_states=joint_states,
            joint_candidates=final_joint_candidates,
        )

        raw_mlh = self.hypotheses.get_current_mlh()
        top_latent_id = _normalize_graph_id(
            raw_mlh.get("latent_id") or raw_mlh.get("graph_id")
        )
        top_graph_id = top_latent_id
        top_chart_id = raw_mlh.get("chart_id")
        top_evidence = float(raw_mlh.get("evidence", 0.0))
        expected_top = predicted_latent_ids[0] if predicted_latent_ids else None
        mismatch = 0.0
        if expected_top is not None:
            mismatch = 0.0 if top_latent_id == expected_top else 1.0

        confidence_gap = max(0.0, self.boundary_confidence_target - top_evidence)
        prediction_mismatch = max(mismatch, confidence_gap)
        residual = min(
            1.0,
            max(
                prediction_mismatch,
                action_prediction_error,
                appearance_prediction_error,
                change_prediction_error,
            ),
        )
        self._last_residual = residual
        self._last_prediction_mismatch = prediction_mismatch
        self._vote_bias = {
            object_id: bias * 0.5
            for object_id, bias in self._vote_bias.items()
            if abs(bias * 0.5) > 1e-4
        }
        self._vote_chart_bias = {
            chart_target: bias * 0.5
            for chart_target, bias in self._vote_chart_bias.items()
            if abs(bias * 0.5) > 1e-4
        }

        self._last_winner_path = {
            "base": self._top_label(self._last_base_scores),
            "after_temporal_behavior": self._top_label(
                self._last_after_temporal_behavior_scores
            ),
            "after_joint_state": self._top_label(
                self._last_after_joint_hypothesis_scores
            ),
            "final": self._top_label(self._last_final_scores),
        }

        ranked_hypotheses = [
            RankedHypothesisVote.from_dict(hypothesis)
            for hypothesis in self.hypotheses.as_ranked_hypotheses()
        ]
        if learning_latent_id is not None:
            if learning_chart_id is None and learning_latent_id == top_latent_id:
                learning_chart_id = top_chart_id
        learning_confidence = float(
            self.hypotheses.get_evidence().get(learning_latent_id, top_evidence)
        )
        self._record_chart_transition(
            learning_latent_id=learning_latent_id if learn else None,
            chart_id=learning_chart_id,
            confidence=learning_confidence,
        )
        self._update_temporal_state(
            embedding,
            predicted_embedding,
            predicted_appearance_signature,
            predicted_change_signature,
            top_latent_id=top_latent_id,
            prediction_mismatch=prediction_mismatch,
            action_prediction_error=action_prediction_error,
            appearance_prediction_error=appearance_prediction_error,
            change_prediction_error=change_prediction_error,
        )
        # Use the evidence-backed winner as the transition reference so
        # transition support cannot bootstrap and then reinforce its own flips.
        transition_reference_latent_id = (
            self._last_winner_path.get("after_joint_state")
            or self._last_winner_path.get("after_temporal_behavior")
            or top_latent_id
        )
        transition_reference_chart_id = transition_chart_ids.get(transition_reference_latent_id)
        if transition_reference_chart_id is None and (
            transition_reference_latent_id == top_latent_id
        ):
            transition_reference_chart_id = top_chart_id
        self._last_winning_latent_id = transition_reference_latent_id
        self._last_winning_chart_id = transition_reference_chart_id
        self._last_active_cells = (
            self._temporal_state.context_vector.detach().cpu().numpy().astype(np.float32)
        )
        self._last_boundary_pressure = self._temporal_state.boundary_pressure_value
        self._last_message_state = PredictiveMessageState(
            latent_id=top_latent_id,
            confidence=top_evidence,
            residual=residual,
            active_cells=self._temporal_state.context_vector.detach().clone(),
            predicted_appearance_signature=predicted_appearance_signature.detach().clone(),
            predicted_change_signature=predicted_change_signature.detach().clone(),
            appearance_residual=appearance_residual.detach().clone(),
            change_residual=change_residual.detach().clone(),
            ranked_hypotheses=ranked_hypotheses,
        )

        output_evidence, output_ranked_hypotheses, output_mlh = self._output_summary()

        return {
            "detail_packet": packet,
            "observation_field": observation_field,
            "embeddings": embeddings,
            "embedding": embedding.detach().clone(),
            "memory_retrieval": self._memory_retrieval_summary(),
            "evidence": output_evidence,
            "mlh": output_mlh,
            "ranked_hypotheses": [dict(hypothesis) for hypothesis in output_ranked_hypotheses],
            "active_cells": self._last_active_cells.copy(),
            "residual": residual,
            "boundary_pressure": self._last_boundary_pressure,
            "base_evidence": dict(self._last_base_scores),
            "after_temporal_behavior_evidence": dict(
                self._last_after_temporal_behavior_scores
            ),
            "after_joint_hypothesis_evidence": dict(
                self._last_after_joint_hypothesis_scores
            ),
            "final_evidence": dict(self._last_final_scores),
            "temporal_behavior_adjustments": dict(
                self._last_temporal_behavior_adjustments
            ),
            "temporal_behavior_scores": dict(self._last_temporal_behavior_scores),
            "joint_hypothesis_adjustments": dict(
                self._last_joint_hypothesis_adjustments
            ),
            "joint_hypothesis_support": dict(self._last_joint_hypothesis_support),
            "joint_hypothesis_candidates": [
                dict(candidate)
                for candidate in self._last_joint_hypothesis_candidates
            ],
            "self_supervised_adjustments": dict(
                self._last_self_supervised_adjustments
            ),
            "self_supervised_support": dict(self._last_self_supervised_support),
            "winner_path": dict(self._last_winner_path),
            "prediction_mismatch": prediction_mismatch,
            "action_prediction_error": action_prediction_error,
            "appearance_prediction_error": appearance_prediction_error,
            "change_prediction_error": change_prediction_error,
        }

    @staticmethod
    def _top_label(evidence: dict[str, float]) -> str | None:
        if not evidence:
            return None
        return max(
            evidence.items(),
            key=lambda item: (float(item[1]), str(item[0])),
        )[0]

    def get_current_mlh(self) -> dict[str, Any]:
        _, _, mlh = self._output_summary()
        return mlh

    def get_possible_matches(self, min_weight: float = 0.05) -> list[str]:
        evidence, _, _ = self._output_summary()
        return [
            object_id
            for object_id, weight in evidence.items()
            if float(weight) >= float(min_weight)
        ]

    def get_evidence(self) -> dict[str, float]:
        evidence, _, _ = self._output_summary()
        return evidence

    def get_context_signal(self) -> dict[str, Any] | None:
        signal = self.get_context_signal_message()
        return None if signal is None else signal.to_dict()

    def get_context_signal_message(self) -> PredictiveContextSignal | None:
        if not self.hypotheses.state.object_ids:
            return None

        mlh = self.hypotheses.get_current_mlh()
        ranked_context_hypotheses = []
        can_export_context = (
            float(mlh.get("evidence", 0.0)) >= self.boundary_confidence_target
            and float(self._last_residual) <= 0.25
        )
        if can_export_context:
            can_export_predicted_appearance = (
                float(self._last_appearance_prediction_error) <= 0.25
            )
            can_export_predicted_change = (
                float(self._last_change_prediction_error) <= 0.35
            )
        else:
            can_export_predicted_appearance = False
            can_export_predicted_change = False
        if can_export_predicted_appearance:
            predicted_appearance_signature = self._last_predicted_appearance_signature.detach().clone()
        else:
            predicted_appearance_signature = torch.zeros(
                0,
                dtype=torch.float32,
                device=self.device,
            )
        if can_export_predicted_change:
            predicted_change_signature = self._last_predicted_change_signature.detach().clone()
        else:
            predicted_change_signature = torch.zeros(
                0,
                dtype=torch.float32,
                device=self.device,
            )
        if can_export_context:
            appearance_signature = torch.as_tensor(
                np.asarray(mlh.get("appearance_signature", np.zeros(0)), dtype=np.float32).reshape(-1),
                dtype=torch.float32,
                device=self.device,
            )
            ranked_context_hypotheses = [
                RankedHypothesisVote.from_dict(hypothesis)
                for hypothesis in self.hypotheses.as_ranked_hypotheses()
            ]
        else:
            appearance_signature = torch.zeros(
                0,
                dtype=torch.float32,
                device=self.device,
            )
        behavior_signature = torch.as_tensor(
            np.asarray(mlh.get("behavior_signature", np.zeros(0)), dtype=np.float32).reshape(-1),
            dtype=torch.float32,
            device=self.device,
        )

        return PredictiveContextSignal(
            latent_id=None,
            chart_id=mlh.get("chart_id") if can_export_context else None,
            confidence=float(mlh.get("evidence", 0.0)),
            residual=float(self._last_residual),
            active_cells=self._temporal_state.context_vector.detach().clone(),
            predicted_appearance_signature=predicted_appearance_signature,
            predicted_change_signature=predicted_change_signature,
            appearance_signature=appearance_signature,
            behavior_label=mlh.get("behavior_label"),
            behavior_signature=behavior_signature,
            ranked_hypotheses=ranked_context_hypotheses,
        )

    def get_last_detail_packet(self) -> dict[str, Any] | None:
        return None if self._last_packet is None else dict(self._last_packet)

    def get_last_observation_field(self) -> ObservationField:
        return self._last_observation_field

    def get_last_embeddings(self) -> ObservationEmbeddings:
        return self._last_embeddings

    def get_memory_slots(self) -> MemorySlots:
        return self.appearance_memory.get_slot_state()

    def get_change_memory_slots(self) -> MemorySlots:
        return self.change_memory.get_slot_state()

    def get_last_memory_retrieval(self) -> HopfieldRetrievalState:
        return self._last_memory_retrieval

    def get_last_change_memory_retrieval(self) -> HopfieldRetrievalState:
        return self._last_change_memory_retrieval

    def get_last_stream_memory_retrievals(self) -> dict[str, HopfieldRetrievalState]:
        return {
            "appearance": self._last_memory_retrieval,
            "change": self._last_change_memory_retrieval,
        }

    def get_chart_transition_counts(self) -> dict[str, dict[str, float]]:
        return {
            str(chart_id): {
                str(next_chart_id): float(count)
                for next_chart_id, count in sorted(next_counts.items())
            }
            for chart_id, next_counts in sorted(self._chart_transition_counts.items())
        }

    def get_chart_state_prototypes(self) -> dict[str, dict[str, Any]]:
        return {
            str(chart_id): self._chart_state_snapshot(str(chart_id))
            for chart_id in sorted(self._chart_state_prototypes.keys())
        }

    def get_all_known_object_ids(self) -> list[str]:
        return sorted(
            set(self.appearance_memory.get_all_known_object_ids())
            | set(self.change_memory.get_all_known_object_ids())
        )

    def get_all_known_latent_ids(self) -> list[str]:
        return self.get_all_known_object_ids()

    def get_hypothesis_state(self) -> HypothesisBankState:
        return self.hypotheses.get_state()

    def get_temporal_state(self) -> TemporalState:
        return self._temporal_state

    def get_last_message_state(self) -> PredictiveMessageState:
        return self._last_message_state

    def get_evidence_debug(self) -> dict[str, Any]:
        packet = self.get_last_detail_packet() or {}
        return {
            "detail_packet_type": packet.get("packet_type"),
            "detail_shard_count": int(packet.get("shard_count", 0)),
            "memory_retrieval": self._memory_retrieval_summary(),
            "base_evidence": dict(self._last_base_scores),
            "after_temporal_behavior_evidence": dict(
                self._last_after_temporal_behavior_scores
            ),
            "after_joint_hypothesis_evidence": dict(
                self._last_after_joint_hypothesis_scores
            ),
            "final_evidence": dict(self._last_final_scores),
            "temporal_behavior_adjustments": dict(
                self._last_temporal_behavior_adjustments
            ),
            "temporal_behavior_scores": dict(self._last_temporal_behavior_scores),
            "joint_hypothesis_adjustments": dict(
                self._last_joint_hypothesis_adjustments
            ),
            "joint_hypothesis_support": dict(self._last_joint_hypothesis_support),
            "joint_hypothesis_candidates": [
                dict(candidate)
                for candidate in self._last_joint_hypothesis_candidates
            ],
            "self_supervised_adjustments": dict(
                self._last_self_supervised_adjustments
            ),
            "self_supervised_support": dict(self._last_self_supervised_support),
            "memory_stream_scores": {
                stream_name: dict(stream_scores)
                for stream_name, stream_scores in self._last_memory_stream_scores.items()
            },
            "memory_stream_weights": dict(self._last_memory_stream_weights),
            "winner_path": dict(self._last_winner_path),
            "prediction_mismatch": float(self._last_prediction_mismatch),
            "action_prediction_error": float(self._last_action_prediction_error),
            "appearance_prediction_error": float(self._last_appearance_prediction_error),
            "change_prediction_error": float(self._last_change_prediction_error),
            "stream_prediction_norms": {
                "predicted_appearance_norm": float(
                    self._last_predicted_appearance_signature.norm(p=2).item()
                ),
                "predicted_change_norm": float(
                    self._last_predicted_change_signature.norm(p=2).item()
                ),
                "appearance_residual_norm": float(
                    self._last_appearance_residual.norm(p=2).item()
                ),
                "change_residual_norm": float(
                    self._last_change_residual.norm(p=2).item()
                ),
            },
            "query_bias_norms": {
                "action_context_norm": float(np.linalg.norm(self._action_context)),
                "context_norm": float(
                    0.0
                    if self._external_context is None
                    else self._external_context.norm(p=2).item()
                ),
                "predicted_appearance_prior_norm": float(
                    self._external_predicted_appearance_signature.norm(p=2).item()
                ),
                "predicted_change_prior_norm": float(
                    self._external_predicted_change_signature.norm(p=2).item()
                ),
            },
        }

    @staticmethod
    def _single_retrieval_summary(
        retrieval: HopfieldRetrievalState,
    ) -> dict[str, Any]:
        return {
            "iterations": int(retrieval.iteration_count),
            "top_object_id": retrieval.top_object_id,
            "top_chart_id": retrieval.top_chart_id,
            "energy_trace": list(retrieval.energy_trace),
            "top_slot_indices": (
                []
                if retrieval.top_slot_indices is None
                else [
                    int(value)
                    for value in retrieval.top_slot_indices.detach().cpu().reshape(-1).tolist()
                ]
            ),
        }

    def _memory_retrieval_summary(self) -> dict[str, Any]:
        primary_summary = self._single_retrieval_summary(self._last_memory_retrieval)
        primary_summary["stream_weights"] = dict(self._last_memory_stream_weights)
        primary_summary["appearance"] = self._single_retrieval_summary(
            self._last_memory_retrieval
        )
        primary_summary["change"] = self._single_retrieval_summary(
            self._last_change_memory_retrieval
        )
        return primary_summary

    def get_temporal_prediction_status(self, threshold: float = 0.6) -> str | None:
        if not self.hypotheses.entries:
            return None
        return "confident" if self.get_current_mlh().get("evidence", 0.0) >= threshold else "confused"

    def get_temporal_surprise(self) -> float:
        return float(self._last_boundary_pressure)

    def get_temporal_context(self) -> dict[str, Any]:
        return {
            "boundary_pressure": float(self._temporal_state.boundary_pressure_value),
            "mean_surprise": float(self._temporal_state.mean_surprise_value),
            "current_latent_id": self._temporal_state.current_latent_id,
            "current_graph_id": self._temporal_state.current_latent_id,
            "known_latent_ids": list(self._temporal_state.known_latent_ids),
            "known_states": list(self._temporal_state.known_latent_ids),
            "trace_norm": float(self._temporal_state.trace_norm),
            "event_detected": bool(self._temporal_state.event_detected_bool),
            "dwell_steps": int(self._temporal_state.dwell_step_count),
            "prediction_mismatch": float(
                self._temporal_state.prediction_mismatch_value
            ),
            "action_prediction_error": float(
                self._temporal_state.action_prediction_error_value
            ),
            "appearance_prediction_error": float(
                self._temporal_state.appearance_prediction_error_value
            ),
            "change_prediction_error": float(
                self._temporal_state.change_prediction_error_value
            ),
            "predicted_appearance_norm": float(
                self._temporal_state.predicted_appearance_signature.norm(p=2).item()
            ),
            "predicted_change_norm": float(
                self._temporal_state.predicted_change_signature.norm(p=2).item()
            ),
            "timescale_trace_norms": list(self._temporal_state.trace_bank_norms),
            "active_self_supervised_latent_id": self._active_self_supervised_latent_id,
            "active_self_supervised_object_id": self._active_self_supervised_latent_id,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "context_dim": self.context_dim,
            "memory_layout": self.memory_layout,
            "memory_substrate_cell_count": self.memory_substrate_cell_count,
            "memory_active_cell_sparsity": self.memory_active_cell_sparsity,
            "memory_seed": self.memory_seed,
            "prediction_bias_weight": self.prediction_bias_weight,
            "vote_bias_weight": self.vote_bias_weight,
            "action_context_weight": self.action_context_weight,
            "temporal_score_weight": self.temporal_score_weight,
            "joint_appearance_score_weight": self.joint_appearance_score_weight,
            "joint_pose_score_weight": self.joint_pose_score_weight,
            "joint_behavior_score_weight": self.joint_behavior_score_weight,
            "context_behavior_score_weight": self.context_behavior_score_weight,
            "vote_appearance_score_weight": self.vote_appearance_score_weight,
            "vote_behavior_score_weight": self.vote_behavior_score_weight,
            "boundary_confidence_target": self.boundary_confidence_target,
            "self_supervised_object_reset_distance": self.self_supervised_object_reset_distance,
            "self_supervised_boundary_reset_threshold": self.self_supervised_boundary_reset_threshold,
            "self_supervised_memory_reuse_threshold": self.self_supervised_memory_reuse_threshold,
            "transition_score_weight": self.transition_score_weight,
            "anchor_bootstrap_novel_min_posterior": self.anchor_bootstrap_novel_min_posterior,
            "anchor_bootstrap_existing_overwhelm_margin": self.anchor_bootstrap_existing_overwhelm_margin,
            "anchor_bootstrap_existing_overwhelm_ratio": self.anchor_bootstrap_existing_overwhelm_ratio,
            "anchor_continuity_switch_margin": self.anchor_continuity_switch_margin,
            "anchor_continuity_switch_ratio": self.anchor_continuity_switch_ratio,
            "chart_transition_counts": self.get_chart_transition_counts(),
            "chart_state_prototypes": self.get_chart_state_prototypes(),
            "latent_object_counter": self._latent_object_counter,
            "memory": self.appearance_memory.state_dict(),
            "appearance_memory": self.appearance_memory.state_dict(),
            "change_memory": self.change_memory.state_dict(),
            "hypotheses": self.hypotheses.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.memory_substrate_cell_count = state_dict.get(
            "memory_substrate_cell_count",
            self.memory_substrate_cell_count,
        )
        if self.memory_substrate_cell_count is not None:
            self.memory_substrate_cell_count = max(
                int(self.memory_substrate_cell_count),
                32,
            )
        self.memory_active_cell_sparsity = float(
            state_dict.get(
                "memory_active_cell_sparsity",
                self.memory_active_cell_sparsity,
            )
        )
        self.memory_seed = int(state_dict.get("memory_seed", self.memory_seed))
        self.prediction_bias_weight = float(
            state_dict.get("prediction_bias_weight", self.prediction_bias_weight)
        )
        self.vote_bias_weight = float(
            state_dict.get("vote_bias_weight", self.vote_bias_weight)
        )
        self.action_context_weight = float(
            state_dict.get("action_context_weight", self.action_context_weight)
        )
        self.temporal_score_weight = float(
            state_dict.get("temporal_score_weight", self.temporal_score_weight)
        )
        self.joint_appearance_score_weight = float(
            state_dict.get(
                "joint_appearance_score_weight",
                self.joint_appearance_score_weight,
            )
        )
        self.joint_pose_score_weight = float(
            state_dict.get("joint_pose_score_weight", self.joint_pose_score_weight)
        )
        self.joint_behavior_score_weight = float(
            state_dict.get("joint_behavior_score_weight", self.joint_behavior_score_weight)
        )
        self.context_behavior_score_weight = float(
            state_dict.get(
                "context_behavior_score_weight",
                self.context_behavior_score_weight,
            )
        )
        self.vote_appearance_score_weight = float(
            state_dict.get(
                "vote_appearance_score_weight",
                self.vote_appearance_score_weight,
            )
        )
        self.vote_behavior_score_weight = float(
            state_dict.get("vote_behavior_score_weight", self.vote_behavior_score_weight)
        )
        self.boundary_confidence_target = float(
            state_dict.get(
                "boundary_confidence_target",
                self.boundary_confidence_target,
            )
        )
        self.self_supervised_object_reset_distance = float(
            state_dict.get(
                "self_supervised_object_reset_distance",
                self.self_supervised_object_reset_distance,
            )
        )
        self.self_supervised_boundary_reset_threshold = float(
            state_dict.get(
                "self_supervised_boundary_reset_threshold",
                self.self_supervised_boundary_reset_threshold,
            )
        )
        self.self_supervised_memory_reuse_threshold = float(
            state_dict.get(
                "self_supervised_memory_reuse_threshold",
                self.self_supervised_memory_reuse_threshold,
            )
        )
        self.transition_score_weight = float(
            state_dict.get("transition_score_weight", self.transition_score_weight)
        )
        self.anchor_bootstrap_novel_min_posterior = float(
            state_dict.get(
                "anchor_bootstrap_novel_min_posterior",
                self.anchor_bootstrap_novel_min_posterior,
            )
        )
        self.anchor_bootstrap_existing_overwhelm_margin = float(
            state_dict.get(
                "anchor_bootstrap_existing_overwhelm_margin",
                self.anchor_bootstrap_existing_overwhelm_margin,
            )
        )
        self.anchor_bootstrap_existing_overwhelm_ratio = float(
            state_dict.get(
                "anchor_bootstrap_existing_overwhelm_ratio",
                self.anchor_bootstrap_existing_overwhelm_ratio,
            )
        )
        self.anchor_continuity_switch_margin = float(
            state_dict.get(
                "anchor_continuity_switch_margin",
                self.anchor_continuity_switch_margin,
            )
        )
        self.anchor_continuity_switch_ratio = float(
            state_dict.get(
                "anchor_continuity_switch_ratio",
                self.anchor_continuity_switch_ratio,
            )
        )
        appearance_memory_state = state_dict.get(
            "appearance_memory",
            state_dict.get("memory", {}),
        )
        saved_memory_layout = str(
            appearance_memory_state.get(
                "memory_layout",
                state_dict.get("memory_layout", self.memory_layout),
            )
        )
        if saved_memory_layout != self.memory_layout and not (
            saved_memory_layout == "shared_slot_bank_v2"
            and self.memory_layout in {"fixed_sparse_recurrent", "fixed_sparse_recurrent_v1"}
        ):
            self._rebuild_memories(saved_memory_layout)
        self.appearance_memory.load_state_dict(appearance_memory_state)
        self.change_memory.load_state_dict(state_dict.get("change_memory", {}))
        self.memory = self.appearance_memory
        self.hypotheses.load_state_dict(state_dict.get("hypotheses", {}))
        self._chart_transition_counts = {
            str(chart_id): {
                str(next_chart_id): float(count)
                for next_chart_id, count in (next_counts or {}).items()
            }
            for chart_id, next_counts in state_dict.get(
                "chart_transition_counts",
                {},
            ).items()
        }
        self._chart_state_prototypes = {
            str(chart_id): {
                "observation_count": float((prototype or {}).get("observation_count", 0.0)),
                "location": np.asarray(
                    (prototype or {}).get("location", np.zeros(3, dtype=np.float32)),
                    dtype=np.float32,
                ).reshape(3),
                "pose_vectors": _normalize_pose_matrix(
                    (prototype or {}).get("pose_vectors", np.eye(3, dtype=np.float32))
                ),
                "appearance_vector": _as_float32_vector(
                    (prototype or {}).get("appearance_vector")
                ),
                "behavior_vector": _as_float32_vector(
                    (prototype or {}).get("behavior_vector")
                ),
                "behavior_label": (prototype or {}).get("behavior_label"),
                "behavior_label_counts": {
                    str(label): float(count)
                    for label, count in ((prototype or {}).get("behavior_label_counts") or {}).items()
                },
            }
            for chart_id, prototype in state_dict.get("chart_state_prototypes", {}).items()
        }
        self._latent_object_counter = int(state_dict.get("latent_object_counter", 0))
        self._active_self_supervised_latent_id = None
        self._last_learning_latent_id = None
        self._last_learning_chart_id = None
        self._last_winning_latent_id = None
        self._last_winning_chart_id = None
        self._last_self_supervised_support = {}
