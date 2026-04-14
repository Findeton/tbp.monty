from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


def _empty_f32(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    return torch.zeros(shape, dtype=torch.float32, device=device)


def _empty_i64(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    return torch.zeros(shape, dtype=torch.int64, device=device)


def _tensor_from_any(
    value: Any,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if value is None:
        return torch.zeros(0, dtype=dtype, device=device)
    if torch.is_tensor(value):
        return value.detach().to(device=device, dtype=dtype).clone()
    return torch.as_tensor(value, dtype=dtype, device=device).clone()


@dataclass
class ObservationField:
    packet_type: str
    sender_id: str | None
    grid_shape: tuple[int, int]
    micro_patch_shape: tuple[int, int]
    rgb: torch.Tensor
    depth: torch.Tensor
    support: torch.Tensor
    xyz: torch.Tensor
    uv: torch.Tensor
    valid_cells: torch.Tensor
    flow_direction: torch.Tensor
    flow_magnitude: torch.Tensor
    temporal_feature_stats: torch.Tensor
    anchor_location: torch.Tensor
    anchor_pose: torch.Tensor
    anchor_confidence: torch.Tensor
    anchor_on_object: torch.Tensor
    camera_forward: torch.Tensor
    child_locations: torch.Tensor
    child_confidences: torch.Tensor
    child_graph_hash: torch.Tensor
    raw_active_cells: torch.Tensor

    @classmethod
    def empty(
        cls,
        *,
        device: torch.device,
        packet_type: str = "unknown_packet",
        sender_id: str | None = None,
        micro_patch_shape: tuple[int, int] = (4, 4),
        grid_shape: tuple[int, int] = (0, 0),
    ) -> "ObservationField":
        patch_rows = max(int(micro_patch_shape[0]), 1)
        patch_cols = max(int(micro_patch_shape[1]), 1)
        return cls(
            packet_type=packet_type,
            sender_id=sender_id,
            grid_shape=tuple(grid_shape),
            micro_patch_shape=(patch_rows, patch_cols),
            rgb=_empty_f32((0, patch_rows, patch_cols, 3), device),
            depth=_empty_f32((0, patch_rows, patch_cols), device),
            support=_empty_f32((0, patch_rows, patch_cols), device),
            xyz=_empty_f32((0, patch_rows, patch_cols, 3), device),
            uv=_empty_f32((0, 2), device),
            valid_cells=_empty_f32((0,), device),
            flow_direction=_empty_f32((3,), device),
            flow_magnitude=_empty_f32((1,), device),
            temporal_feature_stats=_empty_f32((0, 3), device),
            anchor_location=_empty_f32((3,), device),
            anchor_pose=_empty_f32((3, 3), device),
            anchor_confidence=_empty_f32((1,), device),
            anchor_on_object=_empty_f32((1,), device),
            camera_forward=_empty_f32((3,), device),
            child_locations=_empty_f32((0, 3), device),
            child_confidences=_empty_f32((0,), device),
            child_graph_hash=_empty_f32((0, 8), device),
            raw_active_cells=_empty_f32((0,), device),
        )

    @property
    def cell_count(self) -> int:
        return int(self.rgb.shape[0])

    @property
    def child_count(self) -> int:
        return int(self.child_locations.shape[0])

    @property
    def shard_count(self) -> int:
        return max(self.cell_count, self.child_count)


@dataclass
class ObservationEmbeddings:
    appearance: torch.Tensor
    geometry: torch.Tensor
    temporal: torch.Tensor
    joint: torch.Tensor

    @classmethod
    def empty(
        cls,
        *,
        stream_dim: int,
        context_dim: int,
        device: torch.device,
    ) -> "ObservationEmbeddings":
        return cls(
            appearance=_empty_f32((stream_dim,), device),
            geometry=_empty_f32((stream_dim,), device),
            temporal=_empty_f32((stream_dim,), device),
            joint=_empty_f32((context_dim,), device),
        )


@dataclass
class RankedHypothesisVote:
    object_id: str
    probability: float
    evidence: float
    rank: int
    chart_id: str | None = None
    sender_id: str | None = None
    sender_ids: list[str] = field(default_factory=list)
    behavior_label: str | None = None
    behavior_signature: torch.Tensor = field(
        default_factory=lambda: torch.zeros(0, dtype=torch.float32)
    )
    appearance_signature: torch.Tensor = field(
        default_factory=lambda: torch.zeros(0, dtype=torch.float32)
    )
    inferred_state: int | None = None

    @property
    def latent_id(self) -> str:
        return self.object_id

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RankedHypothesisVote":
        sender_id = payload.get("sender_id")
        sender_ids = [
            str(item)
            for item in (payload.get("sender_ids") or [])
            if item is not None
        ]
        if sender_id is not None and str(sender_id) not in sender_ids:
            sender_ids = [str(sender_id), *sender_ids]

        inferred_state = payload.get("inferred_state")
        if inferred_state is not None:
            try:
                inferred_state = int(inferred_state)
            except (TypeError, ValueError):
                inferred_state = None

        return cls(
            object_id=str(payload.get("latent_id", payload.get("object_id", "unknown"))),
            chart_id=(
                None
                if payload.get("chart_id") is None
                else str(payload.get("chart_id"))
            ),
            probability=float(payload.get("probability", 0.0)),
            evidence=float(payload.get("evidence", 0.0)),
            rank=int(payload.get("rank", 1)),
            sender_id=None if sender_id is None else str(sender_id),
            sender_ids=sender_ids,
            behavior_label=(
                None
                if payload.get("behavior_label") is None
                else str(payload.get("behavior_label"))
            ),
            behavior_signature=_tensor_from_any(
                payload.get("behavior_signature"),
                device=torch.device("cpu"),
            ).reshape(-1),
            appearance_signature=_tensor_from_any(
                payload.get("appearance_signature"),
                device=torch.device("cpu"),
            ).reshape(-1),
            inferred_state=inferred_state,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "object_id": self.object_id,
            "latent_id": self.object_id,
            "probability": float(self.probability),
            "evidence": float(self.evidence),
            "scaled_evidence": float(self.probability),
            "rank": int(self.rank),
        }
        if self.chart_id is not None:
            payload["chart_id"] = self.chart_id
        if self.sender_id is not None:
            payload["sender_id"] = self.sender_id
        if self.sender_ids:
            payload["sender_ids"] = list(self.sender_ids)
        if self.behavior_label is not None:
            payload["behavior_label"] = self.behavior_label
        if self.behavior_signature.numel() > 0:
            payload["behavior_signature"] = (
                self.behavior_signature.detach().cpu().float().numpy().copy()
            )
        if self.appearance_signature.numel() > 0:
            payload["appearance_signature"] = (
                self.appearance_signature.detach().cpu().float().numpy().copy()
            )
        if self.inferred_state is not None:
            payload["inferred_state"] = int(self.inferred_state)
        return payload


@dataclass(init=False)
class PredictiveMessageState:
    latent_id: str | None
    confidence: float
    residual: float
    active_cells: torch.Tensor
    predicted_appearance_signature: torch.Tensor = field(
        default_factory=lambda: torch.zeros(0, dtype=torch.float32)
    )
    predicted_change_signature: torch.Tensor = field(
        default_factory=lambda: torch.zeros(0, dtype=torch.float32)
    )
    appearance_residual: torch.Tensor = field(
        default_factory=lambda: torch.zeros(0, dtype=torch.float32)
    )
    change_residual: torch.Tensor = field(
        default_factory=lambda: torch.zeros(0, dtype=torch.float32)
    )
    ranked_hypotheses: list[RankedHypothesisVote] = field(default_factory=list)

    def __init__(
        self,
        latent_id: str | None = None,
        confidence: float = 0.0,
        residual: float = 0.0,
        active_cells: torch.Tensor | None = None,
        predicted_appearance_signature: torch.Tensor | None = None,
        predicted_change_signature: torch.Tensor | None = None,
        appearance_residual: torch.Tensor | None = None,
        change_residual: torch.Tensor | None = None,
        ranked_hypotheses: list[RankedHypothesisVote] | None = None,
        *,
        graph_id: str | None = None,
    ) -> None:
        resolved_latent_id = latent_id if latent_id is not None else graph_id
        self.latent_id = None if resolved_latent_id is None else str(resolved_latent_id)
        self.confidence = float(confidence)
        self.residual = float(residual)
        self.active_cells = (
            torch.zeros(0, dtype=torch.float32)
            if active_cells is None
            else active_cells
        )
        self.predicted_appearance_signature = (
            torch.zeros(0, dtype=torch.float32)
            if predicted_appearance_signature is None
            else predicted_appearance_signature
        )
        self.predicted_change_signature = (
            torch.zeros(0, dtype=torch.float32)
            if predicted_change_signature is None
            else predicted_change_signature
        )
        self.appearance_residual = (
            torch.zeros(0, dtype=torch.float32)
            if appearance_residual is None
            else appearance_residual
        )
        self.change_residual = (
            torch.zeros(0, dtype=torch.float32)
            if change_residual is None
            else change_residual
        )
        self.ranked_hypotheses = list(ranked_hypotheses or [])

    @classmethod
    def empty(
        cls,
        *,
        context_dim: int,
        device: torch.device,
    ) -> "PredictiveMessageState":
        return cls(
            latent_id=None,
            confidence=0.0,
            residual=0.0,
            active_cells=_empty_f32((context_dim,), device),
            predicted_appearance_signature=_empty_f32((0,), device),
            predicted_change_signature=_empty_f32((0,), device),
            appearance_residual=_empty_f32((0,), device),
            change_residual=_empty_f32((0,), device),
            ranked_hypotheses=[],
        )

    def ranked_hypotheses_as_dicts(self) -> list[dict[str, Any]]:
        return [hypothesis.to_dict() for hypothesis in self.ranked_hypotheses]

    @property
    def graph_id(self) -> str | None:
        return self.latent_id

    @graph_id.setter
    def graph_id(self, value: str | None) -> None:
        self.latent_id = None if value is None else str(value)


@dataclass(init=False)
class PredictiveContextSignal:
    latent_id: str | None
    chart_id: str | None
    confidence: float
    residual: float
    active_cells: torch.Tensor
    child_latent_ids: list[str] = field(default_factory=list)
    predicted_appearance_signature: torch.Tensor = field(
        default_factory=lambda: torch.zeros(0, dtype=torch.float32)
    )
    predicted_change_signature: torch.Tensor = field(
        default_factory=lambda: torch.zeros(0, dtype=torch.float32)
    )
    appearance_signature: torch.Tensor = field(
        default_factory=lambda: torch.zeros(0, dtype=torch.float32)
    )
    behavior_label: str | None = None
    behavior_signature: torch.Tensor = field(
        default_factory=lambda: torch.zeros(0, dtype=torch.float32)
    )
    sender_id: str | None = None
    sender_step_count: int | None = None
    routing_scope: str | None = None
    target_sender_ids: list[str] = field(default_factory=list)
    ranked_hypotheses: list[RankedHypothesisVote] = field(default_factory=list)

    def __init__(
        self,
        latent_id: str | None = None,
        chart_id: str | None = None,
        confidence: float = 0.0,
        residual: float = 0.0,
        active_cells: torch.Tensor | None = None,
        child_latent_ids: list[str] | None = None,
        predicted_appearance_signature: torch.Tensor | None = None,
        predicted_change_signature: torch.Tensor | None = None,
        appearance_signature: torch.Tensor | None = None,
        behavior_label: str | None = None,
        behavior_signature: torch.Tensor | None = None,
        sender_id: str | None = None,
        sender_step_count: int | None = None,
        routing_scope: str | None = None,
        target_sender_ids: list[str] | None = None,
        ranked_hypotheses: list[RankedHypothesisVote] | None = None,
        *,
        graph_id: str | None = None,
        child_graph_ids: list[str] | None = None,
    ) -> None:
        resolved_latent_id = latent_id if latent_id is not None else graph_id
        resolved_child_latent_ids = (
            child_latent_ids if child_latent_ids is not None else child_graph_ids
        )
        self.latent_id = None if resolved_latent_id is None else str(resolved_latent_id)
        self.chart_id = None if chart_id is None else str(chart_id)
        self.confidence = float(confidence)
        self.residual = float(residual)
        self.active_cells = (
            torch.zeros(0, dtype=torch.float32)
            if active_cells is None
            else active_cells
        )
        self.child_latent_ids = [
            str(value) for value in (resolved_child_latent_ids or []) if value is not None
        ]
        self.predicted_appearance_signature = (
            torch.zeros(0, dtype=torch.float32)
            if predicted_appearance_signature is None
            else predicted_appearance_signature
        )
        self.predicted_change_signature = (
            torch.zeros(0, dtype=torch.float32)
            if predicted_change_signature is None
            else predicted_change_signature
        )
        self.appearance_signature = (
            torch.zeros(0, dtype=torch.float32)
            if appearance_signature is None
            else appearance_signature
        )
        self.behavior_label = None if behavior_label is None else str(behavior_label)
        self.behavior_signature = (
            torch.zeros(0, dtype=torch.float32)
            if behavior_signature is None
            else behavior_signature
        )
        self.sender_id = None if sender_id is None else str(sender_id)
        self.sender_step_count = (
            None if sender_step_count is None else int(sender_step_count)
        )
        self.routing_scope = None if routing_scope is None else str(routing_scope)
        self.target_sender_ids = [
            str(value) for value in (target_sender_ids or []) if value is not None
        ]
        self.ranked_hypotheses = list(ranked_hypotheses or [])

    @classmethod
    def empty(
        cls,
        *,
        context_dim: int,
        device: torch.device,
    ) -> "PredictiveContextSignal":
        return cls(
            latent_id=None,
            confidence=0.0,
            residual=0.0,
            active_cells=_empty_f32((context_dim,), device),
            predicted_appearance_signature=_empty_f32((0,), device),
            predicted_change_signature=_empty_f32((0,), device),
            appearance_signature=_empty_f32((0,), device),
            behavior_signature=_empty_f32((0,), device),
        )

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        *,
        device: torch.device,
    ) -> "PredictiveContextSignal":
        child_latent_ids = payload.get("child_latent_ids")
        if child_latent_ids is None:
            child_latent_ids = payload.get("child_graph_ids")
        return cls(
            latent_id=None
            if payload.get("latent_id", payload.get("graph_id")) is None
            else str(payload.get("latent_id", payload.get("graph_id"))),
            chart_id=(
                None
                if payload.get("chart_id") is None
                else str(payload.get("chart_id"))
            ),
            confidence=float(payload.get("confidence", 0.0)),
            residual=float(payload.get("residual", 0.0)),
            active_cells=_tensor_from_any(
                payload.get("active_cells"),
                device=device,
            ).reshape(-1),
            child_latent_ids=[
                str(item)
                for item in (child_latent_ids or [])
                if item is not None
            ],
            predicted_appearance_signature=_tensor_from_any(
                payload.get("predicted_appearance_signature"),
                device=device,
            ).reshape(-1),
            predicted_change_signature=_tensor_from_any(
                payload.get("predicted_change_signature"),
                device=device,
            ).reshape(-1),
            appearance_signature=_tensor_from_any(
                payload.get("appearance_signature"),
                device=device,
            ).reshape(-1),
            behavior_label=(
                None
                if payload.get("behavior_label") is None
                else str(payload.get("behavior_label"))
            ),
            behavior_signature=_tensor_from_any(
                payload.get("behavior_signature"),
                device=device,
            ).reshape(-1),
            sender_id=None if payload.get("sender_id") is None else str(payload.get("sender_id")),
            sender_step_count=None
            if payload.get("sender_step_count") is None
            else int(payload.get("sender_step_count")),
            routing_scope=(
                None
                if payload.get("routing_scope") is None
                else str(payload.get("routing_scope"))
            ),
            target_sender_ids=[
                str(item)
                for item in (payload.get("target_sender_ids") or [])
                if item is not None
            ],
            ranked_hypotheses=[
                RankedHypothesisVote.from_dict(hypothesis)
                for hypothesis in (payload.get("ranked_hypotheses") or [])
                if isinstance(hypothesis, dict)
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "graph_id": self.latent_id,
            "latent_id": self.latent_id,
            "confidence": float(self.confidence),
            "residual": float(self.residual),
            "active_cells": self.active_cells.detach().cpu().float().numpy().copy(),
        }
        if self.chart_id is not None:
            payload["chart_id"] = self.chart_id
        if self.child_latent_ids:
            payload["child_graph_ids"] = list(self.child_latent_ids)
            payload["child_latent_ids"] = list(self.child_latent_ids)
        if self.predicted_appearance_signature.numel() > 0:
            payload["predicted_appearance_signature"] = (
                self.predicted_appearance_signature.detach().cpu().float().numpy().copy()
            )
        if self.predicted_change_signature.numel() > 0:
            payload["predicted_change_signature"] = (
                self.predicted_change_signature.detach().cpu().float().numpy().copy()
            )
        if self.appearance_signature.numel() > 0:
            payload["appearance_signature"] = (
                self.appearance_signature.detach().cpu().float().numpy().copy()
            )
        if self.behavior_label is not None:
            payload["behavior_label"] = self.behavior_label
        if self.behavior_signature.numel() > 0:
            payload["behavior_signature"] = (
                self.behavior_signature.detach().cpu().float().numpy().copy()
            )
        if self.sender_id is not None:
            payload["sender_id"] = self.sender_id
        if self.sender_step_count is not None:
            payload["sender_step_count"] = int(self.sender_step_count)
        if self.routing_scope is not None:
            payload["routing_scope"] = self.routing_scope
        if self.target_sender_ids:
            payload["target_sender_ids"] = list(self.target_sender_ids)
        if self.ranked_hypotheses:
            payload["ranked_hypotheses"] = [
                hypothesis.to_dict() for hypothesis in self.ranked_hypotheses
            ]
        return payload

    @property
    def graph_id(self) -> str | None:
        return self.latent_id

    @graph_id.setter
    def graph_id(self, value: str | None) -> None:
        self.latent_id = None if value is None else str(value)

    @property
    def child_graph_ids(self) -> list[str]:
        return list(self.child_latent_ids)

    @child_graph_ids.setter
    def child_graph_ids(self, values: list[str]) -> None:
        self.child_latent_ids = [str(value) for value in values if value is not None]


@dataclass
class PredictiveVoteMessage:
    sender_id: str | None
    sensed_pose_rel_body: np.ndarray
    active_cells: torch.Tensor = field(
        default_factory=lambda: torch.zeros(0, dtype=torch.float32)
    )
    appearance_signature: torch.Tensor = field(
        default_factory=lambda: torch.zeros(0, dtype=torch.float32)
    )
    behavior_label: str | None = None
    behavior_signature: torch.Tensor = field(
        default_factory=lambda: torch.zeros(0, dtype=torch.float32)
    )
    ranked_hypotheses: list[RankedHypothesisVote] = field(default_factory=list)
    possible_states: dict[str, list[Any]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PredictiveVoteMessage":
        pose = np.asarray(
            payload.get("sensed_pose_rel_body", np.zeros((4, 3), dtype=np.float64)),
            dtype=np.float64,
        )
        if pose.size == 0:
            pose = np.zeros((4, 3), dtype=np.float64)
        pose = pose.reshape(-1, 3)

        ranked_hypotheses = [
            RankedHypothesisVote.from_dict(hypothesis)
            for hypothesis in (payload.get("ranked_hypotheses") or [])
            if isinstance(hypothesis, dict)
        ]
        possible_states = {
            str(object_id): list(state_list or [])
            for object_id, state_list in (payload.get("possible_states") or {}).items()
            if object_id is not None
        }
        sender_id = payload.get("sender_id")

        return cls(
            sender_id=None if sender_id is None else str(sender_id),
            sensed_pose_rel_body=pose,
            active_cells=_tensor_from_any(
                payload.get("active_cells"),
                device=torch.device("cpu"),
            ).reshape(-1),
            appearance_signature=_tensor_from_any(
                payload.get("appearance_signature"),
                device=torch.device("cpu"),
            ).reshape(-1),
            behavior_label=(
                None
                if payload.get("behavior_label") is None
                else str(payload.get("behavior_label"))
            ),
            behavior_signature=_tensor_from_any(
                payload.get("behavior_signature"),
                device=torch.device("cpu"),
            ).reshape(-1),
            ranked_hypotheses=ranked_hypotheses,
            possible_states=possible_states,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "possible_states": copy.deepcopy(self.possible_states),
            "sensed_pose_rel_body": np.asarray(
                self.sensed_pose_rel_body,
                dtype=np.float64,
            ).copy(),
            "ranked_hypotheses": [
                hypothesis.to_dict() for hypothesis in self.ranked_hypotheses
            ],
            "sender_id": self.sender_id,
        }
        if self.active_cells.numel() > 0:
            payload["active_cells"] = (
                self.active_cells.detach().cpu().float().numpy().copy()
            )
        if self.appearance_signature.numel() > 0:
            payload["appearance_signature"] = (
                self.appearance_signature.detach().cpu().float().numpy().copy()
            )
        if self.behavior_label is not None:
            payload["behavior_label"] = self.behavior_label
        if self.behavior_signature.numel() > 0:
            payload["behavior_signature"] = (
                self.behavior_signature.detach().cpu().float().numpy().copy()
            )
        return payload


@dataclass
class TemporalState:
    trace_decay_rates: torch.Tensor
    trace_bank: torch.Tensor
    context_vector: torch.Tensor
    latest_embedding: torch.Tensor
    predicted_embedding: torch.Tensor
    predicted_appearance_signature: torch.Tensor
    predicted_change_signature: torch.Tensor
    mean_surprise: torch.Tensor
    boundary_pressure: torch.Tensor
    prediction_mismatch: torch.Tensor
    action_prediction_error: torch.Tensor
    appearance_prediction_error: torch.Tensor
    change_prediction_error: torch.Tensor
    dwell_steps: torch.Tensor
    event_detected: torch.Tensor
    current_latent_id: str | None = None
    known_latent_ids: list[str] = field(default_factory=list)

    @classmethod
    def empty(
        cls,
        *,
        context_dim: int,
        device: torch.device,
        trace_decay_rates: tuple[float, ...] = (0.2, 0.5, 0.8),
    ) -> "TemporalState":
        decay_values = [
            min(max(float(rate), 0.0), 0.999)
            for rate in trace_decay_rates
        ]
        if not decay_values:
            decay_values = [0.5]

        trace_count = len(decay_values)
        return cls(
            trace_decay_rates=torch.as_tensor(
                decay_values,
                dtype=torch.float32,
                device=device,
            ),
            trace_bank=_empty_f32((trace_count, context_dim), device),
            context_vector=_empty_f32((context_dim,), device),
            latest_embedding=_empty_f32((context_dim,), device),
            predicted_embedding=_empty_f32((context_dim,), device),
            predicted_appearance_signature=_empty_f32((0,), device),
            predicted_change_signature=_empty_f32((0,), device),
            mean_surprise=_empty_f32((1,), device),
            boundary_pressure=_empty_f32((1,), device),
            prediction_mismatch=_empty_f32((1,), device),
            action_prediction_error=_empty_f32((1,), device),
            appearance_prediction_error=_empty_f32((1,), device),
            change_prediction_error=_empty_f32((1,), device),
            dwell_steps=_empty_i64((1,), device),
            event_detected=_empty_f32((1,), device),
            current_latent_id=None,
            known_latent_ids=[],
        )

    @property
    def trace_norm(self) -> float:
        if self.context_vector.numel() == 0:
            return 0.0
        return float(self.context_vector.norm(p=2).item())

    @property
    def trace_bank_norms(self) -> list[float]:
        if self.trace_bank.numel() == 0:
            return []
        return [
            float(value)
            for value in self.trace_bank.norm(dim=1, p=2).detach().cpu().tolist()
        ]

    @property
    def current_graph_id(self) -> str | None:
        return self.current_latent_id

    @current_graph_id.setter
    def current_graph_id(self, value: str | None) -> None:
        self.current_latent_id = None if value is None else str(value)

    @property
    def known_states(self) -> list[str]:
        return list(self.known_latent_ids)

    @known_states.setter
    def known_states(self, values: list[str]) -> None:
        self.known_latent_ids = [str(value) for value in values if value is not None]

    @property
    def mean_surprise_value(self) -> float:
        return float(self.mean_surprise[0].item()) if self.mean_surprise.numel() else 0.0

    @property
    def boundary_pressure_value(self) -> float:
        return (
            float(self.boundary_pressure[0].item())
            if self.boundary_pressure.numel()
            else 0.0
        )

    @property
    def prediction_mismatch_value(self) -> float:
        return (
            float(self.prediction_mismatch[0].item())
            if self.prediction_mismatch.numel()
            else 0.0
        )

    @property
    def action_prediction_error_value(self) -> float:
        return (
            float(self.action_prediction_error[0].item())
            if self.action_prediction_error.numel()
            else 0.0
        )

    @property
    def appearance_prediction_error_value(self) -> float:
        return (
            float(self.appearance_prediction_error[0].item())
            if self.appearance_prediction_error.numel()
            else 0.0
        )

    @property
    def change_prediction_error_value(self) -> float:
        return (
            float(self.change_prediction_error[0].item())
            if self.change_prediction_error.numel()
            else 0.0
        )

    @property
    def dwell_step_count(self) -> int:
        return int(self.dwell_steps[0].item()) if self.dwell_steps.numel() else 0

    @property
    def event_detected_bool(self) -> bool:
        return bool(self.event_detected[0].item() >= 0.5) if self.event_detected.numel() else False


@dataclass
class MemorySlots:
    object_ids: list[str] = field(default_factory=list)
    slot_embeddings: torch.Tensor | None = None
    slot_object_indices: torch.Tensor | None = None
    slot_counts: torch.Tensor | None = None
    slot_chart_ids: list[str] = field(default_factory=list)

    @classmethod
    def empty(
        cls,
        *,
        embedding_dim: int,
        device: torch.device,
    ) -> "MemorySlots":
        return cls(
            object_ids=[],
            slot_embeddings=_empty_f32((0, embedding_dim), device),
            slot_object_indices=_empty_i64((0,), device),
            slot_counts=_empty_f32((0,), device),
            slot_chart_ids=[],
        )

    @property
    def num_slots(self) -> int:
        return int(0 if self.slot_embeddings is None else self.slot_embeddings.shape[0])

    @property
    def latent_ids(self) -> list[str]:
        return list(self.object_ids)

    @latent_ids.setter
    def latent_ids(self, value: list[str]) -> None:
        self.object_ids = [str(item) for item in value]

    @property
    def slot_latent_indices(self) -> torch.Tensor | None:
        return self.slot_object_indices

    @slot_latent_indices.setter
    def slot_latent_indices(self, value: torch.Tensor | None) -> None:
        self.slot_object_indices = value


@dataclass
class HopfieldRetrievalState:
    object_ids: list[str] = field(default_factory=list)
    slot_chart_ids: list[str] = field(default_factory=list)
    best_chart_ids: list[str | None] = field(default_factory=list)
    query: torch.Tensor | None = None
    retrieved: torch.Tensor | None = None
    settled: torch.Tensor | None = None
    slot_similarities: torch.Tensor | None = None
    slot_attention: torch.Tensor | None = None
    slot_object_indices: torch.Tensor | None = None
    top_slot_indices: torch.Tensor | None = None
    object_scores: torch.Tensor | None = None
    energies: torch.Tensor | None = None
    iterations: torch.Tensor | None = None

    @classmethod
    def empty(
        cls,
        *,
        embedding_dim: int,
        device: torch.device,
    ) -> "HopfieldRetrievalState":
        return cls(
            object_ids=[],
            slot_chart_ids=[],
            best_chart_ids=[],
            query=_empty_f32((embedding_dim,), device),
            retrieved=_empty_f32((embedding_dim,), device),
            settled=_empty_f32((embedding_dim,), device),
            slot_similarities=_empty_f32((0,), device),
            slot_attention=_empty_f32((0,), device),
            slot_object_indices=_empty_i64((0,), device),
            top_slot_indices=_empty_i64((0,), device),
            object_scores=_empty_f32((0,), device),
            energies=_empty_f32((0,), device),
            iterations=_empty_i64((1,), device),
        )

    @property
    def iteration_count(self) -> int:
        return int(self.iterations[0].item()) if self.iterations is not None and self.iterations.numel() else 0

    @property
    def energy_trace(self) -> list[float]:
        if self.energies is None:
            return []
        return [float(value) for value in self.energies.detach().cpu().reshape(-1).tolist()]

    @property
    def latent_ids(self) -> list[str]:
        return list(self.object_ids)

    @latent_ids.setter
    def latent_ids(self, value: list[str]) -> None:
        self.object_ids = [str(item) for item in value]

    @property
    def slot_latent_indices(self) -> torch.Tensor | None:
        return self.slot_object_indices

    @slot_latent_indices.setter
    def slot_latent_indices(self, value: torch.Tensor | None) -> None:
        self.slot_object_indices = value

    @property
    def latent_scores(self) -> torch.Tensor | None:
        return self.object_scores

    @latent_scores.setter
    def latent_scores(self, value: torch.Tensor | None) -> None:
        self.object_scores = value

    @property
    def top_object_id(self) -> str | None:
        if (
            not self.object_ids
            or self.object_scores is None
            or self.object_scores.numel() == 0
        ):
            return None
        index = int(torch.argmax(self.object_scores).item())
        return self.object_ids[index]

    @property
    def top_latent_id(self) -> str | None:
        return self.top_object_id

    @property
    def top_chart_id(self) -> str | None:
        if (
            not self.best_chart_ids
            or self.object_scores is None
            or self.object_scores.numel() == 0
        ):
            return None
        index = int(torch.argmax(self.object_scores).item())
        return self.best_chart_ids[index]


@dataclass
class HypothesisBankState:
    object_ids: list[str] = field(default_factory=list)
    chart_ids: list[str | None] = field(default_factory=list)
    scores: torch.Tensor | None = None
    weights: torch.Tensor | None = None
    locations: torch.Tensor | None = None
    pose_vectors: torch.Tensor | None = None
    appearance_vectors: torch.Tensor | None = None
    behavior_vectors: torch.Tensor | None = None
    behavior_labels: list[str | None] = field(default_factory=list)
    inferred_states: list[int | None] = field(default_factory=list)
    ages: torch.Tensor | None = None
    sources: list[str] = field(default_factory=list)

    @classmethod
    def empty(cls, *, device: torch.device) -> "HypothesisBankState":
        return cls(
            object_ids=[],
            chart_ids=[],
            scores=_empty_f32((0,), device),
            weights=_empty_f32((0,), device),
            locations=_empty_f32((0, 3), device),
            pose_vectors=_empty_f32((0, 3, 3), device),
            appearance_vectors=_empty_f32((0, 0), device),
            behavior_vectors=_empty_f32((0, 0), device),
            behavior_labels=[],
            inferred_states=[],
            ages=_empty_i64((0,), device),
            sources=[],
        )

    @property
    def size(self) -> int:
        return len(self.object_ids)

    @property
    def latent_ids(self) -> list[str]:
        return list(self.object_ids)

    @latent_ids.setter
    def latent_ids(self, value: list[str]) -> None:
        self.object_ids = [str(item) for item in value]
