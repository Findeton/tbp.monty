from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from tbp.monty.frameworks.models.cortical_column_torch.dendrites import (
    SparseDendrites,
)
from tbp.monty.frameworks.models.predictive_hypothesis_torch.tensor_objects import (
    HopfieldRetrievalState,
    MemorySlots,
    ObservationField,
)


def _normalize_graph_id(graph_id: Any) -> str | None:
    if graph_id is None:
        return None

    normalized = str(graph_id)
    if normalized in {"", "unknown", "no_observations_yet", "None"}:
        return None
    return normalized


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


def _normalize_vector(
    vector: torch.Tensor,
    *,
    target_dim: int,
    device: torch.device,
) -> torch.Tensor:
    tensor = vector.detach().to(device=device, dtype=torch.float32).reshape(-1)
    if tensor.numel() != target_dim:
        tensor = _resize_vector(tensor, target_dim)
    norm = float(tensor.norm(p=2).item())
    if norm > 1e-8:
        tensor = tensor / norm
    return tensor


def _normalize_rows(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.numel() == 0:
        return matrix
    norms = matrix.norm(dim=1, p=2, keepdim=True).clamp_min(1e-8)
    return matrix / norms


class FixedSparseRecurrentMemory:
    """Fixed sparse recurrent memory with local patch motifs and recurrent context.

    The substrate uses a fixed cell population and sparse recurrent dendrites.
    Object identities and chart IDs remain in a lightweight outer readout layer
    so the rest of Track 14 can keep its current interfaces.
    """

    def __init__(
        self,
        learning_rate: float = 0.15,
        device: str = "cpu",
        max_slots_per_object: int = 8,
        insertion_similarity_threshold: float = 0.92,
        topk_score_pool: int = 2,
        embedding_dim: int = 256,
        substrate_cell_count: int | None = None,
        active_cell_sparsity: float = 0.08,
        detail_dim: int | None = None,
        settle_iters: int = 3,
        convergence_threshold: float = 1e-4,
        detail_weight: float = 0.60,
        recurrent_weight: float = 0.22,
        object_pattern_weight: float = 0.18,
        learning_object_bias_weight: float = 0.22,
        competitor_inhibition_weight: float = 0.35,
        pattern_separation_weight: float = 0.40,
        membrane_decay: float = 0.82,
        refractory_decay: float = 0.75,
        eligibility_decay: float = 0.90,
        beta: float = 8.0,
        seed: int = 42,
    ) -> None:
        self.learning_rate = float(max(min(learning_rate, 1.0), 1e-4))
        self.device = torch.device(device)
        self.max_slots_per_object = max(int(max_slots_per_object), 1)
        self.insertion_similarity_threshold = float(
            max(min(insertion_similarity_threshold, 0.999), -0.999)
        )
        self.topk_score_pool = max(int(topk_score_pool), 1)
        self._embedding_dim = max(int(embedding_dim), 1)
        self._detail_dim = max(int(detail_dim or max(self._embedding_dim, 64)), 16)
        self.n_cells = max(
            int(substrate_cell_count or max(self._embedding_dim * 6, 128)),
            32,
        )
        sparsity = float(max(min(active_cell_sparsity, 0.5), 0.01))
        self.active_cell_count = max(4, min(self.n_cells, int(round(self.n_cells * sparsity))))
        self.settle_iters = max(int(settle_iters), 1)
        self.convergence_threshold = float(max(convergence_threshold, 1e-8))
        self.detail_weight = float(max(detail_weight, 0.0))
        self.recurrent_weight = float(max(recurrent_weight, 0.0))
        self.object_pattern_weight = float(max(object_pattern_weight, 0.0))
        self.learning_object_bias_weight = float(
            max(learning_object_bias_weight, 0.0)
        )
        self.competitor_inhibition_weight = float(
            max(competitor_inhibition_weight, 0.0)
        )
        self.pattern_separation_weight = float(max(pattern_separation_weight, 0.0))
        self.membrane_decay = float(max(min(membrane_decay, 0.999), 0.0))
        self.refractory_decay = float(max(min(refractory_decay, 0.999), 0.0))
        self.eligibility_decay = float(max(min(eligibility_decay, 0.999), 0.0))
        self.beta = float(max(beta, 1e-3))
        self.seed = int(seed)
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(self.seed)

        # Stage tensor creation through locals before assigning onto self. On the
        # macOS/torch 1.13.1 runtime used for Track 14, the previous direct
        # constructor path could segfault during larger predictive-memory
        # initialization even though the underlying allocations were individually
        # stable.
        query_templates = self._normalized_random_matrix(
            self.n_cells,
            self._embedding_dim,
        )
        detail_templates = self._normalized_random_matrix(
            self.n_cells,
            self._detail_dim,
        )
        cell_usage_counts = torch.zeros(
            self.n_cells,
            dtype=torch.float32,
            device=self.device,
        )
        membrane = torch.zeros(
            self.n_cells,
            dtype=torch.float32,
            device=self.device,
        )
        refractory_trace = torch.zeros(
            self.n_cells,
            dtype=torch.float32,
            device=self.device,
        )
        eligibility_trace = torch.zeros(
            self.n_cells,
            dtype=torch.float32,
            device=self.device,
        )
        prev_active = torch.zeros(
            self.n_cells,
            dtype=torch.float32,
            device=self.device,
        )
        object_patterns = torch.zeros(
            (0, self.n_cells),
            dtype=torch.float32,
            device=self.device,
        )
        object_counts = torch.zeros(
            0,
            dtype=torch.float32,
            device=self.device,
        )
        assemblies = MemorySlots.empty(
            embedding_dim=self.n_cells,
            device=self.device,
        )
        assembly_query_prototypes = torch.zeros(
            (0, self._embedding_dim),
            dtype=torch.float32,
            device=self.device,
        )
        last_retrieval = HopfieldRetrievalState.empty(
            embedding_dim=self._embedding_dim,
            device=self.device,
        )
        dendrites = SparseDendrites(
            n_cells=self.n_cells,
            max_segments_per_cell=16,
            max_synapses_per_segment=24,
            activation_threshold=0.18,
            sigmoid_temp=0.10,
            connected_threshold=0.30,
            permanence_increment=0.06,
            permanence_decrement=0.015,
            initial_permanence=0.42,
            seed=self.seed,
            device=str(self.device),
        )

        self._query_templates = query_templates
        self._detail_templates = detail_templates
        self._cell_usage_counts = cell_usage_counts
        self._membrane = membrane
        self._refractory_trace = refractory_trace
        self._eligibility_trace = eligibility_trace
        self._prev_active = prev_active
        self._object_ids = []
        self._object_patterns = object_patterns
        self._object_counts = object_counts
        self._assemblies = assemblies
        self._assembly_query_prototypes = assembly_query_prototypes
        self._last_retrieval = last_retrieval
        self._dendrites = dendrites

    def _initialize_state_storage(self) -> None:
        # Stage tensor creation through locals before assigning onto self. On the
        # macOS/torch 1.13.1 runtime used for Track 14, the previous inline
        # constructor path could segfault during larger predictive-memory
        # initialization even though the underlying allocations were individually
        # stable.
        query_templates = self._normalized_random_matrix(
            self.n_cells,
            self._embedding_dim,
        )
        detail_templates = self._normalized_random_matrix(
            self.n_cells,
            self._detail_dim,
        )
        cell_usage_counts = torch.zeros(
            self.n_cells,
            dtype=torch.float32,
            device=self.device,
        )
        membrane = torch.zeros(
            self.n_cells,
            dtype=torch.float32,
            device=self.device,
        )
        refractory_trace = torch.zeros(
            self.n_cells,
            dtype=torch.float32,
            device=self.device,
        )
        eligibility_trace = torch.zeros(
            self.n_cells,
            dtype=torch.float32,
            device=self.device,
        )
        prev_active = torch.zeros(
            self.n_cells,
            dtype=torch.float32,
            device=self.device,
        )
        object_patterns = torch.zeros(
            (0, self.n_cells),
            dtype=torch.float32,
            device=self.device,
        )
        object_counts = torch.zeros(
            0,
            dtype=torch.float32,
            device=self.device,
        )
        assemblies = MemorySlots.empty(
            embedding_dim=self.n_cells,
            device=self.device,
        )
        assembly_query_prototypes = torch.zeros(
            (0, self._embedding_dim),
            dtype=torch.float32,
            device=self.device,
        )
        last_retrieval = HopfieldRetrievalState.empty(
            embedding_dim=self._embedding_dim,
            device=self.device,
        )
        dendrites = SparseDendrites(
            n_cells=self.n_cells,
            max_segments_per_cell=16,
            max_synapses_per_segment=24,
            activation_threshold=0.18,
            sigmoid_temp=0.10,
            connected_threshold=0.30,
            permanence_increment=0.06,
            permanence_decrement=0.015,
            initial_permanence=0.42,
            seed=self.seed,
            device=str(self.device),
        )

        self._query_templates = query_templates
        self._detail_templates = detail_templates
        self._cell_usage_counts = cell_usage_counts
        self._membrane = membrane
        self._refractory_trace = refractory_trace
        self._eligibility_trace = eligibility_trace
        self._prev_active = prev_active
        self._object_ids = []
        self._object_patterns = object_patterns
        self._object_counts = object_counts
        self._assemblies = assemblies
        self._assembly_query_prototypes = assembly_query_prototypes
        self._last_retrieval = last_retrieval
        self._dendrites = dendrites

    def _normalized_random_matrix(self, rows: int, cols: int) -> torch.Tensor:
        matrix = torch.randn(
            (rows, cols),
            generator=self._rng,
            dtype=torch.float32,
        )
        return _normalize_rows(matrix.to(self.device))

    def _effective_learning_rate(self, count: float, scale: float = 1.0) -> float:
        stabilized_count = max(float(count), 1.0)
        adapted = self.learning_rate / (stabilized_count**0.5)
        minimum = self.learning_rate * 0.25
        return float(max(minimum, min(self.learning_rate, adapted)) * float(scale))

    def _object_index(self, object_id: str) -> int:
        if object_id in self._object_ids:
            return self._object_ids.index(object_id)

        self._object_ids.append(object_id)
        self._object_patterns = torch.cat(
            [
                self._object_patterns,
                torch.zeros(
                    (1, self.n_cells),
                    dtype=torch.float32,
                    device=self.device,
                ),
            ],
            dim=0,
        )
        self._object_counts = torch.cat(
            [
                self._object_counts,
                torch.zeros(1, dtype=torch.float32, device=self.device),
            ],
            dim=0,
        )
        return len(self._object_ids) - 1

    def _object_assembly_indices(self, object_index: int) -> torch.Tensor:
        if self._assemblies.num_slots == 0 or self._assemblies.slot_object_indices is None:
            return torch.zeros(0, dtype=torch.int64, device=self.device)
        return torch.nonzero(
            self._assemblies.slot_object_indices == int(object_index),
            as_tuple=False,
        ).reshape(-1)

    def _next_chart_id(self, object_index: int) -> str:
        object_id = self._object_ids[object_index]
        next_index = int(self._object_assembly_indices(object_index).numel())
        return f"{object_id}#chart{next_index}"

    def _append_assembly(
        self,
        object_index: int,
        pattern: torch.Tensor,
        query: torch.Tensor,
    ) -> None:
        self._assemblies.slot_embeddings = torch.cat(
            [self._assemblies.slot_embeddings, pattern.reshape(1, -1)],
            dim=0,
        )
        self._assembly_query_prototypes = torch.cat(
            [self._assembly_query_prototypes, query.reshape(1, -1)],
            dim=0,
        )
        self._assemblies.slot_object_indices = torch.cat(
            [
                self._assemblies.slot_object_indices,
                torch.as_tensor([object_index], dtype=torch.int64, device=self.device),
            ],
            dim=0,
        )
        self._assemblies.slot_counts = torch.cat(
            [
                self._assemblies.slot_counts,
                torch.as_tensor([1.0], dtype=torch.float32, device=self.device),
            ],
            dim=0,
        )
        self._assemblies.slot_chart_ids.append(self._next_chart_id(object_index))

    def _best_assembly(
        self,
        pattern: torch.Tensor,
        slot_indices: torch.Tensor | None = None,
    ) -> tuple[int | None, float]:
        if self._assemblies.num_slots == 0:
            return None, -1.0

        if slot_indices is None:
            slot_indices = torch.arange(self._assemblies.num_slots, device=self.device)
        if int(slot_indices.numel()) == 0:
            return None, -1.0

        slot_patterns = self._assemblies.slot_embeddings.index_select(0, slot_indices)
        similarities = torch.matmul(slot_patterns, pattern)
        best_local_index = int(torch.argmax(similarities).item())
        return int(slot_indices[best_local_index].item()), float(
            similarities[best_local_index].item()
        )

    def _extract_patch_descriptors(
        self,
        observation_field: ObservationField | None,
    ) -> torch.Tensor:
        if observation_field is None or observation_field.cell_count <= 0:
            return torch.zeros(
                (0, self._detail_dim),
                dtype=torch.float32,
                device=self.device,
            )

        descriptors: list[torch.Tensor] = []
        for patch_index in range(observation_field.cell_count):
            support_patch = observation_field.support[patch_index]
            support_gate = support_patch.unsqueeze(-1)
            rgb_patch = observation_field.rgb[patch_index] * support_gate
            depth_patch = observation_field.depth[patch_index] * support_patch
            xyz_patch = observation_field.xyz[patch_index] * support_gate
            rgb_dx = observation_field.rgb[patch_index].diff(dim=0).abs().reshape(-1)
            rgb_dy = observation_field.rgb[patch_index].diff(dim=1).abs().reshape(-1)
            depth_dx = observation_field.depth[patch_index].diff(dim=0).abs().reshape(-1)
            depth_dy = observation_field.depth[patch_index].diff(dim=1).abs().reshape(-1)
            seed = torch.cat(
                [
                    rgb_patch.reshape(-1),
                    depth_patch.reshape(-1),
                    xyz_patch.reshape(-1),
                    support_patch.reshape(-1),
                    rgb_dx,
                    rgb_dy,
                    depth_dx,
                    depth_dy,
                    observation_field.uv[patch_index].reshape(-1),
                    observation_field.valid_cells[patch_index : patch_index + 1],
                    support_patch.mean().reshape(1),
                    support_patch.std(unbiased=False).reshape(1),
                ]
            )
            descriptors.append(
                _normalize_vector(
                    seed,
                    target_dim=self._detail_dim,
                    device=self.device,
                )
            )

        return torch.stack(descriptors, dim=0)

    def _detail_scores(
        self,
        patch_descriptors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if patch_descriptors.numel() == 0:
            return (
                torch.zeros(self.n_cells, dtype=torch.float32, device=self.device),
                torch.zeros(self.n_cells, dtype=torch.int64, device=self.device),
            )

        similarities = torch.matmul(patch_descriptors, self._detail_templates.t())
        # Torch 1.13.1 on this macOS runtime can segfault on Tensor.max(dim=0)
        # for these descriptor-derived similarity tensors even when the tensor is
        # finite and contiguous. topk/amax stay stable and preserve semantics.
        top_values, best_patch_indices = torch.topk(similarities, k=1, dim=0)
        max_scores = top_values.squeeze(0)
        best_patch_indices = best_patch_indices.squeeze(0)
        mean_scores = similarities.mean(dim=0)
        return (0.70 * max_scores) + (0.30 * mean_scores), best_patch_indices

    def _sparsify(self, scores: torch.Tensor) -> torch.Tensor:
        if scores.numel() == 0:
            return torch.zeros(0, dtype=torch.float32, device=self.device)

        k = min(self.active_cell_count, int(scores.numel()))
        top_values, top_indices = torch.topk(scores, k=k)
        active = torch.zeros_like(scores)
        clipped = F.relu(top_values)
        if float(clipped.abs().sum().item()) <= 1e-8:
            clipped = torch.ones_like(top_values)
        active.scatter_(0, top_indices, clipped)
        norm = float(active.norm(p=2).item())
        if norm > 1e-8:
            active = active / norm
        return active

    def _pattern_to_embedding(self, pattern: torch.Tensor) -> torch.Tensor:
        if pattern.numel() == 0:
            return torch.zeros(
                self._embedding_dim,
                dtype=torch.float32,
                device=self.device,
            )

        reconstructed = torch.mv(self._query_templates.t(), pattern)
        norm = float(reconstructed.norm(p=2).item())
        if norm > 1e-8:
            reconstructed = reconstructed / norm
        return reconstructed

    def _activity_state(
        self,
        query: torch.Tensor,
        observation_field: ObservationField | None,
        *,
        previous_active: torch.Tensor | None = None,
        membrane_state: torch.Tensor | None = None,
        object_bias: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        query_vector = _normalize_vector(
            query,
            target_dim=self._embedding_dim,
            device=self.device,
        )
        patch_descriptors = self._extract_patch_descriptors(observation_field)
        detail_scores, best_patch_indices = self._detail_scores(patch_descriptors)
        active_source = self._prev_active if previous_active is None else previous_active
        recurrent_scores = self._dendrites.predict(active_source)
        membrane_source = self._membrane if membrane_state is None else membrane_state
        feedforward_scores = torch.mv(self._query_templates, query_vector)

        combined = (
            (self.membrane_decay * membrane_source)
            + feedforward_scores
            + (self.detail_weight * detail_scores)
            + (self.recurrent_weight * recurrent_scores)
            - (0.25 * self._refractory_trace)
        )
        if object_bias is not None and object_bias.numel() == self.n_cells:
            combined = combined + (self.object_pattern_weight * object_bias)

        active_pattern = self._sparsify(combined)
        predicted_binary = (recurrent_scores >= 0.5).float()
        return {
            "query": query_vector,
            "patch_descriptors": patch_descriptors,
            "best_patch_indices": best_patch_indices,
            "feedforward_scores": feedforward_scores,
            "detail_scores": detail_scores,
            "recurrent_scores": recurrent_scores,
            "membrane": combined,
            "active_pattern": active_pattern,
            "predicted_binary": predicted_binary,
        }

    def _assembly_bias(self, pattern: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._assemblies.num_slots == 0:
            return (
                torch.zeros(self.n_cells, dtype=torch.float32, device=self.device),
                torch.zeros(0, dtype=torch.float32, device=self.device),
                torch.zeros(0, dtype=torch.float32, device=self.device),
            )

        similarities = torch.matmul(self._assemblies.slot_embeddings, pattern)
        attention = torch.softmax(self.beta * similarities, dim=0)
        assembly_bias = torch.mv(self._assemblies.slot_embeddings.t(), attention)
        norm = float(assembly_bias.norm(p=2).item())
        if norm > 1e-8:
            assembly_bias = assembly_bias / norm
        return assembly_bias, similarities, attention

    def _object_bias(self, query_pattern: torch.Tensor) -> torch.Tensor:
        if self._object_patterns.numel() == 0:
            return torch.zeros(self.n_cells, dtype=torch.float32, device=self.device)

        object_similarities = torch.matmul(self._object_patterns, query_pattern)
        attention = torch.softmax(self.beta * object_similarities, dim=0)
        object_bias = torch.mv(self._object_patterns.t(), attention)
        norm = float(object_bias.norm(p=2).item())
        if norm > 1e-8:
            object_bias = object_bias / norm
        return object_bias

    def _current_object_bias(self, object_index: int) -> torch.Tensor:
        if not (0 <= object_index < self._object_patterns.shape[0]):
            return torch.zeros(self.n_cells, dtype=torch.float32, device=self.device)

        object_pattern = self._object_patterns[object_index]
        if float(object_pattern.abs().sum().item()) <= 1e-8:
            return torch.zeros(self.n_cells, dtype=torch.float32, device=self.device)
        return object_pattern.detach().clone()

    def _competitor_object_bias(
        self,
        pattern: torch.Tensor,
        object_index: int,
    ) -> torch.Tensor:
        if self._object_patterns.shape[0] <= 1:
            return torch.zeros(self.n_cells, dtype=torch.float32, device=self.device)

        valid_mask = self._object_counts > 0
        if 0 <= object_index < valid_mask.shape[0]:
            valid_mask = valid_mask.clone()
            valid_mask[object_index] = False
        if not bool(valid_mask.any().item()):
            return torch.zeros(self.n_cells, dtype=torch.float32, device=self.device)

        competitor_patterns = self._object_patterns[valid_mask]
        similarities = torch.matmul(competitor_patterns, pattern)
        if similarities.numel() == 0 or float(torch.max(similarities).item()) <= 1e-8:
            return torch.zeros(self.n_cells, dtype=torch.float32, device=self.device)

        attention = torch.softmax(self.beta * similarities, dim=0)
        competitor_bias = torch.mv(competitor_patterns.t(), attention)
        norm = float(competitor_bias.norm(p=2).item())
        if norm > 1e-8:
            competitor_bias = competitor_bias / norm
        return competitor_bias

    def _competitor_assembly_bias(
        self,
        pattern: torch.Tensor,
        object_index: int,
    ) -> torch.Tensor:
        if self._assemblies.num_slots == 0 or self._assemblies.slot_object_indices is None:
            return torch.zeros(self.n_cells, dtype=torch.float32, device=self.device)

        competitor_mask = self._assemblies.slot_object_indices != int(object_index)
        if not bool(competitor_mask.any().item()):
            return torch.zeros(self.n_cells, dtype=torch.float32, device=self.device)

        competitor_assemblies = self._assemblies.slot_embeddings[competitor_mask]
        similarities = torch.matmul(competitor_assemblies, pattern)
        if similarities.numel() == 0 or float(torch.max(similarities).item()) <= 1e-8:
            return torch.zeros(self.n_cells, dtype=torch.float32, device=self.device)

        attention = torch.softmax(self.beta * similarities, dim=0)
        competitor_bias = torch.mv(competitor_assemblies.t(), attention)
        norm = float(competitor_bias.norm(p=2).item())
        if norm > 1e-8:
            competitor_bias = competitor_bias / norm
        return competitor_bias

    def _combined_competitor_bias(
        self,
        pattern: torch.Tensor,
        object_index: int,
    ) -> torch.Tensor:
        object_bias = self._competitor_object_bias(pattern, object_index)
        assembly_bias = self._competitor_assembly_bias(pattern, object_index)
        combined = object_bias + assembly_bias
        norm = float(combined.norm(p=2).item())
        if norm > 1e-8:
            combined = combined / norm
        return combined

    def _separate_pattern(
        self,
        pattern: torch.Tensor,
        object_index: int,
        *,
        sparsify: bool,
        competitor_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.pattern_separation_weight <= 0.0:
            return pattern.detach().clone()

        if competitor_bias is None:
            competitor_bias = self._combined_competitor_bias(pattern, object_index)
        if float(competitor_bias.abs().sum().item()) <= 1e-8:
            return pattern.detach().clone()

        separated = torch.relu(
            pattern - (self.pattern_separation_weight * competitor_bias)
        )
        if float(separated.abs().sum().item()) <= 1e-8:
            return pattern.detach().clone()
        if sparsify:
            return self._sparsify(separated)

        norm = float(separated.norm(p=2).item())
        if norm > 1e-8:
            separated = separated / norm
        return separated

    def _energy(self, pattern: torch.Tensor) -> torch.Tensor:
        if self._assemblies.num_slots == 0:
            recurrent = self._dendrites.predict(pattern)
            return -pattern.dot(recurrent)

        similarities = torch.matmul(self._assemblies.slot_embeddings, pattern)
        return -torch.logsumexp(self.beta * similarities, dim=0) + (0.5 * pattern.dot(pattern))

    def _aggregate_object_scores(
        self,
        query_pattern: torch.Tensor,
        settled_pattern: torch.Tensor,
        query_embedding: torch.Tensor,
    ) -> tuple[dict[str, float], dict[str, str | None], torch.Tensor]:
        if self._assemblies.num_slots == 0:
            empty_scores = torch.zeros(0, dtype=torch.float32, device=self.device)
            return {}, {}, empty_scores

        query_support = F.relu(
            torch.matmul(self._assemblies.slot_embeddings, query_pattern)
        )
        settled_support = F.relu(
            torch.matmul(self._assemblies.slot_embeddings, settled_pattern)
        )
        query_prototype_support = (
            F.relu(torch.matmul(self._assembly_query_prototypes, query_embedding))
            if self._assembly_query_prototypes.numel() > 0
            else torch.zeros(0, dtype=torch.float32, device=self.device)
        )

        scores: dict[str, float] = {}
        best_chart_ids: dict[str, str | None] = {}
        object_scores = torch.zeros(
            len(self._object_ids),
            dtype=torch.float32,
            device=self.device,
        )
        for object_index, object_id in enumerate(self._object_ids):
            mask = self._assemblies.slot_object_indices == int(object_index)
            if bool(mask.any().item()):
                slot_indices = torch.nonzero(mask, as_tuple=False).reshape(-1)
                object_query_support = query_support[mask]
                object_settled_support = settled_support[mask]
                object_query_prototype_support = (
                    query_prototype_support[mask]
                    if query_prototype_support.numel() > 0
                    else torch.zeros(
                        object_settled_support.shape[0],
                        dtype=torch.float32,
                        device=self.device,
                    )
                )
                topk = min(self.topk_score_pool, int(object_settled_support.numel()))
                top_values = torch.topk(object_settled_support, k=topk).values
                best_score = float(top_values[0].item())
                pooled_score = float(top_values.mean().item())
                query_bonus = float(object_query_support.max().item())
                best_query_prototype = float(
                    object_query_prototype_support.max().item()
                )
                best_local_index = int(
                    torch.argmax(
                        object_settled_support
                        + (0.20 * object_query_support)
                        + (0.35 * object_query_prototype_support)
                    ).item()
                )
                best_slot_index = int(slot_indices[best_local_index].item())
                best_chart_ids[object_id] = self._assemblies.slot_chart_ids[best_slot_index]
            else:
                best_score = 0.0
                pooled_score = 0.0
                query_bonus = 0.0
                best_query_prototype = 0.0
                best_chart_ids[object_id] = None

            object_pattern = self._object_patterns[object_index]
            if float(object_pattern.abs().sum().item()) > 1e-8:
                object_proto_support = max(
                    0.0,
                    float(object_pattern.dot(settled_pattern).item()),
                )
            else:
                object_proto_support = 0.0

            score = (
                (0.05 * best_score)
                + (0.90 * best_query_prototype)
                + (0.05 * object_proto_support)
            )
            scores[object_id] = float(score)
            object_scores[object_index] = float(score)

        return scores, best_chart_ids, object_scores

    def _update_cell_templates(
        self,
        query: torch.Tensor,
        patch_descriptors: torch.Tensor,
        best_patch_indices: torch.Tensor,
        active_pattern: torch.Tensor,
    ) -> None:
        active_indices = torch.nonzero(active_pattern, as_tuple=True)[0].tolist()
        if not active_indices:
            return

        for cell_index in active_indices:
            count = float(self._cell_usage_counts[cell_index].item())
            query_lr = self._effective_learning_rate(count)
            updated_query = (
                ((1.0 - query_lr) * self._query_templates[cell_index])
                + (query_lr * query)
            )
            self._query_templates[cell_index] = _normalize_vector(
                updated_query,
                target_dim=self._embedding_dim,
                device=self.device,
            )

            if patch_descriptors.numel() > 0:
                patch_index = int(best_patch_indices[cell_index].item())
                detail_lr = self._effective_learning_rate(count, scale=0.75)
                updated_detail = (
                    ((1.0 - detail_lr) * self._detail_templates[cell_index])
                    + (detail_lr * patch_descriptors[patch_index])
                )
                self._detail_templates[cell_index] = _normalize_vector(
                    updated_detail,
                    target_dim=self._detail_dim,
                    device=self.device,
                )

            self._cell_usage_counts[cell_index] = self._cell_usage_counts[cell_index] + 1.0

    def _update_object_pattern(self, object_index: int, active_pattern: torch.Tensor) -> None:
        count = float(self._object_counts[object_index].item())
        learning_rate = self._effective_learning_rate(count, scale=0.80)
        previous = self._object_patterns[object_index]
        updated = ((1.0 - learning_rate) * previous) + (learning_rate * active_pattern)
        updated = self._separate_pattern(updated, object_index, sparsify=False)
        norm = float(updated.norm(p=2).item())
        if norm > 1e-8:
            updated = updated / norm
        self._object_patterns[object_index] = updated
        self._object_counts[object_index] = self._object_counts[object_index] + 1.0

    def _update_assemblies(
        self,
        object_index: int,
        active_pattern: torch.Tensor,
        query: torch.Tensor,
        novelty_signal: float,
    ) -> None:
        same_object_slots = self._object_assembly_indices(object_index)
        effective_threshold = float(
            max(
                -0.999,
                min(
                    0.999,
                    self.insertion_similarity_threshold + (0.05 * float(np.clip(novelty_signal, 0.0, 1.0))),
                ),
            )
        )
        if int(same_object_slots.numel()) == 0:
            self._append_assembly(object_index, active_pattern.clone(), query.clone())
            return

        best_slot_index, best_similarity = self._best_assembly(active_pattern, same_object_slots)
        if (
            best_slot_index is None
            or (
                best_similarity < effective_threshold
                and int(same_object_slots.numel()) < self.max_slots_per_object
            )
        ):
            self._append_assembly(object_index, active_pattern.clone(), query.clone())
            return

        slot_count = float(self._assemblies.slot_counts[best_slot_index].item())
        learning_rate = self._effective_learning_rate(slot_count, scale=0.85)
        updated = (
            ((1.0 - learning_rate) * self._assemblies.slot_embeddings[best_slot_index])
            + (learning_rate * active_pattern)
        )
        norm = float(updated.norm(p=2).item())
        if norm > 1e-8:
            updated = updated / norm
        self._assemblies.slot_embeddings[best_slot_index] = updated
        updated_query = (
            ((1.0 - learning_rate) * self._assembly_query_prototypes[best_slot_index])
            + (learning_rate * query)
        )
        self._assembly_query_prototypes[best_slot_index] = _normalize_vector(
            updated_query,
            target_dim=self._embedding_dim,
            device=self.device,
        )
        self._assemblies.slot_counts[best_slot_index] = self._assemblies.slot_counts[best_slot_index] + 1.0

    def _post_observe_update(
        self,
        active_pattern: torch.Tensor,
        predicted_binary: torch.Tensor,
        membrane: torch.Tensor,
    ) -> None:
        active_binary = (active_pattern > 0.0).float()
        self._dendrites.learn(
            active_binary,
            self._prev_active,
            learning_rate=self.learning_rate,
        )
        self._dendrites.grow_for_unpredicted(
            active_binary,
            self._prev_active,
            predicted_binary,
        )
        self._eligibility_trace = (
            (self.eligibility_decay * self._eligibility_trace) + active_pattern
        )
        self._refractory_trace = (
            (self.refractory_decay * self._refractory_trace) + active_binary
        )
        self._membrane = membrane.detach().clone()
        self._prev_active = active_pattern.detach().clone()

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

        object_index = self._object_index(normalized_id)
        object_bias = self._current_object_bias(object_index)
        activity_state = self._activity_state(
            embedding,
            observation_field,
            object_bias=(
                object_bias
                if float(object_bias.abs().sum().item()) > 1e-8
                else None
            ),
        )
        competitor_bias = self._combined_competitor_bias(
            activity_state["active_pattern"],
            object_index,
        )
        learning_scores = activity_state["membrane"].detach().clone()
        if float(object_bias.abs().sum().item()) > 1e-8:
            learning_scores = learning_scores + (
                self.learning_object_bias_weight * object_bias
            )
        if float(competitor_bias.abs().sum().item()) > 1e-8:
            learning_scores = learning_scores - (
                self.competitor_inhibition_weight * competitor_bias
            )

        active_pattern = self._sparsify(learning_scores)
        active_pattern = self._separate_pattern(
            active_pattern,
            object_index,
            sparsify=True,
            competitor_bias=competitor_bias,
        )
        self._update_cell_templates(
            activity_state["query"],
            activity_state["patch_descriptors"],
            activity_state["best_patch_indices"],
            active_pattern,
        )
        self._update_object_pattern(object_index, active_pattern)
        self._update_assemblies(
            object_index,
            active_pattern,
            activity_state["query"],
            novelty_signal,
        )
        self._post_observe_update(
            active_pattern,
            activity_state["predicted_binary"],
            learning_scores,
        )

    def retrieve(
        self,
        embedding: torch.Tensor,
        *,
        store_as_last: bool = True,
        observation_field: ObservationField | None = None,
    ) -> HopfieldRetrievalState:
        if not self._object_ids:
            state = HopfieldRetrievalState.empty(
                embedding_dim=self._embedding_dim,
                device=self.device,
            )
            if store_as_last:
                self._last_retrieval = state
            return state

        initial_state = self._activity_state(
            embedding,
            observation_field,
            previous_active=torch.zeros_like(self._prev_active),
            membrane_state=torch.zeros_like(self._membrane),
        )
        query_pattern = initial_state["active_pattern"]
        settled_pattern = query_pattern.detach().clone()
        energies: list[float] = []
        iteration_count = 0

        for step in range(self.settle_iters):
            assembly_bias, _, _ = self._assembly_bias(settled_pattern)
            recurrent = self._dendrites.predict(settled_pattern)
            object_bias = self._object_bias(settled_pattern)
            updated_scores = (
                (0.45 * settled_pattern)
                + (0.25 * query_pattern)
                + (0.15 * assembly_bias)
                + (0.15 * object_bias)
                + (self.recurrent_weight * recurrent)
            )
            updated_pattern = self._sparsify(updated_scores)
            iteration_count = step + 1
            energies.append(float(self._energy(updated_pattern).item()))
            delta = float((updated_pattern - settled_pattern).norm(p=2).item())
            settled_pattern = updated_pattern
            if delta < self.convergence_threshold:
                break

        raw_similarities = (
            torch.matmul(self._assemblies.slot_embeddings, settled_pattern)
            if self._assemblies.num_slots > 0
            else torch.zeros(0, dtype=torch.float32, device=self.device)
        )
        attention = (
            torch.softmax(self.beta * raw_similarities, dim=0)
            if raw_similarities.numel() > 0
            else torch.zeros(0, dtype=torch.float32, device=self.device)
        )
        scores, best_chart_ids, object_scores = self._aggregate_object_scores(
            query_pattern,
            settled_pattern,
            initial_state["query"],
        )
        topk_slots = min(self.topk_score_pool, int(attention.numel()))
        top_slot_indices = (
            torch.topk(attention, k=topk_slots).indices
            if topk_slots > 0
            else torch.zeros(0, dtype=torch.int64, device=self.device)
        )
        settled_embedding = self._pattern_to_embedding(settled_pattern)
        state = HopfieldRetrievalState(
            object_ids=list(self._object_ids),
            slot_chart_ids=list(self._assemblies.slot_chart_ids),
            best_chart_ids=[best_chart_ids.get(object_id) for object_id in self._object_ids],
            query=initial_state["query"].detach().clone(),
            retrieved=settled_embedding.detach().clone(),
            settled=settled_embedding.detach().clone(),
            slot_similarities=raw_similarities.detach().clone(),
            slot_attention=attention.detach().clone(),
            slot_object_indices=self._assemblies.slot_object_indices.detach().clone(),
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

    def score(
        self,
        embedding: torch.Tensor,
        *,
        observation_field: ObservationField | None = None,
    ) -> dict[str, float]:
        retrieval_state = self.retrieve(
            embedding,
            store_as_last=True,
            observation_field=observation_field,
        )
        return self.retrieval_scores(retrieval_state)

    def latent_score(
        self,
        embedding: torch.Tensor,
        *,
        observation_field: ObservationField | None = None,
    ) -> dict[str, float]:
        return self.score(embedding, observation_field=observation_field)

    def get_all_known_object_ids(self) -> list[str]:
        return list(self._object_ids)

    def get_all_known_latent_ids(self) -> list[str]:
        return self.get_all_known_object_ids()

    def get_slot_state(self) -> MemorySlots:
        return MemorySlots(
            object_ids=list(self._object_ids),
            slot_embeddings=self._assemblies.slot_embeddings.detach().clone(),
            slot_object_indices=self._assemblies.slot_object_indices.detach().clone(),
            slot_counts=self._assemblies.slot_counts.detach().clone(),
            slot_chart_ids=list(self._assemblies.slot_chart_ids),
        )

    def get_last_retrieval_state(self) -> HopfieldRetrievalState:
        return self._last_retrieval

    def reset_episode_state(self) -> None:
        self._membrane.zero_()
        self._refractory_trace.zero_()
        self._eligibility_trace.zero_()
        self._prev_active.zero_()
        self._last_retrieval = HopfieldRetrievalState.empty(
            embedding_dim=self._embedding_dim,
            device=self.device,
        )

    def _reset_state(self) -> None:
        self._initialize_state_storage()

    def _load_from_legacy_slot_bank(self, state_dict: dict[str, Any]) -> None:
        self.learning_rate = float(state_dict.get("learning_rate", self.learning_rate))
        self.max_slots_per_object = int(state_dict.get("max_slots_per_object", self.max_slots_per_object))
        self.insertion_similarity_threshold = float(
            state_dict.get("insertion_similarity_threshold", self.insertion_similarity_threshold)
        )
        self.topk_score_pool = int(state_dict.get("topk_score_pool", self.topk_score_pool))
        self._reset_state()

        self._object_ids = [
            str(object_id)
            for object_id in state_dict.get(
                "latent_ids",
                state_dict.get("object_ids", []),
            )
        ]
        self._object_patterns = torch.zeros(
            (len(self._object_ids), self.n_cells),
            dtype=torch.float32,
            device=self.device,
        )
        self._object_counts = torch.zeros(
            len(self._object_ids),
            dtype=torch.float32,
            device=self.device,
        )

        slot_embeddings = state_dict.get("slot_embeddings")
        slot_object_indices = state_dict.get("slot_object_indices")
        slot_counts = state_dict.get("slot_counts")
        slot_chart_ids = [
            str(chart_id)
            for chart_id in (state_dict.get("slot_chart_ids") or [])
            if chart_id is not None
        ]
        if slot_embeddings is None or slot_object_indices is None:
            return

        legacy_embeddings = torch.as_tensor(
            slot_embeddings,
            dtype=torch.float32,
            device=self.device,
        )
        if legacy_embeddings.ndim == 1:
            legacy_embeddings = legacy_embeddings.reshape(1, -1)
        legacy_object_indices = torch.as_tensor(
            slot_object_indices,
            dtype=torch.int64,
            device=self.device,
        ).reshape(-1)
        legacy_counts = torch.as_tensor(
            slot_counts if slot_counts is not None else np.ones(len(legacy_object_indices)),
            dtype=torch.float32,
            device=self.device,
        ).reshape(-1)

        for slot_index in range(legacy_embeddings.shape[0]):
            object_index = int(legacy_object_indices[slot_index].item())
            if not (0 <= object_index < len(self._object_ids)):
                continue

            query = _normalize_vector(
                legacy_embeddings[slot_index],
                target_dim=self._embedding_dim,
                device=self.device,
            )
            active_pattern = self._sparsify(torch.mv(self._query_templates, query))
            self._update_cell_templates(
                query,
                torch.zeros((0, self._detail_dim), dtype=torch.float32, device=self.device),
                torch.zeros(self.n_cells, dtype=torch.int64, device=self.device),
                active_pattern,
            )
            self._assemblies.slot_embeddings = torch.cat(
                [self._assemblies.slot_embeddings, active_pattern.reshape(1, -1)],
                dim=0,
            )
            self._assembly_query_prototypes = torch.cat(
                [self._assembly_query_prototypes, query.reshape(1, -1)],
                dim=0,
            )
            self._assemblies.slot_object_indices = torch.cat(
                [
                    self._assemblies.slot_object_indices,
                    torch.as_tensor([object_index], dtype=torch.int64, device=self.device),
                ],
                dim=0,
            )
            self._assemblies.slot_counts = torch.cat(
                [
                    self._assemblies.slot_counts,
                    legacy_counts[slot_index : slot_index + 1],
                ],
                dim=0,
            )
            chart_id = (
                slot_chart_ids[slot_index]
                if slot_index < len(slot_chart_ids)
                else self._next_chart_id(object_index)
            )
            self._assemblies.slot_chart_ids.append(chart_id)

        for object_index in range(len(self._object_ids)):
            mask = self._assemblies.slot_object_indices == int(object_index)
            if not bool(mask.any().item()):
                continue
            patterns = self._assemblies.slot_embeddings[mask]
            counts = self._assemblies.slot_counts[mask].reshape(-1, 1)
            weighted = (patterns * counts).sum(dim=0)
            norm = float(weighted.norm(p=2).item())
            if norm > 1e-8:
                weighted = weighted / norm
            self._object_patterns[object_index] = weighted
            self._object_counts[object_index] = float(counts.sum().item())

    def state_dict(self) -> dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "max_slots_per_object": self.max_slots_per_object,
            "insertion_similarity_threshold": self.insertion_similarity_threshold,
            "topk_score_pool": self.topk_score_pool,
            "embedding_dim": self._embedding_dim,
            "memory_layout": "fixed_sparse_recurrent_v1",
            "substrate_cell_count": self.n_cells,
            "active_cell_count": self.active_cell_count,
            "detail_dim": self._detail_dim,
            "settle_iters": self.settle_iters,
            "convergence_threshold": self.convergence_threshold,
            "detail_weight": self.detail_weight,
            "recurrent_weight": self.recurrent_weight,
            "object_pattern_weight": self.object_pattern_weight,
            "learning_object_bias_weight": self.learning_object_bias_weight,
            "competitor_inhibition_weight": self.competitor_inhibition_weight,
            "pattern_separation_weight": self.pattern_separation_weight,
            "membrane_decay": self.membrane_decay,
            "refractory_decay": self.refractory_decay,
            "eligibility_decay": self.eligibility_decay,
            "beta": self.beta,
            "seed": self.seed,
            "object_ids": list(self._object_ids),
            "latent_ids": list(self._object_ids),
            "object_patterns": self._object_patterns.detach().cpu().numpy(),
            "object_counts": self._object_counts.detach().cpu().numpy(),
            "slot_embeddings": self._assemblies.slot_embeddings.detach().cpu().numpy(),
            "slot_query_prototypes": self._assembly_query_prototypes.detach().cpu().numpy(),
            "slot_object_indices": self._assemblies.slot_object_indices.detach().cpu().numpy(),
            "slot_counts": self._assemblies.slot_counts.detach().cpu().numpy(),
            "slot_chart_ids": list(self._assemblies.slot_chart_ids),
            "query_templates": self._query_templates.detach().cpu().numpy(),
            "detail_templates": self._detail_templates.detach().cpu().numpy(),
            "cell_usage_counts": self._cell_usage_counts.detach().cpu().numpy(),
            "membrane": self._membrane.detach().cpu().numpy(),
            "refractory_trace": self._refractory_trace.detach().cpu().numpy(),
            "eligibility_trace": self._eligibility_trace.detach().cpu().numpy(),
            "prev_active": self._prev_active.detach().cpu().numpy(),
            "dendrites": self._dendrites.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        memory_layout = str(state_dict.get("memory_layout", "shared_slot_bank_v2"))
        if memory_layout == "shared_slot_bank_v2":
            self._load_from_legacy_slot_bank(state_dict)
            self._last_retrieval = HopfieldRetrievalState.empty(
                embedding_dim=self._embedding_dim,
                device=self.device,
            )
            return

        self.learning_rate = float(state_dict.get("learning_rate", self.learning_rate))
        self.max_slots_per_object = int(state_dict.get("max_slots_per_object", self.max_slots_per_object))
        self.insertion_similarity_threshold = float(
            state_dict.get("insertion_similarity_threshold", self.insertion_similarity_threshold)
        )
        self.topk_score_pool = int(state_dict.get("topk_score_pool", self.topk_score_pool))
        self._embedding_dim = int(state_dict.get("embedding_dim", self._embedding_dim))
        self._detail_dim = int(state_dict.get("detail_dim", self._detail_dim))
        self.n_cells = int(state_dict.get("substrate_cell_count", self.n_cells))
        self.active_cell_count = int(state_dict.get("active_cell_count", self.active_cell_count))
        self.settle_iters = int(state_dict.get("settle_iters", self.settle_iters))
        self.convergence_threshold = float(
            state_dict.get("convergence_threshold", self.convergence_threshold)
        )
        self.detail_weight = float(state_dict.get("detail_weight", self.detail_weight))
        self.recurrent_weight = float(state_dict.get("recurrent_weight", self.recurrent_weight))
        self.object_pattern_weight = float(
            state_dict.get("object_pattern_weight", self.object_pattern_weight)
        )
        self.learning_object_bias_weight = float(
            state_dict.get(
                "learning_object_bias_weight",
                self.learning_object_bias_weight,
            )
        )
        self.competitor_inhibition_weight = float(
            state_dict.get(
                "competitor_inhibition_weight",
                self.competitor_inhibition_weight,
            )
        )
        self.pattern_separation_weight = float(
            state_dict.get(
                "pattern_separation_weight",
                self.pattern_separation_weight,
            )
        )
        self.membrane_decay = float(state_dict.get("membrane_decay", self.membrane_decay))
        self.refractory_decay = float(state_dict.get("refractory_decay", self.refractory_decay))
        self.eligibility_decay = float(state_dict.get("eligibility_decay", self.eligibility_decay))
        self.beta = float(state_dict.get("beta", self.beta))
        self.seed = int(state_dict.get("seed", self.seed))
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(self.seed)
        self._reset_state()

        query_templates = state_dict.get("query_templates")
        detail_templates = state_dict.get("detail_templates")
        cell_usage_counts = state_dict.get("cell_usage_counts")
        membrane = state_dict.get("membrane")
        refractory_trace = state_dict.get("refractory_trace")
        eligibility_trace = state_dict.get("eligibility_trace")
        prev_active = state_dict.get("prev_active")
        object_patterns = state_dict.get("object_patterns")
        object_counts = state_dict.get("object_counts")
        slot_embeddings = state_dict.get("slot_embeddings")
        slot_query_prototypes = state_dict.get("slot_query_prototypes")
        slot_object_indices = state_dict.get("slot_object_indices")
        slot_counts = state_dict.get("slot_counts")

        if query_templates is not None:
            self._query_templates = _normalize_rows(
                torch.as_tensor(query_templates, dtype=torch.float32, device=self.device)
            )
        if detail_templates is not None:
            self._detail_templates = _normalize_rows(
                torch.as_tensor(detail_templates, dtype=torch.float32, device=self.device)
            )
        if cell_usage_counts is not None:
            self._cell_usage_counts = torch.as_tensor(
                cell_usage_counts,
                dtype=torch.float32,
                device=self.device,
            ).reshape(-1)
        if membrane is not None:
            self._membrane = torch.as_tensor(
                membrane,
                dtype=torch.float32,
                device=self.device,
            ).reshape(-1)
        if refractory_trace is not None:
            self._refractory_trace = torch.as_tensor(
                refractory_trace,
                dtype=torch.float32,
                device=self.device,
            ).reshape(-1)
        if eligibility_trace is not None:
            self._eligibility_trace = torch.as_tensor(
                eligibility_trace,
                dtype=torch.float32,
                device=self.device,
            ).reshape(-1)
        if prev_active is not None:
            self._prev_active = torch.as_tensor(
                prev_active,
                dtype=torch.float32,
                device=self.device,
            ).reshape(-1)

        self._object_ids = [
            str(object_id)
            for object_id in state_dict.get(
                "latent_ids",
                state_dict.get("object_ids", []),
            )
        ]
        if object_patterns is not None:
            self._object_patterns = _normalize_rows(
                torch.as_tensor(object_patterns, dtype=torch.float32, device=self.device)
            )
        else:
            self._object_patterns = torch.zeros(
                (len(self._object_ids), self.n_cells),
                dtype=torch.float32,
                device=self.device,
            )
        if object_counts is not None:
            self._object_counts = torch.as_tensor(
                object_counts,
                dtype=torch.float32,
                device=self.device,
            ).reshape(-1)
        else:
            self._object_counts = torch.zeros(
                len(self._object_ids),
                dtype=torch.float32,
                device=self.device,
            )

        self._assemblies = MemorySlots.empty(embedding_dim=self.n_cells, device=self.device)
        if slot_embeddings is not None:
            self._assemblies.slot_embeddings = _normalize_rows(
                torch.as_tensor(slot_embeddings, dtype=torch.float32, device=self.device)
            )
        if slot_query_prototypes is not None:
            self._assembly_query_prototypes = _normalize_rows(
                torch.as_tensor(
                    slot_query_prototypes,
                    dtype=torch.float32,
                    device=self.device,
                )
            )
        elif self._assemblies.slot_embeddings.numel() > 0:
            reconstructed = [
                self._pattern_to_embedding(pattern)
                for pattern in self._assemblies.slot_embeddings
            ]
            self._assembly_query_prototypes = torch.stack(reconstructed, dim=0)
        else:
            self._assembly_query_prototypes = torch.zeros(
                (0, self._embedding_dim),
                dtype=torch.float32,
                device=self.device,
            )
        if slot_object_indices is not None:
            self._assemblies.slot_object_indices = torch.as_tensor(
                slot_object_indices,
                dtype=torch.int64,
                device=self.device,
            ).reshape(-1)
        if slot_counts is not None:
            self._assemblies.slot_counts = torch.as_tensor(
                slot_counts,
                dtype=torch.float32,
                device=self.device,
            ).reshape(-1)
        self._assemblies.slot_chart_ids = [
            str(chart_id)
            for chart_id in (state_dict.get("slot_chart_ids") or [])
            if chart_id is not None
        ]

        dendrite_state = state_dict.get("dendrites")
        if isinstance(dendrite_state, dict) and dendrite_state:
            self._dendrites.load_state_dict(dendrite_state)

        self._last_retrieval = HopfieldRetrievalState.empty(
            embedding_dim=self._embedding_dim,
            device=self.device,
        )