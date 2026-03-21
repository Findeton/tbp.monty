"""Hippocampal Module for tbp.monty.

Sits at the top of the LM heterarchy (no SM connections). Receives State outputs
from top-level EvidenceGraphLMs via lm_to_lm_matrix and implements four functions
inspired by the hippocampal formation:

1. Fast binding (Hebbian co-occurrence association between concepts)
2. Episodic memory (timestamped sequences of active concepts per episode)
3. Relational memory (typed graph of relations between concepts)
4. Global context signal (broadcast downward to all lower LMs)

Connectivity example (2 level-0 LMs -> 1 level-1 LM -> HPC):
    lm_to_lm_matrix = [[], [], [0, 1], [2]]
    sm_to_lm_matrix  = [[0], [1], [], []]

The HippocampalModule is the last LM in topological order; it receives LM inputs,
performs its updates, and exposes get_context_signal() for MontyBase to broadcast
downward via _dispatch_context_signals().
"""

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING

import numpy as np

from tbp.monty.frameworks.models.abstract_monty_classes import LearningModule
from tbp.monty.frameworks.experiments.mode import ExperimentMode

if TYPE_CHECKING:
    from tbp.monty.context import RuntimeContext
    from tbp.monty.frameworks.models.states import GoalState

try:
    import networkx as nx
    _HAS_NETWORKX = True
except ImportError:
    _HAS_NETWORKX = False

__all__ = ["HippocampalModule"]

logger = logging.getLogger(__name__)


class HippocampalModule(LearningModule):
    """Hippocampal-inspired binding and memory module.

    Sits at the top of the LM heterarchy with no direct SM connections.
    Receives State objects from upstream LMs, extracts high-confidence concept
    identities, and maintains four memory systems.

    Args:
        context_dim: Dimensionality of the context vector broadcast downward.
        max_episodes: Maximum number of episodes to retain in episodic memory.
        min_confidence_to_bind: Minimum LM confidence threshold for a concept to
            be considered "active" and included in binding/episodic updates.
        learning_module_id: Identifier string for this module.
    """

    def __init__(
        self,
        context_dim: int = 32,
        max_episodes: int = 500,
        min_confidence_to_bind: float = 0.5,
        learning_module_id: str = "hippocampal_module",
    ):
        self.context_dim = context_dim
        self.max_episodes = max_episodes
        self.min_confidence_to_bind = min_confidence_to_bind
        self.learning_module_id = learning_module_id
        self.experiment_mode = None

        self._init_memory()

    def _init_memory(self):
        # ---- Fast binding (Hebbian co-occurrence) ----
        # Maps frozenset(active_ids) -> count
        self.co_occurrence_counts = {}

        # Index bookkeeping for the association matrix
        self._obj_to_idx = {}
        self._idx_to_obj = []
        self.association_matrix = np.zeros((0, 0), dtype=np.float32)
        self._assoc_dirty = False

        # ---- Episodic memory ----
        # Bounded deque of episodes; each episode is a list of step-dicts
        self.episodic_memory = deque(maxlen=self.max_episodes)
        self._current_episode_buffer = []

        # ---- Relational memory (graph) ----
        if _HAS_NETWORKX:
            self.relational_graph = nx.DiGraph()
        else:
            # Fallback: plain dicts / lists
            self._nodes = {}
            self._edges = []

        # ---- Context signal ----
        self.context_vector = np.zeros(self.context_dim, dtype=np.float32)
        self._active_concepts = []
        self._association_strengths = {}

        # ---- Temporal tracking ----
        self._prev_concepts = []

    # ------------------------------------------------------------------
    # LearningModule interface
    # ------------------------------------------------------------------

    def reset(self):
        self._init_memory()

    def pre_episode(self):
        if self._current_episode_buffer:
            self.episodic_memory.append(list(self._current_episode_buffer))
        self._current_episode_buffer = []
        self._prev_concepts = []
        self.context_vector = np.zeros(self.context_dim, dtype=np.float32)
        self._active_concepts = []
        self._association_strengths = {}

    def post_episode(self):
        if self._current_episode_buffer:
            self.episodic_memory.append(list(self._current_episode_buffer))
        self._current_episode_buffer = []

    def set_experiment_mode(self, mode: ExperimentMode) -> None:
        self.experiment_mode = mode

    def matching_step(self, ctx: RuntimeContext, observations) -> None:
        # ---- 1. Extract active concepts from upstream LM states ----
        active_ids, locations = self._extract_active_concepts(observations)
        self._active_concepts = active_ids

        # ---- 2. Fast binding (only when >= 2 concepts co-active) ----
        if len(active_ids) >= 2:
            self._update_fast_binding(active_ids)

        # ---- 3. Episodic memory step ----
        self._record_episode_step(active_ids, ctx)

        # ---- 4. Relational graph ----
        self._update_relational_graph(active_ids, locations)

        # ---- 5. Context vector ----
        self.context_vector = self._compute_context_vector(active_ids)

        self._prev_concepts = list(active_ids)

        logger.debug(
            f"[HPC] active={active_ids}"
            f" co_occ_keys={len(self.co_occurrence_counts)}"
        )

    def exploratory_step(self, ctx: RuntimeContext, observations) -> None:
        self.matching_step(ctx, observations)

    def receive_votes(self, votes) -> None:
        return None

    def send_out_vote(self):
        return None

    def propose_goal_states(self) -> list[GoalState]:
        return []

    def get_output(self):
        return None

    # ------------------------------------------------------------------
    # Context signal
    # ------------------------------------------------------------------

    def get_context_signal(self) -> dict | None:
        """Return a dict describing the current hippocampal context.

        Returns None if there are no active concepts (nothing to broadcast).

        Keys:
            context_vector: np.ndarray of shape (context_dim,)
            active_concepts: list[str]
            association_strengths: dict[str, float]
            episode_count: int
        """
        if not self._active_concepts:
            return None

        return dict(
            context_vector=self.context_vector.copy(),
            active_concepts=list(self._active_concepts),
            association_strengths=dict(self._association_strengths),
            episode_count=len(self.episodic_memory),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_active_concepts(
        self, observations
    ) -> tuple[list[str], dict[str, np.ndarray]]:
        """Pull high-confidence object identities from upstream LM states.

        Args:
            observations: list of State objects forwarded from upstream LMs.

        Returns:
            (active_ids, locations) where *active_ids* is a deduplicated list of
            string concept identifiers and *locations* maps each id to a numpy
            location array when available.
        """
        if not observations:
            return ([], {})

        active_ids = []
        locations = {}

        for state in observations:
            if state is None or not state.use_state:
                continue
            if state.confidence < self.min_confidence_to_bind:
                continue
            obj_id = state.non_morphological_features.get("object_id")
            if obj_id is None:
                continue
            obj_id_str = str(obj_id)
            if obj_id_str not in active_ids:
                active_ids.append(obj_id_str)
                if state.location is not None:
                    locations[obj_id_str] = np.array(state.location)

        return (active_ids, locations)

    # ---- fast binding ----

    def _update_fast_binding(self, active_ids: list[str]) -> None:
        """Increment co-occurrence count for the current set of active concepts.

        Only called when len(active_ids) >= 2.
        """
        key = frozenset(active_ids)
        self.co_occurrence_counts[key] = (
            self.co_occurrence_counts.get(key, 0) + 1
        )
        self._assoc_dirty = True

    def _rebuild_association_matrix(self) -> None:
        """Rebuild the normalised association matrix from raw co-occurrence counts.

        Uses a cosine-like normalisation:
            A[i,j] = co_matrix[i,j] / sqrt(appearance[i] * appearance[j])
        where *appearance[i]* is the total number of co-occurrence events in
        which concept *i* participated.

        Skipped if the matrix is already up to date (``_assoc_dirty == False``).
        """
        if not self._assoc_dirty:
            return None

        # Collect the full vocabulary of concept ids seen so far
        all_ids = set()
        for key in self.co_occurrence_counts:
            all_ids.update(key)

        # Ensure every id has an index
        for obj_id in sorted(all_ids):
            if obj_id not in self._obj_to_idx:
                self._obj_to_idx[obj_id] = len(self._idx_to_obj)
                self._idx_to_obj.append(obj_id)

        n = len(self._idx_to_obj)

        # Build the raw co-occurrence matrix
        co_matrix = np.zeros((n, n), dtype=np.float32)
        for key, count in self.co_occurrence_counts.items():
            ids = list(key)
            for a in ids:
                for b in ids:
                    if a != b:
                        i, j = self._obj_to_idx[a], self._obj_to_idx[b]
                        co_matrix[i, j] += count

        # Compute per-concept appearance counts
        appearance = np.zeros(n, dtype=np.float32)
        for key, count in self.co_occurrence_counts.items():
            for obj_id in key:
                appearance[self._obj_to_idx[obj_id]] += count

        # Normalise
        norm = np.sqrt(np.outer(appearance, appearance))
        norm[norm == 0] = 1.0
        self.association_matrix = co_matrix / norm

        self._assoc_dirty = False

    # ---- episodic memory ----

    def _record_episode_step(
        self, active_ids: list[str], ctx: RuntimeContext
    ) -> None:
        """Append a timestamped record of the current active concepts."""
        global_step = ctx.timer.global_step if ctx.timer is not None else -1
        episode_step = ctx.timer.episode_step if ctx.timer is not None else -1
        self._current_episode_buffer.append(
            dict(
                global_step=global_step,
                episode_step=episode_step,
                active_concepts=list(active_ids),
            )
        )

    # ---- relational memory ----

    def _update_relational_graph(
        self, active_ids: list[str], locations: dict[str, np.ndarray]
    ) -> None:
        """Dispatch to networkx or fallback graph implementation."""
        if _HAS_NETWORKX:
            self._update_nx_graph(active_ids, locations)
        else:
            self._update_fallback_graph(active_ids, locations)

    def _update_nx_graph(
        self, active_ids: list[str], locations: dict[str, np.ndarray]
    ) -> None:
        """Update the relational DiGraph (networkx backend)."""
        # ---- Ensure all active concepts are nodes ----
        for obj_id in active_ids:
            if obj_id not in self.relational_graph:
                self.relational_graph.add_node(
                    obj_id, activation_count=0, last_seen_step=-1
                )
            self.relational_graph.nodes[obj_id]["activation_count"] += 1

        # ---- Co-occurrence edges ----
        for i, a in enumerate(active_ids):
            for b in active_ids[i + 1:]:
                for src, dst in ((a, b), (b, a)):
                    if self.relational_graph.has_edge(src, dst):
                        data = self.relational_graph[src][dst]
                        if data.get("type") == "cooccurrence":
                            data["count"] += 1
                        else:
                            # Edge exists but is a different type; bump count
                            data["count"] = data.get("count", 0) + 1
                    else:
                        self.relational_graph.add_edge(
                            src, dst,
                            type="cooccurrence", count=1, strength=0.0,
                        )

        # ---- Spatial edges (displacement between co-active concepts) ----
        for i, a in enumerate(active_ids):
            for b in active_ids[i + 1:]:
                if a in locations and b in locations:
                    displacement = locations[b] - locations[a]
                    for src, dst, disp in (
                        (a, b, displacement),
                        (b, a, -displacement),
                    ):
                        if self.relational_graph.has_edge(src, dst):
                            # Update running mean displacement
                            data = self.relational_graph[src][dst]
                            old = data.get(
                                "displacement", np.zeros(3)
                            )
                            count = data.get("spatial_count", 0)
                            data["displacement"] = (
                                (old * count + disp) / (count + 1)
                            )
                            data["spatial_count"] = count + 1
                        else:
                            self.relational_graph.add_edge(
                                src, dst,
                                type="spatial",
                                displacement=disp,
                                spatial_count=1,
                            )

        # ---- Temporal edges (lag-1 transitions from previous step) ----
        for prev in self._prev_concepts:
            for curr in active_ids:
                if prev != curr:
                    if self.relational_graph.has_edge(prev, curr):
                        data = self.relational_graph[prev][curr]
                        if data.get("type") == "temporal":
                            data["count"] = data.get("count", 0) + 1
                        else:
                            pass  # already handled above
                    else:
                        self.relational_graph.add_edge(
                            prev, curr,
                            type="temporal", lag=1, count=1,
                        )

    def _update_fallback_graph(
        self, active_ids: list[str], locations: dict[str, np.ndarray]
    ) -> None:
        """Update relational memory using plain dicts/lists (no networkx)."""
        # ---- Nodes ----
        for obj_id in active_ids:
            if obj_id not in self._nodes:
                self._nodes[obj_id] = {"activation_count": 0}
            self._nodes[obj_id]["activation_count"] += 1

        # ---- Co-occurrence edges ----
        for i, a in enumerate(active_ids):
            for b in active_ids[i + 1:]:
                self._upsert_edge(a, b, "cooccurrence", count=1)
                self._upsert_edge(b, a, "cooccurrence", count=1)

        # ---- Spatial edges ----
        for i, a in enumerate(active_ids):
            for b in active_ids[i + 1:]:
                if a in locations and b in locations:
                    disp = locations[b] - locations[a]
                    self._upsert_edge(a, b, "spatial", displacement=disp)
                    self._upsert_edge(b, a, "spatial", displacement=-disp)

        # ---- Temporal edges ----
        for prev in self._prev_concepts:
            for curr in active_ids:
                if prev != curr:
                    self._upsert_edge(prev, curr, "temporal", lag=1, count=1)

    def _upsert_edge(self, src: str, dst: str, edge_type: str, **attrs) -> None:
        """Insert or update an edge in the fallback edge list."""
        for edge in self._edges:
            if (
                edge["src"] == src
                and edge["dst"] == dst
                and edge["type"] == edge_type
            ):
                for k, v in attrs.items():
                    if k == "count":
                        edge["count"] = edge.get("count", 0) + v
                    elif k == "displacement":
                        old = edge.get("displacement", np.zeros(3))
                        c = edge.get("count", 1)
                        edge["displacement"] = (old * c + v) / (c + 1)
                        edge["count"] = c + 1
                    else:
                        edge[k] = v
                return None

        # No matching edge found -- create a new one
        entry = dict(src=src, dst=dst, type=edge_type)
        entry.update(attrs)
        self._edges.append(entry)

    # ---- context vector ----

    def _compute_context_vector(
        self, active_ids: list[str]
    ) -> np.ndarray:
        """Compute a context vector by averaging association-matrix rows.

        For each active concept that appears in the association matrix its
        normalised row is projected into a vector of length ``context_dim``
        (truncated or zero-padded as needed).  The vectors are averaged to
        produce a single context vector.

        Also populates ``self._association_strengths`` with per-concept mean
        association values.
        """
        if not active_ids:
            return np.zeros(self.context_dim, dtype=np.float32)

        self._rebuild_association_matrix()
        n_known = len(self._idx_to_obj)

        vecs = []
        strengths = {}
        for obj_id in active_ids:
            if obj_id in self._obj_to_idx and n_known > 0:
                idx = self._obj_to_idx[obj_id]
                row = self.association_matrix[idx]
                vec = np.zeros(self.context_dim, dtype=np.float32)
                copy_len = min(n_known, self.context_dim)
                vec[:copy_len] = row[:copy_len]
                vecs.append(vec)
                strengths[obj_id] = (
                    float(np.mean(row)) if len(row) > 0 else 0.0
                )
            else:
                vec = np.zeros(self.context_dim, dtype=np.float32)
                vecs.append(vec)
                strengths[obj_id] = 0.0

        self._association_strengths = strengths

        result = np.mean(
            np.stack(vecs, axis=0), axis=0
        ).astype(np.float32)
        return result

    # ------------------------------------------------------------------
    # Public query helpers
    # ------------------------------------------------------------------

    def recall_episode(self, index: int) -> list[dict] | None:
        """Return a stored episode by index, or None if out of range.

        Args:
            index: Zero-based episode index into ``episodic_memory``.

        Returns:
            A list of step-dicts for that episode, or ``None``.
        """
        episodes = list(self.episodic_memory)
        if 0 <= index < len(episodes):
            return episodes[index]
        return None

    def recall_associations(self, concept_id: str) -> dict[str, float]:
        """Return association strengths from *concept_id* to every other concept.

        The returned dict maps other concept ids to normalised association
        scores from the ``association_matrix``.  Returns an empty dict if the
        concept has not been observed.
        """
        self._rebuild_association_matrix()
        if concept_id not in self._obj_to_idx:
            return {}
        idx = self._obj_to_idx[concept_id]
        row = self.association_matrix[idx]
        return {
            self._idx_to_obj[j]: float(row[j])
            for j in range(len(self._idx_to_obj))
            if j != idx
        }

    def get_relational_edges(
        self, concept_id: str | None = None, edge_type: str | None = None
    ) -> list[dict]:
        """Query relational edges, optionally filtered by concept or type.

        Args:
            concept_id: If given, only return edges where *concept_id* is
                either the source or the destination.
            edge_type: If given, only return edges whose ``type`` field
                matches (e.g. ``'cooccurrence'``, ``'temporal'``,
                ``'spatial'``).  For ``'spatial'``, edges are identified by
                the presence of a ``'displacement'`` key.

        Returns:
            A list of edge dicts, each containing at least ``src``, ``dst``,
            and ``type`` keys.
        """
        if _HAS_NETWORKX:
            edges = []
            for src, dst, data in self.relational_graph.edges(data=True):
                if concept_id is not None and concept_id not in (src, dst):
                    continue
                if edge_type is not None:
                    if edge_type == "spatial":
                        if "displacement" not in data:
                            continue
                    elif data.get("type") != edge_type:
                        continue
                edges.append({"src": src, "dst": dst, **data})
            return edges
        else:
            result = self._edges
            if concept_id is not None:
                result = [
                    e for e in result
                    if e["src"] == concept_id or e["dst"] == concept_id
                ]
            if edge_type is not None:
                result = [
                    e for e in result if e["type"] == edge_type
                ]
            return list(result)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        co_occ = {str(list(k)): v for k, v in self.co_occurrence_counts.items()}
        episodes = [list(ep) for ep in self.episodic_memory]
        current_ep = list(self._current_episode_buffer)

        if _HAS_NETWORKX:
            relational = dict(
                nodes=dict(self.relational_graph.nodes(data=True)),
                edges=[
                    {"src": u, "dst": v, **d}
                    for u, v, d in self.relational_graph.edges(data=True)
                ],
            )
        else:
            relational = dict(
                nodes=dict(self._nodes),
                edges=list(self._edges),
            )

        return dict(
            co_occurrence_counts=co_occ,
            obj_to_idx=dict(self._obj_to_idx),
            idx_to_obj=list(self._idx_to_obj),
            association_matrix=self.association_matrix.tolist(),
            episodic_memory=episodes,
            current_episode_buffer=current_ep,
            relational=relational,
            context_vector=self.context_vector.tolist(),
            active_concepts=list(self._active_concepts),
            prev_concepts=list(self._prev_concepts),
            context_dim=self.context_dim,
            max_episodes=self.max_episodes,
            min_confidence_to_bind=self.min_confidence_to_bind,
        )

    def load_state_dict(self, state_dict: dict) -> None:
        self.context_dim = state_dict.get("context_dim", self.context_dim)
        self.max_episodes = state_dict.get("max_episodes", self.max_episodes)
        self.min_confidence_to_bind = state_dict.get(
            "min_confidence_to_bind", self.min_confidence_to_bind
        )

        # Restore co-occurrence counts (keys are stringified lists -> frozenset)
        self.co_occurrence_counts = {
            frozenset(eval(k)): v
            for k, v in state_dict["co_occurrence_counts"].items()
        }

        self._obj_to_idx = state_dict["obj_to_idx"]
        self._idx_to_obj = state_dict["idx_to_obj"]
        self.association_matrix = np.array(
            state_dict["association_matrix"], dtype=np.float32
        )
        self._assoc_dirty = False

        # Restore episodic memory
        self.episodic_memory = deque(
            state_dict["episodic_memory"], maxlen=self.max_episodes
        )
        self._current_episode_buffer = state_dict["current_episode_buffer"]

        # Restore relational graph
        relational = state_dict.get(
            "relational", dict(nodes={}, edges=[])
        )
        if _HAS_NETWORKX:
            self.relational_graph = nx.DiGraph()
            for node, attrs in relational["nodes"].items():
                self.relational_graph.add_node(node, **attrs)
            for edge in relational["edges"]:
                src = edge.pop("src")
                dst = edge.pop("dst")
                self.relational_graph.add_edge(src, dst, **edge)
        else:
            self._nodes = relational["nodes"]
            self._edges = relational["edges"]

        # Restore context state
        self.context_vector = np.array(
            state_dict["context_vector"], dtype=np.float32
        )
        self._active_concepts = state_dict.get("active_concepts", [])
        self._prev_concepts = state_dict.get("prev_concepts", [])
