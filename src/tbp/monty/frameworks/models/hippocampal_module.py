"""Hippocampal Module for tbp.monty.

Sits at the top of the LM heterarchy (no SM connections). Receives State outputs
from top-level EvidenceGraphLMs via lm_to_lm_matrix and implements five functions
inspired by the hippocampal formation:

1. Fast binding (Hebbian co-occurrence association between concepts)
2. Episodic memory (timestamped sequences of active concepts per episode)
3. Relational memory (typed graph of relations between concepts)
4. Global context signal (broadcast downward to all lower LMs)
5. Temporal prediction (cross-episode sequence learning and next-state prediction)

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
        initial_associations: list | None = None,
        initial_association_count: int = 10,
        temporal_prediction_weight: float = 0.5,
        temporal_dim: int = 512,
        temporal_sparsity: float = 0.04,
        hebbian_learning_rate: float = 1.0,
        surprise_learning_boost: float = 0.0,
    ):
        """Initialize the HippocampalModule.

        Args:
            context_dim: Dimensionality of the context vector.
            max_episodes: Maximum episodes to retain in episodic memory.
            min_confidence_to_bind: Minimum confidence for concept activation.
            learning_module_id: Identifier for this module.
            initial_associations: Optional list of [concept_a, concept_b]
                pairs to pre-load as co-occurrence associations. Enables
                HPC to provide useful context from the first episode.
                Example: [["053_mini_soccer_ball", "bouncy"],
                          ["011_banana", "edible"]]
            initial_association_count: Synthetic co-occurrence count per pair.
            temporal_prediction_weight: Weight for temporal predictions when
                merged into association_strengths for downstream LM priming.
                Higher values give temporal predictions more influence on
                evidence matching. Range [0, 1].
            temporal_dim: Dimensionality of sparse distributed representations
                (SDRs) used for temporal association.  Each concept gets a
                random sparse binary vector of this length.  Higher values
                increase associative capacity (~1/sparsity associations).
            temporal_sparsity: Fraction of active bits in each SDR.  Lower
                values increase capacity but reduce overlap-based similarity.
                Biological analog: ~2-5% of neurons active in a cortical
                minicolumn at any time.
            hebbian_learning_rate: Learning rate for the Hebbian outer-product
                update rule.  Controls how quickly temporal associations are
                formed.  Higher values enable one-shot learning.
        """
        self.context_dim = context_dim
        self.max_episodes = max_episodes
        self.min_confidence_to_bind = min_confidence_to_bind
        self.learning_module_id = learning_module_id
        self.temporal_prediction_weight = temporal_prediction_weight
        self.temporal_dim = temporal_dim
        self.temporal_sparsity = temporal_sparsity
        self.hebbian_learning_rate = hebbian_learning_rate
        # T6.3: Surprise-modulated cross-episode learning.
        # Modulated LR = hebbian_learning_rate * (1 + surprise_learning_boost * surprise).
        # Novel transitions (surprise=1.0) get amplified, predicted ones use base rate.
        self.surprise_learning_boost = surprise_learning_boost
        self.experiment_mode = None

        # GraphMatchingMonty compatibility attributes
        self.buffer = self._StubBuffer()
        # Use a non-terminal state so the HPC doesn't prematurely end
        # episodes.  "match" would count toward min_lms_match and cause
        # the episode to terminate at min_eval_steps even when the real
        # LM hasn't converged yet.
        self.terminal_state = None
        self.primary_target = None
        self.primary_target_rotation_quat = None
        self.detected_object = None
        self.detected_rotation_r = None

        self._init_memory()

        if initial_associations:
            pairs = [(a, b) for a, b in initial_associations]
            self.preload_associations(pairs, count=initial_association_count)

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

        # ---- State discrimination ----
        # Maps state-level concept IDs to their parent object identity.
        # E.g., {"stapler_open": "stapler", "stapler_closed": "stapler"}
        # Enables the HPC to reason about "same object, different state."
        self._state_to_object = {}
        # Reverse: object → set of known states
        self._object_states = {}

        # ---- Temporal tracking ----
        self._prev_concepts = []

        # ---- Cross-episode temporal prediction (SDR + Hebbian) ----
        # Sparse distributed representations for each concept.
        # Each concept gets a random sparse binary vector — the biological
        # equivalent of a cell assembly or SDR in a cortical minicolumn.
        self._concept_sdrs = {}
        # Hebbian temporal association matrix.  W[i,j] encodes the learned
        # association strength from SDR dimension j (source) to dimension i
        # (target).  Updated via outer-product Hebbian rule:
        #   W += η * outer(sdr_current, sdr_previous)
        # This is biologically analogous to synaptic strengthening between
        # neurons that fire in temporal sequence (STDP / Hebb's rule).
        self._temporal_W = np.zeros(
            (self.temporal_dim, self.temporal_dim), dtype=np.float32
        )
        # Action-conditioned association matrices.  Each action gets its own
        # weight matrix, so prediction can be conditioned on what action was
        # taken.  Analogous to the hippocampus binding motor sequences to
        # state transitions via separate synaptic pathways.
        self._action_W = {}
        # Terminal concept from the most recent completed episode
        self._last_episode_terminal = None
        # Temporal predictions for the current episode (computed in pre_episode)
        self._current_predictions = {}
        # Prediction accuracy tracking
        self._prediction_history = []
        # Action label for the current inter-episode transition
        self._pending_action = None

    # ------------------------------------------------------------------
    # LearningModule interface
    # ------------------------------------------------------------------

    def reset(self):
        self._init_memory()

    def pre_episode(self, primary_target=None):
        if self._current_episode_buffer:
            self.episodic_memory.append(list(self._current_episode_buffer))
        self._current_episode_buffer = []
        self._prev_concepts = []
        self.context_vector = np.zeros(self.context_dim, dtype=np.float32)
        self._active_concepts = []
        self._association_strengths = {}

        # Compute temporal predictions for the upcoming episode based on
        # the last episode's terminal concept.  These predictions prime
        # downstream LMs through the context signal.
        self._current_predictions = self._compute_temporal_predictions()
        if self._current_predictions:
            logger.info(
                f"[HPC] temporal predictions from "
                f"'{self._last_episode_terminal}': "
                f"{self._current_predictions}"
            )

    def post_episode(self):
        # Determine the terminal concept for this episode
        terminal = self._resolve_terminal_concept()

        # T6.3: Compute prediction surprise BEFORE updating weights,
        # so the surprise reflects how unexpected this transition was
        # given the model's state before learning.
        if terminal:
            surprise = self._get_prediction_surprise(terminal)
            self._record_episode_transition(terminal, surprise=surprise)
            self._validate_prediction(terminal)
            self._last_episode_terminal = terminal

        self._pending_action = None

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
    # GraphMatchingMonty compatibility stubs
    # ------------------------------------------------------------------
    # These attributes/methods are accessed by MontyForGraphMatching on ALL
    # LMs. HPC doesn't need graph matching, but must not crash the loop.

    class _StubBuffer:
        """Minimal buffer stub so lm.buffer.method() calls don't crash."""

        def __init__(self):
            self.on_object = []
            self.stats = {
                "time": [],
                "lm_processed_steps": [],
            }

        def get_num_observations_on_object(self):
            return 0

        def get_last_obs_processed(self):
            return False

        def update_stats(self, stats_dict, update_time=True, append=True):
            pass

        def update_last_stats_entry(self, stats):
            pass

        def get_num_matching_steps(self):
            return 0

        def reset(self):
            self.on_object = []
            self.stats = {}

        def __len__(self):
            return 0

    def add_lm_processing_to_buffer_stats(self, lm_processed):
        pass

    def update_terminal_condition(self):
        pass  # HPC is always in terminal "match" state

    def get_possible_matches(self):
        return []

    def set_individual_ts(self, terminal_state=None):
        pass

    def collect_stats_to_save(self):
        return {"possible_matches": self.get_possible_matches()}

    def get_all_known_object_ids(self):
        return []

    @property
    def stepwise_target_object(self):
        return None

    @stepwise_target_object.setter
    def stepwise_target_object(self, value):
        pass

    @property
    def stepwise_targets_list(self):
        if not hasattr(self, "_stepwise_targets_list"):
            self._stepwise_targets_list = []
        return self._stepwise_targets_list

    @stepwise_targets_list.setter
    def stepwise_targets_list(self, value):
        self._stepwise_targets_list = value

    # ------------------------------------------------------------------
    # Context signal
    # ------------------------------------------------------------------

    def get_context_signal(self) -> dict | None:
        """Return a dict describing the current hippocampal context.

        Returns None if there are no active concepts AND no temporal
        predictions (nothing to broadcast).

        Keys:
            context_vector: np.ndarray of shape (context_dim,)
            active_concepts: list[str]
            association_strengths: dict[str, float] - per-concept-pair
                associations for all concepts associated with the active ones,
                PLUS temporal predictions merged in with temporal_prediction_weight.
                Keys are concept IDs; values are max association strength
                across all active concepts.
            episode_count: int
            temporal_predictions: dict[str, float] - predicted next concepts
                with transition probabilities from cross-episode sequence memory.
        """
        if not self._active_concepts and not self._current_predictions:
            return None

        # Build per-pair association strengths: for each active concept,
        # collect its full association row so downstream LMs can look up
        # specific graph_id matches.
        assoc_strengths = {}
        self._rebuild_association_matrix()
        for concept in self._active_concepts:
            if concept in self._obj_to_idx:
                idx = self._obj_to_idx[concept]
                row = self.association_matrix[idx]
                for j, other_id in enumerate(self._idx_to_obj):
                    if j != idx and row[j] > 0:
                        # Take max across active concepts
                        assoc_strengths[other_id] = max(
                            assoc_strengths.get(other_id, 0.0),
                            float(row[j]),
                        )
            # Also include the active concept itself with a self-strength
            # so direct matches work in _apply_hippocampal_bias
            if concept not in assoc_strengths:
                assoc_strengths[concept] = self._association_strengths.get(
                    concept, 0.0
                )

        # Merge temporal predictions into association_strengths so that
        # downstream LMs automatically prime evidence for predicted objects
        # through the existing _apply_hippocampal_bias() pathway.
        # This is biologically analogous to hippocampal pre-play: the HPC
        # replays expected future states, priming cortical columns before
        # sensory input arrives.
        w = self.temporal_prediction_weight
        for concept_id, prob in self._current_predictions.items():
            weighted = prob * w
            assoc_strengths[concept_id] = max(
                assoc_strengths.get(concept_id, 0.0), weighted
            )

        return dict(
            context_vector=self.context_vector.copy(),
            active_concepts=list(self._active_concepts),
            association_strengths=assoc_strengths,
            episode_count=len(self.episodic_memory),
            temporal_predictions=dict(self._current_predictions),
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
            # Prefer graph_id (human-readable object name like "055_baseball")
            # over object_id (numeric hash used for similarity features).
            obj_id = state.non_morphological_features.get("graph_id")
            if obj_id is None:
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
    # Cross-episode temporal prediction (SDR + Hebbian)
    # ------------------------------------------------------------------

    def _resolve_terminal_concept(self) -> str | None:
        """Determine the terminal concept for the current episode.

        Prefers the current active concepts; falls back to the last step
        of the episode buffer.
        """
        if self._active_concepts:
            return self._active_concepts[0]
        if self._current_episode_buffer:
            last = self._current_episode_buffer[-1]
            concepts = last.get("active_concepts", [])
            if concepts:
                return concepts[0]
        return None

    def _get_concept_sdr(self, concept_id: str) -> np.ndarray:
        """Get or create a sparse distributed representation for a concept.

        Each concept gets a deterministic random sparse binary vector.
        Biologically, this represents the cell assembly (population code)
        that activates when this concept is recognized — analogous to
        a place cell ensemble for a specific location, or a concept cell
        assembly for a specific object.

        The SDR is seeded from the concept ID hash, making it deterministic
        and reproducible across sessions.

        Args:
            concept_id: String identifier for the concept.

        Returns:
            Binary numpy array of shape (temporal_dim,) with
            temporal_sparsity fraction of bits set to 1.
        """
        if concept_id not in self._concept_sdrs:
            n_active = max(1, int(self.temporal_dim * self.temporal_sparsity))
            # Use a hash that is stable across Python invocations
            # (built-in hash() is randomized by PYTHONHASHSEED).
            import hashlib
            h = int(hashlib.sha256(concept_id.encode()).hexdigest(), 16)
            rng = np.random.RandomState(h % (2**31))
            sdr = np.zeros(self.temporal_dim, dtype=np.float32)
            indices = rng.choice(
                self.temporal_dim, size=n_active, replace=False
            )
            sdr[indices] = 1.0
            self._concept_sdrs[concept_id] = sdr
        return self._concept_sdrs[concept_id]

    def _get_prediction_surprise(self, observed: str) -> float:
        """Compute surprise for a cross-episode transition.

        Returns a value in [0, 1] where 0 = perfectly predicted and
        1 = completely unexpected. Based on whether the observed concept
        was in the temporal predictions and its predicted probability.

        This is the cross-episode analog of TemporalMemory's within-episode
        surprise: both measure prediction error, but at different time scales
        (episodes vs. sensory steps).

        Args:
            observed: The concept ID actually observed.

        Returns:
            Surprise value in [0, 1].
        """
        if not self._current_predictions:
            return 1.0  # No prediction = max surprise
        if observed in self._current_predictions:
            # Surprise decreases with predicted probability.
            # P(observed) = 1.0 → surprise = 0.0 (perfectly predicted)
            # P(observed) = 0.1 → surprise = 0.9 (weakly predicted)
            return 1.0 - self._current_predictions[observed]
        return 1.0  # Concept not predicted at all

    def _hebbian_update(
        self, current_sdr: np.ndarray, previous_sdr: np.ndarray,
        action: str | None = None,
        surprise: float | None = None,
    ) -> None:
        """Update temporal association via Hebbian outer-product rule.

        Implements the biological learning rule: neurons that fire in
        temporal sequence strengthen their connections.  This is analogous
        to spike-timing-dependent plasticity (STDP) — the synapse from
        the previous state's cell assembly to the current state's cell
        assembly is strengthened.

        T6.3: When surprise_learning_boost > 0, the learning rate is
        modulated by prediction surprise: novel transitions (high surprise)
        get amplified learning, while predicted transitions use base rate.
        Analogous to neuromodulatory gating of STDP by dopamine/norepinephrine.

        W += η_modulated * outer(current_sdr, previous_sdr)

        After this update, W @ previous_sdr will have increased overlap
        with current_sdr, meaning the previous state now predicts the
        current state more strongly.

        Args:
            current_sdr: SDR of the state that was just observed.
            previous_sdr: SDR of the state that preceded it.
            action: If given, also update the action-conditioned matrix.
            surprise: Prediction surprise in [0, 1]. If provided and
                surprise_learning_boost > 0, modulates the learning rate.
        """
        # T6.3: Surprise-modulated learning rate
        if surprise is not None and self.surprise_learning_boost > 0:
            modulated_lr = self.hebbian_learning_rate * (
                1.0 + self.surprise_learning_boost * surprise
            )
        else:
            modulated_lr = self.hebbian_learning_rate

        update = modulated_lr * np.outer(current_sdr, previous_sdr)
        self._temporal_W += update

        action = action or self._pending_action or "episode_transition"
        if action not in self._action_W:
            self._action_W[action] = np.zeros(
                (self.temporal_dim, self.temporal_dim), dtype=np.float32
            )
        self._action_W[action] += update

    def _record_episode_transition(
        self, terminal: str, surprise: float | None = None,
    ) -> None:
        """Record a cross-episode transition via Hebbian learning.

        Updates the temporal association matrix so that the previous
        episode's terminal SDR now predicts the current terminal SDR
        through spreading activation.

        T6.3: Pass surprise to _hebbian_update for error-modulated learning.

        Args:
            terminal: The concept ID for this episode's terminal state.
            surprise: Prediction surprise in [0, 1], or None.
        """
        src = self._last_episode_terminal
        if not src:
            return

        src_sdr = self._get_concept_sdr(src)
        tgt_sdr = self._get_concept_sdr(terminal)
        self._hebbian_update(tgt_sdr, src_sdr, surprise=surprise)

        logger.debug(
            f"[HPC] Hebbian update: {src} → {terminal} "
            f"(action={self._pending_action or 'episode_transition'}, "
            f"surprise={surprise})"
        )

    def _spreading_activation(
        self, source_sdr: np.ndarray, W: np.ndarray
    ) -> dict[str, float]:
        """Predict next concepts via spreading activation through W.

        Multiplies the weight matrix by the source SDR to produce an
        activation pattern, then finds the known concepts whose SDRs
        have the highest overlap with this activation.  This is the
        biological mechanism: activation spreads through learned synaptic
        connections to pre-activate the predicted cell assembly.

        Args:
            source_sdr: SDR of the source concept.
            W: Association weight matrix to use.

        Returns:
            Dict mapping concept_id -> normalised activation (probability).
        """
        activation = W @ source_sdr

        results = {}
        for concept_id, sdr in self._concept_sdrs.items():
            overlap = float(np.dot(activation, sdr))
            if overlap > 0:
                results[concept_id] = overlap

        if not results:
            return {}

        total = sum(results.values())
        return {k: v / total for k, v in results.items()}

    def _validate_prediction(self, observed: str) -> None:
        """Compare temporal prediction with the actually observed concept."""
        if not self._current_predictions:
            return
        hit = observed in self._current_predictions
        top = max(
            self._current_predictions, key=self._current_predictions.get
        )
        self._prediction_history.append(dict(
            predicted=dict(self._current_predictions),
            observed=observed,
            hit=hit,
            top_prediction=top,
            top_correct=(top == observed),
        ))
        if hit:
            logger.info(
                f"[HPC] prediction HIT: predicted {top} "
                f"(p={self._current_predictions[top]:.2f}), "
                f"observed '{observed}'"
            )
        else:
            logger.info(
                f"[HPC] prediction MISS: predicted {top} "
                f"(p={self._current_predictions[top]:.2f}), "
                f"observed '{observed}'"
            )

    def _compute_temporal_predictions(
        self, from_concept: str | None = None, action: str | None = None
    ) -> dict[str, float]:
        """Compute P(next | from_concept) via spreading activation.

        The source concept's SDR is multiplied by the Hebbian association
        matrix.  The resulting activation pattern is compared against all
        known concept SDRs by dot product.  Concepts with positive overlap
        are returned as predictions, normalised to sum to 1.

        This is the biological prediction mechanism: activating the source
        cell assembly causes spreading activation through learned synapses,
        pre-activating the cell assemblies of predicted successor concepts.

        Args:
            from_concept: Source concept. Defaults to _last_episode_terminal.
            action: If given, use action-conditioned association matrix.

        Returns:
            Dict mapping concept_id -> probability (sums to 1.0).
        """
        src = from_concept or self._last_episode_terminal
        if not src:
            return {}

        source_sdr = self._get_concept_sdr(src)

        if action is not None:
            W = self._action_W.get(action)
            if W is None:
                return {}
        else:
            W = self._temporal_W

        return self._spreading_activation(source_sdr, W)

    def get_temporal_predictions(
        self, from_concept: str | None = None, action: str | None = None
    ) -> dict[str, float]:
        """Public API for temporal next-state predictions.

        Predicts what concept is likely to appear next via spreading
        activation through the Hebbian association matrix.  Optionally
        conditions on a specific action label.

        Args:
            from_concept: Source concept. Defaults to last episode's terminal.
            action: Optional action label for conditioned prediction.

        Returns:
            Dict mapping concept_id -> probability.
        """
        return self._compute_temporal_predictions(from_concept, action)

    def record_action(self, action_label: str) -> None:
        """Record the action associated with the current episode transition.

        Call between episodes to annotate the transition with an action
        label (e.g., "push", "rotate", "wait").  Without this, transitions
        are labeled "episode_transition" by default.

        Args:
            action_label: Human-readable action category string.
        """
        self._pending_action = action_label

    def get_prediction_accuracy(self) -> dict:
        """Return prediction accuracy statistics.

        Returns:
            Dict with keys:
                total: number of episodes with predictions
                hits: predictions where observed concept was in predicted set
                top_correct: predictions where top prediction was correct
                accuracy: hit rate (hits / total)
                top_accuracy: top-1 accuracy (top_correct / total)
                history: full prediction history list
        """
        total = len(self._prediction_history)
        hits = sum(1 for p in self._prediction_history if p["hit"])
        top_correct = sum(
            1 for p in self._prediction_history if p.get("top_correct")
        )
        return dict(
            total=total,
            hits=hits,
            top_correct=top_correct,
            accuracy=hits / total if total > 0 else 0.0,
            top_accuracy=top_correct / total if total > 0 else 0.0,
            history=list(self._prediction_history),
        )

    # ------------------------------------------------------------------
    # Episodic replay (offline consolidation)
    # ------------------------------------------------------------------

    def replay(
        self,
        n_replays: int = 1,
        learning_rate: float | None = None,
        reverse: bool = False,
    ) -> int:
        """Replay stored episodes to consolidate temporal associations.

        During rest/sleep the hippocampus replays stored episode sequences,
        re-firing cell assemblies in temporal order.  This strengthens
        Hebbian associations without new experience — consolidation.

        Forward replay strengthens the learned sequence order.  Reverse
        replay (goal→start) strengthens backward associations, which can
        improve planning efficiency.

        Args:
            n_replays: Number of times to replay each stored episode.
            learning_rate: Hebbian learning rate for replay updates.
                Defaults to ``self.hebbian_learning_rate``.  A lower rate
                provides gentler consolidation without overwriting.
            reverse: If True, replay episodes in reverse order (backward
                replay for planning consolidation).

        Returns:
            Number of Hebbian updates applied.
        """
        lr = learning_rate if learning_rate is not None \
            else self.hebbian_learning_rate
        old_lr = self.hebbian_learning_rate
        self.hebbian_learning_rate = lr

        n_updates = 0
        episodes = list(self.episodic_memory)

        for _ in range(n_replays):
            for episode in episodes:
                # Extract unique concept sequence from episode steps
                concepts = []
                for step in episode:
                    for c in step.get("active_concepts", []):
                        if not concepts or concepts[-1] != c:
                            concepts.append(c)

                if len(concepts) < 2:
                    continue

                if reverse:
                    concepts = list(reversed(concepts))

                # Re-apply Hebbian learning on consecutive pairs
                for i in range(len(concepts) - 1):
                    prev_sdr = self._get_concept_sdr(concepts[i])
                    curr_sdr = self._get_concept_sdr(concepts[i + 1])
                    # Use unconditioned update (replay doesn't know actions)
                    saved_action = self._pending_action
                    self._pending_action = None
                    self._hebbian_update(curr_sdr, prev_sdr)
                    self._pending_action = saved_action
                    n_updates += 1

        self.hebbian_learning_rate = old_lr
        logger.info(
            f"[HPC] replay: {n_updates} Hebbian updates "
            f"({'reverse' if reverse else 'forward'}, "
            f"lr={lr}, n_replays={n_replays})"
        )
        return n_updates

    def replay_cross_episode(
        self,
        n_replays: int = 1,
        learning_rate: float | None = None,
    ) -> int:
        """Replay cross-episode transitions to consolidate sequence memory.

        Instead of replaying within-episode steps, replays the sequence
        of terminal concepts across episodes — strengthening the cross-
        episode temporal model that drives prediction.

        This is the key consolidation for the temporal world model:
        re-experiencing the order in which objects were encountered.

        Args:
            n_replays: Number of times to replay the episode sequence.
            learning_rate: Hebbian learning rate for replay.

        Returns:
            Number of Hebbian updates applied.
        """
        lr = learning_rate if learning_rate is not None \
            else self.hebbian_learning_rate
        old_lr = self.hebbian_learning_rate
        self.hebbian_learning_rate = lr

        # Extract terminal concept from each episode
        terminals = []
        for episode in self.episodic_memory:
            if not episode:
                continue
            last_step = episode[-1]
            concepts = last_step.get("active_concepts", [])
            if concepts:
                terminals.append(concepts[0])

        n_updates = 0
        for _ in range(n_replays):
            for i in range(len(terminals) - 1):
                prev_sdr = self._get_concept_sdr(terminals[i])
                curr_sdr = self._get_concept_sdr(terminals[i + 1])
                saved_action = self._pending_action
                self._pending_action = None
                self._hebbian_update(curr_sdr, prev_sdr)
                self._pending_action = saved_action
                n_updates += 1

        self.hebbian_learning_rate = old_lr
        logger.info(
            f"[HPC] cross-episode replay: {n_updates} updates "
            f"from {len(terminals)} terminal concepts"
        )
        return n_updates

    # ------------------------------------------------------------------
    # State discrimination (T2.2)
    # ------------------------------------------------------------------

    def register_object_states(
        self, object_id: str, state_ids: list[str]
    ) -> None:
        """Declare that multiple concepts are states of the same object.

        This is the mechanism for state discrimination: the HPC learns
        that ``stapler_open`` and ``stapler_closed`` are the SAME object
        in different states.  Enables queries like "what states can this
        object be in?" and "given I see state X, what object is it?"

        Biologically, this corresponds to the hippocampus binding multiple
        sensory patterns (different views/states) to a single concept
        node in the cognitive map.

        Args:
            object_id: Parent object identity (e.g., "stapler").
            state_ids: List of state-level concept IDs
                (e.g., ["stapler_open", "stapler_closed"]).
        """
        for sid in state_ids:
            self._state_to_object[sid] = object_id
        if object_id not in self._object_states:
            self._object_states[object_id] = set()
        self._object_states[object_id].update(state_ids)

    def get_object_identity(self, concept_id: str) -> str:
        """Return the parent object identity for a state-level concept.

        If *concept_id* is a registered state (e.g., "stapler_open"),
        returns the parent object (e.g., "stapler").  Otherwise returns
        the concept_id itself (it IS the object identity).

        Args:
            concept_id: State-level or object-level concept ID.

        Returns:
            Object identity string.
        """
        return self._state_to_object.get(concept_id, concept_id)

    def get_object_states(self, object_id: str) -> set[str]:
        """Return all known states for an object.

        Args:
            object_id: Parent object identity.

        Returns:
            Set of state-level concept IDs, or empty set if unknown.
        """
        return set(self._object_states.get(object_id, set()))

    def get_current_state(self, object_id: str) -> str | None:
        """Return the most recently observed state for an object.

        Searches the current active concepts and recent episode history
        for the latest state of the given object.

        Args:
            object_id: Parent object identity.

        Returns:
            Most recent state-level concept ID, or None.
        """
        known_states = self._object_states.get(object_id, set())
        if not known_states:
            return None

        # Check current active concepts first
        for concept in self._active_concepts:
            if concept in known_states:
                return concept

        # Check last episode terminal
        if self._last_episode_terminal in known_states:
            return self._last_episode_terminal

        # Search episode buffer backward
        for step in reversed(self._current_episode_buffer):
            for c in step.get("active_concepts", []):
                if c in known_states:
                    return c

        return None

    def predict_state_after_action(
        self, object_id: str, action: str
    ) -> str | None:
        """Predict what state an object will be in after an action.

        Combines state discrimination with temporal prediction: finds the
        object's current state, then predicts the next state conditioned
        on the action.

        Args:
            object_id: Parent object identity.
            action: Action label.

        Returns:
            Predicted next state-level concept ID, or None.
        """
        current = self.get_current_state(object_id)
        if current is None:
            return None

        preds = self.get_temporal_predictions(
            from_concept=current, action=action
        )
        if not preds:
            return None

        # Prefer predictions that are known states of the same object
        known_states = self._object_states.get(object_id, set())
        state_preds = {
            k: v for k, v in preds.items() if k in known_states
        }

        if state_preds:
            return max(state_preds, key=state_preds.get)
        # Fall back to best overall prediction
        return max(preds, key=preds.get)

    # ------------------------------------------------------------------
    # Behavior recognition (T2.6 — temporal pattern matching)
    # ------------------------------------------------------------------

    def register_behavior(
        self, name: str, sequence: list[str],
        cyclic: bool = False,
    ) -> None:
        """Register a named behavior as a concept sequence template.

        A behavior is a temporal pattern — a sequence of concepts that
        occurs in a specific order.  Once registered, ``recognize_behavior``
        can match observed sequences against stored templates.

        Biologically, this corresponds to the hippocampus storing an
        event schema — a learned pattern of sequential activations that
        represents a familiar behavior (pouring, cutting, walking).

        Args:
            name: Behavior name (e.g., "pouring", "cutting").
            sequence: Ordered list of concept IDs that define the behavior.
            cyclic: If True, the sequence repeats (e.g., walking, cutting).
        """
        if not hasattr(self, "_behaviors"):
            self._behaviors = {}
        self._behaviors[name] = dict(
            sequence=list(sequence),
            cyclic=cyclic,
        )

    def recognize_behavior(
        self, observed: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Match an observed sequence against registered behavior templates.

        Computes a similarity score between the observed concept sequence
        and each registered behavior.  Uses subsequence matching — the
        observed sequence doesn't need to match the full template, just
        a contiguous portion.

        Biologically, this is pattern completion on temporal sequences:
        partial observation of a behavior activates the full template,
        enabling recognition before the behavior is complete.

        Args:
            observed: Sequence of concept IDs to match.  Defaults to the
                sequence of terminal concepts from recent episodes.

        Returns:
            List of ``(behavior_name, similarity)`` tuples, sorted by
            similarity (highest first).  Similarity is in [0, 1].
        """
        if not hasattr(self, "_behaviors") or not self._behaviors:
            return []

        if observed is None:
            observed = self._extract_recent_sequence()

        if not observed:
            return []

        results = []
        for name, info in self._behaviors.items():
            template = info["sequence"]
            sim = self._sequence_similarity(observed, template, info["cyclic"])
            if sim > 0:
                results.append((name, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def _extract_recent_sequence(self) -> list[str]:
        """Extract the sequence of terminal concepts from recent episodes."""
        terminals = []
        for episode in self.episodic_memory:
            if not episode:
                continue
            last = episode[-1]
            concepts = last.get("active_concepts", [])
            if concepts:
                c = concepts[0]
                if not terminals or terminals[-1] != c:
                    terminals.append(c)
        # Also include current episode if active
        if self._last_episode_terminal:
            c = self._last_episode_terminal
            if not terminals or terminals[-1] != c:
                terminals.append(c)
        return terminals

    def _sequence_similarity(
        self, observed: list[str], template: list[str], cyclic: bool
    ) -> float:
        """Compute similarity between observed sequence and template.

        Uses longest contiguous subsequence matching.  For cyclic
        templates, the template is doubled to handle wrap-around.

        Returns:
            Similarity score in [0, 1].  1.0 = perfect match of full
            template.  Partial matches score proportionally.
        """
        if not observed or not template:
            return 0.0

        search_template = template + template if cyclic else template
        t_len = len(template)

        # Find longest contiguous match of observed within search_template
        best_match_len = 0
        for start in range(len(search_template)):
            match_len = 0
            for j in range(len(observed)):
                if (start + j < len(search_template)
                        and observed[j] == search_template[start + j]):
                    match_len += 1
                else:
                    break
            best_match_len = max(best_match_len, match_len)

        return best_match_len / t_len if t_len > 0 else 0.0

    # ------------------------------------------------------------------
    # Goal-directed planning (forward replay via spreading activation)
    # ------------------------------------------------------------------

    def plan_action_sequence(
        self, current: str, goal: str, max_depth: int = 10
    ) -> list[tuple[str, str]] | None:
        """Find an action sequence from *current* to *goal* via forward replay.

        Iteratively applies spreading activation to simulate forward
        trajectories from the current state.  At each step, picks the
        predicted concept closest to the goal (greedy heuristic) and
        records the best action for that transition.

        Biologically, this is hippocampal forward replay: the HPC
        sequentially activates learned state representations, propagating
        activation through the association matrix until the goal
        representation is reached.

        Args:
            current: Current concept/state.
            goal: Desired concept/state.
            max_depth: Maximum replay steps before giving up.

        Returns:
            List of ``(action, resulting_state)`` tuples, or ``None``
            if the goal is not reachable within *max_depth* steps.
        """
        if current == goal:
            return []

        goal_sdr = self._get_concept_sdr(goal)
        visited = {current}
        state = current
        plan = []

        for _ in range(max_depth):
            preds = self._compute_temporal_predictions(from_concept=state)
            if not preds:
                return None

            # Pick predicted concept most similar to goal (greedy replay)
            best_concept = None
            best_score = -1.0
            for concept_id, prob in preds.items():
                if concept_id in visited and concept_id != goal:
                    continue
                concept_sdr = self._get_concept_sdr(concept_id)
                goal_sim = float(np.dot(concept_sdr, goal_sdr))
                score = prob * (1.0 + goal_sim)
                if score > best_score:
                    best_score = score
                    best_concept = concept_id

            if best_concept is None:
                return None

            action = self._best_action_from_activation(state, best_concept)
            plan.append((action, best_concept))

            if best_concept == goal:
                return plan

            visited.add(best_concept)
            state = best_concept

        return None

    def _best_action_from_activation(
        self, source: str, target: str
    ) -> str:
        """Find the action whose matrix gives strongest source→target activation.

        For each action-conditioned weight matrix, compute the activation
        of the target SDR when spreading from the source SDR.  Return the
        action with the highest activation.

        Args:
            source: Source concept ID.
            target: Target concept ID.

        Returns:
            Action label string.
        """
        source_sdr = self._get_concept_sdr(source)
        target_sdr = self._get_concept_sdr(target)

        best_action = "episode_transition"
        best_score = 0.0

        for action, W in self._action_W.items():
            activation = W @ source_sdr
            score = float(np.dot(activation, target_sdr))
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def simulate_trajectory(
        self, current: str, actions: list[str]
    ) -> list[tuple[str | None, dict[str, float]]]:
        """Predict the state trajectory given a sequence of actions.

        Performs mental simulation via sequential spreading activation:
        starting from *current*, applies each action's association matrix
        in sequence.  Analogous to hippocampal vicarious trial and error
        (VTE) — rapidly simulating possible futures before committing
        to an action.

        Args:
            current: Starting concept/state.
            actions: Sequence of action labels to simulate.

        Returns:
            List of ``(predicted_state, probability_dict)`` tuples.
        """
        trajectory = []
        state = current

        for action in actions:
            preds = self._compute_temporal_predictions(
                from_concept=state, action=action
            )
            if not preds:
                trajectory.append((None, {}))
                break

            next_state = max(preds, key=preds.get)
            trajectory.append((next_state, dict(preds)))
            state = next_state

        return trajectory

    # ------------------------------------------------------------------
    # Manipulation via prediction (T2.7)
    # ------------------------------------------------------------------

    def suggest_action(
        self, goal_state: str, object_id: str | None = None,
    ) -> str | None:
        """Suggest an action to move toward a desired goal state.

        Uses the temporal world model to determine what action should be
        taken next: finds the current state, plans a path to the goal
        via forward replay, and returns the first action in the plan.

        This closes the perception→prediction→action loop:
        1. LM recognizes current state
        2. HPC plans route to goal state via temporal associations
        3. HPC suggests the next action
        4. Motor system executes the action
        5. New state is observed → loop repeats

        Biologically, this is the hippocampus driving goal-directed
        behavior through prospective coding: the HPC pre-activates the
        goal state, reverse-chains through learned associations to find
        the action that leads there, and biases motor output accordingly.

        Args:
            goal_state: Desired state-level concept ID.
            object_id: If given, uses state discrimination to find the
                current state of this object.  Otherwise uses the last
                episode's terminal concept.

        Returns:
            Action label string, or None if no plan is found.
        """
        if object_id is not None:
            current = self.get_current_state(object_id)
        else:
            current = self._last_episode_terminal

        if current is None:
            return None

        plan = self.plan_action_sequence(current, goal_state)
        if plan:
            return plan[0][0]  # first action in the plan
        return None

    def evaluate_action(
        self, action: str, object_id: str | None = None,
    ) -> dict[str, float]:
        """Predict the outcome of taking an action.

        Returns the probability distribution over possible next states.
        If *object_id* is given, filters predictions to states of that
        object.

        Args:
            action: Action label to evaluate.
            object_id: If given, filter predictions to known states.

        Returns:
            Dict mapping concept_id -> probability.
        """
        if object_id is not None:
            current = self.get_current_state(object_id)
        else:
            current = self._last_episode_terminal

        if current is None:
            return {}

        preds = self.get_temporal_predictions(
            from_concept=current, action=action
        )

        if object_id is not None:
            known = self._object_states.get(object_id, set())
            if known:
                filtered = {k: v for k, v in preds.items() if k in known}
                if filtered:
                    total = sum(filtered.values())
                    return {k: v / total for k, v in filtered.items()}

        return preds

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

    def preload_associations(
        self, pairs: list[tuple[str, str]], count: int = 10
    ) -> None:
        """Pre-load co-occurrence associations without running episodes.

        This seeds the association matrix with synthetic co-occurrence data,
        enabling the HPC to provide useful context signals before any
        episodes have been run. Used for Phase 3 (behavioral feature
        injection via pre-loaded associations).

        Args:
            pairs: List of (concept_a, concept_b) pairs to associate.
                Each pair will be recorded as if the two concepts were
                observed together ``count`` times.
            count: Number of synthetic co-occurrence events per pair.
        """
        for concept_a, concept_b in pairs:
            key = frozenset([concept_a, concept_b])
            self.co_occurrence_counts[key] = (
                self.co_occurrence_counts.get(key, 0) + count
            )
        self._assoc_dirty = True
        self._rebuild_association_matrix()

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
            # Temporal prediction state (SDR + Hebbian)
            concept_sdrs={
                k: v.tolist() for k, v in self._concept_sdrs.items()
            },
            temporal_W=self._temporal_W.tolist(),
            action_W={
                k: v.tolist() for k, v in self._action_W.items()
            },
            last_episode_terminal=self._last_episode_terminal,
            prediction_history=list(self._prediction_history),
            temporal_prediction_weight=self.temporal_prediction_weight,
            temporal_dim=self.temporal_dim,
            temporal_sparsity=self.temporal_sparsity,
            hebbian_learning_rate=self.hebbian_learning_rate,
            # State discrimination
            state_to_object=dict(self._state_to_object),
            object_states={
                k: list(v) for k, v in self._object_states.items()
            },
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

        # Restore temporal prediction state (SDR + Hebbian)
        self.temporal_dim = state_dict.get(
            "temporal_dim", self.temporal_dim
        )
        self.temporal_sparsity = state_dict.get(
            "temporal_sparsity", self.temporal_sparsity
        )
        self.hebbian_learning_rate = state_dict.get(
            "hebbian_learning_rate", self.hebbian_learning_rate
        )
        self._concept_sdrs = {
            k: np.array(v, dtype=np.float32)
            for k, v in state_dict.get("concept_sdrs", {}).items()
        }
        tw = state_dict.get("temporal_W")
        if tw is not None:
            self._temporal_W = np.array(tw, dtype=np.float32)
        else:
            self._temporal_W = np.zeros(
                (self.temporal_dim, self.temporal_dim), dtype=np.float32
            )
        self._action_W = {
            k: np.array(v, dtype=np.float32)
            for k, v in state_dict.get("action_W", {}).items()
        }
        self._last_episode_terminal = state_dict.get(
            "last_episode_terminal", None
        )
        self._prediction_history = state_dict.get(
            "prediction_history", []
        )
        self.temporal_prediction_weight = state_dict.get(
            "temporal_prediction_weight", self.temporal_prediction_weight
        )

        # Restore state discrimination
        self._state_to_object = state_dict.get("state_to_object", {})
        self._object_states = {
            k: set(v)
            for k, v in state_dict.get("object_states", {}).items()
        }
