# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Unified cortical column with SDR-based spatial and temporal computation.

Implements the core computation of a neocortical column:

1. **Spatial pooling with learning**: Sensory input encoded as SDR activates
   a sparse set of minicolumns. Proximal dendrite permanences adapt via
   competitive Hebbian learning. Homeostatic boosting maintains stable sparsity.

2. **Contextual prediction**: Basal dendritic segments on cells within each
   minicolumn predict which cell should fire based on temporal context.

3. **Predicted vs burst firing**: If a cell in an active minicolumn was
   predicted, only that cell fires (confident). Otherwise all cells burst.

4. **Three-factor learning**: Dendritic segments grow and strengthen based
   on pre x post x modulator (burst=strong, predicted=maintenance).

5. **Attractor dynamics**: Recurrent connections between cells form attractor
   basins. After feedforward activation, a settling loop converges to the
   nearest learned pattern, providing pattern completion and noise robustness.

6. **Object recognition**: Settled patterns are matched against stored
   prototypes (attractor memory) or SDR snapshots (legacy mode).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from tbp.monty.frameworks.models.cortical_column.associative_memory import (
    HeteroAssociativeMemory,
)
from tbp.monty.frameworks.models.cortical_column.attractor_memory import (
    AttractorMemory,
)
from tbp.monty.frameworks.models.cortical_column.dendrites import DendriteSegments
from tbp.monty.frameworks.models.cortical_column.encoders import FeatureSDREncoder
from tbp.monty.frameworks.models.cortical_column.motor_prediction import (
    MotorPrediction,
)
from tbp.monty.frameworks.models.cortical_column.neuromodulators import (
    NeuromodulatoryState,
)
from tbp.monty.frameworks.models.cortical_column.recurrent import (
    RecurrentConnections,
)
from tbp.monty.frameworks.models.cortical_column.sdr_memory import SDRObjectMemory

logger = logging.getLogger(__name__)


class CorticalColumn:
    """Biologically plausible cortical column using SDR computation.

    Parameters
    ----------
    n_minicolumns : int
        Number of minicolumns (determines feedforward SDR width).
    n_cells_per_minicolumn : int
        Cells per minicolumn (determines context capacity).
    sparsity : float
        Target fraction of active minicolumns per step.
    encoder_kwargs : dict or None
        Arguments for FeatureSDREncoder.
    dendrite_kwargs : dict or None
        Arguments for DendriteSegments.
    memory_kwargs : dict or None
        Arguments for SDRObjectMemory.
    burst_learning_rate : float
        Dendritic learning rate for burst (unpredicted) firing.
    predicted_learning_rate : float
        Dendritic learning rate for predicted firing.
    evidence_decay : float
        Per-step decay factor for accumulated evidence (0 = no decay).
    sp_learning : bool
        Enable spatial pooler learning on proximal dendrites.
    sp_connected_threshold : float
        Permanence threshold for a proximal synapse to be "connected".
    sp_permanence_increment : float
        Permanence increment for active inputs on winning minicolumns.
    sp_permanence_decrement : float
        Permanence decrement for inactive inputs on winning minicolumns.
    boost_strength : float
        Strength of homeostatic boosting (0 = no boosting).
    seed : int
        Random seed.
    """

    def __init__(
        self,
        n_minicolumns: int = 2048,
        n_cells_per_minicolumn: int = 8,
        sparsity: float = 0.03,
        encoder_kwargs: dict = None,
        dendrite_kwargs: dict = None,
        apical_dendrite_kwargs: dict = None,
        memory_kwargs: dict = None,
        recurrent_kwargs: dict = None,
        burst_learning_rate: float = 0.1,
        predicted_learning_rate: float = 0.01,
        evidence_decay: float = 0.0,
        sp_learning: bool = True,
        sp_connected_threshold: float = 0.5,
        sp_permanence_increment: float = 0.05,
        sp_permanence_decrement: float = 0.02,
        boost_strength: float = 3.0,
        use_attractor: bool = False,
        use_apical: bool = False,
        continuous_plasticity: bool = False,
        novelty_threshold: float = 0.8,
        use_motor_prediction: bool = False,
        motor_prediction_kwargs: dict = None,
        use_neuromodulation: bool = False,
        use_weight_memory: bool = False,
        associative_memory_kwargs: dict = None,
        seed: int = 42,
    ):
        self.n_minicolumns = n_minicolumns
        self.n_cells_per_minicolumn = n_cells_per_minicolumn
        self.n_cells = n_minicolumns * n_cells_per_minicolumn
        self._sparsity = sparsity
        self._n_active_minicolumns = max(1, int(sparsity * n_minicolumns))
        self._burst_lr = burst_learning_rate
        self._predicted_lr = predicted_learning_rate
        self._evidence_decay = evidence_decay
        self._continuous_plasticity = continuous_plasticity
        self._novelty_threshold = novelty_threshold
        self._sp_learning = sp_learning
        self._sp_connected_threshold = sp_connected_threshold
        self._sp_increment = sp_permanence_increment
        self._sp_decrement = sp_permanence_decrement
        self._boost_strength = boost_strength
        self._use_attractor = use_attractor
        self._use_apical = use_apical
        self._rng = np.random.RandomState(seed)

        # Encoder
        enc_kwargs = encoder_kwargs or {}
        self._encoder = FeatureSDREncoder(seed=seed, **enc_kwargs)

        # Feedforward proximal dendrites (spatial pooler)
        self._init_feedforward(seed)

        # Basal dendritic segments (context prediction)
        dend_kwargs = dict(
            n_cells=self.n_cells,
            max_segments_per_cell=32,
            max_synapses_per_segment=24,
            activation_threshold=10,
            initial_permanence=0.5,
            connected_threshold=0.3,
            permanence_increment=0.05,
            permanence_decrement=0.02,
            seed=seed,
        )
        if dendrite_kwargs:
            dend_kwargs.update(dendrite_kwargs)
        self._dendrites = DendriteSegments(**dend_kwargs)

        # Apical dendritic segments (top-down modulation — Phase 2)
        if self._use_apical:
            apical_kwargs = dict(
                n_cells=self.n_cells,
                max_segments_per_cell=16,
                max_synapses_per_segment=20,
                activation_threshold=8,
                initial_permanence=0.5,
                connected_threshold=0.3,
                permanence_increment=0.04,
                permanence_decrement=0.02,
                seed=seed + 500,
            )
            if apical_dendrite_kwargs:
                apical_kwargs.update(apical_dendrite_kwargs)
            self._apical_dendrites = DendriteSegments(**apical_kwargs)
        else:
            self._apical_dendrites = None

        # Object memory (legacy SDR snapshots — always available)
        mem_kwargs = dict(
            location_overlap_threshold=0.3,
            feature_overlap_threshold=0.2,
            max_observations_per_object=500,
        )
        if memory_kwargs:
            mem_kwargs.update(memory_kwargs)
        self._memory = SDRObjectMemory(**mem_kwargs)

        # Recurrent connections + attractor memory (Phase 1)
        rec_kwargs = dict(
            n_cells=self.n_cells,
            n_minicolumns=n_minicolumns,
            n_cells_per_minicolumn=n_cells_per_minicolumn,
            sparsity=sparsity,
            seed=seed,
        )
        if recurrent_kwargs:
            rec_kwargs.update(recurrent_kwargs)
        self._recurrent = RecurrentConnections(**rec_kwargs)
        self._attractor_memory = AttractorMemory(n_cells=self.n_cells)

        # Hetero-associative memory (Phase 7a — weight-based recognition)
        self._use_weight_memory = use_weight_memory
        if use_weight_memory:
            am_kwargs = dict(
                n_cells=self.n_cells,
                n_input=self._encoder.total_bits,
                n_label_bits=256,
                n_active_label=10,
                learning_rate=0.01,
                decay_rate=0.0001,
                cell_weight=0.5,
                ff_weight=0.5,
                seed=seed,
            )
            if associative_memory_kwargs:
                am_kwargs.update(associative_memory_kwargs)
            self._associative_memory = HeteroAssociativeMemory(**am_kwargs)
        else:
            self._associative_memory = None

        # Motor-conditional prediction (Phase 4)
        self._use_motor_prediction = use_motor_prediction
        if use_motor_prediction:
            mp_kwargs = dict(
                location_bits=self._encoder._location_bits,
            )
            if motor_prediction_kwargs:
                mp_kwargs.update(motor_prediction_kwargs)
            self._motor_prediction = MotorPrediction(**mp_kwargs)
        else:
            self._motor_prediction = None
        self._prev_location = None  # for computing displacement

        # Neuromodulatory gating (Phase 5)
        self._use_neuromodulation = use_neuromodulation
        self._neuromodulators = NeuromodulatoryState() if use_neuromodulation else None
        self._surprise_ema = 0.5  # running average of surprise

        # --- Episode state (numpy arrays) ---
        self._prev_active_mask = np.zeros(self.n_cells, dtype=np.bool_)
        self._prev_winner_mask = np.zeros(self.n_cells, dtype=np.bool_)
        self._active_mask = np.zeros(self.n_cells, dtype=np.bool_)
        self._winner_mask = np.zeros(self.n_cells, dtype=np.bool_)
        self._predicted_mask = np.zeros(self.n_cells, dtype=np.bool_)
        self._apical_predicted_mask = np.zeros(self.n_cells, dtype=np.bool_)
        self._context_active_mask = np.zeros(self.n_cells, dtype=np.bool_)
        self._active_mc_mask = np.zeros(n_minicolumns, dtype=np.bool_)
        self._bursting_mc_mask = np.zeros(n_minicolumns, dtype=np.bool_)
        self._last_ff_overlap = None  # cached for settling loop
        self._last_surprise = 1.0  # lagged surprise for modulation (start high)
        self._mode = "eval"
        self._current_object = None
        self._step_count = 0
        self._evidence = {}

    # ------------------------------------------------------------------
    # Feedforward (proximal dendrite) initialization
    # ------------------------------------------------------------------

    def _init_feedforward(self, seed: int) -> None:
        """Initialize proximal dendrite connections and permanences.

        Each minicolumn has potential connections to a random ~10% of
        input bits. Initial permanences are drawn from a normal distribution
        centered around the connected threshold so that some start connected
        and others don't — learning refines which connections matter.
        """
        rng = np.random.RandomState(seed + 1000)
        n_input = self._encoder.total_bits
        n_potential = max(1, n_input // 10)

        self._ff_potential = np.zeros(
            (self.n_minicolumns, n_input), dtype=np.bool_
        )
        self._ff_permanences = np.zeros(
            (self.n_minicolumns, n_input), dtype=np.float32
        )

        for mc in range(self.n_minicolumns):
            indices = rng.choice(n_input, size=n_potential, replace=False)
            self._ff_potential[mc, indices] = True
            self._ff_permanences[mc, indices] = rng.normal(
                self._sp_connected_threshold, 0.05, size=n_potential
            ).clip(0.0, 1.0).astype(np.float32)

        # Boosting state
        self._boost_factors = np.ones(self.n_minicolumns, dtype=np.float32)
        self._mc_duty_cycle = np.full(
            self.n_minicolumns, self._sparsity, dtype=np.float32
        )

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def pre_episode(self, mode: str = "eval", object_name: str = None):
        """Reset state for a new episode."""
        self._mode = mode
        self._current_object = object_name
        self._prev_active_mask[:] = False
        self._prev_winner_mask[:] = False
        self._active_mask[:] = False
        self._winner_mask[:] = False
        self._predicted_mask[:] = False
        self._apical_predicted_mask[:] = False
        self._context_active_mask[:] = False
        self._active_mc_mask[:] = False
        self._bursting_mc_mask[:] = False
        self._last_ff_overlap = None
        self._last_surprise = 1.0
        self._prev_location = None
        self._surprise_ema = 0.5
        if self._motor_prediction is not None:
            self._motor_prediction.reset()
        if self._neuromodulators is not None:
            self._neuromodulators.reset()
        self._step_count = 0
        # Initialize evidence for all known objects
        if self._use_weight_memory and self._associative_memory is not None:
            known = self._associative_memory.known_objects
        else:
            known = self._memory.known_objects
        self._evidence = {obj: 0.0 for obj in known}

    def post_episode(self):
        """Finalize an episode."""
        if self._mode == "train" and self._current_object:
            if self._use_weight_memory and self._associative_memory is not None:
                logger.info(
                    "Learned object '%s' via associative memory "
                    "(total weight=%.1f, %d objects)",
                    self._current_object,
                    self._associative_memory.total_weight,
                    len(self._associative_memory.known_objects),
                )
            else:
                logger.info(
                    "Saved object '%s' with %d observations",
                    self._current_object,
                    self._memory.n_observations(self._current_object),
                )

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def step(self, state, **kwargs):
        """Process one observation through the cortical column.

        Returns
        -------
        dict with keys: surprise, n_bursting, n_predicted, evidence, mlh,
        active_cells
        """
        if not hasattr(state, "use_state") or not state.use_state:
            return self._empty_result()

        # Save previous state
        np.copyto(self._prev_active_mask, self._active_mask)
        np.copyto(self._prev_winner_mask, self._winner_mask)

        # 1. Encode
        input_sdr = self._encoder.encode(state)
        current_location = np.asarray(state.location, dtype=np.float64)
        location_sdr = self._encoder.encode_location(current_location)
        feature_sdr = self._encoder.encode_features(state)

        # 1b. Motor-conditional prediction (Phase 4)
        motor_pred_error = 0.0
        if self._use_motor_prediction and self._motor_prediction is not None:
            if self._prev_location is not None:
                displacement = current_location - self._prev_location
            else:
                displacement = np.zeros(3)
            mp_result = self._motor_prediction.step(location_sdr, displacement)
            motor_pred_error = mp_result["prediction_error"]
        self._prev_location = current_location.copy()

        # 2. Spatial pooling (with learning + boosting)
        self._spatial_pooling(input_sdr)

        # 3. Basal dendritic prediction
        prev_padded = np.zeros(self.n_cells + 1, dtype=np.bool_)
        prev_padded[:self.n_cells] = self._prev_active_mask
        self._predicted_mask = self._dendrites.compute_predicted_cells_mask(
            prev_padded
        )
        active_segments = self._dendrites.get_active_segments_from_mask(
            prev_padded
        )

        # 3b. Apical dendritic prediction (from top-down context)
        if self._use_apical and self._apical_dendrites is not None:
            ctx_padded = np.zeros(self.n_cells + 1, dtype=np.bool_)
            ctx_padded[:self.n_cells] = self._context_active_mask
            self._apical_predicted_mask = (
                self._apical_dendrites.compute_predicted_cells_mask(ctx_padded)
            )
            apical_active_segments = (
                self._apical_dendrites.get_active_segments_from_mask(ctx_padded)
            )
        else:
            self._apical_predicted_mask[:] = False
            apical_active_segments = {}

        # 4. Cell activation: predicted vs burst (4-state with apical)
        self._activate_cells(active_segments, apical_active_segments)

        # 5. Attractor settling (if enabled)
        if self._use_attractor:
            settled, settled_mc, n_iters = self._recurrent.settle(
                self._active_mask,
                ff_overlap=self._last_ff_overlap,
                boost_factors=self._boost_factors,
            )
            self._active_mask[:] = settled
            self._active_mc_mask[:] = settled_mc

        # 6. Surprise
        n_active = int(self._active_mc_mask.sum())
        n_burst = int(self._bursting_mc_mask.sum())
        surprise = n_burst / max(n_active, 1)

        # 7. Learning (basal + apical)
        if self._prev_active_mask.any():
            self._learn(active_segments)
        if self._use_apical and self._context_active_mask.any():
            self._learn_apical(apical_active_segments)

        # 8. Recurrent learning (attractor formation)
        # With continuous_plasticity: recurrent learning runs in both modes
        should_learn_recurrent = self._use_attractor and (
            self._mode == "train" or self._continuous_plasticity
        )
        if should_learn_recurrent:
            self._recurrent.learn(self._active_mask)
            self._recurrent.decay()

        # 9. Object memory + evidence update
        if self._use_weight_memory and self._associative_memory is not None:
            # --- Weight-based memory (Phase 7a) ---
            if self._mode == "train" and self._current_object:
                self._associative_memory.learn(
                    self._active_mask, input_sdr, self._current_object
                )
                # Also learn attractor if enabled (strengthens basins)
                if self._use_attractor:
                    self._attractor_memory.store(
                        self._current_object, self._active_mask
                    )
            elif self._mode == "eval" and self._continuous_plasticity:
                # Reconsolidation: re-learn on low-surprise known objects
                mlh = self._get_mlh()
                if (
                    mlh["graph_id"] is not None
                    and surprise < self._novelty_threshold
                ):
                    self._associative_memory.learn(
                        self._active_mask, input_sdr, mlh["graph_id"]
                    )

            # Evidence: weight-based readout
            if self._mode == "eval":
                match_scores = self._associative_memory.recall(
                    self._active_mask, input_sdr
                )
                for obj_name, score in match_scores.items():
                    if self._evidence_decay > 0:
                        self._evidence[obj_name] *= (1 - self._evidence_decay)
                    self._evidence[obj_name] = (
                        self._evidence.get(obj_name, 0.0) + score
                    )
        else:
            # --- Legacy SDR lookup memory ---
            if self._mode == "train" and self._current_object:
                cells_sdr = self._active_mask.astype(np.float64)
                self._memory.store(
                    self._current_object, location_sdr, feature_sdr, cells_sdr,
                )
                if self._use_attractor:
                    self._attractor_memory.store(
                        self._current_object, self._active_mask
                    )
            elif self._mode == "eval" and self._continuous_plasticity:
                mlh = self._get_mlh()
                if (
                    mlh["graph_id"] is not None
                    and surprise < self._novelty_threshold
                ):
                    if self._use_attractor:
                        self._attractor_memory.store(
                            mlh["graph_id"], self._active_mask
                        )

            # Evidence: SDR snapshot matching (legacy)
            if self._mode == "eval":
                known = self._memory.known_objects
                if known:
                    match_scores = self._memory.match(
                        location_sdr, feature_sdr
                    )
                    for obj_name, score in match_scores.items():
                        if self._evidence_decay > 0:
                            self._evidence[obj_name] *= (
                                1 - self._evidence_decay
                            )
                        self._evidence[obj_name] = (
                            self._evidence.get(obj_name, 0.0) + score
                        )

        self._step_count += 1
        self._last_surprise = surprise

        # Update neuromodulators (Phase 5)
        if self._use_neuromodulation and self._neuromodulators is not None:
            self._surprise_ema = 0.9 * self._surprise_ema + 0.1 * surprise
            self._neuromodulators.update(
                burst_ratio=surprise,
                surprise_ema=self._surprise_ema,
            )

        # Structural plasticity: age segments, periodic pruning (Phase 6)
        if self._dendrites._graded or self._dendrites._pruning_age > 0:
            self._dendrites.age_segments(prev_padded)
            # Prune dead segments every 100 steps
            if self._step_count % 100 == 0 and self._step_count > 0:
                self._dendrites.prune_old_segments()

        return {
            "surprise": surprise,
            "n_bursting": n_burst,
            "n_predicted": n_active - n_burst,
            "evidence": dict(self._evidence),
            "mlh": self._get_mlh(),
            "active_cells": set(np.where(self._active_mask)[0].tolist()),
            "motor_prediction_error": motor_pred_error,
        }

    # ------------------------------------------------------------------
    # Spatial pooling with learning + boosting
    # ------------------------------------------------------------------

    def _spatial_pooling(self, input_sdr: np.ndarray) -> None:
        """Compute active minicolumns and optionally learn.

        1. Overlap: count connected active inputs per minicolumn
        2. Boost: multiply by homeostatic boost factors
        3. Inhibition: top-k wins
        4. Learn: strengthen/weaken proximal permanences (train only)
        5. Update boosting duty cycles
        """
        active_input = input_sdr > 0.5  # bool mask

        # Overlap: count connected inputs that are active
        connected = (
            (self._ff_permanences >= self._sp_connected_threshold)
            & self._ff_potential
        )
        overlap = (connected & active_input[np.newaxis, :]).sum(axis=1)
        overlap = overlap.astype(np.float32)

        # Boosting
        boosted = overlap * self._boost_factors

        # Winner-take-all via partial sort (O(n) instead of O(n log n))
        # Neuromodulation: arousal broadens/narrows competition
        k = self._n_active_minicolumns
        if self._use_neuromodulation and self._neuromodulators is not None:
            k = max(1, int(k * self._neuromodulators.competition_width_scale()))
            k = min(k, self.n_minicolumns)
        if k >= len(boosted):
            winners = np.arange(len(boosted))
        else:
            winners = np.argpartition(boosted, -k)[-k:]

        self._active_mc_mask[:] = False
        self._active_mc_mask[winners] = True

        # Cache overlap for settling loop
        self._last_ff_overlap = boosted

        # Spatial pooler learning
        # With continuous_plasticity: always learn, modulated by surprise
        # Without: only learn in train mode (backward compatible)
        should_learn = self._sp_learning and (
            self._mode == "train"
            or self._continuous_plasticity
        )
        if should_learn:
            self._sp_learn(winners, active_input)
            self._update_boosting()

    def _sp_learn(self, winning_mcs: np.ndarray, active_input: np.ndarray):
        """Hebbian learning on proximal dendrites of winning minicolumns.

        Vectorized: all winning minicolumns updated in one batch operation.
        With continuous_plasticity during eval, learning rates are modulated
        by surprise (burst ratio): high surprise → full learning, low
        surprise → minimal maintenance learning.
        """
        # Learning rate modulation
        modulation = 1.0
        # Continuous plasticity: surprise-modulated during eval
        if self._continuous_plasticity and self._mode == "eval":
            modulation *= self._last_surprise
        # Neuromodulation: novelty scales SP learning
        if self._use_neuromodulation and self._neuromodulators is not None:
            modulation *= self._neuromodulators.sp_learning_rate_scale()

        inc = self._sp_increment * modulation
        dec = self._sp_decrement * modulation

        w_pot = self._ff_potential[winning_mcs]                    # (W, n_input)
        active_and_pot = active_input[np.newaxis, :] & w_pot       # (W, n_input)
        inactive_and_pot = (~active_input)[np.newaxis, :] & w_pot  # (W, n_input)

        self._ff_permanences[winning_mcs] = np.clip(
            self._ff_permanences[winning_mcs]
            + inc * active_and_pot
            - dec * inactive_and_pot,
            0.0,
            1.0,
        )

    def _update_boosting(self):
        """Update homeostatic boost factors from duty cycle tracking."""
        alpha = 0.01  # EMA smoothing rate
        self._mc_duty_cycle *= (1.0 - alpha)
        self._mc_duty_cycle[self._active_mc_mask] += alpha

        if self._boost_strength > 0:
            self._boost_factors = np.exp(
                self._boost_strength * (self._sparsity - self._mc_duty_cycle)
            )

    # ------------------------------------------------------------------
    # Cell activation
    # ------------------------------------------------------------------

    def _activate_cells(
        self, active_segments: dict, apical_active_segments: dict = None,
    ) -> None:
        """Determine active and winner cells using 4-state activation logic.

        When apical dendrites are enabled, cells have four activation modes:
        1. Basal + apical predicted → fire (highest confidence)
        2. Basal predicted only → fire (normal confidence)
        3. Apical predicted only → biased (partially depolarized, favoured
           in best_matching_cell but still bursts at minicolumn level)
        4. Neither → burst (full surprise)

        Without apical: collapses to the original 2-state (predicted vs burst).
        """
        self._active_mask[:] = False
        self._winner_mask[:] = False
        self._bursting_mc_mask[:] = False

        k = self.n_cells_per_minicolumn
        active_2d = self._active_mask.reshape(self.n_minicolumns, k)
        winner_2d = self._winner_mask.reshape(self.n_minicolumns, k)
        predicted_2d = self._predicted_mask.reshape(self.n_minicolumns, k)
        apical_2d = self._apical_predicted_mask.reshape(self.n_minicolumns, k)
        active_mcs = np.where(self._active_mc_mask)[0]

        if len(active_mcs) == 0:
            return

        # Vectorized check: which active minicolumns have basal-predicted cells?
        mc_has_basal = predicted_2d[active_mcs].any(axis=1)
        predicted_mcs = active_mcs[mc_has_basal]
        non_predicted_mcs = active_mcs[~mc_has_basal]

        # --- State 1 & 2: Basal predicted (with or without apical) ---
        if len(predicted_mcs) > 0:
            pred_patterns = predicted_2d[predicted_mcs]  # (P, k)
            active_2d[predicted_mcs] = pred_patterns
            winner_2d[predicted_mcs] = pred_patterns

        # --- States 3 & 4 for non-basal-predicted minicolumns ---
        if len(non_predicted_mcs) > 0:
            if self._use_apical:
                # Check which non-predicted MCs have apical predictions
                mc_has_apical = apical_2d[non_predicted_mcs].any(axis=1)
                apical_only_mcs = non_predicted_mcs[mc_has_apical]
                fully_bursting_mcs = non_predicted_mcs[~mc_has_apical]

                # State 3: Apical-only → burst at MC level but bias winner
                # toward apical-predicted cells
                if len(apical_only_mcs) > 0:
                    active_2d[apical_only_mcs] = True
                    self._bursting_mc_mask[apical_only_mcs] = True
                    for mc in apical_only_mcs:
                        winner = self._best_matching_cell(
                            int(mc), active_segments, apical_active_segments,
                        )
                        self._winner_mask[winner] = True

                # State 4: Neither → full burst
                if len(fully_bursting_mcs) > 0:
                    active_2d[fully_bursting_mcs] = True
                    self._bursting_mc_mask[fully_bursting_mcs] = True
                    for mc in fully_bursting_mcs:
                        winner = self._best_matching_cell(
                            int(mc), active_segments,
                        )
                        self._winner_mask[winner] = True
            else:
                # No apical: all non-predicted MCs burst
                active_2d[non_predicted_mcs] = True
                self._bursting_mc_mask[non_predicted_mcs] = True
                for mc in non_predicted_mcs:
                    winner = self._best_matching_cell(
                        int(mc), active_segments,
                    )
                    self._winner_mask[winner] = True

    def _best_matching_cell(
        self,
        mc: int,
        active_segments: dict,
        apical_active_segments: dict = None,
    ) -> int:
        """Choose best cell in a bursting minicolumn for learning.

        Priority order:
        1. Cell with most active basal segments (best context match)
        2. Cell with apical prediction (top-down bias, Phase 2)
        3. Cell with fewest existing segments (least committed)
        """
        k = self.n_cells_per_minicolumn
        start = mc * k

        # Check which cells in this minicolumn have active basal segments
        best_cell = None
        best_n = -1
        for cell in range(start, start + k):
            if cell in active_segments:
                n = len(active_segments[cell])
                if n > best_n:
                    best_n = n
                    best_cell = cell

        if best_cell is not None:
            return best_cell

        # Apical bias: prefer cells predicted by top-down context
        if apical_active_segments is not None:
            apical_cells = []
            for cell in range(start, start + k):
                if cell in apical_active_segments:
                    apical_cells.append(cell)
            if apical_cells:
                return int(self._rng.choice(apical_cells))

        # Also check apical predicted mask directly (covers cases where
        # apical segments are active but weren't passed explicitly)
        if self._use_apical:
            apical_pred = self._apical_predicted_mask[start:start + k]
            if apical_pred.any():
                candidates = np.where(apical_pred)[0] + start
                return int(self._rng.choice(candidates))

        # Fallback: cell with fewest existing segments (most available)
        seg_counts = self._dendrites._segment_count[start:start + k]
        min_count = seg_counts.min()
        candidates = np.where(seg_counts == min_count)[0] + start
        return int(self._rng.choice(candidates))

    # ------------------------------------------------------------------
    # Three-factor learning
    # ------------------------------------------------------------------

    def _learn(self, active_segments: dict) -> None:
        """Three-factor Hebbian learning on basal dendrites.

        Uses batch operations for strengthen and punish. The active mask is
        built once and reused across all segments.
        """
        k = self.n_cells_per_minicolumn

        # Build padded active mask once for all strengthen operations
        prev_padded = np.zeros(self.n_cells + 1, dtype=np.bool_)
        prev_padded[:self.n_cells] = self._prev_active_mask

        # --- Collect (cell, seg) pairs for batch operations ---
        str_cells = []
        str_segs = []
        pun_cells = []
        pun_segs = []

        for cell, seg_indices in active_segments.items():
            mc = cell // k
            if self._winner_mask[cell] and not self._bursting_mc_mask[mc]:
                # Strengthen: predicted winner in non-bursting minicolumn
                for seg_idx in seg_indices:
                    str_cells.append(cell)
                    str_segs.append(seg_idx)
            elif not self._active_mc_mask[mc]:
                # Punish: false prediction (minicolumn not even active)
                for seg_idx in seg_indices:
                    pun_cells.append(cell)
                    pun_segs.append(seg_idx)

        # Batch strengthen
        if str_cells:
            self._dendrites.strengthen_segments_batch(
                np.array(str_cells, dtype=np.int32),
                np.array(str_segs, dtype=np.int32),
                prev_padded,
            )

        # Batch punish
        if pun_cells:
            self._dendrites.punish_segments_batch(
                np.array(pun_cells, dtype=np.int32),
                np.array(pun_segs, dtype=np.int32),
            )

        # Grow new segments on winner cells in bursting minicolumns
        # (per-cell random sampling — kept as loop)
        prev_active_set = set(np.where(self._prev_active_mask)[0].tolist())
        bursting_mcs = np.where(self._bursting_mc_mask)[0]
        for mc in bursting_mcs:
            s, e = int(mc) * k, (int(mc) + 1) * k
            mc_winners = np.where(self._winner_mask[s:e])[0] + s
            for winner in mc_winners:
                self._dendrites.grow_segment(int(winner), prev_active_set)

    # ------------------------------------------------------------------
    # Apical learning (Phase 2)
    # ------------------------------------------------------------------

    def _learn_apical(self, apical_active_segments: dict) -> None:
        """Hebbian learning on apical dendrites using context signal.

        Mirrors basal learning: strengthen correct apical predictions,
        punish false ones, grow new segments for unpredicted winners.
        """
        if self._apical_dendrites is None:
            return

        k = self.n_cells_per_minicolumn

        # Build padded context mask for strengthen
        ctx_padded = np.zeros(self.n_cells + 1, dtype=np.bool_)
        ctx_padded[:self.n_cells] = self._context_active_mask

        str_cells = []
        str_segs = []
        pun_cells = []
        pun_segs = []

        for cell, seg_indices in apical_active_segments.items():
            mc = cell // k
            if self._winner_mask[cell] and not self._bursting_mc_mask[mc]:
                for seg_idx in seg_indices:
                    str_cells.append(cell)
                    str_segs.append(seg_idx)
            elif not self._active_mc_mask[mc]:
                for seg_idx in seg_indices:
                    pun_cells.append(cell)
                    pun_segs.append(seg_idx)

        if str_cells:
            self._apical_dendrites.strengthen_segments_batch(
                np.array(str_cells, dtype=np.int32),
                np.array(str_segs, dtype=np.int32),
                ctx_padded,
            )

        if pun_cells:
            self._apical_dendrites.punish_segments_batch(
                np.array(pun_cells, dtype=np.int32),
                np.array(pun_segs, dtype=np.int32),
            )

        # Grow apical segments on winner cells in bursting minicolumns
        ctx_active_set = set(np.where(self._context_active_mask)[0].tolist())
        if ctx_active_set:
            bursting_mcs = np.where(self._bursting_mc_mask)[0]
            for mc in bursting_mcs:
                s, e = int(mc) * k, (int(mc) + 1) * k
                mc_winners = np.where(self._winner_mask[s:e])[0] + s
                for winner in mc_winners:
                    self._apical_dendrites.grow_segment(
                        int(winner), ctx_active_set
                    )

    # ------------------------------------------------------------------
    # Context signals (Phase 2 — Monty integration)
    # ------------------------------------------------------------------

    def receive_context(self, **context_signal) -> None:
        """Receive top-down context from a parent column or LM.

        The context signal's ``active_cells`` field (a bool mask or set of
        cell indices) feeds the apical dendrites. This is how hierarchical
        columns communicate: a parent broadcasts its state via
        ``get_context_signal()`` and the child's apical dendrites learn to
        associate parent patterns with local predictions.

        Parameters
        ----------
        **context_signal : dict
            Must contain ``active_cells`` (np.ndarray bool mask of shape
            (n_cells,) or set of int cell indices).
        """
        active = context_signal.get("active_cells")
        if active is None:
            return

        if isinstance(active, set):
            self._context_active_mask[:] = False
            if active:
                idx = np.fromiter(active, dtype=np.int64)
                # Only set bits within our cell range
                valid = idx[idx < self.n_cells]
                if len(valid) > 0:
                    self._context_active_mask[valid] = True
        elif isinstance(active, np.ndarray):
            if active.dtype == np.bool_:
                n = min(len(active), self.n_cells)
                self._context_active_mask[:n] = active[:n]
                self._context_active_mask[n:] = False
            else:
                self._context_active_mask[:] = False
                valid = active[active < self.n_cells]
                if len(valid) > 0:
                    self._context_active_mask[valid] = True

    def get_context_signal(self) -> dict:
        """Broadcast this column's current state for top-down modulation.

        Returns a dict suitable for ``receive_context()`` on a child column.
        """
        return {
            "active_cells": self._active_mask.copy(),
            "winner_cells": self._winner_mask.copy(),
            "surprise": self.surprise,
        }

    # ------------------------------------------------------------------
    # Evidence and MLH
    # ------------------------------------------------------------------

    def _get_mlh(self):
        if not self._evidence:
            return {"graph_id": None, "evidence": 0.0}
        best_obj = max(self._evidence, key=self._evidence.get)
        return {"graph_id": best_obj, "evidence": self._evidence[best_obj]}

    def get_current_mlh(self):
        """Public interface for most likely hypothesis."""
        return self._get_mlh()

    def get_all_known_object_ids(self):
        """Return list of all trained object names."""
        if self._use_weight_memory and self._associative_memory is not None:
            return self._associative_memory.known_objects
        return self._memory.known_objects

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def surprise(self) -> float:
        n_active = int(self._active_mc_mask.sum())
        n_burst = int(self._bursting_mc_mask.sum())
        return n_burst / max(n_active, 1)

    @property
    def encoder(self) -> FeatureSDREncoder:
        return self._encoder

    @property
    def memory(self) -> SDRObjectMemory:
        return self._memory

    @property
    def dendrites(self) -> DendriteSegments:
        return self._dendrites

    @property
    def recurrent(self) -> RecurrentConnections:
        return self._recurrent

    @property
    def attractor_memory(self) -> AttractorMemory:
        return self._attractor_memory

    @property
    def associative_memory(self):
        """Hetero-associative memory (None if use_weight_memory=False)."""
        return self._associative_memory

    @property
    def apical_dendrites(self):
        """Apical dendritic segments (None if use_apical=False)."""
        return self._apical_dendrites

    @property
    def motor_prediction(self):
        """Motor prediction module (None if use_motor_prediction=False)."""
        return self._motor_prediction

    @property
    def neuromodulators(self):
        """Neuromodulatory state (None if use_neuromodulation=False)."""
        return self._neuromodulators

    def receive_reward(self, reward: float) -> None:
        """Receive an external reward signal for neuromodulatory gating.

        Parameters
        ----------
        reward : float
            Reward signal in [-1, 1]. Positive → consolidate, negative → decay.
        """
        if self._neuromodulators is not None:
            self._neuromodulators.update(
                burst_ratio=self._last_surprise,
                surprise_ema=self._surprise_ema,
                reward_signal=reward,
            )

    def _empty_result(self):
        return {
            "surprise": 1.0,
            "n_bursting": 0,
            "n_predicted": 0,
            "evidence": dict(self._evidence),
            "mlh": {"graph_id": None, "evidence": 0.0},
            "active_cells": set(),
            "motor_prediction_error": 0.0,
        }
