# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Unified cortical column with modern Hopfield dynamics in PyTorch.

Architecture follows the cortical LM / HPC separation:

**Cortical column (LM)** — slow, incremental learning:
  - Hopfield memory stores a small number of object-level attractors
    (one per learned object), updated incrementally via EMA prototypes.
  - Settling denoises the current observation toward the nearest attractor.

**Episodic memory (HPC)** — fast, one-shot storage:
  - Stores individual observations with labels and novelty gating.
  - Used for offline replay / consolidation (not for settling).

Pipeline per step (flat mode — Track 9 default):
1. Encode sensory input → feedforward float tensor
2. Spatial pooling: overlap + boosting + top-k
3. Dendritic prediction: basal predict(prev_active) + apical predict(context)
4. Cell activation: graded 4-state logic
5. Hopfield settling: denoise toward cortical object attractors
6. Surprise: mismatch between predicted and settled patterns
7. Learning: Hebbian on all pathways, modulated by surprise
8. Evidence update: attention over prototypes → per-object scores
9. Episodic storage: one-shot store in HPC memory

Laminar mode (Track 10, laminar=True):
1. Encode → thalamic relay (L6 gating) → L4 spatial pooling
2. L4 → L2/3 (Hopfield settling with optional plateau context)
3. L2/3 + apical context → L5 (output, plateau trigger)
4. L5 → L6 (feedback, thalamic gate update)
5. Surprise, learning (STDP or Hebbian), evidence
All Track 10 features default off — flat mode reproduces Track 9 exactly.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from tbp.monty.frameworks.models.cortical_column_torch.associative_memory import (
    HopfieldAssociativeMemory,
)
from tbp.monty.frameworks.models.cortical_column_torch.dendrites import (
    SparseDendrites,
)
from tbp.monty.frameworks.models.cortical_column_torch.encoders import (
    TorchFeatureEncoder,
)
from tbp.monty.frameworks.models.cortical_column_torch.episodic_memory import (
    EpisodicMemory,
)
from tbp.monty.frameworks.models.cortical_column_torch.location_feature_memory import (
    LocationFeatureMemory,
)
from tbp.monty.frameworks.models.cortical_column_torch.predictive_tracker import (
    PredictiveTracker,
)
from tbp.monty.frameworks.models.cortical_column_torch.hopfield import (
    ModernHopfieldMemory,
)
from tbp.monty.frameworks.models.cortical_column_torch.motor_prediction import (
    ContrastiveMotorPrediction,
)
from tbp.monty.frameworks.models.cortical_column_torch.neuromodulators import (
    NeuromodulatoryGating,
)
from tbp.monty.frameworks.models.cortical_column_torch.sparse_activations import (
    active_minicolumn_mask,
    enforce_sparsity,
)

logger = logging.getLogger(__name__)


class CorticalColumnTorch:
    """Modern Hopfield cortical column in PyTorch.

    Parameters
    ----------
    n_minicolumns : int
        Number of minicolumns.
    n_cells_per_minicolumn : int
        Cells per minicolumn.
    sparsity : float
        Target fraction of active minicolumns.
    base_learning_rate : float
        Base Hebbian learning rate (modulated by surprise).
    beta : float
        Hopfield inverse temperature. Higher = sharper retrieval.
    max_settle_iters : int
        Max Hopfield settling iterations. Should be >= 10 for reliable
        convergence with many stored patterns.
    use_apical : bool
        Enable apical dendritic prediction (top-down context).
    use_motor_prediction : bool
        Enable contrastive Hebbian motor prediction.
    use_neuromodulation : bool
        Enable neuromodulatory gating.
    evidence_decay : float
        Per-step evidence decay factor.
    laminar : bool
        Enable laminar structure (L4/L2/3/L5/L6). Default False = flat mode.
    multi_head : bool
        Enable multi-head dendritic attention. Default False.
    n_heads : int
        Number of dendritic heads per cell (when multi_head=True).
    use_plateau : bool
        Enable plateau potential working memory. Default False.
    use_interneurons : bool
        Enable PV+/SST+/VIP+ interneuron circuit. Default False.
    use_stdp : bool
        Enable STDP (replaces symmetric Hebbian). Default False.
    use_eligibility : bool
        Enable eligibility traces. Default False.
    use_phase_coding : bool
        Enable oscillatory phase coding. Default False.
    use_thalamic_relay : bool
        Enable thalamocortical loop gating. Default False.
    laminar_kwargs : dict or None
        Override LaminarConfig parameters.
    plateau_kwargs : dict or None
        Override PlateauPotentialMemory parameters.
    oscillator_kwargs : dict or None
        Override CorticalOscillator parameters.
    seed : int
        Random seed.
    device : str
        PyTorch device.
    """

    def __init__(
        self,
        n_minicolumns: int = 2048,
        n_cells_per_minicolumn: int = 8,
        sparsity: float = 0.03,
        base_learning_rate: float = 0.05,
        beta: float = 12.0,
        max_settle_iters: int = 10,
        use_apical: bool = False,
        use_motor_prediction: bool = False,
        use_neuromodulation: bool = False,
        evidence_decay: float = 0.0,
        laminar: bool = False,
        multi_head: bool = False,
        n_heads: int = 4,
        use_plateau: bool = False,
        use_interneurons: bool = False,
        use_stdp: bool = False,
        use_eligibility: bool = False,
        use_phase_coding: bool = False,
        use_thalamic_relay: bool = False,
        use_location_feature_memory: bool = False,
        encoder_kwargs: dict = None,
        dendrite_kwargs: dict = None,
        hopfield_kwargs: dict = None,
        memory_kwargs: dict = None,
        episodic_kwargs: dict = None,
        lfm_kwargs: dict = None,
        motor_kwargs: dict = None,
        neuromod_kwargs: dict = None,
        laminar_kwargs: dict = None,
        plateau_kwargs: dict = None,
        oscillator_kwargs: dict = None,
        seed: int = 42,
        device: str = "cpu",
    ):
        self.n_minicolumns = n_minicolumns
        self.n_cells_per_minicolumn = n_cells_per_minicolumn
        self.n_cells = n_minicolumns * n_cells_per_minicolumn
        self._sparsity = sparsity
        self._n_active_mc = max(1, int(sparsity * n_minicolumns))
        self._base_lr = base_learning_rate
        self._beta = beta
        self._max_settle_iters = max_settle_iters
        self._use_apical = use_apical
        self._use_motor_prediction = use_motor_prediction
        self._use_neuromodulation = use_neuromodulation
        self._evidence_decay = evidence_decay
        self.device = torch.device(device)

        torch.manual_seed(seed)

        # ---- Encoder ----
        enc_kw = dict(
            location_bits=320,
            feature_bits_per_dim=32,
            n_feature_dims=10,
            seed=seed,
            device=device,
        )
        if encoder_kwargs:
            enc_kw.update(encoder_kwargs)
        self._encoder = TorchFeatureEncoder(**enc_kw)

        # ---- Feedforward proximal dendrites ----
        self._init_feedforward(seed)

        # ---- Basal dendrites ----
        dend_kw = dict(
            n_cells=self.n_cells,
            max_segments_per_cell=32,
            max_synapses_per_segment=24,
            activation_threshold=0.3,
            connected_threshold=0.3,
            seed=seed,
            device=device,
        )
        if dendrite_kwargs:
            dend_kw.update(dendrite_kwargs)
        self._dendrites = SparseDendrites(**dend_kw)

        # ---- Apical dendrites ----
        if self._use_apical:
            apical_kw = dict(
                n_cells=self.n_cells,
                max_segments_per_cell=16,
                max_synapses_per_segment=20,
                activation_threshold=0.2,
                connected_threshold=0.3,
                seed=seed + 500,
                device=device,
            )
            self._apical_dendrites = SparseDendrites(**apical_kw)
        else:
            self._apical_dendrites = None

        # ---- Cortical Hopfield memory (L2/3 attractors) ----
        # Stores a small number of object-level attractors (~1 per object),
        # NOT individual observations.  Patterns are synced from associative
        # memory prototypes.  No phase augmentation — attractors are
        # phase-independent object representations.
        hop_kw = dict(
            n_cells=self.n_cells,
            beta=beta,
            max_stored=1000,
            max_settle_iters=max_settle_iters,
            device=device,
        )
        if hopfield_kwargs:
            # Filter out novelty_threshold — that's an episodic concept
            hop_kw_filtered = {
                k: v for k, v in hopfield_kwargs.items()
                if k != "novelty_threshold"
            }
            hop_kw.update(hop_kw_filtered)
        self._hopfield = ModernHopfieldMemory(**hop_kw)

        # ---- Associative memory ----
        mem_kw = dict(
            n_cells=self.n_cells,
            n_label_bits=256,
            beta=beta,
            learning_rate=0.01,
            device=device,
        )
        if memory_kwargs:
            mem_kw.update(memory_kwargs)
        self._associative_memory = HopfieldAssociativeMemory(**mem_kw)

        # ---- Episodic memory (HPC) ----
        # One-shot storage of individual observations with novelty gating.
        # Separate from the cortical column's slow attractor dynamics.
        ep_kw = dict(
            n_cells=self.n_cells,
            max_episodes=2000,
            novelty_threshold=0.7,
            device=device,
        )
        if episodic_kwargs:
            ep_kw.update(episodic_kwargs)
        # Legacy: if hopfield_kwargs had novelty_threshold, use it
        if hopfield_kwargs and "novelty_threshold" in hopfield_kwargs:
            ep_kw["novelty_threshold"] = hopfield_kwargs["novelty_threshold"]
        self._episodic_memory = EpisodicMemory(**ep_kw)

        # ---- Location-Feature memory (Phase 11) ----
        # Stores composite (location, feature) patterns tagged with object IDs.
        # Replaces EMA prototype evidence with many-pattern-per-object evidence.
        self._use_lfm = use_location_feature_memory
        if self._use_lfm:
            _lfm_kw = dict(
                d_loc=self._encoder._location_bits,
                d_feat=self._encoder.total_bits - self._encoder._location_bits,
                beta=beta,
                novelty_threshold=0.7,
                max_patterns=5000,
                device=device,
            )
            if lfm_kwargs:
                _lfm_kw.update(lfm_kwargs)
            self._lfm = LocationFeatureMemory(**_lfm_kw)
            self._predictive_tracker = PredictiveTracker(
                surprise_threshold=0.3,
                anchor_evidence_threshold=0.1,
                prediction_bonus_weight=0.5,
                max_consecutive_drops=5,
            )
        else:
            self._lfm = None
            self._predictive_tracker = None

        # ---- Motor prediction ----
        if self._use_motor_prediction:
            mp_kw = dict(
                location_dim=self._encoder._location_bits,
                motor_dim=3,
                device=device,
            )
            if motor_kwargs:
                mp_kw.update(motor_kwargs)
            self._motor_prediction = ContrastiveMotorPrediction(**mp_kw)
        else:
            self._motor_prediction = None

        # ---- Neuromodulation ----
        if self._use_neuromodulation:
            nm_kw = neuromod_kwargs or {}
            self._neuromod = NeuromodulatoryGating(**nm_kw)
        else:
            self._neuromod = None

        # ---- Track 10: Laminar features (all default off) ----
        self._laminar = laminar
        self._multi_head = multi_head
        self._n_heads = n_heads
        self._use_plateau = use_plateau
        self._use_interneurons = use_interneurons
        self._use_stdp = use_stdp
        self._use_eligibility = use_eligibility
        self._use_phase_coding = use_phase_coding
        self._use_thalamic_relay = use_thalamic_relay

        self._laminar_layers = None
        self._laminar_config = None
        self._multihead_dendrites = None
        self._plateau = None
        self._interneurons = None
        self._stdp = None
        self._eligibility = None
        self._oscillator = None
        self._thalamic_relay = None

        if self._laminar:
            self._init_laminar(laminar_kwargs, seed)
        if self._multi_head:
            self._init_multihead(n_heads, dendrite_kwargs, seed)
        if self._use_plateau:
            self._init_plateau(plateau_kwargs)
        if self._use_interneurons:
            self._init_interneurons(n_heads)
        if self._use_stdp:
            self._init_stdp()
        if self._use_eligibility:
            self._init_eligibility()
        if self._use_phase_coding:
            self._init_oscillator(oscillator_kwargs)
        if self._use_thalamic_relay:
            self._init_thalamic_relay()

        # ---- Feedforward proximal state ----
        # (initialized in _init_feedforward)

        # ---- Episode state ----
        self._active = torch.zeros(self.n_cells, dtype=torch.float32,
                                   device=self.device)
        self._prev_active = torch.zeros(self.n_cells, dtype=torch.float32,
                                        device=self.device)
        self._predicted = torch.zeros(self.n_cells, dtype=torch.float32,
                                      device=self.device)
        self._apical_predicted = torch.zeros(self.n_cells, dtype=torch.float32,
                                             device=self.device)
        self._context = torch.zeros(self.n_cells, dtype=torch.float32,
                                    device=self.device)
        self._active_mc = torch.zeros(self.n_minicolumns, dtype=torch.bool,
                                      device=self.device)
        self._mc_overlap = torch.zeros(self.n_minicolumns, dtype=torch.float32,
                                       device=self.device)
        self._prev_location: np.ndarray | None = None
        self._mode = "eval"
        self._current_object: str | None = None
        self._step_count = 0
        self._evidence: dict[str, float] = {}
        self._surprise = 1.0
        self._activation_history: list[torch.Tensor] = []

    def _init_feedforward(self, seed: int) -> None:
        """Initialize proximal dendrite permanences."""
        rng = np.random.RandomState(seed + 1000)
        n_input = self._encoder.total_bits
        n_potential = max(1, n_input // 10)

        self._ff_potential = torch.zeros(
            self.n_minicolumns, n_input, dtype=torch.bool, device=self.device
        )
        self._ff_permanences = torch.zeros(
            self.n_minicolumns, n_input, dtype=torch.float32, device=self.device
        )
        self._sp_connected_threshold = 0.5
        self._sp_increment = 0.05
        self._sp_decrement = 0.02

        for mc in range(self.n_minicolumns):
            indices = rng.choice(n_input, size=n_potential, replace=False)
            self._ff_potential[mc, indices] = True
            perms = rng.normal(self._sp_connected_threshold, 0.10, n_potential)
            perms = np.clip(perms, 0.0, 1.0).astype(np.float32)
            self._ff_permanences[mc, indices] = torch.from_numpy(perms)

        # Boosting
        self._boost_factors = torch.ones(self.n_minicolumns, dtype=torch.float32,
                                         device=self.device)
        self._mc_duty_cycle = torch.full(
            (self.n_minicolumns,), self._sparsity, dtype=torch.float32,
            device=self.device,
        )
        self._boost_strength = 3.0

        # Developmental wiring: fixed random bias per cell within each
        # minicolumn.  In biology, cells in the same minicolumn have
        # different lateral wiring from birth — this gives each cell an
        # innate, input-dependent preference before dendrites learn.
        k = self.n_cells_per_minicolumn
        cell_rng = np.random.RandomState(seed + 2000)
        self._cell_bias = torch.from_numpy(
            cell_rng.standard_normal((self.n_minicolumns, k)).astype(np.float32)
        ).to(self.device)
        # Normalize so the bias magnitude is controlled
        self._cell_bias = self._cell_bias / (
            self._cell_bias.norm(dim=1, keepdim=True) + 1e-8
        )

    # ------------------------------------------------------------------
    # Track 10: Laminar feature initialization
    # ------------------------------------------------------------------

    def _init_laminar(self, laminar_kwargs: dict | None, seed: int) -> None:
        """Initialize laminar layer modules."""
        from tbp.monty.frameworks.models.cortical_column_torch.layers import (
            L4Layer, L23Layer, L5Layer, L6Layer, LaminarConfig,
        )
        lk = laminar_kwargs or {}
        self._laminar_config = LaminarConfig(**lk)

        n_input = self._encoder.total_bits
        self._l4 = L4Layer(
            n_minicolumns=self.n_minicolumns,
            n_input=n_input,
            n_cells_per_mc=self._laminar_config.n_L4,
            sparsity=self._sparsity,
            seed=seed + 3000,
            device=str(self.device),
        )
        self._l23 = L23Layer(
            n_minicolumns=self.n_minicolumns,
            n_cells_per_mc=self._laminar_config.n_L23,
            device=str(self.device),
        )
        self._l5 = L5Layer(
            n_minicolumns=self.n_minicolumns,
            n_cells_per_mc=self._laminar_config.n_L5,
            device=str(self.device),
        )
        self._l6 = L6Layer(
            n_minicolumns=self.n_minicolumns,
            n_input=n_input,
            n_cells_per_mc=self._laminar_config.n_L6,
            device=str(self.device),
        )
        # L6 feedback from previous step (one-step delay)
        self._prev_l6_activation = torch.zeros(
            self._l6.n_cells, dtype=torch.float32, device=self.device
        )
        self._laminar_layers = {
            "L4": self._l4, "L23": self._l23,
            "L5": self._l5, "L6": self._l6,
        }

    def _init_multihead(
        self, n_heads: int, dendrite_kwargs: dict | None, seed: int
    ) -> None:
        """Initialize multi-head dendritic attention."""
        from tbp.monty.frameworks.models.cortical_column_torch.multihead_dendrites import (
            MultiHeadDendrites,
        )
        mk = dict(
            n_cells=self.n_cells,
            n_heads=n_heads,
            max_segments_per_head=8,
            max_synapses_per_segment=24,
            activation_threshold=0.3,
            connected_threshold=0.3,
            seed=seed + 4000,
            device=str(self.device),
        )
        if dendrite_kwargs:
            for k in ("activation_threshold", "connected_threshold",
                       "max_synapses_per_segment"):
                if k in dendrite_kwargs:
                    mk[k] = dendrite_kwargs[k]
        self._multihead_dendrites = MultiHeadDendrites(**mk)

    def _init_plateau(self, plateau_kwargs: dict | None) -> None:
        """Initialize plateau potential working memory."""
        from tbp.monty.frameworks.models.cortical_column_torch.plateau import (
            PlateauPotentialMemory,
        )
        pk = dict(
            n_cells=self.n_cells,
            device=str(self.device),
        )
        if plateau_kwargs:
            pk.update(plateau_kwargs)
        self._plateau = PlateauPotentialMemory(**pk)

    def _init_interneurons(self, n_heads: int) -> None:
        """Initialize PV+/SST+/VIP+ interneuron circuit."""
        from tbp.monty.frameworks.models.cortical_column_torch.interneurons import (
            InterneuronCircuit,
        )
        self._interneurons = InterneuronCircuit(
            n_minicolumns=self.n_minicolumns,
            n_cells=self.n_cells,
            n_heads=n_heads,
            device=str(self.device),
        )

    def _init_stdp(self) -> None:
        """Initialize STDP rule."""
        from tbp.monty.frameworks.models.cortical_column_torch.stdp import (
            STDPRule,
        )
        self._stdp = STDPRule(
            n_cells=self.n_cells,
            device=str(self.device),
        )

    def _init_eligibility(self) -> None:
        """Initialize eligibility traces."""
        from tbp.monty.frameworks.models.cortical_column_torch.stdp import (
            EligibilityTrace,
        )
        self._eligibility = EligibilityTrace(
            device=str(self.device),
        )

    def _init_oscillator(self, oscillator_kwargs: dict | None) -> None:
        """Initialize oscillatory phase coding."""
        from tbp.monty.frameworks.models.cortical_column_torch.oscillator import (
            CorticalOscillator,
        )
        ok = oscillator_kwargs or {}
        self._oscillator = CorticalOscillator(
            device=str(self.device), **ok,
        )

    def _init_thalamic_relay(self) -> None:
        """Initialize thalamocortical relay."""
        from tbp.monty.frameworks.models.cortical_column_torch.thalamic_relay import (
            ThalamicRelay,
        )
        n_l6 = self.n_minicolumns  # L6 has 1 cell per MC by default
        if self._laminar_config is not None:
            n_l6 = self.n_minicolumns * self._laminar_config.n_L6
        self._thalamic_relay = ThalamicRelay(
            n_input=self._encoder.total_bits,
            n_l6_cells=n_l6,
            device=str(self.device),
        )

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def pre_episode(self, mode: str = "eval", object_name: str | None = None):
        """Reset state for a new episode."""
        self._mode = mode
        self._current_object = object_name
        self._active.zero_()
        self._prev_active.zero_()
        self._predicted.zero_()
        self._apical_predicted.zero_()
        self._context.zero_()
        self._active_mc.zero_()
        self._mc_overlap.zero_()
        self._last_pre_settle = torch.zeros(self.n_cells, dtype=torch.float32,
                                            device=self.device)
        self._prev_location = None
        self._step_count = 0
        self._surprise = 1.0
        self._activation_history.clear()

        if self._motor_prediction is not None:
            self._motor_prediction.reset()
        if self._neuromod is not None:
            self._neuromod.reset()

        # Track 10 resets
        if self._plateau is not None:
            self._plateau.reset()
        if self._stdp is not None:
            self._stdp.reset()
        if self._eligibility is not None:
            self._eligibility.clear()
        if self._oscillator is not None:
            self._oscillator.reset()
        if self._thalamic_relay is not None:
            self._thalamic_relay.reset()
        if self._interneurons is not None:
            self._interneurons.reset()
        if self._laminar and hasattr(self, "_prev_l6_activation"):
            self._prev_l6_activation.zero_()

        # Reset predictive tracker per eval episode
        if self._predictive_tracker is not None:
            self._predictive_tracker.reset()

        # Initialize evidence for known objects
        if self._use_lfm and self._lfm is not None:
            known = self._lfm.known_objects
        else:
            known = self._associative_memory.known_objects
        self._evidence = {obj: 0.0 for obj in known}

        # Sync cortical attractors from current prototypes
        self._sync_cortical_attractors()

    def _sync_cortical_attractors(self):
        """Sync Hopfield attractor patterns from associative memory prototypes.

        In cortical attractor mode, the Hopfield network stores one
        pattern per known object — the slowly-learned prototype from
        the associative memory.  This is biologically analogous to L2/3
        recurrent connectivity forming object-level attractors through
        experience.
        """
        protos = self._associative_memory.get_prototypes()
        if protos is not None and protos.shape[0] > 0:
            self._hopfield.set_patterns(protos)
        else:
            self._hopfield.clear()

    def post_episode(self):
        """Finalize episode."""
        if self._mode == "train":
            # Auto-generate label if none was provided and step() didn't
            # already generate one (e.g. fewer than 5 steps).
            if not self._current_object and self._activation_history:
                self._current_object = (
                    self._associative_memory.auto_label(
                        self._activation_history
                    )
                )

            if self._current_object:
                # Store final observation in episodic memory (HPC)
                pat = self._last_pre_settle
                if pat.abs().sum() > 0:
                    self._episodic_memory.store(
                        pat, label=self._current_object,
                    )

                # Sync cortical attractors from updated prototypes
                self._sync_cortical_attractors()

                logger.info(
                    "CorticalColumnTorch: learned '%s', "
                    "%d cortical attractors, %d episodes, %d objects",
                    self._current_object,
                    self._hopfield.n_stored,
                    self._episodic_memory.n_stored,
                    len(self._associative_memory.known_objects),
                )

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def step(self, state, **kwargs) -> dict:
        """Process one observation through the column.

        Routes to laminar pipeline when laminar=True, otherwise flat.

        Returns dict with: surprise, evidence, mlh, active_cells,
        settling_iterations, motor_prediction_error
        """
        if not hasattr(state, "use_state") or not state.use_state:
            return self._empty_result()

        if self._laminar:
            return self._step_laminar(state, **kwargs)

        with torch.no_grad():
            # Save previous
            self._prev_active.copy_(self._active)

            # 1. Encode
            input_vec = self._encoder.encode(state)
            current_location = np.asarray(state.location, dtype=np.float64)

            # 1a. Split for location-feature memory (Phase 11)
            lfm_loc_enc = lfm_feat_enc = None
            lfm_world_disp = np.zeros(3, dtype=np.float64)
            if self._use_lfm:
                _lb = self._encoder._location_bits
                lfm_loc_enc = input_vec[:_lb]
                lfm_feat_enc = input_vec[_lb:]
                # Compute displacement before _prev_location is updated
                if self._prev_location is not None:
                    lfm_world_disp = current_location - self._prev_location

            # 1b. Motor prediction
            motor_pred_error = 0.0
            if self._use_motor_prediction and self._motor_prediction is not None:
                if self._prev_location is not None:
                    disp = torch.from_numpy(
                        (current_location - self._prev_location).astype(np.float32)
                    ).to(self.device)
                else:
                    disp = torch.zeros(3, device=self.device)
                loc_enc = self._encoder.encode_location(current_location)
                mp_result = self._motor_prediction.step(loc_enc, disp)
                motor_pred_error = mp_result["prediction_error"]
            self._prev_location = current_location.copy()

            # 2. Spatial pooling
            self._spatial_pooling(input_vec)

            # 3. Basal dendritic prediction
            if self._multi_head and self._multihead_dendrites is not None:
                if self._use_interneurons and self._interneurons is not None:
                    gate = self._interneurons.compute_branch_gate(
                        self._prev_active, novelty=self._surprise,
                    )
                    self._multihead_dendrites.apply_gate(gate)
                    self._predicted = self._multihead_dendrites.predict_gated(
                        self._prev_active
                    )
                else:
                    self._predicted = self._multihead_dendrites.predict(
                        self._prev_active
                    )
            else:
                self._predicted = self._dendrites.predict(self._prev_active)

            # 3b. Apical prediction
            if self._use_apical and self._apical_dendrites is not None:
                self._apical_predicted = self._apical_dendrites.predict(
                    self._context
                )
            else:
                self._apical_predicted.zero_()

            # 4. Cell activation
            self._activate_cells()

            # Save pre-settle activation.  This is the raw cortical
            # pattern before attractor denoising — used for episodic
            # storage and associative memory learning.
            pre_settle_active = self._active.clone()
            self._last_pre_settle = pre_settle_active

            # 5. Hopfield settling — denoise toward cortical attractors
            # Cortical attractors are object-level prototypes (no phase
            # augmentation), so we always use the base settling path.
            beta = self._beta
            if self._use_neuromodulation and self._neuromod is not None:
                beta *= self._neuromod.beta_scale()

            # Plateau enrichment before settling
            query = self._active
            if self._use_plateau and self._plateau is not None:
                query = self._plateau.enrich_query(query)

            def sparsity_fn(x):
                return enforce_sparsity(
                    x, self.n_minicolumns, self.n_cells_per_minicolumn
                )
            settled, n_iters = self._hopfield.settle(
                query, beta=beta, sparsity_fn=sparsity_fn,
            )
            self._active = settled

            # 6. Surprise: 1 - overlap between predicted and active
            pred_overlap = (self._predicted * self._active.abs()).sum()
            active_norm = self._active.abs().sum() + 1e-8
            surprise = 1.0 - float(pred_overlap / active_norm)
            surprise = max(0.0, min(1.0, surprise))
            self._surprise = surprise

            # 7. Learning
            lr = self._base_lr * (0.1 + 0.9 * surprise)
            if self._use_neuromodulation and self._neuromod is not None:
                lr *= self._neuromod.learning_rate_scale()

            # Learn basal dendrites
            if self._prev_active.abs().sum() > 0:
                active_binary = (self._active.abs() > 1e-6).float()
                prev_binary = (self._prev_active.abs() > 1e-6).float()
                predicted_binary = (self._predicted > 0.3).float()

                if self._use_stdp and self._stdp is not None:
                    self._stdp.update_traces(prev_binary, active_binary)
                    updates = self._stdp.compute_sparse_update(
                        prev_binary, active_binary,
                        self._dendrites._segments,
                    )
                    if self._use_eligibility and self._eligibility is not None:
                        self._eligibility.accumulate(updates)
                        nm_signal = surprise
                        if self._neuromod is not None:
                            nm_signal *= self._neuromod.consolidation_scale()
                        self._eligibility.consolidate(
                            nm_signal, self._dendrites._segments,
                        )
                    else:
                        self._stdp.apply_sparse_update(
                            self._dendrites._segments, updates,
                        )
                else:
                    self._dendrites.learn(
                        active_binary, prev_binary, learning_rate=lr,
                    )

                self._dendrites.grow_for_unpredicted(
                    active_binary, prev_binary, predicted_binary,
                )

                # Multi-head learning (flat mode)
                if self._multi_head and self._multihead_dendrites is not None:
                    self._multihead_dendrites.learn(
                        active_binary, prev_binary, learning_rate=lr,
                    )
                    self._multihead_dendrites.grow_for_unpredicted(
                        active_binary, prev_binary, predicted_binary,
                    )

            # Plateau update (flat mode)
            if self._use_plateau and self._plateau is not None:
                # In flat mode, use basal+apical coincidence heuristic
                has_ba = (
                    (self._predicted > 0.3) & (self._apical_predicted > 0.2)
                )
                self._plateau.update(self._active, has_ba)

            # Interneuron learning (flat mode)
            if self._use_interneurons and self._interneurons is not None:
                self._interneurons.learn(self._active, surprise, lr)

            # Learn apical
            if (
                self._use_apical
                and self._apical_dendrites is not None
                and self._context.abs().sum() > 0
            ):
                active_binary = (self._active.abs() > 1e-6).float()
                ctx_binary = (self._context.abs() > 1e-6).float()
                apical_pred_binary = (self._apical_predicted > 0.3).float()
                self._apical_dendrites.learn(
                    active_binary, ctx_binary, learning_rate=lr,
                )
                self._apical_dendrites.grow_for_unpredicted(
                    active_binary, ctx_binary, apical_pred_binary,
                )

            # 7b. Episodic storage (HPC): one-shot store of raw observation
            if self._mode == "train" and pre_settle_active.abs().sum() > 0:
                self._episodic_memory.store(
                    pre_settle_active, label=self._current_object,
                )

            # SP learning
            if self._mode == "train":
                self._sp_learn(input_vec, lr)

            # 8. Object memory + evidence
            # Pre-settle activation for both train and eval: keeps the
            # prototype space consistent (prototypes are learned from
            # pre-settle, queries should match).
            if self._mode == "train":
                self._activation_history.append(pre_settle_active.clone())
                # Auto-generate label if none was provided.
                if not self._current_object and len(self._activation_history) >= 5:
                    self._current_object = (
                        self._associative_memory.auto_label(
                            self._activation_history
                        )
                    )
                if self._current_object:
                    self._associative_memory.learn(
                        pre_settle_active, self._current_object, lr=lr * 0.1,
                    )
                    # Sync cortical attractors after prototype update
                    self._sync_cortical_attractors()
                    # Store in location-feature memory (Phase 11)
                    if self._use_lfm and self._lfm is not None:
                        self._lfm.store(
                            lfm_loc_enc, lfm_feat_enc, self._current_object,
                            raw_location=current_location,
                        )
            elif self._mode == "eval":
                if self._use_lfm and self._lfm is not None:
                    tracker = self._predictive_tracker
                    _lb = self._encoder._location_bits

                    # -- Anchor-Track-Predict-Surprise loop --
                    #
                    # 1. Feature-only query (always): rotation-invariant
                    #    baseline evidence + predicted object locations
                    _, feat_ev, pred_locs = self._lfm.query_features(
                        lfm_feat_enc
                    )

                    # 2. Predictive tracking per object
                    #    (lfm_world_disp computed at step 1a, before
                    #    _prev_location was overwritten)
                    prediction_bonus: dict[str, float] = {}
                    for obj_name in list(feat_ev.keys()):
                        if tracker.is_anchored(obj_name):
                            # Track: path integration (world displacement
                            # as object-frame approximation)
                            tracker.track(obj_name, lfm_world_disp)
                            tracked_loc = tracker.get_location(obj_name)

                            # Predict: what features at this location?
                            loc_enc = self._encoder.encode_location(
                                tracked_loc,
                            )
                            retrieved, _ = self._lfm.query_location(
                                loc_enc,
                            )
                            predicted_feat = retrieved[_lb:]

                            # Surprise: predicted vs actual features
                            cos_sim = float(
                                torch.nn.functional.cosine_similarity(
                                    predicted_feat.unsqueeze(0),
                                    lfm_feat_enc.unsqueeze(0),
                                ).item()
                            )

                            if cos_sim > tracker.surprise_threshold:
                                # Low surprise — prediction confirmed
                                prediction_bonus[obj_name] = cos_sim
                                tracker.confirm(obj_name)
                            else:
                                # High surprise — drop and re-anchor
                                tracker.drop(obj_name)

                        # Anchor if not currently tracking
                        if (
                            not tracker.is_anchored(obj_name)
                            and not tracker.is_retired(obj_name)
                            and feat_ev.get(obj_name, 0.0)
                            > tracker.anchor_evidence_threshold
                            and obj_name in pred_locs
                        ):
                            tracker.anchor(obj_name, pred_locs[obj_name])

                    # 4. Accumulate evidence: feature baseline +
                    #    prediction confirmation bonus
                    for obj_name, score in feat_ev.items():
                        prev = self._evidence.get(obj_name, 0.0)
                        if self._evidence_decay > 0:
                            prev *= (1 - self._evidence_decay)
                        bonus = prediction_bonus.get(obj_name, 0.0)
                        total = score + bonus * tracker.prediction_bonus_weight
                        self._evidence[obj_name] = prev + total
                else:
                    scores = self._associative_memory.recall(pre_settle_active)
                    for obj_name, score in scores.items():
                        prev = self._evidence.get(obj_name, 0.0)
                        if self._evidence_decay > 0:
                            prev *= (1 - self._evidence_decay)
                        self._evidence[obj_name] = prev + score

            # Update neuromodulators
            if self._use_neuromodulation and self._neuromod is not None:
                self._neuromod.update(surprise)

            # Advance oscillator
            if self._use_phase_coding and self._oscillator is not None:
                self._oscillator.step()

            self._step_count += 1

            return {
                "surprise": surprise,
                "evidence": dict(self._evidence),
                "mlh": self._get_mlh(),
                "active_cells": set(
                    torch.nonzero(self._active.abs() > 1e-6, as_tuple=True)[0]
                    .cpu().tolist()
                ),
                "settling_iterations": n_iters,
                "motor_prediction_error": motor_pred_error,
            }

    # ------------------------------------------------------------------
    # Laminar step pipeline (Track 10)
    # ------------------------------------------------------------------

    def _step_laminar(self, state, **kwargs) -> dict:
        """Laminar pipeline: L4 → L2/3 → L5 → L6 → thalamic gate."""
        with torch.no_grad():
            self._prev_active.copy_(self._active)

            # 1. Encode
            input_vec = self._encoder.encode(state)
            current_location = np.asarray(state.location, dtype=np.float64)

            # 1b. Motor prediction
            motor_pred_error = 0.0
            if self._use_motor_prediction and self._motor_prediction is not None:
                if self._prev_location is not None:
                    disp = torch.from_numpy(
                        (current_location - self._prev_location).astype(
                            np.float32)
                    ).to(self.device)
                else:
                    disp = torch.zeros(3, device=self.device)
                loc_enc = self._encoder.encode_location(current_location)
                mp_result = self._motor_prediction.step(loc_enc, disp)
                motor_pred_error = mp_result["prediction_error"]
            self._prev_location = current_location.copy()

            # 1c. Thalamic relay gating
            if self._use_thalamic_relay and self._thalamic_relay is not None:
                l6_act = (
                    self._prev_l6_activation
                    if hasattr(self, "_prev_l6_activation")
                    else None
                )
                input_vec = self._thalamic_relay.forward(input_vec, l6_act)

            # 2. L4: spatial pooling (feedforward)
            n_active = self._n_active_mc
            if self._use_neuromodulation and self._neuromod is not None:
                n_active = max(1, int(n_active * self._neuromod.sparsity_scale()))
                n_active = min(n_active, self.n_minicolumns)

            active_mc, l4_act = self._l4.forward(
                input_vec, n_active_override=n_active,
            )
            self._active_mc = active_mc

            # 3. Basal dendritic prediction for L2/3
            if self._multi_head and self._multihead_dendrites is not None:
                # Multi-head with optional interneuron gating
                if (self._use_interneurons and self._interneurons is not None):
                    novelty = self._surprise
                    gate = self._interneurons.compute_branch_gate(
                        self._prev_active, novelty=novelty,
                    )
                    self._multihead_dendrites.apply_gate(gate)
                    basal_pred = self._multihead_dendrites.predict_gated(
                        self._prev_active
                    )
                else:
                    basal_pred = self._multihead_dendrites.predict(
                        self._prev_active
                    )
            else:
                basal_pred = self._dendrites.predict(self._prev_active)

            # Reshape prediction to L2/3 size
            lc = self._laminar_config
            n_l23_cells = self.n_minicolumns * lc.n_L23
            # Map flat prediction → L2/3 cells (take first n_l23 from full)
            basal_pred_l23 = basal_pred[:n_l23_cells]

            # 4. L2/3: representation with Hopfield settling
            l23_act = self._l23.forward(
                active_mc, l4_act, basal_pred_l23,
                n_L4_cells_per_mc=lc.n_L4,
            )

            # Hopfield settling on L2/3
            beta = self._beta
            if self._use_neuromodulation and self._neuromod is not None:
                beta *= self._neuromod.beta_scale()

            # Pad L2/3 activation to full n_cells for Hopfield
            padded = torch.zeros(
                self.n_cells, dtype=torch.float32, device=self.device
            )
            padded[:n_l23_cells] = l23_act

            # Save pre-settle activation
            pre_settle_active = padded.clone()
            self._last_pre_settle = pre_settle_active

            # Plateau enrichment before settling
            if self._use_plateau and self._plateau is not None:
                padded = self._plateau.enrich_query(padded)

            # Cortical attractor settling (no phase augmentation)
            def sparsity_fn(x):
                return enforce_sparsity(
                    x, self.n_minicolumns, self.n_cells_per_minicolumn
                )
            settled, n_iters = self._hopfield.settle(
                padded, beta=beta, sparsity_fn=sparsity_fn,
            )

            self._active = settled

            # 5. Apical prediction for L5
            n_l5_cells = self.n_minicolumns * lc.n_L5
            if self._use_apical and self._apical_dendrites is not None:
                self._apical_predicted = self._apical_dendrites.predict(
                    self._context
                )
                apical_for_l5 = self._apical_predicted[:n_l5_cells]
            else:
                apical_for_l5 = torch.zeros(
                    n_l5_cells, dtype=torch.float32, device=self.device
                )

            # 6. L5: output layer integrating L2/3 + apical
            l5_act = self._l5.forward(
                l23_act, apical_for_l5, active_mc,
                n_L23_cells_per_mc=lc.n_L23,
            )

            # Plateau update (L5 cells with basal+apical → plateau trigger)
            if self._use_plateau and self._plateau is not None:
                # Pad L5 activation to n_cells
                l5_padded = torch.zeros(
                    self.n_cells, dtype=torch.float32, device=self.device
                )
                l5_padded[:n_l5_cells] = l5_act
                ba_padded = torch.zeros(
                    self.n_cells, dtype=torch.bool, device=self.device
                )
                ba_padded[:n_l5_cells] = self._l5._has_basal_and_apical[
                    :n_l5_cells
                ]
                self._plateau.update(l5_padded, ba_padded)

            # 7. L6: feedback layer
            l6_act = self._l6.forward(
                l23_act, l5_act, active_mc,
                n_L23_cells_per_mc=lc.n_L23,
                n_L5_cells_per_mc=lc.n_L5,
            )
            self._prev_l6_activation = l6_act.clone()

            # Thalamic learning
            if self._use_thalamic_relay and self._thalamic_relay is not None:
                self._thalamic_relay.learn(
                    self._surprise, l6_act,
                )

            # 8. Surprise
            pred_overlap = (basal_pred * self._active.abs()).sum()
            active_norm = self._active.abs().sum() + 1e-8
            surprise = 1.0 - float(pred_overlap / active_norm)
            surprise = max(0.0, min(1.0, surprise))
            self._surprise = surprise

            # 9. Learning
            lr = self._base_lr * (0.1 + 0.9 * surprise)
            if self._use_neuromodulation and self._neuromod is not None:
                lr *= self._neuromod.learning_rate_scale()

            if self._prev_active.abs().sum() > 0:
                active_binary = (self._active.abs() > 1e-6).float()
                prev_binary = (self._prev_active.abs() > 1e-6).float()
                predicted_binary = (basal_pred > 0.3).float()

                if self._use_stdp and self._stdp is not None:
                    # STDP learning
                    self._stdp.update_traces(prev_binary, active_binary)
                    updates = self._stdp.compute_sparse_update(
                        prev_binary, active_binary,
                        self._dendrites._segments,
                    )
                    if self._use_eligibility and self._eligibility is not None:
                        self._eligibility.accumulate(updates)
                        nm_signal = surprise
                        if self._neuromod is not None:
                            nm_signal *= self._neuromod.consolidation_scale()
                        self._eligibility.consolidate(
                            nm_signal, self._dendrites._segments,
                        )
                    else:
                        self._stdp.apply_sparse_update(
                            self._dendrites._segments, updates,
                        )
                else:
                    # Standard Hebbian
                    self._dendrites.learn(
                        active_binary, prev_binary, learning_rate=lr,
                    )

                self._dendrites.grow_for_unpredicted(
                    active_binary, prev_binary, predicted_binary,
                )

                # Multi-head learning
                if self._multi_head and self._multihead_dendrites is not None:
                    self._multihead_dendrites.learn(
                        active_binary, prev_binary, learning_rate=lr,
                    )
                    self._multihead_dendrites.grow_for_unpredicted(
                        active_binary, prev_binary, predicted_binary,
                    )

            # Apical learning
            if (
                self._use_apical
                and self._apical_dendrites is not None
                and self._context.abs().sum() > 0
            ):
                active_binary = (self._active.abs() > 1e-6).float()
                ctx_binary = (self._context.abs() > 1e-6).float()
                apical_pred_binary = (self._apical_predicted > 0.3).float()
                self._apical_dendrites.learn(
                    active_binary, ctx_binary, learning_rate=lr,
                )
                self._apical_dendrites.grow_for_unpredicted(
                    active_binary, ctx_binary, apical_pred_binary,
                )

            # Interneuron learning
            if self._use_interneurons and self._interneurons is not None:
                self._interneurons.learn(self._active, surprise, lr)

            # L4 SP learning
            if self._mode == "train":
                self._l4.learn(input_vec, lr)
                # L6 feedback learning
                self._l6.learn(surprise, lr)

            # Episodic storage (HPC)
            if self._mode == "train" and pre_settle_active.abs().sum() > 0:
                self._episodic_memory.store(
                    pre_settle_active, label=self._current_object,
                )

            # 10. Object memory + evidence
            if self._mode == "train":
                self._activation_history.append(pre_settle_active.clone())
                if not self._current_object and len(self._activation_history) >= 5:
                    self._current_object = (
                        self._associative_memory.auto_label(
                            self._activation_history
                        )
                    )
                if self._current_object:
                    self._associative_memory.learn(
                        pre_settle_active, self._current_object, lr=lr * 0.1,
                    )
                    self._sync_cortical_attractors()
            elif self._mode == "eval":
                scores = self._associative_memory.recall(pre_settle_active)
                for obj_name, score in scores.items():
                    prev = self._evidence.get(obj_name, 0.0)
                    if self._evidence_decay > 0:
                        prev *= (1 - self._evidence_decay)
                    self._evidence[obj_name] = prev + score

            # Update neuromodulators
            if self._use_neuromodulation and self._neuromod is not None:
                self._neuromod.update(surprise)

            # Advance oscillator
            if self._use_phase_coding and self._oscillator is not None:
                self._oscillator.step()

            self._step_count += 1

            return {
                "surprise": surprise,
                "evidence": dict(self._evidence),
                "mlh": self._get_mlh(),
                "active_cells": set(
                    torch.nonzero(self._active.abs() > 1e-6, as_tuple=True)[0]
                    .cpu().tolist()
                ),
                "settling_iterations": n_iters,
                "motor_prediction_error": motor_pred_error,
            }

    # ------------------------------------------------------------------
    # Spatial pooling
    # ------------------------------------------------------------------

    def _spatial_pooling(self, input_vec: torch.Tensor) -> None:
        """Compute active minicolumns from input via proximal dendrites."""
        active_input = (input_vec.abs() > 0.1).float()

        # Overlap: connected permanences * active input
        connected = (
            (self._ff_permanences >= self._sp_connected_threshold)
            & self._ff_potential
        ).float()
        overlap = (connected * active_input.unsqueeze(0)).sum(dim=1)

        # Save per-minicolumn overlap for cell activation (Level 3).
        # Different overlap magnitudes interact with the developmental
        # cell bias to select different cells within each minicolumn.
        self._mc_overlap = overlap

        # Boost
        boosted = overlap * self._boost_factors

        # Top-k inhibition
        k = self._n_active_mc
        if self._use_neuromodulation and self._neuromod is not None:
            k = max(1, int(k * self._neuromod.sparsity_scale()))
            k = min(k, self.n_minicolumns)

        if k >= self.n_minicolumns:
            winners = torch.arange(self.n_minicolumns, device=self.device)
        else:
            _, winners = torch.topk(boosted, k)

        self._active_mc.zero_()
        self._active_mc[winners] = True

    def _sp_learn(self, input_vec: torch.Tensor, lr: float) -> None:
        """Hebbian learning on proximal dendrites."""
        active_input = (input_vec.abs() > 0.1)
        winner_idx = torch.nonzero(self._active_mc, as_tuple=True)[0]

        if len(winner_idx) == 0:
            return

        inc = self._sp_increment * lr
        dec = self._sp_decrement * lr

        w_pot = self._ff_potential[winner_idx]
        active_pot = active_input.unsqueeze(0) & w_pot
        inactive_pot = (~active_input).unsqueeze(0) & w_pot

        self._ff_permanences[winner_idx] = torch.clamp(
            self._ff_permanences[winner_idx]
            + inc * active_pot.float()
            - dec * inactive_pot.float(),
            0.0, 1.0,
        )

        # Update boosting
        alpha = 0.01
        self._mc_duty_cycle *= (1.0 - alpha)
        self._mc_duty_cycle[self._active_mc] += alpha
        if self._boost_strength > 0:
            self._boost_factors = torch.exp(
                self._boost_strength * (self._sparsity - self._mc_duty_cycle)
            )

    # ------------------------------------------------------------------
    # Cell activation
    # ------------------------------------------------------------------

    def _activate_cells(self) -> None:
        """Determine active cells using graded 4-state logic."""
        self._active.zero_()

        k = self.n_cells_per_minicolumn
        active_mcs = torch.nonzero(self._active_mc, as_tuple=True)[0]

        if len(active_mcs) == 0:
            return

        pred_2d = self._predicted.reshape(self.n_minicolumns, k)
        apical_2d = self._apical_predicted.reshape(self.n_minicolumns, k)
        active_2d = self._active.reshape(self.n_minicolumns, k)

        for mc_idx in active_mcs:
            mc = mc_idx.item()
            basal = pred_2d[mc]
            apical = apical_2d[mc]

            has_basal = basal.max() > 0.3
            has_apical = apical.max() > 0.2 if self._use_apical else False

            if has_basal and has_apical:
                # State 1: both predicted — highest confidence
                combined = basal * 0.7 + apical * 0.3
                active_2d[mc] = combined
            elif has_basal:
                # State 2: basal only — normal confidence
                active_2d[mc] = basal
            elif has_apical:
                # State 3: apical only — biased burst
                # Apical biases which cell fires but still "bursts"
                noise = torch.rand(k, device=self.device) * 0.3
                active_2d[mc] = apical * 0.5 + noise
            else:
                # State 4: burst with developmental wiring bias.
                # Each cell has a fixed random preference (self._cell_bias)
                # modulated by the minicolumn's overlap score.  Different
                # inputs produce different overlaps, which interact with
                # the per-cell bias to select different winners — analogous
                # to innate wiring diversity in cortical development.
                overlap_val = self._mc_overlap[mc]
                bias = self._cell_bias[mc] * (overlap_val * 0.001)
                active_2d[mc] = (1.0 / k) + bias

        # Enforce sparsity: top-1 per minicolumn
        self._active = enforce_sparsity(
            self._active, self.n_minicolumns, k, k_per_mc=1,
        )

    # ------------------------------------------------------------------
    # Context and output
    # ------------------------------------------------------------------

    def receive_context(self, **context_signal) -> None:
        """Receive top-down context for apical dendrites.

        If the incoming context has a different dimension than ``n_cells``,
        it is linearly interpolated to match.  This mirrors the topographic
        projection neurons between cortical areas of different sizes.
        """
        active_cells = context_signal.get("active_cells")
        if active_cells is not None:
            if isinstance(active_cells, np.ndarray):
                ctx = torch.from_numpy(
                    active_cells.astype(np.float32)
                ).to(self.device)
            elif isinstance(active_cells, torch.Tensor):
                ctx = active_cells.to(self.device).float()
            else:
                return

            # Project to n_cells when sender has a different cell count.
            if ctx.shape[0] != self.n_cells:
                ctx = torch.nn.functional.interpolate(
                    ctx.unsqueeze(0).unsqueeze(0),
                    size=self.n_cells,
                    mode="linear",
                    align_corners=False,
                ).squeeze()

            self._context = ctx

    def step_from_context(self) -> dict:
        """Run a reduced step using only received context (no sensor input).

        Designed for parent columns that have no sensor module. Uses the
        received context vector as the activation, settles it in Hopfield
        memory, and queries the associative memory for evidence.

        Returns the same dict format as :meth:`step`.
        """
        import torch

        if self._context is None or self._context.abs().sum() < 1e-8:
            return self._empty_result()

        with torch.no_grad():
            # Use context as activation
            self._active = self._context.clone()

            # Enforce sparsity
            self._active = enforce_sparsity(
                self._active, self.n_minicolumns,
                self.n_cells_per_minicolumn,
            )

            # Save pre-settle for Hopfield storage
            pre_settle_active = self._active.clone()

            # Hopfield settling
            beta = self._beta
            if self._use_neuromodulation and self._neuromod is not None:
                beta *= self._neuromod.beta_scale()

            def sparsity_fn(x):
                return enforce_sparsity(
                    x, self.n_minicolumns, self.n_cells_per_minicolumn,
                )

            settled, n_iters = self._hopfield.settle(
                self._active, beta=beta, sparsity_fn=sparsity_fn,
            )
            self._active = settled

            # Surprise from prediction overlap
            pred_overlap = (self._predicted * self._active.abs()).sum()
            active_norm = self._active.abs().sum() + 1e-8
            surprise = 1.0 - float(pred_overlap / active_norm)
            surprise = max(0.0, min(1.0, surprise))
            self._surprise = surprise

            # Object memory
            if self._mode == "train":
                lr = self._base_lr * (0.1 + 0.9 * surprise)
                # Episodic storage (HPC)
                self._episodic_memory.store(
                    pre_settle_active, label=self._current_object,
                )
                if self._current_object:
                    self._associative_memory.learn(
                        self._active, self._current_object, lr=lr,
                    )
                    self._sync_cortical_attractors()
                else:
                    # Auto-label for parent
                    self._auto_label_patterns.append(
                        self._active.detach().cpu()
                    )
                    if len(self._auto_label_patterns) >= 5:
                        label = self._associative_memory.auto_label(
                            self._auto_label_patterns
                        )
                        self._current_object = label
                        self._associative_memory.learn(
                            self._active, label, lr=lr,
                        )
                        self._sync_cortical_attractors()

            elif self._mode == "eval":
                scores = self._associative_memory.recall(self._active)
                for name, score in scores.items():
                    self._evidence[name] = (
                        self._evidence.get(name, 0.0)
                        + score - self._evidence_decay
                    )

            # Update prediction for next step
            self._predicted = self._dendrites.predict(self._active)
            self._step_count += 1

        return {
            "surprise": self._surprise,
            "evidence": dict(self._evidence),
            "mlh": self._get_mlh(),
            "active_cells": self._active.cpu().numpy(),
            "settling_iterations": n_iters,
            "motor_prediction_error": 0.0,
        }

    def get_context_signal(self) -> dict | None:
        """Return context signal for child LMs."""
        if self._step_count == 0:
            return None
        return {"active_cells": self._active.cpu().numpy()}

    @property
    def surprise(self) -> float:
        return self._surprise

    def get_current_mlh(self) -> dict:
        """Return most likely hypothesis."""
        return self._get_mlh()

    def get_all_known_object_ids(self) -> list[str]:
        if self._use_lfm and self._lfm is not None:
            return self._lfm.known_objects
        return self._associative_memory.known_objects

    def _get_mlh(self) -> dict:
        if not self._evidence:
            return {
                "graph_id": None,
                "evidence": 0.0,
                "active_cells_sdr": set(),
            }
        best = max(self._evidence.items(), key=lambda x: x[1])
        return {
            "graph_id": best[0],
            "evidence": best[1],
            "active_cells_sdr": set(
                torch.nonzero(self._active.abs() > 1e-6, as_tuple=True)[0]
                .cpu().tolist()
            ),
        }

    def _empty_result(self) -> dict:
        return {
            "surprise": 0.0,
            "evidence": dict(self._evidence),
            "mlh": self._get_mlh(),
            "active_cells": set(),
            "settling_iterations": 0,
            "motor_prediction_error": 0.0,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        sd = {
            "hopfield": self._hopfield.state_dict(),
            "associative_memory": self._associative_memory.state_dict(),
            "episodic_memory": self._episodic_memory.state_dict(),
            "ff_permanences": self._ff_permanences.cpu(),
            "ff_potential": self._ff_potential.cpu(),
            "boost_factors": self._boost_factors.cpu(),
            "mc_duty_cycle": self._mc_duty_cycle.cpu(),
            "evidence": dict(self._evidence),
        }
        if self._use_lfm and self._lfm is not None:
            sd["location_feature_memory"] = self._lfm.state_dict()
        return sd

    def load_state_dict(self, sd: dict) -> None:
        self._hopfield.load_state_dict(sd["hopfield"])
        self._associative_memory.load_state_dict(sd["associative_memory"])
        if "episodic_memory" in sd:
            self._episodic_memory.load_state_dict(sd["episodic_memory"])
        if "location_feature_memory" in sd and self._lfm is not None:
            self._lfm.load_state_dict(sd["location_feature_memory"])
        self._ff_permanences = sd["ff_permanences"].to(self.device)
        self._ff_potential = sd["ff_potential"].to(self.device)
        self._boost_factors = sd["boost_factors"].to(self.device)
        self._mc_duty_cycle = sd["mc_duty_cycle"].to(self.device)
        self._evidence = dict(sd.get("evidence", {}))
