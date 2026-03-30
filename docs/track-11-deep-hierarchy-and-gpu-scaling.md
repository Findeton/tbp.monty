# Track 11: Deep Hierarchy and GPU-Parallel Scaling

## Goal

Scale from individual laminar columns (Track 10) to a **deep, GPU-parallel
cortical hierarchy**: multiple columns per cortical area running as batched
tensor operations, 5+ area levels with feedforward/feedback streams
connected via thalamic relays, cross-modal convergence, and efficient
GPU memory management. The full system runs end-to-end through
`MontyObjectRecognitionExperiment` with real 3D models.

## Depends On

Track 10 (Intra-Column Biological Depth). The laminar column with L1–L6,
multi-head dendrites, inhibitory interneurons, STDP, eligibility traces,
and thalamocortical gating must be complete before Track 11 begins.

## Problem

After Track 10, each column is biologically deep but runs in isolation.
There is no GPU-parallel multi-column execution, no dense lateral
connectivity within a cortical area, no deep feedforward/feedback
hierarchy across areas, and no thalamic routing between areas. The Monty
vote protocol handles inter-column communication but operates through
Python-level message passing, not batched tensor ops.

---

## Sections

### B1: GPU-Parallel Multi-Column Execution

Run N columns simultaneously as a batched tensor operation.

**Architecture**:

All columns within a cortical area share the same tensor layout.
Per-area tensors have a leading batch dimension:

```
# N columns in one area, each with n_cells:
x_area: (N, n_cells)           # all column activations
Ξ_area: (N, n_patterns, n_cells) # per-column Hopfield memories
D_area: list of N sparse tensors  # or batched sparse (if same topology)
```

All per-column operations (spatial pooling, Hopfield settling, dendritic
prediction, Hebbian learning) are batched matmuls operating on the first
dimension. No Python loops over columns.

**Lateral connections within an area**:

Columns within an area are connected laterally (this is the voting
mechanism, but now implemented as direct tensor operations instead of
the Monty vote protocol):

```
# Lateral excitation: each column receives weighted input from neighbors
lateral_input = W_lateral @ x_area.T  # (N, N) @ (N, n_cells).T
# → (N, n_cells): each column gets a contribution from all others

# This replaces send_out_vote / receive_votes for within-area communication
# The vote protocol is still used for between-area communication via Monty
```

Lateral connections are sparse (each column connects to ~10% of others
in the area) and learned via Hebbian rules (columns that co-fire
strengthen their lateral connections).

**Implementation**:
- `CorticalArea` class: manages a batch of `CorticalColumnTorch` as a
  single set of batched tensors
- `CorticalArea.step(inputs)`: runs all columns in parallel, applies
  lateral connections, returns all outputs
- `CorticalAreaLM(LearningModule)`: wraps `CorticalArea` in the Monty
  interface. From Monty's perspective, it's one LM (one entry in
  `sm_to_lm_matrix`), but internally it's N parallel columns.

**Tests**:
- N columns produce same output as N sequential column.step() calls
  (before lateral connections)
- Lateral connections shift column activations toward consensus
- GPU speedup: N=64 columns in batched mode faster than sequential
- Memory usage scales linearly with N

---

### B2: Deep Feedforward/Feedback Hierarchy

Stack cortical areas into a hierarchy with feedforward and feedback
streams, connected via thalamic relays.

**Architecture**:

```
Area 1 (V1-like: edges, colors)
  ↓ feedforward via thalamus
Area 2 (V2-like: textures, contours)
  ↓
Area 3 (V4-like: shapes, object parts)
  ↓
Area 4 (IT-like: whole objects)
  ↓
Area 5 (PFC-like: categories, goals)

  ↑ feedback via thalamus (all levels)
```

Each area receives feedforward input from the area below (via thalamic
relay) and feedback from the area above (via thalamic relay). The
thalamic relays implement the gating from Track 10 A7, so each area
can modulate what it receives from below and above.

**Inter-area connections** (using Track 10 A1 laminar subtypes):
- Feedforward: **L5b** (thin-tufted, cortical feedforward) of lower area
  → thalamic relay → L4 of higher area. L5b carries the hierarchical
  representation signal, NOT the motor command (which is L5a).
- Feedback: **L6b** (corticocortical feedback) of higher area → thalamic
  relay → L1 of lower area (apical dendrites of L5a cells). L6b handles
  inter-area feedback, while L6a is reserved for intra-column
  thalamocortical modulation (Track 10 A7).
- Motor output: **L5a** (thick-tufted, subcortical) projects to motor
  system / striatum / brainstem. This is independent of the hierarchy
  feedforward stream. A column can send a motor command (L5a) while
  simultaneously propagating a different representation upward (L5b).
- Feedforward connections (L5b → thal → L4) are learned Hebbian
  (what patterns in the lower area predict patterns in the higher area)
- Feedback connections (L6b → thal → L1) are also Hebbian (what
  higher-area states predict lower-area states)

**Representations become increasingly abstract**:
- Area 1: encodes raw sensory features (edges, colors, curvatures)
  — direct encoder output
- Area 2: encodes combinations of Area 1 features (textures, local shape)
  — learned by Hopfield settling over Area 1 outputs
- Area 3: encodes parts (handle, spout, rim of a mug)
  — learned by Hopfield settling over Area 2 outputs
- Area 4: encodes whole objects (mug, banana)
  — learned by Hopfield settling over Area 3 outputs
- Area 5: encodes categories and task context
  — learned from reward signals and cross-modal input

Each area has its own Hopfield memory (Ξ), so storage capacity
multiplies across levels. 5 areas × exponential capacity per area =
compositional representation power.

**Implementation**:
- `CorticalHierarchy` class: ordered list of `CorticalArea` instances
  with inter-area connection matrices and thalamic relays
- `CorticalHierarchy.step(sensory_input)`: propagates through areas
  bottom-up, then top-down (one step of each per call, matching
  biological propagation delays)
- `CorticalHierarchyLM(LearningModule)`: wraps the full hierarchy in
  the Monty interface. Single LM from Monty's perspective.

**Tests**:
- 3-area hierarchy: lower areas activate before higher areas (temporal
  ordering matches biology)
- Feedback from Area 3 biases Area 1 representation (top-down priming)
- Object recognition accuracy increases with hierarchy depth (1 vs 3
  vs 5 areas) on YCB objects
- Each area's Hopfield memories store increasingly abstract patterns
- Disabling feedback reproduces pure feedforward behavior

---

### B3: Thalamic Routing Between Areas

Track 10 A7 introduces the thalamic relay for within-column predictive
gating (L6 → thalamus → L4). This section extends the same relay
architecture to **inter-area** routing: the thalamus routes information
between cortical areas based on attention and task demands, not just
fixed anatomical connections.

**Architecture**:

Each inter-area connection goes through a thalamic relay with a
learned gate:

```
# Feedforward: Area_low.L5 → Thalamic relay → Area_high.L4
relay_ff = ThalamicRelay(
    input_dim=area_low.n_output,
    output_dim=area_high.n_input,
)

# The relay gate is modulated by:
# 1. Area_high.L6 feedback (top-down expectation)
# 2. Neuromodulatory state (arousal, attention)
# 3. Learned routing weights

gate = sigmoid(
    W_feedback @ area_high.L6_activation
    + W_arousal * arousal
    + bias
)
relayed_input = gate * (W_transform @ area_low.L5_output)
```

This enables:
- **Selective attention**: Higher areas can suppress irrelevant lower-area
  input by closing the thalamic gate
- **Task-dependent routing**: Different tasks activate different higher-area
  patterns → different gating → different information flows up
- **Pulvinar-like coordination**: The pulvinar nucleus coordinates
  information flow between cortical areas that aren't directly connected.
  Implement as a "hub" thalamic relay that receives from multiple areas
  and broadcasts to multiple areas.

**Tests**:
- Thalamic gate modulates inter-area signal strength
- Top-down attention (high-area L6) selectively gates lower-area input
- Different task contexts produce different routing patterns
- Removing thalamic gating (gate=1 always) reproduces direct connections

---

### B4: Cross-Modal Integration

Multiple sensory modalities (vision, touch, proprioception) feed separate
low-level hierarchies that converge at higher levels.

**Architecture**:

```
Visual hierarchy:   V1 → V2 → V4 → IT ─┐
                                         ├→ PFC (multi-modal)
Tactile hierarchy:  S1 → S2 → SII ─────┘
                                    ↑
Motor hierarchy:    M1 ← PM ← PFC ─┘
```

Each modality has its own low-level areas with modality-specific
encoders. Higher areas receive convergent input from multiple
modalities via thalamic relays. The PFC-like top area integrates
all modalities and sends feedback to all.

**Implementation**:
- Each modality maps to a Monty sensor module (CameraSM, etc.)
- Each modality feeds its own low-level `CorticalArea`
- Higher areas receive from multiple lower areas via `CorticalHierarchy`
- The full system is one `CorticalHierarchyLM` from Monty's perspective

**Tests**:
- Visual-only input activates visual hierarchy but not tactile
- Multi-modal input produces stronger higher-area activation than
  single-modality (supralinear integration at convergence)
- Top-down feedback from PFC biases both visual and tactile areas

---

### B5: Efficient GPU Memory Management

At scale (64 columns × 5 areas × 16K cells/column = ~5M cells), memory
becomes a concern. Strategies:

- **Sparse storage**: Dendritic matrices and lateral connections stored as
  sparse tensors (COO/CSR). Only nonzero permanences consume memory.
- **Gradient-free**: No computation graph, no autograd, no optimizer state.
  All memory is weights + activations. This is ~3× more efficient than
  equivalent-parameter transformers during training.
- **Mixed precision**: Weights in float16, activations in float32 during
  settling (for numerical stability of softmax), learning updates in
  float32 accumulated into float16 weights.
- **Streaming patterns**: Hopfield pattern storage (Ξ) can use a
  fixed-size ring buffer. Old patterns are overwritten by new ones.
  The weight matrix `W_assoc` retains the Hebbian trace of all patterns
  ever learned, even after Ξ forgets them.
- **Multi-GPU**: Different cortical areas on different GPUs, with
  inter-area communication via NCCL. Each area is a self-contained
  batch computation.

**Tests**:
- Memory usage for 5-area × 64-column hierarchy fits in 16GB GPU
- Mixed precision produces same accuracy as float32 (within tolerance)
- Ring buffer pattern storage: old patterns gracefully forgotten,
  recent patterns retrieved accurately

---

### B6: Communication Through Coherence (Inter-Area Oscillatory Gating)

Track 10 A8 introduces oscillatory dynamics within a column: gamma
emerges from PV+ interneuron circuits, theta from thalamocortical/motor
loops. This section extends oscillations to **inter-area communication**.

**Problem**: Thalamic gating (B3) provides a learned, feedforward/feedback
gate between areas, but it's a static gain per connection. In real
cortex, the *timing* of communication matters: two areas exchange
information effectively only when their gamma oscillations are
**phase-aligned** (Fries 2015, "Communication Through Coherence").
This is a fundamentally different gating mechanism from thalamic
amplitude gating — it gates by *when* signals arrive, not *how strong*
they are.

**Limitations without CTC** (tests that MUST FAIL with thalamic gating alone):

1. **Selective listening**: Area 3 receives feedforward input from
   both Area 1 (visual) and Area 2 (tactile). With thalamic gating,
   suppressing one modality requires explicitly closing its thalamic
   gate. There's no mechanism for Area 3 to selectively "tune in" to
   Area 1 while Area 2's gate remains open. Expected: when both
   modalities provide conflicting evidence, Area 3 cannot consistently
   prefer one over the other without explicit gate changes.

2. **Flexible routing without weight changes**: Present the same
   stimulus to two different task contexts. The routing should change
   (different higher areas should receive the information) without
   any weight updates — only phase alignment shifts. Thalamic gating
   alone requires learning new gate weights for each context.
   Expected: novel task context produces chance routing accuracy.

**Architecture**:

Each area's PV+ circuit oscillates at its own intrinsic gamma frequency
(emergent from Track 10 A4/A8). Inter-area communication strength
depends on gamma **phase coherence** between sender and receiver:

```
# Each area has its own emergent gamma phase from PV+ dynamics:
gamma_phase_area1 = area1.pv_circuit.current_phase()
gamma_phase_area3 = area3.pv_circuit.current_phase()

# Phase coherence: how aligned are the two areas' gamma oscillations?
phase_diff = gamma_phase_area1 - gamma_phase_area3
coherence = cos(phase_diff)  # 1.0 = perfectly aligned, -1.0 = anti-phase

# Effective inter-area transmission = thalamic_gate × coherence_gate
coherence_gate = relu(coherence)  # only positive coherence transmits
effective_signal = thalamic_gate * coherence_gate * feedforward_signal
```

**Phase alignment is controlled by top-down attention**:
- A higher area can shift its gamma phase (by modulating its PV+
  inhibition timing) to align with a selected lower area
- This is how selective attention works: attend to visual input by
  synchronizing PFC gamma with V1 gamma; ignore tactile by letting
  PFC desynchronize from S1
- The phase shift is fast (single gamma cycle to realign) and requires
  no weight changes — attention is purely dynamic

```
# Top-down attention shifts gamma phase to match target area:
attention_to_area1 = softmax(task_context @ area_keys)
phase_nudge = attention_to_area1 * align_strength
area3.pv_circuit.nudge_phase(phase_nudge_toward=area1_phase)
```

**Why areas have different intrinsic frequencies (and why it helps)**:
- Each area's PV+ population has different parameters → different
  natural gamma frequency (e.g., V1 ≈ 60Hz, PFC ≈ 35Hz)
- Two areas synchronize by **entraining** to a common frequency —
  the sender speeds up or the receiver slows down slightly
- When attention releases, areas return to their natural frequencies
  and communication drops — automatic disengagement
- If all areas had identical gamma, you couldn't selectively pair
  them because all would always be in sync

**Theta coordination across areas**:
- Theta provides a slower coordination frame: all areas in a
  hierarchy share a common theta rhythm (driven by thalamocortical
  loops)
- The theta phase resets at significant events (saccade onset,
  object contact) to align all areas' temporal frames
- Within the shared theta cycle, gamma-phase coherence determines
  which specific area pairs communicate

**Biological basis**: Fries (2015) — "Rhythms for Cognition:
Communication through Coherence"; Bastos et al. (2015) — "Visual
areas exert feedforward and feedback influences through distinct
frequency channels" — feedforward in gamma, feedback in alpha/beta;
Gregoriou et al. (2009) — attention increases gamma coherence between
V4 and FEF.

**Tests** (after implementation):
1. **Phase coherence gates transmission**: Aligned areas transmit;
   misaligned areas don't, even with thalamic gate open.
2. **Selective listening via phase alignment**: Area 3 attends to
   Area 1 (synchronized gamma) while Area 2's signal is attenuated
   (desynchronized), without closing Area 2's thalamic gate.
3. **Flexible routing without weight changes**: Same stimulus, two
   task contexts → different phase alignments → different routing
   → correct task-dependent behavior. No weight updates needed.
4. **Phase entrainment dynamics**: Two areas with different intrinsic
   gamma frequencies synchronize within 2-3 gamma cycles when
   coupling is activated, and desynchronize within 1-2 cycles when
   coupling is released.
5. **Feedforward gamma, feedback alpha/beta**: Feedforward signals
   (lower→higher) preferentially transmit during gamma coherence;
   feedback signals (higher→lower) preferentially during alpha/beta
   coherence (matching Bastos et al. 2015).
6. **Disabling CTC** (coherence_gating=False) reproduces B3 behavior
   with thalamic gating only.

---

## Implementation Phases

### Phase 1: GPU-Parallel Multi-Column (B1)

Create `CorticalArea` with batched tensor operations. Add lateral
connections within area.

**Tests**: N=64 columns run in parallel. Lateral connections produce
consensus. GPU speedup measurable.

---

### Phase 2: Deep Hierarchy (B2)

Stack `CorticalArea` instances into `CorticalHierarchy`. Add inter-area
feedforward/feedback via thalamic relays.

**Tests**: 5-area hierarchy on YCB objects. Accuracy increases with
depth. Feedback improves lower-area representations.

---

### Phase 3: Thalamic Routing (B3)

Learned thalamic gating between areas. Task-dependent routing.

**Tests**: Top-down attention selectively gates inter-area signals.
Different contexts produce different routing.

---

### Phase 4: Cross-Modal Integration (B4)

Multi-modal convergence at higher levels. CameraSM + ChangeDetectingSM
feeding separate low-level areas.

**Tests**: Multi-modal integration stronger than single-modal.
Top-down feedback biases both modalities.

---

### Phase 5: Memory Optimization (B5)

Mixed precision, sparse storage, multi-GPU support.

**Tests**: 5-area × 64-column hierarchy fits in 16GB GPU. Mixed
precision matches float32 accuracy.

---

### Phase 5b: Communication Through Coherence (B6)

First: run the two CTC limitation tests (selective listening, flexible
routing) with thalamic gating only and verify they FAIL. Then: add
inter-area gamma phase coherence gating, top-down phase alignment,
theta coordination. Verify the same tests now PASS.

**Tests**: Limitation tests fail → implement → limitation tests pass.
Phase coherence gates inter-area transmission. Selective attention
via phase alignment works without weight changes.

---

### Phase 6: Full Eval

Full benchmark on YCB + animated models through
`MontyObjectRecognitionExperiment`.

**Integration tests**:

1. **Single-area, 16 columns, 5 YCB objects**: Basic parallel column
   recognition with lateral voting.
2. **3-area hierarchy, 8 columns/area, 10 YCB objects**: Deep
   hierarchy benchmark. Report accuracy vs depth.
3. **5-area hierarchy, 64 columns/area, 15 YCB objects**: Full-scale
   benchmark. Target: >90% accuracy within 50 steps.
4. **Multi-modal**: CameraSM + ChangeDetectingSM feeding separate
   low-level areas, converging at Area 4. Animated Fox.glb.
5. **Novelty + online learning**: Train on 10 objects. Present 5 novel
   objects. System detects novelty (energy-based), learns them online,
   re-recognizes on second presentation.
6. **Attention routing**: Train on all 15 YCB objects. At eval time,
   provide top-down "look for kitchen objects" context to PFC area.
   Verify thalamic routing biases recognition toward mug/bowl/plate.
7. **Communication Through Coherence**: 3-area hierarchy with two
   conflicting input streams. Top-down gamma phase alignment selectively
   routes one stream. Routing changes with task context without weight
   updates. Phase-selected stream accuracy > 80%, suppressed stream
   accuracy < 30%.

**Acceptance criteria**:
- All 7 integration tests pass
- 15-object YCB benchmark at >90% accuracy within 50 steps
- Full hierarchy runs on single GPU (≤16GB)
- No backprop anywhere — verified by absence of .backward() calls
- All tests use real Panda3D rendering with real .glb meshes

---

## Execution Order & Dependencies

```
Track 10 complete
  └→ Phase 1 (GPU parallel columns)
       └→ Phase 2 (deep hierarchy)
            └→ Phase 3 (thalamic routing)
                 ├→ Phase 4 (cross-modal)
                 ├→ Phase 5 (memory optimization)
                 └→ Phase 5b (communication through coherence)
                      └→ Phase 6 (full eval, depends on 4 + 5 + 5b)
```

Phases 1–3 are sequential (each builds on the previous). Phases 4, 5,
and 5b can be developed in parallel once Phase 3 is done: cross-modal
integration (4) and memory optimization (5) don't depend on CTC (5b),
and vice versa. Phase 6 depends on all previous phases.

---

## File Structure

```
src/tbp/monty/frameworks/models/cortical_column_torch/
├── (Track 9 + 10 files unchanged)
├── cortical_area.py          # CorticalArea (batched multi-column)
├── cortical_hierarchy.py     # CorticalHierarchy (deep stack)
├── inter_area_relay.py       # Inter-area thalamic routing
├── coherence.py              # Inter-area gamma phase coherence (CTC)
├── cortical_area_lm.py       # CorticalAreaLM (Monty adapter)
└── cortical_hierarchy_lm.py  # CorticalHierarchyLM (Monty adapter)

tests/unit/frameworks/models/test_cortical_column_torch/
├── (Track 9 + 10 tests unchanged)
├── test_cortical_area.py
├── test_cortical_hierarchy.py
├── test_inter_area_relay.py
├── test_coherence.py
└── test_coherence_limitations.py  # Thalamic-gating-only failure cases

tests/integration/frameworks/models/
└── test_cortical_hierarchy_monty.py

src/tbp/monty/conf/experiment/
└── cortical_hierarchy_eval.yaml
```

---

## Principles

1. **Additive, not destructive**: All Track 9 and 10 tests pass throughout.
2. **Feature flags**: Hierarchy depth, lateral connectivity, thalamic
   routing, cross-modal integration all configurable. Minimal config
   (1 area, 1 column) reproduces Track 10 exactly.
3. **Biology drives architecture**: Every inter-area connection pattern
   maps to known cortical anatomy (V1→V2 feedforward, V2→V1 feedback,
   pulvinar coordination).
4. **GPU-first**: All computation is batched tensor ops. No Python loops
   over columns, areas, or connections in the inner loop.
5. **Monty integration**: From Monty's perspective, the entire hierarchy
   is one LM (`CorticalHierarchyLM`). The internal multi-area, multi-column
   structure is invisible to the experiment harness.
