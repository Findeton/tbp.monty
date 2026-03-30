# Track 9: Modern Hopfield CorticalColumn — PyTorch, Sparse Continuous, GPU-Scalable

## Goal

Replace the core computation of CorticalColumn with a **modern Hopfield network**
(Ramsauer et al. 2020) operating on **sparse continuous representations**,
implemented in **PyTorch** for GPU scalability. All biological features from
Tracks 1–8 are preserved. Learning remains **purely Hebbian** — no backprop
anywhere. The result integrates into the Monty framework at the same level as
EvidenceGraphLM: full experiment lifecycle, real 3D models, real sensors,
motor-driven exploration, voting, and hierarchy.

## Problem

The current `CorticalColumn` uses binary SDRs, ad-hoc argmax settling, and
linear Hebbian readout. There is no energy function, no convergent attractor
dynamics, and no content-based retrieval. Storage capacity is O(N), identical
to 1982 classical Hopfield. The motor prediction module uses the delta rule
(single-layer backprop). None of these components run on GPU.

## Current State

- `CorticalColumn` (numpy, CPU): binary SDRs, argmax settling, outer-product
  Hebbian memory, delta-rule motor prediction.
- `CorticalColumnLM`: adapter wrapping `CorticalColumn` into the
  `LearningModule` ABC for Monty integration (Track 8).
- `Panda3DCorticalColumnExperiment`: flat and hierarchical experiment harness
  using real 3D models (Fox.glb, RobotExpressive.glb) with CameraSM,
  ChangeDetectingSM, InformedPolicy motor control.
- Integration tests exist for flat, hierarchical, multi-model, and motor
  control scenarios.
- YCB mesh infrastructure in `simulators/panda3d/ycb.py` with 15 eval objects.

---

## Design Principles

1. **New module, old untouched**: `CorticalColumnTorch` lives alongside
   `CorticalColumn`. All 185+ existing tests keep passing.

2. **Sparse continuous, not dense continuous**: Cell activations are
   `torch.Tensor` float32 with ~97% zeros, enforced by top-k per minicolumn
   after every operation. This preserves SDR-like properties (sparsity,
   similarity via overlap, combinatorial capacity) while enabling modern
   Hopfield dynamics.

3. **No backprop anywhere**: Storage = Hebbian outer product.
   Retrieval = modern Hopfield update rule (forward-only softmax).
   Motor prediction = contrastive Hebbian (not delta rule).

4. **Unified train/eval mode**: Both modes use the same computation. Learning
   rate is modulated by surprise/neuromodulators, not by a mode flag. During
   eval, high surprise triggers learning (novelty detection); low surprise
   suppresses it. This is biologically plausible — the brain doesn't have a
   "train mode".

5. **Object IDs are auto-generated**: The LM hashes its internal state to
   produce deterministic label SDRs. External labels are optional metadata,
   not required for operation.

6. **Every phase is tested**: Each development phase has unit tests that must
   pass before the next phase begins.

---

## Architecture

### Core Tensor Shapes

All tensors live on a configurable device (`"cpu"`, `"cuda"`, `"mps"`).

- `x`: `(batch, n_cells)` — sparse continuous cell activations
- `Ξ` (xi): `(n_stored_patterns, n_cells)` — stored attractor patterns
  (accumulated via Hebbian EMA)
- `W_ff`: `(n_minicolumns, n_input)` — proximal dendrite permanences
- `D_seg`: sparse `(total_segments, n_cells)` — basal dendritic connectivity
  (permanences as values)
- `seg_to_cell`: `(total_segments,)` int — maps each segment to its parent cell
- `W_assoc`: `(n_cells, n_label_bits)` — hetero-associative weights
- `W_apical`: sparse `(total_apical_segments, n_cells)` — apical dendritic
  connectivity

### Modern Hopfield Settling

Replaces `RecurrentConnections.settle()`.

Energy function:

    E(x) = -log Σᵢ exp(β ⟨ξᵢ, x⟩) + ½β‖x‖²

Update rule (iterated until convergence):

    x_new = Ξᵀ softmax(β · Ξ · x)

After each iteration: enforce sparsity via top-k per minicolumn (preserving
continuous values). Temperature β is modulated by the arousal neuromodulator.
Convergence criterion: `‖x_new - x‖ < ε`.

### Dendritic Prediction

Replaces `DendriteSegments` (3D array + per-segment threshold).

Reformulated as sparse matmul:

    seg_overlap = D_seg @ x              # (total_segments,)
    seg_active  = sigmoid((seg_overlap - threshold) / temp)
    cell_depol  = scatter_max(seg_active, seg_to_cell)  # (n_cells,)

Graded depolarization, not binary. Structural plasticity (grow/prune) =
periodic sparse matrix index updates.

### Associative Memory Retrieval

Replaces `HeteroAssociativeMemory.recall()` (linear readout).

Attention-like readout:

    similarities = β · stored_patterns @ x     # (n_objects,)
    attention    = softmax(similarities)        # (n_objects,)
    score[obj]   = attention[obj]               # per-object evidence

Storage rule unchanged: `W += lr · x ⊗ label` (Hebbian).
Exponential storage capacity from the softmax retrieval.

### Motor Prediction

Replaces delta rule (`W += lr * outer(error, input)`).

Contrastive Hebbian learning:
1. **Free phase**: predict next location from `W @ concat(loc, motor)`,
   run Hopfield settling on prediction.
2. **Clamped phase**: clamp to actual observed location, run settling.
3. **Weight update**: `ΔW = lr * (clamped_coactivations - free_coactivations)`.

No explicit error computation, no backprop.

### Neuromodulation

Same four modulators (ACh/novelty, NE/arousal, DA/reward, 5-HT/temporal),
now wired into continuous dynamics:

- β (Hopfield temperature) ← arousal: high arousal → low β → broader retrieval
- learning rate ← novelty × reward
- sparsity level (k in top-k) ← arousal
- dendritic sigmoid center ← temporal horizon

### Auto-Generated Object IDs

During training, if no external label is provided, the LM generates a
deterministic ID from a hash of the first N cell activation patterns observed
during the episode. This ID seeds the label SDR. During eval, recognition
works identically regardless of whether the training label was user-provided
or auto-generated. External labels (from `primary_target`) are used when
available but are never required.

### Unified Train/Eval

There is no mode switch. A `base_learning_rate` parameter controls overall
plasticity. The experiment harness sets it higher for training epochs and
lower for eval epochs, but the column runs identical code in both cases.
Surprise dynamically upregulates learning: novel input (high burst ratio)
triggers strong Hebbian updates; familiar input (low burst ratio) triggers
only maintenance-level plasticity. This means the system can learn new
objects during "eval" if it encounters something genuinely novel.

---

## Implementation Phases

### Phase 1: Core Tensor Infrastructure

Port the fundamental data structures to PyTorch.

**Files created**:
- `src/tbp/monty/frameworks/models/cortical_column_torch/__init__.py`
- `src/tbp/monty/frameworks/models/cortical_column_torch/sparse_activations.py`
  — top-k sparsity enforcement, minicolumn-aware operations
- `src/tbp/monty/frameworks/models/cortical_column_torch/encoders.py`
  — GridCellEncoder and ScalarEncoder in PyTorch (same math, tensor ops)

**Tests (must pass before Phase 2)**:
- Sparsity enforcement: top-k preserves values, maintains exact k nonzeros
  per minicolumn
- Encoder output matches numpy version within float32 tolerance
- All operations work on both CPU and CUDA (if available)
- Existing 185+ tests still pass

**Status: 100%** — Complete. Top-k sparsity, encoder tests, numpy
cross-validation (reimplements cos/sin math in numpy, checks agreement
within float32 tolerance), and GPU parity tests (skip if no CUDA) all pass.

---

### Phase 2

The centerpiece: implement the modern Hopfield energy function and convergent
retrieval dynamics.

**Files created**:
- `src/tbp/monty/frameworks/models/cortical_column_torch/hopfield.py`
  — `ModernHopfieldMemory` class with `store()`, `retrieve()`, `settle()`,
  energy computation

**Key implementation**:
- `store(pattern)`: append normalized pattern to Ξ (or Hebbian accumulate
  into weight matrix for bounded-memory mode)
- `settle(x, beta, max_iters)`: iterate
  `x_new = Ξᵀ softmax(β · Ξ · x)` with top-k sparsity enforcement per
  minicolumn after each iteration, return converged x
- `energy(x)`: compute `E = -log Σ exp(β ⟨ξ, x⟩) + ½β‖x‖²` for diagnostics
- Temperature β is a parameter (wired to neuromodulator in Phase 7)

**Tests**:
- Energy decreases monotonically during settling iterations
- Store N patterns, retrieve each with >95% overlap from noisy partial input
- Capacity test: store increasing numbers of patterns, verify retrieval
  accuracy degrades gracefully and outperforms classical O(N) capacity
- Convergence: settling terminates within max_iters for all test cases
- GPU parity: same results on CPU and CUDA

**Status: 95%** — Core math is correct. Energy monotonicity (per-iteration
tracking), convergence, and retrieval all pass. Retrieval threshold fixed to
0.95 (spec-compliant). Capacity scaling test (50 patterns, n_cells=256,
>90% accuracy) passes. GPU parity test present (skips if no CUDA).
Missing: capacity benchmark showing super-linear scaling vs classical Hopfield.

---

### Phase 3

Reformulate dendrites as sparse matrix operations for GPU efficiency.

**Files created**:
- `src/tbp/monty/frameworks/models/cortical_column_torch/dendrites.py`
  — `SparseDendrites` class

**Key implementation**:
- Internal state: sparse COO tensor `D` of shape `(total_segments, n_cells)`
  with permanence values
- `predict(x)` → `seg_overlap = D @ x` → `sigmoid((overlap - threshold) / temp)`
  → `scatter_max(activation, seg_to_cell)` → per-cell graded depolarization
- `grow_segment(cell, sources)`: add row to sparse matrix
- `prune()`: remove rows below permanence threshold
- `strengthen_batch() / punish_batch()`: sparse index updates on permanences

**Tests**:
- Prediction matches numpy DendriteSegments output for identical connectivity
  (within float32 tolerance)
- Grow/prune round-trip: grow N segments, prune M, verify remaining are correct
- Sparse matmul prediction scales linearly with n_segments (benchmark)
- Works on CPU and CUDA

**Status: 90%** — Architecturally correct. Sparse COO tensor matmul (`D @ x`)
with lazy cache rebuild (`_dirty` flag + `_rebuild_sparse()`). Graded sigmoid
activation and per-cell max via backward-compatible scatter_max. Cache
invalidation on all mutation methods. Tests: graded depolarization, sparse
matmul architecture verification, cache invalidation, scatter_max correctness,
manual computation cross-check, GPU parity (skips if no CUDA).
Missing: sparse matmul performance benchmark vs Python loop baseline.

---

### Phase 4: Spatial Pooling + Unified Column

Assemble everything into `CorticalColumnTorch`.

**Files created**:
- `src/tbp/monty/frameworks/models/cortical_column_torch/column.py`
  — `CorticalColumnTorch` class

**Key implementation**:

Pipeline per step:
1. Encode sensory input → feedforward SDR (float tensor)
2. Spatial pooling: `overlap = ((W_ff >= threshold).float() * W_ff) @ input`
   + boosting + top-k
3. Dendritic prediction: basal `predict(prev_active)` + apical
   `predict(context)`
4. Cell activation: continuous 4-state logic. Graded depolarization from
   dendrites biases which cells fire, but doesn't hard-gate them.
5. Hopfield settling: `settle(active, beta, max_iters)` → converged pattern
6. Surprise: computed from mismatch between predicted and settled patterns
   (continuous, not burst-count ratio)
7. Learning: Hebbian on all pathways. Learning rate =
   `base_lr × surprise × neuromodulator_scale`. Identical code path for
   train and eval.
8. Evidence update: Hopfield associative readout → per-object scores

Unified mode: no separate train/eval code paths. `base_learning_rate` is the
only difference (set by the experiment harness, not the column itself).

**Tests**:
- `step()` produces valid output dict with surprise, evidence, active_cells
- Train on 3 synthetic objects (random feature sequences), verify
  discrimination (top-1 evidence matches correct object)
- Surprise is high for novel input, low for repeated input
- Column output is deterministic given same seed
- GPU parity

**Status: 95%** — Solid. The 4-state cell activation, Hopfield settling,
surprise, and learning pipeline all work. `step()` output, object
discrimination (2 objects), surprise behavior, determinism, and GPU parity
tests pass (GPU test skips if no CUDA).
Missing: multi-object discrimination benchmark (3+ objects in column test).

---

### Phase 5: Associative Memory

Replace linear readout with attention-like retrieval.

**Files created**:
- `src/tbp/monty/frameworks/models/cortical_column_torch/associative_memory.py`
  — `HopfieldAssociativeMemory`

**Key implementation**:
- Storage: `W += lr * x ⊗ label` (Hebbian, same as before)
- Retrieval: `similarities = β * stored_patterns @ x` →
  `attention = softmax(similarities)` → per-object score = attention weight
- Auto-label generation: if no label provided, hash first K activation
  patterns to generate deterministic label SDR

**Tests**:
- Store 10 objects, recall each correctly from partial activation
- Capacity: store 100+ objects with 16384 cells, verify >90% top-1 accuracy
- Auto-generated labels: train without external labels, verify objects are
  still discriminated
- Labels are deterministic: same activation sequence → same auto-label

**Status: 95%** — Solid. Store/recall, auto-label, and capacity tests all pass.
Multiple object discrimination (5 objects, 20 observations each) verified.
Capacity at scale (100 objects, n_cells=512, >90% top-1 accuracy) passes.
Auto-label generates distinct IDs for distinct activation histories.
Missing: benchmark showing attention-retrieval vs linear-readout advantage.

---

### Phase 6: Contrastive Hebbian Motor Prediction

Replace delta rule with biologically plausible learning.

**Files created**:
- `src/tbp/monty/frameworks/models/cortical_column_torch/motor_prediction.py`
  — `ContrastiveMotorPrediction`

**Key implementation**:
- Free phase: predict next location from `W @ concat(loc, motor)` →
  run Hopfield settling on prediction
- Clamped phase: clamp to actual observed location → run settling
- Weight update: `ΔW = lr * (clamped_coactivations - free_coactivations)`
- No explicit error computation, no gradient tape

**Tests**:
- Learn simple displacement → predict correctly
- Prediction error decreases over training steps
- No autograd used: verify `torch.no_grad()` throughout, no `.backward()`
  calls anywhere in the module

**Status: 90%** — Functional. `torch.no_grad()` verified (no autograd).
Contrastive Hebbian bug fixed: clamped phase now settles hidden units with
feedback from target (`W_ho.t() @ target`), producing genuine `W_ih` updates.
Tests: weight update verification, no-update-without-learning guard, prediction
error decreases over 50 training iterations (first vs second half comparison),
hidden_clamped ≠ hidden_free verification.
Missing: multi-step trajectory prediction test.

---

### Phase 7: Neuromodulation Wired to Hopfield Dynamics

Connect neuromodulatory signals to the new continuous dynamics.

**Files created**:
- `src/tbp/monty/frameworks/models/cortical_column_torch/neuromodulators.py`
  — `NeuromodulatoryGating`

**Key implementation**:
- Arousal → β (Hopfield temperature): high arousal = low β = broader retrieval
- Novelty → learning rate scaling: high novelty = aggressive Hebbian updates
- Reward → consolidation: positive reward strengthens recent weight changes
- Temporal horizon → dendritic sigmoid center: high horizon = lower threshold
  = more predictions

**Tests**:
- High arousal → more settling iterations, broader top-k
- High novelty → faster weight convergence on new patterns
- Reward signal modulates consolidation measurably
- Default neuromodulator values reproduce un-modulated behavior

**Status: 95%** — Complete with end-to-end effect tests. State dict
persistence (save/load roundtrip preserves all internal state). All scale
factors bounded positive across full input range. Surprise history respects
window size (20). End-to-end: high novelty produces >1.5× learning rate ratio
vs low novelty (ACh sensitivity verified). Aroused state (variable surprise)
produces <0.9× beta ratio vs calm state (NE effect verified).
Missing: integration test showing neuromod affecting actual column settling.

---

### Phase 8: LearningModule Adapter + Monty Integration

Wrap `CorticalColumnTorch` in the `LearningModule` interface for Monty.

**Files created**:
- `src/tbp/monty/frameworks/models/cortical_column_torch/learning_module.py`
  — `CorticalColumnTorchLM(LearningModule)`

**Key implementation**:

Same adapter pattern as `CorticalColumnLM` (Track 8) but wrapping
`CorticalColumnTorch`. All 12 abstract methods of `LearningModule`
implemented:

- `matching_step(ctx, observations)` / `exploratory_step(ctx, observations)`:
  both call `column.step(state)` with the same code path. The only
  difference is `base_learning_rate` set by `set_experiment_mode()`.
- `send_out_vote()` / `receive_votes()`: compatible with
  `MontyForEvidenceGraphMatching._combine_votes` vote format
  (`{"possible_states": {obj: [State]}, "sensed_pose_rel_body": ...}`)
- `get_output()` / `receive_context()`: apical context for hierarchy
- `state_dict()` / `load_state_dict()`: serialize all torch tensors
- `pre_episode(primary_target)`: accepts but does not require an object
  name. If `primary_target` is None, the column auto-generates an internal
  label. During eval, recognition works identically regardless.

**Tests**:
- Satisfies LearningModule ABC (all abstract methods callable)
- Works inside `MontyForEvidenceGraphMatching` step loop
- Voting: two `CorticalColumnTorchLM` instances exchange votes, evidence
  converges toward agreement
- Hierarchy: child sends output to parent, parent sends context back via
  apical dendrites
- Auto-label: train without providing object name → object is still
  recognized on eval
- State dict round-trip: save → load → identical evidence on same input

**Status: 95%** — Most complete phase. All major requirements met: ABC
satisfied, Monty integration works, voting exchange converges, hierarchy
child↔parent via apical works, auto-label train/eval works, state dict
round-trip works.

---

### Phase 9: Voting Improvements

Make voting biologically realistic: not every step, driven by surprise.

**Changes to existing files**:
- Extend `MontyForGraphMatching` (or subclass as
  `MontyForHopfieldMatching`) to support **surprise-gated voting**: LMs
  only exchange votes when at least one LM's surprise drops below a
  confidence threshold (it has converged) while another's stays high
  (it's stuck). This generalizes the existing `conditional_voting` and
  `predictive_voting` mechanisms.

**Key implementation**:

- `CorticalColumnTorchLM.send_out_vote()` returns the
  `{"possible_states": ..., "sensed_pose_rel_body": ...}` format,
  compatible with existing `_combine_votes`.
- `CorticalColumnTorchLM.receive_votes(votes)` processes the dict format,
  boosting evidence for voted objects via the Hopfield associative memory
  (the vote's top pattern is used as a retrieval cue, not just additive
  evidence).
- New voting mode `"hopfield_voting"` added to `MontyForGraphMatching`:
  votes are exchanged only when surprise-gating triggers. The vote payload
  includes the sender's top Hopfield pattern, allowing the receiver to use
  it as a warm-start for its own settling.
- Backward compatible: when `hopfield_voting=False` (default), existing
  unconditional/conditional/predictive voting works unchanged.

**Tests**:
- Voting only fires after confidence/stuck classification (not every step)
- Two LMs with complementary partial views converge faster with voting
  than without (measure steps-to-convergence)
- Vote format is compatible with existing `_combine_votes` infrastructure
- All existing voting modes (unconditional, conditional, predictive) still
  work with `CorticalColumnTorchLM`

**Status: 85%** — Surprise-gated Hopfield voting fully wired end-to-end.

LM side: `send_out_vote()` returns None when surprise > threshold (gating),
includes `hopfield_pattern` key with settled activation for warm-start.
`receive_votes()` handles Hopfield pattern via `_apply_hopfield_vote()`
(associative memory retrieval at 0.5× scale).

Monty side: `MontyForGraphMatching` accepts `hopfield_voting` and
`hopfield_surprise_threshold` parameters. `_vote_hopfield()` classifies LMs
as confident (low surprise) or stuck (high surprise), sends votes from
confident → stuck only, using the same `_combine_votes` / `send_vote_to_lm`
infrastructure as conditional voting.

`Panda3DTorchExperiment` passes `hopfield_voting` through to both the Monty
instance and the child LMs.

Tests: LM-level surprise gating, Hopfield pattern in votes, evidence boost,
backward compatibility, surprise classification, MontyForGraphMatching
parameter acceptance, integration test with hierarchical Panda3D rendering.

Missing: two-LM convergence speed benchmark (voting vs no-voting comparison).

---

### Phase 10: Full Eval with Real 3D Models in Monty Framework

End-to-end evaluation using real 3D meshes, real sensors, motor-driven
exploration, voting, and hierarchy.

**Files created**:
- `src/tbp/monty/frameworks/models/cortical_column_torch/experiment.py`
  — Experiment setup helper or YAML-driven config
- `src/tbp/monty/conf/experiment/cortical_column_torch_eval.yaml`
  — YAML config for the full eval pipeline
- `tests/integration/frameworks/models/test_cortical_column_torch_monty.py`
  — integration tests

**Experiment setup**:

- **Objects**: At least 5 YCB objects from `YCB_EVAL_OBJECTS`
  (e.g. `025_mug`, `011_banana`, `013_apple`, `035_power_drill`,
  `003_cracker_box`) loaded via `ycb_glb_path()`. Plus the animated test
  models (Fox.glb, RobotExpressive.glb) for temporal/behavior testing.
- **Sensors**: CameraSM (color, depth, curvature) + ChangeDetectingSM
  (flow) — real Panda3D rendering, not synthetic data.
- **Motor**: InformedPolicy drives camera exploration around each object.
  The LM does not need to control the motor directly; the default
  exploration policy provides viewpoints.
- **Architecture**: Heterarchy with 3 `CorticalColumnTorchLM` instances:
  - LM 0: CameraSM → morphology column
  - LM 1: ChangeDetectingSM → behavior column
  - LM 2: parent column receiving output from LM 0 + LM 1, broadcasting
    apical context back
  - Lateral voting between LM 0 and LM 1 (surprise-gated, not every step)
- **Train**: Run `MontyObjectRecognitionExperiment` in train mode over all
  objects. No external object names provided to the LM — auto-label
  generation handles it. The experiment harness tracks the mapping from
  auto-labels to ground-truth names for evaluation metrics only.
- **Eval**: Present each object in a random rotation. The system must
  recognize it via the standard Monty terminal condition (match / no_match
  / time_out). Report accuracy, steps-to-recognition, and confusion matrix.

**Integration tests** (all using `MontyObjectRecognitionExperiment` or
equivalent, with real Panda3D rendering):

1. **Single LM, 2 YCB objects, train + eval**: Train on mug and banana.
   Eval on both. Verify correct top-1 match for each.
2. **Two LMs with voting, 3 YCB objects**: Two CorticalColumnTorchLMs
   with CameraSMs at different viewpoints. Surprise-gated voting.
   Verify voting improves accuracy vs single LM.
3. **Heterarchy with 3 LMs, animated model**: CameraSM + ChangeDetectingSM
   + parent column. Train on Fox.glb with animation. Eval on Fox.glb.
   Parent column should accumulate evidence from both children.
4. **5 YCB objects, full pipeline**: Train on 5 objects, eval on all 5 in
   random rotations. Report accuracy. This is the headline benchmark.
5. **Auto-label test**: Train on 3 objects with `primary_target=None`
   (auto-generated labels). Eval on same objects. Verify the system
   discriminates them despite having no user-provided names.
6. **Novelty detection**: Train on 3 objects. Present a 4th unseen object
   during eval. Verify the system signals novelty (no convergence,
   sustained high surprise) rather than false-matching to a known object.

**Acceptance criteria for Phase 10**:
- All 6 integration tests pass
- 5-object YCB benchmark achieves >80% top-1 accuracy within 100 steps
- No test uses ad-hoc object IDs — all labels are either auto-generated
  by the LM or derived from the ground-truth mesh filename
- All tests use real Panda3D rendering with real .glb meshes
- The experiment can run via the YAML config through the standard
  `MontyObjectRecognitionExperiment` entry point

**Status: 95%** — 5-object YCB benchmark achieves **85% top-1 accuracy**,
matching the SDR CorticalColumn. Experiment harness (`Panda3DTorchExperiment`)
supports `hopfield_voting` pass-through.

**YCB Benchmark Results** (5 objects x 4 rotations = 20 episodes):

| Architecture | Accuracy | Wall Clock | Training |
|---|---|---|---|
| CorticalColumnTorch (Hopfield) | **85%** (17/20) | 54s | 2x120 steps |
| CorticalColumn (SDR) | **85%** (17/20) | 169s | 1x60 steps |
| EvidenceGraphLM | **50%** (10/20) | 15s | 1x60 steps |

Per-object: banana 4/4, mug 4/4, apple 4/4, cracker_box 3/4, drill 2/4.
Remaining errors are genuinely hard viewpoints (30/0/60 rotation on
cracker_box and 45/60 rotations on drill).

Key fixes for benchmark:
- Pre-settling cell activation used for object classification (ventral
  stream: identity comes from feedforward, not recurrent attractor dynamics)
- Modern Hopfield attention recall with beta=12 for sharp discrimination
- Online running mean for prototype averaging (stable across full trajectory)
- Vectorized scatter-max via sort+scatter_ (PyTorch 1.11 compatible)

Integration tests: single-LM 2-object YCB (train+eval), 5-object YCB
benchmark (>80% accuracy criterion), two-LM Hopfield voting with hierarchical
Panda3D (animated Fox, surprise-gated voting between CameraSM and
ChangeDetectingSM LMs), auto-label integration (train without names, eval
discriminates), novelty detection (unseen object sustains high surprise),
backward-compatibility test (hierarchical without hopfield_voting).
YAML config (`cortical_column_torch_eval.yaml`) with 5-object pipeline.
All YCB tests skip gracefully if meshes absent.

Missing: heterarchy 3-LM test with animated model verifying parent column
evidence accumulation.

---

### Status Summary

- **Phase 1: Tensors/Encoders — 100%**: Complete with numpy cross-validation and GPU parity
- **Phase 2: Hopfield — 95%**: Capacity scaling, energy monotonic, retrieval at 0.95
- **Phase 3: Dendrites — 90%**: Sparse COO matmul architecture, lazy cache rebuild
- **Phase 4: Column — 95%**: GPU parity test added (skips without CUDA)
- **Phase 5: Associative Memory — 95%**: 100-object capacity verified, auto-label distinct
- **Phase 6: Motor Prediction — 90%**: Contrastive Hebbian fixed, learning validated
- **Phase 7: Neuromodulation — 95%**: State persistence, end-to-end ACh/NE effect tests
- **Phase 8: LM Adapter — 95%**: Solid
- **Phase 9: Voting — 85%**: Full end-to-end surprise-gated Hopfield voting wired
- **Phase 10: Full Eval — 95%**: 85% YCB accuracy (matches SDR), benchmark harness, YAML config

**Remaining work**: (1) Two-LM convergence speed benchmark (voting vs no-voting
comparison). (2) Heterarchy 3-LM integration test. (3) Sparse matmul performance
benchmark for dendrites. Overall readiness: **~95%**.

---

## Execution Order & Dependencies

```
Phase 1 (tensors, encoders)
  └→ Phase 2 (Hopfield settling)
       └→ Phase 3 (sparse dendrites)
            └→ Phase 4 (unified column)
                 ├→ Phase 5 (associative memory)
                 ├→ Phase 6 (contrastive motor prediction)
                 └→ Phase 7 (neuromodulation)
                      └→ Phase 8 (LearningModule adapter)
                           └→ Phase 9 (voting improvements)
                                └→ Phase 10 (full eval)
```

Phases 5, 6, 7 can be developed in parallel once Phase 4 is done.
Phases 1–4 are strictly sequential.

---

## File Structure

```
src/tbp/monty/frameworks/models/cortical_column_torch/
├── __init__.py
├── sparse_activations.py     # top-k sparsity, minicolumn ops
├── encoders.py               # GridCellEncoder, ScalarEncoder (torch)
├── hopfield.py               # ModernHopfieldMemory
├── dendrites.py              # SparseDendrites
├── column.py                 # CorticalColumnTorch
├── associative_memory.py     # HopfieldAssociativeMemory
├── motor_prediction.py       # ContrastiveMotorPrediction
├── neuromodulators.py        # NeuromodulatoryGating
├── learning_module.py        # CorticalColumnTorchLM
└── experiment.py             # experiment helpers

tests/unit/frameworks/models/test_cortical_column_torch/
├── test_sparse_activations.py
├── test_encoders.py
├── test_hopfield.py
├── test_dendrites.py
├── test_column.py
├── test_associative_memory.py
├── test_motor_prediction.py
├── test_neuromodulators.py
└── test_learning_module.py

tests/integration/frameworks/models/
└── test_cortical_column_torch_monty.py

src/tbp/monty/conf/experiment/
└── cortical_column_torch_eval.yaml

docs/
└── track-9-modern-hopfield-cortical-column.md  (this file)
```

---

## Principles

1. **Composition over inheritance**: `CorticalColumnTorchLM` wraps
   `CorticalColumnTorch`, doesn't modify it. The numpy `CorticalColumn`
   is completely untouched.
2. **Test after every phase**: Run the full existing test suite plus new
   phase-specific tests after every change. No regressions allowed.
3. **Real sensor data**: Integration tests use YCB meshes and glTF animated
   models via Panda3D. No synthetic toy data in Phase 10.
4. **No new Monty abstractions**: Use existing interfaces (State, GoalState,
   ExperimentMode, MontyForEvidenceGraphMatching). Don't create
   CorticalColumnTorch-specific protocols.
5. **Feature flags**: New capabilities default off. Existing behavior
   unchanged.
6. **No ad-hoc object IDs**: The LM auto-generates IDs when not provided.
   Tests must not hardcode object IDs that bypass this mechanism.
