# Track 6: Predictive Coding Heterarchy

## Purpose

Unify inter-LM communication, error-modulated learning, and hypothesis
management into a coherent predictive coding architecture. Build on burst
sampling's prediction-error-driven hypothesis management to extend the same
principle — computation proportional to surprise — to learning rates,
inter-LM communication, and hierarchical prediction.

This track consolidates work previously scattered across:
- T1.5 (top-down biasing), T1.6 (attention routing) from Track 1
- T2.E (error-modulated temporal learning) from Track 2
- T4.4–T4.7 (faster matching, hypothesis management, parallelism)
  from Systems Engineering

## Roadmap

See [full-system-roadmap-v2.md](full-system-roadmap-v2.md) for full program
context, restriction analysis, concept layers, and priority stack.

## Current Status

Added 2026-03-28. Status: **Phase 1 (T6.1-T6.4) + Phase 2 (T6.5-T6.8) + Phase 4a (T6.13-T6.13a) implemented.**

### What exists today

**Within-LM prediction:**
- TemporalMemory: SDR → Hebbian → spreading activation → predict next
  observation → compute surprise ∈ [0, 1] (56 tests)
- HPC: concept SDR → Hebbian → spreading activation → predict next concept
  → validate prediction (36 tests)
- Surprise → evidence modulation: V1 done (scales evidence delta by surprise,
  off by default: `surprise_boost=0.0`, `surprise_penalty=0.0`)

**Inter-LM communication (three modes, two already condition-gated):**
- **Unconditional voting** (`conditional_voting=False`, default): every LM
  votes to every connected LM every step
- **Conditional voting** (`conditional_voting=True`): LMs classified as
  confident (≤2 possible matches), stuck (>2 after N steps), or undecided.
  Voting fires only when all LMs have settled; confident → stuck only.
  Predates temporal prediction; uses hypothesis cardinality as confusion proxy
- **Temporal voting** (`temporal_voting=True`): runs alongside spatial voting.
  Uses `get_temporal_prediction_status()` — confident (correct prediction)
  LMs vote to confused (wrong prediction) LMs

**Bottom-up (child → parent):**
- `get_output()` sends raw content: object_id + pose + optional packet
  (top-k SDRs, support weights)
- No prediction-error signal; parent always receives full child output

**Top-down (parent → child):**
- **Does not exist.** T1.5 is "not started." The `add-top-down-connections`
  future-work spec describes the intent but no code exists
- HPC → all LMs context broadcast (`_dispatch_context_signals()`) is the
  only downward signal, and it's global, not parent-to-child

**Hypothesis management (burst sampling — the working foundation):**
- `BurstSamplingHypothesesUpdater` (768 lines, now the default for
  benchmarks) replaced `DefaultHypothesesUpdater`. Key mechanisms:
  - **Evidence slope as prediction error**: `EvidenceSlopeTracker` computes
    a sliding-window average of step-to-step evidence changes for every
    hypothesis. The max global slope across all objects and channels is the
    system's prediction-error signal.
  - **Burst trigger**: When max slope ≤ `burst_trigger_slope` (default 1.0),
    a burst starts: new informed hypotheses are sampled for
    `sampling_burst_duration` (default 5) consecutive steps.
  - **Informed top-k sampling**: `_sample_informed()` computes feature
    evidence for all nodes and picks top-k by evidence score, not
    brute-force. Each burst step uses the current observation's features.
  - **Continuous pruning**: Hypotheses with evidence slope < 
    `deletion_trigger_slope` (default 0.5) and age ≥ 5 steps are deleted.
  - **Dynamic hypothesis space**: No fixed size. The space grows during
    bursts and shrinks through pruning. Replaces episode-boundary supervision.
  - **Multi-object capability**: Monty can now move between objects without
    episode boundaries; prediction error triggers resampling automatically.
- Benchmarked: accuracy improvements on similar-object configs, rotation
  error reduction, runtime improvement especially at scale (77 objects).
- Preliminary compositional results: lower-level LM bursts when moving
  between child objects (e.g., logo ↔ disk on mug), passes recognised
  IDs up to higher-level LM.

### Two distinct prediction-error signals

Burst sampling and TemporalMemory compute different prediction errors.
Track 6 must distinguish them or risk conflating unrelated signals:

| Signal | What it measures | Where | Range | Drives |
|---|---|---|---|---|
| **Evidence slope** | "Am I making matching progress?" — rate of evidence accumulation for the best hypothesis | `EvidenceSlopeTracker._max_global_slope()` | [-1, 2] | Burst trigger (hypothesis resampling) |
| **TM surprise** | "Did I predict this observation?" — overlap between predicted and observed SDR | `TemporalMemory.compute_surprise()` | [0, 1] | Nothing real yet (V1 off by default) |

These are **complementary, not redundant**. Currently never connected:

| Evidence slope | TM surprise | Meaning | Action |
|---|---|---|---|
| High | Low | Matching well, predictions correct | Quiescence |
| High | High | Matching well, but unexpected transition | No burst; boost TM learning |
| Low | Low | Matching stalled (e.g. symmetry), prediction correct | Burst needed; TM learning rate normal |
| Low | High | Object changed or major mismatch | Burst + boosted learning |

**Design principle**: Evidence slope drives hypothesis management (burst
sampling). TM surprise drives learning modulation and communication gating.
They should not be unified into one number.

### What's missing

1. **No `receive_prediction()` on LMs** — only `receive_votes()` and
   `receive_context()`
2. **No upward prediction-error signal** — `get_output()` sends content,
   not residuals
3. **No parent-side part-location prediction** — parent stores child
   `object_id` as a feature but can't invert its model to predict which
   part should appear where
4. **No surprise-gated communication** — votes/outputs sent regardless of
   TM surprise level
5. **No unified voting mode** — spatial confusion (cardinality) and
   temporal confusion (prediction error) are independent, parallel passes
6. **Surprise doesn't modulate learning** — Hebbian updates are blind to
   TM prediction error; evidence slope drives burst resampling but not
   learning rates
7. **No offspring/refinement hypotheses** — burst sampling only adds
   informed hypotheses (from scratch per observation). Cannot sample
   refined poses near high-evidence hypotheses for the same object.
   This is the particle-filter-like sharpening needed for precise pose
   estimation once object ID is confident.
8. **No context-based hypothesis biasing** — HPC context dispatch exists
   but doesn't influence which graph IDs burst sampling selects. Scene/task
   priors ("I'm in the kitchen, likely objects: cup, plate, fork") could
   narrow sampling. Independent of hierarchy.

## First-Principles Analysis

### The predictive coding frame

In predictive coding (Rao & Ballard 1999, Friston's free energy):
- Top-down connections carry **predictions** from higher to lower levels
- Bottom-up connections carry **prediction errors** (residuals)
- Each level minimises its prediction error by updating its model (learning)
  or updating its predictions (inference)
- Computation is proportional to prediction error, not data volume

In Monty's current architecture, burst sampling already makes hypothesis
management proportional to prediction error (evidence slope). But three
things remain unconditional:
- **Learning**: Hebbian updates run at fixed rate regardless of TM surprise
- **Communication**: Votes and upward outputs sent every step
- **Init scope**: Even burst sampling samples all graph IDs; no context
  or top-down signal narrows the search space

### Biological mapping

| Cortical circuit | Monty equivalent | Predictive coding role |
|---|---|---|
| L2/3 pyramidal → L2/3 in peer column | Lateral voting | Consensus / mutual error correction |
| L5 pyramidal → higher-area L4 | `get_output()` → parent LM input | Bottom-up: should carry prediction error |
| L6 → L1 apical dendrites in child | `receive_prediction()` (new) | Top-down: carries prediction |
| L6 local → L6 in child | Top-down pose prediction | Location-specific part expectation |
| Hippocampus → cortical columns | `_dispatch_context_signals()` | Episodic context / temporal priors |
| Thalamic relay (pulvinar) | Not yet modeled | Attention-gated routing |
| Dendritic prediction on pyramidal cells | TemporalMemory surprise | Within-column prediction error |
| Burst vs sparse firing | **Burst sampling** (evidence slope triggers hypothesis burst) + confused vs confident LM states | Error magnitude → computation budget |

The TemporalMemory's surprise signal maps to dendritic prediction errors:
when apical prediction matches basal input, only predicted cells fire (sparse,
"confident"); when prediction fails, all cells in the minicolumn burst (dense,
"confused"). This is already described in the V1 docstring and is the
biological basis for surprise gating both learning and communication.

### How the pieces unify

Burst sampling is the working foundation. Everything below extends the
same prediction-error → adaptive-response principle to new dimensions:

```
                    PREDICTION (top-down)
                    ┌──────────────────┐
                    │                  ▼
              ┌─────┴─────┐    ┌──────────────┐
              │ Parent LM │    │   Child LM   │
              │ (abstract) │    │  (concrete)  │
              └─────▲─────┘    └──────┬───────┘
                    │                  │
                    └──────────────────┘
                    PREDICTION ERROR (bottom-up)
```

| Dimension | What already works | What Track 6 adds |
|---|---|---|
| Hypothesis count | Burst sampling: evidence slope → burst/prune | Context biasing, offspring refinement, top-down narrowing |
| Learning rate | Fixed Hebbian LR | TM surprise → modulated LR |
| Communication | 3 independent voting modes | Unified surprise-gated voting |
| Bottom-up signal | Always full content | Prediction-error gated (confirmed vs error) |
| Top-down signal | Does not exist | Parent predictions → child hypothesis seeding |

1. **Burst sampling is the within-LM adaptive compute mechanism.** It
   already makes hypothesis management proportional to evidence slope.
   Track 6 extends this: (a) offspring hypotheses for pose refinement
   when object ID is confident, (b) HPC context biasing to narrow which
   graph IDs get sampled, (c) top-down predictions from parent LMs.

2. **TM surprise modulates learning, not hypothesis management.** The
   evidence slope (burst sampling's signal) and TM surprise are
   complementary. Evidence slope = "am I matching well?" TM surprise =
   "did I predict correctly?" High TM surprise with high evidence slope
   means the matching is right but the temporal model is wrong → boost
   learning, don't burst. Low evidence slope with low TM surprise means
   matching stalled on a symmetric object → burst, don't change LR.

3. **Bottom-up messages become prediction errors.** If the parent's
   prediction was correct, the child sends minimal information ("confirmed").
   Only when the prediction is wrong does the child send a full error signal,
   causing the parent to update. Quiescent children cost almost nothing.

4. **Surprise modulates communication bandwidth.** What drives what:

   | What | Signal used | How | Status |
   |---|---|---|---|
   | Hypothesis resampling | Evidence slope | Burst trigger (already working) | **Done** (burst sampling) |
   | Hypothesis pruning | Evidence slope | Deletion trigger (already working) | **Done** (burst sampling) |
   | Evidence accumulation | TM surprise | Scale delta by surprise | V1 done (off by default) |
   | Hebbian learning rate | TM surprise | `lr × (1 + α × surprise)` | Not started |
   | Weight decay | TM surprise | Low surprise → mild decay | Not started |
   | Bottom-up bandwidth | TM surprise | Low surprise → suppress upward signal | Not started |
   | Top-down prediction weight | Parent confidence | High confidence → stronger child bias | Not started (T1.5) |
   | Hypothesis init scope | HPC context | Context narrows graph IDs sampled | Not started |
   | Lateral vote gating | TM surprise + cardinality | Unified confused/confident | Partially (modes B, C separate) |

5. **Lateral voting becomes error-driven consensus.** Conditional voting
   (Mode B) already gates on spatial confusion. Temporal voting (Mode C)
   already gates on prediction error. The missing piece: a unified mode
   where confusion = (high TM surprise OR high spatial cardinality) and
   confidence = (low TM surprise AND low spatial cardinality).

## Architecture Design

### Message protocol extensions

**Top-down prediction (parent → child):**

```python
# New method on EvidenceGraphLM:
def receive_prediction(self, prediction):
    """Accept top-down prediction from parent LM.
    
    prediction = {
        "predicted_object_ids": ["tire", "door"],
        "predicted_poses": [Rotation, Rotation],
        "predicted_locations": [ndarray(3,), ndarray(3,)],
        "predicted_scales": [1.0, 1.0],
        "confidence": 0.85,
        "sender_id": "parent_lm_1"
    }
    """
    self._pending_prediction = prediction
```

Used in `_get_initial_hypothesis_space_single()`: if a pending prediction
exists, initialise hypotheses from predictions + exploration budget instead
of full enumeration.

**Prediction-error upward signal (child → parent):**

```python
# Extended get_output() variant:
def get_prediction_error(self):
    """Return residual between parent's prediction and child's evidence.
    
    Returns None if no prediction was received or prediction was confirmed.
    Returns error dict if prediction was wrong.
    """
    if self._pending_prediction is None:
        return self._get_full_output()  # No prediction → full content
    
    pred = self._pending_prediction
    actual = self.current_mlh
    
    if self._prediction_confirmed(pred, actual):
        return {"confirmed": True, "surprise": 0.0}
    
    return {
        "confirmed": False,
        "surprise": self._compute_prediction_mismatch(pred, actual),
        "actual_object_id": actual["graph_id"],
        "actual_pose": actual["rotation"],
        "pose_residual": actual_pose - predicted_pose,
    }
```

**Unified vote gating:**

```python
def _is_confused(self, lm):
    """Unified confusion check: spatial OR temporal."""
    spatial = len(lm.get_possible_matches()) > self.vote_confident_threshold
    temporal = (
        hasattr(lm, "get_temporal_prediction_status")
        and lm.get_temporal_prediction_status() == "confused"
    )
    return spatial or temporal

def _is_confident(self, lm):
    """Unified confidence check: spatial AND temporal (if available)."""
    spatial = 0 < len(lm.get_possible_matches()) <= self.vote_confident_threshold
    temporal_ok = (
        not hasattr(lm, "get_temporal_prediction_status")
        or lm.get_temporal_prediction_status() != "confused"
    )
    return spatial and temporal_ok
```

### Error-modulated learning

**Within-episode (TemporalMemory):**

```python
# In step(), after computing surprise:
modulated_lr = self.learning_rate * (1.0 + self.surprise_learning_boost * surprise)
self._W += modulated_lr * np.outer(sdr, prev_sdr)

# Surprise-gated decay for predicted transitions:
if surprise < self.low_surprise_decay_threshold:
    mask = np.outer(sdr, prev_sdr) > 0
    self._W[mask] *= (1.0 - self.low_surprise_decay_rate)
```

**Cross-episode (HPC):**

```python
# In _hebbian_update(), after _validate_prediction():
surprise = self._get_prediction_surprise(observed)
modulated_lr = self.hebbian_learning_rate * (1.0 + self.surprise_learning_boost * surprise)
update = modulated_lr * np.outer(current_sdr, previous_sdr)
self._temporal_W += update
```

All new parameters default to 0.0 for full backward compatibility.

### Hypothesis proposal (builds on burst sampling)

Burst sampling's `_sample_informed()` already does **top-k feature-evidence
ranking** — it computes feature evidence for all nodes and picks the best
matches, not brute force. This is the analytical fingerprint baseline.

What's missing from burst sampling:

1. **Context biasing**: `_sample_informed()` samples from all graph IDs
   equally. HPC context or parent predictions could weight which graphs
   get sampled. Mechanistically: multiply feature-evidence scores by a
   prior weight per graph_id before top-k selection.

2. **Offspring/refinement hypotheses**: Burst sampling only creates
   hypotheses *informed by the current observation* (from-scratch per
   step). It cannot take an existing high-evidence hypothesis and create
   variants with nearby orientations or locations for the same object.
   This is the particle-filter-like refinement needed once object ID is
   confident but pose is imprecise.

3. **Learned retrieval** (only if needed): If the analytical ranking
   (feature evidence) doesn't correlate well with the correct node at
   early steps, a contrastive MLP could improve ranking quality:

```
Observation encoder:
  f(pose_vectors, features) → query embedding q ∈ ℝ^D

Node encoder (precomputed per graph, cached):
  g(node_pose_vectors, node_features) → key embedding k ∈ ℝ^D

Score: q · k → top-K (graph_id, node_id) pairs
Rotation: closed-form via align_multiple_orthonormal_vectors()
Scale: observed/stored curvature ratio
```

One-shot compatible: new objects get node embeddings cached immediately.

### Computational scaling properties

| Scenario | Current (with burst sampling) | With Track 6 additions |
|---|---|---|
| Parent predicts correctly | N/A (no top-down) | Child confirms, near-zero cost |
| Parent predicts wrong | N/A (no top-down) | Child sends error, parent updates |
| Peer LMs agree | All vote every step (unconditional) | Quiescent, no votes (TM surprise gated) |
| Peer LMs disagree | Conditional vote once settled (Mode B) | Unified surprise-gated voting |
| Known object, confident | Hypothesis space already pruned small | Same (burst sampling handles this) |
| Object changes | Burst triggers resampling from all graph IDs | Context-biased resampling (fewer candidates) |
| Object ID confident, pose imprecise | No refinement mechanism | Offspring hypotheses sharpen pose |
| Novel object, high TM surprise | TM surprise unused for learning | Boosted Hebbian LR, faster temporal model improvement |
| Symmetric object (matching stalls) | Burst triggers, many hypotheses persist | Same (correct behavior — genuinely ambiguous) |

## Implementation Plan

### Phase 1: Error-modulated learning within each LM

Make each column a better predictor before building inter-column prediction.

| ID | Milestone | Lines | Dependencies | Status |
|---|---|---|---|---|
| T6.0 | Analytical surprise signal validation | 0 (logging only) | TemporalMemory | **Done** (27 tests) |
| T6.1 | Surprise-modulated Hebbian rate (TM) | ~30 | T6.0 pass | **Done** |
| T6.2 | Surprise-gated decay (TM) | ~40 | T6.1 | **Done** |
| T6.3 | HPC cross-episode error modulation | ~50 | T6.0 pass | **Done** |
| T6.4 | Combined learning + evidence modulation | ~20 | T6.1, V1 | **Done** |

**T6.0 validation gate:** Run extended YCB eval with TemporalMemory enabled
and `learn=True`. Log `(step, surprise, transition_was_novel)`. Surprise must
separate novel from repeated transitions with AUC > 0.7. If not, the signal
is too noisy and Phase 1 stops here.

**T6.1 implementation:**
1. Add `surprise_learning_boost: float = 0.0` to `TemporalMemory.__init__`
2. Modulate Hebbian update: `lr × (1 + boost × surprise)`
3. Default 0.0 = backward compatible

Tests:
- `test_surprise_modulated_lr_noop_when_zero`
- `test_surprise_modulated_lr_novel_transition_stronger`
- `test_surprise_modulated_lr_prediction_improves_faster`

**T6.2 implementation:**
1. Add `low_surprise_decay_rate: float = 0.0`,
   `low_surprise_decay_threshold: float = 0.2`
2. After Hebbian update, decay participating weights if surprise < threshold

Tests:
- `test_decay_prevents_unbounded_growth`
- `test_decay_preserves_novel_transitions`

**T6.3 implementation:**
1. Add `surprise_learning_boost: float = 0.0` to HPC
2. Add `_get_prediction_surprise(observed)` method
3. Pass surprise from `_validate_prediction()` to `_hebbian_update()`

Tests:
- `test_hpc_surprise_novel_concept_stronger`
- `test_hpc_surprise_action_conditioned`

**T6.4:** Enable both surprise-modulated learning and surprise-modulated
evidence together. Sweep `surprise_boost`, `surprise_penalty`,
`surprise_learning_boost` on extended YCB.

### Phase 2: Unified voting, context biasing, and surprise-gated communication

Merge the three independent voting modes and add context-based narrowing.

| ID | Milestone | Lines | Dependencies | Status |
|---|---|---|---|---|
| T6.5 | Unified confusion/confidence classification | ~40 | T6.1 | **Done** |
| T6.6 | Surprise-gated upward messaging | ~80 | T6.5 | **Done** |
| T6.6a | HPC context biasing for burst sampling | ~30 | None | **Done** |
| T6.7 | KDTree workers=-1 + FAISS batch queries | ~50 | None | **Done** (workers=-1; FAISS optional) |
| T6.8 | GIL-free per-object updates | ~80 | None | **Done** (ThreadPoolExecutor) |

**T6.5 implementation:**
Replace `_vote_conditional()` and `_vote_temporal()` with
`_vote_predictive()`:
- Confusion = high **TM surprise** OR high spatial cardinality
- Confidence = low **TM surprise** AND low spatial cardinality
- Note: TM surprise (observation prediction), not evidence slope
  (hypothesis matching progress). An LM can match well (high evidence
  slope) but be temporally confused (high TM surprise) — it should
  still receive votes to help its temporal model.
- Single pass, single temporary vote matrix
- Backward compatible: when `temporal_voting=False` and
  `surprise_learning_boost=0`, reduces to existing Mode B behavior

Tests:
- `test_unified_vote_matches_conditional_when_no_temporal`
- `test_unified_vote_spatially_confident_but_temporally_confused_receives`
- `test_unified_vote_both_confident_suppresses_vote`

**T6.6 implementation:**
Add `surprise_gated_output: bool = False` to EvidenceGraphLM.
When enabled, `get_output()` returns a minimal confirmation dict when
surprise is below threshold. Parent LM skips hypothesis update for
confirmed children.

Tests:
- `test_gated_output_sends_full_when_surprised`
- `test_gated_output_sends_minimal_when_predicted`
- `test_parent_skips_update_on_confirmation`

**T6.6a (context biasing — quick, independent):**
Wire HPC context signals into burst sampling's `_sample_informed()`.
When `_dispatch_context_signals()` sends a context (e.g., scene priors,
task goal), the LM stores a graph-ID weight dictionary. During burst
sampling, multiply each graph's feature-evidence scores by its context
weight before top-k selection. Graphs not in the context get a small
but non-zero weight (exploration budget).

Implementation: ~30 lines in `BurstSamplingHypothesesUpdater._sample_informed`.
Add `context_graph_weights: Optional[dict[str, float]]` field.

This is what Jeff described as "I'm in the kitchen looking for my coffee
cup" — narrowing the hypothesis space via scene/task priors without
requiring any hierarchy. Independent of compositional models.

Tests:
- `test_context_biasing_prefers_context_objects`
- `test_context_biasing_still_finds_unexpected_objects`

**T6.7 (independent quick wins):**
- Change `workers=1` → `workers=-1` in `GridObjectModel.find_nearest_neighbors()`
- Replace scipy KDTree with FAISS `IndexFlatL2` for batch queries
- For T5's per-state KDTrees, use combined index with state-ID filters

**T6.8 (independent quick win):**
- Replace `threading.Thread` in `receive_votes()` with
  `multiprocessing.Pool` or batched vectorised update
- Increase `maxtasksperchild` in run_parallel.py from 1 to 5–10

### Phase 3: Top-down prediction (the hard piece)

Requires compositional models that encode part-location relationships.

| ID | Milestone | Lines | Dependencies | Status |
|---|---|---|---|---|
| T6.9 | `receive_prediction()` method on EvidenceGraphLM | ~60 | T1.R2 (done) | Not started |
| T6.10 | Parent-side part-location prediction | ~200 | T6.9, compositional learning | Not started |
| T6.11 | Prediction-seeded hypothesis initialisation | ~100 | T6.9, T6.10 | Not started |
| T6.12 | Bottom-up prediction-error signal | ~80 | T6.9 | Not started |

**T6.9 implementation:**
Add `receive_prediction(prediction_dict)` to EvidenceGraphLM. Stores
prediction as `_pending_prediction`. During next hypothesis init, if
prediction exists, create hypotheses from predicted (object, pose, scale)
tuples + 10% exploration budget.

**T6.10 implementation (research):**
The parent LM stores child `object_id` as a feature at each node. To
predict which part should appear where, the parent must invert its model:
"given my pose hypothesis for the car, predict tire at world position X."

This requires the parent's `GridObjectModel` to store the child's
`object_id` → location mapping and the current parent MLH to transform
stored locations to sensor frame. The `upward_hypothesis_packet` already
carries top-k child IDs with support weights — the parent needs to learn
which child IDs appear at which locations in its reference frame.

**T6.11 implementation:**
Modify `_get_initial_hypothesis_space_single()`:
- Check `_pending_prediction`
- If present: create hypotheses for predicted objects at predicted poses
  only, plus random exploration budget (10% of normal K)
- If absent: fall back to existing full enumeration or proposal network

**T6.12 implementation:**
Add `get_prediction_error()` to EvidenceGraphLM. Returns minimal
confirmation when prediction was correct, full error signal otherwise.
Parent LM checks children's prediction errors before updating its
own hypotheses.

### Phase 4: Extending burst sampling (offspring, ranking, learned retrieval)

Burst sampling's `_sample_informed()` already does top-k feature-evidence
ranking. Phase 4 adds the capabilities it's missing: pose refinement,
ranking quality validation, and (if needed) learned retrieval.

| ID | Milestone | Lines | Dependencies | Status |
|---|---|---|---|---|
| T6.13 | Validate burst sampling ranking quality | ~50 (logging) | None | **Done** (telemetry + logging) |
| T6.13a | Offspring/refinement hypotheses | ~100 | None | **Done** |
| T6.14 | Learned retrieval encoder (only if T6.13 fails) | ~200 | T6.13 fail | Not started |
| T6.15 | Integration with hypothesis pipeline | ~100 | T6.13 or T6.14 | Not started |
| T6.16 | Self-supervised online training | ~50 | T6.14 | Not started |

**T6.13 (validate existing ranking, no new code):**
Burst sampling already picks top-k nodes by feature evidence. Measure
whether the correct node is in the top-200 candidates at burst steps.
Log `(step, burst_step, correct_node_rank, correct_node_in_top_k)`.
Recall@200 ≥ 90% means the analytical ranking works. If it does,
T6.14 is unnecessary.

**T6.13a (offspring/refinement hypotheses):**
Add a second sampling mode to `BurstSamplingHypothesesUpdater`:
when object ID is confident (few graph IDs remaining, high evidence)
but pose might be imprecise, sample new hypotheses near the best
existing ones:
- Same graph_id as the high-evidence hypothesis
- Location: jittered ±ε around the hypothesis's current location
- Orientation: small rotation perturbations (e.g., ±5° per axis)

This is complementary to informed sampling (which starts from scratch
per observation). Offspring refines; informed discovers.

Trigger: evidence slope is high (matching well) but pose error metric
or angular distance between top hypotheses suggests orientation
uncertainty. Or: explicit "refine" mode when an LM reaches a state
where it knows the object but not the precise pose.

Implementation: Add `_sample_offspring()` method alongside
`_sample_informed()` in `update_hypotheses()`. Called when
`sampling_burst_steps == 0` (not bursting) and a refinement condition
is met.

Tests:
- `test_offspring_reduces_pose_error_on_known_object`
- `test_offspring_does_not_change_object_id`
- `test_offspring_not_triggered_during_burst`

**T6.14 (only if T6.13 shows low recall):**
Two small MLPs: observation encoder + node encoder → shared embedding
space. Contrastive loss: positive = correct node, negatives = all others.
Cache node embeddings per graph, recompute when graph memory changes.
This improves burst sampling's ranking, not replaces it.

**T6.15:** If T6.14 is implemented, wire learned retrieval into
`_sample_informed()`. When `use_learned_retrieval=True`, use MLP
scores instead of feature evidence for top-k selection. Add 10%
exploration budget. Everything downstream unchanged.

**T6.16:** Self-supervised: after each episode, store (step-0 observation,
terminal MLH) as training pair. Every N episodes, run gradient steps.

## Validation Gates

| Gate | Metric | Pass | Fail |
|---|---|---|---|
| T6.0 signal quality | TM surprise AUC (novel vs repeated) | > 0.7 | < 0.6 |
| T6.1 learning speed | Episodes to 80% prediction accuracy | ≥10% fewer | No improvement |
| T6.2 saturation | W.max() after 1000 episodes | Plateaus | Still growing |
| T6.4 accuracy | Extended YCB recognition | Within 2pp | > 2pp drop |
| T6.4 convergence | Steps to correct MLH | ≥5% fewer | Slower |
| T6.5 vote parity | Accuracy under unified voting | ≥ conditional baseline | Regression |
| T6.6 throughput | Steps/second with gated output | ≥20% faster | No improvement |
| T6.6a context | Accuracy on kitchen-context eval | ≥ baseline with faster runtime | Accuracy drop |
| T6.11 accuracy | Recognition with predicted hypotheses | Within 2pp | > 2pp drop |
| T6.11 speed | Wall-clock per episode | ≥3× faster | < 2× |
| T6.13 recall | Recall@200 of correct node in burst sampling | ≥ 90% | < 80% |
| T6.13a pose | Rotation error on confident-object episodes | ≥10% lower | No improvement |

## Risks

- **Phase 3 depends on compositional learning.** Phase 1 (composition) failed
  its gate (39.88% vs 50.60% monolithic). Top-down prediction requires the
  parent to know what parts compose its object. If compositional models remain
  broken, T6.10–T6.12 cannot proceed. Mitigation: Phases 1–2 and Phase 4
  deliver value independently of Phase 3.

- **Surprise noise.** Sensory surprise may be dominated by sensor noise
  (slightly different curvatures at the same location). Mitigation: use
  smoothed surprise (mean of last 5 steps via `get_temporal_surprise()`).

- **Decay rate sensitivity.** Too much decay forgets; too little doesn't
  prevent saturation. Mitigation: sweep in [0.001, 0.01, 0.05] on a
  controlled sequence before full eval.

- **Double modulation.** Surprise modulating both learning AND evidence
  could amplify errors. Mitigation: test independently before combining
  (T6.1–T6.2 without evidence modulation; T6.4 combines both).

- **Proposal false negatives.** Burst sampling's top-k ranking may miss
  correct nodes. Mitigation: exploration budget + T6.13 validates recall.

- **Offspring hypothesis explosion.** Refinement sampling could create too
  many hypotheses if the trigger is too sensitive. Mitigation: cap offspring
  count per step; only trigger when evidence slope is high (matching well)
  and pose uncertainty is measurably high.

- **T5 compatibility.** State-conditioned models multiply sub-graphs.
  Include state_id in node encoder input and part-location predictions.

## Priority Order

Given the dependency structure, burst sampling as the working foundation,
and independent value of each phase:

```
Phase 1 (T6.0–T6.4):  Error-modulated learning (TM surprise → LR)
    ↓ makes each column a better predictor
Phase 2 (T6.5–T6.8):  Unified voting + context biasing + infrastructure
    ↓ enables surprise-gated inter-LM communication
    T6.6a: context biasing (independent, quick, no hierarchy needed)
Phase 4 (T6.13–T6.16): Burst sampling extensions (offspring, ranking)
    │ independent of Phases 1–3; extends working code
Phase 3 (T6.9–T6.12): Top-down prediction (blocked by compositional)
```

Phases 1, 2, and 4 are independent and can be parallelised. Phase 3 depends
on Phase 1 (each LM must be a good predictor) and on compositional models
working.

**Recommended attack order within phases:**
1. T6.6a (context biasing, ~30 lines, immediate value, no dependencies)
2. T6.0 (validate TM surprise signal quality — gates all of Phase 1)
3. T6.13 (validate burst sampling ranking — determines if T6.14 is needed)
4. T6.13a (offspring hypotheses, ~100 lines, addresses a real gap)
5. T6.7 (KDTree workers=-1, one-line speedup)
6. T6.1–T6.4 (error-modulated learning, gated by T6.0)
7. T6.5–T6.6 (unified voting, gated by Phase 1 results)
8. T6.8 (GIL-free updates, engineering)
9. T6.9–T6.12 (top-down, when compositional models work)

The quickest wins: T6.6a (~30 lines), T6.7 (one line), T6.13 (logging only).
The biggest architectural win: T6.9–T6.12 (top-down prediction).
The biggest gap in current capabilities: T6.13a (offspring refinement).
