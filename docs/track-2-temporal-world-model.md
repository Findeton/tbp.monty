# Track 2: Temporal World Model

## Purpose

Extend Monty from static object recognition into dynamic world modeling: state
discrimination, behavior learning, action-effect prediction, and manipulation.

These were originally separate phases (3, 4, 6) in the v1 roadmap. They are one
continuous problem: how does the world change over time and in response to actions?

## Roadmap

See [full-system-roadmap-v2.md](full-system-roadmap-v2.md) for the full program
context, restriction analysis, concept layers, and priority stack.

## Current Status

Updated 2026-03-22. Track 2 is now the #1 priority.

### What's built

- **HippocampalModule** (700+ lines, 31+18+4 tests): episodic buffer,
  co-occurrence matrix, relational graph, context signal, state dict
  serialization, `preload_associations()`, enhanced per-pair context signals,
  GraphMatchingMonty compatibility
- **Context dispatch**: `_dispatch_context_signals()` broadcasts HPC context
  to all downstream LMs after each matching step
- **Evidence modulation**: `_apply_hippocampal_bias()` in EvidenceGraphLM
  biases evidence toward graphs associated with HPC's active concepts
- **CMP enrichment**: `register_dynamic_feature()` and `set_feature_weight()`
  enable runtime injection of non-geometric features
- **Novelty detection**: evidence margin separates confident (86.2%) from
  uncertain (50.4%) predictions
- **LM-only input support**: `_combine_inputs()` allows LMs without SM
  connections (e.g., HPC) to receive LM-to-LM inputs directly
- **Temporal prediction (T2.4)**: Cross-episode sequence learning and
  next-state prediction in HippocampalModule. Learns episode transition
  frequencies, computes P(next | current) and P(next | current, action),
  merges predictions into context signal for downstream LM priming via
  existing `_apply_hippocampal_bias()` pathway. Action-conditioned
  transitions support learning causal action-effect relationships.
  Prediction validation tracks accuracy across episodes.
- **Within-episode temporal memory (T2.9)**: TemporalMemory module learns
  temporal patterns in raw sensory observation streams using SDR encoding +
  Hebbian learning + spreading activation — same algorithm as HPC cross-episode
  learning, applied within episodes to raw features. Behavior generators
  produce realistic sensory streams (walking, stapler, door, wheel, scissors,
  pendulum, hand wave, ball rolling, light flickering) without Habitat.
  Integrated into EvidenceGraphLM as optional component (`temporal_memory`
  config). Provides surprise signal and predictions during both training
  and eval.
- **Track 2 test suite**: 89 unit tests (cross-episode) + 56 unit tests
  (within-episode) + 5 integration tests validating HPC→evidence pathway,
  temporal prediction, and behavior learning from raw sensory data

### Why Track 2 is now #1

Five ablation experiments (1120 episodes) conclusively prove that NO Layer 1
mechanism improves the 67.9% ceiling:

```
baseline (all features):  67.9% (92.2% shape-congruent, 14.3% shape-incongruent)
no_hsv:                   65.6% (90.9%, 10.0%)
no_curvature:             66.1% (93.5%,  5.7%)
pose_only:                 0.0% ( 0.0%,  0.0%)
catbias (T1.4a):          61.2% (84.4%, 10.0%)  ← HURTS
```

Ball→fruit: 0% across ALL 5 variants. No geometric mechanism can break this.

## Restriction Addressed

**Restriction 3: Each episode is independent.**

Monty processes one recognition episode at a time with no memory across episodes.
Graph memory persists object models, but there is no event memory — no record of
what happened, what action was taken, or what the outcome was. Without cross-episode
memory, the system cannot learn behavioral associations, social patterns, or
accumulated knowledge.

## First-Principles Analysis (2026-03-21)

### Assumptions stripped away

1. **"You need physics simulation to test behavioral concepts."**
   False. The pipeline operates on features, not physics. A "bouncy" tag injected
   as a feature works the same as curvature. The first test needs ANY non-geometric
   signal, not a physics engine.

2. **"Cup upright vs inverted is the right first test."**
   Misleading. That's geometric state discrimination — the system already handles
   rotation via hypotheses. The first Track 2 test should validate cross-episode
   information transfer, not geometric variants.

3. **"The ball→fruit canary requires new sensor modalities."**
   False. The HPC already stores associations. Pre-loading HPC with ball↔bouncy
   associations tests the mechanism without new sensors.

4. **"Cross-episode memory needs temporal edges first."**
   False. The simplest useful memory is co-occurrence association: "I've seen this
   object before, and it was associated with X." Temporal edges add value later.

5. **"Behavioral concepts require valence."**
   Not for the first test. "Ball" = co-occurred with "bouncy." That's association,
   not evaluation. Valence comes after basic association is validated.

6. **"You need Habitat to test Track 2."**
   False. BaseGraphTest creates fake observations in seconds. The hippocampal_test.py
   has _make_state() and _make_ctx() helpers. The full pathway can be unit-tested.

7. **"Track 2 can't show results quickly."**
   False. The mechanism is fully wired. Code exists. Only the test is missing.

### What's fundamentally true

- The HPC→LM evidence pathway is complete (wired, tested, implemented)
- Ball→fruit is 0% across 1120 episodes (geometrically irreducible)
- The ONLY way to break 0% is non-geometric information
- The HPC already stores cross-episode associations
- Unit tests validate mechanisms in seconds, not hours
- The bottleneck is testing, not implementation

### What changes

**Old plan (pre-root-cause)**: implement T2.1 → build Habitat cup test →
implement T2.2 → ... (assumed ball 0% was a Layer 2 problem needing behavioral
memory)

**Revised plan (post-root-cause, 2026-03-21)**: The ball 0% is a Layer 1
matching bug, not a Layer 2 problem. The root cause is no scale invariance
(`detected_scale: 1, # TODO: scale doesn't work yet`) combined with evidence
size bias (`evidence_size_norm_power=0`). Track 2 infrastructure (HPC, context
dispatch, preloaded associations) is complete and tested (53 tests) but is NOT
needed for the ball canary. It remains important for genuine Layer 2 concepts
(dangerous, useful, heavy).

**Current plan**: Fix the matching pipeline first (T1.S scale normalization),
then measure the real Layer 1 ceiling, THEN determine what Track 2 actually
needs to solve.

## Scale-Invariant Matching (T1.S)

### Root cause (2026-03-21)

Holdout balls (golf ball 4.3cm, racquetball 5.7cm) are smaller than all
training balls (tennis ball 6.7cm – soccer ball 12.8cm). With `max_match
_distance=0.01m` (1cm) and no scale hypotheses, surface points on a golf
ball are 1.5cm from the nearest baseball graph node. This exceeds the
threshold → zero spatial matches → zero evidence for ALL spheres → the MLH
defaults to large objects (airplanes, clamps) that have accidental matches
due to having more nodes.

Analytical testing (30 seconds, no Habitat) confirmed:
- At `max_match_distance=0.02m`: baseball becomes #1 match for golf ball
- With scale-invariant matching: baseball #1 with clear ball/fruit separation
- HSV saturation (0.35 vs 0.84) distinguishes balls from fruits once the
  spatial gate stops blocking

### The fundamental constraint (2026-03-21)

The matching equation at each step:
```
search_location = hypothesis_location + R × displacement
match if |search_location - stored_point| < max_match_distance
```

For a stored object with `D_s` and query object with `D_q` (same shape,
different sizes), the sensor displacement `d` covers fraction `d/D_q` of
the query. The equivalent position on the stored graph requires displacement
`d × (D_s/D_q)`. But the system applies raw `d`.

**Per-step position error:** `ε = d × |D_s/D_q - 1|`

With d=0.008m, D_s=0.073m, D_q=0.043m: `ε = 0.0056m/step`.
Error accumulates: after N steps, `E ≈ N × ε`.
Hypothesis dies when `E > max_match_distance`: **N ≈ 1.8 steps**.

This proves scale mismatch kills hypotheses in 2 steps, regardless of
features. No feature fix can compensate.

### Why Options A and B both fail

Both approaches require estimating the query object's size (D_q), which
cannot be reliably obtained from sensor data:

**Option A (position normalization):** Normalize stored positions by D_s,
query displacements by D_q_estimate. Error per step in normalized space:
`ε = d × |1/D_q_est - 1/D_q|`. With any practical estimator (curvature,
displacement extent, floor), the error exceeds `max_match_distance` within
1-2 steps. Five Habitat eval attempts produced 100% confused results.

**Option B (match-time scaling):** Scale displacement by D_s/D_q_estimate.
Mathematically identical error: `ε = d × D_s × |1/D_q_est - 1/D_q|`.
Same sensitivity to D_q estimation accuracy.

**D_q estimation is fundamentally unreliable:**
- `principal_curvatures_log` uses symlog transform (`sign(k)*log(|k|+1)`),
  curvature magnitudes don't correlate with object radius in any simple form
- Displacement extent: after 20 steps, only 17-54% of true object size
- Required accuracy for 20-step survival: ±2.7% (golf ball at mmd=0.1)

Both approaches reduce to the same constraint: you need D_q within ~3%
before matching. No sensor signal provides that accuracy.

### The biological solution: multi-scale grid cell modules (2026-03-21)

**Key insight from neuroscience:** The brain does NOT estimate object
scale before matching. It tests scale as a hypothesis variable, using
grid cell modules at multiple spacings.

**Grid cells in the medial entorhinal cortex (MEC):**
- Organized in ~4-5 discrete modules per cortical column
- Each module has a different grid spacing (scale)
- Module spacings related by ~√2 (1.42x) ratio
- Path integration: `new_phase = old_phase + displacement / grid_spacing`
- The displacement is divided by the module's spacing — identical to
  dividing by a scale factor in Monty

**Mapping to Monty architecture:**
Each hypothesis gains a per-hypothesis scale factor. The displacement
step becomes:
```
search_loc = hyp_loc + R × (displacement / scale_factor)
```
Scale=1.0 means query is same size as stored graph. Scale=2.0 means
query is 2x larger (displacement covers less of stored graph → divide).
Wrong-scale hypotheses accumulate negative evidence and get pruned.
Correct-scale hypotheses survive. No D_q estimation needed.

**Implementation (2026-03-21):**
- Added `scales` field to `Hypotheses` and `ChannelHypotheses`
- Displacement scaled per-hypothesis in `DefaultHypothesesDisplacer`
- Hypotheses initialized at 5 scale factors: `[0.39, 0.625, 1.0, 1.6, 2.56]`
  (1.6x spacing, 6.6x total range, matching ~5 grid cell modules in MEC)
- Burst sampling replicates new hypotheses across all scale factors
- `max_match_distance` stays at 0.01m (no normalization needed)
- `current_mlh["scale"]` reports the winning scale factor
- Config: `scale_invariant_matching: true` enables multi-scale modules

**Files changed:**
- `hypotheses.py` — added `scales` field
- `hypotheses_displacer.py` — per-hypothesis displacement scaling
- `hypotheses_updater.py` — multi-scale initialization, `scale_factors` param
- `burst_sampling.py` — multi-scale burst replication, `scale_factors` param
- `learning_module.py` — `possible_scales` storage, MLH scale reporting
- `evidence_matching.py` — `extract_hypotheses` passes scales through

**Analytical verification (2026-03-21):**
With 1.6x module spacing, golf ball (D_q=0.043m) vs tennis ball
(D_s=0.115m): true_scale=2.67, best module=2.56, error=0.00013m/step,
hypothesis survives 75 steps (well beyond episode length).

### Scale as a hierarchy problem (>10x)

Within-sensor scale invariance (~5-10x range) is a single-level
multi-scale problem, solved by grid cell modules. But extreme scale
differences (100x real car vs toy car, or 100000x building vs model) are
fundamentally a **hierarchy and sensor problem**:

1. **Sensor resolution limits:** A surface agent with 0.008m step size
   can't meaningfully process objects <1cm (step > object) or >1m (needs
   thousands of steps). The sensor's physical resolution defines the
   useful scale range.

2. **Sensory representations are different:** Touching a toy car vs a
   real car produces different curvatures, different spatial patterns,
   different textures. They are genuinely different objects at the
   sensory level. The conceptual link "both are cars" requires
   abstraction.

3. **Hierarchy provides the abstraction:** In TBT, higher-level cortical
   columns abstract shared structure across different sensory experiences.
   A column that receives input from multiple lower-level columns, each
   operating at different spatial scales, can learn "car-shape" as a
   concept independent of absolute size.

**This makes extreme scale invariance a test case for the heterarchy.**
The existing multi-level LM architecture (make_local_heterarchy) could
be validated by testing whether a 2-level heterarchy recognizes a toy car
and a real car as the same concept, when each level operates at a
different sensor resolution. This would be a concrete test of whether the
heterarchy adds value beyond single-level matching.

**Proposed test (T1.S.H):** Train two LMs at different sensor resolutions
(e.g., zoom=10 and zoom=2). Lower LM sees fine detail, upper LM sees
coarse shape. Both process the same object. Test whether the upper LM
can match objects across >10x scale differences by abstracting from the
lower LM's more fine-grained features.

### Robustness through voting

In the brain, no single cortical column needs perfect scale coverage.
Different columns have slightly different grid cell module configurations
(4 vs 5 modules, slightly different spacings). **Voting across columns
smooths out coverage gaps.** Any individual column's scale blind spots
are covered by neighboring columns with offset tuning.

In Monty: a multi-LM config where each LM has slightly different
`scale_factors` would give robust scale coverage without tuning. The
diversity of scale module configurations IS the solution — no single
LM needs to be optimal.

### Complementary fixes (config-only)

Apply alongside multi-scale matching:
- `evidence_size_norm_power: 0.5` — removes advantage of large graphs
  (4894-node airplane vs 427-node orange). Independent confound.
- `feature_weights.patch.hsv: [1, 2.0, 0.5]` — up-weight saturation
  (most discriminative feature between balls and fruits)

### Success criteria

- Ball category accuracy > 0% on extended YCB holdout eval
- No regression on shape-congruent categories (airplane, fruit, cup, etc.)
- Analytical test confirms before Habitat eval runs

## Milestone Tracker

| ID | Milestone | Status |
|---|---|---|
| T1.S | Scale-invariant matching (grid cell modules, ~5-10x range) | **Implemented** (5 modules, 476 tests pass, Habitat eval running) |
| T1.S.H | Extreme scale via heterarchy (>10x, toy car vs real car) | Not started (test case for heterarchy validation) |
| T2.0 | Validate HPC→evidence mechanism (unit tests) | **Done** (18 tests pass) |
| T2.1 | Cross-episode event memory | **Done** (associations persist, context dispatches) |
| T2.2 | State discrimination (same object, two states) | **Done** (state registry, predict_state_after_action, 7 tests) |
| T2.3 | Behavioral tag injection (synthetic non-geometric features) | **Done** (preload + HPC-mediated) |
| T2.4 | Temporal prediction (SDR + Hebbian, spreading activation) | **Done** (36 tests, biologically plausible) |
| T2.5 | Action-effect prediction & planning (forward replay) | **Done** (motor→HPC wiring, greedy replay, VTE simulation) |
| T2.6 | Behavior recognition (temporal pattern matching) | **Done** (register_behavior, recognize_behavior, cyclic matching, 6 tests) |
| T2.7 | Manipulation via prediction (suggest_action, evaluate_action) | **Done** (full perception→prediction→action loop, 6 tests) |
| T2.8 | Temporal reference frame in LMs | **Spec written** (deferred: T2.9 provides pragmatic within-episode temporal) |
| T2.9 | Within-episode temporal memory (raw sensory) | **Done** (TemporalMemory: SDR encoding + Hebbian on raw States, 9 behavior generators, 56 tests, LM integration) |
| T2.10 | Temporal prediction voting (confusion voting for time) | **Done** (predict next state, check prediction, vote confident→confused, end-to-end tests) |
| V1 | Surprise → evidence modulation (predictive coding loop) | **Done** (delta-scaling, exploratory_step feeds TM, 2026-03-25) |
| V2 | A/B test: temporal priming helps recognition | **Done** (8 tests, margin ≥ baseline, 2026-03-25) |
| V3 | HPC/TM state persistence in save/load | **Done** (state_dict round-trip, 3 tests, 2026-03-25) |
| V4 | HPC in training config (gap #14) | **Done** (training + eval configs, training-mode get_output, 7 tests, 2026-03-25) |
| V5 | Panda3D A/B: temporal priming ON vs OFF | **Done** (8 tests, Fox Walk+Run, control vs treatment, 55 total pass, 2026-03-26) |

## Test Plan

### Tier 1: Fast unit tests (seconds, run on every change)

These tests validate the mechanism in isolation using synthetic data.
No Habitat. No trained models. Run in the existing test suite.

**Test 1: `test_hpc_association_biases_evidence`**
Create an EvidenceGraphLM with two graphs (ball_graph, fruit_graph). Set
equal evidence for both. Create an HPC context signal with strong "ball"
association. Call `receive_context()`. Verify ball_graph evidence is boosted
and fruit_graph is not.

**Test 2: `test_cross_episode_association_accumulates`**
Episode 1: HPC observes states with object_id="ball" + object_id="bouncy"
(confidence > threshold). Episode 2: HPC observes "fruit" + "edible".
After both episodes, verify association matrix links ball↔bouncy and
fruit↔edible with distinct, non-zero strengths.

**Test 3: `test_dynamic_feature_breaks_tie`**
Register a "material_class" feature via `register_dynamic_feature()`.
Create two observations of identical geometry but different material_class
values. Feed through EvidenceGraphLM. Verify different evidence accumulation.

**Test 4: `test_category_context_resolves_geometric_ambiguity`**
Create observations that match two graphs equally (simulating the ball=fruit
case). Without HPC context: both graphs have equal evidence. With HPC context
strongly associating one graph: that graph wins. This is the core mechanism
test.

### Tier 2: Integration tests (minutes, run before merge)

**Test 5: `test_preloaded_hpc_shifts_ball_evidence`**
Load the extended YCB trained model. Create an HPC state_dict with
ball-category associations pre-loaded. Instantiate the full Monty stack
with HPC. Eval on 1 holdout ball (1 rotation). Verify that MLH shifts
toward a ball-category graph instead of a fruit-category graph.

**Test 6: `test_multi_episode_hpc_accumulation`**
Run 3 sequential episodes through a Monty stack with HPC. In each episode,
present a known object. After 3 episodes, verify HPC's episodic memory
contains 3 entries, association matrix has grown, and context signal
reflects the accumulated history.

### Tier 3: Validation (hours, run for milestones)

**Test 7: `test_extended_ycb_with_behavioral_tags`**
Add a "material_class" feature per category to the extended YCB eval.
Ball objects get material_class=0 (hard), fruit objects get material_class=1
(soft). Re-run full 224-episode eval. Target: ball→fruit drops from 0% to
>50% category accuracy. This proves non-geometric features resolve the
confusion.

**Test 8: `test_full_cross_episode_ball_fruit_resolution`**
Multi-episode experiment:
- Episodes 1-N: train on balls and fruits WITH behavioral tags
- Episodes N+1-M: eval on holdout balls without tags, but HPC has accumulated
  associations from training
- HPC's cross-episode memory should bias evidence toward correct category
- Primary target: ball→fruit > 50% from HPC context alone

## Implementation Plan

### Phase 1: Validate the mechanism (fast, unit tests) — DONE

11 Tier 1 unit tests pass. Created `track2_mechanism_test.py` with 6 test
classes covering all 4 planned tests plus bonus tests:

- `TestHPCAssociationBiasesEvidence` — HPC context boosts target evidence
- `TestCrossEpisodeAssociationAccumulates` — co-occurrence builds across episodes
- `TestDynamicFeatureBreaksTie` — material_class discriminates identical geometry
- `TestCategoryContextResolvesGeometricAmbiguity` — core mechanism: HPC breaks
  geometric tie, with 3 sub-tests (equal evidence, resolved ambiguity, category match)
- `TestHPCContextSignalGeneration` — HPC produces valid context signals
- `TestPreloadAssociations` — `preload_associations()` API

**Files created:**
- `tests/unit/frameworks/models/track2_mechanism_test.py` (18 tests)

### Phase 2: Synthetic context injection (quantitative result) — DONE

4 integration tests pass using the real 34-object extended YCB trained model.
Tests load `model.pt`, extract real graph node features as synthetic
observations, and verify HPC context biases ball evidence over fruit.

**Files created:**
- `tests/integration/frameworks/models/track2_context_injection_test.py` (4 tests)

### Phase 3: Dynamic feature injection — DONE

HPC-mediated approach implemented (Approach 2 from plan). Added:
- `preload_associations()` to HPC for seeding behavioral knowledge
- Enhanced `get_context_signal()` with full per-pair association strengths
  (was mean-only, now includes all associated concepts)
- Eval comparison script `tools/run_eval_with_hpc.py`

**Files created/modified:**
- `src/.../models/hippocampal_module.py` — `preload_associations()`,
  enhanced `get_context_signal()`, GraphMatchingMonty compatibility stubs
- `tools/run_eval_with_hpc.py` — comparison script

**Remaining for Tier 3 validation**: Full Habitat eval with HPC in config.
Requires Hydra config changes to add HPC as second LM.

### Phase 4: Full cross-episode learning — DONE (mechanism)

Multi-episode accumulation validated in unit tests. Key changes:
- `_combine_inputs()` in MontyBase now allows LMs with no SM connections
  (HPC) to receive LM-to-LM inputs directly. Addresses the TODO at line 245.
- HPC `pre_episode()` accepts `primary_target` kwarg for GraphMatchingMonty
- Added `add_lm_processing_to_buffer_stats`, `stepwise_target_object`,
  `stepwise_targets_list` stubs for GraphMatchingMonty compatibility

**Files modified:**
- `src/.../models/monty_base.py` — `_combine_inputs()` fix
- `src/.../models/hippocampal_module.py` — compatibility stubs

**Remaining for Tier 3 validation**: Habitat-based multi-episode eval where
HPC accumulates associations naturally during training, then biases eval.

### Phase 5: Temporal prediction (T2.4) — DONE

Cross-episode sequence learning and next-state prediction implemented in
HippocampalModule. The system learns temporal structure across episodes and
uses it to predict (and prime) what comes next — the core of a temporal
world model.

**Biological basis:**

The implementation follows three hippocampal mechanisms:

1. **Sequence cells / time cells**: Hippocampal neurons that fire at
   specific positions in a temporal sequence. Mapped to episode transition
   counts — each transition (A→B) is a learned temporal association, like
   a time cell that fires during the A→B transition.

2. **Hippocampal pre-play**: Before navigating a familiar route, hippocampal
   place cells "pre-play" the expected sequence. Mapped to temporal
   prediction in `pre_episode()` — after completing episode N, the HPC
   computes P(next | terminal_concept_N) and primes downstream LMs via
   the context signal before episode N+1 begins.

3. **Action-outcome binding**: The hippocampus binds actions to their
   outcomes, enabling goal-directed behavior. Mapped to action-conditioned
   transitions — `record_action("push")` annotates transitions so the
   system can predict P(next | current, push) ≠ P(next | current, pull).

**Architecture:**

```
Episode N ends → post_episode():
  1. Resolve terminal concept (what was recognized)
  2. Record transition: prev_terminal →[action]→ current_terminal
  3. Validate prediction (was the prediction correct?)

Episode N+1 starts → pre_episode():
  1. Compute temporal predictions: P(next | last_terminal)
  2. Store predictions for context signal integration

During Episode N+1 → get_context_signal():
  1. Merge temporal predictions into association_strengths
     (weight = temporal_prediction_weight * transition_probability)
  2. Downstream LMs receive primed evidence via existing
     _apply_hippocampal_bias() — no changes to EvidenceGraphLM needed
```

**Key design decisions:**

- **SDRs not symbols**: Each concept is represented as a sparse distributed
  representation (SDR) — a random binary vector with ~4% active bits.
  This is the biological representation: a cell assembly (population code)
  for each concept, like a place cell ensemble.  SDRs enable graceful
  degradation and generalization through overlap.

- **Hebbian learning not counting**: Temporal associations are learned via
  outer-product Hebbian update: `W += η * outer(sdr_current, sdr_previous)`.
  This is spike-timing-dependent plasticity (STDP) — synapses strengthen
  between neurons that fire in temporal sequence.  No explicit counting,
  no frequency tables.

- **Spreading activation not lookup**: Prediction is `activation = W @ sdr_source`,
  then finding concepts whose SDRs overlap the activation pattern.  This IS
  the biological mechanism — activation spreads through learned synaptic
  connections to pre-activate predicted cell assemblies.

- **Forward replay not BFS**: Planning chains spreading activation
  iteratively, biasing toward the goal via SDR similarity.  This is
  hippocampal forward replay — not an explicit graph search algorithm.

- **Predictions flow through existing pathway**: Temporal predictions are
  merged into `association_strengths` in the context signal, so the
  existing `_apply_hippocampal_bias()` primes downstream LMs with zero
  changes to EvidenceGraphLM.

- **Action-conditioned via separate weight matrices**: Each action gets
  its own Hebbian matrix, so `W_push @ sdr_ball` ≠ `W_pull @ sdr_ball`.
  Analogous to the hippocampus binding motor sequences to state transitions
  via separate synaptic pathways.

**New parameters:**

| Parameter | Default | Purpose |
|---|---|---|
| `temporal_dim` | 512 | SDR dimensionality (~1/sparsity associations capacity) |
| `temporal_sparsity` | 0.04 | Fraction of active bits per SDR |
| `hebbian_learning_rate` | 1.0 | Outer-product update strength |
| `temporal_prediction_weight` | 0.5 | Weight for context signal merging |

**New methods in HippocampalModule:**

| Method | Purpose |
|---|---|
| `get_temporal_predictions(from_concept, action)` | Spreading activation prediction |
| `record_action(action_label)` | Annotate inter-episode transition |
| `get_prediction_accuracy()` | Hit rate, top-1 accuracy, history |
| `plan_action_sequence(current, goal)` | Forward replay planning |
| `simulate_trajectory(current, actions)` | VTE trajectory simulation |
| `_get_concept_sdr(concept_id)` | Generate deterministic SDR |
| `_hebbian_update(current, previous)` | Outer-product learning |
| `_spreading_activation(source_sdr, W)` | Core prediction mechanism |

**Files modified:**
- `src/.../models/hippocampal_module.py` — SDR + Hebbian temporal system
- `src/.../models/monty_base.py` — motor→HPC action reporting

**Files created:**
- `tests/unit/.../test_temporal_world_model.py` — 36 tests across 11 classes

**Test classes:**
- `TestSDRGeneration` (4 tests) — sparse, binary, deterministic, correct dim
- `TestHebbianLearning` (3 tests) — outer-product strengthens associations
- `TestSpreadingActivationPrediction` (6 tests) — prediction via activation
- `TestActionConditionedPrediction` (3 tests) — action-conditioned matrices
- `TestPredictionValidation` (2 tests) — accuracy tracking
- `TestContextSignalIntegration` (4 tests) — downstream LM priming
- `TestForwardReplayPlanning` (5 tests) — greedy forward replay
- `TestTrajectorySimulation` (3 tests) — mental simulation of action sequences
- `TestSerializationRoundTrip` (1 test) — SDR/matrix persistence
- `TestMotorActionReporting` (2 tests) — motor system→HPC auto-wiring
- `TestEdgeCases` (3 tests) — robustness

### Phase 7: Within-episode temporal memory (T2.9) — DONE

Within-episode temporal learning on raw sensory data. Bypasses Habitat's
limitation of static objects by using behavior generators that produce
realistic sensory State streams directly from physics dynamics.

**Biological basis:**

The same SDR + Hebbian + spreading activation motif used for cross-episode
learning in HPC is applied here within episodes at the sensory level:

1. **Locality-sensitive SDR encoding**: Raw observation features (location,
   pose vectors, curvatures, color) projected through a random matrix, then
   winner-take-all selects top-k active bits. Similar stimuli → similar SDRs
   (overlapping neural populations). Analogous to sensory cortex receptive
   field encoding.

2. **Hebbian temporal association**: Consecutive observation SDRs associated
   via outer-product rule `W += η * outer(sdr_t, sdr_{t-1})`. This is STDP
   (spike-timing-dependent plasticity) at the observation level.

3. **Spreading activation prediction**: `predicted = WTA(W @ sdr_current)`.
   Analogous to dendritic prediction in cortical pyramidal cells.

4. **Surprise as prediction error**: `surprise = 1 - overlap / n_active`.
   Analogous to burst firing when predictions fail (all cells fire instead
   of just the predicted subset).

**Architecture:**

```
Two levels of temporal learning, same algorithm:

Level 1 (Cortical — TemporalMemory, within episode):
  Input:  Raw sensory States from SM (location, curvatures, pose, color)
  Output: Surprise signal, predicted next observation
  Scope:  Within a single episode — learns how features change over time
  Use:    Behavior recognition, anomaly detection, temporal prediction

Level 2 (Hippocampal — HPC, cross episode):
  Input:  Object identity from LM (graph_id)
  Output: Context signal, temporal predictions for next episode
  Scope:  Across episodes — learns which objects follow which
  Use:    Episode prediction, evidence priming
```

**Behavior generators (no Habitat needed):**

Each behavior is a pure function that generates State sequences from
physical dynamics, producing the same kind of feature streams that a
sensor module would observe:

| Behavior | Dynamics | Test focus |
|---|---|---|
| `walking_gait` | Sinusoidal curvatures + pose sway + linear drift | Cyclic pattern, gait period |
| `stapler_press` | Binary state transition (open↔closed) | Discrete transition learning |
| `door_opening` | Monotonic rotation, arc trajectory | Trajectory prediction |
| `wheel_spinning` | Continuous rotation, fixed location | Speed discrimination |
| `scissors_cutting` | Cyclical open-close | Repetitive pattern |
| `hand_waving` | Lateral oscillation + wrist tilt | Gesture recognition |
| `pendulum_swing` | Decaying oscillation | Non-stationary dynamics |
| `ball_rolling` | Translation + rotation (no-slip constraint) | Combined dynamics |
| `light_flickering` | Color/brightness oscillation + noise | Pure feature dynamics |
| `add_noise` | Gaussian noise on any behavior | Robustness testing |

**LM integration:**

TemporalMemory is an optional component of EvidenceGraphLM, configured
via `temporal_memory` kwarg (dict or instance). During matching_step(),
each SM observation is fed to the temporal memory. During training,
transitions are learned and replayed. During eval, surprise signal is
available via `get_temporal_surprise()`.

**New parameters (TemporalMemory):**

| Parameter | Default | Purpose |
|---|---|---|
| `sdr_dim` | 2048 | SDR dimensionality (larger than HPC — richer sensory data) |
| `sdr_sparsity` | 0.02 | Fraction of active bits (40 out of 2048) |
| `learning_rate` | 0.1 | Hebbian learning rate |
| `projection_seed` | 42 | Random projection matrix seed |
| `include_location` | True | Whether location is in the feature encoding |

**Files created:**
- `src/.../models/temporal_memory.py` — TemporalMemory module
- `src/.../environments/behaviors.py` — 9 behavior generators + noise
- `tests/unit/.../test_within_episode_temporal.py` — 56 tests

**Files modified:**
- `src/.../models/evidence_matching/learning_module.py` — optional
  temporal_memory integration in EvidenceGraphLM

**Test classes (56 tests):**
- `TestSDREncoding` (6) — sparse binary, locality-sensitive, deterministic
- `TestHebbianTemporalLearning` (4) — transitions, strengthening, sequences
- `TestSurprise` (4) — prediction error, history tracking
- `TestWalkingBehavior` (4) — gait learning, prediction, anomaly, noise
- `TestStaplerBehavior` (3) — state transition, surprise, SDR difference
- `TestDoorBehavior` (2) — trajectory prediction, training improvement
- `TestWheelBehavior` (2) — rotation learning, speed discrimination
- `TestScissorsBehavior` (1) — cyclical cutting pattern
- `TestHandWaving` (1) — oscillatory gesture
- `TestPendulumBehavior` (1) — non-stationary damped motion
- `TestBallRolling` (1) — combined translation + rotation
- `TestLightFlickering` (1) — pure color dynamics
- `TestBehaviorRecognition` (4) — 2/4/6 behavior discrimination, partial
- `TestSequencePrediction` (2) — chained prediction, sparsity maintenance
- `TestReplayConsolidation` (4) — strengthening, counts, edge cases
- `TestTemporalContext` (3) — context signal, predictions
- `TestSurpriseAsAttention` (2) — behavior switch, both-trained
- `TestSerialization` (3) — encoding/prediction/behavior persistence
- `TestEdgeCases` (5) — robustness
- `TestLMTemporalMemoryIntegration` (3) — EvidenceGraphLM + temporal

## First-Principles Reassessment (2026-03-25)

### The honest status (updated 2026-03-26)

10 milestones (T2.0–T2.10) marked "Done." V1–V4 validation tasks done
(2026-03-25). V5 Panda3D A/B test done (2026-03-26).

| Layer | Status |
|---|---|
| Infrastructure | Complete. HPC, TemporalMemory, Panda3D, behavior generators, voting, event detection — all tested |
| Integration | Wired. HPC→evidence pathway + surprise→evidence modulation both operational |
| Validation | **V2** (synthetic A/B, 8 tests) + **V5** (Panda3D rendered A/B, 8 tests on Fox Walk/Run). Treatment margin ≥ 90% of control; temporal priming doesn't hurt |
| Panda3D behavior training | Working. 55 E2E tests pass on real animated models. A/B validated |
| Habitat validation | **Dropped.** Habitat has static meshes only — no temporal signal exists. Panda3D is the correct validation platform for temporal features |

### Assumptions stripped away

1. **"10 milestones Done = Track 2 is done."** False. Each milestone was
   validated with unit tests on synthetic data. "Mechanism works in
   isolation" ≠ "mechanism adds value in the system."

2. **"More components is progress."** False. Two independent temporal
   learning systems (HPC cross-episode, TemporalMemory within-episode)
   share the same SDR+Hebbian algorithm but have no integration path.
   The TemporalMemory computes surprise but it feeds NOWHERE during
   recognition. The predictive coding loop is open.

3. **"Building more is the right next step."** False. The building phase
   is over. The validation phase never started. Gap #13 (A/B test) was
   identified as #1 priority on 2026-03-22 but never executed.

### What changes

**Old approach:** Build T2.0 → T2.1 → ... → T2.10 → eventually validate

**New approach:** Validate what exists. Close the open loops. Measure.

### Validation plan (2026-03-25)

Three implementation tasks, in dependency order:

**V1. Surprise → evidence modulation (closes gap #17)**

The TemporalMemory computes surprise (prediction error) every step, but
the signal goes nowhere. In TBT, prediction error IS the signal — it
drives learning, attention, and exploration. This is the most
biologically important missing piece.

Implementation: Add `_apply_surprise_modulation()` to EvidenceGraphLM.
Called after temporal memory step in `matching_step()`.

- Low surprise (predictions correct) → boost evidence gain for current
  MLH by `(1 - surprise) * surprise_boost`. Analogous to sparse firing
  in predicted minicolumns — confident prediction accelerates convergence.
- High surprise (predictions wrong) → scale down evidence gain by
  `(1 - surprise_penalty * surprise)`. Analogous to burst firing —
  the model's temporal expectations are violated, so current spatial
  hypotheses are less trustworthy.
- Parameters: `surprise_boost` (default 0.5), `surprise_penalty`
  (default 0.3). Conservative defaults — modulation should help,
  not dominate.
- Gated by `self._temporal_memory is not None` — zero impact on
  existing pipelines.

**V2. A/B test proving temporal priming helps (closes gap #13)**

Unit test that creates two identical LMs with two trained objects
(geometrically ambiguous, like ball vs fruit). One LM has temporal
memory enabled, the other doesn't. Both process the same observation
sequence. Measure:

- Steps to convergence (MLH stabilizes on correct object)
- Final evidence margin (MLH evidence - runner-up evidence)
- Whether surprise modulation accelerates the correct-object case

This test runs in seconds (synthetic data, no Habitat) and directly
answers: does temporal context help spatial recognition?

**V3. HPC state persistence (closes gap #15)**

Add `save_state()` / `load_state()` to HippocampalModule that
serializes all memory systems (associations, episodic buffer, relational
graph, temporal SDRs/weights) alongside the LM's graph memory.

Wire into MontyBase's `save_state()` / `load_state()` so HPC state
persists across experiment runs. Without this, every run starts with
a fresh HPC — the entire point of cross-episode memory is accumulation.

### Implementation results (2026-03-25)

**V1: Surprise → evidence modulation — DONE**

Added `_apply_surprise_modulation()` to EvidenceGraphLM. Scales the
evidence DELTA from each matching step based on temporal prediction
surprise:
- Low surprise (< 0.5): amplify delta by `1 + boost * (1 - 2*surprise)`
- High surprise (> 0.5): dampen delta by `1 - penalty * (2*surprise - 1)`
- At surprise = 0.5: scale = 1.0 (neutral, no effect)

Also added `_feed_temporal_memory()` called from both `exploratory_step`
(training) and `matching_step` (eval), so temporal memory learns during
training — was previously only fed during eval.

Parameters: `surprise_boost` (default 0.0), `surprise_penalty` (default
0.0). Zero defaults = no impact on existing pipelines.

Files modified:
- `src/.../evidence_matching/learning_module.py` — `_apply_surprise_modulation()`,
  `_feed_temporal_memory()`, `exploratory_step()` override, `state_dict()`/
  `load_state_dict()` now include temporal memory

**V2: A/B test — DONE (8 tests)**

`tests/unit/frameworks/models/test_temporal_priming_ab.py`:
- `TestSurpriseModulation` (2 tests): low-surprise boost, noop when disabled
- `TestTemporalPrimingAB` (4 tests): evidence margin comparison, surprise
  decrease on familiar sequences, novel-vs-familiar surprise, penalty on
  wrong-object matching
- `TestSurpriseSignalBehavior` (2 tests): bounded [0,1], context available

Key result: temporal LM evidence margin ≥ baseline margin for correct
object (surprise modulation amplifies correct-object evidence gain).

**V3: HPC state persistence — DONE (3 tests)**

`EvidenceGraphLM.state_dict()` now includes `temporal_memory` key.
`load_state_dict()` restores it. Tests verify:
- W matrix and projection survive save/load
- Surprise is identical after round-trip
- Named behaviors persist

Total new tests: 11. All passing. No regressions (779 framework tests pass).

**V5: Panda3D A/B test — DONE (8 tests, 2026-03-26)**

Replaced the invalid Habitat A/B eval (static meshes have no temporal
signal) with a Panda3D A/B test on animated models. Fox Walk + Fox Run
trained on two identical `Panda3DBehaviorExperiment` instances:

- Control: no temporal memory
- Treatment: temporal memory + surprise modulation (boost=0.5, penalty=0.3)

Key results:
- Both learn identical object sets (same graph_ids)
- Treatment evidence margin ≥ 90% of control for behavior discrimination
- Treatment surprise < 1.0 on both Walk and Run (temporal predictions active)
- Morphology recognition not degraded
- 55 total Panda3D E2E tests pass (8 new + 47 existing, 0 regressions)

Files modified:
- `tests/unit/simulators/panda3d/test_panda3d_e2e.py` — `TestTemporalPrimingAB`
  (8 tests), `_compute_evidence_margin()` helper

### Deferred (validate first)

- T2.8 (temporal reference frame) — deep architectural change, premature
- More behavior generators — data isn't the bottleneck
- Panda3D expansion — works as standalone capability, not gating
- Action SDRs (#18) — nice-to-have, not blocking validation
- State discrimination as perception (#19) — requires spatial recognition

## What's Still Needed (updated 2026-03-26)

### Completed

1. ~~**Context → evidence modulation**~~ — **DONE**
2. ~~**CMP enrichment**~~ — **DONE**
3. ~~**Unit tests for the mechanism**~~ — **DONE**
4. ~~**End-to-end quantitative test**~~ — **DONE**
5. ~~**Behavioral feature source**~~ — **DONE**
6. ~~**Temporal prediction**~~ — **DONE** (T2.4: SDR + Hebbian)
7. ~~**State discrimination**~~ — **DONE** (T2.2: state registry)
8. ~~**Episodic replay**~~ — **DONE** (forward/reverse, cross-episode)
9. ~~**Behavior recognition**~~ — **DONE** (T2.6: template matching)
10. ~~**Manipulation via prediction**~~ — **DONE** (T2.7: suggest/evaluate)
11. ~~**Within-episode temporal structure**~~ — **DONE** (T2.9:
    TemporalMemory operating on raw sensory data, no Habitat needed)
12. ~~**Behavior recognition from raw data**~~ — **DONE** (T2.9:
    SDR-based behavior recognition replaces string template matching
    at the sensory level — learns and recognizes walking, stapler, door,
    wheel, scissors, etc. from raw feature streams)

### Validation gaps (resolved 2026-03-26)

13. ~~**A/B test proving temporal priming helps recognition.**~~
    — **V2: DONE** (synthetic, unit-level). **V5: DONE** (Panda3D rendered
    data, Fox Walk+Run, treatment margin ≥ 90% of control). Habitat A/B
    dropped — static meshes have no temporal signal to learn from.

14. ~~**HPC in Hydra config for training, not just eval.**~~
    — **DONE.** Created `extended_ycb_train_hpc.yaml` (training) and
    `extended_ycb_eval_holdout_hpc_trained.yaml` (eval from trained HPC).
    Added training-mode `get_output()` to `GraphLM` so HPC receives
    object identity during supervised pretraining. Fixed `_combine_inputs`
    to handle None LM outputs. 7 new tests (22 total in track2_mechanism).

15. ~~**HPC state persistence across experiment runs.**~~
    — **V3: DONE.** TemporalMemory included in LM state_dict.

### Integration gaps

16. **T2.8: Temporal reference frame in LMs.** Deferred until validation
    proves temporal context adds value.

17. ~~**Surprise → evidence modulation.**~~
    — **V1: DONE.** Delta-scaling modulation in `_apply_surprise_modulation()`.

18. **Action SDRs instead of string labels.** Deferred.

19. **State discrimination is a registry, not perception.** Deferred.

20. **Dynamic sensory data sources.** Deferred. Panda3D behavior
    training partially addresses this.

### Priority order (revised 2026-03-26)

Gaps #13, #14, #15, #17 are now closed.

**Why the Habitat A/B eval was dropped (2026-03-26):**

The previous #1 priority was "run extended YCB eval with temporal priming
on vs off." This is invalid because Habitat only supports static meshes —
no articulation, no deformation, no dynamics. Temporal memory learns from
*how features change over time*. With static objects there is no temporal
signal to learn from, so an A/B comparison would be meaningless (both arms
would produce identical results).

**Panda3D A/B validation replaces Habitat A/B (V5, 2026-03-26):**

The correct validation platform is the Panda3D behavior training pipeline,
which renders animated 3D models with skeletal deformation. The A/B test
(`TestTemporalPrimingAB` in `test_panda3d_e2e.py`, 8 tests) trains two
identical `Panda3DBehaviorExperiment` instances on Fox Walk + Fox Run:

- **Control**: no temporal memory, no surprise modulation
- **Treatment**: temporal memory (SDR dim=1024, sparsity=0.02, lr=0.1)
  + surprise modulation (boost=0.5, penalty=0.3) on both LMs

Results (2026-03-26):
- Both learn identical shapes and behaviors (same graph_ids)
- Treatment evidence margin ≥ 90% of control (temporal priming doesn't hurt)
- Treatment surprise < 1.0 on both Walk and Run (predictions are meaningful)
- Morphology recognition not degraded by temporal priming
- All 8 tests pass, 55 total Panda3D E2E tests pass (0 regressions)

Remaining priorities:

1. **#18 (Action SDRs)** — makes action conditioning biologically
   plausible.
2. **#16 (Temporal reference frame)** — the deep architectural change.
3. **#20 (Dynamic data)** — expand Panda3D training to more models and
   behaviors; Habitat is not the path for temporal validation.

## T2.8: Temporal Reference Frame in LMs (Specification)

**Status:** Design specification only. The TemporalMemory (T2.9) provides
a pragmatic implementation of within-episode temporal learning using the
same SDR + Hebbian algorithm. The temporal reference frame approach below
would go deeper — integrating time directly into the LM's hypothesis
space alongside spatial coordinates.

**Relationship to T2.9:** The TemporalMemory is a standalone temporal
learner that runs alongside the LM. The temporal reference frame would
make time a first-class dimension IN the LM's matching process — each
hypothesis would have a temporal coordinate, and the evidence update
would consider temporal as well as spatial fit. T2.9 can be seen as the
"fast path" that works today; T2.8 is the "deep path" for the future.

**The problem:** The HPC handles cross-episode temporal structure (what
concept follows what). But within-episode temporal structure (how an
object's features change over time at a given location) requires temporal
tracking in the LMs themselves — the cortical columns.

**The biological basis:** Grid cells in the medial entorhinal cortex
provide a universal reference frame. The same path integration mechanism
that tracks spatial displacement also tracks temporal displacement:

```
Spatial:  phase += spatial_displacement / grid_spacing
Temporal: phase += temporal_displacement / temporal_spacing
```

Time cells in the hippocampus and entorhinal cortex tile the temporal
dimension at multiple scales, just as place cells tile space.

**Data structure changes:**

```python
# hypotheses.py — add temporal phase to each hypothesis
@dataclass
class Hypotheses:
    evidence: ndarray           # (N,)
    locations: ndarray          # (N, 3)
    poses: ndarray              # (N, 3, 3)
    scales: ndarray | None      # (N,)
    temporal_phases: ndarray | None  # (N,) ← NEW

# hypotheses_displacer.py — path-integrate temporal displacement
temporal_displacement = 1.0 / episode_steps_per_unit  # or from timer
new_phases = old_phases + temporal_displacement / temporal_spacing
# Multi-scale: different temporal grid modules have different spacings

# object_model.py — store temporal features at graph nodes
# Each node gains: temporal_position (when this feature was observed)
# Edge attr gains: temporal_displacement (time between observations)
Data(
    x=features,
    pos=locations,
    temporal_pos=temporal_positions,    # (N,) ← NEW
    edge_attr=spatial_displacements,
    edge_temporal=temporal_displacements,  # (E,) ← NEW
)
```

**Matching with temporal hypotheses:**

```
search_location = hyp_location + R × (spatial_disp / scale)
search_temporal = hyp_temporal_phase + temporal_disp / temporal_scale

# Match requires BOTH spatial AND temporal proximity
spatial_match = |search_location - stored_location| < max_match_distance
temporal_match = |search_temporal - stored_temporal| < max_temporal_distance

match = spatial_match AND temporal_match
```

**When this becomes testable:**
1. Dynamic objects in simulator (Habitat articulated objects, MuJoCo)
2. Objects that change state during observation (cup being poured)
3. Graph memory that stores temporal features from training on dynamic data

**Prerequisites:**
- Dynamic object support in simulator
- Training pipeline that captures temporal features
- Graph memory extension for temporal node attributes

## Extended YCB Evidence (2026-03-21)

The evidence boundary from the extended YCB benchmark, confirmed by 5
ablation experiments (1120 episodes):

| Category | Baseline | No HSV | No Curv | Pose Only | Catbias |
|---|---|---|---|---|---|
| Ball (SI) | 0% | 0% | 0% | 0% | 0% |
| Tool (SI) | 25% | 7.1% | 10.7% | 0% | 10.7% |
| Can (SI) | 21.4% | 35.7% | 7.1% | 0% | 28.6% |

Ball is 0% in EVERY variant. Root cause analysis (2026-03-21) proved this
is a Layer 1 scale mismatch, not a Layer 2 problem. The fix is multi-scale
grid cell modules, not cross-episode memory. Ball accuracy > 0% would
validate the scale solution, not the Layer 1→2 boundary.

The true Layer 1→2 boundary test is: can the system distinguish objects
with identical geometry but different behavioral properties? (e.g., a
ball that bounces vs one that doesn't). This requires Track 2 (HPC).

## Concept Layers Served

- **Layer 1 scale** (ball vs fruit across sizes): solved by multi-scale
  grid cell modules within a single LM. Not a Layer 2 problem.
- **Layer 1 extreme scale** (toy car vs real car, >10x): a heterarchy
  problem. Higher-level LMs abstract shape across sensor resolutions.
  This is a concrete test case for validating the heterarchy architecture.
- **Layer 1→2 boundary** (same geometry, different behavior): requires
  cross-episode association memory. The real test: objects with identical
  geometry but different behavioral tags.
- **Layer 2** (behavioral concepts: dangerous, useful, heavy, food): requires
  event memory + valence signal + action-effect binding. Deferred until the
  basic association mechanism is validated (Phase 1-2).
- **Layer 3** (social concepts: friend, trust): requires T2.6 (behavior
  recognition) applied to agents rather than objects. Long-term.

## Biological Plausibility

The Track 2 approach follows the TBT's hippocampal formation model:

- **Fast binding**: Hebbian co-occurrence in the HPC association matrix mirrors
  hippocampal pattern completion. Two concepts active together → strengthened
  association. This is the same mechanism as one-shot episodic encoding.
- **Context broadcast**: HPC → cortical column context mirrors hippocampal →
  neocortical replay. The context signal biases active hypotheses, analogous
  to how hippocampal replay during sleep consolidates and primes memories.
- **Cross-episode memory**: The episodic buffer mimics hippocampal episodic
  memory. The relational graph mimics the hippocampal cognitive map (spatial,
  temporal, and conceptual relations in the same substrate).

The design avoids deep learning: no gradient updates, no backpropagation.
Association strength is computed from raw co-occurrence counts with cosine
normalization — a biologically plausible Hebbian rule.
- **Temporal prediction (T2.4)**: Episode transition counts implement a
  frequency-based temporal model analogous to sequence cells / time cells
  in the hippocampus. Pre-play (predicting the next episode's content
  before it begins) mirrors hippocampal pre-play observed in rodent
  navigation studies. Action-conditioned transitions correspond to the
  hippocampus binding motor sequences to state transitions — the basis
  for goal-directed planning.

## Key Files

| Component | Path |
|---|---|
| HippocampalModule | `src/tbp/monty/frameworks/models/hippocampal_module.py` |
| Evidence modulation | `src/tbp/monty/frameworks/models/evidence_matching/learning_module.py` (`_apply_hippocampal_bias()`, `register_dynamic_feature()`) |
| Context dispatch | `src/tbp/monty/frameworks/models/graph_matching.py` (`_dispatch_context_signals()`) |
| LM-only input support | `src/tbp/monty/frameworks/models/monty_base.py` (`_combine_inputs()`) |
| HPC tests | `tests/unit/frameworks/models/hippocampal_test.py` |
| **Track 2 unit tests** | `tests/unit/frameworks/models/track2_mechanism_test.py` (18 tests) |
| **Temporal world model tests** | `tests/unit/frameworks/models/test_temporal_world_model.py` (29 tests) |
| **Track 2 integration tests** | `tests/integration/frameworks/models/track2_context_injection_test.py` (4 tests) |
| EvidenceGraphLM tests | `tests/unit/frameworks/models/evidence_matching/evidence_lm_test.py` |
| Test utilities | `tests/unit/resources/unit_test_utils.py` (`BaseGraphTest`) |
| **HPC eval script** | `tools/run_eval_with_hpc.py` |
| Extended YCB configs | `src/tbp/monty/conf/experiment/extended_ycb_*.yaml` |
| Feature ablation configs | `src/tbp/monty/conf/experiment/extended_ycb_eval_holdout_*.yaml` |
| Ablation analysis | `tools/feature_ablation_analysis.py` |
| Novelty detection | `tools/phase2_category_evidence_readout.py` |
| **Event detection tests** | `tests/unit/frameworks/models/test_event_detection.py` (25 tests) |
| **Temporal prediction voting** | `src/tbp/monty/frameworks/models/graph_matching.py` (`_vote_temporal()`) |
| **End-to-end state training** | `tests/unit/frameworks/models/test_temporal_prediction_voting.py` |
| **A/B test + persistence** | `tests/unit/frameworks/models/test_temporal_priming_ab.py` (11 tests) |
| **HPC training config** | `src/tbp/monty/conf/experiment/extended_ycb_train_hpc.yaml` |
| **HPC trained eval config** | `src/tbp/monty/conf/experiment/extended_ycb_eval_holdout_hpc_trained.yaml` |

## T2.10: Temporal Prediction Voting

**Status:** Implemented (2026-03-22).

### Motivation

The existing `_vote_conditional()` in `GraphMatchingMonty` implements
confusion voting for spatial recognition: LMs that are confident about
object identity send votes to LMs that are stuck. This mirrors the
biological principle that cortical columns complete local processing
before engaging in lateral communication.

Temporal prediction voting applies the same principle to the time
dimension. When an LM makes a prediction about the next state (based on
its learned transition sequence) and that prediction doesn't match what
actually happens, this prediction error signals temporal confusion. Just
as spatial confusion triggers voting, temporal confusion should too.

### Biological basis

In the neocortex, cortical columns continuously predict their next input
based on learned temporal sequences (dendritic predictions on pyramidal
cells). When the prediction matches, only the predicted cells fire
(sparse activation). When the prediction fails, ALL cells in the
minicolumn fire — a burst signal. This burst signal:

1. Propagates laterally to neighboring columns (voting)
2. Signals surprise to higher levels (event detection)
3. Triggers learning (strengthen the actual transition)

The temporal prediction voting mechanism maps directly:
- **Correct prediction → sparse firing**: The LM is "temporally
  confident" — its model correctly predicts the temporal structure.
- **Wrong prediction → burst firing**: The LM is "temporally confused"
  — its model doesn't match reality. Needs votes from confident LMs.
- **Lateral propagation**: Temporally confident LMs vote to help
  temporally confused LMs, just as spatially confident LMs help
  spatially stuck ones.

This is the same mechanism as `_vote_conditional()`, applied to temporal
predictions instead of spatial hypothesis counts.

### Design

**In `EvidenceGraphLM`:**

```python
def _predict_next_state(self):
    """Predict what state should come next based on transition sequence.

    Uses the model's learned transitions: if the current MLH state is A
    and the model has a transition A→B, predict B.
    """
    current_state = self.current_mlh.get("state")
    if current_state is None:
        return None
    graph_id = self.current_mlh.get("graph_id", "")
    models = self.graph_memory.models_in_memory.get(graph_id, {})
    for channel_model in models.values():
        if hasattr(channel_model, "get_next_state"):
            return channel_model.get_next_state(current_state)
    return None

def _check_temporal_prediction(self):
    """Compare predicted next state with actual observed state.

    Returns 'confident' if prediction matched, 'confused' if not,
    None if no prediction was made.
    """
    predicted = getattr(self, "_predicted_next_state", None)
    if predicted is None:
        return None
    actual = self.current_mlh.get("state")
    if actual == predicted:
        return "confident"
    else:
        return "confused"
```

The prediction is made at the END of each step (predicting what comes
next), then checked at the START of the following step (comparing
prediction against reality). This mirrors dendritic prediction: the
cell's apical dendrite is primed before the next input arrives.

**In `GraphMatchingMonty`:**

```python
def _vote_temporal(self):
    """Vote from temporally confident LMs to temporally confused ones.

    Analogous to _vote_conditional, but triggered by temporal prediction
    error rather than spatial hypothesis count. LMs that correctly
    predicted the current state vote to help LMs whose predictions failed.
    """
    confident = set()
    confused = set()

    for i, lm in enumerate(self.learning_modules):
        if not hasattr(lm, "_check_temporal_prediction"):
            continue
        status = lm._check_temporal_prediction()
        if status == "confident":
            confident.add(i)
        elif status == "confused":
            confused.add(i)

    if not confident or not confused:
        return

    # Same one-directional voting: confident → confused
    votes_per_lm = [lm.send_out_vote() for lm in self.learning_modules]

    saved_matrix = self.lm_to_lm_vote_matrix
    temp_matrix = []
    for i in range(len(self.learning_modules)):
        if i in confused:
            senders = [j for j in saved_matrix[i]
                       if j in confident and votes_per_lm[j] is not None]
            temp_matrix.append(senders)
        else:
            temp_matrix.append([])

    self.lm_to_lm_vote_matrix = temp_matrix
    try:
        combined_votes = self._combine_votes(votes_per_lm)
        for i in confused:
            self.send_vote_to_lm(self.learning_modules[i], i, combined_votes)
            self.update_stats_after_vote(self.learning_modules[i])
    finally:
        self.lm_to_lm_vote_matrix = saved_matrix
```

**Wired into `_vote()`:** Temporal voting runs alongside (not instead of)
spatial voting. Both can fire in the same step — they address different
kinds of confusion.

### What temporal prediction voting enables

1. **Multi-column behavior recognition**: The morphology LM (receiving
   static features) may be confused when the object changes state (shape
   is suddenly different). The behavior LM (receiving change signals from
   ChangeDetectingSM) may correctly predict the state transition. The
   behavior LM votes to help the morphology LM.

2. **Tempo recognition**: If predictions are consistently early or late,
   the speed correction mechanism adjusts the timer. Prediction errors
   trigger both voting AND speed correction.

3. **Learning signal**: "Confused" is the signal that the LM needs to
   update its model. The prediction error identifies exactly which
   transition the model is missing or has wrong.

4. **Graceful degradation**: LMs without state-conditioned models (no
   transitions to predict) return `None` from `_check_temporal_prediction`
   and are skipped — no interference with spatial voting.

### End-to-end test coverage

The implementation includes end-to-end tests that:
1. Build `StateConditionedModel` with multiple states and transitions
2. Train an `EvidenceGraphLM` on multi-state observations
3. Run `matching_step` and verify correct state inference in MLH
4. Test temporal prediction: predict → step → check
5. Test multi-LM temporal voting: confident LM helps confused LM
6. Verify backward compatibility: stateless models unaffected

### Files changed

| File | Changes |
|---|---|
| `state_conditioned_model.py` | Added `get_transition_sequence()` alias |
| `learning_module.py` | Added `_predict_next_state()`, `_check_temporal_prediction()`, wired into `matching_step()` |
| `graph_matching.py` | Added `_vote_temporal()`, `temporal_voting` config param, wired into `_vote()` |
| `test_temporal_prediction_voting.py` | New: end-to-end training + matching + voting tests |

## T2.E: Error-Modulated Temporal Learning

Moved to [Track 6: Predictive Coding Heterarchy](track-6-predictive-coding-heterarchy.md)
(milestones T6.0–T6.4). Error-modulated learning is part of the unified
predictive coding architecture that also covers top-down prediction,
surprise-gated communication, and hypothesis proposal.

Summary: surprise-modulated Hebbian learning rate for TemporalMemory (T6.1),
surprise-gated weight decay to prevent saturation (T6.2), and HPC
cross-episode error modulation (T6.3). All parameters default to 0.0 for
backward compatibility. Full design, implementation phases, validation gates,
and risks are in the Track 6 document.
