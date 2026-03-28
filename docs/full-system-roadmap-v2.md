# Full-System Roadmap v2 (Rebuilt From First Principles)

## Purpose

This document replaces the original full-system roadmap. It preserves the
scientific goals and operating discipline but rebuilds the program structure
from first principles, informed by the empirical results of Phases 0-2.

The original roadmap defined 10 sequential phases. This rebuild reorganizes
around 4 parallel research tracks that reflect the actual dependency structure
discovered during implementation.

## What Changed And Why

The original roadmap assumed a strict linear sequence:

    Composition → Categories → Time → Prediction → Language →
    Manipulation → Heterarchy → Scale → Social → Integration

Implementation revealed that this ordering was wrong in several critical ways:

1. **Phase 1 (Composition) failed its gate.** The compositional hierarchy did not
   beat the monolithic baseline (39.88% vs 50.60%). The roadmap said to
   re-evaluate. Instead, Phase 2 succeeded via a completely different mechanism
   (lateral voting) that had nothing to do with compositional hierarchy.

2. **Phase 2's winning mechanism came from Phase 7.** Lateral voting between peer
   LMs — a heterarchy mechanism the original roadmap placed at Phase 7 — produced
   the only quantitative win (+10pp on Omniglot). Every hierarchical, encoding,
   and scoring approach from the original Phase 2 plan failed (46+ experiments).

3. **Heterarchy is foundational, not an optimization.** The original roadmap
   treated heterarchy as a late-stage scaling concern. In practice, it was the
   core cognitive mechanism. Without it, Phase 2 would have achieved 0pp
   improvement.

4. **Most phases are parallel, not sequential.** Categories don't require
   composition. Language doesn't require behavior. Manipulation doesn't require
   language. The actual dependencies are much sparser than the original ordering
   assumed.

## Starting Point (Updated)

### Proven capabilities
- Grounded static object modeling via reference-frame-based graphs
- Sensorimotor matching with evidence accumulation
- Category generalization via lateral voting + category-aggregated evidence
  readout (+10pp on Omniglot cross-version benchmark)
- Multi-LM voting with 44% convergence speedup (3 LMs vs 1 LM)
- Category subgraph extraction from existing graph memories
- Evidence-per-graph logging for post-hoc analysis
- HippocampalModule with 4 memory systems (fast binding, episodic, relational,
  context signal) — scaffolded, 27 unit tests passing
- Extended YCB benchmark (50 objects, 9 categories, 34 train / 16 holdout)
- T3.1-T3.2 trivial language bridge (object + category naming)

### Evidence boundary (2026-03-20)

The extended YCB evaluation (224 episodes, 16 holdout objects, 9 categories)
revealed a fundamental boundary in the system's capabilities. Category accuracy
decomposes into two qualitatively different regimes:

**Shape-congruent categories** (where shape ≈ category):
airplane 100%, fruit 100%, cup 95.2%, utensil 92.9%, box 82.1%, clamp 71.4%.
Weighted average: **93.2%**. Near ceiling for geometric matching.

**Shape-incongruent categories** (where shape ≠ category):
tool 25.0%, can 21.4%, ball 0.0%.
Weighted average: **15.7%**. Irreducible by shape alone.

The confusions are geometrically correct:
- Ball → fruit (both spherical; golf ball ≈ strawberry in geometry)
- Tool → utensil (screwdrivers ≈ spatulas; elongated with handles)
- Can → box (potted meat can is rectangular)

The system is not confused — it is correctly identifying the nearest known
geometry. The "errors" mark the exact boundary where Layer 1 (structural
concepts) ends and Layer 2 (behavioral/functional concepts) begins.

**Evidence margins predict correctness.** Categories with high evidence margin
(utensil: 100.3, airplane: 8.6, cup: 7.1) are always correctly classified.
Categories with low margin (can: 2.9, tool: 1.4, ball: 1.3) are always wrong.
The system has the information to say "I'm not sure" but no mechanism to act
on it.

**All 224 episodes ended "confused"** — the system never converged on any
holdout object. This is correct behavior (holdout objects are not in the
training set), but there is no pathway for "novel object from known category."

These findings restructure the priority stack: Layer 1 refinements (T1.3,
T1.4a) have diminishing returns. The highest-leverage work is at the
Layer 1 → Layer 2 boundary: novelty detection, feature contribution analysis,
and cross-episode memory.

### Feature ablation + category bias results (2026-03-21)

Five eval runs (224 episodes each, 1120 total) confirm the Layer 1 ceiling
and reveal the contribution of each feature:

```
VARIANT             OVERALL  SHAPE-CONG  SHAPE-INC
--------------------------------------------------
baseline (all)        67.9%       92.2%      14.3%
no_hsv                65.6%       90.9%      10.0%
no_curvature          66.1%       93.5%       5.7%
pose_only              0.0%        0.0%       0.0%
catbias (0.1)         61.2%       84.4%      10.0%
```

Key findings:
1. **Pose vectors carry zero discriminative signal** (0% across the board).
   Essential for exploration but not identification.
2. **Curvature is the primary discriminative feature.** Removing it drops
   shape-incongruent from 14.3% → 5.7%. It's the main signal distinguishing
   cans from boxes and tools from utensils.
3. **HSV has mixed effects.** Helps utensils and fruits slightly, hurts cups.
   Net contribution small but positive.
4. **Category bias (T1.4a) HURTS overall** (67.9% → 61.2%). Exactly as
   predicted: it destroys fruit accuracy (100% → 64.3%) by biasing toward
   wrong categories. Only helps clamp (71% → 86%) and can (21% → 29%).
5. **The baseline feature set is already near-optimal for Layer 1.** No
   ablation or bias mechanism improves overall accuracy. The 67.9% ceiling
   is the empirical limit of shape-based matching with current features.

**Novelty detection (T1.N)** validated: evidence margin separates confident
predictions (86.2% accuracy) from uncertain ones (50.4%), a 35.8pp split.

### Root cause analysis (2026-03-21)

First-principles analysis of the ball 0% failure mode revealed that the
previous diagnosis ("balls need behavioral/non-geometric information") was
**wrong**. The actual root cause is a scale mismatch in spatial matching:

1. **No scale invariance.** (`detected_scale: 1, # TODO: scale doesn't work
   yet`). All hypotheses are initialized at scale=1. Holdout balls (golf ball
   4.3cm, racquetball 5.7cm) are smaller than all training balls (6.7-12.8cm).
2. **Spatial gate blocks all evidence.** `max_match_distance = 0.01m` (1cm).
   A golf ball surface point is 1.5cm away from a baseball surface point at
   scale=1. This exceeds the threshold → zero spatial matches → zero evidence
   for ALL spheres (balls AND fruits).
3. **Large graphs win by accident.** With `evidence_size_norm_power = 0`, a
   4894-node airplane has 8x more accidental spatial matches than a 427-node
   orange, regardless of feature quality. Holdout balls match airplanes and
   clamps, not fruits.
4. **HSV saturation DOES distinguish balls from fruits** (0.35 vs 0.84, a
   2.4x difference), but this signal never fires because the spatial gate
   blocks evidence before features are compared.

Analytical simulation (seconds, no Habitat) confirmed: at
`max_match_distance = 0.02m`, baseball becomes #1 match for a golf-ball-sized
query. With scale-invariant matching, baseball is #1 and ball/fruit separation
is clear. The features were always sufficient — the matching pipeline
couldn't access them.

**The path forward is fixing the matching pipeline (scale invariance,
evidence size normalization, feature weights), not Track 2 behavioral
memory.** Track 2 remains important for genuine Layer 2 concepts (dangerous,
useful, heavy) but is NOT needed for the ball→fruit canary.

## Methodology: Analytical-First Testing

**Every hypothesis MUST be tested analytically before running Habitat evals.**

Full eval runs take 1-4 hours per variant. Analytical tests using stored graph
data take seconds. The analytical-first methodology:

1. **Load trained graphs** from `model.pt` (has node positions, features,
   graph structure for all 34 objects)
2. **Simulate the query** — generate synthetic observation points matching
   the holdout object's geometry (sphere for balls, etc.)
3. **Compute matching analytically** — for each training graph, find spatial
   neighbors, compute feature similarity, sum evidence
4. **Sweep parameters** — test different match distances, feature weights,
   normalization powers in a loop
5. **Only run Habitat eval for the winning config** — confirms that the
   analytical prediction holds in the full iterative pipeline

This methodology caught the scale mismatch root cause in seconds. Running
5 Habitat ablation experiments (1120 episodes, hours) did NOT catch it
because the experiments varied features, not matching parameters.

**Rule: if you can test it with stored graph data, do that first.**

### Three fundamental restrictions
All remaining cognitive capabilities are blocked by three restrictions in the
current implementation. These are not bugs or missing features — they are
architectural boundaries that define where the system stops.

**Restriction 1: Positions must come from sensors.**
Every graph node has a 3D position derived from a depth sensor. But abstract
concepts (numbers, social relationships, temporal sequences) require reference
frames where "position" means something other than physical location. The brain
solves this via hippocampal replay: stored episodes are replayed to cortical
columns as synthetic input, allowing columns to build models from non-sensory
patterns. The HippocampalModule's `get_context_signal()` is the designed
mechanism for this, but it is not yet connected to downstream LMs as input.

**Restriction 2: Features are fixed at config time.**
The features an LM observes (HSV, curvature, surface normal) are defined in
YAML at config time. A column processing social interactions would need features
like valence, agent identity, and action type — computed from other columns'
outputs, not from sensors. The CMP already supports LM-to-LM output (the
compositional hierarchy uses `object_id` as a feature from child to parent),
but this is rigid and limited. The solution: make any LM's output available as
features to any connected LM, dynamically.

**Restriction 3: Each episode is independent.**
Monty processes one recognition episode at a time with no memory across episodes.
Graph memory persists object models (that's how training works), but there is no
event memory — no record of what happened, what action was taken, or what the
outcome was. Without cross-episode memory, the system cannot learn behavioral
associations (Layer 2), social patterns (Layer 3), or cultural norms (Layer 4).
The HippocampalModule's episodic buffer is designed for this but is not yet
connected to the matching pipeline.

## The Core Theoretical Claim

The Thousand Brains Theory claims that the neocortex uses one mechanism —
cortical columns with reference frames — for all cognition: spatial, temporal,
social, and abstract.

The evidence from neuroscience:
- The neocortex has uniform 6-layer columnar architecture everywhere
- Abstract reasoning activates spatial brain regions
- Mathematical concepts are systematically structured by spatial metaphors
  (numbers as points on a path, sets as containers, operations as movements)
- Language understanding activates sensorimotor cortex
- Children learn abstract concepts through concrete spatial experience first

The implication: reference frames are not inherently spatial. They are
**relational structures** — any organized space where things have positions
relative to each other. The same columnar mechanism that models "the surface
of a cup" can model "the number line" or "the relationship between me and
my friend" — if given the right input.

The 3 restrictions above are exactly the gap between "same mechanism" and
"different input." Lifting them, in order, is the path to higher cognition.

## Concept Layers

From first principles, concepts form layers where each layer requires
the previous plus new infrastructure:

### Layer 1: Structural concepts (cup, sphere, red, curved, alphabet)
Patterns in spatial and feature structure directly observed by sensors.
**Status: PROVEN, CEILING MEASURED.** Phase 2 demonstrated +10pp category
generalization. Extended YCB shows 93.2% accuracy on shape-congruent categories
(near ceiling) and 15.7% on shape-incongruent categories (irreducible by shape).
The ball→fruit, screwdriver→spatula, and can→box confusions mark the exact
empirical boundary of Layer 1.
Requires: sensors + graphs + multi-column voting + evidence readout.
Heterarchy needed: static peer voting (already works).

### Layer 1→2 boundary (revised 2026-03-21)
The original analysis claimed that ball→fruit confusion marks the Layer 1→2
boundary (needing behavioral memory). Root cause analysis proved this wrong:
holdout balls fail because of **scale mismatch** (no scale hypotheses) and
**evidence size bias** (large graphs win by node count), not because geometry
is insufficient. HSV saturation (0.35 vs 0.84) already distinguishes balls
from fruits — the spatial gate blocks this signal from ever being evaluated.

The true Layer 1→2 boundary is NOT yet measured. It can only be measured
after scale-invariant matching is implemented, which will raise the Layer 1
ceiling above 67.9%. The remaining failures after that fix will mark the
real boundary where behavioral/functional knowledge is needed.

### Layer 2: Behavioral/associative concepts (dangerous, useful, heavy, food)
Associations between structural patterns and interaction outcomes.
"Dangerous" = things that look like this have hurt me before.
"Food" = things that look like this were edible when I interacted with them.
Requires: Layer 1 + cross-episode memory (Restriction 3) + valence signal +
action-effect binding.
Heterarchy needed: context-dependent routing (action ↔ perception modules).

### Layer 3: Social/relational concepts (friend, enemy, trust, love)
Patterns across repeated interactions with specific agents over time.
"Friend" = this agent consistently produces positive outcomes across many episodes.
Requires: Layer 2 + persistent agent identity + theory of mind (modeling another
agent's internal state as a reference frame).
Heterarchy needed: cross-type routing (visual ↔ memory ↔ behavior ↔ mind model).

### Layer 4: Cultural/normative concepts (justice, politics, democracy)
Shared models across multiple agents about how groups should behave.
Requires: Layer 3 + language + multi-agent models + normative knowledge.
Heterarchy needed: full dynamic cross-type routing.

### Layer 5: Formal/abstract concepts (number, proof, infinity, function)
Abstract reference frames with no sensory grounding, operated on by rules.
Requires: abstract entity creation (Restriction 1 fully lifted) + universal
quantification + rule-based transformation + compositional reasoning.
Heterarchy needed: sequential logical chaining (may require architectural changes).
**Status: THEORETICAL.** No clear path from current Monty.

## Program Structure: Four Parallel Tracks

The original 10 sequential phases are reorganized into 4 parallel research
tracks plus operating infrastructure carried forward from v1.

### Track 1: Multi-Column Intelligence

**What it enables**: Categories, composition, abstraction, attention, routing
**Concept layers served**: 1, 2 (partially), 3 (partially)
**Restriction addressed**: None directly (this track exploits what already works)

This is the most proven track. The +10pp result came from lateral voting +
category-aggregated evidence readout — both multi-column mechanisms.

| Milestone | Description | Status |
|---|---|---|
| T1.1: Static peer voting | Fixed vote matrix, all-to-all | **Done** (Phase 2) |
| T1.2: Category-aggregated readout | Sum evidence by category post-hoc | **Done** (+10pp) |
| T1.N: Novelty detection | Margin-based "novel object from category X" | **Next** |
| T1.FA: Feature ablation | HSV on/off, curvature on/off contribution analysis | **Next** |
| T1.3: Online category readout | Category aggregation during matching, not post-hoc | After T1.N |
| T1.4a: Category-biased evidence (single LM) | Self-bias within one LM | After T1.FA (risk: see below) |
| T1.R2: CMP enrichment (lifts Restriction 2) | Make LM outputs consumable as features by connected LMs | After T1.4a |
| T1.4b: Cross-LM category bias | LM0's category evidence biases LM1 via CMP | After T1.R2 |
| T1.5: Top-down biasing | Parent LM constrains child hypotheses via CMP | After T1.R2 |
| T1.6: Attention-based routing | Dynamic who-talks-to-whom based on context | Planned |
| T1.7: Learned routing | Route discovery from evidence statistics | Research |

**Immediate next step (T1.N)**: Exploit the evidence margin signal. The extended
YCB data proves that evidence margin predicts correctness: high-margin categories
(>6.0) are always right, low-margin categories (<3.0) are always wrong. A
threshold on `(best_category_evidence - second_best) / best_category_evidence`
enables the system to say "I don't recognize this exact object, but it's most
similar to category X (confidence: high/low)." This is ~10 lines in the
evidence readout and gives the system a qualitatively new capability: knowing
what it doesn't know.

**Then (T1.FA)**: Feature ablation study. The eval pipeline uses HSV, pose
vectors, and principal curvatures for scoring. But balls still match fruits
despite HSV being available. Before building more complex mechanisms, measure
what each feature actually contributes. Run eval with: (a) all features,
(b) geometry only (no HSV), (c) HSV amplified (weight 2.0). If HSV adds noise,
remove it. If it helps shape-congruent but not shape-incongruent categories,
document that boundary.

**Risk assessment for T1.4a**: The extended YCB evidence matrix shows that
category bias is a double-edged sword. It will amplify the strongest category
signal — which is WRONG for shape-incongruent categories (balls → fruit bias
gets reinforced). Prediction before implementation: T1.4a improves shape-
congruent accuracy (93.2% → ~96-98%) but degrades shape-incongruent accuracy
(15.7% → possibly lower). The experiment MUST measure both splits separately,
not a single accuracy number. If the prediction holds, the mechanism is
understood and the result is informative even if overall accuracy stays flat.

**Then (T1.R2)**: Lift Restriction 2 to enable cross-LM feature flow. The
first-principles analysis shows this is a small change (~20 lines):
1. Make `feature_weights` and `tolerances` updatable at runtime
2. Inject LM outputs into the observation dict alongside sensor features
3. Use variance-based auto-tolerance for new features
The matching pipeline already iterates over registered features — the change
is making registration dynamic, not redesigning the pipeline.

**Then (T1.4b-T1.5)**: Use the enriched CMP for cross-LM category bias and
top-down biasing. Also enables the HippocampalModule's context signal to
reach matching LMs.

### Track 2: Temporal World Model

**What it enables**: State recognition, behavior learning, prediction,
action-effect models, manipulation
**Concept layers served**: 2, 3 (partially)
**Restriction addressed**: Restriction 3 (cross-episode memory)

This track turns Monty from a static recognition system into a dynamic world
model. The original roadmap split this across Phases 3, 4, and 6. They are
one continuous problem: how does the world change over time and in response
to actions?

| Milestone | Description | Status |
|---|---|---|
| T2.1: Cross-episode event memory | Store (object, action, outcome, time) tuples that persist | Next |
| T2.2: State discrimination | Same object, two states (cup upright vs inverted) | Next |
| T2.3: Temporal edges in graphs | Edges that encode "after I did X, the state changed to Y" | Planned |
| T2.4: Action-effect prediction | Given current state + action, predict next state | Planned |
| T2.5: Behavior recognition | Recognize behavior independent of object (opening, pouring) | Planned |
| T2.6: Manipulation | Use prediction to plan and execute multi-step actions | Planned |

**Immediate next step (T2.1-T2.2)**: Wire the HippocampalModule's episodic
buffer to persist across episodes. Two test cases:

1. **State discrimination** (original plan): Train on a cup upright, train on
   the same cup inverted. Test whether the system can discriminate the two
   states of the same object.
2. **Ball→fruit canary** (new, from extended YCB): After training on balls and
   fruits, add interaction episodes (e.g., bounce vs squish). Test whether
   episodic memory resolves the ball→fruit confusion that is irreducible by
   shape. This is the primary canary metric for Layer 2 capability — the day
   balls stop matching fruits is the day we have genuinely advanced beyond
   shape matching.

The HippocampalModule already has `_record_episode_step()` and an episodic
deque. The gap is: connecting this to the experiment loop so episodes are stored
and queryable, and creating an environment where objects have states or
interaction outcomes.

### Track 3: Grounding Bridge

**What it enables**: Language grounding, semantic naming, instruction following,
agent modeling
**Concept layers served**: 3, 4
**Restrictions addressed**: Restriction 1 (non-sensory input), Restriction 2
(flexible features)

This track connects Monty's grounded world model to external systems: language,
other agents, and abstract domains.

| Milestone | Description | Status |
|---|---|---|
| T3.1: Object naming | Map Monty object IDs to words (lookup table) | Trivial |
| T3.2: Category naming | Map category-aggregated evidence to category words | Trivial |
| T3.3: Spatial relation language | "The mug is on the table" from object positions | Planned |
| T3.4: Instruction following | "Pick up the red mug" → object identification + motor goal | Planned |
| T3.5: Agent identity | Distinguish objects from agents (agents move on their own) | Planned |
| T3.6: Behavioral language | "The agent is approaching the cup" | Requires T2.5 |
| T3.7: Theory of mind | Model another agent's reference frame / knowledge state | Research |

**Immediate next step (T3.1-T3.2)**: Build the trivial language bridge — a
mapping from object IDs and category names to words. This costs almost nothing
and immediately enables "tell me what you see" functionality. The TextSM and
TextOutputHook from the world model extension provide the input/output
scaffolding.

### Track 4: Systems Engineering

**What it enables**: Speed, reliability, scalability
**Cross-cutting**: supports all other tracks

| Milestone | Description | Status |
|---|---|---|
| T4.1: Evidence logging | Per-graph evidence in eval stats | **Done** |
| T4.2: Distance-aware metrics | Graded evaluation replacing binary accuracy | **Done** |
| T4.3: Category readout tool | Post-hoc category-aggregated scoring | **Done** |
| T4.4: Faster matching | Approximate nearest neighbor, GPU acceleration | Planned (see [Track 6](track-6-predictive-coding-heterarchy.md) T6.7) |
| T4.5: Dynamic hypothesis management | Prune low-evidence hypotheses during matching | Planned (see [Track 6](track-6-predictive-coding-heterarchy.md) T6.11–T6.16) |
| T4.6: Checkpoint infrastructure | Reliable save/load/merge across experiments | Ongoing |
| T4.7: Parallel execution | Multi-process LMs, distributed experiments | Planned (see [Track 6](track-6-predictive-coding-heterarchy.md) T6.8) |

Detailed plans for T4.4, T4.5, and T4.7 (including learned hypothesis
proposal, KDTree/FAISS acceleration, and GIL-free parallelism) are in
[Track 6: Predictive Coding Heterarchy](track-6-predictive-coding-heterarchy.md),
which consolidates all inter-LM communication, error-modulated learning,
and scalability work.

## Heterarchy Is Not A Phase

The original roadmap placed heterarchy at Phase 7. This was the single largest
structural error. Heterarchy is not a capability to be added later — it is the
**communication substrate** that every other capability depends on.

Heterarchy comes in layers of sophistication, each needed by different
capabilities:

| Level | Mechanism | What needs it | Status |
|---|---|---|---|
| Static peer voting | Fixed vote matrix, pose-space agreement | Categories (Layer 1) | **Working** |
| Conditional voting | Confidence/confusion-gated directed voting | Convergence speed | **Working** (Mode B) |
| Temporal voting | Prediction-error-gated directed voting | Behavior, state tracking | **Working** (Mode C) |
| Unified predictive voting | Combined surprise + cardinality gating | All tracks | Planned ([Track 6](track-6-predictive-coding-heterarchy.md) T6.5) |
| Top-down prediction | Parent predicts child state, child sends errors | Composition, scalability | Planned ([Track 6](track-6-predictive-coding-heterarchy.md) T6.9–T6.12) |
| Cross-modal routing | Visual ↔ text ↔ motor module communication | Language grounding | Not started |
| Recursive composition | Modules chaining logical steps sequentially | Formal reasoning | Theoretical |

Each level should be developed alongside the track that needs it, not as a
separate phase. The full predictive coding heterarchy plan (error-modulated
learning, surprise-gated communication, top-down prediction, hypothesis
proposal) is in [Track 6](track-6-predictive-coding-heterarchy.md).

## Hippocampal Integration: The Bridge To Higher Layers

The HippocampalModule is the single most important piece of infrastructure for
progressing beyond Layer 1. It provides:

- **Episodic memory**: storage that survives across episodes (lifts Restriction 3)
- **Replay**: synthetic input to downstream LMs (lifts Restriction 1)
- **Association matrix**: detects co-activation patterns across instances
  (the same mechanism that produced the +10pp category readout, but accumulated
  over time rather than computed per-episode)
- **Relational graph**: typed edges (cooccurrence, spatial, temporal) between
  concepts — the substrate for Layer 2 and Layer 3 concepts

**The key architectural insight**: The hippocampus is to temporal/social concepts
what lateral voting is to structural concepts. Lateral voting extracts categories
from the pattern of instance co-activation within an episode. The hippocampus
extracts behavioral and social concepts from the pattern of instance
co-activation across episodes. Same principle, different timescale.

## Restriction Resolution Order

The three restrictions have a dependency chain, but a lot of valuable work
can happen BEFORE any restriction is lifted. The execution plan interleaves
restriction-free work with restriction lifting.

Updated 2026-03-20 based on extended YCB evidence boundary analysis.
The key finding: Layer 1 is approaching its ceiling (93.2% on shape-congruent
categories). This shifts priority toward Layer 1→2 boundary work and
Restriction 3 (cross-episode memory), which is the gate to Layer 2.

```
── No restriction needed ──────────────────────────────────
T1.N:  Novelty detection via evidence margin (~10 lines)
T1.FA: Feature ablation (HSV / curvature contribution)
T3.1-T3.2: Trivial language bridge (DONE)
T1.3:  Online category readout (single LM)
T1.4a: Category-biased evidence (RISK: hurts shape-incongruent categories)

── Restriction 3 (cross-episode memory) ───────────────────
T2.1-T2.2: Wire HippocampalModule episodic buffer
           Store events, not just object models
           Test 1: state discrimination (cup upright vs inverted)
           Test 2: ball→fruit canary (Layer 2 gate metric)

── Restriction 2 (flexible features / CMP enrichment) ─────
T1.R2: Make feature_weights/tolerances updatable at runtime
       Inject LM outputs into observation dict
       Auto-tolerance for dynamically registered features
       ~20 lines of core change, NOT a redesign
       Informed by T1.FA results (which features matter)

── Restriction 2 enables ──────────────────────────────────
T1.4b: Cross-LM category bias via enriched CMP
T1.5: Top-down biasing (parent constrains child via CMP)
HPC→LMs: Hippocampal context signal reaches matching LMs

── Restriction 1 (non-sensory input) ──────────────────────
Hippocampal replay feeds stored episodes as synthetic input
Columns build models from non-sensory patterns
Enables: Layer 2+ concepts (behavioral, social, abstract)
```

Key insight from first-principles analysis of Restriction 2: the matching
pipeline is already feature-agnostic (iterates over whatever is registered).
The change is making registration dynamic, not redesigning the pipeline. The
hard part is not the plumbing but the design question: what representation
should flow between LMs (winner only, confidence, full evidence distribution,
or category-aggregated evidence). T1.FA and T1.4a will inform this design
question experimentally.

Key insight from extended YCB (2026-03-20): Layer 1 refinements face
diminishing returns. The 93.2% on shape-congruent categories is near ceiling.
The 15.7% on shape-incongruent categories is irreducible by any Layer 1
mechanism. The highest-leverage path forward is the Layer 1→2 boundary:
novelty detection (T1.N), feature understanding (T1.FA), and then cross-episode
memory (T2.1-T2.2) to access Layer 2 concepts.

## Definition Of Success

For this roadmap, a "full system" means a system that can:

- learn grounded models from sensorimotor interaction *(proven)*
- generalize from instances to categories by shape *(proven, 93.2% ext. YCB)*
- generalize from instances to categories by function *(Layer 2, 15.7% baseline)*
- know what it doesn't know (novelty detection) *(T1.N, signal proven)*
- model time, state, and behavior *(Track 2)*
- predict future observations and action outcomes *(Track 2)*
- ground language in learned world models *(Track 3, T3.1-T3.2 done)*
- route information across a heterarchical network *(Track 1)*
- accumulate knowledge across episodes *(Restriction 3)*
- build models of entities it has never directly sensed *(Restriction 1)*
- scale to large memory and many interacting modules *(Track 4)*

## Go/No-Go Gates

### Gate 1: Novelty Detection (Track 1, T1.N) — PASSED (2026-03-21)
Evidence margin reliably separates correct from incorrect: 86.2% accuracy
on confident predictions vs 50.4% on uncertain (35.8pp separation).
Implemented in `logging_utils.py` and `phase2_category_evidence_readout.py`.

### Gate 2: Category Bias Double-Edge (Track 1, T1.4a) — FAILED (2026-03-21)
Prediction was: shape-congruent improves, shape-incongruent degrades. Actual
result: BOTH degrade. Overall: 67.9% → 61.2% (-6.7pp). Fruit destroyed
(100% → 64.3%). The normalized category evidence creates misleading signals.
**Conclusion**: T1.4a is a dead end. Category bias should NOT be used for
evidence accumulation on this benchmark. T1.4b (cross-LM bias) is also
deprioritized as it depends on the same mechanism.

### Gate 3: Feature Ablation (Track 1, T1.FA) — PASSED (2026-03-21)
Five runs (1120 episodes) confirm:
- Curvature is the primary discriminative feature (removing it: 14.3% → 5.7%)
- HSV has mixed effects (net slightly positive)
- Pose vectors carry zero discriminative signal (0% across the board)
- Baseline feature set is already near-optimal for Layer 1
**Conclusion**: No feature engineering path improves the 67.9% ceiling.
The path forward is Track 2 (behavioral memory).

### Gate 4: Ball→Fruit Canary — REDIAGNOSED (2026-03-21)
The ball 0% was originally attributed to the Layer 1→2 boundary (needing
behavioral memory). First-principles root cause analysis proved this wrong:
the actual cause is **scale mismatch + evidence size bias** in the matching
pipeline. Analytical testing confirms that scale-invariant matching puts
ball as #1 in seconds. This gate is now a **Layer 1 matching fix**, not a
Layer 2 problem.

### Gate 5: Hippocampal Replay (Restriction 1) — OPEN
If replaying stored episodes to downstream LMs does not cause them to build
useful non-sensory models, the reference-frame-reuse hypothesis may be wrong
and abstract concepts may require genuinely different computational mechanisms.

## Immediate Priority Stack

Updated 2026-03-21 based on root cause analysis of ball 0%.

The ball→fruit canary is NOT a Layer 2 problem — it's a Layer 1 matching
pipeline bug (no scale invariance, no evidence size normalization). The
67.9% ceiling is NOT the true Layer 1 ceiling; it's an artifact of the
spatial matching gate blocking cross-scale evidence.

1. **T1.S: Scale-invariant matching** — the #1 priority.
   Add scale as a hypothesis dimension. Analytical test proves ball becomes #1
   match. Expected impact: ball 0% → >0%, shape-incongruent improves broadly.
   Also fixes any cross-scale confusion (small objects matching large objects
   by accident).
2. **T1.SN: Evidence size normalization** — complementary to T1.S.
   Set `evidence_size_norm_power > 0`. Removes the advantage of large graphs
   (4894-node airplane vs 427-node orange). Config-only change.
3. **T1.SW: Saturation weight tuning** — use the signal that's already there.
   Increase HSV saturation weight from 0.5 to 2.0. Saturation is the most
   discriminative feature between balls (0.35) and fruits (0.84) but is
   currently under-weighted. Config-only change.
4. **T2.1-T2.2**: Cross-episode memory + genuine Layer 2 concepts.
   Still important for real behavioral concepts (dangerous, useful, heavy),
   but no longer needed for the ball canary. Deprioritized until T1.S proves
   that the matching pipeline is working correctly.
5. **T1.5**: Top-down biasing. After T1.S validates scale matching.
6. **T3.3**: Spatial relation language.

## Scorecard

| Area | Readiness | Evidence |
|---|---|---|
| Static grounded recognition | L4 | Robust YCB results |
| Category gen. (shape-congruent) | L4 | 92.2% on ext. YCB; confirmed ceiling by 5 ablations |
| Category gen. (shape-incongruent) | L1 | 14.3% on ext. YCB; ball 0% caused by scale mismatch not feature gap |
| Category gen. (Omniglot) | L3 | +10pp cross-version; 70% ceiling from spatial features |
| Novelty detection | L4 | 86.2% confident vs 50.4% uncertain (35.8pp separation) |
| Feature contribution analysis | L4 | 5 runs, 1120 eps: curvature > HSV > pose (for identity) |
| Category bias (T1.4a) | L4 | Tested, failed: -6.7pp overall (dead end on ext. YCB) |
| CMP enrichment (T1.R2) | L3 | API implemented, ready for Track 2 integration |
| Compositional part-whole | L2 | Prototype exists, negative result vs monolithic |
| Lateral voting / heterarchy | L4 | 44% faster convergence, category subgraph wins |
| Category subgraph extraction | L3 | 100% offline accuracy, 25% online win rate (Omniglot) |
| Evidence-per-graph logging | L5 | Integrated into eval pipeline |
| Distance-aware evaluation | L3 | Tool built, validated on YCB and Omniglot |
| Extended YCB benchmark | L3 | 50 objects, 9 categories, decomposed metric, reproducible |
| Hippocampal module | L3 | 700+ lines, 53 tests, 4 memory systems, wired + integration tested |
| Scale-invariant matching | L1 | Root cause identified, analytical fix validated, code TODO |
| Evidence size normalization | L1 | Confound identified, config-only fix ready |
| Analytical-first testing | L5 | Methodology proven: seconds vs hours, caught root cause |
| Time and state | L0 | No implementation |
| Behavior learning | L0 | No implementation |
| Prediction | L1 | Implicit in matching (hypothesis-conditioned expectations) |
| Language grounding | L2 | T3.1-T3.2 done (naming bridge), no grounded mapping |
| Manipulation | L0 | Motor system exists for exploration only |
| Multi-agent / social | L0 | No implementation |
| Abstract reasoning | L0 | Theoretical only |

Readiness levels: L0 = no implementation, L1 = theory only, L2 = prototype,
L3 = reproducible benchmark, L4 = strong result against baseline, L5 = integrated

## Document Organization

Each track has a lightweight **track index** document that links to individual
**investigation** documents. Investigation documents follow the template in
[phase-document-template.md](phase-document-template.md) and contain the full
technical detail (benchmarks, execution registry, decision log, etc.).

Multiple investigations can run within a track. The track index maintains the
overall status, links to all investigations, and records which investigation
is currently active.

### Track Index Registry

| Track | Index Document | Active Investigation |
|---|---|---|
| (pre-track) | — | [Phase 0: Baseline Discipline](phase-0-baseline-discipline.md) (completed) |
| Track 1 | [track-1-multi-column-intelligence.md](track-1-multi-column-intelligence.md) | **T1.S scale-invariant matching (#1 priority)** |
| Track 2 | [track-2-temporal-world-model.md](track-2-temporal-world-model.md) | Mechanism complete (53 tests); deprioritized until T1.S done |
| Track 3 | [track-3-grounding-bridge.md](track-3-grounding-bridge.md) | T3.1-T3.2 done (naming bridge) |
| Track 4 | (in roadmap) | Cross-cutting engineering |

### Relationship To v1 Roadmap

This document supersedes `full-system-roadmap.md`. The mapping:

| v1 Phase | v2 Track | Notes |
|---|---|---|
| Phase 0: Baseline | (completed) | Pre-track work |
| Phase 1: Composition | Track 1 | Negative result; lateral voting succeeded instead |
| Phase 2: Categories | Track 1 | +10pp via voting + readout |
| Phase 3: Time/State | Track 2 | |
| Phase 4: Prediction | Track 2 | Implicit in matching; make it explicit |
| Phase 5: Language | Track 3 | Moved earlier; trivial bridge is nearly free |
| Phase 6: Manipulation | Track 2 | Depends on action-effect prediction, not language |
| Phase 7: Heterarchy | Track 1 | Promoted to foundational; not a late optimization |
| Phase 8: Scale | Track 4 | Cross-cutting engineering |
| Phase 9: Social | Track 3 | Depends on Track 2 (behavior modeling) |
| Phase 10: Integration | (deferred) | Meaningful only after Tracks 1-3 mature |

## Operating Rules For Prompt Loops

If this roadmap is used as the basis of an implementation prompt, the following rules
should be treated as part of the operating contract.

- Default to the Thousand Brains Theory and Jeff Hawkins style of reasoning as the
  starting prior, not as untouchable dogma.
- For every major design decision, ask whether it preserves the core TBT ideas:
  reference frames, sensorimotor learning, object-centric structure, cortical-column
  reuse, and heterarchical coordination.
- When a phase proposes the use of deep neural networks, justify clearly why they are
  needed, what role they play, and why a more brain-like mechanism is insufficient.
- If a non-TBT or non-biological mechanism is chosen, the phase document must say so
  explicitly and explain the tradeoff.
- If CMP, voting, routing, timing, or LM internals are modified, the phase document
  must include a biological plausibility note describing whether the change is
  consistent with the broader theory, merely engineering-motivated, or in direct
  tension with the theory.
- Always preserve the distinction between grounded world modeling and helper systems.
  External neural components should support Monty, not silently replace its core job.
- Every phase should end with a recommendation that is evidence-based: continue,
  revise, pause, or abandon.

## Stability Rules

To keep the development loop stable and useful, every active investigation should
maintain the following artifacts.

- one investigation document that acts as the single source of truth
- one benchmark definition or experiment list tied to the investigation goal
- one explicit decision log in the investigation document
- one current blocker section in the investigation document
- one observability artifact or tool that shows what the system is doing internally
- one execution registry for every non-trivial test, training run, or evaluation run

The loop is considered unstable if any of the following happen.

- benchmark status is spread across chat history only
- key design decisions are not written down
- an investigation advances without a concrete test plan
- an investigation introduces opaque machinery without an inspection method
- results cannot be reproduced from the investigation document
- long-running commands are launched without being written down first
- a run cannot be monitored, resumed, or safely restarted after a machine or editor
  restart

## Execution Safety Protocol

Before launching any non-trivial test, training job, evaluation run, or data pipeline,
the following must be true.

- the exact launch command is written into the active investigation document first
- the document includes a command to check run state
- the document includes the expected output location
- the document includes a recovery or resume command if the run is interrupted
- the document includes a stop or cleanup command if the run must be halted
- those same commands are also shown to the user in chat for visibility at the time the
  run is proposed or launched

For this roadmap, a non-trivial run means any command that is expected to take more
than a quick local validation step or that may continue in the background.

### Required Run Metadata

Every recorded run should include:

- purpose
- launch command
- monitoring command
- expected outputs
- resume or recovery command
- stop command
- resource profile
- current status

### Restart And Recovery Requirement

Assume that VS Code, the terminal session, or the entire machine may restart at any
time.

No long-running work should depend on fragile in-memory context only.

- background runs should write logs to stable files
- output directories should be named predictably
- recovery steps should be documented before launch, not after failure
- if a job cannot be resumed cleanly, the investigation document must say so explicitly
  and explain the workaround

### Chat Visibility Requirement

Commands stored only in documentation are not enough.

Whenever a run is about to be launched, the user should also see in chat:

- the launch command
- the monitor command
- the output path
- the resume or recovery command

This is required even if the same commands are already in the investigation document.

## Resource Budget Policy

This machine is resource-constrained relative to typical research workloads.

Current default planning assumptions:

- 8 CPU cores
- 16 GB RAM

Therefore the default local execution policy is conservative.

- prefer one heavy run at a time
- prefer `num_parallel=1` unless an investigation document justifies more
- prefer limiting math-library thread counts for heavy Python workloads
- avoid multiple simultaneous Habitat or similarly heavy jobs
- prefer detached or restart-safe runs for long jobs
- prefer smaller validation runs before full runs
- record resource-saving flags directly in the investigation document

For CPU-heavy Python jobs, the default safe profile should usually include:

- `OMP_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`
- `OPENBLAS_NUM_THREADS=1`

If an investigation needs a more aggressive resource profile, the document must explain:

- why it is necessary
- what the expected RAM and CPU load is
- what the rollback plan is if the machine becomes unstable

## Biological Plausibility Policy

The intended stance is critical biological realism.

- Start by asking what a brain-like solution would look like.
- Prefer explicit TBT-compatible mechanisms when they are good enough.
- Consider HTM-like or SDR-like mechanisms seriously where they improve biological
  plausibility and preserve interpretability.
- Use standard deep learning components only when they clearly solve a problem that the
  current Monty substrate cannot yet solve practically, and document the reason.
- Do not justify an addition only by saying that it performs better. Also ask whether
  it damages the long-term scientific goal of building a more brain-like system.
- Do not reject neural network components on ideology alone. Reject them only if they
  weaken the architecture, obscure understanding, or replace the core problem rather
  than help solve it.

## Observability First

Every investigation must make the system easier to inspect.

- In the fast.ai style of practical interpretability, every important capability should
  be accompanied by a way to look at what the system is doing.
- New algorithms should ship with at least one inspection tool, visualization, replay,
  trace collector, state browser, or debug report.
- If an investigation cannot be represented through a tool, add one before calling it
  stable.

Acceptable observability artifacts include:

- hypothesis visualizers
- graph memory browsers
- evidence heatmaps
- state and timer traces
- track-specific dashboards
- prediction versus reality replays
- routing and voting inspection tools

## Investigation Document Protocol

When a new investigation starts within a track, create or update a dedicated
investigation document and link it from the track index.

Every investigation document must include:

- purpose and scope
- investigation summary
- biological plausibility review
- design options considered
- decision log
- implementation plan and current status
- benchmark and test plan
- execution registry for all non-trivial runs
- observability tools and inspection plan
- current blockers and open questions
- exit criteria and failure criteria

Use the template in [docs/phase-document-template.md](phase-document-template.md).

## Investigation Implementation Checklist

Before starting implementation work on any new investigation, verify the following.

- a dedicated investigation document exists
- the track index links to that document
- the document contains a benchmark list and explicit tests
- the document contains launch, monitor, and recovery commands for any planned
  non-trivial runs
- the document includes a biological plausibility section
- the document lists whether deep learning, HTM, SDRs, or other helper methods
  are being considered and why
- the document names at least one required observability tool
- the run plan respects the default resource budget unless explicitly justified
- success and failure conditions are written before implementation begins
