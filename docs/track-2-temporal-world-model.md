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

Updated 2026-03-20 with extended YCB evidence boundary analysis.

T2.1-T2.2 infrastructure wired (2026-03-20). HippocampalModule is connected to
the main matching loop via `_dispatch_context_signals()`. Episodic memory persists
across episodes. 31 unit tests pass (4 new for context dispatch).

**Importance elevated**: Extended YCB results prove that ~30% of object categories
(ball, tool, can) are irreducible by shape matching alone (15.7% accuracy).
The ball→fruit confusion (0% — all balls classified as fruit) is the primary
canary metric for this track. Layer 2 (behavioral concepts) is the ONLY path
to resolving these confusions. Track 2 is now the highest-leverage research
direction for advancing beyond the Layer 1 ceiling.

## Restriction Addressed

**Restriction 3: Each episode is independent.**

Monty processes one recognition episode at a time with no memory across episodes.
Graph memory persists object models, but there is no event memory — no record of
what happened, what action was taken, or what the outcome was. Without cross-episode
memory, the system cannot learn behavioral associations, social patterns, or
accumulated knowledge.

## Milestone Tracker

| ID | Milestone | Status |
|---|---|---|
| T2.1 | Cross-episode event memory | **Infrastructure wired** |
| T2.2 | State discrimination (same object, two states) | **Infrastructure wired** |
| T2.3 | Temporal edges in graphs | Not started |
| T2.4 | Action-effect prediction | Not started |
| T2.5 | Behavior recognition | Not started |
| T2.6 | Manipulation via prediction | Not started |

## Implementation (2026-03-20)

### Context Dispatch Wiring

Added `_dispatch_context_signals()` to `MontyForGraphMatching._step_learning_modules()`.
After all LMs have taken their matching/exploratory step, any LM that implements
`get_context_signal()` (currently only `HippocampalModule`) has its signal broadcast
to all other LMs via `receive_context()`.

**Files modified:**
- `src/tbp/monty/frameworks/models/graph_matching.py`: Added `_dispatch_context_signals()`
  method, called at end of `_step_learning_modules()`.
- `src/tbp/monty/frameworks/models/evidence_matching/learning_module.py`: Added
  `receive_context()` to `EvidenceGraphLM`, stores signal in `_hippocampal_context`.

### HippocampalModule Capabilities (already implemented)

The HippocampalModule provides:

- **Episodic buffer** (`_record_episode_step()`): Records timestamped
  (active_ids, locations, step_number) tuples per step. Bounded deque of
  episodes with configurable `max_episodes`.
- **Relational graph** (`_update_relational_graph()`): Typed edges between
  concepts — cooccurrence, spatial (displacement vectors), and temporal (lag-1
  sequence). Uses networkx when available, falls back to dict-based storage.
- **Association matrix** (`_rebuild_association_matrix()`): Cosine-normalized
  co-occurrence matrix tracking which concepts appear together.
- **Context signal** (`get_context_signal()`): Returns context_vector,
  active_concepts, association_strengths, episode_count. Broadcast to all LMs.
- **State dict** serialization for persistence across sessions.

### What's Wired

- HPC `get_context_signal()` → `MontyForGraphMatching._dispatch_context_signals()` →
  `EvidenceGraphLM.receive_context()` → `_hippocampal_context` dict stored
- Episodic memory persists across episodes (verified in tests)
- Association matrix accumulates across episodes (verified in tests)
- Temporal edges track concept transitions (verified in tests)

### What's Still Needed

1. **End-to-end Habitat test**: Create an environment with the same object in two
   states (e.g., upright/inverted cup). Run two training episodes, then test whether
   the HPC's episodic memory + relational graph can discriminate the states.
2. **Context → evidence modulation**: EvidenceGraphLM currently stores the context
   signal but doesn't use it. The next step is to use `_hippocampal_context` to
   bias evidence accumulation (e.g., "you've seen this pattern before, expect cups").
3. **HPC as a full LM in Hydra config**: Create a config that includes the HPC in
   the `learning_module_configs` with appropriate `lm_to_lm_matrix` wiring.

## First Investigation Plan

**T2.1-T2.2: Cross-episode memory + state discrimination.**

Two test cases, in order of difficulty:

### Test 1: State discrimination (original plan)
A cup upright vs a cup inverted. Train on both states (two separate episodes).
Test whether the system can (a) recognize it's the same object and (b)
discriminate which state it's in.

### Test 2: Ball→fruit canary (Layer 2 gate)
The extended YCB ball→fruit confusion (0% accuracy, evidence margin 1.3) is
the primary canary for Layer 2 capability. After training on balls and fruits
with their standard shape models, add interaction episodes encoding behavioral
properties (e.g., "bounced" vs "squished"). Test whether episodic memory
resolves the confusion that is irreducible by geometry alone.

**Why this is the right test**: A golf ball and a strawberry have nearly
identical surface geometry (~40mm smooth sphere). No amount of visual features
will reliably distinguish them. But a child knows the difference because one
bounces and one doesn't. This test directly measures whether the system can
form Layer 2 (behavioral) concepts.

**Success criterion**: Ball→fruit confusion drops from 0% to >50% category
accuracy when interaction history is available.

### Requirements (both tests)
1. An environment where objects can have states or interaction outcomes
2. The HippocampalModule wired to store episodes after each training episode
   **[DONE — wiring implemented]**
3. A query mechanism: "have I seen this object before? in what state?"
   **[PARTIAL — `recall_episode()` and `recall_associations()` exist but not
   integrated into matching]**
4. Context → evidence modulation: EvidenceGraphLM must use `_hippocampal_context`
   to bias evidence accumulation

Investigation document: pending creation.

## Extended YCB Evidence (2026-03-20)

The evidence boundary from the extended YCB benchmark directly motivates
Track 2. Categories defined by function rather than shape:

| Category | Accuracy | Confused as | Why |
|---|---|---|---|
| Ball | 0% | Fruit | Both spherical; geometry identical |
| Tool | 25% | Utensil | Screwdrivers ≈ spatulas geometrically |
| Can | 21.4% | Box | Potted meat can is rectangular |

These confusions are irreducible by Layer 1 (shape) mechanisms. They require
Layer 2 (behavioral) information: what happens when you interact with the
object. This is exactly what Track 2's cross-episode memory is designed to
provide.

**Benchmark**: `extended_ycb_eval_holdout.yaml`
**Results**: `~/tbp/results/monty/projects/phase2_review_runs/extended_ycb_eval_holdout/`

## Concept Layers Served

- **Layer 1→2 boundary** (ball vs fruit, tool vs utensil): requires T2.1
  (event memory) + interaction episodes. The extended YCB quantifies this
  boundary at 15.7% accuracy for shape-incongruent categories.
- **Layer 2** (behavioral concepts: dangerous, useful, heavy, food): requires
  T2.1 (event memory) + a valence signal + action-effect binding
- **Layer 3** (social concepts: friend, trust): requires T2.5 (behavior
  recognition) applied to agents rather than objects
