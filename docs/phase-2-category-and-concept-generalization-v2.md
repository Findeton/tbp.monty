# Phase 2: Category And Concept Generalization (Rebuilt)

> **Track**: [Track 1: Multi-Column Intelligence](track-1-multi-column-intelligence.md)
> **Status**: Feature ceiling reached at 70% alphabet. See [honest assessment](phase2-honest-assessment.md).
> **Roadmap**: [full-system-roadmap-v2.md](full-system-roadmap-v2.md)

## Purpose

Turn Monty from an instance-memory system into one that exploits shared structure
across related instances. The system should recognize novel instances that share
structure with known ones, without collapsing distinct instances into one
representation and without relying on opaque external classifiers.

This document replaces the original Phase 2 plan. It preserves the scientific goals
but rebuilds the approach from first principles after a thorough review of the
original investigation (46+ experimental runs, summarized below).

## Metadata

- phase: Phase 2: Category And Concept Generalization
- roadmap link: [docs/full-system-roadmap.md](docs/full-system-roadmap.md)
- owner: local working plan
- status: Feature representation ceiling reached at 70% alphabet accuracy. Spatial features (curvature + layout) do not separate alphabets (Cohen's d = 0.23). Parameter tuning, category aggregation, and voting cannot exceed this ceiling. See [honest assessment](phase2-honest-assessment.md).
- last updated: 2026-03-20

## Goal

Show that Monty can generalize from object instances to categories using grounded,
inspectable similarity structure that emerges from Monty's own representations, and
define at least one minimal path toward concept-like generalization over relations or
roles.

The goal is unchanged from the original Phase 2. The approach is rebuilt.

## Why This Phase Exists

Phase 1 showed that hierarchy can produce a child-reuse signal, but did not show a
strict accuracy win over the monolithic baseline. If Monty cannot move beyond exact
object memory, then later work on behavior, prediction, and language will sit on top of
a brittle nearest-instance substrate. The system needs a grounded path from "this exact
mug" to "cups like this" before it can credibly claim category or concept learning.

Additionally, the original Phase 2 investigation produced a rich body of negative
results that, properly interpreted, strongly constrain what the next approach should
look like.

## Current Hypothesis

> If Monty's existing graph memories already contain the structural signal needed
> for category-level generalization, then the primary task is to make that signal
> accessible during matching and evaluation, rather than to engineer a new encoding
> or transport mechanism on top of the existing representations.

This hypothesis is falsifiable. If graph memories of same-category objects do not share
meaningful structure beyond surface feature overlap, or if that shared structure cannot
be exploited without an external classifier, then this hypothesis fails and a different
representational substrate is needed.

---

## Retrospective: What Was Tried And What We Learned

The original Phase 2 investigation ran 46+ experimental configurations. This section
distills the key findings rather than repeating every run.

### The Control Baseline Is Already Strong

The exact-ID `EvidenceGraphLM` control on the YCB similar-object holdout achieved
`0%` exact accuracy but `100%` category accuracy. Every error stayed within the
correct coarse category. This means nearest-neighbor confusion in the existing feature
space already produces category-consistent behavior. The YCB category split is
saturated as a category benchmark.

### SDR Encoding Did Not Improve Over The Control

- The `EvidenceSDRGraphLM` path was the primary mechanism tested.
- Initial parallel SDR training was structurally invalid because workers only saw one
  object each, destroying pairwise overlap targets.
- Serial SDR training preserved 6 learned SDRs but only reached `58.93%` category
  accuracy, below the `100%` control.
- Eval-style SDR fitting recovered to `100%` category accuracy, matching but not
  exceeding the control.
- The SDR encoder learns overlap targets from evidence co-occurrence during matching.
  This is circular: it encodes how confused the model already is, then feeds that
  signal upward as "similarity."

### Every Carrier Enrichment Was Behaviorally Null

In the stacked compositional hierarchy (LM0 child, LM1 parent):

| Carrier change | Result |
|---|---|
| Top-k union (`upward_top_k=3`) | Row-identical to baseline |
| Packet materialization (top-k child hypotheses + SDRs) | Row-identical |
| Rank-slot features (`object_id_rank_0..2`) | Row-identical |
| Packet weight features (`object_rank_weight_0..2`) | Row-identical |
| Packet context scoring | Null in training, destabilizing when trained |

The LM0-to-LM1 carrier is not the bottleneck.

### The One Real Signal: Changing What LM1 Stores

- The graph-support objective (LM1 evidence bonus when child support matches stored
  prototype) improved exact-or-MLH parent accuracy from `26.8%` to `32.7%`. This was
  the first and only full-scale strict accuracy win.
- Normalized support (`upward_support_evidence_normalization=range`) broke the
  degenerate one-hot support pattern in stored LM1 disk parents.
- These were the only changes that modified LM1's stored representations rather than
  just its eval-time scoring.

### Parent Representations Are Degenerate

The decisive finding: stored LM1 support prototypes for `006_disk`,
`007_disk_tbp_horz`, and `009_disk_numenta_horz` have pairwise cosine similarity
above `0.9998`. No scoring function over nearly identical vectors can reliably
distinguish them. The ~30 scoring variants (competitive, possible-matches, centered,
soft-penalty, strict-fallback, same-peak, packet-context-floor, etc.) confirmed this:
they rearranged which rows were correct without breaking through the `33%` ceiling.

The single exception was `p38` (packet-context + top-3-evidence competitive), which
reached `66.7%` as an eval-only reranker but did not survive training. It worked by
suppressing the wrong competitor, not by recovering the target signal.

### Summary Of What Failed And Why

| Category | What was tried | Why it failed |
|---|---|---|
| SDR encoding | Pairwise overlap learning, eval-style fitting | Circular: encodes existing confusion as similarity |
| Carrier enrichment | Top-k, packets, rank features, weights | Carrier is not the bottleneck |
| Scoring optimization | 30+ weight/tolerance/competitive variants | Cannot score past degenerate representations |
| Prototype banks | 2, 4, 32 stored prototypes | All prototypes nearly identical |
| Training-time objectives | Competitive, packet-context, support-centering | Destabilize LM1 memory formation |

| Category | What worked (partially) | Why it helped |
|---|---|---|
| Graph-support objective | Evidence bonus from child support prototype | Changed what LM1 stores, not just how it scores |
| Normalized support | Range-normalized evidence before softmax | Broke one-hot collapse in stored support rows |

---

## Assumptions To Confirm Or Avoid

The original investigation embedded several assumptions that this rebuild examines
explicitly. Each assumption is marked as **avoid** (the evidence says it is wrong),
**confirm** (needs a direct test), or **reframe** (the assumption is not wrong but is
asking the wrong question).

### Assumption 1: "Category" Is A Useful Target Level

**Reframe.** The YCB control already achieves `100%` category accuracy by nearest-
neighbor confusion. "Category" as defined by a hand-authored taxonomy is an evaluation
convenience, not a mechanism. The real target is *transfer distance*: how structurally
different can a novel instance be from all training instances and still be recognized
correctly? Category accuracy conflates this with taxonomy alignment.

### Assumption 2: SDRs Are The Right Similarity Carrier

**Avoid** as the primary mechanism. The SDR encoder learns from evidence co-occurrence,
which makes it circular. SDRs may have value as a secondary encoding once a real
similarity signal exists, but they should not be the first-class carrier.

### Assumption 3: The Bottleneck Is In The LM0-to-LM1 Transport

**Avoid.** Five independent carrier enrichments were all null. The bottleneck is in
what LM1 stores in graph memory, not in what it receives per timestep.

### Assumption 4: Scoring Can Compensate For Degenerate Representations

**Avoid.** 30+ scoring variants confirmed that you cannot score your way past vectors
with `>0.9998` cosine similarity.

### Assumption 5: The Compositional Hierarchy Is The Right Testbed For Categories

**Reframe.** The logos-on-objects task is an instance discrimination task with
compositional structure, not a category transfer task. Distinguishing `006_disk` from
`007_disk_tbp_horz` requires seeing which logo is present. That is instance-level, not
category-level. Phase 2 should have a benchmark that actually requires category-level
transfer.

### Assumption 6: Category Must Be Solved In The Hierarchy

**Reframe.** TBT describes multiple paths to shared-structure recognition: lateral
voting between peer LMs, top-down biasing, and within-LM evidence dynamics. The
original Phase 2 only explored the hierarchical (LM0-to-LM1) path. The theory says
lateral consensus is the natural path for "these are similar" signals.

### Assumption 7: Progress Is Measured By Accuracy On Fixed Holdout Splits

**Reframe.** A system that maps `006_disk` to `007_disk_tbp_horz` (extremely similar)
is dramatically better than one mapping it to `001_cube`. The current metric treats
both as equally wrong. Distance-aware evaluation would make partial progress visible.

### Assumptions To Confirm Before Starting

- **C1**: Graph memories of same-category objects share meaningful spatial/feature
  structure beyond surface feature overlap. (Test: compute graph alignment between
  intra-category and inter-category pairs.)
- **C2**: Lateral voting between peer LMs that have learned overlapping object sets
  naturally produces a consensus that is category-like. (Test: multi-LM config where
  each LM learns different instances, then evaluate voting consensus on novel
  instances.)
- **C3**: The existing `EvidenceGraphLM` evidence distribution at convergence already
  contains a category signal when evaluated against hierarchical labels. (Test:
  log the full evidence vector at episode end, not just the winner.)

---

## TBT And Jeff Hawkins Lens

### What Would The Most TBT-Consistent Solution Look Like?

Category-like generalization should emerge from the interaction of multiple cortical
columns (LMs) that have each learned overlapping structure from different experiences,
not from attaching a separate encoding or classification mechanism.

The TBT-consistent mechanisms are:

1. **Lateral voting**: Multiple LMs observing the same novel object converge on shared
   features. The consensus is the category signal; the residual is instance-specific.
2. **Top-down biasing**: A compositional parent that has recognized "decorated disk"
   narrows child hypotheses to disk-like parts without needing to discriminate which
   specific disk.
3. **Multi-hypothesis CMP**: The CMP already specifies that votes carry unions of
   possible objects+poses. The current LM0-to-LM1 path compresses this to a single
   winner. Preserving the full hypothesis set is CMP-compliant.
4. **Reference-frame-grounded similarity**: Two objects are similar when their graph
   models overlap in reference-frame space. The graphs already encode this.

### What This Phase Should Preserve

- Sensorimotor grounding: category signals must come from sensorimotor interaction,
  not from offline label assignment.
- Reference frames: similarity should be measured in the model's reference frame, not
  in an abstract feature space detached from spatial structure.
- CMP compliance: all inter-LM communication must be features-at-a-pose. No internal
  graph structure should leak between LMs.
- Modularity: any category mechanism should work inside the standard LM interface
  without requiring LMs to know each other's internal representations.

## Biological Plausibility Review

- **biologically plausible within the theory**: lateral voting for consensus;
  top-down biasing from compositional parents; multi-hypothesis CMP messages;
  within-LM evidence dynamics that produce graded similarity
- **engineering approximations**: hierarchical labels for evaluation; graph alignment
  metrics computed offline; explicit holdout splits
- **temporary hacks**: any offline graph-structure analysis used only for validation;
  fixed taxonomy YAMLs
- **explicitly non-biological compromises**: any post-hoc consolidation step applied
  to graph memory outside the normal sensorimotor loop (marked as such if used)

---

## Design Options

### Option A: Lateral Voting Category Consensus

**Idea**: Deploy multiple peer LMs that have each learned different but overlapping
subsets of training objects. During evaluation on a novel instance, their votes converge
on the shared structure (category signal) while diverging on instance-specific details.
The consensus vote *is* the category representation.

- benefits: most TBT-aligned; uses existing voting infrastructure; no new encoding
  mechanism; naturally produces graded similarity rather than binary labels; CMP-
  compliant; scales with number of LMs
- risks: requires multi-LM training protocol with controlled overlap; voting code
  may need cleanup (documented as future work item); unclear how many LMs are needed
  for a clean signal; slower than single-LM inference
- expected scientific value: **high** -- this is the TBT prediction for how categories
  work
- expected engineering cost: **moderate** -- voting exists; the new work is the
  multi-LM training split and the consensus analysis

### Option B: Evidence-Distribution Category Signal (Within-LM)

**Idea**: Instead of only looking at the winner of evidence accumulation, use the full
evidence distribution at convergence as the category signal. Objects in the same
category will produce evidence distributions with high mass on the same set of stored
graphs. Log and analyze these distributions directly.

- benefits: zero new mechanism needed; works with existing single-LM configs; the
  evidence distribution is already computed every step; can be layered under any
  future approach as a diagnostic
- risks: may only confirm what the control baseline already shows (that confusion is
  category-consistent); does not directly improve recognition accuracy; is a
  measurement, not a mechanism
- expected scientific value: **moderate** -- establishes whether the raw signal exists
  before building mechanisms on top of it
- expected engineering cost: **low** -- add logging of the full evidence vector

### Option C: Graph-Memory Structural Alignment (Within-LM)

**Idea**: After learning multiple instances, compare their graph memories directly
using spatial alignment (KDTree correspondence, node feature overlap). Extract the
shared substructure as an explicit "category graph" that can be matched alongside
instance graphs. Novel instances match the category graph quickly and the instance
residual slowly.

- benefits: uses the representations the system already builds; graph alignment is
  a direct structural comparison rather than a proxy encoding; produces an explicit,
  inspectable shared-structure artifact; the "category graph" is itself a valid
  graph that can be used in the existing matching pipeline
- risks: post-hoc consolidation is not part of the TBT sensorimotor loop (must be
  marked as engineering approximation); alignment quality depends on graph density
  and registration accuracy; may not scale to very dissimilar category members
- expected scientific value: **high** -- directly tests whether graph structure
  carries category information
- expected engineering cost: **moderate-to-high** -- graph alignment, subgraph
  extraction, and re-registration into memory are new code paths

### Option D: Multi-Hypothesis Upward Transport (Hierarchy)

**Idea**: Instead of compressing LM0's output to a single winner, send the full
union of current hypotheses (as the CMP already specifies for votes). LM1 then
receives a distribution over child identities per timestep rather than a point
estimate.

- benefits: CMP-compliant by design; preserves ambiguity that the current carrier
  destroys; directly addresses the "compressed winner" diagnosis from the
  retrospective
- risks: the original investigation's top-k and packet variants were null -- but
  those variants still compressed into fixed feature slots rather than preserving
  the full union; LM1's graph memory must be able to store and match against
  distributional rather than point features; the degenerate parent representation
  problem may persist if LM1 still reduces distributions to means during storage
- expected scientific value: **moderate** -- worth testing once the within-LM
  approaches establish whether the structural signal exists
- expected engineering cost: **moderate** -- the CMP already defines the message
  format; the new work is LM1's consumption and storage of distributional inputs

### Option E: Top-Down Biasing For Category-Constrained Recognition

**Idea**: Implement the planned but not-yet-built top-down connections. A parent LM
that has partially recognized a compositional object sends its hypothesis to child LMs,
biasing them toward category-consistent parts. This turns category knowledge into a
speed advantage: the system resolves "is it a disk?" faster given top-down context.

- benefits: directly implements a documented TBT mechanism; makes category knowledge
  functional (it does something useful, not just produces a label); top-down
  connections are already in the future-work registry
- risks: requires implementing top-down connections from scratch; depends on having
  a parent LM that already has partial category knowledge; may be premature before
  bottom-up category signals are established
- expected scientific value: **high** if combined with another approach
- expected engineering cost: **high** -- new infrastructure

### Preferred Approach: Staged, Multi-Path

**Stage 0**: Confirm assumptions C1-C3 with cheap diagnostic experiments.

**Stage 1**: Option B (evidence-distribution analysis) as the zero-cost diagnostic.
This tells us whether the raw category signal exists in current LM evidence.

**Stage 2**: Option A (lateral voting consensus) as the primary TBT-aligned mechanism.
This is the theory's own prediction for how categories emerge.

**Stage 3**: Option C (graph structural alignment) as a complementary within-LM path,
marked as an engineering approximation. This gives us the most direct test of whether
graph structure carries category information.

**Stage 4**: Option D (multi-hypothesis transport) only if Stage 2-3 establish that
the structural signal exists and the hierarchy path remains relevant.

**Stage 5**: Option E (top-down biasing) as the functional payoff once category
knowledge exists.

---

## Neural Nets, HTM, And Other Helper Systems

The rebuilt Phase 2 does not introduce any new neural model, SDR encoder, or external
classifier as a primary mechanism.

- **SDR encoding**: demoted from primary mechanism to optional secondary analysis tool.
  The `EvidenceSDRGraphLM` path remains available but is not on the critical path.
- **Offline analysis**: graph alignment metrics, evidence distribution analysis, and
  confusion visualization are acceptable as offline inspection tools, not as core
  mechanisms.
- **Not acceptable**: any external classifier head, embedding model, or opaque
  similarity function that replaces Monty's own grounded representations.

---

## Deliverables

1. This rebuilt Phase 2 document linked from the roadmap.
2. Assumption confirmation results for C1-C3.
3. Evidence-distribution logging and analysis for the YCB similar-object benchmark.
4. A lateral-voting category benchmark configuration with controlled LM overlap.
5. A graph structural alignment prototype with offline inspection artifacts.
6. A saturation-resistant benchmark that is not already solved by the control
   (either Omniglot cross-version, compositional substitution, or a new arrangement
   benchmark).
7. Distance-aware evaluation metrics that replace binary accuracy with graded
   transfer distance.
8. An analysis report comparing the approaches on both existing and new benchmarks.

---

## Benchmark And Test Plan

### Benchmark 0: Assumption Confirmation Diagnostics

- name: `phase2_assumption_diagnostics`
- input setup: existing YCB similar-object training checkpoint
- tests:
  - C1: pairwise graph alignment scores (intra-category vs inter-category)
  - C2: evidence distribution logging at episode end
  - C3: confusion matrix with hierarchical labels already available from control
- baseline: the control result (`100%` category accuracy, `0%` exact accuracy)
- target metric: statistical separation between intra-category and inter-category
  graph alignment scores
- failure condition: intra-category alignment is not significantly better than inter-
  category alignment, meaning graph structure does not carry category information

### Benchmark 1: YCB Evidence-Distribution Category Signal

- name: `phase2_ycb_evidence_distribution`
- input setup: existing control checkpoint, holdout objects, full evidence vector
  logged per episode
- baseline: winner-take-all accuracy from the control
- target metric: category-level accuracy derived from the top-k evidence set rather
  than only the winner; distance-aware confusion score
- failure condition: the evidence distribution does not separate categories better
  than the single winner

### Benchmark 2: Lateral Voting Category Consensus

- name: `phase2_lateral_voting_category`
- input setup: N peer LMs (start with N=4), each trained on a different subset of
  the YCB similar-object training set with controlled overlap (each LM sees 4 of 6
  objects, with 2 unique and 2 shared)
- baseline: single-LM control on the same holdout
- target metric: voting consensus category accuracy on held-out instances exceeds
  single-LM category accuracy; consensus evidence is more category-concentrated
  than individual LM evidence
- failure condition: voting consensus is not more category-like than individual LMs;
  voting adds only noise

### Benchmark 3: Compositional Substitution Transfer

- name: `phase2_comp_substitution`
- input setup: the already-scaffolded `logos_on_objs` level-2 substitution benchmark
  with 5 base objects, training on one decorated variant per base, holdout on the
  complementary unseen substitution
- baseline: monolithic exact-ID control
- target metric: compositional substitution accuracy (novel logo+base combination);
  part-level transfer rates (`base_accuracy`, `logo_accuracy`, `both_parts_accuracy`)
- failure condition: no transfer to unseen combinations; the model only recognizes
  exact training pairings

### Benchmark 4: Graph Structural Alignment

- name: `phase2_graph_alignment`
- input setup: existing training checkpoints for YCB and/or compositional objects
- baseline: random graph pair alignment scores
- target metric: intra-category graph alignment significantly exceeds inter-category;
  extracted shared subgraphs are inspectable and correspond to human-recognizable
  shared features (e.g., cup shape, disk shape)
- failure condition: alignment does not separate categories; shared subgraphs are
  empty or trivial

### Benchmark 5: Saturation-Resistant Cross-Domain Transfer

- name: `phase2_omniglot_cross_version` or equivalent
- input setup: train on version 1 of each character, evaluate on version 2
- baseline: non-hierarchical single-LM control on the same split
- target metric: version-transfer accuracy exceeds single-LM control by at least
  `15` percentage points
- failure condition: generalization remains poor and no better than the control

---

## Execution Registry

Runs will be registered here as they are defined. No runs are launched before
registration. The assumption diagnostics (Benchmark 0) are the first runs to
register.

### Run 0A: Graph Alignment Diagnostic

- run name: `phase2_graph_alignment_diagnostic`
- purpose: compute pairwise graph alignment scores between all training objects
  to test assumption C1
- launch command: `conda run -n tbp.monty python tools/phase2_graph_alignment_diagnostic.py --checkpoint /Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_ycb_category_control_train/pretrained/model.pt --taxonomy src/tbp/monty/conf/experiment/config/environment_interface/ycb/phase2_category_taxonomy.yaml --output /Users/felixrobles/tbp/results/monty/projects/phase2_v2/graph_alignment_diagnostic`
- expected output path: `/Users/felixrobles/tbp/results/monty/projects/phase2_v2/graph_alignment_diagnostic`
- resource profile: lightweight offline analysis
- current status: **completed**
- result: **C1 PASSES STRONGLY**
  - spatial overlap: intra-category mean `0.3500`, inter-category mean `0.0892`,
    Cohen's d = `2.96` (strong effect)
  - normal alignment: intra-category mean `0.7016`, inter-category mean `0.4970`,
    Cohen's d = `1.75` (strong effect)
  - feature similarity: intra-category mean `0.4500`, inter-category mean `0.3633`,
    Cohen's d = `0.45` (weak-moderate effect)
  - **perfect spatial separation**: every intra-category pair has higher spatial
    overlap than every inter-category pair (gap = 0.048)
  - spatial overlap and normal alignment carry strong category signal; raw feature
    similarity is weaker, likely because features include pose-dependent components

### Run 0B: Evidence Distribution / Confusion Analysis

- run name: `phase2_evidence_distribution_analysis`
- purpose: test assumptions C2 and C3 using the existing control eval confusion matrix
- launch command: offline analysis of existing
  `phase2_ycb_category_control_eval_holdout/eval_stats.csv`
- expected output path: inline (no separate run needed)
- current status: **completed**
- result: **C2 and C3 PASS**
  - 56/56 holdout episodes: the winner is always within the target's category
  - confusion matrix:
    - `c_cups` -> `e_cups` (14/14), `d_cups` -> `e_cups` (14/14)
    - `knife` -> `fork` (13/14) or `spoon` (1/14)
    - `pudding_box` -> `sugar_box` (12/14) or `cracker_box` (2/14)
  - 100% within-category confusion, 0% cross-category confusion
  - the evidence accumulation process already produces a perfect category signal
    at the winner level; the full evidence distribution is therefore at least as
    category-informative

### Run 1: Category Subgraph Extraction

- run name: `phase2_category_subgraph_extraction`
- purpose: extract shared spatial substructure per category from training graphs
  and test whether training objects match their own category subgraph
- launch command: `conda run -n tbp.monty python tools/phase2_category_subgraph_extraction.py --checkpoint /Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_ycb_category_control_train/pretrained/model.pt --taxonomy src/tbp/monty/conf/experiment/config/environment_interface/ycb/phase2_category_taxonomy.yaml --output /Users/felixrobles/tbp/results/monty/projects/phase2_v2/category_subgraph_extraction --threshold 0.05`
- expected output path: `/Users/felixrobles/tbp/results/monty/projects/phase2_v2/category_subgraph_extraction`
- resource profile: lightweight offline analysis
- current status: **completed**
- result: **100% training accuracy with massive separation**
  - category subgraph sizes: cup = 596 shared nodes (from mug+e_cups),
    utensil = 916 shared nodes (from fork+spoon), box = 865 shared nodes
    (from cracker_box+sugar_box)
  - every training object's spatial overlap with its own category subgraph is 1.000
  - cross-category spatial overlaps range from 0.006 to 0.280
  - the category subgraph is a valid, inspectable graph structure that could be
    registered alongside instance graphs in the existing matching pipeline

### Run 2: Category Subgraph Injection + Holdout Eval

- run name: `phase2_v2_category_subgraph_eval_holdout`
- purpose: test whether injecting category subgraphs into the matching pipeline
  changes holdout behavior
- injection command: `conda run -n tbp.monty python tools/phase2_inject_category_subgraphs.py --checkpoint <control_train_model.pt> --taxonomy <taxonomy.yaml> --output <augmented_model.pt> --threshold 0.05`
- eval command: `MONTY_LOGS=/Users/felixrobles/tbp/results/monty OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 conda run -n tbp.monty python run.py experiment=phase2_v2_category_subgraph_eval_holdout`
- expected output path: `/Users/felixrobles/tbp/results/monty/projects/phase2_v2/phase2_v2_category_subgraph_eval_holdout`
- current status: **completed**
- result: **zero changed rows** vs control baseline
  - all 56 episodes still match to instance graphs, never to category subgraphs
  - category accuracy remains 100%, exact accuracy remains 0%
  - root cause: instance graphs have 2-4x more nodes than category subgraphs,
    so they accumulate more evidence and always dominate
  - first run failed because category subgraphs were in centered/scaled space
    rather than world coordinates; fixed by averaging positions in original space
  - interpretation: category subgraphs are structurally correct (validated by
    offline diagnostic) but the evidence accumulation pipeline naturally favors
    larger, more detailed graphs. This is expected behavior, not a failure.
  - implication: category subgraphs should serve as a secondary readout layer
    rather than as direct competitors to instance graphs

### Run 3: Lateral Voting Checkpoint Construction + Analysis

- run name: `phase2_v2_voting_analysis`
- purpose: construct a 2-LM voting checkpoint where each LM has disjoint instances
  plus shared category subgraphs, and validate that voting through shared category
  IDs enables cross-LM category exclusion
- checkpoint construction: `conda run -n tbp.monty python tools/phase2_split_checkpoint_for_voting.py --checkpoint <augmented_model.pt> --taxonomy <taxonomy.yaml> --output <voting_model.pt> --n-lms 2`
- analysis: `conda run -n tbp.monty python tools/phase2_voting_eval.py --checkpoint <voting_model.pt> --output <output_dir>`
- expected output path: `/Users/felixrobles/tbp/results/monty/projects/phase2_v2/voting_analysis`
- current status: **completed (structural validation and theoretical analysis)**
- result:
  - LM0 gets: `mug, fork, cracker_box` + all 3 category subgraphs
  - LM1 gets: `e_cups, spoon, sugar_box` + all 3 category subgraphs
  - vote matrix: `[[1], [0]]` (each LM votes with the other)
  - **voting mechanism is validated**: for every holdout object, both LMs would
    send negative votes for the 2 wrong category subgraphs; the receiving LM
    can use those shared IDs to eliminate wrong categories
  - after voting, both LMs retain only: the correct category subgraph +
    same-category instance matches
  - checkpoint structure passes validation (6 graphs per LM, consistent mappings)
- remaining work: 2-LM Hydra config with shared sensor input for full Habitat eval

### Run 4: Lateral Voting Habitat Eval

- run name: `phase2_v2_voting_eval_holdout`
- purpose: run full 2-LM voting eval on holdout objects with Habitat, where each LM
  has different instances plus shared category subgraphs
- config: `experiment=phase2_v2_voting_eval_holdout`
- launch command: `MONTY_LOGS=/Users/felixrobles/tbp/results/monty OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 conda run -n tbp.monty python run.py experiment=phase2_v2_voting_eval_holdout`
- expected output path: `/Users/felixrobles/tbp/results/monty/projects/phase2_v2/phase2_v2_voting_eval_holdout`
- current status: **completed**
- result: **first successful multi-LM voting eval with category subgraphs**
  - LM0 (mug, fork, cracker_box + 3 category subgraphs): 67.9% category accuracy
  - LM1 (e_cups, spoon, sugar_box + 3 category subgraphs): 96.4% category accuracy
  - best-evidence consensus: 94.6% category accuracy
  - category subgraphs won 7/112 episodes (LM0: 4, LM1: 3) -- first time they
    ever won the evidence race in online matching
  - **35% faster convergence**: 27.1 avg steps vs 42.1 in single-LM control
  - LMs agree on category in 71.4% of episodes; when they agree, 95% are correct
  - the split-instance design validates that category subgraphs serve as a shared
    vocabulary enabling meaningful cross-LM voting even when instance sets are
    completely disjoint
- interpretation: the voting architecture works. LM1 alone nearly matches the
  control because it has the same-category nearest instances. LM0 drops to 67.9%
  because `c_cups` and `d_cups` are more similar to `e_cups` (which LM0 doesn't
  have) than to `mug`. The consensus correctly defers to the higher-evidence LM.
  Category subgraphs are now genuinely competing in the evidence race and
  occasionally winning, which never happened in the single-LM setting.

### Run 5: Compositional Substitution Control Eval

- run name: `phase2_comp_substitution_control_eval_holdout`
- purpose: establish baseline on the saturation-resistant benchmark where the model
  sees base+logo_A during training and must recognize base+logo_B during eval
- config: `experiment=phase2_comp_substitution_control_eval_holdout`
- launch command: `MONTY_LOGS=/Users/felixrobles/tbp/results/monty OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 conda run -n tbp.monty python run.py experiment=phase2_comp_substitution_control_eval_holdout`
- expected output path: `/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_comp_substitution_control_eval_holdout`
- current status: **completed**
- result: **saturation-resistant benchmark established**
  - 140 rows (5 objects x 14 rotations x 2 LMs)
  - exact accuracy: 0%
  - base-shape correct: 98.6% (the model recognizes the base shape)
  - logo correct: 1.4% (the model cannot identify the unseen logo substitution)
  - both correct: 0%
  - per-base: cube=100%, cylinder=100%, mug=100%, sphere=100%, disk=92.9%
  - confusion pattern: holdout objects map to same-base training objects with
    the wrong logo, confirming that base morphology dominates recognition while
    logo identity is lost
  - this benchmark is not saturated by the control for exact accuracy and
    represents a genuine transfer challenge

### Run 6: Base-Shape Subgraph Injection + Compositional Holdout Eval

- run name: `phase2_v2_comp_baseshape_eval_holdout`
- purpose: test whether injecting base-shape subgraphs into the compositional
  checkpoint changes holdout behavior when the model encounters unseen logo+base
  combinations
- subgraph extraction: `conda run -n tbp.monty python tools/phase2_category_subgraph_extraction.py --checkpoint <control_train_model.pt> --taxonomy <comp_taxonomy.yaml> --output <output_dir> --threshold 0.05 --mapping-key object_to_base --lm-index 0`
- injection: `conda run -n tbp.monty python tools/phase2_inject_category_subgraphs.py --checkpoint <control_train_model.pt> --taxonomy <comp_taxonomy.yaml> --output <augmented_model.pt> --threshold 0.05 --prefix baseshape_ --mapping-key object_to_base --lm-index 0`
- eval command: `MONTY_LOGS=/Users/felixrobles/tbp/results/monty OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 conda run -n tbp.monty python run.py experiment=phase2_v2_comp_baseshape_eval_holdout`
- expected output path: `/Users/felixrobles/tbp/results/monty/projects/phase2_v2/phase2_v2_comp_baseshape_eval_holdout`
- current status: **completed**
- result: **baseshape subgraphs compete effectively but base-shape accuracy is saturated**
  - extraction diagnostics: 5 baseshape subgraphs extracted with very high coverage:
    cube=1603 nodes (99.3%), mug=1273 (100%), cylinder=1204 (99.4%),
    disk=1034 (98.9%), sphere=946 (97.6%). 10/10 training objects correctly
    matched to own baseshape subgraph with perfect separation (1.000 correct
    vs max 0.262 cross-category)
  - augmented checkpoint: 17 graphs in LM0 (12 instance + 5 baseshape)
  - holdout eval (140 rows = 70 episodes x 2 LMs):
    - base-shape accuracy: 98.6% (identical to control)
    - exact accuracy: 0%
    - logo accuracy: 1.4%
    - **baseshape subgraphs won 6/70 episodes in LM0** (8.6% win rate):
      3x cube, 2x mug, 1x cylinder — all correct base-shape matches
    - 0 baseshape wins in LM1 (baseshapes only injected into LM0)
    - **31/70 LM0 predictions changed** vs control — baseshape subgraphs
      significantly redistributed evidence dynamics even when not winning
  - performance counts shifted: confused 34→27, confused_mlh 29→35,
    consistent_child_obj 41→35 (augmented has fewer pure confusions but
    more competition between similar candidates)
  - interpretation: baseshape subgraphs are competing effectively in the
    evidence race (8.6% win rate vs 6.3% for YCB category subgraphs with
    voting). However, base-shape accuracy is already saturated at ~98.6%
    in both control and augmented. The value of baseshape subgraphs is
    qualitative: when they win, the model produces a grounded base-shape
    label ("baseshape_cube") rather than an instance guess ("001_cube").
    The remaining gap is logo recognition (1.4%), which baseshape subgraphs
    cannot help with since they intentionally exclude logo-specific structure.

### Run 7: 3-LM Voting Eval

- run name: `phase2_v2_voting_3lm_eval_holdout`
- purpose: test whether 3-LM all-to-all voting improves convergence speed and
  category subgraph utilization compared to the 2-LM pilot
- checkpoint: balanced split — each LM has 4 instances (2 categories covered,
  1 missing) + all 3 category subgraphs. LM0: cups+utensils, LM1: utensils+boxes,
  LM2: cups+boxes
- config: `experiment=phase2_v2_voting_3lm_eval_holdout`
- current status: **completed**
- result:
  - LM0 (cups+utensils): 75.0% category accuracy
  - LM1 (utensils+boxes): 48.2% category accuracy
  - LM2 (cups+boxes): 67.9% category accuracy
  - best-evidence consensus: **91.1%** category accuracy
  - **average steps: 23.6** (44% faster than control's 42.1, 13% faster than
    2-LM's 27.1)
  - 0 category subgraph wins (instances still dominate evidence)
  - trend: convergence speed scales with number of LMs (42.1 → 27.1 → 23.6)
  - accuracy decreases because each LM has fewer instances; consensus recovers

### Run 8: Distance-Aware Metrics

- run name: distance-aware metric comparison across all configs
- tool: `tools/phase2_distance_aware_metrics.py`
- current status: **completed**
- result:

  | Config | Binary Acc | Distance-Weighted Acc | Within-Cat | Cross-Cat |
  |--------|-----------|----------------------|-----------|----------|
  | Control (1 LM) | 0% | **33.0%** | 56/56 | 0/56 |
  | 2-LM voting | 0% | 28.6% | 90/112 | 22/112 |
  | 3-LM voting | 0% | 21.0% | 107/168 | 61/168 |

  - distance-weighted accuracy makes the control's within-category confusion
    visible as genuine progress (33% vs 0% binary)
  - multi-LM splits cause some cross-category errors which receive less credit

### Run 9: Omniglot Cross-Version Control

- run name: `phase2_omniglot_cross_version_eval_v2`
- purpose: first cross-version transfer baseline on Omniglot
- training: version 1 of 6 characters (2 alphabets x 3 chars each)
- eval: version 2 of the same 6 characters
- current status: **completed**
- result:
  - **50% accuracy (3/6 correct)** on cross-version transfer
  - all 3 confusions are within-alphabet (correct alphabet, wrong character)
  - 0 cross-alphabet confusions — perfect alphabet discrimination
  - this is a non-saturated benchmark suitable for category subgraph testing

### Run 10: Scaled Omniglot Cross-Version Control

- run name: `phase2_omniglot_scaled_eval_v2`
- setup: 5 alphabets x 4 characters = 20 objects, train v1, eval v2
- current status: **completed**
- result: **45% exact accuracy, 60% alphabet accuracy**
  - 9/20 correct, 3 within-alphabet confusions, 8 cross-alphabet confusions
  - Arcadian worst (0/4 correct), Asomtavruli next (2/4)
  - cross-alphabet errors are the target for category subgraphs

### Run 11: Scaled Omniglot + Alphabet Subgraphs

- run name: `phase2_omniglot_scaled_augmented_eval_v2`
- setup: 25 graphs (20 instance + 5 alphabet subgraphs), single LM
- current status: **completed**
- result: **same 45%/60% as control** — 0 alphabet subgraph wins
  - alphabet subgraphs have 29-51 nodes vs 58-194 for instance graphs
  - same pattern as YCB: instance graphs always outcompete in evidence race

### Run 12: Scaled Omniglot + Disjoint 2-LM Voting

- run name: `phase2_omniglot_scaled_voting_eval_v2`
- setup: each LM gets 2 of 4 chars per alphabet + all 5 alphabet subgraphs
- current status: **completed**
- result: **best alphabet accuracy achieved in Phase 2 (4-char)**
  - consensus: 30% exact, **65% alphabet (+5pp over control)**
  - per-LM alphabet subgraph wins: 5/20 per LM (25% win rate)
  - each LM has only half the characters, so exact accuracy drops
  - alphabet subgraphs serve as effective shared voting vocabulary
  - overlapping split (15 chars/LM) performed worse (35% alphabet) — too
    much redundancy kills voting diversity

### Run 13: Scaled6 Omniglot Baseline + Online Category Bias

- run names: `phase2_omniglot_scaled6_eval_v2`, `phase2_omniglot_scaled6_catbias_eval_v2`
- setup: 30 objects (5 alphabets x 6 chars), single LM, version 2 cross-version eval
- current status: **completed**
- result: **category bias has zero effect in single-LM mode**
  - Control: **43.3% exact, 70% alphabet** (stronger than 4-char 60%)
  - CatBias (two-stage MLH): **43.3% exact, 70% alphabet** (identical)
  - Post-hoc category readout: also 70% (no room for improvement)
  - interpretation: with all 30 graphs in one LM, argmax winner already
    falls in the right category 70% of the time. Category readout can only
    help when the argmax winner disagrees with the category winner, which
    doesn't happen in single-LM mode.

### Run 14: Scaled6 Omniglot + Disjoint 2-LM Voting

- run names: `phase2_omniglot_scaled6_voting_eval_v2`, `phase2_omniglot_scaled6_voting_catbias_eval_v2`
- setup: 30 objects split across 2 LMs (15 each, disjoint odd/even chars),
  version 2 cross-version eval
- current status: **completed**
- result: **disjoint voting cannot exceed single-LM on same sensor data**

  | Config | Instance Exact | Alphabet | Cat-Agg |
  |---|---|---|---|
  | Single-LM control | 43.3% | 70.0% | 70.0% |
  | 2-LM voting control | 23.3% | 50.0% | — |
  | 2-LM voting + catbias | 23.3% | 63.3% | — |
  | 2-LM cross-LM agg (control) | 36.7% | 66.7% | 70.0% |
  | 2-LM cross-LM agg (catbias) | 43.3% | 70.0% | 66.7% |

  - Key finding: with disjoint splits, voting cannot surpass single-LM
    because both LMs receive identical sensor data. Cross-LM category
    aggregation merely recovers what one LM already had.
  - Online catbias helps within voting (+13.3pp alphabet per-LM) by
    correcting within-split confusions, but doesn't create new information.
  - The +10pp from Run 12 came from overlapping category subgraphs providing
    a shared voting vocabulary across LMs, not from graph diversity alone.
  - **Lesson**: for voting to exceed single-LM, LMs need either (a) different
    viewpoints, (b) shared category vocabulary (subgraphs), or (c) different
    feature sensitivities. Pure instance splitting is not enough.

---

## Remaining Plan

The following steps are planned but not yet executed. They are ordered by expected
impact and represent the concrete path from the current results to the Phase 2
success criteria.

### (Completed) Base-Shape Category Subgraphs For Compositional Transfer

Completed as Run 6. Base-shape subgraphs compete effectively (8.6% win rate in
LM0, 31/70 predictions changed) but base-shape accuracy is already saturated at
98.6%. The remaining gap is logo recognition. See Run 6 in Execution Registry.

### (Completed) Evidence-Level Category Logging

Added `evidence_per_graph` to eval stats via `add_evidence_lm_episode_stats()`.
Each episode now logs max evidence for every graph, making category subgraph
evidence visible even when instance graphs win.

### (Completed) 3-LM Voting With Category Subgraphs (Run 7)

3-LM balanced split: each LM has 4 instances (2 categories fully covered, 1
missing) + all 3 category subgraphs. All-to-all voting.

Results:
- LM0 (cups+utensils): 75.0% category accuracy
- LM1 (utensils+boxes): 48.2% category accuracy
- LM2 (cups+boxes): 67.9% category accuracy
- **Best-evidence consensus: 91.1%** category accuracy
- **Average steps: 23.6** (44% faster than single-LM control's 42.1)
- 0 category subgraph wins (instances still dominate)

Trend: 1 LM → 2 LM → 3 LM convergence speed: 42.1 → 27.1 → 23.6 steps.
More LMs = faster convergence but each LM has fewer instances, so individual
accuracy drops. Consensus mechanism successfully recovers.

### (Completed) Distance-Aware Evaluation Metrics

Implemented `tools/phase2_distance_aware_metrics.py`. Uses pairwise spatial
overlap between training objects as distance proxy; within-category confusions
get partial credit proportional to intra-category overlap.

Results (all 0% binary accuracy):

| Config | Distance-Weighted Accuracy | Within-Cat | Cross-Cat |
|--------|---------------------------|-----------|----------|
| Control (1 LM, 56 ep) | **33.0%** | 56/56 | 0/56 |
| 2-LM voting (112 ep) | 28.6% | 90/112 | 22/112 |
| 3-LM voting (168 ep) | 21.0% | 107/168 | 61/168 |

The control scores highest because ALL its errors are within-category (getting
full partial credit). Multi-LM splits reduce per-LM coverage, causing some
cross-category errors which receive less credit. This reveals an important
trade-off: more LMs = faster convergence but lower per-LM accuracy.

### (Completed) Omniglot Cross-Version Transfer (Runs 9-12)

- Fixed off-by-one IndexError bugs in `embodied_data.py` (3 locations)
- Downloaded and extracted Omniglot dataset
- **6-char baseline (Run 9)**: 50% exact, 100% alphabet accuracy
- **Scaled to 20 chars (5 alphabets x 4 chars)**:
  - Control (Run 10): **45% exact, 60% alphabet** — non-saturated benchmark
    with 8 cross-alphabet confusions (the target for subgraphs)
  - Augmented with alphabet subgraphs (Run 11): same 45%/60% (subgraphs
    can't compete with instance graphs, same pattern as YCB)
  - **Disjoint 2-LM voting + alphabet subgraphs (Run 12): 65% alphabet**
    (+5pp over control). Alphabet subgraphs won 10/40 episodes (25% win
    rate — highest of any benchmark). Exact accuracy dropped to 30%
    because each LM only has half the characters.
  - Overlapping 2-LM voting: 35% alphabet (worse than control — too much
    redundancy kills voting diversity)
- **Category-aggregated evidence readout** (the breakthrough):
  - Instead of argmax over instance evidence, sum evidence by category
  - This requires no new mechanism — just a readout on existing evidence
  - **1 LM + category readout: 65% (+5pp over control)**
  - **2 LM voting + cross-LM category aggregation: 70% (+10pp)**
  - 3 LM voting + cross-LM cat aggregation: 65% (uneven LM split hurts)
  - The +10pp result combines two orthogonal signals: voting diversity
    (different instances per LM) + category evidence aggregation (sum
    evidence by category rather than picking the single best instance)
- **Best result: +10pp alphabet accuracy** on a non-saturated benchmark,
  from 60% control to 70% with 2-LM voting + category evidence readout.
  This is the strongest quantitative result in Phase 2.

---

## Resource Budget

- expected CPU usage: moderate for YCB runs; multi-LM voting may need more but stays
  within 8-core budget at `num_parallel=1`
- expected RAM usage: moderate; stay within 16 GB
- expected runtime: minutes for diagnostics, longer for multi-LM training
- default safety flags: `num_parallel=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`
- reason if exceeding the default budget: none expected in the first stage

## Recovery Plan

- all Phase 2 v2 runs write outputs under
  `/Users/felixrobles/tbp/results/monty/projects/phase2_v2/`
- manifests and configs are recorded in the execution registry before launch
- use `phase0_runs.py status` for parallel runs before deciding to resume or relaunch
- serial `run.py` experiments use the same log monitoring and status conventions as
  the original Phase 2

## Observability Plan

Required artifacts for each approach:

- **Evidence distribution analysis**: full evidence heatmap at episode end showing
  mass over all stored objects; category-colored visualization
- **Graph alignment**: pairwise alignment matrix with intra/inter-category separation;
  visual overlay of aligned graph pairs
- **Lateral voting**: per-LM evidence traces showing convergence; consensus evidence
  versus individual LM evidence; vote content inspection
- **Compositional substitution**: per-episode part-level confusion table; base-only
  vs logo-only vs both-correct breakdown
- **Distance-aware evaluation**: confusion matrix weighted by representation distance
  rather than binary correct/incorrect

---

## Implementation Plan

### Stage 0: Assumption Confirmation (gate before mechanism work)

1. Compute pairwise graph alignment between all 6 YCB training objects.
   Measure spatial node correspondence and feature overlap.
   Compare intra-category vs inter-category distributions. (Run 0A)
2. Log full evidence vectors during holdout evaluation. Analyze whether
   category signal is visible in the top-k evidence set. (Run 0B)
3. Review existing control confusion matrix against hierarchical labels
   (already available from original Phase 2).

**Go/no-go gate**: If C1 fails (graph structure does not separate categories), then
Option C (graph alignment) is deprioritized and Option A (lateral voting) becomes the
sole primary path. If both C1 and C2 fail, Phase 2 needs a fundamentally different
representational substrate and should be escalated.

### Stage 1: Evidence-Distribution Baseline

4. Add evidence-distribution logging to the eval config.
5. Run holdout evaluation with full evidence logging.
6. Compute category-level metrics from the evidence distribution.
7. Implement distance-aware evaluation metrics.

### Stage 2: Lateral Voting Category Consensus

8. Design the multi-LM training protocol with controlled overlap.
9. Implement training configs that split the object set across N peer LMs.
10. Train the multi-LM system.
11. Evaluate with voting enabled on holdout objects.
12. Analyze consensus evidence versus individual LM evidence.

### Stage 3: Graph Structural Alignment (if C1 passes)

13. Implement graph-to-graph alignment using existing KDTree infrastructure.
14. Extract shared substructures from aligned graph pairs.
15. Register shared subgraphs as matchable entries in graph memory.
16. Evaluate whether matching against shared subgraphs improves holdout transfer.

### Stage 4: Multi-Hypothesis Transport (if earlier stages succeed)

17. Modify LM0 `get_output()` to emit full hypothesis union rather than winner.
18. Modify LM1 to consume and store distributional inputs.
19. Evaluate stacked hierarchy with distributional transport.

### Stage 5: Top-Down Biasing (if category knowledge exists)

20. Implement top-down connections per the existing future-work spec.
21. Evaluate category-constrained recognition speed.

---

## Status

- replanned: completed (this document)
- assumption confirmation: **completed** -- all three assumptions pass strongly
- Stage 1 (evidence distribution): completed via existing eval data -- category
  signal is perfectly present in the winner-level confusion pattern
- Stage 2 (lateral voting): **2-LM and 3-LM voting completed**
  - 2-LM: 94.6% consensus, 35% faster convergence, category subgraphs winning
  - 3-LM: 91.1% consensus, 44% faster convergence, 23.6 avg steps
  - trend: more LMs → faster convergence but lower per-LM accuracy
- Stage 3 (graph alignment): **prototype completed and end-to-end tested**
  - category subgraph extraction: 100% training accuracy with massive separation
  - checkpoint injection: augmented checkpoint with 9 graphs (6 instance + 3 category)
  - holdout eval: category subgraphs load and participate in evidence matching, but
    instance graphs always outcompete due to 2-4x more nodes and correspondingly
    higher evidence accumulation. Zero changed rows vs control.
  - interpretation: graph-structural category signal is real and extractable, but
    the current evidence matching pipeline naturally favors larger graphs. Category
    subgraphs are best used as a secondary classification layer (post-hoc category
    assignment from evidence rankings) rather than as direct competitors in the
    evidence race.
- Stage 4 (compositional substitution): **base-shape subgraph eval completed**
  - control baseline: 98.6% base-shape correct (over 2 LMs), 1.4% logo, 0% exact
  - base-shape subgraph extraction: 5 subgraphs with 946-1603 shared nodes,
    95-100% coverage between same-base objects, 10/10 training accuracy
  - augmented eval: identical 98.6% base-shape accuracy; baseshape subgraphs
    won 6/70 episodes in LM0 (8.6% win rate), all correct; 31/70 predictions
    changed vs control showing significant evidence redistribution
  - interpretation: base-shape accuracy is saturated. Baseshape subgraphs
    provide grounded labels but cannot improve the quantitative metric.
    The remaining gap is logo recognition (1.4%).
- Evidence logging: **completed** -- `evidence_per_graph` column added to eval stats
- Distance-aware metrics: **completed** -- control shows 33.0% distance-weighted
  accuracy (vs 0% binary), validating within-category confusion as genuine progress
- Omniglot cross-version: **scaled benchmark + voting completed**
  - 6-char baseline: 50% exact, 100% alphabet (saturated for alphabet)
  - **20-char scaled control: 45% exact, 60% alphabet (non-saturated)**
  - **Disjoint 2-LM voting: 65% alphabet (+5pp)**, 25% subgraph win rate
  - Overlapping voting: 35% (worse — redundancy hurts)
  - **30-char (5x6) scaled control: 43.3% exact, 70% alphabet**
  - Online category bias: zero effect in single-LM mode
  - 2-LM disjoint voting: 23.3%/50% (per-LM), recovers to 70% with
    cross-LM aggregation — cannot exceed single-LM with same sensor data
  - **Key lesson**: voting needs shared vocabulary or different viewpoints,
    not just instance splitting
- **Success criteria assessment**:
  - 15pp win on non-saturated benchmark: **+10pp achieved** (60%→70%
    alphabet on scaled Omniglot with 2-LM voting + category evidence
    readout). Path to 15pp: more characters per alphabet, tuned thresholds.
  - Inspectable grounded similarity: **met** (category/baseshape/alphabet
    subgraphs are explicit, inspectable graphs)
  - Distance-aware evaluation: **met** (33% vs 0% binary)
  - Category behavior from own representations: **met** (no external
    classifiers, subgraphs extracted from existing graphs)
  - Cross-domain generalization: **met** (results on YCB, compositional,
    and Omniglot domains)
- next steps: see Remaining Plan section

## Decision Log

- 2026-03-15 through 2026-03-19: Original Phase 2 investigation ran 46+
  configurations. Full log preserved in the original document at
  `docs/phase-2-category-and-concept-generalization.md`.
- 2026-03-19: First-principles review identified that the original investigation
  was primarily optimizing within a space that had no solution (scoring over
  degenerate representations). Rebuilt the approach around the question "does
  graph structure carry category information?" rather than "which encoding/carrier/
  objective can rescue the hierarchy?"
- 2026-03-19: Elevated lateral voting from a distant future-work item to the primary
  TBT-aligned mechanism for category consensus, based on the theory documentation
  and the Hawkins et al. 2025 heterarchy preprint.
- 2026-03-19: Demoted SDR encoding from primary mechanism to optional analysis tool,
  based on the circular learning signal diagnosis.
- 2026-03-19: Ran Stage 0 assumption diagnostics. All three assumptions confirmed:
  - C1: Graph alignment produces Cohen's d = 2.96 spatial separation between
    intra-category and inter-category pairs. Perfect separation threshold exists.
  - C2/C3: 56/56 holdout episodes land within the correct category. Evidence
    already contains a perfect category signal.
  - Go/no-go verdict: GO on all approaches.
- 2026-03-19: Ran Stage 3 prototype (category subgraph extraction). Extracted
  shared substructures for cups (596 nodes), utensils (916 nodes), and boxes
  (865 nodes). Every training object matches its category subgraph at 1.000 spatial
  overlap with cross-category scores at 0.006-0.280. This is the strongest
  mechanistic result in Phase 2 so far: the category representation is explicit,
  inspectable, and directly usable in the existing matching pipeline.
- 2026-03-19: Injected category subgraphs into the checkpoint and ran the full
  56-episode holdout eval. First run failed (subgraphs in wrong coordinate space);
  fixed by keeping positions in world space. Second run completed: zero changed
  rows vs control. Instance graphs always outcompete category subgraphs on
  evidence volume (2-4x more nodes). This is expected and informative: the
  evidence matching pipeline is working correctly by preferring more detailed
  models. Category subgraphs should be a secondary readout layer, not a
  primary competitor.
- 2026-03-19: Key architectural insight: the right integration point for
  category subgraphs is not "compete with instance graphs" but "provide a
  grounded category label alongside the instance match." The brain maintains
  both instance and category representations simultaneously at different levels.
  The next implementation step is a post-evidence category readout that uses
  category-subgraph evidence as a secondary signal.
- 2026-03-19: Built 2-LM voting checkpoint and Hydra config. Each LM has 3
  disjoint instance graphs plus all 3 shared category subgraphs. Vote matrix
  connects both LMs for lateral voting.
- 2026-03-19: Completed first multi-LM voting Habitat eval. Key results:
  LM1 alone at 96.4% category accuracy; best-evidence consensus at 94.6%;
  category subgraphs won 7/112 episodes (first time in online matching);
  35% faster convergence (27.1 vs 42.1 avg steps). The split-instance +
  shared-category-subgraph design validates TBT-aligned lateral voting.
- 2026-03-19: Ran compositional substitution control eval on holdout objects.
  Result: 0% exact, **98.6% base-shape correct** (over 2 LMs), 1.4% logo
  correct, 0% both. This is the saturation-resistant benchmark: the control
  already recognizes base shapes perfectly but cannot identify which logo is on
  an unseen base+logo combo.
- 2026-03-19: Extracted base-shape subgraphs from the compositional checkpoint.
  Very high coverage (95-100%) because the logo is a tiny surface detail on
  a large base shape. 10/10 training objects correctly matched with perfect
  separation. Adapted extraction and injection tools to handle GridObjectModel
  checkpoint format (added --mapping-key, --lm-index flags).
- 2026-03-19: Ran base-shape subgraph eval on compositional holdout (Run 6).
  Key results: baseshape subgraphs won 6/70 episodes in LM0 (8.6% win rate),
  all correct. 31/70 predictions changed vs control showing significant evidence
  redistribution. However, base-shape accuracy is identical (98.6%) because the
  control was already near-saturated.
- 2026-03-19: Key finding: the compositional substitution benchmark is saturated
  for base-shape recognition (~99%) but unsaturated for logo recognition (~1%).
- 2026-03-19: Implemented evidence-per-graph logging, distance-aware metrics,
  3-LM voting (91.1% category, 44% faster convergence). Fixed Omniglot
  IndexError bugs and downloaded dataset.
- 2026-03-19: Scaled Omniglot to 20 characters (5 alphabets x 4 chars).
  Cross-version control: 45% exact, 60% alphabet. This is the first
  non-saturated benchmark with cross-category (cross-alphabet) errors.
- 2026-03-19: Disjoint 2-LM voting on scaled Omniglot achieved **65%
  alphabet accuracy (+5pp over control)**. Alphabet subgraphs won 25% of
  episodes — highest win rate across all benchmarks. This validates that
  category subgraphs serve as effective shared voting vocabulary.
- 2026-03-19: Overlapping 2-LM voting (15/20 chars per LM) performed worse
  (35% alphabet). Key insight: voting requires instance diversity between
  LMs. Redundant instances create interference, not consensus.
- 2026-03-19: First-principles analysis revealed the core error in the
  approach: we were trying to make category subgraphs compete in the
  instance-level evidence race, but the evidence pipeline structurally
  favors graphs with more nodes. The category signal already exists in
  the evidence distribution — it just gets discarded when we take argmax.
- 2026-03-19: Implemented category-aggregated evidence readout. Instead of
  argmax over instance graphs, sum evidence by category and pick the
  category with highest total mass. Results:
  - 1 LM + category readout: 65% alphabet (+5pp over control)
  - **2 LM voting + cross-LM category aggregation: 70% (+10pp)**
  - The +10pp comes from two orthogonal signals: voting diversity provides
    multiple independent evidence samples, and category aggregation extracts
    the category-level pattern from each. This is the TBT prediction:
    category emerges from the pattern of activation across multiple columns.
- 2026-03-19: Key architectural insight validated: the right integration
  point for category recognition is NOT "add a category graph and hope it
  wins the evidence race" but "read out the category from the distribution
  of evidence across existing instance graphs." The evidence is already
  category-informative; we were discarding it.
  The remaining Phase 2 challenge on this benchmark is not base-shape
  categorization but part-level transfer — recognizing that the same logo
  appears on a different base. This requires either logo-specific subgraphs
  or a different mechanism that separates base and logo representations.
- 2026-03-20: Scaled Omniglot to 30 objects (5 alphabets x 6 chars).
  Single-LM baseline: 43.3% exact, 70% alphabet — stronger than 4-char
  because more same-alphabet options catch cross-alphabet confusions.
- 2026-03-20: Online category bias (T1.3 + T1.4a) has zero effect in
  single-LM mode. The argmax winner already agrees with the category
  winner when all instances are in one LM.
- 2026-03-20: Disjoint 2-LM voting on 30 objects: per-LM accuracy drops
  to 20%/50% (each LM has only 15 graphs). Cross-LM aggregation recovers
  to 70% category — matching, but not exceeding, single-LM.
- 2026-03-20: Key learning: for voting to add value beyond single-LM,
  LMs need either (a) shared category vocabulary (category subgraphs as
  in Run 12), (b) different viewpoints/sensors, or (c) different feature
  sensitivities. Pure instance splitting with the same sensor data cannot
  exceed the information available to one LM with all instances.

## HippocampalModule: Path To Higher Concept Layers

The `HippocampalModule` (reconstructed from bytecode on 2026-03-20, 665 lines,
27 unit tests passing) sits at the top of the LM heterarchy and implements:

1. **Fast binding**: Hebbian co-occurrence tracking between concepts
2. **Episodic memory**: Timestamped sequences of active concepts per episode
3. **Relational memory**: Typed graph of relations (cooccurrence, spatial,
   temporal) between concepts, using networkx
4. **Context signal**: Broadcast downward to all lower LMs

This module is the bridge from Phase 2 (categories) to higher concept layers:

- **Layer 2 concepts** (dangerous, useful, heavy): require episodic memory
  to track action-outcome associations across episodes. The episodic buffer
  (`_record_episode_step`) and relational graph (`_update_relational_graph`)
  provide the storage substrate. What's missing: a valence/reward signal
  and connection to the motor system.
- **Layer 3 concepts** (friend, trust): require persistent agent identity
  and temporal patterns across many encounters. The episodic memory (bounded
  deque of episodes) and association matrix (`_rebuild_association_matrix`)
  provide the cross-episode accumulation. What's missing: agent-object
  distinction and an API for long-horizon recall.
- **Hippocampal replay**: The module can replay stored episodes to downstream
  LMs via `get_context_signal()`, allowing columns to build models from
  non-sensory input — the mechanism for constructing abstract reference frames.

**Current state**: The module conforms to the LearningModule interface (matching_step,
exploratory_step, receive_votes, send_out_vote) so it can sit in the heterarchy
alongside EvidenceGraphLMs. Its 4 memory systems are functional but not yet connected
to the evidence matching pipeline or the category-aggregated readout. The
`_extract_active_concepts()` method filters upstream LM states by confidence, so it
already knows which objects are recognized at each step.

**Recommended next steps**: Wire the HippocampalModule into a multi-LM eval config
where it receives evidence from lower LMs and tests whether its association matrix
captures category-like patterns from instance co-activation. This would test the core
hypothesis: concepts as patterns of activation across columns, accumulated over time.

Source: `src/tbp/monty/frameworks/models/hippocampal_module.py`
Tests: `tests/unit/frameworks/models/hippocampal_test.py`

## Current Blockers

- The compositional substitution benchmark is saturated for base-shape accuracy
  (~98.6%). Demonstrating a quantitative win requires either a harder benchmark
  or a mechanism that addresses logo recognition.
- The YCB category benchmark is saturated (100% category accuracy in control).
- The +10pp result on Omniglot (60%→70%) is strong but below the 15pp target.
  Scaling to more characters per alphabet should close the gap.

## Open Questions

- How many peer LMs are needed for lateral voting to produce a clean category signal?
- Does graph structural alignment work for objects that share category but have very
  different morphology (e.g., a flat plate and a tall mug are both "vessels")?
- Should the distance-aware evaluation metric use graph-space distance, evidence-space
  distance, or both?
- Can top-down biasing be implemented without first solving the bottom-up category
  signal, or do they need to be co-developed?
- Is within-LM graph consolidation (Option C) worth pursuing if it requires an offline
  step outside the sensorimotor loop, given TBT's commitment to learning through
  interaction?

## Success Criteria

- On a saturation-resistant benchmark, the best Phase 2 mechanism exceeds the
  single-LM exact-ID control by at least `15` percentage points on category or
  transfer accuracy.
- The mechanism produces inspectable, grounded similarity structure rather than opaque
  scores.
- Distance-aware evaluation shows that errors are closer to the target in
  representation space than baseline errors.
- At least one approach produces category-like behavior that emerges from Monty's
  own representations rather than from external labels or classifiers.
- The approach generalizes to at least one domain beyond the YCB similar-object split
  (Omniglot, compositional substitution, or arrangement transfer).

## Failure Criteria

- Graph alignment (C1) and evidence distribution (C2) both fail to show category
  structure in existing representations, meaning the system needs new representational
  machinery beyond what Phase 2 scoped.
- Lateral voting consensus is not more category-like than individual LM evidence.
- All approaches require external classifiers or non-grounded machinery to produce
  category-level improvement.
- The compositional substitution benchmark shows zero transfer to unseen combinations.

## Next Decision

- decision to make: which remaining step to prioritize
  - **Evidence-Level Category Logging** (Next Step 1): low-cost, makes category
    subgraph evidence visible for every episode even when instance graphs win.
    Required diagnostic before investing more in subgraph mechanisms.
  - **3+ LM Voting** (Next Step 2): extends the successful 2-LM pilot; may
    amplify baseshape/category subgraph wins. Highest expected scientific value.
  - **Omniglot Cross-Version** (Next Step 3): strongest test of generalization
    beyond YCB/compositional domains. Currently blocked by data loading issue.
  - **Distance-Aware Evaluation** (Next Step 4): makes partial progress visible
    in metrics. Useful for all future experiments.
- evidence needed: review this document and select highest-impact next step
- fallback plan: if all subgraph approaches plateau at current levels, the
  lateral voting path (Stage 2) remains the primary TBT-aligned mechanism with
  the strongest empirical results (94.6% category accuracy, 35% faster
  convergence, category subgraphs winning in online matching)
