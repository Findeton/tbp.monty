# Phase 2: Category And Concept Generalization

## Purpose

This phase turns Monty from an instance recognizer into a system that can generalize
across related instances and start forming grounded concept-like abstractions.

The immediate goal is to prove that Monty can recognize unseen instances at the
category level without relying on an opaque external classifier.

The broader goal is to show that grounded similarity can support a path from object
memory to reusable concepts before later phases add time, prediction, language, and
semantic bridge mechanisms.

## Metadata

- phase: Phase 2: Category And Concept Generalization
- roadmap link: [docs/full-system-roadmap.md](docs/full-system-roadmap.md)
- owner: local working plan
- status: planning for review
- last updated: 2026-03-15

## Goal

Show that Monty can generalize from object instances to categories using grounded,
inspectable similarity structure, and define one minimal path toward concept-like
generalization over relations or roles.

## Why This Phase Exists

Phase 1 showed that hierarchy can produce a child-reuse signal, but it did not show a
strict accuracy win over the monolithic baseline. That result increases the importance
of Phase 2 rather than reducing it.

If Monty cannot move beyond exact object memory, then later work on behavior,
prediction, and language will sit on top of a brittle nearest-instance substrate. The
system needs a grounded path from "this exact mug" to "cups like this" before it can
credibly claim category or concept learning.

## Current Hypothesis

The main hypothesis is:

> If lower-level LMs expose similarity-aware outputs instead of naive exact object IDs,
> and evaluation uses hierarchical labels that separate instance from category, then
> Monty should be able to recognize held-out instances at category level and transfer at
> least one simple concept-like pattern across similar constituent objects.

This hypothesis is falsifiable. If Monty only succeeds by memorizing exact instances,
or if category-level improvements come only from indiscriminate graph merging, then the
phase has failed.

## Investigation Summary

The repo already contains several important pieces of substrate for this phase.

- There is already a YCB `similar_objects` split under
  `config/environment_interface/ycb/similar_objects` containing cups, utensils, and box
  variants. This is the smallest existing candidate for a category-transfer benchmark.
- The benchmark documentation already states that similar-object experiments need more
  fine-grained metrics with hierarchical labels so that instance and category accuracy
  are not conflated.
- The Omniglot environment and tutorial are already in the repo. The tutorial also
  explicitly notes that generalization to unseen versions of the same characters is poor
  without hierarchy.
- The default `EvidenceGraphLM` path still uses a placeholder object-ID encoding. The
  current `_object_id_to_features` implementation is just the sum of character codes,
  and the code comments explicitly say this should be replaced by a real similarity
  measure.
- There is already an implemented similarity-aware path in
  `EvidenceSDRGraphLM` and `EvidenceSDRLMMixin`, plus the
  `SDRFeatureEvidenceCalculator`. This is not hypothetical infrastructure; it is already
  part of the repo and documented as a way to encode object similarity.
- There is also a future-work note specifically proposing to send similarity-encoded
  object IDs to higher levels for generalization across related compositional objects.
- The current Phase 0 tooling only wraps `run_parallel.py`. This works well for YCB
  parallel experiments, but Omniglot tutorial flows currently use `run.py`, so Phase 2
  must either extend the execution discipline or keep explicit manual run registries for
  non-parallel experiments.
- There is no existing category-aware logging path that cleanly separates exact-instance
  correctness, within-category confusion, and category-level correctness.

These facts suggest a staged Phase 2 plan rather than a single large leap.

## TBT And Jeff Hawkins Lens

Phase 2 should remain grounded in the Thousand Brains style picture of abstraction.

- Categories should emerge from shared grounded structure across object models, not from
  attaching an unrelated classifier head on top of Monty.
- The lower level should remain instance-grounded and sensorimotor.
- The next level should learn reusable structure over similar grounded objects, not just
  over arbitrary labels.
- Similarity-aware encodings are more consistent with cortical reuse than exact integer
  IDs or opaque class embeddings, because they preserve graded relationships between
  known objects.
- A concept-like benchmark should remain grounded in object arrangements or structured
  relations, not pure text or symbolic lookup.

The most TBT-consistent result would be a system where category and concept signals
emerge from reuse of grounded structure, while the supporting metrics and labels remain
engineering scaffolding rather than the main mechanism.

## Biological Plausibility Review

- biologically plausible within the theory: category-like generalization via similarity
  across grounded object models; hierarchical reuse; stroke-to-character structure in
  Omniglot-like settings
- engineering approximations: explicit hierarchical labels; hand-authored YCB taxonomy;
  fixed train and holdout splits; category-level summary metrics
- temporary hacks: SDR overlap thresholds; YAML taxonomies; draft synthetic relation
  benchmark definitions
- explicitly non-biological compromises: any offline clustering or analysis helper used
  only to inspect results; any hand-built concept dataset for the first relation test

This phase should not use a deep neural classifier as Monty's category mechanism. A
neural or statistical helper may be acceptable as an analysis baseline, but not as the
core success path.

## Design Options Considered

### Option A: Metric-First Category Benchmark With Current Instance Models

- benefits: lowest engineering cost; establishes a clean control; isolates whether the
  current system already contains hidden category signal
- risks: may prove only that the metrics were missing, not that the model generalizes;
  likely limited by exact-ID outputs
- expected scientific value: moderate
- expected engineering cost: low

### Option B: Similarity-Aware Category Transfer On YCB Similar Objects

- benefits: reuses an existing dataset split and an already implemented similarity-aware
  LM path; directly attacks the known `object_id` limitation; stays close to grounded
  3D object recognition
- risks: still requires new category labels, metric logic, and benchmark configs;
  success may depend heavily on taxonomy quality
- expected scientific value: high
- expected engineering cost: moderate

### Option C: Omniglot As The Primary Phase 2 Benchmark

- benefits: strong test of cross-instance generalization; already documented as a weak
  point; naturally supports instance-versus-category-like splits
- risks: entangles Phase 2 with unfinished compositional hierarchy work; tutorial path
  currently uses `run.py`; may overload this phase with new hierarchy scaffolding
- expected scientific value: high
- expected engineering cost: high

### Option D: Concept-First Relation Benchmark

- benefits: directly addresses the roadmap requirement for abstract relation or role;
  strongest long-term concept signal if it works
- risks: requires the most new benchmark machinery; too easy to build something
  underconstrained or toy-like before category metrics are stable
- expected scientific value: high but delayed
- expected engineering cost: high

Preferred approach: start with Option A plus Option B as the required first milestone,
keep Option C as the secondary validation benchmark, and treat Option D as the stretch
benchmark that converts category transfer into a genuine concept-like test.

## Neural Nets, HTM, And Other Helper Systems

The preferred Phase 2 path does not introduce a new deep neural model as the category
engine.

- preferred helper mechanism: the existing SDR-based similarity encoding path
  (`EvidenceSDRGraphLM`)
- acceptable analysis helpers: offline clustering, overlap visualization, or confusion
  analysis if used only to inspect Monty's learned structure
- not acceptable as the primary solution: a separate classifier head that performs
  category recognition while Monty remains instance-only internally

The open technical question is whether SDR overlap is the right similarity carrier or
whether the direct evidence-similarity matrix should be passed upward instead. Phase 2
should be designed so that both can be compared without changing the benchmark itself.

## Deliverables

- this Phase 2 document linked from the roadmap
- a hierarchical-label taxonomy for the 10-object YCB similar-object benchmark
- category-aware evaluation outputs that separate instance accuracy from category
  accuracy
- a control benchmark using current exact-ID outputs on the YCB similar-object split
- a similarity-aware benchmark using `EvidenceSDRGraphLM` or an equivalent
  evidence-similarity transport mechanism
- one secondary generalization benchmark, preferably Omniglot unseen-version transfer
  or a minimal relation benchmark if scope permits
- an analysis report comparing exact-instance behavior, category behavior, and any
  concept-like transfer signal

## Benchmark And Test Plan

### Benchmark A: YCB Similar-Object Category Control

- name: phase2_ycb_category_control
- input setup: the existing 10-object `similar_objects` YCB split grouped into cups,
  utensils, and boxes
- baseline: current exact-ID `EvidenceGraphLM` path without similarity-aware object
  encodings
- target metric: held-out instance category accuracy, held-out instance exact accuracy,
  within-category confusion rate, and graph merge statistics
- failure condition: category accuracy improves only because graphs collapse multiple
  objects indiscriminately, or category accuracy remains near chance on held-out
  instances

### Benchmark B: YCB Similar-Object Category Transfer With Similarity-Aware Outputs

- name: phase2_ycb_category_sdr
- input setup: same YCB split and taxonomy as Benchmark A, but with similarity-aware
  lower-level outputs using `EvidenceSDRGraphLM` or direct evidence-similarity transport
- baseline: Benchmark A plus the current scalar or placeholder object-ID hierarchy path
- target metric: at least one meaningful gain in held-out category accuracy while
  preserving inspectable representations and acceptable seen-instance behavior
- failure condition: no gain over Benchmark A, unstable training, or unreadable learned
  similarity structure

### Benchmark C: Omniglot Unseen-Version Generalization

- name: phase2_omniglot_unseen_versions
- input setup: existing Omniglot tutorial path, training on one drawing version of a
  character and evaluating on unseen versions of the same character; later extension can
  test new characters inside seen alphabets
- baseline: the current tutorial configuration and any non-hierarchical control on the
  same split
- target metric: version-transfer accuracy, degradation relative to seen-version
  controls, and evidence traces that remain category-like rather than drawing-specific
- failure condition: generalization remains poor and no better than the existing
  tutorial control

### Benchmark D: Minimal Relation Or Role Transfer Benchmark

- name: phase2_relation_transfer
- input setup: a small arrangement benchmark, such as a dinner-table-like scene, where
  one set of child instances is used for learning and a similar but unseen set is used
  for evaluation
- baseline: exact child-ID hierarchy and monolithic arrangement recognition
- target metric: relation or arrangement accuracy under child-instance substitution
- failure condition: no transfer when constituent objects change, or success depends on
  a non-grounded external classifier

## Execution Registry

The commands below are draft review commands for planned configs. Most of these configs
do not exist yet and are not meant to be launched until the plan is approved.

### Run A: YCB Category Control Training

- run name: `phase2_ycb_category_control_train`
- purpose: train the control model for the YCB category benchmark using current exact-ID
  outputs
- prepare command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py prepare --experiment phase2_ycb_category_control_train --override ++experiment.config.logging.output_dir=/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs --override ++experiment.config.logging.run_name=phase2_ycb_category_control_train --override ++experiment.config.logging.wandb_handlers=[] --num-parallel 1`
- monitor command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py status --manifest /Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_ycb_category_control_train/phase0_manifest.json`
- expected output path: `/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_ycb_category_control_train`
- resume or recovery command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py resume --manifest /Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_ycb_category_control_train/phase0_manifest.json`
- stop command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py stop --manifest /Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_ycb_category_control_train/phase0_manifest.json`
- resource profile: `num_parallel=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`
- current status: planned, config not implemented

### Run B: YCB Category Control Held-Out Evaluation

- run name: `phase2_ycb_category_control_eval_holdout`
- purpose: evaluate the control model on held-out similar-object instances with both
  exact-instance and category scoring
- prepare command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py prepare --experiment phase2_ycb_category_control_eval_holdout --override ++experiment.config.logging.output_dir=/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs --override ++experiment.config.logging.run_name=phase2_ycb_category_control_eval_holdout --override ++experiment.config.logging.wandb_handlers=[] --num-parallel 1`
- monitor command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py status --manifest /Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_ycb_category_control_eval_holdout/phase0_manifest.json`
- expected output path: `/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_ycb_category_control_eval_holdout`
- resume or recovery command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py resume --manifest /Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_ycb_category_control_eval_holdout/phase0_manifest.json`
- stop command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py stop --manifest /Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_ycb_category_control_eval_holdout/phase0_manifest.json`
- resource profile: `num_parallel=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`
- current status: planned, config not implemented

### Run C: YCB Similarity-Aware Training

- run name: `phase2_ycb_category_sdr_train`
- purpose: train a similarity-aware category-transfer model using `EvidenceSDRGraphLM`
  or an equivalent evidence-similarity transport path
- prepare command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py prepare --experiment phase2_ycb_category_sdr_train --override ++experiment.config.logging.output_dir=/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs --override ++experiment.config.logging.run_name=phase2_ycb_category_sdr_train --override ++experiment.config.logging.wandb_handlers=[] --num-parallel 1`
- monitor command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py status --manifest /Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_ycb_category_sdr_train/phase0_manifest.json`
- expected output path: `/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_ycb_category_sdr_train`
- resume or recovery command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py resume --manifest /Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_ycb_category_sdr_train/phase0_manifest.json`
- stop command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py stop --manifest /Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_ycb_category_sdr_train/phase0_manifest.json`
- resource profile: `num_parallel=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`
- current status: planned, config not implemented

### Run D: YCB Similarity-Aware Held-Out Evaluation

- run name: `phase2_ycb_category_sdr_eval_holdout`
- purpose: evaluate the similarity-aware category model on held-out similar-object
  instances
- prepare command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py prepare --experiment phase2_ycb_category_sdr_eval_holdout --override ++experiment.config.logging.output_dir=/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs --override ++experiment.config.logging.run_name=phase2_ycb_category_sdr_eval_holdout --override ++experiment.config.logging.wandb_handlers=[] --num-parallel 1`
- monitor command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py status --manifest /Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_ycb_category_sdr_eval_holdout/phase0_manifest.json`
- expected output path: `/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_ycb_category_sdr_eval_holdout`
- resume or recovery command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py resume --manifest /Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_ycb_category_sdr_eval_holdout/phase0_manifest.json`
- stop command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py stop --manifest /Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_ycb_category_sdr_eval_holdout/phase0_manifest.json`
- resource profile: `num_parallel=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`
- current status: planned, config not implemented

### Run E: Omniglot Unseen-Version Evaluation

- run name: `phase2_omniglot_unseen_versions`
- purpose: evaluate generalization to unseen character versions using the Omniglot
  environment
- launch command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python run.py experiment=phase2_omniglot_unseen_versions`
- monitor command: `tail -f /Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_omniglot_unseen_versions/log.txt`
- expected output path: `/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_omniglot_unseen_versions`
- resume or recovery command: `not yet recoverable under the current Phase 0 tool; either extend the tool to wrap run.py or relaunch from the saved config and output directory`
- stop command: `pkill -f 'experiment=phase2_omniglot_unseen_versions'`
- resource profile: `one run at a time`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`
- current status: planned, config not implemented

## Resource Budget

- expected CPU usage: moderate for YCB runs and low-to-moderate for Omniglot controls
- expected RAM usage: moderate; stay within the current one-heavy-run-at-a-time policy
- expected runtime: multi-minute for YCB evaluations and shorter for reduced Omniglot
  validations
- default safety flags: `num_parallel=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`
- reason if exceeding the default budget: none in the first implementation pass

## Recovery Plan

- YCB parallel experiments should use the existing Phase 0 manifest workflow
- all Phase 2 YCB runs should write manifests, config snapshots, logs, and helper
  scripts under `/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs`
- if VS Code or the machine restarts, use the recorded `status` command for any YCB run
  before deciding whether to resume or relaunch
- Omniglot-style `run.py` experiments currently require either a manual stable log file
  plus saved overrides, or an extension of the Phase 0 tool before they can meet the
  same restart discipline
- the first implementation task should not launch long Omniglot runs until the recovery
  story is explicit and review-approved

## Observability Plan

Phase 2 needs more observability than Phase 1 because raw accuracy is not enough.

Required artifacts:

- instance versus category confusion tables from the same evaluation CSV
- within-category versus cross-category confusion summaries
- graph merge statistics such as `mean_objects_per_graph` and
  `mean_graphs_per_object` where relevant
- similarity inspection artifact, such as an SDR overlap heatmap, evidence-similarity
  matrix, or t-SNE of learned object representations
- for Omniglot or relation benchmarks, one detailed trace or replay artifact showing why
  the model generalized or failed

## Implementation Plan

1. Decide whether Phase 2 should begin immediately after review or wait for a targeted
   Phase 1.5 ablation.
2. Define the first approved taxonomy and holdout split for the YCB similar-object
   benchmark.
3. Add category-aware metrics and summaries before changing the model path.
4. Run the exact-ID control benchmark on the approved YCB split.
5. Add the similarity-aware path using `EvidenceSDRGraphLM` or direct evidence-similarity
   transport.
6. Re-run the YCB holdout benchmark and compare against the control on both instance and
   category metrics.
7. If the YCB benchmark shows real category transfer, run a secondary validation on
   Omniglot unseen versions.
8. If category transfer is confirmed and scope still permits, design the minimal
   relation or role benchmark and decide whether it belongs inside Phase 2 or as a
   Phase 2B follow-up.

## Status

- investigating: completed
- prototyping: not started
- running benchmarks: not started
- blocked: awaiting review and approval

## Decision Log

- 2026-03-15: Chose the YCB similar-object split as the first required Phase 2
  benchmark because the data split already exists in the repo and is the lowest-cost
  path to category-transfer evidence.
- 2026-03-15: Chose similarity-aware object outputs as the preferred first mechanism
  because the repo already contains `EvidenceSDRGraphLM` while the default object-ID
  feature path is explicitly marked as a placeholder.
- 2026-03-15: Kept Omniglot and concept-like relation transfer as secondary validation
  benchmarks so that Phase 2 does not immediately over-expand into new dataset and
  hierarchy scaffolding.

## Current Blockers

- no approved hierarchical-label taxonomy or holdout policy exists yet for the YCB
  similar-object benchmark
- no category-aware evaluation summary currently exists in the standard logging path
- the default `object_id` feature path is still similarity-poor by construction
- `run.py` experiments do not yet benefit from the same Phase 0 orchestration used for
  parallel runs
- it is not yet decided whether the roadmap should require a Phase 1.5 ablation before
  Phase 2 implementation starts

## Open Questions

- should similarity be carried upward via SDR overlap, direct evidence-similarity
  values, or both as competing implementations?
- should the first YCB category benchmark stay on the 10-object split or expand to a
  broader set after the control metric exists?
- should Omniglot be treated as Phase 2 proper, or as a secondary validation after the
  YCB benchmark is working?
- what is the smallest credible relation or role benchmark that proves concept-like
  transfer without forcing a large new environment build?

## Success Criteria

- on held-out YCB instances, category accuracy reaches at least `60%`
- on that same held-out evaluation, category accuracy exceeds held-out exact-instance
  accuracy by at least `20` percentage points
- the similarity-aware run exceeds the exact-ID control by at least `15` percentage
  points on held-out category accuracy while losing no more than `10` percentage points
  of seen-instance exact accuracy
- the observability artifacts show structured within-category reuse rather than
  indiscriminate graph collapse
- if a secondary benchmark is run, it exceeds its exact-ID or non-hierarchical control
  by at least `15` percentage points on the transfer metric

## Failure Criteria

- category accuracy improves only because graphs merge many objects without preserving
  useful distinctions
- category-level success requires a separate external classifier instead of grounded
  Monty representations
- the similarity-aware path produces no meaningful gain over the exact-ID control
- the concept-like benchmark requires unrelated non-grounded machinery to work at all
- the benchmark suite cannot be run reproducibly under a documented execution and
  recovery plan

## Next Decision

- decision to make: whether to approve Phase 2A as YCB category transfer with a
  similarity-aware LM path, or redirect the phase toward Omniglot first
- evidence needed: review of this document, approval of the provisional success bars,
  and agreement on the first taxonomy and holdout split
- fallback plan: if the similarity-aware path is judged too much scope for the first
  pass, implement the metric-first YCB control benchmark first and use that result to
  refine the second half of the phase