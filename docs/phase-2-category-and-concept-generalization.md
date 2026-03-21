# Phase 2: Category And Concept Generalization (Original)

> **Track**: [Track 1: Multi-Column Intelligence](track-1-multi-column-intelligence.md)
> **Status**: Superseded by [phase-2-category-and-concept-generalization-v2.md](phase-2-category-and-concept-generalization-v2.md)
> **Roadmap**: [full-system-roadmap-v2.md](full-system-roadmap-v2.md)
>
> This document is retained for historical reference. It records 46+ experimental
> configurations that all failed, and the diagnostic analysis that led to the
> rebuilt approach in v2. Read the v2 document for current work.

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
- status: investigated with control, valid eval-style SDR fitting, full compositional SDR bridge train/eval, frozen-LM0 diagnostic, top-k union pilot, parent-support pilot, temporal-support accumulator pilot, and a stronger held-out compositional substitution benchmark scaffold recorded
- last updated: 2026-03-19

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

Initial execution decision for this phase:

- begin with the metric-first control benchmark on the YCB similar-object split
- do not start the SDR-based similarity path until the control metric and holdout split
  are validated end to end
- keep Omniglot as the secondary benchmark after the YCB control path produces a clean
  baseline

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

## Improvement Plan

The next implementation pass should proceed in four ordered workstreams.

### Improvement 1: Hierarchical SDR Composition Bridge

- goal: test whether similarity-aware `object_id` transport improves the existing
  stacked high-level and low-level compositional LM setup from Phase 1
- motivation: the current Phase 1 compositional stack still uses the placeholder
  scalar `object_id` path, so it does not actually test whether SDR-style similarity
  transport improves parent-object learning or generalization
- first implementation target: add stacked compositional configs where the lower LM and
  higher LM both use `EvidenceSDRGraphLM`, and use those runs to reevaluate the current
  logos-on-objects hierarchy against the monolithic baseline
- expected outcome: either a measurable compositional improvement, or a clean negative
  result that shows SDR transport does not rescue the current Phase 1 benchmark

### Improvement 2: Saturation-Resistant Phase 2 Benchmark

- goal: fully test Phase 2 on a benchmark where the exact-ID control cannot already
  saturate category behavior
- motivation: the current YCB held-out category benchmark is valuable as a control, but
  the baseline already reaches `100.0%` category accuracy, so it cannot tell us whether
  similarity-aware transport adds value beyond category-consistent nearest-instance
  confusion
- planned benchmark path: keep the YCB similar-object benchmark as the control, then add
  a stronger transfer benchmark such as a dinner-table-style arrangement task, a held-out
  compositional substitution task, or Omniglot unseen-version transfer with hierarchy
- expected outcome: a benchmark where exact IDs fail by construction but similarity-aware
  transport has a plausible route to generalization

### Improvement 3: Full Phase 2 Similarity Stack

- goal: implement similarity-aware object transport as a first-class mechanism rather
  than as a single experimental LM variant
- motivation: upward hierarchy, multimodal bridging, and top-down biasing should all be
  able to consume a richer similarity carrier than the current scalar placeholder
- implementation targets:
  `object_id` transport should be pluggable between scalar IDs, SDRs, and direct
  evidence-similarity vectors; hierarchical configs should support SDR fitting before
  downstream use; multimodal and top-down experiments should reuse the same carrier once
  the bottom-up path is validated
- expected outcome: a reusable Phase 2 substrate instead of a one-off YCB SDR branch

### Improvement 4: Lateral Associative Voting

- goal: extend similarity-aware communication to same-level LM-to-LM voting after the
  upward hierarchy path is stable
- motivation: same-level communication has a different role from upward hierarchy and
  should only be generalized after the lower-risk hierarchical transport path is working
- implementation target: learned associative voting or mapping between peer LM object
  representations, potentially using SDR-style encodings as one candidate substrate
- sequencing rule: do this last, after Improvements 1 through 3 have produced stable
  upward and benchmarked similarity transport

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
- prepare command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py prepare --experiment phase2_ycb_category_control_train --num-parallel 1`
- launch command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py launch --manifest /Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_ycb_category_control_train/phase0_manifest.json`
- monitor command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py status --manifest /Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_ycb_category_control_train/phase0_manifest.json`
- expected output path: `/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_ycb_category_control_train`
- resume or recovery command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py resume --manifest /Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_ycb_category_control_train/phase0_manifest.json`
- stop command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py stop --manifest /Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_ycb_category_control_train/phase0_manifest.json`
- resource profile: `num_parallel=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`
- current status: running after local YCB Habitat assets were provisioned and the manifest was resumed successfully
- dataset monitor command: `tail -n 40 /Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/dataset_logs/phase2_ycb_dataset_download.log`

### Run B: YCB Category Control Held-Out Evaluation

- run name: `phase2_ycb_category_control_eval_holdout`
- purpose: evaluate the control model on held-out similar-object instances with both
  exact-instance and category scoring
- prepare command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py prepare --experiment phase2_ycb_category_control_eval_holdout --num-parallel 1`
- launch command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py launch --manifest /Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_ycb_category_control_eval_holdout/phase0_manifest.json`
- monitor command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py status --manifest /Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_ycb_category_control_eval_holdout/phase0_manifest.json`
- expected output path: `/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_ycb_category_control_eval_holdout`
- resume or recovery command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py resume --manifest /Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_ycb_category_control_eval_holdout/phase0_manifest.json`
- stop command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py stop --manifest /Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_ycb_category_control_eval_holdout/phase0_manifest.json`
- resource profile: `num_parallel=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`
- current status: completed; `eval_stats.csv` and `category_metrics.json` written under `/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_ycb_category_control_eval_holdout`
- control result summary: `exact_accuracy=0.0%`, `category_accuracy=100.0%`, `within_category_confusion_rate=100.0%`, `cross_category_confusion_rate=0.0%`
- observed mapping pattern: held-out cups mapped to `e_cups`, `knife` mapped almost entirely to `fork`, and `pudding_box` mapped mostly to `sugar_box`

### Run C: YCB Similarity-Aware Training

- run name: `phase2_ycb_category_sdr_train`
- purpose: train a similarity-aware category-transfer model using `EvidenceSDRGraphLM`
  or an equivalent evidence-similarity transport path
- prepare command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py prepare --experiment phase2_ycb_category_sdr_train --override ++experiment.config.logging.output_dir=/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs --override ++experiment.config.logging.run_name=phase2_ycb_category_sdr_train --override ++experiment.config.logging.wandb_handlers=[] --num-parallel 1`
- launch command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py launch --manifest /Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_ycb_category_sdr_train/phase0_manifest.json`
- monitor command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py status --manifest /Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_ycb_category_sdr_train/phase0_manifest.json`
- expected output path: `/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_ycb_category_sdr_train`
- resume or recovery command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py resume --manifest /Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_ycb_category_sdr_train/phase0_manifest.json`
- stop command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py stop --manifest /Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_ycb_category_sdr_train/phase0_manifest.json`
- resource profile: `num_parallel=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`
- current status: config implemented, manifest prepared, and training launched after clearing inherited displacement-only learning-module args

### Run D: YCB Similarity-Aware Held-Out Evaluation

- run name: `phase2_ycb_category_sdr_eval_holdout`
- purpose: evaluate the similarity-aware category model on held-out similar-object
  instances
- prepare command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py prepare --experiment phase2_ycb_category_sdr_eval_holdout --override ++experiment.config.logging.output_dir=/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs --override ++experiment.config.logging.run_name=phase2_ycb_category_sdr_eval_holdout --override ++experiment.config.logging.wandb_handlers=[] --num-parallel 1`
- launch command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py launch --manifest /Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_ycb_category_sdr_eval_holdout/phase0_manifest.json`
- monitor command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py status --manifest /Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_ycb_category_sdr_eval_holdout/phase0_manifest.json`
- expected output path: `/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_ycb_category_sdr_eval_holdout`
- resume or recovery command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py resume --manifest /Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_ycb_category_sdr_eval_holdout/phase0_manifest.json`
- stop command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py stop --manifest /Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_ycb_category_sdr_eval_holdout/phase0_manifest.json`
- resource profile: `num_parallel=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`
- current status: config implemented and manifest prepared; waiting on SDR training output

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

### Run F: YCB Similarity-Aware Serial Training

- run name: `phase2_ycb_category_sdr_train_serial_fix2`
- purpose: train the SDR model in a single `run.py` process so one LM instance can see all six training objects and accumulate pairwise overlap targets
- launch command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python run.py experiment=phase2_ycb_category_sdr_train ++experiment.config.logging.run_name=phase2_ycb_category_sdr_train_serial_fix2`
- monitor command: `ps -ax -o pid,etime,command | grep 'phase2_ycb_category_sdr_train_serial_fix2' | grep -v grep || true`
- log command: `tail -n 80 /Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_ycb_category_sdr_train_serial_fix2/run.log`
- expected output path: `/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_ycb_category_sdr_train_serial_fix2`
- stop command: `pkill -f 'phase2_ycb_category_sdr_train_serial_fix2'`
- resource profile: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`
- current status: completed; checkpoint contains `6` graphs and `6` learned SDR embeddings

### Run G: YCB Similarity-Aware Serial Held-Out Evaluation

- run name: `phase2_ycb_category_sdr_eval_holdout_serial_fix2`
- purpose: evaluate the first SDR checkpoint that preserved all learned SDR embeddings at eval time
- launch command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python run.py experiment=phase2_ycb_category_sdr_eval_holdout ++experiment.config.logging.run_name=phase2_ycb_category_sdr_eval_holdout_serial_fix2 ++experiment.config.model_name_or_path=/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_ycb_category_sdr_train_serial_fix2/pretrained/`
- monitor command: `ps -ax -o pid,etime,command | grep 'phase2_ycb_category_sdr_eval_holdout_serial_fix2' | grep -v grep || true`
- log command: `tail -n 80 /Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_ycb_category_sdr_eval_holdout_serial_fix2/run.log`
- expected output path: `/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_ycb_category_sdr_eval_holdout_serial_fix2`
- stop command: `pkill -f 'phase2_ycb_category_sdr_eval_holdout_serial_fix2'`
- resource profile: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`
- current status: completed; `eval_stats.csv` written and category metrics computed

### Run H: YCB Eval-Style SDR Fitting On Training Objects

- run name: `phase2_ycb_category_sdr_fit_eval_train_objects_pilot1`
- purpose: fit SDR overlap targets during evaluation-style matching episodes on the six training objects while leaving graph memory fixed, then save the fitted checkpoint at the end
- launch command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python run.py experiment=phase2_ycb_category_sdr_fit_eval_train_objects ++experiment.config.n_eval_epochs=1 ++experiment.config.logging.run_name=phase2_ycb_category_sdr_fit_eval_train_objects_pilot1`
- monitor command: `tail -f /Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_ycb_category_sdr_fit_eval_train_objects_pilot1/run.log`
- expected output path: `/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_ycb_category_sdr_fit_eval_train_objects_pilot1`
- stop command: `pkill -f 'phase2_ycb_category_sdr_fit_eval_train_objects_pilot1'`
- resource profile: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`
- current status: completed; checkpoint written with non-zero `target_overlaps.counts`, proving that eval-style fitting can accumulate real pairwise overlap targets

### Run I: YCB Eval-Style Fitted SDR Held-Out Evaluation

- run name: `phase2_ycb_category_sdr_eval_holdout_fit_eval_full1`
- purpose: evaluate the checkpoint produced by eval-style SDR fitting on the held-out similar-object instances
- launch command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python run.py experiment=phase2_ycb_category_sdr_eval_holdout ++experiment.config.logging.run_name=phase2_ycb_category_sdr_eval_holdout_fit_eval_full1 ++experiment.config.model_name_or_path=/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_ycb_category_sdr_fit_eval_train_objects_pilot1`
- monitor command: `tail -f /Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_ycb_category_sdr_eval_holdout_fit_eval_full1/run.log`
- expected output path: `/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_ycb_category_sdr_eval_holdout_fit_eval_full1`
- stop command: `pkill -f 'phase2_ycb_category_sdr_eval_holdout_fit_eval_full1'`
- resource profile: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`
- current status: completed; full `56/56` held-out episodes evaluated and category metrics computed

### Run J: Level-1 Compositional SDR Bridge Training

- run name: `phase2_comp_sdr_lvl1_train`
- purpose: test whether stacked SDR-based object transport improves the existing
  level-1 compositional hierarchy by replacing the placeholder scalar `object_id`
  feature with similarity-aware SDRs in both the lower and higher LM
- launch command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python run.py experiment=phase2_comp_sdr_lvl1_train`
- monitor command: `tail -f /Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_comp_sdr_lvl1_train/log.txt`
- expected output path: `/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_comp_sdr_lvl1_train`
- stop command: `pkill -f 'phase2_comp_sdr_lvl1_train'`
- resource profile: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`
- current status: pilot training stabilized as `phase2_comp_sdr_lvl1_train_pilot2`, and full training completed as `phase2_comp_sdr_lvl1_train_full2`
- pilot result summary: both LM0 and LM1 checkpoints were written with SDR-specific state; LM1 learned parent-object graphs for all six level-1 compositional training objects
- full result summary: `phase2_comp_sdr_lvl1_train_full2` wrote `model.pt`, `config.pt`, and `exp_state_dict.pt`; LM0 preserved `4` child graphs and LM1 preserved `6` parent graphs, with SDR state saved at both hierarchy levels
- implementation note: LM1 required a reduced parent graph grid (`num_model_voxels_per_dim: 20`) because graph reconstruction densifies feature grids and otherwise a `200^3 x 2048` SDR-valued `object_id` channel is not tractable

### Run K: Level-1 Compositional SDR Bridge Evaluation

- run name: `phase2_comp_sdr_lvl1_eval`
- purpose: evaluate the level-1 compositional SDR hierarchy against the existing
  compositional benchmark once the bridge training run is available
- launch command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python run.py experiment=phase2_comp_sdr_lvl1_eval`
- monitor command: `tail -f /Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_comp_sdr_lvl1_eval/log.txt`
- expected output path: `/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_comp_sdr_lvl1_eval`
- stop command: `pkill -f 'phase2_comp_sdr_lvl1_eval'`
- resource profile: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`
- current status: pilot evaluation completed as `phase2_comp_sdr_lvl1_eval_pilot4`, and full evaluation completed as `phase2_comp_sdr_lvl1_eval_full2` against the `phase2_comp_sdr_lvl1_train_full2` checkpoint
- pilot result summary: `12` pilot eval rows completed with `3` exact or MLH-correct cases, `2` `consistent_child_obj` cases, `6` confused or confused-MLH cases, and `1` `no_match`
- full result summary: `168` eval rows completed with `27` `correct`, `18` `correct_mlh`, `82` `consistent_child_obj`, `39` `confused_mlh`, and `2` `confused`, which is `26.8%` exact-or-MLH correct, `48.8%` `consistent_child_obj`, and `24.4%` confused or confused-MLH
- implementation note: ordinary eval now freezes SDR encoders by default; the earlier pilot emitted overflow warnings because the SDR LM was still fitting during eval, and the special fit-on-eval path is now opt-in only

### Run L: Level-1 Compositional Frozen-LM0 Diagnostic

- run name: `phase2_comp_sdr_lvl1_train_lm0_frozen_full1`
- purpose: test whether lower-level SDR drift is a primary cause of the stacked bridge failure by preserving LM0's learned SDR vocabulary while resetting and retraining only LM1 parent state
- launch command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python run.py experiment=phase2_comp_sdr_lvl1_train ++experiment.config.logging.run_name=phase2_comp_sdr_lvl1_train_lm0_frozen_full1 ++experiment.config.model_name_or_path=/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_comp_sdr_lvl1_lm0_frozen_seed/pretrained/ ++experiment.config.monty_config.learning_module_configs.learning_module_0.learning_module_args.sdr_args.n_sdr_epochs=0`
- eval command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python run.py experiment=phase2_comp_sdr_lvl1_eval ++experiment.config.logging.run_name=phase2_comp_sdr_lvl1_eval_lm0_frozen_full1 ++experiment.config.model_name_or_path=/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_comp_sdr_lvl1_train_lm0_frozen_full1/pretrained/`
- expected output path: `/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_comp_sdr_lvl1_train_lm0_frozen_full1` and `/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_comp_sdr_lvl1_eval_lm0_frozen_full1`
- current status: completed
- result summary: training completed with LM0 preserving `4` child graphs and LM1 preserving `6` parent graphs; the matching full eval completed all `168` rows with `27` `correct`, `18` `correct_mlh`, `83` `consistent_child_obj`, `38` `confused_mlh`, and `2` `confused`, which is still `26.8%` exact-or-MLH correct
- interpretation note: freezing LM0 during parent retraining shifted only one case from `confused_mlh` to `consistent_child_obj` and did not improve strict parent accuracy

### Run M: Level-1 Compositional Top-K Union Pilot

- run name: `phase2_comp_sdr_lvl1_topk_train_pilot1`
- purpose: test whether replacing one-best child transport with a top-k union SDR carrier improves the stacked compositional pilot before changing the higher-level learning objective
- launch command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python run.py experiment=phase2_comp_sdr_lvl1_train ++experiment.config.n_train_epochs=1 ++experiment.config.logging.run_name=phase2_comp_sdr_lvl1_topk_train_pilot1`
- eval command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python run.py experiment=phase2_comp_sdr_lvl1_eval ++experiment.config.n_eval_epochs=1 ++experiment.config.logging.run_name=phase2_comp_sdr_lvl1_topk_eval_pilot1 ++experiment.config.model_name_or_path=/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_comp_sdr_lvl1_topk_train_pilot1/pretrained/`
- expected output path: `/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_comp_sdr_lvl1_topk_train_pilot1` and `/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_comp_sdr_lvl1_topk_eval_pilot1`
- current status: completed
- result summary: the saved pilot config confirms `learning_module_0.learning_module_args.sdr_args.upward_top_k=3`, and the matching `12`-row eval completed with `2` `correct`, `1` `correct_mlh`, `2` `consistent_child_obj`, `3` `confused_mlh`, `3` `confused`, and `1` `no_match`, which is `25.0%` exact-or-MLH correct, `16.7%` `consistent_child_obj`, and `50.0%` confused or confused-MLH
- interpretation note: this pilot was row-identical to the earlier `phase2_comp_sdr_lvl1_eval_pilot4` baseline, so a top-k union carrier alone did not shift parent behavior at pilot scale

### Run N: Level-1 Compositional Parent-Support Pilot

- run name: `phase2_comp_sdr_lvl1_support_train_pilot1d`
- purpose: test whether adding a quantitative `object_support` feature to LM0 output and teaching LM1 to score it alongside `object_id` changes parent behavior at pilot scale
- launch command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python run.py experiment=phase2_comp_sdr_lvl1_train ++experiment.config.n_train_epochs=1 ++experiment.config.logging.run_name=phase2_comp_sdr_lvl1_support_train_pilot1d`
- eval command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python run.py experiment=phase2_comp_sdr_lvl1_eval ++experiment.config.n_eval_epochs=1 ++experiment.config.logging.run_name=phase2_comp_sdr_lvl1_support_eval_pilot1d ++experiment.config.model_name_or_path=/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_comp_sdr_lvl1_support_train_pilot1d/pretrained/`
- expected output path: `/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_comp_sdr_lvl1_support_train_pilot1d` and `/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_comp_sdr_lvl1_support_eval_pilot1d`
- current status: completed
- result summary: the train pilot completed cleanly and LM1 parent graphs now store both `object_id` and `object_support` (`2059:2063`) in the `learning_module_0` channel, but the matching `12`-row eval still completed with `2` `correct`, `1` `correct_mlh`, `2` `consistent_child_obj`, `3` `confused_mlh`, `3` `confused`, and `1` `no_match`, which is again `25.0%` exact-or-MLH correct, `16.7%` `consistent_child_obj`, and `50.0%` confused or confused-MLH
- interpretation note: this pilot was also row-identical to `phase2_comp_sdr_lvl1_eval_pilot4`, so adding parent-side quantitative child support as an extra feature still did not change pilot behavior

### Run O: Level-1 Compositional Temporal-Support Accumulator Pilot

- run name: `phase2_comp_sdr_lvl1_temporal_train_pilot1`
- purpose: test whether LM1 should accumulate lower-level `object_support` evidence over episode time instead of matching only the latest child support snapshot
- launch command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python run.py experiment=phase2_comp_sdr_lvl1_train ++experiment.config.n_train_epochs=1 ++experiment.config.logging.run_name=phase2_comp_sdr_lvl1_temporal_train_pilot1`
- eval command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python run.py experiment=phase2_comp_sdr_lvl1_eval ++experiment.config.n_eval_epochs=1 ++experiment.config.logging.run_name=phase2_comp_sdr_lvl1_temporal_eval_pilot1 ++experiment.config.model_name_or_path=/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_comp_sdr_lvl1_temporal_train_pilot1/pretrained/`
- expected output path: `/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_comp_sdr_lvl1_temporal_train_pilot1` and `/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_comp_sdr_lvl1_temporal_eval_pilot1`
- current status: completed
- result summary: the train pilot completed cleanly with LM1 configured to temporally accumulate incoming `learning_module_0` `object_support` and smoothed `object_id` features before matching, but the matching `12`-row eval again completed with `2` `correct`, `1` `correct_mlh`, `2` `consistent_child_obj`, `3` `confused_mlh`, `3` `confused`, and `1` `no_match`, which is still `25.0%` exact-or-MLH correct, `16.7%` `consistent_child_obj`, and `50.0%` confused or confused-MLH
- interpretation note: this pilot was also row-identical to `phase2_comp_sdr_lvl1_eval_pilot4`, so accumulating the transported child support over time without changing the parent objective still did not shift pilot behavior

### Run P: Level-1 Compositional Graph-Support Objective Full Run

- run name: `phase2_comp_sdr_lvl1_graphobj_train_full2` / `phase2_comp_sdr_lvl1_graphobj_eval_full2`
- purpose: change the parent-side objective itself by giving LM1 a graph-level evidence bonus when the accumulated child `object_support` matches a parent graph's stored support prototype
- launch command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python run.py experiment=phase2_comp_sdr_lvl1_train ++experiment.config.logging.run_name=phase2_comp_sdr_lvl1_graphobj_train_full2`
- eval command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python run.py experiment=phase2_comp_sdr_lvl1_eval ++experiment.config.logging.run_name=phase2_comp_sdr_lvl1_graphobj_eval_full2 ++experiment.config.model_name_or_path=/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_comp_sdr_lvl1_graphobj_train_full2/pretrained/`
- expected output path: `/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_comp_sdr_lvl1_graphobj_train_full2` and `/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_comp_sdr_lvl1_graphobj_eval_full2`
- current status: completed
- training note: the first full training attempt (`phase2_comp_sdr_lvl1_graphobj_train_full1`) failed while extending LM1 graph memory because adding the new `object_support` feature block widened the feature grid without widening the sparse update tensor shape. Fixing `GridObjectModel` to resize sparse feature grids before mixed-schema updates removed that runtime blocker, and `phase2_comp_sdr_lvl1_graphobj_train_full2` then completed cleanly with `model.pt`, `config.pt`, and `exp_state_dict.pt` written.
- result summary: the matching full eval completed all `168` rows with `27` `correct`, `28` `correct_mlh`, `77` `consistent_child_obj`, `36` `confused_mlh`, and `0` `confused`, which is `32.7%` exact-or-MLH correct, `45.8%` `consistent_child_obj`, and `21.4%` confused or confused-MLH
- comparison note: relative to the earlier full bridge eval (`phase2_comp_sdr_lvl1_eval_full2`), this changed `37/168` rows and improved strict exact-or-MLH parent accuracy by `10` rows (`45 -> 55`, `26.8% -> 32.7%`)
- interpretation note: this is the first parent-side mechanism in the stacked SDR hierarchy that produced a full-scale strict accuracy improvement rather than a row-identical null result or a pure reshuffling of pilot errors

### Run Q: Level-1 Compositional Graph-Support Objective With Frozen LM0

- run name: `phase2_comp_sdr_lvl1_graphobj_lm0_frozen_full1` /
  `phase2_comp_sdr_lvl1_graphobj_eval_lm0_frozen_full1`
- purpose: causal follow-up to test whether the graph-support-objective gain depends on
  LM0 changing during retraining, or whether LM1 can recover the same improvement while
  LM0 is held fixed
- launch command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python run.py experiment=phase2_comp_sdr_lvl1_train ++experiment.config.logging.run_name=phase2_comp_sdr_lvl1_graphobj_lm0_frozen_full1 ++experiment.config.model_name_or_path=/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_comp_sdr_lvl1_lm0_frozen_seed/pretrained/ ++experiment.config.monty_config.learning_module_configs.learning_module_0.learning_module_args.sdr_args.n_sdr_epochs=0`
- eval command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python run.py experiment=phase2_comp_sdr_lvl1_eval ++experiment.config.logging.run_name=phase2_comp_sdr_lvl1_graphobj_eval_lm0_frozen_full1 ++experiment.config.model_name_or_path=/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_comp_sdr_lvl1_graphobj_lm0_frozen_full1/pretrained/`
- expected output path: `/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_comp_sdr_lvl1_graphobj_lm0_frozen_full1` and `/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_comp_sdr_lvl1_graphobj_eval_lm0_frozen_full1`
- current status: completed
- result summary: the matching full eval completed all `168` rows with `27` `correct`, `28` `correct_mlh`, `79` `consistent_child_obj`, `34` `confused_mlh`, and `0` `confused`, which is `32.7%` exact-or-MLH correct, `47.0%` `consistent_child_obj`, and `20.2%` confused or confused-MLH
- comparison note: relative to the earlier full bridge eval (`phase2_comp_sdr_lvl1_eval_full2`), this changed `38/168` rows and again improved strict exact-or-MLH parent accuracy by `10` rows (`45 -> 55`, `26.8% -> 32.7%`); relative to the unfrozen graph-objective eval (`phase2_comp_sdr_lvl1_graphobj_eval_full2`), only `4/168` rows changed and the strict exact-or-MLH total stayed identical
- interpretation note: the graph-support-objective gain survives intact when LM0 is frozen. This materially weakens the hypothesis that the improvement is primarily caused by LM0-side SDR drift or LM0 retraining effects, and strengthens the reading that the dominant lever is LM1-side storage and scoring of distributed child evidence

## Latest Results

- control baseline (`phase2_ycb_category_control_eval_holdout`): `exact_accuracy=0.0%`, `category_accuracy=100.0%`, `within_category_confusion_rate=100.0%`, `cross_category_confusion_rate=0.0%`
- first parallel SDR eval (`phase2_ycb_category_sdr_eval_holdout` and `phase2_ycb_category_sdr_eval_holdout_fix1`): `exact_accuracy=0.0%`, `category_accuracy=53.57%`, `within_category_confusion_rate=53.57%`, `cross_category_confusion_rate=46.43%`
- serial SDR eval with preserved SDR state (`phase2_ycb_category_sdr_eval_holdout_serial_fix2`): `exact_accuracy=0.0%`, `category_accuracy=58.93%`, `within_category_confusion_rate=58.93%`, `cross_category_confusion_rate=41.07%`
- eval-style SDR fitting checkpoint (`phase2_ycb_category_sdr_fit_eval_train_objects_pilot1`): first checkpoint with non-zero learned overlap statistics; the overlap matrix is sparse and only partially populated, but no longer entirely empty
- interrupted held-out pilot note (`phase2_ycb_category_sdr_eval_holdout_fit_eval_pilot1`): reached `100.0%` category accuracy on `54` logged episodes, but this run was interrupted by `KeyboardInterrupt` during evaluation and should not be treated as final
- eval-style fitted SDR held-out eval (`phase2_ycb_category_sdr_eval_holdout_fit_eval_full1`): `exact_accuracy=0.0%`, `category_accuracy=100.0%`, `within_category_confusion_rate=100.0%`, `cross_category_confusion_rate=0.0%`
- compositional SDR bridge training pilot (`phase2_comp_sdr_lvl1_train_pilot2`): completed successfully after reducing the parent LM grid size; the checkpoint preserves SDR state for both the child LM and the parent LM
- compositional SDR bridge full training (`phase2_comp_sdr_lvl1_train_full2`): completed successfully after fixing vector-valued `object_id` parent-memory updates; the final checkpoint wrote `model.pt`, `config.pt`, and `exp_state_dict.pt`, with LM0 saving `4` child graphs and LM1 saving `6` parent graphs plus SDR state at both levels
- compositional SDR bridge eval pilot (`phase2_comp_sdr_lvl1_eval_pilot4`): operational end to end on `12` pilot eval rows, but early accuracy is mixed (`25.0%` exact-or-MLH correct, `16.7%` `consistent_child_obj`, `50.0%` confused or confused-MLH, `8.3%` `no_match`) and does not yet support any improvement claim
- compositional top-k union pilot (`phase2_comp_sdr_lvl1_topk_eval_pilot1`): completed cleanly with the new `upward_top_k=3` carrier enabled in LM0, but the `12` eval rows were row-for-row identical to `phase2_comp_sdr_lvl1_eval_pilot4`, so this ablation did not change pilot behavior
- compositional parent-support pilot (`phase2_comp_sdr_lvl1_support_eval_pilot1d`): completed cleanly with LM1 parent graphs storing both `object_id` and `object_support`, but the `12` eval rows were still row-for-row identical to `phase2_comp_sdr_lvl1_eval_pilot4`, so the extra quantitative support feature also did not change pilot behavior
- compositional temporal-support accumulator pilot (`phase2_comp_sdr_lvl1_temporal_eval_pilot1`): completed cleanly with LM1 temporally accumulating `learning_module_0` support across steps before matching, but the `12` eval rows were still row-for-row identical to `phase2_comp_sdr_lvl1_eval_pilot4`, so temporal accumulation without a different parent objective also did not change pilot behavior
- compositional SDR bridge full eval (`phase2_comp_sdr_lvl1_eval_full2`): completed all `168` eval rows with `27` `correct`, `18` `correct_mlh`, `82` `consistent_child_obj`, `39` `confused_mlh`, and `2` `confused`; the bridge is operational at full scale, but most of the gain lands in child-consistent reuse rather than exact parent-object correctness
- frozen-LM0 compositional eval (`phase2_comp_sdr_lvl1_eval_lm0_frozen_full1`): completed all `168` eval rows with `27` `correct`, `18` `correct_mlh`, `83` `consistent_child_obj`, `38` `confused_mlh`, and `2` `confused`; exact-or-MLH correct stayed flat at `26.8%`
- graph-support-objective full eval (`phase2_comp_sdr_lvl1_graphobj_eval_full2`): completed all `168` eval rows with `27` `correct`, `28` `correct_mlh`, `77` `consistent_child_obj`, `36` `confused_mlh`, and `0` `confused`; this is `32.7%` exact-or-MLH correct, `45.8%` `consistent_child_obj`, and `21.4%` confused or confused-MLH, improving strict parent accuracy by `10` rows over the earlier full bridge baseline
- frozen-LM0 graph-support-objective eval (`phase2_comp_sdr_lvl1_graphobj_eval_lm0_frozen_full1`): completed all `168` eval rows with `27` `correct`, `28` `correct_mlh`, `79` `consistent_child_obj`, `34` `confused_mlh`, and `0` `confused`; this is again `32.7%` exact-or-MLH correct, now with slightly more `consistent_child_obj` and slightly less confused or confused-MLH mass than the unfrozen graph-objective run
- result interpretation: the original supervised-pretraining SDR paths were invalid or underfit for overlap learning, but the eval-style fitting path is operational and recovers the same category-level behavior as the control baseline on the full held-out set. In the stacked compositional hierarchy, carrier-only changes remained null: top-k union, parent-support as an extra feature, and temporal accumulation all failed to move pilot behavior. Changing the parent objective finally did matter. The graph-support objective was first visible as a non-null pilot perturbation, then became the first full-scale strict accuracy win in the stacked bridge, improving exact-or-MLH parent accuracy from `26.8%` to `32.7%`. The completed frozen-LM0 graph-objective follow-up now shows that this gain survives intact when LM0 is held fixed, which shifts the main causal reading even more strongly toward LM1-side storage and scoring of distributed child evidence rather than LM0-side retraining effects.

## Immediate Execution Pass: 2026-03-17

This pass turns the current interpretation into concrete execution rather than more
open-ended speculation.

### Execution Goals

- run the missing frozen-LM0 graph-support-objective causal test to determine whether
  the current full-scale gain survives when LM0 is held fixed
- convert the baseline-versus-graph-objective difference into a reusable changed-row
  report instead of relying only on aggregate counts
- scaffold a stronger saturation-resistant control benchmark so Phase 2 no longer
  depends only on the already-saturated YCB category split

### Execution Artifacts Produced

- changed-row report written to
  `/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_comp_sdr_lvl1_graphobj_vs_base_changed_rows.txt`
- new benchmark configs added for an Omniglot cross-version control path:
  `phase2_omniglot_cross_version_control_train` and
  `phase2_omniglot_cross_version_control_eval`

### Changed-Row Analysis Summary

- `37/168` rows changed between `phase2_comp_sdr_lvl1_eval_full2` and
  `phase2_comp_sdr_lvl1_graphobj_eval_full2`
- the changed rows span all six parent objects rather than concentrating on one corner
  case: `001_cube` (`5`), `002_cube_tbp_horz` (`6`), `004_cube_numenta_horz` (`5`),
  `006_disk` (`7`), `007_disk_tbp_horz` (`8`), and `009_disk_numenta_horz` (`6`)
- the strongest positive transition is `confused_mlh -> correct_mlh` (`9` rows), which
  means the main gain is mostly better resolution of near-miss parent hypotheses rather
  than a broad increase in exact `correct` rows
- the graph-support objective still introduces churn in both directions: `18` improved
  rows, `9` regressed rows, and `10` lateral reshuffles with no ordinal improvement
- the most concerning negative transition is `correct_mlh -> confused_mlh` (`4` rows),
  so the graph-support objective is helping on net but is not yet a uniformly safer
  parent-side scoring rule

### Frozen-LM0 Graph-Objective Status

- run name: `phase2_comp_sdr_lvl1_graphobj_lm0_frozen_full1`
- purpose: causal follow-up to test whether the graph-support-objective gain depends on
  LM0 changing during retraining, or whether LM1 can recover the same improvement while
  LM0 is held fixed
- launch command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python run.py experiment=phase2_comp_sdr_lvl1_train ++experiment.config.logging.run_name=phase2_comp_sdr_lvl1_graphobj_lm0_frozen_full1 ++experiment.config.model_name_or_path=/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_comp_sdr_lvl1_lm0_frozen_seed/pretrained/ ++experiment.config.monty_config.learning_module_configs.learning_module_0.learning_module_args.sdr_args.n_sdr_epochs=0`
- monitor command: `grep -c 'Going from' /Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_comp_sdr_lvl1_graphobj_lm0_frozen_full1/run.log`
- expected output path: `/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_comp_sdr_lvl1_graphobj_lm0_frozen_full1`
- eval output path: `/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_comp_sdr_lvl1_graphobj_eval_lm0_frozen_full1`
- current status: completed
- result summary: `27` `correct`, `28` `correct_mlh`, `79` `consistent_child_obj`, `34` `confused_mlh`, and `0` `confused` over `168` rows; exact-or-MLH parent accuracy stays at `32.7%`
- conclusion: the graph-support-objective gain survives with LM0 frozen, so the main positive lever is not LM0-only

### LM1 Eval-Only Sweep Status

- purpose: test whether a small LM1-side scoring change can keep the frozen-LM0
  graph-objective gain while reducing the remaining cube-versus-disk churn
- baseline frozen graph-objective eval (`phase2_comp_sdr_lvl1_graphobj_eval_lm0_frozen_full1`): `27` `correct`, `28` `correct_mlh`, `79` `consistent_child_obj`, `34` `confused_mlh`, `0` `confused`, `32.7%` exact-or-MLH
- reduced-weight eval (`phase2_comp_sdr_lvl1_graphobj_eval_lm0_frozen_weight1` with `graph_support_objective_weight=1.0`): `27` `correct`, `26` `correct_mlh`, `81` `consistent_child_obj`, `34` `confused_mlh`, `0` `confused`, `31.5%` exact-or-MLH
- tolerance eval (`phase2_comp_sdr_lvl1_graphobj_eval_lm0_frozen_tol01` with `graph_support_objective_tolerance=0.1`): `27` `correct`, `29` `correct_mlh`, `79` `consistent_child_obj`, `33` `confused_mlh`, `0` `confused`, `33.3%` exact-or-MLH
- tolerance eval (`phase2_comp_sdr_lvl1_graphobj_eval_lm0_frozen_tol02` with `graph_support_objective_tolerance=0.2`): `27` `correct`, `29` `correct_mlh`, `78` `consistent_child_obj`, `34` `confused_mlh`, `0` `confused`, `33.3%` exact-or-MLH
- tolerance eval (`phase2_comp_sdr_lvl1_graphobj_eval_lm0_frozen_tol03` with `graph_support_objective_tolerance=0.3`): `27` `correct`, `30` `correct_mlh`, `76` `consistent_child_obj`, `35` `confused_mlh`, `0` `confused`, `33.9%` exact-or-MLH
- reduced-weight comparison versus the frozen graph-objective baseline: `9` changed rows, `2` improved, `3` regressed, and `4` lateral reshuffles; the three regressions include one cube-target `correct_mlh -> consistent_child_obj` and two disk-target `correct_mlh -> confused_mlh`, so lowering weight is not a cleaner parent scoring rule
- tolerance `0.1` comparison versus the frozen graph-objective baseline: only `1` changed row and it is an improvement (`confused_mlh -> correct_mlh`), so this is a small pure gain with slightly lower confused-or-MLH mass
- tolerance `0.2` comparison versus the frozen graph-objective baseline: only `1` changed row and it is an improvement (`consistent_child_obj -> correct_mlh`), giving another small pure gain without measured regressions
- tolerance `0.3` comparison versus the frozen graph-objective baseline: `5` changed rows, `3` improved, `1` regressed, and `1` lateral reshuffle; this yields the best exact-or-MLH point so far, but it also reintroduces a cube-target `correct_mlh -> confused_mlh` regression and increases confused-or-MLH mass overall
- current reading: the immediate follow-up should stay on the LM1-side tolerance/scoring axis rather than weakening the graph-support objective weight, but the likely sweet spot is now around `0.1` to `0.2` if we prioritize cleaner behavior over the slightly larger but noisier `0.3` gain

### LM1 Graph-Support Debug Readout

- debug eval (`phase2_comp_sdr_lvl1_eval_graph_debug_tol01`): matched the clean `tolerance=0.1` result at `33.3%` exact-or-MLH and wrote per-episode LM1 graph-support debug logs under `/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_comp_sdr_lvl1_eval_graph_debug_tol01/sdr_logs_lm1`
- summary artifact: `/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_comp_sdr_lvl1_eval_graph_debug_tol01/graph_support_debug_summary.txt`
- aggregate debug finding: all `84` LM1 episodes produced positive graph-support bonuses, with mean target bonus `1.4590` and mean winner bonus `1.4591`; the top bonus graph matched the final LM1 winner in `39/84` episodes and the target in only `28/84`
- interpretation: at `tolerance=0.1`, the graph-support bonus is still too broad. It is active everywhere, but it usually does not separate the target from the final competing winner because the target and winner bonuses are nearly identical on average
- changed-row detail: the only ordinal improvement versus the frozen graph-objective baseline is episode `67`, where a disk target (`006_disk`) moved from `confused_mlh` to `correct_mlh`, and in that episode the top bonus graph also became the target
- next move implied by the debug output: tighten the tolerance band before doing any broader grid search, starting with `graph_support_objective_tolerance=0.05`, because the current bonus is still rewarding too many competing parent graphs per episode
- debug eval (`phase2_comp_sdr_lvl1_eval_graph_debug_tol005`): completed with local logs preserved under `/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_comp_sdr_lvl1_eval_graph_debug_tol005/sdr_logs_lm1` and summary artifact `/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_comp_sdr_lvl1_eval_graph_debug_tol005/graph_support_debug_summary.txt`
- `tolerance=0.05` result: `32.1%` LM1 exact-or-MLH, `0` changed rows versus the frozen graph-objective baseline, mean positive-bonus count `41.798`, and top-bonus target matches unchanged at `28/84`
- updated interpretation: tightening below `0.1` did not make the graph-support term more selective. It regressed the clean `0.1` gain back to baseline while still rewarding many competing parent graphs, so the lower edge of the useful region is now likely at or above `0.1`
- observability fix: the debug config now writes LM1 logs to a run-name-specific directory so future debug sweeps no longer overwrite each other in the shared `phase2_comp_sdr_lvl1_eval_graph_debug` path
- debug eval (`phase2_comp_sdr_lvl1_eval_graph_debug_tol015`): completed with local logs under `/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_comp_sdr_lvl1_eval_graph_debug_tol015/sdr_logs_lm1` and summary artifact `/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_comp_sdr_lvl1_eval_graph_debug_tol015/graph_support_debug_summary.txt`
- `tolerance=0.15` result: `33.3%` LM1 exact-or-MLH, `1` changed row versus the frozen graph-objective baseline (`consistent_child_obj -> correct_mlh` on `002_cube_tbp_horz`), mean positive-bonus count `37.560`, and the same `28/84` top-bonus target matches as `0.1`
- debug eval (`phase2_comp_sdr_lvl1_eval_graph_debug_tol02`): completed with local logs under `/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_comp_sdr_lvl1_eval_graph_debug_tol02/sdr_logs_lm1` and summary artifact `/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_comp_sdr_lvl1_eval_graph_debug_tol02/graph_support_debug_summary.txt`
- `tolerance=0.2` result: `33.3%` LM1 exact-or-MLH, the same single `consistent_child_obj -> correct_mlh` improvement row as the earlier eval-only `0.2` run, mean positive-bonus count `36.179`, and again `28/84` top-bonus target matches
- updated reading after `0.05`, `0.1`, `0.15`, and `0.2`: the clean region is now best understood as roughly `0.15–0.2`, with `0.2` currently the most selective logged point because it preserves the gain while producing the fewest positive-bonus updates per episode among the clean winners
- joint debug eval (`phase2_comp_sdr_lvl1_eval_graph_debug_w15_tol02` with `graph_support_objective_weight=1.5`, `graph_support_objective_tolerance=0.2`): completed with LM1 logs under `/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_comp_sdr_lvl1_eval_graph_debug_w15_tol02/sdr_logs_lm1` and comparison artifact `/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/phase2_comp_sdr_lvl1_eval_graph_debug_w15_tol02/joint_compare.txt`
- `weight=1.5`, `tolerance=0.2` result: stayed at `33.3%` LM1 exact-or-MLH, improved one frozen-baseline row (`004_cube_numenta_horz` from `confused_mlh -> correct_mlh`), but versus the clean `tolerance=0.2` point it traded one previous win for one regression (`002_cube_tbp_horz` fell back from `correct_mlh -> consistent_child_obj`)
- updated reading after the first joint point: lowering weight from the default `2.0` to `1.5` at `tolerance=0.2` does not look like a clean improvement. It mostly redistributes which cube-confusion row gets corrected, while the aggregate score and mean bonus volume stay essentially unchanged. The next weight probe should therefore move back toward the default, starting with `graph_support_objective_weight=1.75` at `tolerance=0.2`

## Specific Next Steps Plan: 2026-03-17

The calibration pass has now done its job. It showed that the current LM1 graph-support
term can move a few rows, but only by a few rows. That is useful diagnosis, not a path
from `33%` to `99%`. The problem is no longer “find a slightly better scalar weight or
tolerance”; the problem is that LM1 is being asked to infer parents from a message that
is too compressed and with a parent model that is too rigid.

This is the concrete Phase 2 plan from here, optimized for order-of-magnitude gain
rather than marginal cleanup:

1. Freeze the scalar-calibration branch.
  Treat the current `graph_support_objective_tolerance=0.2`, default-weight setup as
  the reference point. Do not spend more cycles on weight/tolerance sweeps unless a
  later architecture change requires re-centering the scorer.

2. Replace the LM0-to-LM1 carrier with a message that preserves ambiguity.
  The current upward path behaves too much like a compressed winner. The next message
  should carry a compact hypothesis packet: top-k child candidates, confidence or
  support weights, and the SDR/support information associated with each hypothesis.
  The immediate goal is not elegance; it is giving LM1 access to the uncertainty it is
  currently missing.

3. Make LM1 score parent graphs against sets of child hypotheses, not a single support
  prototype.
  The first parent-side change should be a multi-prototype or mixture-style support
  representation per parent graph, or an equivalent scorer that can reward consistency
  with several plausible child states. The current single support prototype is too rigid
  for near-sibling parents like the cube variants.

4. Train the parent scorer against the actual hard negatives.
  The big remaining errors are not generic confusion; they are highly structured sibling
  confusions. The training objective should explicitly separate targets from the nearest
  parent competitors that share child evidence, rather than only increasing generic
  support similarity. This is the first route that plausibly changes dozens of rows
  rather than one or two.

5. Add temporal or multi-observation aggregation only after the message and parent scorer
  are upgraded.
  Temporal accumulation on the current carrier is unlikely to close the gap to `99%`
  because it mostly reinforces the same compressed ambiguity. Once LM1 can consume a
  richer hypothesis packet, then temporal aggregation becomes meaningful.

6. Use a success ladder that forces the work to target step changes, not cosmetic gains.
  The target progression should be: `33% -> 50%+` from richer LM0-to-LM1 messages,
  `50%+ -> 70%+` from LM1 multi-prototype or mixture scoring, `70%+ -> 90%+` from hard
  negative training plus better parent evidence aggregation, and only then pursue the
  final residual-error work needed to approach `99%`.

Immediate implementation order:

1. Make the current compression point explicit in code and logs.
  The sender-side seam is `EvidenceSDRGraphLM.get_output()`, which currently collapses
  child ambiguity into one upward `object_id` carrier plus one normalized
  `object_support` vector. The first code task is to expose the exact ranked top-k
  child packet being compressed there so later message upgrades can be compared against
  the current baseline directly.
2. Add a structured top-k LM0-to-LM1 message path beside the current carrier.
  The receiver-side seam is the LM-observation preprocessing path in
  `EvidenceSDRLMMixin._preprocess_temporal_lm_observations()`. The next behavior
  change should thread a top-k child hypothesis packet through this path without
  deleting the current carrier yet, so eval-only experiments can isolate whether the
  richer message changes parent ranking at all.
3. If parent ranking moves materially, implement the LM1 multi-prototype or mixture
  scorer so it can exploit that richer message.
4. After that, retrain the parent-side objective with explicit hard-negative separation.

Implementation update:

- Step 1 is complete. `EvidenceSDRGraphLM.get_output()` now emits a structured
  `upward_hypothesis_packet` alongside the legacy `object_id` and `object_support`
  carriers, and the packet includes the sender-side child SDR vectors for each ranked
  hypothesis rather than only graph ids.
- Step 2 has now started in code. LM-side preprocessing in
  `EvidenceSDRLMMixin._preprocess_temporal_lm_observations()` no longer just carries
  the raw nested packet through; it materializes fixed numeric receiver features
  `object_id_rank_0..2` from the packet and strips the raw packet before matching or
  graph storage.
- The stacked SDR train and inference configs now assign tolerances and non-zero
  feature weights to those rank-slot features, so new LM1 training runs can actually
  learn and score against them.
- This is intentionally still the first consumption path, not the final scorer.
  Existing checkpoints remain backward-compatible because old parent graphs simply do
  not contain the new rank-slot features, so they will be ignored at eval time.
- The next causal question is now sharply defined: after retraining LM1 with these
  rank-slot child SDR features, does parent ranking move materially beyond the current
  `33%` regime, or is the next real bottleneck already the parent scorer itself?
- The first end-to-end retraining check is now running as
  `phase2_comp_sdr_lvl1_packet_train_pilot1` with `n_train_epochs=1`. The first
  relaunch exposed a pre-existing startup crash in `get_evidence_for_each_graph()`
  when LM output was requested before any graph ids were available; that path is now
  guarded in the SDR mixin and the pilot proceeds into training.

Packet-message pilot update:

- The first packet pilot (`phase2_comp_sdr_lvl1_packet_train_pilot1` /
  `phase2_comp_sdr_lvl1_packet_eval_pilot1`) was invalid as a scientific readout.
  The saved LM1 parent graphs contained `object_id_rank_*` feature blocks, but those
  blocks and the base `object_id` block were all zeros. Root cause: LM0 could emit
  upward SDR-valued child features before its `obj2id` / `id2obj` registry had been
  synchronized from loaded graph memory.
- That sender-side sync bug is now fixed in `EvidenceSDRLMMixin.get_output()` and on
  LM state load. Focused unit coverage was extended, and the SDR LM unit suite passes
  with the new regression test.
- The corrected pilot (`phase2_comp_sdr_lvl1_packet_train_pilot2` /
  `phase2_comp_sdr_lvl1_packet_eval_pilot2`) now stores non-zero `object_id`,
  `object_support`, and `object_id_rank_0..2` blocks in the LM1
  `learning_module_0` graph channel, so this is the first valid end-to-end packet
  consumption readout.
- Result: the corrected packet pilot still regresses the 6-row LM1 pilot from
  `1/6 = 16.7%` exact-or-MLH on the old baseline pilot to `0/6 = 0.0%`, with all six
  rows landing in `confused_mlh`.
- Eval-only rank-weight ablation (`phase2_comp_sdr_lvl1_packet_eval_pilot2_rank0`)
  sets `object_id_rank_0..2` feature weights to `0.0` while using the corrected
  packet-trained checkpoint. It remains `0/6 = 0.0%` and row-identical to the
  corrected packet pilot.
- Packet scorer audit: LM1 `object_id_rank_*` features were initially being scored
  incorrectly because the SDR feature calculator only treated the literal
  `object_id` key as an SDR overlap feature. That meant the rank slots fell back to
  dense similarity, and with the configured overlap tolerance of `16` they were
  effectively contributing zero evidence. Fixing the calculator so
  `object_id_rank_*` also uses SDR overlap semantics, then rerunning the eval-only
  packet checkpoint as `phase2_comp_sdr_lvl1_packet_eval_pilot2_rankfix`, still
  leaves LM1 at `0/6 = 0.0%` with the same row-identical decisions as the earlier
  corrected packet pilot.
- Train-plus-eval rank-weight ablation
  (`phase2_comp_sdr_lvl1_packet_train_pilot3_rank0` /
  `phase2_comp_sdr_lvl1_packet_eval_pilot3_rank0`) sets
  `object_id_rank_0..2` feature weights to `0.0` during both LM1 training and eval.
  It still remains `0/6 = 0.0%`, and the LM1 rows are decision-identical to the
  corrected packet pilot apart from small runtime or evidence-value drift.
- No-packet control (`phase2_comp_sdr_lvl1_packet_train_pilot4_nopacket` /
  `phase2_comp_sdr_lvl1_packet_eval_pilot4_nopacket`) disables LM1 packet
  materialization entirely by forcing `upward_packet_top_k=0` under the current code
  path. It still remains `0/6 = 0.0%` and is again LM1 decision-identical to the
  corrected packet pilot.
- Packet weight follow-up (`phase2_comp_sdr_lvl1_packetweight_train_pilot1b` /
  `phase2_comp_sdr_lvl1_packetweight_eval_pilot1b`) augments each packet rank slot
  with an explicit scalar `object_rank_weight_*` feature so LM1 can store and match
  both the ranked child SDRs and their support weights. This also remains
  `0/6 = 0.0%` and row-identical to the corrected packet pilot, so richer packet-side
  weighting still does not rescue the pilot regime.
- Old-regime recreation (`phase2_comp_sdr_lvl1_train_pilot5_objectid_only` /
  `phase2_comp_sdr_lvl1_eval_pilot5_objectid_only`) strips LM1 back to an
  object-id-only matcher under the current code: no packet, no graph-support
  objective, no temporal support path, and zeroed extra LM1 feature weights. That
  control recovers the old pilot regime at `1/6 = 16.7%` exact-or-MLH, with the same
  single correct LM1 row on `007_disk_tbp_horz`.
- Minimal graph-support control (`phase2_comp_sdr_lvl1_train_pilot6_graphobj_min` /
  `phase2_comp_sdr_lvl1_eval_pilot6_graphobj_min`) keeps packet disabled and keeps
  LM0 `object_id` one-best while re-enabling LM1 `object_support`, temporal support,
  and graph-support scoring. This also lands at `1/6 = 16.7%` exact-or-MLH, but now
  the single recovered row is `004_cube_numenta_horz` rather than `007_disk_tbp_horz`.
- Support-top-k control (`phase2_comp_sdr_lvl1_train_pilot7_graphobj_support3` /
  `phase2_comp_sdr_lvl1_eval_pilot7_graphobj_support3`) keeps packet disabled and
  keeps `object_id` one-best, but widens only the lower-level `object_support`
  carrier to top-3 while keeping the minimal graph-support stack active. This drops
  back to `0/6 = 0.0%`, with all six LM1 rows in `confused_mlh`.
- No-temporal support-top-k control
  (`phase2_comp_sdr_lvl1_train_pilot8_support3_notemporal` /
  `phase2_comp_sdr_lvl1_eval_pilot8_support3_notemporal`) keeps packet disabled,
  keeps `object_id` one-best, keeps top-3 `object_support`, and keeps
  graph-support scoring active, but turns temporal accumulation off. This recovers to
  `1/6 = 16.7%` exact-or-MLH, with the recovered row again on `007_disk_tbp_horz`.
- No-graph-support support-top-k control
  (`phase2_comp_sdr_lvl1_train_pilot9_support3_nographobj` /
  `phase2_comp_sdr_lvl1_eval_pilot9_support3_nographobj`) keeps packet disabled,
  keeps `object_id` one-best, keeps top-3 `object_support`, and keeps temporal
  accumulation off, but turns graph-support scoring off as well. This also lands at
  `1/6 = 16.7%` exact-or-MLH, again recovering `007_disk_tbp_horz`.
- Interpretation: the current regression is not just eval-time over-weighting of the
  new rank-slot features. More strongly: packet materialization itself is not the
  operative cause of the `0/6` pilot regression.
- Refined interpretation: the regression was introduced by the broader LM1 enriched
  matching regime that was layered in around packet work, not by the packet transport
  patch in isolation. The remaining live suspects are the support-carrying and
  graph-support stack: `object_support`, temporal support accumulation, graph-support
  scoring, and any interaction between those mechanisms and one-epoch pilot training.
- Stronger packet interpretation after the scorer audit: packet transport is now a
  confirmed null path in two senses. It stays null when rank slots are scored
  correctly as SDR overlaps, and it stays null when those rank slots are augmented
  with explicit support-weight features. That strengthens the current bottleneck read
  that LM1's stored `object_support` target, not missing packet detail, is the main
  failure point in this pilot regime.
- Sharper interpretation: graph-support alone is not the regressor. The pilot only
  falls back to the bad `0/6` regime when ambiguous lower-level support is reintroduced.
  In the current evidence, the decisive trigger is support ambiguity on top of the
  graph-support stack, not packet transport and not one-best-support graph-objective
  scoring by itself.
- Refined interpretation: ambiguous support alone is still not enough. The current
  pilot regression requires at least the combination of ambiguous top-k
  `object_support` with temporal accumulation active. Turning temporal accumulation off
  recovers the old `1/6` pilot regime even while graph-support stays on.
  - Temporal-only ambiguous-support control
    (`phase2_comp_sdr_lvl1_train_p10_temporal_only` /
    `phase2_comp_sdr_lvl1_eval_p10_temporal_only`) keeps packet disabled, keeps
    `object_id` one-best, keeps top-3 `object_support`, turns temporal accumulation
    back on, and leaves graph-support scoring off. This again lands at
    `1/6 = 16.7%` exact-or-MLH and is decision-identical to the no-graph-support
    control, including the same single recovered `007_disk_tbp_horz` row.
  - Final interpretation: temporal accumulation alone is not enough to trigger the bad
    pilot regime under ambiguous support. The `0/6` collapse requires the interaction
    between ambiguous top-k `object_support`, temporal accumulation, and graph-support
    scoring together. Packet transport and rank-slot consumption were red herrings for
    this specific regression.
- LM1 graph-support debug sweep in the actual failure regime:
  `phase2_comp_sdr_lvl1_eval_graph_debug_p7_t0`,
  `phase2_comp_sdr_lvl1_eval_graph_debug_p7_t01`,
  `phase2_comp_sdr_lvl1_eval_graph_debug_p7_t02`, and
  `phase2_comp_sdr_lvl1_eval_graph_debug_p7_w175_t02` all reused the
  `phase2_comp_sdr_lvl1_train_pilot7_graphobj_support3` checkpoint and wrote
  per-episode LM1 payloads under each run's `sdr_logs_lm1` directory.
- Failure-regime debug readout: `tolerance=0.0` and `0.1` both stay collapsed at
  `0/6` exact-or-MLH with all six LM1 rows still `confused_mlh`. The graph-support
  bonus remains extremely broad there, with mean positive-bonus counts of `146.0`
  and `136.2` graph updates per episode respectively, mean target bonus slightly
  below mean winner bonus (`1.3929` vs `1.3962` at `0.0`, `1.3484` vs `1.3521` at
  `0.1`), top-bonus target matches only `1/6`, and top-bonus winner matches only
  `2/6`.
- Selectivity recovery point: `tolerance=0.2`
  (`phase2_comp_sdr_lvl1_eval_graph_debug_p7_t02`) is the first non-collapsed point.
  It recovers to `1/6 = 16.7%` exact-or-MLH, reduces mean positive-bonus count to
  `127.2`, and nearly equalizes mean target and winner bonus (`1.2938` vs `1.2936`).
  Relative to the collapsed pilot7 baseline it changes two rows: `006_disk` softens
  from `confused_mlh` to `confused`, and `007_disk_tbp_horz` flips from
  `confused_mlh` to `correct_mlh`; in that recovered row the top bonus graph is also
  the target and final winner.
- Weight-only follow-up: lowering the weight to `1.75` at `tolerance=0.2`
  (`phase2_comp_sdr_lvl1_eval_graph_debug_p7_w175_t02`) keeps the same aggregate
  `1/6` score and the same mean positive-bonus count (`127.2`), but it does not make
  the bonus cleaner. Instead it shifts the single rescued row from `007_disk_tbp_horz`
  to `009_disk_numenta_horz`, while mean target bonus falls back below mean winner
  bonus (`1.1210` vs `1.1246`).
- Updated interpretation from the failure-regime sweep: under ambiguous top-3
  support, graph-support tolerance is the useful selectivity lever and lower weight
  alone is not. The collapse at `0.0` and `0.1` happens because the bonus stays too
  diffuse across too many parent graphs; `0.2` is the first point that reduces bonus
  breadth enough to let one disk case recover without disabling graph-support
  entirely.
- Null structural follow-up: an eval-only node-level scorer
  (`phase2_comp_sdr_lvl1_eval_graph_debug_p7_node_max_t0` and
  `phase2_comp_sdr_lvl1_eval_graph_debug_p7_node_max_t02`) replaced the mean support
  prototype with a max-over-stored-node support match, but it produced the exact same
  LM1 row patterns as the prototype scorer: `0/6` at `tolerance=0.0` and `1/6` at
  `tolerance=0.2`. This rules out the mean-prototype reduction itself as the primary
  cause of the failure.
- Stronger diagnosis: the actual stored LM1 disk support prototypes are already
  degenerate. In the `phase2_comp_sdr_lvl1_train_pilot7_graphobj_support3`
  checkpoint, the graph-level `object_support` prototypes for `006_disk`,
  `007_disk_tbp_horz`, and `009_disk_numenta_horz` have pairwise cosine similarity
  `1.0`, so the current support carrier gives LM1 no disk-level separation to score.
- Bottleneck shift: this is not simply an LM0 failure. In the collapsed pilot7 eval,
  LM0 is still `correct` on the `006_disk` episode while LM1 flips that case to
  `007_disk_tbp_horz`. The decisive loss therefore happens in LM1's stored support
  representation and parent-side objective, not purely in lower-level child matching.
- Updated plan: stop spending time on graph-support scalar tuning. The path toward a
  real jump now requires a richer child-to-parent message or a different LM1 support
  target, because the current `object_support` prototype is too lossy to distinguish
  similar parent graphs once it is written into graph memory.
- First non-null support-target redesign: LM0 upward support now has an opt-in
  `upward_support_evidence_normalization=range` mode that divides the top-k
  evidence gaps by `max(ptp(top_k_evidences), 1.0)` before the support softmax.
  This keeps the support carrier sensitive to relative child ranking instead of
  letting absolute accumulated evidence scale collapse the weights to one-hot.
- Normalized-support pilot result:
  `phase2_comp_sdr_lvl1_train_pilot10_support3_range` /
  `phase2_comp_sdr_lvl1_eval_pilot10_support3_range` reused the same pilot7-style
  regime (packet disabled, `object_id` one-best, top-3 `object_support`, temporal
  accumulation on, graph-support on at `tolerance=0.0`) and recovered from `0/6`
  to `1/6 = 16.7%` exact-or-MLH. LM1 now gets `002_cube_tbp_horz` back as
  `correct_mlh`; the other five LM1 rows remain `confused_mlh`.
- Storage-side confirmation on the normalized checkpoint: the LM1 disk parent
  nodes no longer store repeated one-hot support rows. For `006_disk`,
  `007_disk_tbp_horz`, and `009_disk_numenta_horz`, every nonzero stored
  `object_support` row is now unique and carries meaningful off-target mass
  (roughly `0.21 / 0.56 / 0.23 / 0.0`), rather than the old repeated
  `[0.0, 1.0, 0.0, 0.0]` pattern.
- Prototype-separation follow-up: the normalized-support checkpoint also breaks the
  exact disk-prototype degeneracy. Pairwise LM1 disk prototype cosine similarity
  drops from `1.0` to `0.999839` (`006_disk` vs `007_disk_tbp_horz`), `0.999907`
  (`006_disk` vs `009_disk_numenta_horz`), and `0.999991`
  (`007_disk_tbp_horz` vs `009_disk_numenta_horz`). That separation is still weak,
  but it is the first direct evidence that changing the support target itself, not
  packet tweaks or graph-support scalar sweeps, moves the actual LM1 memory.
- Combined-selectivity follow-up: the normalized-support checkpoint does compose with
  the earlier graph-support tolerance lever. Eval-only
  `phase2_comp_sdr_lvl1_eval_pilot10_support3_range_t02` keeps the same normalized
  training checkpoint but raises LM1 `graph_support_objective_tolerance` from `0.0`
  to `0.2`; that improves the pilot from `1/6` to `2/6 = 33.3%` exact-or-MLH.
  `002_cube_tbp_horz` stays `correct_mlh`, and `009_disk_numenta_horz` newly flips
  from `confused_mlh` to `correct_mlh`. The `006_disk` row still misses, but now the
  final winner shifts from `002_cube_tbp_horz` to the closer sibling
  `007_disk_tbp_horz`.
- Over-tightened follow-up: pushing the same normalized checkpoint to
  `graph_support_objective_tolerance=0.3`
  (`phase2_comp_sdr_lvl1_eval_pilot10_support3_range_t03`) regresses back to
  `1/6 = 16.7%` exact-or-MLH. It loses the rescued `009_disk_numenta_horz` row and
  only changes the remaining `007_disk_tbp_horz` error from one wrong sibling to
  another, so the extra selectivity is no longer helping.
- Updated interpretation: the collapse was partly caused by using raw absolute
  child evidence scale to form upward support weights. Normalizing that scale is a
  real step in the right direction, but the remaining prototype similarities show
  that more separation is still needed if this line is going to move beyond the
  recovered `1/6` pilot ceiling.
- Current best pilot reading: in this line, the best known setting is now the
  combination of LM0 `upward_support_evidence_normalization=range` with LM1
  `graph_support_objective_tolerance=0.2`. That is the first pilot configuration to
  move beyond the old `1/6` ceiling without reintroducing packet machinery, and it
  reinforces the current diagnosis that support-target quality and parent-side
  selectivity need to improve together.

### Larger Execution Plan

- Phase 1: lock the causal baseline. Keep the collapsed pilot7 checkpoint and the
  current best normalized-support-plus-`tolerance=0.2` point as the two reference
  pilot states. Future work should be judged against those row-level outcomes, not
  only against aggregate scores.
- Phase 2: strengthen the LM0-to-LM1 support carrier. Keep pushing on support-target
  quality rather than packet complexity first. The current `range` normalization is
  the first non-null win here because it changed stored LM1 support rows instead of
  only changing eval-time scoring.
- Phase 3: replace the single-prototype LM1 support target. The remaining parent
  confusion still points to a lossy parent memory: even the improved checkpoint keeps
  sibling disk prototypes extremely close. The next structural step after carrier
  improvement is a richer parent support representation, such as multi-prototype or
  mode-preserving support storage.
- Phase 4: redesign the LM1 parent objective around hard sibling negatives. Once the
  stored support target is less collapsed, parent scoring should explicitly separate
  targets from their nearest structured competitors instead of only rewarding broad
  support similarity.
- Phase 5: reopen packet and rank-weight transport only after the support-target and
  parent-objective path is stronger. Packet work was null in the old collapsed regime,
  but that does not yet prove it will stay null once LM1 sees genuinely distributed
  support targets.
- Phase 6: move from eval-only rescue to training-time consolidation. If a setting is
  only good when patched in at eval time, it is not yet the right training rule. The
  current best eval-only point must therefore be retrained and re-evaluated end to end.
- Phase 7: scale only after the pilot mechanism is real. The first scaling target is
  the broader compositional Phase 2 eval, not extra LMs or same-level voting.
- Phase 8: keep the stronger benchmark path alive in parallel. Omniglot or another
  saturation-resistant control benchmark still matters, but it should follow the core
  LM1 support-target work rather than distract from it.

### Current Execution Gate

- Immediate objective: test whether the current best eval-only point survives when the
  same selectivity rule is active during LM1 training, not just during final scoring.
- Concrete run order:
  `phase2_comp_sdr_lvl1_train_pilot11_support3_range_t02` with LM0
  `upward_support_evidence_normalization=range` and LM1
  `graph_support_objective_tolerance=0.2`, followed by
  `phase2_comp_sdr_lvl1_eval_pilot11_support3_range_t02`.
- Decision rule:
  if the retrained checkpoint stays at least as good as the eval-only rescue point,
  the next branch is support-target consolidation under training; if it falls back,
  the next branch is to improve LM1 stored support structure directly rather than keep
  tuning the current objective.
- Training-time consolidation result:
  `phase2_comp_sdr_lvl1_eval_pilot11_support3_range_t02` is row-identical to the
  earlier eval-only rescue point `phase2_comp_sdr_lvl1_eval_pilot10_support3_range_t02`.
  The gain therefore survives checkpoint formation, but training with
  `graph_support_objective_tolerance=0.2` does not add any new pilot win beyond the
  eval-only setting.
- First multi-prototype parent-memory probe:
  LM1 graph-support scoring now supports
  `graph_support_num_prototypes > 1`, which keeps the old mean prototype and adds a
  small diverse subset of stored `object_support` rows as extra comparison targets.
  The comparable eval-only probe
  `phase2_comp_sdr_lvl1_eval_pilot13_support3_range_t02_proto2_n1` used the stable
  pilot11 checkpoint with `graph_support_num_prototypes=2` and stayed at
  `2/6 = 33.3%` exact-or-MLH. The only row change was `006_disk`, where the wrong
  winner changed from `007_disk_tbp_horz` to `002_cube_tbp_horz`. That is enough to
  show the extra prototype path is active, but not enough to claim a real behavioral
  improvement yet.
- Updated interpretation:
  the remaining bottleneck still looks like LM1 support-target quality, but the
  naive "mean prototype plus one diverse stored row" probe is not sufficient by
  itself. The next support-target redesign should preserve more structure than a
  tiny prototype bank, or should use those prototypes in a more selective objective.
- Graph-debug readout on the comparable multi-prototype probe:
  `phase2_comp_sdr_lvl1_eval_graph_debug_p13_proto2_n1` shows that the changed
  `006_disk` row is not being redirected by a genuinely distinct disk prototype.
  The extra disk prototype selected by `graph_support_num_prototypes=2` is almost
  identical to the disk mean (`006_disk` mean about `0.2128 / 0.5751 / 0.2121`
  versus extra row about `0.2121 / 0.5748 / 0.2131`), and the logged LM1
  graph-support similarities are still consistently highest for `002_cube_tbp_horz`.
  In the `006_disk` episode the graph-support bonus is therefore still being driven
  by a cube-like support shape rather than by a more selective disk-specific mode.
- Packet/rank revisit under the improved regime:
  `phase2_comp_sdr_lvl1_train_pilot14_packet_range_t02` /
  `phase2_comp_sdr_lvl1_eval_pilot14_packet_range_t02` reopened packet materialization
  and packet rank-weight features under the best normalized-support setup:
  LM0 `upward_support_evidence_normalization=range`, LM1
  `graph_support_objective_tolerance=0.2`, `upward_packet_top_k=3`, and
  `upward_packet_weight_features=true` during both training and eval.
  The stored LM1 disk parent graphs now contain nonzero `object_id_rank_0..2`
  rows and nonzero `object_rank_weight_0..2` scalars (for example on `006_disk`,
  all three rank slots are populated on all 13 nonzero parent rows and the
  maximum rank weights are about `0.576 / 0.214 / 0.212`). Despite that, the LM1
  eval is row-identical to the stable `phase2_comp_sdr_lvl1_eval_pilot11_support3_range_t02`
  baseline at `2/6 = 33.3%` exact-or-MLH.
- Revised branch decision:
  identity-bound packet transport is no longer failing because the features are
  missing or collapsed, but it is still behaviorally null in the improved support
  regime. That pushes the next structural step away from more carrier enrichment
  and toward a harder LM1 parent objective, where support similarity must compete
  against its nearest structured alternatives instead of only adding an independent
  positive bonus.
- Competitive LM1 support objective follow-up:
  eval-only `phase2_comp_sdr_lvl1_eval_p15_competitive_t02_m0` enabled a global
  competitive support objective on the stable pilot11 checkpoint by rewarding a
  parent only when its support similarity beat the nearest stored competitor.
  That regressed LM1 from the stable `2/6 = 33.3%` exact-or-MLH point down to
  `1/6 = 16.7%`, keeping only `002_cube_tbp_horz` correct while turning
  `009_disk_numenta_horz` into a miss and collapsing several previous
  `confused_mlh` rows into hard `confused` false negatives.
- Competitive graph-debug interpretation:
  `phase2_comp_sdr_lvl1_eval_graph_debug_p15_competitive_t02_m0` shows the
  global competitive scorer is subtracting one generic cube-like support shape
  from another rather than sharpening the true target. Across all six episodes,
  the strongest positive bonus is still assigned to `002_cube_tbp_horz`, with raw
  support similarities around `0.88-0.94`, nearest-competitor similarities around
  `0.81-0.88` (usually `001_cube`), and only tiny residual competitive scores of
  about `0.066-0.071`. The objective is therefore active, but it mostly erases
  support mass instead of redirecting it toward the target.
- Candidate-scoped competitive follow-up:
  eval-only `phase2_comp_sdr_lvl1_eval_p16_competitive_top3_t02_m0` restricted
  the competitive objective to the top-3 current evidence candidates via
  `graph_support_competitive_top_k=3`. That probe is row-identical to the global
  competitive result at `1/6`, so narrowing competition to the current high-evidence
  set is not enough when those candidates are already dominated by the same cube-like
  family.
- Updated interpretation:
  the failure is no longer just "absolute vs relative" support scoring. The current
  support manifold is so dominated by broad cube-like alternatives that both global
  and top-candidate competitive objectives collapse onto the same wrong LM1 winner.
  The next objective redesign likely needs a more structured competitor set than
  "all graphs" or "current top evidence," for example family-aware / sibling-aware
  competition or a target-conditioned hard-negative set built from possible matches.
- Possible-matches competitive follow-up:
  LM1 competitive support scoring now also supports
  `graph_support_competitive_scope=possible_matches`, which restricts each parent's
  hard negatives to LM1's current thresholded candidate set rather than all stored
  graphs or the raw top-evidence pool. The eval-only probe
  `phase2_comp_sdr_lvl1_eval_p17_competitive_pm_t02_m0` used that scope on the
  stable pilot11 checkpoint and recovered to `2/6 = 33.3%` exact-or-MLH, but only
  as a lateral swap: `007_disk_tbp_horz` improved from `confused_mlh` to `correct`
  while `009_disk_numenta_horz` regressed from `correct_mlh` to `confused`.
- Possible-matches competitive debug interpretation:
  `phase2_comp_sdr_lvl1_eval_graph_debug_p17_competitive_pm_t02_m0` shows that this
  structured competitor set changes the loser more than it helps the winner. In the
  disk episodes the true target still receives zero positive graph-support bonus
  (`positive_bonus_count = 0`, `mean_bonus = 0.0`, `mean_support_similarity = 0.0`),
  and the improved row comes from suppressing a wrong competitor rather than from the
  target graph being matched more selectively.
- Competitive margin follow-up:
  adding a positive margin on that same possible-matches branch
  (`phase2_comp_sdr_lvl1_eval_p18_competitive_pm_t02_m01`) regressed LM1 back down
  to `1/6 = 16.7%`. The margin removed too much of the already-thin useful residual,
  turning `002_cube_tbp_horz` from `correct_mlh` into `confused` and pushing
  `004_cube_numenta_horz` all the way to `no_match`.
- Prototype-bank scaling follow-up:
  the support-target branch was then revisited by scaling the stored LM1 support bank
  beyond the original `proto2` probe. Both
  `phase2_comp_sdr_lvl1_eval_p19_proto4_t02_n1` (`graph_support_num_prototypes=4`)
  and `phase2_comp_sdr_lvl1_eval_p20_proto32_t02_n1`
  (`graph_support_num_prototypes=32`, effectively all stored rows) are row-identical
  to the earlier `phase2_comp_sdr_lvl1_eval_pilot13_support3_range_t02_proto2_n1`
  result at `2/6 = 33.3%` exact-or-MLH. That rules out prototype-cap saturation as
  the explanation for the earlier null result.
- Stored-support collision diagnosis:
  direct inspection of the stable pilot11 checkpoint shows the LM1 support carrier is
  collapsing the hard disk parents before the objective ever sees them. The normalized
  mean support vectors for `006_disk`, `007_disk_tbp_horz`, and
  `009_disk_numenta_horz` are approximately
  `[0.2128, 0.5751, 0.2121, 0.0]`, `[0.2093, 0.5688, 0.2219, 0.0]`, and
  `[0.2100, 0.5704, 0.2196, 0.0]`, with pairwise cosine similarities above `0.9998`.
  Even the raw normalized row banks overlap heavily (minimum cross-graph row distance
  down to about `6.7e-05` between `007_disk_tbp_horz` and `009_disk_numenta_horz`).
  Support alone is therefore encoding broad child-family composition much more than
  instance-specific parent identity.
- Object-id-conditioned support follow-up:
  a new optional LM1 mode,
  `graph_support_object_id_context=true`, was added so graph-support similarity can be
  scored on row-aligned `(object_id, object_support)` pairs instead of on pooled
  support rows alone. The focused unit module still passes cleanly (`40 passed`).
  However the eval-only probe `phase2_comp_sdr_lvl1_eval_p21_objctx_t02_n1`
  regressed sharply to `1/6 = 16.7%` exact-or-MLH.
- Object-id-conditioned debug interpretation:
  `phase2_comp_sdr_lvl1_eval_graph_debug_p21_objctx_t02_n1` shows why the direct
  context gate fails. The wrong disk parents still receive almost the same
  `object_id_context_similarity` as the target (for example in the `006_disk` episode,
  `009_disk_numenta_horz` logs about `0.974` and `007_disk_tbp_horz` about `0.957`),
  so the context term is not selective enough to separate sibling disks. In the
  `009_disk_numenta_horz` episode the target does receive the strongest support bonus,
  but LM1 still finishes on `002_cube_tbp_horz`, which means the bonus remains too weak
  to overturn the base evidence on its own.
- Revised branch decision:
  the support carrier is now well characterized. Increasing the number of stored
  support prototypes does not help, and directly multiplying support by row-level
  `object_id` context is too brittle. The next LM1-side step should therefore move
  away from pure support-space matching and toward a richer parent objective that uses
  row-level child context without collapsing it into a single scalar support score.
- Packet-context measurement follow-up:
  direct inspection of the packet-enabled pilot14 checkpoint shows that LM1's stored
  packet/rank rows are more separated than raw support, but only partially. Averaging
  the concatenated `object_id_rank_0..2` plus `object_rank_weight_0..2` rows gives
  pairwise cosine similarities of about `0.605` for `006_disk` vs
  `007_disk_tbp_horz`, `0.644` for `006_disk` vs `009_disk_numenta_horz`, and
  `0.975` for `007_disk_tbp_horz` vs `009_disk_numenta_horz`. That is a much stronger
  signal than the `> 0.9998` support-space collapse, but it still leaves the sibling
  logo disks tightly coupled.
- Packet-context objective follow-up:
  a new optional LM1 mode, `graph_support_packet_context=true`, now scores each stored
  support row in the context of its aligned packet rank features instead of using only
  pooled support or direct `object_id` context. The focused LM unit module still passes
  cleanly (`41 passed`). The eval-only probe
  `phase2_comp_sdr_lvl1_eval_p22_packetctx_t02_n1` stayed at `2/6 = 33.3%`
  exact-or-MLH, but it was not row-identical to either the stable pilot11 baseline or
  the earlier packet carrier null result: `004_cube_numenta_horz` improved from
  `confused_mlh` to `correct_mlh`, while `002_cube_tbp_horz` regressed from
  `correct_mlh` to `confused_mlh` and `006_disk` switched to the wrong sibling disk
  `009_disk_numenta_horz`.
- Packet-context debug interpretation:
  `phase2_comp_sdr_lvl1_eval_graph_debug_p22_packetctx_t02_n1` shows that the richer
  row context is real but not sufficient on its own. In the `006_disk` episode,
  `009_disk_numenta_horz` now has the highest mean packet-conditioned support score
  (`~0.533`) with the true `006_disk` target slightly lower (`~0.487`), so the packet
  context is redirecting disk support away from the cube family but still not selecting
  the right disk sibling. In the `007_disk_tbp_horz` episode the target remains absent
  from the top packet-conditioned support ranks entirely, which explains why the pure
  packet-context scorer cannot beat the old `2/6` plateau.
- Packet-context plus possible-matches competition follow-up:
  combining that richer row context with the least-bad relative objective,
  `graph_support_competitive_objective=true` plus
  `graph_support_competitive_scope=possible_matches`, produced the eval-only probe
  `phase2_comp_sdr_lvl1_eval_p23_packetctx_pm_t02_n1`. That branch improved LM1 to
  `3/6 = 50.0%` exact-or-MLH, the first pilot in this investigation to beat the long
  `2/6` plateau. The gain comes from restoring `002_cube_tbp_horz` to `correct_mlh`
  while keeping the `004_cube_numenta_horz` and `009_disk_numenta_horz` wins.
- Packet-context competitive debug interpretation:
  `phase2_comp_sdr_lvl1_eval_graph_debug_p23_packetctx_pm_t02_n1` shows why this
  branch improves. In the `002_cube_tbp_horz` episode the relative objective suppresses
  the nearly tied `004_cube_numenta_horz` competitor and leaves the target with the
  dominant mean support bonus (`~0.148` versus `~0.003`). In the `004_cube_numenta_horz`
  and `009_disk_numenta_horz` episodes the target also retains the strongest positive
  mean bonus. However the same debug run confirms the remaining bottleneck: in the
  `006_disk` and `007_disk_tbp_horz` episodes the competitive packet-context objective
  still assigns the strongest positive mean bonus to `009_disk_numenta_horz`, while the
  true `006_disk` target drops to zero and `007_disk_tbp_horz` remains non-dominant.
  The branch therefore improves cube-vs-cube ranking but has not yet solved the disk
  sibling collision.
- Training-time consolidation follow-up:
  the matching pilot train/eval pair
  `phase2_comp_sdr_lvl1_train_p24_packetctx_pm_t02_n1` /
  `phase2_comp_sdr_lvl1_eval_p24_packetctx_pm_t02_n1` then activated the same
  packet-context plus `possible_matches` competitive objective during LM1 training,
  not only during final eval. That branch regressed sharply to `1/6 = 16.7%`
  exact-or-MLH, keeping only `002_cube_tbp_horz` correct.
- Updated interpretation:
  the `p23` gain does not survive checkpoint formation. Relative to the eval-only
  rescue point, `004_cube_numenta_horz` fell from `correct_mlh` to `confused_mlh`
  under `002_cube_tbp_horz`, and `009_disk_numenta_horz` fell from `correct_mlh`
  to `confused` under that same wrong cube parent. This means the current
  packet-context competitive objective is useful as a late reranker on top of the
  packet-enabled checkpoint, but it is destabilizing when allowed to shape LM1 memory
  formation directly. The next branch should therefore return to improving stored LM1
  support or packet structure for the hard disk siblings rather than promoting this
  relative objective into the training loop unchanged.
- Graph-debug diagnosis on the training collapse:
  `phase2_comp_sdr_lvl1_eval_graph_debug_p24_packetctx_pm_t02_n1` makes the failure
  mode explicit. The disk rows did not improve at all: `006_disk` and
  `007_disk_tbp_horz` still assign zero target bonus while `009_disk_numenta_horz`
  keeps the top positive mean bonus (`~0.260` and `~0.263`). What changed is that the
  training-time competitive objective also strengthened the wrong cube attractor.
  In the `001_cube` episode the top bonus moved from a tiny `004_cube_numenta_horz`
  edge (`~0.003`) to a much larger `002_cube_tbp_horz` bonus (`~0.097`). In the
  `004_cube_numenta_horz` episode the target bonus collapsed from `~0.022` to `0.0`
  while `002_cube_tbp_horz` took over with `~0.105`. In the `009_disk_numenta_horz`
  episode the target still retained a small positive mean bonus (`~0.020`), but the
  same wrong cube parent dominated with `~0.107` and flipped the final winner. So the
  training regression is not a generic loss of packet context; it is a selective
  amplification of the existing `002_cube_tbp_horz` attractor plus no progress on the
  disk-sibling collision.
- Next branch after the debug readout:
  the immediate follow-up should isolate packet-context training from the destabilizing
  relative objective. Concretely, the next train/eval branch should enable
  `graph_support_packet_context=true` during LM1 training while leaving
  `graph_support_competitive_objective=false`, then test whether the resulting
  checkpoint remains stable on its own and whether the eval-only
  `possible_matches` reranker still helps on top of it.
- Packet-context training isolation result:
  that follow-up was run immediately as
  `phase2_comp_sdr_lvl1_train_p25_packetctx_t02_n1`, with matching evals
  `phase2_comp_sdr_lvl1_eval_p25_packetctx_t02_n1` and
  `phase2_comp_sdr_lvl1_eval_p26_packetctx_pm_t02_n1`. The isolation branch also
  collapsed to `1/6 = 16.7%` exact-or-MLH. Plain `p25` kept only
  `004_cube_numenta_horz` correct and redirected four of the six rows to the wrong
  `004_cube_numenta_horz` parent, including both `002_cube_tbp_horz` and
  `009_disk_numenta_horz`. Re-adding the eval-only `possible_matches` reranker on top
  as `p26` did not rescue the checkpoint; it stayed at `1/6` and only softened two
  misses from `confused_mlh` to `confused`.
- Updated branch decision:
  packet-context scoring is therefore not just incompatible with the competitive
  objective during LM1 training; it is unstable as a training-time graph-support rule
  in its own right. The best packet-context result remains the eval-only reranker
  `p23` on top of the older `pilot14` checkpoint. The next real train-time branch
  should now return to improving LM1 stored support structure or a new sibling-aware
  parent objective, not more packet-context training variants.
- Support-carrier widening follow-up:
  the simplest stored-support hypothesis was that LM0 was truncating away useful
  parent evidence by only shipping top-3 support mass upward. That was tested as
  `phase2_comp_sdr_lvl1_train_p27_support4_range_t02` /
  `phase2_comp_sdr_lvl1_eval_p27_support4_range_t02`, which regressed from the stable
  `pilot11` checkpoint at `2/6` down to `0/6`. Direct checkpoint inspection showed
  that top-4 support did change LM1 memory: the disk parents gained a real fourth
  support channel (`~0.168-0.171`) and some pairwise distances widened slightly.
  But behaviorally that extra mass only diffused the useful structure; a follow-up
  eval-only tolerance sweep at `0.0` (`p28`) and `0.1` (`p29`) stayed row-identical
  to the `0/6` failure. So the issue is not just that top-4 support needs a different
  threshold; the broader carrier itself degrades LM1 parent selection.
- Support-residual centering follow-up:
  numeric inspection of the stable `pilot11` checkpoint suggested that subtracting a
  sibling-family mean could expose a latent residual signal: after centering the disk
  support means, `006_disk` became nearly anti-parallel to both `007_disk_tbp_horz`
  and `009_disk_numenta_horz`, and the `002_cube_tbp_horz` /
  `004_cube_numenta_horz` pair also flipped to an anti-aligned residual. That led to
  an eval-only scorer probe using centered graph-support similarity. The direct
  support-only variants failed: global centering (`p31`) and the corrected
  `possible_matches` centering (`p32`) both regressed the stable `pilot11`
  checkpoint from `2/6` to `1/6`. Folding the same idea into the best eval-only
  packet-context reranker (`phase2_comp_sdr_lvl1_eval_p33_packetctx_pm_t02_centerpm`)
  also hurt, dropping `p23` from `3/6` to `2/6` by trading the rescued
  `009_disk_numenta_horz` row for `007_disk_tbp_horz`. So the residual signal is
  real in stored means, but simple graph-support centering does not convert it into a
  usable LM1 scoring rule.
- Soft-competition packet-context follow-up:
  because `p23` improves the cube rows but zeros out the true `006_disk` support bonus,
  the next eval-only probe softened the competitive residual instead of changing the
  stored LM1 memory. Two variants were tested on the `p23` shape with the same
  `pilot14` checkpoint: `phase2_comp_sdr_lvl1_eval_p34_packetctx_pm_t02_pen085`
  and `phase2_comp_sdr_lvl1_eval_p35_packetctx_pm_t02_pen095`. The softer `0.85`
  penalty regressed badly, collapsing the restored `002_cube_tbp_horz` win and falling
  to `1/6`; the near-hard `0.95` penalty still lost `002_cube_tbp_horz` and only
  reached `2/6`, with no rescue of `006_disk` or `007_disk_tbp_horz`. So the failure
  mode is not just that the competitive subtraction is slightly too harsh; softening it
  erodes the useful cube-side separation before it fixes the disk sibling collision.
- Packet-context plus direct object-id row-context follow-up:
  the next hypothesis was that the remaining disk ambiguity might need both kinds of
  row context at once: retain the `p23` packet-conditioned scorer, but also require the
  aligned stored `object_id` row to match. That was implemented and tested as the
  eval-only branch `phase2_comp_sdr_lvl1_eval_p36_packet_objctx_pm_t02_n1`. The idea
  did not hold up. Relative to `p23`, LM1 dropped from `3/6` to `2/6` exact-or-MLH:
  `006_disk` improved only from `confused` to `confused_mlh`, while the useful
  `002_cube_tbp_horz` rescue regressed from `correct_mlh` to `confused_mlh`, and
  `007_disk_tbp_horz` also got worse. The matching debug run
  `phase2_comp_sdr_lvl1_eval_graph_debug_p36_packet_objctx_pm_t02_n1` showed why:
  the true `006_disk` target still kept zero mean bonus, but the added row-level
  object-id constraint also collapsed the target bonus on `002_cube_tbp_horz`, so this
  combined scorer sharpened the wrong side of the trade-off instead of solving the disk
  sibling collision.
- Negative-margin competitive follow-up:
  a cheaper scalar probe then tested whether `p23` was simply clipping too aggressively.
  `phase2_comp_sdr_lvl1_eval_p37_packetctx_pm_t02_mneg005` kept the same packet-context
  plus `possible_matches` setup but relaxed the relative objective with
  `graph_support_competitive_margin=-0.05`. That branch was row-identical to `p23` and
  stayed at `3/6` exact-or-MLH, so the current failure mode is not a small thresholding
  artifact around the nearest-competitor margin.
- Top-k-evidence competition follow-up:
  the next probe changed the competition set instead of the scoring formula. Keeping the
  same `pilot14` checkpoint and packet-context scorer, but restricting competitor search
  to the current evidence leaders with
  `graph_support_competitive_scope=top_k_evidence` and
  `graph_support_competitive_top_k=3`, produced
  `phase2_comp_sdr_lvl1_eval_p38_packetctx_top3_t02_m0`. On the old strict
  `correct_mlh` count that branch ties `p23` at `3/6`, but on the broader exact-or-MLH
  measure it becomes the new best local eval-only point at `4/6 = 66.7%` because
  `007_disk_tbp_horz` flips from `confused_mlh` under `009_disk_numenta_horz` to exact
  `correct` while preserving the restored `002_cube_tbp_horz`, `004_cube_numenta_horz`,
  and `009_disk_numenta_horz` wins.
- Top-k-evidence debug interpretation:
  `phase2_comp_sdr_lvl1_eval_graph_debug_p38_packetctx_top3_t02_m0` shows that this is
  not a true target-bonus recovery. In the rescued `007_disk_tbp_horz` episode the true
  target still receives zero graph-support bonus, exactly as in `p23`. What changes is
  the competing `009_disk_numenta_horz` branch: its mean bonus drops from `~0.258` in
  `p23` down to only `~0.029`, while the top updates are redirected toward unrelated
  cube and `006_disk` rows. So `p38` works by stopping LM1 from over-rewarding the wrong
  disk sibling, which is enough to let the underlying packet evidence keep
  `007_disk_tbp_horz` on top even without a positive target bonus.
- Top-k sweep follow-up:
  reducing the same `top_k_evidence` scope from `3` down to `2` as
  `phase2_comp_sdr_lvl1_eval_p39_packetctx_top2_t02_m0` immediately broke the gain.
  LM1 fell to `2/6` exact-or-MLH, `007_disk_tbp_horz` collapsed to `confused` under
  `002_cube_tbp_horz`, and `009_disk_numenta_horz` also regressed to `confused` under
  `004_cube_numenta_horz`. So the improvement is narrow: the useful setting on this axis
  is currently `top_k_evidence` with `top_k=3`, not a general monotonic benefit from
  shrinking the competitor set.
- Competitive object-id-context revisit on the stronger checkpoint:
  because the live scorer prioritizes packet-context rows over direct object-id row
  context, the next cheap probe replayed the `p38` competitive shape on the stronger
  `pilot14` packet checkpoint but swapped the row scorer itself to
  `graph_support_object_id_context=true`. That eval-only branch,
  `phase2_comp_sdr_lvl1_eval_p40_objctx_top3_t02_m0`, collapsed all the way to
  `0/6 = 0.0%` exact-or-MLH. LM1 lost every prior rescue: `002_cube_tbp_horz`,
  `004_cube_numenta_horz`, and `009_disk_numenta_horz` all fell out of the correct or
  `correct_mlh` bucket, while the disk rows remained unmatched. So direct object-id row
  context is not just weaker than packet-context on the old `pilot11` checkpoint; it
  also fails badly when replayed inside the stronger `pilot14` competitive reranker.
- Top-k-within-possible-matches follow-up:
  the next step then implemented a narrower hard-negative scope,
  `graph_support_competitive_scope=top_k_possible_matches`, which first restricts
  competitors to LM1's current `possible_matches` set and only then keeps the top-K by
  evidence. The focused LM unit module stayed green (`42 passed`). But the first eval-only
  branch on that new scope,
  `phase2_comp_sdr_lvl1_eval_p41_packetctx_top3pm_t02_m0`, regressed from `p38` back
  to `3/6 = 50.0%` exact-or-MLH. It preserved the `002_cube_tbp_horz`,
  `004_cube_numenta_horz`, and `009_disk_numenta_horz` wins, but lost the `p38`
  `007_disk_tbp_horz` rescue and reverted that row to `confused_mlh` under
  `009_disk_numenta_horz`. So narrowing the hard negatives to top evidence *within*
  possible matches does not beat the raw `top_k_evidence` scope; the current best local
  eval-only branch remains `p38`.
- Live competitive-penalty replay on the `p38` scorer shape:
  the current LM1 scorer now explicitly consumes
  `graph_support_competitive_penalty`, exposes it in the graph-support debug payload,
  and covers the path in focused LM unit tests (`11` graph-support tests passed after
  the change). Replaying the best local `p38` packet-context top-3-evidence branch with
  `graph_support_competitive_penalty=0.95` as
  `phase2_comp_sdr_lvl1_eval_p42_packetctx_top3_t02_pen095` regressed sharply from
  `4/6` exact-or-MLH down to `2/6`. It only sharpened `002_cube_tbp_horz` from
  `correct_mlh` to `correct`, but it lost the key `p38` rescue of `007_disk_tbp_horz`
  (falling back to `confused` under `002_cube_tbp_horz`) and also degraded
  `009_disk_numenta_horz` from `correct_mlh` to `confused_mlh` under
  `004_cube_numenta_horz`. So on the strongest local frontier, softening the top-3
  competitive subtraction does not recover the missing disk target bonus and instead
  weakens the useful sibling suppression that made `p38` work.
- Same-peak rival filtering was the wrong sibling proxy:
  adding `graph_support_competitive_same_peak_only` and replaying the same `p38`
  scorer shape as `phase2_comp_sdr_lvl1_eval_p43_packetctx_top3_samepeak_t02_m0`
  also regressed from `4/6` down to `2/6`. It preserved `004_cube_numenta_horz` and
  sharpened `009_disk_numenta_horz` to exact `correct`, but it lost both
  `002_cube_tbp_horz` and `007_disk_tbp_horz`, with LM1 falling back to
  `009_disk_numenta_horz` on both rows. The graph-debug replay showed why: support-slot
  peak index is too crude a family proxy, so the scorer can miss the meaningful sibling
  rival and either fall back to a weak unrelated competitor or actively reward the wrong
  same-peak disk sibling.
- Packet-context-gated rival selection is safer than same-peak filtering but still does
  not beat `p38`:
  the next scorer refinement added
  `graph_support_competitive_packet_context_floor`, requiring a rival to clear a packet-
  context similarity floor relative to the current target row before it can become the
  competitive subtractor. Focused LM graph-support tests stayed green (`13 passed`).
  Replaying the `p38` branch with `graph_support_competitive_packet_context_floor=0.9`
  as `phase2_comp_sdr_lvl1_eval_p44_packetctx_top3_ctxfloor09_t02_m0` recovered to
  `4/6` exact-or-MLH, matching `p38` rather than improving it. Relative to `p43`, it
  restored both `002_cube_tbp_horz` and `007_disk_tbp_horz`; relative to `p38`, it only
  sharpened `002_cube_tbp_horz` and `009_disk_numenta_horz` from `correct_mlh` to exact
  `correct` while leaving the still-failed `001_cube` and `006_disk` rows unchanged. So
  packet-context is a better rival filter than support peak, but it does not yet open a
  new local frontier beyond `p38`.
- Strict packet-context fallback also failed on this axis:
  the next follow-up added optional
  `graph_support_competitive_packet_context_strict` behavior so LM1 can skip
  competitive subtraction when no rival clears the packet-context floor. Replaying the
  same branch with `graph_support_competitive_packet_context_floor=0.9` and strict mode
  enabled as `phase2_comp_sdr_lvl1_eval_p45_packetctx_top3_ctxfloor09_strict_t02_m0`
  regressed from `4/6` to `3/6`, losing the `007_disk_tbp_horz` rescue while leaving
  `001_cube` and `006_disk` unchanged. The debug readout shows why: removing fallback
  subtraction also removes the useful suppression on high-context wrong winners such as
  `009_disk_numenta_horz`.
- Ceiling-limited strict fallback closed the packet-context-floor branch as a dead end:
  limiting strict fallback to lower-context candidates via
  `graph_support_competitive_packet_context_strict_ceiling=0.9` was intended to spare
  high-context winners while still helping `006_disk`, but the first replay,
  `phase2_comp_sdr_lvl1_eval_p46_packetctx_top3_ctxfloor09_strictc09_t02_m0`,
  collapsed to `1/6` exact-or-MLH. It lost the `002_cube_tbp_horz`,
  `007_disk_tbp_horz`, and `009_disk_numenta_horz` wins without rescuing either of the
  still-failed rows, so the packet-context-floor family is now empirically closed: plain
  floor ties `p38`, while both strict variants regress.

6. Hold same-level voting and extra level-1 modules until the single-LM1 path is
  better understood.
  More LMs or voting are currently lower priority because the best causal evidence so
  far points to LM1 scoring, not to missing consensus capacity.

7. In parallel, finish the stronger benchmark path.
  Repair the local Omniglot setup and run the cross-version control benchmark so Phase
  2 does not remain pinned to a partially saturated YCB setting.

### Immediate Execution Order

- implement Step 1 now by adding LM1 graph-support debug logging and a dedicated debug
  eval config
- then use that debug path to execute Step 2 on the frozen-LM0 graph-objective
  checkpoint in the clean tolerance region
- after the first debug readout, tighten the region before broadening it: run
  `tolerance=0.05` next so the graph-support bonus is forced to become more selective
- because `tolerance=0.05` reverted to baseline without improving selectivity, the
  next midpoint probe should be `tolerance=0.15` rather than an even smaller value
- after the `tolerance=0.15` debug result, the next tolerance-side step should be
  a `tolerance=0.2` debug run only if we want a logged comparison against the already
  known clean eval-only `0.2` gain; otherwise the current evidence is already strong
  enough to keep the preferred band at `0.1–0.15`
- after the logged `0.2` confirmation, the tolerance band is characterized well
  enough to move to the small joint weight-tolerance grid, centered on
  `graph_support_objective_tolerance in [0.15, 0.2]`
- only after the logged tolerance runs are reviewed should Step 3 or Step 4 begin

### Saturation-Resistant Benchmark Setup

- the first concrete stronger benchmark candidate is Omniglot cross-version transfer:
  train on version `1` of each character and evaluate on version `2`
- this is not yet the full concept benchmark, but it is saturation-resistant in a way
  the current YCB category split is not, because the model must generalize across new
  drawings rather than only stay within the same coarse object category
- the new control configs are meant to establish the benchmark discipline first before
  deciding whether the next mechanism comparison should be monolithic versus hierarchy,
  exact-ID versus SDR, or both
- local validation of the Omniglot configs currently reaches the stock Omniglot
  environment setup and then fails with `IndexError: list index out of range`, which
  indicates the immediate blocker is missing or incomplete Omniglot data rather than a
  Hydra composition error in the new configs

### Held-Out Compositional Substitution Benchmark

- a second stronger benchmark is now scaffolded on `logos_on_objs` level 2 using five
  base objects: `cube`, `disk`, `cylinder`, `sphere`, and `mug`
- training exposes every base object in bare form plus exactly one decorated variant,
  so the model sees both constituents but not every combination:
  `001_cube`, `006_disk`, `011_cylinder`, `016_sphere`, `023_mug`,
  `002_cube_tbp_horz`, `009_disk_numenta_horz`, `012_cylinder_tbp_horz`,
  `019_sphere_numenta_horz`, `024_mug_tbp_horz`
- holdout evaluation uses the complementary unseen logo substitutions:
  `004_cube_numenta_horz`, `007_disk_tbp_horz`, `014_cylinder_numenta_horz`,
  `017_sphere_tbp_horz`, `026_mug_numenta_horz`
- this benchmark is stronger than the current YCB category split because success
  requires transferring a known logo constituent onto a known base object in a pairing
  not seen during training, rather than merely staying within a coarse category bucket
- added configs:
  - `experiment=phase2_comp_substitution_control_train`
  - `experiment=phase2_comp_substitution_control_eval_holdout`
  - `experiment=phase2_comp_substitution_comp_train`
  - `experiment=phase2_comp_substitution_comp_eval_holdout`
- added taxonomy and split files:
  - `config/environment_interface/logos_on_objs/phase2_comp_substitution_train_objects`
  - `config/environment_interface/logos_on_objs/phase2_comp_substitution_holdout_objects`
  - `config/environment_interface/logos_on_objs/phase2_comp_substitution_taxonomy`
- added compositional summary helper:
  `python tools/phase2_compositional_metrics.py --eval-stats <eval_stats.csv> --taxonomy src/tbp/monty/conf/experiment/config/environment_interface/logos_on_objs/phase2_comp_substitution_taxonomy.yaml`
- the helper reports exact accuracy plus part-level transfer summaries:
  `base_accuracy`, `logo_accuracy`, `both_parts_accuracy`, and separate rates for
  base-only, logo-only, cross-part, and no-prediction failures
- this benchmark is ready for first train/eval runs; it does not yet replace Omniglot
  as the best cross-domain validation target, but it gives Phase 2 a stronger local
  benchmark that is not pinned to a partially saturated YCB setting

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
- prototyping: completed for control and SDR debugging paths
- running benchmarks: completed for control, invalid SDR pretraining paths, valid eval-style SDR fitting plus held-out rerun, and full compositional SDR bridge train/eval
- blocked: no active runtime blocker; the remaining issue is scientific rather than engineering, because the current SDR paths still do not show a strict accuracy win over the already-strong baselines

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
- 2026-03-15: Moved from planning to investigation by implementing the YCB control split,
  taxonomy, category metrics utility, and dedicated Phase 2 control train/eval configs
  before launching the first long-running run.
- 2026-03-15: Attempted the first `phase2_ycb_category_control_train` launch and
  confirmed the config path was valid, but the run failed immediately because the local
  `MONTY_DATA` tree only contained `compositional_objects` and not the required YCB
  Habitat assets.
- 2026-03-15: Started recovery by launching Habitat's built-in YCB downloader to
  `/Users/felixrobles/tbp/data/habitat` and recorded a persistent log under
  `/Users/felixrobles/tbp/results/monty/projects/phase2_review_runs/dataset_logs`.
- 2026-03-15: Verified Habitat's downloader created the expected symlink at
  `/Users/felixrobles/tbp/data/habitat/objects/ycb` and resumed the
  `phase2_ycb_category_control_train` manifest successfully.
- 2026-03-15: Completed the control held-out evaluation and observed a sharp control
  baseline: `0%` exact held-out accuracy but `100%` category accuracy, with every
  error staying within the correct coarse category.
- 2026-03-15: Implemented the Phase 2 SDR train/eval configs, validated both
  manifests, and launched `phase2_ycb_category_sdr_train`. The first launch failed
  due to inherited displacement-only LM args; clearing the inherited learning-module
  config resolved the issue.
- 2026-03-16: Root-caused the SDR eval crash to `np.max()` on empty evidence arrays in
  `EvidenceSDRLMMixin.collect_evidences()` and fixed it by skipping empty evidence
  buffers and by refusing to train SDRs when only diagonal or empty overlap targets are
  present.
- 2026-03-16: Added focused regression coverage for the empty-evidence crash and the
  pairwise-target gate, then validated the fix with passing unit and integration tests.
- 2026-03-16: Root-caused a second SDR issue to serialization: the base LM save/load
  path persisted graph memory but dropped SDR-specific state, so the first parallel SDR
  evals did not actually reuse the trained SDR encoder.
- 2026-03-16: Added SDR encoder and target-overlap serialization plus a single-worker
  `load_state_dict_from_parallel()` fast path that preserves full LM state, then
  verified the new checkpoint structure in tests.
- 2026-03-16: Determined that the existing Phase 0 parallel pretraining workflow is
  structurally incompatible with SDR training for this benchmark, because each worker
  only learns one object and therefore cannot accumulate the pairwise overlap targets
  the SDR encoder needs.
- 2026-03-16: Ran a serial SDR retrain and serial holdout eval with all six training
  objects in one LM instance. The resulting checkpoint preserved `6` learned SDRs and
  improved held-out category accuracy from `53.57%` to `58.93%`, but it still remained
  far below the `100%` control baseline.
- 2026-03-16: Inspected the serial SDR checkpoint and found that its `target_overlaps`
  matrix was still empty, which showed that the supervised pretraining workflow was not
  actually producing the pairwise overlap targets required for SDR learning on this
  benchmark.
- 2026-03-16: Added an eval-style SDR fitting experiment that loads the control
  checkpoint, runs evidence-bearing matching episodes on the six training objects, and
  saves the fitted SDR state without mutating graph memory.
- 2026-03-16: Verified that the eval-style fitting checkpoint contains non-zero
  `target_overlaps.counts`, establishing the first valid SDR fitting workflow for this
  benchmark.
- 2026-03-16: Re-ran the held-out evaluation to completion against the eval-style fitted
  checkpoint after an earlier pilot was interrupted. The completed run restored the full
  `56/56` held-out episodes and recovered `100.0%` category accuracy with `0.0%`
  cross-category confusion, matching the control baseline.
- 2026-03-16: Root-caused the first stacked compositional SDR bridge crash to parent-LM
  graph reconstruction densifying a `200^3 x 2048` SDR-valued `object_id` channel.
  Reduced the parent LM grid to `20^3`, reran the bridge training pilot successfully,
  and verified that the saved checkpoint now contains SDR state for both hierarchy levels.
- 2026-03-16: Found that ordinary stacked SDR evaluation was still retraining SDR
  encoders during `post_episode()`, producing numerical overflow warnings. Fixed this
  by making eval-time SDR fitting opt-in, preserved the dedicated eval-style fitting path
  with an explicit config flag, disabled default W&B for the bridge eval config, and
  reran a clean compositional bridge eval pilot.
- 2026-03-16: Root-caused the first full stacked bridge training failure
  (`phase2_comp_sdr_lvl1_train_full1`) to scalar assumptions in parent graph-memory
  updates: vector-valued SDR `object_id` features triggered an ambiguous NumPy truth
  value check and row-collapsing majority logic. Fixed both issues with array-safe
  equality checks and vector-preserving majority aggregation, then added a focused
  regression for vector `object_id` updates.
- 2026-03-16: Completed full stacked compositional SDR bridge training as
  `phase2_comp_sdr_lvl1_train_full2`; the checkpoint wrote `model.pt`, `config.pt`,
  and `exp_state_dict.pt`, with LM0 preserving `4` child graphs and LM1 preserving
  `6` parent graphs plus SDR state at both levels.
- 2026-03-16: Completed full stacked compositional SDR bridge evaluation as
  `phase2_comp_sdr_lvl1_eval_full2`; across `168` eval rows it produced `26.8%`
  exact-or-MLH correct, `48.8%` `consistent_child_obj`, and `24.4%` confused or
  confused-MLH, so the bridge is operational end to end but still does not support an
  accuracy-improvement claim.
- 2026-03-16: Ran the frozen-LM0 causal diagnostic by seeding LM0 from the learned
  full bridge checkpoint, resetting LM1, disabling LM0 SDR updates during parent
  retraining, and re-running the full compositional eval. The resulting
  `phase2_comp_sdr_lvl1_eval_lm0_frozen_full1` run kept exact-or-MLH accuracy flat at
  `26.8%`, showing that lower-level SDR drift is not by itself sufficient to explain the
  parent-recognition failure.
- 2026-03-16: Implemented a top-k union upward carrier in `EvidenceSDRGraphLM`,
  validated it with new unit coverage plus the existing SDR integration suite, and ran
  the matching compositional pilot as `phase2_comp_sdr_lvl1_topk_train_pilot1` /
  `phase2_comp_sdr_lvl1_topk_eval_pilot1`. The saved config confirms
  `upward_top_k=3`, but the `12` eval rows were row-identical to the earlier bridge
  pilot, so carrier broadening alone did not change parent behavior.
- 2026-03-16: Added an `object_support` feature to LM0 output, extended the LM-side
  SDR feature calculator so LM1 can score both `object_id` and `object_support`, fixed
  the mixed-schema graph-memory edge cases this exposed, and ran the support-aware pilot
  as `phase2_comp_sdr_lvl1_support_train_pilot1d` /
  `phase2_comp_sdr_lvl1_support_eval_pilot1d`. LM1 parent graphs now store
  `object_support`, but the `12` eval rows were still row-identical to the earlier
  pilot baseline, so appending quantitative child support as an extra feature did not
  change parent behavior by itself.
- 2026-03-16: Implemented a receiver-side temporal support accumulator in
  `EvidenceSDRGraphLM` so LM1 can accumulate incoming `learning_module_0`
  `object_support` and smoothed `object_id` features across episode time before normal
  matching, validated it with focused SDR tests, and ran the temporal pilot as
  `phase2_comp_sdr_lvl1_temporal_train_pilot1` /
  `phase2_comp_sdr_lvl1_temporal_eval_pilot1`. The pilot again produced a row-identical
  `12`-row eval relative to `phase2_comp_sdr_lvl1_eval_pilot4`, so temporal
  accumulation without a different parent objective did not change parent behavior.
- 2026-03-16: Implemented a graph-level LM1 support objective that adds a parent-graph
  evidence bonus when the accumulated lower-level `object_support` matches a graph's
  stored support prototype. The first full training attempt
  (`phase2_comp_sdr_lvl1_graphobj_train_full1`) then failed in `GridObjectModel`
  because mixed-schema graph updates widened the feature mapping without widening the
  sparse feature-grid update tensors.
- 2026-03-16: Fixed the mixed-schema sparse-grid bug by resizing feature grids before
  sparse updates and by treating newly added feature blocks as genuinely new rather than
  averaging them against nonexistent prior values. Added a focused regression for adding
  `object_support` onto an already-occupied LM voxel, reran focused tests successfully,
  and completed `phase2_comp_sdr_lvl1_graphobj_train_full2` end to end.
- 2026-03-16: Completed the matching full graph-objective eval as
  `phase2_comp_sdr_lvl1_graphobj_eval_full2`; across `168` rows it produced `32.7%`
  exact-or-MLH correct versus `26.8%` for the earlier full bridge baseline, making this
  the first full-scale parent-side accuracy win in the stacked SDR Phase 2 line.

## Current Blockers

- the default `object_id` feature path is still similarity-poor by construction
- the current Phase 0 parallel pretraining workflow is not a valid training path for
  SDR similarity learning on this benchmark because it shards objects across workers and
  destroys pairwise overlap learning signal
- the current graph-memory implementation densifies per-channel feature grids during
  graph reconstruction, so stacked SDR transport is only tractable with a reduced
  parent-LM grid or with a future model-memory refactor
- `run.py` experiments do not yet benefit from the same Phase 0 orchestration used for
  serial restart, monitoring, and recovery
- it is not yet decided whether the roadmap should require a Phase 1.5 ablation before
  Phase 2 implementation starts
- the valid SDR fitting path now matches the simpler control baseline on held-out
  category behavior, but it still does not provide an accuracy gain over that baseline
- the learned overlap matrix from eval-style fitting is sparse and only partially
  populated, so the remaining scientific question is whether richer overlap coverage can
  produce a result that exceeds the already-perfect category control rather than merely
  matching it
- the full stacked compositional SDR bridge remains accuracy-limited: LM0 often lands
  in `consistent_child_obj` rather than exact parent correctness, and LM1 still shows a
  large `confused_mlh` mass, so the current hierarchy is not yet converting child reuse
  into a decisive parent-object win
- the frozen-LM0 causal diagnostic did not improve exact parent accuracy, so fixing LM0
  drift alone is not enough; the dominant remaining bottlenecks are likely one-best
  child transport, stale child-code storage already written into parent graphs, and the
  lack of quantitative confidence-weighted upward evidence
- the top-k union pilot did not change any pilot eval row relative to the earlier
  bridge pilot, so broadening the child carrier without also changing the parent-side
  representation or scoring objective is not sufficient on its own
- the parent-support pilot also did not change any pilot eval row relative to the
  earlier bridge pilot, so adding quantitative child support as an extra feature is
  likewise insufficient without a different parent-side objective
- the temporal-support accumulator pilot also did not change any pilot eval row
  relative to the earlier bridge pilot, so preserving child ambiguity over time still
  does not help if LM1 continues to use the same parent matching objective
- the graph-support objective now shows the first full-scale strict parent-accuracy win,
  but the gain is still modest (`32.7%` exact-or-MLH correct) and needs follow-up to
  determine whether it is robust, saturates quickly, or can be strengthened without
  trading too much mass into child-consistent but still inexact parent matches

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
- the similarity-aware path produces no meaningful gain over the exact-ID control, even
  if a valid fitting workflow can now match the control baseline
- the concept-like benchmark requires unrelated non-grounded machinery to work at all
- the benchmark suite cannot be run reproducibly under a documented execution and
  recovery plan

## Next Decision

- decision to make: whether to continue narrow LM1-side graph-support scoring sweeps,
  now that the first eval-only tolerance change produced a small clean gain, or to stop
  tuning and freeze the current Phase 2 claim around the existing graph-objective win
- evidence needed: review of the final comparison between the control baseline
  (`100.0%` category accuracy), the first broken-eval SDR result (`53.57%`), the
  corrected serial SDR result (`58.93%`), the completed eval-style fitted SDR result
  (`100.0%`), the completed full compositional bridge eval (`26.8%` exact-or-MLH
  correct, `48.8%` `consistent_child_obj`, `24.4%` confused or confused-MLH), and the
  completed frozen-LM0 diagnostic (`26.8%` exact-or-MLH correct, `49.4%`
  `consistent_child_obj`, `23.8%` confused or confused-MLH), plus the completed top-k
  union pilot and parent-support pilot that were both row-identical to the earlier
  bridge pilot, plus the changed-row analysis, the completed frozen-LM0
  graph-objective follow-up (`32.7%` exact-or-MLH), and the first LM1 eval-only sweeps
  showing that `graph_support_objective_weight=1.0` regresses to `31.5%`, while
  tolerance values `0.1` and `0.2` both produce small clean gains to `33.3%` and
  tolerance `0.3` reaches `33.9%` but with new churn and one regression
- fallback plan: if SDR iteration is deprioritized, freeze the current debugging result,
  document that the control already supplies strong category-consistent nearest-instance
  transfer, and move the next work to richer diagnostics or an alternative similarity
  transport mechanism
