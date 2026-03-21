# Phase 1: Make Composition Real

> **Track**: [Track 1: Multi-Column Intelligence](track-1-multi-column-intelligence.md)
> **Status**: Completed (negative result on primary criterion)
> **Roadmap**: [full-system-roadmap-v2.md](full-system-roadmap-v2.md)

## Purpose

This document captures the full plan behind the current work in `tbp.monty`.

The immediate goal is to test whether a higher-level learning module (LM) can take
lower-level recognized object IDs as features and learn compositional objects better
than a monolithic baseline.

The larger goal is to establish a practical path toward a system where:

- Monty learns grounded structure through sensorimotor interaction.
- A standard multimodal AI acts as a teacher, tutor, and synthetic curriculum source.
- Higher-level Monty modules gradually learn reusable abstractions built from grounded
  lower-level parts.

This is not yet a language benchmark or a full multimodal agent benchmark. It is the
first controlled test of hierarchical compositional learning in the current codebase.

## Metadata

- phase: Phase 1: Make Composition Real
- roadmap link: [docs/full-system-roadmap.md](docs/full-system-roadmap.md)
- owner: local working plan
- status: completed with negative result on primary exit criterion
- last updated: 2026-03-16

## Goal

Prove that higher-level LMs can reuse grounded lower-level parts in a way that matters
on the current logos-on-objects benchmark.

## Why This Phase Exists

If Monty cannot show a concrete compositional advantage here, the broader claim that
hierarchy supports grounded abstraction is still unproven.

## Investigation Summary

- the full logos-on-objects training ladder has already been executed locally through
  level 4 under `my_trained_models`
- the remaining benchmark gap is evaluation and comparison, not child-object training
- the repo already has level-1 compositional and monolithic inference configs plus
  higher-level compositional inference configs for levels 2 through 4
- level 2 expands beyond the level-1 objects into additional parent objects such as
  cylinder/logo, sphere/logo, and mug/logo combinations, making it the smallest useful
  follow-up benchmark beyond the basic level-1 comparison

## TBT And Jeff Hawkins Lens

This phase is directly about testing a Thousand Brains style claim.

- lower-level grounded object memories should remain reusable rather than being discarded
  into a flat parent template
- higher-level structure should be built by composition over grounded child outputs
- a positive result here supports hierarchical reuse as a plausible path toward more
  abstract concepts

## Biological Plausibility Review

- biologically plausible within the theory: reuse of grounded parts by higher-level
  modules
- engineering approximations: the current parent supervision scheme and benchmark labels
- temporary hacks: benchmark-specific object and parent-to-child mappings
- explicitly non-biological compromises: supervised experiment scaffolding and synthetic
  dataset construction

This phase does not introduce new neural network helper systems.

## Design Options Considered

### Option A: Stop at training completion only

- benefits: minimal extra work
- risks: no real evidence that composition helps
- expected scientific value: low
- expected engineering cost: low

### Option B: Run level-1 compositional versus monolithic inference and one higher-level follow-up benchmark

- benefits: directly tests the roadmap claim with existing repo configs and trained models
- risks: only partially covers data efficiency and noise-specific ablations
- expected scientific value: high
- expected engineering cost: moderate

### Option C: Build new benchmark machinery before evaluating current models

- benefits: potentially stronger long-term benchmark design
- risks: delays the core go or no-go decision
- expected scientific value: unclear until the current benchmark is actually measured
- expected engineering cost: high

Preferred option: Option B.

## Core Hypothesis

The main hypothesis is:

> A compositional Monty hierarchy should be able to reuse previously learned grounded
> part representations to build larger parent-object models more effectively than
> learning each parent object monolithically from scratch.

In practical terms, if the system already knows objects such as `mug` and `tbp_logo`,
then a higher-level LM should be able to learn `mug_with_tbp_logo` by consuming
lower-level object IDs and their relative structure, rather than treating the entire
composition as a completely new undifferentiated object.

## What We Are Trying To Prove

This plan is intended to test five claims.

### 1. Hierarchical composition works in the current implementation

The first claim is purely implementation-level:

- lower-level LMs can learn parts
- those part IDs can be emitted as features
- a higher-level LM can consume them
- the higher-level LM can store parent object models
- the full training and inference pipeline runs end to end

If this fails, then hierarchy is still only a conceptual direction in this repo and not
yet a working mechanism for compositional abstraction.

### 2. Higher-level LMs can represent parent objects using child structure

The second claim is representational:

- the parent LM should use child object IDs and relative arrangement as part of its
  representation
- parent objects should not need to be modeled only as flat appearance templates

This is the simplest concrete version of abstraction in the current codebase.

### 3. Compositional learning should beat or at least justify itself against a monolithic baseline

The compositional hierarchy is only worth keeping if it yields at least one real
advantage over a monolithic model.

Acceptable advantages include:

- better parent-object accuracy
- better robustness when lower-level detections are noisy
- fewer required labeled parent examples
- better generalization to new compositions of known parts

If none of these appear, then the hierarchy is only extra complexity.

### 4. Reuse of known parts is a viable path toward abstraction

If this benchmark succeeds, it supports a broader claim:

- grounded parts can be reused to form higher-level models
- abstraction in Monty may emerge by composition rather than by directly injecting an
  external symbolic layer everywhere

This matters for future benchmarks like:

- strokes to letters
- letters to words
- word-image compositions
- scene-like object arrangements

### 5. This benchmark can act as the bridge between Monty and a future teacher-student system

The broader teacher-student idea depends on Monty being able to internalize some higher
level structure, not just memorize isolated grounded objects.

If the hierarchy works here, then a standard AI teacher could plausibly help Monty learn
curricula of compositional concepts while Monty still owns the grounded representations.

## Teacher-Student Idea

The intended long-term architecture is a hybrid system.

### Teacher

The teacher is a standard multimodal AI model or toolchain.

Its roles are:

- generate or retrieve examples
- provide text labels, captions, and instructions
- propose curricula
- create synthetic pairings such as word-image-object combinations
- tutor Monty during early training

The teacher is not the final memory substrate for Monty.

### Student

Monty is the grounded student.

Its roles are:

- learn from sensorimotor interaction
- build object and pose models
- detect objects and parts from real or simulated experience
- eventually compose higher-level representations from lower-level ones

Monty is responsible for the grounded, reusable world model.

### Bridge

There are two candidate bridge mechanisms between teacher knowledge and Monty.

#### Option A: Explicit semantic bridge

This is a separate symbolic or relational memory that maps concepts, words, and labels to
Monty graph IDs.

This is the lowest-risk engineering option.

#### Option B: Higher-level LM hierarchy

This is the Monty-native option where higher LMs consume lower LM outputs and gradually
learn more abstract or compositional models.

This benchmark is specifically testing whether Option B is viable.

The likely long-term system may need both:

- explicit bridge memory for flexible symbolic lookup
- higher-level LMs for grounded composition and abstraction

## Why Start With Logos On Objects

The current benchmark is intentionally simple.

The dataset combines:

- base objects such as cube, disk, cylinder, sphere, mug
- logo parts such as `logo_tbp` and `logo_numenta`
- composed parent objects such as `cube_tbp_horz`, `mug_numenta_vert`, and
  `mug_tbp_horz_bent`

This makes it a good first test because:

- the child objects are clear and reusable
- the parent objects are structured compositions
- the lower-level LM and higher-level LM responsibilities are easy to interpret
- the repo already contains configs, mappings, and metrics for this benchmark

This benchmark is the smallest available controlled test of compositional abstraction in
the current repo.

## Current Repo-Specific Benchmark Structure

The benchmark already exists in the repository.

Relevant pieces include:

- `supervised_pre_training_flat_objects_wo_logos`
- `supervised_pre_training_logos_after_flat_objects`
- `supervised_pre_training_curved_objects_after_flat_and_logo`
- `supervised_pre_training_objects_with_logos_lvl1_comp_models`
- `supervised_pre_training_objects_with_logos_lvl1_monolithic_models`
- `supervised_pre_training_objects_with_logos_lvl2_comp_models`
- `supervised_pre_training_objects_with_logos_lvl3_comp_models`
- `supervised_pre_training_objects_with_logos_lvl4_comp_models`
- `infer_comp_lvl1_with_comp_models`
- `infer_comp_lvl1_with_monolithic_models`

The parent-to-child object mapping is defined by the repo and already used by the logging
code for compositional metrics.

## Training Plan

### Phase 0: Environment and dataset preparation

Requirements:

- compositional dataset available under `MONTY_DATA`
- working `tbp.monty` Python environment
- low-memory execution settings to avoid Habitat overload during long runs

Execution policy:

- use one worker (`num_parallel=1`) for remaining long runs when stability matters
- cap math library thread counts
- avoid running multiple Habitat training jobs at once

### Phase 1: Learn flat child objects

Run:

- `supervised_pre_training_flat_objects_wo_logos`

Purpose:

- teach lower-level LM the base shape objects first

Expected outcome:

- stable child-object memory for flat object parts

### Phase 2: Learn logo child objects

Run:

- `supervised_pre_training_logos_after_flat_objects`

Purpose:

- add logo child objects while preserving previous base shapes

Expected outcome:

- lower-level LM can now emit shape IDs and logo IDs as reusable components

### Phase 3: Learn curved child objects and richer base objects

Run:

- `supervised_pre_training_curved_objects_after_flat_and_logo`

Purpose:

- extend child-object repertoire to curved and more realistic base objects

Expected outcome:

- lower-level LM has the component vocabulary needed for compositional training

### Phase 4: Train compositional level-1 higher LM

Run:

- `supervised_pre_training_objects_with_logos_lvl1_comp_models`

Purpose:

- first higher-level LM consumes lower LM output as a feature
- learns parent objects under supervised parent labels

Expected outcome:

- parent-object models stored using lower-level `object_id` signals

### Phase 5: Train monolithic baseline

Run:

- `supervised_pre_training_objects_with_logos_lvl1_monolithic_models`

Purpose:

- provide the baseline for comparison

Expected outcome:

- a non-hierarchical model family trained on the same benchmark distribution

### Phase 6: Expand compositional training through higher levels

Run in order:

- `supervised_pre_training_objects_with_logos_lvl2_comp_models`
- `supervised_pre_training_objects_with_logos_lvl3_comp_models`
- `supervised_pre_training_objects_with_logos_lvl4_comp_models`

Purpose:

- test whether the hierarchy continues to support richer or more numerous
  compositions

Expected outcome:

- increasingly complete parent-model checkpoints at higher compositional levels

## Inference and Evaluation Plan

### Compositional evaluation

Run:

- `infer_comp_lvl1_with_comp_models`

Purpose:

- measure how the hierarchical model performs at recognizing compositional objects

### Monolithic evaluation

Run:

- `infer_comp_lvl1_with_monolithic_models`

Purpose:

- establish whether the compositional hierarchy actually helps compared with a flat
  parent-model baseline

### Metrics to collect

The repo already supports the main compositional metrics.

Primary ones:

- overall parent-object accuracy
- `consistent_child_obj`
- `mlh_prediction_error`

Interpretation:

- high parent accuracy means the higher LM recognizes parent objects correctly
- high `consistent_child_obj` means the lower LM is identifying plausible children even
  when it does not own the full parent model
- lower prediction error suggests better alignment of inferred model and sensed input

## Deliverables

- level-1 compositional inference run completed under the Phase 0 manifest discipline
- level-1 monolithic comparison run completed under the same discipline
- one follow-up compositional benchmark beyond level 1 completed under the same
  discipline
- machine-readable result summary for parent accuracy, child consistency, and prediction
  error comparisons
- updated phase document with a recommendation based on the measured results

## Success Criteria

The benchmark is successful if at least one of the following holds for the compositional
model relative to the monolithic baseline:

- higher parent accuracy
- better `consistent_child_obj` behavior while preserving parent performance
- better robustness to ambiguity or viewpoint changes
- fewer parent-labeled examples required for similar performance
- evidence that known child parts are truly reused in parent recognition

The benchmark is especially valuable if the compositional model performs comparably now,
but shows a clearer path toward generalization and reuse than the monolithic model.

## Failure Criteria

The benchmark does not support the hierarchy if:

- the higher LM adds no measurable benefit
- the child-object signal is too noisy to support stable parent learning
- training is too brittle to reproduce reliably
- monolithic models dominate on both accuracy and simplicity

In that case, the broader teacher-student plan should lean more heavily on an explicit
semantic bridge and treat higher-level LM composition as future work.

## Why This Matters For The Bigger Vision

The broader long-term vision is:

- Monty sees, reads, and acts in the world
- a standard multimodal AI teaches and tutors Monty early on
- Monty gradually internalizes grounded reusable structure from interaction
- higher-level Monty modules learn concepts built from lower-level grounded parts

This benchmark is the first practical controlled test of whether Monty can start moving
from grounded local object memory toward reusable abstraction.

If it succeeds, the next research steps become more credible:

- strokes to letters
- letters to words
- words to grounded concepts
- image-word and object-word composition
- screen-text-image interaction tasks

If it fails, then the path toward those goals likely requires either:

- more robust higher-level LM mechanisms
- stronger top-down feedback
- an explicit semantic bridge layer
- or all three

## Operational Constraints

The recent runs showed that stability matters.

To reduce the risk of freezing the machine:

- avoid multiple parallel Habitat jobs during long phases
- run one worker at a time for remaining expensive stages
- cap numerical library threads
- checkpoint between phases
- prefer resuming partial runs over restarting whole stages

## Benchmark And Test Plan

### Benchmark A: Level-1 compositional inference

- name: `infer_comp_lvl1_with_comp_models`
- input setup: existing level-1 compositional pretrained model under `my_trained_models`
- baseline: level-1 monolithic inference on the same object set and rotations
- target metric: parent accuracy, `consistent_child_obj`, and `mlh_prediction_error`
- failure condition: no measurable compositional advantage or unstable execution

### Benchmark B: Level-1 monolithic comparison

- name: `infer_comp_lvl1_with_monolithic_models`
- input setup: existing level-1 monolithic pretrained model under `my_trained_models`
- baseline: level-1 compositional inference on the same object set and rotations
- target metric: parent accuracy, `consistent_child_obj`, and `mlh_prediction_error`
- failure condition: benchmark not directly comparable to Benchmark A

### Benchmark C: Level-2 compositional follow-up

- name: `infer_comp_lvl2_with_comp_models`
- input setup: existing level-2 compositional pretrained model under `my_trained_models`
- baseline: level-1 compositional performance and expected part reuse on a richer object
  set
- target metric: successful recognition on additional parent combinations built from
  known parts and child-consistency behavior on the expanded benchmark
- failure condition: execution failure or no evidence that the hierarchy scales beyond
  the initial level-1 set

## Execution Registry

Record every non-trivial run here before launching it.

### Run A: Level-1 compositional inference

- run name: `phase1_infer_lvl1_comp`
- purpose: measure level-1 compositional parent-object recognition with the already
  trained compositional model
- prepare command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py prepare --experiment infer_comp_lvl1_with_comp_models --override ++benchmarks.pretrained_dir=/Users/felixrobles/tbp/results/monty/pretrained_models/my_trained_models --override ++experiment.config.logging.output_dir=/Users/felixrobles/tbp/results/monty/projects/phase1_eval_runs --override ++experiment.config.logging.run_name=phase1_infer_lvl1_comp --override ++experiment.config.logging.wandb_handlers=[] --num-parallel 1`
- launch command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py launch --manifest /Users/felixrobles/tbp/results/monty/projects/phase1_eval_runs/phase1_infer_lvl1_comp/phase0_manifest.json --foreground`
- monitor command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py status --manifest /Users/felixrobles/tbp/results/monty/projects/phase1_eval_runs/phase1_infer_lvl1_comp/phase0_manifest.json`
- expected output path: `/Users/felixrobles/tbp/results/monty/projects/phase1_eval_runs/phase1_infer_lvl1_comp`
- resume or recovery command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py resume --manifest /Users/felixrobles/tbp/results/monty/projects/phase1_eval_runs/phase1_infer_lvl1_comp/phase0_manifest.json`
- stop command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py stop --manifest /Users/felixrobles/tbp/results/monty/projects/phase1_eval_runs/phase1_infer_lvl1_comp/phase0_manifest.json`
- summary command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py summarize --manifest /Users/felixrobles/tbp/results/monty/projects/phase1_eval_runs/phase1_infer_lvl1_comp/phase0_manifest.json`
- resource profile: `num_parallel=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`
- current status: completed
- status result: `state=completed`, `outputs_complete=true`, `completed_episodes=[0..83]`, `missing_episodes=[]`
- summary result: `eval_rows=168`, `percent_correct=39.88095238095239`, `performance_counts={"consistent_child_obj": 75, "correct_mlh": 39, "correct": 28, "confused_mlh": 26}`

### Run B: Level-1 monolithic inference

- run name: `phase1_infer_lvl1_monolithic`
- purpose: establish the direct monolithic baseline on the same level-1 benchmark
- prepare command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py prepare --experiment infer_comp_lvl1_with_monolithic_models --override ++benchmarks.pretrained_dir=/Users/felixrobles/tbp/results/monty/pretrained_models/my_trained_models --override ++experiment.config.logging.output_dir=/Users/felixrobles/tbp/results/monty/projects/phase1_eval_runs --override ++experiment.config.logging.run_name=phase1_infer_lvl1_monolithic --override ++experiment.config.logging.wandb_handlers=[] --num-parallel 1`
- launch command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py launch --manifest /Users/felixrobles/tbp/results/monty/projects/phase1_eval_runs/phase1_infer_lvl1_monolithic/phase0_manifest.json --foreground`
- monitor command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py status --manifest /Users/felixrobles/tbp/results/monty/projects/phase1_eval_runs/phase1_infer_lvl1_monolithic/phase0_manifest.json`
- expected output path: `/Users/felixrobles/tbp/results/monty/projects/phase1_eval_runs/phase1_infer_lvl1_monolithic`
- resume or recovery command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py resume --manifest /Users/felixrobles/tbp/results/monty/projects/phase1_eval_runs/phase1_infer_lvl1_monolithic/phase0_manifest.json`
- stop command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py stop --manifest /Users/felixrobles/tbp/results/monty/projects/phase1_eval_runs/phase1_infer_lvl1_monolithic/phase0_manifest.json`
- summary command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py summarize --manifest /Users/felixrobles/tbp/results/monty/projects/phase1_eval_runs/phase1_infer_lvl1_monolithic/phase0_manifest.json`
- resource profile: `num_parallel=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`
- current status: completed after repair of resumed-eval consolidation artifacts
- status result: `state=completed`, `outputs_complete=true`, `completed_episodes=[0..83]`, `missing_episodes=[]`
- summary result: `eval_rows=168`, `percent_correct=50.595238095238095`, `performance_counts={"correct_mlh": 71, "confused_mlh": 37, "consistent_child_obj": 24, "confused": 22, "correct": 14}`

### Run C: Level-2 compositional follow-up

- run name: `phase1_infer_lvl2_comp`
- purpose: test whether composition remains useful on a richer object set containing additional parent combinations built from known parts
- prepare command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py prepare --experiment infer_comp_lvl2_with_comp_models --override ++benchmarks.pretrained_dir=/Users/felixrobles/tbp/results/monty/pretrained_models/my_trained_models --override ++experiment.config.logging.output_dir=/Users/felixrobles/tbp/results/monty/projects/phase1_eval_runs --override ++experiment.config.logging.run_name=phase1_infer_lvl2_comp --override ++experiment.config.logging.wandb_handlers=[] --num-parallel 1`
- launch command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py launch --manifest /Users/felixrobles/tbp/results/monty/projects/phase1_eval_runs/phase1_infer_lvl2_comp/phase0_manifest.json --foreground`
- monitor command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py status --manifest /Users/felixrobles/tbp/results/monty/projects/phase1_eval_runs/phase1_infer_lvl2_comp/phase0_manifest.json`
- expected output path: `/Users/felixrobles/tbp/results/monty/projects/phase1_eval_runs/phase1_infer_lvl2_comp`
- resume or recovery command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py resume --manifest /Users/felixrobles/tbp/results/monty/projects/phase1_eval_runs/phase1_infer_lvl2_comp/phase0_manifest.json`
- stop command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py stop --manifest /Users/felixrobles/tbp/results/monty/projects/phase1_eval_runs/phase1_infer_lvl2_comp/phase0_manifest.json`
- summary command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py summarize --manifest /Users/felixrobles/tbp/results/monty/projects/phase1_eval_runs/phase1_infer_lvl2_comp/phase0_manifest.json`
- resource profile: `num_parallel=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`
- current status: completed
- status result: `state=completed`, `outputs_complete=true`, `completed_episodes=[0..209]`, `missing_episodes=[]`
- summary result: `eval_rows=420`, `percent_correct=34.523809523809526`, `performance_counts={"consistent_child_obj": 184, "confused_mlh": 88, "correct_mlh": 75, "correct": 70, "confused": 3}`

## Resource Budget

- expected CPU usage: moderate
- expected RAM usage: moderate but bounded by single-worker execution
- expected runtime: multi-minute per evaluation run
- default safety flags: `num_parallel=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`
- reason if exceeding the default budget: none

## Recovery Plan

- each run writes a manifest, config snapshot, stable log path, and helper scripts under
  `/Users/felixrobles/tbp/results/monty/projects/phase1_eval_runs/<run_name>`
- after any interruption, use the recorded `status` command first
- if a run is partial, use the recorded `resume` command rather than relaunching from
  scratch
- use the recorded `summary` command to regenerate the current machine-readable output
  summary after restart

## Observability Plan

- primary observability artifact: `eval_stats.csv` plus the Phase 0 machine-readable
  status and summary outputs for each run
- comparison artifact to produce in this phase: a concise result summary comparing
  parent accuracy, child consistency, and prediction-error behavior across the level-1
  compositional and monolithic runs

## Implementation Plan

1. Run the level-1 compositional inference benchmark using Run A.
2. Run the level-1 monolithic comparison using Run B.
3. Run the level-2 compositional follow-up using Run C.
4. Summarize the resulting evaluation CSVs and decide whether the compositional path has
   demonstrated a clear enough advantage to pass the phase.

## Final Result Summary

- level-1 compositional result: `168` rows across `84` episode seeds, `39.88095238095239%`
  strict correct (`correct` or `correct_mlh`), and `75` `consistent_child_obj`
  outcomes
- level-1 monolithic result: `168` rows across `84` episode seeds,
  `50.595238095238095%` strict correct, and `24` `consistent_child_obj` outcomes
- level-2 compositional follow-up: `420` rows across `210` episode seeds,
  `34.523809523809526%` strict correct, and `184` `consistent_child_obj` outcomes
- interpretation: the compositional hierarchy does show a much stronger child-reuse
  signature than the monolithic baseline on level 1, but it does not win the primary
  accuracy comparison; under the original Phase 1 exit criterion, this is not yet a
  positive pass
- reliability note: the monolithic run initially suffered from a resumed-eval cleanup
  bug that renamed the previous top-level `eval_stats.csv` and only wrote the resumed
  batch; that behavior is now fixed in code, Phase 0 completion detection now treats
  leftover `parallel_eval_episode_*` directories as partial state, and the final
  monolithic CSV was repaired from disjoint surviving artifacts covering all `84`
  episode seeds before the manifest was revalidated as completed

## Current Execution Status

Final state:

- environment and data are ready
- config validation is done
- flat, logo, and curved child-object training are complete
- level-1 compositional training is complete
- level-1 monolithic baseline training is complete
- level-2 compositional training is complete
- level-3 compositional training is complete
- level-4 compositional training is complete
- level-1 compositional inference is complete
- level-1 monolithic inference is complete after repaired resumed-eval output
- level-2 compositional follow-up is complete
- code fixes for resumed-eval merge behavior and partial-output detection are complete
- unit and targeted integration validation for the runner changes are complete

## Status

- investigating: completed
- prototyping: completed
- running benchmarks: completed
- blocked: no current blocker

## Decision Log

- 2026-03-15: Chose level-1 compositional versus monolithic inference as the primary
  go or no-go comparison for Phase 1.
- 2026-03-15: Chose level-2 compositional inference as the minimum follow-up benchmark
  because it expands to additional parent combinations without requiring new benchmark
  machinery.
- 2026-03-16: Fixed resumed parallel evaluation so partial eval reruns merge into the
  existing top-level `eval_stats.csv` instead of renaming it away and silently losing
  already consolidated episodes.
- 2026-03-16: Marked Phase 1 execution complete with a negative result on the primary
  exit criterion because the monolithic level-1 baseline outperformed the compositional
  model on strict correctness, despite the compositional model showing much stronger
  `consistent_child_obj` behavior.

## Current Blockers

- none for Phase 1 execution; the remaining blocker is scientific rather than
  operational, namely whether the child-consistency signal is strong enough to justify
  additional targeted ablations before moving on

## Open Questions

- can a targeted noise or data-efficiency ablation convert the observed
  `consistent_child_obj` advantage into a measurable end-task win over monolithic
  training?
- should the next phase depend on a Phase 1.5 ablation, or is the current negative
  result enough to prioritize a heavier explicit semantic bridge?

## Immediate Next Actions

1. Record this phase as execution-complete but not passed on the original exit
  criterion.
2. Design one focused follow-up ablation on noise robustness or parent-data efficiency.
3. Decide whether Phase 2 should wait for that ablation or proceed with a weaker claim
  about compositional reuse.

## Next Decision

- decision to make: whether to treat Phase 1 as sufficient evidence to continue the
  hierarchy-first roadmap, or require a Phase 1.5 ablation before proceeding
- evidence needed: the completed level-1 comparison, the completed level-2 follow-up,
  and at least one targeted ablation if a stronger claim is still desired
- fallback plan: if no near-term ablation is planned, proceed with roadmap updates that
  explicitly state Phase 1 completed execution but failed the original success bar

## Bottom Line

The practical reason for doing all of this is simple:

we are trying to determine whether Monty can build higher-level reusable concepts from
lower-level grounded parts, rather than learning every composed object from scratch as a
flat appearance pattern.

If that claim holds, then Monty has a plausible path toward grounded abstraction.
If it does not, then the teacher-student system will need a heavier explicit semantic
bridge and the current hierarchical LM path is not yet strong enough on its own.