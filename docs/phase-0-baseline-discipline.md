# Phase 0: Baseline Discipline

## Purpose

This phase makes the current Monty benchmark workflow reliable enough that later work
is not built on brittle commands, undocumented recovery tricks, or manual checkpoint
surgery.

The goal is not to redesign Monty's learning algorithms yet. The goal is to make the
current benchmark stack reproducible, restart-safe, monitorable, and inspectable.

## Metadata

- phase: Phase 0: Baseline Discipline
- roadmap link: [docs/full-system-roadmap.md](docs/full-system-roadmap.md)
- owner: local working plan
- status: completed
- last updated: 2026-03-15

## Goal

Create a stable baseline layer for current benchmark execution.

## Why This Phase Exists

The current system already has meaningful benchmarks, but execution is still too
fragile. Later phases will be hard to evaluate honestly unless runs are recorded,
restart-safe, resumable where possible, and summarized automatically.

## Current Hypothesis

> A thin orchestration layer around the existing `run_parallel.py` workflow is enough
> to make current Monty benchmark execution reproducible and restart-safe without
> changing the underlying learning algorithms.

## Investigation Summary

Current baseline facts:

- `run_parallel.py` already supports episode filtering with `episodes=`.
- parallel evaluation already consolidates `eval_stats.csv` and logs.
- parallel training already consolidates `train_stats.csv` and merged checkpoints.
- interrupted runs are still awkward because run metadata, missing episodes, and output
  expectations are not recorded in one place.
- recent compositional work required manual checkpoint handling and ad hoc monitoring.

## TBT And Jeff Hawkins Lens

This phase is aligned with the spirit of Monty and TBT.

- It does not replace the core modeling system.
- It improves scientific discipline around benchmark execution.
- It preserves interpretable, inspectable workflows.
- It helps us evaluate whether the architecture actually works instead of hiding behind
  one-off runs.

## Biological Plausibility Review

This phase is almost entirely engineering-facing.

- biologically plausible within the theory: not directly applicable
- engineering approximations: run manifests, status tools, and summaries
- temporary hacks: none intended
- explicitly non-biological compromises: orchestration scripts and process tracking

This phase does not change CMP, LM internals, routing, voting, or timing.

## Design Options Considered

### Option A: Keep using raw `run_parallel.py` directly

- benefits: no new code
- risks: repeated undocumented operational mistakes
- expected scientific value: low
- expected engineering cost: low short-term, high long-term

### Option B: Build a thin run-management layer around existing runners

- benefits: keeps algorithm path unchanged while adding manifests, status, resume, and
  summaries
- risks: some duplication of run metadata logic
- expected scientific value: high
- expected engineering cost: moderate

### Option C: Rewrite the full runner stack

- benefits: potentially cleaner long-term
- risks: too much scope for Phase 0
- expected scientific value: unclear
- expected engineering cost: high

Preferred option: Option B.

## Neural Nets, HTM, And Other Helper Systems

This phase does not introduce deep neural nets, HTM mechanisms, SDR changes, or helper
models.

## Deliverables

- `tools/phase0_runs.py` CLI wrapper
- importable run-management logic in `src/tbp/monty/frameworks/utils/phase0_runs.py`
- run manifest and config snapshots written before launch
- status, resume, stop, and summary commands
- per-run helper scripts saved into the output directory
- Phase 0 document linked from the roadmap

## Benchmark And Test Plan

### Lightweight validation benchmark

- name: `test/supervised_pre_training`
- input setup: repo test config
- baseline: existing direct `run_parallel.py` behavior
- target metric: manifest creation, run launch, result summary, and expected model file
- failure condition: missing manifest, missing summary, or wrong output detection

### Lightweight evaluation benchmark

- name: `test/eval`
- input setup: repo test config
- baseline: existing direct `run_parallel.py` behavior
- target metric: manifest creation, status reporting, and `eval_stats.csv` summary
- failure condition: wrong output path or wrong completion detection

## Execution Registry

Record every non-trivial run here before launching it.

### Run A: Lightweight supervised pretraining validation

- run name: `phase0_final_supervised_pre_training`
- purpose: validate manifest creation, real training execution, status recovery, and summary generation for parallel pretraining
- prepare command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py prepare --experiment test/supervised_pre_training --override ++experiment.config.logging.output_dir=/tmp/tbp-phase0-final/train --override ++experiment.config.logging.run_name=phase0_final_supervised_pre_training --num-parallel 1`
- launch command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py launch --manifest /tmp/tbp-phase0-final/train/phase0_final_supervised_pre_training/phase0_manifest.json --foreground`
- monitor command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py status --manifest /tmp/tbp-phase0-final/train/phase0_final_supervised_pre_training/phase0_manifest.json`
- expected output path: `/tmp/tbp-phase0-final/train/phase0_final_supervised_pre_training`
- resume or recovery command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py resume --manifest /tmp/tbp-phase0-final/train/phase0_final_supervised_pre_training/phase0_manifest.json`
- stop command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py stop --manifest /tmp/tbp-phase0-final/train/phase0_final_supervised_pre_training/phase0_manifest.json`
- summary command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py summarize --manifest /tmp/tbp-phase0-final/train/phase0_final_supervised_pre_training/phase0_manifest.json`
- resource profile: `num_parallel=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`
- current status: completed
- observed outputs: `pretrained/model.pt`, `parallel_log.txt`, `phase0_manifest.json`, `phase0_config.yaml`, helper scripts, `log.txt`
- status result: `state=completed`, `outputs_complete=true`, `completed_episodes=[0,1]`, `missing_episodes=[]`
- summary result: `model_exists=true`, `train_stats_exists=false`, `lm_graph_counts={"0": 2}`

### Run B: Lightweight evaluation validation

- run name: `phase0_final_eval`
- purpose: validate manifest creation, real evaluation execution, status recovery, and summary generation for parallel evaluation using a freshly produced lightweight pretrained model
- prepare command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py prepare --experiment test/eval --override ++experiment.config.logging.output_dir=/tmp/tbp-phase0-final/eval --override ++experiment.config.logging.run_name=phase0_final_eval --override ++experiment.config.model_name_or_path=/tmp/tbp-phase0-final/train/phase0_final_supervised_pre_training/pretrained --num-parallel 1`
- launch command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py launch --manifest /tmp/tbp-phase0-final/eval/phase0_final_eval/phase0_manifest.json --foreground`
- monitor command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py status --manifest /tmp/tbp-phase0-final/eval/phase0_final_eval/phase0_manifest.json`
- expected output path: `/tmp/tbp-phase0-final/eval/phase0_final_eval`
- resume or recovery command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py resume --manifest /tmp/tbp-phase0-final/eval/phase0_final_eval/phase0_manifest.json`
- stop command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py stop --manifest /tmp/tbp-phase0-final/eval/phase0_final_eval/phase0_manifest.json`
- summary command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/phase0_runs.py summarize --manifest /tmp/tbp-phase0-final/eval/phase0_final_eval/phase0_manifest.json`
- resource profile: `num_parallel=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`
- current status: completed
- observed outputs: `eval_stats.csv`, `detailed_run_stats.json`, `parallel_log.txt`, `phase0_manifest.json`, `phase0_config.yaml`, helper scripts, `log.txt`
- status result: `state=completed`, `outputs_complete=true`, `completed_episodes=[0,1,2,3,4,5]`, `missing_episodes=[]`
- summary result: `eval_stats_exists=true`, `eval_rows=6`, `percent_correct=16.666666666666664`, `performance_counts={"correct": 1, "no_match": 5}`

Rules:

- Do not launch long-running work before filling in this section.
- Commands written here must also be shown to the user in chat when the run is being
  proposed or started.
- Assume the machine, terminal, or editor can restart at any time.
- If the run is not recoverable, say so explicitly and explain the fallback.

## Resource Budget

- expected CPU usage: low to moderate for test configs
- expected RAM usage: low relative to full Habitat benchmark runs
- expected runtime: short
- default safety flags: `num_parallel=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`
- reason if exceeding the default budget: none for current validation plan

## Recovery Plan

- manifests are written to the output directory before launch
- config snapshots are written to `phase0_config.yaml`
- logs are written to `phase0_run.log`
- helper scripts for launch, status, resume, stop, and summary are written to the same
  output directory
- after restart, use `python tools/phase0_runs.py status --manifest <manifest>` to
  determine whether a run is active, partial, or complete
- if interrupted and resumable, use `python tools/phase0_runs.py resume --manifest <manifest>`
- use `python tools/phase0_runs.py summarize --manifest <manifest>` to regenerate the
  current output summary

## Observability Plan

Phase 0 observability focuses on execution state rather than model internals.

Required artifacts:

- manifest JSON
- config snapshot YAML
- run log
- machine-readable status output
- machine-readable summary output

## Implementation Plan

1. Add an importable Phase 0 run-management module.
2. Add a CLI wrapper in `tools/`.
3. Add unit tests for episode formatting, output detection, and manifest building.
4. Link the phase document into the roadmap registry.
5. Validate the tool against the repo's lightweight test configs.

## Validation Results

- unit tests: `tests/unit/frameworks/utils/phase0_runs_test.py` passed in the repo conda environment after adding resolver-reset and training-artifact regression coverage; final result was `11 passed`
- live training validation: completed successfully through `prepare`, `launch`, `status`, and `summarize`
- live evaluation validation: completed successfully through `prepare`, `launch`, `status`, and `summarize`
- real defect fixed during validation: manifest-based `status` and `summarize` failed in a fresh CLI process until custom Hydra resolvers were re-registered before config composition
- real defect fixed during validation: training artifact prediction incorrectly required `train_stats.csv` even when the configured logging handlers did not emit it

## Status

- investigating: completed
- prototyping: completed
- running benchmarks: completed
- blocked: no current blocker

## Decision Log

- 2026-03-15: Chose a thin orchestration layer around `run_parallel.py` instead of
  rewriting the runner stack.
- 2026-03-15: Chose stable per-run manifests and helper scripts as the restart-safe
  baseline format.
- 2026-03-15: Chose conservative local resource defaults because this machine is
  limited to 8 CPU cores and 16 GB RAM.
- 2026-03-15: Fixed fresh-process manifest reload by explicitly re-registering custom
  Hydra resolvers before Phase 0 config composition.
- 2026-03-15: Fixed training output detection so Phase 0 only requires
  `train_stats.csv` when the configured logging handlers actually emit CSV stats.

## Current Blockers

- none for Phase 0 completion

## Open Questions

- should the same tool later support `run.py` as well as `run_parallel.py`?
- should future summaries include automatic benchmark dashboards or stay file-based?
- how much generic checkpoint merge logic should stay in Phase 0 versus later phases?

## Success Criteria

- a run manifest is always written before launch
- a user can check status from a saved manifest after restart
- missing episodes can be detected and converted into a resume command
- current training and evaluation outputs can be summarized automatically
- the workflow remains conservative on local CPU and RAM usage by default

All Phase 0 success criteria were satisfied on the 2026-03-15 lightweight validation
runs.

## Failure Criteria

- output detection is wrong for current benchmark configs
- resume logic produces invalid episode selections
- manifest and helper scripts drift from actual command behavior
- the tool increases operational complexity more than it reduces it

## Next Decision

- decision to make: whether to apply the Phase 0 workflow directly to the active
  compositional benchmark workflow and Phase 1 experiments now that the lightweight
  validation path is complete
- evidence needed: a first real Phase 1 run recorded using the same manifest/status/
  summary discipline
- fallback plan: keep the Phase 0 tool as the default launcher for new Phase 1 runs and
  extend resume coverage only when a concrete benchmark exposes a missing case