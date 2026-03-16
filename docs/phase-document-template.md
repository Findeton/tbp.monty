# Phase Document Template

## Purpose

Use this template for any active phase in [docs/full-system-roadmap.md](docs/full-system-roadmap.md).

The purpose of a phase document is to keep the loop stable.

- It should be the single source of truth for the phase.
- It should make implementation status visible.
- It should make scientific reasoning explicit.
- It should make failures and tradeoffs easy to inspect.

## Metadata

- phase:
- roadmap link:
- owner:
- status:
- last updated:

## Goal

State exactly what this phase is trying to prove or build.

## Why This Phase Exists

Explain why this phase is necessary for the long-term system.

## Current Hypothesis

Write the main hypothesis in a falsifiable way.

## Investigation Summary

Summarize relevant repo context, prior experiments, and external ideas that matter.

## TBT And Jeff Hawkins Lens

Describe how the phase aligns with the core conceptual ideas of Monty and the Thousand
Brains Theory.

Questions to answer:

- What would the most TBT-consistent solution look like?
- Does this phase preserve reference frames, sensorimotor structure, and reusable
  cortical-column-like computation?
- Is this phase moving Monty toward or away from a brain-like architecture?

## Biological Plausibility Review

State clearly which parts are:

- biologically plausible within the current theory
- engineering approximations
- temporary hacks
- explicitly non-biological compromises

If CMP, routing, timing, voting, state, or LM internals are changed, explain whether
the change makes biological sense.

## Design Options Considered

List the main options considered and why one is preferred.

For each option, include:

- benefits
- risks
- expected scientific value
- expected engineering cost

## Neural Nets, HTM, And Other Helper Systems

If deep neural nets, HTM-like mechanisms, SDRs, or any external helper models are part
of the phase, explain:

- why they are being considered
- what exact role they play
- whether they support Monty or replace a core Monty function
- why this is acceptable or not acceptable

## Deliverables

List the concrete outputs expected from this phase.

## Benchmark And Test Plan

List the experiments that will decide whether the phase succeeded.

For each benchmark, include:

- name
- input setup
- baseline
- target metric
- failure condition

## Execution Registry

Record every non-trivial run here before launching it.

For each run, include:

- run name:
- purpose:
- launch command:
- monitor command:
- expected output path:
- resume or recovery command:
- stop command:
- resource profile:
- current status:

Rules:

- Do not launch long-running work before filling in this section.
- Commands written here must also be shown to the user in chat when the run is being
  proposed or started.
- Assume the machine, terminal, or editor can restart at any time.
- If the run is not recoverable, say so explicitly and explain the fallback.

## Resource Budget

State the expected resource use and the safety limits.

Minimum fields:

- expected CPU usage:
- expected RAM usage:
- expected runtime:
- default safety flags:
- reason if exceeding the default budget:

For this machine, default safe assumptions are:

- 8 CPU cores total
- 16 GB RAM total
- one heavy run at a time
- `num_parallel=1` unless justified otherwise
- `OMP_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`
- `OPENBLAS_NUM_THREADS=1`

## Recovery Plan

Describe what to do if VS Code, the terminal session, or the whole computer restarts.

Include:

- where logs are written
- how to confirm whether the run is still active
- how to resume if interrupted
- how to determine whether outputs are complete or partial

## Observability Plan

In the fast.ai style, define how we will inspect what the system is doing.

At least one tool or artifact is required.

Examples:

- evidence visualization
- graph/state browser
- timer trace viewer
- prediction replay tool
- route and vote inspection tool
- benchmark dashboard

## Implementation Plan

Break the work into execution steps.

If a step involves running code, reference the corresponding entry in the execution
registry.

## Status

Use a short status block such as:

- not started
- investigating
- prototyping
- running benchmarks
- blocked
- completed

## Decision Log

Record major design decisions here with dates.

## Current Blockers

List the blockers that are preventing progress.

## Open Questions

List questions that still need evidence.

## Success Criteria

Write the exact conditions for considering this phase successful.

## Failure Criteria

Write the exact conditions that would cause the phase to be revised, paused, or
abandoned.

## Next Decision

- decision to make:
- evidence needed:
- fallback plan: