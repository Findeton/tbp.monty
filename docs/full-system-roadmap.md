# Full-System Roadmap (v1 — Superseded)

> **This document has been superseded by
> [full-system-roadmap-v2.md](full-system-roadmap-v2.md)**, which reorganizes the
> program from 10 sequential phases into 4 parallel research tracks based on
> empirical results from Phases 0-2. All operational rules, safety protocols, and
> policies have been carried forward into v2.
>
> This document is retained as historical reference for the original phase
> definitions and the v1→v2 mapping rationale documented in the v2 roadmap.

## Purpose (Historical)

This document turns the broad discussion about Monty's current limits into a concrete
development roadmap.

The goal is not to produce a vague vision statement. The goal is to define a sequence
of capabilities, benchmarks, decision points, and failure conditions that can be used
to guide implementation and evaluate progress.

This is a proposed roadmap for building a much more complete system around the core
ideas in `tbp.monty`. It should be treated as a working engineering and research plan,
not as official TBP project policy.

## Starting Point

Monty is currently strongest at the following:

- grounded static object modeling
- recognition in reference frames
- sensorimotor matching over explicit structured memories
- interpretable graph-based learning modules
- early hierarchy and compositional experiments

Monty is currently weak, incomplete, or still mostly conceptual in the following:

- robust heterarchy with rich top-down and lateral routing
- category-level and abstract conceptual generalization
- temporal modeling and prediction
- behavior and state learning
- imagination and counterfactual simulation
- language grounding and text-native interaction loops
- manipulation and motor skill learning
- scale invariance in the main default LM path
- broad multimodal transfer at system level
- memory consolidation and long-horizon semantic memory
- large-scale systems engineering and parallel execution
- multi-agent and social cognition

## Guiding Principles

The roadmap assumes the following design choices.

- Monty should remain the grounded structured world-modeling core.
- The final system should be hybrid, not purity-driven. If neural components are the
  best tool for compression, perception, policy distillation, or language priors, use
  them.
- Progress should be benchmarked phase by phase. We do not assume that later phases
  are worth doing if earlier ones fail.
- We should prefer a small number of decisive benchmarks over a large number of weak
  demonstrations.
- Every major phase needs a go or no-go checkpoint.

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

To keep the development loop stable and useful, every active phase should maintain the
following artifacts.

- one phase document that acts as the single source of truth for that phase
- one benchmark definition or experiment list tied to the phase goal
- one explicit decision log in the phase document
- one current blocker section in the phase document
- one observability artifact or tool that shows what the system is doing internally
- one execution registry for every non-trivial test, training run, or evaluation run

The loop is considered unstable if any of the following happen.

- benchmark status is spread across chat history only
- key design decisions are not written down
- a phase advances without a concrete test plan
- a phase introduces opaque machinery without an inspection method
- results cannot be reproduced from the phase document
- long-running commands are launched without being written down first
- a run cannot be monitored, resumed, or safely restarted after a machine or editor
  restart

## Execution Safety Protocol

Before launching any non-trivial test, training job, evaluation run, or data pipeline,
the following must be true.

- the exact launch command is written into the active phase document first
- the phase document includes a command to check run state
- the phase document includes the expected output location
- the phase document includes a recovery or resume command if the run is interrupted
- the phase document includes a stop or cleanup command if the run must be halted
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

No long-running phase work should depend on fragile in-memory context only.

- background runs should write logs to stable files
- output directories should be named predictably
- recovery steps should be documented before launch, not after failure
- if a job cannot be resumed cleanly, the phase document must say so explicitly and
  explain the workaround

### Chat Visibility Requirement

Commands stored only in documentation are not enough.

Whenever a run is about to be launched, the user should also see in chat:

- the launch command
- the monitor command
- the output path
- the resume or recovery command

This is required even if the same commands are already in the phase document.

## Resource Budget Policy

This machine is resource-constrained relative to typical research workloads.

Current default planning assumptions:

- 8 CPU cores
- 16 GB RAM

Therefore the default local execution policy is conservative.

- prefer one heavy run at a time
- prefer `num_parallel=1` unless a phase document justifies more
- prefer limiting math-library thread counts for heavy Python workloads
- avoid multiple simultaneous Habitat or similarly heavy jobs
- prefer detached or restart-safe runs for long jobs
- prefer smaller validation runs before full runs
- record resource-saving flags directly in the phase document

For CPU-heavy Python jobs, the default safe profile should usually include:

- `OMP_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`
- `OPENBLAS_NUM_THREADS=1`

If a phase needs a more aggressive resource profile, the phase document must explain:

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

Every phase must make the system easier to inspect.

- In the fast.ai style of practical interpretability, every important capability should
  be accompanied by a way to look at what the system is doing.
- New algorithms should ship with at least one inspection tool, visualization, replay,
  trace collector, state browser, or debug report.
- If a phase cannot be represented through a tool, add one before calling the phase
  stable.

Acceptable observability artifacts include:

- hypothesis visualizers
- graph memory browsers
- evidence heatmaps
- state and timer traces
- phase-specific dashboards
- prediction versus reality replays
- routing and voting inspection tools

## Phase Document Protocol

When a phase becomes active, create or update a dedicated phase document and link it in
this roadmap.

Every phase document must include:

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

Use the template in [docs/phase-document-template.md](docs/phase-document-template.md).

## Definition Of Success

For the purposes of this roadmap, a "full system" means a system that can do all of
the following in a single integrated architecture:

- learn grounded models from sensorimotor interaction
- generalize from instances to categories and concepts
- model time, state, behavior, and transformations
- predict likely future observations and action outcomes
- manipulate the world using model-based and compiled policies
- ground language in its learned world models
- compose concepts across modalities and levels of abstraction
- route information across a large heterarchical network efficiently
- scale to large memory, long tasks, and many interacting subsystems

It does not mean "equal to humans in every respect". It means reaching a system design
that is plausibly on that path and can be evaluated honestly.

## Program Structure

The roadmap is split into three kinds of work.

- Capability phases: the ordered dependencies that unlock new classes of cognition.
- System tracks: cross-cutting engineering work needed in parallel.
- Scorecard metrics: the fixed measurements used to assess progress over time.

## Scorecard

This scorecard should be updated whenever a phase meaningfully advances.

| Area | Current State | Target For Completion | Primary Benchmark Type |
| --- | --- | --- | --- |
| Static grounded recognition | Early strong point | Robust across domains and sensors | object recognition |
| Compositional part-whole learning | Prototype stage | Clear win over monolithic baselines | compositional objects |
| Category generalization | Weak | Generalize to new instances | category benchmarks |
| Abstract concept learning | Missing | Reusable concept graphs and relations | abstract task suite |
| Time and state | Missing in main path | State-aware inference and prediction | state/sequence benchmarks |
| Behavior learning | Theory plus prototype plans | Learn and recognize behaviors | behavior testbed |
| Imagination and counterfactuals | Missing | Roll out future hypotheses | prediction/planning tasks |
| Language grounding | Missing | Text linked to grounded models | language grounding tasks |
| Manipulation | Early conceptual stage | Multi-step object tasks | manipulation benchmarks |
| Heterarchy and routing | Partial | Efficient large-network routing | multi-LM tasks |
| Scale and systems | Research-grade | Stable large runs and faster matching | throughput and robustness |
| Multi-agent and theory of mind | Missing | Social and multi-agent modeling | multi-agent environments |

## Readiness Levels

Use the following levels when updating each scorecard area.

- `L0`: no implementation
- `L1`: theory only
- `L2`: prototype implementation
- `L3`: reproducible benchmark
- `L4`: strong result against baseline
- `L5`: integrated system capability

## Phase Document Registry

This registry links the currently active or already created phase documents.

| Phase | Status | Phase Document |
| --- | --- | --- |
| Phase 0: Baseline Discipline | completed | [docs/phase-0-baseline-discipline.md](docs/phase-0-baseline-discipline.md) |
| Phase 1: Make Composition Real | completed | [docs/teacher-student-compositional-plan.md](docs/teacher-student-compositional-plan.md) |
| Phase 2: Category And Concept Generalization | not started | [docs/phase-2-category-and-concept-generalization.md](docs/phase-2-category-and-concept-generalization.md) |
| Phase 3: Time, State, And Behavior | not started | pending |
| Phase 4: Prediction And Imagination | not started | pending |
| Phase 5: Language And Semantic Bridge | not started | pending |
| Phase 6: Manipulation And Skill Learning | not started | pending |
| Phase 7: Real Heterarchy, Routing, And Attention | not started | pending |
| Phase 8: Systems Scale | not started | pending |
| Phase 9: Multi-Agent And Social Modeling | not started | pending |
| Phase 10: Integrated Demonstrators | not started | pending |

## Phase 0: Baseline Discipline

### Goal

Make the current Monty evaluation loop reliable enough that later phases are not built
on noise, one-off runs, or path-dependent checkpoint accidents.

### Deliverables

- reproducible benchmark scripts for current object and compositional tasks
- reliable checkpoint merge and resume behavior
- standard result collection for train and eval runs
- versioned benchmark manifests for data, config, and model outputs

### Tests

- rerun the same benchmark twice and compare metric stability
- resume interrupted training without manual checkpoint surgery
- collect comparable output tables for compositional and monolithic baselines

### Exit Criteria

- core benchmark suite can run end to end with documented commands
- output metrics are collected automatically into stable locations
- interrupted runs can be resumed without custom ad hoc fixes

### Failure Signal

If baseline execution is still brittle, later capability work will be too expensive to
validate and Phase 0 is not complete.

Current state: completed on 2026-03-15 with lightweight live validation runs for both
parallel training and parallel evaluation, plus manifest-based status and summary
checks.

## Phase 1: Make Composition Real

### Goal

Prove that higher-level LMs can reuse grounded lower-level parts in a way that matters.

### Why This Comes First

If Monty cannot reuse grounded parts better than monolithic memorization, then the core
claim that structured hierarchy leads toward abstraction is still unproven.

### Deliverables

- compositional inference runs completed for current logo-on-object benchmark
- monolithic comparison runs completed on the same levels
- result summary for parent accuracy, child consistency, and robustness
- at least one follow-up benchmark where novel combinations of known parts appear

### Tests

- compare compositional versus monolithic accuracy
- compare data efficiency for new parent objects
- compare robustness to lower-level detection noise
- test generalization to new compositions of known parts

### Exit Criteria

- compositional hierarchy demonstrates at least one clear advantage over monolithic
  learning
- results are reproducible and not dependent on hand-tuned manual recovery

### Failure Signal

If monolithic models dominate on accuracy, simplicity, and data efficiency, then Monty
needs a different abstraction mechanism before moving forward.

Current state: execution completed on 2026-03-16. The level-1 monolithic baseline
outperformed the compositional model on strict correctness (`50.60%` vs `39.88%`),
while the compositional model showed a much stronger `consistent_child_obj` reuse
signal (`75` vs `24`). Phase 1 therefore completed with a negative result on the
original success bar and should feed either a targeted ablation or a stronger semantic
bridge plan before the roadmap claims a compositional win.

## Phase 2: Category And Concept Generalization

### Goal

Move from object-instance recognition to category-like and concept-like generalization.

### Missing Pieces

- better similarity-aware outputs from LMs
- category-aware datasets and metrics
- representations for transformations and part discovery
- cross-instance reuse beyond exact object memory

### Deliverables

- category benchmark with hierarchical labels
- evaluation on new instances of known categories
- stronger similarity-based higher-level inputs than the current default object ID path
- at least one benchmark for abstract relation or role, not just morphology

### Tests

- train on some instances of a category and evaluate on unseen instances
- test whether known parts emerge and are reused across objects
- measure category accuracy separately from exact instance accuracy

### Exit Criteria

- Monty can classify unseen instances correctly at category level with useful margins
- category-level representations are inspectable and not just hidden correlations in an
  external helper model

### Failure Signal

If Monty still behaves like a nearest-instance memory system, Phase 2 is not complete.

Current state: planning document drafted on 2026-03-15 in
[docs/phase-2-category-and-concept-generalization.md](docs/phase-2-category-and-concept-generalization.md).
Implementation has not started.

## Phase 3: Time, State, And Behavior

### Goal

Extend Monty from static structure into dynamic world modeling.

### Deliverables

- change-detecting sensor module
- state-conditioned models in the LM path
- state-aware hypotheses and CMP outputs
- timer or equivalent temporal-conditioning mechanism
- behavior testbed with at least one hinge-like object such as a stapler

### Tests

- recognize open versus closed object state
- recognize behavior independent of object orientation
- recognize the same behavior across different morphologies
- recognize behavior at different speeds

### Exit Criteria

- Monty can infer morphology, state, and behavior as separate but linked quantities
- state transitions and behavior recognition work in a reproducible benchmark

### Failure Signal

If time remains external bookkeeping instead of part of the representational system,
Monty is still a static recognition system.

## Phase 4: Prediction And Imagination

### Goal

Turn learned models into executable predictive models.

### Scope

This phase is not about image generation for its own sake. It is about counterfactual
rollout, expected observations, expected state transitions, and action-conditioned
prediction.

### Deliverables

- prediction interface for future observations given current hypotheses
- state-transition prediction for behaviors
- action-conditioned rollout for at least one manipulation domain
- evaluation metrics for predictive accuracy and planning usefulness

### Tests

- predict next observed features under current behavior state
- predict object state after a controlled action
- use prediction to reduce steps needed in recognition or manipulation tasks

### Exit Criteria

- predictions improve planning or recognition, not just visualization
- system can run short counterfactual rollouts that remain grounded in learned models

### Failure Signal

If prediction is only descriptive logging and does not influence behavior or inference,
Phase 4 is not complete.

## Phase 5: Language And Semantic Bridge

### Goal

Add text input and output in a way that is grounded in learned models instead of being
just an attached chatbot.

### Design Assumption

The most realistic path is hybrid.

- Monty owns grounded object, behavior, and concept models.
- A language layer maps text to those grounded structures and queries them.
- Standard neural language models can be used as teachers, parsers, or planners where
  they help.

### Deliverables

- text-to-grounded-reference mapping layer
- grounded concept naming and retrieval
- text-driven curricula or supervision interface
- benchmark for word-object, phrase-object, and text-action grounding

### Tests

- map multiple words to the same grounded concept across modalities
- answer text queries about grounded scenes or objects
- follow simple text instructions in a grounded environment

### Exit Criteria

- text is not merely proxied into unrelated embeddings
- language queries can retrieve and manipulate grounded Monty representations

### Failure Signal

If language remains only a wrapper around external models with no deep grounding,
Phase 5 is not complete.

## Phase 6: Manipulation And Skill Learning

### Goal

Make Monty act on the world, not just infer it.

### Deliverables

- manipulation environment and benchmark suite
- object-centric goal decomposition
- model-based action planning for simple tasks
- compiled model-free or partially model-free motor skills for repeated tasks

### Tests

- press or toggle a simple object at a target location
- reposition an object to match a learned goal state
- compare slow model-based action to distilled faster policy execution

### Exit Criteria

- the system can use its models to complete object-centered tasks
- repeated tasks become more efficient through skill compilation

### Failure Signal

If the system can describe what should happen but cannot reliably cause it to happen,
Phase 6 is not complete.

## Phase 7: Real Heterarchy, Routing, And Attention

### Goal

Scale beyond simple stacked hierarchies into richer large-network coordination.

### Deliverables

- routing policy for deciding which modules communicate
- associative voting and top-down biasing beyond current simple patterns
- attention mechanism for limiting active hypothesis flow
- benchmarks with many active LMs and multiple competing objects or tasks

### Tests

- compare recognition speed and accuracy with and without routing
- test whether top-down context reduces ambiguity correctly
- test multi-object and multi-task scenes with many active modules

### Exit Criteria

- larger networks gain capability rather than just cost
- routing and attention reduce compute while improving or preserving accuracy

### Failure Signal

If more modules mostly add confusion, latency, or instability, Phase 7 is not complete.

## Phase 8: Systems Scale

### Goal

Turn Monty from a delicate research stack into a scalable systems substrate.

### Deliverables

- faster search than the current kd-tree-heavy path where needed
- dynamic hypothesis space management
- better memory and checkpoint infrastructure
- parallel execution that works without manual avoidance tactics
- standardized experiment analysis and dashboards

### Tests

- benchmark throughput versus object count, LM count, and hypothesis count
- benchmark resume reliability under interruption
- benchmark distributed or parallel runs on realistic workloads

### Exit Criteria

- multi-hour and multi-stage experiments run routinely without bespoke repair work
- the main algorithmic path scales better than linearly in the most critical bottlenecks

### Failure Signal

If each larger experiment still requires manual shepherding, Phase 8 is not complete.

## Phase 9: Multi-Agent And Social Modeling

### Goal

Model other agents, not just objects.

### Deliverables

- support for multiple independent agents in the environment
- agent identity and behavior models
- task benchmarks requiring prediction of another agent's actions

### Tests

- recognize a second agent's behavior independent of exact appearance
- infer another agent's likely goal in a toy task
- coordinate or compete in a minimal multi-agent benchmark

### Exit Criteria

- the system can represent and reason about another active entity as a structured model

### Failure Signal

If other agents are still treated only as moving objects with no persistent modeled
intent or role, Phase 9 is not complete.

## Phase 10: Integrated Demonstrators

### Goal

Show that the whole stack works together on tasks that matter.

### Candidate Demonstrators

- a grounded assistant that sees a scene, answers questions, and manipulates objects
- a reading-writing agent that grounds words in objects, actions, and behaviors
- a long-horizon household-style task with perception, planning, and execution

### Tests

- end-to-end task completion under partial observability
- transfer to new combinations of known objects and behaviors
- instruction following with grounded explanations of decisions

### Exit Criteria

- integrated tasks work because the pieces cooperate, not because one helper module
  quietly solves the whole task alone

## Parallel System Tracks

The phases above are ordered, but these tracks should run throughout the program.

### Track A: Benchmarking

- keep a small decisive benchmark suite per capability
- insist on monolithic and neural baselines where appropriate
- publish failure cases, not just best runs

### Track B: Architecture Hygiene

- reduce implicit assumptions in CMP and LM internals
- keep interfaces flexible enough for non-graph LM embodiments
- avoid locking the roadmap to one temporary implementation detail

### Track C: Hybrid Integration

- identify where standard neural models clearly help
- keep those components modular and replaceable
- never let helper models erase the value of grounded internal structure

### Track D: Tooling

- automate result collection
- make checkpoint layouts and resumes predictable
- build profiling and debugging support into normal workflow

## Program-Level Go Or No-Go Gates

These gates decide whether the roadmap is still on a credible path.

### Gate 1: Composition

If Phase 1 does not show a meaningful advantage for compositional reuse, re-evaluate the
core abstraction strategy before investing heavily in later phases.

### Gate 2: Generalization

If Phase 2 does not produce real category-level transfer, Monty risks remaining an
instance-memory system and needs a representational rethink.

### Gate 3: Dynamics

If Phase 3 cannot add time and behavior cleanly, Monty will remain too static to serve
as a general world model.

### Gate 4: Action

If Phase 6 cannot convert models into reliable manipulation, Monty will not become a
full sensorimotor intelligence system.

### Gate 5: Scale

If Phase 8 cannot make the system operationally stable and scalable, the architecture
may still be scientifically interesting but not viable as a general platform.

## Near-Term Priority Stack

The recommended immediate execution order is:

1. Finish Phase 0 for the current benchmark stack.
2. Finish Phase 1 and publish a clean compositional versus monolithic result.
3. Start Phase 2 with one hard category-generalization benchmark.
4. Start Phase 3 with the simplest state and behavior prototype.
5. Only after that, begin a serious language bridge.

## Working Update Template

When updating this roadmap, use the following structure.

### Last Updated

- date:
- editor:

### Current Phase Focus

- active phase:
- active benchmarks:
- current blocker:

### Progress Since Last Update

- capability advanced:
- metric improved:
- benchmark added:
- benchmark failed:
- observability added:
- biological plausibility concern:

### Next Decision

- decision to make:
- evidence needed:
- fallback plan:

## Phase Implementation Checklist

Before starting implementation work on any new phase, verify the following.

- a dedicated phase document exists
- the roadmap links to that phase document
- the phase document contains a benchmark list and explicit tests
- the phase document contains launch, monitor, and recovery commands for any planned
  non-trivial runs
- the phase document includes a biological plausibility section
- the phase document lists whether deep learning, HTM, SDRs, or other helper methods
  are being considered and why
- the phase document names at least one required observability tool
- the run plan respects the default resource budget unless explicitly justified
- success and failure conditions are written before implementation begins

## Current Recommendation

Do not try to jump directly from current Monty to a full human-competitive system in a
single line of work.

The best path is:

- prove grounded compositional reuse
- prove category and behavior learning
- add predictive rollout and manipulation
- connect language through grounding
- then scale heterarchy, routing, and systems engineering

That sequence gives the project repeated opportunities to fail honestly, adapt, or
double down based on evidence instead of theory alone.