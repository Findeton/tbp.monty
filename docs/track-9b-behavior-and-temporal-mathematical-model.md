# Track 9b: Behavior And Temporal Mathematical Model

## Goal

Investigate from first principles the best mathematical model for a Monty
learning module that approximates what biology is doing for both:

- static shape and pose recognition
- state changes, movement, and behaviors over time

This document is intentionally upstream of implementation. Its job is to
choose the right theoretical target before more Track 9 or Track 10
engineering hardens the wrong abstraction.

## Status

Track 9b should now be read as an exploratory bridge document.

Its theory-level role has been superseded by Track 12: Cortical Time And
Predictive State, which is the tighter successor for cortical timing,
predictive state, and boundary formation.

Track 9b remains useful for:

- historical context on the shift away from primitive phase
- early surrogate implementation framing
- empirical notes from the current benchmark path

## Depends On

- Track 5 (object behaviors)
- Track 9 (modern Hopfield cortical column)
- Track 10 (biological depth within a column)

## Core Question

Can one cortical-column-like LM family model both shape and behavior using the
same local computation but different sensory inputs, or does biology require a
different mathematical model for behavior than for morphology?

More concretely:

1. Can one LM class handle static shape, object state, movement, and behavior?
2. If yes, what must be shared, and what must be factorized?
3. If no, where does the math have to split?

## Why This Track Exists

Track 9 gives Monty a strong model of local current-state inference. It is
good at answering questions of the form:

- what object or object-state does this local sensory patch currently support?
- what location is associated with this feature pattern?

It is not yet a principled model of time.

The current repo already contains useful ingredients:

- `CorticalColumnTorch` for sparse continuous attractor dynamics
- `LocationFeatureMemory` for content-addressable spatial snapshots
- `ChangeDetectingSM` for change-sensitive sensory input
- `TemporalMemory` for within-episode sequence learning
- `GlobalIntervalTimer` for duration and tempo-like signals
- `StateConditionedModel` for discrete state transitions

These components matter, but they do not yet answer the central theoretical
question. The recent two-LM Panda3D experiment made that obvious: the current
split of geometry versus appearance on one static patch stream is not a real
morphology-versus-behavior factorization, and the full run regressed badly.

That does not prove the Track 9 direction is wrong. It does show that weak
factorizations and weak tests can easily look like progress while missing the
actual biological target.

## Main Position

The earlier Track 9b framing of a controlled hierarchical semi-Markov
switching attractor system was a useful engineering surrogate, but it is too
symbolic and too clean to be the primary cortical target.

The stronger target should be:

- one LM family, not one identical memory object
- the same local cortical motif across modalities
- local recurrent cleanup for current-state inference
- distributed multi-timescale temporal context, not one global clock
- event boundaries as emergent regime-change signals, not primitive labels
- action-conditioned prediction as part of the state definition
- discrete states, hazards, tempo, and phase treated as derived summaries when
  useful, not as the first ontology

Stated sharply:

Track 9b should target a controlled hierarchical predictive attractor system
with distributed temporal traces.

Semi-Markov models, hazard heads, explicit anchors, and phase labels should be
treated as approximation layers or analysis layers, not as the primary theory
of what a cortical column is doing.

## What This Track Should Approximate

The column should primarily approximate:

- a local recurrent state that settles ambiguous current input
- a distributed temporal state that summarizes recent sensorimotor history at
  multiple scales
- action-conditioned predictions of what should happen next
- emergent event boundaries when predictive dynamics change regime
- compositional local evidence that can be integrated hierarchically

It should not primarily approximate:

- a pre-given finite state machine
- a single global episode clock
- a phase labeler
- a memoryless next-state table
- a generic sequence model with cortical vocabulary pasted on top

## First-Principles Requirements

Any serious Track 9b target should satisfy all of the following.

### R1. Partial Observability

The current sensory patch is not enough to determine object identity, object
state, or temporal regime. Hidden state is required.

### R2. Local Current-State Inference

The column must settle noisy, ambiguous input toward a stable local hypothesis.
This is the main virtue of the Track 9 Hopfield direction and should remain
central.

### R3. Predictive Sufficiency

The internal state should be defined by what it predicts about future local
sensorimotor input, not by how easily a human can assign it a symbolic label.

### R4. Distributed Time

Time should first appear as a distributed multi-timescale state, not as one
scalar timer or one universal phase coordinate.

### R5. Event-Relative Resets

The system needs a resettable temporal context tied to meaningful regime
changes. It should know "how long since the last relevant event" before it
tries to know "where am I in the whole episode?"

### R6. Action Dependence

The next observation depends on both object dynamics and the agent's action.
This is a controlled process, not a passive video model.

### R7. Local Learning

The learning rules should be at least compatible with local Hebbian,
prediction-error-gated, or predictive-coding-style mechanisms. End-to-end
sequence backprop is not the theory target.

### R8. Reusable Cortical Motif

Morphology and behavior should reuse the same local computation family when
possible. That does not require identical sensory inputs, identical latent
variables, or identical memory banks.

### R9. Derived Regimes

Discrete behavior states, anchors, hazards, tempo, and phase should be derived
from more primary predictive dynamics unless the evidence clearly forces them
to be explicit primitives.

### R10. Realistic Failure Modes

If the system cannot identify a behavior from current evidence, it should
remain uncertain, carry predictions forward, and use surprise or boundary
signals to refine the hypothesis. It should not hallucinate certainty from an
overly rigid model.

## Stronger Mathematical Target

### Normative Object

The cleanest mathematical target is a local predictive-state representation:

$$
h_t \approx \Psi\!\left(
\mathbb{E}\big[\Phi(Y_{t+1:t+H}) \mid y_{\le t}, u_{t:t+H-1}, c_t\big]
\right)
$$

where:

- `y_{\le t}` is the sensory history available to the column
- `u_{t:t+H-1}` is the future action context or motor policy context
- `c_t` is lateral and hierarchical context
- `h_t` is the local internal state

This says the column should carry whatever local state is needed to predict the
near future of the sensorimotor stream. It does not say that state must be a
symbolic behavior label or a phase index.

### Local State

The local state should be written first as:

$$
h_t = (x_t, m_t, c_t)
$$

where:

- `x_t` = fast recurrent local state used for current-state inference
- `m_t` = distributed temporal context carried as multi-timescale traces
- `c_t` = action, lateral, and hierarchical context

In this formulation, morphology and behavior are not separate first-principles
state types. They are different readouts and different dynamical regimes of the
same local predictive machinery under different sensory inputs.

If an explicit behavior summary is needed, it should be a derived object:

$$
z^{\mathrm{beh}}_t = G(x_t, m_t, c_t)
$$

Potential outputs of `G` include regime identity, anchor identity, elapsed
time, tempo, or phase, but those are downstream summaries rather than the core
state itself.

### Fast Recurrent Inference

The fast recurrent state should still be inferred by local attractor-like
settling:

$$
x_t^\star = \arg\min_x
E_{\mathrm{obs}}(x; y_t)
+ \lambda E_{\mathrm{mem}}(x; r_t)
+ \eta E_{\mathrm{ctx}}(x; c_t)
+ \rho E_{\mathrm{pred}}(x; h_{t-1}, u_{t-1})
$$

where:

- `E_obs` is sensory fit
- `E_mem` is fit to associative memory retrieval
- `E_ctx` is fit to lateral and hierarchical context
- `E_pred` is fit to the predicted continuation of recent dynamics

This keeps the strongest part of Track 9 intact: the column is still a local
recurrent inference system, not just a feed-forward classifier.

### Distributed Temporal Context

Time should first be represented as a family of stateful traces at different
timescales:

$$
m_{k,t+1} = (1 - \beta_t)\, \alpha_k m_{k,t}
+ \psi_k(x_t^\star, y_t, u_t, \epsilon_t)
$$

for `k = 1, \dots, K`, where:

- `\alpha_k \in (0, 1)` are log-spaced decay factors
- `\epsilon_t` is local surprise or prediction error
- `\beta_t` is a soft reset or boundary signal
- `\psi_k` is a local update drive

This is the primary temporal state. It is more biologically plausible than a
single global clock and more flexible than a symbolic duration variable.

If a feed-forward temporal feature is needed for readouts, define it as a
projection of the distributed temporal state:

$$
\phi_t = \Phi(m_t)
$$

This makes `\phi_t` derived from `m_t`, not an independent ontological object.

### Boundary And Reset Signal

Event boundaries should arise from instability and predictive failure, not from
an externally named state machine. A simple abstract form is:

$$
\beta_t = \sigma\!\left(
w^\top
[\epsilon_t,
\|x_t - x_{t-1}\|,
\|m_t - m_{t-1}\|,
\|\Delta u_t\|]
\right)
$$

The exact terms can change, but the logic should remain the same:

- boundary pressure rises when prediction error rises
- boundary pressure rises when the local attractor state changes abruptly
- boundary pressure rises when temporal context becomes inconsistent
- action changes can legitimately contribute to resets

### Generic Hopfield Memory

Modern Hopfield machinery should be treated as a generic associative memory,
not as a separate bank of behavior-state labels.

Use a slot-based memory:

$$
K \in \mathbb{R}^{N \times d_k}, \qquad V \in \mathbb{R}^{N \times d_v}
$$

where `N` is an engineering capacity, not the number of true semantic states.

Form a query from sensory evidence, temporal context, and action or hierarchy:

$$
q_t = Q(y_t, \phi_t, c_t, u_t)
$$

Retrieve with:

$$
\alpha_t = \operatorname{softmax}(\beta K q_t)
$$

$$
r_t = V^\top \alpha_t
$$

This does two important things:

1. It makes the Hopfield memory itself the prototype bank.
2. It allows the meaningful object to be a retrieval basin or sparse mixture,
   not a brittle one-slot-per-state mapping.

The effective number of active prototypes should emerge from usage. There is no
need to pretend the model knows the true number of cortical or behavioral
states in advance.

### Derived Regimes And Summaries

If a discrete or low-dimensional behavior summary is useful, it should be
derived from the distributed state, not built into the first principles.

#### Regime identity

$$
b_t = C(x_t, m_t)
$$

where `C` denotes a coarse-graining of the continuous predictive state into
metastable regimes or basins.

#### Anchor identity

$$
a_t = A(x_t, m_t, \beta_t, c_t)
$$

An anchor is a summary of the current regime and recent reset history. It is
not necessarily a primitive state variable.

#### Elapsed time and tempo

$$
(\tau_t, r_t) = R(x_t, m_t, c_t)
$$

Elapsed time and tempo are best viewed as derived coordinates or summaries of
the distributed temporal context, not as the first thing the column must store
explicitly.

#### Phase

$$
\varphi_t = \Phi_{\mathrm{cyclic}}(x_t, m_t)
$$

only when the current regime admits a cyclic coordinate at all.

Phase should be the last derived quantity, not the first primitive.

## Semi-Markov As Approximation, Not Target

Semi-Markov structure is still useful, but it should be treated as an analysis
layer or engineering approximation over metastable predictive regimes.

For example:

$$
s_t = C(x_t, m_t)
$$

$$
\lambda_{\mathrm{leave}}(t) = g(x_t, m_t, c_t, \epsilon_t)
$$

$$
p(s_{t+1} \mid \text{leave}) = h(x_t, m_t, c_t)
$$

This is a defensible surrogate for implementation because it gives clean event
boundaries, dwell summaries, and benchmarks. But it should not be mistaken for
the primary biological theory.

The current `SemiMarkovTransitionMemory` and
`SelfSupervisedSemiMarkovMemory` belong in this approximation layer.

## Consequences For Morphology And Behavior

### Short answer

Yes for one LM family.

No for one undifferentiated representation.

### What should be shared

The following should probably be shared across morphology and behavior LMs:

- sparse local encoding
- dendritic prediction or predictive modulation
- recurrent cleanup of local hypotheses
- surprise computation
- local Hebbian or prediction-error-gated plasticity
- LM interfaces for voting, hierarchy, and motor control

### What should differ

The following should probably differ by pathway, parameterization, or readout:

- sensory inputs
- observation likelihood terms
- temporal trace emphases and timescale distributions
- predictive readouts
- action dependence
- boundary statistics

### Best system factorization

The stronger theory still supports the same engineering conclusion as before:

- first target: one LM family with sibling morphology and behavior pathways
- parent LM integrates their evidence
- unified single-instance LM remains a later research question, not the first
  implementation target

The reason is not just engineering convenience. It is that static and
change-sensitive pathways probably carry genuinely different local evidence
even if the underlying cortical computation family is shared.

## Implications For Monty Implementation

The practical implications are different from the older Track 9b framing.

1. Temporal state should move into the main local state, not remain only as a
   sidecar memory that adds transient evidence after the fact.
2. The Hopfield memory should be treated as the primary associative dictionary
   for both static and temporal regimes, with generic slots and retrieval
   basins.
3. Boundary or reset logic should be driven by surprise, instability, and
   action context, not only by explicit transition tables.
4. Semi-Markov summaries should be retained as an approximation layer for
   analysis, debugging, and early benchmarks.
5. Tempo and phase should be derived if the data supports them, not forced into
   the ontology from the start.
6. The stronger theoretical claim is not "we have an HSMM inside a column." It
   is "the column carries distributed predictive temporal context and can be
   approximated by HSMM-like summaries when convenient."

## Evaluation Program

The current benchmark emphasis is too phase-centric for the biological goal.
Phase decoding can remain a secondary diagnostic, but it should not be the
main success criterion.

### Primary cortex-relevant tests

The stronger theory should be tested with tasks like:

1. Omission timing:
   does the model maintain temporal expectations when a predicted event is
   omitted?
2. Perturbation recovery:
   after an unexpected change, does the model recover the temporal regime or
   drift permanently?
3. Event-boundary detection:
   do resets align with meaningful changes in predictive dynamics rather than
   arbitrary frame bins?
4. Active versus passive timing:
   does timing change appropriately when the agent controls the sensory stream
   versus when it passively observes it?
5. Cross-morphology temporal transfer:
   can similar local temporal regimes transfer across different shapes?
6. Tempo scaling:
   does the model retain regime identity under compression and dilation without
   inventing unrelated states?
7. Hierarchical composition:
   can higher-level LMs compose child temporal evidence into object plus
   behavior judgments?

### Secondary diagnostic tests

These are still useful, but secondary:

- latent-to-phase decoding
- next-phase decoding
- dwell statistics under synthetic loops
- explicit semi-Markov accuracy under known state labels

Those tests tell us whether the approximation layer is working. They do not by
themselves tell us whether the theory is biologically right.

## Current Repo Status

The repo already contains a useful but narrower prototype. It should now be
understood as a surrogate implementation, not the full theory target.

### What currently exists

The current opt-in Track 9b path includes:

- `TemporalMemory` as a within-episode behavior-sequence head
- `SemiMarkovTransitionMemory` as an explicit dwell-aware transition model
- `SelfSupervisedSemiMarkovMemory` as an online latent discovery plus
  dwell-aware transition module over torch embeddings
- transient temporal evidence bias inside `CorticalColumnTorchLM`
- separate LM kwargs for morphology, behavior, and parent torch LMs

### What that prototype currently shows

- the repo can now benchmark behavior-like temporal structure on synthetic and
  real animated data
- self-supervised temporal labels can be kept separate from benchmark labels
- motion-grounded labels are materially better than equal frame buckets
- evidence arbitration inside the LM matters immediately

### What it does not yet show

- that the temporal state is integrated into the core column dynamics
- that anchor inference is biologically grounded
- that distributed multi-timescale traces are the actual mechanism in code
- that Hopfield memory is being used as a generic temporal regime dictionary
- that the benchmarks are yet aligned with cortical timing rather than mainly
  phase decodability

### Current empirical snapshot

These numbers are still useful context, but should now be interpreted as
surrogate diagnostics rather than the main theory verdict.

- synthetic ambiguous-order and tempo-invariance tasks remain at baseline `1/2`
  in the self-supervised path, with only one discovered latent state
- real `Fox.glb` foot-cycle benchmark currently discovers 12 latent states with
  matched current or next phase accuracy `0.6667/0.6`, but stretched
  performance remains weak at about `0.2558/0.2381`
- real multi-model self-supervised 1-cycle rerun currently has morphology top-1
  `0.3333` matched and stretched, while behavior current or next phase means
  are `0.6159/0.5659` matched and `0.3411/0.3519` stretched

These results establish that the pipeline is alive. They do not yet establish
that the theory of cortical time is right.

## Current Blockers

1. Temporal state is still mostly a sidecar around the column rather than part
   of the column's primary recurrent state.
2. There is no implemented distributed multi-timescale trace state tied to the
   main Torch LM dynamics.
3. There is no implemented boundary or reset mechanism grounded in predictive
   instability and action context.
4. The current Hopfield usage is not yet expressed as a generic slot-based
   temporal regime dictionary.
5. The benchmark program is still too tied to phase-decoding success.
6. There is no strong active sensorimotor timing benchmark yet.

## Open Questions

1. How much temporal state should live in dendritic prediction versus separate
   slow traces or auxiliary state variables?
2. What local signals should drive boundary or reset pressure in a biologically
   plausible approximation?
3. How should generic Hopfield memory slots be created, reused, merged, or
   forgotten over long experience?
4. When should a discrete regime readout be formed at all, and at what level of
   the hierarchy?
5. How explicit should tempo be versus remaining an implicit property of the
   distributed temporal state?
6. Is a unified single-instance LM eventually tractable, or are sibling LMs the
   stable long-term answer?
7. What experimental result would genuinely falsify the distributed predictive
   trace target rather than merely falsify a weak benchmark proxy?

## Success Criteria

Track 9b is successful if it produces:

1. A falsifiable mathematical target centered on distributed predictive temporal
   state rather than only symbolic sequence structure.
2. A clear recommendation on how one LM family can cover both morphology and
   behavior without forcing one undifferentiated representation.
3. Benchmarks that test predictive timing, boundary resets, tempo scaling, and
   cross-morphology transfer, not only phase decoding.
4. A concrete implementation path that keeps Monty's cortical-column premise
   intact.

## Failure Criteria

Track 9b should be revised if:

1. The model only works when explicit phase-like supervision or externally named
   states do most of the conceptual work.
2. The strongest implementation requires replacing local learning with a generic
   sequence model that no longer resembles a cortical approximation.
3. Better cortex-relevant tests show that the system is tracking benchmark
   labels without maintaining a useful predictive temporal state.

## Decision Log

- 2026-04-01: Created Track 9b to investigate the behavior and temporal math
  from scratch rather than continue extending Track 9 under implicit
  assumptions.
- 2026-04-01: Early Track 9b framing preferred a controlled hierarchical
  semi-Markov switching attractor system.
- 2026-04-01: Early engineering preference remained one LM family with separate
  morphology and behavior pathways.
- 2026-04-02: The theory shifted toward event-relative time, separate tempo,
  and resettable temporal context instead of a primitive global phase.
- 2026-04-03: The primary target was rewritten more strongly as a controlled
  hierarchical predictive attractor system with distributed temporal traces.
- 2026-04-03: Semi-Markov states, hazard heads, and explicit anchors were
  demoted from primary theory to approximation and analysis layers.

## Next Decision

- decision to make: whether the next implementation step should integrate
  distributed temporal traces and boundary pressure directly into the Torch LM,
  or continue using the current sidecar temporal surrogate as a temporary
  scaffold
- evidence needed: a minimal integrated prototype plus tests for omission
  timing, boundary detection, and tempo scaling under action
- fallback plan: keep the current semi-Markov sidecar as an explicit surrogate
  layer, but stop treating it as the biological target