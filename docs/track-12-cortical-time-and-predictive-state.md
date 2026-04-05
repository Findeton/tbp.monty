# Track 12: Cortical Time And Predictive State

## Goal

Tighten the temporal theory that began in Track 9b into a form that is both:

- biologically defensible relative to Tracks 9 to 11
- narrow enough to guide implementation without theory drift

The goal is not another benchmark note. The goal is to identify the right
mathematical object for local cortical timing, prediction, and regime change in
Monty's column framework.

## Depends On

- Track 5 (object behaviors)
- Track 9 (modern Hopfield cortical column)
- Track 10 (intra-column biological depth)
- Track 11 (deep hierarchy and GPU scaling)
- Track 9b (behavior and temporal mathematical model) as the exploratory
  predecessor that this track tightens and supersedes at the theory level

## Problem

Track 9b made the right move away from primitive global phase and toward
event-relative temporal context.

But the earlier Track 12 rewrite was still too loose in one critical way: it
let three different roles blur together:

- online temporal state used for inference now
- predictive and gating signals derived from that state
- slower plasticity traces used for learning

That blur makes the theory hard to falsify. If too many variables can each be
said to "carry time", the implementation can silently choose the real theory
later.

## Main Claim

The best current target is a predictive attractor system with one primary
online temporal carrier.

Stated sharply:

The local cortical target is a recurrent inference state plus a distributed
multi-timescale trace state. Predictive gating, boundary pressure, tempo,
phase, and symbolic summaries are derived from that state. Plasticity traces
are separate.

This is narrower than the earlier Track 12 draft, and that narrowing is not a
stylistic choice. It is forced by the biology in Track 10:

- Hopfield recurrence in L2/3 must remain symmetric
- directed temporal learning should live on predictive dendritic pathways, not
  in recurrent attractor weights
- thalamocortical gating modulates input, but is not itself the primary memory
  of elapsed local time
- eligibility traces support plasticity over longer windows, but should not be
  treated as the main online inference state

## What This Track Adds Beyond 9b

Track 12 now makes four commitments relative to Track 9b.

1. It separates normative target, mechanistic hypothesis, and engineering
   surrogate.
2. It narrows the mechanistic story to one primary online temporal state.
3. It separates online temporal inference from plasticity traces.
4. It focuses evaluation on tests that directly falsify predictive timing,
   rather than on secondary phase-decoding diagnostics.

## Level 1: Normative Target

### Predictive-State Principle

The primary state of a cortical column should be defined by what it predicts
about the near future of its local sensorimotor stream.

The cleanest mathematical target is a local predictive-state representation:

$$
h_t^* = \Psi\!\left(
\mathbb{E}\big[\Phi(Y_{t+1:t+H}) \mid y_{\le t}, u_{t:t+H-1}, c_t\big]
\right)
$$

where:

- `y_{\le t}` is the local sensory history
- `u_{t:t+H-1}` is relevant action or policy context
- `c_t` is lateral and hierarchical context
- `H` is a finite but behaviorally meaningful horizon
- `h_t^*` is the ideal local predictive state

This says the column should carry whatever state is sufficient for predicting
future local sensorimotor consequences. It does not say that the primary state
must be a symbolic behavior label, anchor label, timer, or phase coordinate.

### What The Normative Target Must Explain

Any acceptable approximation should explain how a local column can:

- settle ambiguous current input
- preserve recent temporal context across several timescales
- anticipate near-future changes under action
- detect regime breaks when prediction fails
- support compositional hierarchical integration

### What The Normative Target Should Not Collapse Into

The primary target should not collapse into:

- a finite-state machine
- a single global timer
- a phase-label decoder
- a memoryless next-step predictor
- a generic sequence model wrapped in cortical vocabulary

## Level 2: Mechanistic Hypothesis

This level answers a different question:

How could a cortical column approximately implement the predictive-state target
using the ingredients already implied by Tracks 9 to 11?

### What Must Be Separated

The narrowing pass starts by separating three different objects.

1. Online inference state: what the column is using now to interpret the
   current regime.
2. Derived predictive and gating signals: what the column expects next and how
   strongly it should keep or reset the current regime.
3. Plasticity traces: slower variables that support learning, especially on
   dendritic predictive pathways.

Those objects may interact, but they should not be treated as the same state.

### Core Online State

The proposed primary online mechanistic state is:

$$
h_t = (x_t, d_t)
$$

where:

- `x_t` = fast recurrent local attractor state, mainly the settled L2/3
  representation
- `d_t` = distributed multi-timescale temporal trace state used for online
  inference

Action and hierarchical context still matter, but for the first mechanistic
pass they should be treated as exogenous drives or modulators, not as separate
primary temporal state variables.

### What d_t Is

`d_t` is the column's compressed online memory of how the current local regime
has been unfolding over the recent past.

It is best thought of as a bank of traces:

$$
d_t = \big(d_t^{(1)}, d_t^{(2)}, \dots, d_t^{(K)}\big)
$$

with fast and slow components that remember different temporal horizons.

`d_t` is not:

- a global clock
- a symbolic behavior state
- a phase variable
- a next-state table
- an eligibility trace for delayed learning

`d_t` is:

- an online temporal context for local inference
- a summary of recent regime progression
- the main source of countdown-like expectation and tempo adaptation

### Biological Allocation Of Roles

The mechanistic interpretation should align with Track 10.

- `L4` carries current feedforward drive.
- `L2/3` carries the fast settled state `x_t` and its recurrent attractor
  cleanup.
- basal dendrites provide short-lag sequential and contextual bias that helps
  maintain and read out `d_t`.
- apical dendrites in `L1` carry top-down and policy context that modulates
  what continuation is plausible.
- `L6a` plus thalamic relay implement predicted-input gating, but do not serve
  as the primary memory store for elapsed local time.
- `L5a` and `L5b` are the most plausible broadcast path for regime-change
  signals when mismatch and state discontinuity rise.
- slower eligibility traces remain on predictive dendritic pathways and should
  be treated as plasticity variables, not as the primary online state.

### Minimal Causal Loop

The mechanistic story should now be written as one explicit causal loop.

#### Step 1: Predict expected input

$$
\hat{i}_t = P(x_{t-1}, d_{t-1}, c_t, u_{t-1})
$$

This is the thalamocortical or dendritically informed expectation of the next
feedforward drive.

#### Step 2: Compute mismatch

$$
\epsilon_t = \mathcal{D}(i_t, \hat{i}_t)
$$

This mismatch is a live signal, not merely an offline loss.

#### Step 3: Form a trace-biased query

$$
\chi_t = E(i_t) + U_d d_{t-1} + U_c c_t
$$

The temporal state biases which stored regime should be retrieved, but does not
replace current sensory evidence.

#### Step 4: Retrieve associative prior

$$
\alpha_t = \operatorname{softmax}(\gamma K \chi_t)
$$

$$
m_t = V^\top \alpha_t
$$

Modern Hopfield memory is the generic regime dictionary. The meaningful object
is typically the retrieval mixture `\alpha_t` or embedding `m_t`, not a single
slot index.

#### Step 5: Settle the fast recurrent state

$$
x_t^\star = \arg\min_x
E_{\mathrm{obs}}(x; i_t)
+ \lambda E_{\mathrm{mem}}(x; m_t)
+ \eta E_{\mathrm{ctx}}(x; c_t)
$$

This preserves the strongest part of Track 9: local recurrent cleanup remains
the core of current-state inference.

#### Step 6: Compute boundary pressure

For the first mechanistic pass, boundary pressure should be minimal and not
overstuffed:

$$
\beta_t = \sigma\!\left(
w_\epsilon \epsilon_t
+ w_x \|x_t^\star - x_{t-1}^\star\|
- \theta_\beta
\right)
$$

Mismatch and attractor discontinuity are enough for the first serious test. If
that signal is not useful, more elaborate reset terms will likely just hide a
wrong mechanism.

#### Step 7: Update the online trace state

$$
d_t = (1 - \beta_t)\, \Lambda d_{t-1} + B x_t^\star + C u_t
$$

where `\Lambda` is a bank of log-spaced decays.

This update says:

- keep recent regime momentum when boundary pressure is low
- partly reset that momentum when boundary pressure is high
- drive the traces from the settled local state and relevant action context

### What The Narrowing Removes

This narrowing is deliberately restrictive.

- `p_t` is no longer a primary state variable; prediction is derived.
- `m_t` is retrieval output, not core state.
- `\beta_t` is derived boundary pressure, not primitive symbolic transition.
- `\epsilon_t` is not part of the online trace content in the first pass.
- eligibility traces are not folded into `d_t`.

This is necessary for falsifiability. Otherwise too many variables can each be
said to "carry time" and the theory becomes underdetermined.

### Why This Narrowing Is Biologically Forced

The biology in Track 10 forces this narrowing.

- Hopfield recurrence in L2/3 must remain symmetric, so directed temporal
  asymmetry should not be written into recurrent attractor weights.
- plateau-like and dendritic states can bias the retrieval query, which makes
  them plausible contributors to `d_t`.
- thalamocortical loops gate what reaches `L4`, but they are not the primary
  local time store.
- eligibility traces solve delayed plasticity, not immediate inference.

### What This Form Can Explain

This narrowed form should explain:

- countdown-like local expectation for imminent events
- omission timing through mismatch when expected events do not occur
- moderate tempo adaptation through multi-timescale traces
- event boundaries as predictive instability rather than symbolic labels

It should not yet be expected to explain:

- exact long-duration symbolic counting
- arbitrarily large tempo invariance
- stable phase variables without extra cyclic structure
- full hierarchy-level temporal composition in the first implementation pass

### Derived Summaries

Discrete state, anchor, elapsed time, tempo, and phase remain derived summaries
over the predictive state.

#### Regime identity

$$
b_t = C(x_t, d_t, \alpha_t)
$$

#### Anchor identity

$$
a_t = A(b_t, \beta_t, c_t)
$$

#### Elapsed time and tempo

$$
(\tau_t, \nu_t) = R(x_t, d_t, c_t)
$$

`\tau_t` and `\nu_t` should be read as summaries of predictive progression,
not as primary stored variables.

#### Phase

Only if the regime is genuinely cyclic and admits a stable low-dimensional
progress coordinate should phase be introduced:

$$
\varphi_t = \Phi_{\mathrm{cyclic}}(x_t, d_t)
$$

Phase is a late diagnostic, not the first ontology.

### Separate Plasticity Variables

Eligibility traces should remain explicit plasticity variables on predictive
dendritic pathways.

They may be necessary later for learning what transitions or temporal biases to
store, but they should not be folded into the primary online temporal state.

## Level 3: Engineering Surrogate

This level answers a different question:

What approximate model should be built now while the stronger mechanism is
still incomplete?

### Semi-Markov Summary Layer

Semi-Markov structure remains useful, but only as a surrogate over the more
primary predictive state.

Define a coarse summary:

$$
s_t = C(x_t, d_t, \alpha_t)
$$

Then use leave pressure and next-summary prediction as readouts:

$$
\lambda_{\mathrm{leave}}(t) = g(d_t, \epsilon_t, \beta_t)
$$

$$
p(s_{t+1} \mid \text{leave}) = h(x_t, d_t, c_t)
$$

This remains useful for:

- event-boundary summaries
- dwell diagnostics
- tractable debugging
- benchmarkable temporal reports

But it is not the biological target.

### Where Current Repo Code Fits

Under this cleaner separation:

- `SelfSupervisedSemiMarkovMemory` is a surrogate summary learner
- `SemiMarkovTransitionMemory` is a surrogate explicit transition summary
- transient temporal evidence bias inside `CorticalColumnTorchLM` is an early
  sidecar approximation

Those are acceptable scaffolds. They should not be mistaken for the target
mechanism.

## Falsifiers

This theory should now be easier to falsify than both Track 9b and the earlier
Track 12 draft were.

### F1. Omission Countdown Failure

If the model does not show sharpening event-relative expectation and elevated
mismatch when an expected local event is omitted, then `d_t` is probably not
functioning as predictive temporal context.

### F2. Tempo Adaptation Failure

If moderate dilation or compression of the same regime forces unrelated latent
states or destroys regime continuity, then the trace basis is too weak or too
mis-specified.

### F3. Boundary Pressure Failure

If mismatch plus attractor discontinuity do not produce cleaner boundary
detection and recovery than surrogate label transitions, then the reset story
is probably wrong.

### F4. Explicit Phase Dependence

If the strongest implementation still needs explicit phase supervision or
hand-authored temporal labels to work, then the theory has failed its intended
target.

### F5. Active Versus Passive Equivalence

If active control of the same local regime produces essentially the same
temporal state as passive replay, then the action-conditioned predictive claim
is too weak.

## Consequences For Morphology And Behavior

### Short answer

Yes for one LM family.

No for one undifferentiated representation.

### Shared computation

The following should remain shared across morphology and behavior pathways:

- sparse local encoding
- recurrent cleanup of ambiguous local hypotheses
- associative memory retrieval
- dendritic predictive modulation
- local surprise computation
- local Hebbian or prediction-error-gated plasticity
- hierarchical and lateral interfaces

### Factorized inputs and readouts

The following should differ by pathway or readout:

- sensory streams
- observation likelihoods
- trace emphases and timescales
- readouts of local predictive state
- action dependence and boundary statistics

### Best current system factorization

The best first engineering target remains:

- sibling morphology and behavior LMs of the same family
- parent LM integrates their evidence

Unified single-instance LM remains a later research problem.

## Implementation Consequences

Track 12 now implies a more specific implementation roadmap.

1. The first engineering carrier of time should be the core trace bank `d_t`.
2. Predicted input, mismatch, and boundary pressure should be derived each step
   from `x_t`, `d_t`, and context.
3. Hopfield retrieval should receive a trace-biased query rather than only a
   post-hoc temporal evidence adjustment.
4. Plasticity traces should remain distinct from online inference state.
5. Semi-Markov modules should remain summary heads or diagnostics only.
6. Tempo and phase should remain derived diagnostics unless the data forces a
   stronger ontology later.

## Evaluation Program

### Asset Constraint

For this track, evaluation should not use spheres, boxes, bars, or other
programmatically generated stand-in geometries as the primary test bed.

Stage 1 and later temporal tests should use:

- real animated 3D models
- or real audio data once an audio timing path exists

Synthetic schedules, omissions, and tempo manipulations are still allowed, but
they should be applied to real asset streams rather than to synthetic geometry
fixtures.

### Learning And Control Constraint

For this track, the temporal part of the evaluation should use self-supervised
or unsupervised temporal learning rather than supervised phase or state labels
during training.

That means:

- no phase `state_provider` labels in training for the behavior LM
- no hand-authored temporal-state supervision as the main training signal
- phase decoding, when used, should be fit post hoc as a diagnostic only

For visual timing tests, sensing should remain closed-loop and motor should be
controlled by the LM through Monty's own policy, rather than by a fixed scripted
camera path as the primary evaluation mode.

### What Counts As A Sound First Test Program

The best first tests are not the broadest tests. They are the tests that most
directly interrogate the narrowed mechanism.

That means:

- omission timing, tempo perturbation, shared-prefix disambiguation, and
  boundary recovery are sound first tests
- cross-morphology transfer and hierarchical composition are important, but are
  too confounded to use as the first gate on this theory

The reason is simple. The first set tests whether `d_t` is doing real local
predictive work. The second set mixes temporal theory with broader questions
about morphology generalization, hierarchy quality, and evidence aggregation.

### Stage 1: Decisive Local Tests

These should be the first gating tests for implementation.

#### T1. Shared-prefix branch disambiguation

Train two local regimes on real asset streams that share the same early
observations but diverge later. In practice this can be implemented by splicing
real animation segments from the same or related real assets so that the early
lead-in is shared and the continuation differs.

After the shared prefix, the model should maintain distinct predictive
continuations.

Why it is sound:

- it directly tests whether `d_t` is carrying informative recent history
- it can be implemented with real-asset schedule control rather than synthetic
  geometry fixtures

What should be measured:

- next-regime prediction margin after the shared prefix
- whether the model collapses both branches into the same latent summary

#### T2. Omission countdown test

Train a regime with a predictable local event. At evaluation, remove that event
after the same lead-in.

Why it is sound:

- it directly tests event-relative expectation rather than label decoding
- it is the cleanest behavioral consequence of a useful `d_t`

What should be measured:

- rising mismatch around the expected omission point
- whether regime identity is briefly destabilized and then recovered
- whether the system distinguishes omission from a normal within-regime frame

#### T3. Tempo dilation and compression test

Train at one tempo and evaluate at moderate stretch or compression.

Why it is sound:

- it directly tests whether the trace basis supports tempo adaptation
- the repo already has nearby synthetic and Panda3D schedule scaffolding for
  this class of test

What should be measured:

- regime identity stability under `1.5x` to `2x` dilation or compression
- boundary placement stability
- recovery of the same regime rather than invention of unrelated states

#### T4. Boundary perturbation recovery test

Inject a transient off-regime perturbation or noisy interruption into a learned
regime.

Why it is sound:

- it tests whether boundary pressure is actually useful rather than decorative
- it separates a healthy reset-and-recovery mechanism from brittle label
  tracking

What should be measured:

- mismatch spike at perturbation
- temporary regime destabilization
- re-entry into the original regime after the perturbation ends

### Stage 2: Stronger But More Confounded Tests

These are important, but they should not be the first theory gate.

#### T5. Active versus passive timing

The same nominal observation stream should produce different temporal state if
the agent actively controls it versus passively replays it.

Why it matters:

- it tests action-conditioned prediction directly

Why it is not a first gate:

- it depends heavily on policy and sensing details, not only on temporal state

#### T6. Cross-morphology temporal transfer

Similar local regimes across different shapes should partially transfer.

Why it matters:

- it tests whether the local temporal representation is more generic than a
  model-specific label table

Why it is not a first gate:

- current multi-model behavior and morphology results are still too unstable to
  make this a clean first falsifier

### Stage 3: Later System Tests

These should be treated as later-stage system tests.

#### T7. Hierarchical composition of child temporal evidence

This matters eventually, but it is too much of a departure to use as the first
validation of Track 12. It conflates temporal theory with deep hierarchy
quality.

#### T8. Phase diagnostics

Latent-to-phase and next-phase decoding remain useful diagnostics.
They should not be treated as the primary verdict on the theory.

### Practical Recommendation For The Repo

The most practical near-term evaluation plan is:

1. keep temporal training self-supervised and do not use temporal `state_provider`
  labels
2. keep sensing closed-loop, with motor controlled by the LM through Monty's
  own policy
3. move the decisive Stage 1 falsifiers onto real animated Panda3D assets
4. use frame-schedule control on those real assets for moderate stretch,
   compression, omission, and perturbation tests
5. keep phase decoding only as a secondary post hoc report

That is not too much of a departure. It is the right amount of departure. It
stays close to existing repo machinery while changing the target of evaluation
from label recovery to predictive temporal competence.

## Relationship To Track 9b

Track 9b should now be read as an exploratory bridge document.

This Track 12 note is the tighter successor for the temporal theory.

- Track 9b contains the exploratory move away from primitive phase.
- Track 12 adds the narrower causal backbone.
- Track 9b implementation results remain useful context, but they are not the
  main theory verdict anymore.

## Success Criteria

Track 12 is successful if it produces:

1. A sharper and more falsifiable cortical-time theory than Track 9b.
2. A mechanistic story that preserves Track 9 Hopfield invariants and Track 10
   biological constraints.
3. A clean separation between online temporal inference and plasticity traces.
4. A concrete path toward implementation and testing without collapsing into
   generic sequence-model engineering.

## Failure Criteria

Track 12 should be revised if:

1. the narrowed trace state cannot support omission timing or tempo adaptation
2. the theory only succeeds when semi-Markov summaries do the real work
3. the best implementation still depends primarily on explicit phase labels
4. broader benchmark wins appear without improvement on the decisive local
   predictive tests

## First Decision

- recommended first decision: implement `d_t` as the one primary online
  temporal carrier inside the Torch LM core
- defer: full laminar side-state detail, explicit oscillatory phase structure,
  and broader hierarchical timing claims
- evidence needed: improvement on Stage 1 local tests, especially omission
  timing and tempo perturbation, without relying on explicit phase supervision
- fallback plan: keep the semi-Markov sidecar for diagnostics, but do not let
  it define the theory