# Track 8: CorticalColumn Integration with MontyBase

## Goal

Make CorticalColumn a drop-in replacement for EvidenceGraphLM inside the Monty
framework. After this track, CorticalColumn should be usable in any experiment
config that currently uses EvidenceGraphLM, participate in multi-LM voting,
support behavior recognition, and have integration tests proving it works
end-to-end through the standard MontyBase pipeline.

## Current State

CorticalColumn is a standalone class with no inheritance from `LearningModule`.
It has its own `step(state)` / `pre_episode()` / `post_episode()` API that
doesn't match MontyBase's calling conventions. It runs only in the isolated
Panda3D eval harness.

EvidenceGraphLM inherits from `GraphLM → LearningModule` and implements 12
abstract methods that MontyBase depends on to orchestrate the step loop, voting,
hierarchy, motor policies, and experiment lifecycle.

## Feature Parity Checklist

| # | Feature | EvidenceGraphLM | CorticalColumn now | Target |
|---|---------|-----------------|-------------------|--------|
| 1 | LearningModule interface | inherits | standalone class | inherits |
| 2 | matching_step / exploratory_step | implemented | only step(state) | implemented |
| 3 | Evidence accumulation | continuous per-hypothesis | weight readout per-object | weight readout (sufficient) |
| 4 | Voting (send_out_vote / receive_votes) | State objects with evidence | none | implemented |
| 5 | Hierarchy (get_output → parent LM) | State with MLH | none | implemented |
| 6 | State-conditioned models (behaviors) | StateConditionedModel | none | implemented |
| 7 | Change detection (ChangeDetectingSM) | wired via config | none | wired |
| 8 | Motor policies (propose_goal_states) | GSG integration | motor_prediction standalone | GSG or equivalent |
| 9 | Top-down context (receive_context) | dispatch wired | method exists, unused | tested in hierarchy |
| 10 | Surprise-gated output | T6.6 implemented | none | implemented |
| 11 | Burst sampling | BurstSamplingHypothesesUpdater | none | not needed* |
| 12 | Experiment configs | 15+ YAML configs | 0 | at least 1 |
| 13 | Integration tests (full pipeline) | 19 | 0 | at least 5 |
| 14 | Autonomous novelty detection | manual train/eval split | detects but can't act | self-supervised |

*Burst sampling is an EvidenceGraphLM-specific mechanism for managing hypothesis
space. CorticalColumn's equivalent is attractor settling + dendritic prediction
— it achieves the same goal (efficient search) through biology rather than
bookkeeping.

---

## Implementation Plan

### Step 1: LearningModule Interface Adapter

**What**: Make CorticalColumn inherit from `LearningModule` without breaking
any existing code (standalone usage, Panda3D harness, 185 existing tests).

**How**:
- Create `CorticalColumnLM` in a new file
  `src/tbp/monty/frameworks/models/cortical_column/learning_module.py`
- `CorticalColumnLM(LearningModule)` wraps `CorticalColumn` via composition
  (not inheritance — CorticalColumn stays clean)
- Implement all 12 abstract methods by delegating to the inner column
- `matching_step(ctx, observations)` extracts State from observations list,
  calls `column.step(state)`
- `exploratory_step(ctx, observations)` does the same but in train mode
- `set_experiment_mode(mode)` maps ExperimentMode to train/eval
- `state_dict()` / `load_state_dict()` serialize column state
- Stub implementations for voting/output/goal_states (return None/empty)

**Tests**:
- CorticalColumnLM instantiates and satisfies LearningModule ABC
- matching_step with fake observations produces evidence
- exploratory_step learns an object
- Existing 185 tests still pass unchanged

**Parity after this step**: Interface ✓, everything else still stubbed.

---

### Step 2: Observations Bridge

**What**: CorticalColumnLM needs to consume the same observation format that
MontyBase provides (list of State objects from SMs), not the single State that
CorticalColumn.step() expects.

**How**:
- `matching_step(ctx, observations)` receives `list[State]` (one per connected SM)
- Pick the first usable State (where `use_state=True`)
- Pass it to `column.step(state)` and store the result
- Handle edge case: no usable states → skip step (like EvidenceGraphLM does)
- Handle LM-to-LM inputs: if an input has `sender_type="LM"`, treat it as
  top-down context (call `receive_context`)

**Tests**:
- Feed CorticalColumnLM a list of States matching CameraSM output format
- Feed it mixed SM + LM inputs
- Feed it empty/no-use-state inputs → no crash

**Parity after this step**: Can receive standard Monty observations.

---

### Step 3: Voting

**What**: Implement `send_out_vote()` and `receive_votes()` so CorticalColumnLM
can participate in multi-LM lateral communication.

**How**:
- `send_out_vote()` returns the same format as EvidenceGraphLM:
  ```python
  {
      "possible_states": {
          object_id: [State(location=..., confidence=scaled_evidence, ...)]
          for object_id in top_hypotheses
      },
      "sensed_pose_rel_body": current_pose
  }
  ```
  Evidence scores scaled to [-1, 1] for compatibility.
- `receive_votes(votes)` processes incoming votes:
  - For each voted object, add/subtract evidence based on vote confidence
  - This is simpler than EvidenceGraphLM's KDTree spatial matching because
    CorticalColumn accumulates evidence per-object (not per-hypothesis-pose)
  - Good enough for lateral agreement; spatial pose voting can come later

**Tests**:
- Two CorticalColumnLMs vote and agree on same object
- Voting shifts evidence toward consensus
- send_out_vote returns None before first step
- receive_votes with empty votes doesn't crash

**Parity after this step**: Multi-LM voting works.

---

### Step 4: Hierarchy Output

**What**: Implement `get_output()` so CorticalColumnLM can feed its recognition
result to a parent LM in a hierarchy.

**How**:
- `get_output()` returns a `State` with:
  - `location`: last observed location from input State
  - `morphological_features`: pose from input State
  - `non_morphological_features`: `{"object_id": mlh_id, "evidence": mlh_ev}`
  - `confidence`: normalized evidence for top hypothesis
  - `use_state`: True if evidence above threshold, False otherwise
  - `sender_id`: learning_module_id
  - `sender_type`: "LM"
- When surprise is low (prediction matches), set `use_state=False` (surprise-gated)

**Tests**:
- get_output returns valid State after training + eval step
- get_output returns use_state=False when prediction is accurate (low surprise)
- Parent CorticalColumnLM can receive child's output as input
- Two-level hierarchy: child feeds parent, parent accumulates evidence

**Parity after this step**: Hierarchy ✓, surprise-gated output ✓.

---

### Step 5: Goal States & Motor Integration

**What**: Implement `propose_goal_states()` so CorticalColumnLM can drive
motor policies for active sensing.

**How**:
- Option A (minimal): Return empty list — motor system uses default policy
- Option B (full): Create a lightweight GSG that proposes goal states based on:
  - If evidence is low → propose exploratory goals (move to new viewpoint)
  - If evidence is high → propose confirmatory goals (revisit key features)
  - Motor prediction error → move toward locations with high prediction error
    (information-seeking)

Start with Option A, implement Option B as a follow-up.

**Tests**:
- propose_goal_states returns list (possibly empty)
- With Option B: motor prediction error drives exploration direction

**Parity after this step**: Motor integration ✓ (minimal).

---

### Step 6: State-Conditioned Behavior Support

**What**: Enable CorticalColumn to recognize objects in different behavioral
states (e.g., stapler-open vs stapler-closed).

**How**:
- CorticalColumn's basal dendrites already learn temporal sequences within
  an episode. The gap is discrete state labeling.
- Add `state_id` tracking: during training, the harness provides state labels
  alongside object labels
- Associative memory learns `(cell_pattern, state_id)` → object label, where
  state_id modifies the label SDR (e.g., hash of "stapler:open")
- During eval, evidence accumulates per (object, state) pair
- The dendritic predictions naturally separate states: different behavioral
  sequences activate different dendritic contexts → different cell patterns →
  different associative readouts

**Tests**:
- Train on object with 2 states (different feature sequences)
- Eval correctly identifies which state is active
- Evidence separates (object, state1) from (object, state2)

**Parity after this step**: State-conditioned ✓.

---

### Step 7: ChangeDetectingSM Compatibility

**What**: Wire CorticalColumn to work with ChangeDetectingSM for behavior
recognition (motion/flow features instead of static features).

**How**:
- ChangeDetectingSM produces States with `flow_direction`, `flow_magnitude`
  in non_morphological_features
- CorticalColumn's encoder needs to handle these features:
  - Add flow_direction (3D unit vector) and flow_magnitude (scalar) to
    FeatureSDREncoder's feature list
  - Encode them the same way as HSV/curvatures: ScalarEncoder → sparse bits
- The rest of the column pipeline (SP, dendrites, attractor, memory) works
  unchanged — it doesn't care what the features represent

**Tests**:
- FeatureSDREncoder encodes flow features correctly
- CorticalColumnLM trains and evals on synthetic flow States
- Two-column setup: column0 (CameraSM, shape) + column1 (ChangeDetectingSM,
  behavior), both voting → combined recognition

**Parity after this step**: Behavior recognition ✓.

---

### Step 8: Top-Down Context in Hierarchy

**What**: Wire and test `receive_context()` in a real multi-level hierarchy
where a parent column's recognition biases a child column's predictions.

**How**:
- `receive_context()` already exists on CorticalColumn — it sets the apical
  predicted mask
- CorticalColumnLM.receive_context() delegates to column.receive_context()
- MontyBase's `_dispatch_context_signals()` calls get_context_signal() on
  parent and receive_context() on children
- Test that top-down context actually improves recognition:
  - Train child on parts (wheel, door), parent on wholes (car, truck)
  - At eval time, parent recognizes "car" → sends context to child
  - Child should bias toward "wheel" over "door" given car context

**Tests**:
- 2-level hierarchy: parent receives child output, sends context back
- Context biases child's cell selection (apical-predicted cells win)
- Recognition accuracy improves with context vs without

**Parity after this step**: Top-down context ✓.

---

### Step 9: Autonomous Novelty Detection

**What**: Enable CorticalColumn to detect novel objects during eval and
create new object representations without being told the object name.

**How**:
- Already has surprise signal (high burst ratio = novel input)
- Add novelty detection logic to `matching_step`:
  - If surprise stays above `novelty_threshold` for N consecutive steps AND
    evidence for all known objects is below a threshold → declare "new object"
  - Auto-generate a temporary label (e.g., "unknown_001")
  - Switch to learning mode for this object (continuous_plasticity already
    supports learning during eval)
  - After episode, the new label persists in associative memory
- This is the "hippocampal novelty signal" that triggers cortical learning

**Tests**:
- Train on objects A, B. Present novel object C during eval.
- Column detects novelty (high surprise, no convergence)
- With auto_novelty=True, column creates new representation
- Re-presenting C → recognized (lower surprise)

**Parity after this step**: Self-supervised novelty ✓.

---

### Step 10: Experiment Config & Integration Tests

**What**: Create YAML configs and integration tests that run CorticalColumnLM
through the full MontyBase pipeline, matching the integration test coverage
of EvidenceGraphLM.

**How**:
- Create `default_cortical_column.yaml` config file
- Create at least 5 integration tests:
  1. Single CorticalColumnLM, train + eval on 2 objects (basic pipeline)
  2. Two CorticalColumnLMs with voting (lateral communication)
  3. Two-level hierarchy: child → parent (hierarchical recognition)
  4. Shape + behavior: CameraSM + ChangeDetectingSM → 2 columns (multi-modal)
  5. Novelty detection: train on known, present unknown

**Tests**: The 5 integration tests above, running through MontyBase with
real (or realistic synthetic) sensor data.

**Parity after this step**: Full feature parity with EvidenceGraphLM.

---

## Execution Order & Dependencies

```
Step 1 (interface)
  └→ Step 2 (observations bridge)
       ├→ Step 3 (voting)
       ├→ Step 4 (hierarchy + surprise-gated output)
       └→ Step 5 (goal states)
            └→ Step 10 (configs + integration tests) [after all above]

Step 6 (state-conditioned) — can run after Step 2
Step 7 (ChangeDetectingSM) — can run after Step 2
Step 8 (top-down context) — needs Step 4
Step 9 (novelty detection) — needs Step 2
```

Steps 1-2 are blocking. Steps 3-5 and 6-9 can be interleaved. Step 10 is
the capstone.

## Principles

1. **Composition over inheritance**: CorticalColumnLM wraps CorticalColumn,
   doesn't modify it. All 185 existing tests keep passing.
2. **Same test after each step**: Run the full test suite after every change.
3. **Real sensor data**: Integration tests use YCB meshes via Panda3D or
   the standard Monty test fixtures — no synthetic toy data.
4. **No new abstractions**: Use existing Monty interfaces (State, GoalState,
   ExperimentMode). Don't create CorticalColumn-specific protocols.
5. **Feature flags**: New capabilities default off. Existing behavior unchanged.
