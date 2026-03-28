# Track 5: Object Behavior Modeling

## Purpose

Extend Monty to learn and recognize dynamic object behaviors using the same
LM algorithm as static object recognition. Following the TBP theory
([object-behaviors.md](theory/object-behaviors.md)), behaviors are modeled in
**separate cortical columns** with their own reference frames, making them
compositional and object-independent. A "hinge" behavior learned on a stapler
transfers to a laptop lid, a door, or any new hinge object.

## Problem Statement

1. **Monty only models static morphology.** Objects are stored as spatial
   graphs of features at locations. If the object changes (a fox walks, a
   stapler opens), the LM sees unexpected features and its evidence drops.
2. **Dynamic changes are noise, not signal.** The pattern of change carries
   information (walking vs running), but the LM can't use it.
3. **No concept of behavioral state.** An open stapler and a closed stapler
   are the same object, but the LM can't distinguish or predict transitions.
4. **No timing.** Behaviors have temporal structure (the interval between a
   stapler pressing and rebounding matters), but there's no clock input.

## Design Principles

1. **Same algorithm, different input.** The LM is unchanged. A behavior LM
   receives from a change-detecting SM instead of a static-feature SM. This
   follows the parvocellular (static) vs magnocellular (change) pathway
   analogy from the theory.
2. **Behaviors are object-independent.** A behavior model has its own
   reference frame. The same "leg swing" behavior applies to a fox, a human,
   or a robot. Associations between behavior and morphology are learned via
   hierarchy (a higher-level LM), not embedded in the behavior model.
3. **State is a hypothesis dimension.** Hypothesis = (location, rotation,
   state). Evidence accumulates per-state. State is treated as a sub-object-ID
   in voting. This is a multiplicative extension of the existing hypothesis
   space, not a new algorithm.
4. **Backward compatible.** Every new field defaults to None. All existing
   tests pass unchanged. Static objects are a special case (single state,
   no timer, no change SM).

## Architecture Analysis

### What's Reusable (zero new code needed)

| Component | Location | Why Reusable |
|---|---|---|
| `EvidenceGraphLM` | `models/evidence_matching/learning_module.py` | Same algorithm for both morphology and behavior |
| `GridObjectModel` | `models/object_model.py` | Each state's sub-graph is a standard model |
| `ObservationProcessor` | `models/sensor_modules.py` | Reused by ChangeDetectingSM for feature extraction |
| Voting protocol | `learning_module.py` + `model.py` | Extended with state, same mechanism |
| Heterarchy connections | `monty_base.py` | Behavior LMs are children in compositional hierarchy |
| `RuntimeContext` | `context.py` | Already has `timer` field (unused until now) |
| Panda3D rendering | `simulators/panda3d/` | Animated objects for testing |
| Behavior generators | `environments/behaviors.py` | Quick testing without GPU |

### What Needs New Implementation

| Component | Purpose |
|---|---|
| `State.inferred_state` | State dimension in CMP messages |
| `StateConditionedModel` | Multiple sub-graphs per object (one per behavioral state) |
| `Hypotheses.states` | State dimension in hypothesis arrays |
| `GlobalIntervalTimer` | Discrete timer for temporal context |
| `ChangeDetectingSM` | SM that detects local changes (optic flow, feature deltas) |
| Behavior testbed | Configs, synthetic environment, integration tests |

### Data Flow (behavior column)

```
                    NEW                          EXISTING (extended)
                    ───                          ───────────────────
Object animates  →  RGBA+depth frame t-1  ─┐
                    RGBA+depth frame t    ─┤→ ChangeDetectingSM
                                           │     ├─ ObservationProcessor (reused)
                                           │     ├─ Compute local flow (new)
                                           │     └─ Suppress global flow (new)
                                           │              ↓
                                           │     State(location, flow_direction,
                                           │           flow_magnitude, feature_deltas)
                                           │              ↓
                                           │     EvidenceGraphLM (behavior)
                                           │       hypothesis = (loc, rot, STATE)
                                           │       evidence per-STATE sub-graph
                                           │              ↓
GlobalIntervalTimer ──time_encoding──────────→   Timer input to L1
                    ←──reset/speed signals──┘
```

## Implementation Phases

### Phase 1: State in CMP

Add `inferred_state: int | None = None` to the `State` class.

- **Modify**: `states.py` — new parameter in `__init__`, update `__repr__`,
  skip in `_check_all_attributes` when None.
- **Test**: `test_state_cmp.py` — create State with/without inferred_state,
  verify backward compat, serialization.
- **Acceptance**: All 600+ existing tests pass. State with inferred_state=3
  round-trips correctly.

### Phase 2: State-Conditioned Models

A single object ID stores multiple sub-graphs indexed by state.

- **Create**: `state_conditioned_model.py` — `StateConditionedModel` wrapping
  `dict[int, GridObjectModel]`. API: `build_model(locs, feats, state_id)`,
  `get_states()`, `get_model_for_state(sid)`, `get_num_states()`. Stores
  transition sequence `[(from, to, interval), ...]`.
- **Modify**: `graph_memory.py` — `_build_graph()` accepts optional `state_id`,
  builds `StateConditionedModel` when present. Getters gain optional
  `state_id` param.
- **Test**: `test_state_conditioned_model.py` — build 1-state (compat) and
  3-state models, query each, nearest-neighbor scoped to state.
- **Quick test**: Use `stapler_press()` from behaviors.py with manual
  state_id tags.

### Phase 3: State in Hypotheses

Hypothesis = (location, rotation, state). Evidence tested per-state.

- **Modify**: `hypotheses.py` — add `states: Optional[NDArray[int64]]`.
  `hypotheses_updater.py` — initialize across all states.
  `hypotheses_displacer.py` — per-state KDTree lookup.
  `learning_module.py` — `possible_states` dict, MLH includes state,
  `get_output()` sets `inferred_state`.
- **Test**: `test_state_hypotheses.py` — verify hypothesis count =
  N_loc * N_pose * N_state, MLH converges to correct state.
- **Quick test**: Train stapler 2-state, present "open" → MLH.state=open.

### Phase 3b: Vote on State

State is sub-object-ID in voting.

- **Modify**: `learning_module.py` — `send_out_vote()` includes
  `inferred_state`. `_update_evidence_with_vote()` filters by state.
  `model.py` — `_combine_votes()` preserves `inferred_state`.
- **Test**: `test_state_voting.py` — same-state votes reinforce,
  different-state don't cross-reinforce.

### Phase 4: Global Interval Timer

Discrete timer counting elapsed time since last event.

- **Create**: `interval_timer.py` — `GlobalIntervalTimer(n_time_cells=32)`.
  Methods: `step()`, `reset()`, `get_active_cell()`,
  `get_time_encoding() -> ndarray(n,)`, `adjust_speed(factor)`.
- **Modify**: `context.py` — type hint. `monty_experiment.py` — create
  timer, call `timer.step()` each Monty step.
- **Test**: `test_interval_timer.py` — step/reset, encoding shape, speed.

### Phase 5: Change-Detecting Sensor Module

SM that detects local changes (optic flow, feature deltas).

- **Create**: `change_detecting_sm.py` — `ChangeDetectingSM(SensorModule)`.
  Reuses `ObservationProcessor`. Computes per-pixel displacement, subtracts
  global flow, checks threshold. Outputs State with flow features when
  change detected, `use_state=False` otherwise.
- **Test**: `test_change_detecting_sm.py` — static scene → False, object
  moves → True with correct flow, sensor moves → suppressed.

### Phase 6: Behavior Testbed

End-to-end testing at 5 difficulty levels.

- **Create**: `behavior_configs.py` — 2-SM/2-LM factory (CameraSM→morphology
  LM, ChangeDetectingSM→behavior LM). `behavior_environment.py` — synthetic
  replay environment.
- **Tests**: Level 1 (repeated motion), Level 2 (cross-morphology),
  Level 3 (multi-behavior). Panda3D: Fox.glb 3 animations.
- **Modify**: `behaviors.py` — add `state_id` tags and `speed` parameter.

### Phase 7: Event Detection and Timer Speed Adjustment

LMs detect state transitions and adjust timer speed.

- **Modify**: `learning_module.py` — `_detect_event()`, `_adjust_timer_speed()`.
  `interval_timer.py` — `receive_reset_signal()`, `receive_speed_adjustment()`.
  `monty_base.py` — collect signals, pass to timer.
- **Test**: `test_event_detection.py` — transition triggers reset,
  speed mismatch triggers adjustment.

## Key Design Decisions

1. **State as sub-graphs, not 4th dimension.** Theory doc explicitly rejects
   continuous 4D. States are discrete sub-models.
2. **None defaults everywhere.** `inferred_state=None`, `states=None`,
   `timer=None`. Existing code paths unaffected.
3. **ChangeDetectingSM is a new class.** Not a mode of CameraSM. Follows
   parvocellular/magnocellular analogy.
4. **Same LM algorithm.** No new matching logic.
5. **Heterarchy-compatible.** Behavior models are children of morphology
   models in hierarchy. No hierarchy code changes needed.

## References

- [Object Behaviors Theory](theory/object-behaviors.md)
- [Change-Detecting SM](future-work/sensor-module-improvements/change-detecting-sm.md)
- [State in Models](future-work/learning-module-improvements/include-state-in-models.md)
- [State in Hypotheses](future-work/learning-module-improvements/include-state-in-hypotheses.md)
- [State in CMP](future-work/cmp-hierarchy-improvements/include-state-in-CMP.md)
- [Vote on State](future-work/voting-improvements/vote-on-state.md)
- [Global Interval Timer](future-work/cmp-hierarchy-improvements/global-interval-timer.md)
- [Behavior Testbed](future-work/environment-improvements/object-behavior-test-bed.md)
- Hawkins, Lewis, et al. (2019). [A Framework for Intelligence and Cortical
  Function Based on Grid Cells in the Neocortex.](https://doi.org/10.3389/fncir.2018.00121)
