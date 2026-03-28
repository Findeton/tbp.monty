# Track 6 Validation Infrastructure Plan

## Problem Statement

Track 6 (Predictive Coding Heterarchy) implemented 13 milestones across 3 phases.
All features default to off. Zero validation gates have been evaluated. The reason:
**no test bed exists that provides meaningful temporal structure.**

### Why Habitat YCB Doesn't Work for Temporal Features

1. **Within-episode**: Agent follows a deterministic naive spiral on a single static
   object. TemporalMemory would memorize the scanning trajectory, not learn
   meaningful perceptual predictions. The surprise signal is degenerate: high on
   first pass, low on repeat, with no real content beyond "have I done this exact
   spiral before."

2. **Cross-episode**: YCB eval runs objects in arbitrary order. "After banana I see
   softball" has no semantic structure. HPC temporal learning is meaningful only
   when episode order reflects real-world structure.

### What CAN Be Validated on Habitat YCB (No Temporal Signal Needed)

| Feature | Why it works without temporal |
|---|---|
| T6.6a (context biasing) | Uses HPC association_strengths, not TM surprise |
| T6.7 (KDTree workers=-1) | Pure performance metric |
| T6.8 (ThreadPoolExecutor) | Pure performance metric |
| T6.13 (burst ranking recall) | Feature evidence quality, no TM dependency |
| T6.13a (offspring hypotheses) | Pose refinement, no TM dependency |

### What NEEDS Meaningful Temporal Structure

| Feature | What it needs |
|---|---|
| T6.0-T6.4 (surprise-modulated learning) | Surprise must separate novel from repeated — needs real temporal sequences |
| T6.5 (unified voting) | Temporal confusion classification needs TM surprise to carry signal |
| T6.6 (surprise-gated output) | Needs surprise to discriminate predicted vs novel observations |

## Available 3D Assets

### YCB Dataset (79 objects, already on disk as .glb)

All 79 YCB objects exist as textured `.glb` files at:
```
~/tbp/data/habitat/objects/ycb/meshes/<object_name>/google_16k/textured.glb
```

These are real photogrammetry-scanned household objects with textures and normals.
They can be loaded directly by Panda3D's `AssetRegistry` — no conversion needed.

**Representative subset for eval (15 objects, geometrically diverse):**

| Category | Objects | YCB IDs |
|---|---|---|
| Kitchen | mug, bowl, plate, fork, knife | 025, 024, 029, 030, 032 |
| Food | banana, apple, pear | 011, 013, 016 |
| Tools | power drill, wrench, scissors | 035, 042, 037 |
| Containers | cracker box, master chef can | 003, 002 |
| Other | tennis ball, foam brick | 056, 061 |

### Animated glTF Models (8 files, in repo)

Located at `tests/unit/simulators/panda3d/test_assets/animated/`:
- Fox.glb, BrainStem.glb, CesiumMan.glb, CesiumMilkTruck.glb
- RobotExpressive.glb, BoxAnimated.glb, RecursiveSkeletons.glb, InterpolationTest.glb

Useful for V2 (dynamic scenes) but not primary eval objects.

### Additional Sources (if needed)

- **Numenta Lab Objects**: `${MONTY_DATA}/numenta_lab` — photogrammetry scans
- **Compositional Objects**: `${MONTY_DATA}/compositional_objects`
- **Khronos glTF Samples**: Free, CC0 licensed, high-quality test models
  (already using Fox.glb, CesiumMan.glb etc. from this collection)

## What Panda3D Already Provides

The Panda3D simulator is more capable than currently utilized:

- **Multi-object scenes**: `add_object()` tracks multiple objects with independent
  IDs, positions, semantic IDs. Renders all objects together.
- **glTF/GLB loading**: Any mesh via `AssetRegistry`. YCB `.glb` files load directly.
- **Skeletal animation**: `AnimatedObject` with frame-level control.
- **Observation format**: Identical to Habitat (RGBA + depth per agent per sensor).
- **Implements same protocol**: `SimulatedObjectEnvironment` — `add_object()`,
  `remove_all_objects()`, `step()`, `reset()`, `close()`.

What's missing: an evaluation harness that runs structured multi-object episodes
through real EvidenceGraphLMs with proper metrics.

## Infrastructure Plan

### Phase V1: Panda3D Object Recognition Eval (Non-Temporal)

**Goal**: Run the non-temporal Track 6 features (T6.6a, T6.13, T6.13a) through a
real object recognition benchmark using Panda3D with real YCB meshes.

**Why this first**: Validates that Panda3D can replace Habitat for recognition
benchmarks. Unblocks all non-temporal validation. No new temporal infrastructure
needed.

#### V1.1: YCB Object Resolver

Small utility to resolve YCB object names to `.glb` paths:

```python
YCB_MESH_ROOT = Path(os.environ.get(
    "MONTY_DATA", "~/tbp/data"
)).expanduser() / "habitat/objects/ycb/meshes"

def ycb_glb_path(object_name: str) -> Path:
    """Resolve '011_banana' → full .glb path."""
    return YCB_MESH_ROOT / object_name / "google_16k" / "textured.glb"
```

~20 lines. Allows the eval harness and scene builder to reference YCB objects
by name while Panda3D loads the actual textured meshes.

#### V1.2: Panda3D Eval Harness

New file: `src/tbp/monty/simulators/panda3d/evaluation.py`

```
Panda3DEvalHarness
  - Takes: list of YCB object names, list of rotations,
    EvidenceGraphLM config, sensor/transform config
  - Training phase:
    For each object_name:
      1. Clear scene, add YCB .glb at identity rotation
      2. Run orbital exploration policy (N steps)
      3. Feed observations through transform pipeline → CameraSM → State
      4. Feed States through EvidenceGraphLM exploratory_step()
      5. Complete training episode (post_episode)
  - Eval phase:
    For each (object_name, rotation):
      1. Clear scene, add YCB .glb at rotation
      2. Run orbital exploration policy (N steps)
      3. Feed States through EvidenceGraphLM matching_step()
      4. Record: detected_object, confidence, rotation_error, steps_to_converge
  - Returns: per-episode results + aggregate metrics
```

This is essentially what `MontyObjectRecognitionExperiment.run_epoch()` does, but
wired to Panda3D with real YCB meshes. No Hydra config dependency, no habitat-sim
dependency, directly programmable.

**Scope**: ~250 lines. Reuses existing `_RealPipeline` pattern from the Track 6
test suite (Panda3D → transforms → CameraSM → State → LM).

**Object set**: 15 YCB objects (the representative subset above). All real
textured meshes — no primitives.

**Validation targets**:
- Baseline: YCB recognition accuracy with and without burst sampling
- T6.6a: Context biasing → faster convergence when context is correct,
  still recognizes when context is wrong
- T6.13: Log burst ranking recall@K at each burst step
- T6.13a: Offspring → rotation error reduction on confident episodes
- T6.7/T6.8: Wall-clock comparison single-thread vs multi-thread

#### V1.3: Metrics Collection

Structured output per eval run:

```python
@dataclass
class EvalEpisodeResult:
    object_name: str
    rotation: tuple[float, float, float]
    detected_object: str | None
    correct: bool
    steps_to_converge: int | None
    rotation_error_deg: float | None
    max_evidence: float
    wall_clock_seconds: float

@dataclass
class EvalRunResult:
    episodes: list[EvalEpisodeResult]
    accuracy: float
    mean_steps_to_converge: float
    mean_rotation_error_deg: float
    wall_clock_total: float
    config: dict  # All params for reproducibility
```

### Phase V2: Multi-Object Scene Navigation (Temporal Within-Episode)

**Goal**: Create episodes where the agent navigates between multiple real YCB
objects in a scene, providing meaningful within-episode temporal structure for TM.

**Why**: TM surprise becomes meaningful when the observation sequence reflects
genuine perceptual transitions — moving from a banana to a mug, experiencing
occlusion, seeing unexpected objects. Not "step 12 of the spiral" but "I was
looking at a curved yellow surface and now I see a flat brown box."

#### V2.1: Scene Builder

New file: `src/tbp/monty/simulators/panda3d/scenes.py`

```python
@dataclass
class SceneObject:
    ycb_name: str           # e.g. "011_banana"
    position: VectorXYZ     # world position
    rotation: QuaternionWXYZ

@dataclass
class SceneSpec:
    objects: list[SceneObject]
    name: str               # e.g. "kitchen_counter"

# Predefined scenes using real YCB objects
KITCHEN_SCENE = SceneSpec(
    name="kitchen_counter",
    objects=[
        SceneObject("025_mug",        (0.0, 0.0, 0.0), (1,0,0,0)),
        SceneObject("030_fork",       (0.3, 0.0, 0.0), (1,0,0,0)),
        SceneObject("029_plate",      (-0.3, 0.0, 0.0), (1,0,0,0)),
        SceneObject("003_cracker_box", (0.0, 0.0, 0.3), (1,0,0,0)),
    ],
)

WORKSHOP_SCENE = SceneSpec(
    name="workshop_bench",
    objects=[
        SceneObject("035_power_drill",   (0.0, 0.0, 0.0), (1,0,0,0)),
        SceneObject("042_adjustable_wrench", (0.3, 0.0, 0.0), (1,0,0,0)),
        SceneObject("037_scissors",      (-0.3, 0.0, 0.0), (1,0,0,0)),
        SceneObject("048_hammer",        (0.0, 0.0, 0.3), (1,0,0,0)),
    ],
)

FRUIT_BOWL_SCENE = SceneSpec(
    name="fruit_bowl",
    objects=[
        SceneObject("011_banana",  (0.0, 0.0, 0.0), (1,0,0,0)),
        SceneObject("013_apple",   (0.15, 0.0, 0.1), (1,0,0,0)),
        SceneObject("016_pear",    (-0.15, 0.0, 0.1), (1,0,0,0)),
        SceneObject("024_bowl",    (0.0, -0.05, 0.0), (1,0,0,0)),
    ],
)
```

Scenes place real YCB meshes at realistic relative positions. The agent navigates
between them using a waypoint policy.

#### V2.2: Navigation Motor Policy

New motor policy: `WaypointMotorPolicy`

```
Given: list of (position, look_at) waypoints + dwell_steps per waypoint
Produces: camera pose sequence that:
  1. Orbits around waypoint[0] for dwell_steps (exploring one object)
  2. Interpolates camera to waypoint[1] over transit_steps
  3. Orbits around waypoint[1] for dwell_steps
  4. ...
```

This gives the agent structured movement through a multi-object scene with
predictable "explore → transit → explore" temporal rhythm. TM can learn this
rhythm; surprise measures deviations from it.

The temporal signal is now meaningful:
- "After seeing banana curvatures → transit → mug handle geometry" is a real
  perceptual sequence that repeats when revisiting the same scene
- TM surprise spikes at object transitions, drops during steady exploration
- Replacing one object in the scene → surprise spikes at that position
- This is genuinely different from memorizing a deterministic spiral

#### V2.3: Temporal Validation Metrics

**T6.0 (signal quality)**:
- Run agent through kitchen scene with known YCB objects, then swap mug for drill
- Measure TM surprise at the swap point → should spike
- AUC of surprise for "object changed" vs "object same" trials across 50 trials
- Gate: AUC > 0.7 to proceed with Phase 1 features

**T6.1-T6.2 (surprise-modulated learning)**:
- Repeated navigation of same kitchen scene → TM surprise should decrease
- Compare convergence with boost=0 vs boost=2.0
- Metric: episodes to reach mean surprise < 0.3

**T6.5 (unified voting)**:
- Multi-LM setup with shared scene
- LM_0 trained on kitchen objects; LM_1 trained on workshop objects
- Navigate kitchen scene → LM_0 confident, LM_1 confused
- Swap to hybrid scene → both partially confused
- Metric: vote accuracy under unified vs conditional voting

**T6.6 (surprise-gated output)**:
- In a stable kitchen scene, after learning, most observations are predicted
- Count: fraction of steps where get_output() returns confirmed vs full
- Metric: bandwidth reduction (confirmed steps / total steps) without
  accuracy loss

### Phase V3: Structured Episode Sequences (Temporal Cross-Episode)

**Goal**: Create evaluation protocols where episode order has semantic structure,
making HPC temporal learning meaningful.

#### V3.1: Scene Contexts

Use the predefined scenes from V2.1 to create structured episode sequences:

```python
CONTEXT_SEQUENCE = [
    # Morning routine: kitchen → fruit → kitchen
    ("kitchen_counter", "fruit_bowl", "kitchen_counter"),
    # Work session: workshop → workshop → workshop
    ("workshop_bench", "workshop_bench", "workshop_bench"),
]
```

Each "episode" is a scene navigation (V2). The sequence of episodes reflects
real-world patterns: you visit the kitchen, then the fruit bowl, then back.
HPC learns: "after kitchen scene, likely fruit bowl next."

This uses the same real YCB objects throughout — banana, mug, drill, wrench,
etc. No primitives anywhere in the pipeline.

#### V3.2: Context Prediction Metrics

**T6.3 (HPC surprise modulation)**:
- Run 5 cycles of kitchen → workshop → fruit_bowl sequence
- Measure HPC prediction accuracy at cycle 1 vs cycle 5
- With boost=0 vs boost=2.0: does boosted HPC learn faster?

**T6.6a (context biasing)**:
- After learning kitchen context, start new episode with HPC predicting "kitchen"
- Measure: steps to recognize mug (in-context) vs drill (out-of-context)
- Context should accelerate in-context, not prevent out-of-context

## Priority Order

```
V1.1 (YCB resolver)         — ~20 lines, maps names to .glb paths
V1.2 (eval harness)         — ~250 lines, unblocks 5 validation gates
V1.3 (metrics)              — ~80 lines, structured output
  ↓ validates: T6.6a, T6.7, T6.8, T6.13, T6.13a
  ↓ all using real YCB meshes (banana, mug, drill, etc.)

V2.1 (scene builder)        — ~150 lines, multi-object YCB scenes
V2.2 (waypoint policy)      — ~100 lines, structured navigation
V2.3 (temporal metrics)     — logging + analysis
  ↓ validates: T6.0, T6.1, T6.2, T6.5, T6.6
  ↓ using kitchen/workshop/fruit_bowl scenes with real meshes

V3.1 (scene contexts)       — ~50 lines, episode sequencing
V3.2 (context metrics)      — ~100 lines, HPC validation
  ↓ validates: T6.3, T6.6a (with real HPC)
```

**V1 is the critical path.** It validates the non-temporal features that are
most likely to actually work (they don't depend on noisy temporal signals) and
proves that Panda3D can run real YCB recognition benchmarks. V2 and V3 build
on it.

## Risks

- **Panda3D rendering vs Habitat rendering**: YCB `.glb` files were created for
  habitat-sim. Panda3D's glTF loader (via panda3d-gltf) may render them
  differently — different lighting model, different normal handling, different
  texture interpolation. This could affect feature extraction (HSV, curvatures).
  Mitigation: render a few YCB objects in both simulators and visually compare
  before building the full eval pipeline.

- **Single ShowBase constraint**: Panda3D allows only one ShowBase per process.
  The eval harness must manage this carefully — no parallel Panda3D instances.
  Sequential episodes within one ShowBase session.

- **Temporal signal might still be weak**: Even with multi-object scenes (V2),
  TM surprise at object transitions might not be a reliable signal if the SDR
  encoding doesn't distinguish objects well enough. Mitigation: V2.3 measures
  this directly — if AUC < 0.7, we know the signal is too noisy and Phase 1
  features should remain off.

- **Object scale**: YCB objects have varied real-world scales (banana ~20cm,
  cracker box ~30cm). The eval harness needs to handle this — either normalize
  object bounding boxes or adjust camera distance per object. The existing
  Habitat configs handle this via per-object scale in YAML; the Panda3D harness
  should do the same.

- **Multi-object occlusion**: In V2 scenes, objects may occlude each other from
  certain camera angles. This is realistic (and desirable for temporal
  validation) but may confuse the depth pipeline. Mitigation: place objects with
  enough spacing that partial occlusion occurs naturally but total occlusion
  is rare.

## Relationship to Track 6

This infrastructure does NOT implement new Track 6 features. It provides the
test bed to validate features that already exist but have never been measured.
All objects are real YCB meshes — no synthetic primitives.

After V1, we can answer: "Do T6.6a, T6.13, T6.13a actually improve recognition
on real YCB objects rendered through Panda3D?"

After V2, we can answer: "Is TM surprise a useful signal when navigating between
real objects in a scene?"

After V3, we can answer: "Does HPC context biasing accelerate recognition in
structured environments with real object categories?"

If the answer to V2's question is "no," then T6.0-T6.6 should remain off and
the code should be considered speculative scaffolding. If "yes," then we have
the infrastructure to tune parameters and run proper ablations.
