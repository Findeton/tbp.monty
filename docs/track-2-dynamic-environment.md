# Track 2 Side Project: Dynamic 3D Environment

## Purpose

Replace the synthetic behavior generators with a real 3D rendering pipeline
that supports animated, deformable models. The brain learns temporal patterns
from real sensory data — curvatures that change as a mesh deforms, normals
that rotate as a joint articulates. Synthetic generators approximate this;
a GPU-rendered animated model provides it.

## Problem Statement

1. **Habitat objects are static meshes.** No articulation, no deformation, no
   dynamics. A "walking person" cannot exist in Habitat.
2. **Behavior generators fake sensory data.** They produce plausible State
   objects but the features (curvatures, normals, colors) are hand-crafted,
   not extracted from actual geometry.
3. **Temporal training needs real rendered data.** The TemporalMemory should
   learn from the same feature extraction pipeline the LM uses — depth maps
   → 3D point clouds → surface normals → principal curvatures.

## Design Principles

1. **No duplicated code.** The existing feature extraction pipeline
   (DepthTo3DLocations → ObservationProcessor → CameraSM → State) is
   simulator-agnostic. It works on RGBA + depth buffers regardless of
   their source. The new simulator produces these buffers; everything
   downstream is reused.
2. **Backwards compatible.** All existing tests pass. The Habitat backend
   continues to work. The new backend is an alternative, not a replacement.
3. **Protocol-driven.** The new simulator implements the same `Simulator`
   protocol as Habitat and MuJoCo. It plugs into the same
   `SimulatedObjectEnvironment` → `EnvironmentInterface` → `SensorModule`
   pipeline.
4. **Static first, then motion.** Phase 1 proves the pipeline works with
   static objects (same primitives as Habitat). Phase 2 adds animation.

## Architecture Analysis

### What's Reusable (zero new code needed)

| Component | Location | Why Reusable |
|---|---|---|
| `Simulator` protocol | `simulators/simulator.py` | Pure protocol |
| `Environment` protocols | `environments/environment.py` | Pure protocols |
| `EnvironmentInterface` | `environments/embodied_data.py` | Works with any `SimulatedObjectEnvironment` |
| `Transform` pipeline | `environment_utils/transforms.py` | Operates on generic RGBA+depth arrays |
| `DepthTo3DLocations` | `environment_utils/transforms.py` | Camera intrinsics + rigid transforms (pure math) |
| `ObservationProcessor` | `models/sensor_modules.py` | Extracts features from `semantic_3d` arrays |
| `CameraSM` | `models/sensor_modules.py` | Produces `State` from processed observations |
| All `Action` classes | `actions/actions.py` | Backend-agnostic action definitions |
| Motor systems & policies | `models/motor_system.py`, `motor_policies.py` | Backend-agnostic |
| All learning modules | `models/evidence_matching/` etc. | Consume `State` objects only |

### What Needs New Implementation

| Component | Purpose |
|---|---|
| `Panda3DSimulator` | Offscreen GPU rendering, scene management |
| `Panda3DAgent` | Camera configuration, observation packaging |
| `Panda3DActuator` | Action → camera transform mapping |
| `Panda3DSensorConfig` | RGBD sensor setup |
| `Panda3DEnvironment` | `SimulatedObjectEnvironment` wrapper |
| `AnimationController` | Skeletal animation, morph targets (Phase 2) |
| `AssetRegistry` | glTF/GLB model loading and caching |

### Observation Pipeline

```
                    NEW (Panda3D)              EXISTING (reused as-is)
                    ─────────────              ─────────────────────────
Panda3D renders  →  RGBA (H,W,4)  ─┐
RGBA + depth        depth (H,W,1)  ─┤→ DepthTo3DLocations → semantic_3d (N,4)
                    semantic (H,W) ─┘       ↓               → sensor_frame_data
                                     ObservationProcessor    → world_camera (4,4)
                                            ↓
                                     CameraSM.step()
                                            ↓
                                     State(location, pose_vectors,
                                           curvatures, hsv, ...)
                                            ↓
                                     LearningModule.matching_step()
```

The critical contract: Panda3D must produce `SensorObservation` dicts with:
- `"rgba"`: `np.ndarray` shape `(H, W, 4)` dtype `uint8`
- `"depth"`: `np.ndarray` shape `(H, W, 1)` dtype `float32`
  (linearized metric distance, 0 = at camera, 1 = at far plane)
- `"semantic"`: `np.ndarray` shape `(H, W, 1)` dtype `int32` (optional)

## Framework Choice: Panda3D

| Criterion | Panda3D | Alternatives |
|---|---|---|
| Skeletal animation | Native (Actor/Joint, GPU vertex skinning) | ModernGL: manual shader |
| Morph targets | Shader-based | ModernGL: manual |
| glTF support | `panda3d-gltf` plugin | trimesh: import only |
| Offscreen rendering | Built-in GraphicsBuffer | All: possible |
| Scene graph | Full | ModernGL: none |
| Physics | Bullet/ODE integration | PyBullet: separate |
| pip installable | `pip install panda3d` | All: yes |
| Python 3.8 | Supported (1.10.x) | All: yes |

Panda3D wins because skeletal animation with GPU vertex skinning is the hard
part. Building it from scratch with raw OpenGL would add weeks.

## Model Format: glTF 2.0 / GLB

- Industry standard for animated 3D content
- Skeletal animation, morph targets, keyframe animation
- Free asset libraries: Mixamo (characters), Sketchfab
- Compact binary (GLB)
- Supported by `panda3d-gltf`

## File Structure

```
src/tbp/monty/simulators/panda3d/
    __init__.py
    simulator.py          # Panda3DSimulator (Simulator protocol)
    agents.py             # Panda3DAgent
    actuator.py           # Action → camera transform
    sensors.py            # Panda3DSensorConfig
    environment.py        # Panda3DEnvironment (SimulatedObjectEnvironment)
    animation.py          # AnimationController (Phase 2)
    asset_registry.py     # glTF/GLB model loading
    primitives.py         # Procedural sphere/cube/cone generation

tests/unit/simulators/panda3d/
    __init__.py
    test_panda3d_simulator.py    # Core rendering tests
    test_panda3d_actuator.py     # Action execution tests
    test_panda3d_features.py     # Feature extraction parity
    test_panda3d_environment.py  # Full pipeline tests
    test_panda3d_animation.py    # Animation tests (Phase 2)
```

## Implementation Phases

### Phase 1: Offscreen Rendering (static objects)

**Goal:** Render primitive shapes (sphere, cube, cone) to RGBA + depth
buffers in the correct format for the existing transform pipeline.

1. Install `panda3d` as optional dependency
2. Create `Panda3DSimulator` with offscreen `GraphicsBuffer`
3. Generate primitive meshes (sphere, cube, cone) procedurally
4. Render RGBA and depth from configurable camera pose
5. Linearize depth buffer to metric distances
6. Package as `SensorObservation` matching Habitat format
7. **Test:** render sphere at known distance, verify depth value

### Phase 2: Actions and Motor System

**Goal:** Camera can move around objects via the standard Action types.

1. Create `Panda3DActuator` implementing all actuator methods
2. Create `Panda3DAgent` with camera node
3. Map coordinate system (Panda3D Y-up/Z-forward → Monty convention)
4. **Test:** execute MoveForward, verify position changes

### Phase 3: Feature Extraction Parity

**Goal:** Same object rendered in Panda3D and Habitat produces
equivalent features (normals, curvatures, HSV).

1. Wire Panda3D depth into `DepthTo3DLocations`
2. Verify `ObservationProcessor` produces valid normals and curvatures
3. **Test:** sphere curvatures should be uniform and positive
4. **Test:** cube face normals should be axis-aligned

### Phase 4: Environment Integration

**Goal:** Full Monty pipeline works with Panda3D backend.

1. Create `Panda3DEnvironment` (SimulatedObjectEnvironment)
2. Wire into `EnvironmentInterface`
3. **Test:** full episode — add object, step, get observations, features valid

### Phase 5: Semantic Segmentation

**Goal:** Multi-object scenes with per-pixel semantic IDs.

1. Implement object-ID render pass (flat colors, decode to semantic ID)
2. **Test:** 3 objects in scene, verify correct semantic IDs

### Phase 6: glTF Model Loading

**Goal:** Load real 3D models (YCB objects, animated characters).

1. Create `AssetRegistry` for glTF/GLB files
2. Load via `panda3d-gltf`
3. **Test:** load real model, render, verify valid features

### Phase 7: Animation

**Goal:** Animated models with skeletal deformation.

1. Create `AnimationController`
2. Load glTF with embedded animations
3. Advance animation per step, verify mesh deforms (depth changes)
4. **Test:** animated model at frame 0 vs frame 10 → different depth

### Phase 8: Temporal Training Integration

**Goal:** TemporalMemory learns from rendered animated data.

1. Create temporal training episodes with animated models
2. Compare against behavior generator baseline
3. **Test:** temporal predictions from rendered data vs synthetic

## Depth Buffer Calibration

Critical detail. Panda3D depth buffer is in clip space [0, 1].
Must linearize to metric distance:

```python
z_linear = (near * far) / (far - depth_clip * (far - near))
```

Then normalize to Habitat's convention where depth values represent
distance from the camera plane, with max depth = 1.0 for the far plane.

Validate with known geometry: render a sphere of radius R at distance D,
verify center pixel depth = D - R (distance to nearest surface point).

## Dependencies

```toml
[project.optional-dependencies]
simulator_panda3d = [
    'panda3d>=1.10.14',
    'panda3d-gltf>=1.2.0',
]
```

Optional — core Monty does not require Panda3D.

### Phase 9: TBT-Native Sensorimotor Fixes

**Goal:** Fix two structural limitations in the behavior training pipeline
that violate Thousand Brains Theory principles, causing poor behavior and
shape discrimination across real 3D models.

#### Problem 1: Camera-induced flow noise in ChangeDetectingSM

**Root cause:** ChangeDetectingSM computes flow as centroid displacement of
all visible on-object 3D points between frames. Camera motion changes *which*
surface points are visible (occlusion/dis-occlusion), shifting the centroid
even for a completely static object. This visibility-change flow can exceed
the animation-induced flow, making behavior features noisy.

**TBT principle violated:** Cortical columns use motor efference copies to
predict expected sensory input. Prediction error — not raw displacement — is
the meaningful signal. Camera-induced centroid shift is predictable given the
motor command and should produce zero surprise.

**Fix: Point-correspondence flow**

Replace centroid displacement with nearest-neighbor point correspondence:

1. Store the previous frame's full on-object point cloud (world coords)
2. For each point in the current frame, find its nearest neighbor in the
   previous frame's cloud
3. Points with a neighbor within `correspondence_threshold` are "persistent"
   — same physical surface location visible in both frames
4. Compute RMS of per-point displacement magnitudes as flow magnitude.
   RMS captures non-rigid deformation even when net displacement ≈ 0
   (e.g., symmetric walking: left leg forward, right leg backward → mean
   displacement cancels, but RMS reflects genuine limb motion)
5. For a static object under camera motion: persistent points have identical
   world coords → RMS ≈ 0 (ego-motion naturally canceled)
6. For an animated object: mesh deformation moves surface points →
   RMS reflects true animation motion regardless of symmetry

This is the TBT-aligned approach: world-coordinate point correspondence
implicitly compensates for ego-motion without needing to explicitly model
the camera transform, because the camera's motion doesn't change world
coordinates of rigid surfaces.

**Files modified:**
- `src/tbp/monty/frameworks/models/change_detecting_sm.py`

#### Problem 2: Poor shape discrimination across models (deferred)

**Root cause:** The morphology LM in `Panda3DBehaviorExperiment` only uses
`pose_vectors` (surface normals), `on_object`, and `hsv` features. Without
curvature features, shapes with similar normals but different curvature
(e.g., flat robot panel vs curved fox ear) produce similar evidence.

**TBT principle:** Cortical columns use all available sensory features.
Curvature is a fundamental discriminative surface property.

**Status:** Deferred. Adding curvature features improves discrimination on
real curved models but **hurts** flat-faced synthetic models (boxes) where
curvature is uniformly zero, diluting the HSV signal that distinguishes
them. The fix requires curvature-aware weighting (scale curvature weight by
observed curvature variance) — a deeper change. Curvature can be enabled
via the `camera_features` parameter for pipelines using real models.

#### Validation

- 39 passed, 1 pre-existing failure (bar_nod twist/nod swap)
- Cross-morphology discrimination improved: 1 pre-existing failure
  resolved by correspondence flow (was 2 failures before)
- Fox Walk vs Survey: all 4 discrimination tests pass
- Robot Walking vs Dance: all 3 tests pass
- Three-model sensorimotor tests: all 5 pass (when run individually)
- 12+ non-robot diverse_models tests unaffected

### Phase 10: Temporal Priming A/B Validation (2026-03-26)

**Goal:** Prove temporal priming adds value on real rendered animated data.

Habitat A/B was the original validation plan, but Habitat only supports
static meshes — no temporal signal exists. Panda3D with animated models is
the correct platform.

**Test:** `TestTemporalPrimingAB` in `test_panda3d_e2e.py` (8 tests).
Trains two identical pipelines on Fox Walk + Fox Run:
- Control: no temporal memory
- Treatment: temporal memory (SDR 1024, sparsity 0.02) + surprise
  modulation (boost=0.5, penalty=0.3)

**Results:**
- Treatment evidence margin ≥ 90% of control (not hurting)
- Temporal predictions active (surprise < 1.0)
- Morphology recognition not degraded
- 55 total E2E tests pass (0 regressions)

## Risks

| Risk | Mitigation |
|---|---|
| Coordinate system mismatch | Explicit mapping + parity tests vs Habitat |
| Depth linearization errors | Known-geometry validation (sphere at known distance) |
| panda3d-gltf incomplete animation | Blender as offline converter to BAM |
| OpenGL context conflicts with Habitat | Separate processes; not used simultaneously |
| CI without GPU | Mark tests as optional (like Habitat/MuJoCo tests) |
