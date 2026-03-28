# Track 4: Audio Sensorimotor System

## Objective

Add an auditory modality to Monty, starting with the absolute minimum viable
system — 1 SM, 1 LM, synthetic beep sequences — and scaling toward full
human-level audio (20 Hz–20 kHz, ~120 dB, binaural, active head rotation).

This document analyzes the problem from first principles, identifies where the
existing CMP architecture fits naturally and where it requires genuine extension,
and proposes a phased implementation plan starting from the simplest possible
audio system.

---

## 0. Phase 0: Minimum Viable Audio (1 SM, 1 LM, beep sequences)

### 0.1 What is the simplest audio input that exercises the Monty pipeline?

A **sequence of pure-tone beeps** at different frequencies and amplitudes.
Each beep is fully described by three numbers: frequency (Hz), amplitude (dB),
and duration (ms). Between beeps there is silence.

This is the auditory equivalent of touching colored dots on a surface — the
simplest possible input that has both features (frequency, amplitude) and
temporal structure (sequence order, timing).

Example "audio objects" — patterns of beeps:

```
"doorbell":   beep(880 Hz, 70 dB, 200ms) → silence(100ms) → beep(660 Hz, 65 dB, 300ms)
"alarm":      beep(1000 Hz, 80 dB, 100ms) → silence(50ms) → beep(1000 Hz, 80 dB, 100ms) → silence(50ms) → ...  (repeating)
"three_tones": beep(440 Hz, 60 dB, 150ms) → beep(554 Hz, 60 dB, 150ms) → beep(659 Hz, 60 dB, 150ms)
"descending":  beep(659 Hz, 60 dB, 150ms) → beep(554 Hz, 60 dB, 150ms) → beep(440 Hz, 60 dB, 150ms)
"siren":      beep(600 Hz, 75 dB, 100ms) → beep(900 Hz, 75 dB, 100ms) → beep(600 Hz, 75 dB, 100ms) → ...
```

### 0.2 Why beeps are the right starting point

1. **No cochlear model needed.** Each beep has exactly one frequency — no need
   for a filterbank to decompose a complex waveform. The SM can directly receive
   `(frequency_hz, amplitude_db, is_active)` per time step, bypassing all signal
   processing.

2. **Exercises the core pipeline.** Despite being trivial acoustically, beep
   sequences exercise:
   - SM → CMP State conversion (frequency → location, amplitude → feature)
   - Displacement computation (Δfrequency, Δamplitude between consecutive beeps)
   - Evidence accumulation in the LM (does the displacement sequence match?)
   - Temporal learning (TemporalMemory predicts the next beep from the previous)
   - Silence handling (`use_state=False` between beeps)

3. **Directly analogous to existing behaviors.py generators.** The `behaviors.py`
   file generates synthetic State sequences for temporal learning (walking gait,
   stapler press, pendulum, etc.) — no Habitat, no real sensor, just physics →
   State. A beep sequence generator is exactly the same pattern.

4. **"Three_tones" vs. "descending" tests temporal directionality.** These are
   the same three frequencies in opposite order. If the system can distinguish
   them, temporal order is working. If it can't, we know exactly what to fix.

### 0.3 Architecture: 1 SM, 1 LM

```
beep_generator() → [State, State, State, ...]
                          │
                    ┌─────┴─────┐
                    │  AudioSM  │   (1 SM — passes through features, computes displacement)
                    └─────┬─────┘
                          │
                    ┌─────┴─────┐
                    │  LM + TM  │   (1 EvidenceGraphLM with optional TemporalMemory)
                    └───────────┘
```

No motor system. No binaural processing. No filterbank. No hierarchy. No HPC.

### 0.4 CMP State for beeps

Each beep becomes a State. The mapping from beep parameters to CMP fields:

```python
State(
    location=np.array([
        np.log2(frequency_hz),   # log-frequency: ~6.5 (100 Hz) to ~13.3 (10 kHz)
        amplitude_db / 120.0,    # normalized: 0.0 to 1.0
        0.0,                     # no spatial dimension yet (mono)
    ]),
    morphological_features={
        "pose_vectors": np.eye(3),  # identity — beeps have no orientation
        "pose_fully_defined": False, # pure tones have no "shape" to orient
    },
    non_morphological_features={
        "frequency_hz": frequency_hz,    # raw value for matching
        "amplitude_db": amplitude_db,    # raw value for matching
        "is_onset": is_first_frame,      # True on beep start
        "is_offset": is_last_frame,      # True on beep end
    },
    confidence=1.0 if is_beep_active else 0.0,
    use_state=is_beep_active,      # False during silence gaps
    sender_id="audio_sm_0",
    sender_type="SM",
)
```

**Key design choices for the minimum:**

- **location[0] = log2(freq)**: Log-frequency is the natural metric for pitch.
  An octave is a constant displacement of 1.0, regardless of absolute frequency.
  This gives us pitch transposition invariance for free — the same melody at a
  higher pitch produces the same displacement sequence, just shifted.

- **location[1] = normalized dB**: Amplitude is a coordinate, not just a feature
  (as discussed in §1.5 of the full plan). But for pure tones, there's no
  intensity-dependent spectral distortion, so this axis is simple.

- **location[2] = 0**: Placeholder for spatial dimension. Unused in Phase 0.

- **pose_vectors = identity**: Pure tones have no spectral "shape" — they're
  points in frequency space, not extended surfaces. `pose_fully_defined=False`
  tells the LM not to use pose for matching. This is honest — there's genuinely
  no orientation information in a single-frequency beep.

- **use_state=False during silence**: The SM emits nothing during gaps. The LM
  sees a sequence of beep-States with no silence-States interleaved. This is the
  simplest silence handling.

### 0.5 What the LM learns

For "doorbell" = [880 Hz @ 70 dB] → [660 Hz @ 65 dB]:

```
Evidence graph for "doorbell":
  Node 0: loc=(9.78, 0.583, 0)  feat={freq=880, amp=70, onset=T}
     │
     │ edge: displacement=(-0.42, -0.042, 0)    [pitch drops, gets slightly quieter]
     ▼
  Node 1: loc=(9.37, 0.542, 0)  feat={freq=660, amp=65, onset=T}
```

For "three_tones" = [440 Hz] → [554 Hz] → [659 Hz]:

```
Evidence graph for "three_tones":
  Node 0: loc=(8.78, 0.50, 0)  feat={freq=440, amp=60}
     │ displacement=(+0.33, 0, 0)     [pitch rises by major third]
     ▼
  Node 1: loc=(9.11, 0.50, 0)  feat={freq=554, amp=60}
     │ displacement=(+0.25, 0, 0)     [pitch rises by minor third]
     ▼
  Node 2: loc=(9.36, 0.50, 0)  feat={freq=659, amp=60}
```

For "descending" = [659 Hz] → [554 Hz] → [440 Hz]:

```
Evidence graph for "descending":
  Node 0: loc=(9.36, 0.50, 0)  feat={freq=659, amp=60}
     │ displacement=(-0.25, 0, 0)     [pitch drops by minor third]
     ▼
  Node 1: loc=(9.11, 0.50, 0)  feat={freq=554, amp=60}
     │ displacement=(-0.33, 0, 0)     [pitch drops by major third]
     ▼
  Node 2: loc=(8.78, 0.50, 0)  feat={freq=440, amp=60}
```

"Three_tones" and "descending" have the same nodes but different displacement
signs. If the LM uses signed displacements (which it does — `State.location`
differences are signed vectors), these are distinguishable even in Phase 0 without
any special temporal directionality mechanism. The evidence graph naturally stores
directed edges because the displacement from Node 0 → Node 1 has the opposite
sign from Node 1 → Node 0.

### 0.6 TemporalMemory integration

The `TemporalMemory` already accepts `State` objects, extracts features, encodes
them as SDRs, and learns Hebbian transitions. For beep sequences:

- Each beep-State encodes to an SDR based on (log-freq, amplitude)
- The Hebbian matrix learns: "after the 880 Hz beep, expect 660 Hz"
- On the second hearing of "doorbell", the TM predicts 660 Hz after hearing
  880 Hz — low surprise
- On hearing "alarm" after training on "doorbell", the TM expects 660 Hz but
  gets 1000 Hz — high surprise

This works out of the box with zero modifications to `TemporalMemory`.

### 0.7 Beep generator (new function in behaviors.py pattern)

```python
def beep_sequence(beeps, frames_per_beep=5, silence_frames=2):
    """Generate a State sequence from a list of (freq_hz, amplitude_db) beeps.

    Analogous to walking_gait() and stapler_press() — generates a sequence
    of States representing an auditory behavior, no real audio required.

    Parameters
    ----------
    beeps : list of (float, float)
        Each tuple is (frequency_hz, amplitude_db).
    frames_per_beep : int
        How many State frames each beep occupies.
    silence_frames : int
        How many silence frames between beeps.
    """
    states = []
    for freq_hz, amp_db in beeps:
        for f in range(frames_per_beep):
            states.append(State(
                location=np.array([
                    np.log2(freq_hz),
                    amp_db / 120.0,
                    0.0,
                ]),
                morphological_features={
                    "pose_vectors": np.eye(3),
                    "pose_fully_defined": False,
                },
                non_morphological_features={
                    "frequency_hz": freq_hz,
                    "amplitude_db": amp_db,
                    "is_onset": f == 0,
                    "is_offset": f == frames_per_beep - 1,
                },
                confidence=1.0,
                use_state=True,
                sender_id="audio_sm_0",
                sender_type="SM",
            ))
        # Silence gap
        for _ in range(silence_frames):
            states.append(State(
                location=np.array([0.0, 0.0, 0.0]),
                morphological_features={
                    "pose_vectors": np.eye(3),
                    "pose_fully_defined": False,
                },
                non_morphological_features={},
                confidence=0.0,
                use_state=False,
                sender_id="audio_sm_0",
                sender_type="SM",
            ))
    return states
```

Named patterns:

```python
DOORBELL  = [(880, 70), (660, 65)]
ALARM     = [(1000, 80), (1000, 80), (1000, 80)]
THREE_UP  = [(440, 60), (554, 60), (659, 60)]
THREE_DOWN = [(659, 60), (554, 60), (440, 60)]
SIREN     = [(600, 75), (900, 75), (600, 75), (900, 75)]
```

### 0.8 What this tests

| Test | What it validates |
|------|-------------------|
| Learn "doorbell", recognize on re-presentation | Basic evidence accumulation works with audio States |
| Learn "doorbell" at 880/660 Hz, test at 1760/1320 Hz (octave up) | Pitch transposition invariance via log-freq displacement matching |
| Learn "doorbell" at 70/65 dB, test at 50/45 dB (quieter) | Amplitude invariance via normalized-dB displacement matching |
| Learn "three_up", reject "three_down" | Temporal directionality (signed displacements distinguish sequences) |
| Learn "alarm" (repeating), predict 4th beep | TemporalMemory Hebbian prediction on periodic sequence |
| Train TM on "doorbell", present "siren" → high surprise | Surprise signal for novel sequences |
| Silence gaps handled (no LM crash, no false evidence) | `use_state=False` pipeline correctness |

### 0.9 Implementation plan for Phase 0

**New files:**

| File | Contents | Lines (est.) |
|------|----------|-------------|
| `environments/audio_behaviors.py` | `beep_sequence()` + named patterns (DOORBELL, ALARM, etc.) | ~80 |
| `models/audio_sm.py` | `AudioSM(SensorModule)` — minimal: passes through beep States, no signal processing | ~60 |
| `tests/unit/audio/test_audio_beeps.py` | Tests from §0.8 table | ~150 |

**Modified files:**

None. Phase 0 uses the existing `EvidenceGraphLM` and `TemporalMemory` with no
modifications. If those can't handle audio States, that tells us exactly what
needs changing.

**Dependencies:** None beyond numpy (already available).

### 0.10 What Phase 0 deliberately omits

- No waveform input (beeps are synthetic State sequences, like behaviors.py)
- No filterbank (no frequency decomposition — each beep is a single known freq)
- No binaural processing (mono, no spatial dimension)
- No motor system (no head rotation, no exploration policy)
- No hierarchy (1 SM → 1 LM, flat)
- No cochlear compression (pure tones don't have spectral spread)
- No HRTF
- No real audio files

Each of these is added in subsequent phases only when Phase 0 is working.

### 0.11 First-principles check: is this too simple?

**Could this be done without Monty at all?** Yes — a lookup table could
distinguish 5 beep patterns. But that misses the point. The question is: does
Monty's displacement-matching + evidence-accumulation + temporal-memory pipeline
*naturally handle* audio-like input without special-casing? If Phase 0 works with
zero modifications to the LM, that validates the Thousand Brains Theory claim
that the same cortical algorithm works across modalities.

**What can't this do that a human can?** Everything involving real-world sound:
recognizing speech, separating sources, localizing in space, handling noise. But
it can do what a newborn's auditory system does in the first days: detect that a
pattern of tones has occurred before, distinguish rising from falling sequences,
and be surprised by a novel sequence.

---

---

## 1. First-Principles Analysis

### 1.1 What is a sensor module, really?

A sensor module in Monty does one thing: convert raw sensory data into a `State`
that a learning module can use for evidence accumulation. The `State` format
(defined in `states.py`) requires:

```
location:                 np.ndarray, shape (3,)
morphological_features:   {pose_vectors: (3,3), pose_fully_defined: bool}
non_morphological_features: dict of arbitrary features
confidence:               float in [0,1]
use_state:                bool
sender_id:                str
sender_type:              "SM"
```

The LM then:
1. Computes **displacements** between consecutive locations.
2. Matches features + displacements against stored graph nodes.
3. Accumulates **evidence** for object hypotheses.

This is the contract. Anything that satisfies it can drive learning. The question
is: what should `location`, `pose_vectors`, and `features` *mean* for audio?

### 1.2 The geometry of hearing vs. seeing

In vision, the SM explores a **2D surface embedded in 3D Euclidean space**. The
sensor touches a point on the surface. The LM builds a graph where:
- Nodes = features at surface points
- Edges = spatial displacement vectors between points
- Object identity = a graph that matches regardless of the object's 3D pose

In hearing, there is no persistent surface to touch. Instead, the cochlea
decomposes sound into a **spectrotemporal field** — energy distributed across
frequency and time, at some intensity. This field has three axes:

| Axis | Nature | Range | Metric |
|------|--------|-------|--------|
| **Frequency** | Quasi-spatial, persistent, explorable | 20 Hz – 20 kHz | Log-frequency (octaves) |
| **Intensity** | Quasi-spatial, observable | 0 – 120 dB SPL | Decibels (log power) |
| **Time** | Sequential, irreversible | Continuous | Seconds |

The first-principles question: **is this 3D spectrotemporal space analogous
enough to 3D Euclidean space for the existing LM algorithm to work?**

### 1.3 Where the analogy holds

| Visual concept | Audio analogue | Why it works |
|----------------|----------------|--------------|
| Surface point at (x,y,z) | Spectrotemporal point at (log-freq, dB, time) | Both are locations in a 3D space |
| Spatial displacement between points | Δ(log-freq, dB, time) between successive frames | The LM accumulates evidence from displacement sequences |
| Surface normal (pose_vectors[0]) | Spectral gradient direction | Describes "which way" energy is changing in freq×dB space |
| Curvature directions (pose_vectors[1,2]) | Temporal and intensity gradient directions | Second-order structure of the spectral surface |
| Color (non-morphological feature) | Harmonicity, spectral shape, onset strength | Features at a location that help identify the object |
| Object = graph of (features, displacements) | Auditory object = graph of (spectral features, Δfreq/ΔdB/Δtime) | A spoken word, a bird call, a door slam — all are characteristic trajectories through spectrotemporal space |
| Motor action = move sensor | Motor action = rotate head | Changes the observation (binaural cues change with head pose) |

### 1.4 Where the analogy breaks

These are genuine problems, not hand-waveable:

**Problem 1: Time is irreversible.**
In vision, the sensor can revisit any surface point. The LM exploits this: it
moves to a new location, checks features, moves back to confirm. In audio, time
only moves forward. You cannot re-observe the /k/ in "cat" after hearing the /t/.

*Consequence:* The auditory LM cannot do iterative hypothesis refinement on a
single utterance the way a visual LM refines on a single object. Instead, it must
accumulate evidence in a single forward pass — or rely on repetition (hearing the
word again). This is biologically accurate: humans also need repetition to learn
new words, and "replay" in hippocampus serves a similar re-traversal function.

*Mitigation:* The existing `TemporalMemory` (SDR + Hebbian) already handles
forward-only sequences. The evidence graph LM can also work in a forward-only
mode — it receives displacements and accumulates evidence without requiring
backtracking. The `motor_only_step` mechanism already supports steps where the
sensor moves but the LM doesn't update, which maps to "silence between sounds."

**Problem 2: Intensity is not independent of shape.**
In vision, moving an object farther away changes its apparent size but not its
shape (to first order). In audio, changing intensity changes spectral shape:
- **Upward spread of masking**: loud sounds bleed into higher frequency channels
- **Equal-loudness contours** (Fletcher-Munson): perceived relative loudness
  across frequencies changes with overall level
- **Cochlear compression**: OHC active gain is level-dependent, nonlinear

This means a simple translation along the dB axis does not give intensity
invariance. The spectral "surface" deforms with intensity.

*Consequence:* The LM cannot treat intensity translation the same as spatial
translation. The graph representation must either (a) store multiple
intensity-dependent versions of the same object, or (b) learn the systematic
deformation as a function of overall level.

*Mitigation:* Apply cochlear compression (log + OHC nonlinearity) in the SM
*before* constructing the State. This is what the biological cochlea does — outer
hair cells compress the dynamic range and partially normalize level-dependent
spectral changes. After compression, the residual level-dependence is small enough
that the LM's feature tolerances can absorb it. This is analogous to how retinal
adaptation normalizes luminance before cortical processing.

**Problem 3: Auditory objects are inherently temporal.**
A visual object exists simultaneously — all its surface points are present at once.
An auditory object unfolds over time. The word "cat" is not the simultaneous
presence of /k/, /æ/, /t/ — it is their *sequence*. The graph edges along the time
axis are therefore not merely spatial displacements but **ordered transitions**.

*Consequence:* The evidence graph needs directed edges for the time dimension
(freq and dB edges can remain undirected). The existing `EvidenceGraphLM` uses
undirected spatial displacements. This is the most significant architectural
extension required.

*Mitigation:* The graph already stores displacement vectors with signs. Making
evidence accumulation direction-sensitive along the time axis (penalize
hypotheses where observed Δtime is negative relative to expected) is a targeted
change to the hypothesis updater, not a rewrite of the architecture.

**Problem 4: Polyphony — multiple simultaneous sources.**
Vision has occlusion (one object hides another), but each pixel belongs to one
object. Audio has superposition — at each frequency and time, energy from multiple
sources adds together. The cocktail party problem.

*Consequence:* A single auditory SM output at any moment may contain features
from multiple objects. The LM's evidence accumulation will be noisy.

*Mitigation:* This is actually analogous to the visual "clutter" problem that
Monty already handles through multi-column voting. Multiple auditory columns
(SMs at different frequency bands) will each see different mixtures. A source
with strong harmonics at 200, 400, 600 Hz will activate the SMs covering those
bands consistently, while a noise source at 2 kHz activates a different SM.
The voting mechanism across LMs separates these. Additionally, binaural
processing (ITD/ILD) provides a spatial separation cue that vision gets for
free from pixel assignment.

### 1.5 The intensity axis in detail

Intensity deserves special treatment because it is the axis most unlike anything
in vision. Consider what happens at a single cochlear place (one frequency
channel) as intensity varies:

| dB SPL | Auditory nerve response | Information available |
|--------|------------------------|----------------------|
| 0–20 | Threshold, few spikes | Detect presence/absence |
| 20–60 | Rate increases ~linearly with dB | Encode fine intensity differences |
| 60–90 | Rate saturates for best-frequency fibers; high-threshold fibers recruited | Wider neural population active |
| 90–120 | Maximum recruitment; spread to adjacent frequency channels | Spectral shape distortion |

The key insight: **intensity is not just "signal strength" — it determines which
neural population is active and how the spectral representation is distributed**.
At low intensity, a tone activates a narrow frequency band. At high intensity, it
activates a wide band (upward spread). This is geometrically meaningful.

For the CMP State, this means:
- `location[1]` (intensity) should be the **compressed** intensity after OHC
  modeling, not raw dB SPL. This partially absorbs the nonlinear spread.
- A non-morphological feature like `spectral_bandwidth` captures the residual
  spread effect — at a given log-frequency, a loud tone will have wider bandwidth
  than a quiet one. This is analogous to how a visual SM reports `object_coverage`
  — how much of the patch is filled.

### 1.6 Summary: what's genuinely new vs. what maps directly

| Component | Maps directly to existing Monty | Requires new work |
|-----------|-------------------------------|-------------------|
| SM → State conversion | Yes — same contract | New: cochlear filterbank, binaural processing |
| 3D location space | Yes — (log-freq, dB, azimuth) | New: coordinate semantics differ from Euclidean |
| pose_vectors | Partially — gradient directions work | New: interpretation differs (not surface normal) |
| Feature matching in LM | Yes — tolerances absorb domain differences | Tolerances must be tuned for audio features |
| Displacement-based evidence | Yes — same algorithm | New: time-axis directionality constraint |
| Multi-column voting | Yes — same mechanism | No change needed |
| Heterarchy topology | Yes — same `make_local_heterarchy` | New: binaural merge level |
| Motor system | Yes — same interface | New: head rotation policy, HRTF filtering |
| TemporalMemory integration | Yes — SDR projection works on spectral frames | No change needed |
| HPC cross-modal binding | Yes — top-level LM feeds HPC | No change needed |

---

## 2. Human Auditory System: Target Specifications

These numbers define what "as good as a human" means quantitatively.

### 2.1 Peripheral (cochlea)

| Parameter | Human value | Implementation target |
|-----------|-------------|----------------------|
| Frequency range | 20 Hz – 20 kHz | 20 Hz – 20 kHz at 44.1 kHz sample rate |
| Frequency resolution | ~3,500 inner hair cells; ~24 Bark / ~38 ERB bands | 64 ERB-spaced gammatone filters (matches effective cortical resolution) |
| Dynamic range | ~120 dB SPL | 24-bit audio (144 dB theoretical); compress to ~40 dB internal range via OHC model |
| Temporal resolution | ~2–5 ms (phase-locking up to ~5 kHz) | 2.9 ms hop size (512 samples @ 44.1 kHz → 345 frames/sec) |
| Cochlear compression | ~0.2–0.3 dB/dB at moderate levels | Power-law compression: out = sign(in) × |in|^0.3 per channel |
| Phase locking | Up to ~4–5 kHz | Preserve instantaneous phase in channels < 5 kHz |

### 2.2 Binaural

| Parameter | Human value | Implementation target |
|-----------|-------------|----------------------|
| Interaural Time Difference (ITD) | Max ~700 μs; resolve ~10 μs | Cross-correlation per low-freq channel; 44.1 kHz → 22.7 μs resolution |
| Interaural Level Difference (ILD) | Up to ~20 dB at high frequencies | Level difference per high-freq channel |
| Minimum Audible Angle (MAA) | ~1° azimuth (broadband, straight ahead) | Azimuth estimate from ITD/ILD with ~1° resolution at 0° |
| Head-Related Transfer Function | Individual-specific pinna filtering | Generic HRTF (CIPIC or MIT KEMAR dataset); encode elevation via spectral notch features |
| Cone of confusion resolution | Via head movement | Head rotation motor actions disambiguate front/back |

### 2.3 Cortical

| Parameter | Human value | Implementation target |
|-----------|-------------|----------------------|
| Spectrotemporal receptive field | ~1–2 octaves × 20–100 ms | Each SM covers ~1.25 octaves; temporal context ~30 ms (10 frames) |
| Spectral modulation preference | 0.5–8 cycles/octave | Compute spectral modulation features per SM |
| Temporal modulation preference | 4–32 Hz (cortex); up to 63 Hz (thalamus) | Amplitude modulation extraction at 4–64 Hz |
| Cortical column count (A1) | ~10,000–50,000 | 16 SMs × 13 LMs = 208 (sufficient for proof of concept) |
| Cross-frequency integration | Higher cortex integrates across bands | LM hierarchy: level-2 LMs combine frequency bands |

---

## 3. Sensor Module Count and Topology

### 3.1 How many SMs?

The number of SMs is determined by how we tile the cochleagram — the same
question as how many camera patches tile the visual field.

**Biological basis:** Each cortical column in primary auditory cortex (A1)
has a spectrotemporal receptive field (STRF) spanning ~1–2 octaves in frequency
and ~20–100 ms in time. The cochlea spans ~10 octaves (20 Hz to 20 kHz).
With ~1.25 octave bandwidth and 50% overlap, we get 8 frequency bands per ear.

```
Ear  SM  Center freq (Hz)  Band (Hz)         Octaves
───  ──  ────────────────  ──────────────    ────────
L    0   45                20 – 100          ~2.3
L    1   150               70 – 330          ~2.2
L    2   500               230 – 1,100       ~2.3
L    3   1,700             770 – 3,600       ~2.2
L    4   5,500             2,500 – 12,000    ~2.3
L    5   12,000            8,000 – 20,000    ~1.3
R    6–11  (mirror of 0–5)
```

Revised down to **6 SMs per ear, 12 total**. Each SM processes 10–12 gammatone
filter outputs from within its frequency band. Overlap between adjacent SMs
ensures no spectral gaps.

Additionally, **2 broadband SMs** (one per ear) receive the full 64-channel
cochleagram. These detect features that span the full spectrum (e.g., onset of a
broadband click, overall loudness envelope). These are analogous to the
low-resolution "surface SM" in vision that detects overall object coverage.

**Total: 14 SMs** (6 narrowband left + 6 narrowband right + 1 broadband left +
1 broadband right).

### 3.2 LM hierarchy

```
Layer 0 — Narrowband SMs:        SM_L0  SM_L1  SM_L2  SM_L3  SM_L4  SM_L5
                                  SM_R0  SM_R1  SM_R2  SM_R3  SM_R4  SM_R5
          Broadband SMs:          SM_LB  SM_RB

Layer 1 — Monaural LMs (6):      LM_0 (L-low)    ← SM_L0, SM_L1
          fan_in=2 per ear        LM_1 (L-mid)    ← SM_L2, SM_L3
                                  LM_2 (L-high)   ← SM_L4, SM_L5
                                  LM_3 (R-low)    ← SM_R0, SM_R1
                                  LM_4 (R-mid)    ← SM_R2, SM_R3
                                  LM_5 (R-high)   ← SM_R4, SM_R5

Layer 2 — Monaural full-band (2): LM_6 (L-full)  ← LM_0, LM_1, LM_2, SM_LB
                                   LM_7 (R-full)  ← LM_3, LM_4, LM_5, SM_RB

Layer 3 — Binaural (1):           LM_8           ← LM_6, LM_7

Layer 4 — HPC:                    LM_HPC         ← LM_8 + visual LM_14 + text LM_21
```

**Total: 14 SMs, 9 LMs** (+ HPC connection to existing cross-modal system).

### 3.3 What each layer learns

| Layer | What it represents | Auditory example | Visual analogue |
|-------|-------------------|------------------|-----------------|
| **L1: Monaural narrowband** | Spectrotemporal fragments within 2–3 octaves: single formants, harmonic groups, noise bursts | F2 formant of /æ/, attack transient of a snare drum | Oriented edges, local texture |
| **L2: Monaural full-band** | Complete spectral objects from one ear: vowel identity, instrument timbre, entire phonemes | The vowel /i/ (all formants), a piano note (fundamental + overtones) | Object parts (handle, rim) |
| **L3: Binaural** | Spatially-grounded auditory objects: what + where | "A voice saying 'hello' at 30° left" | Full 3D object with pose |
| **HPC** | Cross-modal binding: audio + visual + text | Seeing a dog and hearing it bark → same entity | Multi-sensory object identity |

### 3.4 Why this number, not more or fewer?

**Why not 1 SM per ear (broadband only)?**
Loses the core TBT property: each cortical column maintains its own
independent model. With 1 SM, you have 1 opinion per ear. With 6 narrowband
SMs, you have 6 independent frequency-band perspectives that vote. This is how
the biological auditory cortex works — tonotopically organized columns each
contribute evidence.

**Why not 64 SMs (one per gammatone channel)?**
Diminishing returns. Adjacent gammatone filters are highly correlated. An SM
covering 10–12 adjacent filters can extract the same spectrotemporal features
(spectral slope, bandwidth, center frequency, energy) that individual-filter
SMs would, but with less noise and lower computational cost. The biological
cortex has ~10,000 columns in A1, but each column's STRF spans many hair cells —
columns do not map 1:1 to hair cells.

**Why 6 per ear and not 4 or 8?**
6 gives ~1.5–2.3 octave bandwidth per SM, matching the ~1–2 octave STRF bandwidth
measured in primate A1 (Atiani et al. 2014). 4 would give ~2.5 octave bands
(too coarse to distinguish nearby formants). 8 would give ~1.25 octave bands
(viable but more complexity for marginal benefit at this stage).

---

## 4. The Geometry of Auditory Object Representations

### 4.1 The CMP State for audio

```python
State(
    location=np.array([
        log2_center_freq,   # log2(Hz): ~4.3 (20 Hz) to ~14.3 (20 kHz)
        compressed_dB,       # After OHC compression: ~0.0 to ~40.0
        azimuth_rad,         # Estimated source direction: -π to +π
    ]),
    morphological_features={
        "pose_vectors": np.array([
            spectral_gradient,     # (3,) unit vector: direction of steepest
                                   # spectral energy change in (freq, dB, azimuth)
            temporal_envelope_dir, # (3,) unit vector: direction of amplitude
                                   # modulation (rising/falling/steady)
            cross_product,         # (3,) completes the orthonormal frame
        ]),  # shape (3,3)
        "pose_fully_defined": is_tonal,
            # True: well-defined spectral peak (harmonic/tonal sound)
            # False: broadband noise, no clear spectral orientation
    },
    non_morphological_features={
        "energy":               float,  # Total energy in this SM's band
        "harmonicity":          float,  # 0.0 (noise) to 1.0 (pure tone)
        "spectral_centroid":    float,  # Weighted mean frequency within band
        "spectral_bandwidth":   float,  # Spread of energy around centroid
        "spectral_slope":       float,  # Tilt of spectral envelope
        "spectral_flux":        float,  # Frame-to-frame spectral change
        "onset_strength":       float,  # Transient detection strength
        "amplitude_modulation": float,  # Dominant AM rate (Hz)
        "pitch_estimate":       float,  # F0 if detectable, NaN otherwise
        "itd":                  float,  # Interaural time difference (seconds)
        "ild":                  float,  # Interaural level difference (dB)
    },
    confidence=...,     # Higher when SNR is good, on_object is clear
    use_state=...,      # False during silence or motor_only_step
    sender_id="SM_L3",  # Identifies which SM
    sender_type="SM",
)
```

### 4.2 What "on_object" means for audio

In vision, `on_object` means the center pixel falls on a non-background surface.
In audio, the equivalent is: **is there sufficient energy in this frequency band
to constitute a signal above the noise floor?**

```python
on_object = (energy_in_band > noise_floor_estimate + snr_threshold_dB)
```

This determines `use_state`. SMs in frequency bands with no signal energy
report `use_state=False`, just as CameraSM does when pointing at empty space.
The LM ignores these frames.

### 4.3 Displacement semantics

The LM computes displacements between consecutive `State.location` values:

```python
displacement = current_state.location - previous_state.location
# = [Δlog2_freq, Δcompressed_dB, Δazimuth]
```

**What each displacement component means:**

| Component | Physical meaning | Example |
|-----------|-----------------|---------|
| Δlog2_freq > 0 | Energy moved to higher frequency | Rising pitch, formant transition /æ/→/i/ |
| Δlog2_freq = 0 | Same frequency band | Sustained vowel, steady tone |
| ΔdB > 0 | Sound got louder | Onset, crescendo, approaching source |
| ΔdB < 0 | Sound got quieter | Offset, decrescendo, receding source |
| Δazimuth | Source moved | Head turn, or source physically moving |

An auditory object graph stores these displacements as edge attributes, exactly
as the visual graph stores spatial displacements. Recognition = finding a stored
graph whose displacement sequence matches the observed one, up to translation
in (log-freq, dB, azimuth).

### 4.4 The word "cat" as an evidence graph

```
Node 0: loc=(11.3, 28, 0.0)  feat={harmonicity=0.1, bandwidth=wide, onset=0.9}
   │                                                    [/k/ burst]
   │ edge: Δ=(-1.6, +2, 0.0), Δt=30ms
   ▼
Node 1: loc=(9.7, 30, 0.0)   feat={harmonicity=0.8, bandwidth=narrow, F0=150Hz}
   │                                                    [/æ/ vowel onset]
   │ edge: Δ=(0.0, -3, 0.0), Δt=80ms
   ▼
Node 2: loc=(9.7, 27, 0.0)   feat={harmonicity=0.8, bandwidth=narrow, F0=148Hz}
   │                                                    [/æ/ vowel steady-state decay]
   │ edge: Δ=(+2.2, -12, 0.0), Δt=40ms
   ▼
Node 3: loc=(11.9, 15, 0.0)  feat={harmonicity=0.0, bandwidth=wide, onset=0.7}
                                                        [/t/ release burst]
```

The same word spoken at a higher pitch shifts all `loc[0]` values up by some
constant — a translation in log-frequency space. The LM recognizes it via the
same displacement-matching mechanism used for visual pose invariance.

The same word spoken more loudly shifts all `loc[1]` values up — a translation in
compressed-dB space. After cochlear compression, this shift is approximately
uniform, making it handleable by the existing pose hypothesis mechanism.

The same word spoken more slowly scales all Δt values — the temporal edges stretch.
This is the one invariance that requires explicit handling, discussed in §5.

### 4.5 How intensity changes the graph shape

At low intensity (whisper, ~40 dB SPL):
- Spectral peaks are narrow (minimal upward spread)
- Only low-threshold auditory nerve fibers active
- High-frequency components may be below threshold

At high intensity (shout, ~90 dB SPL):
- Spectral peaks are broad (upward spread of masking)
- High-threshold fibers recruited, response nonlinear
- High-frequency components now audible

The cochlear compression model in the SM (§6.1) normalizes much of this.
Remaining distortion is captured by the `spectral_bandwidth` feature — a wide
bandwidth at the same center frequency indicates high intensity. The LM's
feature tolerances can match "narrow peak at 1 kHz" with "broad peak at 1 kHz"
if the `spectral_bandwidth` tolerance is set appropriately.

This is an imperfect invariance, biologically accurate: humans also struggle to
recognize whispered words compared to normally-spoken ones, and very loud sounds
in reverberant environments are harder to identify.

---

## 5. Temporal Directionality

### 5.1 The problem

In the existing `EvidenceGraphLM`, evidence is accumulated based on
displacement magnitude and feature similarity. A displacement of (Δx, Δy, Δz)
contributes the same evidence regardless of traversal order. This works for
spatial objects because you can walk around them in any direction.

For audio, temporal order matters. The sequence /k/→/æ/→/t/ ("cat") is not the
same as /t/→/æ/→/k/ (which is nothing). The time component of the displacement
must be direction-sensitive.

### 5.2 Proposed solution: signed temporal evidence

Modify the hypothesis updater to treat the time component of displacements
asymmetrically:

```python
# In the evidence update for a hypothesis:
expected_displacement = graph_edge.displacement  # (Δfreq, ΔdB, Δtime)
observed_displacement = current_loc - previous_loc

# Spatial components (freq, dB): standard symmetric matching
freq_match = within_tolerance(observed[0], expected[0], freq_tolerance)
db_match = within_tolerance(observed[1], expected[1], db_tolerance)

# Temporal component: direction-sensitive
time_match = (
    sign(observed[2]) == sign(expected[2])  # Must go same direction
    and within_tolerance(abs(observed[2]), abs(expected[2]), time_tolerance)
)

evidence_delta = freq_match * db_match * time_match * feature_similarity
```

This is a targeted change to the hypothesis updater, not a rewrite. The graph
data structure, the evidence accumulation framework, and the voting mechanism
are untouched.

### 5.3 Tempo invariance

The `time_tolerance` parameter provides some tempo flexibility — a 30ms gap
matching a 40ms expected gap within tolerance. For larger tempo variations (2x
speed), the LM would need to store the graph with normalized temporal
displacements (Δtime as fraction of total duration) or maintain multiple
time-scaled hypothesis tracks. This is a future extension, not required for the
initial implementation.

---

## 6. Processing Pipeline

### 6.1 Cochlear front-end

```
Stereo waveform (44.1 kHz, 2 channels, 24-bit)
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│  Gammatone Filterbank (per ear)                          │
│  64 filters, ERB-spaced from 20 Hz to 20 kHz            │
│  4th-order gammatone impulse response                    │
│  Reference: Patterson et al. 1992                        │
└──────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│  Half-Wave Rectification + Cochlear Compression          │
│  Models inner hair cell transduction:                    │
│    y = max(0, x)            (half-wave rectify)          │
│    y_c = sign(y) * |y|^0.3  (power-law compress)        │
│  Compresses ~120 dB input to ~40 dB internal range       │
│  Mimics OHC active gain + IHC saturation                 │
└──────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│  Temporal Framing                                        │
│  512-sample frames @ 44.1 kHz = 11.6 ms windows          │
│  128-sample hop = 2.9 ms resolution                       │
│  Per frame: 64-dim compressed cochleagram vector          │
└──────────────────────────────────────────────────────────┘
         │
    64-channel compressed cochleagram per ear per frame
```

**Why gammatone and not FFT/mel-spectrogram?**
FFT is a mathematical decomposition with no biological basis. Gammatone filters
model the basilar membrane's mechanical resonance. They produce asymmetric
filter shapes (steeper on the high-frequency side) matching measured cochlear
tuning curves. This matters because upward spread of masking is a geometrically
significant phenomenon (§1.5) that symmetric filters miss.

**Why power-law compression and not log?**
Log compression (as in dB) is infinite at zero — it maps silence to -∞. The
cochlea's compression is finite and approximately power-law with exponent
~0.2–0.4 at moderate levels (Glasberg & Moore 2006). Power-law compression
preserves the zero point (silence = 0) and is differentiable, making gradient
computation for pose_vectors clean.

### 6.2 Binaural processing

```
Left cochleagram (64 ch × T frames)     Right cochleagram (64 ch × T frames)
              │                                      │
              └──────────────┬───────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────┐
│  Per-Channel Cross-Correlation (channels 1–24, <1.5 kHz) │
│  ITD = argmax of normalized cross-correlation            │
│  Resolution: 1 sample = 22.7 μs → ~0.7° azimuth         │
│  Max ITD: ~700 μs = ~31 samples                          │
└──────────────────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│  Per-Channel Level Difference (channels 25–64, >1.5 kHz) │
│  ILD = 20 * log10(energy_L / energy_R) per channel       │
│  Azimuth estimation via lookup table (HRTF-derived)      │
└──────────────────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│  Azimuth Fusion                                          │
│  Weighted combination of ITD and ILD estimates            │
│  Low-freq channels contribute ITD; high-freq → ILD       │
│  Output: per-frame azimuth estimate + confidence          │
│  Ambiguity flag when front/back cannot be resolved        │
└──────────────────────────────────────────────────────────┘
```

### 6.3 Feature extraction (per SM)

Each narrowband SM receives 10–12 adjacent gammatone channels from both ears.
Per frame, it computes:

| Feature | Computation | Biological basis |
|---------|-------------|-----------------|
| `energy` | Sum of squared compressed amplitudes across channels | Overall firing rate in this tonotopic region |
| `spectral_centroid` | Energy-weighted mean frequency | Center of mass of activation on basilar membrane |
| `spectral_bandwidth` | Energy-weighted standard deviation of frequency | Spread of activation (changes with intensity, §1.5) |
| `spectral_slope` | Linear regression slope of energy vs. log-frequency | Spectral tilt — distinguishes vowels, fricatives |
| `spectral_flux` | L2 norm of frame-to-frame spectral difference | Rate of spectral change — onset/offset detection |
| `onset_strength` | Half-wave rectified spectral flux | Transient detection (stops, clicks, attacks) |
| `harmonicity` | Autocorrelation peak ratio | Periodic (voiced) vs. aperiodic (noise) |
| `pitch_estimate` | Autocorrelation peak position → F0 | Fundamental frequency (if harmonic) |
| `amplitude_modulation` | Peak of temporal modulation spectrum (4–64 Hz) | Syllabic rate, tremolo, roughness |
| `itd` | From binaural processor (for this frequency band) | Interaural time difference |
| `ild` | From binaural processor (for this frequency band) | Interaural level difference |

The broadband SMs compute the same features over all 64 channels.

### 6.4 State construction

```python
def step(self, ctx, observation, motor_only_step=False) -> State | None:
    cochleagram = observation["cochleagram"]  # shape (n_channels, n_frames)
    my_channels = cochleagram[self.channel_start:self.channel_end, :]

    energy = np.sum(my_channels[:, -1] ** 2)
    if energy < self.noise_floor:
        return State(
            location=self._last_location,
            morphological_features=self._null_morph,
            non_morphological_features={},
            confidence=0.0,
            use_state=False,
            sender_id=self.sensor_module_id,
            sender_type="SM",
        )

    features = self._extract_features(my_channels)
    location = np.array([
        features["spectral_centroid"],   # log2(Hz)
        features["energy"],              # compressed dB
        features["azimuth"],             # radians
    ])
    pose_vectors = self._compute_gradients(my_channels)

    return State(
        location=location,
        morphological_features={
            "pose_vectors": pose_vectors,
            "pose_fully_defined": features["harmonicity"] > 0.5,
        },
        non_morphological_features=features,
        confidence=min(1.0, energy / self.saturation_energy),
        use_state=not motor_only_step,
        sender_id=self.sensor_module_id,
        sender_type="SM",
    )
```

---

## 7. Motor System: Active Listening

### 7.1 Motor actions

| Action | Parameters | Effect on observation | Purpose |
|--------|-----------|----------------------|---------|
| `rotate_yaw` | angle (degrees) | ITD/ILD change for all sources | Improve azimuth estimate; resolve front/back ambiguity |
| `rotate_pitch` | angle (degrees) | HRTF pinna filtering changes | Estimate source elevation |
| `rotate_roll` | angle (degrees) | ITD axis rotation | Disambiguate complex scenes |
| `attend_frequency` | center_freq (Hz) | Shift processing focus (narrow bandwidth) | Frequency "saccade" — attend to specific band |

### 7.2 Motor policy: HeadRotationPolicy

The policy decides which motor action to take based on current auditory state.
It is analogous to the visual surface-following policy:

```python
class HeadRotationPolicy(MotorPolicy):
    def __call__(self, ctx, observations, state):
        # 1. If azimuth uncertainty is high (front/back ambiguity):
        #    → rotate yaw by 15° to change ITD/ILD and disambiguate
        # 2. If multiple sources detected (competing ITD peaks):
        #    → rotate toward strongest source to separate it
        # 3. If all azimuth estimates are confident:
        #    → small random yaw jitter (sensorimotor exploration)
        # 4. If elevation is unknown:
        #    → rotate pitch to induce HRTF changes
```

### 7.3 HRTF as the motor-observation coupling

In vision, moving the camera changes which surface points are visible. In audio,
rotating the head changes how each source is filtered by the HRTF. This is the
coupling between motor actions and sensory changes that the LM uses to build
spatial models.

The HRTF is modeled as a set of direction-dependent filters applied to the
stereo input before the gammatone filterbank:

```
Raw stereo audio → HRTF(head_pose) → Filtered stereo audio → Cochlear front-end
```

When the head rotates, the HRTF changes, and the SM output changes. The LM
observes this change as a displacement in (log-freq, dB, azimuth) space and
uses it to refine spatial hypotheses — exactly as the visual LM uses camera
motion to refine 3D pose hypotheses.

---

## 8. Key Architectural Decisions

### 8.1 Location space: spectrotemporal vs. Euclidean

**Decision:** Use (log-freq, compressed-dB, azimuth) as the 3D location space
for all auditory SMs and LMs, up to and including the binaural integration
layer. Only the HPC connection needs to translate between auditory and visual
coordinate systems.

**Rationale:** Forcing auditory SMs to output (x, y, z) Euclidean locations
would discard spectral information. The LM's evidence graph is indexed by
location — if two sounds at different frequencies map to the same spatial
location, the graph conflates them. Spectrotemporal coordinates keep the graph
structure clean.

The HPC already handles cross-modal binding by comparing abstract identity
(object ID, confidence) rather than raw locations. A voice at 30° left and a
face at 30° left bind because they share spatial direction and temporal
co-occurrence, not because their location vectors are numerically identical.

### 8.2 Pose vectors: what do they mean?

**Decision:** For audio, pose_vectors encode the local gradient of the
spectrotemporal field, not surface normals.

| pose_vectors[i] | Vision (CameraSM) | Audio (AudioSM) |
|-----------------|-------------------|------------------|
| [0] | Surface normal | Spectral gradient: direction of steepest energy change in (freq, dB, azimuth) space |
| [1] | Curvature direction 1 | Temporal envelope direction: rising/falling/steady |
| [2] | Curvature direction 2 | Cross product (completes frame) |

**Rationale:** pose_vectors serve two purposes in the LM:
1. Determining `pose_fully_defined` — whether the local geometry is distinctive
   enough to constrain pose hypotheses.
2. Rotating between reference frames via `transform_morphological_features()`.

For audio, a tonal sound has a well-defined spectral peak (like a surface with
clear curvature), so `pose_fully_defined=True`. Broadband noise has no preferred
direction (like a flat surface), so `pose_fully_defined=False`. This maps
naturally.

### 8.3 Single-pass vs. iterative evidence accumulation

**Decision:** Auditory evidence accumulation runs in a single forward pass per
utterance/sound event. No backtracking.

**Rationale:** Time is irreversible. The visual LM's ability to revisit surface
points is a luxury that audio does not have. However:
- The evidence graph can still accumulate evidence across *repetitions* of the
  same sound (hearing "cat" multiple times).
- The TemporalMemory provides prediction within an episode, flagging when the
  observed sequence deviates from expectation (surprise signal).
- The HPC stores cross-episode associations, enabling "I've heard this before."

This is biologically faithful. Humans learn new words through repetition, not
single-pass memorization.

---

## 9. Implementation Phases

### Phase 0: Minimum viable audio (1 SM, 1 LM, beep sequences)

See §0 above. No signal processing, no waveforms. Synthetic beep State
sequences exercise the full Monty pipeline.

**New files:** `environments/audio_behaviors.py`, `models/audio_sm.py` (minimal),
`tests/unit/audio/test_audio_beeps.py`

**Modified files:** None.

**Success criteria:** Distinguish 5 beep patterns; pitch transposition invariance;
temporal order discrimination (ascending vs descending); TemporalMemory
prediction and surprise.

### Phase 1: Real waveforms, single SM (mono, no motor)

**New files:**
- `models/cochlear_filterbank.py` — Gammatone filterbank + compression
- Extend `models/audio_sm.py` — process real waveform frames
- `environments/audio_interface.py` — `AudioSource` (WAV file or microphone)

**Scope:** Single-ear, single broadband SM processing mono audio through
gammatone filterbank. Still 1 SM → 1 LM. Train on isolated sounds (pure tones,
chords, simple instrument notes) and verify evidence graphs are meaningful.

**Success criterion:** Distinguish 5–10 isolated instrument notes or simple
sound effects from WAV files.

### Phase 2: Multiple SMs, narrowband hierarchy

**Scope:** Scale from 1 SM to 6 narrowband + 1 broadband per ear (14 SMs total).
Add the 3-layer monaural hierarchy. Still mono (no binaural).

**Success criterion:** Distinguish 5 spoken words from single-speaker recordings.

### Phase 3: Binaural processing

**New files:**
- `models/binaural_processor.py` — ITD/ILD computation, azimuth estimation
- Update `audio_sm.py` to add `itd`/`ild` features and azimuth to location[2]

**Scope:** Stereo input, binaural feature extraction, azimuth-aware States.

**Success criterion:** Separate two simultaneous talkers at different azimuths.

### Phase 4: Motor system

**New files:**
- `models/audio_motor_policy.py` — `HeadRotationPolicy`
- `models/hrtf_model.py` — HRTF filter application based on head pose

**Success criterion:** Active head rotation localizes a sound source with <5°
azimuth error.

### Phase 5: Temporal directionality in LM

**Modified files:**
- `models/evidence_matching/hypotheses_updater.py` — direction-sensitive
  matching for temporal displacement component

**Success criterion:** Reject time-reversed versions of learned words.

### Phase 6: Hierarchy and cross-modal

**Modified files:**
- `config_utils/world_model.py` — Add `build_audio_hierarchy()` and integrate
  with existing visual/text hierarchies
- `models/monty_base.py` — Ensure audio LMs participate in voting and HPC routing

**Scope:** Full 14-SM, 9-LM audio hierarchy connected to the existing world
model. Audio objects learned at the binaural level feed into HPC alongside visual
and text objects.

**Success criterion:** The system learns that a cat's meow and the visual
appearance of a cat belong to the same entity, using HPC cross-modal binding.

---

## 10. Audio-Visual Fusion: Multimodal Behaviors and Binding

The Thousand Brains Theory predicts that cross-modal binding uses the same
mechanism as within-modal binding — co-occurrence in the HPC. This section works
through how audio and video merge at each level, starting from Phase 0.

### 10.1 The binding problem, concretely

When you see a dog and hear it bark, you know the bark belongs to the dog.
Three things must happen:

1. **Temporal co-occurrence**: the bark starts when the dog's mouth opens (HPC
   fast binding — already implemented as `_update_fast_binding`)
2. **Spatial co-location**: the bark comes from the direction of the dog (HPC
   relational graph spatial edges — already implemented as
   `_update_relational_graph` with displacement vectors)
3. **Causal consistency**: the bark's onset aligns with the visual mouth-open
   onset (HPC temporal edges — already implemented as lag-1 transitions)

The HPC already handles all three for concepts from any upstream LM. The only
requirement is that audio and visual top-level LMs both feed into the same HPC
via `lm_to_lm_matrix`. This is the architecture already described in §3.2.

### 10.2 Where the current HPC is insufficient

**Problem: location spaces don't match.** The visual LM reports location as
(x, y, z) in Euclidean meters. The audio LM reports location as (log-freq,
compressed-dB, azimuth). The HPC's `_update_relational_graph` computes spatial
displacement as `locations[b] - locations[a]`. Subtracting a visual (x, y, z)
from an audio (log-freq, dB, azimuth) is meaningless.

**Solution: shared spatial dimension.** The one location dimension that both
modalities share is **azimuth** — the direction to the object/source. For visual
objects, azimuth can be derived from the camera direction. For audio, azimuth is
location[2]. The HPC spatial binding should operate on this shared dimension.

Two approaches:

**(a) Project both locations to shared spatial coords before HPC.**
Each top-level LM outputs a State where `location` is in world-spatial
coordinates. The visual LM already does this (x, y, z). The audio LM's binaural
integration layer (LM_8 in §3.2) would translate its best azimuth estimate into
a 3D direction vector: `(cos(az), 0, sin(az))` × estimated_distance. This
puts both modalities in the same coordinate system before reaching HPC.

**(b) HPC uses modality-aware displacement.**
The HPC stores displacement with a modality tag and only computes spatial
displacement between concepts in the same coordinate system, or converts
on the fly. More complex, less clean.

**Recommendation: approach (a).** The binaural LM's output State should use
world-spatial location. This is what the biological auditory system does — the
"where" pathway (dorsal stream) outputs spatial coordinates compatible with the
visual system's spatial map, enabling cross-modal binding in parietal cortex
and hippocampus.

### 10.3 Multimodal behaviors: audio + visual State sequences

The existing `behaviors.py` generators produce visual State sequences. Extending
them to multimodal means each behavior produces **two parallel State streams** —
one for the visual SM and one for the audio SM — synchronized by time step.

This directly mirrors reality: when a door opens, you simultaneously see it
swing (visual States) and hear it creak (audio States). Both streams arrive at
their respective SMs on the same time step.

#### Phase 0 multimodal behaviors (beeps + visual)

The simplest multimodal behaviors combine existing visual generators with beep
sequences:

```python
def doorbell_press(n_steps=40):
    """Finger pressing a doorbell button, which emits a two-tone chime.

    Visual stream: button depression (location sinks, curvature changes)
    Audio stream:  two-tone beep sequence (880 Hz → 660 Hz)

    Returns:
        (visual_states, audio_states) — parallel lists, same length
    """
    visual_states = []
    audio_states = []

    for t in range(n_steps):
        # --- Visual: button press and release ---
        if t < 10:
            depth = 0.0                          # button at rest
        elif t < 15:
            depth = -0.01 * (t - 10) / 5         # pressing in
        elif t < 20:
            depth = -0.01                         # held down
        else:
            depth = -0.01 * max(0, 1 - (t - 20) / 5)  # releasing

        location = np.array([0.0, 0.0, 0.1 + depth])
        button_curv = np.log1p(0.5 + abs(depth) * 20)

        visual_states.append(State(
            location=location,
            morphological_features={
                "pose_vectors": np.eye(3),
                "pose_fully_defined": True,
                "on_object": 1,
            },
            non_morphological_features={
                "hsv": [0.1, 0.8, 0.9],   # yellow button
                "principal_curvatures_log": [button_curv, np.log1p(0.05)],
            },
            confidence=1.0,
            use_state=True,
            sender_id="visual_sm_0",
            sender_type="SM",
        ))

        # --- Audio: chime triggered after button press ---
        if 15 <= t < 22:
            freq, amp = 880, 70      # first tone
        elif 24 <= t < 34:
            freq, amp = 660, 65      # second tone
        else:
            freq, amp = 0, 0         # silence

        is_active = freq > 0
        audio_states.append(State(
            location=np.array([
                np.log2(freq) if is_active else 0.0,
                amp / 120.0 if is_active else 0.0,
                0.0,
            ]),
            morphological_features={
                "pose_vectors": np.eye(3),
                "pose_fully_defined": False,
            },
            non_morphological_features={
                "frequency_hz": freq,
                "amplitude_db": amp,
            } if is_active else {},
            confidence=1.0 if is_active else 0.0,
            use_state=is_active,
            sender_id="audio_sm_0",
            sender_type="SM",
        ))

    return visual_states, audio_states
```

More Phase 0 multimodal patterns:

```python
def clapping_hands(n_steps=60, clap_period=10):
    """Hands clapping rhythmically.

    Visual: hands converge → contact → separate (location oscillates,
            curvature spikes at contact)
    Audio:  broadband click on each contact (short loud beep at ~2 kHz)

    The audio onset is precisely aligned with the visual contact frame.
    """

def bouncing_ball(n_steps=80, bounce_times=[15, 35, 50, 60]):
    """Ball bouncing on floor, losing height.

    Visual: ball follows parabolic arcs (location traces y-parabola,
            curvature constant = sphere)
    Audio:  impact thud at each bounce (low-freq beep ~100 Hz, amplitude
            decreasing with each bounce as energy is lost)

    Tests: audio amplitude correlating with visual impact velocity.
    """

def scissors_with_sound(n_steps=50, cut_period=15):
    """Scissors cutting — extends scissors_cutting() with audio.

    Visual: existing scissors_cutting() behavior
    Audio:  metallic snip sound at each blade-close (high-freq beep ~4 kHz,
            onset aligned with minimum blade angle)
    """

def alarm_with_flash(n_steps=60, period=8):
    """Emergency alarm — flashing light with siren.

    Visual: color alternates red/dark on each period (hsv hue flips)
    Audio:  alternating high/low beeps (900/600 Hz) synchronized with
            color change

    Tests: audio frequency change aligned with visual color change.
    The HPC should bind these into a single "alarm" concept.
    """
```

#### Key design principle: temporal alignment is the signal

The critical feature of multimodal behaviors is **precise temporal alignment**
between audio and visual events. When the ball hits the floor, the visual
contact frame and the audio onset frame are the **same time step**. This is
how the HPC detects that they're related:

1. At time step 15: visual LM activates "ball_contact" concept, audio LM
   activates "impact_thud" concept
2. HPC `_update_fast_binding` increments co-occurrence("ball_contact",
   "impact_thud")
3. After seeing this pattern 3–4 times across bounces, the association is
   strong enough to be included in `association_strengths` in the context signal
4. On next episode: hearing the thud alone primes the visual system to expect
   a ball (temporal prediction via `_current_predictions`)

This is exactly how human infants learn multimodal associations — through
temporal coincidence, not labeled training data.

### 10.4 Architecture for multimodal Phase 0

```
doorbell_press() → (visual_states, audio_states)
                         │              │
                   ┌─────┴─────┐  ┌────┴─────┐
                   │ Visual SM │  │ Audio SM  │
                   └─────┬─────┘  └────┬─────┘
                         │              │
                   ┌─────┴─────┐  ┌────┴─────┐
                   │ Visual LM │  │ Audio LM  │
                   │ + TM      │  │ + TM      │
                   └─────┬─────┘  └────┬─────┘
                         │              │
                         └──────┬───────┘
                          ┌─────┴─────┐
                          │    HPC    │
                          └───────────┘
```

**Connectivity:**
```python
sm_to_lm_matrix = [[0], [1], [], []]   # SM0→LM0, SM1→LM1
lm_to_lm_matrix = [[], [], [0, 1], [2]] # LM0,LM1→LM2(=HPC)... but HPC needs
                                         # no additional LM; it IS the top.
# Simplified for Phase 0:
sm_to_lm_matrix = [[0], [1], []]    # Visual SM→LM0, Audio SM→LM1, HPC gets none
lm_to_lm_matrix = [[], [], [0, 1]]  # HPC (LM2) ← Visual LM0 + Audio LM1
lm_to_lm_vote_matrix = [[1], [0], []]  # Visual and audio LMs can vote laterally
```

This is 2 SMs, 2 LMs, 1 HPC. The minimum multimodal system.

### 10.5 What the HPC learns from multimodal behaviors

After training on `doorbell_press()`:

**Co-occurrence associations:**
```
frozenset({"button_depression", "chime_880"}) → count: 7  (button + first tone)
frozenset({"button_depression", "chime_660"}) → count: 10 (button + second tone)
frozenset({"chime_880", "chime_660"})         → count: 10 (both tones co-active)
```

**Relational graph:**
```
button_depression → chime_880  [type=temporal, lag=1, count=5]
                               (button triggers first tone)
chime_880 → chime_660          [type=temporal, lag=1, count=5]
                               (first tone followed by second)
```

**Cross-episode temporal predictions:**
After pressing the doorbell in episode 1, on episode 2:
```
_current_predictions = {"chime_880": 0.7, "button_depression": 0.3}
```
Hearing the first chime tone primes the system to expect the second.
Seeing the button primes the system to expect the chime.

**This is cross-modal priming** — the exact phenomenon observed in human
perception. Seeing a dog primes the auditory system to expect a bark.
Hearing a doorbell primes the visual system to expect someone at the door.
It falls out of the existing HPC architecture with zero modifications.

### 10.6 What doesn't work yet: real-time synchronization

In Phase 0, the visual and audio State lists are pre-generated and indexed
by the same time step. This is synchronous by construction. In a real-time
system (Phase 3+), visual and audio streams arrive at different rates:

- Visual: ~30 fps (every ~33 ms)
- Audio: ~345 frames/sec (every ~2.9 ms)

The audio SM produces ~11 States for every 1 visual State. The Monty step loop
calls all SMs once per step. Two approaches:

**(a) Audio SM buffers and summarizes.**
The audio SM accumulates ~11 cochlear frames internally and emits one State per
Monty step, summarizing the ~33 ms of audio. Features like spectral centroid and
energy are averaged; onset events are flagged if any occurred within the window.
This matches the visual frame rate. Simple. Loses ~2.9 ms temporal resolution.

**(b) Monty steps at audio rate, visual SM repeats.**
The system steps at ~345 Hz. The visual SM re-emits its most recent State
(with `use_state=False` on repeat frames so the visual LM doesn't double-count).
The audio SM emits fresh States at full resolution. The audio LM gets fine
temporal detail; the visual LM only updates on real visual frames.

**Recommendation: (a) for Phase 0–2, (b) from Phase 3 onward.**

Phase 0 doesn't have this problem — both streams are synthetic and same-length.

### 10.7 Multimodal TemporalMemory

Each LM has its own TemporalMemory that learns within-modality sequences:
- Visual TM: learns "button goes down, then comes back up"
- Audio TM: learns "880 Hz tone, then 660 Hz tone"

But the interesting predictions are **cross-modal**: hearing the first chime
predicts the button releasing (visual), and seeing the button depress predicts
the chime (audio).

The HPC's temporal prediction mechanism (`_compute_temporal_predictions`)
already provides this. After learning "button_depression → chime_880" as a
temporal edge, the HPC predicts "chime_880" upon seeing "button_depression"
and broadcasts this in the context signal. The audio LM receives this as
hippocampal bias, boosting evidence for the expected chime.

For Phase 0, this works without modification. The HPC context signal
(`get_context_signal()` → `association_strengths` dict) is already dispatched
to all LMs via `_dispatch_context_signals()` in MontyBase.

### 10.8 Full multimodal behavior library (phased)

| Phase | Behavior | Visual | Audio | Tests |
|-------|----------|--------|-------|-------|
| **0** | `doorbell_press` | Button depression | 880→660 Hz beeps | Cross-modal co-occurrence binding |
| **0** | `clapping_hands` | Hand convergence/divergence | Click on contact | Audio onset = visual contact frame |
| **0** | `bouncing_ball` | Parabolic arcs | Low thud at each bounce | Amplitude tracks impact velocity |
| **0** | `alarm_with_flash` | Color alternation | Siren beeps | Synchronized frequency + color changes |
| **0** | `scissors_with_sound` | Blade angle oscillation | High click at close | Extends existing `scissors_cutting()` |
| **1** | `speech_face` | Mouth opening/closing (curvature) | Vowel formants (real waveform) | Lip movement predicts vowel onset |
| **2** | `piano_key` | Key depression (location shift) | Piano note (narrowband SMs) | Pitch varies with key position |
| **3** | `moving_speaker` | Person walking (location changes) | Voice azimuth tracks visual position | Spatial co-location binding |
| **4** | `dog_barking` | Dog mouth (visual object model) | Bark (head-rotation motor resolves direction) | Full cross-modal object identity |

### 10.9 The invariances that multimodal binding provides

Single-modality recognition is fragile:
- Visually, a dog and a wolf are hard to distinguish
- Auditorily, a bark and a cough sound similar at a distance

Cross-modal binding provides **complementary invariances**:
- A barking dog is unambiguous (bark + dog shape = dog, not wolf)
- A coughing person is unambiguous (cough + human shape = person, not dog)

The HPC's co-occurrence mechanism captures this naturally. After experiencing
"bark always co-occurs with dog_shape" and "cough always co-occurs with
human_shape", the association matrix encodes these as strong bindings.

More subtly, **temporal structure** differs:
- A dog barks in bursts (2–3 barks, pause, repeat)
- A person coughs with different rhythm

The TemporalMemory in each modality's LM captures these rhythmic patterns.
The HPC binds "burst-pattern audio sequence" with "dog visual concept" and
"irregular-pattern audio sequence" with "human visual concept." This is
**temporal structure as a binding cue** — not just "these co-occur" but
"these co-occur with this specific temporal relationship."

### 10.10 Plumbing: how multimodal behaviors feed through MontyBase

The existing Monty step loop works like this (from `monty_base.py`):

```
env_interface.step(actions) → observations: Dict[AgentID, Dict[SensorID, SensorObservation]]
                                    │
                     ┌──────────────┴──────────────┐
                     ▼                             ▼
        observations["agent_0"]["visual_sm_0"]   observations["agent_0"]["audio_sm_0"]
                     │                             │
              visual_sm.step(obs)            audio_sm.step(obs)
                     │                             │
                     ▼                             ▼
           sensor_module_outputs[0]       sensor_module_outputs[1]
                     │                             │
        (sm_to_lm_matrix[0]=[0])      (sm_to_lm_matrix[1]=[1])
                     │                             │
                     ▼                             ▼
              visual_lm.matching_step()    audio_lm.matching_step()
                     │                             │
        (lm_to_lm_matrix[2]=[0,1])                │
                     └──────────┬──────────────────┘
                                ▼
                      hpc.matching_step([visual_state, audio_state])
```

Each SM's `step()` method receives its own `SensorObservation` dict. For Phase 0,
the visual SM receives `{"rgba": ..., "depth": ..., "semantic_3d": ...}` (or
whatever a minimal visual observation needs), and the audio SM receives
`{"audio_beep": {"frequency_hz": 880, "amplitude_db": 70}}`.

**The existing loop already handles multiple SMs of different types.** The only
new piece is an environment interface that serves both visual and audio
observations on each step.

#### BehaviorReplayEnvironment

For Phase 0, we need a minimal environment that replays pre-generated behavior
State sequences. This bypasses Habitat entirely — no rendering, no physics, just
playback of the (visual_states, audio_states) tuples from the behavior generators.

```python
class BehaviorReplayEnvironment:
    """Replays pre-generated multimodal behavior sequences as observations.

    Wraps a list of (visual_states, audio_states) pairs and serves them
    as Observations dicts keyed by SensorID. Each step() call advances
    one time step through the sequence.

    For audio-only Phase 0: visual_states can be None.
    For multimodal Phase 0: both streams are present.
    """

    def __init__(self, behavior_fn, agent_id="agent_0",
                 visual_sm_id="visual_sm_0", audio_sm_id="audio_sm_0"):
        visual_states, audio_states = behavior_fn()
        self._visual_states = visual_states
        self._audio_states = audio_states
        self._step = 0
        self._agent_id = agent_id
        self._visual_sm_id = visual_sm_id
        self._audio_sm_id = audio_sm_id

    def step(self, actions, first=False):
        """Return observations for current time step.

        The AudioSM doesn't need raw waveforms in Phase 0 — the behavior
        generator already produced CMP States. But MontyBase expects
        SensorObservation dicts, not States. So we wrap the State's fields
        into a SensorObservation that the AudioSM can unpack.
        """
        t = min(self._step, len(self._audio_states) - 1)
        obs = {self._agent_id: {}}

        if self._visual_states is not None:
            vs = self._visual_states[t]
            obs[self._agent_id][self._visual_sm_id] = {
                "location": vs.location,
                "morphological_features": vs.morphological_features,
                "non_morphological_features": vs.non_morphological_features,
                "confidence": vs.confidence,
                "use_state": vs.use_state,
            }

        ast = self._audio_states[t]
        obs[self._agent_id][self._audio_sm_id] = {
            "location": ast.location,
            "morphological_features": ast.morphological_features,
            "non_morphological_features": ast.non_morphological_features,
            "confidence": ast.confidence,
            "use_state": ast.use_state,
        }

        self._step += 1
        is_done = self._step >= len(self._audio_states)
        return obs, is_done
```

The AudioSM and VisualSM in Phase 0 are thin pass-throughs: they receive the
pre-computed fields from the SensorObservation dict and construct a State. This
tests the full MontyBase pipeline without requiring signal processing.

#### Alternative: direct TemporalMemory testing (even simpler)

For the very first tests, we can skip MontyBase entirely and test the components
directly, same as the existing `behaviors.py` tests:

```python
# Direct TM test — no MontyBase, no experiment loop
visual_states, audio_states = doorbell_press()
visual_tm = TemporalMemory()
audio_tm = TemporalMemory()
hpc = HippocampalModule()

for t in range(len(visual_states)):
    # Feed both TMs
    if visual_states[t].use_state:
        visual_tm.step(visual_states[t])
    if audio_states[t].use_state:
        audio_tm.step(audio_states[t])

    # Feed HPC with "mock" upstream LM outputs
    # (In Phase 0, treat the TM surprise/prediction as the LM's output)
```

This is the fastest path to validating that temporal co-occurrence works.

### 10.11 Cross-modal voting: what does it mean?

The `lm_to_lm_vote_matrix` enables lateral voting between LMs. In the visual
system, two LMs looking at the same object from different patches vote on object
identity — "I think it's a mug" / "I agree, it's a mug." This consensus
mechanism is core to the Thousand Brains Theory.

**Can a visual LM and an audio LM vote with each other?**

In principle, yes — if they're both trying to identify the same entity. But
there's an asymmetry: the visual LM's object graph lives in Euclidean
(x, y, z) location space, while the audio LM's graph lives in (log-freq, dB,
azimuth) space. A vote is a set of hypotheses: "I think this is object X at
pose Y." The pose makes no sense across modalities — a visual rotation has
no audio equivalent.

**What cross-modal voting can do:**

Object identity votes work. If the visual LM votes "I think this is a dog"
and the audio LM votes "I think this is a bark-source," the HPC can bind these
because the object IDs have been previously associated. The vote doesn't need
spatial compatibility — it's an identity vote.

**What it can't do (yet):**

Spatial votes. The visual LM can't tell the audio LM "the sound source is at
(1.2, 0.3, 0.5) meters" because the audio LM doesn't use that coordinate
system. This requires the shared-spatial-coordinate projection described in
§10.2.

**Phase 0 approach:** Enable identity-only lateral voting between audio and
visual LMs via lm_to_lm_vote_matrix. Spatial voting deferred to Phase 3.

### 10.12 The full Phase 0 multimodal test matrix

| # | Test | Setup | Assert |
|---|------|-------|--------|
| 1 | Audio-only beep recognition | Train LM on DOORBELL, re-present | Evidence converges to doorbell |
| 2 | Pitch transposition | Train DOORBELL at 880/660, test at 1760/1320 | Still recognized (log-freq displacement unchanged) |
| 3 | Amplitude invariance | Train at 70/65 dB, test at 50/45 dB | Still recognized (dB displacement unchanged) |
| 4 | Temporal order | Train THREE_UP, present THREE_DOWN | Not recognized (opposite displacement signs) |
| 5 | TM prediction | Train TM on ALARM (repeating), check prediction | After 2 beeps, TM predicts 3rd with low surprise |
| 6 | TM surprise | Train TM on DOORBELL, present SIREN | High surprise on 2nd beep (expected 660, got 900) |
| 7 | Multimodal co-occurrence | Train HPC on doorbell_press() (visual + audio) | co_occurrence_counts includes {button, chime} |
| 8 | Temporal binding | Train HPC on doorbell_press() | relational_graph has temporal edge button→chime |
| 9 | Cross-modal priming | After test 7, present visual button only | HPC context_signal includes chime prediction |
| 10 | Cross-modal priming (reverse) | After test 7, present audio chime only | HPC context_signal includes button prediction |
| 11 | Multimodal TM | Train visual TM + audio TM on doorbell_press() | Visual TM predicts button release; audio TM predicts 660 Hz |
| 12 | Dissociation | Train on doorbell + alarm_with_flash, test alarm audio only | HPC predicts flash, not button (different binding) |

Tests 1–6 validate audio-only operation. Tests 7–12 validate multimodal binding.
All use synthetic beep sequences + simple visual States. No signal processing,
no waveforms, no Habitat.

### 10.13 Implementation for Phase 0 multimodal

**Changes to Phase 0 plan (§0.9):**

| File | Additional contents |
|------|-------------------|
| `environments/audio_behaviors.py` | Add multimodal generators: `doorbell_press()`, `clapping_hands()`, `bouncing_ball()`, `alarm_with_flash()`, `scissors_with_sound()`. Each returns `(visual_states, audio_states)`. Also add `BehaviorReplayEnvironment` for full pipeline testing. |
| `tests/unit/audio/test_audio_beeps.py` | Tests 1–6: audio-only beep recognition, invariances, temporal order, TM |
| `tests/unit/audio/test_multimodal_beeps.py` | Tests 7–12: HPC binding, cross-modal priming, dissociation |

**No modifications to existing files.** The existing HPC, MontyBase dispatch, and
EvidenceGraphLM handle multimodal input without modification — the only new
code is the behavior generators, thin SM pass-throughs, and tests.

This is the acid test for the Thousand Brains Theory's strongest claim:
**one algorithm, any modality, cross-modal binding for free.**

---

## 11. Risks and Open Questions

### 11.1 Computational cost

64-channel gammatone filterbank at 44.1 kHz × 2 ears × 345 frames/sec = ~44,160
filter operations per second. Each of 14 SMs extracts ~10 features per frame.
Each of 9 LMs runs evidence matching. This is within real-time budget on modern
hardware but will need profiling.

### 11.2 Silence handling

Audio has long stretches of silence (unlike vision, where the object is always
visible). The SM should emit `use_state=False` during silence. The LM needs to
handle "no input for N steps" without forgetting the current hypothesis. The
existing motor_only_step mechanism may suffice.

### 11.3 What constitutes an "episode"?

In vision, an episode = exploration of one object. In audio, the natural episode
boundary is less clear. Options:
- One utterance (onset-to-offset as detected by energy threshold)
- One "auditory scene" (everything between silence gaps > 500ms)
- Continuous (no episode boundaries; the LM maintains running hypotheses)

This needs empirical investigation. Start with onset-to-offset segmentation.

### 11.4 Tempo invariance

The initial implementation handles small tempo variations via tolerance in
temporal displacement matching. Large tempo variations (2x speed) require
explicit time-warping or normalized temporal representations. This is a known
hard problem (dynamic time warping) and can be deferred.

### 11.5 The pose_vectors interpretation

Using spectral gradients as pose_vectors is a novel interpretation. It's unclear
whether the existing `transform_morphological_features()` (which applies rotation
matrices) will produce meaningful results when the "rotation" is in
spectrotemporal space. This needs empirical validation. If it fails, we may need
an audio-specific LM variant that handles morphological features differently.

### 11.6 Phase 0 as a diagnostic

Phase 0 (beep sequences) is deliberately designed to isolate the question:
"does the existing LM algorithm work for sequential audio-like input without
modification?" Every subsequent risk (cochlear fidelity, binaural accuracy, HRTF
realism) is irrelevant if the basic pipeline can't handle a sequence of
`(log-freq, dB)` States. Phase 0 answers this in days, not weeks. If it fails,
the failure mode tells us exactly what needs changing before investing in signal
processing infrastructure.

### 11.7 Multimodal step rate mismatch

Audio and visual streams have fundamentally different temporal resolution:
visual ~30 Hz, audio ~345 Hz. In Phase 0 this is a non-issue (both are
synthetic, same length). From Phase 1 onward, we need a strategy:

- **(a) Audio SM buffers internally**, emits one State per Monty step
  (~33 ms worth of audio summarized). Loses fine temporal detail. Simple.
- **(b) Monty steps at audio rate**, visual SM re-emits with
  `use_state=False` on non-keyframes. Preserves audio resolution. Heavier.
- **(c) Decoupled stepping**, where each SM steps at its own rate and the
  LM accumulates asynchronously. Requires architectural change to MontyBase
  (currently assumes all SMs step together).

Start with (a). Graduate to (b) if temporal resolution proves important for
recognition. (c) is a deeper change — defer unless empirically necessary.

### 11.8 Cross-modal concept identity

The HPC binds concepts by string ID (e.g., `"doorbell"`). For cross-modal
binding to work, the visual LM and audio LM must both produce concept IDs
that the HPC can match. In Phase 0 this is trivial — we assign IDs manually
in the behavior generators. In later phases, the LMs learn their own object
graphs with auto-generated IDs (like `"graph_0001"`). These won't match
across modalities.

Solutions:
- **HPC learns the mapping.** After enough co-occurrences of visual
  `"graph_0001"` and audio `"graph_0042"`, the HPC's association matrix
  encodes them as equivalent. This is the default — it works, but requires
  repeated exposure.
- **Shared naming via HPC broadcast.** Once the HPC binds two IDs, it
  could broadcast "graph_0001 = graph_0042" so both LMs use the same label.
  This is a future enhancement (not needed for Phase 0).

### 11.9 What if the LM can't handle forward-only evidence accumulation?

The `EvidenceGraphLM` was designed for visual exploration where the sensor
revisits surface points. In audio, the sensor never revisits a time step. If
the LM's evidence accumulation requires re-encountering previously seen nodes
to converge (e.g., to break symmetries), then audio recognition will fail or
require many repetitions.

Phase 0 test case: present a 3-beep pattern once. Does the LM accumulate
enough evidence to recognize it on second presentation? If it needs 10+
presentations, the forward-only constraint is too severe and we need either:
- A specialized `SequentialEvidenceGraphLM` that scores based on complete
  sequence match rather than iterative point-by-point refinement
- Heavier use of TemporalMemory as the primary sequence matcher, with the
  LM handling only aggregate identity

This is the most important architectural question Phase 0 answers.

---

## 12. Literature References

| Topic | Reference |
|-------|-----------|
| Gammatone filterbank | Patterson, Robinson, Holdsworth, McKeown, Zhang & Allerhand (1992). Complex sounds and auditory images. *Auditory Physiology and Perception*. |
| Cochlear compression | Glasberg & Moore (2006). Prediction of absolute thresholds and equal-loudness contours using a modified loudness model. *JASA*. |
| CARFAC cochlear model | Lyon, R.F. (2017). *Human and Machine Hearing*. Cambridge University Press. |
| ERB scale | Glasberg & Moore (1990). Derivation of auditory filter shapes from notched-noise data. *Hearing Research*. |
| Thousand Brains Theory | Hawkins, Lewis, Klukas, Purdy & Ahmad (2019). A Framework for Intelligence and Cortical Function Based on Grid Cells in the Neocortex. *Frontiers in Neural Circuits*. |
| TBP Monty paper | Clay, Hawkins et al. (2024). The Thousand Brains Project: A New Paradigm for Sensorimotor Intelligence. *arXiv:2412.18354*. |
| Cortical column universality | Mountcastle, V.B. (1978). An organizing principle for cerebral function. *The Mindful Brain*. |
| SDR encoding | Purdy, S. (2016). Encoding Data for HTM Systems. *arXiv:1602.05925*. |
| STRFs in auditory cortex | Theunissen & Elie (2014). Neural processing of natural sounds. *Nature Reviews Neuroscience*. |
| STRF bandwidth | Atiani, David, Elgueda, Locastro, Radtke-Schuller, Shamma & Fritz (2014). Emergent selectivity for task-relevant stimuli in higher-order auditory cortex. *Neuron*. |
| DNN-auditory hierarchy | Millet, Caucheteux, Orhan, Boubenec, Gramfort, Dunbar, Pallier & King (2023). Toward a realistic model of speech processing in the brain with self-supervised learning. *Nature Neuroscience*. |
| ICNet subcortical model | Saddler & McDermott (2025). ICNet. *Nature Machine Intelligence*. |
| Laminar cortical model | Pinotsis & Miller (2023). *NeuroImage*. |
| Sound localization precision | Makous & Middlebrooks (1990). Two-dimensional sound localization by human listeners. *JASA*. |
| Upward spread of masking | Moore, B.C.J. (2013). *An Introduction to the Psychology of Hearing*. 6th ed. Brill. |
| Equal-loudness contours | Fletcher & Munson (1933). Loudness, its definition, measurement, and calculation. *JASA*. ISO 226:2003 (revision). |
| Cross-modal binding | Stein & Meredith (1993). *The Merging of the Senses*. MIT Press. |
| Temporal coincidence in cross-modal learning | Bahrick & Lickliter (2000). Intersensory redundancy guides attentional selectivity and perceptual learning in infancy. *Developmental Psychology*. |
| Multisensory integration in cortex | Ghazanfar & Schroeder (2006). Is neocortex essentially multisensory? *Trends in Cognitive Sciences*. |
| Audio-visual speech (McGurk effect) | McGurk & MacDonald (1976). Hearing lips and seeing voices. *Nature*. |
