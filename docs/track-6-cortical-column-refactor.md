# Track 6 Refactor: Biologically Plausible Cortical Columns

## Problem

Track 6 implemented 13 milestones of predictive coding features, all bolted onto
a dense evidence-accumulation architecture (EvidenceGraphLM). The result:

- **Two incompatible representation regimes**: TemporalMemory uses SDRs; EvidenceGraphLM
  uses dense float feature grids + KDTree matching. Bridging them requires squeezing
  rich SDR prediction errors into a single scalar "surprise" value.
- **13 separate parameters** (surprise_learning_boost, low_surprise_decay_rate,
  surprise_gated_output, etc.) that approximate what a real cortical column does
  with ONE mechanism: predicted vs burst firing.
- **Dense storage**: 50^3 x 16 float32 feature grid per object (~200MB). Doesn't scale.
- **Phase 3 blocked**: Top-down prediction requires compositional models, which failed.
  The dense architecture makes compositional hierarchy fundamentally difficult.
- **V1 eval results**: 33% accuracy at 3 objects — the dense approach already struggles.

## What a Cortical Column Actually Does

A cortical column has ~100,000 neurons in 6 layers. The core computation:

### 1. Minicolumn Activation (Feedforward: "What Am I Sensing?")

Each minicolumn (~100 neurons tall) has feedforward connections from thalamus.
When the input matches a minicolumn's receptive field, that minicolumn activates.
~2% of minicolumns are active at any time (sparse distributed representation).

This is the **spatial pooling** step: raw input -> active minicolumns.

### 2. Cell Activation Within Minicolumns (Context: "Was This Predicted?")

Each minicolumn has multiple cells (biologically ~100, practically 8-32).
Each cell has ~100 **basal dendritic segments**, each independently detecting
a pattern of ~20 previously active cells.

- If ANY dendritic segment on a cell is active -> the cell is **predicted**
- When a minicolumn activates:
  - If predicted cells exist -> only predicted cells fire (**sparse, confident**)
  - If NO predicted cells -> ALL cells burst (**dense, surprised**)

This IS the prediction error signal. It's not a scalar — it's a rich,
high-dimensional pattern of which minicolumns burst and which fire predictively.

### 3. Learning (Three-Factor Hebbian)

Learning happens at dendritic segments:
- **Burst firing** (unpredicted): Grow new dendritic segments connecting the
  bursting cell to the previously active cells. This learns "in this context,
  predict THIS cell." Learning rate is HIGH.
- **Predicted firing**: Strengthen the segment that correctly predicted.
  Learning rate is LOW (maintenance).
- **No firing**: Slightly weaken connections (decay). Prevents saturation.

This single mechanism replaces T6.1 (surprise boost), T6.2 (decay),
T6.4 (combined modulation), and T6.6 (surprise-gated output).

### 4. Object Recognition (Location + Feature Matching)

The Thousand Brains Theory adds **grid cells** for location:
- Each column maintains a location representation in the object's reference frame
- The combined representation is: active_minicolumns (features) x active_cells (location + context)
- Object memory stores these combined patterns
- Recognition = the pattern of active cells matches a stored object pattern

### 5. The Hierarchy (Predictive Coding)

In a hierarchy:
- **Feedforward** (L2/3 -> L4 of parent): Send active cell pattern upward
- **Feedback** (L6 of parent -> L1 of child): Send predictions downward
- Child's **apical dendrites** receive top-down predictions
- If top-down matches bottom-up: smooth predicted firing
- If mismatch: burst -> error propagates upward

This replaces T6.5 (unified voting), T6.9-T6.12 (top-down prediction),
and Phase 3 entirely — it's not a bolt-on, it's the architecture.

## Architecture

### Encoding Layer

```
State (location, hsv, curvatures, pose)
    |
    v
GridCellEncoder(location) -> location_SDR (2048 bits, ~64 active)
ScalarEncoder(hsv)        -> feature_SDR  (768 bits, ~30 active)
ScalarEncoder(curvatures) -> curvature_SDR (256 bits, ~10 active)
    |
    v
Union -> input_SDR (3072 bits, ~104 active, ~3.4% sparsity)
```

**GridCellEncoder**: Multi-scale periodic encoding (like real grid cells).
Each module at scale s maps location to a phase pattern. Different scales
give coarse-to-fine spatial resolution. 8 modules x 256 cells = 2048 bits.

**ScalarEncoder**: Continuous value -> contiguous active bits in a circular
buffer. Adjacent values share active bits (smooth similarity). Standard
HTM encoder.

### Minicolumn Layer

```
input_SDR (3072 bits)
    |
    v
Spatial Pooling -> active_minicolumns (~60 out of 2048)
    |
    v
Dendritic Prediction -> predicted_cells (subset of cells in active minicolumns)
    |
    v
Cell Activation:
  - Predicted minicolumns: only predicted cells fire
  - Unpredicted minicolumns: ALL cells burst
    |
    v
active_cells SDR (variable sparsity: sparse if predicted, dense if surprised)
```

- **n_minicolumns**: 2048
- **n_cells_per_minicolumn**: 8
- **Total cells**: 16,384
- **Active minicolumns per step**: ~60 (3% of 2048)
- **Active cells**: 60-480 (60 if all predicted, 480 if all burst)

### Dendritic Segments

Each cell has S=32 dendritic segments. Each segment:
- Connects to C=24 other cells (sparse random connectivity)
- Has a permanence value per connection (float, [0, 1])
- Is "active" when >= T=12 connected cells are currently active
- Grows new connections to active cells when the cell bursts

Stored as a sparse data structure with a reverse index for efficient
prediction computation.

### Object Memory (SDR-Based)

Replaces GridObjectModel's dense feature grid:

```python
# Training: store SDR snapshots
object_memory["banana"] = {
    (location_sdr_1, active_cells_1),
    (location_sdr_2, active_cells_2),
    ...  # one per training observation
}

# Matching: compute overlap
for obj_name, patterns in object_memory.items():
    overlap = max(sdr_overlap(current_cells, stored_cells)
                  for stored_cells in patterns
                  if sdr_overlap(current_location, stored_loc) > threshold)
    evidence[obj_name] += overlap
```

Memory per object: N observations x ~500 bytes = ~50KB (vs ~200MB for dense grid).

### Three-Factor Learning

```
delta_W = learning_rate * pre * post * modulator

where:
  pre     = was the presynaptic cell active at t-1?
  post    = is the postsynaptic cell active at t?
  modulator:
    - burst_learning_rate (default 0.1) if cell burst (unpredicted)
    - predicted_learning_rate (default 0.01) if cell was predicted
    - decay_rate (default 0.001) for inactive connections
```

## How Track 6 Features Map to This Architecture

| Track 6 Feature | What It Did | New Architecture Equivalent |
|---|---|---|
| T6.1 surprise_learning_boost | Scalar surprise modulates TM learning rate | Burst firing = high learning, predicted = low learning (intrinsic) |
| T6.2 low_surprise_decay | Decay weights on well-predicted transitions | Inactive segment connections decay (intrinsic) |
| T6.3 HPC surprise modulation | Same for cross-episode learning | Same mechanism at HPC level |
| T6.4 Combined modulation | Enable T6.1+T6.2 together | Single mechanism, no parameter needed |
| T6.5 Unified voting | Confusion = high TM surprise OR high cardinality | Burst pattern IS the confusion signal |
| T6.6 Surprise-gated output | Low surprise -> minimal output | Predicted firing = small signal; burst = full signal |
| T6.6a Context biasing | HPC weights burst sampling candidates | Apical dendrites bias cell predictions |
| T6.7 KDTree workers | Performance optimization | Not needed (SDR overlap is O(k)) |
| T6.8 ThreadPool | Performance optimization | Not needed (no per-object threads) |
| T6.9-T6.12 Top-down prediction | BLOCKED | Apical dendrites (L1 feedback) — natural |
| T6.13 Burst ranking | Log ranking quality | Dendritic segment matching IS ranking |
| T6.13a Offspring hypotheses | Jitter high-evidence hypotheses | Not needed (no continuous hypotheses) |

**Key insight**: 13 parameters and features collapse into ONE mechanism with
3 parameters (burst_lr, predicted_lr, decay_rate).

## Implementation Plan

### Files

```
src/tbp/monty/frameworks/models/cortical_column/
  __init__.py
  encoders.py      # GridCellEncoder, ScalarEncoder
  dendrites.py     # DendriteSegments
  column.py        # CorticalColumn (unified module)
  sdr_memory.py    # SDRObjectMemory
```

### Interface Compatibility

CorticalColumn implements the same abstract interface as EvidenceGraphLM:
- `pre_episode(primary_target)` — reset state, set mode
- `exploratory_step(ctx, observations)` — training step
- `matching_step(ctx, observations)` — evaluation step
- `post_episode()` — save learned object
- `get_current_mlh()` — most likely hypothesis
- `get_all_known_object_ids()` — list known objects

This allows direct comparison via the V1 eval harness.

## Expected Outcomes

### What Should Improve

1. **Memory**: ~50KB per object vs ~200MB (4000x reduction)
2. **Temporal integration**: Prediction is intrinsic, not bolted on
3. **Surprise signal**: High-dimensional (which minicolumns burst), not scalar
4. **Learning speed**: One-pass Hebbian, no separate training phase needed
5. **Scalability**: O(k) per step where k=active bits, not O(grid_size)
6. **Phase 3 unblocked**: Top-down prediction is natural via apical dendrites

### What Might Be Worse Initially

1. **Spatial precision**: Grid cell encoding has finite resolution vs continuous float
2. **Feature discrimination**: Fixed encoding vs learned feature weights
3. **Rotation handling**: No continuous pose hypotheses to refine
4. **Benchmark numbers**: May need tuning to match current accuracy

### Validation

Run both architectures through V1 eval harness:
- Same YCB objects, same rotations, same observation pipeline
- Compare: accuracy, convergence speed, memory usage, wall clock
- If cortical column accuracy < evidence-graph accuracy with same data:
  diagnose whether the bottleneck is encoding quality or matching algorithm
