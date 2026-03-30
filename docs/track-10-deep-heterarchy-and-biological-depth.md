# Track 10: Intra-Column Biological Depth

## Goal

Add biological depth **within** a single `CorticalColumnTorch` (Track 9):
laminar structure (L1–L6), inhibitory interneuron diversity, multi-head
dendritic attention, thalamocortical loops, plateau-potential working
memory, spike-timing-dependent plasticity, and eligibility traces.

After this track, each column is a biologically realistic cortical column
with distinct laminar computation, not a flat bag of identical cells.
Inter-column scaling and deep hierarchy are deferred to Track 11.

## Depends On

Track 9 (Modern Hopfield CorticalColumn). All Track 9 phases must be
complete before Track 10 begins.

## Problem

After Track 9, a single `CorticalColumnTorch` is equivalent to **one
transformer attention head** — powerful retrieval, but no multi-head
attention, no context window, and no biological laminar structure. All
cells are identical and there is no distinction between input, recurrent,
output, and feedback processing. Real cortical columns have 6 layers with
distinct cell types, connectivity, and computational roles.

---

Everything in this track happens inside a single `CorticalColumnTorch`
instance without changing the inter-column architecture.

### Relationship to Hopfield Dynamics

Track 9's modern Hopfield settling (`x_new = Ξᵀ softmax(β · Ξ · x)`)
lives in **L2/3** — the representation layer. The recurrent weight matrix
`W ∝ ΞᵀΞ` is **symmetric by construction**, which guarantees a
well-defined energy function and convergent attractor dynamics.

Every feature in this track must preserve that guarantee. Here is how
each section relates to Hopfield:

| Section | Relationship | Where it acts |
|---|---|---|
| A1 Laminar | Partitions the column; Hopfield settling is confined to L2/3 | Framework around Hopfield |
| A2 Multi-head dendrites | Biases which cells *enter* Hopfield settling (depolarization/prediction) | Input to Hopfield |
| A3 Plateau potentials | Enriches the Hopfield *query*: `query = x + γ·plateau` | Modifies Hopfield query |
| A4 PV+ interneurons | Top-k sparsity enforcement = the sparsity step after each Hopfield iteration | Part of Hopfield |
| A4 SST+/VIP+ | Gate dendritic branches (prediction), not Hopfield recurrence | Orthogonal to Hopfield |
| A5 STDP | Applies to **dendritic** connections only, NOT L2/3 recurrent weights | Does not touch Hopfield |
| A6 Eligibility traces | Gates dendritic STDP updates; Hopfield storage unchanged | Does not touch Hopfield |
| A7 Thalamocortical | Gates what input reaches L4 → L2/3 | Input to Hopfield |
| A8 Oscillatory coding | Gamma: emergent from PV+ (part of Hopfield sparsity step); Theta: augments stored patterns in Ξ with phase vectors | Gamma = part of Hopfield; Theta = modifies stored patterns |
| A9 Homeostatic plasticity | Adds per-cell bias terms θⱼ to energy function; preserves convergence, redistributes attractor basin volumes | Bias on Hopfield energy |
| A10 Short-term plasticity | Modulates effective weights W_eff = W ⊙ u ⊙ x per-step; stored W = ΞᵀΞ unchanged | Time-varying input to Hopfield |

**Critical invariant**: L2/3 recurrent connections remain symmetric
Hebbian (`W = ΞᵀΞ`) throughout all phases. STDP (A5) is restricted to
dendritic connections (basal, apical) which serve prediction, not pattern
completion. Applying STDP to L2/3 recurrence would destroy the energy
function and convergence guarantee — this is explicitly prohibited.

**A8 note**: Theta phase-augmented patterns increase the effective
dimensionality of Ξ (from `n_cells` to `n_cells + 2`). The Hopfield
mathematics is unchanged — it's the same energy function in a
higher-dimensional space. Capacity scales with the augmented dimension,
so the same spatial pattern at different theta phases stores as distinct
attractors. Gamma oscillations, by contrast, are not a Hopfield
augmentation — they ARE the sparsity enforcement step (PV+ top-k)
running at a natural rhythm determined by inhibition dynamics.

## Sections

### A1: Laminar Structure (L1–L6)

Real cortical columns have 6 layers with distinct cell types, connectivity,
and computational roles. The current design uses a flat set of minicolumns
where every cell is identical. This phase separates them.

**Limitations of flat column** (tests that MUST FAIL before laminar structure):

1. **Top-down context cannot bias representation**: Present an ambiguous
   input that partially matches two stored objects. Provide top-down
   context (apical signal) indicating which object is expected. In a
   flat column, all cells are equivalent — there is no L1 apical zone
   for context to arrive at, and no L5 cells to integrate bottom-up
   with top-down. The column’s output is unaffected by the context
   signal. Expected: recognition accuracy with and without context
   is statistically identical (Δ < 5%).

2. **No output/representation separation**: The flat column uses the
   same cells for internal representation (Hopfield settling) and
   external output. Training on 5 objects, read out the column’s
   internal representation and its output signal. In a flat column
   they are identical — there is no distinct output transformation.
   Expected: correlation between internal state and output = 1.0.

3. **Feedback cannot refine feedforward**: Present a sequence of
   increasingly refined inputs (coarse → detailed). In a flat column,
   there is no L6→L4 feedback pathway — each input is processed
   independently. Expected: recognition after 5 refinement steps is
   NOT better than recognition from the final input alone.

**Architecture**:

Each minicolumn now contains cells organized into layers:

- **L4** (input layer): Receives feedforward sensory input from the
  encoder (or from lower cortical areas via thalamus). Spiny stellate
  cells — local processing, no long-range projections. Spatial pooling
  happens here: `overlap = W_ff @ input`, top-k competition.
  L4 is the entry point. Its output feeds L2/3.

- **L2/3** (representation layer): Pyramidal cells that form the learned
  object representations. This is where Hopfield settling occurs —
  recurrent connections are within L2/3. Lateral projections to L2/3
  of other columns carry voting signals. The settled L2/3 pattern IS
  the column's "opinion" about what it's sensing.

- **L5a** (subcortical/motor output): Thick-tufted pyramidal cells with
  long apical dendrites reaching L1. These cells integrate bottom-up
  (from L2/3 via basal dendrites) and top-down (from higher areas via
  apical dendrites in L1). L5a projects to subcortical targets:
  - Motor system (brainstem, spinal cord — motor commands)
  - Striatum (action selection)
  - `get_output()` in Monty terms for motor/action signals
  Biologically these are L5 ET (extratelencephalic) cells. They generate
  burst firing (intrinsically bursting, IB), which is important for
  reliably driving distant subcortical targets.

- **L5b** (cortical feedforward output): Thin-tufted pyramidal cells
  with shorter apical dendrites. These carry the feedforward signal to
  higher cortical areas via thalamus — the inter-area hierarchy in
  Track 11. L5b projects to:
  - Higher cortical area's L4 (via thalamic relay — feedforward stream)
  - Contralateral cortex (callosal projections)
  Biologically these are L5 IT (intratelencephalic) cells. They fire in
  regular-spiking (RS) mode. The separation of L5a (motor) from L5b
  (hierarchy) ensures the inter-area feedforward signal and the motor
  output can be independently controlled — a column can output a motor
  command (L5a) while sending a different hierarchical representation
  (L5b) to the next area.

- **L6a** (corticothalamic feedback): Projects to the thalamic relay,
  modulating the gate that controls what input L4 sees (see A7). This
  closes the thalamocortical loop. L6a cells receive input from L4
  (copies of thalamic input) and L2/3 (settled representation), and
  learn to predict expected thalamic input. L6a is strictly intra-column
  — it talks to the column's own thalamic relay, not to other areas.

- **L6b** (corticocortical feedback): Projects back to lower cortical
  areas, providing top-down feedback in the deep hierarchy (Track 11).
  L6b cells receive input from L2/3 and from thalamic relay, and send
  feedback to lower areas' L1 (apical dendrites of L5a cells). These
  are remnants of the embryonic subplate and maintain some of its
  connectivity properties. L6b handles inter-area feedback, while L6a
  handles intra-column thalamic modulation.

- **L1** (context/apical layer): Almost no cell bodies — this layer
  contains the apical dendrite tufts from L5 pyramidal cells and axon
  terminals from higher cortical areas. Top-down context arrives here.
  When apical input in L1 coincides with basal input from L2/3, L5 cells
  generate plateau potentials (see A3).

**Tensor layout**:

```
cells_per_minicolumn = n_L4 + n_L23 + n_L5a + n_L5b + n_L6a + n_L6b
# L1 has no cell bodies — it's a dendritic zone on L5a/L5b cells

# Per-minicolumn cell allocation (configurable):
# L4:   2 cells  (input, local)
# L2/3: 4 cells  (representation, Hopfield settling)
# L5a:  1 cell   (subcortical/motor output, thick-tufted)
# L5b:  1 cell   (cortical feedforward output, thin-tufted)
# L6a:  1 cell   (corticothalamic feedback, A7 loop)
# L6b:  1 cell   (corticocortical feedback, Track 11 hierarchy)
```

Each layer has its own:
- Connectivity matrix (who projects to whom)
- Learning rules (layer-specific rates and modulation)
- Activation dynamics (L4 = feedforward only, L2/3 = recurrent Hopfield,
  L5 = dendritic integration, L6 = modulatory)

**Implementation**:
- `CorticalColumnTorch.step()` becomes a sequential pipeline through
  layers: input → L4 → L2/3 (settle) → L5a (motor output) + L5b
  (hierarchy output) → L6a (thalamic feedback) + L6b (cortical
  feedback) → thalamic gate update
- Each layer is a separate torch module with its own parameters
- Backward (feedback) connections from L5 → L2/3 and L6 → L4 run on the
  *next* step (one-step delay, biologically accurate)

**Tests** (after laminar implementation — same 3 scenarios must now PASS):

1. **Top-down context biases representation**: Ambiguous input +
   apical context → correct object recognized. Context-aided accuracy
   > baseline by at least 20 percentage points.
2. **Output/representation separation**: L2/3 internal state and L5a/L5b
   outputs are correlated but not identical (correlation < 0.9).
   L5a output reflects motor/subcortical intent; L5b output reflects
   hierarchical feedforward signal. L5a and L5b outputs are distinct
   from each other (correlation < 0.95) — motor and hierarchy signals
   diverge because L5a has stronger L1 apical drive and burst dynamics.
3. **Feedback refines feedforward**: 5 refinement steps with L6→L4
   feedback produce higher accuracy than the final input alone.
   Feedback advantage > 10 percentage points.

**Additional unit tests**:
- L4 activation is purely feedforward (no recurrence)
- L2/3 Hopfield settling converges as in Track 9
- L5a output depends on both L2/3 (basal) and L1 context (apical),
  produces burst-mode firing for motor commands
- L5b output depends on L2/3 (basal), produces regular-spiking
  feedforward signal for hierarchy
- L5a and L5b have distinct output profiles (motor vs hierarchy)
- L6a feedback modulates thalamic gate on subsequent steps
- L6b feedback signal is available for inter-area connections (Track 11)
- Full column step with all layers produces valid output
- Disabling layers (e.g., L6 feedback off) reproduces Track 9 behavior

---

### A2: Multi-Head Dendritic Attention

Each pyramidal neuron has ~50 independent basal dendritic branches.
Each branch can independently recognize a different context pattern via
NMDA spikes. This is multi-head attention within a single neuron — but
the integration follows biology, not a transformer.

**Limitations of single-segment prediction** (tests that MUST FAIL before
multi-head):

1. **Context capacity**: Train a cell to be predicted by 8 distinct
   context patterns (8 different preceding states that should each
   predict this cell). With n_heads=1 (single segment), the segment
   can learn at most one pattern well — later patterns overwrite
   earlier ones via Hebbian interference. Present all 8 contexts at
   eval. Expected: prediction accuracy ≤ 25% (only the most recent
   1–2 patterns are recalled).

2. **Context-specific learning interference**: Train on two tasks
   where the same cell should be predicted by context A (task 1) and
   context B (task 2), but the correct *response* of the cell is
   different in each task. With one segment, both contexts write to
   the same synapses → catastrophic interference. Expected: accuracy
   on task 1 drops below 60% after training on task 2.

3. **Discrimination of overlapping contexts**: Two contexts share 70%
   of their active cells but should predict different target cells.
   A single segment conflates them. Expected: discrimination accuracy
   < 65% (near chance for binary classification).

**Two levels of integration** (matching cortical physiology):

1. **Within a compartment type (basal branches): disjunctive (OR/MAX)**.
   If *any* basal branch recognizes its pattern, the cell is depolarized.
   A cell with 8 branches can be predicted by 8 different contexts —
   the power is in the disjunction. This matches HTM and the existing
   `CorticalColumn` behavior, where `any(segment_active) → cell predicted`.

2. **Between compartment types (basal × apical): conjunctive (AND)**.
   The BAC (backpropagation-activated calcium) firing mechanism requires
   coincident basal depolarization AND apical input to trigger a full
   calcium spike. This is genuinely multiplicative — already handled by
   the L5 laminar integration in A1 (basal from L2/3 × apical from L1).

**Architecture**:

Currently each cell has multiple segments, but they contribute to a single
binary prediction. The new design makes branches graded and independent
pattern detectors with proper disjunctive somatic integration:

- Each cell has `n_heads` dendritic branches per compartment type
- Each branch has its own sparse connectivity (a row block in the
  dendritic sparse matrix)
- Each branch produces an **independent graded activation** via:
  `branch_activation = sigmoid((D_branch @ x - threshold) / temp)`
- Somatic integration is **max-over-branches** (OR), not product (AND):

  ```
  # Each branch independently detects its learned pattern
  # Batched: D_branches is (n_heads, n_synapses_per_branch) sparse
  branch_overlaps = D_branches @ x    # (n_heads,) per cell
  branch_acts = sigmoid((branch_overlaps - threshold) / temp)

  # Disjunctive integration: cell fires if ANY branch matches
  cell_depol = max(branch_acts)       # OR semantics
  ```

  This is GPU-scalable: all branches across all cells in a layer are
  stored as a single sparse matrix with a leading (cell, head) index.
  The max reduction is a `scatter_max` over the `(cell, head) → cell`
  mapping — identical in structure to the existing Track 9
  `scatter_max(seg_active, seg_to_cell)` but with an additional head
  dimension:

  ```
  # All branches for all cells in one batched sparse matmul:
  # D: sparse (total_branches, n_cells), total_branches = n_cells * n_heads
  all_branch_overlaps = D @ x                    # (total_branches,)
  all_branch_acts = sigmoid((all_branch_overlaps - threshold) / temp)

  # Map each branch to its parent cell, take max
  cell_depol = scatter_max(all_branch_acts, branch_to_cell)  # (n_cells,)

  # Which branch won? (needed for branch-specific learning)
  winning_branch = scatter_argmax(all_branch_acts, branch_to_cell)
  ```

- **Branch-specific Hebbian learning**: Only the winning branch (the one
  that matched best) gets its synapses strengthened. Non-winning branches
  are untouched. This allows different branches to specialize in different
  context patterns without interference:

  ```
  # Only strengthen synapses on the branch that actually matched
  ΔD[winning_branch] += lr * x  # Hebbian: strengthen active synapses
  ```

- **Capacity scales linearly**: n_heads branches per cell means n_heads
  independent context patterns can predict that cell. Total branch storage
  is O(n_cells × n_heads × synapses_per_branch). No quadratic blowup.
  The sparse matrix grows linearly with n_heads.

**Why OR and not AND?** Consider a cell that should fire when seeing
a mug. Branch A learns "cylindrical curvature at this location."
Branch B learns "handle shape nearby." Branch C learns "seen this
object before at a different rotation." Any ONE of these is sufficient
evidence to predict the cell — they're alternative contexts, not
required co-conditions. Requiring all branches to agree (AND) would
mean the cell ONLY fires when every context is simultaneously present,
which is almost never. The AND operation is reserved for the basal ×
apical coincidence at the compartment level (A1/A3).

**Biological basis**: Major et al. (2013) — "Active properties of
neocortical pyramidal neuron dendrites" — individual branches generate
local NMDA spikes that propagate to the soma; Poirazi et al. (2003) —
"Pyramidal Neuron as a Two-Layer Neural Network" — somatic output is a
nonlinear function of independent dendritic subunit outputs; Larkum
(2013) — the multiplicative BAC mechanism operates between apical and
basal *compartments*, not within branches of the same type.

**Tests** (after multi-head implementation — same 3 scenarios must now PASS):

1. **Context capacity**: 8 distinct contexts each predict the cell.
   With n_heads=8, each branch learns one context without interference.
   Prediction accuracy > 85% across all 8 contexts.
2. **Context-specific learning**: Two tasks with different responses
   coexist on different branches. Accuracy on both tasks > 80% after
   sequential training (no catastrophic interference).
3. **Overlapping context discrimination**: Two 70%-overlapping contexts
   activate different branches. Discrimination accuracy > 85%.

**Additional unit tests**:
- Multiple branches on one cell learn different patterns independently
- Disjunctive integration: cell fires when ANY one branch matches,
  even if all other branches are silent
- Branch-specific learning: strengthen winning branch, others unchanged
- Number of branches per cell is configurable via n_heads
- Disabling multi-head (n_heads=1) reproduces Track 9 single-segment
  behavior (`scatter_max` with one branch per cell = identity)
- GPU scaling: n_heads=32 runs without memory blowup or numerical
  instability (unlike multiplicative formula which explodes at n_heads>4)

---

### A3: Plateau Potential Working Memory (Context Window)

When L5 cells receive coincident basal (from L2/3) and apical (from L1)
input, they generate **dendritic calcium spikes** — plateau potentials
lasting 200–500ms. This sustained depolarization maintains a working
memory trace of recent inputs.

**Limitations of memoryless retrieval** (tests that MUST FAIL before
plateau potentials):

1. **Multi-viewpoint convergence**: Present the same object from 5
   successive viewpoints (saccade sequence). Each viewpoint partially
   matches the object. Without working memory, the column treats each
   viewpoint independently — the 5th viewpoint has no benefit from
   the first 4. Expected: recognition accuracy at viewpoint 5 is NOT
   significantly higher than at viewpoint 1 (<10% improvement).

2. **Temporal context aids disambiguation**: Two similar objects (mug
   and cup) share many features. Show 3 viewpoints of one object.
   Without working memory, the 3rd viewpoint produces the same
   ambiguity as the 1st — no accumulation of disambiguating evidence.
   Expected: confusion rate between mug/cup at viewpoint 3 is NOT
   lower than at viewpoint 1 (< 10% reduction).

3. **Sequence completion from partial cue**: Train on 4-element
   sequence A→B→C→D. Present A, B, then remove input. Without
   persistent activity, the column immediately forgets A and B.
   Expected: retrieval of C after input removal is at chance.

**Architecture**:

- Each L5 cell has a `plateau_state` tensor: a decaying trace of
  recent activations, not just the current activation
- When a cell fires with both basal and apical support (high confidence),
  its plateau state is set to a high value and decays exponentially
  over subsequent steps
- The plateau state is included in the Hopfield retrieval query:

  ```
  # Context-enriched query for Hopfield retrieval:
  query = current_activation + gamma * plateau_state
  x_new = Ξᵀ softmax(β · Ξ · query)
  ```

  This means the retrieval considers not just "what am I seeing now"
  but "what have I confidently identified in the last W steps."

- The effective context window size is controlled by the decay constant
  τ (wired to the 5-HT/temporal-horizon neuromodulator):
  - High temporal horizon → slow decay → long context window
  - Low temporal horizon → fast decay → focus on immediate input

**Biological basis**: Major & Bhatt (2018) — plateau potentials as
cortical working memory; Takahashi et al. (2020) — plateau-dependent
persistent activity in sensory cortex.

**Tests** (after plateau implementation — same 3 scenarios must now PASS):

1. **Multi-viewpoint convergence**: Recognition accuracy increases
   monotonically across viewpoints. Viewpoint 5 accuracy > viewpoint 1
   by at least 25 percentage points.
2. **Temporal context aids disambiguation**: Confusion rate between
   similar objects drops after 3 viewpoints. Viewpoint 3 confusion
   rate < 50% of viewpoint 1 confusion rate.
3. **Sequence completion from partial cue**: After A→B presented,
   input removed. Plateau state sustains A+B context. Retrieval of C
   accuracy > 75%.

**Additional unit tests**:
- Plateau state decays exponentially with configurable time constant
- Coincident basal + apical input → plateau triggered; basal alone → no plateau
- Hopfield retrieval with plateau context is more accurate than without
  for sequential object recognition (same object, multiple viewpoints)
- Context window length scales with τ parameter
- Neuromodulator (temporal horizon) controls τ dynamically

---

### A4: Inhibitory Interneuron Diversity

Real cortex has three main inhibitory interneuron types with distinct
computational roles. The current design uses only top-k (one type of
inhibition). Adding the others enables learned dynamic gating.

**Limitations of top-k-only inhibition** (tests that MUST FAIL before
inhibitory interneuron diversity):

1. **No learned attention routing**: Present an input that activates
   multiple dendritic branches (multi-head, from A2). All branches
   contribute equally to the cell’s response regardless of which
   branches are relevant to the current task. With only top-k (PV+),
   there is no mechanism to selectively suppress irrelevant branches.
   Expected: branch-level attention cannot be modulated by context —
   all branches contribute equally (entropy of branch activations is
   maximal for multi-branch cells).

2. **No novelty-dependent learning speed**: Train on a known object
   (low novelty), then present a novel object. Without VIP+
   disinhibition, the learning rate is the same for both — there is
   no mechanism to "open up" the cell to new patterns when novelty
   is high. Expected: learning speed (weight change magnitude per
   step) for novel stimuli is NOT significantly higher than for
   familiar stimuli (< 20% difference).

3. **Uniform sparsity across layers**: L4 (input processing) and L2/3
   (associative) may need different sparsity levels, but global top-k
   enforces the same k everywhere. Expected: recognition accuracy
   with forced uniform sparsity is lower than with independently
   tuned per-layer sparsity. Accuracy difference NOT testable with
   current system (no layer distinction). This test becomes meaningful
   only after A1.

**Architecture**:

Three inhibitory populations per minicolumn:

- **PV+ (basket cells)**: Fast perisomatic inhibition.
  Implements winner-take-all competition within a layer.
  This is what top-k already approximates, but now it's layer-specific:
  PV+ cells in L4 inhibit other L4 cells, PV+ in L2/3 inhibit L2/3, etc.
  Each layer can have different sparsity levels.

  ```
  # Layer-specific top-k (PV+ inhibition)
  L4_active  = topk(L4_overlap, k=k_L4)
  L23_active = topk(L23_hopfield, k=k_L23)
  ```

  **PV+ inhibition is the gamma oscillator.** The E→PV+→E feedback
  loop naturally creates oscillatory dynamics: excitatory cells fire →
  PV+ cells fire (suppressing excitatory) → inhibition decays →
  excitatory cells fire again. The cycle period is set by PV+
  inhibition strength and decay time constant, NOT by a free
  `period` parameter. Stronger input drive → faster gamma. Different
  layers can oscillate at different intrinsic frequencies because
  they have different PV+ populations. This emergent gamma is the
  timing reference for phase coding (A8).

  **Gap junction coupling between PV+ cells.** PV+ basket cells are
  electrically coupled via connexin-36 (Cx36) gap junctions. Unlike
  chemical synapses, gap junctions are bidirectional, symmetric, and
  non-plastic — they are a fixed structural property. ~50% of nearby
  PV+ pairs are coupled (Galarreta & Hestrin 1999, 2001).

  Gap junctions synchronize PV+ firing within <1ms precision. Without
  them, PV+ cells oscillate at gamma individually but with random
  phase offsets between cells — producing incoherent, noisy inhibition.
  With gap junctions, PV+ cells synchronize into a coherent gamma
  rhythm that provides a stable timing reference.

  The coupling follows Ohm's law, implemented as graph Laplacian
  diffusion on PV+ activations:

  ```
  # G_gap: symmetric sparse matrix (~50% of PV+ pairs, non-plastic)
  # g_gap: coupling conductance (~0.1-0.3)
  # L_gap = D - G_gap (graph Laplacian, D = degree matrix)
  #
  # Each PV+ cell is pulled toward the mean of its coupled neighbors:
  pv_coupled = pv_activation - g_gap * L_gap @ pv_activation
  ```

  This is a diffusion/smoothing step: coupled PV+ cells converge
  toward a shared activation level, synchronizing their oscillatory
  dynamics. Because G_gap is symmetric, the coupled dynamics preserve
  the symmetric structure needed for oscillatory stability. The
  Laplacian diffusion has a well-defined energy function (Dirichlet
  energy: ½ xᵀ L x), which only decreases — so gap junction coupling
  provably reduces phase dispersion across the PV+ network.

  **Biological reference**: Galarreta & Hestrin (1999) — "A network of
  fast-spiking cells in the neocortex connected by electrical
  synapses"; Connors & Long (2004) — "Electrical synapses in the
  mammalian brain."

- **SST+ (Martinotti cells)**: Dendritic inhibition.
  Projects to the dendrites of pyramidal cells, selectively gating
  specific dendritic branches. This controls **which attention heads
  are active** on each step. SST+ cells learn which branches to
  inhibit based on context:

  ```
  # SST+ inhibition: gate dendritic branches
  branch_gate = 1 - SST_activation  # learned, context-dependent
  gated_branch_acts = branch_acts * branch_gate
  ```

  When SST+ strongly inhibits a branch, that "attention head" is shut
  off. This is biologically-grounded learned attention routing.

- **VIP+ cells**: Inhibit SST+ cells (disinhibition).
  When VIP+ fires, it releases SST+ inhibition, unmasking dendritic
  branches. VIP+ is driven by top-down attention signals and
  neuromodulators (especially ACh/novelty):

  ```
  # VIP+ disinhibition circuit:
  VIP_activation = f(top_down_attention, novelty)
  SST_effective = SST_activation * (1 - VIP_activation)
  branch_gate = 1 - SST_effective
  ```

  High novelty → VIP+ fires → SST+ inhibited → all branches unmasked
  → the cell "opens up" to learn new associations. Low novelty →
  VIP+ quiet → SST+ active → only relevant branches contribute →
  efficient, focused processing.

**Biological basis**: Tremblay et al. (2016) — "GABAergic Interneurons
in the Neocortex: From Cellular Properties to Circuits"; Karnani et al.
(2016) — VIP-SST disinhibition.

**Limitation of no gap junctions** (test that MUST FAIL before adding
gap junction coupling to PV+):

4. **Gamma phase coherence**: Run 16 PV+ cells in an E→PV+→E loop
   for 100 steps. Without gap junctions, each PV+ cell oscillates at
   gamma frequency but with random initial phase offsets that never
   synchronize. Measure mean pairwise phase coherence across PV+ cells.
   Expected: phase coherence < 0.5 (incoherent).

**Tests** (after inhibitory interneuron implementation — same 3+1
scenarios must now PASS):

1. **Learned attention routing**: SST+ cells learn to gate irrelevant
   branches based on context. Branch activation entropy is significantly
   lower than uniform (< 50% of maximal entropy) — the column focuses
   on relevant branches.
2. **Novelty-dependent learning speed**: VIP+ disinhibition on novel
   stimuli produces > 2× weight change magnitude per step compared to
   familiar stimuli.
3. **Per-layer sparsity**: L4 and L2/3 independently maintain different
   sparsity levels (k_L4 ≠ k_L23). Per-layer tuned sparsity improves
   recognition accuracy by > 5% over uniform sparsity.
4. **Gamma phase coherence with gap junctions**: Same 16 PV+ cells,
   now with gap junction coupling (g_gap=0.2, ~50% sparse G_gap).
   Mean pairwise phase coherence > 0.8 within 20 steps. Gap junctions
   pull PV+ cells into synchronized gamma.

**Additional unit tests**:
- PV+ inhibition: layer-specific sparsity is maintained independently
- SST+ inhibition: gating one branch suppresses its contribution to
  cell depolarization
- VIP+ disinhibition: activating VIP unmasks SST-gated branches
- Full circuit: high novelty → VIP → SST suppressed → all branches
  active → faster learning on novel input
- Gap junction coupling: G_gap is symmetric, non-plastic, and sparse
- Gap junction diffusion reduces PV+ activation variance across cells
- Disabling gap junctions (g_gap=0) reverts to uncoupled PV+ behavior
- Disabling interneuron diversity (all gating = 1) reproduces Track 9

---

### A5: Spike-Timing-Dependent Plasticity (STDP)

Symmetric Hebbian learning (`W += lr * pre * post`) treats co-activation
as sufficient evidence for association. It cannot represent *direction*:
the fact that A caused B is lost. This section first demonstrates
concrete failures of symmetric Hebbian, then implements STDP to fix them.

**Limitations of symmetric Hebbian** (tests that MUST FAIL before STDP):

These tests establish the value of STDP by showing what the current
system cannot do. Each test is run with symmetric Hebbian and must
produce the wrong answer or chance-level performance:

1. **Directed sequence recall**: Train on sequence A→B→C→D (four
   distinct activation patterns presented in order). Present only A.
   Symmetric Hebbian retrieves a cloud of {B, C, D} with roughly
   equal similarity — it cannot isolate B as the *next* element.
   Expected: top-1 retrieval accuracy at chance (≈33% for 3 successors).
   Always use the same realistic 3d model, perhaps with different
   behaviours and/or orientations.

2. **Motion direction discrimination**: Present the same 5 activation
   patterns in forward order (1→2→3→4→5) and reverse order (5→4→3→2→1)
   as two distinct "objects." Train on both. At eval, present a forward
   sequence. Symmetric Hebbian gives equal evidence to both objects —
   the pairwise co-activations are identical regardless of direction.
   Expected: classification accuracy ≈ 50% (chance).

3. **Ambiguous sequence disambiguation**: Train on A→B→C and D→B→E
   (shared element B). Present A then B, ask for next-element prediction.
   Symmetric Hebbian gives equal weight to C and E from B, regardless
   of whether A or D preceded it. Expected: prediction accuracy ≈ 50%.

4. **Temporal prediction error (skip detection)**: Train on A→B→C.
   Present A→C (skipping B). Symmetric Hebbian sees A and C, both
   associated with the sequence — no surprise signal. Expected:
   surprise/energy for the A→C transition is NOT significantly higher
   than for the trained A→B transition.

**Architecture** (STDP implementation that fixes these limitations):

- Maintain per-synapse **timing traces**: exponentially decaying records
  of when the pre- and post-synaptic cells were last active
- Pre trace: `trace_pre[syn] = trace_pre[syn] * decay + pre_active`
- Post trace: `trace_post[cell] = trace_post[cell] * decay + post_active`
- Weight update:

  ```
  # Causal (pre before post) → LTP
  ΔW_LTP = lr_LTP * post_active * trace_pre

  # Acausal (post before pre) → LTD
  ΔW_LTD = -lr_LTD * pre_active * trace_post

  ΔW = ΔW_LTP + ΔW_LTD
  ```

- This replaces the symmetric Hebbian rule with a temporally asymmetric
  one: A→B strengthens A-to-B connections and weakens B-to-A connections
- STDP applies to **dendritic connections only**: basal dendrites and
  apical dendrites. These serve temporal prediction ("what comes next")
  and top-down prediction ("what the hierarchy expects"), respectively.
- **L2/3 recurrent connections are excluded from STDP.** Those connections
  form the Hopfield weight matrix `W = ΞᵀΞ`, which must remain symmetric
  to preserve the energy function and convergent attractor dynamics
  (see "Relationship to Hopfield Dynamics" above). Applying STDP to
  L2/3 recurrence would destroy the energy landscape — the network
  could oscillate or diverge instead of settling to attractors.
  Biologically, L2/3 recurrent excitatory connections are local and
  largely reciprocal (symmetric), consistent with this constraint.
- The STDP window (decay constant) is modulated by the temporal-horizon
  neuromodulator

**Implementation note**: In continuous-time (rate-coded) networks, STDP
is approximated by trace-based rules. Each step approximates a time
window of ~20ms. The traces decay with a time constant of ~5–10 steps.

**Tests** (after STDP implementation — same 4 scenarios must now PASS):

1. **Directed sequence recall**: Present A from trained A→B→C→D.
   Top-1 retrieval is B (the successor). Accuracy > 90%.
2. **Motion direction discrimination**: Forward vs reverse sequences
   classified correctly. Accuracy > 85%.
3. **Ambiguous sequence disambiguation**: A→B predicts C (not E);
   D→B predicts E (not C). Accuracy > 85%.
4. **Temporal prediction error**: A→C (skip B) produces significantly
   higher surprise than A→B (trained transition). Energy difference
   is > 2× the baseline noise.

**Additional unit tests**:
- Pre-before-post → synapse strengthened
- Post-before-pre → synapse weakened
- Symmetric co-activation → net zero change (LTP and LTD cancel)
- Trace decay constant affects learning window as expected

**Limitation of STDP**: STDP captures *ordering* (A before B) but not
*precise inter-event timing* (whether A is 200ms or 500ms before B).
Precise temporal interval encoding requires oscillatory phase coding,
addressed in A8.

---

### A6: Eligibility Traces (Three-Factor with Temporal Bridge)

The weight change from STDP (A5) should not be applied immediately.
Instead, it creates an **eligibility trace** — a potential weight change
that only materializes when a neuromodulatory signal arrives.

**Limitations of immediate STDP weight updates** (tests that MUST FAIL
before eligibility traces):

1. **Delayed reward credit assignment**: Train on a sequence A→B→C
   where a reward signal arrives 20 steps after C. With immediate STDP,
   weight updates happen at each step and are done long before the
   reward arrives. The reward has no way to retroactively strengthen
   the A→B and B→C connections that led to it. Expected: re-presenting
   A after reward produces no better recall of the rewarded sequence
   than an unrewarded sequence.

2. **Selective consolidation**: Train on two sequences: A→B→C (rewarded
   after 15 steps) and D→E→F (unrewarded). With immediate STDP, both
   sequences are learned equally — the reward signal has no retroactive
   effect on which synapses are consolidated. Expected: recall accuracy
   for both sequences is statistically identical.

3. **Noise resilience**: Present a sequence with occasional random
   noise patterns interspersed (A→noise→B→noise→C). With immediate
   STDP, noise→B connections are strengthened just as much as A→B
   connections (both are causal). Expected: noise patterns predict B
   with similar strength as A does.

**Architecture**:

```
# Step 1: STDP computes proposed weight change
ΔW_proposed = STDP(pre_trace, post_trace)

# Step 2: Store as eligibility trace (decays over seconds = many steps)
eligibility[syn] = eligibility[syn] * decay_elig + ΔW_proposed

# Step 3: Actual weight change only when neuromodulator arrives
ΔW_actual = eligibility[syn] * neuromodulator_signal
W += ΔW_actual
```

- The eligibility trace has a longer time constant (~50–100 steps) than
  the STDP traces (~5–10 steps). It bridges the gap between "these two
  neurons were co-active" and "a reward/surprise signal arrived later."
- The neuromodulator signal comes from the DA/reward and ACh/novelty
  channels: positive reward → consolidate eligible synapses, negative →
  decay them, high novelty → consolidate (learn from surprising events)
- This is the biological implementation of temporal credit assignment
  without backprop

**Biological basis**: Gerstner et al. (2018) — "Eligibility Traces and
Plasticity on Behavioral Time Scales"; Izhikevich (2007) — "Solving the
Distal Reward Problem through Linkage of STDP and Dopamine Signaling."

**Tests** (after eligibility trace implementation — same 3 scenarios
must now PASS):

1. **Delayed reward credit assignment**: Reward at step +20 retroactively
   consolidates the A→B→C sequence (within eligibility window).
   Re-presenting A after reward recalls B with > 80% accuracy. Without
   reward, recall is at baseline.
2. **Selective consolidation**: Rewarded sequence A→B→C is consolidated;
   unrewarded D→E→F decays. After 50 steps, recall accuracy for
   A→B→C > 80%, for D→E→F < 40%.
3. **Noise resilience**: Only reward-validated connections are
   consolidated. After reward, A predicts B (accuracy > 80%) but
   noise patterns do NOT predict B (accuracy < 30%).

**Additional unit tests**:
- Eligible synapse + reward → weight change
- Eligible synapse + no reward → eligibility decays, no weight change
- Eligible synapse + negative reward → weight weakened
- Delayed reward (N steps after co-activation) still consolidates
  if within eligibility window
- Eligibility trace decay constant is configurable

---

### A7: Thalamocortical Loop (Within-Column)

The thalamus is not a passive relay. It actively gates what information
reaches L4 and maintains persistent activity that supports working memory.

**Limitations of ungated feedforward** (tests that MUST FAIL before
thalamocortical gating):

1. **Noise robustness**: Add Gaussian noise (SNR = 2.0) to the sensory
   input. Without thalamic gating, all noise passes directly to L4
   and corrupts the representation. Expected: recognition accuracy
   drops by > 30% compared to clean input.

2. **Redundant processing**: Present the same input 10 times in a row
   (the column has already recognized the object). Without gating,
   each presentation drives L4 with full strength, wasting compute
   and potentially overwriting settled representations. Expected:
   compute cost (number of L2/3 settling iterations) is the same at
   step 10 as at step 1. No efficiency gain from familiarity.

3. **Prediction error detection**: After training, present a familiar
   object, then suddenly switch to a novel one. Without thalamic
   gating there is no predictive coding mechanism — the switch is
   processed the same way as any input. Expected: surprise signal
   at the switch is NOT significantly higher than at any other step
   (< 1.5× baseline).

**Architecture**:

Each column has an associated **thalamic relay unit**:

```
Sensory input → Thalamic relay → L4
                     ↑
                L6 (feedback/modulation)
```

- The relay has a gate value in [0, 1] per input channel
- L6 cells learn to modulate the gate based on the column's current
  state (expectations, attention):

  ```
  gate = sigmoid(W_L6_to_thal @ L6_activation + bias)
  L4_input = gate * sensory_input
  ```

- When the column "expects" certain input (low surprise), L6 partially
  closes the gate → attenuates redundant feedforward drive → saves
  energy, reduces noise
- When surprise is high, L6 opens the gate → full feedforward input
  reaches L4
- This implements **predictive coding at the thalamic level**: the
  column predicts its input via L6 → thalamus, and only prediction
  errors (unexpected input) fully propagate to L4

**Tests** (after thalamocortical implementation — same 3 scenarios must
now PASS):

1. **Noise robustness**: With thalamic gating, L6 learns to partially
   close the gate for predicted input. Noisy input (SNR = 2.0)
   accuracy drops by < 10% (vs > 30% without gating).
2. **Redundant processing**: After recognition, L6 closes the gate for
   predicted (redundant) input. L2/3 settling iterations at step 10
   are < 30% of step 1. Column efficiently suppresses redundant
   feedforward drive.
3. **Prediction error detection**: Object switch produces a surprise
   signal > 3× baseline (gate was closed for predicted input, novel
   input forces it wide open → large prediction error signal).

**Additional unit tests**:
- L6 feedback modulates L4 input magnitude
- High surprise → gate opens → full input
- Low surprise → gate partially closes → attenuated input
- Learning in L6 feedback weights reduces surprise over training
- Disabling thalamic gating reproduces Track 9 behavior

---

### A8: Oscillatory Phase Coding (Precise Temporal Intervals)

STDP (A5) captures temporal *ordering* but not *interval*: it knows A
comes before B, but cannot distinguish whether A is 4 steps or 40 steps
before B. Real cortex encodes precise timing information through the
*phase* of a cell's firing relative to ongoing oscillations.

Critically, the relevant oscillations are **not a bolted-on sinusoid**
with a free `period` parameter. Gamma oscillations emerge naturally
from the PV+ interneuron circuit (A4), and theta oscillations arise
from thalamocortical/motor loops. This section connects them to temporal
encoding.

**Limitations of STDP alone** (tests that MUST FAIL without phase coding):

1. **Interval discrimination**: Train on two sequences with the same
   elements but different inter-event intervals: fast (A→[2 steps]→B)
   vs slow (A→[10 steps]→B). STDP alone learns A→B in both cases —
   the trace decays differently but the learned weight direction is
   the same. Present a fast A→B at eval. STDP cannot distinguish which
   training sequence it came from. Expected: classification ≈ 50%.

2. **Rhythm reproduction**: Train on a repeating pattern with specific
   timing: A(step 0), B(step 3), C(step 4), A(step 8), B(step 11),
   C(step 12) — a rhythm with a long gap then two quick events. The
   column should reproduce this timing pattern when primed with A.
   STDP alone predicts the *order* A→B→C but generates it at uniform
   intervals. Expected: inter-event timing variance from trained
   rhythm is high (no rhythm preservation).

3. **Duration-sensitive recognition**: Two "objects" defined by the
   same spatial features but different dynamics — object X rotates
   slowly (features change every 10 steps), object Y rotates fast
   (features change every 2 steps). After training on both, present
   one at eval. STDP sees the same feature transitions in the same
   order. Expected: classification ≈ 50%.

**Architecture — Two Nested Oscillatory Bands**:

Real cortex uses multiple nested oscillations at different frequency
bands. Two are essential for temporal encoding within a column:

#### Gamma (~30–100 Hz): Emergent from PV+ Inhibition

Gamma is NOT a free parameter — it emerges from the E→PV+→E feedback
loop defined in A4. The implementation makes this explicit:

```
# Instead of instantaneous top-k, PV+ inhibition has temporal dynamics:
pv_drive = W_E_to_PV @ excitatory_activation
pv_activation = pv_activation * pv_decay + pv_drive
inhibition = W_PV_to_E @ pv_activation

# Excitatory cells compete through this dynamic inhibition:
excitatory_activation = relu(raw_drive - inhibition)

# The cycle period emerges from pv_decay and connection strengths:
# faster input drive → faster gamma, stronger PV → slower gamma.
# Typical emergent period: 3–8 steps (≈ 30–100 Hz at 20ms/step)
```

Each gamma cycle corresponds to **one "slot"** — one discrete item
in a sequence or working memory. Multiple items map to successive
gamma cycles. The phase within a gamma cycle encodes fine-grained
temporal position.

**Gamma frequency adapts to input drive** (not a tunable parameter):
- Strong sensory input → fast gamma → high temporal resolution
- Weak/ambiguous input → slow gamma → coarser timing
- This is the ING (Interneuron Network Gamma) model
  (Whittington et al. 2000)

**Each layer has its own intrinsic gamma** because layers have
different PV+ populations. L4 (strong feedforward drive, fast PV+)
typically oscillates faster than L2/3 (recurrent, slower PV+).
This frequency difference is functional, not a bug.

#### Theta (~4–8 Hz): Thalamocortical/Motor Pacemaker

Theta is a slower oscillation driven by non-local sources:

```
# Theta phase advances with motor/exploration steps:
theta_phase[t] = theta_phase[t-1] + theta_rate
theta_rate = base_theta_rate * motor_speed_factor

# motor_speed_factor: derived from the motor policy's step size.
# Fast saccades → faster theta → more compressed sequences per cycle.
# Slow careful exploration → slower theta → finer per-cycle resolution.
```

- Theta is modulated by **behavioral state**: the motor policy's
  step size and the arousal neuromodulator (NE) both influence
  `theta_rate`. High arousal / fast exploration → faster theta.
- Each theta cycle frames a **sequence chunk**: ~4–8 gamma cycles
  fit inside one theta cycle, defining a natural sequence window
  of 4–8 items (matching human working memory capacity).
- Theta comes from the thalamocortical loop (A7) — the L6→thalamus
  feedback circuit provides the slow modulatory drive.

#### Theta-Gamma Coupling

Gamma amplitude is modulated by theta phase — gamma bursts are
strongest at the trough of theta, weakest at the peak:

```
# Theta-gamma coupling: gamma gain modulated by theta
gamma_gain = 0.5 * (1 + cos(theta_phase[t] - preferred_theta_phase))
pv_drive = gamma_gain * W_E_to_PV @ excitatory_activation
```

This nesting means:
- **Within a gamma cycle**: fine temporal position (which step
  within the slot)
- **Gamma cycle index within theta cycle**: which item in a
  sequence (1st, 2nd, 3rd...)
- **Theta cycle count**: which chunk of the exploration episode

#### Phase Encoding in Hopfield Patterns

Cell activations are augmented with theta phase (not gamma —
gamma is too fast and is captured by the PV+ dynamics themselves):

```
# Theta phase tag at the moment of cell activation:
theta_tag[cell] = theta_phase[t]

# Encoded as 2D unit vector for smooth wraparound:
phase_vec[cell] = [cos(theta_tag), sin(theta_tag)]

# Augmented Hopfield pattern:
pattern_augmented = concat(x_spatial, phase_vec)  # (n_cells + 2,)
```

Two patterns with identical spatial features but different theta
phases store as **different attractors** in Ξ. This is how the
same feature (e.g., "cylindrical curvature") at different points
in an exploration trajectory is distinguished.

#### Phase Precession

As a sequence progresses through gamma slots within a theta cycle,
successive items fire at progressively earlier theta phases
(matching hippocampal theta-phase precession):

```
# Items presented later in a sequence fire earlier in theta:
precession_shift = -precession_rate * gamma_cycle_index
effective_theta_phase = theta_phase[t] + precession_shift
```

This compresses the temporal sequence into a phase code readable
within a single theta cycle — enabling one-shot sequence retrieval.

**Biological basis**: O'Keefe & Recce (1993) — theta-phase precession
in hippocampal place cells; Lisman & Jensen (2013) — theta-gamma
neural code for multi-item working memory; Whittington et al. (2000)
— ING model of gamma from PV+ interneuron network; Fries (2015) —
"Rhythms for Cognition: Communication through Coherence."

**Tests** (after implementation — same 3 scenarios must now PASS):

1. **Interval discrimination**: Fast A→B vs slow A→B classified
   correctly based on different theta phase differences. > 80%.
2. **Rhythm reproduction**: Primed with A, column reproduces trained
   inter-event timing (long gap then two quick events) because gamma
   slot count between events is captured in the theta-phase code.
   Timing error < 20% of trained intervals.
3. **Duration-sensitive recognition**: Slow-rotating vs fast-rotating
   object classified correctly because different motor speeds produce
   different theta rates → different phase signatures. > 80%.

**Additional unit tests**:
- PV+ E-I loop produces stable oscillation with measurable period
- Gamma frequency increases with input drive strength
- Different layers produce different intrinsic gamma frequencies
- Theta rate scales with motor step size
- ~4-8 gamma cycles fit within one theta cycle
- Theta-phase-augmented patterns are distinguishable by Hopfield
  retrieval when spatial features are identical but phases differ
- Phase precession shifts earlier with sequence position
- Disabling oscillatory coding (phase_coding=False) reproduces A5
  behavior exactly (PV+ reverts to instantaneous top-k)

**Interaction with other features**:
- **A4 (PV+ interneurons)**: Gamma IS the PV+ dynamics, not a
  separate mechanism. Phase 5b extends A4's PV+ from instantaneous
  top-k to dynamic E-I oscillation. This is why Phase 5b depends
  on Phase 4.
- **STDP (A5)**: STDP learns *directed* connections, phase coding
  adds *interval* information. Together they capture the full
  temporal structure of sequences.
- **Plateau potentials (A3)**: Plateau decay τ sets the *working
  memory window*; theta period sets the *sequence chunk size*;
  gamma period sets the *item resolution* within a chunk.
  These are complementary.
- **Neuromodulation**: ACh (novelty) promotes gamma power and
  suppresses alpha — this is biologically well-established
  (Muthukumaraswamy et al. 2009). NE (arousal) modulates theta
  rate. These are already in Track 9 Phase 7.
- **Track 11 inter-area**: Gamma phase coherence between areas
  determines communication efficiency — see Track 11 B6
  (Communication Through Coherence).

---

### A9: Homeostatic / Intrinsic Plasticity

Hebbian learning is inherently unstable: cells that participate in many
stored patterns receive stronger drive (more active → stronger synapses →
even more active). Without a compensatory mechanism, the network
degenerates into two populations: overactive "hub" cells that fire for
everything and silent "dead" cells that never participate.

**Limitations of unregulated Hebbian learning** (tests that MUST FAIL
before homeostatic plasticity):

1. **Runaway excitation after extended training**: Train on 50 objects
   sequentially. After storing all 50, evaluate on each object. Some
   L2/3 cells participate in many stored patterns (high overlap). Without
   homeostasis, these cells fire for every input — they've become
   non-discriminative. Expected: for cells in >10 patterns, mean
   activation specificity (fraction of objects that trigger the cell) is
   > 0.5 (cell fires for more than half of all objects).

2. **Silent cell death**: After training on 50 diverse objects, measure
   what fraction of L2/3 cells have zero firing rate over 100 evaluation
   steps across all trained objects. Without homeostasis, cells that were
   never incorporated into any pattern during initial training remain
   permanently silent — they have low initial overlap with any input, so
   they never win competition, so their synapses are never strengthened.
   Expected: > 25% of L2/3 cells have zero firing rate (wasted capacity).

3. **Scale instability across parallel columns**: Run N=8 identical
   columns in parallel (same architecture, different random initialization)
   on the same training data. Without homeostasis, small initial
   differences amplify through Hebbian feedback: some columns develop
   stronger representations (higher firing rates), others become
   relatively silent. Expected: coefficient of variation of mean firing
   rates across 8 columns > 0.5 after 50-object training.

**Architecture**:

Each cell maintains a running estimate of its own firing rate and adjusts
its threshold to converge toward a target rate. This is a slow negative
feedback loop — the timescale is ~100-1000× slower than Hebbian learning,
matching biological synaptic scaling (Turrigiano 2008, 2012).

```
# Running estimate of each cell's firing rate (very slow exponential average)
running_rate = running_rate * rate_decay + (1 - rate_decay) * is_active
# rate_decay ≈ 0.999 (much slower than Hebbian lr)

# Threshold adjustment: nudge threshold toward target firing rate
threshold += lr_homeo * (running_rate - target_rate)
# If cell fires too much: running_rate > target → threshold increases → harder to fire
# If cell fires too little: running_rate < target → threshold decreases → easier to fire

# Optional: gain modulation (multiplicative, preserves relative synapse strengths)
gain = target_rate / (running_rate + epsilon)
gain = clamp(gain, 0.9, 1.1)  # bounded to prevent instability
effective_drive = gain * raw_drive
```

**Key design choices**:

- **Threshold adjustment, not weight scaling**: The primary mechanism is
  threshold adjustment because it's local (per-cell), fast to compute,
  and preserves the Hopfield weight matrix exactly. Synaptic scaling
  (multiplying all incoming weights) would modify W = ΞᵀΞ, potentially
  breaking the symmetric structure.
- **Very slow timescale**: rate_decay ≈ 0.999 means the running average
  integrates over ~1000 steps. This prevents homeostasis from interfering
  with fast Hebbian dynamics — Hebbian learning selects patterns on each
  step, homeostasis corrects long-term drift over many episodes.
- **Bounded gain**: The gain factor is clamped to [0.9, 1.1] per step
  to prevent catastrophic over-correction. Over many steps, the
  cumulative effect can be large, but each step's adjustment is gentle.
- **Per-layer target rates**: L4 (feedforward) may need higher target
  rates than L2/3 (sparse representation). The target_rate is
  configurable per layer.

**Relationship to Hopfield dynamics**: Homeostatic threshold adjustment
adds a **bias term** to the Hopfield energy function:

  E(x) = -log Σ exp(β ⟨ξᵢ, x⟩) + ½β‖x‖² + Σⱼ θⱼ xⱼ

where θⱼ is cell j's homeostatic threshold offset. This is still a
valid energy function — the bias terms don't affect the convexity of
the softmax term or the convergence of the settling dynamics. They
shift which attractors are preferred (overactive cells get biased away
from firing, underactive cells get biased toward it), redistributing
the attractor basin volumes more evenly across stored patterns.

**Biological basis**: Turrigiano (2008) — "The Self-Tuning Neuron:
Synaptic Scaling of Excitatory Synapses"; Turrigiano (2012) —
"Homeostatic Synaptic Plasticity: Local and Global Mechanisms for
Stabilizing Neuronal Function."

**Tests** (after homeostatic implementation — same 3 scenarios must
now PASS):

1. **No runaway excitation**: After 50-object training, cells in >10
   patterns have activation specificity < 0.2 (fire for < 20% of
   objects). Homeostatic threshold increase prevents hub cells from
   becoming non-discriminative.
2. **No silent cell death**: < 5% of L2/3 cells have zero firing rate
   over 100 evaluation steps. Homeostatic threshold decrease recruits
   previously silent cells into new patterns.
3. **Scale stability**: Coefficient of variation of mean firing rates
   across 8 parallel columns < 0.15 after 50-object training.
   Homeostasis ensures each column converges to similar activity levels
   regardless of initialization.

**Additional unit tests**:
- Running rate converges to target rate over ~1000 steps for a cell
  with constant drive
- Threshold increases for overactive cells, decreases for silent cells
- rate_decay parameter controls integration timescale as expected
- Per-layer target rates are independently configurable
- Gain modulation is bounded within [0.9, 1.1] per step
- Disabling homeostasis (lr_homeo=0) reproduces pre-homeostatic behavior
- Hopfield settling converges with homeostatic bias terms
  (energy monotonically decreases)

---

### A10: Short-Term Synaptic Plasticity (STP)

Every chemical synapse in cortex exhibits activity-dependent changes in
transmission efficacy on fast timescales (~100ms–1s). Unlike STDP/Hebbian
learning (which changes stored weights), STP modulates the *effective*
weight on each step without altering the underlying learned connections.
This provides automatic temporal filtering: depression-dominated synapses
act as novelty detectors (strong response to new input, weak to repeated);
facilitation-dominated synapses act as temporal integrators (build up
response to sustained input).

**Limitations of static synaptic weights** (tests that MUST FAIL before
STP):

1. **No automatic novelty detection**: Present the same input pattern 5
   times consecutively, then present a novel pattern. With static
   weights, both the 5th repetition and the novel input produce the same
   activation strength — there is no activity-dependent adaptation.
   Expected: ratio of novel response to 5th-repetition response is
   ≈ 1.0 (no novelty enhancement, ratio within [0.9, 1.1]).

2. **No temporal integration of sustained drive**: Present a weak but
   sustained excitatory signal to PV+ cells for 20 steps (simulating
   a prolonged stimulus). With static E→PV+ weights, PV+ activation
   is constant across all 20 steps — no buildup. Expected: PV+
   activation at step 20 is NOT significantly higher than at step 1
   (ratio < 1.2). In reality, PV+ should progressively increase as
   facilitation builds, providing stable gamma ramp-up.

3. **No adaptation to noise**: Present a stream of meaningful patterns
   interspersed with a repeated noise pattern (M₁, N, M₂, N, M₃, N...).
   With static weights, the noise pattern N drives the same response at
   every presentation. Expected: response to the 5th presentation of N
   is NOT significantly weaker than the 1st (ratio > 0.8). STP
   depression should automatically attenuate repeated noise.

**Architecture** — Tsodyks-Markram Model:

The standard computational model of STP (Tsodyks & Markram 1997) uses
two per-synapse state variables:

- **x** (depression): fraction of available neurotransmitter resources.
  Starts at 1.0. Each presynaptic spike depletes resources. Recovers
  toward 1.0 with time constant τ_D.
- **u** (facilitation): effective release probability. Starts at U
  (baseline). Each presynaptic spike increases u (residual calcium).
  Decays back to U with time constant τ_F.

```
# Per connection-type state variables (not per individual synapse —
# grouped by type for efficiency):
# x: (n_connections,) depression, initialized to 1.0
# u: (n_connections,) facilitation, initialized to U

# Recovery phase (every step):
x = x + (1 - x) * dt / tau_D    # resources recover
u = u + (U - u) * dt / tau_F    # calcium decays

# Transmission phase (on presynaptic activity):
u_eff = u + U * (1 - u)         # transient calcium increase
effective_weight = W * u_eff * x # actual postsynaptic effect
x = x * (1 - u_eff * pre_rate)  # resources depleted proportional to activity
```

The effective synaptic weight is `W * u_eff * x`, where W is the
learned (Hebbian/STDP) weight, u_eff is the facilitation factor, and
x is the depression factor. STP modulates transmission without
touching the stored weight W.

**Connection-type-specific STP profiles**:

| Connection | U (baseline prob) | τ_D (depression recovery) | τ_F (facilitation decay) | Dominant mode |
|---|---|---|---|---|
| E→E (basal, apical) | 0.5 | 10 steps (~200ms) | 0.5 steps (~10ms) | Depression |
| E→PV+ | 0.1 | 2 steps (~40ms) | 8 steps (~160ms) | Facilitation |
| E→SST+ | 0.2 | 5 steps (~100ms) | 5 steps (~100ms) | Balanced |
| E→VIP+ | 0.15 | 3 steps (~60ms) | 6 steps (~120ms) | Facilitation |

- **E→E depression-dominant**: High initial release probability (U=0.5)
  depletes resources quickly. Novel input → strong first response →
  rapid depression → repeated input produces progressively weaker
  responses. This is an automatic **high-pass temporal filter** /
  novelty detector. In Hopfield terms: the first presentation of a
  pattern gets full-strength retrieval; repeated re-presentation
  gets attenuated retrieval, naturally shifting attention to novel
  features.

- **E→PV+ facilitation-dominant**: Low initial release probability
  (U=0.1) means weak initial PV+ drive. Sustained excitatory input
  gradually builds calcium → u increases → PV+ drive ramps up. This
  provides a **temporal integrator** for PV+ inhibition: initial
  excitation passes through with minimal inhibition (allowing fast
  exploration), then gamma stabilizes as PV+ facilitation builds.
  This replaces the need for hand-tuned "gamma ramp-up" parameters.

**Relationship to Hopfield dynamics**: STP creates a time-varying
effective weight matrix W_eff(t) = W ⊙ U(t) ⊙ X(t) where ⊙ is
element-wise product. The stored weights (W = ΞᵀΞ for L2/3 recurrence)
are UNCHANGED. STP is equivalent to a time-varying query transformation:
instead of retrieving with x, we retrieve with a modulated query where
recently-presented features are de-emphasized (depression) and sustained
context features are amplified (facilitation). The Hopfield energy
function remains valid — STP modulates the input, not the energy
landscape.

**Biological basis**: Tsodyks & Markram (1997) — "The neural code
between neocortical pyramidal neurons depends on neurotransmitter
release probability"; Zucker & Regehr (2002) — "Short-term synaptic
plasticity"; Abbott & Regehr (2004) — "Synaptic computation."

**Tests** (after STP implementation — same 3 scenarios must now PASS):

1. **Automatic novelty detection**: After 5 repetitions of the same
   input, present a novel pattern. Novel response / 5th-repetition
   response ratio > 2.0. E→E depression attenuates repeated input,
   making novel input relatively stronger.
2. **PV+ temporal integration**: Sustained 20-step excitatory drive.
   PV+ activation at step 20 is > 2× step 1 activation due to E→PV+
   facilitation buildup. Gamma stabilizes progressively rather than
   appearing instantly.
3. **Noise adaptation**: After 5 presentations of noise pattern N,
   response to N is < 50% of initial response (depression). Response
   to novel meaningful patterns M is NOT attenuated (different
   synapses, no depression). Discrimination ratio (M/N) > 3.0 by
   the 5th cycle.

**Additional unit tests**:
- Depression: repeating same pre-synaptic input N times decreases
  effective weight monotonically
- Facilitation: repeating same pre-synaptic input N times increases
  effective weight monotonically (for facilitating synapses)
- Recovery: after input stops, x recovers toward 1.0 with τ_D,
  u recovers toward U with τ_F
- Connection-type parameters: E→E is depressing, E→PV+ is facilitating
- STP does not modify stored weights W (only effective transmission)
- Disabling STP (U=1.0, τ_D=∞, τ_F=0) reproduces static weight behavior
- STP + Hopfield: settling converges with STP-modulated effective weights

**Interaction with other features**:
- **STDP (A5)**: STDP modifies the *stored* weight W. STP modulates
  the *effective* weight per-step. They are independent: STDP
  determines what the column has learned, STP determines how strongly
  that learning is expressed given recent activity.
- **Homeostasis (A9)**: Both are stabilizing mechanisms but at different
  timescales. Homeostasis operates over ~1000 steps (adjusting
  thresholds for long-term balance). STP operates over ~1-10 steps
  (adapting transmission for immediate temporal context). They don't
  interfere because their timescales are well-separated.
- **Oscillations (A8)**: E→PV+ facilitation means gamma ramp-up is
  automatic — PV+ inhibition builds gradually as facilitation
  increases, producing a natural onset envelope for gamma oscillations.
  This replaces any need for hand-tuned ramp parameters.
- **VIP+ disinhibition (A4)**: Novel input → strong E→E response
  (no depression yet) AND VIP+ activation (novelty signal). Together
  these "open up" the column for learning. Familiar input → depressed
  E→E (weak) AND low VIP+ (SST+ gating active). Natural synergy.

---

## Implementation Phases

### Phase 1: Laminar Cell Types (A1)

Restructure `CorticalColumnTorch` to have layered cell populations.
Initially just separate L4, L2/3, L5a, L5b, L6a, L6b with distinct
connectivity but same learning rules. This is a refactor of the cell
layout, not new computation. L5a (thick-tufted, subcortical/motor
output) and L5b (thin-tufted, cortical feedforward) get separate output
pathways. L6a (corticothalamic) drives the thalamic relay; L6b
(corticocortical) provides the inter-area feedback signal for Track 11.

**Tests**: All Track 9 tests still pass. Layer-specific activation
patterns visible in diagnostics. L5a and L5b produce distinct outputs.
L6a and L6b have separate projection targets.

---

### Phase 2: Multi-Head Dendritic Attention (A2)

Add per-branch computation with nonlinear integration to
`SparseDendrites`. Configurable n_heads per cell.

**Tests**: Multi-head improves context discrimination on synthetic
patterns. n_heads=1 reproduces Track 9.

---

### Phase 3: Plateau Potentials + Context Window (A3)

Add decaying plateau state to L5a cells (thick-tufted, with long apical
dendrites reaching L1). Integrate into Hopfield retrieval query. L5a
cells are the primary site for plateau potentials because their
extensive L1 apical tuft provides the coincidence detection surface
for BAC firing (basal + apical calcium spike).

**Tests**: Plateau-enriched retrieval outperforms memoryless retrieval
on sequential recognition tasks.

---

### Phase 4: Inhibitory Interneurons (A4)

Add PV+, SST+, VIP+ populations. Wire SST+ to dendritic branch gating.
Wire VIP+ to SST+ disinhibition. Wire ACh/novelty to VIP+.

**Tests**: Full PV → SST → VIP circuit functional. High novelty unmasks
branches. Low novelty gates irrelevant branches.

---

### Phase 5: STDP + Eligibility Traces (A5, A6)

First: run the four limitation tests (directed sequence recall, motion
direction discrimination, ambiguous sequence disambiguation, temporal
prediction error) with symmetric Hebbian and verify they FAIL. Then:
replace symmetric Hebbian with trace-based STDP, add eligibility traces
bridging STDP to neuromodulator-gated consolidation. Verify the same
four tests now PASS.

**Tests**: Limitation tests fail → implement → limitation tests pass.
Delayed reward consolidates eligible synapses.

---

### Phase 5b: Oscillatory Phase Coding (A8)

First: run the three interval limitation tests (interval discrimination,
rhythm reproduction, duration-sensitive recognition) with STDP alone
and verify they FAIL. Then: extend PV+ inhibition from instantaneous
top-k to dynamic E-I oscillation (emergent gamma), add theta pacemaker
from thalamocortical/motor input, implement theta-gamma coupling and
phase-augmented Hopfield patterns. Verify the same three tests now PASS.

**Tests**: Interval limitation tests fail → implement → interval
limitation tests pass. PV+ dynamics produce measurable gamma. Theta
scales with motor step rate. Phase-augmented Hopfield retrieval
distinguishes same-feature, different-timing patterns.

---

### Phase 5c: Homeostatic Plasticity + Short-Term Plasticity (A9, A10)

Add per-cell running firing rate estimates and homeostatic threshold
adjustment (A9). Add per-connection-type STP state variables
(depression x, facilitation u) with Tsodyks-Markram dynamics (A10).
Homeostatic plasticity operates on the ~1000-step timescale, STP on
the ~1-10 step timescale. Both are stabilizing mechanisms that
preserve the Hopfield weight matrix.

First: run the six limitation tests (3 for homeostasis + 3 for STP)
and verify they FAIL with the current system. Then implement both
mechanisms and verify the tests PASS.

**Tests**: Homeostatic limitation tests fail → implement → pass.
STP limitation tests fail → implement → pass. Running rates converge
to target. E→E depression attenuates repeated input. E→PV+
facilitation ramps up PV+ drive.

---

### Phase 6: Thalamocortical Loop (A7)

Add thalamic relay unit per column. Wire L6a → thalamus → L4.
Implement predictive gating. L6a (corticothalamic) cells drive the
thalamic gate; L6b is left available for Track 11 inter-area feedback.

**Tests**: L6a feedback attenuates predicted input. Surprise opens gate.
L6b output is independent of thalamic relay modulation.

---

### Phase 7: Full Eval

End-to-end evaluation of laminar column on YCB objects through
`MontyObjectRecognitionExperiment`, verifying that intra-column
biological features improve recognition accuracy and temporal
sequence learning vs Track 9 baseline.

**Integration tests**:

1. **Laminar column, 5 YCB objects, train + eval**: Verify laminar
   pipeline (L4→L2/3→L5a+L5b→L6a+L6b) produces correct recognition.
   L5a and L5b produce distinct output signals.
2. **Multi-head dendrites improve discrimination**: Same 5 objects,
   compare n_heads=1 vs n_heads=8. Multi-head should have higher
   accuracy on similar objects.
3. **Plateau context window aids sequential recognition**: Present
   same object from multiple viewpoints. Plateau-enabled column
   should converge faster than memoryless.
4. **STDP directed sequence recall on saccade paths**: Train on 5 YCB
   objects with consistent saccade trajectories. At eval, present the
   first saccade viewpoint only. STDP column predicts the next viewpoint
   features; symmetric Hebbian column cannot. STDP accuracy > 80%,
   symmetric Hebbian < 40%.
5. **Thalamocortical gating reduces noise**: Add sensor noise. Column
   with thalamic gating should be more robust than without.
6. **Phase coding enables speed-sensitive recognition**: Train on
   animated Fox.glb at two speeds (normal and 2× fast). Column with
   phase coding distinguishes the two speeds; column without phase
   coding cannot. Phase-coded accuracy > 75%, non-phase-coded ≈ 50%.
7. **Homeostatic stability under scale**: Train on 50 objects across
   8 parallel columns. Homeostasis-enabled columns show < 5% silent
   cells and < 0.15 CV of firing rates across columns. Without
   homeostasis: > 25% silent cells and > 0.5 CV.
8. **STP novelty enhancement on repeated vs novel**: Train on 5 objects,
   then present the same object 5 times followed by a novel object.
   STP-enabled column’s response to the novel object is > 2× the 5th
   repetition. Without STP: ratio ≈ 1.0.

**Acceptance criteria**:
- All 8 integration tests pass
- Laminar column with all features enabled outperforms Track 9 baseline
  on 5-object YCB benchmark
- Each biological feature can be independently enabled/disabled via
  feature flags
- All tests use real Panda3D rendering with real .glb meshes

---

## Execution Order & Dependencies

```
Track 9 complete
  └→ Phase 1 (laminar structure: L4, L2/3, L5a, L5b, L6a, L6b)
       ├→ Phase 2 (multi-head dendrites)
       ├→ Phase 3 (plateau potentials — on L5a thick-tufted cells)
       ├→ Phase 4 (inhibitory interneurons + gap junctions)
       ├→ Phase 5 (STDP + eligibility)
       │    └→ Phase 5b (oscillatory phase coding — depends on 4 + 5)
       ├→ Phase 5c (homeostatic plasticity + STP)
       └→ Phase 6 (thalamocortical loop — L6a drives gate)
            └→ Phase 7 (full eval)
```

Phases 2, 3, 4, 5, 5c, 6 can be developed in parallel once Phase 1 is done.
Eligibility traces (Phase 5) need DA/reward and ACh/novelty signals,
which are already provided by Track 9 Phase 7 (`NeuromodulatoryGating`);
they do not depend on the VIP+ interneuron circuit (Phase 4).
Phase 5b depends on BOTH Phase 4 (gamma oscillation emerges from PV+
dynamics) AND Phase 5 (phase coding augments STDP-learned sequences
with interval information). Phase 5c (homeostatic + STP) depends only on
Phase 1 (needs distinct layers and connection types for per-layer target
rates and per-connection-type STP parameters). Phase 6 depends on 1
(needs L6a layer). Phase 7 depends on all previous phases.

---

## File Structure

```
src/tbp/monty/frameworks/models/cortical_column_torch/
├── (Track 9 files unchanged)
├── layers.py                 # L4, L23, L5a, L5b, L6a, L6b layer modules
├── interneurons.py           # PV+, SST+, VIP+ populations + gap junctions
├── plateau.py                # Plateau potential working memory (on L5a)
├── stdp.py                   # STDP + eligibility traces
├── oscillator.py             # Oscillatory phase coding
├── homeostasis.py            # Homeostatic / intrinsic plasticity
├── stp.py                    # Short-term synaptic plasticity (Tsodyks-Markram)
└── thalamic_relay.py         # Thalamocortical loop gating (via L6a)

tests/unit/frameworks/models/test_cortical_column_torch/
├── (Track 9 tests unchanged)
├── test_layers.py
├── test_interneurons.py      # includes gap junction coupling tests
├── test_plateau.py
├── test_stdp.py
├── test_stdp_limitations.py  # Symmetric Hebbian failure cases
├── test_oscillator.py
├── test_oscillator_limitations.py  # STDP-only failure cases
├── test_homeostasis.py
├── test_homeostasis_limitations.py  # Unregulated Hebbian failure cases
├── test_stp.py
├── test_stp_limitations.py   # Static weight failure cases
└── test_thalamic_relay.py

tests/integration/frameworks/models/
└── test_laminar_column_monty.py
```

---

## Principles

1. **Additive, not destructive**: Each phase adds to the column without
   breaking previous phases. All Track 9 tests pass throughout.
2. **Feature flags**: Every new biological feature defaults off. Enabling
   laminar=True, multi_head=True, stdp=True, etc. progressively adds
   complexity. The minimal config reproduces Track 9 exactly.
3. **Biology drives architecture**: Every computational choice maps to a
   specific biological mechanism with cited experimental evidence. No
   "it works better so we added it" without a biological counterpart.
4. **GPU-first**: All new computation is expressed as batched tensor
   operations. No Python loops over cells, segments, columns, or areas
   in the inner loop.
5. **Monty integration**: The laminar column is still wrapped by
   `CorticalColumnTorchLM`. No changes to the Monty interface.
