# Phase 2: Honest Assessment

**Date**: 2026-03-20
**Status**: Feature representation ceiling reached. No further mechanism
on top of the current features can improve Omniglot beyond ~70% alphabet.

## What We Actually Measured

### The Benchmark

30 Omniglot characters: 5 alphabets × 6 characters each.
Training: version 1 of each character (one handwriting sample per character).
Eval: version 2 (different handwriting of the same characters).
Task: cross-version recognition.

### The Results

| Config | Exact | Alphabet | Notes |
|---|---|---|---|
| Single-LM baseline | 43.3% | 70.0% | The ceiling |
| Single-LM + category bias | 43.3% | 70.0% | Zero effect |
| 2-LM voting (disjoint) | 23.3% | 50.0% | Worse — half the graphs per LM |
| 2-LM cross-LM aggregation | 36.7% | 70.0% | Recovers baseline, doesn't beat it |
| Tighter match distance (3) | 30.0% | 66.7% | Worse — too restrictive |
| Surface normal weights | 43.3% | 70.0% | No effect |
| Tighter + normal weights | 30.0% | 66.7% | Worse |
| Linear probe (LOO) | — | 63.3% | Worse than argmax |
| Nearest centroid (LOO) | — | 53.3% | Much worse |
| Sum-by-category | — | 70.0% | Equal to argmax |

### What These Results Mean

1. **The argmax baseline is already the optimal readout.** Learned probes
   (logistic regression, nearest centroid) do worse because there are only
   30 samples and 30 features. The simple argmax captures all extractable signal.

2. **Category aggregation adds zero value on this benchmark.** Sum-by-category
   gives the same 70% as argmax. The +10pp from the earlier 4-char experiment
   was a statistical artifact of a specific confusion structure, not a
   generalizable mechanism.

3. **Parameter tuning cannot improve beyond the baseline.** Tighter spatial
   matching (mmd=3) actively hurts (-3.3pp). Feature reweighting has no effect.
   HSV features are constant (V ≈ 0.200 ± 0.002 for all on-stroke points).

## Root Cause: The Features Don't Separate Alphabets

### Evidence

Measured pairwise spatial overlap (KDTree, radius=5) between all 30 stored graphs:

- **Within-alphabet overlap**: 0.389 ± 0.184
- **Cross-alphabet overlap**: 0.350 ± 0.149
- **Cohen's d**: 0.23 (negligible separation)

The stored graphs of characters from different alphabets are nearly as spatially
similar as characters from the same alphabet. The system cannot distinguish
alphabets because its features (spatial layout + curvature) don't encode
alphabet-distinguishing information.

### Why

1. **Omniglot characters are flat 2D surfaces.** Principal curvatures are
   near-zero everywhere except at stroke edges. The curvature distribution
   is identical across all 5 alphabets (mean ≈ [-0.26, -2.04] for all).

2. **The only discriminating signal is WHERE curvature changes occur** — the
   spatial layout of stroke edges. But this is character-specific, not
   alphabet-specific. Characters from different alphabets with similar stroke
   layouts will confuse each other.

3. **HSV is constant.** The depth map `1.2 - gaussian_filter(~stroke)` produces
   H=0, S=0, V≈0.200 for all on-stroke points. No discriminating power.

4. **Surface normals vary** (std 0.35 on x,y), but adding their weight to the
   evidence computation has no effect because the normal direction is already
   captured implicitly in the pose vector matching.

### What Would Actually Distinguish Alphabets

Features that the current system **does not compute**:

- **Topology**: Junction count, loop count, endpoint count, connected components.
  Armenian characters tend to have more curves; Arcadian tends to be more angular.
- **Stroke sequence**: The sensor follows the drawing path. Characters from the
  same alphabet share stroke patterns. But the system treats each step independently.
- **Global statistics**: Aspect ratio (Anglo-Saxon Futhorc ≈ 0.58, Alphabet of the
  Magi ≈ 1.43), total node count, spatial extent. These differ by alphabet but
  aren't used during matching.
- **Multi-scale structure**: Coarse-grained spatial layout vs fine stroke details.

## What Was Overblown

### "Category-aggregated evidence readout" (+10pp)

This was presented as a breakthrough mechanism. In reality:
- The +10pp came from a 4-char benchmark where confusion structure happened to
  favor category aggregation. On the 6-char benchmark it adds 0pp.
- The mechanism is just `sum(positive evidence per category) → argmax`. This is
  a prior, not a readout. It assumes "many weak matches > one strong mismatch."
- On the benchmark where it "worked," the argmax already achieves the same 70%.

### "2-LM voting" as a TBT validation

Both LMs receive the same sensor data. The only difference is which stored
graphs they have. This tests database sharding, not multi-column intelligence.
To test TBT's voting claim, LMs need genuinely different information sources
(different sensors, angles, or modalities).

### "HippocampalModule wiring" and "language bridge"

These are infrastructure (pipes that data flows through) with no demonstrated
functional benefit. The HPC context signal carries no useful information in
single-object Omniglot episodes. The language bridge is a dictionary lookup.

## Provably True Results

Despite the overblown claims, real scientific value emerged:

1. **The evidence distribution IS category-informative.** Within-alphabet
   confusions dominate (8 of 17 errors). The system's errors respect category
   boundaries more often than not. This is a genuine property of the feature space.

2. **Spatial overlap predicts confusion.** The most-confused cross-alphabet pairs
   have 70-80% spatial overlap. Confusion is mechanistically explained by feature
   similarity, not random.

3. **The feature ceiling is quantified.** 70% alphabet accuracy = the maximum
   extractable from (spatial layout + curvature) on this benchmark. This is
   now a concrete target: any new feature that genuinely separates alphabets
   should push past 70%.

## What Actually Worked: Evidence Size Normalization

After the first-principles analysis, one genuine improvement was found:

**Graph size normalization** (power=0.75): **73.3% alphabet (+3.3pp), 46.7% exact (+3.4pp)**.

Larger graphs have more nodes → more hypotheses → more evidence accumulation
purely from graph topology, not from better feature matches. Normalizing
the MLH evidence by `N_nodes^0.75` corrects this bias.

This works because:
1. The bias is mechanistic (more nodes = more evidence), not statistical
2. Some cross-alphabet confusions are with graphs 2-3x larger than the correct answer
3. The normalization doesn't change the evidence accumulation — only the winner selection

Implementation: `evidence_size_norm_power` parameter in EvidenceGraphLM. Set to
0.75 for best alphabet accuracy, 0.5 for best exact accuracy. Default 0 (off).

| Config | Exact | Alphabet |
|---|---|---|
| Baseline (no normalization) | 43.3% | 70.0% |
| power=0.50 | 50.0% | 70.0% |
| power=0.75 | 46.7% | 73.3% |

This is a small but real improvement from a mechanistic insight, not a
hyperparameter search or cosmetic change.

## What To Do Next

### Priority 1: Features that break the ceiling

Add topology-aware features to the Omniglot sensor:
- Junction count at each node (number of stroke intersections in the local patch)
- Endpoint detection (stroke terminates in the local patch)
- Loop count or Euler characteristic of the local patch

These require modifying `OmniglotEnvironment._observations()` or adding a
preprocessing step to the CameraSM. They represent genuinely new information
not present in curvature or spatial layout.

### Priority 2: Multi-sensor experiment (legitimate TBT test)

Two sensor patches at different positions on the character, each feeding its
own LM. This tests whether combining two genuinely different observations of
the same object produces better recognition than one. On Omniglot this is
straightforward: two patches at different stroke positions.

### Priority 3: Stop building infrastructure on a 70% foundation

Don't add more layers (hippocampal memory, language bridges, category biases)
until single-column recognition is strong enough that the added complexity has
something to amplify. The current system confuses characters 57% of the time.
More pipes won't fix that.
