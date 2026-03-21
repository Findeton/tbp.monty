# Track 1: Multi-Column Intelligence

## Purpose

Develop the multi-LM voting, routing, and consensus mechanisms that enable
category generalization, compositional reuse, attention, and abstraction.

This is the most proven track. The +10pp result on Omniglot came from lateral
voting + category-aggregated evidence readout — both multi-column mechanisms.

## Roadmap

See [full-system-roadmap-v2.md](full-system-roadmap-v2.md) for the full program
context, restriction analysis, concept layers, and priority stack.

## Current Status

Updated 2026-03-20 with extended YCB evidence boundary analysis.

- **Best result**: +10pp alphabet accuracy on scaled Omniglot (60%→70%)
  via 2-LM voting + cross-LM category-aggregated evidence readout
- **Convergence speed**: 44% faster with 3 LMs vs 1 LM on YCB
- **Category subgraph win rate**: 25% on Omniglot (highest of any benchmark)
- **Extended YCB benchmark**: 67.9% overall, decomposed into:
  - Shape-congruent categories: **93.2%** (near ceiling for geometry matching)
  - Shape-incongruent categories: **15.7%** (irreducible by shape; requires Layer 2)
- **Evidence margin signal**: Proven predictive of correctness (high margin → right,
  low margin → wrong). Not yet exploited as a novelty detection mechanism.
- Layer 1 ceiling empirically measured; priority shifts to Layer 1→2 boundary

## Milestone Tracker

| ID | Milestone | Restriction needed | Status |
|---|---|---|---|
| T1.1 | Static peer voting | None | **Done** |
| T1.2 | Category-aggregated evidence readout (post-hoc) | None | **Done** |
| T1.N | Novelty detection via evidence margin | None | **Next** |
| T1.FA | Feature ablation (HSV/curvature contribution) | None | **Next** |
| T1.3 | Online category readout during matching | None | After T1.N |
| T1.4a | Category-biased evidence (single LM, self-bias) | None | After T1.FA (risk: see roadmap) |
| T1.R2 | CMP enrichment (lifts Restriction 2) | — | After T1.4a |
| T1.4b | Cross-LM category bias via enriched CMP | Restriction 2 | After T1.R2 |
| T1.5 | Top-down biasing from parent to child LMs | Restriction 2 | After T1.R2 |
| T1.6 | Attention-based dynamic routing | Restriction 2 | Planned |
| T1.7 | Learned routing from evidence statistics | Restriction 2 | Research |

## Investigation Registry

Investigations are listed in chronological order. Each contains full technical
detail (benchmarks, execution registry, decision log, results).

| Investigation | Status | Document | Key Result |
|---|---|---|---|
| Composition (Phase 1) | Completed (negative) | [teacher-student-compositional-plan.md](teacher-student-compositional-plan.md) | Monolithic 50.6% > compositional 39.9%; but compositional showed stronger child-reuse signal (75 vs 24) |
| Categories v1 (Phase 2 original) | Completed (negative) | [phase-2-category-and-concept-generalization.md](phase-2-category-and-concept-generalization.md) | 46+ experiments; all hierarchical, SDR, and scoring approaches failed; degenerate parent representations (>0.9998 cosine similarity) |
| Categories v2 (Phase 2 rebuilt) | **Active** | [phase-2-category-and-concept-generalization-v2.md](phase-2-category-and-concept-generalization-v2.md) | +10pp via voting + category readout; category subgraphs validated; HippocampalModule reconstructed |

## Key Findings

1. **Hierarchy failed; lateral voting succeeded.** 46 hierarchical experiments
   produced 0pp improvement. 2-LM lateral voting produced +10pp. This is the
   most important empirical finding in the program.

2. **Category subgraphs are structurally valid but can't compete in evidence.**
   Instance graphs always outcompete category subgraphs (more nodes = more
   evidence). Category subgraphs work as shared voting vocabulary (25% win rate
   when they're the only representation for a category in an LM).

3. **The category signal is already in the evidence distribution.** The +10pp
   came from summing evidence by category — a pure readout on existing data.
   No new encoding, scoring, or transport mechanism was needed.

4. **Voting diversity matters more than voting quantity.** Disjoint instance
   splits (+10pp) outperform overlapping splits (-25pp). Redundancy kills
   consensus quality.

5. **Convergence speed scales with LMs.** 42→27→24 steps for 1→2→3 LMs.
   More columns = faster convergence via parallel hypothesis elimination.

6. **Layer 1 has a measured ceiling (2026-03-20).** Extended YCB decomposes
   into shape-congruent (93.2%, near ceiling) vs shape-incongruent (15.7%,
   irreducible by geometry). Ball→fruit (0%), screwdriver→spatula, can→box
   confusions are geometrically correct — they mark the boundary where shape
   matching ends and behavioral/functional knowledge begins.

7. **Evidence margin is an unexploited confidence signal.** High margin
   (>6.0 per-object-normalized) → always correct. Low margin (<3.0) → always
   wrong. The system has the information to detect novelty but no mechanism
   to act on it.

8. **Category bias is a double-edged sword.** It amplifies the strongest
   category signal, which is WRONG for shape-incongruent categories (balls
   → fruit bias gets reinforced). Any T1.4a experiment must measure both
   regimes separately.

## Next Steps

Updated 2026-03-20 based on extended YCB evidence boundary analysis.

**T1.N: Novelty detection via evidence margin.**
The extended YCB data proves evidence margin predicts correctness. Threshold on
`(best_cat - second_cat) / best_cat` enables "novel object, category X,
confidence high/low." ~10 lines in evidence readout. Tests on both Omniglot
and extended YCB. Qualitatively new capability: the system knows what it
doesn't know.

**T1.FA: Feature ablation study.**
Run eval with modified feature_weights: (a) all features (baseline), (b) no
HSV, (c) HSV amplified, (d) no curvature. Measure impact on shape-congruent
vs shape-incongruent categories separately. No retraining needed (graphs
already contain all features). Directly informs T1.R2 design decisions.

**T1.4a: Online category bias (with risk assessment).**
Prediction: improves shape-congruent (93.2% → 96-98%), degrades shape-
incongruent (15.7% → lower). If prediction holds, mechanism understood.
Must measure both splits separately on extended YCB, not just overall.
Also run on Omniglot for comparison.

**Then T1.R2: CMP enrichment (lifts Restriction 2).** Small change (~20 lines):
make feature_weights/tolerances updatable at runtime, inject LM outputs into
observation dict. Informed by T1.FA (which features matter) and T1.4a (what
signal to pass between LMs). Unlocks T1.4b (cross-LM bias), T1.5 (top-down),
and hippocampal context → matching LMs.

## Extended YCB Evidence Boundary (2026-03-20)

The extended YCB benchmark (50 objects, 9 categories, 34 train / 16 holdout)
quantifies where Layer 1 ends. Normalized evidence matrix (avg per training
object):

```
           fruit   ball    box    can    cup  utensil  tool   clamp  airplane
fruit      24.3*   17.5    4.6    4.8    5.8    1.2    6.4    3.8    12.8
ball       14.5*    7.8    4.9    6.8    7.0    2.2    9.2    5.9    13.2
box         3.5    3.2   14.0*    6.9    4.6    4.2    5.6    6.6     6.1
can         2.7    3.5   10.4*    7.5    5.2    3.4    5.1    4.4     6.3
cup         1.7    2.0    3.4    4.9   12.0*    1.9    3.0    2.6     3.6
utensil     2.5    2.3    4.3    4.0    3.4  108.8*    8.5    5.5     5.9
tool        4.7    3.3    5.8    4.8    3.8   13.0*   11.2   11.5     9.8
clamp       2.8    3.3    4.9    4.5    3.6    6.2    5.5   13.5*     4.4
airplane    1.5    0.8    1.4    1.5    1.6   10.2    7.1    6.3    18.8*
```

Rows = true category, cols = predicted. `*` = highest. Key observations:
- Ball row: fruit (14.5) dominates ball (7.8) → spheres match spheres
- Tool row: utensil (13.0) beats tool (11.2) → screwdrivers ≈ spatulas
- Can row: box (10.4) beats can (7.5) → rectangular can matches boxes
- Utensil margin (100.3) is enormous → knife matches fork/spoon perfectly

The ball→fruit confusion is the primary canary metric for Track 2 (Layer 2).

**Configs**: `extended_ycb_train.yaml`, `extended_ycb_eval_holdout.yaml`
**Results**: `~/tbp/results/monty/projects/phase2_review_runs/extended_ycb_eval_holdout/`
**Taxonomy**: `config/environment_interface/ycb/extended_taxonomy.yaml`

## HippocampalModule Connection

The HippocampalModule (665 lines, 31 tests) bridges Track 1 to higher concept
layers. Its association matrix accumulates co-activation patterns across
episodes — the same mechanism as category-aggregated readout, but over time
rather than within a single episode. See the Track 2 index for temporal
integration plans.

Source: `src/tbp/monty/frameworks/models/hippocampal_module.py`
