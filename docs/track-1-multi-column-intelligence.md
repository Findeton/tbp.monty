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
  - Shape-congruent categories: **92.2%** (near ceiling for geometry matching)
  - Shape-incongruent categories: **14.3%** (irreducible by shape; requires Layer 2)
- **Novelty detection (T1.N)**: Validated. Evidence margin separates confident
  predictions (86.2% accuracy) from uncertain (50.4%), 35.8pp separation.
- **Feature ablation (T1.FA)**: Complete. Curvature is the primary discriminative
  feature. HSV has mixed effects. Pose vectors carry zero discriminative signal.
  No ablation improves overall accuracy — baseline is near-optimal for Layer 1.
- **Category bias (T1.4a)**: Tested and FAILED. Hurts overall (67.9% → 61.2%).
  Destroys fruit (100→64%), only helps clamp and can slightly.
- **CMP enrichment (T1.R2)**: `register_dynamic_feature()` and
  `set_feature_weight()` implemented. Ready for use by Track 2.
- Layer 1 ceiling empirically confirmed at 67.9%; priority shifts to Track 2.

## Milestone Tracker

| ID | Milestone | Restriction needed | Status |
|---|---|---|---|
| T1.1 | Static peer voting | None | **Done** |
| T1.2 | Category-aggregated evidence readout (post-hoc) | None | **Done** |
| T1.N | Novelty detection via evidence margin | None | **Done** (86.2% vs 50.4%) |
| T1.FA | Feature ablation (HSV/curvature contribution) | None | **Done** (5 runs, 1120 eps) |
| T1.4a | Category-biased evidence (single LM, self-bias) | None | **Done** (failed: -6.7pp) |
| T1.3 | Online category readout during matching | None | Deprioritized |
| T1.R2 | CMP enrichment (lifts Restriction 2) | — | **Done** (API added) |
| T1.4b | Cross-LM category bias via enriched CMP | Restriction 2 | Deprioritized |
| T1.5 | Top-down biasing from parent to child LMs | Restriction 2 | Planned → **Track 6** (T6.9–T6.12) |
| T1.6 | Attention-based dynamic routing | Restriction 2 | Planned → **Track 6** (T6.13–T6.16) |
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
   into shape-congruent (92.2%, near ceiling) vs shape-incongruent (14.3%,
   irreducible by geometry). Ball→fruit (0%), screwdriver→spatula, can→box
   confusions are geometrically correct — they mark the boundary where shape
   matching ends and behavioral/functional knowledge begins.

7. **Evidence margin IS a reliable confidence signal (2026-03-21, T1.N).**
   Novelty detection validated: confident predictions (margin ≥ 0.65) achieve
   86.2% accuracy vs 50.4% for uncertain predictions (35.8pp separation).
   Implemented in `logging_utils.py` and `phase2_category_evidence_readout.py`.

8. **Category bias HURTS overall accuracy (2026-03-21, T1.4a).**
   Prediction was: helps shape-congruent, hurts shape-incongruent. Actual
   result: hurts BOTH. Overall: 67.9% → 61.2% (-6.7pp). Fruit destroyed
   (100% → 64.3%). Only clamp (71→86%) and can (21→29%) improved. The
   normalized category evidence creates misleading signals when categories
   have different geometric distributions. T1.4a is a dead end on this
   benchmark.

9. **Feature ablation confirms curvature is king (2026-03-21, T1.FA).**
   Five eval runs (1120 episodes total):
   - Pose vectors alone: 0% (no discriminative signal for identity)
   - No curvature: shape-incongruent drops 14.3% → 5.7%
   - No HSV: mixed effects, net slightly negative
   - Baseline (all features): 67.9% — already near-optimal
   No ablation variant improves overall accuracy. The feature set is
   at its ceiling.

10. **CMP enrichment is ready (2026-03-21, T1.R2).**
    `register_dynamic_feature()` and `set_feature_weight()` added to
    EvidenceGraphLM. Runtime feature registration with auto-tolerance.
    Ready for Track 2 to inject behavioral features.

## Next Steps

Updated 2026-03-21. Track 1 Layer 1 work is complete. All milestones through
T1.R2 are done. The remaining Track 1 milestones (T1.3, T1.4b, T1.5, T1.6,
T1.7) are deprioritized because:
- T1.4a proved that category bias hurts on extended YCB
- T1.3 (online category readout) feeds into T1.4a which failed
- T1.4b (cross-LM bias) depends on T1.4a which failed
- The Layer 1 ceiling (67.9%) is confirmed by 5 ablation runs

**Priority shifts to Track 2** (cross-episode memory, behavioral concepts).
The ball→fruit canary (0% across all ablations) is the primary gate metric.
The HPC→evidence modulation is wired (`_apply_hippocampal_bias()`); the next
step is creating a functional test with interaction episodes.

**Track 1 remaining value**: T1.5 (top-down biasing) and T1.6 (attention
routing) are now planned in [Track 6](track-6-predictive-coding-heterarchy.md)
as part of the unified predictive coding heterarchy design (T6.9–T6.12 for
top-down prediction, T6.13–T6.16 for attention routing). T1.R2's dynamic
feature registration enables this path.

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
