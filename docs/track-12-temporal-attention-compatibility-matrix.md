# Track 12 Temporal And Attention Compatibility Matrix

Repo-grounded snapshot of what is implemented now, what is default in current Track 12 runs, and which features compose cleanly versus conflict, shadow, or depend on each other.

Primary grounding:

- [track-12-cortical-time-and-predictive-state.md](track-12-cortical-time-and-predictive-state.md)
- [track12-benchmark-schematic-and-math.md](track12-benchmark-schematic-and-math.md)
- [track-10-deep-heterarchy-and-biological-depth.md](track-10-deep-heterarchy-and-biological-depth.md)
- [track-6-predictive-coding-heterarchy.md](track-6-predictive-coding-heterarchy.md)
- [track-1-multi-column-intelligence.md](track-1-multi-column-intelligence.md)
- [learning_module.py](../src/tbp/monty/frameworks/models/cortical_column_torch/learning_module.py)
- [column.py](../src/tbp/monty/frameworks/models/cortical_column_torch/column.py)
- [benchmark_track12_real_time_falsifiers.py](../scripts/benchmark_track12_real_time_falsifiers.py)

## How To Read This

- `Default in Track 12` means the current real-asset Track 12 falsifier benchmark turns it on by default.
- `Cross-compatible` means it can coexist in the same code path or same LM stack without a hard collision.
- `Conflict / dependency` is used for three different situations:
  - true exclusivity: one mechanism replaces or shadows another
  - soft dependency: a feature technically enables but is mostly inert without another flag
  - separate path: implemented elsewhere, but not wired into the default Track 12 benchmark
- `Compatible` here means code-level coexistence or explicit test coverage where available. It does not mean every pairwise combination has been benchmarked exhaustively.

## Short Compatibility Summary

| Topic | Current answer |
|---|---|
| Safest default temporal bundle | Trace bank `d_t` + action-conditioned prediction + inferred-state correction + hierarchical child/parent loop + conditional lateral voting |
| Main real exclusivity | `self_supervised_temporal_config` and `temporal_transition_config` are both constructible, but the self-supervised path wins inside `_update_temporal_state()`, so the transition-memory path is shadowed if both are set |
| Main soft dependencies | `use_eligibility` is only behaviorally meaningful with `use_stdp`; `use_interneurons` is mainly meaningful with `multi_head`; `use_thalamic_relay` is mainly meaningful with `laminar`; plateau memory matters most with `use_apical` |
| Main separate-but-implemented systems | `TemporalMemory`, `GlobalIntervalTimer`, `StateConditionedModel`, salience SM, Track 10 laminar stack |
| Main missing integration | Parent-to-child predictive path, surprise-gated upward/downward residual communication, unified predictive routing, dynamic attention routing |

## 1. Core Track 12 Runtime

| Feature | Status | Default in Track 12 | Cross-compatible | Conflict / dependency | Notes |
|---|---|---|---|---|---|
| Trace-bank temporal state `d_t` | Implemented | Yes | Compatible with action-conditioned prediction, inferred-state correction, `TemporalMemory`, self-supervised temporal memory, transition memory, lateral voting, parent LM | No hard conflict; this is the current primary online temporal carrier | This is the benchmarked `core_temporal_state_mode = "trace_bank_d_t"`, with `h_t = (x_t, d_t)` |
| Boundary pressure and trace discontinuity | Implemented | Yes | Compatible with omission, perturbation, action-blind replay, and temporal-status reporting | No hard conflict | Derived from surprise, prediction mismatch, and trace discontinuity; used as the main boundary-style diagnostic |
| Action-conditioned prediction and query bias | Implemented | Yes | Compatible with trace bank, boundary pressure, action-blind falsifier, parent/child hierarchy | If disabled, the active-vs-action-blind replay comparison loses most of its intended meaning | Wired through `action_predictive_config`; benchmark default uses it on both child LMs |
| Inferred-state correction and hidden-state geometry | Implemented | Yes | Compatible with trace bank and child parity defaults | Benchmark path, not a separate old timer path | Reported as `input_geometry_mode = "inferred_relative_hidden_state"` |
| Temporal status, predicted label, event signal, context export | Implemented | Yes | Compatible with trace bank and optional temporal learners | If no temporal learner is present, status falls back to surprise-based confident/confused heuristics | Exposed by `get_temporal_prediction_status()` and `get_temporal_context()` |
| Context-only parent stepping from child context | Implemented | Yes in hierarchical Track 12 runs | Compatible with current child/parent hierarchy and goal-state motor loop | Not the same thing as a missing predictive top-down `receive_prediction()` path | Parent LMs can step from received child context even without direct sensor input |
| Conditional lateral voting | Implemented | Yes | Compatible with the current Track 12 child pair and parent LM | Not equivalent to planned dynamic attention routing | Track 12 uses settle-gated conditional voting, not full predictive routing |
| Goal-state-driven motor control with parent authority | Implemented | Yes | Compatible with parent LFM and current hierarchy | Depends on parent LM authority; not a general attention router | Reported as `goal_state_driven_motor = true` and `goal_state_authority = ["lm_parent"]` |

## 2. Optional Temporal Learners And Older Temporal Systems

| Feature | Status | Default in Track 12 | Cross-compatible | Conflict / dependency | Notes |
|---|---|---|---|---|---|
| `TemporalMemory` | Implemented | No | Compatible with trace bank, action prediction, self-supervised temporal memory, and benchmark evidence modulation | No hard conflict | Within-episode SDR plus Hebbian sequence learner over raw sensory features |
| Temporal behavior evidence bias | Implemented | No | Compatible with trace bank and `TemporalMemory` | Depends on `TemporalMemory`; without it this path is inert | Uses behavior scoring to temporarily bias object evidence |
| `SelfSupervisedSemiMarkovMemory` | Implemented | No in the default Track 12 child preset, yes in specific self-supervised/Track 9b paths | Compatible with trace bank, action prediction, and `TemporalMemory` | Shadows `SemiMarkovTransitionMemory` inside `_update_temporal_state()` if both are configured | Online HSMM-like latent-state discovery over embeddings; not the default inference path in current Track 12 |
| Self-supervised temporal evidence / transition / prediction boosts | Implemented | No | Compatible with trace bank and self-supervised memory | Depends on `SelfSupervisedSemiMarkovMemory`; otherwise inert | These are evidence-bias knobs, not separate memories |
| `SemiMarkovTransitionMemory` | Implemented | No | Compatible with trace bank and `TemporalMemory` | Mutually exclusive as the active temporal-state owner whenever self-supervised memory is also configured, because the self-supervised branch is checked first | Discrete label plus dwell-time model over settled LM labels |
| Fully self-supervised LM object-identity mode | Implemented | No | Compatible with trace bank and Track 12 benchmark harness | Changes training and decoding assumptions; introduces graph-ID ambiguity that is resolved post hoc by joint child decoder then parent fallback | This is a benchmark mode switch, not the default named-object Track 12 training path |
| `GlobalIntervalTimer` | Implemented | No | Separate from the current Track 12 default stack | Separate older timing path; not integrated into the current default trace-bank inference loop | Global time-cell-like timer with multi-LM reset and speed adjustment |
| `StateConditionedModel` | Implemented | No | Separate explicit-state modeling path | Separate older object-state path; not the default Track 12 temporal carrier | One object with multiple discrete state-specific subgraphs |

## 3. Attention And Biological-Depth Mechanisms

| Feature | Status | Default in Track 12 | Cross-compatible | Conflict / dependency | Notes |
|---|---|---|---|---|---|
| Modern Hopfield settling (`ModernHopfieldMemory`) | Implemented | Yes | Core with all current column modes | No conflict | The main attractor retrieval mechanism; this is the core non-transformer attention-like retrieval primitive |
| Hopfield associative-memory softmax readout | Implemented | Yes | Core with all current temporal modes | No conflict | Softmax attention over object prototypes in `HopfieldAssociativeMemory.recall()` |
| Location-feature memory partial-cue retrieval | Implemented | Parent LFM is enabled in Track 12 benchmark | Compatible with hierarchy, goal-state motor, and reference-frame estimation | Not a transformer router; separate optional memory path | Supports feature-to-location, location-to-feature, and full partial-cue retrieval |
| LFM-driven goal-state targeting | Implemented | Yes on the parent side in Track 12 benchmark | Compatible with parent authority and goal-state motor loop | Depends on LFM plus goal-state-driven motor control | Used to turn retrieved likely locations into motor targets |
| Laminar mode (`laminar=True`) | Implemented | No | Compatible with multi-head dendrites, plateau, interneurons, STDP, eligibility, phase coding, thalamic relay | No known hard conflict; explicit tests run all Track 10 features together | This is the main Track 10 structural switch |
| Multi-head dendritic attention | Implemented | No | Compatible in flat and laminar modes; compatible with plateau, STDP, phase coding, thalamic relay | No hard conflict | True multi-head dendritic branch integration, not QKV transformer attention |
| Plateau-potential working memory | Implemented | No | Compatible with flat or laminar mode | Behaviorally strongest with `use_apical`; otherwise much weaker or effectively inert | Used to enrich the Hopfield query via apical-plus-basal coincidence |
| PV/SST/VIP interneuron circuit | Implemented | No | Compatible with laminar or flat Track 10 stack | Mostly meaningful when `multi_head=True`, because SST/VIP branch gating acts on dendritic heads | Provides PV competition plus SST/VIP branch gating/disinhibition |
| STDP on dendritic pathways | Implemented | No | Compatible with flat or laminar mode and with eligibility traces | No hard conflict; deliberately does not touch Hopfield recurrent weights | This preserves the symmetric Hopfield core while adding dendritic plasticity |
| Eligibility traces | Implemented | No | Compatible with STDP and neuromodulation | Mostly inert without `use_stdp=True` because consolidation is applied in the STDP branch | Soft dependency, not a hard conflict |
| Oscillatory phase coding | Implemented | No | Compatible with flat or laminar mode and the other Track 10 features | No hard conflict | Tested in isolation and in the full all-features Track 10 bundle |
| Thalamic relay gating | Implemented | No | Compatible with laminar stack and Track 10 features | Mostly meaningful with `laminar=True`; outside laminar flow it is effectively not part of the main computation path | L6-modulated sensory gating, not the primary temporal state |
| Salience sensor module | Implemented | No | Compatible with goal-state-driven motor selection | Separate salience path, not in the default Track 12 benchmark loop | Produces salience-weighted goal states from sensor observations |

## 4. Benchmark And Diagnostic Features

| Feature | Status | Default in Track 12 | Cross-compatible | Conflict / dependency | Notes |
|---|---|---|---|---|---|
| Real animated-asset falsifier benchmark | Implemented | Yes | Compatible with the default Track 12 child pair, parent LM, and goal-state motor loop | None | Uses real Panda3D animated assets only |
| Matched, stretched, and compressed replay conditions | Implemented | Yes | Compatible with the default trace-bank stack | None | These are the main tempo-preservation checks |
| Omission and perturbation windowed deltas | Implemented | Yes | Compatible with boundary pressure and temporal surprise metrics | None | These are the main disturbance-response falsifiers |
| Action-blind replay versus active replay | Implemented | Yes | Compatible with action-conditioned prediction | Most informative when action-conditioned prediction is enabled | This is the direct efference-copy falsifier |
| Motion-grounded phase decoding | Implemented | Yes as a secondary report | Compatible with the current benchmark | Secondary only; explicitly not used as a training signal | Useful for diagnostics, not the primary theory target |
| Child-LM temporal parity report | Implemented | Yes | Compatible with the current Track 12 child setup | Specific to the benchmark harness | The benchmark records whether morphology and behavior child LMs share the same temporal config |
| Object-identity decoder and joint child signature fallback | Implemented | No in named-object mode, yes in fully self-supervised mode | Compatible with fully self-supervised benchmark runs | Only relevant when graph IDs are auto-labeled and ambiguous | Resolves ambiguous graph IDs by joint child signature, then parent fallback |

## 5. Missing, Partial, Or Planned Features

| Feature | Current status | Cross-compatible answer | Conflict / limitation | Notes |
|---|---|---|---|---|
| Parent-to-child predictive path (`receive_prediction`) | Missing | Not available yet | Current downward signal is context broadcast or context-only stepping, not true predictive coding | Explicitly called out as missing in Track 6 |
| Surprise-gated upward residual communication | Missing / partial | Not available as a unified architecture | Current voting and outputs do not send true residual prediction errors upward | Track 6 says communication is still largely unconditional content passing |
| Unified predictive voting mode | Missing | Not available yet | Spatial confusion and temporal confusion are still parallel, not unified | Track 6 explicitly lists this as missing |
| Attention-based dynamic routing | Planned only | Not implemented | Current Track 12 uses conditional voting and goal-state motor control, not true dynamic routing | Track 1 moves this to planned Track 6 work |
| Learned routing from evidence statistics | Research only | Not implemented | No stable implementation path exists | Still listed as research in Track 1 |
| Transformer-style QKV attention | Not implemented | Not part of the current architecture | The codebase uses Hopfield retrieval, prototype softmax, dendritic gating, and partial-cue memory instead | Important negative fact: the attention stack is not a transformer stack |
| Broad cortical-time integration beyond the narrowed Track 12 target | Partial | Pieces exist, but not as one integrated default architecture | The missing gap is integration: top-down prediction, surprise-gated communication, and full laminar temporal control working together | This is why the narrow Track 12 target looks more complete than the broader cortical-time vision |

## Practical Compatibility Verdicts

| Combination | Verdict | Why |
|---|---|---|
| Trace bank + action-conditioned prediction + inferred-state correction | Recommended | This is the current default Track 12 child stack |
| Trace bank + `TemporalMemory` | Safe | `TemporalMemory` adds behavior-side temporal evidence without replacing the trace bank |
| Trace bank + self-supervised temporal memory | Safe | The trace bank remains the online carrier; the self-supervised module adds latent-state semantics |
| Self-supervised temporal memory + transition memory | Avoid if you expect both to drive state updates | The self-supervised branch is checked first and effectively shadows transition memory |
| Laminar + multi-head + plateau + interneurons + STDP + eligibility + phase coding + thalamic relay | Safe, explicitly tested | The repo has Track 10 tests that enable the whole bundle together |
| `use_eligibility=True` without `use_stdp=True` | Technically allowed, weak practical value | Eligibility consolidation is only used in the STDP learning branch |
| `use_interneurons=True` without `multi_head=True` | Technically allowed, weak practical value | The SST/VIP branch-gating effect is mainly used in the multi-head path |
| `use_thalamic_relay=True` without `laminar=True` | Technically allowed, weak practical value | The thalamic relay is meaningfully used in the laminar step path |
| Plateau memory without apical input | Technically allowed, weak practical value | Plateau updates are most meaningful when apical predictions exist |
| Current Track 12 stack + dynamic attention routing | Not possible yet | Dynamic routing is still planned, not implemented |

## Bottom Line

The current repo is not a bag of unrelated temporal ideas. It has a real default Track 12 stack centered on trace-bank state, action-conditioned prediction, hierarchical context, and falsifier-style diagnostics.

The main compatibility story is simple:

- most Track 10 biological-depth features compose
- most optional temporal learners can sit next to the trace bank
- the main real exclusivity is the self-supervised versus explicit transition-memory state-owner path
- the main missing layer is predictive-coding integration, not the existence of more standalone timing modules