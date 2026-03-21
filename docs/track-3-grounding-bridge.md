# Track 3: Grounding Bridge

## Purpose

Connect Monty's grounded world model to external systems: language, other agents,
and abstract domains.

This track lifts Restriction 1 (positions from sensors only) and Restriction 2
(features fixed at config time) by creating mapping layers between Monty's internal
representations and external modalities.

## Roadmap

See [full-system-roadmap-v2.md](full-system-roadmap-v2.md) for the full program
context, restriction analysis, concept layers, and priority stack.

## Current Status

T3.1-T3.2 completed (2026-03-20). Language bridge maps object IDs → words and
aggregated evidence → category names. Validated on scaled6 Omniglot: 43.3% object
naming accuracy, 70% category naming accuracy.

## Milestone Tracker

| ID | Milestone | Status |
|---|---|---|
| T3.1 | Object naming (object ID → word) | **Completed** |
| T3.2 | Category naming (category evidence → category word) | **Completed** |
| T3.3 | Spatial relation language | Not started |
| T3.4 | Instruction following | Not started |
| T3.5 | Agent identity (agent vs object distinction) | Not started |
| T3.6 | Behavioral language | Not started (requires Track 2) |
| T3.7 | Theory of mind | Research |

## Available Infrastructure

From the world model extension:

- **TextSM** (`src/tbp/monty/frameworks/models/text_sm.py`): Hash-embedding
  text token sensor module that converts text tokens to CMP State format.
- **LiveTextSource** (`src/tbp/monty/frameworks/environments/live_text_interface.py`):
  Text input interface (stdin/queue or token list).
- **TextOutputHook** (`src/tbp/monty/frameworks/experiments/text_output_hook.py`):
  Template or ollama-based text output generation.
- **Category readout** (`tools/phase2_category_evidence_readout.py`): Maps
  evidence distributions to category labels — the basis for T3.2.

From T3.1-T3.2 implementation:

- **LanguageBridge** (`src/tbp/monty/frameworks/models/language_bridge.py`):
  Core grounding module. Maps graph IDs → object names, aggregates evidence by
  category → category names, produces natural language descriptions of recognition.
  - `name_object(graph_id)` → human-readable object name (T3.1)
  - `name_category(evidence_per_graph)` → (category_name, evidence, breakdown) (T3.2)
  - `describe_recognition(graph_id, evidence, evidence_per_graph)` → sentence
  - Auto-generates names from underscored IDs when no explicit mapping given
- **NamingLogger** (`src/tbp/monty/frameworks/loggers/naming_logger.py`):
  BaseMontyLogger that reads LM results post-episode and outputs named results
  via LanguageBridge. Writes `naming_results.jsonl` and `naming_summary.json`.
- **Tests**: 14 tests in `tests/unit/frameworks/models/language_bridge_test.py`

## T3.1-T3.2 Results

Validated on scaled6 Omniglot (30 objects, 5 alphabets x 6 characters, cross-version):

| Metric | Accuracy |
|---|---|
| Object naming (exact instance match) | 43.3% (13/30) |
| Category naming (aggregated evidence) | 70.0% (21/30) |

Example output:
```
Episode 1: Target="Alphabet of the Magi 1"
  Recognized "Alphabet of the Magi 3" (evidence: 12.0).
  Category: "Alphabet of the Magi" (total: 32.4)
  Instance: ✗  Category: ✓

Episode 4: Target="Alphabet of the Magi 4"
  Recognized "Asomtavruli (Georgian) 4" (evidence: 658.7).
  Category evidence suggests "Alphabet of the Magi" (total: 3358.6)
  over instance category "Asomtavruli (Georgian)"
  Instance: ✗  Category: ✓
```

The bridge reveals cases where the instance prediction is wrong but category
evidence correctly identifies the right alphabet — the category signal that
was already present in the evidence distribution but invisible in argmax output.

## Next Steps (T3.3+)

**T3.3: Spatial relation language.**

After recognition, the system knows object locations from detected poses. A
spatial relation module would compute pairwise relations (left-of, above,
inside, near) and express them in natural language: "The mug is to the left
of the bowl."

Requires: detected locations from multiple recognized objects in the same scene
(currently episodes are single-object). Multi-object scenes are a prerequisite.

**T3.4: Instruction following.**

Requires bidirectional grounding: language input → motor goals. The TextSM
provides the input channel; the motor system provides the action channel.
The bridge needs to map verbs to motor primitives and nouns to graph IDs.

Requires: CMP enrichment (Restriction 2 lift) so language module output can
become context for the visual matching pipeline.

## Restrictions Addressed

- **Restriction 2 (features fixed at config time)**: T3.4+ requires LM outputs
  to become features for other LMs (e.g., language module output → visual module
  context). This requires CMP enrichment (making LM outputs available as flexible
  features to connected LMs).
- **Restriction 1 (positions from sensors only)**: T3.7 requires modeling another
  agent's reference frame — positions that the system has never directly sensed.
  This requires hippocampal replay of inferred states.

## Concept Layers Served

- **Layer 3** (social concepts): T3.5 (agent identity) + T3.7 (theory of mind)
- **Layer 4** (cultural concepts): requires language + multi-agent models + norms
