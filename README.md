# INTERS: Sim-to-SOP

Sim-to-SOP turns instructor simulator runs into practical, step-by-step **guidelines** for students. Reference SOPs set the hard limits.

> The SOP says what's allowed. The guidelines show how the best instructors do it.

The guidelines are **advisory**: what to do, when, why, and the range instructors typically use.
The existing SOPs, the POH and regulations are read-only **reference SOPs**:
- Every guideline is checked against them.
- A value past a limit is clipped to the limit and flagged.
- Where a guideline and an SOP differ, the SOP wins.
- The system never writes or edits an SOP.

The LLM never sees raw telemetry. Deterministic signal processing builds an event
timeline. A planner → writer → verifier graph writes each guideline, and the verifier
checks every number against the telemetry and the reference limits.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -e ".[app,dev]"
streamlit run app/streamlit_app.py  # the app; on an empty app, Home → "Load sample data" shows the whole workflow

sim2sop demo --provider none      # the same demo script from the command line, no LLM needed
uvicorn sim2sop.api:app --reload  # REST API at http://localhost:8000/docs
pytest
```

## Where the AI is used

| Where | What the AI does | What checks it |
|---|---|---|
| Writing guidelines | Words each step (when, what, why, typical) from the step's verified numbers | The verifier: every number must match the data and sit inside the limits, else up to 2 rewrites; limit text is quoted, never AI-written |
| Official documents → *Read a manual with AI* | Reads a manual with no limits list and proposes hard limits and required actions in the procedure's signals | The quote must be in the manual word for word, the value must be written in the quote, the signal / phases / condition / action must exist in the profile; an instructor ticks what goes in force ([manuals.py](src/sim2sop/agent/manuals.py)) |
| *Patterns* page | Explains the patterns statistics find in the runs (instructor differences, habits, techniques that go together, knock-on effects, limit breaks, least agreement) as a few findings with an action each | Each finding must cite patterns, and every number it writes must be one of theirs ([patterns.py](src/sim2sop/learn/patterns.py), [insights.py](src/sim2sop/agent/insights.py)) |

With the AI writer off, guidelines use the template, manuals need a limits file, and Patterns shows the strongest patterns as found.

## Demo script (`sim2sop demo`)

| Step | Version | What it shows |
|---|---|---|
| 1. Create, data only | v1 | 1 run → guidelines marked **UNCHECKED** |
| 2. Add reference SOPs | v2 | C172 POH + aerodrome SOP: each guideline shows its hard limit. Run 1 selects flaps 20 at 89 kt; the value is **clipped to Vfe 85 kt** and flagged |
| 3. More runs | v3 | 5 runs: typical ranges about 2× tighter, short-final power correction becomes a technique note, the overspeed run is **excluded** from the flaps-20 range |
| 4. Instructor edit | v4, v5 | Edit the Flaps 10° guideline; the next update applies the same wording to Flaps 20° and 30° |
| 5. Other domains | | Land (parallel parking, CARLA format) has a generator and sample rules: pick it in the app's sidebar, or `sim2sop generate-data land`. Sea is still TODO |

Each version is written to `data/guidelines/<procedure>/v<N>.html`. Every step in the demo is auto-approved; in real use the instructor approves or rejects each change.

## Two modes

| | Create (`sim2sop create`) | Update (`sim2sop update`) |
|---|---|---|
| Input | Flight data (+ optional reference SOPs) | Existing guidelines + new runs and/or revised references |
| Output | v1, *limit-checked* or *unchecked* | New version with a step-level diff and the reason for each change |
| What changes | Everything | Only affected phases and steps |
| Approval | `sim2sop approve` publishes | Each changed step is approved or rejected (`--reject s12`), and a rejected step reverts to the previous version |

How update decides what changed:
- **New flights:** every phase learns from them. Only steps whose numbers moved are rewritten, each with the reason.
- **A revised SOP or POH** (same `id`, new `revision`): every step is re-checked, and the steps whose limits changed are re-cited.
- (Dormant for now: runs tagged with an `environment` in `meta.json` can trigger per-phase technique-change detection. Nothing in the app or the sample data tags runs.)
- **New instructor corrections:** matching steps are rewritten in the corrected style.

## Reference SOPs and hard limits

```bash
sim2sop add-reference air_c172_circuit references/air/c172_poh_notes.md references/air/aerodrome_sop_notes.md
sim2sop references air_c172_circuit
```

In the app, the documents in `references/<domain>/` are **built in**: they are added automatically, re-added when their file is edited, and cannot be removed or replaced (Start over keeps them). Edit those files to change the rules. The CLI demo still adds them by hand to show unchecked → checked.

Documents are copied into `data/references/<procedure>/` and never modified. Limits and required actions go in the document's YAML front matter; for a PDF, put the same YAML in a `<name>.limits.yaml` file next to it:

```yaml
---
id: c172_poh          # stable across revisions; a new revision supersedes the old one
title: C172 POH
kind: poh             # sop | poh | regulation | handbook | other
revision: "A"
limits:
  - {id: vfe_full, signal: ias_kt, max: 85, when: "flaps_idx >= 2", text: "Max 85 KIAS with flaps beyond 10° (Vfe)"}
  - {id: circuit_height_max, signal: agl_ft, max: 1100, phases: [crosswind, downwind], text: "..."}
requirements:          # actions an SOP makes mandatory: always core guidelines
  - {id: landing_light, key: "toggle:light_landing:1", text: "Landing light on for take-off"}
---
```

How limits are applied:
- `when` is evaluated on the state *after* the step's own action. Selecting flaps 20 falls under the flaps-20 Vfe.
- `at` chooses whether the limit bounds the trigger, the result or both.
- Run values that break a limit are excluded from that step's typical range. If no compliant run remains, the value is clipped and flagged.
- The limit-violation report lists every breach found in the full telemetry.

## Guideline template

**When** (trigger) → **what to do** → **why** (the instructor's callout) → **typical range** (median and robust spread across runs) → **hard limit** (quoted from the reference, with its citation).

Typical ranges use the median ± a blend of the profile tolerance and 2σ, where σ comes from the IQR (or the MAD for fewer than 4 runs). One bad run cannot widen a range, and ranges are capped at the limits. Hard-limit text is always quoted from the reference: the LLM never writes it, and instructors cannot edit it.

## LLM: Ollama or an API key

Set the provider in `.env` (copy `.env.example`) or override it per command with `--provider/--model`.

| `SIM2SOP_LLM_PROVIDER` | Needs | Notes |
|---|---|---|
| `ollama` (default) | `ollama pull qwen2.5:7b-instruct` | Fully offline |
| `openai` | `SIM2SOP_API_KEY`, optional `SIM2SOP_LLM_BASE_URL` | Any OpenAI-compatible endpoint |
| `anthropic` | `SIM2SOP_API_KEY` | Claude Messages API |
| `none` | nothing | Deterministic template writer |

If the provider is unreachable, the pipeline falls back to the template writer. Run `sim2sop llm-check` to test the connection.

## Code map

| Path | What it is |
|---|---|
| [profiles/](profiles/) | One YAML per procedure: signals (incl. ground track), phases, events, keywords. **New domain = new file** |
| [references/](references/) | Sample reference SOPs per domain (**placeholders**, not real POH text) |
| [src/sim2sop/adapters/](src/sim2sop/adapters/) | MSFS SimConnect, CARLA, NMEA 0183, generic CSV |
| [src/sim2sop/generate/](src/sim2sop/generate/) | Synthetic runs in each sim's format (air: 5 baseline circuits, plus single flights with chosen technique) |
| [src/sim2sop/signals/](src/sim2sop/signals/) | Phases, events, annotation matching, limit-violation report |
| [src/sim2sop/references.py](src/sim2sop/references.py) | Read-only reference library: limits, requirements, revisions |
| [src/sim2sop/learn/](src/sim2sop/learn/) | DTW consensus and typical ranges, hard limits, change detection, correction memory |
| [src/sim2sop/agent/](src/sim2sop/agent/) | LLM clients, planner, writer, verifier, LangGraph graph |
| [src/sim2sop/pipeline.py](src/sim2sop/pipeline.py) | create / update / approve / edit, and the reason for each change |
| [src/sim2sop/guidelines_store.py](src/sim2sop/guidelines_store.py) | Versions and step-level diff |
| [app/](app/) | Streamlit app: Home (with *Load sample data* on first run) → 1 · Instructor flights (a .zip of flight folders or a telemetry CSV) → 2 · Guidelines, with Official documents (the rules, built in) in their own *Rules* section. Each upload or document updates the guidelines and pops up a review window; a rejected change is not proposed again until its data changes. Plus Progress (version history, one guideline across versions), *Start over* in the sidebar, a Flight generator (random flights with chosen technique, a rundown of each, added to the guidelines in one click or downloaded as a zip, and the data-format guide) and a flight data explorer ([views/](app/views/), shared widgets in [ui.py](app/ui.py)) |

## CLI

```bash
sim2sop generate-data air -n 5                     # baseline runs
sim2sop ingest air_c172_circuit data/raw/air
sim2sop add-reference air_c172_circuit <file>...
sim2sop create air_c172_circuit [--provider anthropic]
sim2sop update air_c172_circuit
sim2sop approve air_c172_circuit [--steps s12 s13] [--reject s14]
sim2sop edit air_c172_circuit s14 action "Select Flaps 10°, then check the indicator."
sim2sop show | diff A B | export [--pdf] | references | reset [--references --memory]
```

## Status and team to-dos

Done and tested (air): the generator with the overspeed scenario and the in-app flight generator, MSFS adapter, ground track, phases, events, annotation matching, reference library and limits, consensus with typical ranges, change detection, update with reasons, approval, correction memory, RAG, verifier loop, HTML, CLI, API and UI.

- **Data:** sea (NMEA) generator; its adapter and profile exist but are untested, and the sea folder has no reference SOPs yet. Land (parallel parking) has a generator, sample rules (`references/land/`, placeholders) and tests, but not a real CARLA capture yet: confirm `MAX_STEER_DEG` and the light-state bits. Confirm MSFS sign conventions against a real capture.
- **AI:** pre-test Qwen 2.5 7B vs Llama 3.1 8B, including on *Read a manual with AI* and *Patterns* (both are tested only with a scripted model so far). Replace the placeholder references with real POH and aerodrome SOP extracts.
- **Learning:** student grading and a "Common mistakes" section (stretch).
- **Product:** per-step charts in the HTML (the UI already has them), quick-reference cards, PDF on the demo machine.

## Limitations (say these in the pitch)

- All runs are generated, not recorded from real instructors. Adapters accept real recordings in the same format unchanged.
- "Learning" means memory, retrieval and consensus. No model weights are trained.
- The guidelines are advisory and not certified. The reference SOPs remain the authority.
