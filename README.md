# INTERS: Sim-to-SOP

Sim-to-SOP turns an instructor's simulator run into a verified, step-by-step SOP (standard operating procedure).

> Fly it once, get the manual. Fly it five times, get the standard.

The LLM never sees raw telemetry. Deterministic signal processing turns each run into an
event timeline. A planner → writer → verifier graph writes the steps, and the verifier
checks every number against the data. Multi-run consensus and correction memory make
each new version better than the last.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -e ".[app,dev]"

sim2sop demo --provider none      # full air demo, no LLM needed
streamlit run app/streamlit_app.py
uvicorn sim2sop.api:app --reload  # REST API at http://localhost:8000/docs
pytest
```

`sim2sop demo` generates 5 C172 circuits and builds four SOP versions:

| Version | What happens | What it shows |
|---|---|---|
| v1 | 1 run | Every step cited and verified; default tolerance bands |
| v2 | 5 runs | DTW consensus: bands about 2× tighter, confidence up, short-final power correction becomes a technique note (2/5 runs) |
| v3 | Instructor edits the Flaps 10° step | Edit stored as a before/after diff and re-verified |
| v4 | Regenerate | The correction is reused on Flaps 20° / 30° / UP, each with its own verified numbers |

The HTML output for each version is written to `data/sops/<procedure>/v<N>.html`.

## LLM: Ollama or an API key

Set the provider in `.env` (copy `.env.example`) or with environment variables. You can also override it per command with `--provider/--model`.

| `SIM2SOP_LLM_PROVIDER` | Needs | Notes |
|---|---|---|
| `ollama` (default) | `ollama pull qwen2.5:7b-instruct` | Fully offline |
| `openai` | `SIM2SOP_API_KEY`, optional `SIM2SOP_LLM_BASE_URL` | Any OpenAI-compatible endpoint: OpenAI, OpenRouter, Groq, vLLM, LM Studio |
| `anthropic` | `SIM2SOP_API_KEY` | Claude Messages API |
| `none` | nothing | Deterministic template writer |

If the provider is unreachable, the pipeline falls back to the template writer, so a demo
never dies on stage. Run `sim2sop llm-check` to test the connection. All providers go
through one interface in [src/sim2sop/agent/llm.py](src/sim2sop/agent/llm.py); to add one, subclass `LLMClient`.

## Pipeline

```
raw sim files ─ adapter ─> common schema ─> resample + derived signals
   ─> phase segmentation (profile rules + ruptures change points)
   ─> event extraction (toggles, thresholds, control inputs, turns)
   ─> annotation matching (callout → nearest event; unmatched → flag)
   ─> compliance limits ─> run library (Parquet + DuckDB)
   ─> consensus (DTW across runs → mandatory / technique, tolerance bands, confidence)
   ─> planner → writer (LLM or template, + RAG citation, + correction memory) → verifier ⟲
   ─> versioned SOP (JSON + HTML/PDF)
```

| Path | What it is |
|---|---|
| [profiles/](profiles/) | One YAML per procedure: signals, phases, events, keywords, limits. **New domain = new file** |
| [src/sim2sop/adapters/](src/sim2sop/adapters/) | MSFS SimConnect, CARLA, NMEA 0183, generic CSV with column mapping |
| [src/sim2sop/generate/](src/sim2sop/generate/) | Synthetic runs in each sim's native format (air done) |
| [src/sim2sop/signals/](src/sim2sop/signals/) | Phases, events, annotation matching, compliance |
| [src/sim2sop/learn/](src/sim2sop/learn/) | DTW consensus, tolerance bands, correction memory |
| [src/sim2sop/agent/](src/sim2sop/agent/) | LLM clients, planner, writer, verifier, LangGraph graph |
| [src/sim2sop/rag/](src/sim2sop/rag/) | Reference retrieval (TF-IDF, or Ollama `nomic-embed-text`) |
| [references/](references/) | Reference docs per domain (`.md` with `##` sections, `.txt`, `.pdf`) |
| [src/sim2sop/render/](src/sim2sop/render/) | HTML template, PDF via WeasyPrint (`pip install -e ".[pdf]"`) |
| [app/streamlit_app.py](app/streamlit_app.py) | SOP viewer, instructor edits, metrics, version diff, run explorer |

## CLI

```bash
sim2sop generate-data air -n 5
sim2sop ingest air_c172_circuit data/raw/air          # a run folder, or a folder of runs
sim2sop build air_c172_circuit [--provider anthropic] [--full]
sim2sop show air_c172_circuit [--version 2]
sim2sop edit air_c172_circuit s14 action "Select Flaps 10°, then check the indicator."
sim2sop diff air_c172_circuit 2 4
sim2sop export air_c172_circuit --pdf
sim2sop transcribe callouts.wav data/raw/air/<run> --offset 0   # needs .[voice]
sim2sop reset air_c172_circuit --memory
```

## Status and team to-dos

Done and tested (air): generator, MSFS adapter, schema, phases, events, annotation matching,
compliance flags, consensus and bands, correction memory, RAG, verifier loop, versioning,
HTML render, CLI, API and UI.

- **Data:** land (CARLA) and sea (NMEA) generators. The adapters and profiles exist but no data has been run through them yet, and their thresholds are guesses. Confirm MSFS pitch/bank sign conventions against a real capture.
- **AI:** pre-test Qwen 2.5 7B vs Llama 3.1 8B on the writer prompt. The reference notes in `references/air/` are **placeholders**: replace them with real POH / FAA AFH extracts. Optionally move to LanceDB.
- **Learning:** student grading mode and a "Common mistakes" section (stretch).
- **Product:** per-step chart in the HTML, checklist cards, PDF on the demo machine (WeasyPrint needs GTK on Windows).

## Limitations (say these in the pitch)

- All runs are generated, not recorded from real instructors. Adapters accept real recordings in the same format unchanged.
- "Learning" means memory, retrieval and consensus. No model weights are trained.
- The output is not a certified document.
