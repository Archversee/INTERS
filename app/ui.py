"""Shared state, wording and widgets for the Streamlit app."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sim2sop import config, references, storage
from sim2sop import guidelines_store as store
from sim2sop.agent.llm import get_llm
from sim2sop.generate import generate
from sim2sop.ingest import ingest_run
from sim2sop.profiles import Profile, list_profiles, load_profile
from sim2sop.schema import Guidelines, Step

PAGES: dict = {}   # filled by streamlit_app.py so pages can link to each other

PROFILE_NAMES = {
    "air_c172_circuit": "✈️ Air · Cessna 172 circuit",
    "land_parallel_park": "🚗 Land · Parallel parking",
    "sea_harbour_exit": "🚢 Sea · Harbour exit",
}
PROVIDERS = {"ollama": "Ollama (local / LAN)", "openai": "OpenAI-compatible API",
             "anthropic": "Anthropic API", "none": "No AI (built-in template)"}
DOC_KINDS = {"poh": "Aircraft manual (POH)", "sop": "Standard operating procedure", "notam": "NOTAM",
             "regulation": "Regulation", "handbook": "Handbook (guidance only)", "other": "Document"}
FLAG_LABELS = {
    "limit_violation": ("Instructor flights that broke a hard limit", ":material/warning:"),
    "excluded": ("Values left out of a typical range (they broke a limit)", ":material/remove_circle:"),
    "clipped": ("Values capped at a hard limit", ":material/vertical_align_top:"),
    "technique_change": ("Technique changes detected", ":material/swap_horiz:"),
    "required_missing": ("Required actions nobody flew", ":material/report:"),
    "unmatched_annotation": ("Spoken callouts with no matching action", ":material/mic_off:"),
    "verifier": ("Numbers the checker could not confirm", ":material/error:"),
    "unchecked": ("Not checked against official limits", ":material/gpp_maybe:"),
    "data": ("Data issues", ":material/database:"),
}


# --- session state ------------------------------------------------------------------------

def profile() -> Profile:
    names = list_profiles()
    name = st.session_state.get("profile_name") or ("air_c172_circuit" if "air_c172_circuit" in names else names[0])
    return load_profile(name)


def llm():
    provider = st.session_state.get("llm_provider", config.LLM_PROVIDER)
    model = st.session_state.get("llm_model") or None
    return get_llm(provider, model)


@st.cache_data(ttl=30, show_spinner=False)
def llm_status(provider: str, model: str | None) -> tuple[bool, list[str]]:
    client = get_llm(provider, model)
    if provider == "ollama":
        try:
            return True, client.models()
        except Exception:
            return False, []
    return client.available(), []


@st.cache_data(show_spinner=False)
def telemetry(procedure: str, run_id: str) -> pd.DataFrame:
    return storage.load_telemetry(procedure, run_id)


# --- wording --------------------------------------------------------------------------------

def phase_title(name: str) -> str:
    return name.replace("_", " ").title()


def run_title(run_id: str) -> str:
    n = run_id.rsplit("_", 1)[-1]
    if "obstacle" in run_id:
        return f"Flight {n} (after new obstacle)"
    return f"Flight {n}" if n.isdigit() else run_id


def mode_title(doc: Guidelines) -> str:
    return {"create": "Created", "update": "Updated", "edit": "Instructor edit"}[doc.mode]


def doc_badges(doc: Guidelines) -> str:
    parts = [":green-badge[:material/check_circle: Published]" if doc.status == "published"
             else ":orange-badge[:material/edit_note: Draft — needs review]"]
    parts.append(":blue-badge[:material/verified_user: Checked against official limits]" if doc.checked
                 else ":red-badge[:material/gpp_maybe: Unchecked — no official documents]")
    parts.append(":gray-badge[Advisory]")
    return " ".join(parts)


# --- workflow state ---------------------------------------------------------------------------

def news_since(doc: Guidelines | None, prof: Profile) -> dict:
    runs = storage.list_runs(prof.procedure)
    docs = references.list_references(prof.procedure)
    if doc is None:
        return {"runs": runs, "refs": docs}
    return {"runs": [r for r in runs if r.run_id not in doc.runs],
            "refs": [d for d in docs if d.label not in doc.references]}


def next_action(prof: Profile) -> tuple[str, str, str]:
    """(message, page key, button label) for the most useful next step."""
    runs = storage.list_runs(prof.procedure)
    doc = store.latest(prof.procedure)
    if not runs:
        return "Start by adding instructor flights from the simulator.", "runs", "Add flights"
    if doc is None:
        return (f"You have {len(runs)} flight{'s' if len(runs) != 1 else ''}. Create the first guidelines.",
                "guidelines", "Create guidelines")
    if doc.pending:
        return (f"Version {doc.version} has {len(doc.pending)} change{'s' if len(doc.pending) != 1 else ''} "
                "waiting for your approval.", "review", "Review changes")
    new = news_since(doc, prof)
    if new["runs"] or new["refs"]:
        bits = []
        if new["runs"]:
            bits.append(f"{len(new['runs'])} new flight{'s' if len(new['runs']) != 1 else ''}")
        if new["refs"]:
            bits.append(f"{len(new['refs'])} new document{'s' if len(new['refs']) != 1 else ''}")
        return f"{' and '.join(bits)} since version {doc.version}. Update the guidelines.", "guidelines", "Update guidelines"
    if not doc.checked:
        return ("The guidelines are not checked against official limits yet. Add reference documents.",
                "references", "Add documents")
    return "Everything is up to date.", "guidelines", "View guidelines"


def link_button(page_key: str, label: str, icon: str = ":material/arrow_forward:", primary: bool = True,
                key: str | None = None) -> None:
    """Button that opens another page of the app."""
    if st.button(label, icon=icon, type="primary" if primary else "secondary", key=key or f"go-{page_key}-{label}"):
        st.switch_page(PAGES[page_key])


# --- sample data --------------------------------------------------------------------------------

def sample_dirs(prof: Profile, obstacle: bool = False) -> list[Path]:
    raw = config.RAW_DIR / prof.domain
    dirs = sorted(d for d in raw.glob("*") if d.is_dir() and ("obstacle" in d.name) == obstacle) if raw.exists() else []
    if not dirs and prof.domain == "air":
        dirs = generate(prof.domain, n=3 if obstacle else 5, scenario="obstacle" if obstacle else None)
    return dirs


def add_sample_flights(prof: Profile, count: int | None = None, obstacle: bool = False) -> int:
    known = {r.run_id for r in storage.list_runs(prof.procedure)}
    todo = [d for d in sample_dirs(prof, obstacle) if d.name not in known][:count]
    for d in todo:
        ingest_run(d, prof)
    return len(todo)


# --- long-running builds -----------------------------------------------------------------------

def run_build(label: str, fn):
    """Run fn(progress) inside a status box with a progress bar."""
    with st.status(label, expanded=True) as status:
        bar = st.progress(0.0, text="Analysing flights and planning the guidelines…")

        def progress(i: int, n: int, msg: str) -> None:
            bar.progress(min(1.0, (i + 1) / max(n, 1)), text=f"{msg}  ({i + 1} of {n})")

        result = fn(progress)
        status.update(label=f"{label} — done", state="complete", expanded=False)
    return result


# --- guideline card -----------------------------------------------------------------------------

def step_badges(s: Step) -> str:
    out = []
    if s.required_by:
        out.append(":green-badge[Required by SOP]")
    elif s.mandatory:
        out.append(":blue-badge[Core — every instructor]")
    else:
        out.append(f":orange-badge[Technique — {s.seen_in} of {s.run_count} instructors]")
    if s.change == "added":
        out.append(":violet-badge[New]")
    elif s.change == "modified":
        out.append(":violet-badge[Changed]")
    if s.review == "pending":
        out.append(":orange-badge[Awaiting approval]")
    if s.edited:
        out.append(":gray-badge[Edited by instructor]")
    if not s.verified:
        out.append(":red-badge[Number not verified]")
    return " ".join(out)


def guideline_body(s: Step, doc: Guidelines) -> None:
    rows = [(":material/schedule:", "When", s.trigger), (":material/lightbulb:", "Why", s.intent),
            (":material/straighten:", "Typical", s.typical)]
    for icon, label, text in rows:
        if text:
            st.markdown(f"{icon} **{label}** — {text}")
    if s.limit:
        st.markdown(f":red[:material/block: **Hard limit** — {s.limit}]")
    elif doc.checked:
        st.caption(":material/block: No hard limit in the official documents for this step.")
    else:
        st.caption(":material/gpp_maybe: Not checked — no official documents added yet.")


def step_chart(s: Step, doc: Guidelines, prof: Profile) -> None:
    run_id = s.anchor_run or doc.anchor_run
    if not run_id or run_id not in {r.run_id for r in storage.list_runs(prof.procedure)}:
        st.caption("No flight data available for a chart.")
        return
    df = telemetry(prof.procedure, run_id)
    window = df[(df["t"] >= s.t - 20) & (df["t"] <= s.t + 20)]
    signals = [x for x in prof.trigger_for(s.phase) if x in window]
    if not signals:
        return
    sig = st.segmented_control("Show", signals, default=signals[0], format_func=prof.label,
                               key=f"chart-{doc.version}-{s.id}")
    sig = sig or signals[0]
    fig = go.Figure(go.Scatter(x=window["t"] - s.t, y=window[sig], name=prof.label(sig),
                               line=dict(width=3)))
    fact = s.facts.get(f"trigger.{sig}")
    if fact and fact.lo is not None:
        fig.add_hrect(y0=fact.lo, y1=fact.hi, opacity=0.18, line_width=0, fillcolor="green",
                      annotation_text="typical range", annotation_position="top left")
        for lim, name in ((fact.limit_max, "hard limit"), (fact.limit_min, "hard limit")):
            if lim is not None:
                fig.add_hline(y=lim, line_color="red", line_dash="dot", annotation_text=name)
    fig.add_vline(x=0, line_dash="dash", annotation_text="action")
    unit = prof.unit(sig) or ""
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                      xaxis_title="seconds from the action", yaxis_title=f"{prof.label(sig)} ({unit})")
    st.plotly_chart(fig, width="stretch")
    st.caption(f"From {run_title(run_id)}. Green band = what instructors typically do; red line = official limit.")
