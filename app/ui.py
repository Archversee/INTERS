"""Shared state, wording and widgets for the Streamlit app."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sim2sop import config, references, storage
from sim2sop import guidelines_store as store
from sim2sop.agent.llm import get_llm
from sim2sop.learn.changes import step_slots
from sim2sop.pipeline import approve, create_guidelines, update_guidelines
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
DOC_KINDS = {"poh": "Aircraft manual (POH)", "sop": "Standard operating procedure",
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
FIELDS = [("action", "What to do"), ("trigger", "When"), ("intent", "Why"), ("typical", "Typical"),
          ("limit", "Hard limit")]
# Change reasons that matter to a reviewer (anything else is a routine range refinement).
IMPORTANT = ("Technique change", "See ", "Hard limit", "Value clipped", "Now ", "New action", "Instructor",
             "Re-checked", "New guideline")


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
    if run_id.startswith("generated_"):
        return f"Generated flight #{n}"
    return f"Flight {int(n)}" if n.isdigit() else run_id


def mode_title(doc: Guidelines) -> str:
    return {"create": "Created", "update": "Updated", "edit": "Instructor edit"}[doc.mode]


def doc_badges(doc: Guidelines) -> str:
    parts = [":green-badge[:material/check_circle: Published]" if doc.status == "published"
             else ":orange-badge[:material/edit_note: Draft — needs review]"]
    parts.append(":blue-badge[:material/verified_user: Checked against official limits]" if doc.checked
                 else ":red-badge[:material/gpp_maybe: Unchecked — no official documents]")
    parts.append(":gray-badge[Advisory]")
    return " ".join(parts)


def link_button(page_key: str, label: str, icon: str = ":material/arrow_forward:", primary: bool = True,
                key: str | None = None) -> None:
    """Button that opens another page of the app."""
    if st.button(label, icon=icon, type="primary" if primary else "secondary", key=key or f"go-{page_key}-{label}"):
        st.switch_page(PAGES[page_key])


# --- data ---------------------------------------------------------------------------------------

def generated_dir(prof: Profile) -> Path:
    return config.RAW_DIR / "generated" / prof.domain


def reset_all(prof: Profile, load_rules: bool = True) -> None:
    """Delete every flight, guideline version, document, correction and generated flight for
    the procedure; optionally load the sample official rules again."""
    storage.delete_runs(prof.procedure)
    shutil.rmtree(config.GUIDELINES_DIR / prof.procedure, ignore_errors=True)
    references.clear_references(prof.procedure)
    shutil.rmtree(config.MEMORY_DIR, ignore_errors=True)
    shutil.rmtree(generated_dir(prof), ignore_errors=True)
    telemetry.clear()
    for k in ("review_queue", "gen_current"):
        st.session_state.pop(k, None)
    if load_rules:
        for path in sorted((config.REFERENCES_DIR / prof.references).glob("*.md")):
            if path.name.lower() != "readme.md":
                references.add_reference(prof.procedure, path)


# --- building and reviewing ----------------------------------------------------------------------

def run_build(label: str, fn):
    """Run fn(progress) inside a status box with a progress bar."""
    with st.status(label, expanded=True) as status:
        bar = st.progress(0.0, text="Analysing the flights and checking the official limits…")

        def progress(i: int, n: int, msg: str) -> None:
            bar.progress(min(1.0, (i + 1) / max(n, 1)), text=f"{msg}  ({i + 1} of {n})")

        result = fn(progress)
        status.update(label=f"{label} — done", state="complete", expanded=False)
    return result


def refresh_guidelines(prof: Profile, title: str) -> Guidelines | None:
    """Create or update the guidelines after new data, then queue the review popup."""
    if not storage.list_runs(prof.procedure):
        return None
    exists = store.latest(prof.procedure) is not None
    doc = run_build("Updating the guidelines" if exists else "Creating the guidelines",
                    (lambda p: update_guidelines(prof, llm(), p)) if exists
                    else (lambda p: create_guidelines(prof, llm(), p)))
    if doc.pending:
        st.session_state["review_queue"] = {"version": doc.version, "title": title}
    else:
        st.session_state["toast"] = f"{title}: the guidelines did not need to change."
    return doc


def show_queued() -> None:
    """Toasts and the review popup queued by the previous action (call once per run)."""
    if msg := st.session_state.pop("toast", None):
        st.toast(msg, icon=":material/info:")
    if q := st.session_state.get("review_queue"):
        review_dialog(q["version"], q["title"])


def pending_banner(prof: Profile, key: str) -> None:
    doc = store.latest(prof.procedure)
    if doc and doc.pending and not st.session_state.get("review_queue"):
        with st.container(border=True):
            st.markdown(f":orange[:material/fact_check:] **{len(doc.pending)} change"
                        f"{'s' if len(doc.pending) != 1 else ''} in version {doc.version} "
                        "waiting for your approval.**")
            if st.button("Review now", type="primary", icon=":material/fact_check:", key=key):
                st.session_state["review_queue"] = {"version": doc.version, "title": f"Version {doc.version}"}
                st.rerun()


def _dismiss_review() -> None:
    st.session_state.pop("review_queue", None)


def _is_important(s: Step) -> bool:
    return s.change == "added" or any(r.startswith(IMPORTANT) for r in s.change_reasons)


def _decision(s: Step, version: int) -> str:
    return st.segmented_control("Decision", ["Approve", "Reject"], default="Approve",
                                key=f"dec-{version}-{s.id}", label_visibility="collapsed") or "Approve"


def _before_after(old: Step, new: Step) -> None:
    changed = [(f, label) for f, label in FIELDS if (getattr(old, f) or "") != (getattr(new, f) or "")]
    if not changed:
        return
    a, b = st.columns(2)
    a.caption("Before")
    b.caption("After")
    for f, label in changed:
        a.markdown(f"**{label}** — {getattr(old, f) or '—'}")
        b.markdown(f"**{label}** — {getattr(new, f) or '—'}")


@st.dialog("Review the guideline changes", width="large", on_dismiss=_dismiss_review)
def review_dialog(version: int, title: str) -> None:
    prof = profile()
    doc = store.load(prof.procedure, version)
    if not doc.pending:
        st.success("Nothing left to review.")
        return
    prev = store.load(prof.procedure, doc.parent_version) if doc.parent_version else None
    # The only publish button, top right. It keeps any change marked "Reject" below at its old text.
    rejects = [x.id for x in doc.pending if st.session_state.get(f"dec-{version}-{x.id}") == "Reject"]
    head, btn = st.columns([3, 2], vertical_alignment="center")
    head.markdown(f"### {title}")
    if btn.button("Approve all and publish" if not rejects else f"Publish ({len(rejects)} rejected)",
                  type="primary", icon=":material/done_all:", key="pub-all", width="stretch"):
        _publish(prof, doc, rejects)
    _review_summary(doc, prev)
    capped = [f.message for f in doc.flags if f.kind == "clipped"]
    if capped:
        st.warning("**Capped at an official limit:** " + " ".join(capped), icon=":material/vertical_align_top:")

    if prev is None:
        st.markdown(f"**{len(doc.steps)} new guidelines** from {len(doc.runs)} flight"
                    f"{'s' if len(doc.runs) != 1 else ''}. Read through and publish.")
        with st.expander("See all guidelines"):
            for phase in doc.phases:
                steps = doc.steps_in(phase)
                if steps:
                    st.markdown(f"**{phase_title(phase)}**\n" + "\n".join(f"- {s.action}" for s in steps))
    else:
        slot_of = {s.id: k for k, s in step_slots(doc.steps).items()}
        prev_slots = step_slots(prev.steps)
        important = [s for s in doc.pending if _is_important(s)]
        routine = [s for s in doc.pending if not _is_important(s)]
        st.markdown(f"**{len(doc.pending)} guideline{'s' if len(doc.pending) != 1 else ''} changed.** "
                    "Everything is approved unless you mark it *Reject* (keeps the previous version).")
        for s in important:
            with st.container(border=True):
                head, choice = st.columns([4, 2])
                head.markdown(f"**{phase_title(s.phase)} · {s.action}**")
                head.markdown(step_badges(s))
                with choice:
                    _decision(s, version)
                st.markdown("\n".join(f"- {r}" for r in s.change_reasons))
                old = prev_slots.get(slot_of[s.id])
                if old is not None:
                    _before_after(old, s)
                else:
                    guideline_body(s, doc)
        if routine:
            with st.expander(f"{len(routine)} guideline{'s' if len(routine) != 1 else ''} had their typical "
                             "ranges refined by the new data", icon=":material/tune:", expanded=not important):
                for s in routine:
                    head, choice = st.columns([4, 2])
                    head.markdown(f"**{phase_title(s.phase)} · {s.action}**  \n"
                                  + "; ".join(s.change_reasons[:3]))
                    with choice:
                        _decision(s, version)
        if doc.removed:
            st.markdown("**Removed:** " + ", ".join(f"{phase_title(s.phase)} · {s.label}" for s in doc.removed))


def _publish(prof: Profile, doc: Guidelines, rejects: list[str]) -> None:
    approve(prof, doc.version, None, rejects)
    st.session_state.pop("review_queue", None)
    st.session_state["toast"] = f"Version {doc.version} published."
    st.rerun()


def _review_summary(doc: Guidelines, prev: Guidelines | None) -> None:
    """The headline numbers of the new version."""
    m, pm = doc.metrics, prev.metrics if prev else {}
    c = st.columns(3)
    c[0].metric("Flights", len(doc.runs), f"+{len(doc.runs) - len(prev.runs)}" if prev else None)
    c[1].metric("Guidelines changed", len(doc.pending))
    c[2].metric("Confidence", f"{m['mean_confidence']:.0%}",
                f"{(m['mean_confidence'] - pm['mean_confidence']) * 100:+.0f} pts" if pm else None)


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


def limit_badges(s: Step, prof: Profile) -> str:
    """Short red chips for the hard limits beside a step, e.g. "max 85 kt"."""
    return " ".join(f":red-badge[:material/block: {lim.describe()} {prof.unit(lim.signal) or ''}]".replace(" ]", "]")
                    for lim in s.limits)


def limit_name(lim, prof: Profile) -> str:
    """A limit in a few words, e.g. "Airspeed max 85 kt"."""
    label = prof.label(lim.signal)
    return f"{label[:1].upper()}{label[1:]} {lim.describe()} {prof.unit(lim.signal) or ''}".strip()


def when_text(lim, prof: Profile) -> str:
    """When a limit applies, in plain words: "Flaps 20°/30° · Downwind", or "Always"."""
    parts = []
    if lim.when:
        m = re.fullmatch(r"\s*(\w+)\s*(==|>=|<=|>|<)\s*(-?[\d.]+)\s*", lim.when)
        toggle = prof.events.toggles.get(m.group(1)) if m else None
        if toggle:
            ok = {"==": float.__eq__, ">=": float.__ge__, "<=": float.__le__, ">": float.__gt__, "<": float.__lt__}
            states = [name for v, name in toggle.states.items() if ok[m.group(2)](float(v), float(m.group(3)))]
            parts.append(f"{toggle.name} {'/'.join(states)}".strip())
        else:
            parts.append(lim.when)
    if lim.phases:
        parts.append(", ".join(phase_title(p) for p in lim.phases))
    return " · ".join(parts) or "Always"


def checklist_line(n: int, s: Step, prof: Profile) -> str:
    """One line per step: what to do, when, and the hard limit."""
    when = (s.trigger or "").rstrip(".")
    when = when[:1].lower() + when[1:]
    tags = []
    if s.required_by:
        tags.append(":green-badge[Required]")
    elif not s.mandatory:
        tags.append(f":orange-badge[Technique · {s.seen_in}/{s.run_count}]")
    if s.change:
        tags.append(":violet-badge[New]" if s.change == "added" else ":violet-badge[Changed]")
    line = f"{n}. **{s.action.rstrip('.')}**" + (f" — {when}" if when else "")
    return " ".join([line, limit_badges(s, prof), *tags]).strip()


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
        for lim in (fact.limit_max, fact.limit_min):
            if lim is not None:
                fig.add_hline(y=lim, line_color="red", line_dash="dot", annotation_text="hard limit")
    fig.add_vline(x=0, line_dash="dash", annotation_text="action")
    unit = prof.unit(sig) or ""
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                      xaxis_title="seconds from the action", yaxis_title=f"{prof.label(sig)} ({unit})")
    st.plotly_chart(fig, width="stretch")
    st.caption(f"From {run_title(run_id)}. Green band = what instructors typically do; red line = official limit.")
