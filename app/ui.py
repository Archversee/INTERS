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
# What one recorded run is called in each domain, and the domains the generator can make.
NOUNS = {"air": "flight", "land": "drive", "sea": "run"}
GENERATED = {"air", "land"}
PROVIDERS = {"ollama": "Ollama (local / LAN)", "openai": "OpenAI-compatible API",
             "anthropic": "Anthropic API", "none": "No AI (built-in template)"}
DOC_KINDS = {"poh": "Aircraft manual (POH)", "sop": "Standard operating procedure",
             "regulation": "Regulation", "handbook": "Handbook (guidance only)", "other": "Document"}
FLAG_LABELS = {
    "limit_violation": ("Instructor runs that broke a hard limit", ":material/warning:"),
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

def noun(n: int = 1, cap: bool = False) -> str:
    """What a recorded run is called for the selected procedure: "flight", "drives", ..."""
    name = st.session_state.get("profile_name") or "air"
    word = NOUNS.get(name.split("_", 1)[0], "run") + ("" if n == 1 else "s")
    return word.capitalize() if cap else word


def phase_title(name: str) -> str:
    return name.replace("_", " ").title()


def run_title(run_id: str) -> str:
    n = run_id.rsplit("_", 1)[-1]
    if run_id.startswith("generated_"):
        return f"Generated {noun()} (seed {n})"
    return f"{noun(cap=True)} {int(n)}" if n.isdigit() else run_id


def instructor_name(raw: str | None) -> str:
    """"instructor_3" -> "Instructor 3"; no name (or the generator's placeholder) -> "—"."""
    return "—" if raw in (None, "", "generated") else raw.replace("_", " ").title()


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


def sample_dir(prof: Profile) -> Path:
    return config.RAW_DIR / "sample" / prof.domain


def rules_folder(prof: Profile) -> Path:
    """Where the procedure's built-in official documents live."""
    return config.REFERENCES_DIR / prof.references


def ensure_rules(prof: Profile) -> None:
    """The built-in official documents are always in force: add any that are missing, and add a
    document again when its file was edited (call once per run). Existing guidelines are
    re-checked against the new rules."""
    changed = references.sync_builtin(prof.procedure, rules_folder(prof))
    if changed and store.latest(prof.procedure) is not None:
        refresh_guidelines(prof, "Official rules updated: " + ", ".join(d.label for d in changed))


def start_over(prof: Profile) -> None:
    """Delete everything for the procedure, behind a confirmation (sidebar popover)."""
    st.caption(f"Deletes every {noun()}, guideline version, uploaded document, instructor correction and "
               f"generated {noun()} for this procedure. The built-in official rules stay.")
    sure = st.checkbox("Yes, delete all the data", key="reset-sure")
    if st.button("Delete all data", type="primary", icon=":material/delete_forever:", disabled=not sure,
                 key="reset-go", width="stretch"):
        reset_all(prof)
        st.session_state["toast"] = "All data deleted. The built-in official rules stay in force."
        st.rerun()


def load_sample(prof: Profile) -> None:
    """First-run shortcut: five generated instructor runs, then the guidelines are created
    (against the built-in official rules) and the review pops up."""
    from sim2sop.generate import generate
    from sim2sop.ingest import ingest_run

    references.sync_builtin(prof.procedure, rules_folder(prof))
    with st.spinner(f"Generating five instructor {noun(2)}…"):
        for run_dir in generate(prof.domain, n=5, out_dir=sample_dir(prof)):
            ingest_run(run_dir, prof)
    refresh_guidelines(prof, f"Sample data loaded: 5 {noun(2)}")


def reset_all(prof: Profile) -> None:
    """Delete every run, guideline version, uploaded document, correction and generated run
    for the procedure. The built-in official rules are put straight back."""
    storage.delete_runs(prof.procedure)
    shutil.rmtree(config.GUIDELINES_DIR / prof.procedure, ignore_errors=True)
    references.clear_references(prof.procedure)
    shutil.rmtree(config.MEMORY_DIR, ignore_errors=True)
    shutil.rmtree(generated_dir(prof), ignore_errors=True)
    shutil.rmtree(sample_dir(prof), ignore_errors=True)
    telemetry.clear()
    for k in ("review_queue", "gen_current"):
        st.session_state.pop(k, None)
    references.sync_builtin(prof.procedure, rules_folder(prof))


# --- building and reviewing ----------------------------------------------------------------------

def run_build(label: str, fn):
    """Run fn(progress) inside a status box with a progress bar."""
    with st.status(label, expanded=True) as status:
        provider = st.session_state.get("llm_provider", config.LLM_PROVIDER)
        if provider != "none":
            st.caption(f":material/smart_toy: The AI writer ({st.session_state.get('llm_model') or provider}) is "
                       "wording each guideline, which can take a while per step. If it stops answering, the "
                       "built-in wording is used for the rest. For instant results, switch the AI writer to "
                       "*No AI* in the sidebar.")
        bar = st.progress(0.0, text=f"Analysing the {noun(2)} and checking the official limits…")

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
    if cut := next((f.message for f in doc.flags if f.message.startswith("The AI writer")), None):
        st.session_state["warn"] = cut
    if doc.pending:
        st.session_state["review_queue"] = {"version": doc.version, "title": title}
    else:
        st.session_state["toast"] = f"{title}: the guidelines did not need to change."
    return doc


def show_queued() -> None:
    """Toasts and the review popup queued by the previous action (call once per run)."""
    if msg := st.session_state.pop("toast", None):
        st.toast(msg, icon=":material/info:")
    if msg := st.session_state.pop("warn", None):
        st.toast(msg, icon=":material/warning:", duration="long")
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
    return s.change in ("added", "moved") or any(r.startswith(IMPORTANT) for r in s.change_reasons)


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
    _capped(doc, prev, prof)
    _cited(doc, prev)

    if prev is None:
        st.markdown(f"**{len(doc.steps)} new guidelines** from {len(doc.runs)} {noun(len(doc.runs))}. "
                    "Read them through, then publish.")
        with st.expander("All guidelines", expanded=True):
            for phase in doc.phases:
                steps = doc.steps_in(phase)
                if steps:
                    st.markdown(f"**{phase_title(phase)}**\n" + "\n".join(f"- {s.action}" for s in steps))
    else:
        slot_of = {s.id: k for k, s in step_slots(doc.steps).items()}
        prev_slots = step_slots(prev.steps)
        # Values capped at a limit first: they are what a reviewer most needs to see.
        important = sorted((s for s in doc.pending if _is_important(s)), key=lambda s: not _is_capped(s))
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
                if s.change == "moved":
                    old = next((o for o in prev.steps if o.key == s.key and o.phase == s.moved_from), None)
                if old is not None and any((getattr(old, f) or "") != (getattr(s, f) or "") for f, _ in FIELDS):
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
    """The headline numbers of the new version (no arrow when nothing moved)."""
    m, pm = doc.metrics, prev.metrics if prev else {}
    new_runs = len(doc.runs) - len(prev.runs) if prev else 0
    pts = round((m["mean_confidence"] - pm["mean_confidence"]) * 100) if pm else 0
    c = st.columns(3)
    c[0].metric(noun(2, cap=True), len(doc.runs), f"+{new_runs}" if new_runs else None)
    c[1].metric("Guidelines changed" if prev else "New guidelines", len(doc.pending))
    c[2].metric("Confidence", f"{m['mean_confidence']:.0%}", f"{pts:+d} pts" if pts else None)


def _is_capped(s: Step) -> bool:
    return any(f.clipped for f in s.facts.values())


def _capped(doc: Guidelines, prev: Guidelines | None, prof: Profile) -> None:
    """In plain words: each value this version caps at an official limit, and each new
    run that broke a limit (left out of the typical range)."""
    by_id = {s.id: s for s in doc.steps}
    for s in doc.pending:
        broke = [run_title(f.run_id) for f in doc.flags if f.kind == "clipped" and f.step_id == s.id and f.run_id]
        for f in (f for f in s.facts.values() if f.clipped):
            lim = next((x for x in s.limits if x.signal == f.signal), None)
            source = f" — *{lim.text}* ({lim.doc_title})" if lim else ""
            st.error(f"**{phase_title(s.phase)} · {s.label}:** typical {prof.label(f.signal)} capped at "
                     f"**{f.fmt()}**{source}." + (f"  \nFlown past the limit in: {', '.join(broke)}." if broke else ""),
                     icon=":material/vertical_align_top:")
    known = set(prev.runs) if prev else set()
    for f in doc.flags:
        s = by_id.get(f.step_id or "")
        if f.kind == "excluded" and f.run_id not in known and s is not None:
            lim = next((x for x in s.limits if x.text in f.message), None)
            st.error(f"**{phase_title(s.phase)} · {s.label}:** {run_title(f.run_id)} broke a hard limit"
                     + (f" — *{lim.text}* ({lim.doc_title})" if lim else "")
                     + ". It is left out of the typical range.", icon=":material/remove_circle:")


def _cited(doc: Guidelines, prev: Guidelines | None) -> None:
    """Steps that only gained a citation: their wording did not change, so nothing to review."""
    if prev is None:
        return
    old = {k: s.reference and s.reference.doc for k, s in step_slots(prev.steps).items()}
    cited = [s.reference.doc for k, s in step_slots(doc.steps).items()
             if s.review != "pending" and s.reference and s.reference.doc != old.get(k)]
    if cited:
        n = len(cited)
        st.caption(f":material/menu_book: {n} other guideline{'s now cite' if n != 1 else ' now cites'} "
                   f"{', '.join(sorted(set(cited)))}. The wording is unchanged, so there is nothing to review.")


# --- guideline card -----------------------------------------------------------------------------

def _change_badge(s: Step) -> str | None:
    if s.change == "moved":
        return f":violet-badge[Moved from {phase_title(s.moved_from or '')}]"
    return {"added": ":violet-badge[New]", "modified": ":violet-badge[Changed]"}.get(s.change or "")


def step_badges(s: Step, show_change: bool = True) -> str:
    """`show_change` is off on a first version, where every step is new."""
    out = []
    if s.required_by:
        out.append(":green-badge[Required by SOP]")
    elif s.run_count < 2:
        out.append(f":gray-badge[Seen in 1 {noun()}]")
    elif s.mandatory:
        out.append(":blue-badge[Core — every instructor]")
    else:
        out.append(f":orange-badge[Technique — {s.seen_in} of {s.run_count} instructors]")
    if show_change and (badge := _change_badge(s)):
        out.append(badge)
    if s.review == "pending":
        out.append(":orange-badge[Awaiting approval]")
    if s.edited:
        out.append(":gray-badge[Edited by instructor]")
    if not s.verified:
        out.append(":red-badge[Number not verified]")
    return " ".join(out)


def guideline_body(s: Step, doc: Guidelines) -> None:
    rows = [(":material/schedule:", "When", s.trigger), (":material/lightbulb:", "Why", s.intent),
            (":material/straighten:", "Typical",
             s.typical and f"{s.typical} :gray[({len(s.runs)} {noun(len(s.runs))})]")]
    for icon, label, text in rows:
        if text:
            st.markdown(f"{icon} **{label}** — {text}")
    if s.limit:   # "Unchecked" is shown once at the top of the page, not on every step
        st.markdown(f":red[:material/block: **Hard limit** — {s.limit}]")


def _limit_value(lim, prof: Profile) -> str:
    """"max 85 kt", "max 50°" (% and ° sit on the number, as in the guideline text)."""
    unit = prof.unit(lim.signal) or ""
    return f"{lim.describe()}{'' if unit in ('', '%', '°') else ' '}{unit}"


def limit_badges(s: Step, prof: Profile) -> str:
    """Short red chips for the hard limits beside a step, e.g. "max 85 kt"."""
    return " ".join(f":red-badge[:material/block: {_limit_value(lim, prof)}]" for lim in s.limits)


def limit_name(lim, prof: Profile) -> str:
    """A limit in a few words, e.g. "Airspeed max 85 kt", "Max 11 m from where you stopped"."""
    label, value = prof.label(lim.signal), _limit_value(lim, prof)
    if label.startswith("from "):
        return f"{value[:1].upper()}{value[1:]} {label}"
    return f"{label[:1].upper()}{label[1:]} {value}"


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


def checklist_line(n: int, s: Step, prof: Profile, show_change: bool = True) -> str:
    """One line per step: what to do, when, and the hard limit."""
    when = (s.trigger or "").rstrip(".")
    if when[1:2].islower():   # "At about…" -> "at about…", but keep "IAS 70 kt"
        when = when[:1].lower() + when[1:]
    tags = []
    if s.required_by:
        tags.append(":green-badge[Required]")
    elif not s.mandatory:
        tags.append(f":orange-badge[Technique · {s.seen_in}/{s.run_count}]")
    if show_change and (badge := _change_badge(s)):
        tags.append(badge)
    line = f"{n}. **{s.action.rstrip('.')}**" + (f" — {when}" if when else "")
    return " ".join([line, limit_badges(s, prof), *tags]).strip()


def step_chart(s: Step, doc: Guidelines, prof: Profile) -> None:
    run_id = s.anchor_run or doc.anchor_run
    if not run_id or run_id not in {r.run_id for r in storage.list_runs(prof.procedure)}:
        st.caption(f"No {noun()} data available for a chart.")
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
    limits = []
    if fact and fact.lo is not None:
        fig.add_hrect(y0=fact.lo, y1=fact.hi, opacity=0.18, line_width=0, fillcolor="green",
                      annotation_text="typical range", annotation_position="top left")
        limits = [x for x in (fact.limit_max, fact.limit_min) if x is not None]
        for lim in limits:
            fig.add_hline(y=lim, line_color="red", line_dash="dot", annotation_text="hard limit")
    fig.add_vline(x=0, line_dash="dash", annotation_text="action")
    unit = prof.unit(sig) or ""
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                      xaxis_title="seconds from the action", yaxis_title=f"{prof.label(sig)} ({unit})")
    st.plotly_chart(fig, width="stretch")
    st.caption(f"From {run_title(run_id)}. Green band = what instructors typically do"
               + ("; red line = official limit." if limits else "."))
