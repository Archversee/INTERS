"""Step 2: the guidelines themselves."""

from __future__ import annotations

from collections import defaultdict

import streamlit as st
import ui

from sim2sop import guidelines_store as store
from sim2sop import references, storage
from sim2sop import render as render_html
from sim2sop.pipeline import apply_edit

EDITABLE = {"action": "What to do", "trigger": "When", "intent": "Why", "typical": "Typical"}


@st.dialog("Guideline details", width="large")
def details(step_id: str, version: int) -> None:
    prof = ui.profile()
    doc = store.load(prof.procedure, version)
    s = next(x for x in doc.steps if x.id == step_id)
    st.markdown(f"### {s.action}")
    st.markdown(ui.step_badges(s, show_change=doc.parent_version is not None,
                               old=ui.previous_steps(doc).get(s.id)))
    ui.step_chart(s, doc, prof)

    c1, c2 = st.columns(2)
    c1.metric("Confidence", f"{s.confidence:.0%}", help="How consistently instructors do this, and how tight "
              f"the typical range is. Rises as more {ui.noun(2)} are added.")
    c2.metric("Based on", f"{len(s.runs)} {ui.noun(len(s.runs))}")
    if s.reference:
        st.caption(f":material/menu_book: Source: {s.reference.doc} — {s.reference.section}")
    if s.required_by:
        st.caption(f":material/verified: Required by {s.required_by}")
    for n in s.technique_notes:
        st.caption(f":material/info: {n}")
    for f in s.flags:
        st.caption(f":red[:material/warning: {f}]")
    if s.change_reasons:
        st.markdown("**Why this changed in this version**\n" + "\n".join(f"- {r}" for r in s.change_reasons))

    st.divider()
    st.markdown("**Improve the wording**")
    st.caption("Your edit is saved as a new version and remembered: similar steps will be written the same "
               "way next time. Hard limits come from the official documents and can't be edited here.")
    field = st.segmented_control("Part", list(EDITABLE), default="action", format_func=EDITABLE.get,
                                 key=f"edit-field-{s.id}") or "action"
    text = st.text_area("Text", value=getattr(s, field) or "", key=f"edit-{version}-{s.id}-{field}")
    if st.button("Save edit", type="primary", icon=":material/save:"):
        new = apply_edit(prof, s.id, field, text, version=version)
        edited = next(x for x in new.steps if x.id == s.id)
        st.session_state["flash"] = (f"Saved as version {new.version}." if edited.verified else
                                     f"Saved as version {new.version}, but a number in your text does not "
                                     f"match the {ui.noun()} data — see the step's warnings.")
        # Apply the same wording to similar steps right away; the review pops up if any changed.
        ui.refresh_guidelines(prof, f"Your edit to {s.label}, applied to similar steps")
        st.rerun()


def _build_bar(prof, doc) -> None:
    """Guidelines normally update by themselves when a run or document is added. This covers
    data added another way (e.g. the command line)."""
    runs = storage.list_runs(prof.procedure)
    if doc is None:
        with st.container(border=True):
            st.markdown(f":material/info: The guidelines are created automatically when you add the first "
                        f"{ui.noun()}.")
            ui.link_button("runs", f"Add {ui.noun(2)}", key="g-runs")
        return
    new_runs = [r for r in runs if r.run_id not in doc.runs]
    new_docs = [d for d in references.list_references(prof.procedure) if d.label not in doc.references]
    if new_runs or new_docs:
        with st.container(border=True):
            st.markdown(f":material/update: There is data the guidelines do not include yet "
                        f"({len(new_runs)} {ui.noun(len(new_runs))}, {len(new_docs)} document(s)).")
            if st.button("Update the guidelines", type="primary", icon=":material/auto_awesome:"):
                ui.refresh_guidelines(prof, "Guidelines updated")
                st.rerun()


def render() -> None:
    prof = ui.profile()
    st.title(":material/checklist: Guidelines")
    if msg := st.session_state.pop("flash", None):
        st.success(msg, icon=":material/check_circle:")

    versions = store.versions(prof.procedure)
    latest = store.load(prof.procedure, versions[-1]) if versions else None
    _build_bar(prof, latest)
    if latest is None:
        return

    top = st.columns([3, 1])
    v = top[1].selectbox("Version", versions[::-1], format_func=lambda x: f"Version {x}"
                         + (" (latest)" if x == versions[-1] else ""), label_visibility="collapsed")
    doc = store.load(prof.procedure, v)
    with top[0]:
        st.markdown(f"**{doc.title}** · version {doc.version} · {ui.mode_title(doc).lower()} from "
                    f"{len(doc.runs)} {ui.noun(len(doc.runs))}")
        st.markdown(ui.doc_badges(doc))
    ui.pending_banner(prof, key="g-review")
    st.caption("Advisory guidelines: how instructors typically fly it. Where they differ from an official "
               "document, the official document wins.")

    view = st.segmented_control("View", ["Checklist", "Full detail"], default="Checklist", key="g-view",
                                label_visibility="collapsed") or "Checklist"
    phases = [p for p in doc.phases if doc.steps_in(p)]
    show_change = doc.parent_version is not None   # on a first version every step is new
    before = ui.previous_steps(doc)
    if view == "Checklist":
        st.caption("One line per step: **what to do** — when to do it · :red-badge[hard limit] from the official "
                   "documents · :violet-badge[Changed: Typical] what changed since the previous version. Switch to "
                   "*Full detail* for typical ranges, the instructor's reason, charts and edits.")
        n = 0
        for phase in phases:
            with st.container(border=True):
                st.markdown(f"**{ui.phase_title(phase)}**")
                lines = []
                for s in doc.steps_in(phase):
                    n += 1
                    lines.append(ui.checklist_line(n, s, prof, show_change, before.get(s.id)))
                st.markdown("\n".join(lines))
    else:
        with st.expander("How to read a guideline", icon=":material/help:"):
            st.markdown(
                "- **When** — the moment to act. **Why** — the instructor's own callout.\n"
                f"- **Typical** — the range instructors used across the recorded {ui.noun(2)}.\n"
                "- :red[**Hard limit**] — from the official documents; never crossed.\n"
                "- :blue-badge[Core] every instructor did it · :orange-badge[Technique] only some did · "
                ":green-badge[Required by SOP] an official document requires it.")
        pick = st.pills("Phase", ["All"] + phases, default="All",
                        format_func=lambda p: p if p == "All" else ui.phase_title(p), key=f"phase-{v}")
        shown = phases if pick in (None, "All") else [pick]
        n = 0
        for phase in shown:
            st.subheader(ui.phase_title(phase))
            for s in doc.steps_in(phase):
                n += 1
                with st.container(border=True):
                    head, btn = st.columns([5, 1])
                    head.markdown(f"**{n}. {s.action}**")
                    head.markdown(ui.step_badges(s, show_change, before.get(s.id)))
                    if btn.button("Details", key=f"d-{v}-{s.id}", icon=":material/open_in_full:", width="stretch"):
                        details(s.id, v)
                    ui.guideline_body(s, doc)

    _flags(doc)
    st.download_button("Download printable version (HTML)", render_html.to_html(doc),
                       file_name=f"{doc.procedure}_guidelines_v{doc.version}.html", mime="text/html",
                       icon=":material/download:")


def _flags(doc) -> None:
    groups = defaultdict(list)
    for f in doc.flags:
        groups[f.kind].append(f.message)
    if not groups:
        return
    st.subheader("Things to check")
    for kind, (label, icon) in ui.FLAG_LABELS.items():
        if kind in groups:
            with st.expander(f"{label} ({len(groups[kind])})", icon=icon):
                for m in groups[kind]:
                    st.markdown(f"- {m}")
