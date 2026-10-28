"""Step 3: the guidelines themselves."""

from __future__ import annotations

from collections import defaultdict

import streamlit as st
import ui

from sim2sop import guidelines_store as store
from sim2sop import render as render_html
from sim2sop import storage
from sim2sop.pipeline import apply_edit, create_guidelines, update_guidelines

EDITABLE = {"action": "What to do", "trigger": "When", "intent": "Why", "typical": "Typical"}


@st.dialog("Guideline details", width="large")
def details(step_id: str, version: int) -> None:
    prof = ui.profile()
    doc = store.load(prof.procedure, version)
    s = next(x for x in doc.steps if x.id == step_id)
    st.markdown(f"### {s.action}")
    st.markdown(ui.step_badges(s))
    ui.step_chart(s, doc, prof)

    c1, c2 = st.columns(2)
    c1.metric("Confidence", f"{s.confidence:.0%}", help="How consistently instructors do this, and how tight "
              "the typical range is. Rises as more flights are added.")
    c2.metric("Based on", f"{len(s.runs)} flight{'s' if len(s.runs) != 1 else ''}")
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
                                     "match the flight data — see the step's warnings.")
        st.rerun()


def _build_bar(prof, doc) -> None:
    runs = storage.list_runs(prof.procedure)
    if doc is None:
        with st.container(border=True):
            if not runs:
                st.markdown(":material/info: Add instructor flights first.")
                ui.link_button("runs", "Go to flights", key="g-runs")
                return
            st.markdown(f"Create guidelines from **{len(runs)} flight{'s' if len(runs) != 1 else ''}**.")
            if st.button("Create guidelines", type="primary", icon=":material/auto_awesome:"):
                ui.run_build("Creating guidelines", lambda p: create_guidelines(prof, ui.llm(), p))
                st.rerun()
        return
    new = ui.news_since(doc, prof)
    if new["runs"] or new["refs"]:
        bits = [f"{len(new['runs'])} new flight{'s' if len(new['runs']) != 1 else ''}"] if new["runs"] else []
        bits += [f"{len(new['refs'])} new document{'s' if len(new['refs']) != 1 else ''}"] if new["refs"] else []
        with st.container(border=True):
            st.markdown(f":material/update: **{' and '.join(bits)}** since version {doc.version}. "
                        "Updating rewrites only the steps they affect.")
            if st.button("Update guidelines", type="primary", icon=":material/auto_awesome:"):
                new_doc = ui.run_build("Updating guidelines", lambda p: update_guidelines(prof, ui.llm(), p))
                if new_doc.pending:
                    st.session_state["flash"] = (f"Version {new_doc.version}: {len(new_doc.pending)} step(s) changed "
                                                 "and wait for your approval on the Review page.")
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
                    f"{len(doc.runs)} flight{'s' if len(doc.runs) != 1 else ''}")
        st.markdown(ui.doc_badges(doc))
    if doc.pending:
        st.warning(f"{len(doc.pending)} step(s) in this version changed and need your approval.",
                   icon=":material/fact_check:")
        ui.link_button("review", "Review changes", key="g-review")
    st.caption("Advisory guidelines: how instructors typically fly it. Where they differ from an official "
               "document, the official document wins.")

    with st.expander("How to read a guideline", icon=":material/help:"):
        st.markdown(
            "- **When** — the moment to act. **Why** — the instructor's own callout.\n"
            "- **Typical** — the range instructors used across the recorded flights.\n"
            "- :red[**Hard limit**] — from the official documents; never crossed.\n"
            "- :blue-badge[Core] every instructor did it · :orange-badge[Technique] only some did · "
            ":green-badge[Required by SOP] an official document requires it.")

    phases = [p for p in doc.phases if doc.steps_in(p)]
    pick = st.pills("Phase", ["All"] + phases, default="All", format_func=lambda p: p if p == "All" else ui.phase_title(p),
                    key=f"phase-{v}")
    shown = phases if pick in (None, "All") else [pick]
    n = 0
    for phase in shown:
        st.subheader(ui.phase_title(phase))
        for s in doc.steps_in(phase):
            n += 1
            with st.container(border=True):
                head, btn = st.columns([5, 1])
                head.markdown(f"**{n}. {s.action}**")
                head.markdown(ui.step_badges(s))
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
