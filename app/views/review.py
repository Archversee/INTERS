"""Step 4: approve or reject each changed guideline, then publish."""

from __future__ import annotations

import streamlit as st
import ui

from sim2sop import guidelines_store as store
from sim2sop.learn.changes import step_slots
from sim2sop.pipeline import approve

FIELDS = [("action", "What to do"), ("trigger", "When"), ("intent", "Why"), ("typical", "Typical"),
          ("limit", "Hard limit")]


def render() -> None:
    prof = ui.profile()
    st.title(":material/fact_check: Review changes")
    if msg := st.session_state.pop("review_flash", None):
        st.success(msg, icon=":material/check_circle:")
    doc = store.latest(prof.procedure)
    if doc is None:
        st.info("Nothing to review yet — create guidelines first.", icon=":material/info:")
        ui.link_button("guidelines", "Go to guidelines", key="r-g")
        return
    if not doc.pending:
        st.success(f"Nothing waiting. Version {doc.version} is {doc.status}.", icon=":material/check_circle:")
        ui.link_button("guidelines", "View guidelines", key="r-view")
        return

    prev = store.load(prof.procedure, doc.parent_version) if doc.parent_version else None
    slot_of = {s.id: k for k, s in step_slots(doc.steps).items()}
    prev_slots = step_slots(prev.steps) if prev else {}

    st.write(f"Version {doc.version} ({ui.mode_title(doc).lower()}) has **{len(doc.pending)} change"
             f"{'s' if len(doc.pending) != 1 else ''}**. For each one: **Approve** keeps the new text, "
             "**Reject** keeps the previous version.")
    if prev is None:
        st.caption("This is the first version, so every guideline is new. Read through and approve.")

    decisions = {}
    for s in doc.pending:
        old = prev_slots.get(slot_of[s.id])
        with st.container(border=True):
            head, choice = st.columns([4, 2])
            head.markdown(f"**{ui.phase_title(s.phase)} · {s.action}**")
            head.markdown(ui.step_badges(s))
            decisions[s.id] = choice.segmented_control(
                "Decision", ["Approve", "Reject"], default="Approve", key=f"dec-{doc.version}-{s.id}",
                label_visibility="collapsed") or "Approve"
            if s.change_reasons:
                st.markdown("**Why it changed**\n" + "\n".join(f"- {r}" for r in s.change_reasons))
            if old is None:
                ui.guideline_body(s, doc)
                continue
            changed = [(f, label) for f, label in FIELDS if (getattr(old, f) or "") != (getattr(s, f) or "")]
            if changed:
                a, b = st.columns(2)
                a.caption("Before")
                b.caption("After")
                for f, label in changed:
                    a.markdown(f"**{label}** — {getattr(old, f) or '—'}")
                    b.markdown(f"**{label}** — {getattr(s, f) or '—'}")
    if doc.removed:
        st.markdown("**Removed in this version:** " + ", ".join(f"{ui.phase_title(s.phase)} · {s.label}"
                                                               for s in doc.removed))

    rejects = [sid for sid, d in decisions.items() if d == "Reject"]
    label = f"Publish version {doc.version}" + (f" ({len(rejects)} rejected)" if rejects else "")
    if st.button(label, type="primary", icon=":material/publish:"):
        approve(prof, doc.version, None, rejects)
        st.session_state["review_flash"] = f"Version {doc.version} published."
        st.rerun()
