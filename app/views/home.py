"""Home: what the app does, where you are in the workflow, what to do next."""

from __future__ import annotations

import streamlit as st
import ui

from sim2sop import guidelines_store as store
from sim2sop import references, storage


def render() -> None:
    prof = ui.profile()
    runs = storage.list_runs(prof.procedure)
    docs = references.list_references(prof.procedure)
    doc = store.latest(prof.procedure)

    st.title("Sim-to-SOP")
    st.markdown("#### Turn instructor simulator flights into step-by-step guidelines for students — "
                "always inside the official limits.")
    st.caption(f"Procedure: {ui.PROFILE_NAMES.get(st.session_state['profile_name'], prof.title)}")

    msg, page, label = ui.next_action(prof)
    with st.container(border=True):
        st.markdown(f"**:material/assistant_navigation: Next step:** {msg}")
        ui.link_button(page, label, key="home-next")

    st.subheader("Your progress")
    cols = st.columns(4)
    cards = [
        ("1 · Instructor flights", ":material/flight:", "runs",
         f"**{len(runs)}** flight{'s' if len(runs) != 1 else ''} recorded" if runs else "No flights yet",
         bool(runs)),
        ("2 · Official documents", ":material/menu_book:", "references",
         f"**{len(docs)}** document{'s' if len(docs) != 1 else ''} (optional)" if docs
         else "None yet (optional — sets the hard limits)", bool(docs)),
        ("3 · Guidelines", ":material/checklist:", "guidelines",
         f"**Version {doc.version}** · {len(doc.steps)} steps" if doc else "Not created yet", doc is not None),
        ("4 · Review changes", ":material/fact_check:", "review",
         (f"**{len(doc.pending)}** waiting for approval" if doc.pending else "All approved") if doc
         else "Nothing to review", bool(doc) and not doc.pending),
    ]
    for col, (title, icon, page, text, done) in zip(cols, cards):
        with col.container(border=True, height="stretch"):
            st.markdown(f"{icon} **{title}**")
            st.markdown(f":green[:material/check_circle:] {text}" if done else f":gray[:material/radio_button_unchecked:] {text}")
            ui.link_button(page, "Open", primary=False, key=f"home-{page}")

    st.subheader("How it works")
    a, b, c = st.columns(3)
    with a.container(border=True, height="stretch"):
        st.markdown(":material/flight: **Instructors fly it**")
        st.write("Instructors fly the procedure in the simulator and say what they are doing "
                 "(\"Rotating\", \"Flaps 10 on downwind\"). Each flight is recorded.")
    with b.container(border=True, height="stretch"):
        st.markdown(":material/insights: **The system learns the technique**")
        st.write("It finds every action, when it happened and the range instructors typically use. "
                 "More flights make the ranges tighter and more confident.")
    with c.container(border=True, height="stretch"):
        st.markdown(":material/verified_user: **Official limits always win**")
        st.write("The POH, SOPs and NOTAMs set hard limits. Guidelines never go past them; "
                 "anything that does is capped and flagged for the instructor.")

    st.info("New here? The **Guided demo** walks through the whole story in six clicks.",
            icon=":material/play_circle:")
    ui.link_button("demo", "Open the guided demo", icon=":material/play_circle:", primary=False, key="home-demo")
