"""Home: what the app does, where you are, what to do next."""

from __future__ import annotations

import streamlit as st
import ui

from sim2sop import guidelines_store as store
from sim2sop import references, storage


def _next_step(runs, docs, doc) -> tuple[str, str, str]:
    if not runs:
        return ("Upload the first instructor flight. The guidelines are created from it automatically.", "runs",
                "Upload flights")
    if not docs:
        return ("Add the official documents (POH, SOP) so the guidelines are checked against hard limits.",
                "references", "Add documents")
    return "Add more flights to refine the guidelines, or read them.", "guidelines", "Open the guidelines"


def render() -> None:
    prof = ui.profile()
    runs = storage.list_runs(prof.procedure)
    docs = references.list_references(prof.procedure)
    doc = store.latest(prof.procedure)

    st.title("Sim-to-SOP")
    st.markdown("#### Turn instructor simulator flights into step-by-step guidelines for students — "
                "always inside the official limits.")
    st.caption(f"Procedure: {ui.PROFILE_NAMES.get(st.session_state['profile_name'], prof.title)}")

    ui.pending_banner(prof, key="home-review")
    msg, page, label = _next_step(runs, docs, doc)
    with st.container(border=True):
        st.markdown(f"**:material/assistant_navigation: Next step:** {msg}")
        ui.link_button(page, label, key="home-next")

    cols = st.columns(3)
    cards = [
        ("Instructor flights", ":material/flight:", "runs",
         f"**{len(runs)}** flight{'s' if len(runs) != 1 else ''} recorded" if runs else "No flights yet", bool(runs)),
        ("Official documents", ":material/menu_book:", "references",
         f"**{len(docs)}** document{'s' if len(docs) != 1 else ''} setting the hard limits" if docs
         else "None yet — guidelines will be unchecked", bool(docs)),
        ("Guidelines", ":material/checklist:", "guidelines",
         f"**Version {doc.version}** · {len(doc.steps)} steps · {doc.status}" if doc
         else "Created automatically from the first flight", doc is not None),
    ]
    for col, (title, icon, page, text, done) in zip(cols, cards):
        with col.container(border=True, height="stretch"):
            st.markdown(f"{icon} **{title}**")
            st.markdown(f":green[:material/check_circle:] {text}" if done
                        else f":gray[:material/radio_button_unchecked:] {text}")
            ui.link_button(page, "Open", primary=False, key=f"home-{page}")

    st.subheader("How it works")
    a, b, c = st.columns(3)
    with a.container(border=True, height="stretch"):
        st.markdown(":material/menu_book: **Official limits come first**")
        st.write("The aircraft manual (POH) and the SOPs set hard limits. Guidelines never go past them; "
                 "anything that does is capped and flagged.")
    with b.container(border=True, height="stretch"):
        st.markdown(":material/flight: **Each flight teaches it more**")
        st.write("Every instructor flight is added to the guidelines straight away: typical ranges tighten, "
                 "and actions only some instructors do become technique notes.")
    with c.container(border=True, height="stretch"):
        st.markdown(":material/fact_check: **You approve every change**")
        st.write("After each flight a review window shows exactly what changed and why. "
                 "Approve or reject each change, then publish.")

    st.caption("No flight data yet? Make some in the **Flight generator**, then upload it. To clear everything and "
               "start again, use *Start over* at the bottom of the **Progress** page.")
