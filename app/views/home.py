"""Home: what the app does, where you are, what to do next."""

from __future__ import annotations

import streamlit as st
import ui

from sim2sop import guidelines_store as store
from sim2sop import references, storage


def _next_step(runs, docs, doc) -> tuple[str, str, str]:
    if not runs:
        return (f"Upload the first instructor {ui.noun()}. The guidelines are created from it automatically.",
                "runs", f"Upload {ui.noun(2)}")
    if not docs:
        return ("Add the official documents (manuals, SOPs) so the guidelines are checked against hard limits.",
                "references", "Add documents")
    return f"Add more {ui.noun(2)} to refine the guidelines, or read them.", "guidelines", "Open the guidelines"


def render() -> None:
    prof = ui.profile()
    runs = storage.list_runs(prof.procedure)
    docs = references.list_references(prof.procedure)
    doc = store.latest(prof.procedure)

    st.title("Sim-to-SOP")
    st.markdown(f"#### Turn instructor simulator {ui.noun(2)} into step-by-step guidelines for students — "
                "always inside the official limits.")
    st.caption(f"Procedure: {ui.PROFILE_NAMES.get(st.session_state['profile_name'], prof.title)}")

    ui.pending_banner(prof, key="home-review")
    msg, page, label = _next_step(runs, docs, doc)
    with st.container(border=True):
        st.markdown(f"**:material/assistant_navigation: Next step:** {msg}")
        if runs or prof.domain not in ui.GENERATED:
            ui.link_button(page, label, key="home-next")
        else:
            a, b = st.columns(2)
            with a:
                if st.button("Load sample data", type="primary", icon=":material/auto_awesome:", width="stretch",
                             help=f"Five generated instructor {ui.noun(2)} and the sample official documents. "
                                  "The guidelines are created straight away."):
                    ui.load_sample(prof)
                    st.rerun()
            with b:
                ui.link_button(page, f"Upload my own {ui.noun(2)}", icon=":material/upload:", primary=False,
                               key="home-next")
            st.caption("New here? Sample data shows the whole workflow in one click. *Start over* in the sidebar "
                       "clears it again.")

    cols = st.columns(3)
    cards = [
        (f"1 · Instructor {ui.noun(2)}", ":material/flight:", "runs",
         f"**{len(runs)}** {ui.noun(len(runs))} recorded" if runs else f"No {ui.noun(2)} yet", bool(runs)),
        ("2 · Official documents", ":material/menu_book:", "references",
         f"**{len(docs)}** document{'s' if len(docs) != 1 else ''} setting the hard limits" if docs
         else "None yet — guidelines will be unchecked", bool(docs)),
        ("3 · Guidelines", ":material/checklist:", "guidelines",
         f"**Version {doc.version}** · {len(doc.steps)} steps · {doc.status}" if doc
         else f"Created automatically from the first {ui.noun()}", doc is not None),
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
        st.write("The official manuals and SOPs set hard limits. Guidelines never go past them; "
                 "anything that does is capped and flagged.")
    with b.container(border=True, height="stretch"):
        st.markdown(f":material/flight: **Each {ui.noun()} teaches it more**")
        st.write(f"Every instructor {ui.noun()} is added to the guidelines straight away: typical ranges tighten, "
                 "and actions only some instructors do become technique notes.")
    with c.container(border=True, height="stretch"):
        st.markdown(":material/fact_check: **You approve every change**")
        st.write(f"After each {ui.noun()} a review window shows exactly what changed and why. "
                 "Approve or reject each change, then publish.")

    st.caption(f"Want a particular kind of {ui.noun()}? Make one in the **{ui.noun(cap=True)} generator** and add it "
               "straight to the "
               "guidelines. To clear everything and start again, use *Start over* in the sidebar.")
