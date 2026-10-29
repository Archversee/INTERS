"""See the results: patterns across instructors, found by statistics and explained by the AI."""

from __future__ import annotations

import pandas as pd
import streamlit as st
import ui

from sim2sop import config, storage
from sim2sop import guidelines_store as store
from sim2sop.agent import insights

KINDS = {"instructor": "Instructor difference", "habit": "Instructor habit", "together": "Always together",
         "effect": "Knock-on effect", "limit_break": "Limit broken", "spread": "Least agreement"}


def _find(prof, doc, ai_on: bool) -> None:
    with st.spinner("Comparing the runs" + (" and asking the AI to explain the patterns…" if ai_on else "…")):
        found = insights.explain(doc, storage.list_runs(prof.procedure), prof, ui.llm() if ai_on else None)
    insights.save(found)
    st.rerun()


def render() -> None:
    prof = ui.profile()
    st.title(":material/insights: Patterns")
    st.write(f"How instructors differ from each other. Statistics find the patterns in the {ui.noun(2)}: who does "
             "a step differently, which techniques go together, what the limit-breaking runs have in common. The "
             "AI explains them for the instructor team, and every number it writes is checked against the patterns.")
    doc = store.latest(prof.procedure)
    if doc is None:
        st.info(f"No guidelines yet. Upload {ui.noun(2)} first.", icon=":material/info:")
        return
    ai_on = st.session_state.get("llm_provider", config.LLM_PROVIDER) != "none"
    last = insights.load(prof.procedure)
    if st.button("Find patterns with AI" if ai_on else "Find patterns", type="primary",
                 icon=":material/auto_awesome:", key="patterns-go",
                 help=None if ai_on else "The AI writer is off: you get the patterns without the AI's explanation."):
        _find(prof, doc, ai_on)
    if last is None:
        st.info(f"Press the button to compare the {len(doc.runs)} {ui.noun(len(doc.runs))} behind version "
                f"{doc.version}.", icon=":material/arrow_upward:")
        return
    st.caption(f"From version {last.version} ({last.runs} {ui.noun(last.runs)}) · written by {last.writer} · "
               f"{last.made:%d %b %Y %H:%M} UTC"
               + (f" · the guidelines are now at version {doc.version}: press the button to refresh"
                  if last.version != doc.version else ""))
    if last.note:
        st.info(last.note, icon=":material/info:")

    by_id = {p.id: p for p in last.patterns}
    st.subheader("Findings")
    for f in last.findings:
        with st.container(border=True):
            st.markdown(f"**{f.headline}**")
            if f.detail:
                st.write(f.detail)
            if f.action:
                st.markdown(f":material/task_alt: **What to do:** {f.action}")
            st.caption("Based on: " + " · ".join(f"{p} {by_id[p].text}" for p in f.patterns if p in by_id))
    if last.set_aside:
        with st.expander(f"AI findings set aside by the number check ({len(last.set_aside)})", icon=":material/block:"):
            for f in last.set_aside:
                st.markdown(f"- {f.headline} — :red[{'; '.join(f.problems)}]")

    st.subheader("All patterns found in the data")
    st.dataframe(pd.DataFrame([{"Id": p.id, "Kind": KINDS.get(p.kind, p.kind), "Pattern": p.text,
                                ui.noun(2, cap=True): ", ".join(ui.run_title(r) for r in p.runs)}
                               for p in last.patterns]), hide_index=True, width="stretch")
