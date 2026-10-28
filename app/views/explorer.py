"""Advanced: raw flight data with detected phases and actions."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import ui

from sim2sop import storage


def render() -> None:
    prof = ui.profile()
    st.title(":material/monitoring: Flight data explorer")
    st.write("What the system extracted from one flight: the phases it detected, the actions it found, and "
             "which spoken callout explains each action. The AI never sees this raw data — only the action list.")
    runs = storage.list_runs(prof.procedure)
    if not runs:
        st.info("No flights yet.", icon=":material/info:")
        return
    rid = st.selectbox("Flight", [r.run_id for r in runs], format_func=ui.run_title)
    rec = next(r for r in runs if r.run_id == rid)
    df = ui.telemetry(prof.procedure, rid)
    options = [c for c in df.columns if c != "t" and c in prof.signals]
    sig = st.multiselect("Signals", options, default=[s for s in prof.context[:2] if s in options],
                         format_func=lambda s: f"{prof.label(s)} ({prof.unit(s)})" if prof.unit(s) else prof.label(s))
    fig = go.Figure()
    for s in sig:
        fig.add_trace(go.Scatter(x=df["t"], y=df[s], name=prof.label(s)))
    for ph in rec.phases:
        fig.add_vrect(x0=ph.t_start, x1=ph.t_end, annotation_text=ui.phase_title(ph.name),
                      annotation_position="top left", opacity=0.06, line_width=0)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="seconds")
    st.plotly_chart(fig, width="stretch")

    st.subheader(f"{len(rec.events)} actions found")
    st.dataframe(pd.DataFrame([{"Time (s)": round(e.t, 1), "Phase": ui.phase_title(e.phase or ""),
                                "Action": e.label, "Instructor callout": e.annotation or ""}
                               for e in rec.events]), hide_index=True, width="stretch")
    for f in rec.flags:
        st.warning(f.message, icon=":material/mic_off:")
