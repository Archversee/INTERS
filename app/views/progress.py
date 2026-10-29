"""Progress across versions: does it learn?"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import ui

from sim2sop import guidelines_store as store
from sim2sop.metrics import edits_per_version

# The guideline shown first under "Follow one guideline": one that visibly tightens.
FOLLOW = {"air": "toggle:flaps_idx:2", "land": "toggle:steer_lock:-1"}


def render() -> None:
    prof = ui.profile()
    st.title(":material/trending_up: Progress")
    st.write(f"How the guidelines improve as instructors add {ui.noun(2)}, official documents and corrections.")
    versions = store.versions(prof.procedure)
    if versions:
        docs = [store.load(prof.procedure, v) for v in versions]
        _history(prof, docs)
        _follow(prof, docs)
    else:
        st.info(f"No guidelines yet. Upload a {ui.noun()} on *1 · Instructor {ui.noun(2)}* to create them.",
                icon=":material/info:")


def _follow(prof, docs) -> None:
    """One guideline, version by version: the quickest way to see it learn."""
    latest = docs[-1]
    keys = list(dict.fromkeys(s.key for s in latest.steps))
    labels = {s.key: f"{ui.phase_title(s.phase)} · {s.label}" for s in reversed(latest.steps)}
    st.subheader("Follow one guideline")
    pick = FOLLOW.get(prof.domain)
    key = st.selectbox("Guideline", keys, format_func=labels.get, key="follow",
                       index=keys.index(pick) if pick in keys else 0)
    rows, runs_col = [], ui.noun(2, cap=True)
    for d in docs:
        s = next((x for x in d.steps if x.key == key), None)
        if s is None:
            continue
        sig = next((g for g in prof.trigger_for(s.phase) if f"trigger.{g}" in s.facts), None)
        f = s.facts.get(f"trigger.{sig}") if sig else None
        row = {"Version": d.version, "What happened": ui.mode_title(d), runs_col: len(d.runs),
               "Typically at": f.fmt_range() if f else "—", "Confidence": f"{s.confidence:.0%}",
               "Note": "Capped at a hard limit" if f is not None and f.clipped else
               (f"A {ui.noun()} that broke a limit is left out" if len(s.runs) < len(d.runs) else "")}
        same = (runs_col, "Typically at", "Confidence", "Note")
        if rows and all(rows[-1][c] == row[c] for c in same):
            continue  # only the versions where this guideline moved
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _history(prof, docs) -> None:
    edits = edits_per_version(prof.procedure)
    last, prev = docs[-1], (docs[-2] if len(docs) > 1 else None)
    m, pm = last.metrics, (prev.metrics if prev else {})

    def delta(key, pct=True):
        if not prev or key not in pm:
            return None
        d = m[key] - pm[key]
        if abs(d) < 0.005:
            return None   # shows as "+0%": no arrow
        return f"{d:+.0%}" if pct else f"{d:+.2f}"

    def series(key, scale=1.0):
        return [d.metrics.get(key, 0) * scale for d in docs]

    st.subheader(f"Version {last.version} compared with version {prev.version}" if prev else f"Version {last.version}")
    c = st.columns(4)
    c[0].metric(f"Numbers verified against {ui.noun()} data", f"{m['verifier_pass_rate']:.0%}",
                delta("verifier_pass_rate"), border=True, chart_data=series("verifier_pass_rate", 100),
                help=f"Every number in the guidelines is checked against the recorded {ui.noun(2)}.")
    within = m.get("within_limits", -1)
    c[1].metric("Inside official limits", "Unchecked" if within < 0 else f"{within:.0%}", border=True,
                help="Share of limited values inside the hard limits from the official documents.")
    c[2].metric("Typical-range width", f"{m['range_width_ratio']:.2f}", delta("range_width_ratio", False),
                delta_color="inverse", border=True, chart_data=series("range_width_ratio"),
                help=f"1.00 = the default range for a single {ui.noun()}. Lower = instructors agree more closely.")
    c[3].metric("Confidence", f"{m['mean_confidence']:.0%}", delta("mean_confidence"), border=True,
                chart_data=series("mean_confidence", 100),
                help="Average over all steps: how consistently instructors do each step.")

    st.subheader("Version history")
    rows = [{
        "Version": d.version,
        "What happened": ui.mode_title(d),
        ui.noun(2, cap=True): len(d.runs),
        "Checked against official limits": "Yes" if d.checked else "No",
        "Steps changed": int(d.metrics.get("steps_changed", 0)),
        "Instructor edits made on it": edits.get(d.version, 0),
        "Values capped at a limit": int(d.metrics.get("clipped_values", 0)),
        "Status": d.status.title(),
    } for d in docs]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.subheader("Range width and confidence over versions")
    x = [f"v{d.version}" for d in docs]
    fig = go.Figure([go.Scatter(x=x, y=series("range_width_ratio"), name="Typical-range width (lower is better)"),
                     go.Scatter(x=x, y=series("mean_confidence"), name="Confidence")])
    top = max(1.0, *series("range_width_ratio"), *series("mean_confidence"))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.2),
                      yaxis=dict(range=[0, top * 1.05]))
    st.plotly_chart(fig, width="stretch")
