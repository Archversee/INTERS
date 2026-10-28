"""Progress across versions: does it learn?"""

from __future__ import annotations

import pandas as pd
import streamlit as st
import ui

from sim2sop import guidelines_store as store
from sim2sop.metrics import edits_per_version


def render() -> None:
    prof = ui.profile()
    st.title(":material/trending_up: Progress")
    st.write("How the guidelines improve as instructors add flights, official documents and corrections.")
    versions = store.versions(prof.procedure)
    if not versions:
        st.info("No guidelines yet.", icon=":material/info:")
        return
    docs = [store.load(prof.procedure, v) for v in versions]
    edits = edits_per_version(prof.procedure)
    last, prev = docs[-1], (docs[-2] if len(docs) > 1 else None)
    m, pm = last.metrics, (prev.metrics if prev else {})

    def delta(key, pct=True):
        if not prev or key not in pm:
            return None
        d = m[key] - pm[key]
        return f"{d:+.0%}" if pct else f"{d:+.2f}"

    def series(key, scale=1.0):
        return [d.metrics.get(key, 0) * scale for d in docs]

    st.subheader(f"Version {last.version} compared with version {prev.version}" if prev else f"Version {last.version}")
    c = st.columns(4)
    c[0].metric("Numbers verified against flight data", f"{m['verifier_pass_rate']:.0%}",
                delta("verifier_pass_rate"), border=True, chart_data=series("verifier_pass_rate", 100),
                help="Every number in the guidelines is checked against the recorded flights.")
    within = m.get("within_limits", -1)
    c[1].metric("Inside official limits", "Unchecked" if within < 0 else f"{within:.0%}", border=True,
                help="Share of limited values inside the hard limits from the official documents.")
    c[2].metric("Typical-range width", f"{m['range_width_ratio']:.2f}", delta("range_width_ratio", False),
                delta_color="inverse", border=True, chart_data=series("range_width_ratio"),
                help="1.00 = the default range for a single flight. Lower = instructors agree more closely.")
    c[3].metric("Confidence", f"{m['mean_confidence']:.0%}", delta("mean_confidence"), border=True,
                chart_data=series("mean_confidence", 100),
                help="Average over all steps: how consistently instructors do each step.")

    st.subheader("Version history")
    rows = [{
        "Version": d.version,
        "What happened": ui.mode_title(d),
        "Flights": len(d.runs),
        "Conditions": d.environment or "Normal",
        "Checked against official limits": "Yes" if d.checked else "No",
        "Steps changed": int(d.metrics.get("steps_changed", 0)),
        "Instructor edits made on it": edits.get(d.version, 0),
        "Values capped at a limit": int(d.metrics.get("clipped_values", 0)),
        "Status": d.status.title(),
    } for d in docs]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.subheader("Range width and confidence over versions")
    chart = pd.DataFrame({"Typical-range width (lower is better)": series("range_width_ratio"),
                          "Confidence": series("mean_confidence")},
                         index=[f"v{d.version}" for d in docs])
    st.line_chart(chart)
