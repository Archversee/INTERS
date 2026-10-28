"""Streamlit UI: SOP viewer, version diff, run comparison, instructor edits.

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sim2sop import config, render, sop_store, storage
from sim2sop.agent.llm import get_llm
from sim2sop.generate import generate
from sim2sop.ingest import ingest_run
from sim2sop.metrics import edits_per_version
from sim2sop.pipeline import apply_edit, generate_sop
from sim2sop.profiles import list_profiles, load_profile

st.set_page_config(page_title="Sim-to-SOP", layout="wide")

# --- sidebar: profile, LLM, data ---------------------------------------------------------
with st.sidebar:
    st.title("Sim-to-SOP")
    profile = load_profile(st.selectbox("Procedure profile", list_profiles()))
    provider = st.selectbox("LLM provider", ["ollama", "openai", "anthropic", "none"],
                            index=["ollama", "openai", "anthropic", "none"].index(config.LLM_PROVIDER)
                            if config.LLM_PROVIDER in ("ollama", "openai", "anthropic", "none") else 0)
    model = st.text_input("Model", value=config.LLM_MODEL if provider == config.LLM_PROVIDER else "")
    llm = get_llm(provider, model or None)
    st.caption(f"{llm.name} — {'available' if llm.available() else 'unavailable: template writer'}")

    st.divider()
    runs = storage.list_runs(profile.procedure)
    st.write(f"**{len(runs)} run(s)** ingested")
    raw_dir = config.RAW_DIR / profile.domain
    pending = sorted(d for d in raw_dir.glob("*") if d.is_dir() and d.name not in {r.run_id for r in runs}) \
        if raw_dir.exists() else []
    if st.button("Generate 5 synthetic runs", disabled=profile.domain != "air"):
        generate(profile.domain, n=5)
        st.rerun()
    if pending and st.button(f"Ingest next run ({pending[0].name})"):
        ingest_run(pending[0], profile)
        st.rerun()
    if pending and st.button(f"Ingest all {len(pending)} pending"):
        for d in pending:
            ingest_run(d, profile)
        st.rerun()
    if runs and st.button("Generate next SOP version", type="primary"):
        with st.spinner("Planner → writer → verifier…"):
            generate_sop(profile, llm=llm)
        st.rerun()

versions = sop_store.versions(profile.procedure)
tab_sop, tab_metrics, tab_diff, tab_runs = st.tabs(["SOP", "Metrics", "Version diff", "Runs"])

# --- SOP viewer + edits --------------------------------------------------------------------
with tab_sop:
    if not versions:
        st.info("No SOP yet. Generate or ingest runs, then generate an SOP version.")
    else:
        v = st.selectbox("Version", versions[::-1], format_func=lambda x: f"v{x}")
        sop = sop_store.load(profile.procedure, v)
        left, right = st.columns([3, 2])
        with left:
            # Jinja2 autoescapes all step text (including LLM output), so this HTML is safe to embed.
            st.iframe(render.to_html(sop), height=900)
        with right:
            st.subheader("Instructor edit")
            step = st.selectbox("Step", sop.steps, format_func=lambda s: f"{s.id} [{s.phase}] {s.label}")
            field = st.selectbox("Field", ["action", "trigger", "intent", "expected", "tolerance"])
            text = st.text_area("Text", value=getattr(step, field) or "", key=f"{v}-{step.id}-{field}")
            if st.button("Save edit as new version"):
                new = apply_edit(profile, step.id, field, text, version=v)
                st.success(f"Saved v{new.version}. The correction is stored and reused next generation.")
                st.rerun()

            st.subheader("Step telemetry")
            anchor = next((r for r in runs if r.run_id == sop.anchor_run), None)
            if anchor is not None:
                df = storage.load_telemetry(profile.procedure, anchor.run_id)
                window = df[(df["t"] >= step.t - 20) & (df["t"] <= step.t + 20)]
                fig = go.Figure()
                for sig in profile.context[:3]:
                    if sig in window:
                        fig.add_trace(go.Scatter(x=window["t"], y=window[sig], name=profile.label(sig)))
                fig.add_vline(x=step.t, line_dash="dash")
                fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, width="stretch")
                st.caption(f"Run {anchor.run_id}, ±20 s around the step (times differ per run).")

# --- metrics across versions -----------------------------------------------------------------
with tab_metrics:
    if versions:
        rows = []
        edits = edits_per_version(profile.procedure)
        for v in versions:
            s = sop_store.load(profile.procedure, v)
            rows.append({"version": f"v{v}", **s.metrics, "edits": edits.get(v, 0), "writer": s.writer})
        st.dataframe(pd.DataFrame(rows).set_index("version"), width="stretch")
        dfm = pd.DataFrame(rows)
        st.line_chart(dfm.set_index("version")[["band_width_ratio", "mean_confidence", "verifier_pass_rate"]])

# --- diff ----------------------------------------------------------------------------------
with tab_diff:
    if len(versions) >= 2:
        c1, c2 = st.columns(2)
        a = c1.selectbox("From", versions, index=len(versions) - 2, format_func=lambda x: f"v{x}")
        b = c2.selectbox("To", versions, index=len(versions) - 1, format_func=lambda x: f"v{x}")
        for d in sop_store.diff(sop_store.load(profile.procedure, a), sop_store.load(profile.procedure, b)):
            st.markdown(f"**{d['change']}** — {d['label']}")
            for f, (old, new) in d.get("fields", {}).items():
                st.markdown(f"- *{f}*: ~~{old}~~ → {new}")

# --- runs ----------------------------------------------------------------------------------
with tab_runs:
    if runs:
        rid = st.selectbox("Run", [r.run_id for r in runs])
        rec = storage.load_run(profile.procedure, rid)
        df = storage.load_telemetry(profile.procedure, rid)
        sig = st.multiselect("Signals", [c for c in df.columns if c != "t"], default=profile.context[:2])
        fig = go.Figure()
        for s in sig:
            fig.add_trace(go.Scatter(x=df["t"], y=df[s], name=s))
        for ph in rec.phases:
            fig.add_vrect(x0=ph.t_start, x1=ph.t_end, annotation_text=ph.name, opacity=0.06, line_width=0)
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, width="stretch")
        st.dataframe(pd.DataFrame([{"t": e.t, "phase": e.phase, "event": e.label, "callout": e.annotation}
                                   for e in rec.events]), width="stretch")
        for f in rec.flags:
            st.warning(f.message)
