"""Streamlit UI: guidelines viewer, change review, instructor edits, references, metrics,
version diff and run explorer.

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sim2sop import config, references, render, storage
from sim2sop import guidelines_store as store
from sim2sop.agent.llm import get_llm
from sim2sop.generate import generate
from sim2sop.ingest import ingest_run
from sim2sop.metrics import edits_per_version
from sim2sop.pipeline import apply_edit, approve, create_guidelines, update_guidelines
from sim2sop.profiles import list_profiles, load_profile

st.set_page_config(page_title="Sim-to-SOP", layout="wide")
PROVIDERS = ["ollama", "openai", "anthropic", "none"]

# --- sidebar: profile, LLM, data, actions ---------------------------------------------------
with st.sidebar:
    st.title("Sim-to-SOP")
    st.caption("Advisory guidelines from instructor runs. Reference SOPs set the hard limits.")
    profile = load_profile(st.selectbox("Procedure profile", list_profiles()))
    provider = st.selectbox("LLM provider", PROVIDERS,
                            index=PROVIDERS.index(config.LLM_PROVIDER) if config.LLM_PROVIDER in PROVIDERS else 0)
    model = st.text_input("Model", value=config.LLM_MODEL if provider == config.LLM_PROVIDER else "")
    llm = get_llm(provider, model or None)
    st.caption(f"{llm.name} — {'available' if llm.available() else 'unavailable: template writer'}")

    st.divider()
    runs = storage.list_runs(profile.procedure)
    st.write(f"**{len(runs)} run(s)** ingested")
    raw_dir = config.RAW_DIR / profile.domain
    known = {r.run_id for r in runs}
    pending_raw = sorted(d for d in raw_dir.glob("*") if d.is_dir() and d.name not in known) if raw_dir.exists() else []
    c1, c2 = st.columns(2)
    if c1.button("Generate 5 runs", disabled=profile.domain != "air"):
        generate(profile.domain, n=5)
        st.rerun()
    if c2.button("3 obstacle runs", disabled=profile.domain != "air"):
        generate(profile.domain, n=3, scenario="obstacle")
        st.rerun()
    if pending_raw and st.button(f"Ingest next run ({pending_raw[0].name})"):
        ingest_run(pending_raw[0], profile)
        st.rerun()
    if pending_raw and st.button(f"Ingest all {len(pending_raw)} pending"):
        for d in pending_raw:
            ingest_run(d, profile)
        st.rerun()

    st.divider()
    has_doc = bool(store.versions(profile.procedure))
    if runs and not has_doc and st.button("Create guidelines", type="primary"):
        with st.spinner("Planner → writer → verifier…"):
            create_guidelines(profile, llm=llm)
        st.rerun()
    if has_doc and st.button("Update guidelines", type="primary"):
        with st.spinner("Detecting changes, rewriting affected guidelines…"):
            update_guidelines(profile, llm=llm)
        st.rerun()

versions = store.versions(profile.procedure)
tabs = st.tabs(["Guidelines", "Review", "References", "Metrics", "Version diff", "Runs"])

# --- guidelines viewer + edits --------------------------------------------------------------
with tabs[0]:
    if not versions:
        st.info("No guidelines yet. Generate or ingest runs, then create guidelines.")
    else:
        v = st.selectbox("Version", versions[::-1], format_func=lambda x: f"v{x}")
        doc = store.load(profile.procedure, v)
        left, right = st.columns([3, 2])
        with left:
            # Jinja2 autoescapes all text (including LLM output), so this HTML is safe to embed.
            st.iframe(render.to_html(doc), height=900)
        with right:
            step = st.selectbox("Step", doc.steps, format_func=lambda s: f"{s.id} [{s.phase}] {s.label}")
            st.subheader("Step telemetry")
            anchor = step.anchor_run or doc.anchor_run
            if anchor in known:
                df = storage.load_telemetry(profile.procedure, anchor)
                window = df[(df["t"] >= step.t - 20) & (df["t"] <= step.t + 20)]
                sig = st.selectbox("Signal", [s for s in profile.trigger_for(step.phase) if s in window])
                fig = go.Figure(go.Scatter(x=window["t"], y=window[sig], name=profile.label(sig)))
                fact = step.facts.get(f"trigger.{sig}")
                if fact and fact.lo is not None:
                    fig.add_hrect(y0=fact.lo, y1=fact.hi, opacity=0.15, line_width=0,
                                  annotation_text="typical range")
                    for lim, name in ((fact.limit_max, "hard limit (max)"), (fact.limit_min, "hard limit (min)")):
                        if lim is not None:
                            fig.add_hline(y=lim, line_color="red", annotation_text=name)
                fig.add_vline(x=step.t, line_dash="dash")
                fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, width="stretch")
                st.caption(f"Run {anchor}, ±20 s around the step.")

            st.subheader("Instructor edit")
            field = st.selectbox("Field", ["action", "trigger", "intent", "typical"])
            text = st.text_area("Text", value=getattr(step, field) or "", key=f"{v}-{step.id}-{field}")
            st.caption("Hard limits come from the reference SOPs and cannot be edited here.")
            if st.button("Save edit as new version"):
                new = apply_edit(profile, step.id, field, text, version=v)
                st.success(f"Saved v{new.version}. The correction is stored and reused next time.")
                st.rerun()

# --- review pending changes -----------------------------------------------------------------
with tabs[1]:
    doc = store.latest(profile.procedure)
    if doc is None:
        st.info("Nothing to review.")
    elif not doc.pending:
        st.success(f"v{doc.version} is {doc.status}; no changes awaiting approval.")
    else:
        st.write(f"**v{doc.version}** ({doc.mode}): {len(doc.pending)} change(s) awaiting approval")
        rejects = []
        for s in doc.pending:
            with st.container(border=True):
                st.markdown(f"**{s.id} [{s.phase}] {s.action}** — {s.change}")
                for r in s.change_reasons:
                    st.markdown(f"- {r}")
                if s.typical:
                    st.caption(f"Typical: {s.typical}")
                if s.limit:
                    st.caption(f"Hard limit: {s.limit}")
                if st.checkbox("Reject (keep previous version)", key=f"rej-{doc.version}-{s.id}"):
                    rejects.append(s.id)
        if doc.removed:
            st.write("Removed: " + ", ".join(s.label for s in doc.removed))
        if st.button("Apply review and publish", type="primary"):
            approve(profile, doc.version, None, rejects)
            st.rerun()

# --- reference SOPs ---------------------------------------------------------------------------
with tabs[2]:
    st.write("Reference SOPs, POH, regulations and NOTAMs are **read-only**: they set hard limits and "
             "are never edited. Guidelines without them are *unchecked*.")
    docs = references.list_references(profile.procedure, active_only=False)
    if docs:
        st.dataframe(pd.DataFrame([{"id": d.id, "document": d.label, "kind": d.kind,
                                    "state": "active" if d.active else "superseded",
                                    "limits": len(d.limits), "required actions": len(d.requirements)}
                                   for d in docs]), width="stretch")
        for d in docs:
            if d.active and (d.limits or d.requirements):
                with st.expander(f"{d.label}: limits and required actions"):
                    for lim in d.limits:
                        st.markdown(f"- **{lim.text}** — `{lim.signal}` {lim.describe()}"
                                    + (f" when `{lim.when}`" if lim.when else "")
                                    + (f" in {', '.join(lim.phases)}" if lim.phases else ""))
                    for r in d.requirements:
                        st.markdown(f"- Required: **{r.text}** (`{r.key}`)")
    samples = sorted((config.REFERENCES_DIR / profile.references).glob("*.md"))
    samples = [p for p in samples if p.name.lower() != "readme.md"]
    if samples:
        pick = st.multiselect("Add sample references", samples, format_func=lambda p: p.name)
        if pick and st.button("Add selected"):
            for p in pick:
                references.add_reference(profile.procedure, p)
            st.rerun()
    up = st.file_uploader("Or upload a reference (.md with front matter, .pdf, .txt)")
    if up is not None and st.button("Add upload"):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / up.name
            path.write_bytes(up.getvalue())
            references.add_reference(profile.procedure, path)
        st.rerun()

# --- metrics across versions ----------------------------------------------------------------
with tabs[3]:
    if versions:
        edits = edits_per_version(profile.procedure)
        rows = []
        for v in versions:
            d = store.load(profile.procedure, v)
            rows.append({"version": f"v{v}", "mode": d.mode, "status": d.status,
                         "checked": d.checked, **d.metrics, "edits": edits.get(v, 0), "writer": d.writer})
        dfm = pd.DataFrame(rows).set_index("version")
        st.dataframe(dfm, width="stretch")
        st.line_chart(dfm[["range_width_ratio", "mean_confidence", "verifier_pass_rate"]])

# --- diff -------------------------------------------------------------------------------------
with tabs[4]:
    if len(versions) >= 2:
        c1, c2 = st.columns(2)
        a = c1.selectbox("From", versions, index=len(versions) - 2, format_func=lambda x: f"v{x}")
        b = c2.selectbox("To", versions, index=len(versions) - 1, format_func=lambda x: f"v{x}")
        for d in store.diff(store.load(profile.procedure, a), store.load(profile.procedure, b)):
            st.markdown(f"**{d['change']}** — {d['label']}")
            for r in d.get("reasons") or []:
                st.markdown(f"- *why:* {r}")
            for f, (old, new) in (d.get("fields") or {}).items():
                st.markdown(f"- *{f}*: ~~{old}~~ → {new}")

# --- runs -------------------------------------------------------------------------------------
with tabs[5]:
    if runs:
        rid = st.selectbox("Run", [r.run_id for r in runs],
                           format_func=lambda r: r + (f" [{storage.load_run(profile.procedure, r).meta.environment}]"
                                                      if storage.load_run(profile.procedure, r).meta.environment else ""))
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
