"""Step 2: official documents (reference SOPs) that set the hard limits."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
import ui

from sim2sop import config, references
from sim2sop.references import read_doc_meta


def _added(prof, doc) -> None:
    """A new document can change limits: re-check the guidelines (and pop up the review)."""
    ui.refresh_guidelines(prof, f"{doc.label} added")


def _rules(prof, active) -> None:
    """Every hard limit and required action in force, in plain words."""
    limits = [(d, lim) for d in active for lim in d.limits]
    reqs = [(d, r) for d in active for r in d.requirements]
    st.subheader(f"Rules in force ({len(limits) + len(reqs)})")
    if not limits and not reqs:
        st.info("No rules yet. Add the POH and the aerodrome SOP below to check the guidelines against "
                "official limits.", icon=":material/info:")
        return
    st.caption("The guidelines never go past these. A flight that does is flagged, and its value is left out "
               "of the typical range.")
    if limits:
        st.dataframe(pd.DataFrame([{
            "Limit": ui.limit_name(lim, prof),
            "Applies": ui.when_text(lim, prof),
            "The document says": lim.text,
            "From": d.title,
        } for d, lim in limits]), hide_index=True, width="stretch")
    if reqs:
        st.markdown("**Must always be done** (kept in the guidelines even if an instructor skips it)")
        st.dataframe(pd.DataFrame([{"Action": r.text, "From": d.title} for d, r in reqs]),
                     hide_index=True, width="stretch")
    guidance = [d.title for d in active if not d.limits and not d.requirements]
    if guidance:
        st.caption("Guidance only (used for citations, sets no rules): " + ", ".join(guidance))


def render() -> None:
    prof = ui.profile()
    st.title(":material/menu_book: Official documents")
    st.write("The aircraft manual (POH) and the standard operating procedures are the authority. They set the "
             "**hard limits** the guidelines must stay inside. They are only read, never changed.")
    st.caption("Optional: without them, guidelines are still created but marked *Unchecked*. "
               "Adding a document re-checks the guidelines straight away.")
    ui.pending_banner(prof, key="refs-review")

    docs = references.list_references(prof.procedure, active_only=False)
    added_ids = {d.id: d for d in docs if d.active}
    _rules(prof, [d for d in docs if d.active])

    samples = [p for p in sorted((config.REFERENCES_DIR / prof.references).glob("*.md"))
               if p.name.lower() != "readme.md"]
    if samples:
        st.subheader("Add documents")
        cols = st.columns(2)
        for i, path in enumerate(samples):
            meta = read_doc_meta(path)
            doc_id, title = str(meta.get("id") or path.stem), meta.get("title") or path.stem
            kind = ui.DOC_KINDS.get(meta.get("kind", "other"), "Document")
            n_lim, n_req = len(meta.get("limits") or []), len(meta.get("requirements") or [])
            with cols[i % 2].container(border=True):
                st.markdown(f"**{title}**")
                st.caption(f"{kind} · {n_lim} hard limit{'s' if n_lim != 1 else ''}"
                           + (f" · {n_req} required action{'s' if n_req != 1 else ''}" if n_req else ""))
                if doc_id in added_ids:
                    st.markdown(":green[:material/check_circle: Added]")
                elif st.button("Add", key=f"add-{path.name}", icon=":material/add:"):
                    _added(prof, references.add_reference(prof.procedure, path))
                    st.rerun()

    with st.expander("Upload your own document", icon=":material/upload:"):
        st.caption("A Markdown file with its limits listed at the top (see the samples), or a PDF plus a "
                   "`.limits.yaml` file. A newer revision of a document replaces the older one.")
        up = st.file_uploader("Document", type=["md", "txt", "pdf"])
        lim = st.file_uploader("Limits file for a PDF (optional)", type=["yaml", "yml"])
        if up is not None and st.button("Add document", type="primary"):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / up.name
                path.write_bytes(up.getvalue())
                if lim is not None:
                    path.with_suffix(".limits.yaml").write_bytes(lim.getvalue())
                doc = references.add_reference(prof.procedure, path)
            _added(prof, doc)
            st.rerun()

    old = [d for d in docs if not d.active]
    if old:
        st.caption("Replaced by newer revisions: " + ", ".join(d.label for d in old))
    if added_ids:
        ui.link_button("guidelines", "Open the guidelines", key="refs-next", primary=False)
