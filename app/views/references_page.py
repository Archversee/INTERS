"""Step 2: official documents (reference SOPs) that set the hard limits."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
import ui

from sim2sop import config, references
from sim2sop.references import read_doc_meta


def _added(prof, doc) -> None:
    """A new document can change limits: re-check the guidelines (and pop up the review)."""
    ui.refresh_guidelines(prof, f"{doc.label} added")


def _limit_line(lim) -> str:
    where = f" · during {', '.join(ui.phase_title(p) for p in lim.phases)}" if lim.phases else ""
    return f"- {lim.text}{where}"


def render() -> None:
    prof = ui.profile()
    st.title(":material/menu_book: Official documents")
    st.write("The aircraft manual (POH), standard operating procedures and NOTAMs are the authority. "
             "They set the **hard limits** the guidelines must stay inside. They are only read, never changed.")
    st.caption("Optional: without them, guidelines are still created but marked *Unchecked*. "
               "Adding a document re-checks the guidelines straight away.")
    ui.pending_banner(prof, key="refs-review")

    docs = references.list_references(prof.procedure, active_only=False)
    added_ids = {d.id: d for d in docs if d.active}

    samples = [p for p in sorted((config.REFERENCES_DIR / prof.references).glob("*.md"))
               if p.name.lower() != "readme.md"]
    if samples:
        st.subheader("Sample documents")
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

    active = [d for d in docs if d.active]
    st.subheader(f"In use ({len(active)})")
    if not active:
        st.info("No documents yet. Add the POH and the aerodrome SOP to check the guidelines against "
                "official limits.", icon=":material/info:")
        return
    for d in active:
        with st.container(border=True):
            st.markdown(f"**{d.title}** &nbsp; :gray-badge[{ui.DOC_KINDS.get(d.kind, d.kind)}]"
                        + (f" :gray-badge[rev {d.revision}]" if d.revision else ""))
            if d.limits:
                st.markdown("**Hard limits**\n" + "\n".join(_limit_line(lim) for lim in d.limits))
            if d.requirements:
                st.markdown("**Required actions** (always kept in the guidelines)\n"
                            + "\n".join(f"- {r.text}" for r in d.requirements))
            if not d.limits and not d.requirements:
                st.caption("Guidance only: used for citations, sets no limits.")
    old = [d for d in docs if not d.active]
    if old:
        st.caption("Replaced by newer revisions: " + ", ".join(d.label for d in old))
    ui.link_button("guidelines", "Open the guidelines", key="refs-next", primary=False)
