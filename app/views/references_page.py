"""Rules: official documents (reference SOPs) that set the hard limits."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
import ui

from sim2sop import config, references
from sim2sop.agent import manuals
from sim2sop.schema import Limit


def _added(prof, doc) -> None:
    """A new document can change limits: re-check the guidelines (and pop up the review)."""
    ui.refresh_guidelines(prof, f"{doc.label} added")


def _rules(prof, active) -> None:
    """Every hard limit and required action in force, in plain words."""
    limits = [(d, lim) for d in active for lim in d.limits]
    reqs = [(d, r) for d in active for r in d.requirements]
    st.subheader(f"Rules in force ({len(limits) + len(reqs)})")
    if not limits and not reqs:
        st.info("No rules yet. Upload an official document below to check the guidelines against "
                "official limits.", icon=":material/info:")
        return
    st.caption(f"The guidelines never go past these. A {ui.noun()} that does is flagged, and its value is left out "
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
    st.write("The official manuals and standard operating procedures are the authority. They set the "
             "**hard limits** the guidelines must stay inside. They are only read, never changed.")
    st.caption(":material/lock: The built-in documents are always in force and cannot be removed or replaced. "
               "You can upload more of your own; adding one re-checks the guidelines straight away.")
    ui.pending_banner(prof, key="refs-review")

    docs = references.list_references(prof.procedure, active_only=False)
    active = [d for d in docs if d.active]
    _rules(prof, active)

    st.subheader("Documents in force")
    cols = st.columns(2)
    for i, d in enumerate(sorted(active, key=lambda d: not d.builtin)):
        n_lim, n_req = len(d.limits), len(d.requirements)
        with cols[i % 2].container(border=True):
            st.markdown(f"**{d.title}**" + (f" · rev {d.revision}" if d.revision else ""))
            st.caption(f"{ui.DOC_KINDS.get(d.kind, 'Document')} · {n_lim} hard limit{'s' if n_lim != 1 else ''}"
                       + (f" · {n_req} required action{'s' if n_req != 1 else ''}" if n_req else ""))
            st.markdown(":blue-badge[:material/lock: Built in · always in force]" if d.builtin
                        else ":gray-badge[:material/upload: Uploaded]")

    _ai_manual(prof)

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
                try:
                    doc = references.add_reference(prof.procedure, path)
                except references.BuiltinDocument as e:
                    st.error(f"{e}. Give your document a different `id` to add it alongside.",
                             icon=":material/lock:")
                    return
            _added(prof, doc)
            st.rerun()

    old = [d for d in docs if not d.active]
    if old:
        st.caption("Replaced by newer revisions: " + ", ".join(d.label for d in old))
    if active:
        ui.link_button("guidelines", "Open the guidelines", key="refs-next", primary=False)


# --- reading a manual with the AI ----------------------------------------------------------------

def _rule_words(prof, p: manuals.Proposal) -> tuple[str, str]:
    """(the rule in a few words, when it applies)."""
    if p.kind == "requirement":
        return f"Must do: {manuals.actions(prof).get(p.key, p.key)}", "—"
    lim = Limit(**p.as_rule())
    return ui.limit_name(lim, prof), ui.when_text(lim, prof)


def _ai_manual(prof) -> None:
    """Upload a manual with no limits list; the AI proposes its rules; the instructor ticks the
    ones to put in force."""
    ss = st.session_state
    ai_on = ss.get("llm_provider", config.LLM_PROVIDER) != "none"
    with st.container(border=True):
        st.markdown("**:material/smart_toy: Read a manual with AI**")
        st.caption("Upload a manual without a limits list (PDF, Markdown or text). The AI reads each section that "
                   "states a rule and proposes hard limits and required actions in this procedure's terms. A "
                   "proposal must quote the manual word for word and use a value written in that quote, or it is "
                   "set aside. Nothing is in force until you tick it and add the manual.")
        n = ss.setdefault("manual_n", 0)
        up = st.file_uploader("Manual", type=["pdf", "md", "txt"], key=f"manual-up-{n}")
        c1, c2, c3 = st.columns([3, 2, 1])
        title = c1.text_input("Title", value=Path(up.name).stem.replace("_", " ") if up else "",
                              key=f"manual-title-{n}")
        kinds = list(ui.DOC_KINDS)
        kind = c2.selectbox("Kind", kinds, index=kinds.index("sop"), format_func=ui.DOC_KINDS.get,
                            key=f"manual-kind-{n}")
        rev = c3.text_input("Revision", key=f"manual-rev-{n}")
        if not ai_on:
            st.caption(":material/info: Switch the AI writer on in the sidebar to use this.")
        if st.button("Find the rules with AI", type="primary", icon=":material/auto_awesome:",
                     disabled=not (up and ai_on and title.strip()), key="manual-go"):
            folder = config.DATA_DIR / "uploads" / prof.procedure
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / up.name
            path.write_bytes(up.getvalue())
            with st.status(f"Reading {up.name}", expanded=True) as status:
                bar = st.progress(0.0, text="Finding the sections that state a rule…")
                reading = manuals.read_manual(
                    ui.llm(), path, prof, title=title.strip(),
                    progress=lambda i, total, msg: bar.progress(min(1.0, (i + 1) / max(total, 1)),
                                                                text=f"{msg}  ({i + 1} of {total})"))
                status.update(label=f"Read {up.name}", state="complete", expanded=False)
            ss["manual"] = {"path": str(path), "title": title.strip(), "kind": kind,
                            "revision": rev.strip() or None, "reading": reading}
            st.rerun()
        if m := ss.get("manual"):
            _manual_review(prof, m)


def _manual_review(prof, m: dict) -> None:
    ss = st.session_state
    reading: manuals.ManualReading = m["reading"]
    ok, aside = reading.accepted, reading.set_aside
    st.markdown(f"**{m['title']}** — read {reading.sections_read} of {reading.sections_total} sections: "
                f"**{len(ok)} rule{'s' if len(ok) != 1 else ''}** passed the checks"
                + (f", {len(aside)} set aside" if aside else "") + ".")
    if reading.stopped:
        st.warning(f"The AI writer stopped answering after {reading.sections_read} sections ({reading.stopped}). "
                   "The rules found so far are below.", icon=":material/warning:")
    chosen = []
    if ok:
        rows, suspect = [], 0
        for p in ok:
            rule, applies = _rule_words(prof, p)
            breaks = manuals.recorded_breaks(p, prof.procedure, prof)
            every = breaks is not None and breaks[0] == breaks[1] >= 2
            suspect += every
            rows.append({"Use": not every, "Rule": rule, "Applies": applies, "The manual says": p.quote,
                         f"Recorded {ui.noun(2)} breaking it": "—" if breaks is None else
                         f"{breaks[0]} of {breaks[1]}" + (" — check the signal" if every else ""),
                         "Section": p.section})
        edited = st.data_editor(pd.DataFrame(rows), hide_index=True, width="stretch",
                                key=f"manual-pick-{ss['manual_n']}",
                                column_config={"Use": st.column_config.CheckboxColumn(width="small")},
                                disabled=[c for c in rows[0] if c != "Use"])
        chosen = [p for p, use in zip(ok, edited["Use"]) if use]
        if suspect:
            st.caption(f":material/warning: Every recorded {ui.noun()} would break {suspect} of these, so they "
                       "are unticked: the AI most likely picked the wrong signal (e.g. steering angle for angle "
                       "to the kerb). Tick one only if the rule really is that strict.")
    else:
        st.info("No rule passed the checks. The manual can still be added for citations.", icon=":material/info:")
    if aside:
        with st.expander(f"Set aside by the checks ({len(aside)})", icon=":material/block:"):
            st.dataframe(pd.DataFrame([{"Proposed": p.text or p.key or p.signal, "The AI quoted": p.quote,
                                        "Why it was set aside": "; ".join(p.problems)} for p in aside]),
                         hide_index=True, width="stretch")
    b1, b2 = st.columns(2)
    if b1.button(f"Add the manual with {len(chosen)} rule{'s' if len(chosen) != 1 else ''}", type="primary",
                 icon=":material/add_task:", width="stretch", key="manual-add"):
        try:
            doc = manuals.add_manual(prof.procedure, Path(m["path"]), m["title"], m["kind"], m["revision"], chosen)
        except references.BuiltinDocument as e:
            st.error(f"{e}. Give the manual a different title to add it alongside.", icon=":material/lock:")
            return
        ss.pop("manual", None)
        ss["manual_n"] += 1
        _added(prof, doc)
        st.rerun()
    if b2.button("Discard", icon=":material/delete:", width="stretch", key="manual-discard"):
        ss.pop("manual", None)
        ss["manual_n"] += 1
        st.rerun()
