"""Step 1: instructor runs (flights, drives, ...). They come in only by uploading recorded data."""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st
import ui

from sim2sop import storage
from sim2sop.ingest import ingest_run

TELEMETRY = (".csv", ".parquet")


def _flight_dirs(root: Path) -> list[Path]:
    """Every folder in an unpacked upload that holds a telemetry file."""
    found = {p.parent for p in root.rglob("*") if p.suffix.lower() in TELEMETRY and "__MACOSX" not in p.parts}
    return sorted(found)


def _ingest(prof, files, notes) -> list[str]:
    """Ingest the uploaded files; returns the names of the flights added."""
    added = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, f in enumerate(files):
            d = Path(tmp) / str(i)
            d.mkdir()
            stem = Path(f.name).stem
            if f.name.lower().endswith(".zip"):
                with zipfile.ZipFile(f) as z:
                    z.extractall(d)
                dirs = _flight_dirs(d)
                if not dirs:
                    st.error(f"{f.name}: no telemetry file (.csv or .parquet) inside.", icon=":material/error:")
                for run_dir in dirs:
                    # A zip of the flight's files (not of its folder) takes the zip's name,
                    # unless its meta.json names the flight.
                    run_id = stem if run_dir == d and not (d / "meta.json").exists() else None
                    added.append(ingest_run(run_dir, prof, run_id=run_id).run_id)
            else:
                (d / f.name).write_bytes(f.getvalue())
                if notes is not None and len(files) == 1:
                    (d / "annotations.json").write_bytes(notes.getvalue())
                added.append(ingest_run(d, prof, run_id=stem).run_id)
    return added


def _upload(prof) -> None:
    n = st.session_state.setdefault("upload_n", 0)   # bumped after each upload to clear the boxes
    with st.container(border=True):
        st.markdown(f"**Upload {ui.noun()} data**")
        st.caption(f"A **.zip** of one or more {ui.noun()} folders (as downloaded from the {ui.noun(cap=True)} "
                   "generator), or a simulator **telemetry CSV**. The guidelines update straight away and show you "
                   "what changed.")
        files = st.file_uploader(f"{ui.noun(cap=True)} data", type=["zip", "csv", "parquet"],
                                 accept_multiple_files=True, key=f"up-{n}", label_visibility="collapsed")
        notes = None
        if len(files) == 1 and not files[0].name.lower().endswith(".zip"):
            notes = st.file_uploader(f"Callouts for this {ui.noun()} (annotations.json, optional)", type=["json"],
                                     key=f"up-notes-{n}")
        if st.button("Add to the guidelines", type="primary", disabled=not files):
            try:
                added = _ingest(prof, files, notes)
            except Exception as e:  # a malformed file should not crash the page
                st.error(f"Could not read the upload: {e}", icon=":material/error:")
                return
            if added:
                st.session_state["upload_n"] = n + 1
                title = f"{ui.run_title(added[0])} added" if len(added) == 1 else f"{len(added)} {ui.noun(2)} added"
                ui.refresh_guidelines(prof, title)
                st.rerun()
        st.caption(f"Need data? Make a {ui.noun()} in the **{ui.noun(cap=True)} generator** and add it from "
                   "there. The file format is explained there too.")
        ui.link_button("generator", f"Open the {ui.noun(cap=True)} generator",
                       key="runs-gen", primary=False)


def render() -> None:
    prof = ui.profile()
    ui.page_header(f"Instructor {ui.noun(2)}",
                   f"Each {ui.noun()} is one instructor doing the procedure in the simulator, with their spoken "
                   f"callouts. The more {ui.noun(2)} you add, the better the guidelines describe the typical "
                   "technique.", eyebrow="Step 1 · Record")
    ui.pending_banner(prof, key="runs-review")
    _upload(prof)

    runs = storage.list_runs(prof.procedure)
    st.subheader(f"{len(runs)} {ui.noun(len(runs))} recorded")
    if not runs:
        st.info(f"No {ui.noun(2)} yet. Upload the first {ui.noun()} — the guidelines are created from it automatically.",
                icon=":material/info:")
        return
    rows = []
    for r in runs:
        matched = sum(a.event_id is not None for a in r.annotations)
        unmatched = [a.text for a in r.annotations if a.event_id is None]
        mins, secs = divmod(int(r.meta.duration_s), 60)
        rows.append({
            ui.noun(cap=True): ui.run_title(r.run_id),
            "Instructor": ui.instructor_name(r.meta.instructor),
            "Length": f"{mins}:{secs:02d}",
            "Actions found": len(r.events),
            "Callouts matched": f"{matched} of {len(r.annotations)}",
            "Notes": ("Unmatched callout: " + "; ".join(f'"{u}"' for u in unmatched)) if unmatched else "",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    ui.link_button("guidelines", "Open the guidelines", key="runs-next", primary=False)
