"""Step 1: instructor flights."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
import ui

from sim2sop import config, references, storage
from sim2sop.ingest import ingest_run


def render() -> None:
    prof = ui.profile()
    st.title(":material/flight: Instructor flights")
    st.write("Each flight is one instructor flying the procedure in the simulator, with their spoken "
             "callouts. The more flights you add, the better the guidelines describe the typical technique.")

    runs = storage.list_runs(prof.procedure)
    ui.pending_banner(prof, key="runs-review")
    remaining = ui.remaining_samples(prof)
    obstacle_left = ui.remaining_samples(prof, obstacle=True)

    with st.container(border=True):
        st.markdown("**Add sample flights** (generated in the simulator's own data format). "
                    "The guidelines update straight after each flight and show you what changed.")
        c1, c2, c3 = st.columns(3)
        if c1.button(f"Add {ui.run_title(remaining[0].name)}" if remaining else "All 5 flights added",
                     icon=":material/add:", disabled=not remaining, width="stretch", type="primary"):
            ui.add_flights(prof, remaining[:1])
            st.rerun()
        if c2.button(f"Add all {len(remaining)} remaining" if len(remaining) > 1 else "Add all remaining",
                     icon=":material/playlist_add:", disabled=len(remaining) < 2, width="stretch"):
            ui.add_flights(prof, remaining)
            st.rerun()
        if c3.button(f"Add {ui.run_title(obstacle_left[0].name).lower()}" if obstacle_left
                     else "All obstacle flights added", icon=":material/construction:",
                     disabled=not obstacle_left or bool(remaining), width="stretch",
                     help="Instructors flying a wider circuit around a new crane (add the NOTAM document too). "
                          "Available once the 5 normal flights are in."):
            ui.add_flights(prof, obstacle_left[:1])
            st.rerun()
        if prof.domain != "air":
            st.caption("Sample flights are only available for the air procedure so far. Upload your own below.")

    with st.expander("Upload a recorded flight", icon=":material/upload:"):
        tele = st.file_uploader("Telemetry file from the simulator (CSV)", type=["csv", "parquet", "nmea", "log", "txt"])
        notes = st.file_uploader("Callouts (annotations.json, optional)", type=["json"])
        env = st.text_input("Conditions tag (optional)", placeholder="e.g. new-obstacle",
                            help="Tag flights flown under new conditions. Once 3 tagged flights consistently "
                                 "differ, the guidelines treat it as a technique change.")
        if tele is not None and st.button("Add this flight", type="primary"):
            with tempfile.TemporaryDirectory() as tmp:
                d = Path(tmp)
                (d / tele.name).write_bytes(tele.getvalue())
                if notes is not None:
                    (d / "annotations.json").write_bytes(notes.getvalue())
                rec = ingest_run(d, prof, run_id=Path(tele.name).stem, environment=env or None)
            ui.refresh_guidelines(prof, f"{rec.run_id} added")
            st.rerun()

    st.subheader(f"{len(runs)} flight{'s' if len(runs) != 1 else ''} recorded")
    if not runs:
        st.info("No flights yet. Add the first flight — the guidelines are created from it automatically.",
                icon=":material/info:")
        return
    rows = []
    for r in runs:
        matched = sum(a.event_id is not None for a in r.annotations)
        unmatched = [a.text for a in r.annotations if a.event_id is None]
        mins, secs = divmod(int(r.meta.duration_s), 60)
        rows.append({
            "Flight": ui.run_title(r.run_id),
            "Conditions": r.meta.environment or "Normal",
            "Instructor": (r.meta.instructor or "—").replace("_", " ").title(),
            "Length": f"{mins}:{secs:02d}",
            "Actions found": len(r.events),
            "Callouts matched": f"{matched} of {len(r.annotations)}",
            "Notes": ("Unmatched callout: " + "; ".join(f'"{u}"' for u in unmatched)) if unmatched else "",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    ui.link_button("guidelines", "Open the guidelines", key="runs-next", primary=False)

    with st.expander("Start over", icon=":material/delete:"):
        st.write("Deletes all flights, guidelines, added documents and instructor corrections for this procedure.")
        if st.checkbox("Yes, delete everything for this procedure") and st.button("Delete", type="primary"):
            storage.delete_runs(prof.procedure)
            shutil.rmtree(config.GUIDELINES_DIR / prof.procedure, ignore_errors=True)
            references.clear_references(prof.procedure)
            shutil.rmtree(config.MEMORY_DIR, ignore_errors=True)
            ui.telemetry.clear()
            st.session_state.pop("demo_log", None)
            st.session_state.pop("review_queue", None)
            st.rerun()
