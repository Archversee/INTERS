"""Guided demo: official documents are loaded first, then flights are added one at a time and
the guidelines change as needed. Every step does the real work and pops up the review."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import Callable

import streamlit as st
import ui

from sim2sop import config, references, storage
from sim2sop import guidelines_store as store
from sim2sop.learn import memory
from sim2sop.pipeline import apply_edit
from sim2sop.profiles import Profile, load_profile

PROFILE = "air_c172_circuit"
BASE_DOCS = ("c172_poh_notes.md", "aerodrome_sop_notes.md", "faa_afh_traffic_pattern_notes.md")
NOTAM = "notam_crane.md"


@dataclass
class Item:
    title: str
    story: str
    button: str
    done: Callable[[Profile], bool]
    run: Callable[[Profile], None]


def _ingested(prof: Profile) -> set[str]:
    return {r.run_id for r in storage.list_runs(prof.procedure)}


def _flight(i: int, obstacle: bool = False) -> tuple[Callable, Callable]:
    def target(prof):
        return ui.sample_dirs(prof, obstacle)[i]

    return (lambda prof: target(prof).name in _ingested(prof),
            lambda prof: ui.add_flights(prof, [target(prof)]))


def _notam_done(prof: Profile) -> bool:
    return any(d.kind == "notam" for d in references.list_references(prof.procedure))


def _add_notam(prof: Profile) -> None:
    doc = references.add_reference(prof.procedure, config.REFERENCES_DIR / prof.references / NOTAM)
    ui.refresh_guidelines(prof, f"{doc.label} issued")


def _edit_done(prof: Profile) -> bool:
    return any(c.procedure == prof.procedure for c in memory.all_corrections())


def _edit(prof: Profile) -> None:
    doc = store.latest(prof.procedure)
    flaps10 = next(s for s in doc.steps if s.key == "toggle:flaps_idx:1")
    apply_edit(prof, flaps10.id, "action", f"Select {flaps10.label}, then check the flap position indicator.")
    ui.refresh_guidelines(prof, f"Instructor reworded {flaps10.label}; same wording applied to similar steps")


ITEMS = [
    Item("First instructor flight",
         "The first instructor flies the circuit and calls out each action. The guidelines are created from "
         "this one flight. Watch for the flaps: this instructor selected flaps 20 too fast.",
         "Add flight 1", *_flight(0)),
    Item("Second flight", "A second instructor flies the same circuit. Typical ranges start to form.",
         "Add flight 2", *_flight(1)),
    Item("Third flight", "Only some instructors make a power correction on short final — that becomes a "
         "technique note rather than a core step.", "Add flight 3", *_flight(2)),
    Item("Fourth flight", "With more flights, the ranges get tighter and the confidence rises.",
         "Add flight 4", *_flight(3)),
    Item("Fifth flight", "Five flights: the typical technique is well established.", "Add flight 5", *_flight(4)),
    Item("A NOTAM is issued", "A crane goes up under the downwind leg. The NOTAM says to fly downwind at least "
         "1.5 nm from the runway. The guidelines are re-checked against the new limit.",
         "Add the NOTAM", _notam_done, _add_notam),
    Item("First flight around the obstacle", "An instructor flies a wider circuit. One flight is not enough "
         "to change the technique — it is held back.", "Add obstacle flight 1", *_flight(0, obstacle=True)),
    Item("Second flight around the obstacle", "Still held back: the system waits for 3 flights that agree.",
         "Add obstacle flight 2", *_flight(1, obstacle=True)),
    Item("Third flight around the obstacle", "Three flights agree: a technique change. Only the downwind and "
         "base guidelines are rebuilt from the new flights.", "Add obstacle flight 3", *_flight(2, obstacle=True)),
    Item("An instructor improves the wording", "The instructor rewrites the Flaps 10° step. The same wording "
         "is applied to the other flap steps.", "Make the edit", _edit_done, _edit),
]


def _summary(prof: Profile, before: int | None) -> list[str]:
    doc = store.latest(prof.procedure)
    if doc is None or doc.version == before:
        return ["No new version."]
    m = doc.metrics
    changed = int(m.get("steps_changed", 0))
    if doc.mode == "create":
        lines = [f"Version {doc.version}: **{len(doc.steps)} guidelines created**, "
                 + ("checked against the official limits." if doc.checked else "unchecked.")]
    else:
        lines = [f"Version {doc.version}: **{changed} guideline{'s' if changed != 1 else ''} changed**"
                 if changed else f"Version {doc.version}: nothing needed to change."]
    lines.append(f"Range width {m['range_width_ratio']:.2f} · confidence {m['mean_confidence']:.0%} · "
                 f"{int(m['technique_steps'])} technique note(s)")
    for f in doc.flags:
        if f.kind == "technique_change":
            lines.append(f":material/swap_horiz: {f.message}")
    if doc.mode == "create" and any(f.kind == "clipped" for f in doc.flags):
        lines.append(":material/vertical_align_top: Flaps 20 at 89 kt is above the POH limit of 85 kt — "
                     "the guideline is capped at 85 kt and the flight is flagged.")
    return lines


def _reset(prof: Profile) -> None:
    storage.delete_runs(prof.procedure)
    shutil.rmtree(config.GUIDELINES_DIR / prof.procedure, ignore_errors=True)
    references.clear_references(prof.procedure)
    shutil.rmtree(config.MEMORY_DIR, ignore_errors=True)
    ui.telemetry.clear()
    st.session_state["demo_log"] = {}
    st.session_state.pop("review_queue", None)
    for name in BASE_DOCS:
        references.add_reference(prof.procedure, config.REFERENCES_DIR / prof.references / name)
    ui.sample_dirs(prof)
    ui.sample_dirs(prof, obstacle=True)


def render() -> None:
    st.title(":material/play_circle: Guided demo")
    prof = load_profile(PROFILE)
    if st.session_state.get("profile_name") != PROFILE:
        st.session_state["profile_name"] = PROFILE
    st.write("The official documents are loaded first. Then add the instructor flights one at a time: after each "
             "one the guidelines update and a review window shows what changed and why.")
    log: dict = st.session_state.setdefault("demo_log", {})
    docs = references.list_references(prof.procedure)
    started = {d.id for d in docs} >= {"c172_poh", "aerodrome_sop", "faa_afh"}

    with st.container(border=True):
        if started:
            st.markdown(":green[:material/check_circle:] **Official documents loaded:** "
                        + ", ".join(d.label for d in docs if d.kind != "notam"))
            st.caption(f"{sum(len(d.limits) for d in docs)} hard limits and "
                       f"{sum(len(d.requirements) for d in docs)} required actions in force.")
        else:
            st.markdown("**Step 0 · Load the official documents** — the Cessna 172 manual (POH), the aerodrome "
                        "SOP and a flying handbook. They set the hard limits before any flight is added.")
            if store.versions(prof.procedure) or storage.list_runs(prof.procedure):
                st.caption(":material/warning: Starting the demo clears the existing data for this procedure.")
        if st.button("Restart the demo" if started else "Start the demo", icon=":material/restart_alt:",
                     type="secondary" if started else "primary"):
            _reset(prof)
            st.rerun()
    if not started:
        return
    ui.pending_banner(prof, key="demo-review")

    next_i = next((i for i, it in enumerate(ITEMS) if not it.done(prof)), None)
    for i, it in enumerate(ITEMS):
        done = it.done(prof)
        with st.container(border=True):
            mark = (":green[:material/check_circle:]" if done else
                    ":blue[:material/arrow_circle_right:]" if i == next_i else ":gray[:material/radio_button_unchecked:]")
            st.markdown(f"{mark} **{i + 1}. {it.title}**")
            if done or i == next_i:
                st.caption(it.story)
            if done:
                for line in log.get(i, []):
                    st.markdown(line)
            elif i == next_i and st.button(it.button, type="primary", icon=":material/play_arrow:", key=f"demo-{i}"):
                before = store.latest(prof.procedure)
                it.run(prof)
                log[i] = _summary(prof, before.version if before else None)
                st.rerun()
    if next_i is None:
        st.success("Demo complete. The Progress page shows how the guidelines improved, version by version.",
                   icon=":material/celebration:")
        ui.link_button("progress", "Open Progress", key="demo-progress")
