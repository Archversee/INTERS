"""Guided demo: the official rules are loaded first, then flights are added one at a time and
the guidelines learn from each one. Every step does the real work and pops up the review."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import Callable

import pandas as pd
import streamlit as st
import ui

from sim2sop import config, references, storage
from sim2sop import guidelines_store as store
from sim2sop.learn import memory
from sim2sop.pipeline import apply_edit
from sim2sop.profiles import Profile, load_profile

PROFILE = "air_c172_circuit"
BASE_DOCS = ("c172_poh_notes.md", "aerodrome_sop_notes.md", "faa_afh_traffic_pattern_notes.md")
WATCH = ("toggle:flaps_idx:2", "trigger.ias_kt", "Flaps 20°")   # the guideline the demo follows


@dataclass
class Item:
    title: str
    story: str
    button: str
    done: Callable[[Profile], bool]
    run: Callable[[Profile], None]


def _ingested(prof: Profile) -> set[str]:
    return {r.run_id for r in storage.list_runs(prof.procedure)}


def _flight(i: int) -> tuple[Callable, Callable]:
    def target(prof):
        return ui.sample_dirs(prof)[i]

    return (lambda prof: target(prof).name in _ingested(prof),
            lambda prof: ui.add_flights(prof, [target(prof)]))


def _edit_done(prof: Profile) -> bool:
    return any(c.procedure == prof.procedure for c in memory.all_corrections())


def _edit(prof: Profile) -> None:
    doc = store.latest(prof.procedure)
    flaps10 = next(s for s in doc.steps if s.key == "toggle:flaps_idx:1")
    apply_edit(prof, flaps10.id, "action", f"Select {flaps10.label}, then check the flap position indicator.")
    ui.refresh_guidelines(prof, f"Instructor reworded {flaps10.label}; same wording applied to similar steps")


ITEMS = [
    Item("First instructor flight",
         "The guidelines are created from this one flight. This instructor selects flaps 20 at 89 kt — over the "
         "85 kt limit — so that guideline is capped at the limit and the flight is flagged.",
         "Add flight 1", *_flight(0)),
    Item("Second flight", "A second instructor flies the same circuit, within the limit. The flaps-20 guideline now "
         "learns from this flight instead of the one that broke the limit.", "Add flight 2", *_flight(1)),
    Item("Third flight", "Only some instructors add a little power on short final — that becomes a technique note, "
         "not a core step.", "Add flight 3", *_flight(2)),
    Item("Fourth flight", "More flights: the typical ranges get tighter and the confidence rises.",
         "Add flight 4", *_flight(3)),
    Item("Fifth flight", "Five flights: the typical technique is well established.", "Add flight 5", *_flight(4)),
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
        lines = [f"Version {doc.version}: **{len(doc.steps)} guidelines created**, checked against the rules."]
    else:
        lines = [f"Version {doc.version}: **{changed} guideline{'s' if changed != 1 else ''} changed**"
                 if changed else f"Version {doc.version}: nothing needed to change."]
    lines.append(f"Confidence {m['mean_confidence']:.0%} · {int(m['technique_steps'])} technique note(s)")
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


def _rules(prof: Profile, docs) -> None:
    limits = [lim for d in docs for lim in d.limits]
    reqs = [r for d in docs for r in d.requirements]
    st.markdown(":green[:material/check_circle:] **Rules loaded** from "
                + ", ".join(d.title for d in docs if d.limits or d.requirements))
    st.markdown("\n".join(
        [f"- :red-badge[{ui.limit_name(lim, prof)}] {ui.when_text(lim, prof)}" for lim in limits]
        + [f"- :green-badge[Must do] {r.text}" for r in reqs]))


def _watch(prof: Profile) -> None:
    """One guideline, version by version: the quickest way to see it learn."""
    key, fact_name, title = WATCH
    rows = []
    for v in store.versions(prof.procedure):
        doc = store.load(prof.procedure, v)
        s = next((x for x in doc.steps if x.key == key), None)
        f = s.facts.get(fact_name) if s else None
        if f is None or (rows and rows[-1]["Flights"] == len(doc.runs)):
            continue  # only the versions that added flights
        note = "Capped at the 85 kt limit" if f.clipped else ("Flight that broke the limit left out"
                                                             if len(s.runs) < len(doc.runs) else "")
        rows.append({"Version": v, "Flights": len(doc.runs), "Typically at": f.fmt_range(),
                     "Confidence": f"{s.confidence:.0%}", "Note": note})
    if not rows:
        return
    with st.container(border=True):
        st.markdown(f":material/visibility: **Watch one guideline: {title}**")
        st.caption("How the guideline for selecting flaps 20 changes as each flight is added.")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def render() -> None:
    st.title(":material/play_circle: Guided demo")
    prof = load_profile(PROFILE)
    if st.session_state.get("profile_name") != PROFILE:
        st.session_state["profile_name"] = PROFILE
    st.write("The official rules are loaded first. Then add the instructor flights one at a time: after each "
             "one the guidelines learn from it, and a review window shows what changed.")
    log: dict = st.session_state.setdefault("demo_log", {})
    docs = references.list_references(prof.procedure)
    started = {d.id for d in docs} >= {"c172_poh", "aerodrome_sop", "faa_afh"}

    with st.container(border=True):
        if started:
            _rules(prof, docs)
        else:
            st.markdown("**Start** — loads the rules from the Cessna 172 manual (POH) and the aerodrome SOP "
                        "before any flight is added.")
            if store.versions(prof.procedure) or storage.list_runs(prof.procedure):
                st.caption(":material/warning: Starting the demo clears the existing data for this procedure.")
        if st.button("Restart the demo" if started else "Start the demo", icon=":material/restart_alt:",
                     type="secondary" if started else "primary"):
            _reset(prof)
            st.rerun()
    if not started:
        return
    ui.pending_banner(prof, key="demo-review")

    left, right = st.columns([3, 2])
    with left:
        next_i = next((i for i, it in enumerate(ITEMS) if not it.done(prof)), None)
        for i, it in enumerate(ITEMS):
            done = it.done(prof)
            with st.container(border=True):
                mark = (":green[:material/check_circle:]" if done else ":blue[:material/arrow_circle_right:]"
                        if i == next_i else ":gray[:material/radio_button_unchecked:]")
                st.markdown(f"{mark} **{i + 1}. {it.title}**")
                if done or i == next_i:
                    st.caption(it.story)
                if done:
                    for line in log.get(i, []):
                        st.markdown(line)
                elif i == next_i:
                    if st.button(it.button, type="primary", icon=":material/play_arrow:", key=f"demo-{i}"):
                        before = store.latest(prof.procedure)
                        it.run(prof)
                        log[i] = _summary(prof, before.version if before else None)
                        st.rerun()
                    rest = [j for j in range(i, len(ITEMS)) if ITEMS[j].button.startswith("Add flight")]
                    if len(rest) > 1 and st.button(f"Add all {len(rest)} remaining flights at once",
                                                   icon=":material/fast_forward:", key=f"demo-all-{i}"):
                        before = store.latest(prof.procedure)
                        ui.add_flights(prof, ui.remaining_samples(prof))
                        for j in rest:
                            log[j] = ["Added together with the other remaining flights."]
                        log[rest[-1]] = _summary(prof, before.version if before else None)
                        st.rerun()
    with right:
        _watch(prof)
        if store.latest(prof.procedure):
            ui.link_button("guidelines", "See the guidelines", icon=":material/checklist:", primary=False,
                           key="demo-guidelines")
    if next_i is None:
        st.success("Demo complete. The Progress page shows how the guidelines improved, version by version.",
                   icon=":material/celebration:")
        ui.link_button("progress", "Open Progress", key="demo-progress")
