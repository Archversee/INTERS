"""Guided demo: the pitch story, one click per step. Every step does the real work."""

from __future__ import annotations

import shutil

import streamlit as st
import ui

from sim2sop import config, references, storage
from sim2sop import guidelines_store as store
from sim2sop.pipeline import apply_edit, approve, create_guidelines, update_guidelines
from sim2sop.profiles import load_profile

PROFILE = "air_c172_circuit"
BASE_DOCS = ("c172_poh_notes.md", "aerodrome_sop_notes.md", "faa_afh_traffic_pattern_notes.md")


def _publish(prof) -> None:
    doc = store.latest(prof.procedure)
    if doc and doc.pending:
        approve(prof)  # in the demo the instructor approves every change


def _ref(prof, name: str) -> None:
    references.add_reference(prof.procedure, config.REFERENCES_DIR / prof.references / name)


def step1(prof):
    ui.add_sample_flights(prof, 1)
    doc = ui.run_build("Creating guidelines from one flight", lambda p: create_guidelines(prof, ui.llm(), p))
    return [f"Version {doc.version}: **{len(doc.steps)} guidelines** from 1 flight.",
            ":red-badge[Unchecked] — no official documents yet, so no hard limits are shown."]


def step2(prof):
    _publish(prof)
    for name in BASE_DOCS:
        _ref(prof, name)
    doc = ui.run_build("Checking against the official documents", lambda p: update_guidelines(prof, ui.llm(), p))
    capped = [f for f in doc.flags if f.kind == "clipped"]
    lines = [f"Version {doc.version}: :blue-badge[Checked] against {len(doc.references)} documents; "
             f"{doc.metrics['within_limits']:.0%} of limited values are inside the limits."]
    if capped:
        lines.append("The instructor selected **flaps 20 at 89 kt**, above the 85 kt limit in the POH. "
                     "The guideline is capped at 85 kt and the flight is flagged.")
    return lines


def step3(prof):
    _publish(prof)
    before = store.latest(prof.procedure)
    ui.add_sample_flights(prof)
    doc = ui.run_build("Learning from five flights", lambda p: update_guidelines(prof, ui.llm(), p))
    technique = [s for s in doc.steps if not s.mandatory]
    return [f"Version {doc.version}: typical ranges are **{before.metrics['range_width_ratio'] / doc.metrics['range_width_ratio']:.1f}× tighter** "
            f"and confidence rose from {before.metrics['mean_confidence']:.0%} to {doc.metrics['mean_confidence']:.0%}.",
            f"{len(technique)} action(s) only some instructors do became technique notes "
            f"({', '.join(s.label for s in technique)}).",
            "The flight that broke the flaps limit is left out of the flaps-20 typical range."]


def step4(prof):
    _publish(prof)
    _ref(prof, "notam_crane.md")
    ui.add_sample_flights(prof, obstacle=True)
    doc = ui.run_build("Updating for the new obstacle", lambda p: update_guidelines(prof, ui.llm(), p))
    changed = [s for s in doc.steps if s.change]
    phases = sorted({ui.phase_title(s.phase) for s in changed})
    return [f"Version {doc.version}: technique change detected. **Only {len(changed)} guidelines changed**, "
            f"all in {' and '.join(phases)} — every other step is word-for-word the same.",
            "Each change lists its reasons and cites the NOTAM. They are waiting for approval on the Review page."]


def step5(prof):
    _publish(prof)
    doc = store.latest(prof.procedure)
    flaps10 = next(s for s in doc.steps if s.key == "toggle:flaps_idx:1")
    text = f"Select {flaps10.label}, then check the flap position indicator."
    apply_edit(prof, flaps10.id, "action", text)
    new = ui.run_build("Regenerating with the instructor's correction", lambda p: update_guidelines(prof, ui.llm(), p))
    adopted = [s.action for s in new.steps if s.change]
    _publish(prof)
    return [f"The instructor changed **{flaps10.label}** to: *{text}*",
            "On the next update the same wording was applied to:"] + [f"- {a}" for a in adopted]


STEPS = [
    ("Create guidelines from one flight",
     "An instructor flies the circuit once, saying what they do. No official documents yet.", step1, "guidelines"),
    ("Add the official documents",
     "Add the Cessna 172 manual (POH), the aerodrome SOP and a flying handbook. They set the hard limits.",
     step2, "guidelines"),
    ("Add four more flights",
     "More instructors fly the same circuit. The system learns what they typically do.", step3, "progress"),
    ("A new obstacle appears",
     "A crane goes up under the downwind leg. A NOTAM is issued and three instructors fly a wider circuit.",
     step4, "review"),
    ("An instructor improves the wording",
     "The instructor rewrites one step. The system remembers and writes similar steps the same way.",
     step5, "guidelines"),
]


def _reset(prof) -> None:
    storage.delete_runs(prof.procedure)
    shutil.rmtree(config.GUIDELINES_DIR / prof.procedure, ignore_errors=True)
    references.clear_references(prof.procedure)
    shutil.rmtree(config.MEMORY_DIR, ignore_errors=True)
    ui.telemetry.clear()
    st.session_state["demo_results"] = {}


def render() -> None:
    st.title(":material/play_circle: Guided demo")
    st.write("The whole story in five clicks, using the Cessna 172 traffic circuit. Each step does the real "
             "work — open the other pages at any time to look around.")
    prof = load_profile(PROFILE)
    if st.session_state.get("profile_name") != PROFILE:
        st.info("The demo uses the ✈️ Air procedure; it has been selected for you.", icon=":material/info:")
        st.session_state["profile_name"] = PROFILE
    results: dict = st.session_state.setdefault("demo_results", {})

    started = bool(results) or not store.versions(prof.procedure)
    with st.container(border=True):
        if store.versions(prof.procedure) and not results:
            st.markdown(":material/warning: There is existing data for this procedure. Starting the demo clears it.")
        if st.button("Start the demo from scratch" if not results else "Restart the demo",
                     icon=":material/restart_alt:", type="secondary" if results else "primary"):
            _reset(prof)
            st.rerun()
    if not started:
        return

    for i, (title, story, fn, page) in enumerate(STEPS):
        done = i in results
        ready = all(j in results for j in range(i))
        with st.container(border=True):
            mark = ":green[:material/check_circle:]" if done else ":gray[:material/radio_button_unchecked:]"
            st.markdown(f"{mark} **Step {i + 1} · {title}**")
            st.write(story)
            if done:
                for line in results[i]:
                    st.markdown(line)
                ui.link_button(page, "See it", primary=False, key=f"demo-see-{i}")
            elif ready and st.button(f"Run step {i + 1}", type="primary", icon=":material/play_arrow:",
                                     key=f"demo-run-{i}"):
                results[i] = fn(prof)
                st.rerun()
    if len(results) == len(STEPS):
        st.success("Demo complete. The Progress page shows the improvement version by version.",
                   icon=":material/celebration:")
        ui.link_button("progress", "Open Progress", key="demo-progress")
