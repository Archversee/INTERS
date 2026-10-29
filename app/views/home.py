"""Home: what the app does, where you are, what to do next."""

from __future__ import annotations

import streamlit as st
import ui

from sim2sop import guidelines_store as store
from sim2sop import references, storage

HOW = [
    ("Record", "Instructors fly the procedure in the simulator and say what they are doing. Upload the "
               "recording, or load the sample data."),
    ("Learn", "The app finds every action, lines the recordings up, and works out what instructors "
              "typically do and when."),
    ("Approve", "Guidelines are written and checked against the official limits. You see exactly what "
                "changed and publish with one click."),
]
FEATURES = [
    ("Every number is traced", "Each value in a guideline is checked against the recorded data. Nothing is "
                               "made up, even when an AI writes the wording."),
    ("Official limits come first", "Hard limits are quoted word for word from the manuals and SOPs. A value "
                                   "past a limit is capped and flagged, never taught."),
    ("Gets better with every session", "Typical ranges tighten as recordings are added, and actions only "
                                       "some instructors take become technique notes."),
    ("Learns your house style", "Edit a guideline once and similar steps are written the same way next "
                                "time."),
    ("A human signs off", "Each new version opens a review: approve or reject each change, then publish. "
                          "A rejected change is not proposed again."),
    ("Ready for the classroom", "Read it as a checklist or in full detail, and download a printable "
                                "version for students."),
]


def _next_step(runs, docs) -> tuple[str, str, str]:
    if not runs:
        return (f"Upload the first instructor {ui.noun()}. The guidelines are created from it automatically.",
                "runs", f"Upload {ui.noun(2)}")
    if not docs:
        return ("Add the official documents (manuals, SOPs) so the guidelines are checked against hard limits.",
                "references", "Add documents")
    return f"Add more {ui.noun(2)} to refine the guidelines, or read them.", "guidelines", "Open the guidelines"


def _hero() -> None:
    st.html(f"""
<div class="hero">
  <div class="tag">For flight schools, driving schools and maritime academies</div>
  <h1>Turn every simulator session into guidelines <em>students can trust.</em></h1>
  <p>Sim-to-SOP learns how your instructors actually do it from recorded simulator {ui.noun(2)}, writes it up as
     clear step-by-step guidelines, and checks every number against the official limits.</p>
  <div class="proof"><span>Every number traced to recorded data</span>
    <span>Hard limits quoted from your manuals</span><span>You approve every change</span></div>
</div>""")


def _grid(items: list[tuple[str, str]], cls: str, numbered: bool = False) -> None:
    """Items on hairlines in a grid (no cards): the steps in three columns, the features in two."""
    cells = "".join(f"<div>{f'<b>{i:02d}</b>' if numbered else ''}<h4>{t}</h4><p>{x}</p></div>"
                    for i, (t, x) in enumerate(items, start=1))
    st.html(f'<div class="{cls}">{cells}</div>')


def render() -> None:
    prof = ui.profile()
    runs = storage.list_runs(prof.procedure)
    docs = references.list_references(prof.procedure)
    doc = store.latest(prof.procedure)

    _hero()
    ui.pending_banner(prof, key="home-review")

    msg, page, label = _next_step(runs, docs)
    text, action = st.columns([3, 2], vertical_alignment="center")
    text.markdown(f"**Next step** · {ui.PROFILE_NAMES.get(st.session_state['profile_name'], prof.title)}  \n{msg}")
    with action:
        if runs or prof.domain not in ui.GENERATED:
            ui.link_button(page, label, key="home-next")
        else:
            a, b = st.columns(2)
            if a.button("Try with sample data", type="primary", width="stretch", key="tone-attn-sample",
                        help=f"Five generated instructor {ui.noun(2)}, checked against the built-in official "
                             "rules. The guidelines are created straight away. *Start over* in the sidebar "
                             "clears it again."):
                ui.load_sample(prof)
                st.rerun()
            with b:
                ui.link_button(page, f"Upload {ui.noun(2)}", primary=False, key="home-next")

    stats = [
        (f"Instructor {ui.noun(2)}", len(runs), None),
        ("Guideline steps", len(doc.steps) if doc else 0,
         f"Version {doc.version}, {doc.status}" if doc else "Created from the first recording"),
        ("Official hard limits", sum(len(d.limits) for d in docs),
         f"From {len(docs)} official document{'s' if len(docs) != 1 else ''}"),
        ("Confidence", f"{doc.metrics['mean_confidence']:.0%}" if doc else "—",
         "How consistently instructors do each step. Rises as recordings are added."),
    ]
    for i, (col, (label, value, tip)) in enumerate(zip(st.columns(4), stats), start=1):
        with col.container(key=f"stat-{i}"):
            st.metric(label, value, help=tip)

    st.subheader("How it works")
    _grid(HOW, "steps", numbered=True)
    st.subheader("Why instructors use it")
    _grid(FEATURES, "facts")
    st.caption("Procedures: Air · Cessna 172 circuit, Land · Parallel parking, Sea · Harbour exit. Pick one in the "
               f"sidebar. Want a particular kind of {ui.noun()}? Make one in the **{ui.noun(cap=True)} generator** "
               "and add it straight to the guidelines.")
