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
  <span class="tag">For flight schools, driving schools and maritime academies</span>
  <h1>Turn every simulator session into guidelines <em>students can trust.</em></h1>
  <p>Sim-to-SOP learns how your instructors actually do it from recorded simulator {ui.noun(2)}, writes it up as
     clear step-by-step guidelines, and checks every number against the official limits.</p>
  <div class="proof"><span><i></i>Every number traced to recorded data</span>
    <span><i></i>Hard limits quoted from your manuals</span><span><i></i>You approve every change</span></div>
</div>""")


def _cards(items: list[tuple[str, str]], numbered: bool) -> None:
    """Cards in rows of three. The numbered steps carry the accent; the feature cards stay plain."""
    for row in range(0, len(items), 3):
        cols = st.columns(3)
        for i, (col, (title, text)) in enumerate(zip(cols, items[row:row + 3]), start=row + 1):
            num = f'<div class="num">{i}</div>' if numbered else ""
            plain = "" if numbered else " plain"
            col.html(f'<div class="feature{plain}">{num}<h4>{title}</h4><p>{text}</p></div>')


def render() -> None:
    prof = ui.profile()
    runs = storage.list_runs(prof.procedure)
    docs = references.list_references(prof.procedure)
    doc = store.latest(prof.procedure)

    _hero()
    ui.pending_banner(prof, key="home-review")

    msg, page, label = _next_step(runs, docs)
    with st.container(border=True):
        text, action = st.columns([1, 1], vertical_alignment="center")
        text.markdown(f"**Next step** · {ui.PROFILE_NAMES.get(st.session_state['profile_name'], prof.title)}  \n{msg}")
        with action:
            if runs or prof.domain not in ui.GENERATED:
                ui.link_button(page, label, key="home-next")
            else:
                a, b = st.columns(2)
                if a.button("Try with sample data", type="primary", icon=":material/play_arrow:", width="stretch",
                            key="tone-attn-sample",
                            help=f"Five generated instructor {ui.noun(2)}, checked against the built-in "
                                 "official rules. The guidelines are created straight away."):
                    ui.load_sample(prof)
                    st.rerun()
                with b:
                    ui.link_button(page, f"Upload {ui.noun(2)}", icon=":material/upload:", primary=False,
                                   key="home-next")
        if not runs and prof.domain in ui.GENERATED:
            st.caption("New here? The sample data runs the whole workflow in under a minute. *Start over* in the "
                       "sidebar clears it again.")

    st.subheader("Where you are")
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
            st.metric(label, value, border=True, help=tip)
    links = st.columns(4)
    for col, (key, text) in zip(links, [("runs", f"Instructor {ui.noun(2)}"), ("guidelines", "Guidelines"),
                                         ("references", "Official documents"), ("progress", "Progress")]):
        with col:
            ui.link_button(key, text, primary=False, key=f"home-{key}")

    st.subheader("How it works")
    _cards(HOW, numbered=True)
    st.subheader("Why instructors use it")
    _cards(FEATURES, numbered=False)

    st.html('<div class="domains"><span>Air · Cessna 172 circuit</span><span>Land · Parallel parking</span>'
            '<span>Sea · Harbour exit</span></div>')
    st.caption(f"Pick a procedure in the sidebar. Want a particular kind of {ui.noun()}? Make one in the "
               f"**{ui.noun(cap=True)} generator** and add it straight to the guidelines.")
