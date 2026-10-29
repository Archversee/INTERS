"""Shared state, wording and widgets for the Streamlit app."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sim2sop import config, references, storage
from sim2sop import guidelines_store as store
from sim2sop.agent.llm import get_llm
from sim2sop.learn.changes import step_slots
from sim2sop.pipeline import approve, create_guidelines, update_guidelines
from sim2sop.profiles import Profile, list_profiles, load_profile
from sim2sop.schema import Guidelines, Step

PAGES: dict = {}   # filled by streamlit_app.py so pages can link to each other

ASSETS = Path(__file__).parent / "assets"
PROFILE_NAMES = {
    "air_c172_circuit": "Air · Cessna 172 circuit",
    "land_parallel_park": "Land · Parallel parking",
    "sea_harbour_exit": "Sea · Harbour exit",
}
# What one recorded run is called in each domain, and the domains the generator can make.
NOUNS = {"air": "flight", "land": "drive", "sea": "run"}
GENERATED = {"air", "land"}
PROVIDERS = {"ollama": "Ollama (local / LAN)", "openai": "OpenAI-compatible API",
             "anthropic": "Anthropic API", "none": "No AI (built-in template)"}
DOC_KINDS = {"poh": "Aircraft manual (POH)", "sop": "Standard operating procedure",
             "regulation": "Regulation", "handbook": "Handbook (guidance only)", "other": "Document"}
FLAG_LABELS = {
    "limit_violation": ("Instructor runs that broke a hard limit", ":material/warning:"),
    "excluded": ("Values left out of a typical range (they broke a limit)", ":material/remove_circle:"),
    "clipped": ("Values capped at a hard limit", ":material/vertical_align_top:"),
    "technique_change": ("Technique changes detected", ":material/swap_horiz:"),
    "required_missing": ("Required actions nobody flew", ":material/report:"),
    "unmatched_annotation": ("Spoken callouts with no matching action", ":material/mic_off:"),
    "verifier": ("Numbers the checker could not confirm", ":material/error:"),
    "unchecked": ("Not checked against official limits", ":material/gpp_maybe:"),
    "data": ("Data issues", ":material/database:"),
}
FIELDS = [("action", "What to do"), ("trigger", "When"), ("intent", "Why"), ("typical", "Typical"),
          ("limit", "Hard limit")]
# Change reasons that matter to a reviewer (anything else is a routine range refinement).
IMPORTANT = ("Technique change", "See ", "Hard limit", "Value clipped", "Now ", "New action", "Instructor",
             "Re-checked", "New guideline")


# --- session state ------------------------------------------------------------------------

def profile() -> Profile:
    names = list_profiles()
    name = st.session_state.get("profile_name") or ("air_c172_circuit" if "air_c172_circuit" in names else names[0])
    return load_profile(name)


def llm():
    provider = st.session_state.get("llm_provider", config.LLM_PROVIDER)
    model = st.session_state.get("llm_model") or None
    return get_llm(provider, model)


@st.cache_data(ttl=30, show_spinner=False)
def llm_status(provider: str, model: str | None) -> tuple[bool, list[str]]:
    client = get_llm(provider, model)
    if provider == "ollama":
        try:
            return True, client.models()
        except Exception:
            return False, []
    return client.available(), []


@st.cache_data(show_spinner=False)
def telemetry(procedure: str, run_id: str) -> pd.DataFrame:
    return storage.load_telemetry(procedure, run_id)


# --- wording --------------------------------------------------------------------------------

def noun(n: int = 1, cap: bool = False) -> str:
    """What a recorded run is called for the selected procedure: "flight", "drives", ..."""
    name = st.session_state.get("profile_name") or "air"
    word = NOUNS.get(name.split("_", 1)[0], "run") + ("" if n == 1 else "s")
    return word.capitalize() if cap else word


def phase_title(name: str) -> str:
    return name.replace("_", " ").title()


def run_title(run_id: str) -> str:
    n = run_id.rsplit("_", 1)[-1]
    if run_id.startswith("generated_"):
        return f"Generated {noun()} (seed {n})"
    return f"{noun(cap=True)} {int(n)}" if n.isdigit() else run_id


def instructor_name(raw: str | None) -> str:
    """"instructor_3" -> "Instructor 3"; no name (or the generator's placeholder) -> "—"."""
    return "—" if raw in (None, "", "generated") else raw.replace("_", " ").title()


def mode_title(doc: Guidelines) -> str:
    return {"create": "Created", "update": "Updated", "edit": "Instructor edit"}[doc.mode]


def doc_badges(doc: Guidelines) -> str:
    parts = [":green-badge[:material/check_circle: Published]" if doc.status == "published"
             else ":orange-badge[:material/pending: Draft · needs review]"]
    parts.append(":blue-badge[Checked against official limits]" if doc.checked
                 else ":red-badge[Unchecked · no official documents]")
    parts.append(":gray-badge[Advisory]")
    return " ".join(parts)


# Tints are mixed from the text colour, so the same CSS works in the light and the dark theme.
# Palette ("jet age"): space navy, turquoise, coral, amber, violet and green. Each kind of action
# has its own colour: publish / save = green, review = amber, AI = violet, delete = coral; the
# turquoise primary is for everything else. Buttons pick theirs by key prefix ("tone-go-…").
CSS = """
<style>
:root { --navy: #14213D; --teal: #14A3A3; --coral: #F2545B; --amber: #F4B740; --violet: #7B61FF;
        --green: #2E9E6B; --sky: #3B82C4; }
[data-testid="stDecoration"] { display: none; }
.block-container { padding-top: 4.2rem; max-width: 1240px; }
h1, h2, h3 { letter-spacing: -0.02em; }
.eyebrow { text-transform: uppercase; letter-spacing: .12em; font-size: .74rem; font-weight: 700;
           color: var(--coral); margin-bottom: -.9rem; }
.lede { font-size: 1.06rem; opacity: .78; max-width: 52rem; margin: .2rem 0 1rem; }

/* Buttons by role */
[class*="st-key-tone-"] button[data-testid="stBaseButton-primary"] { border: none; transition: filter .15s; }
[class*="st-key-tone-"] button[data-testid="stBaseButton-primary"]:hover { filter: brightness(1.08); }
[class*="st-key-tone-go"] button[data-testid="stBaseButton-primary"] { background: var(--green); color: #fff; }
[class*="st-key-tone-attn"] button[data-testid="stBaseButton-primary"] { background: var(--amber); color: var(--navy); }
[class*="st-key-tone-ai"] button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, var(--violet), #A25CF0); color: #fff; }
[class*="st-key-tone-danger"] button[data-testid="stBaseButton-primary"] { background: var(--coral); color: #fff; }
[class*="st-key-tone-"] button[data-testid="stBaseButton-primary"]:disabled { filter: grayscale(1); opacity: .45; }

/* Approve / Reject in the review */
[class*="st-key-dec-"] button[role="radio"][aria-checked="true"]:first-of-type {
    background: color-mix(in srgb, var(--green) 16%, transparent); color: var(--green); border-color: var(--green); }
[class*="st-key-dec-"] button[role="radio"][aria-checked="true"]:last-of-type {
    background: color-mix(in srgb, var(--coral) 16%, transparent); color: var(--coral); border-color: var(--coral); }

/* Home */
.hero { position: relative; overflow: hidden; border-radius: 1.2rem; padding: 2.7rem 2.7rem 2.3rem;
        margin-bottom: .6rem; color: #fff;
        background: radial-gradient(700px 320px at 92% -10%, rgba(43,196,196,.45), transparent 65%),
                    radial-gradient(520px 260px at -5% 115%, rgba(242,84,91,.38), transparent 60%),
                    linear-gradient(160deg, #101B33 0%, #14213D 60%, #1B2C52 100%); }
.hero::after { content: ""; position: absolute; inset: 0; pointer-events: none; opacity: .09;
               background-image: linear-gradient(rgba(255,255,255,.6) 1px, transparent 1px),
                                 linear-gradient(90deg, rgba(255,255,255,.6) 1px, transparent 1px);
               background-size: 44px 44px; mask-image: linear-gradient(90deg, transparent 30%, #000); }
.hero .tag { display: inline-block; padding: .25rem .75rem; border-radius: 999px; font-size: .78rem;
             font-weight: 700; color: var(--navy); background: var(--amber); }
.hero h1 { color: #fff; font-size: 2.65rem; line-height: 1.12; margin: .9rem 0 .7rem; padding: 0; max-width: 46rem; }
.hero h1 em { font-style: normal; background: linear-gradient(90deg, var(--amber), var(--coral));
              -webkit-background-clip: text; background-clip: text; color: transparent; }
.hero p { color: rgba(255,255,255,.82); font-size: 1.1rem; max-width: 44rem; margin: 0; }
.hero .proof { display: flex; flex-wrap: wrap; gap: .6rem; margin-top: 1.6rem; font-size: .86rem; }
.hero .proof span { padding: .35rem .8rem; border-radius: 999px; color: rgba(255,255,255,.88);
                    background: rgba(255,255,255,.07); border: 1px solid rgba(255,255,255,.14); }
.hero .proof i { display: inline-block; width: .5rem; height: .5rem; border-radius: 50%; margin-right: .45rem; }
.feature { height: 100%; padding: 1.15rem 1.25rem; border-radius: .9rem; border-top: 3px solid var(--accent);
           border-left: 1px solid color-mix(in srgb, currentColor 10%, transparent);
           border-right: 1px solid color-mix(in srgb, currentColor 10%, transparent);
           border-bottom: 1px solid color-mix(in srgb, currentColor 10%, transparent);
           background: linear-gradient(180deg, color-mix(in srgb, var(--accent) 9%, transparent), transparent 70%); }
.feature .num { display: inline-flex; width: 2rem; height: 2rem; border-radius: .6rem; align-items: center;
                justify-content: center; font-weight: 800; font-size: .95rem; color: #fff;
                background: var(--accent); margin-bottom: .6rem; }
.feature h4 { margin: 0 0 .35rem; padding: 0; font-size: 1.02rem; }
.feature p { margin: 0; opacity: .75; font-size: .93rem; line-height: 1.5; }
.a1 { --accent: var(--teal); } .a2 { --accent: var(--coral); } .a3 { --accent: var(--amber); }
.a4 { --accent: var(--violet); } .a5 { --accent: var(--green); } .a6 { --accent: var(--sky); }
[class*="st-key-accent-"] [data-testid="stMetric"] { box-shadow: inset 0 3px 0 var(--accent); }
.st-key-accent-1 { --accent: var(--teal); } .st-key-accent-2 { --accent: var(--violet); }
.st-key-accent-3 { --accent: var(--coral); } .st-key-accent-4 { --accent: var(--amber); }
.domains { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: .3rem; }
.domains span { padding: .3rem .85rem; border-radius: 999px; font-size: .85rem; font-weight: 600;
                color: var(--accent); border: 1px solid color-mix(in srgb, var(--accent) 45%, transparent);
                background: color-mix(in srgb, var(--accent) 8%, transparent); }
.sidebar-foot { font-size: .78rem; opacity: .6; line-height: 1.5; }
@media (max-width: 640px) {
  .hero { padding: 1.6rem 1.3rem; } .hero h1 { font-size: 1.8rem; }
}
</style>
"""


def style() -> None:
    """Brand, fonts and spacing shared by every page (call once per run)."""
    st.logo(str(ASSETS / "logo.svg"), size="large", icon_image=str(ASSETS / "mark.svg"))
    st.html(CSS)


def page_header(title: str, lede: str | None = None, eyebrow: str | None = None) -> None:
    """The same header on every page: a small step label, the title and one line of what it is for."""
    if eyebrow:
        st.markdown(f'<div class="eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.title(title)
    if lede:
        lede = re.sub(r"\*(.+?)\*", r"<i>\1</i>", re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", lede))
        st.markdown(f'<p class="lede">{lede}</p>', unsafe_allow_html=True)


def link_button(page_key: str, label: str, icon: str = ":material/arrow_forward:", primary: bool = True,
                key: str | None = None) -> None:
    """Button that opens another page of the app."""
    if st.button(label, icon=icon, type="primary" if primary else "secondary", key=key or f"go-{page_key}-{label}"):
        st.switch_page(PAGES[page_key])


# --- data ---------------------------------------------------------------------------------------

def generated_dir(prof: Profile) -> Path:
    return config.RAW_DIR / "generated" / prof.domain


def sample_dir(prof: Profile) -> Path:
    return config.RAW_DIR / "sample" / prof.domain


def rules_folder(prof: Profile) -> Path:
    """Where the procedure's built-in official documents live."""
    return config.REFERENCES_DIR / prof.references


def ensure_rules(prof: Profile) -> None:
    """The built-in official documents are always in force: add any that are missing, and add a
    document again when its file was edited (call once per run). Existing guidelines are
    re-checked against the new rules."""
    changed = references.sync_builtin(prof.procedure, rules_folder(prof))
    if changed and store.latest(prof.procedure) is not None:
        refresh_guidelines(prof, "Official rules updated: " + ", ".join(d.label for d in changed))


def start_over(prof: Profile) -> None:
    """Delete everything for the procedure, behind a confirmation (sidebar popover)."""
    st.caption(f"Deletes every {noun()}, guideline version, uploaded document, instructor correction and "
               f"generated {noun()} for this procedure. The built-in official rules stay.")
    sure = st.checkbox("Yes, delete all the data", key="reset-sure")
    if st.button("Delete all data", type="primary", icon=":material/delete_forever:", disabled=not sure,
                 key="tone-danger-reset", width="stretch"):
        reset_all(prof)
        st.session_state["toast"] = "All data deleted. The built-in official rules stay in force."
        st.rerun()


def load_sample(prof: Profile) -> None:
    """First-run shortcut: five generated instructor runs, then the guidelines are created
    (against the built-in official rules) and the review pops up."""
    from sim2sop.generate import generate
    from sim2sop.ingest import ingest_run

    references.sync_builtin(prof.procedure, rules_folder(prof))
    with st.spinner(f"Generating five instructor {noun(2)}…"):
        for run_dir in generate(prof.domain, n=5, out_dir=sample_dir(prof)):
            ingest_run(run_dir, prof)
    refresh_guidelines(prof, f"Sample data loaded: 5 {noun(2)}")


def reset_all(prof: Profile) -> None:
    """Delete every run, guideline version, uploaded document, correction and generated run
    for the procedure. The built-in official rules are put straight back."""
    storage.delete_runs(prof.procedure)
    shutil.rmtree(config.GUIDELINES_DIR / prof.procedure, ignore_errors=True)
    references.clear_references(prof.procedure)
    shutil.rmtree(config.MEMORY_DIR, ignore_errors=True)
    shutil.rmtree(generated_dir(prof), ignore_errors=True)
    shutil.rmtree(sample_dir(prof), ignore_errors=True)
    shutil.rmtree(config.DATA_DIR / "uploads" / prof.procedure, ignore_errors=True)
    telemetry.clear()
    for k in ("review_queue", "gen_current", "manual"):
        st.session_state.pop(k, None)
    references.sync_builtin(prof.procedure, rules_folder(prof))


# --- building and reviewing ----------------------------------------------------------------------

def run_build(label: str, fn):
    """Run fn(progress) inside a status box with a progress bar."""
    with st.status(label, expanded=True) as status:
        provider = st.session_state.get("llm_provider", config.LLM_PROVIDER)
        if provider != "none":
            st.caption(f"The AI writer ({st.session_state.get('llm_model') or provider}) is "
                       "wording each guideline, which can take a while per step. If it stops answering, the "
                       "built-in wording is used for the rest. For instant results, switch the AI writer to "
                       "*No AI* in the sidebar.")
        bar = st.progress(0.0, text=f"Analysing the {noun(2)} and checking the official limits…")

        def progress(i: int, n: int, msg: str) -> None:
            bar.progress(min(1.0, (i + 1) / max(n, 1)), text=f"{msg}  ({i + 1} of {n})")

        result = fn(progress)
        status.update(label=f"{label} — done", state="complete", expanded=False)
    return result


def refresh_guidelines(prof: Profile, title: str) -> Guidelines | None:
    """Create or update the guidelines after new data, then queue the review popup."""
    if not storage.list_runs(prof.procedure):
        return None
    exists = store.latest(prof.procedure) is not None
    doc = run_build("Updating the guidelines" if exists else "Creating the guidelines",
                    (lambda p: update_guidelines(prof, llm(), p)) if exists
                    else (lambda p: create_guidelines(prof, llm(), p)))
    if cut := next((f.message for f in doc.flags if f.message.startswith("The AI writer")), None):
        st.session_state["warn"] = cut
    if doc.pending:
        st.session_state["review_queue"] = {"version": doc.version, "title": title}
    else:
        st.session_state["toast"] = f"{title}: the guidelines did not need to change."
    return doc


def show_queued() -> None:
    """Toasts and the review popup queued by the previous action (call once per run)."""
    if msg := st.session_state.pop("toast", None):
        st.toast(msg, icon=":material/info:")
    if msg := st.session_state.pop("warn", None):
        st.toast(msg, icon=":material/warning:", duration="long")
    if q := st.session_state.get("review_queue"):
        review_dialog(q["version"], q["title"])


def pending_banner(prof: Profile, key: str) -> None:
    doc = store.latest(prof.procedure)
    if doc and doc.pending and not st.session_state.get("review_queue"):
        with st.container(border=True):
            st.markdown(f"**{len(doc.pending)} change"
                        f"{'s' if len(doc.pending) != 1 else ''} in version {doc.version} "
                        "waiting for your approval.**")
            if st.button("Review now", type="primary", icon=":material/fact_check:", key=f"tone-attn-{key}"):
                st.session_state["review_queue"] = {"version": doc.version, "title": f"Version {doc.version}"}
                st.rerun()


def _dismiss_review() -> None:
    st.session_state.pop("review_queue", None)


def _is_important(s: Step) -> bool:
    return s.change in ("added", "moved") or any(r.startswith(IMPORTANT) for r in s.change_reasons)


def _decision(s: Step, version: int) -> str:
    return st.segmented_control("Decision", ["Approve", "Reject"], default="Approve",
                                key=f"dec-{version}-{s.id}", label_visibility="collapsed") or "Approve"


def _before_after(old: Step, new: Step) -> None:
    changed = [(f, label) for f, label in FIELDS if (getattr(old, f) or "") != (getattr(new, f) or "")]
    if not changed:
        return
    a, b = st.columns(2)
    a.caption("Before")
    b.caption("After")
    for f, label in changed:
        a.markdown(f"**{label}** — {getattr(old, f) or '—'}")
        b.markdown(f"**{label}** — {getattr(new, f) or '—'}")


@st.dialog("Review the guideline changes", width="large", on_dismiss=_dismiss_review)
def review_dialog(version: int, title: str) -> None:
    prof = profile()
    doc = store.load(prof.procedure, version)
    if not doc.pending:
        st.success("Nothing left to review.")
        return
    prev = store.load(prof.procedure, doc.parent_version) if doc.parent_version else None
    # The only publish button, top right. It keeps any change marked "Reject" below at its old text.
    rejects = [x.id for x in doc.pending if st.session_state.get(f"dec-{version}-{x.id}") == "Reject"]
    head, btn = st.columns([3, 2], vertical_alignment="center")
    head.markdown(f"### {title}")
    if btn.button("Approve all and publish" if not rejects else f"Publish ({len(rejects)} rejected)",
                  type="primary", icon=":material/done_all:", key="tone-go-publish", width="stretch"):
        _publish(prof, doc, rejects)
    _review_summary(doc, prev)
    _capped(doc, prev, prof)
    _cited(doc, prev)

    if prev is None:
        st.markdown(f"**{len(doc.steps)} new guidelines** from {len(doc.runs)} {noun(len(doc.runs))}. "
                    "Read them through, then publish.")
        with st.expander("All guidelines", expanded=True):
            for phase in doc.phases:
                steps = doc.steps_in(phase)
                if steps:
                    st.markdown(f"**{phase_title(phase)}**\n" + "\n".join(f"- {s.action}" for s in steps))
    else:
        slot_of = {s.id: k for k, s in step_slots(doc.steps).items()}
        prev_slots = step_slots(prev.steps)
        # Values capped at a limit first: they are what a reviewer most needs to see.
        important = sorted((s for s in doc.pending if _is_important(s)), key=lambda s: not _is_capped(s))
        routine = [s for s in doc.pending if not _is_important(s)]
        st.markdown(f"**{len(doc.pending)} guideline{'s' if len(doc.pending) != 1 else ''} changed.** "
                    "Everything is approved unless you mark it *Reject* (keeps the previous version).")
        for s in important:
            with st.container(border=True):
                head, choice = st.columns([4, 2])
                head.markdown(f"**{phase_title(s.phase)} · {s.action}**")
                head.markdown(step_badges(s))
                with choice:
                    _decision(s, version)
                st.markdown("\n".join(f"- {r}" for r in s.change_reasons))
                old = prev_slots.get(slot_of[s.id])
                if s.change == "moved":
                    old = next((o for o in prev.steps if o.key == s.key and o.phase == s.moved_from), None)
                if old is not None and any((getattr(old, f) or "") != (getattr(s, f) or "") for f, _ in FIELDS):
                    _before_after(old, s)
                else:
                    guideline_body(s, doc)
        if routine:
            with st.expander(f"{len(routine)} guideline{'s' if len(routine) != 1 else ''} had their typical "
                             "ranges refined by the new data", icon=":material/tune:", expanded=not important):
                for s in routine:
                    head, choice = st.columns([4, 2])
                    head.markdown(f"**{phase_title(s.phase)} · {s.action}**  \n"
                                  + "; ".join(s.change_reasons[:3]))
                    with choice:
                        _decision(s, version)
        if doc.removed:
            st.markdown("**Removed:** " + ", ".join(f"{phase_title(s.phase)} · {s.label}" for s in doc.removed))


def _publish(prof: Profile, doc: Guidelines, rejects: list[str]) -> None:
    approve(prof, doc.version, None, rejects)
    st.session_state.pop("review_queue", None)
    st.session_state["toast"] = f"Version {doc.version} published."
    st.rerun()


def _review_summary(doc: Guidelines, prev: Guidelines | None) -> None:
    """The headline numbers of the new version (no arrow when nothing moved)."""
    m, pm = doc.metrics, prev.metrics if prev else {}
    new_runs = len(doc.runs) - len(prev.runs) if prev else 0
    pts = round((m["mean_confidence"] - pm["mean_confidence"]) * 100) if pm else 0
    c = st.columns(3)
    c[0].metric(noun(2, cap=True), len(doc.runs), f"+{new_runs}" if new_runs else None)
    c[1].metric("Guidelines changed" if prev else "New guidelines", len(doc.pending))
    c[2].metric("Confidence", f"{m['mean_confidence']:.0%}", f"{pts:+d} pts" if pts else None)


def _is_capped(s: Step) -> bool:
    return any(f.clipped for f in s.facts.values())


def _capped(doc: Guidelines, prev: Guidelines | None, prof: Profile) -> None:
    """In plain words: each value this version caps at an official limit, and each new
    run that broke a limit (left out of the typical range)."""
    by_id = {s.id: s for s in doc.steps}
    for s in doc.pending:
        broke = [run_title(f.run_id) for f in doc.flags if f.kind == "clipped" and f.step_id == s.id and f.run_id]
        for f in (f for f in s.facts.values() if f.clipped):
            lim = next((x for x in s.limits if x.signal == f.signal), None)
            source = f" — *{lim.text}* ({lim.doc_title})" if lim else ""
            st.error(f"**{phase_title(s.phase)} · {s.label}:** typical {prof.label(f.signal)} capped at "
                     f"**{f.fmt()}**{source}." + (f"  \nFlown past the limit in: {', '.join(broke)}." if broke else ""),
                     icon=":material/vertical_align_top:")
    known = set(prev.runs) if prev else set()
    for f in doc.flags:
        s = by_id.get(f.step_id or "")
        if f.kind == "excluded" and f.run_id not in known and s is not None:
            lim = next((x for x in s.limits if x.text in f.message), None)
            st.error(f"**{phase_title(s.phase)} · {s.label}:** {run_title(f.run_id)} broke a hard limit"
                     + (f" — *{lim.text}* ({lim.doc_title})" if lim else "")
                     + ". It is left out of the typical range.", icon=":material/remove_circle:")


def _cited(doc: Guidelines, prev: Guidelines | None) -> None:
    """Steps that only gained a citation: their wording did not change, so nothing to review."""
    if prev is None:
        return
    old = {k: s.reference and s.reference.doc for k, s in step_slots(prev.steps).items()}
    cited = [s.reference.doc for k, s in step_slots(doc.steps).items()
             if s.review != "pending" and s.reference and s.reference.doc != old.get(k)]
    if cited:
        n = len(cited)
        st.caption(f"{n} other guideline{'s now cite' if n != 1 else ' now cites'} "
                   f"{', '.join(sorted(set(cited)))}. The wording is unchanged, so there is nothing to review.")


# --- guideline card -----------------------------------------------------------------------------

def previous_steps(doc: Guidelines) -> dict[str, Step]:
    """Each step's counterpart in the version before (same slot), keyed by step id."""
    if not doc.parent_version:
        return {}
    old = step_slots(store.load(doc.procedure, doc.parent_version).steps)
    return {s.id: old[k] for k, s in step_slots(doc.steps).items() if k in old}


def changed_parts(s: Step, old: Step) -> list[str]:
    """Which parts of a guideline differ from the previous version: "When", "Typical", ..."""
    parts = [label for f, label in FIELDS if (getattr(old, f) or "") != (getattr(s, f) or "")]
    if (old.mandatory, old.required_by) != (s.mandatory, s.required_by):
        parts.append("Core / technique")
    return parts


def _change_badge(s: Step, old: Step | None = None) -> str | None:
    """With the previous version's step, a changed step says what changed ("Changed: Typical"),
    so a line that reads the same in the checklist shows where the difference is."""
    if s.change == "moved":
        return f":blue-badge[:material/swap_vert: Moved from {phase_title(s.moved_from or '')}]"
    if s.change == "modified" and old is not None:
        parts = changed_parts(s, old)
        return f":violet-badge[:material/edit: Changed: {', '.join(parts)}]" if parts else None
    return {"added": ":green-badge[:material/add_circle: New]",
            "modified": ":violet-badge[:material/edit: Changed]"}.get(s.change or "")


def step_badges(s: Step, show_change: bool = True, old: Step | None = None) -> str:
    """`show_change` is off on a first version, where every step is new; `old` is the step in
    the previous version."""
    out = []
    if s.required_by:
        out.append(":green-badge[:material/verified: Required by SOP]")
    elif s.run_count < 2:
        out.append(f":gray-badge[Seen in 1 {noun()}]")
    elif s.mandatory:
        out.append(":blue-badge[Core — every instructor]")
    else:
        out.append(f":orange-badge[Technique — {s.seen_in} of {s.run_count} instructors]")
    if show_change and (badge := _change_badge(s, old)):
        out.append(badge)
    if s.review == "pending":
        out.append(":orange-badge[:material/schedule: Awaiting approval]")
    if s.edited:
        out.append(":gray-badge[:material/person_edit: Edited by instructor]")
    if not s.verified:
        out.append(":red-badge[:material/error: Number not verified]")
    return " ".join(out)


def guideline_body(s: Step, doc: Guidelines) -> None:
    rows = [("When", s.trigger), ("Why", s.intent),
            ("Typical", s.typical and f"{s.typical} :gray[({len(s.runs)} {noun(len(s.runs))})]")]
    lines = [f":gray[{label}]&ensp;{text}" for label, text in rows if text]
    if s.limit:   # "Unchecked" is shown once at the top of the page, not on every step
        lines.append(f":red[Hard limit]&ensp;:red[{s.limit}]")
    st.markdown("  \n".join(lines))


def _limit_value(lim, prof: Profile) -> str:
    """"max 85 kt", "max 50°" (% and ° sit on the number, as in the guideline text)."""
    unit = prof.unit(lim.signal) or ""
    return f"{lim.describe()}{'' if unit in ('', '%', '°') else ' '}{unit}"


def limit_badges(s: Step, prof: Profile) -> str:
    """Short red chips for the hard limits beside a step, e.g. "max 85 kt"."""
    return " ".join(f":red-badge[{_limit_value(lim, prof)}]" for lim in s.limits)


def limit_name(lim, prof: Profile) -> str:
    """A limit in a few words, e.g. "Airspeed max 85 kt", "Max 11 m from where you stopped"."""
    label, value = prof.label(lim.signal), _limit_value(lim, prof)
    if label.startswith("from "):
        return f"{value[:1].upper()}{value[1:]} {label}"
    return f"{label[:1].upper()}{label[1:]} {value}"


def when_text(lim, prof: Profile) -> str:
    """When a limit applies, in plain words: "Flaps 20°/30° · Downwind", or "Always"."""
    parts = []
    if lim.when:
        m = re.fullmatch(r"\s*(\w+)\s*(==|>=|<=|>|<)\s*(-?[\d.]+)\s*", lim.when)
        toggle = prof.events.toggles.get(m.group(1)) if m else None
        if toggle:
            ok = {"==": float.__eq__, ">=": float.__ge__, "<=": float.__le__, ">": float.__gt__, "<": float.__lt__}
            states = [name for v, name in toggle.states.items() if ok[m.group(2)](float(v), float(m.group(3)))]
            parts.append(f"{toggle.name} {'/'.join(states)}".strip())
        else:
            parts.append(lim.when)
    if lim.phases:
        parts.append(", ".join(phase_title(p) for p in lim.phases))
    return " · ".join(parts) or "Always"


def checklist_line(n: int, s: Step, prof: Profile, show_change: bool = True, old: Step | None = None) -> str:
    """One line per step: what to do, when, and the hard limit."""
    when = (s.trigger or "").rstrip(".")
    if when[1:2].islower():   # "At about…" -> "at about…", but keep "IAS 70 kt"
        when = when[:1].lower() + when[1:]
    tags = []
    if s.required_by:
        tags.append(":green-badge[:material/verified: Required]")
    elif not s.mandatory:
        tags.append(f":orange-badge[Technique · {s.seen_in}/{s.run_count}]")
    if show_change and (badge := _change_badge(s, old)):
        tags.append(badge)
    line = f"{n}. **{s.action.rstrip('.')}**" + (f" — {when}" if when else "")
    return " ".join([line, limit_badges(s, prof), *tags]).strip()


def step_chart(s: Step, doc: Guidelines, prof: Profile) -> None:
    run_id = s.anchor_run or doc.anchor_run
    if not run_id or run_id not in {r.run_id for r in storage.list_runs(prof.procedure)}:
        st.caption(f"No {noun()} data available for a chart.")
        return
    df = telemetry(prof.procedure, run_id)
    window = df[(df["t"] >= s.t - 20) & (df["t"] <= s.t + 20)]
    signals = [x for x in prof.trigger_for(s.phase) if x in window]
    if not signals:
        return
    sig = st.segmented_control("Show", signals, default=signals[0], format_func=prof.label,
                               key=f"chart-{doc.version}-{s.id}")
    sig = sig or signals[0]
    fig = go.Figure(go.Scatter(x=window["t"] - s.t, y=window[sig], name=prof.label(sig),
                               line=dict(width=3)))
    fact = s.facts.get(f"trigger.{sig}")
    limits = []
    if fact and fact.lo is not None:
        fig.add_hrect(y0=fact.lo, y1=fact.hi, opacity=0.18, line_width=0, fillcolor="green",
                      annotation_text="typical range", annotation_position="top left")
        limits = [x for x in (fact.limit_max, fact.limit_min) if x is not None]
        for lim in limits:
            fig.add_hline(y=lim, line_color="red", line_dash="dot", annotation_text="hard limit")
    fig.add_vline(x=0, line_dash="dash", annotation_text="action")
    unit = prof.unit(sig) or ""
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                      xaxis_title="seconds from the action", yaxis_title=f"{prof.label(sig)} ({unit})")
    st.plotly_chart(fig, width="stretch")
    st.caption(f"From {run_title(run_id)}. Green band = what instructors typically do"
               + ("; red line = official limit." if limits else "."))
