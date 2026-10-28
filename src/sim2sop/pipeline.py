"""High-level operations used by the CLI, API and UI.

  create_guidelines   new guidelines from every ingested run (v1 of a procedure)
  update_guidelines   next version from the latest published one: new runs, technique
                      changes in a new environment, revised references, corrections;
                      only affected steps change, each with its reasons, pending approval
  approve             instructor approves / rejects changed steps; publishes when none pend
  apply_edit          instructor edit, stored in correction memory, saved as a new version

Reference SOPs are read-only inputs (sim2sop.references); nothing here writes to them.
"""

from __future__ import annotations

from pathlib import Path

from sim2sop import guidelines_store as store
from sim2sop import references, render, storage
from sim2sop.agent.graph import build_graph
from sim2sop.agent.llm import LLMClient, get_llm
from sim2sop.agent.planner import plan
from sim2sop.agent.verifier import verify_step
from sim2sop.agent.writer import LLMWriter, TemplateWriter
from sim2sop.ingest import ingest_run
from sim2sop.learn import memory
from sim2sop.learn.changes import ChangeReport, detect_changes, step_slots
from sim2sop.metrics import compute
from sim2sop.profiles import Profile
from sim2sop.rag.index import ReferenceIndex
from sim2sop.schema import Correction, Flag, Guidelines, RefDoc, Step
from sim2sop.signals.compliance import violation_report
from sim2sop.storage import RunRecord

TEXT_FIELDS = store.TEXT_FIELDS


def make_writer(profile: Profile, refs: ReferenceIndex, llm: LLMClient | None = None):
    llm = llm or get_llm()
    if llm.available():
        return LLMWriter(llm, profile, refs)
    return TemplateWriter(profile)


# --- create / update -----------------------------------------------------------------------

def create_guidelines(profile: Profile, llm: LLMClient | None = None) -> Guidelines:
    """New guidelines from every ingested run of the procedure."""
    runs = storage.list_runs(profile.procedure)
    if not runs:
        raise ValueError(f"No runs ingested for {profile.procedure!r}")
    ids = [r.run_id for r in runs]
    envs = [r.meta.environment for r in runs if r.meta.environment]
    return _build(profile, mode="create", llm=llm, previous=None,
                  phase_runs={p: ids for p in profile.phase_names},
                  environment=envs[-1] if envs else None)


def update_guidelines(profile: Profile, llm: LLMClient | None = None) -> Guidelines:
    """Next version from the latest published guidelines (or the latest draft)."""
    prev = store.latest_published(profile.procedure) or store.latest(profile.procedure)
    if prev is None:
        return create_guidelines(profile, llm)
    runs = storage.list_runs(profile.procedure)
    known = set(prev.runs)
    phase_runs = {p: list(prev.phase_runs.get(p, prev.runs)) for p in profile.phase_names}

    # More runs from the same environment: every phase learns from them.
    same_env = [r.run_id for r in runs if r.run_id not in known and r.meta.environment == prev.environment]
    for p in phase_runs:
        phase_runs[p] += same_env

    # Runs from a new environment: a technique change rebuilds only the affected phases.
    report = detect_changes(prev, [r for r in runs if r.run_id not in known], profile)
    environment = prev.environment
    if report.changed:
        environment = report.environment
        for p in report.changed:
            phase_runs[p] = list(report.post_runs)
    elif report.post_runs and not report.pending:
        environment = report.environment
        for p in phase_runs:
            phase_runs[p] += report.post_runs
    return _build(profile, mode="update", llm=llm, previous=prev, phase_runs=phase_runs,
                  environment=environment, report=report)


def _build(profile: Profile, *, mode: str, llm: LLMClient | None, previous: Guidelines | None,
           phase_runs: dict[str, list[str]], environment: str | None,
           report: ChangeReport | None = None) -> Guidelines:
    all_runs = storage.list_runs(profile.procedure)
    used_ids = {i for ids in phase_runs.values() for i in ids}
    runs = [r for r in all_runs if r.run_id in used_ids]
    docs = references.list_references(profile.procedure)
    limits = references.active_limits(profile.procedure)
    reqs = references.active_requirements(profile.procedure)
    refs = ReferenceIndex.for_procedure(profile.procedure)
    writer = make_writer(profile, refs, llm)

    graph = build_graph(profile, writer, lambda: plan(runs, profile, refs, limits, reqs, previous, phase_runs))
    state = graph.invoke({})

    flags: list[Flag] = []
    if not docs:
        flags.append(Flag(kind="unchecked", message=(
            "No reference SOPs supplied: these guidelines are UNCHECKED against hard limits. "
            "Add references and update to make them limit-checked.")))
    flags += report.flags if report else []
    flags += state["flags"]
    flags += violation_report(runs, limits, profile)
    flags += [f for r in runs for f in r.flags]

    doc = Guidelines(
        procedure=profile.procedure, domain=profile.domain, title=profile.title, mode=mode,
        parent_version=previous.version if previous else None, checked=bool(docs),
        runs=[r.run_id for r in runs], phase_runs=phase_runs, environment=environment,
        anchor_run=state["anchor_run"], references=[d.label for d in docs],
        phases=profile.phase_names, steps=state["steps"], flags=flags, writer=writer.name,
    )
    annotate_changes(doc, previous, profile, report, docs)
    doc.status = "draft" if doc.pending else (previous.status if previous else "draft")
    doc.metrics = compute(doc, profile)
    doc = store.save_new(doc)
    render.write_html(doc)
    return doc


# --- reasons for each change -----------------------------------------------------------------

def _reasons(profile: Profile, s: Step, o: Step) -> list[str]:
    out = []
    old_l, new_l = {lim.id: lim for lim in o.limits}, {lim.id: lim for lim in s.limits}
    for i in new_l.keys() - old_l.keys():
        out.append(f"Hard limit added: {new_l[i].text} ({new_l[i].cite()})")
    for i in new_l.keys() & old_l.keys():
        a, b = old_l[i], new_l[i]
        if a.describe() != b.describe():
            out.append(f"Hard limit revised: {a.describe()} → {b.describe()} ({b.cite()})")
        elif a.revision != b.revision:
            out.append(f"Re-checked against {b.cite()} (limit unchanged)")
    for i in old_l.keys() - new_l.keys():
        out.append(f"Hard limit no longer applies: {old_l[i].text} ({old_l[i].cite()})")
    if any(f.clipped for f in s.facts.values()) and not any(f.clipped for f in o.facts.values()):
        out.append("Value clipped to a hard limit")
    if len(s.runs) != len(o.runs):
        out.append(f"Typical ranges from {len(s.runs)} runs (was {len(o.runs)})")
    for name in profile.stated_facts(s.phase):
        a, b = o.facts.get(name), s.facts.get(name)
        if a and b and a.fmt_range() != b.fmt_range():
            out.append(f"{profile.label(b.signal)}: {a.fmt_range()} → {b.fmt_range()}")
    if s.mandatory != o.mandatory or s.required_by != o.required_by:
        out.append(f"Now required by {s.required_by}" if s.required_by else
                   "Now a core guideline (every run)" if s.mandatory else
                   f"Now a technique note ({s.seen_in}/{s.run_count} runs)")
    if s.corrections_seen != o.corrections_seen and not o.edited:
        out.append("Instructor correction applied (house style)")
    if (s.reference and s.reference.doc) != (o.reference and o.reference.doc):
        out.append(f"Cited: {s.reference.doc} — {s.reference.section}" if s.reference else "Citation removed")
    return out


def annotate_changes(doc: Guidelines, prev: Guidelines | None, profile: Profile,
                     report: ChangeReport | None, docs: list[RefDoc]) -> None:
    """Mark each step added / modified / unchanged with its reasons; changed steps pend."""
    if prev is None:
        for s in doc.steps:
            s.change, s.review = "added", "pending"
            s.change_reasons = [f"New guideline from {len(s.runs)} run{'s' if len(s.runs) != 1 else ''}"]
        return
    new_docs = [d for d in docs if d.label not in prev.references]
    prev_slots, new_slots = step_slots(prev.steps), step_slots(doc.steps)
    for slot, s in new_slots.items():
        o = prev_slots.get(slot)
        reasons = []
        if report and s.phase in report.changed:
            reasons.append(f"Technique change in {s.phase}: {len(report.post_runs)} runs tagged "
                           f"'{report.environment}' fly it differently")
            # Cite what explains the change: documents for this environment, or new NOTAMs.
            reasons += [f"See {d.label}" for d in docs
                        if any(lim.environment == report.environment for lim in d.limits)
                        or (d in new_docs and d.kind == "notam")]
        if o is None:
            s.change = "added"
            reasons.append(f"New action seen in {s.seen_in}/{s.run_count} runs")
        else:
            changed = any(getattr(o, f) != getattr(s, f) for f in TEXT_FIELDS) or o.mandatory != s.mandatory
            if not changed:
                s.change, s.review, s.change_reasons = None, None, []
                continue
            s.change = "modified"
            reasons += _reasons(profile, s, o) or ["Wording updated"]
        s.change_reasons, s.review = reasons, "pending"
    doc.removed = [o.model_copy(update={"change_reasons": ["No longer seen in the runs behind this phase"]})
                   for slot, o in prev_slots.items() if slot not in new_slots]


# --- approval and edits ----------------------------------------------------------------------

def approve(profile: Profile, version: int | None = None, approve_ids: list[str] | None = None,
            reject_ids: list[str] = ()) -> Guidelines:
    """Approve pending steps (all, or `approve_ids`) and reject `reject_ids`: a rejected
    step goes back to the previous version's text. Publishes when nothing is pending."""
    doc = store.load(profile.procedure, version) if version else store.latest(profile.procedure)
    if doc is None:
        raise ValueError("No guidelines to approve")
    prev = store.load(profile.procedure, doc.parent_version) if doc.parent_version else None
    prev_slots = step_slots(prev.steps) if prev else {}
    keep: list[Step] = []
    for slot, s in step_slots(doc.steps).items():
        if s.review == "pending" and s.id in reject_ids:
            o = prev_slots.get(slot)
            if o is None:
                doc.removed.append(s.model_copy(update={"review": "rejected"}))
                continue
            s = o.model_copy(update={"id": s.id, "review": "rejected", "change": None,
                                     "change_reasons": s.change_reasons})
        elif s.review == "pending" and (approve_ids is None or s.id in approve_ids):
            s.review = "approved"
        keep.append(s)
    doc.steps = keep
    if not doc.pending:
        doc.status = "published"
    doc.metrics = compute(doc, profile)
    store.overwrite(doc)
    render.write_html(doc)
    return doc


def apply_edit(profile: Profile, step_id: str, field: str, new_text: str,
               version: int | None = None) -> Guidelines:
    """Instructor edit: store the before/after diff in correction memory, re-verify the
    step, and save the edited guidelines as a new version."""
    if field not in ("trigger", "action", "intent", "typical"):
        raise ValueError("Hard limits come from the reference SOPs and cannot be edited here")
    base = store.load(profile.procedure, version) if version else store.latest(profile.procedure)
    if base is None:
        raise ValueError("No guidelines to edit")
    doc = base.model_copy(deep=True)
    step = next((s for s in doc.steps if s.id == step_id), None)
    if step is None:
        raise ValueError(f"No step {step_id!r} in v{base.version}")
    before = getattr(step, field) or ""
    memory.add_correction(Correction(
        domain=profile.domain, procedure=profile.procedure, phase=step.phase,
        step_kind=step.kind, step_key=step.key, field=field, before=before, after=new_text,
        pattern=memory.to_pattern(new_text, step.facts, step.label, step.intent),
        version=base.version,
    ))
    for s in doc.steps:
        if s.review != "pending":
            s.change, s.change_reasons, s.review = None, [], None
    setattr(step, field, new_text)
    step.edited = True
    step.corrections_seen = memory.relevant_count(profile.domain, step.kind, step.signal)
    step.change, step.change_reasons, step.review = "modified", [f"Instructor edit ({field})"], "approved"
    verdict = verify_step(step, profile)
    step.verified = verdict.ok
    step.flags = [f"Unverified — {e}" for e in verdict.errors] + verdict.warnings
    doc.flags = [f for f in doc.flags if f.step_id != step_id or f.kind != "verifier"] + [
        Flag(kind="verifier", step_id=step_id, message=e) for e in verdict.errors]
    doc.mode, doc.parent_version, doc.removed = "edit", base.version, []
    if not doc.writer.endswith("+ instructor edit"):
        doc.writer = f"{doc.writer} + instructor edit"
    doc.metrics = compute(doc, profile)
    doc = store.save_new(doc)
    render.write_html(doc)
    return doc


def ingest_dir(profile: Profile, raw_dir: Path, environment: str | None = None) -> list[RunRecord]:
    """Ingest every run folder under raw_dir (sorted by name)."""
    return [ingest_run(d, profile, environment=environment)
            for d in sorted(p for p in Path(raw_dir).iterdir() if p.is_dir())]
