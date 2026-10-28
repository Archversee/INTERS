"""High-level operations used by the CLI, API and UI."""

from __future__ import annotations

from pathlib import Path

from sim2sop import render, sop_store, storage
from sim2sop.agent.graph import build_graph
from sim2sop.agent.llm import LLMClient, get_llm
from sim2sop.agent.verifier import verify_step
from sim2sop.agent.writer import LLMWriter, TemplateWriter
from sim2sop.ingest import ingest_run
from sim2sop.learn import memory
from sim2sop.metrics import compute
from sim2sop.profiles import Profile
from sim2sop.rag.index import ReferenceIndex
from sim2sop.schema import SOP, Correction, Flag
from sim2sop.storage import RunRecord


def make_writer(profile: Profile, refs: ReferenceIndex, llm: LLMClient | None = None):
    llm = llm or get_llm()
    if llm.available():
        return LLMWriter(llm, profile, refs)
    return TemplateWriter(profile)


def generate_sop(profile: Profile, runs: list[RunRecord] | None = None,
                 llm: LLMClient | None = None, incremental: bool = True) -> SOP:
    """Run the planner -> writer -> verifier graph over all runs of the procedure and save
    the result as the next SOP version."""
    runs = runs if runs is not None else storage.list_runs(profile.procedure)
    if not runs:
        raise ValueError(f"No runs ingested for {profile.procedure!r}")
    refs = ReferenceIndex.for_domain(profile.references)
    writer = make_writer(profile, refs, llm)
    previous = sop_store.latest(profile.procedure) if incremental else None
    graph = build_graph(profile, refs, writer)
    state = graph.invoke({"runs": runs, "previous": previous})
    sop = SOP(
        procedure=profile.procedure, domain=profile.domain, title=profile.title,
        runs=[r.run_id for r in runs], anchor_run=state["anchor_run"],
        phases=profile.phase_names, steps=state["steps"],
        flags=state["flags"], writer=writer.name,
    )
    sop.metrics = compute(sop, profile)
    sop = sop_store.save(sop)
    render.write_html(sop)
    return sop


def apply_edit(profile: Profile, step_id: str, field: str, new_text: str,
               version: int | None = None) -> SOP:
    """Instructor edit: store the before/after diff in correction memory, re-verify the
    step, and save the edited SOP as a new version."""
    sop = sop_store.load(profile.procedure, version) if version else sop_store.latest(profile.procedure)
    if sop is None:
        raise ValueError("No SOP to edit")
    step = next((s for s in sop.steps if s.id == step_id), None)
    if step is None:
        raise ValueError(f"No step {step_id!r} in v{sop.version}")
    before = getattr(step, field) or ""
    memory.add_correction(Correction(
        domain=profile.domain, procedure=profile.procedure, phase=step.phase,
        step_kind=step.kind, step_key=step.key, field=field, before=before, after=new_text,
        pattern=memory.to_pattern(new_text, step.facts, step.label, step.intent),
        sop_version=sop.version,
    ))
    setattr(step, field, new_text)
    step.edited = True
    step.corrections_seen = memory.relevant_count(profile.domain, step.kind, step.signal)
    verdict = verify_step(step, profile)
    step.verified = verdict.ok
    step.flags = [f"Unverified — {e}" for e in verdict.errors] + verdict.warnings
    sop.flags = [f for f in sop.flags if f.step_id != step_id] + [
        Flag(kind="verifier", step_id=step_id, message=e) for e in verdict.errors]
    if not sop.writer.endswith("+ instructor edit"):
        sop.writer = f"{sop.writer} + instructor edit"
    sop.metrics = compute(sop, profile)
    sop = sop_store.save(sop)
    render.write_html(sop)
    return sop


def ingest_dir(profile: Profile, raw_dir: Path) -> list[RunRecord]:
    """Ingest every run folder under raw_dir (sorted by name)."""
    return [ingest_run(d, profile) for d in sorted(p for p in Path(raw_dir).iterdir() if p.is_dir())]
