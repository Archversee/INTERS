"""Planner: turn multi-run consensus into guideline slots.

- Phases can learn from different runs (`phase_runs`): after a technique change, the
  affected phases use only post-change runs while the others keep their runs. Consensus
  is built once per distinct run set, so untouched phases come out identical.
- Each step gets its typical ranges, the hard limits that apply, a citation, confidence
  and technique notes. Runs that break a limit are excluded from that step's ranges;
  what remains is clipped to the limit (see sim2sop.learn.limits).
- Actions a reference SOP requires are always core guidelines.
- Steps whose numbers, limits and citation did not change since the previous version keep
  their text (including instructor edits) and are not sent to the writer again.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from sim2sop.learn import memory
from sim2sop.learn.changes import step_slots
from sim2sop.learn.consensus import ConsensusItem, build_consensus, confidence, item_facts
from sim2sop.learn.limits import applicable, clip_facts, event_violations, relevant
from sim2sop.profiles import Profile
from sim2sop.rag.index import ReferenceIndex
from sim2sop.schema import Flag, Guidelines, Limit, Requirement, Step
from sim2sop.storage import RunRecord

MERGE_WINDOW_S = 1.0
TEXT_FIELDS = ("trigger", "action", "intent", "typical", "limit")


@dataclass
class PlanResult:
    steps: list[Step]
    dirty: list[str]                         # step ids to (re)write
    flags: list[Flag] = field(default_factory=list)
    anchor_run: str | None = None


def _groups(items: list[ConsensusItem]) -> list[list[ConsensusItem]]:
    """Toggles thrown together (landing light + strobe) become one step."""
    groups: list[list[ConsensusItem]] = []
    for it in items:
        last = groups[-1][-1] if groups else None
        if (last and it.kind == "toggle" and last.kind == "toggle" and it.phase == last.phase
                and it.mandatory == last.mandatory and it.t_anchor - last.t_anchor <= MERGE_WINDOW_S):
            groups[-1].append(it)
        else:
            groups.append([it])
    return groups


def _label(item: ConsensusItem, facts) -> str:
    """Restate the action with the consensus value, not one run's value."""
    if item.template and "action" in facts:
        return item.template.format(value=facts["action"].fmt())
    return item.label


def _signature(s: Step) -> tuple:
    facts = tuple(sorted((n, f.fmt(), f.fmt_range()) for n, f in s.facts.items()))
    limits = tuple(sorted((lim.id, lim.revision or "", lim.describe()) for lim in s.limits))
    ref = (s.reference.doc, s.reference.section) if s.reference else None
    return (s.key, s.phase, s.seen_in, s.run_count, s.mandatory, s.required_by, facts, limits, ref)


def _consensus_by_phase(runs: list[RunRecord], profile: Profile,
                        phase_runs: dict[str, list[str]]) -> tuple[list[tuple[ConsensusItem, str]], str | None]:
    """Consensus items in procedure order, each with the anchor run its times refer to."""
    by_id = {r.run_id: r for r in runs}
    sets: dict[frozenset, list[str]] = defaultdict(list)
    for phase in profile.phase_names:
        ids = frozenset(i for i in phase_runs.get(phase, []) if i in by_id)
        if ids:
            sets[ids].append(phase)
    order = {p: i for i, p in enumerate(profile.phase_names)}
    out, main_anchor, most = [], None, -1
    for ids, phases in sets.items():
        items, anchor = build_consensus([r for r in runs if r.run_id in ids])
        out += [(it, anchor.run_id) for it in items if it.phase in phases]
        if len(phases) > most:
            main_anchor, most = anchor.run_id, len(phases)
    out.sort(key=lambda x: (order.get(x[0].phase, 99), x[0].t_anchor))
    return out, main_anchor


def plan(runs: list[RunRecord], profile: Profile, refs: ReferenceIndex,
         limits: list[Limit], requirements: list[Requirement],
         previous: Guidelines | None = None,
         phase_runs: dict[str, list[str]] | None = None) -> PlanResult:
    phase_runs = phase_runs or {p: [r.run_id for r in runs] for p in profile.phase_names}
    ordered, main_anchor = _consensus_by_phase(runs, profile, phase_runs)
    anchors = {id(it): a for it, a in ordered}
    required = {r.key: r for r in requirements}
    flags: list[Flag] = []
    steps: list[Step] = []

    for k, group in enumerate(_groups([it for it, _ in ordered])):
        head = group[0]
        sid = f"s{k + 1:02d}"
        events = list(head.events.values())
        lims = applicable(limits, head.phase, events)

        # Runs that break a limit at this step: excluded if any compliant run remains.
        bad = {run: event_violations(ev, lims) for run, ev in head.events.items()}
        bad = {run: v for run, v in bad.items() if v}
        exclude = set(bad) if bad and len(bad) < len(head.events) else set()
        for run, viol in bad.items():
            lim, val = viol[0]
            what = "excluded from the typical range" if run in exclude else "no compliant run: value clipped"
            flags.append(Flag(kind="excluded" if run in exclude else "clipped", run_id=run, step_id=sid,
                              phase=head.phase, message=(
                                  f"{run}: {head.label} at {profile.label(lim.signal)} {val:.0f} breaks "
                                  f"'{lim.text}' ({lim.cite()}); {what}")))

        facts = item_facts(head, profile, exclude)
        clipped = clip_facts(facts, lims)
        label = "; ".join(_label(it, item_facts(it, profile, exclude) if it is not head else facts)
                          for it in group)
        intent = next((it.intent for it in group if it.intent), None)
        req = next((required[it.key] for it in group if it.key in required), None)
        used_runs = sorted(r for r in head.events if r not in exclude)
        step = Step(
            id=sid, phase=head.phase, key="+".join(it.key for it in group),
            kind=head.kind, signal=head.signal, label=label, intent=intent,
            facts=facts, limits=relevant(lims, facts), t=head.t_anchor, anchor_run=anchors[id(head)],
            mandatory=head.mandatory or req is not None,
            required_by=f"{req.doc_title}: {req.text}" if req else None,
            seen_in=head.seen_in, run_count=head.run_count, runs=used_runs,
            confidence=confidence(head.seen_in - len(exclude), head.run_count, facts, profile),
            reference=None if refs.empty else refs.best(f"{head.phase} {head.phase} {label} {intent or ''}"),
            corrections_seen=memory.relevant_count(profile.domain, head.kind, head.signal),
        )
        step.flags += [f"Clipped to hard limit — {c}" for c in clipped]
        if not step.mandatory:
            step.technique_notes.append(f"Technique: flown in {head.seen_in} of {head.run_count} runs.")
        variants = sorted({v for it in group for v in it.intents})
        if len(variants) > 1:
            step.technique_notes.append("Callouts used: " + " / ".join(f'"{v}"' for v in variants))
        steps.append(step)

    for req in requirements:
        if not any(req.key in s.key.split("+") for s in steps):
            flags.append(Flag(kind="required_missing", message=(
                f"'{req.text}' is required by {req.doc_title} but was not seen in any run")))

    dirty = [s.id for s in steps]
    if previous is not None:
        prev_slots = step_slots(previous.steps)
        dirty = []
        for slot_key, s in step_slots(steps).items():
            old = prev_slots.get(slot_key)
            new_correction = old is not None and not old.edited and old.corrections_seen != s.corrections_seen
            if old is not None and _signature(s) == _signature(old) and not new_correction:
                for f in TEXT_FIELDS:
                    setattr(s, f, getattr(old, f))
                s.verified, s.edited, s.flags = old.verified, old.edited, old.flags
            else:
                dirty.append(s.id)
    return PlanResult(steps=steps, dirty=dirty, flags=flags, anchor_run=main_anchor)
