"""Planner: turn multi-run consensus into guideline slots.

- Phases can learn from different runs (`phase_runs`): after a technique change, the
  affected phases use only post-change runs while the others keep their runs. Consensus
  is built once per distinct run set, so untouched phases come out identical.
- Each step gets its typical ranges, the hard limits that apply, a citation, confidence
  and technique notes. Runs that break a limit are excluded from that step's ranges;
  what remains is clipped to the limit (see sim2sop.learn.limits).
- Actions a reference SOP requires are always core guidelines.
- Steps whose numbers did not change since the previous version keep their text (including
  instructor edits) and are not sent to the writer again. A new citation changes no text, a
  new or revised limit only re-quotes the limit, and a new instructor correction only
  restyles the fields it corrected.
- A change the instructor rejected is not proposed again until the data behind it changes.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from sim2sop.learn import memory
from sim2sop.learn.changes import slot, step_slots
from sim2sop.learn.consensus import ConsensusItem, build_consensus, confidence, item_facts
from sim2sop.learn.limits import applicable, clip_facts, event_violations, relevant
from sim2sop.profiles import Profile
from sim2sop.rag.index import ReferenceIndex
from sim2sop.schema import Flag, Guidelines, Limit, Requirement, Step, callout
from sim2sop.storage import RunRecord

MERGE_WINDOW_S = 1.0
TEXT_FIELDS = ("trigger", "action", "intent", "typical", "limit")


@dataclass
class PlanResult:
    steps: list[Step]
    dirty: list[str]                         # step ids to (re)write
    flags: list[Flag] = field(default_factory=list)
    partial: dict[str, list[str]] = field(default_factory=dict)  # step id -> only these fields
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


def _signature(s: Step, profile: Profile) -> tuple:
    """What the wording depends on: the stated numbers and whether it is core or technique.
    Numbers the guideline does not state, and the flight count, can move without a rewrite."""
    facts = tuple((n, s.facts[n].fmt(), s.facts[n].fmt_range(), s.facts[n].clipped)
                  for n in profile.stated_facts(s.phase) if n in s.facts)
    technique = None if s.mandatory else (s.seen_in, s.run_count)
    return (s.key, s.phase, s.mandatory, technique, facts)


def _limits(s: Step) -> tuple:
    return tuple(sorted((lim.id, lim.revision or "", lim.describe()) for lim in s.limits))


NEW = "new:"   # prefix of a rejected proposal for a step that did not exist before


def proposal(s: Step, profile: Profile) -> str:
    """Identity of a proposed change, remembered when the instructor rejects it."""
    return repr((_signature(s, profile), _limits(s), s.required_by))


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
    tied: dict[str, set[str]] = {}   # step id -> callouts as common as the one chosen

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
            what = ("left out of the typical range" if run in exclude
                    else "every flight broke it, so the typical value is capped at the limit")
            flags.append(Flag(kind="excluded" if run in exclude else "clipped", run_id=run, step_id=sid,
                              phase=head.phase, message=(
                                  f"{run}: {head.label} at {profile.label(lim.signal)} {val:.0f} breaks "
                                  f"'{lim.text}' ({lim.cite()}); {what}")))

        facts = item_facts(head, profile, exclude)
        clipped = clip_facts(facts, lims)
        label = "; ".join(_label(it, item_facts(it, profile, exclude) if it is not head else facts)
                          for it in group)
        spoken = next((it for it in group if it.intent), None)
        intent = spoken.intent if spoken else None
        tied[sid] = spoken.top_intents if spoken else set()
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
            reference=None if refs.empty else refs.best(f"{head.phase} {head.phase} {label} {intent or ''}",
                                                        phase=head.phase),
            corrections_seen=memory.relevant_count(profile.domain, head.kind, head.signal),
        )
        step.flags += [f"Clipped to hard limit — {c}" for c in clipped]
        if not step.mandatory:
            step.technique_notes.append(f"Technique: flown in {head.seen_in} of {head.run_count} flights.")
        variants = sorted({v for it in group for v in it.intents})
        if len(variants) > 1:
            step.technique_notes.append("Callouts used: " + " / ".join(f'"{v}"' for v in variants))
        steps.append(step)

    for req in requirements:
        if not any(req.key in s.key.split("+") for s in steps):
            flags.append(Flag(kind="required_missing", message=(
                f"'{req.text}' is required by {req.doc_title} but was not seen in any run")))

    dirty = [s.id for s in steps]
    partial: dict[str, list[str]] = {}
    if previous is not None:
        prev_slots = step_slots(previous.steps)
        dirty, kept = [], []
        new_slots = step_slots(steps)
        orphans = [o for k, o in prev_slots.items() if k not in new_slots]
        dropped: dict[tuple[str, str], int] = defaultdict(int)   # rejected new steps left out so far
        for slot_key, s in new_slots.items():
            here = slot(s.phase, s.key, int(slot_key.rsplit("|", 1)[1]) - dropped[(s.phase, s.key)])
            old = prev_slots.get(here)
            rejected = previous.rejected.get(here)
            if rejected == NEW + proposal(s, profile):
                dropped[(s.phase, s.key)] += 1   # later steps with this key keep their old numbering
                continue
            if rejected == proposal(s, profile):
                # A rejected change keeps the previous text; a rejected move stays in its old phase.
                old = old or next((o for o in orphans if o.key == s.key), None)
                if old is not None:
                    kept.append(old.model_copy(update={"id": s.id, "change": None, "change_reasons": [],
                                                       "review": None, "moved_from": None}))
                continue
            kept.append(s)
            if old is not None and callout(old.intent) in tied[s.id]:
                s.intent = callout(old.intent)   # an equally common callout: keep the one already shown
            if old is None or _signature(s, profile) != _signature(old, profile):
                dirty.append(s.id)
                continue
            for f in TEXT_FIELDS:
                setattr(s, f, getattr(old, f))
            s.verified, s.edited, s.flags = old.verified, old.edited, old.flags
            fields = ["limit"] if _limits(s) != _limits(old) else []
            if not old.edited and old.corrections_seen != s.corrections_seen:
                fields += memory.corrected_fields(profile.domain, s.kind, s.signal)
            if fields:
                partial[s.id] = fields
                dirty.append(s.id)
        steps = kept
    return PlanResult(steps=steps, dirty=dirty, flags=flags, anchor_run=main_anchor, partial=partial)
