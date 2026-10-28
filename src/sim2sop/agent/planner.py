"""Planner: turn the multi-run consensus into step slots.

Each consensus item becomes a step with its facts, confidence, reference citation and
technique notes. Toggles thrown together (landing light + strobe) merge into one step.
Steps whose consensus did not change since the previous SOP version keep their text
(including instructor edits) and are not sent to the writer again.
"""

from __future__ import annotations

from sim2sop.learn import memory
from sim2sop.learn.consensus import ConsensusItem, build_consensus, confidence, item_facts
from sim2sop.profiles import Profile
from sim2sop.rag.index import ReferenceIndex
from sim2sop.schema import SOP, Flag, Step
from sim2sop.storage import RunRecord

MERGE_WINDOW_S = 1.0
TEXT_FIELDS = ("trigger", "action", "intent", "expected", "tolerance")


def _groups(items: list[ConsensusItem]) -> list[list[ConsensusItem]]:
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


def _unchanged(new: Step, old: Step) -> bool:
    if (new.key, new.phase, new.seen_in, new.run_count) != (old.key, old.phase, old.seen_in, old.run_count):
        return False
    for name, f in new.facts.items():
        o = old.facts.get(name)
        if o is None or f.fmt() != o.fmt() or f.fmt_tol() != o.fmt_tol():
            return False
    return True


def plan(runs: list[RunRecord], profile: Profile, refs: ReferenceIndex,
         previous: SOP | None = None) -> tuple[list[Step], list[str], list[Flag], RunRecord]:
    """Return (steps, ids of steps to (re)write, flags, anchor run)."""
    items, anchor = build_consensus(runs)
    n = len(runs)
    steps: list[Step] = []
    for k, group in enumerate(_groups(items)):
        head = group[0]
        facts = item_facts(head, profile)
        label = "; ".join(_label(it, item_facts(it, profile) if it is not head else facts) for it in group)
        intent = next((it.intent for it in group if it.intent), None)
        query = f"{head.phase} {head.phase} {label} {intent or ''}"
        step = Step(
            id=f"s{k + 1:02d}", phase=head.phase, key="+".join(it.key for it in group),
            kind=head.kind, signal=head.signal, label=label, intent=intent,
            facts=facts, t=head.t_anchor, mandatory=head.mandatory,
            seen_in=head.seen_in, run_count=n, confidence=confidence(head, facts, profile),
            reference=refs.best(query),
            corrections_seen=memory.relevant_count(profile.domain, head.kind, head.signal),
        )
        if not head.mandatory:
            step.technique_notes.append(f"Technique: flown in {head.seen_in} of {n} runs.")
        variants = sorted({v for it in group for v in it.intents})
        if len(variants) > 1:
            step.technique_notes.append("Callouts used: " + " / ".join(f'"{v}"' for v in variants))
        steps.append(step)

    dirty = [s.id for s in steps]
    if previous is not None:
        prev_by_key = {s.key: s for s in previous.steps}
        dirty = []
        for s in steps:
            old = prev_by_key.get(s.key)
            # Rewrite when the consensus moved, or when a new instructor correction applies
            # (an instructor-edited step keeps its edit until its numbers change).
            new_correction = old is not None and not old.edited and old.corrections_seen != s.corrections_seen
            if old is not None and _unchanged(s, old) and not new_correction:
                for f in TEXT_FIELDS:
                    setattr(s, f, getattr(old, f))
                s.reference, s.verified, s.edited, s.flags = old.reference, old.verified, old.edited, old.flags
            else:
                dirty.append(s.id)

    flags = [f for r in runs for f in r.flags]
    return steps, dirty, flags, anchor
