"""Patterns across instructor runs, found with plain statistics (no AI):

  instructor      one instructor does a step at a clearly different value than the others
  habit           a technique step that belongs to particular instructors
  together        two technique steps that always come together
  effect          runs with a technique do a later step at a clearly different value
  limit_break     which runs and instructors break an official limit
  spread          the steps instructors agree on least

"Clearly different" = more than the signal's tolerance in the profile. Each pattern carries
the numbers its sentence states, so the AI can explain the patterns (sim2sop.agent.insights)
and a checker can confirm every number it writes.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import median
from typing import Literal

from pydantic import BaseModel, Field

from sim2sop.learn.changes import step_slots
from sim2sop.profiles import Profile
from sim2sop.schema import Event, Guidelines, Step
from sim2sop.storage import RunRecord

Kind = Literal["instructor", "habit", "together", "effect", "limit_break", "spread"]


class Pattern(BaseModel):
    id: str
    kind: Kind
    text: str                                  # the finding in one sentence
    numbers: list[float] = Field(default_factory=list)   # every number `text` states
    step: str | None = None                    # "Phase · step" it is about
    runs: list[str] = Field(default_factory=list)
    strength: float = 0.0                      # for ranking: bigger = clearer


def person(raw: str | None) -> str | None:
    """"instructor_2" -> "Instructor 2"; no name (or the generator's placeholder) -> None."""
    return None if raw in (None, "", "generated") else raw.replace("_", " ").title()


def _phase(name: str) -> str:
    return name.replace("_", " ").title()


def _where(s: Step) -> str:
    return f"{_phase(s.phase)} · {s.label}"


def _fmt(profile: Profile, signal: str, v: float) -> str:
    spec = profile.signals.get(signal)
    unit = spec.unit if spec and spec.unit else ""
    text = f"{v:.{spec.decimals if spec else 0}f}"
    text = text.lstrip("-") if float(text) == 0 else text
    return f"{text}{unit}" if unit in ("", "%", "°") else f"{text} {unit}"


def _at(profile: Profile, signal: str, v: float) -> str:
    """"angle to the kerb 43°", "8.6 m from where you stopped" (labels starting "from " go after)."""
    label = profile.label(signal)
    return f"{_fmt(profile, signal, v)} {label}" if label.startswith("from ") else f"{label} {_fmt(profile, signal, v)}"


def _num(profile: Profile, signal: str, v: float) -> float:
    spec = profile.signals.get(signal)
    return abs(round(v, spec.decimals if spec else 0))


def events_per_step(doc: Guidelines, runs: list[RunRecord]) -> dict[str, dict[str, Event]]:
    """step id -> {run id: that run's event for the step}. A run's n-th event with the step's
    key in its phase belongs to the step in the n-th slot of that key, as across versions."""
    out: dict[str, dict[str, Event]] = defaultdict(dict)
    by_slot = {k: s for k, s in step_slots(doc.steps).items()}
    for r in runs:
        seen: dict[tuple[str, str], int] = defaultdict(int)
        for e in r.events:
            n = seen[(e.phase, e.key)]
            seen[(e.phase, e.key)] += 1
            for k, s in by_slot.items():
                phase, key, occ = k.split("|")
                if phase == e.phase and key.split("+")[0] == e.key and int(occ) == n:
                    out[s.id][r.run_id] = e
    return out


def find_patterns(doc: Guidelines, runs: list[RunRecord], profile: Profile, top: int = 12) -> list[Pattern]:
    runs = [r for r in runs if r.run_id in set(doc.runs)]
    who = {r.run_id: person(r.meta.instructor) for r in runs}
    per_step = events_per_step(doc, runs)
    found: list[Pattern] = []

    def tol(signal: str) -> float | None:
        spec = profile.signals.get(signal)
        return spec.tolerance if spec and spec.tolerance else None

    # instructor: one instructor's typical value differs from everyone else's (with two
    # instructors, "A vs the rest" and "B vs the rest" are the same finding: stated once)
    people = sorted({p for p in who.values() if p})
    for s in doc.steps:
        evs = per_step.get(s.id, {})
        for sig in profile.trigger_for(s.phase):
            t = tol(sig)
            vals = {rid: e.state[sig] for rid, e in evs.items() if sig in e.state}
            if t is None or len(people) < 2:
                continue
            for p in people[:1] if len(people) == 2 else people:
                mine = [v for rid, v in vals.items() if who.get(rid) == p]
                rest = [v for rid, v in vals.items() if who.get(rid) != p]
                if len(mine) < 2 or len(rest) < 2:
                    continue
                a, b = median(mine), median(rest)
                if abs(a - b) > t:
                    found.append(Pattern(
                        id="", kind="instructor", step=_where(s), runs=[r for r in vals if who.get(r) == p],
                        strength=abs(a - b) / t,
                        text=(f"{p} does \"{s.label}\" ({_phase(s.phase)}) at {_at(profile, sig, a)}, the "
                              f"others at {_fmt(profile, sig, b)} ({len(mine)} and {len(rest)} runs)."),
                        numbers=[_num(profile, sig, a), _num(profile, sig, b), len(mine), len(rest)]))

    # habit / together: technique steps, and who flies them
    technique = [s for s in doc.steps if not s.mandatory and s.run_count >= 3]
    did = {s.id: set(per_step.get(s.id, {})) for s in technique}
    for s in technique:
        runs_with = did[s.id]
        for p in people:
            mine = [r for r in who if who[r] == p]
            if len(mine) < 2:
                continue
            share_p = sum(r in runs_with for r in mine)
            others = [r for r in who if who[r] != p]
            share_o = sum(r in runs_with for r in others)
            if share_p == len(mine) and share_o == 0:
                found.append(Pattern(
                    id="", kind="habit", step=_where(s), runs=sorted(runs_with), strength=3.0 + len(mine),
                    text=(f"Only {p} does \"{s.label}\" ({_phase(s.phase)}): in {share_p} of {len(mine)} of "
                          f"their runs, and in {share_o} of {len(others)} runs by others."),
                    numbers=[share_p, len(mine), share_o, len(others)]))
    for i, a in enumerate(technique):
        for b in technique[i + 1:]:
            if len(did[a.id]) >= 2 and did[a.id] == did[b.id]:
                found.append(Pattern(
                    id="", kind="together", step=f"{_where(a)} + {_where(b)}", runs=sorted(did[a.id]),
                    strength=2.0 + len(did[a.id]),
                    text=(f"\"{a.label}\" ({_phase(a.phase)}) and \"{b.label}\" ({_phase(b.phase)}) always come "
                          f"together: the same {len(did[a.id])} runs do both."),
                    numbers=[len(did[a.id])]))

    # effect: runs with a technique do a later step differently
    order = {s.id: i for i, s in enumerate(doc.steps)}
    for tech in technique:
        with_ = did[tech.id]
        without = {r.run_id for r in runs} - with_
        if len(with_) < 2 or len(without) < 2:
            continue
        for s in doc.steps:
            if order[s.id] <= order[tech.id] or not s.mandatory:
                continue
            evs = per_step.get(s.id, {})
            for sig in profile.trigger_for(s.phase):
                t = tol(sig)
                a = [e.state[sig] for r, e in evs.items() if r in with_ and sig in e.state]
                b = [e.state[sig] for r, e in evs.items() if r in without and sig in e.state]
                if t is None or len(a) < 2 or len(b) < 2 or abs(median(a) - median(b)) <= t:
                    continue
                found.append(Pattern(
                    id="", kind="effect", step=f"{_where(tech)} → {_where(s)}", runs=sorted(with_),
                    strength=abs(median(a) - median(b)) / t,
                    text=(f"Runs with \"{tech.label}\" do \"{s.label}\" ({_phase(s.phase)}) at "
                          f"{_at(profile, sig, median(a))}, runs without at {_fmt(profile, sig, median(b))} "
                          f"({len(a)} and {len(b)} runs)."),
                    numbers=[_num(profile, sig, median(a)), _num(profile, sig, median(b)), len(a), len(b)]))

    # limit_break: who breaks which rule; rules broken by the same runs are one finding
    broke: dict[str, set[str]] = defaultdict(set)
    for f in doc.flags:
        if f.kind == "limit_violation" and f.run_id and "'" in f.message:
            broke[f.message.split("'")[1]].add(f.run_id)
    same_runs: dict[frozenset, list[str]] = defaultdict(list)
    for rule, rids in broke.items():
        same_runs[frozenset(rids)].append(rule)
    for rids, rules in same_runs.items():
        names = ", ".join(sorted({who.get(r) or r for r in rids}))
        what = (f"break \"{rules[0]}\"" if len(rules) == 1 else
                f"break {len(rules)} rules: " + "; ".join(f'"{r}"' for r in rules))
        found.append(Pattern(
            id="", kind="limit_break", runs=sorted(rids), strength=5.0,
            text=(f"{len(rids)} of {len(runs)} runs {what} — all by {names}; those runs are left out of the "
                  "typical ranges."),
            numbers=[len(rids), len(runs)] + ([len(rules)] if len(rules) > 1 else [])))

    # spread: where instructors agree least (typical range against the profile's tolerance)
    spreads = []
    for s in doc.steps:
        for name in profile.stated_facts(s.phase):   # only numbers the guideline states
            f = s.facts.get(name)
            t = tol(f.signal) if f else None
            if f and name.startswith("trigger.") and t and f.lo is not None and f.n >= 3:
                spreads.append(((f.hi - f.lo) / (2 * t), s, f))
    for ratio, s, f in sorted(spreads, key=lambda x: -x[0])[:2]:
        if ratio > 1:
            found.append(Pattern(
                id="", kind="spread", step=_where(s), strength=ratio,
                text=(f"Instructors agree least on \"{s.label}\" ({_phase(s.phase)}): {profile.label(f.signal)} "
                      f"ranges {f.fmt_range()} over {f.n} runs."),
                numbers=[_num(profile, f.signal, f.lo), _num(profile, f.signal, f.hi), f.n]))

    # Strongest first, a few of each kind so one kind does not crowd out the rest.
    cap = {"instructor": 4, "effect": 3, "spread": 2}
    kept, count = [], defaultdict(int)
    for p in sorted(found, key=lambda p: -p.strength):
        if count[p.kind] < cap.get(p.kind, top):
            kept.append(p)
            count[p.kind] += 1
    found = kept[:top]
    for i, p in enumerate(found, 1):
        p.id = f"P{i}"
    return found
