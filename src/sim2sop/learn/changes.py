"""Per-phase change detection for update mode.

Runs tagged with a new environment (e.g. "new-obstacle") are compared with the current
guidelines. A phase is a technique change when at least `profile.change_min_runs` of
those runs ALL fall outside the current typical range of a value the guidelines state,
on the same side, and their median moved by more than the signal's default tolerance. Fewer runs, or
runs that disagree with each other, are outliers: reported, not learned.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from sim2sop.learn.consensus import ConsensusItem, build_consensus
from sim2sop.profiles import Profile
from sim2sop.schema import Event, Flag, Guidelines, Step
from sim2sop.storage import RunRecord


@dataclass
class PhaseChange:
    phase: str
    shifts: list[str] = field(default_factory=list)


@dataclass
class ChangeReport:
    environment: str | None = None
    post_runs: list[str] = field(default_factory=list)
    changed: dict[str, PhaseChange] = field(default_factory=dict)
    pending: bool = False           # new-environment runs, but not enough to decide
    flags: list[Flag] = field(default_factory=list)

    @property
    def confirmed(self) -> bool:
        return bool(self.changed)


def slot(phase: str, key: str, occurrence: int) -> str:
    """Stable identity of a step across versions (keys can repeat within a phase)."""
    return f"{phase}|{key}|{occurrence}"


def step_slots(steps: list[Step]) -> dict[str, Step]:
    seen: dict[tuple[str, str], int] = defaultdict(int)
    out = {}
    for s in steps:
        n = seen[(s.phase, s.key)]
        seen[(s.phase, s.key)] += 1
        out[slot(s.phase, s.key, n)] = s
    return out


def _item_slots(items: list[ConsensusItem]) -> dict[str, ConsensusItem]:
    seen: dict[tuple[str, str], int] = defaultdict(int)
    out = {}
    for it in items:
        n = seen[(it.phase, it.key)]
        seen[(it.phase, it.key)] += 1
        out[slot(it.phase, it.key, n)] = it
    return out


def _value(e: Event, name: str) -> float | None:
    if name == "action":
        return e.after
    where, sig = name.split(".", 1)
    return (e.state if where == "trigger" else e.result).get(sig)


def detect_changes(previous: Guidelines, runs: list[RunRecord], profile: Profile) -> ChangeReport:
    report = ChangeReport()
    candidates = [r for r in runs if r.meta.environment != previous.environment]
    if not candidates:
        return report
    env = candidates[-1].meta.environment  # newest environment tag
    post = [r for r in candidates if r.meta.environment == env]
    report.environment, report.post_runs = env, [r.run_id for r in post]
    need = profile.change_min_runs
    if len(post) < need:
        report.pending = True
        report.flags.append(Flag(kind="technique_change", message=(
            f"{len(post)} run(s) tagged '{env}' — need {need} consistent runs before treating "
            f"differences as a technique change; held out of learning for now")))
        return report

    items, _ = build_consensus(post)
    prev_slots = step_slots(previous.steps)
    for key, item in _item_slots(items).items():
        prev = prev_slots.get(key)
        if prev is None or item.seen_in < need:
            continue
        for name in profile.stated_facts(prev.phase):
            f = prev.facts.get(name)
            if f is None:
                continue
            spec = profile.signals.get(f.signal)
            if spec is None or spec.angle or not spec.tolerance or f.lo is None:
                continue
            vals = [v for v in (_value(e, name) for e in item.events.values()) if v is not None]
            if len(vals) < need:
                continue
            above, below = all(v > f.hi for v in vals), all(v < f.lo for v in vals)
            shift = float(np.median(vals)) - f.value
            if (above or below) and abs(shift) > spec.tolerance:
                change = report.changed.setdefault(prev.phase, PhaseChange(prev.phase))
                change.shifts.append(
                    f"{prev.label}: {profile.label(f.signal)} {f.fmt_range()} → "
                    f"{f.fmt(float(np.median(vals)))} in {len(vals)}/{len(post)} '{env}' runs")
    for ph in report.changed.values():
        report.flags.append(Flag(kind="technique_change", phase=ph.phase, message=(
            f"Technique change in {ph.phase} after '{env}': " + "; ".join(ph.shifts[:3])
            + (f" (+{len(ph.shifts) - 3} more)" if len(ph.shifts) > 3 else ""))))
    if not report.changed:
        report.flags.append(Flag(kind="technique_change", message=(
            f"{len(post)} runs tagged '{env}' match the current guidelines; added as normal runs")))
    return report
