"""Hard limits from reference SOPs, applied to guideline steps.

For each step:
1. find the limits that apply, separately for its trigger and its result: same phase
   (if the limit names phases), `at` covering that snapshot, and the `when` condition
   true on that snapshot. The trigger snapshot includes the step's own switch change
   (selecting flaps 20 puts you under the flaps-20 Vfe, even though the flaps were still
   at 10 when you moved the lever);
2. runs whose values for this step break a limit are excluded from its typical range,
   as long as at least one compliant run remains;
3. whatever is left is clipped to the limit and flagged, so a guideline never states a
   value past a limit. The reference SOP always wins.

Only limits that matter for the step are shown beside it: conditional or phase-specific
limits, and global ones (e.g. Vne) only when the step's values come within 20 % of them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from sim2sop.schema import Event, Fact, Limit

WHERE = ("trigger", "result")
NEAR = 0.2
NAME_RE = re.compile(r"[A-Za-z_]\w*")


def _eval(expr: str, state: dict[str, float]) -> bool:
    try:
        return bool(eval(expr, {"__builtins__": {}, "abs": abs}, dict(state)))  # noqa: S307
    except Exception:
        return False  # signal not in the snapshot: the condition cannot be shown to hold


def snapshot(event: Event, where: str) -> dict[str, float]:
    if where == "result":
        return event.result
    state = dict(event.state)
    if event.kind == "toggle" and event.after is not None:
        state[event.signal] = event.after
    return state


@dataclass
class StepLimits:
    trigger: list[Limit] = field(default_factory=list)
    result: list[Limit] = field(default_factory=list)

    def at(self, where: str) -> list[Limit]:
        return self.trigger if where == "trigger" else self.result

    @property
    def all(self) -> list[Limit]:
        seen, out = set(), []
        for lim in self.trigger + self.result:
            if lim.id not in seen:
                seen.add(lim.id)
                out.append(lim)
        return out


def applicable(limits: list[Limit], phase: str, events: list[Event]) -> StepLimits:
    """Limits that apply to a step, judged on the median trigger / result state of its events.
    A condition that only becomes true after the step (flaps lowered just after a power
    change) belongs to the step that changed it, not to this one."""
    out = StepLimits()
    if not events:
        return out
    states = {}
    for where in WHERE:
        snaps = [snapshot(e, where) for e in events]
        keys = {k for s in snaps for k in s}
        states[where] = state = {k: float(np.median([s[k] for s in snaps if k in s])) for k in keys}
        for lim in limits:
            if lim.phases and phase not in lim.phases:
                continue
            if lim.at not in (where, "both"):
                continue
            if lim.when and not _eval(lim.when, state):
                continue
            if (where == "result" and lim.when and events[0].signal not in NAME_RE.findall(lim.when)
                    and not _eval(lim.when, states["trigger"])):
                continue
            out.at(where).append(lim)
    return out


def _outside(value: float | None, lim: Limit) -> bool:
    if value is None:
        return False
    return (lim.max is not None and value > lim.max) or (lim.min is not None and value < lim.min)


def event_violations(event: Event, limits: StepLimits) -> list[tuple[Limit, float]]:
    out = []
    for where in WHERE:
        for lim in limits.at(where):
            values = [snapshot(event, where).get(lim.signal)]
            if where == "trigger" and event.signal == lim.signal and event.kind != "toggle":
                values.append(event.after)
            bad = [v for v in values if _outside(v, lim)]
            if bad:
                out.append((lim, bad[0]))
    return out


def _where(fact_name: str) -> str:
    return "trigger" if fact_name == "action" else fact_name.split(".", 1)[0]


def clip_facts(facts: dict[str, Fact], limits: StepLimits) -> list[str]:
    """Attach limits to facts and clip value / range into them. Returns notes for values
    that had to move (a range end capped at a limit is expected and not noted)."""
    notes = []
    for name, f in facts.items():
        for lim in limits.at(_where(name)):
            if lim.signal != f.signal:
                continue
            if lim.max is not None:
                f.limit_max = lim.max if f.limit_max is None else min(f.limit_max, lim.max)
            if lim.min is not None:
                f.limit_min = lim.min if f.limit_min is None else max(f.limit_min, lim.min)
        lo_cap = f.limit_min if f.limit_min is not None else -np.inf
        hi_cap = f.limit_max if f.limit_max is not None else np.inf
        old = f.value
        f.value = float(np.clip(f.value, lo_cap, hi_cap))
        if f.lo is not None:
            f.lo = float(np.clip(f.lo, lo_cap, hi_cap))
        if f.hi is not None:
            f.hi = float(np.clip(f.hi, lo_cap, hi_cap))
        if f.value != old:
            f.clipped = True
            notes.append(f"{name}: {old:.{f.decimals}f} clipped to {f.fmt()}")
    return notes


def relevant(limits: StepLimits, facts: dict[str, Fact]) -> list[Limit]:
    """Limits worth showing beside the step."""
    out = []
    for lim in limits.all:
        if lim.when or lim.phases:
            out.append(lim)
            continue
        for name, f in facts.items():
            if f.signal != lim.signal or lim not in limits.at(_where(name)):
                continue
            hi = f.hi if f.hi is not None else f.value
            lo = f.lo if f.lo is not None else f.value
            if (lim.max is not None and hi >= (1 - NEAR) * lim.max) or \
               (lim.min is not None and lo <= (1 + NEAR) * lim.min):
                out.append(lim)
                break
    return out
