"""Verifier: every number in a guideline must be backed by telemetry and sit inside the
reference limits.

A number passes if it matches one of the step's facts (its typical value or either end
of its range, up to sensible rounding), a number in the profile's action vocabulary (the
step label, e.g. "Flaps 10°"), a number in a hard limit or the cited reference, or a run
count / settle time. A number that matches a fact but lies past that fact's hard limit
fails: guidelines never cross a limit.
Numbers in the instructor's callout (intent) are checked too, but a mismatch there is a
warning on the step, not a failure: the instructor said it, the data disagrees.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from sim2sop.learn.memory import matching_part
from sim2sop.profiles import Profile
from sim2sop.schema import Step

NUM_RE = re.compile(r"(?<![A-Za-z0-9])-?\d+(?:\.\d+)?")
CHECKED_FIELDS = ("trigger", "action", "typical", "limit")


@dataclass
class Verdict:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _numbers(text: str | None) -> list[float]:
    return [float(m) for m in NUM_RE.findall(text or "")]


def _check(x: float, step: Step, exact: set[float]) -> str | None:
    """None if supported; otherwise the reason it is not."""
    if any(abs(abs(x) - e) < 1e-9 for e in exact):
        return None
    for f in step.facts.values():
        if matching_part(x, f) is None:
            continue
        if f.limit_max is not None and x > f.limit_max + 1e-9:
            return f"{x:g} is past the hard limit (max {f.limit_max:g})"
        if f.limit_min is not None and x < f.limit_min - 1e-9:
            return f"{x:g} is past the hard limit (min {f.limit_min:g})"
        return None
    return f"{x:g} is not supported by the telemetry for this step"


def verify_step(step: Step, profile: Profile) -> Verdict:
    exact = {abs(v) for v in _numbers(step.label)}
    exact |= {abs(v) for v in _numbers(step.reference.text if step.reference else "")}
    for lim in step.limits:
        exact |= {abs(v) for v in _numbers(f"{lim.text} {lim.cite()} {lim.describe()}")}
    exact |= {float(step.run_count), float(step.seen_in), float(len(step.runs)),
              float(profile.events.settle_s)}
    v = Verdict()
    for name in CHECKED_FIELDS:
        for x in _numbers(getattr(step, name)):
            problem = _check(x, step, exact)
            if problem:
                v.errors.append(f"{name}: {problem}")
    for x in _numbers(step.intent):
        if _check(x, step, exact):
            v.warnings.append(f"Callout says {x:g}, which the telemetry for this step does not show")
    return v


def facts_table(step: Step) -> dict[str, str]:
    """Facts as the writer sees them: typical value, range, run count and any hard limit."""
    out = {}
    for name, f in step.facts.items():
        lim = []
        if f.limit_min is not None:
            lim.append(f"min {f.fmt(f.limit_min)}")
        if f.limit_max is not None:
            lim.append(f"max {f.fmt(f.limit_max)}")
        out[name] = (f"typical {f.fmt()} (range {f.fmt_range()}, n={f.n})"
                     + (f"; HARD LIMIT {' '.join(lim)}" if lim else ""))
    return out
