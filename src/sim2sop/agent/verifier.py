"""Verifier: every number in a step must be backed by telemetry.

A number passes if it matches one of the step's facts (value or tolerance band, up to
sensible rounding), a number in the profile's action vocabulary (the step label, e.g.
"Flaps 10°"), a number in the cited reference text, or a run count / settle time.
Numbers in the instructor's callout (intent) are checked too, but a mismatch there is a
warning on the step, not a failure: the instructor said it, the data disagrees.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from sim2sop.learn.memory import number_matches
from sim2sop.profiles import Profile
from sim2sop.schema import Fact, Step

NUM_RE = re.compile(r"(?<![A-Za-z])-?\d+(?:\.\d+)?")
CHECKED_FIELDS = ("trigger", "action", "expected", "tolerance")


@dataclass
class Verdict:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _numbers(text: str | None) -> list[float]:
    return [float(m) for m in NUM_RE.findall(text or "")]


def _supported(x: float, step: Step, exact: set[float]) -> bool:
    if any(abs(abs(x) - e) < 1e-9 for e in exact):
        return True
    for f in step.facts.values():
        if number_matches(x, f):
            return True
        if f.tol is not None and number_matches(x, Fact(signal=f.signal, value=f.tol, decimals=f.decimals)):
            return True
    return False


def verify_step(step: Step, profile: Profile) -> Verdict:
    exact = {abs(v) for v in _numbers(step.label)}
    exact |= {abs(v) for v in _numbers(step.reference.text if step.reference else "")}
    exact |= {float(step.run_count), float(step.seen_in), float(profile.events.settle_s)}
    v = Verdict()
    for name in CHECKED_FIELDS:
        for x in _numbers(getattr(step, name)):
            if not _supported(x, step, exact):
                v.errors.append(f"{name}: {x:g} is not supported by the telemetry for this step")
    for x in _numbers(step.intent):
        if not _supported(x, step, exact):
            v.warnings.append(f"Callout says {x:g}, which the telemetry for this step does not show")
    return v


def facts_table(step: Step) -> dict[str, str]:
    """Facts as the writer sees them: formatted value with band and run count."""
    return {name: f"{f.fmt()} ({f.fmt_tol()}, n={f.n})" for name, f in step.facts.items()}
