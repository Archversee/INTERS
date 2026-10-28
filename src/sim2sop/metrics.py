"""Demo metrics: verifier pass rate, grounding, band width, confidence, edits per version."""

from __future__ import annotations

import numpy as np

from sim2sop.learn import memory
from sim2sop.profiles import Profile
from sim2sop.schema import SOP


def compute(sop: SOP, profile: Profile) -> dict[str, float]:
    steps = sop.steps
    n = max(len(steps), 1)
    ratios = []
    for s in steps:
        for name, f in s.facts.items():
            spec = profile.signals.get(f.signal)
            if name.startswith("trigger.") and spec and spec.tolerance and f.tol is not None:
                ratios.append(f.tol / spec.tolerance)
    return {
        "steps": float(len(steps)),
        "runs": float(len(sop.runs)),
        "verifier_pass_rate": round(sum(s.verified for s in steps) / n, 3),
        "grounded_pct": round(sum(s.reference is not None for s in steps) / n, 3),
        # Mean band width relative to the profile default (1.0 = single-run default).
        "band_width_ratio": round(float(np.mean(ratios)), 3) if ratios else 1.0,
        "mean_confidence": round(float(np.mean([s.confidence for s in steps])), 3) if steps else 0.0,
        "mandatory_steps": float(sum(s.mandatory for s in steps)),
        "technique_steps": float(sum(not s.mandatory for s in steps)),
        "flags": float(len(sop.flags)),
    }


def edits_per_version(procedure: str) -> dict[int, int]:
    counts: dict[int, int] = {}
    for c in memory.all_corrections():
        if c.procedure == procedure:
            counts[c.sop_version] = counts.get(c.sop_version, 0) + 1
    return counts


def band_for(sop: SOP, step_key_prefix: str, fact: str) -> tuple[float, float] | None:
    """(value, band) of one fact on the first step whose key starts with a prefix."""
    for s in sop.steps:
        if s.key.startswith(step_key_prefix) and fact in s.facts:
            f = s.facts[fact]
            return f.value, f.tol or 0.0
    return None
