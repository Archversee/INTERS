"""Demo metrics: verifier pass rate, values within limits, range width, confidence,
guidelines changed by an update, edits per version."""

from __future__ import annotations

import numpy as np

from sim2sop.learn import memory
from sim2sop.profiles import Profile
from sim2sop.schema import Guidelines


def compute(doc: Guidelines, profile: Profile) -> dict[str, float]:
    steps = doc.steps
    n = max(len(steps), 1)
    ratios, limited, inside, clipped = [], 0, 0, 0
    for s in steps:
        for name, f in s.facts.items():
            spec = profile.signals.get(f.signal)
            if name.startswith("trigger.") and spec and spec.tolerance and f.lo is not None:
                ratios.append(f.width / (2 * spec.tolerance))
            if f.limit_min is not None or f.limit_max is not None:
                limited += 1
                lo_ok = f.limit_min is None or min(f.value, f.lo if f.lo is not None else f.value) >= f.limit_min - 1e-9
                hi_ok = f.limit_max is None or max(f.value, f.hi if f.hi is not None else f.value) <= f.limit_max + 1e-9
                inside += lo_ok and hi_ok
                clipped += f.clipped
    kinds = [f.kind for f in doc.flags]
    return {
        "steps": float(len(steps)),
        "runs": float(len(doc.runs)),
        "verifier_pass_rate": round(sum(s.verified for s in steps) / n, 3),
        # Share of limited values inside their hard limits (1.0 = all; -1 = unchecked).
        "within_limits": round(inside / limited, 3) if limited else (1.0 if doc.checked else -1.0),
        "limited_values": float(limited),
        "clipped_values": float(clipped),
        "limit_violations": float(kinds.count("limit_violation")),
        "excluded_values": float(kinds.count("excluded")),
        "cited_pct": round(sum(s.reference is not None for s in steps) / n, 3),
        # Mean typical-range width relative to the profile default (1.0 = single-run default).
        "range_width_ratio": round(float(np.mean(ratios)), 3) if ratios else 1.0,
        "mean_confidence": round(float(np.mean([s.confidence for s in steps])), 3) if steps else 0.0,
        "core_steps": float(sum(s.mandatory for s in steps)),
        "technique_steps": float(sum(not s.mandatory for s in steps)),
        "steps_changed": float(sum(s.change is not None for s in steps) + len(doc.removed)),
        "flags": float(len(doc.flags)),
    }


def edits_per_version(procedure: str) -> dict[int, int]:
    counts: dict[int, int] = {}
    for c in memory.all_corrections():
        if c.procedure == procedure:
            counts[c.version] = counts.get(c.version, 0) + 1
    return counts


def fact_for(doc: Guidelines, step_key_prefix: str, fact: str):
    """The fact of the first step whose key starts with a prefix (for demo printouts)."""
    for s in doc.steps:
        if s.key.startswith(step_key_prefix) and fact in s.facts:
            return s.facts[fact]
    return None
