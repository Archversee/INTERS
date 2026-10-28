"""Phase segmentation.

The profile's ordered phase rules name the phases: the segmenter walks forward and enters
the next phase once its `enter` expression has held for `hold_s`. Change-point detection
(ruptures) then proposes boundaries from the raw signals, and each rule boundary snaps to
a change point close by, so boundaries sit where the data actually changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sim2sop.profiles import Profile
from sim2sop.schema import Phase


def _first_held(mask: np.ndarray, start: int, hold: int) -> int | None:
    """First index >= start where mask stays true for `hold` samples."""
    run = 0
    for i in range(start, len(mask)):
        run = run + 1 if mask[i] else 0
        if run >= hold:
            return i - hold + 1
    return None


def rule_boundaries(df: pd.DataFrame, profile: Profile) -> list[tuple[str, int]]:
    hz = profile.sample_hz
    out = [(profile.phases[0].name, 0)]
    cursor = 0
    for spec in profile.phases[1:]:
        if not spec.enter:
            continue
        mask = df.eval(spec.enter).to_numpy(dtype=bool)
        idx = _first_held(mask, cursor + 1, max(1, int(spec.hold_s * hz)))
        if idx is None:
            continue  # phase not flown in this run; try the next one from the same point
        out.append((spec.name, idx))
        cursor = idx
    return out


def change_points(df: pd.DataFrame, profile: Profile, penalty: float = 8.0) -> list[float]:
    """Candidate boundary times from multivariate change-point detection (1 Hz, PELT, L2)."""
    cols = [c for c in profile.segmentation_signals if c in df]
    if not cols:
        return []
    import ruptures as rpt

    step = max(1, int(profile.sample_hz))
    x = df[cols].to_numpy(dtype=float)[::step]
    x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-9)
    bkps = rpt.Pelt(model="l2", min_size=3).fit(x).predict(pen=penalty)
    t = df["t"].to_numpy()[::step]
    return [float(t[b]) for b in bkps[:-1]]


def segment(df: pd.DataFrame, profile: Profile, snap_s: float = 4.0) -> list[Phase]:
    bounds = rule_boundaries(df, profile)
    t = df["t"].to_numpy()
    cps = np.array(change_points(df, profile)) if snap_s > 0 else np.array([])
    starts = []
    for name, idx in bounds:
        ts = float(t[idx])
        if idx > 0 and cps.size:
            near = cps[np.abs(cps - ts) <= snap_s]
            if near.size:
                ts = float(near[np.argmin(np.abs(near - ts))])
        starts.append((name, ts))
    # Keep boundaries strictly increasing after snapping.
    for i in range(1, len(starts)):
        if starts[i][1] <= starts[i - 1][1]:
            starts[i] = (starts[i][0], float(t[bounds[i][1]]))
    phases = []
    for i, (name, ts) in enumerate(starts):
        te = starts[i + 1][1] if i + 1 < len(starts) else float(t[-1])
        phases.append(Phase(name=name, t_start=round(ts, 2), t_end=round(te, 2)))
    return phases


def phase_at(phases: list[Phase], t: float) -> str | None:
    for p in phases:
        if p.t_start <= t < p.t_end:
            return p.name
    return phases[-1].name if phases and t >= phases[-1].t_start else None
