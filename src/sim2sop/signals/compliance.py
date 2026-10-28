"""Limit-violation report: instructor runs that break a reference hard limit (e.g. flaps
above Vfe), found on the full telemetry, for the instructor to review.

A limit with `environment` set (e.g. a NOTAM for a new obstacle) only checks runs from
that environment. Violating values are kept out of the typical ranges by the planner.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sim2sop.profiles import Profile
from sim2sop.schema import Flag, Limit
from sim2sop.signals.events import _segments
from sim2sop.storage import RunRecord, load_telemetry


def _phase_column(df: pd.DataFrame, record: RunRecord) -> np.ndarray:
    phase = np.full(len(df), "", dtype=object)
    t = df["t"].to_numpy()
    for p in record.phases:
        phase[(t >= p.t_start) & (t < p.t_end)] = p.name
    if record.phases:
        phase[t >= record.phases[-1].t_end] = record.phases[-1].name
    return phase


def check_run(df: pd.DataFrame, record: RunRecord, limits: list[Limit], profile: Profile,
              min_duration_s: float = 1.0) -> list[Flag]:
    flags = []
    t = df["t"].to_numpy()
    phase = _phase_column(df, record)
    for lim in limits:
        if lim.environment and lim.environment != record.meta.environment:
            continue
        if lim.signal not in df:
            continue
        x = df[lim.signal].to_numpy()
        mask = np.zeros(len(df), dtype=bool)
        if lim.max is not None:
            mask |= x > lim.max
        if lim.min is not None:
            mask |= x < lim.min
        if lim.when:
            try:
                mask &= df.eval(lim.when).to_numpy(dtype=bool)
            except Exception:
                continue  # condition uses a signal this run does not have
        if lim.phases:
            mask &= np.isin(phase, lim.phases)
        for s, e in _segments(mask):
            dur = (e - s) / profile.sample_hz
            if dur < min_duration_s:
                continue
            worst = x[s:e].max() if lim.max is not None and x[s:e].max() > lim.max else x[s:e].min()
            flags.append(Flag(
                kind="limit_violation", t=float(t[s]), run_id=record.run_id, phase=str(phase[s]),
                message=(f"{record.run_id} at {t[s]:.0f}s ({phase[s]}): {profile.label(lim.signal)} "
                         f"{worst:.{profile.signals[lim.signal].decimals if lim.signal in profile.signals else 0}f} "
                         f"for {dur:.1f}s breaks '{lim.text}' ({lim.cite()})")))
    return flags


def violation_report(runs: list[RunRecord], limits: list[Limit], profile: Profile) -> list[Flag]:
    if not limits:
        return []
    flags = []
    for r in runs:
        flags += check_run(load_telemetry(r.meta.procedure, r.run_id), r, limits, profile)
    return flags
