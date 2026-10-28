"""Compliance flags: instructor actions that break a reference limit (e.g. flaps above Vfe)."""

from __future__ import annotations

import pandas as pd

from sim2sop.profiles import Profile
from sim2sop.schema import Flag
from sim2sop.signals.events import _segments


def check_limits(df: pd.DataFrame, profile: Profile, run_id: str | None = None,
                 min_duration_s: float = 1.0) -> list[Flag]:
    flags = []
    t = df["t"].to_numpy()
    for limit in profile.limits:
        try:
            mask = df.eval(limit.when).to_numpy(dtype=bool)
        except Exception as e:  # a bad expression should not stop ingest
            flags.append(Flag(kind="compliance", run_id=run_id,
                              message=f"Could not evaluate limit {limit.when!r}: {e}"))
            continue
        for s, e in _segments(mask):
            if (e - s) / profile.sample_hz >= min_duration_s:
                ref = f" ({limit.reference})" if limit.reference else ""
                flags.append(Flag(kind="compliance", t=float(t[s]), run_id=run_id,
                                  message=f"{limit.message}{ref} for {(e - s) / profile.sample_hz:.1f}s"))
    return flags
