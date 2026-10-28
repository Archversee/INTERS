"""Domain-independent telemetry utilities: validation, resampling, derived signals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sim2sop.profiles import Profile
from sim2sop.schema import SignalCategory


def validate(df: pd.DataFrame, profile: Profile) -> list[str]:
    """Return warnings for missing signals; raise if there is no time column."""
    if "t" not in df.columns:
        raise ValueError("telemetry must have a 't' column (seconds from run start)")
    missing = [s for s in profile.signals if s not in df.columns]
    return [f"signal {s!r} missing from telemetry" for s in missing]


def resample(df: pd.DataFrame, profile: Profile) -> pd.DataFrame:
    """Put the run on a uniform clock. Continuous signals are interpolated (angles via
    unwrap), discrete ones hold their last value."""
    df = df.sort_values("t").drop_duplicates("t")
    t0 = float(df["t"].iloc[0])
    t = df["t"].to_numpy() - t0
    step = 1.0 / profile.sample_hz
    grid = np.round(np.arange(0.0, t[-1] + 1e-9, step), 6)
    out = {"t": grid}
    for col in df.columns:
        if col == "t" or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        y = df[col].to_numpy(dtype=float)
        spec = profile.signals.get(col)
        if spec is not None and spec.category == SignalCategory.discrete:
            idx = np.clip(np.searchsorted(t, grid, side="right") - 1, 0, len(t) - 1)
            out[col] = y[idx]
        elif spec is not None and spec.angle:
            out[col] = np.degrees(np.interp(grid, t, np.unwrap(np.radians(y)))) % 360.0
        else:
            out[col] = np.interp(grid, t, y)
    return pd.DataFrame(out)


def wrap180(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def add_derived(df: pd.DataFrame, profile: Profile) -> pd.DataFrame:
    df = df.copy()
    for spec in profile.derived.relative_heading:
        mask = df.eval(spec.anchor).to_numpy(dtype=bool)
        if not mask.any():
            ref = float(df[spec.signal].iloc[0])
        else:
            i0 = int(np.argmax(mask))
            t0 = df["t"].iloc[i0]
            window = df[(df["t"] >= t0) & (df["t"] <= t0 + spec.anchor_window_s)][spec.signal]
            ref = _circular_mean(window.to_numpy())
        df[spec.name] = wrap180(df[spec.signal] - ref)
        df.attrs[f"{spec.name}_ref"] = ref
    return df


def heading_rate(df: pd.DataFrame, signal: str, smooth_s: float, hz: float) -> np.ndarray:
    """Degrees per second, unwrapped and smoothed."""
    h = np.degrees(np.unwrap(np.radians(df[signal].to_numpy(dtype=float))))
    rate = np.gradient(h, df["t"].to_numpy())
    win = max(1, int(smooth_s * hz))
    return pd.Series(rate).rolling(win, center=True, min_periods=1).mean().to_numpy()


def snapshot(df: pd.DataFrame, t: float, signals: list[str], half_window_s: float = 0.5) -> dict[str, float]:
    """Median value of each signal around t (robust to sensor noise)."""
    win = df[(df["t"] >= t - half_window_s) & (df["t"] <= t + half_window_s)]
    if win.empty:
        win = df.iloc[[int(np.argmin(np.abs(df["t"].to_numpy() - t)))]]
    return {s: float(win[s].median()) for s in signals if s in win}


def _circular_mean(deg: np.ndarray) -> float:
    r = np.radians(deg)
    return float(np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())) % 360.0)
