"""Event extraction: switch toggles, threshold crossings, control inputs and turns become
a compact event timeline. Each event carries a context snapshot at the trigger and after
settling, which later become the only numbers the SOP is allowed to state."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sim2sop.profiles import Profile
from sim2sop.schema import Event, Phase
from sim2sop.signals.phases import phase_at
from sim2sop.telemetry import heading_rate, snapshot


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of contiguous True runs."""
    edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _toggles(df: pd.DataFrame, profile: Profile, debounce_s: float = 0.5) -> list[dict]:
    out = []
    hold = max(1, int(debounce_s * profile.sample_hz))
    for sig, spec in profile.events.toggles.items():
        if sig not in df:
            continue
        v = df[sig].round().to_numpy()
        current = v[0]
        i = 1
        while i < len(v):
            if v[i] != current and np.all(v[i:i + hold] == v[i]):
                after = int(v[i])
                state = spec.states.get(after, str(after))
                out.append(dict(t=float(df["t"].iloc[i]), kind="toggle", signal=sig,
                                label=f"{spec.name} {state}".strip(), before=float(current),
                                after=float(after), core=f"toggle:{sig}:{after}"))
                current = v[i]
            i += 1
    return out


def _thresholds(df: pd.DataFrame, profile: Profile, phases: list[Phase]) -> list[dict]:
    out = []
    t = df["t"].to_numpy()
    for spec in profile.events.thresholds:
        if spec.signal not in df:
            continue
        x = df[spec.signal].to_numpy()
        mask = x > spec.above if spec.above is not None else x < spec.below
        hold = max(1, int(spec.hold_s * profile.sample_hz))
        level = spec.above if spec.above is not None else spec.below
        direction = "above" if spec.above is not None else "below"
        for s, e in _segments(mask):
            if s == 0 or e - s < hold:
                continue  # already true at start, or a blip
            ph = phase_at(phases, float(t[s]))
            if spec.phases and ph not in spec.phases:
                continue
            out.append(dict(t=float(t[s]), kind="threshold", signal=spec.signal, label=spec.label,
                            before=None, after=float(level), unit=profile.unit(spec.signal),
                            core=f"threshold:{spec.signal}:{direction}:{level:g}"))
            if spec.once:
                break
    return out


def _controls(df: pd.DataFrame, profile: Profile) -> list[dict]:
    out = []
    hz = profile.sample_hz
    t = df["t"].to_numpy()
    for spec in profile.events.controls:
        if spec.signal not in df:
            continue
        x = df[spec.signal].rolling(max(1, int(hz)), center=True, min_periods=1).median().to_numpy()
        rate = np.abs(np.gradient(x, t))
        moving = rate > spec.min_step / spec.settle_s
        # Merge movements separated by short pauses into one input.
        gap = int(1.0 * hz)
        segs = _segments(moving)
        merged: list[list[int]] = []
        for s, e in segs:
            if merged and s - merged[-1][1] <= gap:
                merged[-1][1] = e
            else:
                merged.append([s, e])
        w = int(2 * hz)
        for s, e in merged:
            before = float(np.median(x[max(0, s - w):s + 1]))
            after = float(np.median(x[e - 1:min(len(x), e + w)]))
            if abs(after - before) < spec.min_step:
                continue
            verb = "increase" if after > before else "decrease"
            template = f"{'Increase' if verb == 'increase' else 'Reduce'} {spec.name.lower()} to {{value}}"
            out.append(dict(t=float(t[s]), kind="control", signal=spec.signal,
                            label=template.format(value=f"{after:.0f}{profile.unit(spec.signal) or ''}"),
                            template=template, before=before, after=after,
                            unit=profile.unit(spec.signal), core=f"control:{spec.signal}:{verb}"))
    return out


def _turns(df: pd.DataFrame, profile: Profile, phases: list[Phase]) -> list[dict]:
    spec = profile.events.turns
    if spec is None or spec.signal not in df:
        return []
    hz = profile.sample_hz
    t = df["t"].to_numpy()
    rate = heading_rate(df, spec.signal, smooth_s=2.0, hz=hz)
    hdg = df[spec.signal].to_numpy()
    out = []
    for sign in (1, -1):
        for s, e in _segments(sign * rate > spec.min_rate_dps):
            if (e - s) / hz < spec.min_duration_s:
                continue
            change = float(np.sum(rate[s:e]) / hz)
            if abs(change) < spec.min_change_deg:
                continue
            direction = "right" if sign > 0 else "left"
            after = float(hdg[min(e, len(hdg) - 1)])
            t_end = float(t[min(e, len(t) - 1)])
            out.append(dict(t=float(t[s]), kind="turn", signal=spec.signal,
                            label=f"Turn {direction} to heading {after:03.0f}°",
                            template=f"Turn {direction} to heading {{value}}",
                            before=float(hdg[s]), after=after, unit="°", t_end=t_end,
                            # A turn belongs to the leg it rolls out on.
                            phase=phase_at(phases, t_end + 1.0), core=f"turn:{direction}"))
    return out


def extract_events(df: pd.DataFrame, profile: Profile, phases: list[Phase],
                   phase_lookahead_s: float = 2.0) -> list[Event]:
    """Events sorted by time. An action that starts a phase (full power -> take-off roll)
    belongs to that phase, so the phase is looked up slightly after the event."""
    raw = _toggles(df, profile) + _thresholds(df, profile, phases) + _controls(df, profile) \
        + _turns(df, profile, phases)
    raw.sort(key=lambda r: r["t"])
    settle = profile.events.settle_s
    events = []
    for i, r in enumerate(raw):
        phase = r.pop("phase", None) or phase_at(phases, r["t"] + phase_lookahead_s)
        key = r.pop("core")
        t_result = r.pop("t_end", r["t"]) + (2.0 if r["kind"] == "turn" else settle)
        events.append(Event(
            id=f"e{i:03d}", phase=phase, key=key,
            unit=r.pop("unit", None) or profile.unit(r["signal"]),
            state=snapshot(df, r["t"], profile.context),
            result=snapshot(df, t_result, profile.context),
            **r,
        ))
    return events
