"""Multi-run consensus.

Runs are aligned to an anchor run (the medoid) with DTW over their event sequences.
Matching event keys cost nothing, matching intent annotations act as anchor points
(negative cost), and normalised time breaks ties between repeated keys. Then:

- an event matched in every run is mandatory;
- an event seen in only some runs becomes a technique note;
- numeric values become typical ranges: median ± a robust spread that tightens (or
  widens) as runs are added. Hard limits are applied afterwards (sim2sop.learn.limits).

Half-width blends the profile's default tolerance d (a prior worth one run) with 2σ,
where σ is robust (IQR / 1.349 from 4 runs, MAD otherwise), so one bad run cannot
widen a range on its own: half = (d + (n-1)·2σ) / n, floored at 20 % of d.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from sim2sop.profiles import Profile
from sim2sop.schema import Event, Fact
from sim2sop.storage import RunRecord

ANNOTATION_ANCHOR_BONUS = 0.5
TIME_WEIGHT = 2.0
MERGE_WINDOW_FRAC = 0.03   # same key within 3 % of the run length = same step, different timing


@dataclass
class ConsensusItem:
    key: str
    phase: str
    kind: str
    signal: str
    label: str
    template: str | None
    t_anchor: float                                     # time in the anchor run
    events: dict[str, Event] = field(default_factory=dict)   # run_id -> event
    run_count: int = 1

    @property
    def seen_in(self) -> int:
        return len(self.events)

    @property
    def mandatory(self) -> bool:
        return self.seen_in == self.run_count

    @property
    def intent(self) -> str | None:
        notes = [e.annotation for e in self.events.values() if e.annotation]
        return Counter(notes).most_common(1)[0][0] if notes else None

    @property
    def intents(self) -> list[str]:
        return sorted({e.annotation for e in self.events.values() if e.annotation})


def _norm(text: str | None) -> str:
    return re.sub(r"[^a-z0-9 ]", "", (text or "").lower()).strip()


def _cost(a: Event, b: Event, ta: float, tb: float) -> float:
    if a.key == b.key:
        c = 0.0
    elif a.kind == b.kind and a.signal == b.signal:
        c = 1.0
    else:
        c = 2.0
    if a.annotation and b.annotation and _norm(a.annotation) == _norm(b.annotation):
        c -= ANNOTATION_ANCHOR_BONUS
    return c + TIME_WEIGHT * abs(ta - tb)


def dtw_align(a: list[Event], b: list[Event], dur_a: float, dur_b: float) -> tuple[float, list[tuple[int, int]]]:
    """Classic DTW; returns total cost and the warping path as (i, j) index pairs."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("inf"), []
    ta = [e.t / max(dur_a, 1e-9) for e in a]
    tb = [e.t / max(dur_b, 1e-9) for e in b]
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = _cost(a[i - 1], b[j - 1], ta[i - 1], tb[j - 1]) + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    path, i, j = [], n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = int(np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]]))
        i, j = (i - 1, j - 1) if step == 0 else (i - 1, j) if step == 1 else (i, j - 1)
    return float(D[n, m]), path[::-1]


def _medoid(runs: list[RunRecord]) -> int:
    if len(runs) <= 2:
        return len(runs) - 1  # latest run anchors small sets
    totals = []
    for r in runs:
        totals.append(sum(dtw_align(r.events, o.events, r.meta.duration_s, o.meta.duration_s)[0]
                          for o in runs if o is not r))
    return int(np.argmin(totals))


def build_consensus(runs: list[RunRecord]) -> tuple[list[ConsensusItem], RunRecord]:
    """Align all runs to the medoid run and return consensus items in anchor order."""
    anchor = runs[_medoid(runs)]
    n = len(runs)
    items = [ConsensusItem(key=e.key, phase=e.phase or "", kind=e.kind, signal=e.signal,
                           label=e.label, template=e.template, t_anchor=e.t,
                           events={anchor.run_id: e}, run_count=n)
             for e in anchor.events]

    extras: list[ConsensusItem] = []
    window = MERGE_WINDOW_FRAC * anchor.meta.duration_s
    for run in runs:
        if run is anchor:
            continue
        _, path = dtw_align(anchor.events, run.events, anchor.meta.duration_s, run.meta.duration_s)
        used: set[int] = set()
        # For each anchor event, the best same-key partner on the warping path.
        by_anchor: dict[int, list[int]] = {}
        for i, j in path:
            by_anchor.setdefault(i, []).append(j)
        for i, js in by_anchor.items():
            same = [j for j in js if run.events[j].key == anchor.events[i].key and j not in used]
            if same:
                j = min(same, key=lambda j: abs(run.events[j].t / run.meta.duration_s
                                                - anchor.events[i].t / anchor.meta.duration_s))
                items[i].events[run.run_id] = run.events[j]
                used.add(j)
        # Unmatched events: same action at a slightly different time (DTW paired it with a
        # neighbour), or a technique step the anchor did not fly.
        nearest = {j: i for i, j in path}
        for j, ev in enumerate(run.events):
            if j in used:
                continue
            slot = nearest.get(j, len(items) - 1)
            t_anchor = items[slot].t_anchor if items else ev.t
            home = min((it for it in items + extras
                        if it.key == ev.key and run.run_id not in it.events
                        and abs(it.t_anchor - t_anchor) <= window),
                       key=lambda it: abs(it.t_anchor - t_anchor), default=None)
            if home is None:
                home = ConsensusItem(key=ev.key, phase=ev.phase or "", kind=ev.kind,
                                     signal=ev.signal, label=ev.label, template=ev.template,
                                     t_anchor=t_anchor + 0.01, run_count=n)
                extras.append(home)
            home.events[run.run_id] = ev

    all_items = items + extras
    all_items.sort(key=lambda it: it.t_anchor)
    return all_items, anchor


# --- typical ranges -------------------------------------------------------------------------

def typical_range(values: list[float], default_tol: float | None,
                  angle: bool = False) -> tuple[float, float, float]:
    """Return (median, lo, hi) of the values seen across runs."""
    x = np.asarray(values, dtype=float)
    n = len(x)
    if angle:
        r = np.radians(x)
        centre = float(np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())) % 360.0)
        dev = np.abs((x - centre + 180.0) % 360.0 - 180.0)
        sigma = float(np.median(dev) * 1.4826) if n > 1 else 0.0
    else:
        centre = float(np.median(x))
        if n >= 4:
            q1, q3 = np.percentile(x, [25, 75])
            sigma = float(q3 - q1) / 1.349
        elif n > 1:
            sigma = float(np.median(np.abs(x - centre)) * 1.4826)
        else:
            sigma = 0.0
    if default_tol is None:
        half = 2 * sigma
    else:
        half = max((default_tol + (n - 1) * 2 * sigma) / n, 0.2 * default_tol)
    return centre, centre - half, centre + half


def item_facts(item: ConsensusItem, profile: Profile, exclude: set[str] = frozenset()) -> dict[str, Fact]:
    """The numbers a guideline may state: the action value, plus every context signal at
    the trigger and after settling. The template writer states a subset; the verifier
    accepts any of them. Events from runs in `exclude` are left out."""
    facts: dict[str, Fact] = {}
    evs = [e for r, e in item.events.items() if r not in exclude]

    def add(name: str, signal: str, values: list[float | None]):
        values = [v for v in values if v is not None and np.isfinite(v)]
        if not values:
            return
        spec = profile.signals.get(signal)
        centre, lo, hi = typical_range(values, spec.tolerance if spec else None, bool(spec and spec.angle))
        facts[name] = Fact(signal=signal, value=centre, lo=lo, hi=hi, unit=spec.unit if spec else None,
                           n=len(values), decimals=spec.decimals if spec else 0)

    if item.kind in ("control", "turn", "threshold"):
        add("action", item.signal, [e.after for e in evs])
    for sig in profile.context:
        add(f"trigger.{sig}", sig, [e.state.get(sig) for e in evs])
        add(f"result.{sig}", sig, [e.result.get(sig) for e in evs])
    return facts


def confidence(seen_in: int, run_count: int, facts: dict[str, Fact], profile: Profile) -> float:
    """Presence across runs x how many runs back it x how tight the ranges are."""
    if not run_count:
        return 0.0
    presence = seen_in / run_count
    support = seen_in / (seen_in + 1)
    ratios = []
    for f in facts.values():
        spec = profile.signals.get(f.signal)
        if spec and spec.tolerance and f.lo is not None:
            ratios.append(min(1.0, f.width / (2 * spec.tolerance)))
    tightness = 1.0 - 0.5 * (float(np.mean(ratios)) if ratios else 1.0)
    return round(presence * (0.5 * support + 0.5 * tightness), 3)
