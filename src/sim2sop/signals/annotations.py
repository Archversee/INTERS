"""Link each intent annotation to the detected event it describes.

Score = time distance (as a fraction of the window) minus a bonus when the annotation
text contains a profile keyword pointing at the event's signal. Each event takes at most
one annotation. Annotations with no event in the window are returned as flags.
"""

from __future__ import annotations

from sim2sop.profiles import Profile
from sim2sop.schema import Annotation, Event, Flag

KEYWORD_BONUS = 0.6


def match_annotations(annotations: list[Annotation], events: list[Event],
                      profile: Profile, run_id: str | None = None) -> list[Flag]:
    window = profile.annotation_window_s
    candidates = []
    for ai, a in enumerate(annotations):
        text = a.text.lower()
        hinted = {sig for kw, sigs in profile.annotation_keywords.items() if kw in text for sig in sigs}
        for ev in events:
            dt = a.t - ev.t
            if abs(dt) > window:
                continue
            score = abs(dt) / window - (KEYWORD_BONUS if ev.signal in hinted else 0.0)
            candidates.append((score, ai, ev, dt))

    taken_a: set[int] = set()
    taken_e: set[str] = set()
    for score, ai, ev, dt in sorted(candidates, key=lambda c: c[0]):
        if ai in taken_a or ev.id in taken_e:
            continue
        a = annotations[ai]
        a.event_id, a.match_dt = ev.id, round(dt, 2)
        ev.annotation = a.text
        taken_a.add(ai)
        taken_e.add(ev.id)

    return [
        Flag(kind="unmatched_annotation", t=a.t, run_id=run_id,
             message=f'Annotation "{a.text}" at {a.t:.1f}s has no matching event within ±{window:g}s')
        for i, a in enumerate(annotations) if i not in taken_a
    ]
