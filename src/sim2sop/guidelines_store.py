"""Versioned guidelines: data/guidelines/<procedure>/v<N>.json, plus rendered HTML next to it.

A new version is always a new file. Approving or rejecting steps changes the review
state of that version in place; it does not create a version.
"""

from __future__ import annotations

from sim2sop import config
from sim2sop.learn.changes import step_slots
from sim2sop.schema import Guidelines

TEXT_FIELDS = ("trigger", "action", "intent", "typical", "limit")


def _dir(procedure: str):
    return config.GUIDELINES_DIR / procedure


def versions(procedure: str) -> list[int]:
    d = _dir(procedure)
    if not d.exists():
        return []
    return sorted(int(p.stem[1:]) for p in d.glob("v*.json") if p.stem[1:].isdigit())


def load(procedure: str, version: int) -> Guidelines:
    return Guidelines.model_validate_json((_dir(procedure) / f"v{version}.json").read_text(encoding="utf-8"))


def latest(procedure: str) -> Guidelines | None:
    vs = versions(procedure)
    return load(procedure, vs[-1]) if vs else None


def latest_published(procedure: str) -> Guidelines | None:
    for v in reversed(versions(procedure)):
        doc = load(procedure, v)
        if doc.status == "published":
            return doc
    return None


def html_path(doc: Guidelines):
    return _dir(doc.procedure) / f"v{doc.version}.html"


def save_new(doc: Guidelines) -> Guidelines:
    """Save as the next version number."""
    vs = versions(doc.procedure)
    doc.version = (vs[-1] + 1) if vs else 1
    return overwrite(doc)


def overwrite(doc: Guidelines) -> Guidelines:
    d = _dir(doc.procedure)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"v{doc.version}.json").write_text(doc.model_dump_json(indent=2), encoding="utf-8")
    return doc


def diff(a: Guidelines, b: Guidelines) -> list[dict]:
    """Step-level changes from a to b, matched on (phase, key, occurrence)."""
    old, new = step_slots(a.steps), step_slots(b.steps)
    out = []
    for slot, s in new.items():
        o = old.get(slot)
        if o is None:
            out.append({"slot": slot, "step": s.id, "label": s.label, "change": "added",
                        "reasons": s.change_reasons})
            continue
        changed = {f: (getattr(o, f), getattr(s, f)) for f in TEXT_FIELDS if getattr(o, f) != getattr(s, f)}
        if changed or o.mandatory != s.mandatory:
            out.append({"slot": slot, "step": s.id, "label": s.label, "change": "modified",
                        "fields": changed, "reasons": s.change_reasons})
    out += [{"slot": k, "step": o.id, "label": o.label, "change": "removed"}
            for k, o in old.items() if k not in new]
    return out
