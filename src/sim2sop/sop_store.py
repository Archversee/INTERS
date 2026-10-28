"""Versioned SOPs: data/sops/<procedure>/v<N>.json, plus rendered HTML next to it."""

from __future__ import annotations

from sim2sop import config
from sim2sop.schema import SOP


def _dir(procedure: str):
    return config.SOPS_DIR / procedure


def versions(procedure: str) -> list[int]:
    d = _dir(procedure)
    if not d.exists():
        return []
    return sorted(int(p.stem[1:]) for p in d.glob("v*.json") if p.stem[1:].isdigit())


def latest(procedure: str) -> SOP | None:
    vs = versions(procedure)
    return load(procedure, vs[-1]) if vs else None


def load(procedure: str, version: int) -> SOP:
    return SOP.model_validate_json((_dir(procedure) / f"v{version}.json").read_text(encoding="utf-8"))


def save(sop: SOP) -> SOP:
    """Save as the next version number."""
    vs = versions(sop.procedure)
    sop.version = (vs[-1] + 1) if vs else 1
    d = _dir(sop.procedure)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"v{sop.version}.json").write_text(sop.model_dump_json(indent=2), encoding="utf-8")
    return sop


def diff(a: SOP, b: SOP) -> list[dict]:
    """Step-level changes from a to b, matched on step key."""
    fields = ("trigger", "action", "intent", "expected", "tolerance")
    old = {s.key: s for s in a.steps}
    new = {s.key: s for s in b.steps}
    out = []
    for key, s in new.items():
        o = old.get(key)
        if o is None:
            out.append({"key": key, "label": s.label, "change": "added"})
            continue
        changed = {f: (getattr(o, f), getattr(s, f)) for f in fields if getattr(o, f) != getattr(s, f)}
        if changed or o.mandatory != s.mandatory:
            out.append({"key": key, "label": s.label, "change": "modified", "fields": changed,
                        "mandatory": (o.mandatory, s.mandatory)})
    out += [{"key": k, "label": o.label, "change": "removed"} for k, o in old.items() if k not in new]
    return out
