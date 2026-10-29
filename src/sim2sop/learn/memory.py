"""Correction memory: instructor edits stored as before/after diffs with context, and
retrieved as examples the next time a similar step is written.

Two uses:
- LLM writer: the most relevant corrections become few-shot examples in the prompt.
- Template writer: a correction becomes a learned pattern. Numbers in the corrected text
  that match the step's facts are replaced by placeholders, so the same wording is reused
  with each new step's own (verified) numbers.
"""

from __future__ import annotations

import json
import re

from sim2sop import config
from sim2sop.schema import Correction, Fact

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _path():
    return config.MEMORY_DIR / "corrections.jsonl"


def add_correction(c: Correction) -> None:
    config.MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    with open(_path(), "a", encoding="utf-8") as f:
        f.write(c.model_dump_json() + "\n")


def all_corrections() -> list[Correction]:
    if not _path().exists():
        return []
    return [Correction.model_validate(json.loads(line))
            for line in _path().read_text(encoding="utf-8").splitlines() if line.strip()]


def _signal_of(key: str) -> str:
    parts = key.split(":")
    return parts[1] if len(parts) > 1 else ""


def retrieve(domain: str, procedure: str, phase: str, step_kind: str, step_key: str,
             field: str | None = None, k: int = 3) -> list[Correction]:
    """Rank by context similarity: same step > same kind+signal > same kind, same domain
    > other domain (writing style transfers across domains; numbers never do)."""
    scored = []
    for c in all_corrections():
        if field and c.field != field:
            continue
        s = 0.0
        if c.step_key == step_key:
            s += 3
        elif _signal_of(c.step_key) == _signal_of(step_key) and c.step_kind == step_kind:
            s += 2
        elif c.step_kind == step_kind:
            s += 1
        else:
            continue
        s += 1.0 if c.domain == domain else 0.0
        s += 0.5 if c.procedure == procedure else 0.0
        s += 0.5 if c.phase == phase else 0.0
        scored.append((s, c.created, c))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [c for _, _, c in scored[:k]]


def relevant_count(domain: str, step_kind: str, step_signal: str) -> int:
    """How many corrections apply to this kind of step (same kind and signal, same domain).
    When it grows, the step is rewritten so the new correction can take effect."""
    return sum(1 for c in all_corrections()
               if c.domain == domain and c.step_kind == step_kind and _signal_of(c.step_key) == step_signal)


def corrected_fields(domain: str, step_kind: str, step_signal: str) -> list[str]:
    """The fields those corrections changed: only these are restyled on similar steps."""
    return sorted({c.field for c in all_corrections()
                   if c.domain == domain and c.step_kind == step_kind and _signal_of(c.step_key) == step_signal})


# --- learned patterns for the template writer ---------------------------------------------

def to_pattern(text: str, facts: dict[str, Fact], label: str | None = None,
               intent: str | None = None) -> str:
    """Turn corrected text into a reusable pattern: {label}, {intent}, {fact:<name>} for a
    typical value and {fact:<name>.lo} / {fact:<name>.hi} for the ends of its range."""
    pattern = text
    if label and label in pattern:
        pattern = pattern.replace(label, "{label}")
    if intent and intent in pattern:
        pattern = pattern.replace(intent, "{intent}")

    def repl(m: re.Match) -> str:
        x = float(m.group())
        for name, f in facts.items():
            part = matching_part(x, f)
            if part:
                return "{fact:" + name + ("" if part == "value" else "." + part) + "}"
        return m.group()

    # Leave numbers inside the {label} / {intent} placeholders alone.
    parts = re.split(r"(\{[^}]*\})", pattern)
    return "".join(p if p.startswith("{") else NUM_RE.sub(repl, p) for p in parts)


def fill_pattern(pattern: str, facts: dict[str, Fact], label: str, intent: str | None) -> str | None:
    """Fill a learned pattern; None if it needs a fact this step does not have."""
    missing = False

    def repl(m: re.Match) -> str:
        nonlocal missing
        token = m.group(1)
        if token == "label":
            return label
        if token == "intent":
            if not intent:
                missing = True
            return intent or ""
        if token.startswith("fact:"):
            name, _, part = token[5:].partition(".")
            if name in ("trigger", "result"):  # names contain a dot: trigger.ias_kt[.lo]
                sig, _, part = part.partition(".")
                name = f"{name}.{sig}"
            f = facts.get(name)
            if f is None:
                missing = True
                return ""
            v = {"lo": f.lo, "hi": f.hi}.get(part, f.value)
            return f"{(f.value if v is None else v):.{f.decimals}f}"
        return m.group(0)

    out = re.sub(r"\{([^}]*)\}", repl, pattern)
    return None if missing else out


def matching_part(x: float, f: Fact) -> str | None:
    """Which number of a fact a stated number is: the typical value, or the low / high end
    of its range, up to sensible rounding (display precision or 3 %). Sign is ignored so
    "descend at 400 fpm" matches -400."""
    best, best_d = None, float("inf")
    for part, v in (("value", f.value), ("lo", f.lo), ("hi", f.hi)):
        if v is None:
            continue
        d = abs(abs(x) - abs(v))
        if d <= max(0.5 * 10 ** (-f.decimals), 0.03 * abs(v)) and d < best_d:
            best, best_d = part, d
    return best


def number_matches(x: float, f: Fact) -> bool:
    return matching_part(x, f) is not None
