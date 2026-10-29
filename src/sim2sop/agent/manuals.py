"""AI reads an official manual (POH, SOP, handbook) and proposes the hard limits and required
actions in it, in the procedure's own signals. An instructor confirms them before they are
in force.

The model never decides on its own. Every proposal is checked against the manual and the
profile, and a proposal that fails a check is set aside with the reason:
  - its quote must appear in the manual, word for word (no invented rules);
  - a limit's value must appear in that quote (no unit conversions or guesses);
  - the signal, the phases and the `when` condition must exist in the profile;
  - a required action must be an action the procedure can detect (a toggle state).
Each section with some text is sent to the model, one call each, up to MAX_SECTIONS; when a
manual has more, those that look like rules (a number, or "must", "never", "do not") go first.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from sim2sop import references
from sim2sop.agent.llm import LLMClient, LLMError, LLMUnreachable, parse_json
from sim2sop.profiles import Profile
from sim2sop.schema import RefDoc

MAX_SECTIONS = 24          # model calls per manual
SECTION_CHARS = 2400       # longer sections are split
RULE_WORDS = re.compile(r"\b(must|never|do not|don't|shall|not exceed|maximum|minimum|max|min|limit|"
                        r"required|always|before|at least|no more than|no faster|no slower|below|above)\b", re.I)
NUM_RE = re.compile(r"(?<![A-Za-z0-9])-?\d+(?:\.\d+)?")

SYSTEM_PROMPT = """You read one section of an official training manual and list the HARD RULES it states.
Only rules the text states explicitly; never infer, convert units or round.

- A limit bounds one measurable signal from the SIGNALS list: {"signal", "min" or "max" (a number written in the
  quote, in the signal's unit), "when" (optional condition using SIGNALS, e.g. "flaps_idx >= 2"), "phases"
  (optional, from PHASES), "text" (the rule in a short sentence), "quote" (the exact words from the section)}.
- A required action is one of ACTIONS that the text says must be done: {"key", "text", "quote"}.
- Skip anything that does not map onto SIGNALS or ACTIONS, and advice ("should", "normally").
Reply with one JSON object: {"limits": [...], "requirements": [...]} (empty lists if none)."""


class Proposal(BaseModel):
    """One rule the AI found, with the checks it passed or failed."""

    kind: Literal["limit", "requirement"]
    id: str = ""
    signal: str | None = None
    min: float | None = None
    max: float | None = None
    when: str | None = None
    phases: list[str] | None = None
    key: str | None = None
    text: str = ""
    quote: str = ""
    section: str = ""
    problems: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def as_rule(self) -> dict:
        """The front-matter entry for the document's limits / requirements."""
        if self.kind == "limit":
            out = {"id": self.id, "signal": self.signal, "section": self.section, "text": self.text}
            out |= {k: v for k, v in (("min", self.min), ("max", self.max), ("when", self.when),
                                      ("phases", self.phases)) if v not in (None, [])}
            return out
        return {"id": self.id, "key": self.key, "section": self.section, "text": self.text}


@dataclass
class ManualReading:
    title: str
    proposals: list[Proposal] = field(default_factory=list)
    sections_read: int = 0
    sections_total: int = 0
    stopped: str | None = None     # why reading stopped early (the AI server stopped answering)

    @property
    def accepted(self) -> list[Proposal]:
        return [p for p in self.proposals if p.ok]

    @property
    def set_aside(self) -> list[Proposal]:
        return [p for p in self.proposals if not p.ok]


# --- the manual's text ----------------------------------------------------------------------

def read_sections(path: Path) -> list[tuple[str, str]]:
    """(section, text) pairs: Markdown headings, PDF pages, or paragraphs of plain text."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        raw = [(f"p.{i + 1}", page.extract_text() or "") for i, page in enumerate(PdfReader(str(path)).pages)]
    else:
        _, body = references.split_front_matter(path.read_text(encoding="utf-8", errors="replace"))
        raw, title, buf = [], "Text", []
        for line in body.splitlines():
            if line.startswith("#"):
                if buf:
                    raw.append((title, "\n".join(buf)))
                title, buf = line.lstrip("#").strip() or title, []
            elif not line.startswith(">"):   # ">" lines are editor notes in the samples
                buf.append(line)
        if buf:
            raw.append((title, "\n".join(buf)))
    out = []
    for title, text in raw:
        text = re.sub(r"[ \t]+", " ", text).strip()
        for i in range(0, len(text), SECTION_CHARS):
            if part := text[i:i + SECTION_CHARS].strip():
                out.append((title if i == 0 else f"{title} (cont.)", part))
    return out


def states_a_rule(text: str) -> bool:
    return bool(NUM_RE.search(text)) or bool(RULE_WORDS.search(text))


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[‐-―−]", "-", text)).strip().lower()


# --- what the model may use ------------------------------------------------------------------

def actions(profile: Profile) -> dict[str, str]:
    """Event keys a required action can name, with their wording ("toggle:hand_brake:1": "Handbrake APPLY")."""
    return {f"toggle:{sig}:{state}": f"{spec.name} {label}".strip()
            for sig, spec in profile.events.toggles.items() for state, label in spec.states.items()}


def vocabulary(profile: Profile) -> dict:
    return {
        "SIGNALS": {s: f"{profile.label(s)}" + (f" ({spec.unit})" if spec.unit else "")
                    for s, spec in profile.signals.items() if spec.category.value != "environment"},
        "PHASES": profile.phase_names,
        "ACTIONS": actions(profile),
    }


# --- checks ------------------------------------------------------------------------------------

def _when_ok(expr: str, profile: Profile) -> bool:
    row = pd.DataFrame({s: [0.0] for s in profile.signals} | {"t": [0.0]})
    try:
        out = row.eval(expr)
    except Exception:
        return False
    return np.asarray(out).dtype == bool


def check(p: Proposal, profile: Profile, manual_text: str) -> Proposal:
    problems = []
    if not p.quote or _norm(p.quote) not in _norm(manual_text):
        problems.append("the quote is not in the manual")
    if p.kind == "limit":
        if p.signal not in profile.signals:
            problems.append(f"unknown signal {p.signal!r}")
        if p.min is None and p.max is None:
            problems.append("no min or max")
        quoted = {abs(float(x)) for x in NUM_RE.findall(p.quote or "")}
        for name, v in (("min", p.min), ("max", p.max)):
            if v is not None and abs(v) not in quoted:
                problems.append(f"{name} {v:g} is not written in the quote")
        if p.when and not _when_ok(p.when, profile):
            problems.append(f"condition {p.when!r} does not use the procedure's signals")
        bad = [ph for ph in p.phases or [] if ph not in profile.phase_names]
        if bad:
            problems.append(f"unknown phase {', '.join(bad)}")
    elif p.key not in actions(profile):
        problems.append(f"{p.key!r} is not an action this procedure detects")
    p.problems = problems
    return p


# --- reading -----------------------------------------------------------------------------------

def _num(v) -> float | None:
    """A number, also when written with its unit ("5 km/h")."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    m = NUM_RE.search(str(v)) if v not in (None, "") else None
    return float(m.group()) if m else None


def _flat(text: str) -> str:
    return re.sub(r"[\s_]+", " ", text).strip().lower()


def _name(value, names: dict[str, str]) -> str | None:
    """A small model may answer with a label ("speed", "Reverse In", "Handbrake APPLY") instead
    of the name: map it back when it matches exactly (case and spacing aside)."""
    if value in (None, ""):
        return None
    return names.get(_flat(str(value)), str(value))


def _proposals(data: dict, section: str, profile: Profile) -> list[Proposal]:
    signals = {_flat(s): s for s in profile.signals} | {_flat(profile.label(s)): s for s in profile.signals}
    phases = {_flat(ph): ph for ph in profile.phase_names}
    keys = {_flat(k): k for k in actions(profile)} | {_flat(v): k for k, v in actions(profile).items()}
    out = []
    for raw in data.get("limits") or []:
        if isinstance(raw, dict):
            ph = raw.get("phases")
            ph = [ph] if isinstance(ph, str) else ph if isinstance(ph, list) else []
            out.append(Proposal(kind="limit", signal=_name(raw.get("signal"), signals), min=_num(raw.get("min")),
                                max=_num(raw.get("max")), when=raw.get("when") or None,
                                phases=[_name(x, phases) for x in ph] or None,
                                text=str(raw.get("text") or ""), quote=str(raw.get("quote") or ""), section=section))
    for raw in data.get("requirements") or []:
        if isinstance(raw, dict):
            out.append(Proposal(kind="requirement", key=_name(raw.get("key"), keys), text=str(raw.get("text") or ""),
                                quote=str(raw.get("quote") or ""), section=section))
    return out


def read_manual(llm: LLMClient, path: Path, profile: Profile, title: str | None = None,
                progress: Callable[[int, int, str], None] | None = None) -> ManualReading:
    """Ask the model, section by section, for the rules in a manual; check every proposal."""
    sections = read_sections(path)
    manual_text = "\n".join(t for _, t in sections)
    # Commands with no number or rule word ("Cancel the signal once parked") are rules too:
    # every section is read when they fit, the rule-like ones first when they do not.
    texty = [(s, t) for s, t in sections if len(t) >= 20]
    keep = set(sorted(range(len(texty)), key=lambda i: not states_a_rule(texty[i][1]))[:MAX_SECTIONS])
    todo = [x for i, x in enumerate(texty) if i in keep]
    reading = ManualReading(title=title or Path(path).stem, sections_total=len(sections))
    vocab = json.dumps(vocabulary(profile), indent=1, ensure_ascii=False)
    for i, (section, text) in enumerate(todo):
        if progress:
            progress(i, len(todo), f"Reading {section}")
        prompt = (f"Procedure: {profile.title}\n{vocab}\n\nMANUAL SECTION \"{section}\":\n{text}")
        try:
            data = parse_json(llm.chat([{"role": "user", "content": prompt}], system=SYSTEM_PROMPT,
                                       json_mode=True).text)
        except LLMUnreachable as e:
            reading.stopped = str(e)
            break
        except (LLMError, ValueError):
            data = {}   # an unreadable reply: nothing proposed for this section
        reading.proposals += [check(p, profile, manual_text) for p in _proposals(data, section, profile)]
        reading.sections_read += 1
    _dedupe_and_name(reading)
    return reading


def _dedupe_and_name(reading: ManualReading) -> None:
    seen, kept = set(), []
    for p in reading.proposals:
        ident = (p.kind, p.signal, p.min, p.max, p.when, tuple(p.phases or ()), p.key)
        if ident in seen:
            continue
        seen.add(ident)
        kept.append(p)
    for n, p in enumerate(kept, 1):
        what = p.signal or (p.key or "").replace(":", "_")
        p.id = f"ai_{p.kind}_{n:02d}_{re.sub(r'[^a-z0-9_]', '', what.lower())}"
    reading.proposals = kept


def add_manual(procedure: str, path: Path, title: str, kind: str, revision: str | None,
               rules: list[Proposal]) -> RefDoc:
    """Add the manual with the confirmed rules, as a limits file beside it (the manual itself
    is copied unchanged)."""
    import shutil
    import tempfile

    import yaml

    meta = {"id": re.sub(r"[^a-z0-9_]+", "_", title.lower()).strip("_") or Path(path).stem,
            "title": title, "kind": kind, "revision": revision,
            "limits": [p.as_rule() for p in rules if p.kind == "limit"],
            "requirements": [p.as_rule() for p in rules if p.kind == "requirement"]}
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / Path(path).name
        shutil.copyfile(path, copy)
        copy.with_suffix(".limits.yaml").write_text(yaml.safe_dump(meta, sort_keys=False, allow_unicode=True),
                                                    encoding="utf-8")
        return references.add_reference(procedure, copy)
