"""AI findings for instructors: the patterns found in the runs (sim2sop.learn.patterns),
explained and grouped into a few findings, each with what to do about it.

The model sees the patterns and their numbers, never the telemetry. A finding must cite the
patterns it rests on, and every number it writes must be one of theirs; a finding that fails
is set aside with the reason. Without an AI, each strong pattern becomes a finding as is.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from sim2sop import config
from sim2sop.agent.llm import LLMClient, LLMError, LLMUnreachable, parse_json
from sim2sop.learn.patterns import Pattern, find_patterns
from sim2sop.profiles import Profile
from sim2sop.schema import Guidelines
from sim2sop.storage import RunRecord

NUM_RE = re.compile(r"(?<![A-Za-z0-9])-?\d+(?:\.\d+)?")
MAX_FINDINGS = 5
# Words that claim a repeated behaviour: wrong when every cited pattern rests on one run.
OFTEN_RE = re.compile(r"\b(frequently|often|several|many|repeatedly|regularly|consistently|always|usually|"
                      r"habitually|tends? to)\b", re.I)

SYSTEM_PROMPT = """You help a chief instructor understand how their instructors differ, from PATTERNS found in
simulator runs by statistics. Write up to 5 findings, most important first.

Rules:
- Each finding rests on one or more PATTERNS and lists their ids in "patterns".
- Use ONLY numbers that appear in those patterns' text. Never invent or compute new numbers.
- "headline": the finding in one plain sentence. "detail": what it means for training, 1-2 sentences.
- "action": one concrete thing the instructor team could do (brief instructors, add a note to the guideline,
  check a limit), or "" if none.
- Say only what the patterns show. Never generalise: one run is "one run", one instructor is that instructor,
  never "several" or "instructors" in general.
- Never guess reasons or motives (no "misunderstanding", "preference", "lack of"); if the reason matters,
  the action is to ask the instructor why.
- Group related patterns into one finding; skip patterns that say nothing useful.
Reply with one JSON object: {"findings": [{"headline", "detail", "action", "patterns": ["P1", ...]}]}"""


class Finding(BaseModel):
    headline: str
    detail: str = ""
    action: str = ""
    patterns: list[str] = Field(default_factory=list)
    problems: list[str] = Field(default_factory=list)


class Insights(BaseModel):
    procedure: str
    version: int                       # the guidelines version the runs come from
    runs: int
    writer: str                        # model name, or "template"
    made: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    patterns: list[Pattern] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    set_aside: list[Finding] = Field(default_factory=list)
    note: str | None = None            # e.g. why the AI was not used


def check(f: Finding, by_id: dict[str, Pattern]) -> Finding:
    problems = [f"unknown pattern {p}" for p in f.patterns if p not in by_id]
    cited = [by_id[p] for p in f.patterns if p in by_id]
    if not cited:
        problems.append("cites no pattern")
    allowed = {round(abs(float(n)), 3) for p in cited for n in p.numbers}
    # Numbers inside names count too ("Instructor 2", "Flaps 20°").
    allowed |= {round(abs(float(x)), 3) for p in cited for x in NUM_RE.findall(p.text)}
    # So does the gap between two of a pattern's own numbers ("0.4 m closer": 9.0 vs 8.6).
    allowed |= {round(abs(a - b), 3) for p in cited for a in p.numbers for b in p.numbers if a != b}
    text = f"{f.headline} {f.detail} {f.action}"
    for x in NUM_RE.findall(text):
        if round(abs(float(x)), 3) not in allowed:
            problems.append(f"{x} is not in the patterns it cites")
    if cited and all(len(p.runs) <= 1 for p in cited) and (word := OFTEN_RE.search(text)):
        problems.append(f'says "{word.group()}" but the patterns it cites rest on one run')
    f.problems = problems
    return f


def _template(patterns: list[Pattern]) -> list[Finding]:
    return [Finding(headline=p.text, patterns=[p.id]) for p in patterns[:MAX_FINDINGS]]


def explain(doc: Guidelines, runs: list[RunRecord], profile: Profile, llm: LLMClient | None) -> Insights:
    patterns = find_patterns(doc, runs, profile)
    out = Insights(procedure=profile.procedure, version=doc.version, runs=len(doc.runs),
                   writer="template", patterns=patterns)
    if not patterns:
        out.note = "No clear patterns yet: add more runs, from more than one instructor."
        return out
    if llm is None or not llm.available():
        out.findings = _template(patterns)
        out.note = "AI writer off: these are the strongest patterns as found."
        return out
    payload = [{"id": p.id, "kind": p.kind, "text": p.text} for p in patterns]
    prompt = (f"Procedure: {profile.title}\nRuns: {len(doc.runs)}\n\nPATTERNS:\n"
              + json.dumps(payload, indent=1, ensure_ascii=False))
    try:
        data = parse_json(llm.chat([{"role": "user", "content": prompt}], system=SYSTEM_PROMPT,
                                   json_mode=True).text)
    except (LLMUnreachable, LLMError, ValueError) as e:
        out.findings = _template(patterns)
        out.note = f"The AI writer did not answer ({e}); these are the strongest patterns as found."
        return out
    by_id = {p.id: p for p in patterns}
    for raw in (data.get("findings") or [])[:MAX_FINDINGS]:
        if not isinstance(raw, dict):
            continue
        cites = raw.get("patterns") or []
        f = check(Finding(headline=str(raw.get("headline") or ""), detail=str(raw.get("detail") or ""),
                          action=str(raw.get("action") or ""),
                          patterns=[str(c) for c in (cites if isinstance(cites, list) else [cites])]), by_id)
        (out.set_aside if f.problems or not f.headline else out.findings).append(f)
    out.writer = llm.name
    if not out.findings:
        out.findings = _template(patterns)
        out.note = "No AI finding passed the number check; these are the strongest patterns as found."
    return out


# --- the last analysis, kept per procedure ---------------------------------------------------

def _path(procedure: str):
    return config.DATA_DIR / "insights" / f"{procedure}.json"


def save(ins: Insights) -> None:
    path = _path(ins.procedure)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ins.model_dump_json(indent=2), encoding="utf-8")


def load(procedure: str) -> Insights | None:
    path = _path(procedure)
    return Insights.model_validate_json(path.read_text(encoding="utf-8")) if path.exists() else None
