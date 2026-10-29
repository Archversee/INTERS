"""Common data model shared by every domain.

Telemetry itself lives in a pandas DataFrame with a float `t` column (seconds from run
start) plus one column per canonical signal named in the domain profile. Everything
else (annotations, events, phases, guidelines, references) is a Pydantic model so it
can be validated on load and serialised to JSON.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


class SignalCategory(str, Enum):
    pose = "pose"               # incl. position and ground track
    motion = "motion"
    control = "control"         # continuous controls
    discrete = "discrete"       # switches, levers with detents, gear
    environment = "environment"


class Annotation(BaseModel):
    """An instructor's intent callout ("Rotating", "Flaps 10 on downwind")."""

    t: float
    text: str
    source: Literal["spoken", "typed"] = "typed"
    event_id: str | None = None     # filled by annotation matching
    match_dt: float | None = None   # annotation time minus event time


class Event(BaseModel):
    """One entry of the compact event timeline. This is all the LLM ever sees."""

    id: str
    t: float
    kind: Literal["toggle", "threshold", "control", "turn"]
    signal: str
    label: str                      # human-readable action, e.g. "Flaps 10°"
    template: str | None = None     # label with "{value}" where `after` goes, for restating
    before: float | None = None
    after: float | None = None
    unit: str | None = None
    phase: str | None = None
    key: str = ""                   # stable signature used to align runs
    state: dict[str, float] = Field(default_factory=dict)   # context at t (the trigger)
    result: dict[str, float] = Field(default_factory=dict)  # context after settling
    annotation: str | None = None


class Phase(BaseModel):
    name: str
    t_start: float
    t_end: float


FlagKind = Literal["unmatched_annotation", "verifier", "limit_violation", "excluded",
                   "clipped", "required_missing", "technique_change", "unchecked", "data"]


class Flag(BaseModel):
    kind: FlagKind
    message: str
    t: float | None = None
    step_id: str | None = None
    run_id: str | None = None
    phase: str | None = None


class RunMeta(BaseModel):
    run_id: str
    domain: str
    procedure: str
    source_format: str
    instructor: str | None = None
    environment: str | None = None  # version tag, e.g. "new-obstacle"; None = baseline
    created: datetime = Field(default_factory=_now)
    n_samples: int = 0
    duration_s: float = 0.0


# --- reference SOPs (read-only) ----------------------------------------------------------

class Limit(BaseModel):
    """A hard limit from a reference SOP / POH / NOTAM. Guidelines never cross it."""

    id: str
    signal: str
    min: float | None = None
    max: float | None = None
    when: str | None = None         # condition on the post-action state, e.g. "flaps_idx >= 2"
    phases: list[str] | None = None
    at: Literal["trigger", "result", "both"] = "both"   # which snapshot of a step it bounds
    environment: str | None = None  # runs from this environment are checked; None = all runs
    text: str = ""
    section: str | None = None
    # filled from the owning document
    doc_id: str = ""
    doc_title: str = ""
    revision: str | None = None

    def describe(self) -> str:
        if self.min is not None and self.max is not None:
            return f"{self.min:g} to {self.max:g}"
        return f"max {self.max:g}" if self.max is not None else f"min {self.min:g}"

    def cite(self) -> str:
        rev = f" rev {self.revision}" if self.revision else ""
        sec = f", {self.section}" if self.section else ""
        return f"{self.doc_title}{rev}{sec}"


class Requirement(BaseModel):
    """An action a reference SOP makes mandatory (always a core guideline)."""

    id: str
    key: str                        # event key, e.g. "toggle:light_landing:1"
    text: str = ""
    section: str | None = None
    doc_id: str = ""
    doc_title: str = ""
    revision: str | None = None


class RefDoc(BaseModel):
    """A registered reference document. The file is copied in and never modified."""

    id: str                         # stable across revisions (front matter `id`)
    title: str
    kind: Literal["sop", "poh", "notam", "regulation", "handbook", "other"] = "other"
    revision: str | None = None
    file: str                       # stored file name inside the library
    added: datetime = Field(default_factory=_now)
    active: bool = True             # False once superseded by a newer revision
    limits: list[Limit] = Field(default_factory=list)
    requirements: list[Requirement] = Field(default_factory=list)

    @property
    def label(self) -> str:
        return f"{self.title} rev {self.revision}" if self.revision else self.title


# --- guidelines ------------------------------------------------------------------------------

class Fact(BaseModel):
    """A number a guideline may state: the typical value instructors use and its range,
    with the hard limit it is checked against."""

    signal: str
    value: float                    # median across runs (after clipping)
    lo: float | None = None         # typical range
    hi: float | None = None
    unit: str | None = None
    n: int = 1                      # number of runs behind the value
    decimals: int = 0
    limit_min: float | None = None
    limit_max: float | None = None
    clipped: bool = False

    def _num(self, x: float) -> str:
        v = f"{x:.{self.decimals}f}"
        return v.lstrip("-") if float(v) == 0 else v   # never "-0"

    def fmt(self, value: float | None = None) -> str:
        v = self._num(self.value if value is None else value)
        if not self.unit:
            return v
        return f"{v}{self.unit}" if self.unit in ("%", "°") else f"{v} {self.unit}"

    def fmt_range(self) -> str:
        if self.lo is None or self.hi is None:
            return self.fmt()
        lo = self._num(self.lo)
        if lo == self._num(self.hi):
            return self.fmt()
        sep = " to " if self.lo < 0 or self.hi < 0 else "–"
        return f"{lo}{sep}{self.fmt(self.hi)}"

    @property
    def width(self) -> float:
        return (self.hi - self.lo) if self.lo is not None and self.hi is not None else 0.0


class Citation(BaseModel):
    doc: str
    section: str | None = None
    text: str
    score: float = 0.0


class Step(BaseModel):
    """Guideline template: when (trigger) -> what to do -> why (annotation)
    -> typical range (from runs) -> hard limit (from the reference SOP)."""

    id: str
    phase: str
    key: str
    kind: str
    signal: str = ""
    label: str = ""                 # action vocabulary from the profile, e.g. "Flaps 10°"
    trigger: str = ""
    action: str = ""
    intent: str | None = None
    typical: str = ""
    limit: str = ""
    reference: Citation | None = None
    limits: list[Limit] = Field(default_factory=list)
    facts: dict[str, Fact] = Field(default_factory=dict)
    t: float = 0.0                  # time in anchor_run, for charts
    anchor_run: str | None = None
    mandatory: bool = True          # core guideline (every run, or required by an SOP)
    required_by: str | None = None  # reference that makes it mandatory
    seen_in: int = 1
    run_count: int = 1
    runs: list[str] = Field(default_factory=list)   # runs behind the typical ranges
    confidence: float = 0.5
    technique_notes: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    verified: bool = False
    edited: bool = False
    corrections_seen: int = 0       # relevant corrections in memory when this text was written
    # update mode
    change: Literal["added", "modified", "moved"] | None = None
    moved_from: str | None = None   # previous phase of a "moved" step
    change_reasons: list[str] = Field(default_factory=list)
    review: Literal["pending", "approved", "rejected"] | None = None


class Guidelines(BaseModel):
    """One version of the guidelines for a procedure. Advisory: the reference SOPs win."""

    procedure: str
    domain: str
    title: str
    version: int = 1
    parent_version: int | None = None
    mode: Literal["create", "update", "edit"] = "create"
    status: Literal["draft", "published"] = "draft"
    checked: bool = False           # limit-checked against reference SOPs (else "unchecked")
    created: datetime = Field(default_factory=_now)
    runs: list[str] = Field(default_factory=list)
    phase_runs: dict[str, list[str]] = Field(default_factory=dict)  # runs behind each phase
    environment: str | None = None
    anchor_run: str | None = None   # step times refer to this run (the consensus medoid)
    references: list[str] = Field(default_factory=list)  # RefDoc labels in force
    phases: list[str] = Field(default_factory=list)
    steps: list[Step] = Field(default_factory=list)
    removed: list[Step] = Field(default_factory=list)     # steps dropped by this version
    rejected: dict[str, str] = Field(default_factory=dict)  # step slot -> proposal the instructor rejected
    flags: list[Flag] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    writer: str = "template"

    def steps_in(self, phase: str) -> list[Step]:
        return [s for s in self.steps if s.phase == phase]

    @property
    def pending(self) -> list[Step]:
        return [s for s in self.steps if s.review == "pending"]


class Correction(BaseModel):
    """An instructor edit, stored as a before/after diff with context."""

    domain: str
    procedure: str
    phase: str
    step_kind: str
    step_key: str
    field: Literal["trigger", "action", "intent", "typical"]   # limits are read-only
    before: str
    after: str
    pattern: str | None = None      # `after` with this step's facts as placeholders
    version: int
    created: datetime = Field(default_factory=_now)
