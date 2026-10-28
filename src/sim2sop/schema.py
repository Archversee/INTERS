"""Common data model shared by every domain.

Telemetry itself lives in a pandas DataFrame with a float `t` column (seconds from run
start) plus one column per canonical signal named in the domain profile. Everything
else (annotations, events, phases, steps, SOPs) is a Pydantic model so it can be
validated on load and serialised to JSON.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class SignalCategory(str, Enum):
    pose = "pose"
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


class Flag(BaseModel):
    kind: Literal["unmatched_annotation", "verifier", "compliance", "consensus"]
    message: str
    t: float | None = None
    step_id: str | None = None
    run_id: str | None = None


class RunMeta(BaseModel):
    run_id: str
    domain: str
    procedure: str
    source_format: str
    instructor: str | None = None
    created: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    n_samples: int = 0
    duration_s: float = 0.0


class Fact(BaseModel):
    """A number the SOP is allowed to state, with where it came from."""

    signal: str
    value: float
    unit: str | None = None
    tol: float | None = None        # ± band; from consensus spread or profile default
    n: int = 1                      # number of runs behind the value
    decimals: int = 0

    def fmt(self, value: float | None = None) -> str:
        v = f"{self.value if value is None else value:.{self.decimals}f}"
        if not self.unit:
            return v
        return f"{v}{self.unit}" if self.unit in ("%", "°") else f"{v} {self.unit}"

    def fmt_tol(self) -> str:
        return f"±{self.fmt(self.tol)}" if self.tol is not None else ""


class Citation(BaseModel):
    doc: str
    section: str | None = None
    text: str
    score: float = 0.0


class Step(BaseModel):
    """Step template: trigger -> action -> intent -> expected result -> tolerance -> reference."""

    id: str
    phase: str
    key: str
    kind: str
    signal: str = ""
    label: str = ""                 # action vocabulary from the profile, e.g. "Flaps 10°"
    trigger: str = ""
    action: str = ""
    intent: str | None = None
    expected: str = ""
    tolerance: str = ""
    reference: Citation | None = None
    facts: dict[str, Fact] = Field(default_factory=dict)
    t: float = 0.0                  # time in the anchor run, for charts
    mandatory: bool = True
    seen_in: int = 1
    run_count: int = 1
    confidence: float = 0.5
    technique_notes: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    verified: bool = False
    edited: bool = False
    corrections_seen: int = 0       # relevant corrections in memory when this text was written


class SOP(BaseModel):
    procedure: str
    domain: str
    title: str
    version: int = 1
    created: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    runs: list[str] = Field(default_factory=list)
    anchor_run: str | None = None   # step times refer to this run (the consensus medoid)
    phases: list[str] = Field(default_factory=list)
    steps: list[Step] = Field(default_factory=list)
    flags: list[Flag] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    writer: str = "template"

    def steps_in(self, phase: str) -> list[Step]:
        return [s for s in self.steps if s.phase == phase]


class Correction(BaseModel):
    """An instructor edit, stored as a before/after diff with context."""

    domain: str
    procedure: str
    phase: str
    step_kind: str
    step_key: str
    field: Literal["trigger", "action", "intent", "expected", "tolerance"]
    before: str
    after: str
    pattern: str | None = None      # `after` with this step's facts as placeholders
    sop_version: int
    created: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
