"""YAML domain profiles: the only per-domain configuration besides the adapter.

A profile names the signals, phases, which events to extract, annotation keywords,
reference limits and where the reference documents live. Adding a domain means adding
one YAML file (plus an adapter if the source format is new).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from sim2sop import config
from sim2sop.schema import SignalCategory


class SignalSpec(BaseModel):
    category: SignalCategory
    unit: str | None = None
    label: str | None = None
    tolerance: float | None = None   # default ± band for a single run
    decimals: int = 0
    angle: bool = False              # wraps at 360 (heading)


class PhaseSpec(BaseModel):
    name: str
    enter: str | None = None         # pandas expression; None = initial phase
    hold_s: float = 2.0              # condition must hold this long to enter
    description: str | None = None
    trigger_signals: list[str] | None = None   # override the profile-wide lists in this phase
    result_signals: list[str] | None = None


class ToggleSpec(BaseModel):
    name: str
    states: dict[int, str] = Field(default_factory=dict)


class ThresholdSpec(BaseModel):
    signal: str
    above: float | None = None
    below: float | None = None
    label: str
    hold_s: float = 1.0
    phases: list[str] | None = None  # only detect inside these phases
    once: bool = True

    @model_validator(mode="after")
    def _one_direction(self):
        if (self.above is None) == (self.below is None):
            raise ValueError(f"threshold on {self.signal}: set exactly one of above/below")
        return self


class ControlSpec(BaseModel):
    signal: str
    name: str
    min_step: float                  # minimum settled change to count as an input
    settle_s: float = 3.0


class TurnSpec(BaseModel):
    signal: str
    min_change_deg: float = 45.0
    min_rate_dps: float = 1.0
    min_duration_s: float = 5.0


class EventsSpec(BaseModel):
    toggles: dict[str, ToggleSpec] = Field(default_factory=dict)
    thresholds: list[ThresholdSpec] = Field(default_factory=list)
    controls: list[ControlSpec] = Field(default_factory=list)
    turns: TurnSpec | None = None
    settle_s: float = 8.0            # where the "result" (after the action) snapshot is taken


class RelativeHeadingSpec(BaseModel):
    """Heading relative to the heading held where `anchor` first becomes true
    (e.g. runway heading = heading when the take-off roll passes 30 kt)."""

    name: str
    signal: str
    anchor: str
    anchor_window_s: float = 5.0


class TrackFrameSpec(BaseModel):
    """Position as ground track in a frame anchored where `anchor` first becomes true
    (e.g. the take-off roll): `along` the anchor heading and `offset` across it
    (absolute distance; `side` is signed, + = right). Uses lat/lon if given, else x/y in m.
    `backward` measures `along` behind the anchor (distance reversed)."""

    along: str
    offset: str
    side: str | None = None
    anchor: str
    heading_signal: str
    lat: str | None = None
    lon: str | None = None
    x: str | None = None
    y: str | None = None
    unit: Literal["nm", "m"] = "nm"
    backward: bool = False
    anchor_window_s: float = 5.0


class DerivedSpec(BaseModel):
    """Derived signals computed after the adapter, before segmentation."""

    relative_heading: list[RelativeHeadingSpec] = Field(default_factory=list)
    track_frame: list[TrackFrameSpec] = Field(default_factory=list)


class Profile(BaseModel):
    domain: str
    procedure: str
    title: str
    adapter: str
    sample_hz: float = 5.0
    signals: dict[str, SignalSpec]
    context: list[str]               # signals snapshotted at the trigger and the result
    trigger_signals: list[str] = Field(default_factory=list)   # default: first two of context
    result_signals: list[str] = Field(default_factory=list)
    derived: DerivedSpec = Field(default_factory=DerivedSpec)
    phases: list[PhaseSpec]
    segmentation_signals: list[str] = Field(default_factory=list)
    events: EventsSpec = Field(default_factory=EventsSpec)
    annotation_window_s: float = 10.0
    annotation_keywords: dict[str, list[str]] = Field(default_factory=dict)
    # Update mode: a phase counts as a technique change when at least this many runs from a
    # new environment all fall outside the current typical range, on the same side.
    change_min_runs: int = 3
    references: str                  # sub-folder of references/ holding sample reference docs
    style: str = ""                  # writing guidance for the LLM

    @model_validator(mode="after")
    def _defaults(self):
        self.trigger_signals = self.trigger_signals or self.context[:2]
        self.result_signals = self.result_signals or self.context[:2]
        return self

    def phase_spec(self, name: str) -> PhaseSpec | None:
        return next((p for p in self.phases if p.name == name), None)

    def trigger_for(self, phase: str) -> list[str]:
        spec = self.phase_spec(phase)
        return spec.trigger_signals if spec and spec.trigger_signals else self.trigger_signals

    def result_for(self, phase: str) -> list[str]:
        spec = self.phase_spec(phase)
        return spec.result_signals if spec and spec.result_signals else self.result_signals

    def stated_facts(self, phase: str) -> list[str]:
        """Facts a guideline in this phase states: the action value, the trigger values, and
        only the results this phase adds on purpose (e.g. distance from the runway on downwind)."""
        return (["action"] + [f"trigger.{s}" for s in self.trigger_for(phase)]
                + [f"result.{s}" for s in self.result_for(phase) if s not in self.result_signals])

    def unit(self, signal: str) -> str | None:
        spec = self.signals.get(signal)
        return spec.unit if spec else None

    def label(self, signal: str) -> str:
        spec = self.signals.get(signal)
        return (spec.label if spec and spec.label else signal)

    @property
    def derived_signals(self) -> set[str]:
        """Signals computed from the telemetry rather than read from it."""
        d = self.derived
        return ({r.name for r in d.relative_heading}
                | {x for t in d.track_frame for x in (t.along, t.offset, t.side) if x})

    @property
    def phase_names(self) -> list[str]:
        return [p.name for p in self.phases]


def load_profile(name_or_path: str | Path) -> Profile:
    path = Path(name_or_path)
    if not path.suffix:
        path = config.PROFILES_DIR / f"{name_or_path}.yaml"
    with open(path, encoding="utf-8") as f:
        return Profile.model_validate(yaml.safe_load(f))


def list_profiles() -> list[str]:
    return sorted(p.stem for p in config.PROFILES_DIR.glob("*.yaml"))
