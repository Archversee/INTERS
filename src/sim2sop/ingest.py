"""Ingest one run: adapter -> resample -> derived signals (incl. ground track) -> phases
-> events -> annotation matching -> run library.

Reference limits are not checked here: references can be added later, so violations
are reported each time guidelines are built (sim2sop.signals.compliance).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sim2sop.adapters import get_adapter
from sim2sop.profiles import Profile
from sim2sop.schema import Flag, RunMeta
from sim2sop.signals.annotations import match_annotations
from sim2sop.signals.events import extract_events
from sim2sop.signals.phases import segment
from sim2sop.storage import RunRecord, save_run
from sim2sop.telemetry import add_derived, resample, validate


def analyse_run(source_dir: Path, profile: Profile, run_id: str | None = None,
                environment: str | None = None) -> tuple[RunRecord, pd.DataFrame]:
    """Process a run without storing it (used to preview generated flights)."""
    source_dir = Path(source_dir)
    adapter = get_adapter(profile.adapter)
    df, annotations, meta = adapter.load(source_dir)
    run_id = run_id or meta.get("run_id") or source_dir.name

    warnings = validate(df, profile)
    df = add_derived(resample(df, profile), profile)
    phases = segment(df, profile)
    events = extract_events(df, profile, phases)
    flags = match_annotations(annotations, events, profile, run_id=run_id)
    flags += [Flag(kind="data", message=w, run_id=run_id) for w in warnings]

    record = RunRecord(
        meta=RunMeta(
            run_id=run_id, domain=profile.domain, procedure=profile.procedure,
            source_format=adapter.source_format, instructor=meta.get("instructor"),
            environment=environment or meta.get("environment"),
            n_samples=len(df), duration_s=float(df["t"].iloc[-1]),
        ),
        phases=phases, events=events, annotations=annotations, flags=flags,
    )
    return record, df


def ingest_run(source_dir: Path, profile: Profile, run_id: str | None = None,
               environment: str | None = None) -> RunRecord:
    """`environment` tags the run (e.g. "new-obstacle"); defaults to meta.json's value."""
    record, df = analyse_run(source_dir, profile, run_id, environment)
    save_run(record, df)
    return record
