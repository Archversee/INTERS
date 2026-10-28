"""Run library: Parquet telemetry + JSON run records, queried with DuckDB.

data/runs/<procedure>/<run_id>/
    telemetry.parquet   resampled common-schema telemetry
    run.json            RunRecord: meta, phases, events, annotations, flags
"""

from __future__ import annotations

import shutil
from pathlib import Path

import duckdb
import pandas as pd
from pydantic import BaseModel, Field

from sim2sop import config
from sim2sop.schema import Annotation, Event, Flag, Phase, RunMeta


class RunRecord(BaseModel):
    meta: RunMeta
    phases: list[Phase] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
    annotations: list[Annotation] = Field(default_factory=list)
    flags: list[Flag] = Field(default_factory=list)

    @property
    def run_id(self) -> str:
        return self.meta.run_id


def run_dir(procedure: str, run_id: str) -> Path:
    return config.RUNS_DIR / procedure / run_id


def save_run(record: RunRecord, telemetry: pd.DataFrame) -> Path:
    d = run_dir(record.meta.procedure, record.run_id)
    d.mkdir(parents=True, exist_ok=True)
    telemetry.to_parquet(d / "telemetry.parquet", index=False)
    (d / "run.json").write_text(record.model_dump_json(indent=2), encoding="utf-8")
    return d


def load_run(procedure: str, run_id: str) -> RunRecord:
    return RunRecord.model_validate_json((run_dir(procedure, run_id) / "run.json").read_text(encoding="utf-8"))


def load_telemetry(procedure: str, run_id: str) -> pd.DataFrame:
    return pd.read_parquet(run_dir(procedure, run_id) / "telemetry.parquet")


def list_runs(procedure: str) -> list[RunRecord]:
    base = config.RUNS_DIR / procedure
    if not base.exists():
        return []
    records = [RunRecord.model_validate_json(p.read_text(encoding="utf-8")) for p in base.glob("*/run.json")]
    return sorted(records, key=lambda r: (r.meta.created, r.run_id))


def delete_runs(procedure: str) -> None:
    shutil.rmtree(config.RUNS_DIR / procedure, ignore_errors=True)


def connect() -> duckdb.DuckDBPyConnection:
    """DuckDB over the whole run library. View `telemetry` has every sample of every run,
    with `procedure` and `run_id` columns taken from the path."""
    con = duckdb.connect()
    pattern = (config.RUNS_DIR / "*" / "*" / "telemetry.parquet").as_posix()
    if any(config.RUNS_DIR.glob("*/*/telemetry.parquet")):
        con.execute(f"""
            CREATE VIEW telemetry AS
            SELECT *,
                   split_part(replace(filename, '\\', '/'), '/', -3) AS procedure,
                   split_part(replace(filename, '\\', '/'), '/', -2) AS run_id
            FROM read_parquet('{pattern}', filename = true, union_by_name = true)
        """)
    return con


def signal_stats(procedure: str, signal: str) -> pd.DataFrame:
    """Per-run min / max / mean of one signal across the library (example DuckDB query)."""
    con = connect()
    return con.execute(f"""
        SELECT run_id, min("{signal}") AS min, max("{signal}") AS max, avg("{signal}") AS mean
        FROM telemetry WHERE procedure = ? GROUP BY run_id ORDER BY run_id
    """, [procedure]).df()
