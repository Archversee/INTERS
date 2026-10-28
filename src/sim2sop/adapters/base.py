"""Adapter interface: raw source files -> common schema.

A run directory holds the source telemetry file, optional `annotations.json`
([{t, text, source}], times in the source clock) and optional `meta.json`.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from sim2sop.schema import Annotation


class Adapter(ABC):
    name: str = "base"
    source_format: str = "unknown"

    @abstractmethod
    def read_telemetry(self, run_dir: Path) -> pd.DataFrame:
        """Return a DataFrame with `t` (seconds, source clock) and canonical signal columns."""

    def read_annotations(self, run_dir: Path, df_raw: pd.DataFrame | None = None) -> list[Annotation]:
        path = run_dir / "annotations.json"
        if path.exists():
            return [Annotation.model_validate(a) for a in json.loads(path.read_text(encoding="utf-8"))]
        return []

    def read_meta(self, run_dir: Path) -> dict:
        path = run_dir / "meta.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def load(self, run_dir: Path) -> tuple[pd.DataFrame, list[Annotation], dict]:
        run_dir = Path(run_dir)
        df = self.read_telemetry(run_dir)
        notes = self.read_annotations(run_dir)
        t0 = float(df["t"].min())
        # Annotation times share the source clock; shift both to start at zero.
        df["t"] = df["t"] - t0
        for a in notes:
            a.t = round(a.t - t0, 3)
        return df, notes, self.read_meta(run_dir)


def find_file(run_dir: Path, *patterns: str) -> Path:
    for pattern in patterns:
        hits = sorted(run_dir.glob(pattern))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"No file matching {patterns} in {run_dir}")
