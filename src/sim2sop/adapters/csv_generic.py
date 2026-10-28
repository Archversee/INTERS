"""Generic CSV fallback with a column-mapping step.

`mapping.yaml` next to the CSV maps source columns to canonical names, with an optional
linear conversion:

    time: Time_s
    columns:
      Airspeed: ias_kt
      Altitude_m: {name: alt_ft, scale: 3.28084}
    annotation_column: Note     # optional typed-annotation column

Rows whose annotation column is non-empty become typed annotations.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from sim2sop.adapters.base import Adapter, find_file
from sim2sop.schema import Annotation


class GenericCSVAdapter(Adapter):
    name = "csv"
    source_format = "csv"

    def _mapping(self, run_dir: Path) -> dict:
        path = run_dir / "mapping.yaml"
        return yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}

    def _raw(self, run_dir: Path) -> pd.DataFrame:
        return pd.read_csv(find_file(run_dir, "telemetry.csv", "*.csv"))

    def read_telemetry(self, run_dir: Path) -> pd.DataFrame:
        raw, m = self._raw(run_dir), self._mapping(run_dir)
        out = pd.DataFrame({"t": raw[m.get("time", "t")].astype(float)})
        cols = m.get("columns") or {c: c for c in raw.columns if c != m.get("time", "t")}
        for src, dst in cols.items():
            if src not in raw or not pd.api.types.is_numeric_dtype(raw[src]):
                continue
            if isinstance(dst, dict):
                out[dst["name"]] = raw[src] * dst.get("scale", 1.0) + dst.get("offset", 0.0)
            else:
                out[dst] = raw[src]
        return out

    def read_annotations(self, run_dir: Path, df_raw=None) -> list[Annotation]:
        notes = super().read_annotations(run_dir)
        m = self._mapping(run_dir)
        col = m.get("annotation_column")
        if col:
            raw = self._raw(run_dir)
            for t, text in zip(raw[m.get("time", "t")], raw[col]):
                if isinstance(text, str) and text.strip():
                    notes.append(Annotation(t=float(t), text=text.strip(), source="typed"))
        return sorted(notes, key=lambda a: a.t)
