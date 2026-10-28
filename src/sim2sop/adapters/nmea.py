"""NMEA 0183 adapter for bridge-simulator logs, parsed with pynmea2.

Reads a `.nmea`/`.log`/`.txt` file of sentences. Uses RMC (time, position, SOG, COG),
HDT (heading), RSA (rudder angle) and RPM (engine/shaft RPM, used as the engine order).
Sentences are merged on their timestamp; the latest value of each field is carried
forward. Sentences without a time of their own take the most recent RMC time.

TODO(data): not yet exercised by generated data; build the sea generator against this.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pynmea2

from sim2sop.adapters.base import Adapter, find_file


class NMEAAdapter(Adapter):
    name = "nmea"
    source_format = "nmea0183"

    def read_telemetry(self, run_dir: Path) -> pd.DataFrame:
        path = find_file(run_dir, "*.nmea", "*.log", "*.txt")
        rows: list[dict] = []
        state: dict = {}
        t_cur: float | None = None
        t_start: datetime | None = None
        for line in path.read_text(encoding="ascii", errors="ignore").splitlines():
            try:
                msg = pynmea2.parse(line.strip())
            except pynmea2.ParseError:
                continue
            kind = msg.sentence_type
            if kind == "RMC" and msg.datestamp and msg.timestamp:
                stamp = datetime.combine(msg.datestamp, msg.timestamp)
                t_start = t_start or stamp
                t_cur = (stamp - t_start).total_seconds()
                state.update(lat_deg=msg.latitude, lon_deg=msg.longitude,
                             sog_kt=_f(msg.spd_over_grnd), cog_deg=_f(msg.true_course))
            elif kind == "HDT":
                state["heading_deg"] = _f(msg.heading)
            elif kind == "RSA":
                state["rudder_deg"] = _f(msg.rsa_starboard)
            elif kind == "RPM":
                state["engine_rpm"] = _f(msg.speed)
            else:
                continue
            if t_cur is not None:
                rows.append({"t": t_cur, **state})
        if not rows:
            raise ValueError(f"No usable NMEA sentences in {path}")
        return pd.DataFrame(rows).groupby("t", as_index=False).last()


def _f(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
