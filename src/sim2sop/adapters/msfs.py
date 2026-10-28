"""MSFS adapter: CSV/Parquet logged from SimConnect with Python-SimConnect SimVar names.

Units are SimConnect defaults as returned by Python-SimConnect: lat/lon and attitude in
radians, vertical speed in ft/s. SimConnect reports nose-up pitch and right bank as
negative; both are flipped here so positive = nose up / right bank.
TODO(data): confirm sign conventions against a real capture.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sim2sop.adapters.base import Adapter, find_file

# SimVar -> (canonical name, conversion)
SIMVARS = {
    "PLANE_LATITUDE": ("lat_deg", np.degrees),
    "PLANE_LONGITUDE": ("lon_deg", np.degrees),
    "PLANE_ALTITUDE": ("alt_ft", None),
    "PLANE_ALT_ABOVE_GROUND": ("agl_ft", None),
    "PLANE_HEADING_DEGREES_MAGNETIC": ("heading_deg", lambda r: np.degrees(r) % 360.0),
    "PLANE_PITCH_DEGREES": ("pitch_deg", lambda r: -np.degrees(r)),
    "PLANE_BANK_DEGREES": ("bank_deg", lambda r: -np.degrees(r)),
    "AIRSPEED_INDICATED": ("ias_kt", None),
    "GROUND_VELOCITY": ("gs_kt", None),
    "VERTICAL_SPEED": ("vs_fpm", lambda v: v * 60.0),
    "GENERAL_ENG_THROTTLE_LEVER_POSITION:1": ("throttle_pct", None),
    "GENERAL_ENG_RPM:1": ("rpm", None),
    "ELEVATOR_POSITION": ("elevator", None),
    "AILERON_POSITION": ("aileron", None),
    "FLAPS_HANDLE_INDEX": ("flaps_idx", None),
    "SIM_ON_GROUND": ("on_ground", None),
    "BRAKE_PARKING_POSITION": ("parking_brake", None),
    "LIGHT_LANDING": ("light_landing", None),
    "LIGHT_TAXI": ("light_taxi", None),
    "LIGHT_STROBE": ("light_strobe", None),
    "AMBIENT_WIND_VELOCITY": ("wind_kt", None),
    "AMBIENT_WIND_DIRECTION": ("wind_dir_deg", None),
}


class MSFSAdapter(Adapter):
    name = "msfs"
    source_format = "msfs_simconnect"

    def read_telemetry(self, run_dir: Path) -> pd.DataFrame:
        path = find_file(run_dir, "telemetry.parquet", "telemetry.csv", "*.parquet", "*.csv")
        raw = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        out = pd.DataFrame({"t": raw["SIMULATION_TIME"].astype(float)})
        for simvar, (name, conv) in SIMVARS.items():
            if simvar in raw:
                col = raw[simvar].astype(float)
                out[name] = conv(col) if conv else col
        return out
