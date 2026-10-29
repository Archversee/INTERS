"""CARLA adapter: per-tick vehicle state logged from the CARLA Python API.

Expected CSV columns (one row per world tick), named after the CARLA API fields:
  timestamp                         world snapshot elapsed_seconds
  location.x, location.y            m (CARLA is left-handed: +y is to the right)
  rotation.yaw                      degrees
  velocity.x, velocity.y            m/s
  control.throttle, control.brake   0..1
  control.steer                     -1..1
  control.gear, control.reverse, control.hand_brake
  light_state                       carla.VehicleLightState bitmask (optional)

Speed is a magnitude (reverse says which way). steer_lock turns the steering input into
the positions a driving instructor talks about: -1 full left lock, 0 straight, 1 full right
lock (in between holds the last of those).
TODO(data): confirm MAX_STEER_DEG and the light_state bits against a real capture.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sim2sop.adapters.base import Adapter, find_file

MAX_STEER_DEG = 70.0  # front-wheel angle at steer = 1.0; vehicle-specific
LEFT_BLINKER, RIGHT_BLINKER = 0x20, 0x40  # carla.VehicleLightState bits
LOCK, STRAIGHT = 0.5, 0.1   # |control.steer| counted as full lock / wheels straight


class CarlaAdapter(Adapter):
    name = "carla"
    source_format = "carla_vehicle_state"

    def read_telemetry(self, run_dir: Path) -> pd.DataFrame:
        raw = pd.read_csv(find_file(run_dir, "telemetry.csv", "*.csv"))
        def col(name: str) -> pd.Series:
            return raw[name].astype(float) if name in raw else pd.Series(0.0, index=raw.index)

        speed = np.hypot(raw["velocity.x"], raw["velocity.y"]) * 3.6
        reverse = col("control.reverse")
        steer = raw["control.steer"].astype(float)
        lock = pd.Series(np.select([steer <= -LOCK, steer >= LOCK, steer.abs() <= STRAIGHT], [-1.0, 1.0, 0.0],
                                   np.nan), index=raw.index).ffill().fillna(0.0)
        out = pd.DataFrame({
            "t": raw["timestamp"].astype(float),
            "x_m": raw["location.x"],
            "y_m": -raw["location.y"],  # to right-handed
            "heading_deg": (-raw["rotation.yaw"] + 90.0) % 360.0,
            "speed_kmh": speed,
            "throttle": raw["control.throttle"],
            "brake": raw["control.brake"],
            "steer_deg": steer * MAX_STEER_DEG,
            "steer_lock": lock,
            "gear": col("control.gear"),
            "reverse": reverse,
            "hand_brake": col("control.hand_brake"),
        })
        if "light_state" in raw:
            ls = raw["light_state"].astype(int)
            out["indicator_left"] = ((ls & LEFT_BLINKER) > 0).astype(float)
            out["indicator_right"] = ((ls & RIGHT_BLINKER) > 0).astype(float)
        return out
