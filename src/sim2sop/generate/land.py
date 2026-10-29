"""Generate parallel-parking runs in CARLA vehicle-state format.

Output per run (data/raw/land/<run_id>/):
  telemetry.csv     one row per world tick, columns named after the CARLA Python API fields
                    (left-handed: +y is to the right, yaw in degrees)
  annotations.json  the instructor's spoken callouts [{t, text, source}]
  meta.json         domain / procedure / source format

Driving on the left: the car drives up the road, stops alongside the parked car, then
reverses into the space behind it on the left kerb (full left lock, straighten at about
45°, full right lock, wheels straight). A kinematic bicycle model, not a vehicle-dynamics
model: speed and steering follow the script with simple rate limits, with run-to-run
variation and sensor noise so consensus and typical ranges have something to learn from.

Scenario hook for the demo: variant 0 reverses at about 7 km/h, above the 5 km/h limit.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

DT = 0.1                 # 10 Hz
WHEELBASE_M = 2.7
WHEEL_ANGLE_DEG = 35.0   # road-wheel angle at full lock (kinematics only)
STEER_RATE = 1.5         # full lock per second
LEFT_BLINKER, RIGHT_BLINKER = 0x20, 0x40   # carla.VehicleLightState bits


class _Car:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.t = 0.0
        self.x = 0.0      # metres east (right-handed; road runs north)
        self.y = 0.0      # metres north
        self.hdg = 0.0    # degrees clockwise from north
        self.v = 0.0      # m/s, + forwards
        self.steer = 0.0  # -1 full left .. 1 full right
        self.throttle = 0.0
        self.brake = 1.0
        self.reverse = 0
        self.hand_brake = 1
        self.indicator_left = 0
        self.indicator_right = 0
        # targets
        self.t_speed = 0.0   # km/h, magnitude; direction comes from the gear
        self.t_steer = 0.0
        self.rows: list[dict] = []
        self.notes: list[dict] = []
        self.anchor = (0.0, 0.0, 0.0)   # x, y, heading where reverse was selected

    # --- script helpers ---------------------------------------------------------------
    def say(self, text: str) -> None:
        lag = float(np.clip(self.rng.normal(0.6, 0.4), -0.4, 1.8))
        self.notes.append({"t": round(self.t + lag, 2), "text": text, "source": "spoken"})

    def hold(self, seconds: float) -> None:
        for _ in range(max(1, int(round(seconds / DT)))):
            self.step()

    def until(self, cond, max_s: float = 120.0) -> None:
        end = self.t + max_s
        while not cond() and self.t < end:
            self.step()

    def stop(self) -> None:
        self.t_speed = 0.0
        self.until(lambda: abs(self.v) < 0.02, max_s=20)
        self.v = 0.0

    def backed_m(self) -> float:
        """Distance reversed since reverse was selected, along the original heading."""
        x0, y0, h0 = self.anchor
        h = math.radians(h0)
        return -((self.x - x0) * math.sin(h) + (self.y - y0) * math.cos(h))

    def angle(self) -> float:
        """Angle to the kerb (the heading when reverse was selected); + = rear swung left."""
        return (self.hdg - self.anchor[2] + 180.0) % 360.0 - 180.0

    # --- dynamics ----------------------------------------------------------------------
    def step(self) -> None:
        r = self.rng
        self.steer += float(np.clip(self.t_steer - self.steer, -STEER_RATE * DT, STEER_RATE * DT))
        target = self.t_speed / 3.6 * (-1.0 if self.reverse else 1.0)
        if self.hand_brake:
            target = 0.0
        slowing = abs(target) < abs(self.v) - 0.05 or target == 0.0
        accel = 2.5 if slowing else 1.2
        self.v += float(np.clip(target - self.v, -accel * DT, accel * DT))
        self.throttle = 0.0 if slowing else 0.2 + 0.01 * abs(target) * 3.6
        self.brake = (0.4 if abs(self.v) > 0.01 else 1.0) if slowing else 0.0

        delta = math.radians(self.steer * WHEEL_ANGLE_DEG)
        self.hdg = (self.hdg + math.degrees(self.v / WHEELBASE_M * math.tan(delta)) * DT) % 360.0
        self.x += self.v * math.sin(math.radians(self.hdg)) * DT
        self.y += self.v * math.cos(math.radians(self.hdg)) * DT
        self.t += DT

        moving = abs(self.v) > 0.01
        self.rows.append({
            "t": self.t, "x": self.x + r.normal(0, 0.01), "y": self.y + r.normal(0, 0.01),
            "hdg": (self.hdg + r.normal(0, 0.15)) % 360.0,
            "vx": self.v * math.sin(math.radians(self.hdg)) + r.normal(0, 0.02) * moving,
            "vy": self.v * math.cos(math.radians(self.hdg)) + r.normal(0, 0.02) * moving,
            "throttle": self.throttle, "brake": self.brake, "steer": self.steer,
            "gear": -1 if self.reverse else 1, "reverse": self.reverse, "hand_brake": self.hand_brake,
            "light_state": LEFT_BLINKER * self.indicator_left + RIGHT_BLINKER * self.indicator_right,
        })


def simulate_park(seed: int, variant: int = 0, *,
                  reverse_fast: bool | None = None, late_indicator: bool | None = None,
                  blind_spot_check: bool | None = None,
                  forward_adjust: bool | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """Drive up and reverse-park once. `variant` picks the technique differences between
    runs unless they are given explicitly."""
    reverse_fast = variant == 0 if reverse_fast is None else reverse_fast
    late_indicator = variant % 2 == 1 if late_indicator is None else late_indicator
    blind_spot_check = variant == 2 if blind_spot_check is None else blind_spot_check
    forward_adjust = variant in (1, 3) if forward_adjust is None else forward_adjust
    rng = np.random.default_rng(seed)
    p = {
        "approach_kmh": rng.normal(15, 1.5),
        "approach_m": rng.normal(28, 3),
        "signal_before_m": rng.normal(10, 1.5),
        "reverse_kmh": rng.normal(7.2, 0.4) if reverse_fast else rng.normal(3.0, 0.4),
        "lock_at_m": rng.normal(1.8, 0.2),
        "straighten_deg": rng.normal(43, 2.5),
        "straight_m": rng.normal(0.8, 0.15),
    }
    c = _Car(rng)

    # Move off and drive up to the parked car
    c.hold(2 + rng.uniform(0, 1))
    c.hand_brake = 0
    c.say(rng.choice(["Handbrake off, moving off", "Releasing the handbrake"]))
    c.t_speed = p["approach_kmh"]
    c.until(lambda: c.y >= p["approach_m"] - p["signal_before_m"])
    if not late_indicator:
        c.indicator_left = 1
        c.say(rng.choice(["Signal left", "Indicating left"]))
    c.until(lambda: c.y >= p["approach_m"] - 4.0)
    c.t_speed = 4.0
    c.until(lambda: c.y >= p["approach_m"] - 0.6)
    c.stop()
    c.say(rng.choice(["Stop alongside the parked car", "Stopping level with the car in front"]))
    c.hold(rng.uniform(2.5, 3.5))
    if late_indicator:
        c.indicator_left = 1
        c.say("Signal on")
        c.hold(1)

    # Reverse in
    c.reverse = 1
    c.anchor = (c.x, c.y, c.hdg)
    c.say(rng.choice(["Reverse gear", "Into reverse"]))
    c.hold(1.5)
    if blind_spot_check:
        c.say("Checking mirrors and blind spot")   # no matching control input: should be flagged
    c.hold(rng.uniform(1, 2))
    c.t_speed = p["reverse_kmh"]
    c.until(lambda: c.backed_m() >= p["lock_at_m"])
    c.t_steer = -1.0
    c.say(rng.choice(["Full left lock", "Steer full left"]))
    c.until(lambda: c.angle() >= p["straighten_deg"] - 6)
    c.t_steer = 0.0
    c.say(rng.choice(["Straighten up", "Wheels straight, forty-five degrees"]))
    start, t0 = c.backed_m(), c.t
    c.until(lambda: c.backed_m() >= start + p["straight_m"] and c.t >= t0 + 1.0)
    c.t_steer = 1.0
    c.say(rng.choice(["Full right lock", "Steer right to bring the front in"]))
    # Wheels straight a little before parallel: the car keeps turning while they unwind.
    c.until(lambda: c.angle() <= 1.1 * p["reverse_kmh"] + rng.normal(0, 0.3))
    c.t_steer = 0.0
    c.say("Wheels straight")
    c.hold(1.0)
    c.stop()
    c.say(rng.choice(["Stop, parallel to the kerb", "And stop"]))
    c.hold(rng.uniform(2.5, 3.5))

    if forward_adjust:
        # Technique difference: shuffle forward to centre the car in the space.
        c.reverse = 0
        c.say("Drive, moving up a little")
        c.hold(1)
        c.t_speed = 2.0
        c.hold(rng.normal(2.5, 0.4))
        c.stop()
        c.hold(rng.uniform(2.5, 3.5))

    # Secure
    c.hand_brake = 1
    c.say(rng.choice(["Handbrake on", "Handbrake applied"]))
    c.hold(1.5)
    c.indicator_left = 0
    c.say("Signal off, parked")
    c.hold(3)
    return pd.DataFrame(c.rows), c.notes


def to_carla(df: pd.DataFrame) -> pd.DataFrame:
    """Write the internal state with CARLA API field names and its left-handed frame."""
    return pd.DataFrame({
        "timestamp": df["t"].round(2),
        "location.x": df["x"].round(3),
        "location.y": (-df["y"]).round(3),                  # CARLA: +y is to the right
        "rotation.yaw": ((90.0 - df["hdg"]) % 360.0).round(2),   # CARLA: yaw 0 = +x, clockwise
        "velocity.x": df["vx"].round(3),
        "velocity.y": (-df["vy"]).round(3),
        "control.throttle": df["throttle"].round(3),
        "control.brake": df["brake"].round(3),
        "control.steer": df["steer"].round(3),
        "control.gear": df["gear"],
        "control.reverse": df["reverse"],
        "control.hand_brake": df["hand_brake"],
        "light_state": df["light_state"].astype(int),
    })


def _write_run(out_dir: Path, run_id: str, df: pd.DataFrame, notes: list[dict], meta: dict) -> Path:
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    to_carla(df).to_csv(run_dir / "telemetry.csv", index=False)
    (run_dir / "annotations.json").write_text(json.dumps(notes, indent=2))
    (run_dir / "meta.json").write_text(json.dumps({
        "run_id": run_id, "domain": "land", "procedure": "parallel_park",
        "source_format": "carla_vehicle_state", "generated": True, **meta}, indent=2))
    return run_dir


def generate_land_runs(out_dir: Path, n: int = 5, seed: int = 500) -> list[Path]:
    """Baseline parks. Run 1 (variant 0) reverses faster than the limit."""
    paths = []
    for i in range(n):
        df, notes = simulate_park(seed + i, variant=i)
        paths.append(_write_run(out_dir, f"land_parallel_park_{i + 1:02d}", df, notes, {
            "instructor": f"instructor_{(i % 2) + 1}", "seed": seed + i, "variant": i}))
    return paths


def generate_drive(out_dir: Path, seed: int, reverse_fast: bool = False, late_indicator: bool = False,
                   blind_spot_check: bool = False, forward_adjust: bool = False,
                   instructor: str | None = None) -> Path:
    """One drive with the technique choices made explicitly (the app's generator)."""
    choices = {"reverse_fast": reverse_fast, "late_indicator": late_indicator,
               "blind_spot_check": blind_spot_check, "forward_adjust": forward_adjust}
    df, notes = simulate_park(seed, **choices)
    meta = {"instructor": instructor or "generated", "seed": seed, "choices": choices}
    return _write_run(out_dir, f"generated_{seed}", df, notes, meta)
