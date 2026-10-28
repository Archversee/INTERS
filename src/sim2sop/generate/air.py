"""Generate C172 traffic-pattern runs in MSFS SimConnect format.

Output per run (data/raw/air/<run_id>/):
  telemetry.csv     columns named like Python-SimConnect SimVars, in SimConnect units
                    (radians for angles/lat/lon, ft/s for vertical speed)
  annotations.json  spoken intent callouts [{t, text, source}]
  meta.json         domain / procedure / source format

The flight is a scripted kinematic model, not a flight-dynamics model: targets for
airspeed, vertical speed, heading and throttle are set by the script and the state
follows them with simple lags. Speeds and heights follow C172 POH-style values with
run-to-run variation and sensor noise, so consensus and typical ranges have something
to learn from.

Scenario hooks for the demo:
  variant 0       selects flaps 20 at about 88 kt, above Vfe 85 KIAS (a limit violation)
  obstacle=True   "new obstacle" environment: wider downwind (longer crosswind leg) and
                  a later descent, so the final approach is unchanged
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

DT = 0.2  # 5 Hz

# Seletar-like field. Positions are illustrative, not survey data.
FIELD_LAT, FIELD_LON, FIELD_ELEV_FT = 1.4167, 103.8678, 36.0
RWY_HDG = 30.0


def _wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


class _Sim:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.t = 0.0
        self.x = 0.0  # metres east
        self.y = 0.0  # metres north
        self.agl = 0.0
        self.ias = 0.0
        self.vs = 0.0
        self.hdg = (RWY_HDG - 90.0) % 360.0
        self.throttle = 0.0
        self.rpm = 700.0
        self.bank = 0.0
        self.flaps = 0
        self.on_ground = 1
        self.brake = 1
        self.light_landing = 0
        self.light_taxi = 0
        self.light_strobe = 0
        self.wind_kt = float(rng.uniform(3, 9))
        self.wind_dir = float((RWY_HDG + rng.normal(0, 20)) % 360)
        # targets
        self.t_ias = 0.0
        self.t_vs = 0.0
        self.t_hdg = self.hdg
        self.t_throttle = 0.0
        self.rows: list[dict] = []
        self.notes: list[dict] = []
        self.origin = (0.0, 0.0)  # take-off roll start, for the runway frame
        self.scheduled: list[tuple[float, str, float]] = []

    # --- script helpers ---------------------------------------------------------------
    def say(self, text: str) -> None:
        """Spoken callout; people speak slightly after (sometimes before) acting."""
        lag = float(np.clip(self.rng.normal(0.8, 0.6), -0.5, 2.5))
        self.notes.append({"t": round(self.t + lag, 2), "text": text, "source": "spoken"})

    def hold(self, seconds: float) -> None:
        for _ in range(max(1, int(round(seconds / DT)))):
            self.step()

    def until(self, cond, max_s: float = 900.0) -> None:
        end = self.t + max_s
        while not cond() and self.t < end:
            self.step()

    def at(self, delay_s: float, **targets: float) -> None:
        """Change targets later, e.g. s.at(20, t_vs=-400)."""
        self.scheduled += [(self.t + delay_s, k, v) for k, v in targets.items()]

    def offset_nm(self) -> float:
        """Distance from the extended runway centreline."""
        dx, dy = self.x - self.origin[0], self.y - self.origin[1]
        h = math.radians(RWY_HDG)
        return abs(dx * math.cos(h) - dy * math.sin(h)) / 1852.0

    def turn_to(self, hdg: float) -> None:
        self.t_hdg = hdg % 360.0
        self.until(lambda: abs(_wrap180(self.t_hdg - self.hdg)) < 1.0, max_s=120)

    # --- dynamics ----------------------------------------------------------------------
    def step(self) -> None:
        r = self.rng
        for item in [x for x in self.scheduled if x[0] <= self.t]:
            setattr(self, item[1], item[2])
            self.scheduled.remove(item)
        self.throttle += np.clip(self.t_throttle - self.throttle, -40 * DT, 40 * DT)
        rpm_target = 700 + 18.0 * self.throttle + (80 if not self.on_ground else 0)
        self.rpm += (rpm_target - self.rpm) * DT / 1.5

        if self.on_ground:
            self.ias += np.clip(self.t_ias - self.ias, -3.0 * DT, 4.0 * DT)
            self.vs, self.agl = 0.0, 0.0
            max_rate = 8.0
        else:
            self.ias += (self.t_ias - self.ias) * DT / 5.0
            self.vs += (self.t_vs - self.vs) * DT / 3.0
            self.agl = max(0.0, self.agl + self.vs / 60.0 * DT)
            max_rate = 3.0  # standard-rate turn

        err = _wrap180(self.t_hdg - self.hdg)
        rate = float(np.clip(0.6 * err, -max_rate, max_rate))
        if self.on_ground and self.ias < 2:
            rate = 0.0
        self.hdg = (self.hdg + rate * DT) % 360.0
        bank_target = 0.0 if self.on_ground else 7.0 * rate
        self.bank += (bank_target - self.bank) * DT / 1.0

        gs = max(0.0, self.ias - self.wind_kt * math.cos(math.radians(self.wind_dir - self.hdg)) * (self.ias > 30))
        d = gs * 0.514444 * DT
        self.x += d * math.sin(math.radians(self.hdg))
        self.y += d * math.cos(math.radians(self.hdg))
        self.t += DT

        pitch = 0.0 if self.on_ground else 0.006 * self.vs + 2.0 - 1.5 * self.flaps
        self.rows.append({
            "t": self.t, "x": self.x, "y": self.y,
            "agl": self.agl + (r.normal(0, 3) if not self.on_ground else 0.0),
            "ias": max(0.0, self.ias + r.normal(0, 0.4)),
            "gs": gs, "vs": self.vs + r.normal(0, 25) * (not self.on_ground),
            "hdg": (self.hdg + r.normal(0, 0.3)) % 360.0,
            "pitch": pitch + r.normal(0, 0.3), "bank": self.bank + r.normal(0, 0.5),
            "throttle": self.throttle, "rpm": self.rpm + r.normal(0, 10),
            "elevator": float(np.clip(pitch / 15.0, -1, 1)),
            "aileron": float(np.clip((bank_target - self.bank) / 30.0, -1, 1)),
            "flaps": self.flaps, "on_ground": self.on_ground, "brake": self.brake,
            "light_landing": self.light_landing, "light_taxi": self.light_taxi,
            "light_strobe": self.light_strobe, "wind_kt": self.wind_kt, "wind_dir": self.wind_dir,
        })


def simulate_circuit(seed: int, variant: int = 0, obstacle: bool = False) -> tuple[pd.DataFrame, list[dict]]:
    """Fly one circuit. `variant` adds technique differences between runs; `obstacle`
    flies the wider pattern used after the new-obstacle NOTAM."""
    rng = np.random.default_rng(seed)
    p = {
        "vr": rng.normal(55, 1.2),
        "vy": rng.normal(74, 1.5),
        "crosswind_agl": rng.normal(700, 35),
        "pattern_agl": rng.normal(1000, 15),
        "downwind_ias": rng.normal(92, 2.5),
        "cruise_power": rng.normal(68, 3),
        "abeam_power": rng.normal(45, 3),
        "flaps10_ias": rng.normal(88, 2),
        "approach_ias": rng.normal(70, 1.5),
        "final_ias": rng.normal(65, 1.2),
        "final_turn_nm": rng.normal(0.38, 0.02),
    }
    overspeed = variant == 0
    s = _Sim(rng)

    # Taxi
    s.light_taxi = 1
    s.hold(2 + rng.uniform(0, 1))
    s.brake = 0
    s.say(rng.choice(["Brakes off, taxiing", "Parking brake released, taxi"]))
    s.t_throttle, s.t_ias = 25, 10
    s.hold(rng.normal(20, 3))
    s.t_ias = 6
    s.turn_to(RWY_HDG)
    s.t_ias, s.t_throttle = 0, 10
    s.hold(4)

    # Line up and take off
    s.light_strobe = 1
    if variant % 2 == 0:
        s.light_landing = 1
        s.say("Lights on, lined up")
    s.hold(rng.uniform(2, 4))
    s.t_throttle, s.t_ias = 100, 80
    s.origin = (s.x, s.y)
    s.say(rng.choice(["Full power, rolling", "Full throttle, rolling"]))
    if variant % 2 == 1:
        s.hold(2)
        s.light_landing = 1
    s.until(lambda: s.ias >= p["vr"])
    s.say("Rotating")
    s.hold(1.5)
    s.on_ground = 0
    s.t_vs, s.t_ias = 700, p["vy"]

    # Climb, crosswind
    s.until(lambda: s.agl >= p["crosswind_agl"])
    s.say("Turning crosswind")
    t_cw = s.t
    s.t_hdg = RWY_HDG - 90
    s.until(lambda: s.agl >= p["pattern_agl"] - 40)
    s.t_vs, s.t_ias, s.t_throttle = 0, p["downwind_ias"], p["cruise_power"]
    s.say(rng.choice(["Level at circuit height, power set", "Circuit altitude, cruise power"]))
    crosswind_leg_s = rng.normal(45, 3) + (25 if obstacle else 0)
    s.until(lambda: s.t - t_cw >= crosswind_leg_s)

    # Downwind
    s.say("Turning downwind")
    s.turn_to(RWY_HDG + 180)
    if variant == 2:
        s.hold(12)
        s.say("Checking for traffic")  # no matching control input: should be flagged
    s.hold(max(5.0, rng.normal(30, 4)))
    s.t_throttle, s.t_ias = p["abeam_power"], 82
    s.say(rng.choice(["Abeam the numbers, power back", "Abeam threshold, power 1500"]))
    s.until(lambda: s.ias <= p["flaps10_ias"])
    s.flaps = 1
    s.say(rng.choice(["Flaps 10 on downwind", "Flaps ten"]))
    s.t_ias = 90 if overspeed else 80
    if obstacle:
        # Longer base leg ahead: hold circuit height for longer before descending.
        s.at(28, t_vs=-400)
    else:
        s.t_vs = -400
    s.hold(rng.normal(18, 3))

    # Base
    s.say("Turning base")
    s.turn_to(RWY_HDG + 90)
    s.hold(3.5 if overspeed else 2)
    s.flaps = 2
    s.say("Flaps 20")
    s.t_ias = p["approach_ias"]
    s.until(lambda: s.offset_nm() <= p["final_turn_nm"], max_s=120)

    # Final
    s.say("Turning final")
    s.turn_to(RWY_HDG)
    s.hold(2)
    s.flaps = 3
    s.say(rng.choice(["Flaps 30, on speed", "Full flap, sixty-five knots"]))
    s.t_ias, s.t_vs = p["final_ias"], -450
    if variant in (1, 3):
        # Technique difference: a power correction on short final.
        s.until(lambda: s.agl <= 250)
        s.t_throttle = s.throttle + 15
        s.hold(6)
        s.t_throttle = p["abeam_power"] - 10
    s.until(lambda: s.agl <= 60)
    s.t_throttle = 0
    s.until(lambda: s.agl <= 15)
    s.t_vs, s.t_ias = -120, 50
    s.until(lambda: s.agl <= 0.5, max_s=60)
    s.on_ground, s.agl, s.vs = 1, 0.0, 0.0
    s.say("Touchdown")

    # Roll-out and taxi clear
    s.t_ias = 8
    s.until(lambda: s.ias <= 12, max_s=60)
    s.t_hdg = RWY_HDG + 90
    s.hold(4)
    s.flaps = 0
    s.light_landing, s.light_strobe = 0, 0
    s.say("Flaps up, clear of the runway")
    s.t_throttle = 20
    s.hold(12)
    s.t_ias, s.t_throttle = 0, 0
    s.hold(5)
    s.brake = 1
    s.say("Parking brake set")
    s.hold(3)
    return pd.DataFrame(s.rows), s.notes


def to_simconnect(df: pd.DataFrame) -> pd.DataFrame:
    """Rename / convert the internal state to Python-SimConnect SimVar names and units."""
    lat0 = math.radians(FIELD_LAT)
    lat = lat0 + df["y"] / 6_371_000.0
    lon = math.radians(FIELD_LON) + df["x"] / (6_371_000.0 * math.cos(lat0))
    return pd.DataFrame({
        "SIMULATION_TIME": df["t"].round(2),
        "PLANE_LATITUDE": lat,
        "PLANE_LONGITUDE": lon,
        "PLANE_ALTITUDE": FIELD_ELEV_FT + df["agl"],
        "PLANE_ALT_ABOVE_GROUND": df["agl"],
        "PLANE_HEADING_DEGREES_MAGNETIC": np.radians(df["hdg"]),
        "PLANE_PITCH_DEGREES": -np.radians(df["pitch"]),   # SimConnect: nose-up is negative
        "PLANE_BANK_DEGREES": -np.radians(df["bank"]),     # SimConnect: right bank is negative
        "AIRSPEED_INDICATED": df["ias"],
        "GROUND_VELOCITY": df["gs"],
        "VERTICAL_SPEED": df["vs"] / 60.0,                 # ft/s
        "GENERAL_ENG_THROTTLE_LEVER_POSITION:1": df["throttle"],
        "GENERAL_ENG_RPM:1": df["rpm"],
        "ELEVATOR_POSITION": df["elevator"],
        "AILERON_POSITION": df["aileron"],
        "FLAPS_HANDLE_INDEX": df["flaps"],
        "SIM_ON_GROUND": df["on_ground"],
        "BRAKE_PARKING_POSITION": df["brake"],
        "LIGHT_LANDING": df["light_landing"],
        "LIGHT_TAXI": df["light_taxi"],
        "LIGHT_STROBE": df["light_strobe"],
        "AMBIENT_WIND_VELOCITY": df["wind_kt"],
        "AMBIENT_WIND_DIRECTION": df["wind_dir"],
    })


def _write_run(out_dir: Path, run_id: str, df: pd.DataFrame, notes: list[dict], meta: dict) -> Path:
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    to_simconnect(df).to_csv(run_dir / "telemetry.csv", index=False)
    (run_dir / "annotations.json").write_text(json.dumps(notes, indent=2))
    (run_dir / "meta.json").write_text(json.dumps({
        "run_id": run_id, "domain": "air", "procedure": "c172_circuit",
        "source_format": "msfs_simconnect", "generated": True, **meta}, indent=2))
    return run_dir


def generate_air_runs(out_dir: Path, n: int = 5, seed: int = 172) -> list[Path]:
    """Baseline circuits. Run 1 (variant 0) selects flaps 20 above Vfe."""
    paths = []
    for i in range(n):
        df, notes = simulate_circuit(seed + i, variant=i)
        paths.append(_write_run(out_dir, f"air_c172_circuit_{i + 1:02d}", df, notes, {
            "instructor": f"instructor_{(i % 2) + 1}", "seed": seed + i, "variant": i}))
    return paths


OBSTACLE_ENV = "new-obstacle"


def generate_obstacle_runs(out_dir: Path, n: int = 3, seed: int = 500) -> list[Path]:
    """Circuits flown after the new-obstacle NOTAM, tagged with its environment."""
    paths = []
    for i in range(n):
        variant = 4 + 2 * i  # even variants: no short-final power correction or extra callouts
        df, notes = simulate_circuit(seed + i, variant=variant, obstacle=True)
        paths.append(_write_run(out_dir, f"air_c172_circuit_obstacle_{i + 1:02d}", df, notes, {
            "instructor": f"instructor_{(i % 2) + 1}", "seed": seed + i, "variant": variant,
            "environment": OBSTACLE_ENV}))
    return paths
