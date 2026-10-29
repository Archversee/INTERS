"""Tools: generate a random instructor run (a flight, a drive), see what is in it, then add it or
download it. Also documents the file format, so real recordings can be ingested the same way."""

from __future__ import annotations

import io
import json
import random
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import ui

from sim2sop import config, references, storage
from sim2sop.generate.air import generate_flight
from sim2sop.generate.land import generate_drive
from sim2sop.ingest import analyse_run, ingest_run
from sim2sop.profiles import Profile
from sim2sop.signals.compliance import check_run


@dataclass
class Sim:
    """What the generator page needs to know about one domain's simulator."""

    make: Callable[..., Path]                  # generate_flight / generate_drive
    simulator: str                             # whose file format the run is written in
    choices: dict[str, tuple[str, str]]        # option -> (label, help)
    columns: dict[str, tuple[str, str]]        # telemetry column -> (what it is, unit in the file)
    columns_md: str                            # where the column names come from
    required_md: str                           # which columns are required
    example_csv: str
    callouts: list[dict]
    highlights: Callable
    charts: tuple[Callable, Callable]
    phase_row: Callable                        # (segment) -> extra columns of the phase table


# --- air: Cessna 172 circuit in MSFS SimConnect format ----------------------------------------

def _air_highlights(record, df: pd.DataFrame, meta: dict) -> list[str]:
    first = {}
    for e in record.events:
        first.setdefault(e.key, e)
    phase = {p.name: df[(df["t"] >= p.t_start) & (df["t"] < p.t_end)] for p in record.phases}
    out = []
    if e := first.get("toggle:on_ground:0"):
        out.append(f"Lifts off at **{e.state.get('ias_kt', 0):.0f} kt**")
    if (dw := phase.get("downwind")) is not None and len(dw):
        out.append(f"Downwind at **{dw['agl_ft'].median():,.0f} ft**, "
                   f"**{dw['track_offset_nm'].median():.1f} nm** from the runway, "
                   f"**{dw['ias_kt'].median():.0f} kt**")
    flaps = [f"{deg}° at {first[k].state.get('ias_kt', 0):.0f} kt" for deg, k in
             ((10, "toggle:flaps_idx:1"), (20, "toggle:flaps_idx:2"), (30, "toggle:flaps_idx:3")) if k in first]
    if flaps:
        out.append("Flaps " + ", ".join(flaps))
    if "wind_kt" in df:
        out.append(f"Wind {df['wind_kt'].median():.0f} kt from {df['wind_dir_deg'].median():03.0f}°")
    choices = meta.get("choices", {})
    if choices.get("power_correction"):
        out.append("Power correction on short final")
    if choices.get("light_on_roll"):
        out.append("Landing light switched on during the take-off roll")
    return out


def _air_track(record, df: pd.DataFrame) -> None:
    fig = go.Figure()
    for p in record.phases:
        seg = df[(df["t"] >= p.t_start) & (df["t"] <= p.t_end)]
        fig.add_trace(go.Scatter(x=seg["track_side_nm"], y=seg["track_along_nm"], mode="lines",
                                 name=ui.phase_title(p.name), line=dict(width=3)))
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers+text", text=["Take-off"], textposition="bottom center",
                             marker=dict(size=10, symbol="triangle-up", color="black"), showlegend=False))
    fig.update_yaxes(scaleanchor="x", scaleratio=1, title="nm along the runway")
    fig.update_xaxes(title="nm from the centreline")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), title="Ground track (from above)",
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")


def _air_profile(record, df: pd.DataFrame) -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["t"], y=df["agl_ft"], name="Height (ft)", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=df["t"], y=df["ias_kt"], name="Airspeed (kt)", yaxis="y2", line=dict(width=2)))
    for p in record.phases:
        fig.add_vrect(x0=p.t_start, x1=p.t_end, opacity=0.06, line_width=0)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), title="Height and airspeed",
                      xaxis_title="seconds", yaxis=dict(title="ft"),
                      yaxis2=dict(title="kt", overlaying="y", side="right"), legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")


AIR = Sim(
    make=generate_flight,
    simulator="Microsoft Flight Simulator",
    choices={
        "overspeed": ("Flaps 20 selected too fast", "About 89 kt, above the 85 kt flap limit — a limit violation"),
        "power_correction": ("Power correction on short final", "A brief burst of power below 250 ft — a "
                             "technique only some instructors use"),
        "light_on_roll": ("Landing light on late", "Switched on during the take-off roll instead of when lined up"),
        "traffic_check": ("Extra callout with no action", '"Checking for traffic" on downwind, with no matching '
                          "control input — it gets flagged"),
    },
    columns={
        "SIMULATION_TIME": ("Time", "seconds (any start; shifted to 0)"),
        "PLANE_LATITUDE": ("Latitude", "radians"),
        "PLANE_LONGITUDE": ("Longitude", "radians"),
        "PLANE_ALTITUDE": ("Altitude above sea level", "ft"),
        "PLANE_ALT_ABOVE_GROUND": ("Height above ground", "ft"),
        "PLANE_HEADING_DEGREES_MAGNETIC": ("Heading", "radians"),
        "PLANE_PITCH_DEGREES": ("Pitch", "radians, nose-up negative"),
        "PLANE_BANK_DEGREES": ("Bank", "radians, right bank negative"),
        "AIRSPEED_INDICATED": ("Indicated airspeed", "knots"),
        "GROUND_VELOCITY": ("Groundspeed", "knots"),
        "VERTICAL_SPEED": ("Vertical speed", "ft per second"),
        "GENERAL_ENG_THROTTLE_LEVER_POSITION:1": ("Throttle", "percent, 0–100"),
        "GENERAL_ENG_RPM:1": ("Engine RPM", "rpm"),
        "ELEVATOR_POSITION": ("Elevator", "−1 to 1"),
        "AILERON_POSITION": ("Aileron", "−1 to 1"),
        "FLAPS_HANDLE_INDEX": ("Flap lever", "0 up · 1 = 10° · 2 = 20° · 3 = 30°"),
        "SIM_ON_GROUND": ("On the ground", "0 or 1"),
        "BRAKE_PARKING_POSITION": ("Parking brake", "0 or 1"),
        "LIGHT_LANDING": ("Landing light", "0 or 1"),
        "LIGHT_TAXI": ("Taxi light", "0 or 1"),
        "LIGHT_STROBE": ("Strobe", "0 or 1"),
        "AMBIENT_WIND_VELOCITY": ("Wind speed", "knots"),
        "AMBIENT_WIND_DIRECTION": ("Wind direction", "degrees"),
    },
    columns_md="Columns are the [Python-SimConnect](https://pypi.org/project/SimConnect/) SimVar names, in "
               "SimConnect's own units (the app converts them).",
    required_md="Only SIMULATION_TIME is strictly required. A missing column is reported as a warning, and "
                "anything that depends on it (e.g. flap steps without FLAPS_HANDLE_INDEX) is left out.",
    example_csv="SIMULATION_TIME,PLANE_LATITUDE,PLANE_LONGITUDE,...,AIRSPEED_INDICATED,...,FLAPS_HANDLE_INDEX,...\n"
                "0.2,0.024725,1.812814,...,0.0,...,0,...",
    callouts=[{"t": 52.4, "text": "Rotating", "source": "spoken"},
              {"t": 341.0, "text": "Flaps 10 on downwind", "source": "spoken"}],
    highlights=_air_highlights,
    charts=(_air_track, _air_profile),
    phase_row=lambda seg: {"Airspeed (kt)": round(seg["ias_kt"].median()),
                           "Height (ft)": round(seg["agl_ft"].median()),
                           "From runway (nm)": round(seg["track_offset_nm"].median(), 1)},
)


# --- land: parallel parking in CARLA vehicle-state format ------------------------------------

def _land_highlights(record, df: pd.DataFrame, meta: dict) -> list[str]:
    first = {}
    for e in record.events:
        first.setdefault(e.key, e)
    phase = {p.name: df[(df["t"] >= p.t_start) & (df["t"] < p.t_end)] for p in record.phases}
    out = []
    if (ap := phase.get("approach")) is not None and len(ap):
        out.append(f"Drives up at **{ap['speed_kmh'].quantile(0.9):.0f} km/h** and stops "
                   "alongside the parked car")
    if len(rv := df[df["reverse"] == 1]):
        out.append(f"Reverses at up to **{rv['speed_kmh'].max():.1f} km/h**")
    if e := first.get("toggle:steer_lock:-1"):
        out.append(f"Full left lock after reversing **{e.state.get('back_m', 0):.1f} m**")
    if e := first.get("toggle:steer_lock:1"):
        out.append(f"Full right lock at **{e.state.get('kerb_angle_deg', 0):.0f}°** to the kerb")
    out.append(f"Ends **{abs(df['kerb_angle_deg'].iloc[-1]):.0f}°** off parallel to the "
               "kerb")
    choices = meta.get("choices", {})
    if choices.get("forward_adjust"):
        out.append("Moves forward at the end to centre the car")
    if choices.get("late_indicator"):
        out.append("Left indicator switched on only after stopping")
    return out


def _land_track(record, df: pd.DataFrame) -> None:
    fig = go.Figure()
    for p in record.phases:
        seg = df[(df["t"] >= p.t_start) & (df["t"] <= p.t_end)]
        if seg["speed_kmh"].max() > 0.3:   # stationary phases are a single dot
            fig.add_trace(go.Scatter(x=seg["x_m"], y=seg["y_m"], mode="lines", name=ui.phase_title(p.name),
                                     line=dict(width=3)))
    start = df[df["reverse"] == 1].head(1)
    if len(start):
        fig.add_trace(go.Scatter(x=start["x_m"], y=start["y_m"], mode="markers+text", text=["Reverse from here"],
                                 textposition="middle right", marker=dict(size=10, color="black"),
                                 showlegend=False))
    fig.update_yaxes(scaleanchor="x", scaleratio=1, title="m along the road")
    fig.update_xaxes(title="m across the road")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), title="Path of the car (from above)",
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")


def _land_profile(record, df: pd.DataFrame) -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["t"], y=df["speed_kmh"], name="Speed (km/h)", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=df["t"], y=df["kerb_angle_deg"], name="Angle to the kerb (°)", yaxis="y2",
                             line=dict(width=2)))
    for p in record.phases:
        fig.add_vrect(x0=p.t_start, x1=p.t_end, opacity=0.06, line_width=0)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), title="Speed and angle to the kerb",
                      xaxis_title="seconds", yaxis=dict(title="km/h"),
                      yaxis2=dict(title="°", overlaying="y", side="right"), legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")


LAND = Sim(
    make=generate_drive,
    simulator="the CARLA driving simulator",
    choices={
        "reverse_fast": ("Reverses too fast", "About 7 km/h, above the 5 km/h reversing limit, and runs past "
                         "11 m into the space behind — limit violations"),
        "forward_adjust": ("Moves forward to centre the car", "A short move forward at the end — a technique only "
                           "some instructors use"),
        "late_indicator": ("Indicator on late", "Signals after stopping instead of while slowing down"),
        "blind_spot_check": ("Extra callout with no action", '"Checking mirrors and blind spot" before reversing, '
                             "with no matching control input — it gets flagged"),
    },
    columns={
        "timestamp": ("Time", "seconds (world snapshot elapsed_seconds; any start)"),
        "location.x": ("Position x", "m"),
        "location.y": ("Position y", "m, CARLA is left-handed: +y is to the right"),
        "rotation.yaw": ("Heading (yaw)", "degrees, 0 = +x"),
        "velocity.x": ("Velocity x", "m/s"),
        "velocity.y": ("Velocity y", "m/s"),
        "control.throttle": ("Throttle", "0 to 1"),
        "control.brake": ("Brake", "0 to 1"),
        "control.steer": ("Steering", "−1 full left to 1 full right"),
        "control.gear": ("Gear", "integer"),
        "control.reverse": ("Reverse", "0 or 1"),
        "control.hand_brake": ("Handbrake", "0 or 1"),
        "light_state": ("Lights", "carla.VehicleLightState bitmask (left blinker 32, right 64)"),
    },
    columns_md="Columns are the [CARLA Python API](https://carla.readthedocs.io/en/latest/python_api/) fields, "
               "logged once per world tick from `vehicle.get_transform()`, `get_velocity()`, `get_control()` and "
               "`get_light_state()` (the app converts them).",
    required_md="Required: timestamp, location, rotation.yaw, velocity, and control.throttle / brake / steer. "
                "Gear, reverse, handbrake and light_state are optional; anything that depends on a missing one "
                "(e.g. indicator steps without light_state) is left out.",
    example_csv="timestamp,location.x,location.y,rotation.yaw,velocity.x,velocity.y,control.throttle,...,"
                "control.steer,...,light_state\n0.1,0.0,0.0,90.0,0.0,0.0,0.0,...,0.0,...,0",
    callouts=[{"t": 18.4, "text": "Full left lock", "source": "spoken"},
              {"t": 22.1, "text": "Straighten up", "source": "spoken"}],
    highlights=_land_highlights,
    charts=(_land_track, _land_profile),
    phase_row=lambda seg: {"Speed (km/h)": round(seg["speed_kmh"].median(), 1),
                           "Top speed (km/h)": round(seg["speed_kmh"].max(), 1)},
)

SIMS = {"air": AIR, "land": LAND}


# --- page -----------------------------------------------------------------------------------

def _out_dir(prof: Profile) -> Path:
    return ui.generated_dir(prof)


def _generated(prof: Profile) -> list[Path]:
    d = _out_dir(prof)
    return sorted((x for x in d.glob("*") if x.is_dir()), key=lambda x: x.stat().st_mtime,
                  reverse=True) if d.exists() else []


def _new_seed() -> None:
    st.session_state["gen_seed"] = random.randint(1000, 99999)


def _surprise(sim: Sim) -> None:
    for k in sim.choices:
        st.session_state[f"gen_{k}"] = random.random() < 0.35
    _new_seed()


def _generate(prof: Profile, sim: Sim) -> None:
    ss = st.session_state
    path = sim.make(_out_dir(prof), seed=int(ss["gen_seed"]),
                    instructor=ss["gen_instructor"].strip().replace(" ", "_").lower() or None,
                    **{k: ss[f"gen_{k}"] for k in sim.choices})
    ss["gen_current"] = path.name
    _new_seed()  # the next run is different unless the seed is typed in again


@st.cache_data(show_spinner=False)
def _analyse(path: str, mtime: float, procedure: str):
    return analyse_run(Path(path), ui.profile())


def _limits(prof: Profile):
    """The official documents in use; before any are added, the sample documents."""
    docs = references.list_references(prof.procedure)
    if docs:
        return [lim for d in docs for lim in d.limits], "the official documents in use"
    folder = config.REFERENCES_DIR / prof.references
    sample = [references.read_refdoc(f) for f in sorted(folder.glob("*.md")) if f.name != "README.md"]
    return [lim for d in sample for lim in d.limits], "the sample official documents (not added yet)"


def _mmss(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def _form(prof: Profile, sim: Sim) -> None:
    ss = st.session_state
    ss.setdefault("gen_seed", random.randint(1000, 99999))
    ss.setdefault("gen_instructor", "")
    for k in sim.choices:
        ss.setdefault(f"gen_{k}", False)

    with st.container(border=True):
        st.markdown(f"**1 · Choose the {ui.noun()}**")
        st.text_input("Instructor (optional)", key="gen_instructor", placeholder="e.g. Instructor 3")
        st.caption("What does this instructor do differently? Everything else (speeds, distances, timing, sensor "
                   f"noise) varies randomly from {ui.noun()} to {ui.noun()}.")
        cols = st.columns(2)
        for i, (k, (label, help_)) in enumerate(sim.choices.items()):
            cols[i % 2].checkbox(label, key=f"gen_{k}", help=help_)
        s1, s2, s3, s4 = st.columns([2, 1, 1, 2], vertical_alignment="bottom")
        s1.number_input("Random seed", min_value=0, max_value=999_999, step=1, key="gen_seed",
                        help=f"The same seed and choices always give the same {ui.noun()}.")
        s2.button("New seed", on_click=_new_seed, width="stretch")
        s3.button("Surprise me", on_click=_surprise, args=(sim,), width="stretch",
                  help="Random choices and seed.")
        s4.button(f"Generate {ui.noun()}", type="primary", on_click=_generate,
                  args=(prof, sim), width="stretch", key="generate")


def _zip(run_dir: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(run_dir.iterdir()):
            z.write(f, f"{run_dir.name}/{f.name}")
    return buf.getvalue()


def _rundown(prof: Profile, sim: Sim, run_dir: Path) -> None:
    record, df = _analyse(str(run_dir), run_dir.stat().st_mtime, prof.procedure)
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    added = run_dir.name in {r.run_id for r in storage.list_runs(prof.procedure)}
    limits, source = _limits(prof)
    breaches = check_run(df, record, limits, prof)
    unmatched = [a.text for a in record.annotations if a.event_id is None]

    with st.container(border=True):
        st.markdown(f"**2 · What's in this {ui.noun()} — {ui.run_title(run_dir.name)}**")
        st.caption(f"Instructor: {ui.instructor_name(meta.get('instructor'))} · seed {meta.get('seed')}")
        m = st.columns(4)
        m[0].metric("Length", _mmss(record.meta.duration_s))
        m[1].metric("Data points", f"{record.meta.n_samples:,}", help=f"{prof.sample_hz:g} per second")
        m[2].metric("Spoken callouts", len(record.annotations))
        m[3].metric("Actions detected", len(record.events), help="Switches, levers, steering, stops, turns — "
                    "what the guidelines are built from.")

        st.markdown("\n".join(f"- {h}" for h in sim.highlights(record, df, meta)))
        if breaches:
            for f in breaches:
                st.error(f"**Breaks a hard limit** — {ui.phase_title(f.phase or '')} at {_mmss(f.t or 0)}: "
                         + f.message.split(": ", 1)[-1], icon=":material/report:")
            st.caption(f"Checked against {source}. When this {ui.noun()} is added, the guidelines leave these "
                       "values out of the typical range.")
        elif limits:
            st.success(f"Stays within all {len(limits)} hard limits of {source}.")
        for u in unmatched:
            st.warning(f'Callout "{u}" has no matching action — it will be flagged for the instructor.',
                       icon=":material/mic_off:")

        c1, c2 = st.columns(2)
        with c1:
            sim.charts[0](record, df)
        with c2:
            sim.charts[1](record, df)

        rows = []
        for p in record.phases:
            seg = df[(df["t"] >= p.t_start) & (df["t"] < p.t_end)]
            rows.append({"Phase": ui.phase_title(p.name), "Starts at": _mmss(p.t_start),
                         "Length (s)": round(p.t_end - p.t_start), **sim.phase_row(seg)})
        st.markdown("**Phase by phase** (typical values in each phase)")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        with st.expander(f"Actions and callouts ({len(record.events)})"):
            st.dataframe(pd.DataFrame([{"Time": _mmss(e.t), "Phase": ui.phase_title(e.phase or ""),
                                        "Action": e.label, "Instructor callout": e.annotation or ""}
                                       for e in record.events]), hide_index=True, width="stretch")
        with st.expander("The raw files"):
            t1, t2, t3 = st.tabs(["telemetry.csv", "annotations.json", "meta.json"])
            with t1:
                st.caption("First 10 rows, exactly as written by the simulator logger.")
                st.dataframe(pd.read_csv(run_dir / "telemetry.csv", nrows=10), hide_index=True)
            t2.json(json.loads((run_dir / "annotations.json").read_text(encoding="utf-8")), expanded=False)
            t3.json(meta)

        b1, b2, b3 = st.columns(3)
        if added:
            b1.markdown(":green[:material/check_circle:] Added to the guidelines.")
        elif b1.button("Add to the guidelines", type="primary", width="stretch",
                       help="The guidelines update straight away and show you what changed."):
            ingest_run(run_dir, prof)
            ui.refresh_guidelines(prof, f"{ui.run_title(run_dir.name)} added")
            st.rerun()
        b2.download_button("Download (zip)", _zip(run_dir), file_name=f"{run_dir.name}.zip",
                           mime="application/zip", width="stretch",
                           help=f"The {ui.noun()} in the simulator's file format: an example to upload, or to keep.")
        if b3.button("Discard", width="stretch",
                     help=f"Delete the generated files ({ui.noun(2)} already uploaded stay)."):
            shutil.rmtree(run_dir, ignore_errors=True)
            st.session_state.pop("gen_current", None)
            st.rerun()


def _format_guide(prof: Profile, sim: Sim) -> None:
    run = ui.noun()
    st.subheader(f"Data format — bring your own {ui.noun(2)}")
    st.write(f"A generated {run} is laid out exactly like a recording from {sim.simulator}. "
             f"To add a real {run}, give it the same shape.")
    t1, t2, t3, t4, t5 = st.tabs(["Folder", "telemetry.csv", "annotations.json", "meta.json", "How to add it"])
    with t1:
        st.markdown(f"One folder per {run}:")
        st.code(f"my_{run}_01/\n"
                "├── telemetry.csv      required: one row per sample, the simulator's column names\n"
                "├── annotations.json   optional: the instructor's spoken callouts\n"
                f"└── meta.json          optional: {run} name, instructor", language=None)
        st.caption(f"telemetry.parquet works instead of CSV. Download any generated {run} above for a "
                   "complete example.")
    with t2:
        st.markdown(f"{sim.columns_md} Any sample rate works — the app resamples to {prof.sample_hz:g} per second.")
        st.dataframe(pd.DataFrame([{"Column": k, "What it is": what, "Unit in the file": unit}
                                   for k, (what, unit) in sim.columns.items()]), hide_index=True, width="stretch")
        st.caption(sim.required_md)
        st.code(sim.example_csv, language="text")
    with t3:
        st.markdown(f"A list of callouts. `t` uses the same clock as `{next(iter(sim.columns))}`; each callout is "
                    "matched to the nearest action and becomes the **Why** of a guideline.")
        st.code(json.dumps(sim.callouts, indent=2), language="json")
        st.caption("`source` is spoken or typed. Audio can be transcribed with `sim2sop transcribe`.")
    with t4:
        st.code(json.dumps({"run_id": f"my_{run}_01", "instructor": "instructor_3"}, indent=2), language="json")
        st.caption(f"Both fields are optional: the folder name is used as the {run}'s name if `run_id` is missing.")
    with t5:
        st.markdown(f"**In this app:** *1 · Instructor {ui.noun(2)}* → *Upload {run} data* → choose the .zip, or "
                    "telemetry.csv and annotations.json if you have one.")
        ui.link_button("runs", f"Open Instructor {ui.noun(2)}", key="gen-runs", primary=False)
        st.markdown(f"**From the command line** (a single {run} folder, or a folder of {run} folders):")
        name = st.session_state.get("profile_name", prof.procedure)
        st.code(f"sim2sop ingest {name} path/to/my_{run}_01\n"
                f"sim2sop update {name}", language="bash")


def render() -> None:
    prof = ui.profile()
    ui.page_header(f"{ui.noun(cap=True)} generator",
                   f"Make a random instructor {ui.noun()} in the simulator's own data format, see what's in it, "
                   "then add it to the guidelines with one click, or download it as an example of what a "
                   "recording should look like.", eyebrow="Tools")
    sim = SIMS.get(prof.domain)
    if sim is None:
        st.info("The generator makes Cessna 172 circuits and parallel parking so far. Pick one of those "
                "procedures in the sidebar.")
        return

    _form(prof, sim)
    made = _generated(prof)
    if made:
        names = [d.name for d in made]
        current = st.session_state.get("gen_current")
        if current not in names:
            current = names[0]
        if len(names) > 1:
            current = st.selectbox("Showing", names, index=names.index(current), format_func=ui.run_title,
                                   help=f"{ui.noun(2, cap=True)} you generated earlier.")
        st.session_state["gen_current"] = current
        _rundown(prof, sim, _out_dir(prof) / current)
    else:
        st.info(f"Choose the options above and press **Generate {ui.noun()}**.")
    st.divider()
    _format_guide(prof, sim)
