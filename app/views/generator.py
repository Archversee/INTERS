"""Tools: generate a random instructor flight, see what is in it, then add it or download it.
Also documents the file format, so real recordings can be ingested the same way."""

from __future__ import annotations

import io
import json
import random
import shutil
import zipfile
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import ui

from sim2sop import config, references, storage
from sim2sop.adapters.msfs import SIMVARS
from sim2sop.generate.air import generate_flight
from sim2sop.ingest import analyse_run
from sim2sop.profiles import Profile
from sim2sop.signals.compliance import check_run

CHOICES = {
    "overspeed": ("Flaps 20 selected too fast", "About 89 kt, above the 85 kt flap limit — a limit violation"),
    "power_correction": ("Power correction on short final", "A brief burst of power below 250 ft — a technique "
                         "only some instructors use"),
    "light_on_roll": ("Landing light on late", "Switched on during the take-off roll instead of when lined up"),
    "traffic_check": ("Extra callout with no action", '"Checking for traffic" on downwind, with no matching '
                      "control input — it gets flagged"),
}

# SimVar -> (what it is, unit in the file). Order and names follow the MSFS adapter.
FORMAT = {
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
}


def _out_dir(prof: Profile) -> Path:
    return ui.generated_dir(prof)


def _generated(prof: Profile) -> list[Path]:
    d = _out_dir(prof)
    return sorted((x for x in d.glob("*") if x.is_dir()), key=lambda x: x.stat().st_mtime,
                  reverse=True) if d.exists() else []


def _new_seed() -> None:
    st.session_state["gen_seed"] = random.randint(1000, 99999)


def _surprise() -> None:
    for k in CHOICES:
        st.session_state[f"gen_{k}"] = random.random() < 0.35
    _new_seed()


def _generate(prof: Profile) -> None:
    ss = st.session_state
    path = generate_flight(_out_dir(prof), seed=int(ss["gen_seed"]),
                           instructor=ss["gen_instructor"].strip().replace(" ", "_").lower() or None,
                           **{k: ss[f"gen_{k}"] for k in CHOICES})
    ss["gen_current"] = path.name
    _new_seed()  # the next flight is different unless the seed is typed in again


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


def _form(prof: Profile) -> None:
    ss = st.session_state
    ss.setdefault("gen_seed", random.randint(1000, 99999))
    ss.setdefault("gen_instructor", "")
    for k in CHOICES:
        ss.setdefault(f"gen_{k}", False)

    with st.container(border=True):
        st.markdown("**1 · Choose the flight**")
        st.text_input("Instructor (optional)", key="gen_instructor", placeholder="e.g. Instructor 3")
        st.caption("What does this instructor do differently? Everything else (speeds, heights, wind, sensor "
                   "noise) varies randomly from flight to flight.")
        cols = st.columns(2)
        for i, (k, (label, help_)) in enumerate(CHOICES.items()):
            cols[i % 2].checkbox(label, key=f"gen_{k}", help=help_)
        s1, s2, s3, s4 = st.columns([2, 1, 1, 2], vertical_alignment="bottom")
        s1.number_input("Random seed", min_value=0, max_value=999_999, step=1, key="gen_seed",
                        help="The same seed and choices always give the same flight.")
        s2.button("New seed", icon=":material/casino:", on_click=_new_seed, width="stretch")
        s3.button("Surprise me", icon=":material/shuffle:", on_click=_surprise, width="stretch",
                  help="Random choices and seed.")
        s4.button("Generate flight", type="primary", icon=":material/auto_awesome:", on_click=_generate,
                  args=(prof,), width="stretch")


def _highlights(prof: Profile, record, df: pd.DataFrame, meta: dict) -> list[str]:
    first = {}
    for e in record.events:
        first.setdefault(e.key, e)
    phase = {p.name: df[(df["t"] >= p.t_start) & (df["t"] < p.t_end)] for p in record.phases}
    out = []
    if e := first.get("toggle:on_ground:0"):
        out.append(f":material/flight_takeoff: Lifts off at **{e.state.get('ias_kt', 0):.0f} kt**")
    if (dw := phase.get("downwind")) is not None and len(dw):
        out.append(f":material/height: Downwind at **{dw['agl_ft'].median():,.0f} ft**, "
                   f"**{dw['track_offset_nm'].median():.1f} nm** from the runway, "
                   f"**{dw['ias_kt'].median():.0f} kt**")
    flaps = [f"{deg}° at {first[k].state.get('ias_kt', 0):.0f} kt" for deg, k in
             ((10, "toggle:flaps_idx:1"), (20, "toggle:flaps_idx:2"), (30, "toggle:flaps_idx:3")) if k in first]
    if flaps:
        out.append(":material/tune: Flaps " + ", ".join(flaps))
    if "wind_kt" in df:
        out.append(f":material/air: Wind {df['wind_kt'].median():.0f} kt from {df['wind_dir_deg'].median():03.0f}°")
    choices = meta.get("choices", {})
    if choices.get("power_correction"):
        out.append(":material/bolt: Power correction on short final")
    if choices.get("light_on_roll"):
        out.append(":material/light_mode: Landing light switched on during the take-off roll")
    return out


def _track_chart(prof: Profile, record, df: pd.DataFrame) -> None:
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


def _profile_chart(prof: Profile, record, df: pd.DataFrame) -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["t"], y=df["agl_ft"], name="Height (ft)", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=df["t"], y=df["ias_kt"], name="Airspeed (kt)", yaxis="y2", line=dict(width=2)))
    for p in record.phases:
        fig.add_vrect(x0=p.t_start, x1=p.t_end, opacity=0.06, line_width=0)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), title="Height and airspeed",
                      xaxis_title="seconds", yaxis=dict(title="ft"),
                      yaxis2=dict(title="kt", overlaying="y", side="right"), legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")


def _zip(run_dir: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(run_dir.iterdir()):
            z.write(f, f"{run_dir.name}/{f.name}")
    return buf.getvalue()


def _rundown(prof: Profile, run_dir: Path) -> None:
    record, df = _analyse(str(run_dir), run_dir.stat().st_mtime, prof.procedure)
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    added = run_dir.name in {r.run_id for r in storage.list_runs(prof.procedure)}
    limits, source = _limits(prof)
    breaches = check_run(df, record, limits, prof)
    unmatched = [a.text for a in record.annotations if a.event_id is None]

    with st.container(border=True):
        st.markdown(f"**2 · What's in this flight — {ui.run_title(run_dir.name)}**")
        st.caption(f"Instructor: {meta.get('instructor', '—').replace('_', ' ').title()} · seed {meta.get('seed')}")
        m = st.columns(4)
        m[0].metric("Length", _mmss(record.meta.duration_s))
        m[1].metric("Data points", f"{record.meta.n_samples:,}", help=f"{prof.sample_hz} per second")
        m[2].metric("Spoken callouts", len(record.annotations))
        m[3].metric("Actions detected", len(record.events), help="Switches, flap changes, power changes, turns "
                    "— what the guidelines are built from.")

        st.markdown("\n".join(f"- {h}" for h in _highlights(prof, record, df, meta)))
        if breaches:
            for f in breaches:
                st.error(f"**Breaks a hard limit** — {ui.phase_title(f.phase or '')} at {_mmss(f.t or 0)}: "
                         + f.message.split(": ", 1)[-1], icon=":material/report:")
            st.caption(f"Checked against {source}. When this flight is added, the guidelines leave these values "
                       "out of the typical range.")
        elif limits:
            st.success(f"Stays within all {len(limits)} hard limits of {source}.", icon=":material/verified:")
        for u in unmatched:
            st.warning(f'Callout "{u}" has no matching action — it will be flagged for the instructor.',
                       icon=":material/mic_off:")

        c1, c2 = st.columns(2)
        with c1:
            _track_chart(prof, record, df)
        with c2:
            _profile_chart(prof, record, df)

        rows = []
        for p in record.phases:
            seg = df[(df["t"] >= p.t_start) & (df["t"] < p.t_end)]
            rows.append({"Phase": ui.phase_title(p.name), "Starts at": _mmss(p.t_start),
                         "Length (s)": round(p.t_end - p.t_start), "Airspeed (kt)": round(seg["ias_kt"].median()),
                         "Height (ft)": round(seg["agl_ft"].median()),
                         "From runway (nm)": round(seg["track_offset_nm"].median(), 1)})
        st.markdown("**Phase by phase** (typical values in each phase)")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        with st.expander(f"Actions and callouts ({len(record.events)})", icon=":material/list:"):
            st.dataframe(pd.DataFrame([{"Time": _mmss(e.t), "Phase": ui.phase_title(e.phase or ""),
                                        "Action": e.label, "Instructor callout": e.annotation or ""}
                                       for e in record.events]), hide_index=True, width="stretch")
        with st.expander("The raw files", icon=":material/description:"):
            t1, t2, t3 = st.tabs(["telemetry.csv", "annotations.json", "meta.json"])
            with t1:
                st.caption("First 10 rows, exactly as written by the simulator logger.")
                st.dataframe(pd.read_csv(run_dir / "telemetry.csv", nrows=10), hide_index=True)
            t2.json(json.loads((run_dir / "annotations.json").read_text(encoding="utf-8")), expanded=False)
            t3.json(meta)

        if added:
            st.caption(":green[:material/check_circle:] This flight has been uploaded.")
        else:
            st.caption("To add it to the guidelines, download the zip and upload it on *1 · Instructor flights*.")
        b1, b2, b3 = st.columns(3)
        b1.download_button("Download (zip)", _zip(run_dir), file_name=f"{run_dir.name}.zip", type="primary",
                           mime="application/zip", icon=":material/download:", width="stretch")
        with b2:
            ui.link_button("runs", "Go to upload", icon=":material/upload:", primary=False, key="gen-upload")
        if b3.button("Discard", icon=":material/delete:", width="stretch",
                     help="Delete the generated files (flights already uploaded stay)."):
            shutil.rmtree(run_dir, ignore_errors=True)
            st.session_state.pop("gen_current", None)
            st.rerun()


def _format_guide(prof: Profile) -> None:
    st.subheader(":material/integration_instructions: Data format — bring your own flights")
    st.write("A generated flight is laid out exactly like a recording from Microsoft Flight Simulator. "
             "To add a real flight, give it the same shape.")
    t1, t2, t3, t4, t5 = st.tabs(["Folder", "telemetry.csv", "annotations.json", "meta.json", "How to add it"])
    with t1:
        st.markdown("One folder per flight:")
        st.code("my_flight_01/\n"
                "├── telemetry.csv      required: one row per sample, SimConnect column names\n"
                "├── annotations.json   optional: the instructor's spoken callouts\n"
                "└── meta.json          optional: flight name, instructor", language=None)
        st.caption("telemetry.parquet works instead of CSV. Download any generated flight above for a "
                   "complete example.")
    with t2:
        st.markdown("Columns are the [Python-SimConnect](https://pypi.org/project/SimConnect/) SimVar names, in "
                    "SimConnect's own units (the app converts them). Any sample rate works — the app resamples "
                    f"to {prof.sample_hz} per second.")
        st.dataframe(pd.DataFrame([{"Column": k, "What it is": FORMAT.get(k, ("", ""))[0],
                                    "Unit in the file": FORMAT.get(k, ("", ""))[1]}
                                   for k in ["SIMULATION_TIME", *SIMVARS]]), hide_index=True, width="stretch")
        st.caption("Only SIMULATION_TIME is strictly required. A missing column is reported as a warning, and "
                   "anything that depends on it (e.g. flap steps without FLAPS_HANDLE_INDEX) is left out.")
        st.code("SIMULATION_TIME,PLANE_LATITUDE,PLANE_LONGITUDE,...,AIRSPEED_INDICATED,...,FLAPS_HANDLE_INDEX,...\n"
                "0.2,0.024725,1.812814,...,0.0,...,0,...", language="text")
    with t3:
        st.markdown("A list of callouts. `t` uses the same clock as `SIMULATION_TIME`; each callout is matched "
                    "to the nearest action and becomes the **Why** of a guideline.")
        st.code(json.dumps([{"t": 52.4, "text": "Rotating", "source": "spoken"},
                            {"t": 341.0, "text": "Flaps 10 on downwind", "source": "spoken"}], indent=2),
                language="json")
        st.caption("`source` is spoken or typed. Audio can be transcribed with `sim2sop transcribe`.")
    with t4:
        st.code(json.dumps({"run_id": "my_flight_01", "instructor": "instructor_3"}, indent=2), language="json")
        st.caption("Both fields are optional: the folder name is used as the flight's name if `run_id` is missing.")
    with t5:
        st.markdown("**In this app:** *1 · Instructor flights* → *Upload flight data* → choose "
                    "telemetry.csv, and annotations.json if you have one.")
        ui.link_button("runs", "Open Instructor flights", key="gen-runs", primary=False)
        st.markdown("**From the command line** (a single flight folder, or a folder of flight folders):")
        name = st.session_state.get("profile_name", prof.procedure)
        st.code(f"sim2sop ingest {name} path/to/my_flight_01\n"
                f"sim2sop update {name}", language="bash")


def render() -> None:
    prof = ui.profile()
    st.title(":material/science: Flight generator")
    st.write("Make a random instructor flight in the simulator's own data format, see what's in it, then add it "
             "to the guidelines — or download it as an example of what a recording should look like.")
    if prof.domain != "air":
        st.info("The generator only makes Cessna 172 circuits so far. Pick that procedure in the sidebar.",
                icon=":material/info:")
        _format_guide(prof)
        return

    _form(prof)
    made = _generated(prof)
    if made:
        names = [d.name for d in made]
        current = st.session_state.get("gen_current")
        if current not in names:
            current = names[0]
        if len(names) > 1:
            current = st.selectbox("Showing", names, index=names.index(current), format_func=ui.run_title,
                                   help="Flights you generated earlier.")
        st.session_state["gen_current"] = current
        _rundown(prof, _out_dir(prof) / current)
    else:
        st.info("Choose the options above and press **Generate flight**.", icon=":material/arrow_upward:")
    st.divider()
    _format_guide(prof)
