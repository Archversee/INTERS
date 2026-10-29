from sim2sop import config, references
from sim2sop.agent.llm import NoLLM
from sim2sop.generate.land import generate_land_runs
from sim2sop.ingest import analyse_run, ingest_run
from sim2sop.pipeline import create_guidelines
from sim2sop.profiles import load_profile

PROFILE = "land_parallel_park"
STEERING = ["toggle:steer_lock:-1", "toggle:steer_lock:0", "toggle:steer_lock:1", "toggle:steer_lock:0"]


def test_generated_park_segments_and_steers(tmp_path):
    p = load_profile(PROFILE)
    rec, df = analyse_run(generate_land_runs(tmp_path, n=3)[2], p)  # variant 2: unmatched blind-spot callout
    assert [ph.name for ph in rec.phases] == ["start", "approach", "reverse_in", "secure"]
    reversing = [e for e in rec.events if e.phase == "reverse_in"]
    assert [e.key for e in reversing if e.key.startswith("toggle:steer_lock")] == STEERING
    left = next(e for e in reversing if e.key == "toggle:steer_lock:-1")
    assert 1.0 < left.state["back_m"] < 3.5 and abs(left.state["kerb_angle_deg"]) < 5
    straighten = next(e for e in reversing if e.key == "toggle:steer_lock:0")
    assert 30 < straighten.state["kerb_angle_deg"] < 50
    assert abs(df["kerb_angle_deg"].iloc[-1]) < 5   # parked parallel to the kerb
    assert any("blind spot" in f.message for f in rec.flags)
    assert not any(f.kind == "data" for f in rec.flags)   # derived signals are not "missing"


def test_land_guidelines_cap_reversing_speed(data_dir, tmp_path):
    p = load_profile(PROFILE)
    for f in sorted((config.REFERENCES_DIR / "land").glob("*.md")):
        if f.name != "README.md":
            references.add_reference(p.procedure, f)
    for run in generate_land_runs(tmp_path, n=5):   # run 1 reverses at about 7 km/h
        ingest_run(run, p)
    doc = create_guidelines(p, NoLLM("template"))
    keys = [s.key for s in doc.steps_in("reverse_in")]
    assert [k for k in keys if k.startswith("toggle:steer_lock")] == STEERING
    assert all(s.seen_in == 5 for s in doc.steps_in("reverse_in") if s.key.startswith("toggle:steer_lock"))
    assert any(f.kind == "limit_violation" and "5 km/h" in f.message for f in doc.flags)
    assert {s.key for s in doc.steps if s.required_by} >= {"toggle:indicator_left:1", "toggle:hand_brake:1"}
    stop = [s for s in doc.steps_in("approach") if s.label == "Stop alongside the parked car"]
    assert len(stop) == 1 and stop[0].seen_in == 5   # one step, though the runs differ in length
