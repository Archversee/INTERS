from sim2sop import sop_store
from sim2sop.agent.llm import NoLLM
from sim2sop.ingest import ingest_run
from sim2sop.metrics import band_for
from sim2sop.pipeline import apply_edit, generate_sop
from sim2sop.profiles import list_profiles, load_profile

PROFILE = "air_c172_circuit"


def test_all_profiles_load():
    for name in list_profiles():
        load_profile(name)


def test_ingest_extracts_phases_events_and_flags(data_dir, air_raw):
    p = load_profile(PROFILE)
    rec = ingest_run(air_raw[2], p)  # variant 2 has an unmatched "Checking for traffic" callout
    assert [ph.name for ph in rec.phases] == p.phase_names
    keys = {e.key for e in rec.events}
    assert {"toggle:flaps_idx:1", "toggle:flaps_idx:2", "toggle:flaps_idx:3",
            "threshold:ias_kt:above:55", "toggle:on_ground:0", "toggle:on_ground:1"} <= keys
    rotate = next(e for e in rec.events if e.key == "threshold:ias_kt:above:55")
    assert rotate.annotation == "Rotating"
    assert any("Checking for traffic" in f.message for f in rec.flags)


def test_sop_versions_learn_from_runs_and_edits(data_dir, air_raw):
    p = load_profile(PROFILE)
    llm = NoLLM("template")

    ingest_run(air_raw[0], p)
    v1 = generate_sop(p, llm=llm)
    assert v1.version == 1 and v1.metrics["verifier_pass_rate"] == 1.0
    assert all(s.mandatory for s in v1.steps)

    for d in air_raw[1:]:
        ingest_run(d, p)
    v2 = generate_sop(p, llm=llm)
    assert v2.metrics["verifier_pass_rate"] == 1.0
    assert v2.metrics["band_width_ratio"] < v1.metrics["band_width_ratio"]
    assert v2.metrics["mean_confidence"] > v1.metrics["mean_confidence"]
    assert band_for(v2, "threshold:ias_kt", "trigger.ias_kt")[1] < band_for(v1, "threshold:ias_kt", "trigger.ias_kt")[1]
    # Power correction on short final is flown in 2 of 5 runs -> technique, not mandatory.
    technique = [s for s in v2.steps if not s.mandatory]
    assert technique and all(s.phase == "final" for s in technique)

    flaps10 = next(s for s in v2.steps if s.key == "toggle:flaps_idx:1")
    v3 = apply_edit(p, flaps10.id, "action", f"Select {flaps10.label}, then check the indicator.")
    assert v3.version == 3

    v4 = generate_sop(p, llm=llm)
    flaps20 = next(s for s in v4.steps if s.key == "toggle:flaps_idx:2")
    assert flaps20.action == f"Select {flaps20.label}, then check the indicator."
    assert sop_store.diff(v2, v4)


def test_edit_with_wrong_number_is_flagged(data_dir, air_raw):
    p = load_profile(PROFILE)
    ingest_run(air_raw[0], p)
    v1 = generate_sop(p, llm=NoLLM("template"))
    rotate = next(s for s in v1.steps if s.key == "threshold:ias_kt:above:55")
    v2 = apply_edit(p, rotate.id, "expected", "Climb at 95 kt.")
    edited = next(s for s in v2.steps if s.id == rotate.id)
    assert not edited.verified
    assert any("95" in f.message for f in v2.flags)
