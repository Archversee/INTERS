from sim2sop import guidelines_store as store
from sim2sop import references
from sim2sop.agent.llm import NoLLM
from sim2sop.ingest import ingest_run
from sim2sop.learn.changes import step_slots
from sim2sop.metrics import fact_for
from sim2sop.pipeline import apply_edit, approve, create_guidelines, update_guidelines
from sim2sop.profiles import list_profiles, load_profile

PROFILE = "air_c172_circuit"
LLM = NoLLM("template")
BASE_REFS = ("c172_poh_notes.md", "aerodrome_sop_notes.md", "faa_afh_traffic_pattern_notes.md")


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
    assert "track_offset_nm" in rotate.state


def test_reference_library_parses_limits_and_supersedes(data_dir, refs_dir, tmp_path):
    proc = load_profile(PROFILE).procedure
    doc = references.add_reference(proc, refs_dir / "c172_poh_notes.md")
    assert doc.kind == "poh" and {lim.id for lim in doc.limits} >= {"vfe_10", "vfe_full", "vne"}
    revised = tmp_path / "poh_rev_b.md"
    revised.write_text((refs_dir / "c172_poh_notes.md").read_text(encoding="utf-8")
                       .replace('revision: "A"', 'revision: "B"'), encoding="utf-8")
    references.add_reference(proc, revised)
    active = references.list_references(proc)
    assert [d.revision for d in active] == ["B"]
    assert len(references.list_references(proc, active_only=False)) == 2


def test_create_unchecked_then_limits_clip_and_flag(data_dir, air_raw, refs_dir):
    p = load_profile(PROFILE)
    ingest_run(air_raw[0], p)  # run 1 selects flaps 20 at ~89 kt, above Vfe 85
    v1 = create_guidelines(p, LLM)
    assert not v1.checked and v1.status == "draft" and len(v1.pending) == len(v1.steps)
    assert any(f.kind == "unchecked" for f in v1.flags)
    assert v1.metrics["verifier_pass_rate"] == 1.0
    approve(p)
    assert store.latest(p.procedure).status == "published"

    for f in BASE_REFS:
        references.add_reference(p.procedure, refs_dir / f)
    v2 = update_guidelines(p, LLM)
    assert v2.checked and v2.metrics["within_limits"] == 1.0
    flaps20 = next(s for s in v2.steps if s.key == "toggle:flaps_idx:2")
    f = flaps20.facts["trigger.ias_kt"]
    assert f.clipped and f.value == 85 and f.limit_max == 85
    assert "85" in flaps20.limit and flaps20.verified
    assert any(fl.kind == "limit_violation" for fl in v2.flags)
    assert any(s.required_by for s in v2.steps)  # aerodrome SOP makes the lights mandatory
    assert v2.metrics["cited_pct"] >= 0.8


def test_runs_tighten_ranges_and_exclude_violations(data_dir, air_raw, refs_dir):
    p = load_profile(PROFILE)
    for f in BASE_REFS:
        references.add_reference(p.procedure, refs_dir / f)
    ingest_run(air_raw[0], p)
    v1 = create_guidelines(p, LLM)
    approve(p)
    for d in air_raw[1:]:
        ingest_run(d, p)
    v2 = update_guidelines(p, LLM)
    assert v2.metrics["range_width_ratio"] < v1.metrics["range_width_ratio"]
    assert fact_for(v2, "threshold:ias_kt", "trigger.ias_kt").width < fact_for(v1, "threshold:ias_kt", "trigger.ias_kt").width
    flaps20 = next(s for s in v2.steps if s.key == "toggle:flaps_idx:2")
    assert "air_c172_circuit_01" not in flaps20.runs  # overspeed run excluded from learning
    assert not flaps20.facts["trigger.ias_kt"].clipped
    technique = [s for s in v2.steps if not s.mandatory]
    assert technique and all(s.phase == "final" for s in technique)


def test_obstacle_update_changes_only_downwind_and_base(data_dir, air_raw, obstacle_raw, refs_dir):
    p = load_profile(PROFILE)
    for f in BASE_REFS:
        references.add_reference(p.procedure, refs_dir / f)
    for d in air_raw:
        ingest_run(d, p)
    create_guidelines(p, LLM)
    v1 = approve(p)

    references.add_reference(p.procedure, refs_dir / "notam_crane.md")
    for d in obstacle_raw[:2]:
        ingest_run(d, p)
    held = update_guidelines(p, LLM)  # only 2 runs: not enough to call it a technique change
    assert not any(s.change for s in held.steps if s.phase in ("downwind", "base") and
                   any("Technique change" in r for r in s.change_reasons))
    assert any(f.kind == "technique_change" and "waiting for 3" in f.message for f in held.flags)
    approve(p)

    ingest_run(obstacle_raw[2], p)
    v = update_guidelines(p, LLM)
    changed = {s.phase for s in v.steps if s.change}
    assert changed == {"downwind", "base"}
    assert v.environment == "new-obstacle"
    assert all(any("NOTAM" in r for r in s.change_reasons) for s in v.steps if s.change)
    assert v.status == "draft" and v.pending
    downwind = [s for s in v.steps if s.phase == "downwind"]
    assert all(s.facts["result.track_offset_nm"].lo >= 1.5 for s in downwind)
    assert v.metrics["within_limits"] == 1.0
    # Untouched phases are word-for-word the previous version.
    before, after = step_slots(v1.steps), step_slots(v.steps)
    untouched = [k for k, s in before.items() if s.phase not in ("downwind", "base")]
    assert untouched and all(after[k].typical == before[k].typical for k in untouched)

    reject = v.pending[0]
    published = approve(p, reject_ids=[reject.id])
    assert published.status == "published" and not published.pending
    assert next(s for s in published.steps if s.id == reject.id).review == "rejected"


def test_revised_poh_rechecks_affected_steps(data_dir, air_raw, refs_dir, tmp_path):
    p = load_profile(PROFILE)
    for f in BASE_REFS:
        references.add_reference(p.procedure, refs_dir / f)
    for d in air_raw:
        ingest_run(d, p)
    create_guidelines(p, LLM)
    approve(p)
    rev_b = tmp_path / "poh_rev_b.md"
    rev_b.write_text((refs_dir / "c172_poh_notes.md").read_text(encoding="utf-8")
                     .replace('revision: "A"', 'revision: "B"').replace("max: 85", "max: 75")
                     .replace("Max 85 KIAS", "Max 75 KIAS"), encoding="utf-8")
    references.add_reference(p.procedure, rev_b)
    v = update_guidelines(p, LLM)
    flaps20 = next(s for s in v.steps if s.key == "toggle:flaps_idx:2")
    assert any("85 → max 75" in r for r in flaps20.change_reasons)
    assert flaps20.facts["trigger.ias_kt"].value <= 75 and flaps20.verified
    assert all(s.phase not in ("taxi", "takeoff", "climb") for s in v.steps if s.change)
    assert v.metrics["within_limits"] == 1.0


def test_edit_is_learned_and_wrong_numbers_flagged(data_dir, air_raw):
    p = load_profile(PROFILE)
    for d in air_raw:
        ingest_run(d, p)
    v1 = create_guidelines(p, LLM)
    approve(p)
    flaps10 = next(s for s in v1.steps if s.key == "toggle:flaps_idx:1")
    v2 = apply_edit(p, flaps10.id, "action", f"Select {flaps10.label}, then check the indicator.")
    assert v2.version == 2 and v2.mode == "edit"
    v3 = update_guidelines(p, LLM)
    flaps20 = next(s for s in v3.steps if s.key == "toggle:flaps_idx:2")
    assert flaps20.action == f"Select {flaps20.label}, then check the indicator."
    assert "Instructor correction applied (house style)" in flaps20.change_reasons

    rotate = next(s for s in v3.steps if s.key == "threshold:ias_kt:above:55")
    v4 = apply_edit(p, rotate.id, "typical", "Instructors rotate at 95 kt.")
    edited = next(s for s in v4.steps if s.id == rotate.id)
    assert not edited.verified
    assert any("95" in f.message for f in v4.flags)
