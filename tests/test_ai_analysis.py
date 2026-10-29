import json

from sim2sop import config, references, storage
from sim2sop.agent import insights, manuals
from sim2sop.agent.llm import LLMClient, LLMReply, NoLLM
from sim2sop.generate.land import generate_land_runs
from sim2sop.ingest import ingest_run
from sim2sop.pipeline import create_guidelines
from sim2sop.profiles import load_profile

MANUAL = """# Parallel parking manual

## Reversing
Reverse slowly. The speed must never exceed 6 km/h while reversing into the space.

## Finishing
Apply the handbrake before you leave the car. Park within 0.3 m of the kerb.
"""


class ScriptedLLM(LLMClient):
    """Answers every call with the same JSON reply."""
    provider = "fake"

    def __init__(self, reply: dict):
        super().__init__(model="scripted")
        self.reply, self.calls = reply, 0

    def available(self) -> bool:
        return True

    def chat(self, messages, system=None, tools=None, json_mode=False, temperature=0.2):
        self.calls += 1
        return LLMReply(text=json.dumps(self.reply))


def test_manual_rules_are_checked_against_the_text_and_the_profile(data_dir, tmp_path):
    p = load_profile("land_parallel_park")
    path = tmp_path / "manual.md"
    path.write_text(MANUAL, encoding="utf-8")
    llm = ScriptedLLM({
        "limits": [
            {"signal": "speed_kmh", "max": 6, "when": "reverse == 1", "phases": ["reverse_in"],
             "text": "Reverse no faster than 6 km/h", "quote": "must never exceed 6 km/h while reversing"},
            {"signal": "speed_kmh", "max": 3.7, "text": "converted", "quote": "must never exceed 6 km/h"},
            {"signal": "kerb_gap_m", "max": 0.3, "text": "no such signal", "quote": "within 0.3 m of the kerb"},
            {"signal": "speed_kmh", "max": 4, "text": "invented", "quote": "Never go faster than 4 km/h"},
        ],
        "requirements": [
            {"key": "toggle:hand_brake:1", "text": "Apply the handbrake", "quote": "Apply the handbrake"},
            {"key": "toggle:horn:1", "text": "Sound the horn", "quote": "Reverse slowly"},
        ]})
    reading = manuals.read_manual(llm, path, p, title="Parking manual")
    assert reading.sections_read == 2 and llm.calls == 2   # the title-only section states no rule
    ok = reading.accepted
    assert [(x.kind, x.signal or x.key) for x in ok] == [("limit", "speed_kmh"), ("requirement", "toggle:hand_brake:1")]
    reasons = " | ".join(pr for x in reading.set_aside for pr in x.problems)
    for why in ("not written in the quote", "unknown signal", "not in the manual", "not an action"):
        assert why in reasons

    doc = manuals.add_manual(p.procedure, path, "Parking manual", "sop", "1", ok)
    assert [(lim.signal, lim.max, lim.when) for lim in doc.limits] == [("speed_kmh", 6.0, "reverse == 1")]
    assert [r.key for r in references.active_requirements(p.procedure)] == ["toggle:hand_brake:1"]


def test_ai_findings_must_use_the_patterns_numbers(data_dir, tmp_path):
    p = load_profile("land_parallel_park")
    references.sync_builtin(p.procedure, config.REFERENCES_DIR / "land")
    for run in generate_land_runs(tmp_path, n=5):
        ingest_run(run, p)
    doc = create_guidelines(p, NoLLM("template"))
    runs = storage.list_runs(p.procedure)

    plain = insights.explain(doc, runs, p, NoLLM("template"))
    assert plain.patterns and plain.findings and plain.writer == "template"
    habit = next(x for x in plain.patterns if x.kind == "habit")   # the forward shuffle is instructor 2's
    assert "Instructor 2" in habit.text

    llm = ScriptedLLM({"findings": [
        {"headline": "Only Instructor 2 shuffles forward at the end, in 2 of 2 runs.", "detail": "",
         "action": "Brief the team.", "patterns": [habit.id]},
        {"headline": "Instructor 2 parks 37% better.", "patterns": [habit.id]},
        {"headline": "Something else.", "patterns": ["P99"]},
    ]})
    ai = insights.explain(doc, runs, p, llm)
    assert ai.writer == "fake:scripted" and [f.headline for f in ai.findings] == [
        "Only Instructor 2 shuffles forward at the end, in 2 of 2 runs."]
    assert {pr for f in ai.set_aside for pr in f.problems} >= {"37 is not in the patterns it cites",
                                                               "unknown pattern P99"}
    insights.save(ai)

    # A limit on the wrong signal shows in the data: every drive turns the wheel past 50°.
    wrong = manuals.Proposal(kind="limit", signal="steer_deg", max=50, quote="50")
    right = manuals.Proposal(kind="limit", signal="kerb_angle_deg", max=50, quote="50")
    assert manuals.recorded_breaks(wrong, p.procedure, p) == (5, 5)
    assert manuals.recorded_breaks(right, p.procedure, p) == (0, 5)
    assert insights.load(p.procedure).findings[0].patterns == [habit.id]


def test_loose_answers_from_a_small_model_map_to_the_profile():
    p = load_profile("land_parallel_park")
    got = manuals._proposals({"limits": [{"signal": "Speed", "max": "5 km/h", "phases": "Reverse In"}],
                              "requirements": [{"key": "Handbrake APPLY"}]}, "s", p)
    assert [(x.signal, x.max, x.phases) for x in got if x.kind == "limit"] == [("speed_kmh", 5.0, ["reverse_in"])]
    assert [x.key for x in got if x.kind == "requirement"] == ["toggle:hand_brake:1"]


def test_findings_may_state_a_gap_but_not_overclaim_one_run():
    from sim2sop.learn.patterns import Pattern
    by_id = {"P1": Pattern(id="P1", kind="instructor", text="x", numbers=[8.6, 9.0, 3, 2], runs=["a", "b"]),
             "P2": Pattern(id="P2", kind="limit_break", text="1 of 5 runs break it", numbers=[1, 5], runs=["a"])}
    gap = insights.check(insights.Finding(headline="Instructor 1 stops 0.4 m earlier.", patterns=["P1"]), by_id)
    assert gap.problems == []
    often = insights.check(insights.Finding(headline="Instructor 1 often breaks it.", patterns=["P2"]), by_id)
    assert often.problems == ['says "often" but the patterns it cites rest on one run']
