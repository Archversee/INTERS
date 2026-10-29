import json

import pytest

from sim2sop.agent.graph import build_graph
from sim2sop.agent.llm import LLMClient, LLMReply, LLMUnreachable, ToolCall, get_llm
from sim2sop.agent.planner import plan
from sim2sop.agent.verifier import verify_step
from sim2sop.agent.writer import LLMWriter, _restates
from sim2sop.ingest import ingest_run
from sim2sop.learn import memory
from sim2sop.learn.consensus import typical_range
from sim2sop.profiles import load_profile
from sim2sop.rag.index import ReferenceIndex
from sim2sop.schema import Fact, Limit, Step


def _step(**kw) -> Step:
    facts = {
        "trigger.ias_kt": Fact(signal="ias_kt", value=55.2, lo=54, hi=57, unit="kt", limit_max=60),
        "result.vs_fpm": Fact(signal="vs_fpm", value=-410.0, lo=-450, hi=-370, unit="fpm"),
    }
    return Step(id="s01", phase="takeoff", key="threshold:ias_kt:above:55", kind="threshold",
                signal="ias_kt", label="Rotate", facts=facts, **kw)


def test_verifier_accepts_facts_and_rejects_invented_or_out_of_limit_numbers():
    p = load_profile("air_c172_circuit")
    good = _step(trigger="When airspeed reaches 55 kt.", typical="Instructors: 54–57 kt; then -450 to -370 fpm.")
    assert verify_step(good, p).ok
    v = verify_step(_step(trigger="When airspeed reaches 70 kt."), p)
    assert not v.ok and "70" in v.errors[0]
    callout = _step(trigger="When airspeed reaches 55 kt.", intent='"Power 1500"')
    v = verify_step(callout, p)
    assert v.ok and v.warnings  # callout numbers warn, they do not fail the step
    # A number that matches telemetry but is past the hard limit fails.
    step = _step()
    step.facts["trigger.ias_kt"] = Fact(signal="ias_kt", value=89, lo=88, hi=90, unit="kt", limit_max=85)
    step.trigger = "At 89 kt."
    v = verify_step(step, p)
    assert not v.ok and "hard limit" in v.errors[0]


def test_typical_range_is_robust_to_one_bad_run():
    _, lo, hi = typical_range([55, 55.5, 54.8, 55.2, 70], default_tol=5)
    assert hi - lo < 4  # IQR ignores the 70 kt outlier
    _, lo1, hi1 = typical_range([55], default_tol=5)
    assert (hi1 - lo1) == pytest.approx(10)


class FakeLLM(LLMClient):
    """First draft invents a number; the rewrite uses the facts."""
    provider = "fake"

    def __init__(self):
        super().__init__(model="fake")
        self.calls = 0

    def chat(self, messages, system=None, tools=None, json_mode=False, temperature=0.2):
        self.calls += 1
        prompt = messages[-1]["content"]
        failed = "FAILED VERIFICATION" in prompt
        payload = json.loads(prompt.split("STEP:\n", 1)[1].split("\n\n", 1)[0])
        ias = payload["FACTS"]["trigger.ias_kt"].split()[1]  # "typical 55 kt (range ...)"
        stated = ias if failed else "12345"
        return LLMReply(text=json.dumps({
            "trigger": f"At {stated} kt.", "action": payload["step_label"], "intent": None,
            "typical": "", "limit": ""}))


def test_graph_rewrites_until_verified(data_dir, air_raw):
    p = load_profile("air_c172_circuit")
    run = ingest_run(air_raw[1], p)
    refs = ReferenceIndex([])
    fake = FakeLLM()
    writer = LLMWriter(fake, p, refs, use_tools=False)
    graph = build_graph(p, writer, lambda: plan([run], p, refs, [], []), max_rounds=2)
    state = graph.invoke({})
    assert all(s.verified for s in state["steps"])
    assert fake.calls == 2 * len(state["steps"])  # one failed draft + one rewrite per step


class StuckLLM(LLMClient):
    """A server that is up but never answers in time."""
    provider = "fake"

    def __init__(self):
        super().__init__(model="stuck")
        self.calls = 0

    def chat(self, *args, **kwargs):
        self.calls += 1
        raise LLMUnreachable("stuck: ReadTimeout after 120s")


def test_unresponsive_llm_is_asked_once_then_template_is_used(data_dir, air_raw):
    p = load_profile("air_c172_circuit")
    run = ingest_run(air_raw[1], p)
    refs = ReferenceIndex([])
    stuck = StuckLLM()
    writer = LLMWriter(stuck, p, refs)
    state = build_graph(p, writer, lambda: plan([run], p, refs, [], [])).invoke({})
    assert stuck.calls == 1  # not once per step (each would wait for the full timeout)
    assert writer.down and writer.fallback_steps == len(state["steps"])
    assert all(s.verified and s.action for s in state["steps"])


def test_limit_text_is_quoted_not_model_written(data_dir, air_raw):
    p = load_profile("air_c172_circuit")
    run = ingest_run(air_raw[1], p)
    lim = Limit(id="vfe", signal="ias_kt", max=85, when="flaps_idx >= 2", text="Max 85 KIAS with flaps beyond 10°",
                doc_title="POH", revision="A")
    result = plan([run], p, ReferenceIndex([]), [lim], [])
    flaps20 = next(s for s in result.steps if s.key == "toggle:flaps_idx:2")
    flaps10 = next(s for s in result.steps if s.key == "toggle:flaps_idx:1")
    assert [x.id for x in flaps20.limits] == ["vfe"] and not flaps10.limits
    out = LLMWriter(FakeLLM(), p, ReferenceIndex([]), use_tools=False).write(flaps20)
    assert out["limit"].startswith("Max 85 KIAS")


def test_providers_build_requests_and_parse_replies(monkeypatch):
    seen = {}

    def fake_post(self, url, payload, headers=None):
        seen[self.provider] = (url, payload, headers)
        if self.provider == "anthropic":
            return {"content": [{"type": "text", "text": "hi"},
                                {"type": "tool_use", "id": "tu1", "name": "search_references",
                                 "input": {"query": "flaps"}}]}
        msg = {"content": "hi", "tool_calls": [{"id": "c1", "function": {
            "name": "search_references", "arguments": '{"query": "flaps"}'}}]}
        return {"message": msg} if self.provider == "ollama" else {"choices": [{"message": msg}]}

    monkeypatch.setattr(LLMClient, "_post", fake_post)
    tool = {"name": "search_references", "description": "", "parameters": {"type": "object", "properties": {}}}
    history = [{"role": "user", "content": "q"},
               {"role": "assistant", "content": "", "tool_calls": [ToolCall("c0", "search_references", {})]},
               {"role": "tool", "tool_call_id": "c0", "name": "search_references", "content": "[]"}]
    for provider in ("ollama", "openai", "anthropic"):
        client = get_llm(provider=provider, api_key="k")
        reply = client.chat(history, system="sys", tools=[tool])
        assert reply.text == "hi"
        assert reply.tool_calls[0].name == "search_references"
        assert reply.tool_calls[0].arguments == {"query": "flaps"}

    url, payload, headers = seen["anthropic"]
    assert url.endswith("/v1/messages") and headers["x-api-key"] == "k"
    assert payload["system"] == "sys" and payload["tools"][0]["input_schema"]
    assert payload["messages"][2]["content"][0]["type"] == "tool_result"
    url, payload, headers = seen["openai"]
    assert url.endswith("/chat/completions") and headers["Authorization"] == "Bearer k"
    assert payload["messages"][0] == {"role": "system", "content": "sys"}
    assert seen["ollama"][0].endswith("/api/chat")


def test_unknown_provider_rejected():
    with pytest.raises(ValueError):
        get_llm(provider="nope")


def test_correction_pattern_round_trip():
    facts = {"result.ias_kt": Fact(signal="ias_kt", value=81.3, lo=79, hi=84, unit="kt")}
    pattern = memory.to_pattern("Select Flaps 10°, hold 81 kt (79–84).", facts, label="Flaps 10°")
    assert pattern == "Select {label}, hold {fact:result.ias_kt} kt ({fact:result.ias_kt.lo}–{fact:result.ias_kt.hi})."
    other = {"result.ias_kt": Fact(signal="ias_kt", value=72.0, lo=70, hi=74, unit="kt")}
    assert memory.fill_pattern(pattern, other, "Flaps 20°", None) == "Select Flaps 20°, hold 72 kt (70–74)."
    assert memory.fill_pattern(pattern, {}, "Flaps 20°", None) is None


def test_trigger_that_repeats_the_action_is_caught():
    assert _restates("Flaps 20°", "Set flaps to 20°", "Flaps 20°")
    assert _restates("Turn right to heading 28°", "Turn right to heading 28°", "Turn right to heading 28°")
    assert not _restates("IAS 80 kt", "Set flaps to 20°", "Flaps 20°")
    assert not _restates(None, "Set flaps to 20°", "Flaps 20°")


def test_typical_range_stays_on_the_side_of_zero_the_flights_were_on():
    _, lo, _ = typical_range([0.0, 0.4, 1.0], default_tol=5)
    assert lo == 0.0
    _, _, hi = typical_range([-700.0, -650.0, -600.0], default_tol=150)
    assert hi <= 0.0
