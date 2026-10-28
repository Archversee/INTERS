import json

from sim2sop.agent import llm as llm_mod
from sim2sop.agent.graph import build_graph
from sim2sop.agent.llm import LLMClient, LLMReply, ToolCall, get_llm
from sim2sop.agent.verifier import verify_step
from sim2sop.agent.writer import LLMWriter
from sim2sop.ingest import ingest_run
from sim2sop.learn import memory
from sim2sop.profiles import load_profile
from sim2sop.rag.index import ReferenceIndex
from sim2sop.schema import Fact, Step


def _step(**kw) -> Step:
    facts = {
        "trigger.ias_kt": Fact(signal="ias_kt", value=55.2, unit="kt", tol=1.4),
        "result.vs_fpm": Fact(signal="vs_fpm", value=-410.0, unit="fpm", tol=150),
    }
    return Step(id="s01", phase="takeoff", key="threshold:ias_kt:above:55", kind="threshold",
                signal="ias_kt", label="Rotate", facts=facts, **kw)


def test_verifier_accepts_facts_and_rejects_invented_numbers():
    p = load_profile("air_c172_circuit")
    good = _step(trigger="When airspeed reaches 55 kt.", expected="Descend at 410 fpm.",
                 tolerance="airspeed ±1 kt (from 1 run)")
    assert verify_step(good, p).ok
    bad = _step(trigger="When airspeed reaches 60 kt.")
    v = verify_step(bad, p)
    assert not v.ok and "60" in v.errors[0]
    callout = _step(trigger="When airspeed reaches 55 kt.", intent='"Power 1500"')
    v = verify_step(callout, p)
    assert v.ok and v.warnings  # callout numbers warn, they do not fail the step


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
        ias = next((v.split()[0] for k, v in payload["FACTS"].items() if k == "trigger.ias_kt"), "0")
        stated = ias if failed else "12345"
        return LLMReply(text=json.dumps({
            "trigger": f"At {stated} kt.", "action": payload["step_label"], "intent": None,
            "expected": "", "tolerance": ""}))


def test_graph_rewrites_until_verified(data_dir, air_raw):
    p = load_profile("air_c172_circuit")
    run = ingest_run(air_raw[0], p)
    refs = ReferenceIndex.for_domain(p.references)
    fake = FakeLLM()
    graph = build_graph(p, refs, LLMWriter(fake, p, refs, use_tools=False), max_rounds=2)
    state = graph.invoke({"runs": [run], "previous": None})
    assert all(s.verified for s in state["steps"])
    assert fake.calls == 2 * len(state["steps"])  # one failed draft + one rewrite per step


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
    import pytest
    with pytest.raises(ValueError):
        llm_mod.get_llm(provider="nope")


def test_correction_pattern_round_trip():
    facts = {"result.ias_kt": Fact(signal="ias_kt", value=81.3, unit="kt", tol=3)}
    pattern = memory.to_pattern("Select Flaps 10°, hold 81 kt.", facts, label="Flaps 10°")
    assert pattern == "Select {label}, hold {fact:result.ias_kt} kt."
    other = {"result.ias_kt": Fact(signal="ias_kt", value=72.0, unit="kt", tol=3)}
    assert memory.fill_pattern(pattern, other, "Flaps 20°", None) == "Select Flaps 20°, hold 72 kt."
    assert memory.fill_pattern(pattern, {}, "Flaps 20°", None) is None
