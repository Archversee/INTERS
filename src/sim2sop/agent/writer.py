"""Writer: fill a step's text fields from its facts.

- LLMWriter: prompts the configured model (Ollama / OpenAI-compatible / Anthropic) with
  the step's facts, the instructor's callout, the cited reference and relevant past
  corrections; the model may call `search_references`. Returns JSON.
- TemplateWriter: deterministic sentences from the facts, plus learned patterns from
  instructor corrections. Used when no LLM is configured or the LLM call fails.
"""

from __future__ import annotations

import json

from sim2sop.agent.llm import LLMClient, LLMError, parse_json, run_with_tools
from sim2sop.agent.verifier import facts_table
from sim2sop.learn import memory
from sim2sop.profiles import Profile
from sim2sop.rag.index import ReferenceIndex
from sim2sop.schema import Step

FIELDS = ("trigger", "action", "intent", "expected", "tolerance")


class TemplateWriter:
    name = "template"

    def __init__(self, profile: Profile):
        self.profile = profile

    def _phrase(self, step: Step, prefix: str, signals: list[str]) -> list[str]:
        out = []
        for sig in signals:
            f = step.facts.get(f"{prefix}.{sig}")
            if f is not None:
                out.append(f"{self.profile.label(sig)} {f.fmt()}")
        return out

    def write(self, step: Step, feedback: list[str] | None = None) -> dict[str, str | None]:
        p = self.profile
        if step.kind == "threshold" and "action" in step.facts:
            trigger = f"When {p.label(step.signal)} reaches {step.facts['action'].fmt()}."
        else:
            parts = self._phrase(step, "trigger", p.trigger_for(step.phase))
            trigger = f"At {', '.join(parts)}." if parts else ""
        expected_parts = self._phrase(step, "result", p.result_for(step.phase))
        expected = (f"Within {p.events.settle_s:g} s: {', '.join(expected_parts)}."
                    if expected_parts else "")
        tol_parts: dict[str, str] = {}   # one band per signal
        for name in ["action"] + [f"trigger.{s}" for s in p.trigger_for(step.phase)]:
            f = step.facts.get(name)
            if f is not None and f.tol is not None and f.unit and f.signal not in tol_parts:
                tol_parts[f.signal] = f"{p.label(f.signal)} {f.fmt_tol()}"
        tolerance = "; ".join(tol_parts.values())
        if tolerance:
            tolerance += f" (from {step.seen_in} run{'s' if step.seen_in != 1 else ''})"
        out = {
            "trigger": trigger,
            "action": f"{step.label}.",
            "intent": f'"{step.intent}"' if step.intent else None,
            "expected": expected,
            "tolerance": tolerance,
        }
        # Learned wording from instructor corrections on the same kind of step. Merged steps
        # ("Flaps UP; Landing light OFF") only take corrections made on that exact step.
        for field in ("trigger", "action", "expected", "tolerance"):
            for c in memory.retrieve(p.domain, p.procedure, step.phase, step.kind, step.key, field=field):
                same_kind = memory._signal_of(c.step_key) == step.signal and "+" not in step.key
                if c.pattern and (c.step_key == step.key or same_kind):
                    filled = memory.fill_pattern(c.pattern, step.facts, step.label, step.intent)
                    if filled:
                        out[field] = filled
                        break
        return out


SYSTEM_PROMPT = """You write one step of a Standard Operating Procedure (SOP) from verified simulator telemetry.

Rules:
- Use ONLY numbers that appear in FACTS (rounded to the precision shown) or in the step label. Never invent numbers.
- trigger: the condition that starts the step. action: what to do, imperative voice, keep the step label's wording.
- intent: why, based on the instructor's callout; null if there is none.
- expected: the expected result after the action, using result.* facts.
- tolerance: the ± bands for the values you state.
- Follow the instructor corrections shown: they are the house style.
- Reply with one JSON object with keys: trigger, action, intent, expected, tolerance.
"""

SEARCH_TOOL = {
    "name": "search_references",
    "description": "Search the domain reference documents (POH, handbooks, regulations).",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
}


class LLMWriter:
    def __init__(self, llm: LLMClient, profile: Profile, refs: ReferenceIndex, use_tools: bool = True):
        self.llm = llm
        self.profile = profile
        self.refs = refs
        self.use_tools = use_tools
        self.fallback = TemplateWriter(profile)
        self.name = llm.name

    def _prompt(self, step: Step, feedback: list[str] | None) -> str:
        p = self.profile
        corrections = memory.retrieve(p.domain, p.procedure, step.phase, step.kind, step.key, k=3)
        payload = {
            "procedure": p.title,
            "phase": step.phase,
            "step_label": step.label,
            "event_kind": step.kind,
            "instructor_callout": step.intent,
            "mandatory": step.mandatory,
            "seen_in_runs": f"{step.seen_in}/{step.run_count}",
            "FACTS": facts_table(step),
            "reference": step.reference.model_dump(include={"doc", "section", "text"}) if step.reference else None,
        }
        msg = f"Style: {p.style}\n\nSTEP:\n{json.dumps(payload, indent=2, ensure_ascii=False)}"
        if corrections:
            msg += "\n\nINSTRUCTOR CORRECTIONS ON SIMILAR STEPS (before -> after):\n" + "\n".join(
                f"- [{c.field}] {c.before!r} -> {c.after!r}" for c in corrections)
        if feedback:
            msg += "\n\nYOUR PREVIOUS DRAFT FAILED VERIFICATION. Fix these:\n" + "\n".join(f"- {f}" for f in feedback)
        return msg

    def write(self, step: Step, feedback: list[str] | None = None) -> dict[str, str | None]:
        messages = [{"role": "user", "content": self._prompt(step, feedback)}]
        try:
            if self.use_tools:
                handlers = {"search_references": lambda query: [c.model_dump() for c in self.refs.search(query)]}
                try:
                    reply = run_with_tools(self.llm, messages, [SEARCH_TOOL], handlers, system=SYSTEM_PROMPT)
                except LLMError:
                    self.use_tools = False  # model/provider without tool support: plain JSON from now on
                    reply = self.llm.chat(messages, system=SYSTEM_PROMPT, json_mode=True)
            else:
                reply = self.llm.chat(messages, system=SYSTEM_PROMPT, json_mode=True)
            data = parse_json(reply.text)
        except (LLMError, ValueError) as e:
            step.flags.append(f"LLM writer failed ({e.__class__.__name__}); template used")
            return self.fallback.write(step, feedback)
        return {f: (str(data[f]).strip() if data.get(f) not in (None, "") else None) for f in FIELDS}
