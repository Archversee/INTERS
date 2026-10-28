"""Writer: fill a guideline's text fields from its facts and hard limits.

Guideline template: when (trigger) -> what to do (action) -> why (intent, from the
callout) -> typical range (from runs) -> hard limit (from the reference SOPs).

- LLMWriter: prompts the configured model (Ollama / OpenAI-compatible / Anthropic) with
  the step's facts, limits, the instructor's callout, the cited reference and relevant
  past corrections; the model may call `search_references`. Returns JSON.
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

FIELDS = ("trigger", "action", "intent", "typical", "limit")
EDITABLE = ("trigger", "action", "typical")   # fields learned patterns may rewrite


def limit_text(step: Step) -> str:
    """Hard limits are quoted from the reference, never paraphrased."""
    return "; ".join(f"{lim.text} — {lim.cite()}" for lim in step.limits)


class TemplateWriter:
    name = "template"

    def __init__(self, profile: Profile):
        self.profile = profile

    def _ranges(self, step: Step, prefix: str, signals: list[str]) -> list[str]:
        out = []
        for sig in signals:
            f = step.facts.get(f"{prefix}.{sig}")
            if f is not None:
                out.append(f"{self.profile.label(sig)} {f.fmt_range()}")
        return out

    def write(self, step: Step, feedback: list[str] | None = None) -> dict[str, str | None]:
        p = self.profile
        if step.kind == "threshold" and "action" in step.facts:
            trigger = f"When {p.label(step.signal)} reaches {step.facts['action'].fmt()}."
        else:
            parts = [f"{p.label(s)} {step.facts[f'trigger.{s}'].fmt()}"
                     for s in p.trigger_for(step.phase) if f"trigger.{s}" in step.facts]
            trigger = f"At about {', '.join(parts)}." if parts else ""

        when = self._ranges(step, "trigger", p.trigger_for(step.phase))
        if step.kind in ("control", "turn") and "action" in step.facts:
            when.insert(0, f"{p.label(step.signal)} {step.facts['action'].fmt_range()}")
        after = self._ranges(step, "result", p.result_for(step.phase))
        n = len(step.runs)
        typical = ""
        if when:
            typical = f"Instructors: {', '.join(when)}"
            if after:
                typical += f"; then {', '.join(after)}"
            typical += f" ({n} run{'s' if n != 1 else ''})."
        out = {
            "trigger": trigger,
            "action": f"{step.label}.",
            "intent": f'"{step.intent}"' if step.intent else None,
            "typical": typical,
            "limit": limit_text(step),
        }
        # Learned wording from instructor corrections on the same kind of step. Merged steps
        # ("Flaps UP; Landing light OFF") only take corrections made on that exact step.
        for field in EDITABLE:
            for c in memory.retrieve(p.domain, p.procedure, step.phase, step.kind, step.key, field=field):
                same_kind = memory._signal_of(c.step_key) == step.signal and "+" not in step.key
                if c.pattern and (c.step_key == step.key or same_kind):
                    filled = memory.fill_pattern(c.pattern, step.facts, step.label, step.intent)
                    if filled:
                        out[field] = filled
                        break
        return out


SYSTEM_PROMPT = """You write one step of advisory flight-training GUIDELINES from verified simulator telemetry.
Guidelines describe how experienced instructors do it; the reference SOPs set hard limits and always win.

Rules:
- Use ONLY numbers that appear in FACTS (typical value or range ends, rounded to the precision shown) or in the step label. Never invent numbers.
- Never state a value past a HARD LIMIT shown in FACTS.
- trigger: when to act. action: what to do, imperative voice, keep the step label's wording.
- intent: why, based on the instructor's callout; null if there is none.
- typical: the range instructors typically use, from the FACTS ranges.
- limit: copy LIMITS verbatim, or "" if there are none.
- Follow the instructor corrections shown: they are the house style.
- Reply with one JSON object with keys: trigger, action, intent, typical, limit.
"""

SEARCH_TOOL = {
    "name": "search_references",
    "description": "Search the procedure's reference SOPs, POH, handbooks and NOTAMs.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
}


class LLMWriter:
    def __init__(self, llm: LLMClient, profile: Profile, refs: ReferenceIndex, use_tools: bool = True):
        self.llm = llm
        self.profile = profile
        self.refs = refs
        self.use_tools = use_tools and not refs.empty
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
            "core_guideline": step.mandatory,
            "required_by": step.required_by,
            "seen_in_runs": f"{step.seen_in}/{step.run_count}",
            "FACTS": facts_table(step),
            "LIMITS": limit_text(step),
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
        out = {f: (str(data[f]).strip() if data.get(f) not in (None, "") else None) for f in FIELDS}
        out["limit"] = limit_text(step)  # limits are quoted from the reference, not model-written
        return out
