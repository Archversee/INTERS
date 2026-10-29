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

from sim2sop.agent.llm import LLMClient, LLMError, LLMUnreachable, parse_json, run_with_tools
from sim2sop.agent.verifier import facts_table
from sim2sop.learn import memory
from sim2sop.profiles import Profile
from sim2sop.rag.index import ReferenceIndex
from sim2sop.schema import Step

FIELDS = ("trigger", "action", "intent", "typical", "limit")
EDITABLE = ("trigger", "action", "typical")   # fields learned patterns may rewrite


def limit_text(step: Step) -> str:
    """Hard limits are quoted from the reference, never paraphrased. The full citation
    (revision, section) stays on the step's limits."""
    return "; ".join(f"{lim.text} — {lim.doc_title}" for lim in step.limits)


# Units that say what the number is on their own ("74 kt" needs no "airspeed").
SELF_EXPLANATORY = {"kt", "ft", "fpm", "%", "rpm", "km/h"}


class TemplateWriter:
    name = "template"

    def __init__(self, profile: Profile):
        self.profile = profile

    def _say(self, sig: str, text: str) -> str:
        p = self.profile
        if p.unit(sig) in SELF_EXPLANATORY:
            return text
        label = p.label(sig)
        return f"{text} {label}" if label.startswith("from ") else f"{label} {text}"

    def _ranges(self, step: Step, prefix: str, signals: list[str]) -> list[str]:
        return [self._say(sig, step.facts[f"{prefix}.{sig}"].fmt_range())
                for sig in signals if f"{prefix}.{sig}" in step.facts]

    def write(self, step: Step, feedback: list[str] | None = None) -> dict[str, str | None]:
        p = self.profile
        if step.kind == "threshold" and "action" in step.facts:
            trigger = f"When {p.label(step.signal)} reaches {step.facts['action'].fmt()}."
        else:
            parts = [self._say(s, step.facts[f"trigger.{s}"].fmt())
                     for s in p.trigger_for(step.phase) if f"trigger.{s}" in step.facts]
            trigger = f"At about {', '.join(parts)}." if parts else ""

        # Typical: the action's own range, the conditions it is done at, and only the
        # outcomes this phase states on purpose (e.g. distance from the runway on downwind).
        context = ", ".join(self._ranges(step, "trigger", p.trigger_for(step.phase)))
        head = ""
        if step.kind in ("control", "turn") and "action" in step.facts:
            head = f"{p.label(step.signal)} {step.facts['action'].fmt_range()}"
        extra = [s for s in p.result_for(step.phase) if s not in p.result_signals]
        after = ", ".join(self._ranges(step, "result", extra))
        typical = ""
        if head or context:
            typical = "Usually " + (f"{head} at {context}" if head and context else head or context)
            if after:
                typical += f"; then {after}"
            typical += "."
        out = {
            "trigger": trigger,
            "action": f"{step.label}.",
            "intent": f'"{step.intent}"' if step.intent else None,
            "typical": typical,
            "limit": limit_text(step),
        }
        return apply_learned(out, p, step)


def learned(profile: Profile, step: Step, field: str) -> str | None:
    """Wording learned from instructor corrections on the same kind of step. Merged steps
    ("Flaps UP; Landing light OFF") only take corrections made on that exact step, and a
    correction that kept a literal number ("Select Flaps 10°" where the label was not
    recognised) is never copied onto another step."""
    for c in memory.retrieve(profile.domain, profile.procedure, step.phase, step.kind, step.key, field=field):
        portable = "{" in (c.pattern or "") or not any(ch.isdigit() for ch in c.pattern or "")
        same_kind = memory._signal_of(c.step_key) == step.signal and "+" not in step.key and portable
        if c.pattern and (c.step_key == step.key or same_kind):
            filled = memory.fill_pattern(c.pattern, step.facts, step.label, step.intent)
            if filled:
                return filled
    return None


def apply_learned(out: dict[str, str | None], profile: Profile, step: Step) -> dict[str, str | None]:
    """Instructor corrections are the house style: they win over any writer's wording."""
    for field in EDITABLE:
        if filled := learned(profile, step, field):
            out[field] = filled
    return out


def rewrite_fields(step: Step, fields: list[str], profile: Profile) -> dict[str, str | None]:
    """Only the parts that changed, keeping the rest of the step's text: the quoted hard
    limit, and fields restyled by an instructor correction."""
    out: dict[str, str | None] = {}
    for field in fields:
        if field == "limit":
            out["limit"] = limit_text(step)
        elif field in EDITABLE and (filled := learned(profile, step, field)):
            out[field] = filled
    return out


SYSTEM_PROMPT = """You write one step of advisory flight-training GUIDELINES from verified simulator telemetry.
Guidelines describe how experienced instructors do it; the reference SOPs set hard limits and always win.

Rules:
- Use ONLY numbers that appear in FACTS (typical value or range ends, rounded to the precision shown) or in the step label. Never invent numbers.
- Never state a value past a HARD LIMIT shown in FACTS.
- trigger: when to act: the condition to watch (airspeed, height, position, a callout). Never just repeat the action.
- action: what to do, imperative voice, keep the step label's wording.
- intent: why, based on the instructor's callout; null if there is none.
- typical: the range instructors typically use, written as a range ("lo–hi unit") whenever FACTS give one.
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
        self.down: str | None = None   # set when the server stops answering: the rest of the build
        self.fallback_steps = 0        # then uses the template instead of waiting on every step

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
        if self.down:
            self.fallback_steps += 1
            return self.fallback.write(step, feedback)
        messages = [{"role": "user", "content": self._prompt(step, feedback)}]
        try:
            if self.use_tools:
                handlers = {"search_references": lambda query: [c.model_dump() for c in self.refs.search(query)]}
                try:
                    reply = run_with_tools(self.llm, messages, [SEARCH_TOOL], handlers, system=SYSTEM_PROMPT)
                except LLMUnreachable:
                    raise
                except LLMError:
                    self.use_tools = False  # model/provider without tool support: plain JSON from now on
                    reply = self.llm.chat(messages, system=SYSTEM_PROMPT, json_mode=True)
            else:
                reply = self.llm.chat(messages, system=SYSTEM_PROMPT, json_mode=True)
            data = parse_json(reply.text)
        except LLMUnreachable as e:
            self.down = str(e)
            self.fallback_steps += 1
            return self.fallback.write(step, feedback)
        except (LLMError, ValueError) as e:
            step.flags.append(f"LLM writer failed ({e.__class__.__name__}); template used")
            return self.fallback.write(step, feedback)
        out = {f: (str(data[f]).strip() if data.get(f) not in (None, "") else None) for f in FIELDS}
        out["limit"] = limit_text(step)  # limits are quoted from the reference, not model-written
        if _restates(out["trigger"], out["action"], step.label):
            out["trigger"] = self.fallback.write(step)["trigger"]   # "when" must be a condition, not the action
        return apply_learned(out, self.profile, step)


def _plain(text: str | None) -> str:
    return "".join(ch for ch in (text or "").lower() if ch.isalnum())


def _restates(trigger: str | None, action: str | None, label: str) -> bool:
    """A trigger that only repeats the action ("Flaps 20°" — "Flaps 20°") says nothing about when."""
    t = _plain(trigger)
    return bool(t) and (t in _plain(action) or t in _plain(label))
