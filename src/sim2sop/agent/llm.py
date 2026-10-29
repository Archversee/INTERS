"""One chat interface over Ollama, OpenAI-compatible APIs and Anthropic.

Every provider takes the same inputs (system prompt, messages, optional tools) and
returns an `LLMReply`. Tools use a neutral spec, `{"name", "description", "parameters"}`
with a JSON-schema `parameters`, which each backend converts to its own format.
Switch providers with SIM2SOP_LLM_PROVIDER (see .env.example).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

from sim2sop import config


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMReply:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None


class LLMError(RuntimeError):
    pass


class LLMUnreachable(LLMError):
    """The server did not answer in time or refused the connection (as opposed to an error reply)."""


class LLMClient:
    provider = "base"

    def __init__(self, model: str, base_url: str = "", api_key: str = "", timeout: float = 120):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}"

    def chat(self, messages: list[dict], system: str | None = None,
             tools: list[dict] | None = None, json_mode: bool = False,
             temperature: float = 0.2) -> LLMReply:
        raise NotImplementedError

    def available(self) -> bool:
        return True

    def _post(self, url: str, payload: dict, headers: dict | None = None) -> dict:
        try:
            r = httpx.post(url, json=payload, headers=headers or {}, timeout=self.timeout)
        except httpx.TransportError as e:   # timeouts and connection failures
            raise LLMUnreachable(f"{self.name}: {e.__class__.__name__} after {self.timeout:.0f}s") from e
        except httpx.HTTPError as e:
            raise LLMError(f"{self.name}: {e}") from e
        if r.status_code >= 400:
            raise LLMError(f"{self.name}: HTTP {r.status_code}: {r.text[:500]}")
        return r.json()


class OllamaClient(LLMClient):
    provider = "ollama"

    def available(self) -> bool:
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=3)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    def models(self) -> list[str]:
        """Models pulled on the Ollama server (raises httpx.HTTPError if unreachable)."""
        r = httpx.get(f"{self.base_url}/api/tags", timeout=5)
        r.raise_for_status()
        return sorted(m["name"] for m in r.json().get("models", []))

    def chat(self, messages, system=None, tools=None, json_mode=False, temperature=0.2):
        msgs = ([{"role": "system", "content": system}] if system else []) + _to_openai_messages(messages)
        payload: dict[str, Any] = {
            "model": self.model, "messages": msgs, "stream": False,
            "options": {"temperature": temperature},
        }
        if tools:
            payload["tools"] = [_openai_tool(t) for t in tools]
        if json_mode:
            payload["format"] = "json"
        data = self._post(f"{self.base_url}/api/chat", payload)
        msg = data.get("message", {})
        calls = [
            ToolCall(id=uuid.uuid4().hex[:8], name=c["function"]["name"],
                     arguments=_as_dict(c["function"].get("arguments")))
            for c in msg.get("tool_calls") or []
        ]
        return LLMReply(text=msg.get("content", ""), tool_calls=calls, raw=data)


class OpenAIClient(LLMClient):
    """Any endpoint that speaks the OpenAI /chat/completions protocol."""

    provider = "openai"

    def available(self) -> bool:
        return bool(self.api_key) or "localhost" in self.base_url or "127.0.0.1" in self.base_url

    def chat(self, messages, system=None, tools=None, json_mode=False, temperature=0.2):
        msgs = ([{"role": "system", "content": system}] if system else []) + _to_openai_messages(messages)
        payload: dict[str, Any] = {"model": self.model, "messages": msgs, "temperature": temperature}
        if tools:
            payload["tools"] = [_openai_tool(t) for t in tools]
        if json_mode and not tools:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        data = self._post(f"{self.base_url}/chat/completions", payload, headers)
        msg = data["choices"][0]["message"]
        calls = [
            ToolCall(id=c.get("id") or uuid.uuid4().hex[:8], name=c["function"]["name"],
                     arguments=_as_dict(c["function"].get("arguments")))
            for c in msg.get("tool_calls") or []
        ]
        return LLMReply(text=msg.get("content") or "", tool_calls=calls, raw=data)


class AnthropicClient(LLMClient):
    provider = "anthropic"
    api_version = "2023-06-01"

    def available(self) -> bool:
        return bool(self.api_key)

    def chat(self, messages, system=None, tools=None, json_mode=False, temperature=0.2):
        payload: dict[str, Any] = {
            "model": self.model, "max_tokens": 2048, "temperature": temperature,
            "messages": _to_anthropic_messages(messages),
        }
        sys_prompt = system or ""
        if json_mode:
            sys_prompt += "\n\nRespond with a single JSON object and nothing else."
        if sys_prompt:
            payload["system"] = sys_prompt
        if tools:
            payload["tools"] = [
                {"name": t["name"], "description": t.get("description", ""), "input_schema": t["parameters"]}
                for t in tools
            ]
        headers = {"x-api-key": self.api_key, "anthropic-version": self.api_version}
        data = self._post(f"{self.base_url}/v1/messages", payload, headers)
        text = "".join(b.get("text", "") for b in data["content"] if b["type"] == "text")
        calls = [ToolCall(id=b["id"], name=b["name"], arguments=b.get("input") or {})
                 for b in data["content"] if b["type"] == "tool_use"]
        return LLMReply(text=text, tool_calls=calls, raw=data)


class NoLLM(LLMClient):
    """Placeholder for provider "none": callers fall back to the template writer."""

    provider = "none"

    def available(self) -> bool:
        return False

    def chat(self, *args, **kwargs):
        raise LLMError("No LLM configured (SIM2SOP_LLM_PROVIDER=none)")


_PROVIDERS: dict[str, type[LLMClient]] = {
    "ollama": OllamaClient, "openai": OpenAIClient, "anthropic": AnthropicClient, "none": NoLLM,
}


def get_llm(provider: str | None = None, model: str | None = None,
            base_url: str | None = None, api_key: str | None = None) -> LLMClient:
    """Build the configured client. Arguments override the environment / .env settings."""
    provider = (provider or config.LLM_PROVIDER).lower()
    if provider not in _PROVIDERS:
        raise ValueError(f"Unknown LLM provider {provider!r}; choose from {sorted(_PROVIDERS)}")
    if provider != config.LLM_PROVIDER:
        # Switching provider at call time: use that provider's defaults, not the env ones.
        default_model = config._DEFAULT_MODELS[provider]
        default_url = config._DEFAULT_BASE_URLS[provider]
        default_key = ""
    else:
        default_model, default_url, default_key = config.LLM_MODEL, config.LLM_BASE_URL, config.LLM_API_KEY
    return _PROVIDERS[provider](
        model=model or default_model,
        base_url=base_url or default_url,
        api_key=api_key or default_key,
        timeout=config.LLM_TIMEOUT_S,
    )


def run_with_tools(llm: LLMClient, messages: list[dict], tools: list[dict],
                   handlers: dict[str, Callable[..., Any]], system: str | None = None,
                   max_rounds: int = 6) -> LLMReply:
    """Simple agent loop: let the model call tools until it answers in text."""
    messages = list(messages)
    for _ in range(max_rounds):
        reply = llm.chat(messages, system=system, tools=tools)
        if not reply.tool_calls:
            return reply
        messages.append({"role": "assistant", "content": reply.text, "tool_calls": reply.tool_calls})
        for call in reply.tool_calls:
            fn = handlers.get(call.name)
            try:
                result = fn(**call.arguments) if fn else {"error": f"unknown tool {call.name}"}
            except Exception as e:  # tool errors go back to the model, not up the stack
                result = {"error": str(e)}
            messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name,
                             "content": json.dumps(result, default=str)})
    return llm.chat(messages, system=system)


def parse_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply (tolerates code fences and chatter)."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise LLMError(f"No JSON object in reply: {text[:200]!r}")
    return json.loads(text[start:end + 1])


# --- message format conversion ------------------------------------------------------------
# Internal messages are OpenAI-shaped: {"role": "user"|"assistant"|"tool", "content": str},
# assistant messages may carry "tool_calls": [ToolCall], tool messages carry "tool_call_id".

def _openai_tool(t: dict) -> dict:
    return {"type": "function", "function": {
        "name": t["name"], "description": t.get("description", ""), "parameters": t["parameters"]}}


def _to_openai_messages(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_calls"):
            out.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": [
                {"id": c.id, "type": "function",
                 "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                for c in m["tool_calls"]]})
        elif m["role"] == "tool":
            out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


def _to_anthropic_messages(messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_calls"):
            blocks = [{"type": "text", "text": m["content"]}] if m.get("content") else []
            blocks += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
                       for c in m["tool_calls"]]
            out.append({"role": "assistant", "content": blocks})
        elif m["role"] == "tool":
            block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
            # Consecutive tool results must share one user turn.
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


def _as_dict(args: Any) -> dict:
    if isinstance(args, dict):
        return args
    if isinstance(args, str) and args.strip():
        return json.loads(args)
    return {}
