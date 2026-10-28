"""Project paths and runtime settings, overridable with environment variables or a .env file."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("SIM2SOP_ROOT", Path(__file__).resolve().parents[2]))


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (KEY=VALUE lines); real environment variables win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")

DATA_DIR = Path(os.environ.get("SIM2SOP_DATA", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"          # generated / recorded source files, as the sim writes them
RUNS_DIR = DATA_DIR / "runs"        # ingested runs in the common schema
GUIDELINES_DIR = DATA_DIR / "guidelines"   # versioned guideline outputs (see guidelines_store)
LIBRARY_DIR = DATA_DIR / "references"     # registered reference SOPs (see references.py)
MEMORY_DIR = DATA_DIR / "memory"    # instructor corrections
PROFILES_DIR = Path(os.environ.get("SIM2SOP_PROFILES", ROOT / "profiles"))
REFERENCES_DIR = Path(os.environ.get("SIM2SOP_REFERENCES", ROOT / "references"))

# --- LLM -------------------------------------------------------------------------------
# Provider: "ollama" (local, offline), "openai" (any OpenAI-compatible endpoint: OpenAI,
# OpenRouter, Groq, vLLM, LM Studio...), "anthropic", or "none" (deterministic template
# writer, no model needed).
LLM_PROVIDER = os.environ.get("SIM2SOP_LLM_PROVIDER", "ollama").lower()

_DEFAULT_MODELS = {
    "ollama": "qwen2.5:7b-instruct",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-5-5",
    "none": "template",
}
_DEFAULT_BASE_URLS = {
    "ollama": "http://localhost:11434",
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
    "none": "",
}
_KEY_ENV = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}

LLM_MODEL = os.environ.get("SIM2SOP_MODEL", _DEFAULT_MODELS.get(LLM_PROVIDER, ""))
LLM_BASE_URL = os.environ.get("SIM2SOP_LLM_BASE_URL", _DEFAULT_BASE_URLS.get(LLM_PROVIDER, ""))
LLM_API_KEY = os.environ.get("SIM2SOP_API_KEY") or os.environ.get(_KEY_ENV.get(LLM_PROVIDER, ""), "")
LLM_TIMEOUT_S = float(os.environ.get("SIM2SOP_LLM_TIMEOUT", "120"))

# Embeddings for reference RAG. "ollama" uses nomic-embed-text locally; "tfidf" needs nothing.
EMBED_PROVIDER = os.environ.get("SIM2SOP_EMBED_PROVIDER", "tfidf").lower()
EMBED_MODEL = os.environ.get("SIM2SOP_EMBED_MODEL", "nomic-embed-text")
OLLAMA_URL = os.environ.get("OLLAMA_URL", _DEFAULT_BASE_URLS["ollama"])

# Verifier loop: how many rewrite rounds before a mismatch is flagged instead of fixed.
MAX_VERIFY_ROUNDS = int(os.environ.get("SIM2SOP_MAX_VERIFY_ROUNDS", "2"))
