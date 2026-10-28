"""Reference RAG: chunk the domain's reference documents and retrieve the best section
for each step, so every step can cite the POH / handbook / regulation it follows.

Backends:
  tfidf   pure NumPy TF-IDF cosine, no model needed (default)
  ollama  nomic-embed-text embeddings via Ollama, cached next to the documents

Documents: references/<domain>/*.md|*.txt (Markdown headings become sections) and *.pdf
(one section per page; needs `pypdf`).
TODO(ai): swap the in-memory index for LanceDB once the reference set grows.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np

from sim2sop import config
from sim2sop.schema import Citation

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOP = set("the a an and or of to in on at for with is are be by as it this that from into "
           "when then than your you its not no if".split())


@dataclass
class Chunk:
    doc: str
    section: str
    text: str


def _stem(t: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if len(t) > len(suffix) + 2 and t.endswith(suffix):
            return t[: -len(suffix)]
    return t


def _tokens(text: str) -> list[str]:
    return [_stem(t) for t in TOKEN_RE.findall(text.lower()) if t not in STOP]


def load_chunks(domain_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(domain_dir.glob("*")):
        if path.suffix.lower() in (".md", ".txt"):
            chunks += _markdown_chunks(path)
        elif path.suffix.lower() == ".pdf":
            chunks += _pdf_chunks(path)
    return chunks


def _markdown_chunks(path: Path) -> list[Chunk]:
    doc_title, section, buf, out = path.stem, "", [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            doc_title = line[2:].strip()
        elif line.startswith("#"):
            if buf and section:
                out.append(Chunk(doc_title, section, " ".join(buf).strip()))
            section, buf = line.lstrip("#").strip(), []
        elif line.strip() and not line.startswith(">"):
            buf.append(line.strip())
    if buf and section:
        out.append(Chunk(doc_title, section, " ".join(buf).strip()))
    return out


def _pdf_chunks(path: Path) -> list[Chunk]:
    try:
        from pypdf import PdfReader
    except ImportError:
        return []
    return [Chunk(path.stem, f"p.{i + 1}", (page.extract_text() or "").strip())
            for i, page in enumerate(PdfReader(str(path)).pages) if (page.extract_text() or "").strip()]


class ReferenceIndex:
    def __init__(self, domain_dir: Path, backend: str | None = None):
        self.chunks = load_chunks(domain_dir)
        self.backend = backend or config.EMBED_PROVIDER
        self.domain_dir = domain_dir
        if self.backend == "ollama":
            try:
                self._vecs = self._ollama_matrix()
            except httpx.HTTPError:
                self.backend = "tfidf"  # stay offline-safe
        if self.backend == "tfidf":
            self._build_tfidf()

    @classmethod
    def for_domain(cls, references: str) -> "ReferenceIndex":
        return cls(config.REFERENCES_DIR / references)

    # --- tf-idf ---
    def _build_tfidf(self) -> None:
        docs = [_tokens(f"{c.section} {c.section} {c.text}") for c in self.chunks]
        df = Counter(t for d in docs for t in set(d))
        self._vocab = {t: i for i, t in enumerate(sorted(df))}
        n = max(len(docs), 1)
        self._idf = np.array([math.log((1 + n) / (1 + df[t])) + 1 for t in sorted(df)])
        self._vecs = np.vstack([self._tfidf(d) for d in docs]) if docs else np.zeros((0, 0))

    def _tfidf(self, tokens: list[str]) -> np.ndarray:
        v = np.zeros(len(self._vocab))
        for t, c in Counter(tokens).items():
            if t in self._vocab:
                v[self._vocab[t]] = c
        v *= self._idf
        return v / (np.linalg.norm(v) or 1.0)

    # --- ollama embeddings ---
    def _embed(self, texts: list[str]) -> np.ndarray:
        r = httpx.post(f"{config.OLLAMA_URL}/api/embed",
                       json={"model": config.EMBED_MODEL, "input": texts}, timeout=120)
        r.raise_for_status()
        m = np.array(r.json()["embeddings"], dtype=float)
        return m / np.linalg.norm(m, axis=1, keepdims=True)

    def _ollama_matrix(self) -> np.ndarray:
        texts = [f"{c.section}: {c.text}" for c in self.chunks]
        digest = hashlib.sha1(json.dumps([config.EMBED_MODEL, texts]).encode()).hexdigest()[:12]
        cache = self.domain_dir / f".embeddings-{digest}.npy"
        if cache.exists():
            return np.load(cache)
        m = self._embed(texts)
        np.save(cache, m)
        return m

    def search(self, query: str, k: int = 3) -> list[Citation]:
        if not self.chunks:
            return []
        q = self._embed([query])[0] if self.backend == "ollama" else self._tfidf(_tokens(query))
        scores = self._vecs @ q
        order = np.argsort(-scores)[:k]
        return [Citation(doc=self.chunks[i].doc, section=self.chunks[i].section,
                         text=self.chunks[i].text, score=round(float(scores[i]), 3))
                for i in order if scores[i] > 0]

    def best(self, query: str, min_score: float = 0.12) -> Citation | None:
        hits = self.search(query, k=1)
        return hits[0] if hits and hits[0].score >= min_score else None
