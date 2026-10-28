"""Reference RAG: chunk the procedure's registered reference documents (see
sim2sop.references) and retrieve the best section for each guideline, so it can cite
the SOP / POH / regulation behind it.

Backends:
  tfidf   pure NumPy TF-IDF cosine, no model needed (default)
  ollama  nomic-embed-text embeddings via Ollama, cached next to the documents

Documents: .md/.txt (front matter skipped, `#` headings become sections) and .pdf (one
section per page; needs `pypdf`).
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

from sim2sop import config, references
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


def load_chunks(files: list[tuple[str, Path]]) -> list[Chunk]:
    """files: (document label used in citations, path)."""
    chunks: list[Chunk] = []
    for label, path in files:
        if path.suffix.lower() in (".md", ".txt"):
            chunks += _markdown_chunks(path, label)
        elif path.suffix.lower() == ".pdf":
            chunks += _pdf_chunks(path, label)
    return chunks


def _markdown_chunks(path: Path, doc_title: str) -> list[Chunk]:
    section, buf, out = "", [], []
    _, body = references.split_front_matter(path.read_text(encoding="utf-8"))
    for line in body.splitlines():
        if line.startswith("# "):
            continue
        elif line.startswith("#"):
            if buf and section:
                out.append(Chunk(doc_title, section, " ".join(buf).strip()))
            section, buf = line.lstrip("#").strip(), []
        elif line.strip() and not line.startswith(">"):
            buf.append(line.strip())
    if buf and section:
        out.append(Chunk(doc_title, section, " ".join(buf).strip()))
    return out


def _pdf_chunks(path: Path, doc_title: str) -> list[Chunk]:
    try:
        from pypdf import PdfReader
    except ImportError:
        return []
    return [Chunk(doc_title, f"p.{i + 1}", (page.extract_text() or "").strip())
            for i, page in enumerate(PdfReader(str(path)).pages) if (page.extract_text() or "").strip()]


class ReferenceIndex:
    def __init__(self, files: list[tuple[str, Path]], cache_dir: Path | None = None,
                 backend: str | None = None):
        self.chunks = load_chunks(files)
        self.backend = backend or config.EMBED_PROVIDER
        self.cache_dir = cache_dir or config.DATA_DIR
        if self.backend == "ollama" and self.chunks:
            try:
                self._vecs = self._ollama_matrix()
            except httpx.HTTPError:
                self.backend = "tfidf"  # stay offline-safe
        if self.backend != "ollama" or not self.chunks:
            self.backend = "tfidf"
            self._build_tfidf()

    @classmethod
    def for_procedure(cls, procedure: str) -> "ReferenceIndex":
        """Index the procedure's active reference documents (empty if none are added)."""
        files = [(doc.label, path) for doc, path in references.active_files(procedure)]
        return cls(files, cache_dir=references.library_dir(procedure))

    @property
    def empty(self) -> bool:
        return not self.chunks

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
        cache = self.cache_dir / f".embeddings-{digest}.npy"
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
