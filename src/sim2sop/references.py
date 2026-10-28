"""Reference library: the read-only reference SOPs, POH, regulations and NOTAMs that set
hard limits for a procedure's guidelines.

    data/references/<procedure>/
        manifest.json     list of RefDoc (limits and requirements parsed from each doc)
        <files>           copies of the added documents; never modified

Limits and required actions are declared in YAML front matter of a Markdown document:

    ---
    id: c172_poh
    title: C172 POH
    kind: poh                      # sop | poh | notam | regulation | handbook | other
    revision: "A"
    limits:
      - {id: vfe_full, signal: ias_kt, max: 85, when: "flaps_idx >= 2",
         section: "Airspeed limitations", text: "Max 85 KIAS with flaps beyond 10° (Vfe)"}
    requirements:
      - {id: landing_light, key: "toggle:light_landing:1", text: "Landing light on for take-off"}
    ---

For a PDF (or any file without front matter) put the same YAML in a sidecar file next
to it: `<name>.limits.yaml`. The PDF text is still indexed for citations.
TODO(ai): extract limits from unstructured PDFs with the LLM, then have an instructor confirm.

Adding a document with an `id` that is already in the library supersedes the older
revision, which stays in the manifest as inactive (history).
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import yaml

from sim2sop import config
from sim2sop.schema import Limit, RefDoc, Requirement

FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)


def library_dir(procedure: str) -> Path:
    return config.LIBRARY_DIR / procedure


def split_front_matter(text: str) -> tuple[dict, str]:
    m = FRONT_MATTER_RE.match(text)
    if not m:
        return {}, text
    return (yaml.safe_load(m.group(1)) or {}), text[m.end():]


def read_doc_meta(path: Path) -> dict:
    path = Path(path)
    meta: dict = {}
    if path.suffix.lower() in (".md", ".txt"):
        meta, _ = split_front_matter(path.read_text(encoding="utf-8"))
    sidecar = path.with_suffix(".limits.yaml")
    if sidecar.exists():
        meta = {**meta, **(yaml.safe_load(sidecar.read_text(encoding="utf-8")) or {})}
    return meta


def _to_refdoc(meta: dict, stored_name: str, fallback_id: str) -> RefDoc:
    doc_id = str(meta.get("id") or fallback_id)
    title = str(meta.get("title") or fallback_id)
    revision = str(meta["revision"]) if meta.get("revision") is not None else None
    common = {"doc_id": doc_id, "doc_title": title, "revision": revision}
    limits = []
    for raw in meta.get("limits") or []:
        if raw.get("environment") is None and meta.get("environment"):
            raw = {**raw, "environment": meta["environment"]}
        limits.append(Limit(**{**raw, **common}))
    reqs = [Requirement(**{**r, **common}) for r in meta.get("requirements") or []]
    return RefDoc(id=doc_id, title=title, kind=meta.get("kind", "other"), revision=revision,
                  file=stored_name, limits=limits, requirements=reqs)


def _manifest_path(procedure: str) -> Path:
    return library_dir(procedure) / "manifest.json"


def list_references(procedure: str, active_only: bool = True) -> list[RefDoc]:
    path = _manifest_path(procedure)
    if not path.exists():
        return []
    docs = [RefDoc.model_validate(d) for d in json.loads(path.read_text(encoding="utf-8"))]
    return [d for d in docs if d.active] if active_only else docs


def _save(procedure: str, docs: list[RefDoc]) -> None:
    library_dir(procedure).mkdir(parents=True, exist_ok=True)
    _manifest_path(procedure).write_text(
        json.dumps([d.model_dump(mode="json") for d in docs], indent=2), encoding="utf-8")


def add_reference(procedure: str, path: Path) -> RefDoc:
    """Copy a reference document into the library (read-only from here on)."""
    path = Path(path)
    meta = read_doc_meta(path)
    doc_id = str(meta.get("id") or path.stem)
    rev = str(meta.get("revision") or "")
    stored = f"{doc_id}{'__' + re.sub(r'[^A-Za-z0-9.-]', '_', rev) if rev else ''}{path.suffix}"
    d = library_dir(procedure)
    d.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, d / stored)
    sidecar = path.with_suffix(".limits.yaml")
    if sidecar.exists():
        shutil.copyfile(sidecar, (d / stored).with_suffix(".limits.yaml"))
    new = _to_refdoc(meta, stored, path.stem)

    docs = list_references(procedure, active_only=False)
    for old in docs:
        if old.id == new.id and old.active:
            old.active = False  # superseded by the new revision
    docs = [x for x in docs if not (x.id == new.id and x.revision == new.revision)] + [new]
    _save(procedure, docs)
    return new


def remove_reference(procedure: str, doc_id: str) -> None:
    docs = list_references(procedure, active_only=False)
    _save(procedure, [d for d in docs if d.id != doc_id])


def clear_references(procedure: str) -> None:
    shutil.rmtree(library_dir(procedure), ignore_errors=True)


def active_files(procedure: str) -> list[tuple[RefDoc, Path]]:
    return [(d, library_dir(procedure) / d.file) for d in list_references(procedure)]


def active_limits(procedure: str) -> list[Limit]:
    return [lim for d in list_references(procedure) for lim in d.limits]


def active_requirements(procedure: str) -> list[Requirement]:
    return [r for d in list_references(procedure) for r in d.requirements]
