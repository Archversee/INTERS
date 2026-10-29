"""FastAPI backend: upload runs and reference SOPs, create / update guidelines, review
changes, serve versions.

    uvicorn sim2sop.api:app --reload
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from sim2sop import guidelines_store as store
from sim2sop import references, render, storage
from sim2sop.agent.llm import get_llm
from sim2sop.ingest import ingest_run
from sim2sop.metrics import edits_per_version
from sim2sop.pipeline import apply_edit, approve, create_guidelines, update_guidelines
from sim2sop.profiles import list_profiles, load_profile

app = FastAPI(title="Sim-to-SOP", version="0.2.0",
              description="Advisory guidelines from simulator runs, checked against reference SOPs.")


def _profile(name: str):
    try:
        return load_profile(name)
    except FileNotFoundError:
        raise HTTPException(404, f"Unknown profile {name!r}; available: {list_profiles()}")


def _save_upload(upload: UploadFile, folder: Path, name: str | None = None) -> Path:
    path = folder / Path(name or upload.filename or "upload").name
    with open(path, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return path


@app.get("/profiles")
def profiles():
    return list_profiles()


# --- runs ---------------------------------------------------------------------------------

@app.get("/profiles/{profile}/runs")
def runs(profile: str):
    p = _profile(profile)
    return [{"run_id": r.run_id, "environment": r.meta.environment, "events": len(r.events),
             "duration_s": r.meta.duration_s, "flags": [f.message for f in r.flags]}
            for r in storage.list_runs(p.procedure)]


@app.post("/profiles/{profile}/runs")
async def upload_run(profile: str, telemetry: UploadFile = File(...),
                     annotations: UploadFile | None = File(None),
                     run_id: str | None = Form(None), environment: str | None = Form(None)):
    """Upload one run: the sim's telemetry file (+ optional annotations.json and environment tag)."""
    p = _profile(profile)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _save_upload(telemetry, d)
        if annotations is not None:
            _save_upload(annotations, d, "annotations.json")
        record = ingest_run(d, p, run_id=run_id or Path(telemetry.filename or "run").stem,
                            environment=environment)
    return {"run_id": record.run_id, "environment": record.meta.environment, "events": len(record.events),
            "phases": [ph.name for ph in record.phases], "flags": [f.message for f in record.flags]}


# --- reference SOPs (read-only once added) -------------------------------------------------

@app.get("/profiles/{profile}/references")
def list_refs(profile: str):
    p = _profile(profile)
    return references.list_references(p.procedure, active_only=False)


@app.post("/profiles/{profile}/references")
async def add_ref(profile: str, file: UploadFile = File(...), limits: UploadFile | None = File(None)):
    """Add a reference document (.md with front matter, or .pdf/.txt + optional limits YAML)."""
    p = _profile(profile)
    with tempfile.TemporaryDirectory() as tmp:
        path = _save_upload(file, Path(tmp))
        if limits is not None:
            _save_upload(limits, Path(tmp), path.with_suffix(".limits.yaml").name)
        try:
            doc = references.add_reference(p.procedure, path)
        except references.BuiltinDocument as e:
            raise HTTPException(409, str(e)) from None
    return doc


# --- guidelines -----------------------------------------------------------------------------

class BuildRequest(BaseModel):
    provider: str | None = None
    model: str | None = None


def _llm(req: BuildRequest | None):
    return get_llm(req.provider, req.model) if req and (req.provider or req.model) else None


def _built(doc):
    return {"version": doc.version, "status": doc.status, "checked": doc.checked,
            "pending": [s.id for s in doc.pending], "metrics": doc.metrics, "writer": doc.writer}


@app.post("/profiles/{profile}/guidelines/create")
def create(profile: str, req: BuildRequest | None = None):
    try:
        return _built(create_guidelines(_profile(profile), llm=_llm(req)))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/profiles/{profile}/guidelines/update")
def update(profile: str, req: BuildRequest | None = None):
    try:
        return _built(update_guidelines(_profile(profile), llm=_llm(req)))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/profiles/{profile}/guidelines")
def versions(profile: str):
    p = _profile(profile)
    return {"versions": store.versions(p.procedure), "edits_per_version": edits_per_version(p.procedure)}


@app.get("/profiles/{profile}/guidelines/{version}")
def get_version(profile: str, version: int):
    p = _profile(profile)
    try:
        return store.load(p.procedure, version)
    except FileNotFoundError:
        raise HTTPException(404, f"No v{version}")


@app.get("/profiles/{profile}/guidelines/{version}/html", response_class=HTMLResponse)
def get_html(profile: str, version: int):
    return render.to_html(get_version(profile, version))


@app.get("/profiles/{profile}/guidelines/{a}/diff/{b}")
def diff(profile: str, a: int, b: int):
    return store.diff(get_version(profile, a), get_version(profile, b))


class ReviewRequest(BaseModel):
    version: int | None = None
    approve: list[str] | None = None    # None = approve every pending step not rejected
    reject: list[str] = []


@app.post("/profiles/{profile}/guidelines/review")
def review(profile: str, req: ReviewRequest):
    try:
        doc = approve(_profile(profile), req.version, req.approve, req.reject)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"version": doc.version, "status": doc.status, "pending": [s.id for s in doc.pending]}


class EditRequest(BaseModel):
    step_id: str
    field: str
    text: str
    version: int | None = None


@app.post("/profiles/{profile}/edits")
def edit(profile: str, req: EditRequest):
    try:
        doc = apply_edit(_profile(profile), req.step_id, req.field, req.text, version=req.version)
    except ValueError as e:
        raise HTTPException(400, str(e))
    step = next(s for s in doc.steps if s.id == req.step_id)
    return {"version": doc.version, "verified": step.verified, "flags": step.flags}
