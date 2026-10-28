"""FastAPI backend: upload runs, trigger generation, serve SOPs.

    uvicorn sim2sop.api:app --reload
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from sim2sop import render, sop_store, storage
from sim2sop.agent.llm import get_llm
from sim2sop.ingest import ingest_run
from sim2sop.metrics import edits_per_version
from sim2sop.pipeline import apply_edit, generate_sop
from sim2sop.profiles import list_profiles, load_profile

app = FastAPI(title="Sim-to-SOP", version="0.1.0")


def _profile(name: str):
    try:
        return load_profile(name)
    except FileNotFoundError:
        raise HTTPException(404, f"Unknown profile {name!r}; available: {list_profiles()}")


@app.get("/profiles")
def profiles():
    return list_profiles()


@app.get("/profiles/{profile}/runs")
def runs(profile: str):
    p = _profile(profile)
    return [{"run_id": r.run_id, "events": len(r.events), "duration_s": r.meta.duration_s,
             "flags": [f.message for f in r.flags]} for r in storage.list_runs(p.procedure)]


@app.post("/profiles/{profile}/runs")
async def upload_run(profile: str, telemetry: UploadFile = File(...),
                     annotations: UploadFile | None = File(None),
                     run_id: str | None = Form(None)):
    """Upload one run: the sim's telemetry file (+ optional annotations.json)."""
    p = _profile(profile)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        with open(d / Path(telemetry.filename or "telemetry.csv").name, "wb") as f:
            shutil.copyfileobj(telemetry.file, f)
        if annotations is not None:
            with open(d / "annotations.json", "wb") as f:
                shutil.copyfileobj(annotations.file, f)
        record = ingest_run(d, p, run_id=run_id or Path(telemetry.filename or "run").stem)
    return {"run_id": record.run_id, "events": len(record.events),
            "phases": [ph.name for ph in record.phases], "flags": [f.message for f in record.flags]}


class BuildRequest(BaseModel):
    provider: str | None = None
    model: str | None = None
    full: bool = False


@app.post("/profiles/{profile}/sops")
def build(profile: str, req: BuildRequest | None = None):
    p = _profile(profile)
    req = req or BuildRequest()
    llm = get_llm(req.provider, req.model) if (req.provider or req.model) else None
    try:
        sop = generate_sop(p, llm=llm, incremental=not req.full)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"version": sop.version, "metrics": sop.metrics, "writer": sop.writer}


@app.get("/profiles/{profile}/sops")
def versions(profile: str):
    p = _profile(profile)
    return {"versions": sop_store.versions(p.procedure), "edits_per_version": edits_per_version(p.procedure)}


@app.get("/profiles/{profile}/sops/{version}")
def get_sop(profile: str, version: int):
    p = _profile(profile)
    try:
        return sop_store.load(p.procedure, version)
    except FileNotFoundError:
        raise HTTPException(404, f"No v{version}")


@app.get("/profiles/{profile}/sops/{version}/html", response_class=HTMLResponse)
def get_sop_html(profile: str, version: int):
    return render.to_html(get_sop(profile, version))


@app.get("/profiles/{profile}/sops/{a}/diff/{b}")
def diff(profile: str, a: int, b: int):
    return sop_store.diff(get_sop(profile, a), get_sop(profile, b))


class EditRequest(BaseModel):
    step_id: str
    field: str
    text: str
    version: int | None = None


@app.post("/profiles/{profile}/edits")
def edit(profile: str, req: EditRequest):
    p = _profile(profile)
    try:
        sop = apply_edit(p, req.step_id, req.field, req.text, version=req.version)
    except ValueError as e:
        raise HTTPException(400, str(e))
    step = next(s for s in sop.steps if s.id == req.step_id)
    return {"version": sop.version, "verified": step.verified, "flags": step.flags}
