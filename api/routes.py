"""FastAPI route definitions for the epistemic verification pipeline."""
from __future__ import annotations

import json
import os
from collections import defaultdict

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse

import config
from pipeline.models import OpenAIConfig, PipelineRequest, PipelineResponse, StressRequest
from pipeline.helpers import PipelineError
from pipeline.orchestrator import run_pipeline
from pipeline.stress import generate_stress_results
from api.ui import UI_HTML

router = APIRouter()


# ---- UI ----

@router.get("/", response_class=HTMLResponse)
def ui():
    return UI_HTML


# ---- Health check (for Railway / load balancers) ----

@router.get("/health")
def health():
    return {"status": "ok", "key_set": config.has_api_key()}


# ---- OpenAI config ----

@router.post("/api/openai/config")
def set_openai_config(cfg: OpenAIConfig):
    clean_key = cfg.api_key.strip()
    clean_key = clean_key.encode("ascii", errors="ignore").decode("ascii")
    clean_key = clean_key.replace(" ", "")
    if not clean_key:
        raise HTTPException(status_code=400, detail="Invalid API key.")
    config.set_runtime_config(clean_key, cfg.model)
    return {"status": "ok", "model": cfg.model, "key_set": True}


@router.get("/api/openai/config")
def get_openai_config():
    if not config.has_api_key():
        return {"key_set": False}
    return {
        "key_set": True,
        "model": config.get_model(),
        "key_preview": config.get_key_preview(),
    }


# ---- Pipeline ----

@router.post("/api/pipeline", response_model=PipelineResponse)
def pipeline_endpoint(req: PipelineRequest):
    try:
        return run_pipeline(req)
    except PipelineError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)


# ---- Stress test ----

@router.post("/api/stress")
def stress_endpoint(req: StressRequest):
    """Run stress harness inline -- returns streaming NDJSON progress + final score."""
    if not config.has_api_key():
        raise HTTPException(status_code=400, detail="Set your OpenAI API key first.")

    # Load tests
    tests_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests.json")
    if not os.path.exists(tests_path):
        raise HTTPException(status_code=404, detail="tests.json not found")

    with open(tests_path, "r", encoding="utf-8") as f:
        tests = json.load(f)

    if req.category:
        tests = [t for t in tests if t["category"] == req.category]

    if req.count:
        by_cat = defaultdict(list)
        for t in tests:
            by_cat[t["category"]].append(t)
        tests = []
        for cat in sorted(by_cat):
            tests.extend(by_cat[cat][:req.count])

    if not tests:
        raise HTTPException(status_code=400, detail="No matching test cases.")

    return StreamingResponse(
        generate_stress_results(tests),
        media_type="application/x-ndjson",
    )
