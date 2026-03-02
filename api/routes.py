"""FastAPI route definitions for the epistemic verification pipeline."""
from __future__ import annotations

import json
import os
from collections import defaultdict

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse

from api.rate_limit import rate_limit_dependency

import config
from pipeline.models import OpenAIConfig, TavilyConfig, StageConfig, PipelineRequest, PipelineResponse, StressRequest, FeedbackRequest
from pipeline.helpers import PipelineError
from pipeline.orchestrator import run_pipeline
from pipeline.stress import generate_stress_results
from pipeline.metrics import get_aggregate
from pipeline.feedback import FeedbackEntry, get_feedback_store
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


# ---- Tavily (web search) config ----

@router.post("/api/tavily/config")
def set_tavily_config(cfg: TavilyConfig):
    clean_key = cfg.api_key.strip()
    clean_key = clean_key.encode("ascii", errors="ignore").decode("ascii")
    clean_key = clean_key.replace(" ", "")
    if not clean_key:
        raise HTTPException(status_code=400, detail="Invalid Tavily API key.")
    config.set_tavily_config(clean_key, cfg.enabled)
    return {"status": "ok", "enabled": cfg.enabled, "key_set": True}


@router.get("/api/tavily/config")
def get_tavily_config():
    if not config.has_tavily_key():
        return {"key_set": False, "enabled": False}
    return {
        "key_set": True,
        "enabled": config.is_tavily_enabled(),
        "key_preview": config.get_tavily_key_preview(),
    }


@router.post("/api/tavily/toggle")
def toggle_tavily(enabled: bool = True):
    if not config.has_tavily_key():
        raise HTTPException(status_code=400, detail="Set Tavily API key first.")
    config.set_tavily_enabled(enabled)
    return {"status": "ok", "enabled": enabled}


# ---- Per-stage model config ----

@router.post("/api/stage/config")
def set_stage_config_endpoint(cfg: StageConfig):
    clean_key = cfg.api_key.strip().encode("ascii", errors="ignore").decode("ascii").replace(" ", "")
    config.set_stage_config(cfg.stage, cfg.provider, clean_key, cfg.model, cfg.base_url)
    return {"status": "ok", "stage": cfg.stage, "provider": cfg.provider}


@router.get("/api/stage/config/{stage}")
def get_stage_config_endpoint(stage: str):
    cfg = config.get_stage_config(stage)
    return {
        "stage": stage,
        "provider": cfg["provider"],
        "model": cfg["model"],
        "key_set": bool(cfg["api_key"]),
    }


# ---- Metrics ----

@router.get("/api/metrics")
def metrics_endpoint():
    """Return aggregate pipeline metrics since startup."""
    return get_aggregate().to_dict()


# ---- Feedback ----

@router.post("/api/feedback")
def submit_feedback(req: FeedbackRequest):
    """Submit user feedback on a pipeline result."""
    if req.rating not in ("accurate", "inaccurate", "partially_accurate"):
        raise HTTPException(status_code=400, detail="Rating must be: accurate, inaccurate, or partially_accurate")
    import uuid
    store = get_feedback_store()
    entry = FeedbackEntry(
        feedback_id=uuid.uuid4().hex[:12],
        request_id=req.request_id,
        prompt=req.prompt,
        rating=req.rating,
        verdict_correct=req.verdict_correct,
        confidence_correct=req.confidence_correct,
        comment=req.comment,
    )
    store.add(entry)
    return {"status": "ok", "feedback_id": entry.feedback_id}


@router.get("/api/feedback/summary")
def feedback_summary():
    """Return aggregate feedback statistics."""
    return get_feedback_store().get_summary()


# ---- Pipeline ----

@router.post("/api/pipeline", response_model=PipelineResponse, dependencies=[Depends(rate_limit_dependency)])
def pipeline_endpoint(req: PipelineRequest):
    if len(req.prompt) > config.MAX_PROMPT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Prompt exceeds maximum length of {config.MAX_PROMPT_LENGTH} characters.",
        )
    try:
        return run_pipeline(req)
    except PipelineError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)


# ---- Stress test ----

@router.post("/api/stress", dependencies=[Depends(rate_limit_dependency)])
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
