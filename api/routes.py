"""FastAPI route definitions for the epistemic verification pipeline."""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
from collections import defaultdict

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from api.rate_limit import rate_limit_dependency, get_rate_limit_info

import config
from pipeline.models import OpenAIConfig, TavilyConfig, StageConfig, PipelineRequest, PipelineResponse, StressRequest, FeedbackRequest
from pipeline.helpers import PipelineError
from pipeline.orchestrator import run_pipeline
from pipeline.stress import generate_stress_results
from pipeline.metrics import get_aggregate
from pipeline.feedback import FeedbackEntry, get_feedback_store
from api.ui import UI_HTML

router = APIRouter()


# Allowed base URL patterns for custom provider endpoints.
# Prevents SSRF by restricting stage base_url to known-safe hosts.
_ALLOWED_BASE_URL_PREFIXES = [
    "https://openrouter.ai/",
    "https://api.openai.com/",
    "https://api.anthropic.com/",
    "http://localhost:",
    "http://127.0.0.1:",
    "http://host.docker.internal:",
]

# Allow extending via env var (comma-separated prefixes)
_extra = os.getenv("ALLOWED_BASE_URLS", "")
if _extra:
    _ALLOWED_BASE_URL_PREFIXES.extend(p.strip() for p in _extra.split(",") if p.strip())


def _validate_base_url(url: str):
    """Reject base_url values that don't match the allowlist."""
    url_lower = url.strip().lower()
    if any(url_lower.startswith(prefix.lower()) for prefix in _ALLOWED_BASE_URL_PREFIXES):
        return
    raise HTTPException(
        status_code=400,
        detail=(
            f"base_url not in allowlist. Allowed prefixes: "
            f"{', '.join(_ALLOWED_BASE_URL_PREFIXES[:4])}... "
            f"Set ALLOWED_BASE_URLS env var to add custom prefixes."
        ),
    )


def _require_admin(request: Request):
    """Verify admin token on config-mutation endpoints.

    When ADMIN_TOKEN is set, requests must include Authorization: Bearer <token>.
    When ADMIN_TOKEN is empty (local dev), all requests are allowed.
    """
    token = config.ADMIN_TOKEN
    if not token:
        return  # No token configured — open access (local dev)
    auth = request.headers.get("authorization", "")
    if auth != f"Bearer {token}":
        raise HTTPException(status_code=401, detail="Unauthorized — invalid or missing admin token.")


# ---- UI ----

@router.get("/", response_class=HTMLResponse)
def ui():
    return UI_HTML


# ---- Health check (for Railway / load balancers) ----

@router.get("/health")
def health():
    from pipeline.nli import is_nli_available, NLI_SERVICE_URL
    return {
        "status": "ok",
        "key_set": config.has_api_key(),
        "tavily_enabled": config.is_tavily_enabled(),
        "nli_available": is_nli_available(),
        "nli_mode": "remote" if NLI_SERVICE_URL else ("local" if is_nli_available() else "disabled"),
    }


# ---- NLI status ----

@router.get("/api/nli/status")
def nli_status():
    """Return NLI verification layer availability and configuration."""
    from pipeline.nli import is_nli_available, NLI_SERVICE_URL, NLI_MODEL
    from pipeline.nli import ENTAILMENT_THRESHOLD, CONTRADICTION_THRESHOLD
    available = is_nli_available()
    return {
        "available": available,
        "mode": "remote" if NLI_SERVICE_URL else ("local" if available else "disabled"),
        "model": NLI_MODEL if available and not NLI_SERVICE_URL else None,
        "remote_url_set": bool(NLI_SERVICE_URL),
        "thresholds": {
            "entailment": ENTAILMENT_THRESHOLD,
            "contradiction": CONTRADICTION_THRESHOLD,
        },
    }


# ---- OpenAI config ----

@router.post("/api/openai/config", dependencies=[Depends(_require_admin)])
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

@router.post("/api/tavily/config", dependencies=[Depends(_require_admin)])
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


@router.post("/api/tavily/toggle", dependencies=[Depends(_require_admin)])
def toggle_tavily(enabled: bool = True):
    if not config.has_tavily_key():
        raise HTTPException(status_code=400, detail="Set Tavily API key first.")
    config.set_tavily_enabled(enabled)
    return {"status": "ok", "enabled": enabled}


# ---- Per-stage model config ----

@router.post("/api/stage/config", dependencies=[Depends(_require_admin)])
def set_stage_config_endpoint(cfg: StageConfig):
    clean_key = cfg.api_key.strip().encode("ascii", errors="ignore").decode("ascii").replace(" ", "")
    # Validate base_url against allowlist to prevent SSRF
    if cfg.base_url:
        _validate_base_url(cfg.base_url)
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

# Server-side timeout (seconds) — must finish before the CDN/proxy timeout
# to return a proper JSON error instead of the platform's XML/HTML error page.
# Default 90 s accommodates arbiter + rewrite loops without false timeouts.
_PIPELINE_TIMEOUT = int(os.getenv("PIPELINE_TIMEOUT", "90"))
# 8 workers: allows concurrent pipelines without exhausting Railway CPU budget.
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)

def _stream_pipeline(req: PipelineRequest):
    """Run pipeline and yield true stage-based NDJSON events.

    Uses a thread-safe queue to receive real stage_start / stage_complete
    events emitted by the orchestrator's ``emit`` callback.
    """
    import time
    import threading
    import queue

    event_q: queue.Queue = queue.Queue()
    result_holder: list = []
    error_holder: list = []
    start = time.monotonic()

    def _emit(event: dict):
        """Callback passed to run_pipeline for real stage events."""
        event["elapsed"] = round(time.monotonic() - start, 1)
        event_q.put(event)

    def _run():
        try:
            result = run_pipeline(req, emit=_emit)
            result_holder.append(result)
        except Exception as e:
            error_holder.append(e)
        finally:
            event_q.put(None)  # sentinel: pipeline finished

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    # Drain events until pipeline completes (sentinel = None)
    timeout = _PIPELINE_TIMEOUT
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            yield json.dumps({"type": "error", "detail": f"Pipeline timed out after {timeout}s.", "status_code": 504}) + "\n"
            return
        try:
            event = event_q.get(timeout=min(remaining, 1.0))
        except queue.Empty:
            continue
        if event is None:
            break  # pipeline done
        yield json.dumps(event) + "\n"

    # Emit final result or error
    if error_holder:
        e = error_holder[0]
        if isinstance(e, PipelineError):
            yield json.dumps({"type": "error", "detail": e.detail, "status_code": e.status_code}) + "\n"
        else:
            yield json.dumps({"type": "error", "detail": str(e), "status_code": 500}) + "\n"
    elif result_holder:
        yield json.dumps({"type": "result", "data": result_holder[0].model_dump()}) + "\n"


@router.post("/api/pipeline", response_model=PipelineResponse, dependencies=[Depends(rate_limit_dependency)])
async def pipeline_endpoint(req: PipelineRequest, request: Request):
    # SSE streaming mode
    if getattr(req, "stream", False):
        return StreamingResponse(
            _stream_pipeline(req),
            media_type="application/x-ndjson",
        )

    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(_executor, run_pipeline, req),
            timeout=_PIPELINE_TIMEOUT,
        )
        return result
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"Pipeline timed out after {_PIPELINE_TIMEOUT}s. Try a simpler query or disable web search.",
        )
    except PipelineError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)


@router.post("/v2/pipeline", response_model=PipelineResponse, dependencies=[Depends(rate_limit_dependency)])
async def v2_pipeline_endpoint(req: PipelineRequest, request: Request):
    """V2 pipeline endpoint with true stage streaming and verdict labels.

    Returns:
    - JSON (non-stream): full PipelineResponse with ``verdict_label`` field.
    - NDJSON (stream=true): real stage events emitted from the orchestrator.
    """
    if getattr(req, "stream", False):
        return StreamingResponse(
            _stream_pipeline(req),
            media_type="application/x-ndjson",
        )

    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(_executor, run_pipeline, req),
            timeout=_PIPELINE_TIMEOUT,
        )
        return result
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"Pipeline timed out after {_PIPELINE_TIMEOUT}s. Try a simpler query or disable web search.",
        )
    except PipelineError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)


# ---- V2 admin config (requires auth) ----

@router.post("/v2/admin/config", dependencies=[Depends(_require_admin)])
def v2_admin_config(cfg: StageConfig):
    """Admin-only configuration endpoint (v2).

    Identical behavior to /api/stage/config but namespaced under /v2/admin.
    """
    clean_key = cfg.api_key.strip().encode("ascii", errors="ignore").decode("ascii").replace(" ", "")
    if cfg.base_url:
        _validate_base_url(cfg.base_url)
    config.set_stage_config(cfg.stage, cfg.provider, clean_key, cfg.model, cfg.base_url)
    return {"status": "ok", "stage": cfg.stage, "provider": cfg.provider}


@router.get("/v2/public/capabilities")
def v2_public_capabilities():
    """Public endpoint: available providers, NLI status (no secrets)."""
    from pipeline.nli import is_nli_available, NLI_SERVICE_URL
    return {
        "providers": sorted(config.PROVIDERS),
        "stages": ["gpt1", "gpt2", "gpt3"],
        "tiers": ["strict", "standard", "light"],
        "output_formats": ["auto", "structured", "annotated", "concise"],
        "nli_available": is_nli_available(),
        "nli_mode": "remote" if NLI_SERVICE_URL else ("local" if is_nli_available() else "disabled"),
        "tavily_enabled": config.is_tavily_enabled(),
        "max_prompt_length": config.MAX_PROMPT_LENGTH,
    }


@router.get("/api/rate-limit")
def rate_limit_info(request: Request):
    """Return current rate limit usage for the caller's IP."""
    return get_rate_limit_info(request)


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

    tier = getattr(req, "tier", "strict") or "strict"
    start_index = getattr(req, "start_index", 0) or 0

    return StreamingResponse(
        generate_stress_results(tests, tier=tier, start_index=start_index),
        media_type="application/x-ndjson",
    )
