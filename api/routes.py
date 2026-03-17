"""FastAPI route definitions for the epistemic verification pipeline.

Provides both sync (ThreadPoolExecutor) and async (native asyncio) pipeline paths.
The /api/pipeline uses the async path by default for both streaming and non-streaming.
The V2 endpoints are kept for backward compat.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import time
from collections import defaultdict
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse

from api.rate_limit import rate_limit_dependency, get_rate_limit_info

import config
from pipeline.models import OpenAIConfig, TavilyConfig, StageConfig, PipelineRequest, PipelineResponse
from pipeline.helpers import PipelineError
from pipeline.runner import generate_pipeline, generate_pipeline_async, generate_pipeline_stream

router = APIRouter()


# Allowed hostnames for custom provider endpoints.
# Uses strict hostname parsing to prevent SSRF via URL authority bypass
# (e.g. http://localhost:password@attacker.com would pass a prefix check
#  but route to attacker.com).
_ALLOWED_HOSTS: set[str] = {
    "openrouter.ai",
    "api.openai.com",
    "api.anthropic.com",
    "localhost",
    "127.0.0.1",
    "host.docker.internal",
}

# Hosts that should never allow subdomain matching.
# e.g. "evil.localhost" could resolve to an attacker-controlled IP.
_NO_SUBDOMAIN_HOSTS: set[str] = {"localhost", "127.0.0.1"}

# Allow extending via env var (comma-separated URLs or hostnames)
_extra = os.getenv("ALLOWED_BASE_URLS", "")
if _extra:
    for _p in _extra.split(","):
        _p = _p.strip()
        if not _p:
            continue
        try:
            _h = urlparse(_p).hostname
            if _h:
                _ALLOWED_HOSTS.add(_h.lower())
        except Exception:
            pass


def _validate_base_url(url: str):
    """Reject base_url values whose hostname is not in the allowlist.

    Uses proper URL parsing instead of string prefix matching to prevent
    SSRF via URL authority/userinfo bypass attacks.

    Additional hardening:
    - Subdomain matching is disabled for localhost/127.0.0.1 to prevent
      DNS rebinding via evil.localhost.
    - HTTPS is required for non-local hosts to prevent credential leakage.
    """
    try:
        parsed = urlparse(url.strip())
        hostname = (parsed.hostname or "").lower()
        scheme = (parsed.scheme or "").lower()

        if not hostname:
            raise ValueError("empty hostname")

        # Exact match always accepted
        if hostname in _ALLOWED_HOSTS:
            # Non-local hosts must use HTTPS
            if hostname not in _NO_SUBDOMAIN_HOSTS and hostname != "host.docker.internal":
                if scheme != "https":
                    raise HTTPException(
                        status_code=400,
                        detail=f"External base_url must use HTTPS (got {scheme}://).",
                    )
            return

        # Subdomain match (e.g. staging.api.openai.com → api.openai.com)
        # but NOT for localhost-class hosts (evil.localhost is dangerous)
        subdomain_hosts = _ALLOWED_HOSTS - _NO_SUBDOMAIN_HOSTS
        if any(hostname.endswith("." + h) for h in subdomain_hosts):
            if scheme != "https":
                raise HTTPException(
                    status_code=400,
                    detail=f"External base_url must use HTTPS (got {scheme}://).",
                )
            return

    except HTTPException:
        raise
    except Exception:
        pass

    import itertools
    allowed_subset = list(itertools.islice(sorted(_ALLOWED_HOSTS), 6))
    
    raise HTTPException(
        status_code=400,
        detail=(
            f"base_url hostname not in allowlist. Allowed hosts: "
            f"{', '.join(allowed_subset)}... "
            f"Set ALLOWED_BASE_URLS env var to add custom hosts."
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

@router.get("/", response_class=FileResponse)
def ui():
    # Serve the newly overhauled static HTML file
    return FileResponse("static/html/index.html")


# ---- Health check (for Railway / load balancers) ----

@router.get("/health")
def health():
    return {
        "status": "ok",
        "key_set": config.has_api_key(),
        "tavily_enabled": config.is_tavily_enabled(),
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
        feedback_id=uuid.uuid4().hex,
        request_id=req.request_id,
        prompt=req.prompt,
        rating=req.rating,
        verdict_correct=req.verdict_correct,
        confidence_correct=req.confidence_correct,
    )
    store.add(entry)
    return {"status": "ok", "feedback_id": entry.feedback_id}


@router.get("/api/feedback/summary")
def feedback_summary():
    """Return aggregate feedback statistics."""
    return get_feedback_store().get_summary()


# ---- Ledger ----

@router.get("/api/ledger")
def get_ledger_data():
    """Returns the full Epistemic Knowledge Graph for visualization."""
    return get_full_graph()


# ---- Pipeline ----

@router.post("/api/pipeline", response_model=PipelineResponse, dependencies=[Depends(rate_limit_dependency)])
async def pipeline_endpoint(req: PipelineRequest, request: Request):
    """Main pipeline endpoint."""
    if req.stream:
        return StreamingResponse(
            generate_pipeline_stream(req),
            media_type="text/event-stream",
        )
    return await generate_pipeline_async(req)


@router.post("/v2/pipeline", response_model=PipelineResponse, dependencies=[Depends(rate_limit_dependency)])
async def v2_pipeline_endpoint(req: PipelineRequest, request: Request):
    """V2 pipeline endpoint (backward compatibility)."""
    return await pipeline_endpoint(req, request)


# ---- V2 admin config (requires auth) ----

@router.post("/v2/admin/config", dependencies=[Depends(_require_admin)])
def v2_admin_config(cfg: StageConfig):
    """Admin-only configuration endpoint (v2)."""
    clean_key = cfg.api_key.strip().encode("ascii", errors="ignore").decode("ascii").replace(" ", "")
    if cfg.base_url:
        _validate_base_url(cfg.base_url)
    config.set_stage_config(cfg.stage, cfg.provider, clean_key, cfg.model, cfg.base_url)
    return {"status": "ok", "stage": cfg.stage, "provider": cfg.provider}


@router.get("/v2/public/capabilities")
def v2_public_capabilities():
    """Public endpoint: available providers (no secrets)."""
    return {
        "providers": sorted(config.PROVIDERS),
        "stages": ["gpt1", "gpt2", "gpt3"],
        "tiers": ["strict", "standard", "light"],
        "output_formats": ["auto", "structured", "annotated", "concise"],
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
            import itertools
            tests.extend(list(itertools.islice(by_cat[cat], req.count)))
    if not tests:
        raise HTTPException(status_code=400, detail="No matching test cases.")

    tier = getattr(req, "tier", "strict") or "strict"
    start_index = getattr(req, "start_index", 0) or 0

    return StreamingResponse(
        generate_stress_results(tests, tier=tier, start_index=start_index),
        media_type="application/x-ndjson",
    )
