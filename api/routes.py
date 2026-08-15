"""FastAPI route definitions for the epistemic verification pipeline.

Provides both sync (ThreadPoolExecutor) and async (native asyncio) pipeline paths.
The /api/pipeline uses the async path by default for both streaming and non-streaming.
The V2 endpoints are kept for backward compat.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import hmac
import json
import os
import time
from collections import defaultdict
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse

import config
from api.rate_limit import get_rate_limit_info, rate_limit_dependency
from api.ui import UI_HTML
from pipeline.feedback import FeedbackEntry, get_feedback_store
from pipeline.grounded_rag import (
    GroundedDocumentRequest,
    GroundedDocumentResponse,
    GroundedFolderSyncRequest,
    GroundedQueryRequest,
    GroundedRAGResponse,
    run_grounded_rag,
)
from pipeline.helpers import PipelineError
from pipeline.knowledge_store import KnowledgeStore, KnowledgeStoreError
from pipeline.metrics import get_aggregate
from pipeline.models import (
    FeedbackRequest,
    OpenAIConfig,
    PipelineRequest,
    PipelineResponse,
    StageConfig,
    StressRequest,
    TavilyConfig,
)
from pipeline.orchestrator import run_pipeline, run_pipeline_async
from pipeline.stress import generate_stress_results

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

    raise HTTPException(
        status_code=400,
        detail=(
            f"base_url hostname not in allowlist. Allowed hosts: "
            f"{', '.join(sorted(list(_ALLOWED_HOSTS)[:6]))}... "
            f"Set ALLOWED_BASE_URLS env var to add custom hosts."
        ),
    )


def _require_admin(request: Request):
    """Verify a control-plane token on config-mutation endpoints.

    ADMIN_TOKEN takes precedence. When grounded mode is enabled without a
    separate admin token, its mandatory knowledge token also closes the model
    configuration plane that determines where private evidence is sent.
    """
    token = config.ADMIN_TOKEN or config.KNOWLEDGE_API_TOKEN
    if not token:
        return  # Both protected modes are disabled: preserve local-dev behavior.
    auth = request.headers.get("authorization", "")
    expected = f"Bearer {token}"
    if not hmac.compare_digest(auth.encode("utf-8"), expected.encode("utf-8")):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized — invalid or missing control-plane token.",
        )


def _require_grounded_access(request: Request):
    """Protect all reads and writes to the private local knowledge corpus.

    Unlike the development-friendly admin endpoints, grounded endpoints remain
    disabled when no token is configured.  User data must never become a public
    endpoint because an environment variable was omitted.
    """
    token = config.KNOWLEDGE_API_TOKEN
    if not token:
        raise HTTPException(
            status_code=503,
            detail=(
                "Grounded knowledge endpoints are disabled until "
                "KNOWLEDGE_API_TOKEN is set."
            ),
        )
    supplied = request.headers.get("authorization", "")
    expected = f"Bearer {token}"
    if not hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8")):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized — invalid or missing knowledge token.",
        )


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
    try:
        config.set_stage_config(cfg.stage, cfg.provider, clean_key, cfg.model, cfg.base_url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"status": "ok", "stage": cfg.stage, "provider": cfg.provider}


@router.get("/api/stage/config/{stage}")
def get_stage_config_endpoint(stage: str):
    if stage not in config.VALID_STAGES:
        raise HTTPException(status_code=404, detail=f"Unknown stage: {stage}")
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
    """Sync streaming fallback — used by V2 endpoints.

    Uses a thread-safe queue to receive real stage_start / stage_complete
    events emitted by the orchestrator's ``emit`` callback.
    """
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


async def _stream_pipeline_async(req: PipelineRequest):
    """Native async event streaming — uses run_pipeline_async without blocking threads.

    Uses an asyncio.Queue to receive stage events from the emit callback.
    The emit callback is called from within the async pipeline coroutine
    (same event loop), so put_nowait is safe.
    """
    event_q: asyncio.Queue = asyncio.Queue()
    start = time.monotonic()

    def _emit(event: dict):
        """Callback passed to run_pipeline_async for real stage events."""
        event["elapsed"] = round(time.monotonic() - start, 1)
        event_q.put_nowait(event)

    # Launch the async pipeline as a concurrent task
    pipeline_task = asyncio.create_task(run_pipeline_async(req, emit=_emit))

    deadline = time.monotonic() + _PIPELINE_TIMEOUT
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            pipeline_task.cancel()
            yield json.dumps({"type": "error", "detail": f"Pipeline timed out after {_PIPELINE_TIMEOUT}s.", "status_code": 504}) + "\n"
            return

        try:
            event = await asyncio.wait_for(event_q.get(), timeout=min(remaining, 0.5))
            yield json.dumps(event) + "\n"
        except asyncio.TimeoutError:
            # No event within 0.5s — check if pipeline finished
            if pipeline_task.done():
                # Drain remaining events
                while not event_q.empty():
                    yield json.dumps(event_q.get_nowait()) + "\n"
                break

    # Yield the final result or error
    try:
        result = await pipeline_task
        yield json.dumps({"type": "result", "data": result.model_dump()}) + "\n"
    except PipelineError as e:
        yield json.dumps({"type": "error", "detail": e.detail, "status_code": e.status_code}) + "\n"
    except Exception as e:
        yield json.dumps({"type": "error", "detail": str(e), "status_code": 500}) + "\n"


@router.post("/api/pipeline", response_model=PipelineResponse, dependencies=[Depends(rate_limit_dependency)])
async def pipeline_endpoint(req: PipelineRequest, request: Request):
    """Main pipeline endpoint — uses native async I/O (V4).

    Both streaming and non-streaming modes use the async pipeline,
    avoiding ThreadPoolExecutor entirely. Falls back to sync on error.
    """
    # Native async streaming — uses asyncio.Queue, no OS threads
    if getattr(req, "stream", False):
        return StreamingResponse(
            _stream_pipeline_async(req),
            media_type="application/x-ndjson",
        )

    # Native async path — no ThreadPoolExecutor, no thread locks
    try:
        result = await asyncio.wait_for(
            run_pipeline_async(req),
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
    except Exception:
        # Fallback to sync path if async pipeline fails unexpectedly
        loop = asyncio.get_running_loop()
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


@router.post(
    "/api/grounded/documents/{document_id}",
    response_model=GroundedDocumentResponse,
    dependencies=[Depends(rate_limit_dependency), Depends(_require_grounded_access)],
)
async def grounded_document_upsert(document_id: str, req: GroundedDocumentRequest):
    """Store an immutable document version and atomically update the FTS index."""
    store = KnowledgeStore(config.KNOWLEDGE_ROOT)
    try:
        record = await asyncio.to_thread(
            store.upsert_document,
            document_id,
            req.folder,
            req.title,
            req.content,
            expected_revision_id=req.expected_revision_id,
            revision_reason=req.revision_reason,
        )
        return GroundedDocumentResponse.from_record(record)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except KnowledgeStoreError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.get(
    "/api/grounded/documents",
    response_model=list[GroundedDocumentResponse],
    dependencies=[Depends(rate_limit_dependency), Depends(_require_grounded_access)],
)
async def grounded_documents_list(folder: str | None = None):
    """List active documents in the grounded knowledge store."""
    store = KnowledgeStore(config.KNOWLEDGE_ROOT)
    records = await asyncio.to_thread(store.list_documents, folder=folder)
    return [GroundedDocumentResponse.from_record(r) for r in records]


@router.get(
    "/api/grounded/receipts/{run_id}",
    dependencies=[Depends(rate_limit_dependency), Depends(_require_grounded_access)],
)
async def grounded_receipt_get(run_id: str):
    """Retrieve an immutable cryptographic run receipt for an earlier grounded query."""
    store = KnowledgeStore(config.KNOWLEDGE_ROOT)
    try:
        receipt = await asyncio.to_thread(store.load_run_receipt, run_id)
        return receipt
    except KnowledgeStoreError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post(
    "/api/grounded/sync-folder",
    response_model=list[GroundedDocumentResponse],
    dependencies=[Depends(rate_limit_dependency), Depends(_require_grounded_access)],
)
async def grounded_folder_sync(req: GroundedFolderSyncRequest):
    """Ingest a directory of documents into the grounded knowledge store."""
    store = KnowledgeStore(config.KNOWLEDGE_ROOT)
    try:
        records = await asyncio.to_thread(store.sync_folder, req.folder_path, target_folder=req.target_folder)
        return [GroundedDocumentResponse.from_record(r) for r in records]
    except (KnowledgeStoreError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post(
    "/api/grounded/query",
    response_model=GroundedRAGResponse,
    dependencies=[Depends(rate_limit_dependency), Depends(_require_grounded_access)],
)
async def grounded_query(req: GroundedQueryRequest):
    """Answer from the fixed local corpus through all grounded evidence gates."""
    try:
        return await asyncio.wait_for(run_grounded_rag(req), timeout=_PIPELINE_TIMEOUT)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"Grounded pipeline timed out after {_PIPELINE_TIMEOUT}s.",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except KnowledgeStoreError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except PipelineError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail)


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

    loop = asyncio.get_running_loop()
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
    try:
        config.set_stage_config(cfg.stage, cfg.provider, clean_key, cfg.model, cfg.base_url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
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
