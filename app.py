import asyncio
import logging
import os
from contextlib import asynccontextmanager

import openai
from langsmith import wrappers

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi import HTTPException

import config
from api.routes import router
import database.client as db
from pipeline.helpers import PipelineError

# Lock protecting the shared OpenAI client against concurrent key-rotation races
_client_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create a shared AsyncOpenAI client at startup, close on shutdown."""
    # Initialize database tables on startup
    db.init_db()

    key = config.get_api_key()
    if key:
        raw_client = openai.AsyncOpenAI(api_key=key)
        app.state.openai_client = wrappers.wrap_openai(raw_client)
    else:
        app.state.openai_client = None
    # Track which key the cached client was built with
    app.state.openai_client_key = key
    yield
    if app.state.openai_client is not None:
        await app.state.openai_client.close()


async def get_openai_client(app_state) -> openai.AsyncOpenAI:
    """Return the cached AsyncOpenAI client, recreating if the API key changed."""
    current_key = config.get_api_key()
    if not current_key:
        return None
    # Fast path: no key change, skip lock
    if (
        app_state.openai_client is not None
        and app_state.openai_client_key == current_key
    ):
        return app_state.openai_client
    # Slow path: key changed or client missing — acquire lock to avoid race
    async with _client_lock:
        # Re-check after acquiring lock (another request may have already rotated)
        if (
            app_state.openai_client is not None
            and app_state.openai_client_key == current_key
        ):
            return app_state.openai_client
        old = app_state.openai_client
        raw_client = openai.AsyncOpenAI(api_key=current_key)
        app_state.openai_client = wrappers.wrap_openai(raw_client)
        app_state.openai_client_key = current_key
        if old is not None:
            await old.close()
    return app_state.openai_client


app = FastAPI(
    title="Epistemic Verification Pipeline",
    version="3.0.0",
    lifespan=lifespan,
)

# ---- CORS ----
# ALLOWED_ORIGINS env var: comma-separated list of allowed origins.
# Defaults to same-origin only (empty list) in production.
_allowed_origins = [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()
]
if _allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )


# ---- Security headers middleware ----
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response: Response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    if os.getenv("FORCE_HTTPS", "").lower() in ("true", "1"):
        response.headers["Strict-Transport-Security"] = (
            "max-age=63072000; includeSubDomains"
        )
    return response


# ---- Static Files ----
app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(router)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        status_code = getattr(exc, "status_code", 500)
        detail = getattr(exc, "detail", "Internal Server Error")
        return JSONResponse(
            status_code=status_code, content={"error": True, "detail": detail}
        )
    if isinstance(exc, PipelineError):
        return JSONResponse(
            status_code=exc.status_code, content={"error": True, "detail": exc.detail}
        )
    # Never expose raw exception messages — they can leak API keys, internal URLs
    logging.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500, content={"error": True, "detail": "Internal server error"}
    )
