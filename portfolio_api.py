"""Backward-compatibility shim.

The monolith has been split into modular packages:
  - config.py          -> Environment-based configuration
  - pipeline/           -> Core logic (models, prompts, sanitizer, helpers,
                           verifier, arbiter, orchestrator, stress)
  - api/                -> FastAPI routes and UI
  - app.py              -> Entry point

This file re-exports `app` so that existing references to
`portfolio_api:app` (e.g. in n8n workflows or scripts) continue to work.

New code should use: uvicorn app:app
"""
from app import app  # noqa: F401
