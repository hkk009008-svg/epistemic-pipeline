"""Centralized configuration with environment variable support.

Reads from env vars at import time, with runtime override via API.
"""
from __future__ import annotations

import os

# ---- Env-based defaults ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
PIPELINE_URL = os.getenv("PIPELINE_URL", "http://localhost:8000")
PORT = int(os.getenv("PORT", "8000"))
MAX_REWRITE_LOOPS = 1

# ---- Runtime overrides (set via /api/openai/config) ----
_runtime: dict = {}


def set_runtime_config(api_key: str, model: str | None = None):
    _runtime["api_key"] = api_key
    if model:
        _runtime["model"] = model


def get_api_key() -> str:
    return _runtime.get("api_key") or OPENAI_API_KEY


def get_model() -> str:
    return _runtime.get("model") or OPENAI_MODEL


def has_api_key() -> bool:
    return bool(get_api_key())


def get_key_preview() -> str:
    key = get_api_key()
    if not key:
        return ""
    return key[:8] + "..." + key[-4:]
