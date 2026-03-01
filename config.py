"""Centralized configuration with environment variable support.

Reads from env vars at import time, with runtime override via API.
Thread-safe runtime config access.
"""
from __future__ import annotations

import os
import threading

# ---- Env-based defaults ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
PIPELINE_URL = os.getenv("PIPELINE_URL", "http://localhost:8000")
PORT = int(os.getenv("PORT", "8000"))
MAX_REWRITE_LOOPS = 3
MAX_PROMPT_LENGTH = 10000
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "20"))

# ---- Runtime overrides (set via /api/openai/config) ----
_runtime: dict = {}
_lock = threading.Lock()


def set_runtime_config(api_key: str, model: str | None = None):
    with _lock:
        _runtime["api_key"] = api_key
        if model:
            _runtime["model"] = model


def get_api_key() -> str:
    with _lock:
        return _runtime.get("api_key") or OPENAI_API_KEY


def get_model() -> str:
    with _lock:
        return _runtime.get("model") or OPENAI_MODEL


def has_api_key() -> bool:
    return bool(get_api_key())


def get_key_preview() -> str:
    key = get_api_key()
    if not key:
        return ""
    return key[:8] + "..." + key[-4:]


# ---- Tavily (web search) ----
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

_tavily_runtime: dict = {}
_tavily_lock = threading.Lock()


def set_tavily_config(api_key: str, enabled: bool = True):
    with _tavily_lock:
        _tavily_runtime["api_key"] = api_key
        _tavily_runtime["enabled"] = enabled


def get_tavily_key() -> str:
    with _tavily_lock:
        return _tavily_runtime.get("api_key") or TAVILY_API_KEY


def is_tavily_enabled() -> bool:
    with _tavily_lock:
        # Enabled if there's a key and not explicitly disabled
        if "enabled" in _tavily_runtime:
            return _tavily_runtime["enabled"] and bool(get_tavily_key())
        return bool(TAVILY_API_KEY)


def set_tavily_enabled(enabled: bool):
    with _tavily_lock:
        _tavily_runtime["enabled"] = enabled


def has_tavily_key() -> bool:
    return bool(get_tavily_key())


def get_tavily_key_preview() -> str:
    key = get_tavily_key()
    if not key:
        return ""
    return key[:8] + "..." + key[-4:]


# ---- Per-stage model configuration ----
# Each stage can have its own provider, model, API key, and base URL.
# Falls back to the global OpenAI config if not set.
_stage_configs: dict = {}  # {"gpt1": {...}, "gpt2": {...}, "gpt3": {...}}
_stage_lock = threading.Lock()

# Valid provider types
PROVIDERS = {"openai", "anthropic", "openrouter", "ollama"}


def set_stage_config(stage: str, provider: str = "openai", api_key: str = "",
                     model: str = "", base_url: str = ""):
    """Configure a specific pipeline stage (gpt1, gpt2, gpt3).

    provider: "openai" | "anthropic" | "openrouter" | "ollama"
    api_key: Provider-specific API key (falls back to global OpenAI key)
    model: Model identifier (falls back to global model)
    base_url: Custom API endpoint (optional, needed for openrouter/ollama)
    """
    assert stage in ("gpt1", "gpt2", "gpt3"), f"Invalid stage: {stage}"
    assert provider in PROVIDERS, f"Invalid provider: {provider}"
    with _stage_lock:
        _stage_configs[stage] = {
            "provider": provider,
            "api_key": api_key,
            "model": model,
            "base_url": base_url,
        }


def get_stage_config(stage: str) -> dict:
    """Return config for a stage. Falls back to global OpenAI config."""
    with _stage_lock:
        if stage in _stage_configs and _stage_configs[stage].get("api_key"):
            return _stage_configs[stage]
    # Fallback to global
    return {
        "provider": "openai",
        "api_key": get_api_key(),
        "model": get_model(),
        "base_url": "",
    }
