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
BEST_OF_N = int(os.getenv("BEST_OF_N", "1"))  # 1 = disabled, 2+ = enable best-of-N
# Admin token — when set, POST /api/openai/config and /api/tavily/config require
# Authorization: Bearer <token>. When unset (local dev), config endpoints are open.
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")

# ---- Grounded local knowledge lane ----
# This first slice intentionally exposes one fixed corpus per deployment.  A
# future multi-user service must derive the corpus root from an authenticated
# server-side principal; clients must never choose arbitrary filesystem roots.
KNOWLEDGE_ROOT = os.getenv("KNOWLEDGE_ROOT", "knowledge_data")
KNOWLEDGE_API_TOKEN = os.getenv("KNOWLEDGE_API_TOKEN", "")

# ---- Runtime overrides (set via /api/openai/config) ----
_runtime: dict = {}
_lock = threading.Lock()


def set_runtime_config(api_key: str, model: str | None = None):
    with _lock:
        _runtime["api_key"] = api_key
        if model:
            _runtime["model"] = model
    # Invalidate cached LLM clients so they pick up the new key
    try:
        from pipeline.helpers import invalidate_client_cache
        invalidate_client_cache()
    except ImportError:
        pass


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
    return "sk-..." + key[-4:]


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
        # Inline key lookup to avoid re-acquiring _tavily_lock (non-reentrant)
        key = _tavily_runtime.get("api_key") or TAVILY_API_KEY
        if "enabled" in _tavily_runtime:
            return _tavily_runtime["enabled"] and bool(key)
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
VALID_STAGES = ("gpt1", "gpt2", "gpt3")


def set_stage_config(stage: str, provider: str = "openai", api_key: str = "",
                     model: str = "", base_url: str = ""):
    """Configure a specific pipeline stage (gpt1, gpt2, gpt3).

    provider: "openai" | "anthropic" | "openrouter" | "ollama"
    api_key: Provider-specific API key (falls back to global OpenAI key)
    model: Model identifier (falls back to global model)
    base_url: Custom API endpoint (optional, needed for openrouter/ollama)
    """
    if stage not in VALID_STAGES:
        raise ValueError(f"Invalid stage: {stage}")
    if provider not in PROVIDERS:
        raise ValueError(f"Invalid provider: {provider}")
    with _stage_lock:
        _stage_configs[stage] = {
            "provider": provider,
            "api_key": api_key,
            "model": model,
            "base_url": base_url,
        }
    # Invalidate cached LLM clients so they pick up the new stage config
    try:
        from pipeline.helpers import invalidate_client_cache
        invalidate_client_cache()
    except ImportError:
        pass


def get_stage_config(stage: str) -> dict:
    """Return config for a stage, merging with global defaults.

    Stage overrides are always applied for any field that has a non-empty
    value.  Missing fields inherit from the global OpenAI config.  This
    allows "model-only" overrides without duplicating the API key.
    """
    if stage not in VALID_STAGES:
        raise ValueError(f"Invalid stage: {stage}")
    global_defaults = {
        "provider": "openai",
        "api_key": get_api_key(),
        "model": get_model(),
        "base_url": "",
    }
    with _stage_lock:
        stage_cfg = _stage_configs.get(stage)
        if not stage_cfg:
            return global_defaults

    # Merge: stage values override globals when non-empty
    return {
        "provider": stage_cfg.get("provider") or global_defaults["provider"],
        "api_key": stage_cfg.get("api_key") or global_defaults["api_key"],
        "model": stage_cfg.get("model") or global_defaults["model"],
        "base_url": stage_cfg.get("base_url") or global_defaults["base_url"],
    }
