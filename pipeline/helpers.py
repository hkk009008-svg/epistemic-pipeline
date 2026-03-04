"""Shared helpers: JSON extraction, LLM call wrappers, activation bypass.

Provides both synchronous and async LLM call paths:
- Sync: call_openai(), call_llm() — used by existing pipeline
- Async: call_openai_async(), call_llm_async(), call_llm_structured() — V4 async pipeline
"""
from __future__ import annotations

import asyncio
import json
import re
import threading
import time
from typing import Optional

import openai
import anthropic as anthropic_sdk

from pydantic import BaseModel

from pipeline.prompts import ACTIVATION_PATTERNS

# Pre-compile activation patterns once at import time
_COMPILED_ACTIVATION = [re.compile(p, re.IGNORECASE) for p in ACTIVATION_PATTERNS]


class PipelineError(Exception):
    """Raised for pipeline-level errors (auth, rate limit, etc.)."""
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


def is_activation_phrase(text: str) -> bool:
    """Check if GPT-1 output is an activation/init phrase that should bypass GPT-2."""
    stripped = text.strip()
    if len(stripped) < 100:
        for pattern in _COMPILED_ACTIVATION:
            if pattern.search(stripped):
                return True
    return False


def extract_json(raw: str) -> dict:
    """Hardened JSON extractor. Handles fences, prose wrapping, truncation."""
    cleaned = raw.strip()

    # Strip markdown code fences
    if "```" in cleaned:
        parts = cleaned.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                cleaned = part
                break

    # If it starts with prose before JSON, extract the JSON object
    if not cleaned.startswith("{"):
        match = re.search(r'\{[\s\S]*\}', cleaned)
        if match:
            cleaned = match.group(0)

    # Try parsing as-is
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try fixing truncated JSON by closing brackets
    for suffix in ["}", "]}", '"]}', '"}]}', '"]}]}']:
        try:
            result = json.loads(cleaned + suffix)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            continue

    raise ValueError(f"Could not parse JSON from: {raw[:300]}")


def _is_transient_openai(exc: Exception) -> bool:
    """Return True if the OpenAI exception is transient and worth retrying."""
    return isinstance(exc, (openai.APITimeoutError, openai.RateLimitError,
                            openai.InternalServerError, openai.APIConnectionError))


def call_openai(client, model: str, system: str, user_content: str,
                expect_json: bool = False, _max_retries: int = 3) -> str:
    """Centralized OpenAI call with error handling and retry on transient errors.

    Retries up to *_max_retries* times with exponential backoff (2s, 4s, 8s)
    for transient errors (timeouts, rate limits, 5xx, connection errors).

    When *expect_json* is True and the response is not valid JSON, the
    function retries once with a repair instruction asking for valid JSON.
    """
    last_exc: Exception | None = None

    for attempt in range(_max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_content},
                ],
                timeout=60,
            )
            result = resp.choices[0].message.content

            # Schema-enforced JSON retry
            if expect_json:
                try:
                    extract_json(result)
                except (ValueError, json.JSONDecodeError):
                    retry_resp = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user_content},
                            {"role": "assistant", "content": result},
                            {
                                "role": "user",
                                "content": (
                                    "Your previous response was not valid JSON. "
                                    "Please output ONLY valid JSON matching the required schema."
                                ),
                            },
                        ],
                        timeout=60,
                    )
                    result = retry_resp.choices[0].message.content

            return result
        except openai.AuthenticationError:
            raise PipelineError(401, "Invalid OpenAI API key. Please re-enter your key.")
        except PipelineError:
            raise
        except Exception as e:
            if _is_transient_openai(e) and attempt < _max_retries:
                last_exc = e
                time.sleep(2 ** (attempt + 1))  # 2s, 4s, 8s
                continue
            # Non-transient or exhausted retries — raise descriptive error
            if isinstance(e, openai.APITimeoutError):
                raise PipelineError(504, "OpenAI request timed out after 60s. Try again or use a faster model.")
            if isinstance(e, openai.RateLimitError):
                raise PipelineError(429, "OpenAI rate limit hit. Wait a moment and try again.")
            if isinstance(e, openai.APIError):
                msg = f"OpenAI API error: {str(e)}"
                if last_exc:
                    msg += f" (after {attempt} retries)"
                raise PipelineError(502, msg)
            raise PipelineError(500, f"Error calling OpenAI: {str(e)}")

    # Should not reach here, but just in case
    raise PipelineError(502, f"OpenAI call failed after {_max_retries} retries: {last_exc}")


# Client cache: reuse TCP connections across calls (avoids SSL handshake per call)
_client_cache: dict[tuple, tuple[str, object]] = {}
_client_cache_lock = threading.Lock()


def _make_client(stage_config: dict):
    """Return a cached (provider_type, client) for the given config.

    Clients are cached by (provider, api_key, base_url) to reuse TCP
    connections and benefit from HTTP/2 multiplexing. This avoids the
    overhead of a fresh SSL handshake on every LLM call (3-7 per request).
    """
    provider = stage_config["provider"]
    api_key = stage_config["api_key"]
    base_url = stage_config.get("base_url", "")

    cache_key = (provider, api_key, base_url)

    with _client_cache_lock:
        cached = _client_cache.get(cache_key)
        if cached is not None:
            return cached

    if provider == "anthropic":
        result = ("anthropic", anthropic_sdk.Anthropic(api_key=api_key))
    elif provider in ("openrouter", "ollama"):
        # Both use OpenAI-compatible API with custom base_url
        result = ("openai", openai.OpenAI(api_key=api_key, base_url=base_url))
    else:
        # Default: openai
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        result = ("openai", openai.OpenAI(**kwargs))

    with _client_cache_lock:
        _client_cache[cache_key] = result
    return result


def invalidate_client_cache() -> None:
    """Clear both sync and async client caches (called when API keys change at runtime)."""
    with _client_cache_lock:
        _client_cache.clear()
    # Also clear the async cache if it has been initialized
    try:
        invalidate_async_client_cache()
    except NameError:
        pass  # async cache not yet defined during module load


def call_llm(stage_config: dict, system: str, user_content: str,
             expect_json: bool = False) -> str:
    """Unified LLM call that dispatches to OpenAI or Anthropic SDK.

    stage_config: dict from config.get_stage_config("gpt1"|"gpt2"|"gpt3")
    Returns the text content of the response.
    """
    provider_type, client = _make_client(stage_config)
    model = stage_config["model"]

    if provider_type == "anthropic":
        return _call_anthropic(client, model, system, user_content, expect_json)
    else:
        return call_openai(client, model, system, user_content, expect_json)


def _is_transient_anthropic(exc: Exception) -> bool:
    """Return True if the Anthropic exception is transient and worth retrying."""
    return isinstance(exc, (anthropic_sdk.RateLimitError,
                            anthropic_sdk.InternalServerError,
                            anthropic_sdk.APIConnectionError,
                            anthropic_sdk.APITimeoutError))


def _call_anthropic(client, model: str, system: str, user_content: str,
                    expect_json: bool = False, _max_retries: int = 3) -> str:
    """Call Anthropic Messages API with retry on transient errors."""
    last_exc: Exception | None = None

    for attempt in range(_max_retries + 1):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=4096,
                system=system,
                messages=[{"role": "user", "content": user_content}],
            )
            result = resp.content[0].text

            if expect_json:
                try:
                    extract_json(result)
                except (ValueError, json.JSONDecodeError):
                    retry_resp = client.messages.create(
                        model=model,
                        max_tokens=4096,
                        system=system,
                        messages=[
                            {"role": "user", "content": user_content},
                            {"role": "assistant", "content": result},
                            {"role": "user", "content": (
                                "Your previous response was not valid JSON. "
                                "Please output ONLY valid JSON matching the required schema."
                            )},
                        ],
                    )
                    result = retry_resp.content[0].text

            return result
        except anthropic_sdk.AuthenticationError:
            raise PipelineError(401, "Invalid Anthropic API key.")
        except PipelineError:
            raise
        except Exception as e:
            if _is_transient_anthropic(e) and attempt < _max_retries:
                last_exc = e
                time.sleep(2 ** (attempt + 1))
                continue
            if isinstance(e, anthropic_sdk.RateLimitError):
                raise PipelineError(429, "Anthropic rate limit hit. Wait a moment.")
            if isinstance(e, anthropic_sdk.APIError):
                msg = f"Anthropic API error: {str(e)}"
                if last_exc:
                    msg += f" (after {attempt} retries)"
                raise PipelineError(502, msg)
            raise PipelineError(500, f"Error calling Anthropic: {str(e)}")

    raise PipelineError(502, f"Anthropic call failed after {_max_retries} retries: {last_exc}")


# ---------------------------------------------------------------------------
# Async client cache — mirrors the sync _client_cache for AsyncOpenAI/AsyncAnthropic
# ---------------------------------------------------------------------------
_async_client_cache: dict[tuple, tuple[str, object]] = {}
_async_client_cache_lock = threading.Lock()


def _make_async_client(stage_config: dict):
    """Return a cached (provider_type, async_client) for the given config.

    Async clients are cached by (provider, api_key, base_url) to reuse TCP
    connections and benefit from HTTP/2 multiplexing.
    """
    provider = stage_config["provider"]
    api_key = stage_config["api_key"]
    base_url = stage_config.get("base_url", "")

    cache_key = (provider, api_key, base_url)

    with _async_client_cache_lock:
        cached = _async_client_cache.get(cache_key)
        if cached is not None:
            return cached

    if provider == "anthropic":
        result = ("anthropic", anthropic_sdk.AsyncAnthropic(api_key=api_key))
    elif provider in ("openrouter", "ollama"):
        result = ("openai", openai.AsyncOpenAI(api_key=api_key, base_url=base_url))
    else:
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        result = ("openai", openai.AsyncOpenAI(**kwargs))

    with _async_client_cache_lock:
        _async_client_cache[cache_key] = result
    return result


def invalidate_async_client_cache() -> None:
    """Clear the async client cache (called when API keys change at runtime)."""
    with _async_client_cache_lock:
        _async_client_cache.clear()


# ---------------------------------------------------------------------------
# Async LLM call wrappers
# ---------------------------------------------------------------------------


async def call_openai_async(
    client: openai.AsyncOpenAI,
    model: str,
    system: str,
    user_content: str,
    expect_json: bool = False,
    _max_retries: int = 3,
) -> str:
    """Async OpenAI call with error handling and retry on transient errors.

    Mirrors call_openai() but uses native async I/O instead of blocking threads.
    """
    last_exc: Exception | None = None

    for attempt in range(_max_retries + 1):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_content},
                ],
                timeout=60,
            )
            result = resp.choices[0].message.content

            if expect_json:
                try:
                    extract_json(result)
                except (ValueError, json.JSONDecodeError):
                    retry_resp = await client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user_content},
                            {"role": "assistant", "content": result},
                            {
                                "role": "user",
                                "content": (
                                    "Your previous response was not valid JSON. "
                                    "Please output ONLY valid JSON matching the required schema."
                                ),
                            },
                        ],
                        timeout=60,
                    )
                    result = retry_resp.choices[0].message.content

            return result
        except openai.AuthenticationError:
            raise PipelineError(401, "Invalid OpenAI API key. Please re-enter your key.")
        except PipelineError:
            raise
        except Exception as e:
            if _is_transient_openai(e) and attempt < _max_retries:
                last_exc = e
                await asyncio.sleep(2 ** (attempt + 1))
                continue
            if isinstance(e, openai.APITimeoutError):
                raise PipelineError(504, "OpenAI request timed out after 60s. Try again or use a faster model.")
            if isinstance(e, openai.RateLimitError):
                raise PipelineError(429, "OpenAI rate limit hit. Wait a moment and try again.")
            if isinstance(e, openai.APIError):
                msg = f"OpenAI API error: {str(e)}"
                if last_exc:
                    msg += f" (after {attempt} retries)"
                raise PipelineError(502, msg)
            raise PipelineError(500, f"Error calling OpenAI: {str(e)}")

    raise PipelineError(502, f"Async OpenAI call failed after {_max_retries} retries: {last_exc}")


async def _call_anthropic_async(
    client: anthropic_sdk.AsyncAnthropic,
    model: str,
    system: str,
    user_content: str,
    expect_json: bool = False,
    _max_retries: int = 3,
) -> str:
    """Async Anthropic Messages API call with retry on transient errors."""
    last_exc: Exception | None = None

    for attempt in range(_max_retries + 1):
        try:
            resp = await client.messages.create(
                model=model,
                max_tokens=4096,
                system=system,
                messages=[{"role": "user", "content": user_content}],
            )
            result = resp.content[0].text

            if expect_json:
                try:
                    extract_json(result)
                except (ValueError, json.JSONDecodeError):
                    retry_resp = await client.messages.create(
                        model=model,
                        max_tokens=4096,
                        system=system,
                        messages=[
                            {"role": "user", "content": user_content},
                            {"role": "assistant", "content": result},
                            {"role": "user", "content": (
                                "Your previous response was not valid JSON. "
                                "Please output ONLY valid JSON matching the required schema."
                            )},
                        ],
                    )
                    result = retry_resp.content[0].text

            return result
        except anthropic_sdk.AuthenticationError:
            raise PipelineError(401, "Invalid Anthropic API key.")
        except PipelineError:
            raise
        except Exception as e:
            if _is_transient_anthropic(e) and attempt < _max_retries:
                last_exc = e
                await asyncio.sleep(2 ** (attempt + 1))
                continue
            if isinstance(e, anthropic_sdk.RateLimitError):
                raise PipelineError(429, "Anthropic rate limit hit. Wait a moment.")
            if isinstance(e, anthropic_sdk.APIError):
                msg = f"Anthropic API error: {str(e)}"
                if last_exc:
                    msg += f" (after {attempt} retries)"
                raise PipelineError(502, msg)
            raise PipelineError(500, f"Error calling Anthropic: {str(e)}")

    raise PipelineError(502, f"Async Anthropic call failed after {_max_retries} retries: {last_exc}")


async def call_llm_async(
    stage_config: dict,
    system: str,
    user_content: str,
    expect_json: bool = False,
) -> str:
    """Async unified LLM call that dispatches to OpenAI or Anthropic SDK.

    Mirrors call_llm() but uses native async I/O.
    """
    provider_type, client = _make_async_client(stage_config)
    model = stage_config["model"]

    if provider_type == "anthropic":
        return await _call_anthropic_async(client, model, system, user_content, expect_json)
    else:
        return await call_openai_async(client, model, system, user_content, expect_json)


async def call_llm_structured(
    stage_config: dict,
    system: str,
    user_content: str,
    response_model: type[BaseModel],
    _max_retries: int = 3,
) -> BaseModel:
    """Async LLM call using OpenAI's strict structured outputs (response_format).

    Uses the OpenAI beta parse API to mathematically enforce valid JSON
    matching the Pydantic schema. Eliminates the need for extract_json()
    regex parsing and retry loops.

    Falls back to call_llm_async() + extract_json() for non-OpenAI providers
    that don't support structured outputs.

    Args:
        stage_config: Stage configuration dict with provider/model/key.
        system: System prompt.
        user_content: User content.
        response_model: Pydantic model class for the response schema.
        _max_retries: Max retries for transient errors.

    Returns:
        A fully instantiated, validated Pydantic object.
    """
    provider_type, client = _make_async_client(stage_config)
    model = stage_config["model"]

    # Structured outputs only supported by OpenAI-compatible APIs
    if provider_type != "openai":
        raw = await call_llm_async(stage_config, system, user_content, expect_json=True)
        parsed = extract_json(raw)
        return response_model.model_validate(parsed)

    last_exc: Exception | None = None

    for attempt in range(_max_retries + 1):
        try:
            resp = await client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_content},
                ],
                response_format=response_model,
                timeout=60,
            )
            parsed = resp.choices[0].message.parsed
            if parsed is not None:
                return parsed
            # If parsed is None (refusal or incomplete), fall back to text extraction
            content = resp.choices[0].message.content or ""
            return response_model.model_validate(extract_json(content))
        except openai.AuthenticationError:
            raise PipelineError(401, "Invalid OpenAI API key. Please re-enter your key.")
        except PipelineError:
            raise
        except Exception as e:
            if _is_transient_openai(e) and attempt < _max_retries:
                last_exc = e
                await asyncio.sleep(2 ** (attempt + 1))
                continue
            # For non-transient errors or API versions that don't support parse,
            # fall back to standard async call + JSON extraction
            try:
                raw = await call_llm_async(stage_config, system, user_content, expect_json=True)
                parsed_dict = extract_json(raw)
                return response_model.model_validate(parsed_dict)
            except Exception:
                pass
            if isinstance(e, openai.APIError):
                raise PipelineError(502, f"Structured output call failed: {str(e)}")
            raise PipelineError(500, f"Error in structured LLM call: {str(e)}")

    raise PipelineError(502, f"Structured call failed after {_max_retries} retries: {last_exc}")
