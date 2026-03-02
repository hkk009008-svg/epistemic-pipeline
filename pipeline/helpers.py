"""Shared helpers: JSON extraction, LLM call wrappers, activation bypass."""
from __future__ import annotations

import json
import re
import threading
import time

import openai
import anthropic as anthropic_sdk

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
    """Clear the client cache (called when API keys change at runtime)."""
    with _client_cache_lock:
        _client_cache.clear()


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
