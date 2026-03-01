"""Shared helpers: JSON extraction, LLM call wrappers, activation bypass."""
from __future__ import annotations

import json
import re

import openai
import anthropic as anthropic_sdk

from pipeline.prompts import ACTIVATION_PATTERNS


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
        for pattern in ACTIVATION_PATTERNS:
            if re.search(pattern, stripped, re.IGNORECASE):
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


def call_openai(client, model: str, system: str, user_content: str, expect_json: bool = False) -> str:
    """Centralized OpenAI call with error handling.

    When *expect_json* is True and the response is not valid JSON, the
    function retries once with a repair instruction asking for valid JSON.
    """
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
    except openai.APITimeoutError:
        raise PipelineError(504, "OpenAI request timed out after 60s. Try again or use a faster model.")
    except openai.AuthenticationError:
        raise PipelineError(401, "Invalid OpenAI API key. Please re-enter your key.")
    except openai.RateLimitError:
        raise PipelineError(429, "OpenAI rate limit hit. Wait a moment and try again.")
    except openai.APIError as e:
        raise PipelineError(502, f"OpenAI API error: {str(e)}")
    except PipelineError:
        raise
    except Exception as e:
        raise PipelineError(500, f"Error calling OpenAI: {str(e)}")


def _make_client(stage_config: dict):
    """Create the appropriate client for a provider config.

    Returns (provider_type, client) where provider_type is "openai" or "anthropic".
    """
    provider = stage_config["provider"]
    api_key = stage_config["api_key"]
    base_url = stage_config.get("base_url", "")

    if provider == "anthropic":
        return ("anthropic", anthropic_sdk.Anthropic(api_key=api_key))
    elif provider in ("openrouter", "ollama"):
        # Both use OpenAI-compatible API with custom base_url
        return ("openai", openai.OpenAI(api_key=api_key, base_url=base_url))
    else:
        # Default: openai
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        return ("openai", openai.OpenAI(**kwargs))


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


def _call_anthropic(client, model: str, system: str, user_content: str,
                    expect_json: bool = False) -> str:
    """Call Anthropic Messages API."""
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
    except anthropic_sdk.RateLimitError:
        raise PipelineError(429, "Anthropic rate limit hit. Wait a moment.")
    except anthropic_sdk.APIError as e:
        raise PipelineError(502, f"Anthropic API error: {str(e)}")
    except PipelineError:
        raise
    except Exception as e:
        raise PipelineError(500, f"Error calling Anthropic: {str(e)}")
