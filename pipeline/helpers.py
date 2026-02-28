"""Shared helpers: JSON extraction, OpenAI call wrapper, activation bypass."""
from __future__ import annotations

import json
import re

import openai

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
                )
                result = retry_resp.choices[0].message.content

        return result
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
