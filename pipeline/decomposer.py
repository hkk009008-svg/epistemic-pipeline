"""Atomic claim decomposition — breaks GPT-1 output into individual verifiable claims.

Runs as a separate LLM call BEFORE GPT-2 verification.
Each claim is a single factual assertion that can be independently verified.
"""
from __future__ import annotations

from typing import List

from pipeline.helpers import extract_json, call_llm

DECOMPOSER_SYSTEM = (
    "You are a claim decomposition engine. Your ONLY job is to break text into "
    "atomic claims — single factual assertions that can each be independently "
    "verified as true or false.\n\n"
    "Rules:\n"
    "1. Each claim must be ONE factual assertion (not compound).\n"
    "2. Preserve the original meaning — do not add or remove information.\n"
    "3. Include structural claims like 'The author states confidence is Medium'.\n"
    "4. Flag claims that contain citations (e.g., [1], [Source]) with has_citation: true.\n"
    "5. Flag claims that are framed as Unknown/uncertain with is_unknown: true.\n"
    "6. Flag claims that are user-provided assertions with is_user_provided: true.\n\n"
    "Output VALID JSON ONLY:\n"
    '{"claims": [\n'
    '  {"text": "...", "has_citation": false, "is_unknown": false, "is_user_provided": false},\n'
    "  ...\n"
    "]}"
)


def decompose_claims(stage_config: dict, gpt1_output: str,
                     user_prompt: str = "") -> List[dict]:
    """Decompose GPT-1 output into atomic claims.

    Returns list of dicts: [{"text": "...", "has_citation": bool, "is_unknown": bool, ...}]
    Returns empty list on failure (decomposition is best-effort, not blocking).
    """
    user_content = (
        f"USER PROMPT (for context):\n{user_prompt}\n\n"
        f"TEXT TO DECOMPOSE:\n{gpt1_output}"
    )
    try:
        raw = call_llm(stage_config, DECOMPOSER_SYSTEM, user_content, expect_json=True)
        parsed = extract_json(raw)
        claims = parsed.get("claims", [])
        # Validate structure
        validated = []
        for c in claims:
            if isinstance(c, dict) and "text" in c:
                validated.append({
                    "text": str(c["text"]),
                    "has_citation": bool(c.get("has_citation", False)),
                    "is_unknown": bool(c.get("is_unknown", False)),
                    "is_user_provided": bool(c.get("is_user_provided", False)),
                })
        return validated
    except Exception:
        # Decomposition failure is non-fatal — GPT-2 can still work without it
        return []
