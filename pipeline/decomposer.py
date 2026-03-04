"""Atomic claim decomposition — breaks GPT-1 output into individual verifiable claims.

Runs as a separate LLM call BEFORE GPT-2 verification.
Each claim is a single factual assertion that can be independently verified.

Features:
- UUID-based claim IDs for deterministic arbiter targeting
- Quality validation (completeness, correctness checks)
- Compound claim detection and splitting
- Decomposition quality metrics
"""
from __future__ import annotations

import re
import uuid
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

# Patterns suggesting a compound claim (multiple assertions in one)
_COMPOUND_PATTERNS = [
    re.compile(r"\band\b.*\band\b", re.IGNORECASE),  # multiple "and" conjunctions
    re.compile(r"\bwhile\b.*\balso\b", re.IGNORECASE),
    re.compile(r"\bboth\b.*\band\b", re.IGNORECASE),
]

# Min length for a meaningful claim
_MIN_CLAIM_LENGTH = 8


def _validate_claim(c: dict) -> dict | None:
    """Validate and normalize a single claim dict. Returns None if invalid.

    Each validated claim receives a deterministic UUID (claim_id) for
    downstream targeting by the Arbiter. This enables ID-based edits
    instead of brittle text-matching.
    """
    if not isinstance(c, dict) or "text" not in c:
        return None
    text = str(c["text"]).strip()
    if len(text) < _MIN_CLAIM_LENGTH:
        return None
    return {
        "claim_id": c.get("claim_id") or uuid.uuid4().hex[:8],
        "text": text,
        "has_citation": bool(c.get("has_citation", False)),
        "is_unknown": bool(c.get("is_unknown", False)),
        "is_user_provided": bool(c.get("is_user_provided", False)),
    }


def _is_compound(claim_text: str) -> bool:
    """Heuristic check if a claim text contains multiple assertions."""
    for pat in _COMPOUND_PATTERNS:
        if pat.search(claim_text):
            return True
    # Check for semicolons separating clauses
    if ";" in claim_text:
        parts = [p.strip() for p in claim_text.split(";") if len(p.strip()) >= 10]
        if len(parts) >= 2:
            return True
    return False


def check_decomposition_quality(
    original_text: str,
    claims: List[dict],
) -> dict:
    """Evaluate the quality of a decomposition.

    Returns:
        {
            "completeness_score": float (0-1),  # coverage of original text
            "compound_count": int,  # claims that look like they need further splitting
            "avg_claim_length": float,
            "claim_count": int,
            "quality_tier": "good" | "acceptable" | "poor",
        }
    """
    if not claims:
        return {
            "completeness_score": 0.0,
            "compound_count": 0,
            "avg_claim_length": 0.0,
            "claim_count": 0,
            "quality_tier": "poor",
        }

    # Completeness: check what fraction of original content words appear in claims
    orig_words = set(re.findall(r'\b\w{4,}\b', original_text.lower()))
    claim_words = set()
    for c in claims:
        claim_words.update(re.findall(r'\b\w{4,}\b', c.get("text", "").lower()))

    completeness = len(orig_words & claim_words) / max(len(orig_words), 1)

    # Compound detection
    compound_count = sum(1 for c in claims if _is_compound(c.get("text", "")))

    # Average claim length
    lengths = [len(c.get("text", "")) for c in claims]
    avg_length = sum(lengths) / max(len(lengths), 1)

    # Quality tier
    if completeness >= 0.6 and compound_count == 0:
        tier = "good"
    elif completeness >= 0.4 or compound_count <= 1:
        tier = "acceptable"
    else:
        tier = "poor"

    return {
        "completeness_score": round(completeness, 3),
        "compound_count": compound_count,
        "avg_claim_length": round(avg_length, 1),
        "claim_count": len(claims),
        "quality_tier": tier,
    }


# Patterns for trivial / non-checkworthy claims
_TRIVIAL_PATTERNS = [
    re.compile(r"^(the )?(sky|water|sun|earth|grass) (is|are) ", re.IGNORECASE),
    re.compile(r"^(this|that|it) (is|was) (a|an|the) ", re.IGNORECASE),
    re.compile(r"^(in (summary|conclusion|general|other words))", re.IGNORECASE),
    re.compile(r"^(as (mentioned|noted|stated|discussed))", re.IGNORECASE),
]

# Minimum word count for a checkworthy claim (conservative to avoid false negatives)
_MIN_CHECKWORTHY_WORDS = 3


def filter_checkworthy(claims: List[dict]) -> List[dict]:
    """Filter out trivial/non-checkworthy claims.

    Removes:
    - Tautologies and common knowledge
    - Transition/structural phrases
    - User-provided assertions (already flagged)
    - Claims too short to be meaningful
    """
    filtered = []
    for c in claims:
        text = c.get("text", "").strip()

        # Skip user-provided claims (no need to verify what the user said)
        if c.get("is_user_provided", False):
            filtered.append(c)  # keep but mark as user-provided
            continue

        # Skip trivially short
        if len(text.split()) < _MIN_CHECKWORTHY_WORDS:
            continue

        # Skip trivial patterns
        if any(pat.match(text) for pat in _TRIVIAL_PATTERNS):
            continue

        filtered.append(c)
    return filtered


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
            v = _validate_claim(c)
            if v is not None:
                validated.append(v)
        # Filter out trivial/non-checkworthy claims
        return filter_checkworthy(validated)
    except Exception:
        # Decomposition failure is non-fatal — GPT-2 can still work without it
        return []
