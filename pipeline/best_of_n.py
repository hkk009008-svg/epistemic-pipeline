"""Best-of-N generation — generate multiple GPT-1 responses and select the best.

Generates N candidate responses and uses lightweight heuristics to pick the
most factually grounded one, without requiring full GPT-2 verification.

Heuristics used for scoring:
1. Citation density: responses with proper citations score higher
2. Hedging language: appropriate hedging on uncertain claims
3. Banned phrase avoidance: fewer banned evidence phrases = better
4. Unknown framing: proper use of Unknown(Actionable) when needed
"""
from __future__ import annotations

import re

from pipeline.helpers import call_llm

# Citation patterns (proper sourcing)
_CITATION_RE = re.compile(r"(?:\[[^\]]{1,50}\]|\([^)]{3,50}\))")

# Banned evidence phrases that should be avoided
_BANNED_RE = re.compile(
    r"(?i)\b(?:studies suggest|research shows|data indicates"
    r"|research indicates|studies show|evidence suggests)\b"
)

# Appropriate hedging language
_HEDGING_RE = re.compile(
    r"(?i)\b(?:may|might|could|appears to|seems to|it is possible"
    r"|it is unclear|further research|not definitively|uncertain)\b"
)

# Unknown/actionable framing
_UNKNOWN_RE = re.compile(
    r"(?i)(?:Unknown\s*\(Actionable\)|Unknown\s*\(Structural\)|is unknown|remains uncertain)"
)

# Bare percentages (penalty) — simple match, no negative lookahead
_BARE_PERCENT_RE = re.compile(r"\b\d+(?:\.\d+)?\s*%")


def score_response(text: str, flags: dict) -> float:
    """Score a GPT-1 response for factuality heuristics.

    Returns a float score (higher is better). Range roughly 0-100.
    """
    score = 50.0  # Base score

    # Citation density: +5 per citation, max +20
    citations = len(_CITATION_RE.findall(text))
    score += min(citations * 5, 20)

    # Banned phrases: -10 each, min penalty -30
    banned = len(_BANNED_RE.findall(text))
    score -= min(banned * 10, 30)

    # Appropriate hedging: +3 each, max +12
    hedges = len(_HEDGING_RE.findall(text))
    score += min(hedges * 3, 12)

    # Unknown framing: +5 each, max +10
    unknowns = len(_UNKNOWN_RE.findall(text))
    score += min(unknowns * 5, 10)

    # Bare percentages: -8 each
    bare_pcts = len(_BARE_PERCENT_RE.findall(text))
    score -= bare_pcts * 8

    # Flag-specific adjustments
    if flags.get("legal_mode"):
        # Legal responses should have more citations
        if citations == 0:
            score -= 15

    if flags.get("current_events"):
        # Current events should acknowledge uncertainty
        if unknowns == 0 and hedges == 0:
            score -= 10

    return max(score, 0.0)


def generate_best_of_n(
    stage_config: dict,
    system_prompt: str,
    user_content: str,
    flags: dict,
    n: int = 2,
) -> tuple[str, dict]:
    """Generate N candidate responses and return the best one.

    Returns (best_response, selection_info) where selection_info contains:
        {
            "candidates_generated": int,
            "scores": list[float],
            "selected_index": int,
            "score_spread": float,
        }
    """
    if n < 2:
        # No point in best-of-1
        output = call_llm(stage_config, system_prompt, user_content)
        return output, {
            "candidates_generated": 1,
            "scores": [score_response(output, flags)],
            "selected_index": 0,
            "score_spread": 0.0,
        }

    candidates = []
    for _ in range(n):
        try:
            resp = call_llm(stage_config, system_prompt, user_content)
            candidates.append(resp)
        except Exception:
            # If a generation fails, continue with what we have
            break

    if not candidates:
        raise ValueError("All candidate generations failed")

    scores = [score_response(c, flags) for c in candidates]
    best_idx = scores.index(max(scores))
    score_spread = max(scores) - min(scores) if len(scores) > 1 else 0.0

    return candidates[best_idx], {
        "candidates_generated": len(candidates),
        "scores": [round(s, 1) for s in scores],
        "selected_index": best_idx,
        "score_spread": round(score_spread, 1),
    }
