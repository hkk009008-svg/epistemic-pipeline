"""Deterministic prompt router and output sanitizer.

These run before/after LLM calls to reduce nondeterminism.
"""
from __future__ import annotations

import re

# ---- Prompt Router patterns ----
_ADVICE_RE = re.compile(
    r"(?i)\b(?:what should I do|should I|recommend|steps|best way|would it help"
    r"|what are my options|how do I|how can I|tips for|help me)\b"
)
_PERCENT_RE = re.compile(
    r"(?i)\b(?:percent|percentage|rate|odds|how many|how often"
    r"|probability|fraction|proportion|typically)\b"
)
_LEGAL_RE = re.compile(
    r"(?i)\b(?:legal|illegal|law|regulation|import|export|IRS|SEC"
    r"|compliance|statute|ordinance|ban|prohibited)\b"
)
_JURISDICTION_RE = re.compile(
    r"(?i)\b(?:US|UK|EU|federal|state of"
    r"|United States|United Kingdom|Canada|Australia|Germany|France|India"
    r"|China|Japan|Brazil|Mexico|California|Texas|New York|Florida"
    r"|Ohio|Illinois|Pennsylvania|Georgia|Michigan|Virginia"
    r"|North Carolina|South Carolina|Massachusetts|Washington"
    r"|Arizona|Colorado|Oregon|Nevada|Tennessee|Kentucky"
    r"|Alabama|Louisiana|Maryland|Minnesota|Wisconsin|Missouri"
    r"|Connecticut|Iowa|Arkansas|Mississippi|Kansas|Utah"
    r"|Nebraska|Oklahoma|New Mexico|Hawaii|Idaho|Montana"
    r"|Wyoming|Vermont|Maine|New Hampshire|Rhode Island"
    r"|South Dakota|North Dakota|Delaware|West Virginia|Alaska)\b"
)
_FUTURE_YEAR_RE = re.compile(r"\b(20[3-9]\d|2[1-9]\d{2}|[3-9]\d{3})\b")


def route_prompt(prompt: str) -> dict:
    """Deterministic heuristic router -- classifies prompt features for downstream use."""
    return {
        "advice_requested": bool(_ADVICE_RE.search(prompt)),
        "percent_requested": bool(_PERCENT_RE.search(prompt)),
        "legal_mode": bool(_LEGAL_RE.search(prompt)),
        "jurisdiction_present": bool(_JURISDICTION_RE.search(prompt)),
        "future_year": bool(_FUTURE_YEAR_RE.search(prompt)),
    }


# ---- Output Sanitizer patterns ----
_BANNED_EVIDENCE_RE = re.compile(
    r"(?i)\b(?:studies suggest|research shows|data indicates"
    r"|generally|often|typically|commonly|usually)\b"
    r"(?!\s*\([^)]+\))"    # not followed by a parenthetical citation
    r"(?!\s*\[[^\]]+\])"   # not followed by a bracket citation
)
_BARE_PERCENT_RE = re.compile(
    r"(?i)\b(?:about|roughly|approximately|around|nearly|close to|an estimated|estimated)?\s*"
    r"\d+(?:\.\d+)?\s*(?:%|percent)\b"
)
_OUTCOME_PROMISE_RE = re.compile(
    r"(?i)\b(?:will improve|will reduce|will increase"
    r"|could help|could assist|may improve|may help|could potentially)\b"
)


def sanitize_output(text: str, flags: dict) -> str:
    """Pre-clean GPT-1 output deterministically before GPT-2 verification."""
    result = text
    # 1. Remove banned evidence phrases (not followed by citation)
    result = _BANNED_EVIDENCE_RE.sub("", result)
    # 2. Convert bare % claims
    result = _BARE_PERCENT_RE.sub(
        "Unknown (Actionable): No authoritative dataset available for this figure", result
    )
    # 3. Strip outcome-promise phrases
    result = _OUTCOME_PROMISE_RE.sub("", result)
    # Clean up residual double-spaces / trailing whitespace per line
    result = re.sub(r"  +", " ", result)
    result = re.sub(r" +\n", "\n", result)
    return result.strip()
