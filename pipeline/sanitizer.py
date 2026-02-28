"""Deterministic prompt router and output sanitizer.

These run before/after LLM calls to reduce nondeterminism.
The sanitizer enforces Audit v7 global rules deterministically.
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

# G1/G2: Banned evidence phrases — vague evidence language without citation
_BANNED_EVIDENCE_RE = re.compile(
    r"(?i)\b(?:studies suggest|research shows|data indicates"
    r"|research indicates|studies show|evidence suggests)\b"
    r"(?!\s*\([^)]+\))"    # not followed by a parenthetical citation
    r"(?!\s*\[[^\]]+\])"   # not followed by a bracket citation
)

# G2: Typicality language used to justify claims (not as hedging qualifiers)
_TYPICALITY_RE = re.compile(
    r"(?i)\b(?:generally|often|typically|commonly|usually)\b"
    r"(?!\s*\([^)]+\))"    # not followed by a parenthetical citation
    r"(?!\s*\[[^\]]+\])"   # not followed by a bracket citation
)

# Bare percent/statistic without citation — fixed: % is non-word, use lookahead
_BARE_PERCENT_RE = re.compile(
    r"(?i)\b(?:about|roughly|approximately|around|nearly|close to|an estimated|estimated)?\s*"
    r"\d+(?:\.\d+)?\s*(?:%(?=\s|$|[,;.\)])|percent\b)"
)

# Outcome-promise phrases (G4 violation markers)
_OUTCOME_PROMISE_RE = re.compile(
    r"(?i)\b(?:will improve|will reduce|will increase"
    r"|could help|could assist|may improve|may help|could potentially)\b"
)


def sanitize_output(text: str, flags: dict) -> str:
    """Pre-clean GPT-1 output deterministically before GPT-2 verification.

    Flag-aware behavior:
    - advice_requested: skip outcome-promise stripping (let GPT-2 handle contextually)
    - legal_mode: apply stricter evidence stripping
    - Always enforce G2 (typicality) and G1 (banned evidence)
    """
    result = text

    # 1. G1: Replace banned evidence phrases with abstention marker
    result = _BANNED_EVIDENCE_RE.sub("[Unverified generalization removed]", result)

    # 2. G2: Replace typicality language with abstention marker
    result = _TYPICALITY_RE.sub("[Typicality language removed]", result)

    # 3. Convert bare % claims to Unknown(Actionable)
    result = _BARE_PERCENT_RE.sub(
        "Unknown(Actionable): No authoritative dataset available for this figure", result
    )

    # 4. Strip outcome-promise phrases — but skip if advice was requested
    #    (let GPT-2 handle contextually for advice-requested prompts)
    if not flags.get("advice_requested"):
        result = _OUTCOME_PROMISE_RE.sub("", result)

    # 5. Legal mode: extra strict — flag any remaining vague legal language
    if flags.get("legal_mode"):
        result = re.sub(
            r"(?i)\b(?:is (?:generally )?legal|is (?:generally )?illegal|is (?:generally )?allowed"
            r"|is (?:generally )?prohibited)\b"
            r"(?!\s*(?:under|per|pursuant to)\s+)",
            "[Legal claim requires citation]",
            result,
        )

    # Clean up residual double-spaces / trailing whitespace per line
    result = re.sub(r"  +", " ", result)
    result = re.sub(r" +\n", "\n", result)
    return result.strip()
