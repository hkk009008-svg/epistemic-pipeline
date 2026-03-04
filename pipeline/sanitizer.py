"""Deterministic prompt router and output sanitizer.

These run before/after LLM calls to reduce nondeterminism.
The sanitizer enforces Audit v7 global rules deterministically.
"""
from __future__ import annotations

import re
from datetime import date

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
# Match any 4-digit year (2000+) for runtime comparisons
_YEAR_RE = re.compile(r"\b(\d{4})\b")
_CURRENT_EVENTS_RE = re.compile(
    r"(?i)\b(?:current|latest|recent|right now|today|now|this year"
    r"|as of|who is the|what is the current|new|newest|updated)\b"
)
_COMPARATIVE_RE = re.compile(
    r"(?i)(?:"
    r"\b(?:safer|better|worse|more effective|less effective|superior|inferior"
    r"|as effective as|as reliable as|as good as|more harm|fewer side effects"
    r"|worth the|reduce or increase|cause more|versus|vs\.?)\b"
    r"|(?:which\s+(?:is|are|has|have)\s+(?:the\s+)?(?:best|worst|safest|most|least|fewer))"
    r"|(?:is\s+\w+\s+(?:or)\s+\w+\s+(?:safer|better|worse|more|less))"
    r")"
)


def route_prompt(prompt: str) -> dict:
    """Deterministic heuristic router -- classifies prompt features for downstream use."""
    current_year = date.today().year
    future_year = any(
        int(m.group(1)) > current_year for m in _YEAR_RE.finditer(prompt)
    )
    return {
        "advice_requested": bool(_ADVICE_RE.search(prompt)),
        "percent_requested": bool(_PERCENT_RE.search(prompt)),
        "legal_mode": bool(_LEGAL_RE.search(prompt)),
        "jurisdiction_present": bool(_JURISDICTION_RE.search(prompt)),
        "future_year": future_year,
        "current_events": bool(_CURRENT_EVENTS_RE.search(prompt)),
        "comparative": bool(_COMPARATIVE_RE.search(prompt)),
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

# Citation detection for nearby-citation checks
_CITATION_RE = re.compile(
    r"(?:\([^)]{3,50}\)|\[[^\]]{1,50}\])"  # (Source 2024) or [CDC 2023] etc.
)


def _has_nearby_citation(text: str, match_start: int, match_end: int,
                         window: int = 80) -> bool:
    """Check if there's a citation within `window` chars of the match."""
    search_start = max(0, match_start - 20)  # Citations sometimes precede the stat
    search_end = min(len(text), match_end + window)
    context = text[search_start:search_end]
    return bool(_CITATION_RE.search(context))


def _replace_bare_percents(text: str) -> str:
    """Replace bare percentages that lack nearby citations."""
    matches = list(_BARE_PERCENT_RE.finditer(text))
    if not matches:
        return text
    # Work backwards to preserve indices
    result = text
    for match in reversed(matches):
        if not _has_nearby_citation(text, match.start(), match.end()):
            result = (result[:match.start()] +
                      "Unknown(Actionable): No authoritative dataset available for this figure" +
                      result[match.end():])
    return result

# Outcome-promise phrases (G4 violation markers)
_OUTCOME_PROMISE_RE = re.compile(
    r"(?i)\b(?:will improve|will reduce|will increase"
    r"|could help|could assist|may improve|may help|could potentially)\b"
)

# Stale date qualifier — "as of [Month] [Year]" where year is before current year
# Used when current_events flag is set to catch stale time-sensitive claims.
# Captures the year for runtime comparison instead of fragile regex math.
_STALE_DATE_BASE_RE = re.compile(
    r"(?i)\bas of\s+(?:January|February|March|April|May|June|July|August"
    r"|September|October|November|December)?\s*(20\d{2})\b"
)

# Pre-compiled whitespace cleanup patterns (avoid re.sub recompilation per call)
_MULTI_SPACE_RE = re.compile(r"  +")
_TRAILING_SPACE_RE = re.compile(r" +\n")

# G1: Vague legal claims without statute/regulation citation (pre-compiled)
_VAGUE_LEGAL_RE = re.compile(
    r"(?i)\b(?:is (?:generally )?legal|is (?:generally )?illegal|is (?:generally )?allowed"
    r"|is (?:generally )?prohibited)\b"
    r"(?!\s*(?:under|per|pursuant to)\s+)"
)


def sanitize_output(text: str, flags: dict, tier: str = "strict") -> str:
    """Pre-clean GPT-1 output deterministically before GPT-2 verification.

    Tier-gated behavior:
    - strict: all rules applied (current behavior)
    - standard: G1/G2 applied; G3 only when percent_requested; G4/G6/legal as strict
    - light: all sanitizer rules skipped (only whitespace cleanup)

    Flag-aware behavior (strict and standard tiers):
    - advice_requested: skip outcome-promise stripping (let GPT-2 handle contextually)
    - legal_mode: apply stricter evidence stripping
    - current_events: flag stale date-qualified claims
    """
    result = text

    # ---- G1 + G2: Banned evidence & typicality (strict and standard only) ----
    if tier in ("strict", "standard"):
        result = _BANNED_EVIDENCE_RE.sub("[Unverified generalization removed]", result)
        result = _TYPICALITY_RE.sub("[Typicality language removed]", result)

    # ---- G3: Bare stats (citation-aware) ----
    # strict: always strip bare stats without nearby citations
    # standard: strip only when percent_requested
    # light: skip entirely
    if tier == "strict":
        result = _replace_bare_percents(result)
    elif tier == "standard" and flags.get("percent_requested"):
        result = _replace_bare_percents(result)

    # ---- G4: Outcome promises (strict and standard, skipped in light) ----
    if tier in ("strict", "standard"):
        if not flags.get("advice_requested"):
            result = _OUTCOME_PROMISE_RE.sub("", result)

    # ---- G6: Stale dates (strict and standard only) ----
    if tier in ("strict", "standard"):
        if flags.get("current_events"):
            current_year = date.today().year

            def _replace_stale(match: re.Match) -> str:
                if int(match.group(1)) < current_year:
                    return "[Stale — verify current status from an authoritative source]"
                return match.group(0)

            result = _STALE_DATE_BASE_RE.sub(_replace_stale, result)

    # ---- Legal mode extra (strict and standard only) ----
    if tier in ("strict", "standard"):
        if flags.get("legal_mode"):
            result = _VAGUE_LEGAL_RE.sub("[Legal claim requires citation]", result)

    # Clean up residual double-spaces / trailing whitespace per line (always)
    result = _MULTI_SPACE_RE.sub(" ", result)
    result = _TRAILING_SPACE_RE.sub("\n", result)
    return result.strip()
