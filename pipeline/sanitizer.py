"""Deterministic prompt router and output sanitizer.

These run before/after LLM calls to reduce nondeterminism.
The sanitizer enforces Audit v7 global rules deterministically.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

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
# US is matched case-sensitively: (?i)\bUS\b also matches the pronoun "us".
_JURISDICTION_RE = re.compile(
    r"(?i:\b(?:UK|EU|federal|state of"
    r"|United States|United Kingdom|Canada|Australia|Germany|France|India"
    r"|China|Japan|Brazil|Mexico|California|Texas|New York|Florida"
    r"|Ohio|Illinois|Pennsylvania|Georgia|Michigan|Virginia"
    r"|North Carolina|South Carolina|Massachusetts|Washington"
    r"|Arizona|Colorado|Oregon|Nevada|Tennessee|Kentucky"
    r"|Alabama|Louisiana|Maryland|Minnesota|Wisconsin|Missouri"
    r"|Connecticut|Iowa|Arkansas|Mississippi|Kansas|Utah"
    r"|Nebraska|Oklahoma|New Mexico|Hawaii|Idaho|Montana"
    r"|Wyoming|Vermont|Maine|New Hampshire|Rhode Island"
    r"|South Dakota|North Dakota|Delaware|West Virginia|Alaska)\b)"
    r"|(?<![A-Za-z])(?:US|U\.S\.A?|USA)(?![A-Za-z])"
)
# Match plausible 4-digit years (1900-2199) for runtime comparisons.
# Narrower than \d{4} to avoid false positives on ZIP codes (90210),
# port numbers (8080), or other non-temporal 4-digit sequences.
_YEAR_RE = re.compile(r"\b((?:19|20|21)\d{2})\b")
# Avoid bare "new"/"now"/"current" — they fire on "new data structure", "help us now",
# and "New York", which then skip GPT-2 entirely when search is unavailable.
_CURRENT_EVENTS_RE = re.compile(
    r"(?i)\b(?:latest|recent|right now|today|this year"
    r"|as of|who is the|what is the current|currently"
    r"|current events?|newest|updated|breaking)\b"
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
    current_year = datetime.now(timezone.utc).date().year
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

# Internal sanitizer markers look like bracket citations; ignore them.
_SANITIZER_MARKER_RE = re.compile(
    r"\[(?:Typicality language removed|Unverified generalization removed|"
    r"Legal claim requires citation|Stale [^\]]*)\]"
)


def _has_nearby_citation(text: str, match_start: int, match_end: int,
                         window: int = 80) -> bool:
    """Check if there's a citation within `window` chars of the match."""
    search_start = max(0, match_start - 20)  # Citations sometimes precede the stat
    search_end = min(len(text), match_end + window)
    context = _SANITIZER_MARKER_RE.sub(" ", text[search_start:search_end])
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

# Outcome-promise replacement mapping for natural sentence flow (G4)
_OUTCOME_PROMISE_PATTERNS = [
    (re.compile(r"(?i)\b(?:could help with|may help with)\b"), "addresses"),
    (re.compile(r"(?i)\b(?:could assist with|could assist in|may assist with|may assist in)\b"), "addresses"),
    (re.compile(r"(?i)\b(?:could help to|may help to|could assist to)\b"), "is intended to"),
    (re.compile(r"(?i)\b(?:will improve|may improve)\b"), "addresses"),
    (re.compile(r"(?i)\b(?:will reduce|will increase)\b"), "addresses"),
    (re.compile(r"(?i)\b(?:could help|could assist|may help|may assist)\b"), "addresses"),
    (re.compile(r"(?i)\bcould potentially\b"), "may"),
]

# Outcome-promise phrases (G4 violation markers)
_OUTCOME_PROMISE_RE = re.compile(
    r"(?i)\b(?:will improve|will reduce|will increase"
    r"|could help|could assist|may improve|may help|could potentially)\b"
)


def _replace_outcome_promises(text: str) -> str:
    """Replace outcome promises with neutral phrasing preserving grammar."""
    result = text
    for pattern, replacement in _OUTCOME_PROMISE_PATTERNS:
        result = pattern.sub(replacement, result)
    return result


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

# Punctuation & grammar cleanup regexes
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,;.:!?])")
_MULTI_DOT_RE = re.compile(r"\.\s*\.+")
_COMMA_DOT_RE = re.compile(r",\s*\.")
_DOT_COMMA_RE = re.compile(r"\.\s*,")
_SEMICOLON_DOT_RE = re.compile(r"[;:]\s*\.")
_DOT_SEMICOLON_RE = re.compile(r"\.\s*[;:]")
_SEMICOLON_COMMA_RE = re.compile(r"[;:]\s*,")
_COMMA_SEMICOLON_RE = re.compile(r",\s*[;:]")
_MULTI_COMMA_RE = re.compile(r",\s*,+")
_MULTI_SEMICOLON_RE = re.compile(r";\s*;+")
_DANGLING_CONNECTOR_RE = re.compile(
    r"\b(?:with|in|to|for|on|by|at|from|about|of|and|or|nor|but|whereas|while|yet)\s*([,;.:!?])",
    re.IGNORECASE,
)
_LEADING_PUNCT_RE = re.compile(r"(?m)^[ \t]*[,;:.]+\s*")
_LEADING_COORDINATOR_RE = re.compile(
    r"(?m)^[ \t]*(?:and|but|or|nor|whereas|while|yet)[,\s;:]+",
    re.IGNORECASE,
)
_SENTENCE_START_COORDINATOR_RE = re.compile(
    r"(?<=[.!?]\s)(?:and|but|or|nor|whereas|while|yet)[,\s;:]+",
    re.IGNORECASE,
)
_FULL_PUNCT_RE = re.compile(r"^[\s,;:.—\-]*$")
_CAP_AFTER_PUNCT_RE = re.compile(r"(?<=[.!?]\s)([a-z])")
_LINE_LEADING_PUNCT_RE = re.compile(r"^[ \t]*[,;:.]+\s*")
_LINE_LEADING_COORD_RE = re.compile(
    r"^[ \t]*(?:and|but|or|nor|whereas|while|yet)[,\s;:]+",
    re.IGNORECASE,
)


def clean_grammar_and_punctuation(text: str) -> str:
    """Normalize punctuation, remove dangling prepositions, strip orphaned coordinators, and clean whitespace."""
    if not text:
        return ""

    # If text contains only punctuation and whitespace, return empty string
    if _FULL_PUNCT_RE.match(text):
        return ""

    # Collapse whitespace first
    if "  " in text:
        text = _MULTI_SPACE_RE.sub(" ", text)

    # Clean space before punctuation
    if _SPACE_BEFORE_PUNCT_RE.search(text):
        text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)

    # Clean duplicate / colliding punctuation
    if ";" in text or ":" in text or "," in text or ".." in text:
        text = _COMMA_DOT_RE.sub(".", text)
        text = _DOT_COMMA_RE.sub(".", text)
        text = _SEMICOLON_DOT_RE.sub(".", text)
        text = _DOT_SEMICOLON_RE.sub(".", text)
        text = _SEMICOLON_COMMA_RE.sub(",", text)
        text = _COMMA_SEMICOLON_RE.sub(",", text)
        text = _MULTI_DOT_RE.sub(".", text)
        text = _MULTI_COMMA_RE.sub(",", text)
        text = _MULTI_SEMICOLON_RE.sub(";", text)

    # Remove dangling prepositions and trailing coordinators before sentence/clause terminators
    if _DANGLING_CONNECTOR_RE.search(text):
        text = _DANGLING_CONNECTOR_RE.sub(r"\1", text)
        if _SPACE_BEFORE_PUNCT_RE.search(text):
            text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
        if _COMMA_DOT_RE.search(text):
            text = _COMMA_DOT_RE.sub(".", text)
        if _SEMICOLON_DOT_RE.search(text):
            text = _SEMICOLON_DOT_RE.sub(".", text)
        if _DOT_SEMICOLON_RE.search(text):
            text = _DOT_SEMICOLON_RE.sub(".", text)
        if _SEMICOLON_COMMA_RE.search(text):
            text = _SEMICOLON_COMMA_RE.sub(",", text)
        if _COMMA_SEMICOLON_RE.search(text):
            text = _COMMA_SEMICOLON_RE.sub(",", text)
        if _MULTI_COMMA_RE.search(text):
            text = _MULTI_COMMA_RE.sub(",", text)
        if _DANGLING_CONNECTOR_RE.search(text):
            text = _DANGLING_CONNECTOR_RE.sub(r"\1", text)
            text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
            if _COMMA_DOT_RE.search(text):
                text = _COMMA_DOT_RE.sub(".", text)
            if _SEMICOLON_DOT_RE.search(text):
                text = _SEMICOLON_DOT_RE.sub(".", text)
            if _DOT_SEMICOLON_RE.search(text):
                text = _DOT_SEMICOLON_RE.sub(".", text)
            if _SEMICOLON_COMMA_RE.search(text):
                text = _SEMICOLON_COMMA_RE.sub(",", text)
            if _COMMA_SEMICOLON_RE.search(text):
                text = _COMMA_SEMICOLON_RE.sub(",", text)

    # Iteratively strip leading punctuation and orphaned coordinators
    while True:
        prev = text
        if _LEADING_PUNCT_RE.search(text):
            text = _LEADING_PUNCT_RE.sub("", text)
        if _LEADING_COORDINATOR_RE.search(text):
            text = _LEADING_COORDINATOR_RE.sub("", text)
        if _SENTENCE_START_COORDINATOR_RE.search(text):
            text = _SENTENCE_START_COORDINATOR_RE.sub("", text)
        if _LEADING_PUNCT_RE.search(text):
            text = _LEADING_PUNCT_RE.sub("", text)
        if text == prev:
            break

    # Fix sentence-internal capitalization after punctuation
    if _CAP_AFTER_PUNCT_RE.search(text):
        text = _CAP_AFTER_PUNCT_RE.sub(
            lambda m: m.group(1).upper(),
            text,
        )

    # Line-by-line whitespace and capitalization cleanup
    if "\n" in text:
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            line_s = line.strip()
            while True:
                prev = line_s
                line_s = _LINE_LEADING_PUNCT_RE.sub("", line_s)
                line_s = _LINE_LEADING_COORD_RE.sub("", line_s)
                if line_s == prev:
                    break
            if line_s and line_s[0].islower() and not line_s.startswith(("http", "www")):
                line_s = line_s[0].upper() + line_s[1:]
            cleaned_lines.append(line_s)
        text = "\n".join(cleaned_lines)
    else:
        line_s = text.strip()
        while True:
            prev = line_s
            line_s = _LINE_LEADING_PUNCT_RE.sub("", line_s)
            line_s = _LINE_LEADING_COORD_RE.sub("", line_s)
            if line_s == prev:
                break
        if line_s and line_s[0].islower() and not line_s.startswith(("http", "www")):
            line_s = line_s[0].upper() + line_s[1:]
        text = line_s

    # Final whitespace cleanup
    if "  " in text:
        text = _MULTI_SPACE_RE.sub(" ", text)
    if " \n" in text:
        text = _TRAILING_SPACE_RE.sub("\n", text)
    return text.strip()


_clean_grammar_and_punctuation = clean_grammar_and_punctuation

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

    # ---- G3: Bare stats (citation-aware) ----
    # Run before G1/G2 so sanitizer bracket-markers cannot look like citations.
    # strict: always strip bare stats without nearby citations
    # standard: strip only when percent_requested
    # light: skip entirely
    if tier == "strict" or tier == "standard" and flags.get("percent_requested"):
        result = _replace_bare_percents(result)

    # ---- G1 + G2: Banned evidence & typicality (strict and standard only) ----
    if tier in ("strict", "standard"):
        result = _BANNED_EVIDENCE_RE.sub("[Unverified generalization removed]", result)
        result = _TYPICALITY_RE.sub("[Typicality language removed]", result)

    # ---- G4: Outcome promises (strict and standard, skipped in light) ----
    if tier in ("strict", "standard") and not flags.get("advice_requested"):
        result = _replace_outcome_promises(result)

    # ---- G6: Stale dates (strict and standard only) ----
    if tier in ("strict", "standard") and flags.get("current_events"):
        current_year = datetime.now(timezone.utc).date().year

        def _replace_stale(match: re.Match) -> str:
            if int(match.group(1)) < current_year:
                return "[Stale — verify current status from an authoritative source]"
            return match.group(0)

        result = _STALE_DATE_BASE_RE.sub(_replace_stale, result)

    # ---- Legal mode extra (strict and standard only) ----
    if tier in ("strict", "standard") and flags.get("legal_mode"):
        result = _VAGUE_LEGAL_RE.sub("[Legal claim requires citation]", result)

    # Clean up residual grammar, double-spaces, and punctuation (always)
    return _clean_grammar_and_punctuation(result)
