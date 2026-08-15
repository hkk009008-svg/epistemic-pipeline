"""Deterministic post-processor: fix GPT-2 claim categories and findings using search sources.

GPT-2 (gpt-4o-mini) frequently ignores content-match rules and marks
claims as "Unsupported" even when search source snippets clearly support
them.  It also flags T1/T7 on source-backed claims.  This module provides
a fast, deterministic fix that runs after parse_gpt2.

Two corrections:
  1. recategorize_with_sources — Unsupported → Observed for source-matched claims
  2. filter_findings_with_sources — remove T1/T7 findings on source-backed content

After both, the caller recomputes the verdict from the corrected findings.

Evidence strength: keyword overlap alone is treated as *weak evidence*.
When NLI scores are available (via enriched atomic claims), entailment
scores are used for stronger upgrades. The justification field always
records the match method ("keyword_overlap" vs "nli_entailment") so
downstream consumers can distinguish.
"""
from __future__ import annotations

import functools
import re
import time
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pipeline.models import ClaimEntry, SearchSource
from pipeline.sanitizer import clean_grammar_and_punctuation


class ClauseType(str, Enum):
    """Syntactic clause taxonomy for proposition AST decomposition."""
    INDEPENDENT = "independent"
    CONCESSIVE = "concessive"
    CONDITIONAL = "conditional"
    TEMPORAL = "temporal"
    RELATIVE = "relative"
    COORDINATE = "coordinate"
    PARTICIPIAL = "participial"


@dataclass
class PropositionSpan:
    """Discrete proposition AST node extracted from complex sentence structure."""
    span_id: str
    clause_type: ClauseType
    raw_text: str
    cleaned_text: str
    start_char: int
    end_char: int
    subordinator: str | None = None
    citation_indices: list[int] = field(default_factory=list)
    parent_span_id: str | None = None
    is_matrix: bool = False
    nesting_level: int = 1


class PreflightResult(tuple):
    """Result from pre-flight scan: behaves as 2-tuple (has_hard, findings) and supports dict-like access."""

    def __new__(cls, has_hard: bool, findings: list[dict], latency_ms: float = 0.0):
        obj = super().__new__(cls, (has_hard, findings))
        obj.has_hard_preflight = has_hard
        obj.findings = findings
        obj.preflight_latency_ms = latency_ms
        return obj

    def get(self, key: str, default: Any = None) -> Any:
        if key == "has_hard_preflight":
            return self.has_hard_preflight
        elif key == "findings":
            return self.findings
        elif key == "preflight_latency_ms":
            return self.preflight_latency_ms
        return default

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, str):
            if item == "has_hard_preflight":
                return self.has_hard_preflight
            elif item == "findings":
                return self.findings
            elif item == "preflight_latency_ms":
                return self.preflight_latency_ms
            raise KeyError(item)
        return super().__getitem__(item)

    def __contains__(self, key: Any) -> bool:
        if key in ("has_hard_preflight", "findings", "preflight_latency_ms"):
            return True
        return super().__contains__(key)

    def keys(self):
        return ("has_hard_preflight", "findings", "preflight_latency_ms")

    def values(self):
        return (self.has_hard_preflight, self.findings, self.preflight_latency_ms)

    def items(self):
        return (
            ("has_hard_preflight", self.has_hard_preflight),
            ("findings", self.findings),
            ("preflight_latency_ms", self.preflight_latency_ms),
        )


# Zero-width, directional formatting, invisible characters, and ASCII control chars for pre-flight de-obfuscation
_ZERO_WIDTH_CHARS = (
    "\u200b\u200c\u200d\u200e\u200f\ufeff\u00ad\u2060\u180e"
    "\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"
)
_CONTROL_AND_ZERO_WIDTH_CODEPOINTS = {
    *range(9),  # \x00 - \x08
    11, 12,        # \x0b, \x0c
    *range(14, 27), # \x0e - \x1a (exclude 27 \x1b so ANSI regex can process escape codes)
    *range(28, 32), # \x1c - \x1f
    127,           # DEL
    *[ord(c) for c in _ZERO_WIDTH_CHARS],
}
_TRANSLATE_TABLE = {cp: None for cp in _CONTROL_AND_ZERO_WIDTH_CODEPOINTS}
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def normalize_preflight_text(text: str) -> str:
    """Normalize text by stripping ANSI escapes, zero-width chars, control chars, and applying NFKC decomposition.

    Execution latency: ~0.4 µs on ASCII, ~2-5 µs on Unicode payload.
    """
    if not text:
        return ""
    t = text
    # Guarded ANSI escape sequence removal must run before translate table
    if "\x1b" in t:
        t = _ANSI_ESCAPE_RE.sub("", t)
    # Unicode NFKC normalization (folds full-width chars, ligatures)
    if not t.isascii():
        t = unicodedata.normalize("NFKC", t)
    return t.translate(_TRANSLATE_TABLE)




# Alias for backwards compatibility
normalize_text_for_scan = normalize_preflight_text

# Words too common to be meaningful signal
_STOP_WORDS = frozenset(
    ["a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "shall", "should", "may", "might", "can", "could", "of", "in", "on", "at", "to", "for", "with", "by", "from", "as", "into", "through", "during", "before", "after", "above", "below", "between", "out", "off", "over", "under", "again", "further", "then", "once", "that", "this", "these", "those", "and", "but", "or", "nor", "not", "no", "so", "yet", "both", "each", "every", "all", "any", "few", "more", "most", "other", "some", "such", "than", "too", "very", "also", "just", "about", "its", "it", "their", "them", "they", "he", "she", "his", "her", "which", "who", "whom", "what", "where", "when", "how", "if", "because", "while", "although", "since", "until", "unless", "even", "still", "already", "often", "really", "only", "however", "known", "called", "referred", "many", "well", "much", "like", "including", "based"]
)

# Minimum number of meaningful keywords a claim must have to attempt matching
_MIN_KEYWORDS = 2

# Minimum fraction of claim keywords found in a source to count as a keyword match
_MATCH_THRESHOLD = 0.5

# Stricter threshold for keyword-only upgrades (no NLI)
# Requires higher overlap when entailment can't be verified
_STRICT_MATCH_THRESHOLD = 0.65

# NLI entailment threshold for source-match upgrades
_NLI_ENTAILMENT_THRESHOLD = 0.5

# Finding types that should be removed when their target is source-backed
_SOURCE_OVERRIDABLE_TYPES = {
    "T1", "T7",
    "Fabricated statistic", "Fabricated citation", "Evidence instantiation",
    "Unverified current fact",
}

# Number word lookup for quantitative parsing
_WORD_NUMS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fourty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "thousand": 1_000, "million": 1_000_000,
    "billion": 1_000_000_000, "trillion": 1_000_000_000_000,
}

_SCALE_WORDS = {"k", "thousand", "m", "million", "b", "billion", "t", "trillion"}

_SCALE_FACTORS = {
    "k": 1_000, "thousand": 1_000,
    "m": 1_000_000, "million": 1_000_000,
    "b": 1_000_000_000, "billion": 1_000_000_000,
    "t": 1_000_000_000_000, "trillion": 1_000_000_000_000,
}

# Quantifier context pattern to distinguish "one dollar/one day" from non-quantitative "one of the"
_UNIT_ONE_RE = re.compile(
    r"\bone\s+(?:percent|pct|dollar|dollars|euro|euros|pound|pounds|yen|day|days|year|years|"
    r"month|months|week|weeks|hour|hours|minute|minutes|second|seconds|degree|degrees|"
    r"kg|km|meter|meters|mile|miles|lb|lbs|gb|mb|tb|k|m|b|thousand|million|billion|trillion)\b",
    re.IGNORECASE,
)

_BRACKET_RE = re.compile(r"\[[^\]]*\]")
_KEYWORD_RE = re.compile(r"[a-zA-Z0-9]+(?:'[a-z]+)?")
_DIGIT_OR_CURRENCY_RE = re.compile(r"[\d\$€£¥₹₩]")
_WORD_NUM_RE = re.compile(
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
    r"fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fourty|fifty|sixty|"
    r"seventy|eighty|ninety|hundred|thousand|million|billion|trillion)\b",
    re.IGNORECASE,
)
_SCALE_RE = re.compile(
    r"(?:[\$€£¥₹₩]\s*)?(\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)\s*(k|m|b|t|thousand|million|billion|trillion)\b",
    re.IGNORECASE,
)
_NUM_RE = re.compile(
    r"(?:[\$€£¥₹₩]\s*)?(\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)(?:\s*(?:%|percent|pct))?",
    re.IGNORECASE,
)
_WORD_TOKENS_RE = re.compile(r"[a-zA-Z]+(?:-[a-zA-Z]+)?")
_CITATION_PATTERN = re.compile(r"\[([\d\s,\-]+)\]")


_CITATION_SPLIT_RE = re.compile(r"[,;]+")


def _parse_citation_group(content: str) -> list[int]:
    """Extract all citation integers from bracketed content like '1, 2' or '1-3'."""
    content_s = content.strip()
    if content_s.isdigit():
        return [int(content_s)]
    indices: list[int] = []
    parts = _CITATION_SPLIT_RE.split(content_s)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            sub = part.split("-")
            if len(sub) == 2 and sub[0].strip().isdigit() and sub[1].strip().isdigit():
                s_num, e_num = int(sub[0].strip()), int(sub[1].strip())
                if 0 < s_num <= e_num <= 100:
                    indices.extend(range(s_num, e_num + 1))
                    continue
        if part.isdigit():
            indices.append(int(part))
    return indices


_parse_citation_indices = _parse_citation_group
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?\n])\s+")
_CLAUSE_SPLIT_RE = re.compile(
    r"(?:[,;]|\b(?:whereas|while|but|although|even though|though|despite|because|since|unless)\b)",
    re.IGNORECASE,
)
_INJECTION_PATTERN = re.compile(
    r"(?i)(?:"
    r"(?:[\"']?(?:gpt2_verdict|verdict)[\"']?\s*:\s*[\"']?PASS[\"']?)|"
    r"[\"']arbiter_decision[\"']\s*:\s*[\"']ALLOW[\"']|"
    r"</?(?:system[_-](?:eval[_-]bypass|override|instruction|prompt)|untrusted[_-]override|instruction|verifier|system|instructions|prompt)\b[^>]*>|"
    r"<!\[CDATA\[|"
    r"===\s*(?:TASK|TRIPWIRE\s*REFERENCE|END\s*TRIPWIRE\s*REFERENCE|EVALUATION|SYSTEM|INSTRUCTIONS|INPUT|OUTPUT)\s*===|"
    r"###\s*SYSTEM(?:\s*INSTRUCTION)?(?:\s*[:#]|\b)|"
    r"##\s*Output\s*Schema(?:\s*Override)?\b|"
    r"---\s*(?:INSTRUCTIONS|SYSTEM|RULES)\s*---|[*]{3}\s*(?:INSTRUCTIONS|SYSTEM|RULES)\s*[*]{3}|"
    r"\b(?:"
        r"A(?:DMIN(?:\s*(?:OVERRIDE|INSTRUCTION|DIRECTIVE|COMMAND|PROTOCOL)\b|_MODE\b)|LLOW_PROMPT_OVERRIDE\b|RBITER_DECISION\s*[:=]\s*[\"']?\s*ALLOW\b)|"
        r"C(?:LEAR\s*(?:CONTEXT|CONSTRAINTS|SYSTEM\s*PROMPT)\b)|"
        r"D(?:EVELOPER\s*MODE(?:\s*[:=]|\s*(?:ACTIVE|ENABLED|ON|TRUE|ALLOW|OVERRIDE|BYPASS)\b)|ISREGARD\s*(?:(?:ALL\s*)?(?:PREVIOUS\s*)?(?:INSTRUCTIONS|RULES|CONSTRAINTS|POLICIES|SAFETY)|SAFETY)\b)|"
        r"E(?:MERGENCY\s*ADMINISTRATIVE\s*PROTOCOL\b)|"
        r"G(?:PT-1\s*RESPONSE\s*TO\s*VERIFY\s*:)|"
        r"I(?:GNORE\s*(?:(?:ALL\s*)?(?:PREVIOUS\s*)?(?:INSTRUCTIONS|RULES|CONSTRAINTS|POLICIES|SAFETY)|SAFETY)\b)|"
        r"N(?:LI\s*PRE-VERIFICATION\s*SIGNALS\s*:)|"
        r"O(?:RIGINAL\s*PROMPT\s*:|UTPUT\s*VERDICT\s*:\s*PASS\b|VERRIDE\s*:\s*VERDICT\s*[:=]\s*[\"']?\s*PASS\b)|"
        r"P(?:ROMPT\s*INJECTION\b|RE-DECOMPOSED\s*ATOMIC\s*CLAIMS\s*:)|"
        r"R(?:OOT\s*(?:OVERRIDE|INSTRUCTION|DIRECTIVE|COMMAND|PROTOCOL)\b|ESET\s*(?:CONTEXT|CONSTRAINTS|SYSTEM\s*PROMPT)\b)|"
        r"S(?:YSTEM\s*(?:OVERRIDE|INSTRUCTION|DIRECTIVE|COMMAND|PROTOCOL)\b|UPERUSER\s*(?:OVERRIDE|INSTRUCTION|DIRECTIVE|COMMAND|PROTOCOL)\b)|"
        r"V(?:ERDICT\s*[:=]\s*[\"']?\s*PASS\b)"
    r")"
    r")"
)

_CODEBLOCK_INJECTION_PATTERN = re.compile(
    r"(?i)```(?:json\w*)?\s*\{[\s\S]*?(?:[\"']verdict[\"']\s*:\s*[\"']PASS[\"']|[\"']arbiter_decision[\"']\s*:\s*[\"']ALLOW[\"']|[\"']reasoning_trace[\"']\s*:\s*\[[\s\S]*?(?:override|override\s+active))[\s\S]*?\}"
)


def scan_prompt_injection(text: str) -> tuple[bool, str | None]:
    """Deterministic regex and token scan for prompt injections, system overrides, and delimiter breakouts.

    Returns:
        (is_injected: bool, snippet_or_none: str | None)
    """
    if not text or not text.strip():
        return False, None
    normalized = normalize_preflight_text(text)
    match = _INJECTION_PATTERN.search(normalized)
    if match:
        snippet = match.group(0).strip().replace("\n", " ")[:60]
        return True, snippet
    if "```" in normalized:
        match_cb = _CODEBLOCK_INJECTION_PATTERN.search(normalized)
        if match_cb:
            snippet = match_cb.group(0).strip().replace("\n", " ")[:60]
            return True, snippet
    return False, None


@functools.lru_cache(maxsize=2048)
def _extract_keywords(text: str) -> frozenset[str]:
    """Extract meaningful lowercase keywords from text, filtering stop words."""
    norm_text = normalize_preflight_text(text)
    words = _KEYWORD_RE.findall(norm_text.lower())
    return frozenset(w for w in words if w not in _STOP_WORDS and len(w) > 1)


@functools.lru_cache(maxsize=2048)
def _extract_numbers(text: str) -> frozenset[float]:
    """Extract all numeric values (canonical floats) from text, ignoring citation markers."""
    # Strip citation markers and bracket annotations (e.g. [1], [2], [verified])
    clean = normalize_preflight_text(_BRACKET_RE.sub(" ", text))

    nums: set[float] = set()

    # 1 & 2: Process digits, currency amounts, and percentages if digits/currencies exist
    if _DIGIT_OR_CURRENCY_RE.search(clean):
        for m in _SCALE_RE.finditer(clean):
            raw_num = m.group(1).replace(",", "")
            scale_word = m.group(2).lower()
            try:
                val = float(raw_num)
                factor = _SCALE_FACTORS.get(scale_word, 1)
                nums.add(round(val * factor, 6))
                nums.add(round(val, 6))
            except ValueError:
                pass

        for m in _NUM_RE.finditer(clean):
            raw_num = m.group(1).replace(",", "")
            try:
                val = float(raw_num)
                nums.add(round(val, 6))
            except ValueError:
                pass

    # 3. Word numbers (e.g. fifty, thirty, seven, twenty-five)
    if "one" in clean.lower() and _UNIT_ONE_RE.search(clean):
        nums.add(1.0)

    # Fast check: if no word-number keywords exist in clean text, skip token-level parsing
    if not _WORD_NUM_RE.search(clean):
        return frozenset(nums)

    unscaled_clean = _SCALE_RE.sub(" ", clean)
    word_tokens = _WORD_TOKENS_RE.findall(unscaled_clean.lower())

    tens_dict = {
        "twenty": 20, "thirty": 30, "forty": 40, "fourty": 40,
        "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    }
    ones_dict = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9,
    }
    scale_dict = {
        "hundred": 100, "thousand": 1_000, "million": 1_000_000,
        "billion": 1_000_000_000, "trillion": 1_000_000_000_000,
    }

    i = 0
    while i < len(word_tokens):
        token = word_tokens[i]
        if "-" in token:
            parts = token.split("-")
            if len(parts) == 2 and parts[0] in _WORD_NUMS and parts[1] in _WORD_NUMS:
                nums.add(float(_WORD_NUMS[parts[0]] + _WORD_NUMS[parts[1]]))
            i += 1
            continue

        if token in tens_dict and i + 1 < len(word_tokens) and word_tokens[i + 1] in ones_dict:
            nums.add(float(tens_dict[token] + ones_dict[word_tokens[i + 1]]))
            i += 2
            continue

        if token in _WORD_NUMS and token != "one" and i + 1 < len(word_tokens) and word_tokens[i + 1] in scale_dict:
            base_val = _WORD_NUMS[token]
            scale_val = scale_dict[word_tokens[i + 1]]
            nums.add(float(base_val * scale_val))
            nums.add(float(base_val))
            i += 2
            continue

        if token in _WORD_NUMS and token != "one":
            nums.add(float(_WORD_NUMS[token]))

        i += 1

    return frozenset(nums)


def _has_nli_scores(nli_claims: list[dict] | None) -> bool:
    """True only when at least one claim actually carries an NLI result."""
    if not nli_claims:
        return False
    return any(isinstance(c, dict) and c.get("nli_result") for c in nli_claims)


def _keyword_threshold(nli_claims: list[dict] | None) -> float:
    """Use the strict overlap cutoff only when NLI scores are present."""
    return _STRICT_MATCH_THRESHOLD if _has_nli_scores(nli_claims) else _MATCH_THRESHOLD


def _is_source_backed(
    text: str,
    source_keyword_sets: list[set[str]],
    threshold: float = _MATCH_THRESHOLD,
) -> bool:
    """Check if text content is supported by any source snippet."""
    kw = _extract_keywords(text)
    if len(kw) < _MIN_KEYWORDS:
        return False
    for src_kw in source_keyword_sets:
        overlap = kw & src_kw
        if len(overlap) / len(kw) >= threshold:
            return True
    return False


def build_source_keyword_sets(sources: list[SearchSource]) -> list[set[str]]:
    """Pre-compute keyword sets from sources (call once, pass to both functions).

    When using both recategorize_with_sources and filter_findings_with_sources,
    call this once and pass the result to avoid redundant keyword extraction.
    """
    return [set(_extract_keywords(f"{s.title} {s.snippet}")) for s in sources]


def build_source_number_sets(sources: list[SearchSource]) -> list[set[float]]:
    """Pre-compute numeric value sets from sources."""
    return [set(_extract_numbers(f"{s.title} {s.snippet}")) for s in sources]


def _find_nli_support(claim_text: str, nli_claims: list[dict]) -> dict | None:
    """Find NLI results for a claim by matching text.

    Returns the nli_result dict if the claim has strong entailment support,
    otherwise None.
    """
    if not nli_claims:
        return None
    claim_lower = claim_text.lower().strip()
    for ac in nli_claims:
        ac_text = ac.get("text", "").lower().strip()
        nli = ac.get("nli_result")
        if not nli:
            continue
        # Match by text similarity (exact or substring)
        if (
            (ac_text == claim_lower or ac_text in claim_lower or claim_lower in ac_text)
            and nli.get("best_entailment", 0.0) >= _NLI_ENTAILMENT_THRESHOLD
        ):
            return nli
    return None


def recategorize_with_sources(
    claim_table: list[ClaimEntry],
    sources: list[SearchSource],
    source_keyword_sets: list[set[str]] | None = None,
    nli_claims: list[dict] | None = None,
) -> list[ClaimEntry]:
    """Re-categorize Unsupported claims that are supported by search source snippets.

    Returns a new list of ClaimEntry objects (leaves originals unchanged).
    Only upgrades "Unsupported" → "Observed".  Never downgrades.

    When *nli_claims* are provided (from decompose + NLI verification),
    NLI entailment scores are used for stronger evidence.  Keyword-only
    matches use a stricter threshold and are labeled as such.

    Pass pre-computed *source_keyword_sets* to avoid redundant extraction
    when calling both recategorize_with_sources and filter_findings_with_sources.
    """
    if not sources or not claim_table:
        return claim_table

    if source_keyword_sets is None:
        source_keyword_sets = build_source_keyword_sets(sources)

    result = []
    for entry in claim_table:
        if entry.category.lower().strip() != "unsupported":
            result.append(entry)
            continue

        claim_kw = _extract_keywords(entry.claim)
        if len(claim_kw) < _MIN_KEYWORDS:
            result.append(entry)
            continue

        # Check NLI entailment first (stronger evidence)
        nli_match = _find_nli_support(entry.claim, nli_claims) if nli_claims else None
        if nli_match:
            best_src_idx = nli_match.get("best_source_idx", 0)
            ent_score = nli_match.get("best_entailment", 0.0)
            src_title = sources[best_src_idx].title if 0 <= best_src_idx < len(sources) else "evidence"
            result.append(ClaimEntry(
                claim=entry.claim,
                category="Observed",
                justification=(
                    f"NLI-verified (entailment={ent_score:.2f}) against source "
                    f"[{best_src_idx + 1}]: {src_title} [match_method=nli_entailment]"
                ),
            ))
            continue

        # Fallback: keyword overlap with stricter threshold
        best_overlap = 0.0
        best_source_idx = -1
        for i, src_kw in enumerate(source_keyword_sets):
            overlap = claim_kw & src_kw
            ratio = len(overlap) / len(claim_kw)
            if ratio > best_overlap:
                best_overlap = ratio
                best_source_idx = i

        # Use stricter threshold for keyword-only matches when NLI actually ran
        threshold = _keyword_threshold(nli_claims)
        if best_overlap >= threshold:
            src = sources[best_source_idx]
            result.append(ClaimEntry(
                claim=entry.claim,
                category="Observed",
                justification=(
                    f"Content-matched to source [{best_source_idx + 1}]: {src.title} "
                    f"(overlap={best_overlap:.0%}) [match_method=keyword_overlap]"
                ),
            ))
        else:
            result.append(entry)

    return result


def filter_findings_with_sources(
    findings: list[dict],
    sources: list[SearchSource],
    source_keyword_sets: list[set[str]] | None = None,
    nli_claims: list[dict] | None = None,
) -> list[dict]:
    """Remove T1/T7 findings whose detail text is supported by search sources.

    Returns a new list of findings (leaves originals unchanged).
    Only removes findings of overridable types (T1, T7, etc.).

    Uses the same keyword threshold as recategorize_with_sources so a finding
    is not dropped while its claim stays Unsupported.

    Pass pre-computed *source_keyword_sets* to avoid redundant extraction
    when calling both recategorize_with_sources and filter_findings_with_sources.
    """
    if not sources or not findings:
        return findings

    if source_keyword_sets is None:
        source_keyword_sets = build_source_keyword_sets(sources)

    threshold = _keyword_threshold(nli_claims)

    result = []
    for f in findings:
        ftype = f.get("type", "")
        detail = f.get("detail", "")

        # Only filter overridable finding types
        if ftype not in _SOURCE_OVERRIDABLE_TYPES:
            result.append(f)
            continue

        # Check if the finding's detail text matches source content
        if _is_source_backed(detail, source_keyword_sets, threshold):
            continue  # Drop this finding — it's about source-backed content

        result.append(f)

    return result


def _get_citation_segments(sent: str) -> list[tuple[list[int], str]]:
    """Segment a sentence into spans associated with their respective citation markers."""
    matches = list(_CITATION_PATTERN.finditer(sent))
    if not matches:
        return []
    first_indices = _parse_citation_group(matches[0].group(1))
    if len(matches) == 1:
        return [(first_indices, sent)]

    # Group adjacent citation markers (e.g. [1][2] or [1], [2])
    grouped_citations: list[tuple[list[int], int, int]] = []
    cur_indices = list(first_indices)
    cur_start = matches[0].start()
    cur_end = matches[0].end()

    for i in range(1, len(matches)):
        m = matches[i]
        intervening = sent[cur_end:m.start()].strip(" ,;")
        m_indices = _parse_citation_group(m.group(1))
        if len(intervening) == 0:
            cur_indices.extend(m_indices)
            cur_end = m.end()
        else:
            grouped_citations.append((cur_indices, cur_start, cur_end))
            cur_indices = list(m_indices)
            cur_start = m.start()
            cur_end = m.end()
    grouped_citations.append((cur_indices, cur_start, cur_end))

    if len(grouped_citations) == 1:
        return [(grouped_citations[0][0], sent)]

    # Segment sentence into spans per citation group
    groups: list[tuple[list[int], str]] = []
    span_start = 0

    for i in range(len(grouped_citations) - 1):
        indices, g_start, g_end = grouped_citations[i]
        _next_indices, next_start, _next_end = grouped_citations[i + 1]

        text_before = sent[span_start:g_start].strip()
        intervening = sent[g_end:next_start]

        split_match = _CLAUSE_SPLIT_RE.search(intervening)

        if split_match:
            split_point = g_end + split_match.start()
            if text_before or _extract_numbers(sent[span_start:g_end]):
                split_point = max(g_end, g_end + split_match.start())
            else:
                split_point = g_end + split_match.start()
            groups.append((indices, sent[span_start:split_point]))
            span_start = split_point
        else:
            groups.append((indices, sent[span_start:g_end]))
            span_start = g_end

    last_indices = grouped_citations[-1][0]
    groups.append((last_indices, sent[span_start:]))
    return groups


def verify_citation_grounding(
    text: str,
    sources: list[SearchSource] | None,
    source_keyword_sets: list[set[str]] | None = None,
    source_number_sets: list[set[float]] | None = None,
) -> list[dict]:
    """Deterministically check that all [N] citation markers refer to valid and relevant sources.

    Returns a list of findings for any:
    - Out-of-range citations (e.g. [5] when only 2 sources exist, or citing [1] when 0 sources exist)
    - Unbacked citations (where the sentence referencing [N] has no keyword overlap with source N)
    - Unbacked numeric claims (where numbers/percentages/currencies in sentence do not exist in cited source)
    """
    if not text or not text.strip():
        return []

    findings: list[dict] = []

    # Check for adversarial prompt injection or delimiter breakout markers
    injected, snippet = scan_prompt_injection(text)
    if injected:
        findings.append({
            "type": "T1",
            "severity": "hard",
            "detail": f"Adversarial prompt injection or directive override token detected in draft: '{snippet}'.",
            "target": "draft",
        })

    if "[" not in text:
        return findings

    # Defensive initialization of sources and on-demand source set lookup
    sources_list = sources or []
    num_sources = len(sources_list)

    kw_cache: dict[int, frozenset[str]] = {}
    num_cache: dict[int, frozenset[float]] = {}

    def _get_src_keywords(idx_1based: int) -> frozenset[str] | set[str]:
        if source_keyword_sets is not None and 1 <= idx_1based <= len(source_keyword_sets):
            return source_keyword_sets[idx_1based - 1]
        if idx_1based not in kw_cache and 1 <= idx_1based <= num_sources:
            src = sources_list[idx_1based - 1]
            kw_cache[idx_1based] = _extract_keywords(f"{src.title} {src.snippet}")
        return kw_cache.get(idx_1based, frozenset())

    def _get_src_numbers(idx_1based: int) -> frozenset[float] | set[float]:
        if source_number_sets is not None and 1 <= idx_1based <= len(source_number_sets):
            return source_number_sets[idx_1based - 1]
        if idx_1based not in num_cache and 1 <= idx_1based <= num_sources:
            src = sources_list[idx_1based - 1]
            num_cache[idx_1based] = _extract_numbers(f"{src.title} {src.snippet}")
        return num_cache.get(idx_1based, frozenset())

    # Break text into sentence segments
    sentences = _SENTENCE_SPLIT_RE.split(text)
    seen_violations: set[str] = set()

    for sent in sentences:
        if "[" not in sent:
            continue

        matches = list(_CITATION_PATTERN.finditer(sent))
        if not matches:
            continue

        # 1. Out-of-range check
        for m in matches:
            idxs = _parse_citation_group(m.group(1))
            for idx in idxs:
                if num_sources == 0:
                    key = f"out_of_range_{idx}"
                    if key not in seen_violations:
                        seen_violations.add(key)
                        findings.append({
                            "type": "T1",
                            "severity": "hard",
                            "detail": f"Fabricated citation: referenced non-existent source [{idx}] (available sources: 0).",
                        })
                elif idx < 1 or idx > num_sources:
                    key = f"out_of_range_{idx}"
                    if key not in seen_violations:
                        seen_violations.add(key)
                        findings.append({
                            "type": "T1",
                            "severity": "hard",
                            "detail": f"Fabricated citation: referenced non-existent source [{idx}] (available sources: 1..{num_sources}).",
                        })

        if num_sources == 0:
            continue

        # 2. Source relevance / keyword grounding check
        sent_kw: frozenset[str] | None = None
        for m in matches:
            idxs = _parse_citation_group(m.group(1))
            for idx in idxs:
                if 1 <= idx <= num_sources:
                    if sent_kw is None:
                        sent_kw = _extract_keywords(sent)
                    src_kw = _get_src_keywords(idx)
                    overlap = sent_kw & src_kw
                    if len(sent_kw) >= 3 and len(overlap) == 0:
                        snippet_preview = sent.strip().replace("\n", " ")[:80]
                        key = f"unbacked_{idx}_{snippet_preview}"
                        if key not in seen_violations:
                            seen_violations.add(key)
                            findings.append({
                                "type": "T1",
                                "severity": "hard",
                                "detail": f"Unbacked citation: [{idx}] does not contain facts supporting statement '{snippet_preview}'.",
                            })

        # 3. Quantitative & numeric citation verification
        segments = _get_citation_segments(sent)
        for indices, seg_text in segments:
            valid_indices = [idx for idx in indices if 1 <= idx <= num_sources]
            if not valid_indices:
                continue

            seg_numbers = _extract_numbers(seg_text)
            if not seg_numbers:
                continue

            # Check if numbers in this segment exist in any cited sources for this segment
            cited_numbers: set[float] = set()
            for idx in valid_indices:
                cited_numbers.update(_get_src_numbers(idx))

            missing_numbers = seg_numbers - cited_numbers
            if missing_numbers:
                snippet_preview = sent.strip().replace("\n", " ")[:80]
                idx_label = ",".join(str(i) for i in valid_indices)
                for num in sorted(missing_numbers):
                    num_disp = int(num) if num.is_integer() else num
                    key = f"unbacked_num_{idx_label}_{num}_{snippet_preview}"
                    if key not in seen_violations:
                        seen_violations.add(key)
                        findings.append({
                            "type": "T1",
                            "severity": "hard",
                            "detail": f"Unbacked numeric claim: [{idx_label}] does not contain numeric value {num_disp} from '{snippet_preview}'.",
                        })

    return findings


def run_preflight_scan(
    text: str = "",
    sources: list[SearchSource] | None = None,
    source_keyword_sets: list[set[str]] | None = None,
    source_number_sets: list[set[float]] | None = None,
    prompt: str | None = None,
    source_keywords: list[set[str]] | None = None,
    source_numbers: list[set[float]] | None = None,
) -> PreflightResult:
    """Deterministic pre-flight token, prompt-injection, and citation bounds scanner (<0.5ms).

    Scans:
      1. Prompt target (if provided) for prompt injections, system overrides, and delimiter breakouts.
      2. Draft text target for prompt injections, polyglot JSON codeblocks, XML container tags,
         out-of-bounds citations, unbacked keywords, and unbacked numeric figures.

    Returns:
        PreflightResult behaving as 2-tuple (has_hard_violations, preflight_findings) and dict-like object.
    """
    t0 = time.perf_counter()
    findings: list[dict] = []

    kw_sets = source_keyword_sets if source_keyword_sets is not None else source_keywords
    num_sets = source_number_sets if source_number_sets is not None else source_numbers

    # Pre-compute source keyword and number sets upfront if sources are present
    if sources:
        if kw_sets is None:
            kw_sets = build_source_keyword_sets(sources)
        if num_sets is None:
            num_sets = build_source_number_sets(sources)

    # 1. Scan user prompt if provided
    if prompt and prompt.strip():
        injected, snippet = scan_prompt_injection(prompt)
        if injected:
            findings.append({
                "type": "T1",
                "severity": "hard",
                "detail": f"Adversarial prompt injection or directive override token detected in prompt: '{snippet}'.",
                "target": "prompt",
            })

    # 2. Scan draft text for injections and verify citation/numeric grounding
    if text and text.strip():
        sources_list = sources or []
        draft_findings = verify_citation_grounding(
            text=text,
            sources=sources_list,
            source_keyword_sets=kw_sets,
            source_number_sets=num_sets,
        )
        findings.extend(draft_findings)

    has_hard = any(f.get("severity") == "hard" for f in findings)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return PreflightResult(has_hard, findings, latency_ms)


_SUBORD_MAP = [
    (r"\b(notwithstanding that|in spite of|even though|although|though|whereas|despite)\b", ClauseType.CONCESSIVE),
    (r"\b(on condition that|in the event that|provided that|providing that|as long as|even if|insofar as|given that|assuming that|unless|if|because)\b", ClauseType.CONDITIONAL),
    (r"\b(as soon as|whenever|before|after|since|until|while|when|once|upon)\b", ClauseType.TEMPORAL),
    (r"\b(whereby|wherein|which|whose|whom|where|who)\b", ClauseType.RELATIVE),
    (r"\b(and|but|or|nor|so|yet)\b", ClauseType.COORDINATE),
]

_SUB_PATTERN = re.compile(
    r"(?:^|(?<=[\s,;—\"(]))\b("
    r"notwithstanding that|even though|in spite of|provided that|providing that|"
    r"on condition that|in the event that|as long as|as soon as|even if|insofar as|"
    r"given that|assuming that|although|though|whereas|despite|unless|because|"
    r"while|when|whenever|after|before|since|until|once|upon|"
    r"which|who|whom|whose|where|whereby|wherein"
    r")\b",
    re.IGNORECASE,
)


_DELIMITER_PATTERN = re.compile(r"(?:,\s+(?![0-9])|[;—\n]+|\s+--\s+|(?<=[.!?])\s+)")
_DETERMINER_PREFIX_RE = re.compile(r"^(?:the|a|an|this|that|these|those|our|their|its|every|each|some|any|no)\b", re.IGNORECASE)
_PARTICIPIAL_START_RE = re.compile(r"^(having(?:\s+been)?\s+\w+|being\s+\w+)\b", re.IGNORECASE)
_START_SUBORD_RE = re.compile(
    r"^(?:"
    r"(?P<concessive>notwithstanding that|in spite of|even though|although|though|whereas|despite)|"
    r"(?P<conditional>on condition that|in the event that|provided that|providing that|as long as|even if|insofar as|given that|assuming that|unless|if|because)|"
    r"(?P<temporal>as soon as|whenever|before|after|since|until|while|when|once|upon)|"
    r"(?P<relative>whereby|wherein|which|whose|whom|where|who)|"
    r"(?P<coordinate>and|but|or|nor|so|yet)"
    r")\b",
    re.IGNORECASE,
)
_GROUP_TO_TYPE = {
    "concessive": ClauseType.CONCESSIVE,
    "conditional": ClauseType.CONDITIONAL,
    "temporal": ClauseType.TEMPORAL,
    "relative": ClauseType.RELATIVE,
    "coordinate": ClauseType.COORDINATE,
}
_RELATIVE_SEARCH_RE = re.compile(r"\b(whereby|wherein|which|whose|whom|where|who)\b", re.IGNORECASE)


def parse_clause_ast(sentence: str) -> list[PropositionSpan]:
    """Deterministically decomposes complex sentence into hierarchical PropositionSpan AST nodes."""
    if not sentence or not sentence.strip():
        return []

    s_clean = sentence.strip()
    if not s_clean:
        return []

    raw_segments: list[tuple[int, int, str]] = []
    last_end = 0
    for m in _DELIMITER_PATTERN.finditer(sentence):
        start = m.start()
        if start > last_end:
            seg = sentence[last_end:start]
            if seg.strip():
                raw_segments.append((last_end, start, seg))
        last_end = m.end()
    if last_end < len(sentence):
        seg = sentence[last_end:]
        if seg.strip():
            raw_segments.append((last_end, len(sentence), seg))

    if not raw_segments:
        start_char = sentence.find(s_clean)
        end_char = start_char + len(s_clean) if start_char != -1 else len(sentence)
        citations: list[int] = []
        if "[" in sentence:
            for m in _CITATION_PATTERN.finditer(sentence):
                citations.extend(_parse_citation_group(m.group(1)))
        return [
            PropositionSpan(
                span_id="span_1",
                clause_type=ClauseType.INDEPENDENT,
                raw_text=sentence[start_char:end_char],
                cleaned_text=s_clean.rstrip("."),
                start_char=start_char,
                end_char=end_char,
                citation_indices=citations,
                is_matrix=True,
                nesting_level=1,
            )
        ]

    expanded_segments: list[tuple[int, int, str]] = []
    for start, end, seg in raw_segments:
        sub_matches = list(_SUB_PATTERN.finditer(seg))
        if len(sub_matches) <= 1 or not sub_matches:
            expanded_segments.append((start, end, seg))
        else:
            sub_starts = [m.start(1) for m in sub_matches]
            prev_sub_start = 0
            for ss in sub_starts[1:]:
                sub_seg = seg[prev_sub_start:ss]
                if sub_seg.strip():
                    expanded_segments.append((start + prev_sub_start, start + ss, sub_seg))
                prev_sub_start = ss
            sub_seg = seg[prev_sub_start:]
            if sub_seg.strip():
                expanded_segments.append((start + prev_sub_start, end, sub_seg))

    parsed_infos = []
    for start, end, seg in expanded_segments:
        seg_trimmed = seg.strip(" ,;:\n—\"()")
        seg_type = ClauseType.INDEPENDENT
        matched_subord = None

        prefix = seg_trimmed[:40]
        m = _START_SUBORD_RE.match(prefix)
        if m:
            seg_type = _GROUP_TO_TYPE[m.lastgroup]
            matched_subord = m.group(0).lower()
        elif not _DETERMINER_PREFIX_RE.match(prefix):
            m_part = _PARTICIPIAL_START_RE.match(prefix)
            if m_part:
                seg_type = ClauseType.PARTICIPIAL
                matched_subord = m_part.group(1).lower()

        # Relative pronouns in first half of clause
        if seg_type == ClauseType.INDEPENDENT:
            m_rel = _RELATIVE_SEARCH_RE.search(seg_trimmed)
            if m_rel and m_rel.start() < len(seg_trimmed) // 2:
                seg_type = ClauseType.RELATIVE
                matched_subord = m_rel.group(1).lower()

        parsed_infos.append((start, end, seg, seg_trimmed, seg_type, matched_subord))

    # Group segments by sentence to assign matrix clause per sentence
    sent_groups: list[list[int]] = [[]]
    for idx, (start, end, seg, seg_trimmed, seg_type, matched_subord) in enumerate(parsed_infos):
        if idx > 0:
            _, prev_end, prev_seg, _, _, _ = parsed_infos[idx - 1]
            intervening = sentence[prev_end:start]
            if (
                ("\n" in intervening)
                or (("." in prev_seg or "!" in prev_seg or "?" in prev_seg) and bool(_SENTENCE_BOUNDARY_RE.search(prev_seg)))
                or (("." in intervening or "!" in intervening or "?" in intervening) and bool(_SENTENCE_BOUNDARY_RE.search(intervening)))
            ):
                sent_groups.append([])
        sent_groups[-1].append(idx)

    matrix_indices: set[int] = set()
    for group in sent_groups:
        if not group:
            continue
        indep_indices = [idx for idx in group if parsed_infos[idx][4] == ClauseType.INDEPENDENT]
        if indep_indices:
            best_idx = max(indep_indices, key=lambda i: len(parsed_infos[i][3]))
            matrix_indices.add(best_idx)
        else:
            matrix_indices.add(group[-1])

    spans: list[PropositionSpan] = []
    parent_id = None
    current_nesting = 1
    for i, (start, end, seg, seg_trimmed, seg_type, matched_subord) in enumerate(parsed_infos):
        span_id = f"span_{i+1}"
        c_indices: list[int] = []
        if "[" in seg:
            for m in _CITATION_PATTERN.finditer(seg):
                c_indices.extend(_parse_citation_group(m.group(1)))

        cleaned = seg_trimmed
        if matched_subord and cleaned.lower().startswith(matched_subord):
            cleaned = cleaned[len(matched_subord):].strip(" ,;:\n—\"()")

        if cleaned and cleaned[0].islower() and not cleaned.startswith(("http", "www")):
            cleaned = cleaned[0].upper() + cleaned[1:]

        is_matrix = (i in matrix_indices)
        if is_matrix:
            nesting = 1
        else:
            if i == 0:
                nesting = 2
            else:
                nesting = min(5, current_nesting + 1)
        current_nesting = max(current_nesting, nesting)

        span = PropositionSpan(
            span_id=span_id,
            clause_type=seg_type,
            raw_text=sentence[start:end],
            cleaned_text=cleaned or seg_trimmed,
            start_char=start,
            end_char=end,
            subordinator=matched_subord,
            citation_indices=c_indices,
            parent_span_id=parent_id if not is_matrix else None,
            is_matrix=is_matrix,
            nesting_level=nesting,
        )
        spans.append(span)
        if not is_matrix and parent_id is None or is_matrix:
            parent_id = span_id

    return spans


_FINITE_VERB_PREFIXES = frozenset({
    "is", "was", "are", "were", "am", "be", "been", "being",
    "will", "would", "shall", "should", "can", "could", "may", "might", "must",
    "has", "have", "had", "having",
    "does", "do", "did",
    "remains", "remained", "became", "becomes", "seems", "seemed", "appears", "appeared",
    "reduced", "reduces", "cured", "cures", "demonstrated", "demonstrates",
    "showed", "shows", "improved", "improves", "failed", "fails",
    "achieved", "achieves", "reported", "reports", "decreased", "decreases",
    "increased", "increases", "prevented", "prevents", "resulted", "results",
    "enabled", "enables", "requires", "required", "prohibits", "prohibited",
    "indicated", "indicates", "suggested", "suggests", "confirms", "confirmed",
})


_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[a-zA-Z0-9\]\)\'\"])[.!?](?=\s+|$)")


def disentangle_and_excise(
    text: str,
    unbacked_span_ids: set[str],
    spans: list[PropositionSpan],
) -> str:
    """Surgically excises unbacked sub-clauses and reconstitutes grammatical sentences."""
    if not text or not text.strip() or not spans:
        return ""

    retained_spans = [s for s in spans if s.span_id not in unbacked_span_ids]
    if not retained_spans:
        return ""

    if len(retained_spans) == len(spans):
        res = clean_grammar_and_punctuation(text)
        if res and not res.endswith((".", "!", "?")):
            res += "."
        return res

    # Group original spans into sentence lists
    sent_span_groups: list[list[PropositionSpan]] = [[]]
    for idx, s in enumerate(spans):
        if idx > 0:
            prev_s = spans[idx - 1]
            intervening = text[prev_s.end_char:s.start_char]
            if (
                ("\n" in intervening)
                or (("." in prev_s.raw_text or "!" in prev_s.raw_text or "?" in prev_s.raw_text) and bool(_SENTENCE_BOUNDARY_RE.search(prev_s.raw_text)))
                or (("." in intervening or "!" in intervening or "?" in intervening) and bool(_SENTENCE_BOUNDARY_RE.search(intervening)))
            ):
                sent_span_groups.append([])
        sent_span_groups[-1].append(s)

    reconstituted_sentences: list[str] = []

    for group in sent_span_groups:
        retained_in_group = [s for s in group if s.span_id not in unbacked_span_ids]
        if not retained_in_group:
            continue

        group_has_matrix = any(s.is_matrix for s in retained_in_group)
        retained_segments: list[str] = []

        for idx, span in enumerate(retained_in_group):
            if not group_has_matrix and span.subordinator and idx == 0:
                # Matrix excised in this sentence: promote first subordinate clause to standalone declarative
                retained_segments.append(span.cleaned_text)
            else:
                clean_raw = span.raw_text.strip(" ,;:\n—")
                retained_segments.append(clean_raw)

        if not retained_segments:
            continue

        first_seg = retained_segments[0].rstrip(" ,;:\n—")
        if first_seg and first_seg[0].islower() and not first_seg.startswith(("http", "www")):
            first_seg = first_seg[0].upper() + first_seg[1:]
        sent_res = first_seg

        for i in range(1, len(retained_segments)):
            seg = retained_segments[i]
            s = seg.strip(" ,;:\n—")
            if not s:
                continue
            first_word = s.split()[0].lower() if s.split() else ""
            if first_word in _FINITE_VERB_PREFIXES:
                s_verb = s[0].lower() + s[1:] if s[0].isupper() and first_word in _FINITE_VERB_PREFIXES else s
                sent_res += " " + s_verb
            else:
                sent_res += ", " + s

        cleaned_sent = clean_grammar_and_punctuation(sent_res)
        if cleaned_sent:
            if not cleaned_sent.endswith((".", "!", "?")):
                cleaned_sent += "."
            reconstituted_sentences.append(cleaned_sent)

    if not reconstituted_sentences:
        return ""

    if len(reconstituted_sentences) == 1:
        return reconstituted_sentences[0]

    return " ".join(reconstituted_sentences)



