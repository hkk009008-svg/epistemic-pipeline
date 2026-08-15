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

import base64
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
    t = t.translate(_TRANSLATE_TABLE)
    # Unicode NFKC normalization (folds full-width chars, ligatures)
    if not t.isascii():
        t = unicodedata.normalize("NFKC", t)
    return t




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


@functools.lru_cache(maxsize=2048)
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
    r"(?:(?<!\d)[,;:!?](?!\d)|(?<!\d)\.(?!\d)|\b(?:whereas|while|but|although|even though|though|despite|because|since|unless|and|or|nor|so|yet)\b)",
    re.IGNORECASE,
)

# Polarity and Negation Patterns
_NEGATION_RE = re.compile(
    r"\b(?:"
    r"does\s+not|did\s+not|do\s+not|cannot|can\s+not|is\s+not|are\s+not|was\s+not|were\s+not|"
    r"will\s+not|would\s+not|should\s+not|could\s+not|has\s+not|have\s+not|had\s+not|"
    r"fails?\s+to|failed\s+to|failing\s+to|"
    r"prohibits?|prohibited|prohibiting|"
    r"refutes?|refuted|refuting|"
    r"denies|denied|denying|deny|"
    r"disproves?|disproved|disproving|"
    r"contradicts?|contradicted|contradicting|"
    r"(?:is\s+|are\s+|was\s+|were\s+)?without\s+evidence\s+of|"
    r"no\s+evidence\s+of|"
    r"lacks?|lacked|lacking|"
    r"without|never|neither|nor|none|no|not"
    r")\b",
    re.IGNORECASE,
)

_PREFIX_NEGATION_WORD_RE = re.compile(
    r"\bnon-(?!(?:profit|steroidal|small|disclosure|gaap|linear|alcoholic|invasive)\b)([a-z]+)\b",
    re.IGNORECASE,
)
_DOUBLE_NEGATION_RE = re.compile(
    r"\b(?:"
    r"not\s+un[a-z]+|"
    r"not\s+in(?:direct|frequent|valid|significant|adequate|effective|consistent|accurate|complete|tangible|tolerable|visible|curable|convenient|sufficient|tolerant)|"
    r"not\s+dis[a-z]+|"
    r"not\s+im[a-z]+|"
    r"not\s+without|not\s+fail\w*|no\s+lack|never\s+fails?|"
    r"without\s+(?:adverse|toxicity|toxicities|side\s+effects?|complications?|incident|injury|harm|disease|recurrence|failure|delay)"
    r")\b",
    re.IGNORECASE,
)


@functools.lru_cache(maxsize=4096)
def extract_polarity_state(text: str) -> bool:
    """Extract semantic polarity of text: True for positive/affirmative, False for negative/negated."""
    if not text:
        return True
    text_clean = re.sub(r"\([^)]*\)", " ", text)
    norm = normalize_preflight_text(text_clean).lower()
    if _DOUBLE_NEGATION_RE.search(norm):
        return True
    if _NEGATION_RE.search(norm) or _PREFIX_NEGATION_WORD_RE.search(norm):
        return False
    return True


@functools.lru_cache(maxsize=4096)
def has_polarity_mismatch(claim_text: str, source_text: str) -> bool:
    """Check if a claim and a source express contradictory semantic polarity."""
    if not claim_text or not source_text:
        return False

    norm_claim = normalize_preflight_text(claim_text).lower()
    norm_source = normalize_preflight_text(source_text).lower()

    # 1. Prefix negation check (e.g. non-toxic vs toxic)
    claim_prefixes = set(_PREFIX_NEGATION_WORD_RE.findall(norm_claim))
    src_prefixes = set(_PREFIX_NEGATION_WORD_RE.findall(norm_source))

    claim_bases_without_prefix = set(re.findall(r"\b[a-z]+\b", _PREFIX_NEGATION_WORD_RE.sub(" ", norm_claim)))
    src_bases_without_prefix = set(re.findall(r"\b[a-z]+\b", _PREFIX_NEGATION_WORD_RE.sub(" ", norm_source)))

    if any(base in claim_bases_without_prefix for base in src_prefixes):
        return True
    if any(base in src_bases_without_prefix for base in claim_prefixes):
        return True

    # Check "not without <X>" vs "without <X>" contradiction
    claim_has_not_without = bool(re.search(r"\bnot\s+without\b", norm_claim, re.IGNORECASE))
    src_has_not_without = bool(re.search(r"\bnot\s+without\b", norm_source, re.IGNORECASE))
    claim_has_without = bool(re.search(r"\bwithout\b", norm_claim, re.IGNORECASE))
    src_has_without = bool(re.search(r"\bwithout\b", norm_source, re.IGNORECASE))

    if (claim_has_not_without and not src_has_not_without and src_has_without) or (
        src_has_not_without and not claim_has_not_without and claim_has_without
    ):
        return True

    # 2. Extract best matching clause/sentence in source
    claim_kw = _extract_keywords(norm_claim)
    if not claim_kw:
        return False

    source_segments = _CLAUSE_SPLIT_RE.split(norm_source)
    matching_segments: list[tuple[int, str]] = []

    for seg in source_segments:
        seg_kw = _extract_keywords(seg)
        matched = claim_kw & seg_kw
        if matched:
            score = sum(len(w) for w in matched)
            matching_segments.append((score, seg))

    if not matching_segments:
        best_segment = norm_source
    else:
        matching_segments.sort(key=lambda x: x[0], reverse=True)
        best_segment = matching_segments[0][1]

    # 3. Check inverting / failure verbs
    src_fails_inverting = bool(
        re.search(
            r"\b(?:fails?\s+to|failed\s+to|failing\s+to|does\s+not|did\s+not|never)\s+(?:prevent|reduce|cure|stop|decrease|eliminate|achieve|guarantee)",
            best_segment,
            re.IGNORECASE,
        )
    )
    claim_asserts_inverting = bool(
        re.search(
            r"\b(?:prevents?|prevented|reduces?|reduced|cures?|cured|stops?|stopped|decreases?|decreased|eliminates?|eliminated|achieves?|achieved|guarantees?|guaranteed)\b",
            norm_claim,
            re.IGNORECASE,
        )
    ) and not bool(re.search(r"\b(?:fails?\s+to|failed\s+to|does\s+not|did\s+not|never)\b", norm_claim, re.IGNORECASE))

    if src_fails_inverting and claim_asserts_inverting:
        return True

    # Check safety / prevention semantic equivalence
    if bool(re.search(r"\b(?:prevents?|prevented|preventing|eliminates?|eliminated|eliminating|guarantees?|guaranteed)\b", norm_claim, re.IGNORECASE)):
        if bool(re.search(r"\b(?:no\s+other|never|no\s+different|without|no\s+conflicting)\b", norm_source, re.IGNORECASE)):
            return False

    # 4. Standard polarity comparison against candidate segments
    claim_pol = extract_polarity_state(norm_claim)
    max_score = matching_segments[0][0] if matching_segments else 0
    top_segments = [seg for score, seg in matching_segments if score == max_score]
    if top_segments:
        if any(extract_polarity_state(seg) == claim_pol for seg in top_segments):
            return False
        return True

    return claim_pol != extract_polarity_state(best_segment)


_MARKDOWN_STRIP_TABLE = str.maketrans("*_~`#", "     ")

_BASE64_WRAPPERS_RE = re.compile(
    r"(?i)(?:"
    r"\bbase64:\s*([A-Za-z0-9+/=_\-]{4,})|"
    r"\[base64\]([\s\S]*?)\[/base64\]|"
    r"<base64>([\s\S]*?)</base64>|"
    r"\batob\(\s*[\"']([A-Za-z0-9+/=_\-]{4,})[\"']\s*\)"
    r")"
)


def _decode_b64(raw_b64: str) -> str | None:
    cleaned = re.sub(r"\s+", "", raw_b64.strip())
    if not cleaned or len(cleaned) < 4:
        return None
    pad_len = (4 - (len(cleaned) % 4)) % 4
    padded = cleaned + ("=" * pad_len)
    try:
        decoded_bytes = base64.b64decode(padded, validate=False)
        decoded_str = decoded_bytes.decode("utf-8", errors="ignore")
        return decoded_str if decoded_str.strip() else None
    except Exception:
        return None


_INJECTION_ACTION_VERBS = (
    r"ignore|disregard|forget|bypass|override|drop|disable|dismiss|remove|delete|strip|skip|reset"
)
_INJECTION_MODIFIERS = (
    r"previous|prior|earlier|past|preceding|existing|above"
)
_INJECTION_TARGETS = (
    r"instructions?|directives?|rules?|constraints?|policies|safety|guidelines?|protocols?|requirements?|safeguards?|context|system\s*prompt"
)

_COMBINATORIAL_INJECTION_RE = re.compile(
    rf"\b(?:(?:{_INJECTION_ACTION_VERBS})\s+(?:(?:(?:all|the|your|any|our|these|those)\s+)?(?:{_INJECTION_MODIFIERS})\s+(?:(?:all|the|your|any|our|these|those)\s+)?|(?:all|the|your|any|our|these|those)\s+)?(?:{_INJECTION_TARGETS})|clear\s+(?:(?:all|the|your|any|our|these|those)\s+)?(?:(?:{_INJECTION_MODIFIERS})\s+)?(?:context|constraints?|history|memory|system\s*prompt|prompts?))\b"
)

_ACTION_VERBS_RE = re.compile(r"\b(?:ignore|disregard|forget|bypass|override|drop|clear|reset|disable|dismiss|remove|delete|strip|skip)\b")

_INJECTION_PATTERN = re.compile(
    r"(?:"
    r"\b(?:"
    r"admin\s*(?:override|instruction|directive|command|protocol|_mode)\b|"
    r"developer\s*mode(?:\s*[:=]|\s*(?:active|enabled|on|true|allow|override|bypass)\b)|"
    r"emergency\s*administrative\s*protocol\b|"
    r"root\s*(?:override|instruction|directive|command|protocol)\b|"
    r"system\s*(?:instruction\s*[:\b]|override\b|directive\b|command\b|protocol\b)|"
    r"superuser\s*(?:override|instruction|directive|command|protocol)\b|"
    r"allow_prompt_override\b|"
    r"clear\s*(?:context|constraints|system\s*prompt)\b|"
    r"reset\s*(?:context|constraints|system\s*prompt)\b|"
    r"gpt-1\s*response\s*to\s*verify\s*:|"
    r"original\s*prompt\s*:|"
    r"pre-decomposed\s*atomic\s*claims\s*:|"
    r"nli\s*pre-verification\s*signals\s*:|"
    r"output\s*verdict\s*:\s*pass|"
    r"override\s*:\s*verdict\s*[:=]\s*[\"']?\s*pass|"
    r"verdict\s*[:=]\s*[\"']?\s*pass|"
    r"arbiter_decision\s*[:=]\s*[\"']?\s*allow|"
    r"prompt\s*injection\b|"
    r"(?:adminoverride|systemoverride|rootoverride|superuseroverride|developermode|clearcontext|resetsystemprompt|disregardsafety|disregardallrules|emergencyadministrativeprotocol|ignorepreviousinstructions)\b"
    r")|"
    r"[\"']?(?:gpt2_verdict|verdict)[\"']?\s*:\s*[\"']?pass[\"']?|"
    r"[\"']?arbiter_decision[\"']?\s*[:=]\s*[\"']?allow[\"']?"
    r")"
)

_CONTAINER_TAGS_RE = re.compile(
    r"</?(?:system[_-](?:eval[_-]bypass|override|instruction|prompt)|untrusted[_-]override|instruction|verifier|system|instructions|prompt)\b[^>]*>|<!\[cdata\["
)

_HEADER_DELIMS_RE = re.compile(
    r"===\s*(?:task|tripwire\s*reference|end\s*tripwire\s*reference|evaluation|system|instructions|input|output)\s*===|"
    r"###\s*system(?:\s*instruction)?(?:\s*[:#]|\b)|"
    r"##\s*output\s*schema(?:\s*override)?\b|"
    r"---\s*(?:instructions|system|rules)\s*---|[*]{3}\s*(?:instructions|system|rules)\s*[*]{3}"
)

_CODEBLOCK_INJECTION_PATTERN = re.compile(
    r"(?i)```(?:json\w*)?\s*\{[\s\S]*?(?:[\"']verdict[\"']\s*:\s*[\"']PASS[\"']|[\"']arbiter_decision[\"']\s*:\s*[\"']ALLOW[\"']|[\"']reasoning_trace[\"']\s*:\s*\[[\s\S]*?(?:override|override\s+active))[\s\S]*?\}"
)


@functools.lru_cache(maxsize=4096)
def scan_prompt_injection(text: str, _depth: int = 0) -> tuple[bool, str | None]:
    """Deterministic regex and token scan for prompt injections, system overrides, and delimiter breakouts.

    Returns:
        (is_injected: bool, snippet_or_none: str | None)
    """
    if not text or not text.strip():
        return False, None
    if _depth > 4:
        return False, None

    normalized = normalize_preflight_text(text)
    lower_norm = normalized.lower()

    # 1. Base64 wrapper decoding check
    if "base64" in lower_norm or "atob" in lower_norm:
        for m in _BASE64_WRAPPERS_RE.finditer(normalized):
            b64_content = next((g for g in m.groups() if g is not None), None)
            if b64_content:
                decoded = _decode_b64(b64_content)
                if decoded:
                    inj, snip = scan_prompt_injection(decoded, _depth=_depth + 1)
                    if inj:
                        return True, snip or f"base64:{b64_content[:40]}"

    # 2. Direct words injection pattern check
    match = _INJECTION_PATTERN.search(lower_norm)
    if match:
        snippet = normalized[match.start():match.end()].strip().replace("\n", " ")[:60]
        return True, snippet

    # 3. XML / Container tag injection check
    if "<" in lower_norm:
        match_tag = _CONTAINER_TAGS_RE.search(lower_norm)
        if match_tag:
            snippet = normalized[match_tag.start():match_tag.end()].strip().replace("\n", " ")[:60]
            return True, snippet

    # 4. Header delimiter injection check
    if any(c in lower_norm for c in ("=", "#", "-", "*")):
        match_delim = _HEADER_DELIMS_RE.search(lower_norm)
        if match_delim:
            snippet = normalized[match_delim.start():match_delim.end()].strip().replace("\n", " ")[:60]
            return True, snippet

    # 5. Combinatorial ActionVerb x Modifier x Target matrix
    if _ACTION_VERBS_RE.search(lower_norm):
        match_comb = _COMBINATORIAL_INJECTION_RE.search(lower_norm)
        if match_comb:
            snippet = normalized[match_comb.start():match_comb.end()].strip().replace("\n", " ")[:60]
            return True, snippet

    # 6. Strip markdown/symbol wrapper check (e.g. *** Disregard past constraints ***)
    if any(c in normalized for c in ("*", "~", "`", "#")):
        stripped_sym = lower_norm.translate(_MARKDOWN_STRIP_TABLE)
        match_sym = _INJECTION_PATTERN.search(stripped_sym) or _COMBINATORIAL_INJECTION_RE.search(stripped_sym)
        if match_sym:
            snippet = normalized[match_sym.start():match_sym.end()].strip().replace("\n", " ")[:60]
            return True, snippet

    # 7. Codeblock injection pattern
    if "```" in normalized:
        match_cb = _CODEBLOCK_INJECTION_PATTERN.search(normalized)
        if match_cb:
            snippet = match_cb.group(0).strip().replace("\n", " ")[:60]
            return True, snippet

    return False, None


@functools.lru_cache(maxsize=4096)
def _extract_keywords(text: str) -> frozenset[str]:
    """Extract meaningful lowercase keywords from text, filtering stop words and numeric/scaled tokens."""
    norm_text = normalize_preflight_text(text)
    raw_tokens = _KEYWORD_RE.findall(norm_text)
    res = set()
    for tok in raw_tokens:
        tok_low = tok.lower()
        if tok_low in _STOP_WORDS or re.search(r"\d", tok_low):
            continue
        if len(tok_low) > 1:
            res.add(tok_low)
        elif len(tok) == 1 and tok.isupper() and tok not in ("A", "I"):
            res.add(tok_low)
    return frozenset(res)


@functools.lru_cache(maxsize=4096)
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
            if len(parts) == 2 and parts[0] in tens_dict and parts[1] in ones_dict:
                base_val = tens_dict[parts[0]] + ones_dict[parts[1]]
                if i + 1 < len(word_tokens) and word_tokens[i + 1] in scale_dict:
                    scale_val = scale_dict[word_tokens[i + 1]]
                    nums.add(float(base_val * scale_val))
                    nums.add(float(base_val))
                    i += 2
                    continue
                nums.add(float(base_val))
                i += 1
                continue

        if token in tens_dict:
            base_val = tens_dict[token]
            if i + 1 < len(word_tokens) and word_tokens[i + 1] in ones_dict:
                base_val += ones_dict[word_tokens[i + 1]]
                i += 1
            if i + 1 < len(word_tokens) and word_tokens[i + 1] in scale_dict:
                scale_val = scale_dict[word_tokens[i + 1]]
                nums.add(float(base_val * scale_val))
                nums.add(float(base_val))
                i += 2
                continue
            nums.add(float(base_val))
            i += 1
            continue

        if token in ones_dict and token != "one":
            base_val = ones_dict[token]
            if i + 1 < len(word_tokens) and word_tokens[i + 1] in scale_dict:
                scale_val = scale_dict[word_tokens[i + 1]]
                nums.add(float(base_val * scale_val))
                nums.add(float(base_val))
                i += 2
                continue
            nums.add(float(base_val))
            i += 1
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
    sources: list[SearchSource] | None = None,
) -> bool:
    """Check if text content is supported by any source snippet."""
    kw = _extract_keywords(text)
    if len(kw) < _MIN_KEYWORDS:
        return False
    clean_text = re.sub(
        r"(?i)\b(?:"
        r"without\s+(?:citation|evidence|source|proof|support|verification)|"
        r"lacks?\s+(?:citation|evidence|source|proof|support)|"
        r"(?:with\s+)?no\s+(?:citation|evidence|source|proof|support)|"
        r"has\s+no\s+citation|"
        r"unbacked\s+citation"
        r")\b",
        "",
        text,
    )
    for i, src_kw in enumerate(source_keyword_sets):
        overlap = kw & src_kw
        if len(overlap) / len(kw) >= threshold:
            if sources and 0 <= i < len(sources):
                src = sources[i]
                if has_polarity_mismatch(clean_text, f"{src.title} {src.snippet}"):
                    continue
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
        if best_overlap >= threshold and best_source_idx >= 0:
            src = sources[best_source_idx]
            if has_polarity_mismatch(entry.claim, f"{src.title} {src.snippet}"):
                result.append(entry)
                continue

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
        if _is_source_backed(detail, source_keyword_sets, threshold, sources=sources):
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


_UNBACKED_AUTHORITY_RE = re.compile(
    r"\b(?:"
    r"(?:(?:clinical|scientific|empirical|experimental|medical)\s+evidence|evidence)"
    r"|(?:(?:medical|scientific|expert|market)\s+consensus|consensus)"
    r"|(?:published\s+(?:papers|literature|studies|data|reports))"
    r"|(?:(?:scientific|clinical)\s+(?:studies|trials))"
    r"|(?:studies|study|research|researchers|data|literature)"
    r"|(?:experts|scientists|clinicians|doctors|specialists)"
    r")"
    r"(?:\s+(?:clearly|strongly|consistently|definitely|unequivocally|directly|robustly))?"
    r"\s+(?:demonstrates?|demonstrated|shows?|showed|shown|indicates?|indicated"
    r"|suggests?|suggested|confirms?|confirmed|proves?|proved|proven"
    r"|establish(?:es)?|established|reveals?|revealed|supports?|supported|agrees?|agreed)\b"
    r"(?!\s*\([^)]+\))"
    r"(?!\s*\[[^\]]+\])"
)

_EXPERTS_AGREE_RE = re.compile(
    r"\b(?:experts|scientists|researchers|clinicians|analysts)\s+agree\b"
    r"(?!\s*\([^)]+\))"
    r"(?!\s*\[[^\]]+\])"
)


_STRUCTURAL_NON_METRIC_RE = re.compile(
    r"\b(?:"
    r"(?:every|upon|within|after|for)\s+(?:\d+(?:\.\d+)?|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|forty|fifty)\s*(?:hours?|days?|weeks?|months?|years?|minutes?|seconds?)\s*(?:written\s+)?(?:notice|advance|delay|period|time|duration|window)?"
    r"|(?:\d+(?:\.\d+)?|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|forty|fifty)\s*(?:hours?|days?|weeks?|months?|years?|minutes?|seconds?)\s*(?:a|per)\s*(?:day|week|month|year|patient|hour)"
    r"|(?:phase|tier|stage|level|cohort|group|item|step|version|figure|table|section|chapter|part)\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
    r"|(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)(?:st|nd|rd|th)\s+(?:cohort|phase|tier|stage|level|group|step|version|edition|generation)"
    r"|(?:by|with|among|between|of|from)\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:former\s+|co-)?(?:engineers?|founders?|co-founders?|colleagues?|people|persons?|authors?|researchers?|scientists?|students?|partners?|members?|participants?|executives?|directors?|developers?|employees?)"
    r")\b",
    re.IGNORECASE,
)

_UNCITED_QUANT_TRIGGER_RE = re.compile(
    r"(?:[\$€£¥₹₩%]|\b(?:\d+(?:\.\d+)?|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fourty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|trillion|percent|pct|dollars?|euros?|pounds?|yen|users?|patients?|cases?|transactions?|degrees?|parameters?|nodes?|instances?|clusters?|servers?)\b)"
)


@functools.lru_cache(maxsize=2048)
def _is_unbracketed_quantitative(sent: str) -> frozenset[float]:
    """Returns non-empty set of quantitative numbers if sent contains factual figures that require citation."""
    clean_sent = _STRUCTURAL_NON_METRIC_RE.sub(" ", sent) if _STRUCTURAL_NON_METRIC_RE.search(sent) else sent
    nums = _extract_numbers(clean_sent)
    if not nums:
        return frozenset()

    # Filter out historical years (e.g. 1000 <= val <= 2099 without metric indicators and with historical context)
    has_metric_indicators = (
        bool(re.search(r"[\$€£¥₹₩%]", clean_sent))
        or bool(re.search(r"\b(?:percent|pct|dollars?|euros?|pounds?|yen|k|thousand|million|billion|trillion|users?|patients?|cases?|margin|rate|increase|decrease|growth|nodes?|instances?|clusters?|servers?|transactions?|degrees?|parameters?|subscribers?|sales|revenue|efficacy)\b", clean_sent, re.IGNORECASE))
    )
    if not has_metric_indicators and re.search(r"\b(?:in|since|during|year|circa|around|founded(?:\s+in)?)\s+\d{4}\b", clean_sent, re.IGNORECASE):
        year_matches = {
            float(m.group(1))
            for m in re.finditer(r"\b(?:in|since|during|year|circa|around|founded(?:\s+in)?)\s+(\d{4})\b", clean_sent, re.IGNORECASE)
        }
        filtered_nums = frozenset(n for n in nums if n not in year_matches)
        return filtered_nums

    return nums


def verify_citation_grounding(
    text: str,
    sources: list[SearchSource] | None,
    source_keyword_sets: list[set[str]] | None = None,
    source_number_sets: list[set[float]] | None = None,
    skip_injection_scan: bool = False,
    sentences: list[str] | None = None,
) -> list[dict]:
    """Deterministically check that all [N] citation markers refer to valid and relevant sources,
    and verify unbracketed sentences for uncited quantitative figures or unbacked authority claims.

    Returns a list of findings for any:
    - Out-of-range citations (e.g. [5] when only 2 sources exist, or citing [1] when 0 sources exist)
    - Unbacked citations (where proposition AST node referencing [N] has no keyword overlap with source N)
    - Polarity contradictions (where proposition AST node contradicts polarity of source N)
    - Swapped / unbacked numeric claims (where numbers in proposition span do not exist in bound source)
    - Uncited quantitative figures (T3) or unbacked authority claims (T3) in unbracketed statements
    """
    if not text or not text.strip():
        return []

    findings: list[dict] = []

    # Check for adversarial prompt injection or delimiter breakout markers
    if not skip_injection_scan:
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

    if sentences is None:
        sentences = _SENTENCE_SPLIT_RE.split(text)
    seen_violations: set[str] = set()

    for sent in sentences:
        if "[" not in sent:
            continue
        sent_clean = sent.strip()
        if not sent_clean:
            continue

        matches = list(_CITATION_PATTERN.finditer(sent))
        if not matches:
            continue

        # -------------------------------------------------------------
        # Bracket-cited sentence processing (R1, R3, R4)
        # -------------------------------------------------------------
        # 1. Out-of-range check
        all_sent_valid_indices: list[int] = []
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
                else:
                    all_sent_valid_indices.append(idx)

        if num_sources == 0 or not all_sent_valid_indices:
            continue

        # 2. AST Proposition Span Decomposition & Grounding (R3 & R4)
        spans = parse_clause_ast(sent)
        if not spans:
            spans = [
                PropositionSpan(
                    span_id="span_1",
                    clause_type=ClauseType.INDEPENDENT,
                    raw_text=sent,
                    cleaned_text=sent_clean.rstrip("."),
                    start_char=0,
                    end_char=len(sent),
                    citation_indices=all_sent_valid_indices,
                    is_matrix=True,
                    nesting_level=1,
                )
            ]

        sent_kw = _extract_keywords(sent_clean)
        sent_has_blocking = bool(
            re.search(
                r"\b(?:blocks?|blocked|blocking|prohibits?|prohibited|prohibiting|bans?|banned|banning|rejects?|rejected|rejecting|prevents?|prevented|preventing)\b",
                sent_clean,
                re.IGNORECASE,
            )
        )
        seen_flagged_nums: set[tuple[int, float]] = set()

        for span in spans:
            span_indices = [idx for idx in span.citation_indices if 1 <= idx <= num_sources]
            if not span_indices:
                # Inherit valid sentence-level citations
                span_indices = list(all_sent_valid_indices)

            span_kw = _extract_keywords(span.cleaned_text)
            span_nums = _extract_numbers(span.raw_text)
            span_preview = span.cleaned_text.strip().replace("\n", " ")[:80]

            unique_span_indices = sorted(set(span_indices))
            if len(unique_span_indices) == 1:
                # Single citation span (or multiple citations referring to same source, e.g. [1] with [1])
                idx = unique_span_indices[0]
                src_kw = _get_src_keywords(idx)
                src_nums = _get_src_numbers(idx)
                src_obj = sources_list[idx - 1]
                src_full_text = f"{src_obj.title} {src_obj.snippet}"

                # Keyword grounding:
                # Flag if span has no overlap and either:
                #   a) The whole sentence has 0 overlap with this source (complete unbacked citation), OR
                #   b) Span is a distinct subclause (concessive, coordinate, conditional, relative) with 0 overlap
                overlap = span_kw & src_kw
                sent_overlap = sent_kw & src_kw
                is_subclause = span.clause_type in (
                    ClauseType.CONCESSIVE,
                    ClauseType.COORDINATE,
                    ClauseType.CONDITIONAL,
                    ClauseType.RELATIVE,
                    ClauseType.TEMPORAL,
                    ClauseType.PARTICIPIAL,
                )
                overlap_ratio = len(overlap) / len(span_kw) if span_kw else 1.0
                has_grounded_numbers = bool(span_nums and span_nums <= src_nums)

                is_unbacked_span = False
                if not has_grounded_numbers and len(span_kw) >= 3:
                    if len(overlap) == 0:
                        is_unbacked_span = True
                    elif is_subclause and overlap_ratio < 0.35:
                        is_unbacked_span = True

                if is_unbacked_span and (len(sent_overlap) == 0 or is_subclause):
                    key = f"unbacked_ast_{idx}_{span_preview}"
                    if key not in seen_violations:
                        seen_violations.add(key)
                        findings.append({
                            "type": "T1",
                            "severity": "hard",
                            "detail": f"Unbacked citation: [{idx}] does not contain facts supporting statement '{span_preview}'.",
                        })
                # Polarity mismatch check
                elif not has_grounded_numbers and not sent_has_blocking and has_polarity_mismatch(span.cleaned_text, src_full_text):
                    key = f"polarity_mismatch_{idx}_{span_preview}"
                    if key not in seen_violations:
                        seen_violations.add(key)
                        findings.append({
                            "type": "T1",
                            "severity": "hard",
                            "detail": f"Unbacked citation: [{idx}] contradicts statement '{span_preview}'.",
                        })

                # Numbers check
                if span_nums:
                    missing = span_nums - src_nums
                    if missing:
                        for num in sorted(missing):
                            seen_flagged_nums.add((idx, num))
                            num_disp = int(num) if num.is_integer() else num
                            key = f"unbacked_num_{idx}_{num}_{span_preview}"
                            if key not in seen_violations:
                                seen_violations.add(key)
                                findings.append({
                                    "type": "T1",
                                    "severity": "hard",
                                    "detail": f"Unbacked numeric claim: [{idx}] does not contain numeric value {num_disp} from '{span_preview}'.",
                                })

            elif len(unique_span_indices) > 1:
                # Multi-Citation Entity-Attribute Isolation (R4)
                # Check each sub-segment in the span against its specific cited source
                span_segments = _get_citation_segments(span.raw_text)
                if len(span_segments) > 1:
                    for seg_idxs, seg_text in span_segments:
                        valid_seg_idxs = [i for i in seg_idxs if 1 <= i <= num_sources]
                        if not valid_seg_idxs:
                            continue
                        seg_kw = _extract_keywords(seg_text)
                        seg_nums = _extract_numbers(seg_text)
                        seg_preview = seg_text.strip().replace("\n", " ")[:80]
                        if len(valid_seg_idxs) == 1:
                            s_idx = valid_seg_idxs[0]
                            src_kw = _get_src_keywords(s_idx)
                            src_nums = _get_src_numbers(s_idx)
                            src_obj = sources_list[s_idx - 1]
                            src_full_text = f"{src_obj.title} {src_obj.snippet}"

                            if has_polarity_mismatch(seg_text, src_full_text):
                                key = f"polarity_mismatch_{s_idx}_{seg_preview}"
                                if key not in seen_violations:
                                    seen_violations.add(key)
                                    findings.append({
                                        "type": "T1",
                                        "severity": "hard",
                                        "detail": f"Unbacked citation: [{s_idx}] contradicts statement '{seg_preview}'.",
                                    })

                            if seg_nums:
                                missing = seg_nums - src_nums
                                if missing:
                                    for num in sorted(missing):
                                        seen_flagged_nums.add((s_idx, num))
                                        num_disp = int(num) if num.is_integer() else num
                                        key = f"unbacked_num_{s_idx}_{num}_{seg_preview}"
                                        if key not in seen_violations:
                                            seen_violations.add(key)
                                            findings.append({
                                                "type": "T1",
                                                "severity": "hard",
                                                "detail": f"Unbacked numeric claim: [{s_idx}] does not contain numeric value {num_disp} from '{seg_preview}'.",
                                            })
                        else:
                            all_seg_nums: set[float] = set()
                            for i in valid_seg_idxs:
                                all_seg_nums.update(_get_src_numbers(i))
                            if seg_nums:
                                missing = seg_nums - all_seg_nums
                                if missing:
                                    idx_label = ",".join(str(i) for i in valid_seg_idxs)
                                    for num in sorted(missing):
                                        for i in valid_seg_idxs:
                                            seen_flagged_nums.add((i, num))
                                        num_disp = int(num) if num.is_integer() else num
                                        key = f"unbacked_num_{idx_label}_{num}_{seg_preview}"
                                        if key not in seen_violations:
                                            seen_violations.add(key)
                                            findings.append({
                                                "type": "T1",
                                                "severity": "hard",
                                                "detail": f"Unbacked numeric claim: [{idx_label}] does not contain numeric value {num_disp} from '{seg_preview}'.",
                                            })
                else:
                    # Single text span with stacked citations e.g. [1, 2]
                    kw_overlaps: dict[int, int] = {}
                    for idx in unique_span_indices:
                        src_kw = _get_src_keywords(idx)
                        kw_overlaps[idx] = len(span_kw & src_kw)

                    max_ov = max(kw_overlaps.values()) if kw_overlaps else 0
                    best_candidates = [idx for idx, ov in kw_overlaps.items() if ov == max_ov and ov > 0]

                    if len(best_candidates) == 1:
                        best_idx = best_candidates[0]
                        src_obj = sources_list[best_idx - 1]
                        src_full_text = f"{src_obj.title} {src_obj.snippet}"
                        if has_polarity_mismatch(span.cleaned_text, src_full_text):
                            key = f"polarity_mismatch_{best_idx}_{span_preview}"
                            if key not in seen_violations:
                                seen_violations.add(key)
                                findings.append({
                                    "type": "T1",
                                    "severity": "hard",
                                    "detail": f"Unbacked citation: [{best_idx}] contradicts statement '{span_preview}'.",
                                })

                    cand_indices = best_candidates if len(best_candidates) == 1 else unique_span_indices
                    if len(span_kw) >= 3 and max_ov == 0:
                        idx_label = ",".join(str(i) for i in unique_span_indices)
                        key = f"unbacked_ast_{idx_label}_{span_preview}"
                        if key not in seen_violations:
                            seen_violations.add(key)
                            findings.append({
                                "type": "T1",
                                "severity": "hard",
                                "detail": f"Unbacked citation: [{idx_label}] does not contain facts supporting statement '{span_preview}'.",
                            })
                    if span_nums:
                        all_span_nums: set[float] = set()
                        for idx in unique_span_indices:
                            all_span_nums.update(_get_src_numbers(idx))

                        if len(best_candidates) == 1:
                            best_idx = best_candidates[0]
                            src_nums = _get_src_numbers(best_idx)
                            missing = span_nums - src_nums
                            if missing:
                                for num in sorted(missing):
                                    seen_flagged_nums.add((best_idx, num))
                                    num_disp = int(num) if num.is_integer() else num
                                    idx_label = str(best_idx) if num in all_span_nums else ",".join(str(i) for i in unique_span_indices)
                                    key = f"unbacked_num_{idx_label}_{num}_{span_preview}"
                                    if key not in seen_violations:
                                        seen_violations.add(key)
                                        findings.append({
                                            "type": "T1",
                                            "severity": "hard",
                                            "detail": f"Unbacked numeric claim: [{idx_label}] does not contain numeric value {num_disp} from '{span_preview}'.",
                                        })
                        else:
                            missing = span_nums - all_span_nums
                            if missing:
                                idx_label = ",".join(str(i) for i in unique_span_indices)
                                for num in sorted(missing):
                                    for c_idx in unique_span_indices:
                                        seen_flagged_nums.add((c_idx, num))
                                    num_disp = int(num) if num.is_integer() else num
                                    key = f"unbacked_num_{idx_label}_{num}_{span_preview}"
                                    if key not in seen_violations:
                                        seen_violations.add(key)
                                        findings.append({
                                            "type": "T1",
                                            "severity": "hard",
                                            "detail": f"Unbacked numeric claim: [{idx_label}] does not contain numeric value {num_disp} from '{span_preview}'.",
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
         unbracketed quantitative figures (T3), unbacked authority claims (T3),
         out-of-bounds citations, unbacked keywords, and unbacked numeric figures.

    Returns:
        PreflightResult behaving as 2-tuple (has_hard_violations, preflight_findings) and dict-like object.
    """
    t0 = time.perf_counter()
    findings: list[dict] = []

    kw_sets = source_keyword_sets if source_keyword_sets is not None else source_keywords
    num_sets = source_number_sets if source_number_sets is not None else source_numbers

    # 1. Scan user prompt if provided
    injected_prompt = False
    if prompt and prompt.strip():
        injected, snippet = scan_prompt_injection(prompt)
        if injected:
            injected_prompt = True
            findings.append({
                "type": "T1",
                "severity": "hard",
                "detail": f"Adversarial prompt injection or directive override token detected in prompt: '{snippet}'.",
                "target": "prompt",
            })

    # 2. Scan draft text for injections, unbracketed claims, and verify citation/numeric grounding
    if text and text.strip():
        # Fast path: check prompt injection in draft
        injected_draft, snippet = scan_prompt_injection(text)
        if injected_draft:
            findings.append({
                "type": "T1",
                "severity": "hard",
                "detail": f"Adversarial prompt injection or directive override token detected in draft: '{snippet}'.",
                "target": "draft",
            })

        lower_text = text.lower()
        has_potential_authority = bool(_UNBACKED_AUTHORITY_RE.search(lower_text) or _EXPERTS_AGREE_RE.search(lower_text))
        has_potential_quant = bool(any(c in "$€£¥₹₩%0123456789" for c in text) or _UNCITED_QUANT_TRIGGER_RE.search(lower_text))

        sentences: list[str] | None = None
        has_brackets = "[" in text
        if has_potential_authority or has_potential_quant or has_brackets:
            prose_for_unbracketed = re.sub(r"```[\s\S]*?```", " ", text) if "```" in text else text
            sentences = _SENTENCE_SPLIT_RE.split(prose_for_unbracketed)

        if sentences and (has_potential_authority or has_potential_quant):
            seen_unbracketed: set[str] = set()
            seen_unbracketed_sentences: set[str] = set()
            for sent in sentences:
                if "[" in sent:
                    continue
                sent_clean = sent.strip()
                if not sent_clean or sent_clean in seen_unbracketed_sentences:
                    continue
                seen_unbracketed_sentences.add(sent_clean)
                lower_sent = sent_clean.lower()

                if has_potential_authority:
                    m_auth = _UNBACKED_AUTHORITY_RE.search(lower_sent) or _EXPERTS_AGREE_RE.search(lower_sent)
                    if m_auth:
                        auth_phrase = sent_clean[m_auth.start():m_auth.end()].strip().replace("\n", " ")
                        snippet_preview = sent_clean.replace("\n", " ")[:80]
                        key = f"unbacked_auth_{auth_phrase}_{snippet_preview}"
                        if key not in seen_unbracketed:
                            seen_unbracketed.add(key)
                            findings.append({
                                "type": "T3",
                                "severity": "soft",
                                "detail": f"Unbacked authority assertion in unbracketed statement: '{auth_phrase}' without source citation.",
                                "target": "draft",
                            })

                if has_potential_quant and (any(c in "$€£¥₹₩%0123456789" for c in sent_clean) or _UNCITED_QUANT_TRIGGER_RE.search(lower_sent)):
                    unbracketed_nums = _is_unbracketed_quantitative(sent_clean)
                    if unbracketed_nums:
                        snippet_preview = sent_clean.replace("\n", " ")[:80]
                        key = f"uncited_quant_{sorted(unbracketed_nums)}_{snippet_preview}"
                        if key not in seen_unbracketed:
                            seen_unbracketed.add(key)
                            findings.append({
                                "type": "T3",
                                "severity": "soft",
                                "detail": f"Uncited quantitative claim in unbracketed statement: '{snippet_preview}' contains figures {sorted(unbracketed_nums)} without source citation.",
                                "target": "draft",
                            })

        if has_brackets:
            sources_list = sources or []
            draft_findings = verify_citation_grounding(
                text=text,
                sources=sources_list,
                source_keyword_sets=kw_sets,
                source_number_sets=num_sets,
                skip_injection_scan=True,
                sentences=sentences,
            )
            for df in draft_findings:
                if df not in findings:
                    findings.append(df)

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
    r"which|who|whom|whose|where|whereby|wherein|"
    r"(?:and|but|or|nor|so|yet)\s+(?=(?:(?:[a-zA-Z0-9]+\s+){1,3}(?:is|are|was|were|has|have|had|did|does|do|can|could|will|would|achieved|recorded|admitted|showed|grew|cost|prevented|reduced|enrolled|reached|completed|excelled|supported|failed|proved|demonstrated|indicates|reveals)|(?:is|are|was|were|has|have|had|did|does|do|can|could|will|would|achieved|recorded|admitted|showed|grew|cost|prevented|reduced|enrolled|reached|completed|excelled|supported|failed|proved|demonstrated|indicates|reveals)\b))"
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


@functools.lru_cache(maxsize=4096)
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
        if not sub_matches:
            expanded_segments.append((start, end, seg))
        else:
            prev_sub_start = 0
            for m in sub_matches:
                s_pos = m.start(1)
                if s_pos > prev_sub_start:
                    prefix_seg = seg[prev_sub_start:s_pos]
                    if prefix_seg.strip():
                        expanded_segments.append((start + prev_sub_start, start + s_pos, prefix_seg))
                prev_sub_start = s_pos
            tail_seg = seg[prev_sub_start:]
            if tail_seg.strip():
                expanded_segments.append((start + prev_sub_start, end, tail_seg))

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



