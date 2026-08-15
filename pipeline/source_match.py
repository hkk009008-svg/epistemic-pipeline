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

import re

from pipeline.models import ClaimEntry, SearchSource

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


def _extract_keywords(text: str) -> set[str]:
    """Extract meaningful lowercase keywords from text, filtering stop words."""
    words = re.findall(r"[a-zA-Z0-9]+(?:'[a-z]+)?", text.lower())
    return {w for w in words if w not in _STOP_WORDS and len(w) > 1}


def _extract_numbers(text: str) -> set[float]:
    """Extract all numeric values (canonical floats) from text, ignoring citation markers."""
    # Strip citation markers and bracket annotations (e.g. [1], [2], [verified])
    clean = re.sub(r"\[[^\]]*\]", " ", text)

    nums: set[float] = set()

    # 1. Numbers with scale multipliers (e.g. $1.5M, 50 million, 100k)
    scale_re = re.compile(
        r"(?:[\$€£¥₹₩]\s*)?(\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)\s*(k|m|b|t|thousand|million|billion|trillion)\b",
        re.IGNORECASE,
    )
    for m in scale_re.finditer(clean):
        raw_num = m.group(1).replace(",", "")
        scale_word = m.group(2).lower()
        try:
            val = float(raw_num)
            factor = _SCALE_FACTORS.get(scale_word, 1)
            nums.add(round(val * factor, 6))
            nums.add(round(val, 6))
        except ValueError:
            pass

    # 2. General numbers (integers, floats, currency amounts, percentages)
    num_re = re.compile(
        r"(?:[\$€£¥₹₩]\s*)?(\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)(?:\s*(?:%|percent|pct))?",
        re.IGNORECASE,
    )
    for m in num_re.finditer(clean):
        raw_num = m.group(1).replace(",", "")
        try:
            val = float(raw_num)
            nums.add(round(val, 6))
        except ValueError:
            pass

    # 3. Word numbers (e.g. fifty, thirty, seven, twenty-five)
    if _UNIT_ONE_RE.search(clean):
        nums.add(1.0)

    # Clean text of scaled phrases before word-number scan so "million" in "1.5 million" isn't extracted as 1,000,000
    unscaled_clean = scale_re.sub(" ", clean)

    word_tokens = re.findall(r"[a-zA-Z]+(?:-[a-zA-Z]+)?", unscaled_clean.lower())

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

    return nums


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
    return [_extract_keywords(f"{s.title} {s.snippet}") for s in sources]


def build_source_number_sets(sources: list[SearchSource]) -> list[set[float]]:
    """Pre-compute numeric value sets from sources."""
    return [_extract_numbers(f"{s.title} {s.snippet}") for s in sources]


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
        if ac_text == claim_lower or ac_text in claim_lower or claim_lower in ac_text:
            if nli.get("best_entailment", 0.0) >= _NLI_ENTAILMENT_THRESHOLD:
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
    matches = list(re.finditer(r"\[(\d+)\]", sent))
    if not matches:
        return []
    if len(matches) == 1:
        return [([int(matches[0].group(1))], sent)]

    # Group adjacent citation markers (e.g. [1][2] or [1], [2])
    grouped_citations: list[tuple[list[int], int, int]] = []
    cur_indices = [int(matches[0].group(1))]
    cur_start = matches[0].start()
    cur_end = matches[0].end()

    for i in range(1, len(matches)):
        m = matches[i]
        intervening = sent[cur_end:m.start()].strip(" ,;")
        if len(intervening) == 0:
            cur_indices.append(int(m.group(1)))
            cur_end = m.end()
        else:
            grouped_citations.append((cur_indices, cur_start, cur_end))
            cur_indices = [int(m.group(1))]
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
        next_indices, next_start, next_end = grouped_citations[i + 1]

        text_before = sent[span_start:g_start].strip()
        intervening = sent[g_end:next_start]

        split_match = re.search(r"(?:[,;]|\b(?:whereas|while|but|although)\b)", intervening, re.IGNORECASE)

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

    sources_list = sources or []
    num_sources = len(sources_list)

    if source_keyword_sets is None and num_sources > 0:
        source_keyword_sets = build_source_keyword_sets(sources_list)
    elif source_keyword_sets is None:
        source_keyword_sets = []

    if source_number_sets is None and num_sources > 0:
        source_number_sets = build_source_number_sets(sources_list)
    elif source_number_sets is None:
        source_number_sets = []

    findings: list[dict] = []
    citation_pattern = re.compile(r"\[(\d+)\]")

    # Break text into sentence segments
    sentences = re.split(r"(?<=[.!?\n])\s+", text)
    seen_violations: set[str] = set()

    for sent in sentences:
        matches = list(citation_pattern.finditer(sent))
        if not matches:
            continue

        sent_kw = _extract_keywords(sent)

        # 1. Out-of-range check
        for m in matches:
            idx_str = m.group(1)
            try:
                idx = int(idx_str)
            except ValueError:
                continue

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
        for m in matches:
            try:
                idx = int(m.group(1))
            except ValueError:
                continue
            if 1 <= idx <= num_sources:
                src_kw = source_keyword_sets[idx - 1]
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
                cited_numbers.update(source_number_sets[idx - 1])

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
    text: str,
    sources: list[SearchSource] | None,
    source_keyword_sets: list[set[str]] | None = None,
    source_number_sets: list[set[float]] | None = None,
) -> tuple[bool, list[dict]]:
    """Deterministic pre-flight token and citation bounds scanner (<10ms).

    Validates citation indices against available source counts, verifies keyword grounding,
    and checks quantitative figures without LLM inference.

    Returns:
        (has_hard_violations, preflight_findings)
    """
    if not text or not text.strip():
        return False, []

    findings = verify_citation_grounding(
        text=text,
        sources=sources or [],
        source_keyword_sets=source_keyword_sets,
        source_number_sets=source_number_sets,
    )
    has_hard = any(f.get("severity") == "hard" for f in findings)
    return has_hard, findings
