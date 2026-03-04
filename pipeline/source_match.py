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
from typing import List, Optional

from pipeline.models import ClaimEntry, SearchSource


# Words too common to be meaningful signal
_STOP_WORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would shall should may might can could of in on at to for "
    "with by from as into through during before after above below "
    "between out off over under again further then once that this "
    "these those and but or nor not no so yet both each every all "
    "any few more most other some such than too very also just about "
    "its it their them they he she his her which who whom what where "
    "when how if because while although since until unless even still "
    "already often really only however known called referred many "
    "well much like including based".split()
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


def _extract_keywords(text: str) -> set[str]:
    """Extract meaningful lowercase keywords from text, filtering stop words."""
    words = re.findall(r"[a-zA-Z0-9]+(?:'[a-z]+)?", text.lower())
    return {w for w in words if w not in _STOP_WORDS and len(w) > 1}


def _is_source_backed(text: str, source_keyword_sets: list[set[str]]) -> bool:
    """Check if text content is supported by any source snippet."""
    kw = _extract_keywords(text)
    if len(kw) < _MIN_KEYWORDS:
        return False
    for src_kw in source_keyword_sets:
        overlap = kw & src_kw
        if len(overlap) / len(kw) >= _MATCH_THRESHOLD:
            return True
    return False


def build_source_keyword_sets(sources: List[SearchSource]) -> list[set[str]]:
    """Pre-compute keyword sets from sources (call once, pass to both functions).

    When using both recategorize_with_sources and filter_findings_with_sources,
    call this once and pass the result to avoid redundant keyword extraction.
    """
    return [_extract_keywords(f"{s.title} {s.snippet}") for s in sources]


def _find_nli_support(claim_text: str, nli_claims: list[dict]) -> Optional[dict]:
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
    claim_table: List[ClaimEntry],
    sources: List[SearchSource],
    source_keyword_sets: list[set[str]] | None = None,
    nli_claims: list[dict] | None = None,
) -> List[ClaimEntry]:
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

        # Use stricter threshold for keyword-only matches
        threshold = _STRICT_MATCH_THRESHOLD if nli_claims else _MATCH_THRESHOLD
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
    findings: List[dict],
    sources: List[SearchSource],
    source_keyword_sets: list[set[str]] | None = None,
) -> List[dict]:
    """Remove T1/T7 findings whose detail text is supported by search sources.

    Returns a new list of findings (leaves originals unchanged).
    Only removes findings of overridable types (T1, T7, etc.).

    Pass pre-computed *source_keyword_sets* to avoid redundant extraction
    when calling both recategorize_with_sources and filter_findings_with_sources.
    """
    if not sources or not findings:
        return findings

    if source_keyword_sets is None:
        source_keyword_sets = build_source_keyword_sets(sources)

    result = []
    for f in findings:
        ftype = f.get("type", "")
        detail = f.get("detail", "")

        # Only filter overridable finding types
        if ftype not in _SOURCE_OVERRIDABLE_TYPES:
            result.append(f)
            continue

        # Check if the finding's detail text matches source content
        if _is_source_backed(detail, source_keyword_sets):
            continue  # Drop this finding — it's about source-backed content

        result.append(f)

    return result
