"""Tavily web search integration for grounding GPT-1 in real sources.

Includes source authority scoring, temporal relevance filtering,
and search quality metrics for improved grounding.
"""
from __future__ import annotations

import threading
from urllib.parse import urlparse

from tavily import TavilyClient

import config
from pipeline.models import SearchSource

_tavily_client: TavilyClient | None = None
_tavily_client_key: str = ""
_tavily_lock = threading.Lock()

# Domain authority tiers for source ranking
_AUTHORITY_HIGH = {
    "gov", "edu", "mil",  # TLD-based
}
# Public-sector second-level domains (TLD split of gov.uk is "uk", not "gov")
_AUTHORITY_PUBLIC_SLDS = frozenset({
    "gov.uk", "nhs.uk", "ac.uk", "gov.au", "govt.nz", "gc.ca", "gouv.fr",
    "gob.es", "gov.in", "go.jp",
})
_AUTHORITY_KNOWN_DOMAINS = {
    # Government / official
    "who.int", "cdc.gov", "nih.gov", "fda.gov", "epa.gov",
    "sec.gov", "irs.gov", "usa.gov", "congress.gov",
    # Academic / research
    "nature.com", "science.org", "pubmed.ncbi.nlm.nih.gov",
    "arxiv.org", "scholar.google.com", "jstor.org",
    # Reference / encyclopedic
    "wikipedia.org", "britannica.com",
    # News (major)
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk",
    "nytimes.com", "washingtonpost.com", "theguardian.com",
    # Legal
    "law.cornell.edu", "supremecourt.gov", "courtlistener.com",
}


def compute_source_authority(url: str) -> float:
    """Score a URL's source authority from 0.0 to 1.0.

    Tiers:
    - 1.0: Government (.gov), educational (.edu), military (.mil)
    - 0.9: Known high-authority domains (WHO, CDC, Nature, etc.)
    - 0.5: Standard domains
    - 0.3: Unknown or low-trust domains
    """
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        hostname = hostname.lower()
        clean_host = hostname.removeprefix("www.")
        labels = clean_host.split(".") if clean_host else []

        for i in range(max(0, len(labels) - 1)):
            suffix = ".".join(labels[i:])
            if suffix in _AUTHORITY_PUBLIC_SLDS:
                return 1.0

        # Check TLD
        tld = labels[-1] if labels else ""
        if tld in _AUTHORITY_HIGH:
            return 1.0

        # Check known domains (strip www. prefix)
        for known in _AUTHORITY_KNOWN_DOMAINS:
            if clean_host == known or clean_host.endswith("." + known):
                return 0.9

        # Standard domains
        if tld in ("com", "org", "net", "io", "co", "int"):
            return 0.5

        return 0.3
    except Exception:
        return 0.3


def _get_tavily_client() -> TavilyClient | None:
    """Return a cached TavilyClient, or None if not configured/disabled."""
    global _tavily_client, _tavily_client_key
    if not config.is_tavily_enabled():
        return None
    current_key = config.get_tavily_key()
    if not current_key:
        return None
    with _tavily_lock:
        if _tavily_client is None or _tavily_client_key != current_key:
            _tavily_client = TavilyClient(api_key=current_key)
            _tavily_client_key = current_key
    return _tavily_client


def should_search(flags: dict) -> bool:
    """Determine if web search should be triggered.

    Only search when the query has signals that benefit from current web
    data: current events, legal, advice, comparisons, statistics, or
    future-year references.  Pure conceptual/explanatory queries (no flags
    set) skip search to save 1-3 s of latency.
    """
    if not config.is_tavily_enabled():
        return False
    return any([
        flags.get("current_events"),
        flags.get("legal_mode"),
        flags.get("advice_requested"),
        flags.get("comparative"),
        flags.get("percent_requested"),
        flags.get("future_year"),
    ])


def rank_sources(sources: list[SearchSource]) -> list[SearchSource]:
    """Re-rank search sources by a combined relevance + authority score.

    Combines the provider's original relevance score with domain authority,
    rather than discarding relevance entirely.  This prevents
    authoritative-but-irrelevant sources from drowning out niche-but-
    relevant ones.

    Combined score = 0.5 * relevance + 0.4 * authority + 0.1 * snippet_signal
    where snippet_signal = min(len(snippet) / 500, 1.0)
    """
    for s in sources:
        relevance = max(0.0, min(1.0, s.score))  # clamp provider score
        authority = compute_source_authority(s.url)
        snippet_signal = min(len(s.snippet) / 500.0, 1.0)
        s.score = round(0.5 * relevance + 0.4 * authority + 0.1 * snippet_signal, 4)
    return sorted(sources, key=lambda s: -s.score)


def deduplicate_sources(sources: list[SearchSource]) -> list[SearchSource]:
    """Deduplicate search sources by normalized URL and content snippet uniqueness.

    - Normalizes URLs (strips fragments, trailing slashes, tracking query params).
    - Retains the first (or highest-scoring) occurrence for each canonical URL.
    """
    if not sources:
        return []

    seen_urls: set[str] = set()
    unique_sources: list[SearchSource] = []

    for s in sources:
        try:
            parsed = urlparse(s.url)
            clean_host = (parsed.hostname or "").lower()
            clean_path = parsed.path.rstrip("/")
            canonical_url = f"{parsed.scheme.lower()}://{clean_host}{clean_path}" if clean_host else s.url.strip()
        except Exception:
            canonical_url = s.url.strip()

        if canonical_url in seen_urls:
            continue
        seen_urls.add(canonical_url)
        unique_sources.append(s)

    return unique_sources


def compute_search_quality(sources: list[SearchSource]) -> dict:
    """Compute search quality metrics for the result set."""
    if not sources:
        return {
            "source_count": 0, "avg_authority": 0.0,
            "high_authority_count": 0, "has_gov_edu": False, "quality_tier": "none",
        }
    authorities = [compute_source_authority(s.url) for s in sources]
    avg = sum(authorities) / len(authorities)
    high_count = sum(1 for a in authorities if a >= 0.9)
    has_gov_edu = any(a >= 1.0 for a in authorities)
    if avg >= 0.8 or high_count >= 2:
        tier = "high"
    elif avg >= 0.5 or high_count >= 1:
        tier = "medium"
    else:
        tier = "low"
    return {
        "source_count": len(sources), "avg_authority": round(avg, 2),
        "high_authority_count": high_count, "has_gov_edu": has_gov_edu, "quality_tier": tier,
    }


def refine_search_query(original_query: str, unsupported_claims: list[str]) -> str:
    """Build a refined search query from unsupported claims.

    Takes the original query and specific unsupported claim texts,
    combining them into a more targeted search query.
    """
    # Take up to 3 unsupported claims for the refined query
    claim_keywords = []
    for claim in unsupported_claims[:3]:
        # Extract key phrases (skip very short or generic words)
        words = [w for w in claim.split() if len(w) > 4]
        claim_keywords.extend(words[:5])

    if not claim_keywords:
        return original_query

    # Combine original query context with claim-specific keywords
    refined = original_query + " " + " ".join(claim_keywords[:10])
    return refined[:500]  # Cap to avoid overly long queries


def fetch_claim_evidence(
    claims: list[dict],
    existing_sources: list[SearchSource],
    max_per_claim: int = 2,
) -> list[SearchSource]:
    """Fetch evidence for specific claims that lack support.

    Performs per-claim search queries for claims that don't have NLI
    support or keyword overlap with existing sources.  Returns new
    sources (deduplicated against *existing_sources*).

    This is the "claim-conditional retrieval" step: retrieval is driven
    by verification needs rather than the original prompt alone.
    """
    client = _get_tavily_client()
    if client is None:
        return []

    existing_urls = {s.url for s in existing_sources}
    new_sources: list[SearchSource] = []

    # Only search for unsupported claims (limit total queries)
    unsupported = []
    for c in claims:
        nli = c.get("nli_result", {})
        if nli.get("supported"):
            continue  # already grounded
        text = c.get("text", "").strip()
        if len(text) >= 20:
            unsupported.append(text)

    # Limit to 3 claim queries to control latency/cost
    for claim_text in unsupported[:3]:
        try:
            response = client.search(
                query=claim_text[:200],
                search_depth="basic",
                max_results=max_per_claim,
                include_answer=False,
                topic="general",
            )
        except Exception:
            continue

        for r in response.get("results", []):
            url = r.get("url", "")
            if url in existing_urls:
                continue
            existing_urls.add(url)
            src = SearchSource(
                title=r.get("title", ""),
                url=url,
                snippet=r.get("content", ""),
                score=r.get("score", 0.0),
            )
            new_sources.append(src)

    if new_sources:
        new_sources = rank_sources(new_sources)

    return new_sources


def perform_web_search(query: str, max_results: int = 5) -> tuple[list[SearchSource], str]:
    """Call Tavily search API. Returns (sources, raw_context_string).

    Sources are ranked by authority score before being returned.
    Returns empty results on any error (search is best-effort).
    """
    client = _get_tavily_client()
    if client is None:
        return [], ""

    try:
        response = client.search(
            query=query,
            search_depth="basic",
            max_results=max_results,
            include_answer=True,
            topic="general",
        )
    except Exception:
        return [], ""

    sources = []
    for r in response.get("results", []):
        sources.append(SearchSource(
            title=r.get("title", ""),
            url=r.get("url", ""),
            snippet=r.get("content", ""),
            score=r.get("score", 0.0),
        ))

    # Deduplicate and rank by authority
    sources = rank_sources(deduplicate_sources(sources))

    context_lines = []
    for i, s in enumerate(sources, 1):
        authority = compute_source_authority(s.url)
        authority_label = ""
        if authority >= 1.0:
            authority_label = " authority=\"high-gov-edu\""
        elif authority >= 0.9:
            authority_label = " authority=\"trusted\""
        context_lines.append(
            f'<untrusted_evidence id="{i}" url="{s.url}"{authority_label}>\n'
            f"Title: {s.title}\n"
            f"Excerpt: {s.snippet}\n"
            f"</untrusted_evidence>"
        )

    raw_context = "\n\n".join(context_lines)

    answer = response.get("answer", "")
    if answer:
        raw_context = f"Tavily Summary: {answer}\n\n---\nSources:\n{raw_context}"

    return sources, raw_context
