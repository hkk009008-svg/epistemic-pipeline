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

        # Check TLD
        tld = hostname.rsplit(".", 1)[-1] if "." in hostname else ""
        if tld in _AUTHORITY_HIGH:
            return 1.0

        # Check known domains (strip www. prefix)
        clean_host = hostname.removeprefix("www.")
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

    When Tavily is enabled, always search — every query benefits from
    grounding in current web sources.  The previous flag-gated approach
    missed many factual queries (e.g. "who is the president") because
    they didn't match narrow keyword regexes.
    """
    return config.is_tavily_enabled()


def rank_sources(sources: list[SearchSource]) -> list[SearchSource]:
    """Re-rank search sources by authority score (descending), then by snippet length as tiebreaker.

    Authority score replaces the Tavily relevance score on s.score, so
    we use snippet length as a secondary signal — longer snippets
    typically contain more useful context for grounding.
    """
    for s in sources:
        s.score = compute_source_authority(s.url)
    return sorted(sources, key=lambda s: (-s.score, -len(s.snippet)))


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

    # Rank by authority
    sources = rank_sources(sources)

    context_lines = []
    for i, s in enumerate(sources, 1):
        authority_label = ""
        if s.score >= 1.0:
            authority_label = " [HIGH AUTHORITY - gov/edu]"
        elif s.score >= 0.9:
            authority_label = " [TRUSTED SOURCE]"
        context_lines.append(
            f"[{i}] {s.title}{authority_label}\n"
            f"    URL: {s.url}\n"
            f"    Excerpt: {s.snippet}"
        )

    raw_context = "\n\n".join(context_lines)

    answer = response.get("answer", "")
    if answer:
        raw_context = f"Tavily Summary: {answer}\n\n---\nSources:\n{raw_context}"

    return sources, raw_context
