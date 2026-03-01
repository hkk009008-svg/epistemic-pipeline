"""Tavily web search integration for grounding GPT-1 in real sources."""
from __future__ import annotations

from tavily import TavilyClient

import config
from pipeline.models import SearchSource

_tavily_client: TavilyClient | None = None
_tavily_client_key: str = ""


def _get_tavily_client() -> TavilyClient | None:
    """Return a cached TavilyClient, or None if not configured/disabled."""
    global _tavily_client, _tavily_client_key
    if not config.is_tavily_enabled():
        return None
    current_key = config.get_tavily_key()
    if not current_key:
        return None
    if _tavily_client is None or _tavily_client_key != current_key:
        _tavily_client = TavilyClient(api_key=current_key)
        _tavily_client_key = current_key
    return _tavily_client


def should_search(flags: dict) -> bool:
    """Determine if web search should be triggered based on prompt routing flags."""
    if not config.is_tavily_enabled():
        return False
    return (
        flags.get("percent_requested", False)
        or flags.get("legal_mode", False)
        or flags.get("future_year", False)
        or flags.get("current_events", False)
    )


def perform_web_search(query: str, max_results: int = 5) -> tuple[list[SearchSource], str]:
    """Call Tavily search API. Returns (sources, raw_context_string).

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

    context_lines = []
    for i, s in enumerate(sources, 1):
        context_lines.append(f"[{i}] {s.title}\n    URL: {s.url}\n    Excerpt: {s.snippet}")

    raw_context = "\n\n".join(context_lines)

    answer = response.get("answer", "")
    if answer:
        raw_context = f"Tavily Summary: {answer}\n\n---\nSources:\n{raw_context}"

    return sources, raw_context
