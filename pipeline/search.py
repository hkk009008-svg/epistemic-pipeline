"""Bare-bones Tavily web search integration."""
from __future__ import annotations

import threading
from tavily import TavilyClient

import config

_tavily_client: TavilyClient | None = None
_tavily_client_key: str = ""
_tavily_lock = threading.Lock()

def _get_tavily_client() -> TavilyClient | None:
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

def perform_search_sync(query: str, max_results: int = 5) -> tuple[list[dict], str]:
    """Perform a bare-bones Tavily search. Returns (sources: list[dict], summary_note: str)."""
    client = _get_tavily_client()
    if client is None:
        return [], "Search disabled or key missing."
        
    try:
        # Tavily API has a strict 400 character limit on search queries
        truncated_query = query[:400] if len(query) > 400 else query
        
        response = client.search(
            query=truncated_query,
            search_depth="basic",
            max_results=max_results,
            include_answer=False,
            topic="general",
        )
    except Exception as e:
        return [], f"Search failed: {e}"
        
    sources = []
    for r in response.get("results", []):
        sources.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", ""),
            "score": r.get("score", 0.0),
        })
        
    note = f"Found {len(sources)} sources."
    return sources, note
