from pipeline.state import GraphState
from pipeline.models import HookResult
from pipeline.search import perform_search_sync

async def stage_hooks(state: GraphState) -> dict:
    """Execute all requested hooks (tools) based on route decision."""
    hooks_to_run = state.get("hooks_to_run", [])
    prompt = state.get("prompt", "")
    hook_results = []
    
    if "search" in hooks_to_run:
        # Run Tavily search
        try:
            sources, note = perform_search_sync(prompt)
            hook_results.append(HookResult(
                tool="search",
                data=sources,
                note=note
            ))
            state["emit"]("status", f"Search completed: {note}")
        except Exception as e:
            hook_results.append(HookResult(
                tool_name="search",
                data=[],
                note=f"Search failed: {e}"
            ))

    # In the future, the deploy team will add more hooks here based on their needs.
            
    return {"hook_results": hook_results}
