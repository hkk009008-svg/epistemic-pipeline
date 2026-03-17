from pipeline.state import GraphState
import config

async def stage_route(state: GraphState) -> dict:
    """Determine which hooks/tools to run."""
    state["emit"]("status", "Routing tools...")
    
    hooks_to_run = []
    if config.is_tavily_enabled():
        hooks_to_run.append("search")
        
    return {"hooks_to_run": hooks_to_run}
