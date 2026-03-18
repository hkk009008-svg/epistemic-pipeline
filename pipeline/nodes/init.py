from pipeline.state import GraphState
from pipeline.helpers import PipelineError
import config

async def stage_init(state: GraphState) -> dict:
    """Validate config and return basic initial state."""
    if not config.has_api_key():
        raise PipelineError(400, "Set your OpenAI API key first.")

    req = state["request"]

    return {
        "prompt": req.prompt,
        "mode": getattr(req, "mode", "verify"),
        "hooks_to_run": [],
        "hook_results": [],
        "reasoning": []
    }
