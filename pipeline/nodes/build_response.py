from pipeline.state import GraphState
from pipeline.models import PipelineResponse

async def stage_build_response(state: GraphState) -> dict:
    """Build the final PipelineResponse to send back to the API."""
    req = state["request"]
    
    resp = PipelineResponse(
        prompt=req.prompt,
        hook_results=state.get("hook_results", []),
        final_result=state.get("evaluation_result", ""),
        final_verdict=state.get("final_verdict", "FAIL"),
        verdict_label=state.get("verdict_label", "Error"),
        reasoning="\n".join(state.get("reasoning", []))
    )

    state["emit"]("done", "")
    return {"final_response": resp}
