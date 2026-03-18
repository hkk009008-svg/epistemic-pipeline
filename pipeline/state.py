from typing import TypedDict, Callable, Optional, Any, List

from pipeline.models import (
    PipelineRequest,
    PipelineResponse,
    HookResult
)

class GraphState(TypedDict, total=False):
    """The unified state for the Epistemic Pipeline LangGraph."""

    # -- Inputs --
    request: PipelineRequest
    emit: Callable  # StageEventEmitter
    prompt: str  # req.prompt shortcut
    mode: str

    # -- Hooks Setup --
    hooks_to_run: List[str]  # e.g., ["search", "custom_tool"]
    hook_results: List[HookResult]  # Results from hooks

    # -- Evaluation --
    evaluation_result: str
    final_verdict: str
    verdict_label: str
    reasoning: List[str]

    # -- Final --
    final_response: PipelineResponse

