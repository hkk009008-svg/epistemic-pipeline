import asyncio
import time
from typing import AsyncGenerator

from pipeline.models import PipelineRequest, PipelineResponse
from pipeline.graph import build_pipeline_graph
from pipeline.utils import _emit_stage_start, _emit_stage_complete, PipelineEventEmitter, _yield_sse

# Global graph instance (compiled once)
_graph = build_pipeline_graph()

def generate_pipeline(request: PipelineRequest) -> PipelineResponse:
    """Synchronous pipeline entry point."""
    return asyncio.run(generate_pipeline_async(request))

async def generate_pipeline_async(request: PipelineRequest) -> PipelineResponse:
    """Asynchronous pipeline execution via LangGraph."""
    
    # We use a no-op emitter for non-streaming mode
    def noop_emit(event: str, data: any):
        pass

    initial_state = {
        "request": request,
        "prompt": request.prompt,
        "emit": noop_emit
    }

    final_state = await _graph.ainvoke(initial_state)
    return final_state["final_response"]


async def generate_pipeline_stream(request: PipelineRequest) -> AsyncGenerator[str, None]:
    """Streaming pipeline execution via LangGraph Server-Sent Events (SSE)."""
    
    emitter = PipelineEventEmitter()
    
    async def graph_runner():
        initial_state = {
            "request": request,
            "prompt": request.prompt,
            "emit": emitter.emit
        }
        try:
            final_state = await _graph.ainvoke(initial_state)
            emitter.emit("done", final_state["final_response"].model_dump_json())
        except Exception as e:
            emitter.emit("error", str(e))
        finally:
            emitter.close()

    # Start the LangGraph execution in a background task
    runner_task = asyncio.create_task(graph_runner())
    
    # Yield SSE events as they arrive from the emitter
    yield _yield_sse("status", "Initializing Auditor Pipeline...")
    
    while True:
        event = await emitter.get_event()
        if event is None:
            break
            
        yield _yield_sse(event["type"], event["data"])
        
        if event["type"] in ("done", "error"):
            break

    # Ensure graph task cleans up
    await runner_task

