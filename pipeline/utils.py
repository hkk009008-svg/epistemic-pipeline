import time
import asyncio
from typing import Callable, Any, Dict, Optional, List
import json

def _date_context() -> str:
    """Return the current date for the system prompt."""
    import datetime
    return f"Today is {datetime.date.today()}.\n"

def _emit_stage_start(emit: Callable, stage: str, data: dict = None):
    emit("stage_start", {"stage": stage, **(data or {})})

def _emit_stage_complete(emit: Callable, stage: str, data: dict = None):
    emit("stage_complete", {"stage": stage, **(data or {})})

def _yield_sse(event: str, data: Any) -> str:
    """Format an event and data as a Server-Sent Events stream chunk."""
    if isinstance(data, dict) or isinstance(data, list):
        data = json.dumps(data)
    elif hasattr(data, "model_dump_json"):
        data = data.model_dump_json()
    else:
        data = str(data)
    
    lines = "".join(f"data: {line}\n" for line in data.splitlines())
    return f"event: {event}\n{lines}\n"

class PipelineEventEmitter:
    """Event emitter for streaming pipeline progress."""
    def __init__(self):
        self.queue = asyncio.Queue()
        
    def emit(self, event_type: str, data: Any):
        self.queue.put_nowait({"type": event_type, "data": data})
        
    async def get_event(self) -> Optional[Dict[str, Any]]:
        return await self.queue.get()
        
    def close(self):
        self.queue.put_nowait(None)

