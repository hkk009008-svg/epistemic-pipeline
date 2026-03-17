"""Pydantic models for the bare-bones Auditor pipeline."""
from __future__ import annotations

import os

from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Any


class OpenAIConfig(BaseModel):
    api_key: str
    model: str = "gpt-4o-mini"

class TavilyConfig(BaseModel):
    api_key: str
    enabled: bool = True

class StageConfig(BaseModel):
    stage: str
    provider: str = "openai"
    api_key: str = ""
    model: str = ""
    base_url: str = ""

class PipelineRequest(BaseModel):
    prompt: str = Field(..., max_length=10_000)
    stream: bool = False

class HookResult(BaseModel):
    """Result of a pipeline tool hook execution."""
    tool: str
    data: Any
    note: str = ""

class PipelineResponse(BaseModel):
    """The final response from the bare-bones Auditor pipeline."""
    prompt: str
    hook_results: List[HookResult] = []
    final_verdict: str  # e.g., "PASS", "FAIL"
    verdict_label: str = "" # Human-readable label
    reasoning: str = "" # Final evaluation reasoning
    final_result: str = "" # Final compiled markdown or text response

