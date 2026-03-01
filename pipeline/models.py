"""Pydantic models for pipeline requests, responses, and internal data."""
from __future__ import annotations

from pydantic import BaseModel
from typing import Optional, List


class OpenAIConfig(BaseModel):
    api_key: str
    model: str = "gpt-4o-mini"


class TavilyConfig(BaseModel):
    api_key: str
    enabled: bool = True


class SearchSource(BaseModel):
    title: str
    url: str
    snippet: str
    score: float = 0.0


class ClaimEntry(BaseModel):
    claim: str
    category: str
    justification: str


class EditEntry(BaseModel):
    action: str
    target: str
    replacement: str


class ConfidenceBreakdown(BaseModel):
    observed_pct: float = 0.0
    inference_pct: float = 0.0
    hypothesis_pct: float = 0.0
    unsupported_pct: float = 0.0
    user_provided_pct: float = 0.0
    total_claims: int = 0
    confidence_label: str = "Unknown"  # "High", "Medium", "Low", "Unknown"


class PipelineRequest(BaseModel):
    prompt: str
    gpt1_system: str = ""
    gpt2_system: str = ""
    gpt3_system: str = ""


class PipelineResponse(BaseModel):
    prompt_version: str = ""
    gpt1_input: str
    gpt1_output: str
    bypassed: bool
    # GPT-2 results
    gpt2_raw: str
    claim_table: List[ClaimEntry]
    violations: List[str]
    gpt2_verdict: str
    gpt2_reasoning: List[str] = []  # GPT-2 chain-of-thought reasoning trace
    # GPT-3 arbiter results (only if GPT-2 FAIL)
    arbiter_invoked: bool
    arbiter_decision: str
    arbiter_rationale: List[str]
    arbiter_edits: List[EditEntry]
    arbiter_policy_notes: List[str]
    arbiter_raw: str
    # Rewrite loop (if ALLOW_WITH_EDITS)
    rewrite_occurred: bool
    rewrite_output: str
    rewrite_gpt2_raw: str
    rewrite_claim_table: List[ClaimEntry]
    rewrite_violations: List[str]
    rewrite_verdict: str
    # Final
    final_verdict: str
    final_result: str
    # Prompt routing / sanitizer metadata
    prompt_flags: Optional[dict] = None
    sanitizer_applied: bool = False
    # Web search enrichment
    search_performed: bool = False
    search_query: str = ""
    search_sources: List[SearchSource] = []
    # Confidence scoring
    confidence: ConfidenceBreakdown = ConfidenceBreakdown()


class StressRequest(BaseModel):
    category: Optional[str] = None
    count: Optional[int] = None
