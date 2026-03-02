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


class GroundingInfo(BaseModel):
    """NLI-backed claim grounding rate metrics."""
    grounding_rate: float = 0.0
    grounded_count: int = 0
    ungrounded_count: int = 0
    contradicted_count: int = 0
    neutral_count: int = 0
    total_evaluated: int = 0


class UnsupportedSpan(BaseModel):
    """A span of text identified as unsupported by evidence."""
    text: str = ""
    start: int = -1
    end: int = -1
    reason: str = ""
    confidence_tier: str = ""


class ConfidenceBreakdown(BaseModel):
    observed_pct: float = 0.0
    inference_pct: float = 0.0
    hypothesis_pct: float = 0.0
    unsupported_pct: float = 0.0
    user_provided_pct: float = 0.0
    total_claims: int = 0
    confidence_label: str = "Unknown"  # "High", "Medium", "Low", "Unknown"
    # NLI-backed grounding rate (when NLI + evidence available)
    grounding: Optional[GroundingInfo] = None
    unsupported_spans: List[UnsupportedSpan] = []


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
    # Atomic claim decomposition
    atomic_claims: List[dict] = []
    decomposition_ran: bool = False
    # Meta-verification (high-stakes cross-check)
    meta_verification: Optional[dict] = None


class StageConfig(BaseModel):
    stage: str  # "gpt1", "gpt2", "gpt3"
    provider: str = "openai"
    api_key: str = ""
    model: str = ""
    base_url: str = ""


class StressRequest(BaseModel):
    category: Optional[str] = None
    count: Optional[int] = None


class FeedbackRequest(BaseModel):
    request_id: str = ""
    prompt: str = ""
    rating: str  # "accurate", "inaccurate", "partially_accurate"
    verdict_correct: Optional[bool] = None
    confidence_correct: Optional[bool] = None
    comment: str = ""
