"""Pydantic models for pipeline requests, responses, and internal data."""
from __future__ import annotations

import os

from pydantic import BaseModel, Field
from typing import Literal, Optional, List


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
    target_id: str = ""  # UUID from decomposer — enables deterministic edits


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
    # Human-readable reasoning for the confidence score
    confidence_reasoning: List[str] = []
    # NLI-backed grounding rate (when NLI + evidence available)
    grounding: Optional[GroundingInfo] = None
    unsupported_spans: List[UnsupportedSpan] = []


class PipelineRequest(BaseModel):
    prompt: str = Field(..., max_length=10_000)
    # System prompt overrides — only accepted when ALLOW_PROMPT_OVERRIDE=true (dev only).
    # In production, these are silently ignored to prevent verification bypass.
    gpt1_system: str = ""
    gpt2_system: str = ""
    gpt3_system: str = ""
    # Verification tier: strict (default/current), standard (balanced), light (fact-check only)
    tier: Literal["strict", "standard", "light"] = "strict"
    # Output format: auto (derive from tier), structured, annotated, concise
    output_format: Literal["auto", "structured", "annotated", "concise"] = "auto"
    # Enable SSE streaming of pipeline progress events
    stream: bool = False

    def model_post_init(self, __context):
        if os.getenv("ALLOW_PROMPT_OVERRIDE", "").lower() not in ("true", "1", "yes"):
            object.__setattr__(self, "gpt1_system", "")
            object.__setattr__(self, "gpt2_system", "")
            object.__setattr__(self, "gpt3_system", "")


# Maps (final_verdict, confidence_label) to human-readable verdict descriptions
_VERDICT_LABELS = {
    ("PASS", "High"): "Verified with evidence",
    ("PASS", "Medium"): "Partially supported",
    ("PASS", "Low"): "Insufficient evidence",
    ("PASS", "Unknown"): "Insufficient evidence",
    ("FAIL", "High"): "Blocked due to fabrication risk",
    ("FAIL", "Medium"): "Contradicted by evidence",
    ("FAIL", "Low"): "Blocked due to fabrication risk",
    ("FAIL", "Unknown"): "Blocked due to fabrication risk",
}


def compute_verdict_label(final_verdict: str, confidence_label: str) -> str:
    """Return a human-readable verdict label."""
    return _VERDICT_LABELS.get(
        (final_verdict, confidence_label),
        "Verified with evidence" if final_verdict == "PASS" else "Blocked due to fabrication risk",
    )


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
    rewrite_reasoning: List[str] = []  # GPT-2 reasoning trace from re-verification
    # Final
    final_verdict: str
    final_result: str
    # Human-readable verdict label (e.g. "Verified with evidence")
    verdict_label: str = ""
    # Prompt routing / sanitizer metadata
    prompt_flags: Optional[dict] = None
    sanitizer_applied: bool = False
    # Web search enrichment
    search_performed: bool = False
    search_attempted: bool = False
    search_note: str = ""
    search_query: str = ""
    search_sources: List[SearchSource] = []
    # Confidence scoring
    confidence: ConfidenceBreakdown = ConfidenceBreakdown()
    # Atomic claim decomposition
    atomic_claims: List[dict] = []
    decomposition_ran: bool = False
    # Meta-verification (high-stakes cross-check)
    meta_verification: Optional[dict] = None
    # Tier and output format metadata
    tier: str = "strict"
    output_format: str = "structured"

    def model_post_init(self, __context):
        # Auto-compute verdict_label if not explicitly set
        if not self.verdict_label:
            object.__setattr__(
                self, "verdict_label",
                compute_verdict_label(self.final_verdict, self.confidence.confidence_label),
            )


class StageConfig(BaseModel):
    stage: Literal["gpt1", "gpt2", "gpt3"]
    provider: Literal["openai", "anthropic", "openrouter", "ollama"] = "openai"
    api_key: str = ""
    model: str = ""
    base_url: str = ""


class StressRequest(BaseModel):
    category: Optional[str] = None
    count: Optional[int] = Field(None, ge=1, le=100)
    tier: Literal["strict", "standard", "light"] = "strict"
    start_index: int = Field(0, ge=0, description="Resume from this test index (0-based)")


class FeedbackRequest(BaseModel):
    request_id: str = ""
    prompt: str = ""
    rating: str  # "accurate", "inaccurate", "partially_accurate"
    verdict_correct: Optional[bool] = None
    confidence_correct: Optional[bool] = None
    comment: str = ""


# ---------------------------------------------------------------------------
# Structured Output Schemas — used with OpenAI's response_format parameter
# to enforce guaranteed JSON output from GPT-2 and GPT-3.
# ---------------------------------------------------------------------------

class FindingSchema(BaseModel):
    """A single verification finding from GPT-2."""
    type: str
    severity: Literal["hard", "soft"]
    detail: str


class GPT2ResponseSchema(BaseModel):
    """Structured output schema for GPT-2 Verifier responses.

    Used with OpenAI's response_format to mathematically enforce valid JSON,
    eliminating the need for extract_json() regex parsing and retry loops.
    """
    reasoning_trace: List[str] = []
    claim_table: List[dict] = []
    findings: List[FindingSchema] = []
    verdict: Literal["PASS", "FAIL"]


class ArbiterEditSchema(BaseModel):
    """A single edit instruction from GPT-3 Arbiter.

    Supports both text-based targeting (legacy) and ID-based targeting (V5).
    When target_id is set, Python applies the edit directly to the claim JSON
    without relying on GPT-1 to find and replace strings.
    """
    action: Literal["DELETE", "REWRITE", "MOVE_TO_UNKNOWN"]
    target: str
    replacement: str = ""
    target_id: str = ""  # UUID from decomposer — enables deterministic edits


class GPT3ResponseSchema(BaseModel):
    """Structured output schema for GPT-3 Arbiter responses.

    Used with OpenAI's response_format to mathematically enforce valid JSON,
    eliminating the need for extract_json() regex parsing and retry loops.
    """
    arbiter_decision: Literal["BLOCK", "ALLOW_WITH_EDITS", "ALLOW_AS_UNKNOWN_ONLY"]
    rationale: List[str] = []
    edits_for_gpt1: List[ArbiterEditSchema] = []
    final_policy_notes: List[str] = []


class DecomposerResponseSchema(BaseModel):
    """Structured output schema for claim decomposition."""
    claims: List[dict] = []
