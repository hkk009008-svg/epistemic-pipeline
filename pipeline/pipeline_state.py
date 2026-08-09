"""PipelineState — typed dict accumulating state across pipeline stages.

Each async stage function reads what it needs from the state and returns
a dict of updated fields. The orchestrator merges updates via state.update().

Using total=False so stages only need to set what they produce.
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional, TypedDict

from pipeline.models import ConfidenceBreakdown, PipelineRequest, SearchSource


class PipelineState(TypedDict, total=False):
    """Accumulated state flowing through the async pipeline stages."""

    # -- Inputs --
    request: PipelineRequest
    emit: Callable  # StageEventEmitter
    prompt: str  # req.prompt shortcut

    # -- Config (resolved once) --
    gpt1_cfg: dict
    gpt2_cfg: dict
    gpt3_cfg: dict
    tier: str
    output_format: str

    # -- Routing --
    flags: dict

    # -- Search --
    search_sources: List[SearchSource]
    search_context: str
    search_performed: bool
    search_attempted: bool
    search_note: str
    src_kw_sets: Optional[list]

    # -- System prompts (assembled from flags + date + augmentation) --
    gpt1_system: str
    gpt2_system: str
    gpt3_system: str
    gpt1_user_content: str

    # -- Response-builder kwargs (computed once) --
    search_kwargs: dict
    decomp_kwargs: dict
    empty_arbiter: dict

    # -- GPT-1 output --
    gpt1_output: str

    # -- Sanitization --
    sanitized_output: str
    sanitizer_applied: bool

    # -- Decomposition --
    atomic_claims: list
    decomposition_ran: bool

    # -- NLI --
    nli_grounding: dict
    nli_unsupported_spans: list

    # -- GPT-2 verification --
    gpt2_raw: str
    claim_table: list
    violations: list
    gpt2_verdict: str
    findings: list
    gpt2_reasoning: list

    # -- Arbiter (GPT-3) --
    arbiter_invoked: bool
    arbiter_decision: str
    arbiter_rationale: list
    arbiter_edits: list
    arbiter_policy_notes: list
    gpt3_raw: str

    # -- Rewrite loop --
    rewrite_occurred: bool
    rewrite_output: str
    rewrite_gpt2_raw: str
    rewrite_claim_table: list
    rewrite_violations: list
    rewrite_verdict: str
    rewrite_reasoning: list
    findings_history: list
    max_rewrite_loops: int

    # -- Final --
    final_verdict: str
    final_result: str
    confidence: ConfidenceBreakdown
    meta_verification: Optional[dict]

    # -- Metrics --
    metrics: Any  # PipelineMetrics (avoid circular import)
