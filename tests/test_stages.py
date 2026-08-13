"""Tests for pipeline/stages.py — decomposed async stage functions.

Tests the stage helper functions and response builders using mock state dicts.
No LLM calls needed — tests verify the deterministic logic within stages.
"""
from __future__ import annotations

import asyncio

import pytest

from pipeline.stages import _base_response, _verify_text
from pipeline.pipeline_state import PipelineState
from pipeline.models import PipelineResponse
from pipeline.orchestrator import compute_confidence


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_state():
    """Minimal PipelineState for testing response builders."""
    return {
        "prompt": "What is an LLC?",
        "tier": "strict",
        "output_format": "structured",
        "gpt1_output": "An LLC is a limited liability company.",
        "flags": {"advice_mode": True},
        "search_kwargs": {
            "search_performed": False,
            "search_attempted": False,
            "search_note": "",
            "search_query": "",
            "search_sources": [],
        },
        "decomp_kwargs": {
            "atomic_claims": [],
            "decomposition_ran": False,
        },
        "empty_arbiter": {
            "arbiter_invoked": False, "arbiter_decision": "", "arbiter_rationale": [],
            "arbiter_edits": [], "arbiter_policy_notes": [], "arbiter_raw": "",
            "rewrite_occurred": False, "rewrite_output": "", "rewrite_gpt2_raw": "",
            "rewrite_claim_table": [], "rewrite_violations": [], "rewrite_verdict": "",
        },
    }


# ---------------------------------------------------------------------------
# Tests: _base_response
# ---------------------------------------------------------------------------

class TestBaseResponse:
    """Tests for the _base_response helper."""

    def test_returns_pipeline_response(self, minimal_state):
        resp = _base_response(
            minimal_state,
            bypassed=True,
            gpt2_raw="(bypassed)",
            claim_table=[],
            violations=[],
            gpt2_verdict="PASS",
            final_verdict="PASS",
            final_result="test output",
            sanitizer_applied=False,
            confidence=compute_confidence([]),
        )
        assert isinstance(resp, PipelineResponse)
        assert resp.final_verdict == "PASS"
        assert resp.final_result == "test output"
        assert resp.bypassed is True

    def test_includes_search_kwargs(self, minimal_state):
        resp = _base_response(
            minimal_state,
            bypassed=False,
            gpt2_raw="{}",
            claim_table=[],
            violations=[],
            gpt2_verdict="PASS",
            final_verdict="PASS",
            final_result="output",
            sanitizer_applied=False,
            confidence=compute_confidence([]),
        )
        assert resp.search_performed is False
        assert resp.search_sources == []

    def test_includes_empty_arbiter_defaults(self, minimal_state):
        resp = _base_response(
            minimal_state,
            bypassed=False,
            gpt2_raw="{}",
            claim_table=[],
            violations=[],
            gpt2_verdict="PASS",
            final_verdict="PASS",
            final_result="output",
            sanitizer_applied=False,
            confidence=compute_confidence([]),
        )
        assert resp.arbiter_invoked is False
        assert resp.rewrite_occurred is False

    def test_overrides_take_precedence(self, minimal_state):
        resp = _base_response(
            minimal_state,
            bypassed=False,
            gpt2_raw="{}",
            claim_table=[],
            violations=[],
            gpt2_verdict="FAIL",
            final_verdict="FAIL",
            final_result="blocked",
            sanitizer_applied=True,
            confidence=compute_confidence([]),
            arbiter_invoked=True,
            arbiter_decision="BLOCK",
        )
        assert resp.arbiter_invoked is True
        assert resp.arbiter_decision == "BLOCK"
        assert resp.final_verdict == "FAIL"

    def test_prompt_version_set(self, minimal_state):
        resp = _base_response(
            minimal_state,
            bypassed=True,
            gpt2_raw="(bypassed)",
            claim_table=[],
            violations=[],
            gpt2_verdict="PASS",
            final_verdict="PASS",
            final_result="test",
            sanitizer_applied=False,
            confidence=compute_confidence([]),
        )
        assert resp.prompt_version != ""

    def test_tier_from_state(self, minimal_state):
        minimal_state["tier"] = "light"
        resp = _base_response(
            minimal_state,
            bypassed=True,
            gpt2_raw="(bypassed)",
            claim_table=[],
            violations=[],
            gpt2_verdict="PASS",
            final_verdict="PASS",
            final_result="test",
            sanitizer_applied=False,
            confidence=compute_confidence([]),
        )
        assert resp.tier == "light"


# ---------------------------------------------------------------------------
# Tests: PipelineState TypedDict
# ---------------------------------------------------------------------------

class TestPipelineState:
    """Tests for PipelineState TypedDict construction."""

    def test_minimal_construction(self):
        """PipelineState with total=False allows minimal construction."""
        state: PipelineState = {"prompt": "test", "tier": "strict"}
        assert state["prompt"] == "test"

    def test_all_fields_optional(self):
        """All fields should be optional (total=False)."""
        state: PipelineState = {}
        assert isinstance(state, dict)

    def test_update_pattern(self):
        """The state.update(await stage()) pattern should work."""
        state: PipelineState = {"prompt": "test"}
        updates = {"flags": {"advice_mode": True}, "tier": "standard"}
        state.update(updates)
        assert state["flags"]["advice_mode"] is True
        assert state["tier"] == "standard"


# ---------------------------------------------------------------------------
# Tests: stage_init (import verification)
# ---------------------------------------------------------------------------

class TestStageImports:
    """Verify all stage functions are importable."""

    def test_all_stages_importable(self):
        from pipeline.stages import (
            stage_init,
            stage_route,
            stage_search,
            stage_build_prompts,
            stage_generate,
            stage_check_fast_paths,
            stage_sanitize,
            stage_decompose,
            stage_nli,
            stage_verify,
            stage_soft_retry,
            stage_arbiter,
            stage_rewrite_loop,
        )
        # All should be async functions
        assert asyncio.iscoroutinefunction(stage_init)
        assert asyncio.iscoroutinefunction(stage_route)
        assert asyncio.iscoroutinefunction(stage_search)
        assert asyncio.iscoroutinefunction(stage_build_prompts)
        assert asyncio.iscoroutinefunction(stage_generate)
        assert asyncio.iscoroutinefunction(stage_check_fast_paths)
        assert asyncio.iscoroutinefunction(stage_sanitize)
        assert asyncio.iscoroutinefunction(stage_decompose)
        assert asyncio.iscoroutinefunction(stage_nli)
        assert asyncio.iscoroutinefunction(stage_verify)
        assert asyncio.iscoroutinefunction(stage_soft_retry)
        assert asyncio.iscoroutinefunction(stage_arbiter)
        assert asyncio.iscoroutinefunction(stage_rewrite_loop)

    def test_verify_text_importable(self):
        assert asyncio.iscoroutinefunction(_verify_text)

    def test_base_response_importable(self):
        from pipeline.stages import _base_response
        assert callable(_base_response)


# ---------------------------------------------------------------------------
# Tests: stage_sanitize (deterministic, no LLM)
# ---------------------------------------------------------------------------

class TestStageSanitize:
    """Test the sanitize stage with real inputs."""

    @pytest.mark.asyncio
    async def test_sanitize_marks_applied(self):
        from pipeline.stages import stage_sanitize
        state = {
            "gpt1_output": "Studies suggest this is true 50%",
            "flags": {},
            "tier": "strict",
        }
        result = await stage_sanitize(state)
        assert "sanitized_output" in result
        assert "sanitizer_applied" in result

    @pytest.mark.asyncio
    async def test_sanitize_no_change_marks_false(self):
        from pipeline.stages import stage_sanitize
        # Simple text with nothing to sanitize
        state = {
            "gpt1_output": "The sky is blue.",
            "flags": {},
            "tier": "strict",
        }
        result = await stage_sanitize(state)
        assert result["sanitizer_applied"] is False


# ---------------------------------------------------------------------------
# Tests: stage_route (deterministic, no LLM)
# ---------------------------------------------------------------------------

class TestStageRoute:
    """Test the route stage."""

    @pytest.mark.asyncio
    async def test_route_extracts_flags(self):
        from pipeline.stages import stage_route
        from pipeline.orchestrator import _noop_emit
        from pipeline.metrics import PipelineMetrics
        state = {
            "prompt": "What should I do about my legal case?",
            "emit": _noop_emit,
            "metrics": PipelineMetrics(request_id="test", prompt_length=10),
        }
        result = await stage_route(state)
        assert "flags" in result
        assert isinstance(result["flags"], dict)

    @pytest.mark.asyncio
    async def test_route_returns_dict(self):
        from pipeline.stages import stage_route
        from pipeline.orchestrator import _noop_emit
        from pipeline.metrics import PipelineMetrics
        state = {
            "prompt": "Hello",
            "emit": _noop_emit,
            "metrics": PipelineMetrics(request_id="test", prompt_length=5),
        }
        result = await stage_route(state)
        assert isinstance(result, dict)
        assert "flags" in result


# ---------------------------------------------------------------------------
# Tests: stage_decompose (no LLM, but checks conditional logic)
# ---------------------------------------------------------------------------

class TestStageDecompose:
    """Test the decompose stage conditional execution."""

    @pytest.mark.asyncio
    async def test_decompose_skips_when_not_warranted(self):
        from pipeline.stages import stage_decompose
        from pipeline.orchestrator import _noop_emit
        from pipeline.metrics import PipelineMetrics
        state = {
            "flags": {},  # no high-stakes flags
            "emit": _noop_emit,
            "metrics": PipelineMetrics(request_id="test", prompt_length=10),
            "gpt2_cfg": {},
            "sanitized_output": "test",
            "prompt": "test",
        }
        result = await stage_decompose(state)
        assert result["atomic_claims"] == []
        assert result["decomposition_ran"] is False
        assert result["decomp_kwargs"] == {"atomic_claims": [], "decomposition_ran": False}
