"""Tests for pipeline/stages.py — decomposed async stage functions.

Tests the stage helper functions and response builders using mock state dicts.
No LLM calls needed — tests verify the deterministic logic within stages.
"""
from __future__ import annotations

import asyncio

import pytest

from pipeline.models import PipelineResponse
from pipeline.orchestrator import compute_confidence
from pipeline.pipeline_state import PipelineState
from pipeline.stages import _base_response, _verify_text

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
            stage_arbiter,
            stage_build_prompts,
            stage_check_fast_paths,
            stage_decompose,
            stage_generate,
            stage_init,
            stage_nli,
            stage_rewrite_loop,
            stage_route,
            stage_sanitize,
            stage_search,
            stage_soft_retry,
            stage_verify,
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
        from pipeline.metrics import PipelineMetrics
        from pipeline.orchestrator import _noop_emit
        from pipeline.stages import stage_route
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
        from pipeline.metrics import PipelineMetrics
        from pipeline.orchestrator import _noop_emit
        from pipeline.stages import stage_route
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
        from pipeline.metrics import PipelineMetrics
        from pipeline.orchestrator import _noop_emit
        from pipeline.stages import stage_decompose
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


# ---------------------------------------------------------------------------
# Tests: stage_arbiter
# ---------------------------------------------------------------------------

class TestStageArbiter:
    """Test stage_arbiter with adaptive poisoning threshold and repair routing."""

    @pytest.mark.asyncio
    async def test_stage_arbiter_heavily_poisoned_routes_to_repair(self, monkeypatch):
        from pipeline.metrics import PipelineMetrics
        from pipeline.models import GPT3ResponseSchema
        from pipeline.orchestrator import _noop_emit
        from pipeline.stages import stage_arbiter

        # Mock structured LLM call to return ALLOW_WITH_EDITS
        fake_response = GPT3ResponseSchema(
            arbiter_decision="ALLOW_WITH_EDITS",
            rationale=["Minor issues"],
            edits_for_gpt1=[],
            final_policy_notes=[],
        )
        async def fake_call_structured(*args, **kwargs):
            return fake_response

        monkeypatch.setattr("pipeline.stages.call_llm_structured", fake_call_structured)

        state = {
            "emit": _noop_emit,
            "metrics": PipelineMetrics(request_id="test_arb", prompt_length=10),
            "gpt3_cfg": {"provider": "mock", "model": "mock"},
            "gpt3_system": "mock system",
            "flags": {},
            "prompt": "test prompt",
            "sanitized_output": "test output",
            "gpt2_raw": "{}",
            "claim_table": [
                {"claim": "Good", "category": "supported"},
                {"claim": "Bad 1", "category": "unsupported"},
                {"claim": "Bad 2", "category": "unsupported"},
            ],  # 2/3 = 66.7% > 35%
            "findings": [],
            "max_rewrite_loops": 2,
            "enable_repair": True,
        }

        updates = await stage_arbiter(state)
        assert updates["arbiter_invoked"] is True
        assert updates["arbiter_decision"] == "BLOCK"
        assert "early_return" not in updates  # routes to repair loop, no early return

    @pytest.mark.asyncio
    async def test_stage_arbiter_heavily_poisoned_early_return_when_repair_disabled(self, monkeypatch):
        from pipeline.metrics import PipelineMetrics
        from pipeline.models import GPT3ResponseSchema
        from pipeline.orchestrator import _noop_emit
        from pipeline.stages import stage_arbiter

        fake_response = GPT3ResponseSchema(
            arbiter_decision="BLOCK",
            rationale=["Poisoned draft"],
            edits_for_gpt1=[],
            final_policy_notes=[],
        )
        async def fake_call_structured(*args, **kwargs):
            return fake_response

        monkeypatch.setattr("pipeline.stages.call_llm_structured", fake_call_structured)

        state = {
            "emit": _noop_emit,
            "metrics": PipelineMetrics(request_id="test_arb", prompt_length=10),
            "gpt3_cfg": {"provider": "mock", "model": "mock"},
            "gpt3_system": "mock system",
            "flags": {},
            "prompt": "test prompt",
            "sanitized_output": "test output",
            "gpt2_raw": "{}",
            "gpt2_verdict": "FAIL",
            "gpt2_reasoning": ["Failed"],
            "violations": ["T1"],
            "claim_table": [
                {"claim": "Bad", "category": "unsupported", "justification": "None"},
            ],
            "findings": [{"type": "T1", "severity": "hard", "detail": "Fabricated citation"}],
            "max_rewrite_loops": 0,  # Repair disabled
            "enable_repair": False,
            "tier": "strict",
            "output_format": "auto",
        }

        updates = await stage_arbiter(state)
        assert updates["arbiter_decision"] == "BLOCK"
        assert "early_return" in updates
        assert updates["early_return"].final_verdict == "FAIL"

    @pytest.mark.asyncio
    async def test_stage_arbiter_lightly_poisoned_overrides_to_allow_with_edits(self, monkeypatch):
        from pipeline.metrics import PipelineMetrics
        from pipeline.models import GPT3ResponseSchema
        from pipeline.orchestrator import _noop_emit
        from pipeline.stages import stage_arbiter

        fake_response = GPT3ResponseSchema(
            arbiter_decision="BLOCK",
            rationale=["Over-blocking"],
            edits_for_gpt1=[],
            final_policy_notes=[],
        )
        async def fake_call_structured(*args, **kwargs):
            return fake_response

        monkeypatch.setattr("pipeline.stages.call_llm_structured", fake_call_structured)

        state = {
            "emit": _noop_emit,
            "metrics": PipelineMetrics(request_id="test_arb", prompt_length=10),
            "gpt3_cfg": {"provider": "mock", "model": "mock"},
            "gpt3_system": "mock system",
            "flags": {},
            "prompt": "test prompt",
            "sanitized_output": "test output",
            "gpt2_raw": "{}",
            "claim_table": [
                {"claim": "Supported fact", "category": "supported"},
                {"claim": "Observed fact", "category": "observed"},
                {"claim": "Unsupported fact", "category": "unsupported"},
            ],  # 1/3 = 33.3% <= 35%
            "findings": [{"type": "T1", "severity": "hard", "detail": "Single hard"}],
            "max_rewrite_loops": 2,
        }

        updates = await stage_arbiter(state)
        assert updates["arbiter_decision"] == "ALLOW_WITH_EDITS"
        assert "early_return" not in updates


# ---------------------------------------------------------------------------
# Tests: stage_verify Pre-Flight Deterministic Scanner
# ---------------------------------------------------------------------------

class TestStageVerifyPreflight:
    """Test deterministic pre-flight token & bounds scanner in stage_verify."""

    @pytest.mark.asyncio
    async def test_stage_verify_out_of_bounds_citation_short_circuits_llm(self, monkeypatch):
        from pipeline.metrics import PipelineMetrics
        from pipeline.models import SearchSource
        from pipeline.orchestrator import _noop_emit
        from pipeline.stages import stage_verify

        async def fake_call_structured(*args, **kwargs):
            raise AssertionError("Blind Verifier LLM called despite hard pre-flight violation!")

        monkeypatch.setattr("pipeline.stages.call_llm_structured", fake_call_structured)

        metrics = PipelineMetrics(request_id="test_preflight_1", prompt_length=10)
        state = {
            "emit": _noop_emit,
            "metrics": metrics,
            "gpt2_cfg": {"provider": "mock", "model": "mock"},
            "gpt2_system": "mock system",
            "prompt": "Test prompt",
            "sanitized_output": "According to reports, growth was strong [5].",
            "search_sources": [
                SearchSource(title="S1", url="http://s1.org", snippet="Growth details.", score=0.5),
            ],
            "flags": {},
            "tier": "strict",
            "atomic_claims": [{"text": "Growth was strong [5]."}],
        }

        updates = await stage_verify(state)
        assert updates["gpt2_verdict"] == "FAIL"
        assert updates["violations"] == ["T1"]
        assert len(updates["findings"]) >= 1
        assert "[5]" in updates["findings"][0]["detail"]
        assert metrics.gpt2_verdict == "FAIL"
        assert metrics.hard_findings >= 1

    @pytest.mark.asyncio
    async def test_stage_verify_unbacked_number_short_circuits_llm(self, monkeypatch):
        from pipeline.metrics import PipelineMetrics
        from pipeline.models import SearchSource
        from pipeline.orchestrator import _noop_emit
        from pipeline.stages import stage_verify

        async def fake_call_structured(*args, **kwargs):
            raise AssertionError("Blind Verifier LLM called despite fabricated number!")

        monkeypatch.setattr("pipeline.stages.call_llm_structured", fake_call_structured)

        metrics = PipelineMetrics(request_id="test_preflight_2", prompt_length=10)
        state = {
            "emit": _noop_emit,
            "metrics": metrics,
            "gpt2_cfg": {"provider": "mock", "model": "mock"},
            "gpt2_system": "mock system",
            "prompt": "Test prompt",
            "sanitized_output": "The basic subscription costs $150 per month [1].",
            "search_sources": [
                SearchSource(title="Pricing", url="http://pricing.org", snippet="Basic subscription costs $50 per month.", score=0.5),
            ],
            "flags": {},
            "tier": "strict",
            "atomic_claims": [{"text": "Basic subscription costs $150 per month."}],
        }

        updates = await stage_verify(state)
        assert updates["gpt2_verdict"] == "FAIL"
        assert any("150" in f["detail"] for f in updates["findings"])
        assert metrics.gpt2_verdict == "FAIL"

    @pytest.mark.asyncio
    async def test_stage_verify_zero_sources_with_citation_short_circuits_llm(self, monkeypatch):
        from pipeline.metrics import PipelineMetrics
        from pipeline.orchestrator import _noop_emit
        from pipeline.stages import stage_verify

        async def fake_call_structured(*args, **kwargs):
            raise AssertionError("Blind Verifier LLM called despite zero sources citation!")

        monkeypatch.setattr("pipeline.stages.call_llm_structured", fake_call_structured)

        metrics = PipelineMetrics(request_id="test_preflight_3", prompt_length=10)
        state = {
            "emit": _noop_emit,
            "metrics": metrics,
            "gpt2_cfg": {"provider": "mock", "model": "mock"},
            "gpt2_system": "mock system",
            "prompt": "Test prompt",
            "sanitized_output": "Quantum computing achieved supremacy [1].",
            "search_sources": [],
            "flags": {},
            "tier": "strict",
            "atomic_claims": [],
        }

        updates = await stage_verify(state)
        assert updates["gpt2_verdict"] == "FAIL"
        assert any("available sources: 0" in f["detail"] for f in updates["findings"])

    @pytest.mark.asyncio
    async def test_stage_verify_clean_draft_proceeds_to_llm(self, monkeypatch):
        from pipeline.metrics import PipelineMetrics
        from pipeline.models import GPT2ResponseSchema, SearchSource
        from pipeline.orchestrator import _noop_emit
        from pipeline.stages import stage_verify

        llm_called = False

        async def fake_call_structured(*args, **kwargs):
            nonlocal llm_called
            llm_called = True
            return GPT2ResponseSchema(
                reasoning_trace=["Valid claim verified."],
                claim_table=[{"claim": "Basic subscription costs $50 per month.", "category": "Observed", "justification": "Matches [1]"}],
                findings=[],
                verdict="PASS",
            )

        monkeypatch.setattr("pipeline.stages.call_llm_structured", fake_call_structured)

        metrics = PipelineMetrics(request_id="test_preflight_4", prompt_length=10)
        state = {
            "emit": _noop_emit,
            "metrics": metrics,
            "gpt2_cfg": {"provider": "mock", "model": "mock"},
            "gpt2_system": "mock system",
            "prompt": "Test prompt",
            "sanitized_output": "The basic subscription costs $50 per month [1].",
            "search_sources": [
                SearchSource(title="Pricing", url="http://pricing.org", snippet="Basic subscription costs $50 per month.", score=0.5),
            ],
            "flags": {},
            "tier": "strict",
            "output_format": "structured",
            "atomic_claims": [],
        }

        updates = await stage_verify(state)
        assert llm_called is True
        assert updates["gpt2_verdict"] == "PASS"
        assert "early_return" in updates


# ---------------------------------------------------------------------------
# Tests: Closed-Loop Negative Constraints & stage_rewrite_loop (Milestone 3)
# ---------------------------------------------------------------------------

class TestStageNegativeConstraints:
    """Unit tests for extract_negative_constraints and format_negative_constraints_block."""

    def test_extract_negative_constraints_tripwires(self):
        from pipeline.stages import extract_negative_constraints

        findings = [
            {"type": "T1", "severity": "hard", "detail": "Fabricated statute Code 999"},
            {"type": "T2", "severity": "soft", "detail": "Used typically without citation"},
            {"type": "T3", "severity": "hard", "detail": "Asserted causation between diet and lifespan"},
            {"type": "T4", "severity": "soft", "detail": "Ranked option A above option B"},
            {"type": "T5", "severity": "soft", "detail": "Guarantees 100% ROI"},
            {"type": "T6", "severity": "soft", "detail": "Reassurance phrase: do not worry"},
            {"type": "T7", "severity": "hard", "detail": "Stated 2030 price as factual"},
        ]
        constraints = extract_negative_constraints(findings)
        assert len(constraints) == 7
        assert any("DO NOT introduce fabricated" in c for c in constraints)
        assert any("DO NOT use typicality words" in c for c in constraints)
        assert any("DO NOT assert causal relationships" in c for c in constraints)
        assert any("DO NOT rank, rate, or compare" in c for c in constraints)
        assert any("DO NOT include outcome promises" in c for c in constraints)
        assert any("DO NOT use reassurance framing" in c for c in constraints)
        assert any("DO NOT present time-sensitive facts" in c for c in constraints)

    def test_extract_negative_constraints_arbiter_edits(self):
        from pipeline.models import EditEntry
        from pipeline.stages import extract_negative_constraints

        edits = [
            EditEntry(action="DELETE", target="Unbacked factual claim.", replacement=""),
            EditEntry(action="REWRITE", target="This will boost profit.", replacement="This may assist operations."),
            EditEntry(action="MOVE_TO_UNKNOWN", target="Success rate is 99%.", replacement="Success rate is unknown."),
        ]
        constraints = extract_negative_constraints([], arbiter_edits=edits)
        assert len(constraints) == 3
        assert 'DO NOT include the claim or text: "Unbacked factual claim."' in constraints
        assert 'DO NOT use the unverified phrasing: "This will boost profit." (replace with: "This may assist operations.")' in constraints
        assert 'DO NOT state as an established fact: "Success rate is 99%." (frame strictly as Unknown/Unverified)' in constraints

    def test_extract_negative_constraints_unbacked_numbers_and_citations(self):
        from pipeline.models import ClaimEntry
        from pipeline.stages import extract_negative_constraints

        findings = [
            {"type": "T1", "severity": "hard", "detail": "Fabricated citation: referenced non-existent source [5]"},
            {"type": "T1", "severity": "hard", "detail": "Unbacked numeric claim: [1] does not contain numeric value 150 from '$150M'"},
            {"type": "T1", "severity": "hard", "detail": "source [1] does not contain facts supporting statement 'profit increased'"},
        ]
        claims = [
            ClaimEntry(claim="Fabricated unbacked assertion", category="Unsupported", justification="No source"),
        ]
        constraints = extract_negative_constraints(findings, claim_table=claims, max_source_count=2)
        assert len(constraints) == 4
        assert "DO NOT cite non-existent source [5] (valid source indices are 1..2)." in constraints
        assert "DO NOT introduce the unbacked numeric figure 150." in constraints
        assert "DO NOT attribute statement 'profit increased' to source without direct evidence." in constraints
        assert 'DO NOT make the unbacked assertion: "Fabricated unbacked assertion"' in constraints

    def test_format_negative_constraints_block(self):
        from pipeline.stages import format_negative_constraints_block

        assert format_negative_constraints_block([]) == ""
        constraints = [
            "DO NOT cite non-existent source [5].",
            "DO NOT introduce the unbacked numeric figure $150M.",
        ]
        block = format_negative_constraints_block(constraints)
        assert block.startswith("### Negative Constraints\n")
        assert "- DO NOT cite non-existent source [5]." in block
        assert "- DO NOT introduce the unbacked numeric figure $150M." in block


class TestStageRewriteLoopClosedLoop:
    """Unit tests for stage_rewrite_loop multi-turn closed loop execution."""

    @pytest.mark.asyncio
    async def test_stage_rewrite_loop_monotonic_constraint_accumulation_turn1_and_turn2(self, monkeypatch):
        from pipeline.metrics import PipelineMetrics
        from pipeline.models import ClaimEntry, SearchSource
        from pipeline.orchestrator import _noop_emit
        from pipeline.stages import stage_rewrite_loop

        captured_prompts = []

        async def fake_call_llm(cfg, system, prompt, **kwargs):
            captured_prompts.append(prompt)
            if len(captured_prompts) == 1:
                return "Turn 1 rewrite with number 80 [1]."
            elif len(captured_prompts) == 2:
                return "Turn 2 clean rewrite without violations [1]."
            return "Fallback"

        verify_count = 0

        async def fake_verify_text(state, text):
            nonlocal verify_count
            verify_count += 1
            if verify_count == 1:
                # Turn 1 verification fails with new unbacked number 80
                return {
                    "gpt2_raw": "{}",
                    "claim_table": [ClaimEntry(claim="Contains 80", category="Unsupported", justification="Wrong number")],
                    "violations": ["T1"],
                    "verdict": "FAIL",
                    "findings": [{"type": "T1", "severity": "hard", "detail": "Unbacked numeric claim: [1] does not contain numeric value 80"}],
                    "reasoning": ["Numeric violation"],
                }
            else:
                # Turn 2 verification passes
                return {
                    "gpt2_raw": "{}",
                    "claim_table": [ClaimEntry(claim="Clean claim", category="Observed", justification="Cites [1]")],
                    "violations": [],
                    "verdict": "PASS",
                    "findings": [],
                    "reasoning": ["All clean"],
                }

        monkeypatch.setattr("pipeline.stages.call_llm_async", fake_call_llm)
        monkeypatch.setattr("pipeline.stages._verify_text", fake_verify_text)

        metrics = PipelineMetrics(request_id="test_m3_loop_1", prompt_length=10)
        state = {
            "emit": _noop_emit,
            "metrics": metrics,
            "gpt1_cfg": {"provider": "mock", "model": "mock"},
            "gpt1_system": "mock system",
            "prompt": "Explain revenue",
            "sanitized_output": "Corrupted initial text citing [5].",
            "gpt1_output": "Corrupted initial text citing [5].",
            "search_sources": [SearchSource(title="S1", url="http://s1.org", snippet="Clean facts.")],
            "findings": [{"type": "T1", "severity": "hard", "detail": "Fabricated citation: referenced non-existent source [5]"}],
            "arbiter_edits": [],
            "arbiter_decision": "ALLOW_WITH_EDITS",
            "flags": {},
            "tier": "strict",
            "max_rewrite_loops": 2,
        }

        result = await stage_rewrite_loop(state)
        assert "early_return" in result
        assert result["early_return"].final_verdict == "PASS"

        # Verify monotonic accumulation in state["negative_constraints"]
        assert len(state["negative_constraints"]) == 3
        assert any("source [5]" in c for c in state["negative_constraints"])
        assert any("numeric figure 80" in c for c in state["negative_constraints"])
        assert any('unbacked assertion: "Contains 80"' in c for c in state["negative_constraints"])

        # Verify Turn 2 prompt contained BOTH Turn 0 and Turn 1 constraints
        assert len(captured_prompts) == 2
        turn2_prompt = captured_prompts[1]
        assert "### Negative Constraints" in turn2_prompt
        assert "DO NOT cite non-existent source [5]" in turn2_prompt
        assert "DO NOT introduce the unbacked numeric figure 80." in turn2_prompt

    @pytest.mark.asyncio
    async def test_stage_rewrite_loop_regenerate_mode_on_block(self, monkeypatch):
        from pipeline.metrics import PipelineMetrics
        from pipeline.models import ClaimEntry, SearchSource
        from pipeline.orchestrator import _noop_emit
        from pipeline.stages import stage_rewrite_loop

        captured_prompts = []

        async def fake_call_llm(cfg, system, prompt, **kwargs):
            captured_prompts.append(prompt)
            return "Fresh clean draft [1]."

        async def fake_verify_text(state, text):
            return {
                "gpt2_raw": "{}",
                "claim_table": [ClaimEntry(claim="Fresh clean draft", category="Observed", justification="Cites [1]")],
                "violations": [],
                "verdict": "PASS",
                "findings": [],
                "reasoning": ["Clean"],
            }

        monkeypatch.setattr("pipeline.stages.call_llm_async", fake_call_llm)
        monkeypatch.setattr("pipeline.stages._verify_text", fake_verify_text)

        metrics = PipelineMetrics(request_id="test_m3_block_regen", prompt_length=10)
        state = {
            "emit": _noop_emit,
            "metrics": metrics,
            "gpt1_cfg": {"provider": "mock", "model": "mock"},
            "gpt1_system": "mock system",
            "prompt": "Explain company policy",
            "sanitized_output": "Heavily poisoned text.",
            "gpt1_output": "Heavily poisoned text.",
            "search_sources": [SearchSource(title="Policy", url="http://pol.org", snippet="Policy details.")],
            "findings": [{"type": "T1", "severity": "hard", "detail": "Fabricated entity XYZ"}],
            "arbiter_edits": [],
            "arbiter_decision": "BLOCK",
            "flags": {},
            "tier": "strict",
            "max_rewrite_loops": 2,
        }

        result = await stage_rewrite_loop(state)
        assert "early_return" in result
        assert result["early_return"].final_verdict == "PASS"

        # Check that prompt was in REGENERATE mode (asking for a fresh response)
        turn1_prompt = captured_prompts[0]
        assert "Your previous response was rejected due to heavy poisoning" in turn1_prompt
        assert "### Negative Constraints" in turn1_prompt
        assert "DO NOT introduce fabricated" in turn1_prompt
        assert "Please generate a completely fresh response" in turn1_prompt

    @pytest.mark.asyncio
    async def test_stage_rewrite_loop_fail_closed_fallback_after_two_turns(self, monkeypatch):
        from pipeline.metrics import PipelineMetrics
        from pipeline.models import ClaimEntry, SearchSource
        from pipeline.orchestrator import _noop_emit
        from pipeline.stages import stage_rewrite_loop

        call_count = 0

        async def fake_call_llm(cfg, system, prompt, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "Turn 1 failing draft"
            elif call_count == 2:
                return "Turn 2 failing draft"
            return "Unknown(Actionable): Facts could not be verified."

        async def fake_verify_text(state, text):
            return {
                "gpt2_raw": "{}",
                "claim_table": [ClaimEntry(claim="Bad claim", category="Unsupported", justification="None")],
                "violations": ["T1"],
                "verdict": "FAIL",
                "findings": [{"type": "T1", "severity": "hard", "detail": "Persistent violation"}],
                "reasoning": ["Failed"],
            }

        monkeypatch.setattr("pipeline.stages.call_llm_async", fake_call_llm)
        monkeypatch.setattr("pipeline.stages._verify_text", fake_verify_text)

        metrics = PipelineMetrics(request_id="test_m3_fallback", prompt_length=10)
        state = {
            "emit": _noop_emit,
            "metrics": metrics,
            "gpt1_cfg": {"provider": "mock", "model": "mock"},
            "gpt1_system": "mock system",
            "prompt": "Test query",
            "sanitized_output": "Initial bad draft",
            "gpt1_output": "Initial bad draft",
            "search_sources": [],
            "findings": [{"type": "T1", "severity": "hard", "detail": "Persistent violation"}],
            "arbiter_edits": [],
            "arbiter_decision": "ALLOW_WITH_EDITS",
            "flags": {},
            "tier": "strict",
            "max_rewrite_loops": 2,
        }

        result = await stage_rewrite_loop(state)
        assert "early_return" in result
        resp = result["early_return"]
        assert resp.final_verdict == "PASS"
        assert resp.confidence.confidence_label == "Low"
        assert metrics.convergence_outcome == "fallback"
        assert metrics.rewrite_loops == 2
