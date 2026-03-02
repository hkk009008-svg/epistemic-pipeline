"""Tests for pipeline/orchestrator.py — deterministic logic paths.

Focuses on pure functions (compute_confidence, _fail_message) and
mocked pipeline flow (run_pipeline path coverage).
"""
from __future__ import annotations

import pytest

from pipeline.models import ClaimEntry, PipelineRequest
from pipeline.orchestrator import compute_confidence, _fail_message, clean_for_display


# ---------------------------------------------------------------------------
# compute_confidence — pure function tests (no mocking needed)
# ---------------------------------------------------------------------------


class TestComputeConfidence:
    """Tests for the confidence breakdown calculator."""

    def test_empty_claim_table_returns_defaults(self):
        """Empty claim table should return all-zero defaults without crashing."""
        result = compute_confidence([])
        assert result.total_claims == 0
        assert result.observed_pct == 0.0
        assert result.confidence_label == "Unknown"

    def test_all_observed_yields_high(self):
        """100% observed claims → High confidence."""
        claims = [ClaimEntry(claim="Water boils at 100C.", category="Supported", justification="Physics.")]
        result = compute_confidence(claims)
        assert result.observed_pct == 100.0
        assert result.confidence_label == "High"

    def test_threshold_70_is_high(self):
        """Exactly 70% observed → High (boundary test for >= 70)."""
        claims = [
            ClaimEntry(claim=f"Claim {i}", category="Supported", justification="J") for i in range(7)
        ] + [
            ClaimEntry(claim=f"Claim {i}", category="Inference", justification="J") for i in range(3)
        ]
        result = compute_confidence(claims)
        assert result.observed_pct == 70.0
        assert result.confidence_label == "High"

    def test_threshold_40_is_medium(self):
        """40% observed → Medium (boundary test for >= 40)."""
        claims = [
            ClaimEntry(claim=f"Claim {i}", category="Observed", justification="J") for i in range(4)
        ] + [
            ClaimEntry(claim=f"Claim {i}", category="Unsupported", justification="J") for i in range(6)
        ]
        result = compute_confidence(claims)
        assert result.observed_pct == 40.0
        assert result.confidence_label == "Medium"

    def test_threshold_20_is_low(self):
        """20% observed → Low (boundary test for >= 20)."""
        claims = [
            ClaimEntry(claim="Obs", category="Supported", justification="J"),
        ] + [
            ClaimEntry(claim=f"Claim {i}", category="Hypothesis", justification="J") for i in range(4)
        ]
        result = compute_confidence(claims)
        assert result.observed_pct == 20.0
        assert result.confidence_label == "Low"

    def test_below_20_is_unknown(self):
        """19% observed → Unknown."""
        claims = [
            ClaimEntry(claim="Obs", category="Supported", justification="J"),
        ] + [
            ClaimEntry(claim=f"Claim {i}", category="Unsupported", justification="J") for i in range(99)
        ]
        result = compute_confidence(claims)
        assert result.observed_pct < 20.0
        assert result.confidence_label == "Unknown"

    def test_category_case_insensitive(self):
        """Categories should be case-insensitive: 'SUPPORTED' == 'supported' == 'Supported'."""
        claims = [
            ClaimEntry(claim="A", category="SUPPORTED", justification="J"),
            ClaimEntry(claim="B", category="observed", justification="J"),
            ClaimEntry(claim="C", category="Supported", justification="J"),
        ]
        result = compute_confidence(claims)
        assert result.observed_pct == 100.0
        assert result.total_claims == 3

    def test_user_provided_category(self):
        """'user-provided' category should be tracked separately."""
        claims = [
            ClaimEntry(claim="User said X", category="user-provided", justification="From user"),
        ]
        result = compute_confidence(claims)
        assert result.user_provided_pct == 100.0
        assert result.observed_pct == 0.0

    def test_all_categories_counted(self):
        """All 5 category types should sum to 100%."""
        claims = [
            ClaimEntry(claim="A", category="Supported", justification="J"),
            ClaimEntry(claim="B", category="Inference", justification="J"),
            ClaimEntry(claim="C", category="Hypothesis", justification="J"),
            ClaimEntry(claim="D", category="Unsupported", justification="J"),
            ClaimEntry(claim="E", category="user-provided", justification="J"),
        ]
        result = compute_confidence(claims)
        total = (
            result.observed_pct + result.inference_pct + result.hypothesis_pct
            + result.unsupported_pct + result.user_provided_pct
        )
        assert total == 100.0


# ---------------------------------------------------------------------------
# _fail_message — pure function tests
# ---------------------------------------------------------------------------


class TestFailMessage:
    """Tests for the user-facing failure message helper."""

    def test_current_events_without_search(self):
        """When current_events=True and no search, message should mention Tavily."""
        msg = _fail_message({"current_events": True}, search_performed=False)
        assert "Tavily" in msg
        assert "web search" in msg.lower()

    def test_current_events_with_search(self):
        """When search was performed, return generic fail message."""
        msg = _fail_message({"current_events": True}, search_performed=True)
        assert "NO PASS" in msg

    def test_no_current_events(self):
        """Non-current-events failure returns generic message."""
        msg = _fail_message({"current_events": False}, search_performed=False)
        assert "NO PASS" in msg

    def test_empty_flags(self):
        """Empty flags dict doesn't crash."""
        msg = _fail_message({}, search_performed=False)
        assert "NO PASS" in msg


# ---------------------------------------------------------------------------
# PipelineRequest model_post_init — prompt override gating
# ---------------------------------------------------------------------------


class TestPipelineRequestPromptOverride:
    """Verify that system prompt overrides are blocked in production mode."""

    def test_overrides_stripped_by_default(self, monkeypatch):
        """Without ALLOW_PROMPT_OVERRIDE, system prompt fields are cleared."""
        monkeypatch.delenv("ALLOW_PROMPT_OVERRIDE", raising=False)
        req = PipelineRequest(prompt="test", gpt1_system="evil", gpt2_system="evil", gpt3_system="evil")
        assert req.gpt1_system == ""
        assert req.gpt2_system == ""
        assert req.gpt3_system == ""

    def test_overrides_allowed_in_dev(self, monkeypatch):
        """With ALLOW_PROMPT_OVERRIDE=true, system prompt fields are preserved."""
        monkeypatch.setenv("ALLOW_PROMPT_OVERRIDE", "true")
        req = PipelineRequest(prompt="test", gpt1_system="custom1", gpt2_system="custom2", gpt3_system="custom3")
        assert req.gpt1_system == "custom1"
        assert req.gpt2_system == "custom2"
        assert req.gpt3_system == "custom3"

    def test_prompt_max_length(self):
        """Prompt exceeding max_length should raise validation error."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            PipelineRequest(prompt="x" * 10_001)


# ---------------------------------------------------------------------------
# _resolve_output_format — auto-resolution tests
# ---------------------------------------------------------------------------


class TestResolveOutputFormat:
    """Tests for the auto-resolution of output_format from tier."""

    def test_auto_strict_gives_structured(self):
        from pipeline.orchestrator import _resolve_output_format
        assert _resolve_output_format("strict", "auto") == "structured"

    def test_auto_standard_gives_annotated(self):
        from pipeline.orchestrator import _resolve_output_format
        assert _resolve_output_format("standard", "auto") == "annotated"

    def test_auto_light_gives_concise(self):
        from pipeline.orchestrator import _resolve_output_format
        assert _resolve_output_format("light", "auto") == "concise"

    def test_explicit_overrides_auto(self):
        from pipeline.orchestrator import _resolve_output_format
        assert _resolve_output_format("light", "structured") == "structured"
        assert _resolve_output_format("strict", "concise") == "concise"
        assert _resolve_output_format("standard", "structured") == "structured"


# ---------------------------------------------------------------------------
# PipelineRequest — tier and output_format validation
# ---------------------------------------------------------------------------


class TestPipelineRequestTierValidation:
    """Verify that PipelineRequest validates tier and output_format fields."""

    def test_valid_tiers(self):
        for t in ("strict", "standard", "light"):
            req = PipelineRequest(prompt="test", tier=t)
            assert req.tier == t

    def test_invalid_tier_rejected(self):
        with pytest.raises(Exception):
            PipelineRequest(prompt="test", tier="ultra")

    def test_valid_output_formats(self):
        for f in ("auto", "structured", "annotated", "concise"):
            req = PipelineRequest(prompt="test", output_format=f)
            assert req.output_format == f

    def test_invalid_output_format_rejected(self):
        with pytest.raises(Exception):
            PipelineRequest(prompt="test", output_format="fancy")

    def test_defaults(self):
        req = PipelineRequest(prompt="test")
        assert req.tier == "strict"
        assert req.output_format == "auto"


# ---------------------------------------------------------------------------
# clean_for_display — strip internal sanitizer markers from user output
# ---------------------------------------------------------------------------


class TestCleanForDisplay:
    """Verify that internal sanitizer markers are stripped before user display."""

    def test_typicality_marker_removed(self):
        text = "Time is [Typicality language removed] considered a fundamental concept."
        result = clean_for_display(text)
        assert "[Typicality language removed]" not in result
        assert "Time is" in result

    def test_unverified_generalization_removed(self):
        text = "[Unverified generalization removed] that exercise is healthy."
        result = clean_for_display(text)
        assert "[Unverified generalization removed]" not in result

    def test_stale_marker_removed(self):
        text = "The rate was [Stale — verify current status from an authoritative source] per year."
        result = clean_for_display(text)
        assert "[Stale" not in result

    def test_legal_marker_removed(self):
        text = "This [Legal claim requires citation] under federal law."
        result = clean_for_display(text)
        assert "[Legal claim requires citation]" not in result

    def test_multiple_markers_removed(self):
        text = "[Typicality language removed] time is [Unverified generalization removed] relative."
        result = clean_for_display(text)
        assert "[Typicality" not in result
        assert "[Unverified" not in result
        assert "time is" in result
        assert "relative" in result

    def test_no_markers_returns_unchanged(self):
        text = "A limited liability company (LLC) is a business structure."
        assert clean_for_display(text) == text

    def test_double_spaces_cleaned(self):
        text = "Hello  [Typicality language removed]  world."
        result = clean_for_display(text)
        assert "  " not in result

    def test_blank_lines_collapsed(self):
        text = "Line one.\n\n\n\nLine two."
        result = clean_for_display(text)
        assert "\n\n\n" not in result
        assert "Line one." in result
        assert "Line two." in result
