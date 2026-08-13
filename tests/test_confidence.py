"""Tests for compute_confidence() and verdict labels — upgraded with NLI grounding rate."""
from __future__ import annotations

from pipeline.orchestrator import compute_confidence, _finalize_response
from pipeline.models import ClaimEntry, ConfidenceBreakdown, PipelineResponse, SearchSource, compute_verdict_label
from pipeline.search import rank_sources


def _claims(categories: list[str]) -> list[ClaimEntry]:
    """Helper to build a list of ClaimEntry with given categories."""
    return [ClaimEntry(claim=f"c{i}", category=cat, justification="") for i, cat in enumerate(categories)]


class TestComputeConfidenceBasic:
    """Test basic category-based confidence scoring."""

    def test_empty_claims(self):
        result = compute_confidence([])
        assert result.confidence_label == "Unknown"
        assert result.total_claims == 0

    def test_all_observed(self):
        result = compute_confidence(_claims(["Observed"] * 5))
        assert result.observed_pct == 100.0
        assert result.confidence_label == "High"

    def test_all_unsupported(self):
        result = compute_confidence(_claims(["Unsupported"] * 5))
        assert result.unsupported_pct == 100.0
        assert result.confidence_label == "Unknown"

    def test_mixed_high(self):
        # 80% observed = High
        result = compute_confidence(_claims(["Observed"] * 4 + ["Inference"]))
        assert result.confidence_label == "High"

    def test_mixed_medium(self):
        # 50% observed = Medium
        result = compute_confidence(_claims(["Observed"] * 5 + ["Unsupported"] * 5))
        assert result.confidence_label == "Medium"

    def test_mixed_low(self):
        # 25% observed = Low
        result = compute_confidence(_claims(["Observed"] + ["Unsupported"] * 3))
        assert result.confidence_label == "Low"

    def test_supported_alias(self):
        result = compute_confidence(_claims(["Supported"] * 5))
        assert result.observed_pct == 100.0
        assert result.confidence_label == "High"

    def test_user_provided_category(self):
        result = compute_confidence(_claims(["user-provided"] * 3))
        assert result.user_provided_pct == 100.0


class TestComputeConfidenceHardFindings:
    """Test hard findings penalty on confidence."""

    def test_hard_finding_drops_high_to_medium(self):
        findings = [{"type": "T1", "severity": "hard", "detail": "fab stat"}]
        result = compute_confidence(_claims(["Observed"] * 5), findings=findings)
        # 100% observed but 1 hard finding: can't be High
        assert result.confidence_label == "Medium"

    def test_two_hard_findings_drops_to_low(self):
        findings = [
            {"type": "T1", "severity": "hard", "detail": "a"},
            {"type": "T2", "severity": "hard", "detail": "b"},
        ]
        result = compute_confidence(_claims(["Observed"] * 5), findings=findings)
        assert result.confidence_label == "Low"

    def test_soft_findings_no_penalty(self):
        findings = [{"type": "Overconfidence", "severity": "soft", "detail": "x"}]
        result = compute_confidence(_claims(["Observed"] * 5), findings=findings)
        assert result.confidence_label == "High"

    def test_legacy_fabricated_statistic_uses_t1_weight(self):
        """Legacy type names must not get the default 1.0 weight."""
        legacy = [{"type": "Fabricated statistic", "severity": "hard", "detail": "73%"}]
        coded = [{"type": "T1", "severity": "hard", "detail": "73%"}]
        a = compute_confidence(_claims(["Observed"] * 5), findings=legacy)
        b = compute_confidence(_claims(["Observed"] * 5), findings=coded)
        assert a.confidence_label == b.confidence_label == "Medium"

    def test_t7_hard_blocks_high_confidence(self):
        findings = [{"type": "T7", "severity": "hard", "detail": "stale rate"}]
        result = compute_confidence(_claims(["Observed"] * 5), findings=findings)
        assert result.confidence_label == "Medium"


class TestComputeConfidenceNLIGrounding:
    """Test NLI grounding rate integration into confidence."""

    def test_grounding_info_populated(self):
        grounding = {
            "grounding_rate": 0.8,
            "grounded_count": 4,
            "ungrounded_count": 1,
            "contradicted_count": 0,
            "neutral_count": 1,
            "total_evaluated": 5,
        }
        result = compute_confidence(_claims(["Observed"] * 5), nli_grounding=grounding)
        assert result.grounding is not None
        assert result.grounding.grounding_rate == 0.8
        assert result.grounding.grounded_count == 4

    def test_contradicted_claims_downgrade(self):
        grounding = {
            "grounding_rate": 0.6,
            "grounded_count": 3,
            "ungrounded_count": 2,
            "contradicted_count": 1,
            "neutral_count": 1,
            "total_evaluated": 5,
        }
        result = compute_confidence(_claims(["Observed"] * 5), nli_grounding=grounding)
        # High would be downgraded to Medium due to contradiction
        assert result.confidence_label == "Medium"

    def test_low_grounding_rate_caps_confidence(self):
        grounding = {
            "grounding_rate": 0.2,
            "grounded_count": 1,
            "ungrounded_count": 4,
            "contradicted_count": 0,
            "neutral_count": 4,
            "total_evaluated": 5,
        }
        result = compute_confidence(_claims(["Observed"] * 5), nli_grounding=grounding)
        # High/Medium capped to Low by low grounding rate
        assert result.confidence_label == "Low"

    def test_high_grounding_rescues_low(self):
        grounding = {
            "grounding_rate": 0.8,
            "grounded_count": 4,
            "ungrounded_count": 1,
            "contradicted_count": 0,
            "neutral_count": 1,
            "total_evaluated": 5,
        }
        # 25% observed (Low), but high grounding can rescue to Medium
        result = compute_confidence(
            _claims(["Observed"] + ["Inference"] * 3),
            nli_grounding=grounding,
        )
        assert result.confidence_label == "Medium"

    def test_grounding_no_effect_without_evaluation(self):
        grounding = {
            "grounding_rate": 0.0,
            "grounded_count": 0,
            "ungrounded_count": 0,
            "contradicted_count": 0,
            "neutral_count": 0,
            "total_evaluated": 0,
        }
        result = compute_confidence(_claims(["Observed"] * 5), nli_grounding=grounding)
        # No evaluated claims means grounding has no effect
        assert result.confidence_label == "High"

    def test_empty_claims_with_grounding(self):
        grounding = {
            "grounding_rate": 0.5,
            "grounded_count": 2,
            "ungrounded_count": 2,
            "contradicted_count": 0,
            "neutral_count": 2,
            "total_evaluated": 4,
        }
        result = compute_confidence([], nli_grounding=grounding)
        assert result.grounding is not None
        assert result.grounding.grounding_rate == 0.5


class TestComputeConfidenceUnsupportedSpans:
    """Test unsupported spans in confidence output."""

    def test_spans_included(self):
        spans = [
            {"text": "Claim X", "start": 10, "end": 20, "reason": "contradicted_by_evidence", "confidence_tier": "strong_contradiction"},
        ]
        result = compute_confidence(_claims(["Observed"] * 3), unsupported_spans=spans)
        assert len(result.unsupported_spans) == 1
        assert result.unsupported_spans[0].reason == "contradicted_by_evidence"

    def test_no_spans_when_none(self):
        result = compute_confidence(_claims(["Observed"] * 3))
        assert result.unsupported_spans == []

    def test_no_position_bias(self):
        # Verify that position weighting has been removed
        # All categories should have equal weight regardless of position
        claims = _claims(["Observed"] * 3 + ["Unsupported"] * 7)
        result = compute_confidence(claims)
        assert result.observed_pct == 30.0
        assert result.unsupported_pct == 70.0


class TestVerdictLabels:
    """Test human-readable verdict label generation."""

    def test_pass_high_is_verified(self):
        assert compute_verdict_label("PASS", "High") == "Verified with evidence"

    def test_pass_medium_is_partial(self):
        assert compute_verdict_label("PASS", "Medium") == "Partially supported"

    def test_pass_low_is_insufficient(self):
        assert compute_verdict_label("PASS", "Low") == "Insufficient evidence"

    def test_fail_high_is_blocked(self):
        assert compute_verdict_label("FAIL", "High") == "Blocked due to fabrication risk"

    def test_fail_medium_is_contradicted(self):
        assert compute_verdict_label("FAIL", "Medium") == "Contradicted by evidence"

    def test_unknown_combo_defaults(self):
        # Unknown combos should still return something sensible
        label = compute_verdict_label("PASS", "SomeOther")
        assert label == "Verified with evidence"  # default for PASS

    def test_verdict_label_auto_computed_on_response(self):
        """PipelineResponse should auto-compute verdict_label."""
        resp = PipelineResponse(
            gpt1_input="test", gpt1_output="output", bypassed=False,
            gpt2_raw="", claim_table=[], violations=[], gpt2_verdict="PASS",
            arbiter_invoked=False, arbiter_decision="", arbiter_rationale=[],
            arbiter_edits=[], arbiter_policy_notes=[], arbiter_raw="",
            rewrite_occurred=False, rewrite_output="", rewrite_gpt2_raw="",
            rewrite_claim_table=[], rewrite_violations=[], rewrite_verdict="",
            final_verdict="PASS", final_result="ok",
            confidence=ConfidenceBreakdown(confidence_label="High"),
        )
        assert resp.verdict_label == "Verified with evidence"


class TestComputeConfidenceAuthority:
    """Evidence confidence must use domain authority, not mutated rank scores."""

    def test_rank_score_does_not_inflate_authority(self):
        sources = [
            SearchSource(title="Blog", url="https://blog.example.com/a", snippet="x" * 400, score=0.95),
            SearchSource(title="Blog2", url="https://other.example.com/b", snippet="y" * 400, score=0.95),
        ]
        ranked = rank_sources(sources)
        result = compute_confidence(
            _claims(["Observed"] * 3 + ["Unsupported"] * 2),
            search_sources=ranked,
        )
        assert result.confidence_label == "Medium"


class TestFinalizeResponse:
    def test_strips_sanitizer_markers(self):
        resp = _finalize_response(
            gpt1_input="q", gpt1_output="raw", bypassed=False,
            gpt2_raw="", claim_table=[], violations=[], gpt2_verdict="PASS",
            arbiter_invoked=False, arbiter_decision="", arbiter_rationale=[],
            arbiter_edits=[], arbiter_policy_notes=[], arbiter_raw="",
            rewrite_occurred=False, rewrite_output="", rewrite_gpt2_raw="",
            rewrite_claim_table=[], rewrite_violations=[], rewrite_verdict="",
            final_verdict="PASS",
            final_result="Time is [Typicality language removed] relative.",
            confidence=ConfidenceBreakdown(confidence_label="Low"),
        )
        assert "[Typicality language removed]" not in resp.final_result
        assert "Time is" in resp.final_result
