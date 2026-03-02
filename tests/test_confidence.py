"""Tests for compute_confidence() — upgraded with NLI grounding rate."""
from __future__ import annotations

from pipeline.orchestrator import compute_confidence
from pipeline.models import ClaimEntry


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
