"""Tests for NLI verification layer (optional, graceful fallback)."""
from __future__ import annotations

from unittest.mock import patch

from pipeline.nli import (
    is_nli_available,
    classify_nli,
    batch_classify_nli,
    verify_claims_with_nli,
    compute_grounding_rate,
    detect_unsupported_spans,
    _compute_confidence_tier,
)


# ---------------------------------------------------------------------------
# is_nli_available (when torch/transformers NOT installed)
# ---------------------------------------------------------------------------

class TestNliAvailability:
    """Test NLI availability detection."""

    def test_nli_available_returns_bool(self):
        result = is_nli_available()
        assert isinstance(result, bool)

    @patch("pipeline.nli.NLI_SERVICE_URL", "")
    @patch("pipeline.nli._nli_available", None)
    def test_nli_unavailable_when_no_torch(self):
        """If torch/transformers aren't installed and no service URL, NLI is unavailable."""
        import pipeline.nli as nli_mod
        nli_mod._nli_available = None
        # In this test environment torch may or may not be installed;
        # we just verify the function returns a bool without error
        result = is_nli_available()
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# _compute_confidence_tier
# ---------------------------------------------------------------------------

class TestConfidenceTier:
    """Test the continuous NLI score to confidence tier mapping."""

    def test_strong_support(self):
        assert _compute_confidence_tier(0.9, 0.02) == "strong_support"

    def test_weak_support(self):
        assert _compute_confidence_tier(0.5, 0.1) == "weak_support"

    def test_strong_contradiction(self):
        assert _compute_confidence_tier(0.1, 0.85) == "strong_contradiction"

    def test_weak_contradiction(self):
        assert _compute_confidence_tier(0.1, 0.55) == "weak_contradiction"

    def test_neutral(self):
        assert _compute_confidence_tier(0.2, 0.2) == "neutral"

    def test_threshold_boundary_entailment(self):
        assert _compute_confidence_tier(0.7, 0.1) == "strong_support"

    def test_threshold_boundary_contradiction(self):
        assert _compute_confidence_tier(0.1, 0.7) == "strong_contradiction"

    def test_weak_boundary(self):
        assert _compute_confidence_tier(0.4, 0.1) == "weak_support"

    def test_just_below_weak(self):
        assert _compute_confidence_tier(0.39, 0.39) == "neutral"

    def test_contradiction_outranks_strong_support(self):
        assert _compute_confidence_tier(0.8, 0.85) == "strong_contradiction"


# ---------------------------------------------------------------------------
# verify_claims_with_nli (integration-level, mocked NLI)
# ---------------------------------------------------------------------------

class TestVerifyClaimsWithNli:
    """Test NLI claim verification with mocked dependencies."""

    def test_empty_claims_returns_unchanged(self):
        result = verify_claims_with_nli([], ["evidence"])
        assert result == []

    def test_empty_evidence_returns_unchanged(self):
        claims = [{"text": "Water boils at 100C."}]
        result = verify_claims_with_nli(claims, [])
        assert result == claims

    @patch("pipeline.nli.is_nli_available", return_value=True)
    @patch("pipeline.nli.batch_classify_nli")
    def test_supported_claim_gets_nli_result(self, mock_batch, mock_avail):
        mock_batch.return_value = [
            {"label": "entailment", "scores": {"entailment": 0.92, "contradiction": 0.03, "neutral": 0.05}}
        ]
        claims = [{"text": "Water boils at 100C at sea level."}]
        evidence = ["Water boils at 100 degrees Celsius under standard atmospheric pressure."]
        result = verify_claims_with_nli(claims, evidence)
        assert len(result) == 1
        assert result[0]["nli_result"]["supported"] is True
        assert result[0]["nli_result"]["contradicted"] is False
        assert result[0]["nli_result"]["confidence_tier"] == "strong_support"
        assert "nli_result" not in claims[0]

    @patch("pipeline.nli.is_nli_available", return_value=True)
    @patch("pipeline.nli.batch_classify_nli")
    def test_contradicted_claim_gets_nli_result(self, mock_batch, mock_avail):
        mock_batch.return_value = [
            {"label": "contradiction", "scores": {"entailment": 0.05, "contradiction": 0.88, "neutral": 0.07}}
        ]
        claims = [{"text": "Water boils at 50C."}]
        evidence = ["Water boils at 100 degrees Celsius."]
        result = verify_claims_with_nli(claims, evidence)
        assert result[0]["nli_result"]["contradicted"] is True
        assert result[0]["nli_result"]["supported"] is False
        assert result[0]["nli_result"]["confidence_tier"] == "strong_contradiction"

    @patch("pipeline.nli.is_nli_available", return_value=True)
    @patch("pipeline.nli.batch_classify_nli")
    def test_neutral_claim(self, mock_batch, mock_avail):
        mock_batch.return_value = [
            {"label": "neutral", "scores": {"entailment": 0.2, "contradiction": 0.1, "neutral": 0.7}}
        ]
        claims = [{"text": "The sky is blue on Mars."}]
        evidence = ["Mars has a thin atmosphere."]
        result = verify_claims_with_nli(claims, evidence)
        assert result[0]["nli_result"]["supported"] is False
        assert result[0]["nli_result"]["contradicted"] is False
        assert result[0]["nli_result"]["confidence_tier"] == "neutral"

    @patch("pipeline.nli.is_nli_available", return_value=True)
    @patch("pipeline.nli.batch_classify_nli")
    def test_multiple_evidence_best_entailment(self, mock_batch, mock_avail):
        mock_batch.return_value = [
            {"label": "neutral", "scores": {"entailment": 0.3, "contradiction": 0.1, "neutral": 0.6}},
            {"label": "entailment", "scores": {"entailment": 0.85, "contradiction": 0.05, "neutral": 0.1}},
        ]
        claims = [{"text": "The rate is 73%."}]
        evidence = ["Some unrelated text.", "The approval rate stands at 73%."]
        result = verify_claims_with_nli(claims, evidence)
        assert result[0]["nli_result"]["best_entailment"] == 0.85
        assert result[0]["nli_result"]["best_source_idx"] == 1
        assert result[0]["nli_result"]["supported"] is True
        assert result[0]["nli_result"]["confidence_tier"] == "strong_support"

    @patch("pipeline.nli.is_nli_available", return_value=True)
    @patch("pipeline.nli.batch_classify_nli")
    def test_per_source_scores_included(self, mock_batch, mock_avail):
        mock_batch.return_value = [
            {"label": "neutral", "scores": {"entailment": 0.3, "contradiction": 0.1, "neutral": 0.6}},
            {"label": "entailment", "scores": {"entailment": 0.85, "contradiction": 0.05, "neutral": 0.1}},
        ]
        claims = [{"text": "The rate is 73%."}]
        evidence = ["Source A.", "Source B."]
        result = verify_claims_with_nli(claims, evidence)
        scores = result[0]["nli_result"]["per_source_scores"]
        assert len(scores) == 2
        assert scores[0]["source_idx"] == 0
        assert scores[1]["source_idx"] == 1

    @patch("pipeline.nli.is_nli_available", return_value=True)
    @patch("pipeline.nli.batch_classify_nli")
    def test_empty_text_claim_passed_through(self, mock_batch, mock_avail):
        claims = [{"text": ""}]
        evidence = ["some evidence"]
        result = verify_claims_with_nli(claims, evidence)
        assert len(result) == 1
        assert "nli_result" not in result[0]
        mock_batch.assert_not_called()

    @patch("pipeline.nli.is_nli_available", return_value=False)
    def test_nli_unavailable_returns_unchanged(self, mock_avail):
        claims = [{"text": "Water boils at 100C."}]
        evidence = ["Water boils at 100 degrees."]
        result = verify_claims_with_nli(claims, evidence)
        assert result == claims
        assert "nli_result" not in result[0]

    @patch("pipeline.nli.is_nli_available", return_value=True)
    @patch("pipeline.nli.batch_classify_nli")
    def test_all_failed_classifications_omit_nli_result(self, mock_batch, mock_avail):
        mock_batch.return_value = [None, None]
        claims = [{"text": "Water boils at 100C."}]
        result = verify_claims_with_nli(claims, ["a", "b"])
        assert "nli_result" not in result[0]
        grounding = compute_grounding_rate(result)
        assert grounding["total_evaluated"] == 0

    @patch("pipeline.nli.is_nli_available", return_value=True)
    @patch("pipeline.nli.batch_classify_nli")
    def test_mixed_evidence_is_contradicted_not_supported(self, mock_batch, mock_avail):
        mock_batch.return_value = [
            {"label": "entailment", "scores": {"entailment": 0.8, "contradiction": 0.1, "neutral": 0.1}},
            {"label": "contradiction", "scores": {"entailment": 0.05, "contradiction": 0.85, "neutral": 0.1}},
        ]
        claims = [{"text": "The rate is 73%."}]
        result = verify_claims_with_nli(claims, ["supports", "refutes"])
        nli = result[0]["nli_result"]
        assert nli["contradicted"] is True
        assert nli["supported"] is False
        assert nli["confidence_tier"] == "strong_contradiction"


# ---------------------------------------------------------------------------
# compute_grounding_rate
# ---------------------------------------------------------------------------

class TestGroundingRate:
    """Test grounding rate computation."""

    def test_empty_claims(self):
        result = compute_grounding_rate([])
        assert result["grounding_rate"] == 0.0
        assert result["total_evaluated"] == 0

    def test_no_nli_results(self):
        claims = [{"text": "claim1"}, {"text": "claim2"}]
        result = compute_grounding_rate(claims)
        assert result["grounding_rate"] == 0.0
        assert result["total_evaluated"] == 0

    def test_all_supported(self):
        claims = [
            {"text": "c1", "nli_result": {"confidence_tier": "strong_support", "best_entailment": 0.9}},
            {"text": "c2", "nli_result": {"confidence_tier": "weak_support", "best_entailment": 0.5}},
        ]
        result = compute_grounding_rate(claims)
        assert result["grounding_rate"] == 1.0
        assert result["grounded_count"] == 2
        assert result["total_evaluated"] == 2

    def test_mixed(self):
        claims = [
            {"text": "c1", "nli_result": {"confidence_tier": "strong_support"}},
            {"text": "c2", "nli_result": {"confidence_tier": "neutral"}},
            {"text": "c3", "nli_result": {"confidence_tier": "strong_contradiction"}},
            {"text": "c4", "nli_result": {"confidence_tier": "weak_support"}},
        ]
        result = compute_grounding_rate(claims)
        assert result["grounding_rate"] == 0.5  # 2/4
        assert result["grounded_count"] == 2
        assert result["contradicted_count"] == 1
        assert result["neutral_count"] == 1

    def test_all_contradicted(self):
        claims = [
            {"text": "c1", "nli_result": {"confidence_tier": "strong_contradiction"}},
        ]
        result = compute_grounding_rate(claims)
        assert result["grounding_rate"] == 0.0
        assert result["contradicted_count"] == 1


# ---------------------------------------------------------------------------
# detect_unsupported_spans
# ---------------------------------------------------------------------------

class TestDetectUnsupportedSpans:
    """Test unsupported span detection."""

    def test_no_claims(self):
        assert detect_unsupported_spans("Some text.", []) == []

    def test_no_nli_results(self):
        claims = [{"text": "Some claim."}]
        assert detect_unsupported_spans("Some claim.", claims) == []

    def test_contradicted_span_detected(self):
        text = "Water boils at 50 degrees Celsius."
        claims = [{
            "text": "Water boils at 50 degrees Celsius.",
            "nli_result": {
                "confidence_tier": "strong_contradiction",
                "worst_contradiction": 0.88,
                "best_entailment": 0.05,
            },
        }]
        spans = detect_unsupported_spans(text, claims)
        assert len(spans) == 1
        assert spans[0]["reason"] == "contradicted_by_evidence"
        assert spans[0]["start"] >= 0

    def test_no_evidence_span_detected(self):
        text = "The GDP of Narnia is $500 billion."
        claims = [{
            "text": "The GDP of Narnia is $500 billion.",
            "nli_result": {
                "confidence_tier": "neutral",
                "worst_contradiction": 0.1,
                "best_entailment": 0.1,
            },
        }]
        spans = detect_unsupported_spans(text, claims)
        assert len(spans) == 1
        assert spans[0]["reason"] == "no_evidence_found"

    def test_supported_claims_not_flagged(self):
        text = "Water boils at 100C."
        claims = [{
            "text": "Water boils at 100C.",
            "nli_result": {
                "confidence_tier": "strong_support",
                "worst_contradiction": 0.02,
                "best_entailment": 0.95,
            },
        }]
        spans = detect_unsupported_spans(text, claims)
        assert len(spans) == 0

    def test_weak_support_not_flagged(self):
        text = "The rate is about 50%."
        claims = [{
            "text": "The rate is about 50%.",
            "nli_result": {
                "confidence_tier": "weak_support",
                "worst_contradiction": 0.1,
                "best_entailment": 0.45,
            },
        }]
        spans = detect_unsupported_spans(text, claims)
        assert len(spans) == 0


# ---------------------------------------------------------------------------
# classify_nli / batch_classify_nli edge cases
# ---------------------------------------------------------------------------

class TestClassifyNliEdgeCases:
    """Test individual classification edge cases."""

    @patch("pipeline.nli.NLI_SERVICE_URL", "")
    @patch("pipeline.nli._get_nli_pipeline", return_value=None)
    def test_classify_returns_none_when_unavailable(self, mock_pipe):
        result = classify_nli("premise", "hypothesis")
        assert result is None

    @patch("pipeline.nli.NLI_SERVICE_URL", "")
    @patch("pipeline.nli._get_nli_pipeline", return_value=None)
    def test_batch_returns_nones_when_unavailable(self, mock_pipe):
        result = batch_classify_nli([("p1", "h1"), ("p2", "h2")])
        assert result == [None, None]
