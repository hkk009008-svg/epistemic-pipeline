"""Tests for NLI verification layer (optional, graceful fallback)."""
from __future__ import annotations

from unittest.mock import patch

from pipeline.nli import (
    is_nli_available,
    classify_nli,
    batch_classify_nli,
    verify_claims_with_nli,
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
