"""Tests for atomic claim decomposition module."""
from __future__ import annotations

import json
from unittest.mock import patch

from pipeline.decomposer import (
    decompose_claims,
    check_decomposition_quality,
    _is_compound,
    _validate_claim,
)


# ---------------------------------------------------------------------------
# decompose_claims
# ---------------------------------------------------------------------------

class TestDecomposeClaims:
    """Test the decompose_claims function with mocked LLM calls."""

    @patch("pipeline.decomposer.call_llm")
    def test_valid_decomposition(self, mock_call):
        mock_call.return_value = json.dumps({
            "claims": [
                {"text": "Water boils at 100C.", "has_citation": False,
                 "is_unknown": False, "is_user_provided": False},
                {"text": "The rate is 73%.", "has_citation": False,
                 "is_unknown": False, "is_user_provided": False},
            ]
        })
        cfg = {"provider": "openai", "api_key": "k", "model": "m", "base_url": ""}
        result = decompose_claims(cfg, "Some GPT-1 output", "user prompt")
        assert len(result) == 2
        assert result[0]["text"] == "Water boils at 100C."
        assert result[1]["text"] == "The rate is 73%."

    @patch("pipeline.decomposer.call_llm")
    def test_claims_with_citation_flag(self, mock_call):
        mock_call.return_value = json.dumps({
            "claims": [
                {"text": "According to [1], rate is 60%.", "has_citation": True,
                 "is_unknown": False, "is_user_provided": False},
            ]
        })
        cfg = {"provider": "openai", "api_key": "k", "model": "m", "base_url": ""}
        result = decompose_claims(cfg, "output", "prompt")
        assert result[0]["has_citation"] is True

    @patch("pipeline.decomposer.call_llm")
    def test_claims_with_unknown_flag(self, mock_call):
        mock_call.return_value = json.dumps({
            "claims": [
                {"text": "The exact figure is unknown.", "has_citation": False,
                 "is_unknown": True, "is_user_provided": False},
            ]
        })
        cfg = {"provider": "openai", "api_key": "k", "model": "m", "base_url": ""}
        result = decompose_claims(cfg, "output", "prompt")
        assert result[0]["is_unknown"] is True

    @patch("pipeline.decomposer.call_llm")
    def test_graceful_failure_returns_empty(self, mock_call):
        mock_call.side_effect = Exception("LLM failed")
        cfg = {"provider": "openai", "api_key": "k", "model": "m", "base_url": ""}
        result = decompose_claims(cfg, "output", "prompt")
        assert result == []

    @patch("pipeline.decomposer.call_llm")
    def test_invalid_json_returns_empty(self, mock_call):
        mock_call.return_value = "not json at all"
        cfg = {"provider": "openai", "api_key": "k", "model": "m", "base_url": ""}
        result = decompose_claims(cfg, "output", "prompt")
        assert result == []

    @patch("pipeline.decomposer.call_llm")
    def test_missing_fields_get_defaults(self, mock_call):
        mock_call.return_value = json.dumps({
            "claims": [
                {"text": "Claim without flags."},
            ]
        })
        cfg = {"provider": "openai", "api_key": "k", "model": "m", "base_url": ""}
        result = decompose_claims(cfg, "output", "prompt")
        assert len(result) == 1
        assert result[0]["has_citation"] is False
        assert result[0]["is_unknown"] is False
        assert result[0]["is_user_provided"] is False

    @patch("pipeline.decomposer.call_llm")
    def test_entries_without_text_skipped(self, mock_call):
        mock_call.return_value = json.dumps({
            "claims": [
                {"has_citation": False},
                {"text": "Climate change affects global temperatures."},
                "not a dict",
            ]
        })
        cfg = {"provider": "openai", "api_key": "k", "model": "m", "base_url": ""}
        result = decompose_claims(cfg, "output", "prompt")
        assert len(result) == 1
        assert result[0]["text"] == "Climate change affects global temperatures."

    @patch("pipeline.decomposer.call_llm")
    def test_empty_claims_list(self, mock_call):
        mock_call.return_value = json.dumps({"claims": []})
        cfg = {"provider": "openai", "api_key": "k", "model": "m", "base_url": ""}
        result = decompose_claims(cfg, "output", "prompt")
        assert result == []

    @patch("pipeline.decomposer.call_llm")
    def test_user_prompt_passed_to_llm(self, mock_call):
        mock_call.return_value = json.dumps({"claims": []})
        cfg = {"provider": "openai", "api_key": "k", "model": "m", "base_url": ""}
        decompose_claims(cfg, "gpt1 text", "my question")
        call_args = mock_call.call_args
        assert "my question" in call_args[0][2]  # user_content
        assert "gpt1 text" in call_args[0][2]

    @patch("pipeline.decomposer.call_llm")
    def test_short_claims_filtered(self, mock_call):
        mock_call.return_value = json.dumps({
            "claims": [
                {"text": "OK."},  # Too short (<8 chars)
                {"text": "Water boils at 100 degrees Celsius."},
            ]
        })
        cfg = {"provider": "openai", "api_key": "k", "model": "m", "base_url": ""}
        result = decompose_claims(cfg, "output", "prompt")
        assert len(result) == 1
        assert result[0]["text"] == "Water boils at 100 degrees Celsius."


# ---------------------------------------------------------------------------
# _validate_claim
# ---------------------------------------------------------------------------

class TestValidateClaim:
    def test_valid_claim(self):
        result = _validate_claim({"text": "Water boils at 100C."})
        assert result is not None
        assert result["text"] == "Water boils at 100C."

    def test_short_claim_rejected(self):
        assert _validate_claim({"text": "Hi."}) is None

    def test_no_text_field(self):
        assert _validate_claim({"has_citation": True}) is None

    def test_not_dict(self):
        assert _validate_claim("string") is None

    def test_defaults_flags(self):
        result = _validate_claim({"text": "Some valid claim text."})
        assert result["has_citation"] is False
        assert result["is_unknown"] is False

    def test_strips_whitespace(self):
        result = _validate_claim({"text": "  Padded claim text.  "})
        assert result["text"] == "Padded claim text."


# ---------------------------------------------------------------------------
# _is_compound
# ---------------------------------------------------------------------------

class TestIsCompound:
    def test_simple_claim(self):
        assert _is_compound("Water boils at 100C.") is False

    def test_compound_with_and(self):
        assert _is_compound("Water boils at 100C and ice melts at 0C and steam rises.") is True

    def test_compound_with_semicolons(self):
        assert _is_compound("The rate is 73%; the average approval takes 3 months.") is True

    def test_single_semicolon_short_parts(self):
        assert _is_compound("Short; part.") is False

    def test_both_and_pattern(self):
        assert _is_compound("Both the rate and the average are high.") is True


# ---------------------------------------------------------------------------
# check_decomposition_quality
# ---------------------------------------------------------------------------

class TestDecompositionQuality:
    def test_empty_claims(self):
        result = check_decomposition_quality("Some original text with words.", [])
        assert result["quality_tier"] == "poor"
        assert result["completeness_score"] == 0.0

    def test_good_quality(self):
        original = "Water boils at 100 degrees Celsius at sea level pressure."
        claims = [
            {"text": "Water boils at 100 degrees Celsius."},
            {"text": "This occurs at sea level pressure."},
        ]
        result = check_decomposition_quality(original, claims)
        assert result["quality_tier"] == "good"
        assert result["completeness_score"] > 0.5

    def test_poor_quality_low_coverage(self):
        original = "The economy grew 3% last quarter, unemployment fell to 4%, and inflation remained at 2%."
        claims = [
            {"text": "Something unrelated to original."},
        ]
        result = check_decomposition_quality(original, claims)
        assert result["completeness_score"] < 0.3

    def test_compound_claims_detected(self):
        claims = [
            {"text": "Water boils at 100C and ice melts at 0C and both are important."},
        ]
        result = check_decomposition_quality("some text", claims)
        assert result["compound_count"] == 1

    def test_claim_count(self):
        claims = [
            {"text": "Claim one here."},
            {"text": "Claim two here."},
            {"text": "Claim three here."},
        ]
        result = check_decomposition_quality("some text", claims)
        assert result["claim_count"] == 3

    def test_avg_claim_length(self):
        claims = [
            {"text": "Short claim."},  # 12
            {"text": "A much longer claim text here."},  # 30
        ]
        result = check_decomposition_quality("some text", claims)
        assert result["avg_claim_length"] == 21.0
