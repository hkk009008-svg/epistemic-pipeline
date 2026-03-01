"""Tests for atomic claim decomposition module."""
from __future__ import annotations

import json
from unittest.mock import patch

from pipeline.decomposer import decompose_claims


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
                {"text": "Valid claim."},
                "not a dict",
            ]
        })
        cfg = {"provider": "openai", "api_key": "k", "model": "m", "base_url": ""}
        result = decompose_claims(cfg, "output", "prompt")
        assert len(result) == 1
        assert result[0]["text"] == "Valid claim."

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
