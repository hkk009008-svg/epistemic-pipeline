"""Tests for best-of-N generation scoring heuristics."""
from __future__ import annotations

from unittest.mock import patch

from pipeline.best_of_n import score_response, generate_best_of_n


class TestScoreResponse:
    """Test factuality heuristic scoring."""

    def test_base_score(self):
        score = score_response("Simple text without markers.", {})
        assert score == 50.0

    def test_citations_boost(self):
        text = "According to [CDC 2024], the rate is high. [WHO Report] confirms this."
        score = score_response(text, {})
        assert score > 50.0

    def test_banned_phrases_penalty(self):
        text = "Studies suggest that research shows the data indicates improvement."
        score = score_response(text, {})
        assert score < 50.0

    def test_hedging_bonus(self):
        text = "This may indicate a trend. It is possible that results could vary."
        score = score_response(text, {})
        assert score > 50.0

    def test_unknown_framing_bonus(self):
        text = "The exact rate is unknown. Unknown(Actionable): verify from official sources."
        score = score_response(text, {})
        assert score > 50.0

    def test_bare_percent_penalty(self):
        text = "The rate is 73% and success is 85%."
        score = score_response(text, {})
        assert score < 50.0

    def test_legal_mode_no_citations_penalty(self):
        text = "It is legal to do this."
        score = score_response(text, {"legal_mode": True})
        score_normal = score_response(text, {})
        assert score < score_normal

    def test_current_events_no_hedging_penalty(self):
        text = "The GDP is exactly 5 trillion dollars."
        score = score_response(text, {"current_events": True})
        score_normal = score_response(text, {})
        assert score < score_normal

    def test_well_structured_response(self):
        text = (
            "According to [1] (CDC Report), the vaccination rate may be increasing. "
            "It is possible that coverage varies by region. "
            "Unknown(Actionable): Exact regional data requires verification."
        )
        score = score_response(text, {})
        # Should score well: citations + hedging + unknown framing
        assert score > 65.0

    def test_score_never_negative(self):
        text = "Studies suggest research shows data indicates studies suggest " * 5
        score = score_response(text, {})
        assert score >= 0.0


class TestGenerateBestOfN:
    """Test the best-of-N generation function."""

    @patch("pipeline.best_of_n.call_llm")
    def test_n_equals_1_skips_selection(self, mock_call):
        mock_call.return_value = "Single response."
        cfg = {"provider": "openai", "api_key": "k", "model": "m", "base_url": ""}
        result, info = generate_best_of_n(cfg, "sys", "user", {}, n=1)
        assert result == "Single response."
        assert info["candidates_generated"] == 1
        assert info["selected_index"] == 0
        mock_call.assert_called_once()

    @patch("pipeline.best_of_n.call_llm")
    def test_selects_best(self, mock_call):
        # First response: bad (banned phrase), Second: good (citation)
        mock_call.side_effect = [
            "Studies suggest the rate is high.",
            "According to [CDC 2024], the rate is increasing.",
        ]
        cfg = {"provider": "openai", "api_key": "k", "model": "m", "base_url": ""}
        result, info = generate_best_of_n(cfg, "sys", "user", {}, n=2)
        assert "CDC" in result  # Should pick the cited response
        assert info["candidates_generated"] == 2
        assert info["selected_index"] == 1

    @patch("pipeline.best_of_n.call_llm")
    def test_handles_partial_failure(self, mock_call):
        mock_call.side_effect = [
            "Good response.",
            Exception("LLM failed"),
        ]
        cfg = {"provider": "openai", "api_key": "k", "model": "m", "base_url": ""}
        result, info = generate_best_of_n(cfg, "sys", "user", {}, n=2)
        assert result == "Good response."
        assert info["candidates_generated"] == 1

    @patch("pipeline.best_of_n.call_llm")
    def test_all_fail_raises(self, mock_call):
        mock_call.side_effect = Exception("LLM failed")
        cfg = {"provider": "openai", "api_key": "k", "model": "m", "base_url": ""}
        try:
            generate_best_of_n(cfg, "sys", "user", {}, n=2)
            assert False, "Should have raised"
        except ValueError:
            pass

    @patch("pipeline.best_of_n.call_llm")
    def test_score_spread_reported(self, mock_call):
        mock_call.side_effect = [
            "Studies suggest improvement.",  # bad
            "According to [Source], this may be the case.",  # good
        ]
        cfg = {"provider": "openai", "api_key": "k", "model": "m", "base_url": ""}
        _, info = generate_best_of_n(cfg, "sys", "user", {}, n=2)
        assert info["score_spread"] > 0
