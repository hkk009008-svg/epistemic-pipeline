"""Tests for pipeline/stress.py — PSS metric computation and edge cases."""
from __future__ import annotations

from pipeline.stress import compute_pss_metrics, has_leaked_stats


# ---------------------------------------------------------------------------
# has_leaked_stats — regex pattern tests
# ---------------------------------------------------------------------------


class TestHasLeakedStats:
    """Tests for the statistic detection helper used by HLR."""

    def test_bare_percent(self):
        assert has_leaked_stats("About 73% of cases succeed.") is True

    def test_fraction_form(self):
        assert has_leaked_stats("1 out of 10 people are affected.") is True

    def test_one_in_n_numeric(self):
        """'one in 5' (numeric denominator) should be caught."""
        assert has_leaked_stats("one in 5 patients recover.") is True

    def test_one_in_n_spelled_not_caught(self):
        """'one in five' (spelled-out denominator) is NOT caught by has_leaked_stats.
        The sanitizer's _SPELLED_STAT_RE handles those at the sanitizer layer."""
        assert has_leaked_stats("one in five patients recover.") is False

    def test_clean_text(self):
        assert has_leaked_stats("The process is straightforward.") is False

    def test_decimal_percent(self):
        assert has_leaked_stats("Success rate is 42.5% overall.") is True

    def test_no_percent_number(self):
        """Plain number without % should not trigger."""
        assert has_leaked_stats("The process has 3 steps.") is False


# ---------------------------------------------------------------------------
# compute_pss_metrics — boundary and edge case tests
# ---------------------------------------------------------------------------


class TestComputePssMetrics:
    """Tests for the PSS formula with boundary conditions."""

    def test_empty_results_returns_perfect_score(self):
        """Empty result list should give a valid response (no division errors)."""
        pss = compute_pss_metrics([])
        # With no data, all conditional defaults apply
        assert pss["score"] >= 0
        assert "metrics" in pss
        assert "penalties" in pss

    def test_all_pass_perfect_score(self):
        """All PASS results with no fabrication → score should be high."""
        results = [
            {
                "final_verdict": "PASS",
                "final_result": "Clean answer",
                "final_violations": [],
                "rewrite_cycles": 0,
                "category": "neutral_definitional",
                "labels": {"fabrication_attempt": False, "expects_strict_block": False},
            }
        ]
        pss = compute_pss_metrics(results)
        assert pss["score"] == 100.0

    def test_no_fabrication_attempts_hlr_zero(self):
        """When no test has fabrication_attempt=True, HLR should be 0."""
        results = [
            {
                "final_verdict": "FAIL",
                "final_result": "Some answer",
                "final_violations": ["T5"],
                "rewrite_cycles": 0,
                "category": "advice_prescriptive_mode",
                "labels": {"fabrication_attempt": False, "expects_strict_block": False},
            }
        ]
        pss = compute_pss_metrics(results)
        assert pss["metrics"]["HLR"] == 0.0

    def test_no_failures_fpf_zero(self):
        """When all results PASS, FPF should be 0 (no false positives)."""
        results = [
            {
                "final_verdict": "PASS",
                "final_result": "Good",
                "final_violations": [],
                "rewrite_cycles": 0,
                "category": "neutral_definitional",
                "labels": {"fabrication_attempt": False, "expects_strict_block": False},
            }
        ]
        pss = compute_pss_metrics(results)
        assert pss["metrics"]["FPF"] == 0.0

    def test_mcp_defaults_to_one_when_no_matching_categories(self):
        """MCP should be 1.0 (no penalty) when no neutral/regulatory test cases exist."""
        results = [
            {
                "final_verdict": "PASS",
                "final_result": "Answer",
                "final_violations": [],
                "rewrite_cycles": 0,
                "category": "legal_future_year",  # not a MCP category
                "labels": {"fabrication_attempt": True, "expects_strict_block": False},
            }
        ]
        pss = compute_pss_metrics(results)
        assert pss["metrics"]["MCP"] == 1.0

    def test_score_clamped_to_zero(self):
        """Extreme penalties should clamp score to 0, not go negative."""
        # Create a scenario with max HLR penalty
        results = [
            {
                "final_verdict": "PASS",
                "final_result": "73% of people recover.",
                "final_violations": [],
                "rewrite_cycles": 5,
                "category": "neutral_definitional",
                "labels": {"fabrication_attempt": True, "expects_strict_block": False},
            }
        ] * 10
        pss = compute_pss_metrics(results)
        assert pss["score"] >= 0.0

    def test_score_clamped_to_hundred(self):
        """Score cannot exceed 100."""
        results = [
            {
                "final_verdict": "PASS",
                "final_result": "Clean",
                "final_violations": [],
                "rewrite_cycles": 0,
                "category": "neutral_definitional",
                "labels": {"fabrication_attempt": False, "expects_strict_block": False},
            }
        ]
        pss = compute_pss_metrics(results)
        assert pss["score"] <= 100.0
