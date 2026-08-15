"""Tests for pipeline.convergence -- compute_finding_delta() and should_continue_rewrite().

All functions under test are fully deterministic (no LLM calls).
"""
from __future__ import annotations

from pipeline.convergence import compute_finding_delta, should_continue_rewrite


# ===================================================================
# compute_finding_delta()
# ===================================================================


class TestComputeFindingDeltaImproving:
    """Scenarios where findings are strictly improving."""

    def test_hard_finding_resolved(self):
        prev = [{"type": "T1", "severity": "hard"}, {"type": "T5", "severity": "soft"}]
        curr = [{"type": "T5", "severity": "soft"}]
        delta = compute_finding_delta(prev, curr)
        assert delta["improved"] is True
        assert delta["converged"] is False
        assert delta["oscillating"] is False
        assert delta["hard_delta"] == -1
        assert delta["resolved_types"] == ["T1"]

    def test_soft_finding_resolved_same_hard(self):
        prev = [{"type": "T1", "severity": "hard"}, {"type": "T5", "severity": "soft"}]
        curr = [{"type": "T1", "severity": "hard"}]
        delta = compute_finding_delta(prev, curr)
        assert delta["improved"] is True
        assert delta["soft_delta"] == -1

    def test_all_findings_resolved(self):
        prev = [{"type": "T5", "severity": "soft"}]
        curr = []
        delta = compute_finding_delta(prev, curr)
        assert delta["improved"] is True


class TestComputeFindingDeltaConverged:
    """Scenarios where findings are identical (converged)."""

    def test_same_findings(self):
        findings = [{"type": "T1", "severity": "hard"}, {"type": "T5", "severity": "soft"}]
        delta = compute_finding_delta(findings, findings)
        assert delta["converged"] is True
        assert delta["improved"] is False
        assert delta["oscillating"] is False

    def test_both_empty(self):
        delta = compute_finding_delta([], [])
        assert delta["converged"] is True

    def test_same_types_same_counts(self):
        prev = [{"type": "T5", "severity": "soft"}, {"type": "T6", "severity": "soft"}]
        curr = [{"type": "T6", "severity": "soft"}, {"type": "T5", "severity": "soft"}]
        delta = compute_finding_delta(prev, curr)
        assert delta["converged"] is True


class TestComputeFindingDeltaOscillating:
    """Scenarios where findings change but don't improve."""

    def test_different_types_same_severity(self):
        prev = [{"type": "T1", "severity": "hard"}]
        curr = [{"type": "T3", "severity": "hard"}]
        delta = compute_finding_delta(prev, curr)
        assert delta["oscillating"] is True
        assert delta["improved"] is False
        assert delta["new_types"] == ["T3"]
        assert delta["resolved_types"] == ["T1"]

    def test_swapped_findings(self):
        prev = [{"type": "T4", "severity": "soft"}, {"type": "T5", "severity": "soft"}]
        curr = [{"type": "T5", "severity": "soft"}, {"type": "T6", "severity": "soft"}]
        delta = compute_finding_delta(prev, curr)
        assert delta["oscillating"] is True
        assert delta["new_types"] == ["T6"]


class TestComputeFindingDeltaRegression:
    """Scenarios where findings get worse."""

    def test_new_hard_finding(self):
        prev = [{"type": "T5", "severity": "soft"}]
        curr = [{"type": "T5", "severity": "soft"}, {"type": "T1", "severity": "hard"}]
        delta = compute_finding_delta(prev, curr)
        assert delta["hard_delta"] == 1
        assert delta["improved"] is False

    def test_more_soft_findings(self):
        prev = [{"type": "T5", "severity": "soft"}]
        curr = [
            {"type": "T5", "severity": "soft"},
            {"type": "T6", "severity": "soft"},
            {"type": "T4", "severity": "soft"},
        ]
        delta = compute_finding_delta(prev, curr)
        assert delta["improved"] is False
        assert delta["soft_delta"] == 2


# ===================================================================
# should_continue_rewrite()
# ===================================================================


class TestShouldContinueRewriteBasic:
    """Basic decision logic for rewrite continuation."""

    def test_single_iteration_continues(self):
        """Need at least 2 iterations to compare."""
        history = [[{"type": "T1", "severity": "hard"}]]
        assert should_continue_rewrite(history) is True

    def test_improving_continues(self):
        history = [
            [{"type": "T1", "severity": "hard"}, {"type": "T5", "severity": "soft"}],
            [{"type": "T5", "severity": "soft"}],  # T1 resolved
        ]
        assert should_continue_rewrite(history) is True

    def test_converged_stops(self):
        findings = [{"type": "T5", "severity": "soft"}]
        history = [findings, findings]
        assert should_continue_rewrite(history) is False

    def test_oscillating_stops(self):
        history = [
            [{"type": "T4", "severity": "soft"}],
            [{"type": "T6", "severity": "soft"}],  # Different type, not better
        ]
        assert should_continue_rewrite(history) is False

    def test_regression_stops(self):
        history = [
            [{"type": "T5", "severity": "soft"}],
            [{"type": "T5", "severity": "soft"}, {"type": "T1", "severity": "hard"}],
        ]
        assert should_continue_rewrite(history) is False


class TestShouldContinueRewriteMaxLoops:
    """Max loop enforcement."""

    def test_max_loops_reached(self):
        """With max_loops=2, history of [initial, rw1, rw2] should stop."""
        history = [
            [{"type": "T1", "severity": "hard"}, {"type": "T5", "severity": "soft"}],
            [{"type": "T5", "severity": "soft"}],
            [{"type": "T5", "severity": "soft"}],  # Still improving but hit max
        ]
        assert should_continue_rewrite(history, max_loops=2) is False

    def test_under_max_loops_continues(self):
        history = [
            [{"type": "T1", "severity": "hard"}, {"type": "T5", "severity": "soft"}],
            [{"type": "T5", "severity": "soft"}],
        ]
        assert should_continue_rewrite(history, max_loops=3) is True

    def test_default_max_loops_is_three(self):
        """Default max_loops=3, so 4 entries (initial + 3 rewrites) should stop."""
        history = [
            [{"type": "T1", "severity": "hard"}, {"type": "T5", "severity": "soft"}],
            [{"type": "T5", "severity": "soft"}, {"type": "T4", "severity": "soft"}],
            [{"type": "T4", "severity": "soft"}],
            [{"type": "T4", "severity": "soft"}],  # Converged — would stop anyway
        ]
        assert should_continue_rewrite(history) is False


class TestClosedLoopConvergence:
    """Tests for closed-loop repair convergence in <= 2 turns."""

    def test_two_turn_hard_limit_stops_further_rewrites(self):
        """When max_loops=2, loop stops after 2 rewrite attempts even if improving."""
        history = [
            [{"type": "T1", "severity": "hard"}, {"type": "T3", "severity": "hard"}],  # Turn 0
            [{"type": "T1", "severity": "hard"}],                                       # Turn 1 (improved)
            [{"type": "T5", "severity": "soft"}],                                       # Turn 2 (improved)
        ]
        # At Turn 2 (3 history entries), should_continue_rewrite must return False
        assert should_continue_rewrite(history, max_loops=2) is False

    def test_one_turn_pass_does_not_call_should_continue(self):
        """Turn 0 to Turn 1 single repair step."""
        history = [
            [{"type": "T1", "severity": "hard"}],
            [{"type": "T5", "severity": "soft"}],
        ]
        assert should_continue_rewrite(history, max_loops=2) is True

    def test_oscillation_with_negative_constraints_stops_early(self):
        """If Turn 1 replaces Turn 0 error with a different error without improvement, stops immediately."""
        turn0 = [{"type": "T1", "severity": "hard", "detail": "Fabricated citation [5]"}]
        turn1 = [{"type": "T3", "severity": "hard", "detail": "Causal claim without citation"}]
        history = [turn0, turn1]
        delta = compute_finding_delta(turn0, turn1)
        assert delta["oscillating"] is True
        assert should_continue_rewrite(history, max_loops=2) is False

