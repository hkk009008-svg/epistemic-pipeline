"""Convergence detection for rewrite loops.

Determines whether re-verification is making progress (findings decreasing)
or oscillating (findings changing but not improving). Used by the orchestrator
to decide whether to continue ALLOW_WITH_EDITS rewrite loops.
"""
from __future__ import annotations

from typing import List


def compute_finding_delta(prev_findings: List[dict], curr_findings: List[dict]) -> dict:
    """Compare two sets of findings and return a delta summary.

    Returns:
        {
            "improved": bool,       # True if findings are strictly better
            "converged": bool,      # True if findings are identical
            "oscillating": bool,    # True if findings changed but didn't improve
            "hard_delta": int,      # Change in hard findings count
            "soft_delta": int,      # Change in soft findings count
            "new_types": list,      # Finding types that are new
            "resolved_types": list, # Finding types that were resolved
        }
    """
    prev_hard = sum(1 for f in prev_findings if f.get("severity") == "hard")
    curr_hard = sum(1 for f in curr_findings if f.get("severity") == "hard")
    prev_soft = sum(1 for f in prev_findings if f.get("severity") == "soft")
    curr_soft = sum(1 for f in curr_findings if f.get("severity") == "soft")

    prev_types = set(f.get("type", "") for f in prev_findings)
    curr_types = set(f.get("type", "") for f in curr_findings)

    new_types = sorted(curr_types - prev_types)
    resolved_types = sorted(prev_types - curr_types)

    hard_delta = curr_hard - prev_hard
    soft_delta = curr_soft - prev_soft

    # Improved: fewer hard violations, OR same hard but fewer soft
    improved = (hard_delta < 0) or (hard_delta == 0 and soft_delta < 0)

    # Converged: identical finding types and counts
    converged = (prev_types == curr_types and
                 prev_hard == curr_hard and
                 prev_soft == curr_soft)

    # Oscillating: changed but not better
    oscillating = not improved and not converged and len(new_types) > 0

    return {
        "improved": improved,
        "converged": converged,
        "oscillating": oscillating,
        "hard_delta": hard_delta,
        "soft_delta": soft_delta,
        "new_types": new_types,
        "resolved_types": resolved_types,
    }


def should_continue_rewrite(history: List[List[dict]], max_loops: int = 3) -> bool:
    """Given a history of findings from each iteration, decide whether to continue.

    history: List of findings lists, ordered [initial, rewrite_1, rewrite_2, ...]
    Returns True if another rewrite loop should be attempted.
    """
    if len(history) < 2:
        return True  # Need at least 2 iterations to compare

    if len(history) >= max_loops + 1:
        return False  # Hit max (history includes initial + N rewrites)

    latest_delta = compute_finding_delta(history[-2], history[-1])

    # Stop if converged (same findings) or oscillating (not improving)
    if latest_delta["converged"] or latest_delta["oscillating"]:
        return False

    # Stop if new hard findings appeared (regression)
    if latest_delta["hard_delta"] > 0:
        return False

    # Continue if improving
    return latest_delta["improved"]
