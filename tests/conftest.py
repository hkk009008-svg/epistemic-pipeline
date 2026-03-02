"""Shared pytest fixtures for the epistemic-pipeline test suite.

All fixtures here are available to every test file automatically.
"""
from __future__ import annotations

import json

import pytest

from pipeline.models import EditEntry


# ---------------------------------------------------------------------------
# route_prompt flag presets
# ---------------------------------------------------------------------------

@pytest.fixture
def flags_advice_requested() -> dict:
    """Prompt flags where the user explicitly asked for advice."""
    return {
        "advice_requested": True,
        "percent_requested": False,
        "legal_mode": False,
        "jurisdiction_present": False,
        "future_year": False,
        "current_events": False,
        "comparative": False,
    }


@pytest.fixture
def flags_all_false() -> dict:
    """Prompt flags where nothing special is detected."""
    return {
        "advice_requested": False,
        "percent_requested": False,
        "legal_mode": False,
        "jurisdiction_present": False,
        "future_year": False,
        "current_events": False,
        "comparative": False,
    }


@pytest.fixture
def flags_legal_with_jurisdiction() -> dict:
    """Prompt flags for a legal prompt that includes jurisdiction."""
    return {
        "advice_requested": False,
        "percent_requested": False,
        "legal_mode": True,
        "jurisdiction_present": True,
        "future_year": False,
        "current_events": False,
        "comparative": False,
    }


# ---------------------------------------------------------------------------
# GPT-2 raw JSON payloads
# ---------------------------------------------------------------------------

@pytest.fixture
def gpt2_pass_json() -> str:
    """Valid GPT-2 JSON with PASS verdict and no findings."""
    return json.dumps({
        "claim_table": [
            {
                "claim": "Water boils at 100C at sea level.",
                "category": "Supported",
                "justification": "Established physics.",
            }
        ],
        "findings": [],
        "verdict": "PASS",
    })


@pytest.fixture
def gpt2_fail_hard_json() -> str:
    """GPT-2 JSON with a single hard finding."""
    return json.dumps({
        "claim_table": [
            {
                "claim": "73% of applicants are approved.",
                "category": "Unsupported",
                "justification": "No source provided for percentage.",
            }
        ],
        "findings": [
            {
                "type": "Fabricated statistic",
                "severity": "hard",
                "detail": "73% figure has no source.",
            }
        ],
        "verdict": "FAIL",
    })


@pytest.fixture
def gpt2_fail_soft_accumulation_json() -> str:
    """GPT-2 JSON with 3 soft findings triggering FAIL by accumulation."""
    return json.dumps({
        "claim_table": [],
        "findings": [
            {"type": "Unsupported evidence reference", "severity": "soft", "detail": "Detail A"},
            {"type": "Prescriptive creep", "severity": "soft", "detail": "Detail B"},
            {"type": "Overconfidence", "severity": "soft", "detail": "Detail C"},
        ],
        "verdict": "FAIL",
    })


@pytest.fixture
def gpt2_prescriptive_creep_no_outcome_json() -> str:
    """GPT-2 JSON with a soft Prescriptive creep finding that has NO outcome-promise language."""
    return json.dumps({
        "claim_table": [],
        "findings": [
            {
                "type": "Prescriptive creep",
                "severity": "soft",
                "detail": "GPT-1 provided a list of steps without being asked.",
            }
        ],
        "verdict": "FAIL",
    })


@pytest.fixture
def gpt2_prescriptive_creep_with_outcome_json() -> str:
    """GPT-2 JSON with a soft Prescriptive creep finding that DOES contain outcome-promise language."""
    return json.dumps({
        "claim_table": [],
        "findings": [
            {
                "type": "Prescriptive creep",
                "severity": "soft",
                "detail": "GPT-1 claims this will improve your chances.",
            }
        ],
        "verdict": "FAIL",
    })


# ---------------------------------------------------------------------------
# GPT-3 arbiter raw JSON payloads
# ---------------------------------------------------------------------------

@pytest.fixture
def arbiter_block_json() -> str:
    """GPT-3 arbiter output with BLOCK decision."""
    return json.dumps({
        "arbiter_decision": "BLOCK",
        "rationale": ["Fabricated statistic cannot be removed without harming coherence."],
        "edits_for_gpt1": [],
        "final_policy_notes": ["Response must be regenerated."],
    })


@pytest.fixture
def arbiter_allow_with_edits_json() -> str:
    """GPT-3 arbiter output with ALLOW_WITH_EDITS decision and edits."""
    return json.dumps({
        "arbiter_decision": "ALLOW_WITH_EDITS",
        "rationale": ["Prescriptive language can be rewritten."],
        "edits_for_gpt1": [
            {
                "action": "REWRITE",
                "target": "this will improve your odds",
                "replacement": "this may help with procedure; no guarantee of outcome",
            },
            {
                "action": "DELETE",
                "target": "Studies suggest that most cases succeed.",
            },
        ],
        "final_policy_notes": ["Ensure no outcome promises remain."],
    })


@pytest.fixture
def arbiter_allow_as_unknown_json() -> str:
    """GPT-3 arbiter output with ALLOW_AS_UNKNOWN_ONLY decision."""
    return json.dumps({
        "arbiter_decision": "ALLOW_AS_UNKNOWN_ONLY",
        "rationale": ["Question is inherently indeterminate."],
        "edits_for_gpt1": [
            {
                "action": "MOVE_TO_UNKNOWN",
                "target": "Country X is medically safer",
                "replacement": "It is unknown which country is definitively safer.",
            }
        ],
        "final_policy_notes": [],
    })


# ---------------------------------------------------------------------------
# EditEntry helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_edits() -> list[EditEntry]:
    """A small list of EditEntry objects covering all three action types."""
    return [
        EditEntry(action="DELETE", target="Bad claim here.", replacement=""),
        EditEntry(action="REWRITE", target="will improve your odds", replacement="may help with procedure; no guarantee"),
        EditEntry(action="MOVE_TO_UNKNOWN", target="73% success rate", replacement="Success rate is unknown."),
    ]


# ---------------------------------------------------------------------------
# Atomic claim decomposition fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def decomposed_claims() -> list:
    """Sample atomic claims from decomposer."""
    return [
        {"text": "Water boils at 100C at sea level.", "has_citation": False,
         "is_unknown": False, "is_user_provided": False},
        {"text": "The approval rate is 73%.", "has_citation": False,
         "is_unknown": False, "is_user_provided": False},
    ]


# ---------------------------------------------------------------------------
# NLI result fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def nli_entailment_result() -> dict:
    return {"label": "entailment", "scores": {"entailment": 0.92, "contradiction": 0.03, "neutral": 0.05}}


@pytest.fixture
def nli_contradiction_result() -> dict:
    return {"label": "contradiction", "scores": {"entailment": 0.05, "contradiction": 0.88, "neutral": 0.07}}


# ---------------------------------------------------------------------------
# Convergence / findings history fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def findings_improving() -> list:
    """Two sets of findings showing improvement."""
    return [
        [{"type": "T1", "severity": "hard"}, {"type": "T5", "severity": "soft"}],
        [{"type": "T5", "severity": "soft"}],  # T1 resolved
    ]
