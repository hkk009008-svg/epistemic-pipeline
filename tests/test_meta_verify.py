"""Tests for meta-verification layer."""
from __future__ import annotations

from pipeline.meta_verify import (
    is_high_stakes,
    check_claim_table_consistency,
    meta_verify_pass,
    meta_verify_fail,
    _text_overlap,
)
from pipeline.models import ClaimEntry


# ---------------------------------------------------------------------------
# is_high_stakes
# ---------------------------------------------------------------------------

class TestIsHighStakes:
    def test_legal_mode(self):
        assert is_high_stakes({"legal_mode": True}) is True

    def test_percent_requested(self):
        assert is_high_stakes({"percent_requested": True}) is True

    def test_comparative(self):
        assert is_high_stakes({"comparative": True}) is True

    def test_none_high_stakes(self):
        assert is_high_stakes({"advice_requested": True, "current_events": True}) is False

    def test_empty_flags(self):
        assert is_high_stakes({}) is False


# ---------------------------------------------------------------------------
# _text_overlap
# ---------------------------------------------------------------------------

class TestTextOverlap:
    def test_matching_texts(self):
        assert _text_overlap("Water boils at 100 degrees Celsius.", "Water boils at 100 degrees Celsius at sea level.") is True

    def test_no_overlap(self):
        assert _text_overlap("Apples are red.", "Bananas are yellow.") is False

    def test_empty_strings(self):
        assert _text_overlap("", "something") is False
        assert _text_overlap("something", "") is False

    def test_case_insensitive(self):
        assert _text_overlap("WATER BOILS", "water boils at 100C") is True


# ---------------------------------------------------------------------------
# check_claim_table_consistency
# ---------------------------------------------------------------------------

class TestCheckClaimTableConsistency:
    def test_consistent_no_issues(self):
        claim_table = [ClaimEntry(claim="Water boils at 100C.", category="Observed", justification="Well established fact.")]
        findings = []
        atomic_claims = [{"text": "Water boils at 100C.", "nli_result": {"confidence_tier": "strong_support"}}]
        result = check_claim_table_consistency(claim_table, findings, atomic_claims)
        assert result["consistent"] is True
        assert result["should_downgrade"] is False

    def test_nli_gpt2_mismatch_detected(self):
        claim_table = [ClaimEntry(claim="The rate is 73%.", category="Observed", justification="Common knowledge.")]
        findings = []
        atomic_claims = [{
            "text": "The rate is 73%.",
            "nli_result": {"confidence_tier": "strong_contradiction", "worst_contradiction": 0.9},
        }]
        result = check_claim_table_consistency(claim_table, findings, atomic_claims)
        assert result["consistent"] is False
        assert result["should_downgrade"] is True
        assert any(i["type"] == "nli_gpt2_mismatch" for i in result["issues"])

    def test_shallow_verification_detected(self):
        claim_table = [
            ClaimEntry(claim=f"c{i}", category="Observed", justification="ok")
            for i in range(6)
        ]
        # Make 4 of 6 have short justifications
        claim_table[0].justification = "x"
        claim_table[1].justification = ""
        claim_table[2].justification = "y"
        claim_table[3].justification = "z"
        findings = []
        result = check_claim_table_consistency(claim_table, findings, [])
        assert any(i["type"] == "shallow_verification" for i in result["issues"])

    def test_suspiciously_clean(self):
        claim_table = [
            ClaimEntry(claim=f"c{i}", category="Observed", justification="Good justification here.")
            for i in range(5)
        ]
        findings = []  # Zero findings on 5 factual claims
        result = check_claim_table_consistency(claim_table, findings, [])
        assert any(i["type"] == "suspiciously_clean" for i in result["issues"])

    def test_no_suspicion_with_few_claims(self):
        claim_table = [
            ClaimEntry(claim="c1", category="Observed", justification="Good justification here."),
        ]
        findings = []
        result = check_claim_table_consistency(claim_table, findings, [])
        # Only 1 claim, not suspicious
        assert not any(i["type"] == "suspiciously_clean" for i in result["issues"])

    def test_no_mismatch_on_unsupported_category(self):
        # If GPT-2 already marked as Unsupported, no mismatch even if NLI contradicts
        claim_table = [ClaimEntry(claim="The rate is 73%.", category="Unsupported", justification="No source.")]
        findings = [{"type": "T1", "severity": "hard", "detail": "fab stat"}]
        atomic_claims = [{
            "text": "The rate is 73%.",
            "nli_result": {"confidence_tier": "strong_contradiction"},
        }]
        result = check_claim_table_consistency(claim_table, findings, atomic_claims)
        assert not any(i["type"] == "nli_gpt2_mismatch" for i in result["issues"])


# ---------------------------------------------------------------------------
# meta_verify_pass
# ---------------------------------------------------------------------------

class TestMetaVerifyPass:
    def test_skips_non_high_stakes(self):
        result = meta_verify_pass(
            flags={"advice_requested": True},
            claim_table=[], findings=[], atomic_claims=[],
            confidence_label="High",
        )
        assert result["ran"] is False
        assert result["adjusted_label"] == "High"

    def test_runs_on_legal_mode(self):
        result = meta_verify_pass(
            flags={"legal_mode": True},
            claim_table=[ClaimEntry(claim="c1", category="Observed", justification="Good justification here.")],
            findings=[],
            atomic_claims=[],
            confidence_label="High",
        )
        assert result["ran"] is True

    def test_downgrade_on_nli_mismatch(self):
        claim_table = [ClaimEntry(claim="The rate is 73%.", category="Observed", justification="Ok.")]
        atomic_claims = [{
            "text": "The rate is 73%.",
            "nli_result": {"confidence_tier": "strong_contradiction", "worst_contradiction": 0.9},
        }]
        result = meta_verify_pass(
            flags={"legal_mode": True},
            claim_table=claim_table, findings=[],
            atomic_claims=atomic_claims,
            confidence_label="High",
        )
        assert result["adjusted_label"] == "Medium"
        assert result["should_reverify"] is True

    def test_no_downgrade_when_consistent(self):
        claim_table = [ClaimEntry(claim="Water boils at 100C.", category="Observed", justification="Well known.")]
        atomic_claims = [{
            "text": "Water boils at 100C.",
            "nli_result": {"confidence_tier": "strong_support"},
        }]
        result = meta_verify_pass(
            flags={"percent_requested": True},
            claim_table=claim_table, findings=[],
            atomic_claims=atomic_claims,
            confidence_label="High",
        )
        assert result["adjusted_label"] == "High"
        assert result["should_reverify"] is False


class TestMetaVerifyFail:
    """False-FAIL overrides must not drop hard T5."""

    def test_hard_t5_kept_when_advice_requested(self):
        findings = [{
            "type": "T5",
            "severity": "hard",
            "detail": "GPT-1 told the user this will improve outcomes.",
        }]
        result = meta_verify_fail(
            flags={"legal_mode": True, "advice_requested": True},
            claim_table=[],
            findings=findings,
            atomic_claims=[],
        )
        assert result["ran"] is True
        assert result["override_to_pass"] is False
        assert len(result["adjusted_findings"]) == 1

    def test_soft_t5_dropped_when_advice_requested(self):
        findings = [{
            "type": "T5",
            "severity": "soft",
            "detail": "Suggested next steps for the user.",
        }]
        result = meta_verify_fail(
            flags={"percent_requested": True, "advice_requested": True},
            claim_table=[],
            findings=findings,
            atomic_claims=[],
        )
        assert result["override_to_pass"] is True
        assert result["adjusted_findings"] == []
