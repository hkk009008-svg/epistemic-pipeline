"""Tests for pipeline.arbiter -- parse_gpt3() and apply_edits().

All functions under test are fully deterministic (no LLM calls).
"""
from __future__ import annotations

import json
import pytest

from pipeline.arbiter import (
    parse_gpt3, apply_edits, apply_edits_by_id,
    check_poisoning_threshold, guard_arbiter_decision,
)
from pipeline.models import EditEntry, ClaimEntry


# ===================================================================
# parse_gpt3() -- BLOCK decision
# ===================================================================


class TestParseGpt3Block:
    """BLOCK decision parsing."""

    def test_block_decision(self, arbiter_block_json: str):
        decision, rationale, edits, policy_notes = parse_gpt3(arbiter_block_json)
        assert decision == "BLOCK"
        assert len(rationale) == 1
        assert "Fabricated statistic" in rationale[0]
        assert edits == []
        assert len(policy_notes) == 1

    def test_block_decision_uppercase(self):
        """Decision string should be uppercased."""
        raw = json.dumps({
            "arbiter_decision": "block",
            "rationale": [],
            "edits_for_gpt1": [],
            "final_policy_notes": [],
        })
        decision, _, _, _ = parse_gpt3(raw)
        assert decision == "BLOCK"

    def test_lowercase_edit_action_is_normalized(self):
        raw = json.dumps({
            "arbiter_decision": "ALLOW_WITH_EDITS",
            "rationale": [],
            "edits_for_gpt1": [
                {"action": "delete", "target": "bad claim", "replacement": ""},
            ],
            "final_policy_notes": [],
        })
        _, _, edits, _ = parse_gpt3(raw)
        assert edits[0].action == "DELETE"
        result = apply_edits("Original with bad claim.", edits)
        assert "DELETE the following text" in result


# ===================================================================
# parse_gpt3() -- ALLOW_WITH_EDITS decision
# ===================================================================


class TestParseGpt3AllowWithEdits:
    """ALLOW_WITH_EDITS decision parsing."""

    def test_allow_with_edits(self, arbiter_allow_with_edits_json: str):
        decision, rationale, edits, policy_notes = parse_gpt3(arbiter_allow_with_edits_json)
        assert decision == "ALLOW_WITH_EDITS"
        assert len(rationale) == 1
        assert len(edits) == 2
        assert edits[0].action == "REWRITE"
        assert edits[0].target == "this will improve your odds"
        assert "no guarantee" in edits[0].replacement
        assert edits[1].action == "DELETE"
        assert "Studies suggest" in edits[1].target
        assert len(policy_notes) == 1

    def test_edit_entries_are_edit_entry_models(self, arbiter_allow_with_edits_json: str):
        _, _, edits, _ = parse_gpt3(arbiter_allow_with_edits_json)
        for edit in edits:
            assert isinstance(edit, EditEntry)


# ===================================================================
# parse_gpt3() -- ALLOW_AS_UNKNOWN_ONLY decision
# ===================================================================


class TestParseGpt3AllowAsUnknown:
    """ALLOW_AS_UNKNOWN_ONLY decision parsing."""

    def test_allow_as_unknown(self, arbiter_allow_as_unknown_json: str):
        decision, rationale, edits, policy_notes = parse_gpt3(arbiter_allow_as_unknown_json)
        assert decision == "ALLOW_AS_UNKNOWN_ONLY"
        assert len(rationale) == 1
        assert len(edits) == 1
        assert edits[0].action == "MOVE_TO_UNKNOWN"
        assert "medically safer" in edits[0].target
        assert "unknown" in edits[0].replacement.lower()
        assert policy_notes == []


# ===================================================================
# parse_gpt3() -- missing / default fields
# ===================================================================


class TestParseGpt3Defaults:
    """Missing fields should get sensible defaults."""

    def test_missing_decision_defaults_to_block(self):
        raw = json.dumps({
            "rationale": ["Some rationale."],
            "edits_for_gpt1": [],
            "final_policy_notes": [],
        })
        decision, _, _, _ = parse_gpt3(raw)
        assert decision == "BLOCK"

    def test_missing_rationale_defaults_to_empty(self):
        raw = json.dumps({
            "arbiter_decision": "ALLOW_WITH_EDITS",
            "edits_for_gpt1": [],
            "final_policy_notes": [],
        })
        _, rationale, _, _ = parse_gpt3(raw)
        assert rationale == []

    def test_missing_edits_defaults_to_empty(self):
        raw = json.dumps({
            "arbiter_decision": "BLOCK",
            "rationale": ["Reason."],
            "final_policy_notes": [],
        })
        _, _, edits, _ = parse_gpt3(raw)
        assert edits == []

    def test_missing_policy_notes_defaults_to_empty(self):
        raw = json.dumps({
            "arbiter_decision": "BLOCK",
            "rationale": [],
            "edits_for_gpt1": [],
        })
        _, _, _, policy_notes = parse_gpt3(raw)
        assert policy_notes == []

    def test_edit_entry_missing_fields(self):
        """Edit entries with missing fields get empty-string defaults."""
        raw = json.dumps({
            "arbiter_decision": "ALLOW_WITH_EDITS",
            "rationale": [],
            "edits_for_gpt1": [
                {"action": "DELETE"}
            ],
            "final_policy_notes": [],
        })
        _, _, edits, _ = parse_gpt3(raw)
        assert len(edits) == 1
        assert edits[0].action == "DELETE"
        assert edits[0].target == ""
        assert edits[0].replacement == ""


# ===================================================================
# parse_gpt3() -- malformed input
# ===================================================================


class TestParseGpt3MalformedInput:
    """Malformed input should fallback to BLOCK."""

    def test_garbage_string(self):
        decision, rationale, edits, policy_notes = parse_gpt3("not json!!!")
        assert decision == "BLOCK"
        assert any("parse error" in r.lower() for r in rationale)
        assert edits == []
        assert policy_notes == []

    def test_empty_string(self):
        decision, rationale, edits, policy_notes = parse_gpt3("")
        assert decision == "BLOCK"
        assert len(rationale) == 1

    def test_valid_json_wrong_schema(self):
        raw = json.dumps({"foo": "bar"})
        decision, rationale, edits, policy_notes = parse_gpt3(raw)
        # Missing arbiter_decision -> defaults to "BLOCK"
        assert decision == "BLOCK"
        assert rationale == []
        assert edits == []


class TestParseGpt3InvalidDecisionFailsClosed:
    """Unrecognized arbiter decisions must BLOCK, not fall through to rewrite."""

    def test_unknown_decision_becomes_block(self):
        raw = json.dumps({
            "arbiter_decision": "ALLOW",
            "rationale": ["Looks fine."],
            "edits_for_gpt1": [],
            "final_policy_notes": [],
        })
        decision, rationale, edits, _ = parse_gpt3(raw)
        assert decision == "BLOCK"
        assert edits == []
        assert any("Invalid arbiter_decision" in line for line in rationale)

    def test_empty_decision_becomes_block(self):
        raw = json.dumps({
            "arbiter_decision": "",
            "rationale": [],
            "edits_for_gpt1": [],
            "final_policy_notes": [],
        })
        decision, _, _, _ = parse_gpt3(raw)
        assert decision == "BLOCK"

    def test_string_rationale_is_coerced_to_list(self):
        raw = json.dumps({
            "arbiter_decision": "BLOCK",
            "rationale": "single string",
            "edits_for_gpt1": [],
            "final_policy_notes": "note",
        })
        decision, rationale, _, policy_notes = parse_gpt3(raw)
        assert decision == "BLOCK"
        assert rationale == ["single string"]
        assert policy_notes == ["note"]


# ===================================================================
# parse_gpt3() -- markdown-wrapped JSON
# ===================================================================


class TestParseGpt3MarkdownWrapped:
    """GPT-3 sometimes wraps output in markdown fences."""

    def test_fenced_json(self):
        inner = json.dumps({
            "arbiter_decision": "ALLOW_WITH_EDITS",
            "rationale": ["Fixable issue."],
            "edits_for_gpt1": [
                {"action": "REWRITE", "target": "old text", "replacement": "new text"}
            ],
            "final_policy_notes": ["Check for outcome promises."],
        })
        raw = f"```json\n{inner}\n```"
        decision, rationale, edits, policy_notes = parse_gpt3(raw)
        assert decision == "ALLOW_WITH_EDITS"
        assert len(edits) == 1
        assert edits[0].action == "REWRITE"


# ===================================================================
# apply_edits() -- DELETE action
# ===================================================================


class TestApplyEditsDelete:
    """DELETE action generates correct instruction."""

    def test_single_delete(self):
        edits = [EditEntry(action="DELETE", target="Bad claim here.", replacement="")]
        result = apply_edits("Original response text.", edits)
        assert 'DELETE the following text: "Bad claim here."' in result
        assert "Original response text." in result

    def test_delete_instruction_format(self):
        edits = [EditEntry(action="DELETE", target="remove this", replacement="")]
        result = apply_edits("Some output.", edits)
        assert result.startswith("You previously produced this response:")
        assert "---\nSome output.\n---" in result
        assert "Apply ONLY these edits" in result
        assert "Output the corrected response in full." in result


# ===================================================================
# apply_edits() -- REWRITE action
# ===================================================================


class TestApplyEditsRewrite:
    """REWRITE action generates correct instruction."""

    def test_single_rewrite(self):
        edits = [
            EditEntry(
                action="REWRITE",
                target="will improve your odds",
                replacement="may help with procedure; no guarantee",
            )
        ]
        result = apply_edits("This will improve your odds of success.", edits)
        assert 'REWRITE "will improve your odds" to: "may help with procedure; no guarantee"' in result

    def test_rewrite_preserves_original(self):
        edits = [EditEntry(action="REWRITE", target="old", replacement="new")]
        result = apply_edits("The old approach.", edits)
        assert "The old approach." in result


# ===================================================================
# apply_edits() -- MOVE_TO_UNKNOWN action
# ===================================================================


class TestApplyEditsMoveToUnknown:
    """MOVE_TO_UNKNOWN action generates correct instruction."""

    def test_single_move_to_unknown(self):
        edits = [
            EditEntry(
                action="MOVE_TO_UNKNOWN",
                target="73% success rate",
                replacement="Success rate is unknown.",
            )
        ]
        result = apply_edits("The 73% success rate is notable.", edits)
        assert 'MOVE the following to the Unknowns section: "73% success rate"' in result
        assert 'reframe as: "Success rate is unknown."' in result
        # Check for em-dash
        assert "\u2014" in result


# ===================================================================
# apply_edits() -- multiple edits
# ===================================================================


class TestApplyEditsMultiple:
    """Multiple edits of different types."""

    def test_all_three_actions(self, sample_edits: list):
        result = apply_edits("GPT-1 response with bad claims.", sample_edits)
        # All three instruction types present
        assert "DELETE" in result
        assert "REWRITE" in result
        assert "MOVE the following to the Unknowns section" in result
        # Each is a bullet
        lines = result.split("\n")
        bullet_lines = [line for line in lines if line.strip().startswith("- ")]
        assert len(bullet_lines) == 3

    def test_edit_order_preserved(self, sample_edits: list):
        result = apply_edits("Some output.", sample_edits)
        delete_pos = result.index("DELETE")
        rewrite_pos = result.index("REWRITE")
        move_pos = result.index("MOVE")
        assert delete_pos < rewrite_pos < move_pos


# ===================================================================
# apply_edits() -- empty edits
# ===================================================================


class TestApplyEditsEmpty:
    """Edge case: no edits."""

    def test_empty_edit_list(self):
        result = apply_edits("GPT-1 output.", [])
        assert "GPT-1 output." in result
        assert "Apply ONLY these edits" in result
        # No bullet points
        lines = result.split("\n")
        bullet_lines = [line for line in lines if line.strip().startswith("- ")]
        assert len(bullet_lines) == 0


# ===================================================================
# apply_edits() -- output structure
# ===================================================================


class TestApplyEditsOutputStructure:
    """Verify the overall structure of the rewrite prompt."""

    def test_contains_original_output(self):
        original = "This is the GPT-1 generated response."
        edits = [EditEntry(action="DELETE", target="something", replacement="")]
        result = apply_edits(original, edits)
        assert original in result

    def test_has_fence_markers(self):
        result = apply_edits("Output.", [EditEntry(action="DELETE", target="x", replacement="")])
        assert "---\nOutput.\n---" in result

    def test_ends_with_instruction(self):
        result = apply_edits("Out.", [EditEntry(action="DELETE", target="x", replacement="")])
        assert result.strip().endswith("Output the corrected response in full.")

    def test_multiline_original_preserved(self):
        original = "Line 1.\nLine 2.\nLine 3."
        result = apply_edits(original, [])
        assert "Line 1.\nLine 2.\nLine 3." in result


# ===================================================================
# apply_edits_by_id() -- deterministic claim edits
# ===================================================================


class TestApplyEditsById:
    """Verify deterministic ID-based edits on atomic claim dictionaries."""

    def test_delete_by_id(self):
        claims = [
            {"claim_id": "c1", "text": "Claim 1"},
            {"claim_id": "c2", "text": "Claim 2 to delete"},
            {"claim_id": "c3", "text": "Claim 3"},
        ]
        edits = [EditEntry(action="DELETE", target="", replacement="", target_id="c2")]
        modified, summary = apply_edits_by_id(claims, edits)
        assert len(modified) == 2
        assert [c["claim_id"] for c in modified] == ["c1", "c3"]
        assert "DELETED claim c2" in summary

    def test_delete_claim_with_claim_key_instead_of_text(self):
        claims = [
            {"claim_id": "c1", "claim": "Claim with claim key"},
        ]
        edits = [EditEntry(action="DELETE", target="", replacement="", target_id="c1")]
        modified, summary = apply_edits_by_id(claims, edits)
        assert len(modified) == 0
        assert "DELETED claim c1" in summary

    def test_rewrite_by_id(self):
        claims = [
            {"claim_id": "c1", "text": "Original text"},
        ]
        edits = [EditEntry(action="REWRITE", target="", replacement="Rewritten text", target_id="c1")]
        modified, summary = apply_edits_by_id(claims, edits)
        assert modified[0]["text"] == "Rewritten text"
        assert "REWROTE claim c1" in summary

    def test_move_to_unknown_by_id(self):
        claims = [
            {"claim_id": "c1", "text": "Speculative claim"},
        ]
        edits = [EditEntry(action="MOVE_TO_UNKNOWN", target="", replacement="", target_id="c1")]
        modified, summary = apply_edits_by_id(claims, edits)
        assert modified[0]["is_unknown"] is True
        assert "Unknown(Actionable)" in modified[0]["text"]
        assert "MOVED claim c1" in summary


# ===================================================================
# check_poisoning_threshold() -- Unit Tests
# ===================================================================


class TestCheckPoisoningThreshold:
    """Unit tests for check_poisoning_threshold calculations and boundaries."""

    def test_exact_boundary_35_percent_not_poisoned(self):
        """Exactly 35.0% unsupported claims (<= 0.35) and 0 hard findings -> is_poisoned = False."""
        # 7 unsupported out of 20 = 35.0%
        claims = [
            ClaimEntry(claim=f"Supported claim {i}", category="Supported", justification="Source")
            for i in range(13)
        ] + [
            ClaimEntry(claim=f"Unsupported claim {i}", category="Unsupported", justification="None")
            for i in range(7)
        ]
        res = check_poisoning_threshold(claims, [], unsupported_threshold=0.35, hard_threshold=2)
        assert res["total_claims"] == 20
        assert res["unsupported_count"] == 7
        assert res["unsupported_ratio"] == pytest.approx(0.35)
        assert res["hard_count"] == 0
        assert res["is_poisoned"] is False

    def test_exact_boundary_35_1_percent_is_poisoned(self):
        """35.1% unsupported claims (> 0.35) -> is_poisoned = True."""
        # 351 unsupported out of 1000 = 35.1%
        claims = [
            {"claim": f"Good {i}", "category": "supported"} for i in range(649)
        ] + [
            {"claim": f"Bad {i}", "category": "unsupported"} for i in range(351)
        ]
        res = check_poisoning_threshold(claims, [], unsupported_threshold=0.35, hard_threshold=2)
        assert res["total_claims"] == 1000
        assert res["unsupported_count"] == 351
        assert res["unsupported_ratio"] == pytest.approx(0.351)
        assert res["is_poisoned"] is True

    def test_hard_violation_boundary_1_vs_2(self):
        """1 hard violation is not poisoned; 2 hard violations is poisoned."""
        claims = [
            ClaimEntry(claim="Claim 1", category="Supported", justification="Source"),
            ClaimEntry(claim="Claim 2", category="Supported", justification="Source"),
        ]
        findings_1 = [{"type": "T1", "severity": "hard", "detail": "Single hard violation"}]
        findings_2 = [
            {"type": "T1", "severity": "hard", "detail": "First hard violation"},
            {"type": "T1", "severity": "hard", "detail": "Second hard violation"},
        ]

        res_1 = check_poisoning_threshold(claims, findings_1, hard_threshold=2)
        assert res_1["hard_count"] == 1
        assert res_1["is_poisoned"] is False

        res_2 = check_poisoning_threshold(claims, findings_2, hard_threshold=2)
        assert res_2["hard_count"] == 2
        assert res_2["is_poisoned"] is True

    def test_zero_claims_empty_claim_table(self):
        """Empty claim table: 0 hard findings is not poisoned, 2 hard findings is poisoned."""
        res_empty = check_poisoning_threshold([], [])
        assert res_empty["total_claims"] == 0
        assert res_empty["unsupported_count"] == 0
        assert res_empty["unsupported_ratio"] == 0.0
        assert res_empty["hard_count"] == 0
        assert res_empty["is_poisoned"] is False

        res_with_hard = check_poisoning_threshold(
            [],
            [
                {"type": "T1", "severity": "hard", "detail": "Out of bounds citation"},
                {"type": "T1", "severity": "hard", "detail": "Fabricated statistic"},
            ],
        )
        assert res_with_hard["total_claims"] == 0
        assert res_with_hard["hard_count"] == 2
        assert res_with_hard["is_poisoned"] is True

    def test_all_supported_claims(self):
        """100% supported claims -> ratio 0.0, is_poisoned False."""
        claims = [
            ClaimEntry(claim="Clean claim 1", category="Observed", justification="Ref 1"),
            ClaimEntry(claim="Clean claim 2", category="Supported", justification="Ref 2"),
        ]
        res = check_poisoning_threshold(claims, [])
        assert res["total_claims"] == 2
        assert res["unsupported_count"] == 0
        assert res["unsupported_ratio"] == 0.0
        assert res["is_poisoned"] is False

    def test_all_unsupported_claims(self):
        """100% unsupported claims -> ratio 1.0, is_poisoned True."""
        claims = [
            {"claim": "Bad claim 1", "category": "unsupported"},
            {"claim": "Bad claim 2", "category": "unsupported_inferential"},
        ]
        res = check_poisoning_threshold(claims, [])
        assert res["total_claims"] == 2
        assert res["unsupported_count"] == 2
        assert res["unsupported_ratio"] == 1.0
        assert res["is_poisoned"] is True

    def test_unsupported_inferential_and_contradicted_categories(self):
        """Includes unsupported, unsupported_inferential, contradicted, refuted, fabricated."""
        claims = [
            {"claim": "c1", "category": "unsupported_inferential"},
            {"claim": "c2", "category": "contradicted"},
            {"claim": "c3", "category": "refuted"},
            {"claim": "c4", "category": "fabricated"},
            {"claim": "c5", "category": "supported"},
        ]
        res = check_poisoning_threshold(claims, [])
        assert res["unsupported_count"] == 4
        assert res["unsupported_ratio"] == pytest.approx(0.8)
        assert res["is_poisoned"] is True

    def test_soft_findings_do_not_increase_hard_count(self):
        """Soft findings do not count toward hard_count threshold."""
        claims = [{"claim": "Clean", "category": "supported"}]
        soft_findings = [
            {"type": "T4", "severity": "soft", "detail": f"Soft issue {i}"}
            for i in range(10)
        ]
        res = check_poisoning_threshold(claims, soft_findings, hard_threshold=2)
        assert res["hard_count"] == 0
        assert res["is_poisoned"] is False


# ===================================================================
# guard_arbiter_decision() -- Unit Tests
# ===================================================================


class TestGuardArbiterDecision:
    """Unit tests for guard_arbiter_decision transitions and rationale notes."""

    def test_heavily_poisoned_overrides_allow_with_edits_to_block(self):
        """ALLOW_WITH_EDITS is overridden to BLOCK when unsupported ratio > 35%."""
        claims = [
            {"claim": "Good", "category": "supported"},
            {"claim": "Bad 1", "category": "unsupported"},
            {"claim": "Bad 2", "category": "unsupported"},
        ]  # 2/3 = 66.7%
        decision, notes = guard_arbiter_decision("ALLOW_WITH_EDITS", claims, [])
        assert decision == "BLOCK"
        assert len(notes) >= 1
        assert "overridden" in notes[0].lower()
        assert "heavily poisoned" in notes[0].lower()

    def test_heavily_poisoned_overrides_allow_as_unknown_to_block(self):
        """ALLOW_AS_UNKNOWN_ONLY is overridden to BLOCK when >= 2 hard violations."""
        findings = [
            {"type": "T1", "severity": "hard", "detail": "Fabricated citation"},
            {"type": "T1", "severity": "hard", "detail": "Fabricated number"},
        ]
        decision, notes = guard_arbiter_decision("ALLOW_AS_UNKNOWN_ONLY", [], findings)
        assert decision == "BLOCK"
        assert any("overridden" in n.lower() for n in notes)

    def test_heavily_poisoned_confirms_block(self):
        """BLOCK decision is confirmed when heavily poisoned."""
        claims = [{"claim": "Bad", "category": "unsupported"}]
        decision, notes = guard_arbiter_decision("BLOCK", claims, [])
        assert decision == "BLOCK"
        assert any("confirmed" in n.lower() for n in notes)

    def test_lightly_poisoned_overrides_block_to_allow_with_edits_when_truthful(self):
        """BLOCK is overridden to ALLOW_WITH_EDITS when lightly poisoned with truthful claims."""
        claims = [
            ClaimEntry(claim="Supported fact 1", category="Supported", justification="Source 1"),
            ClaimEntry(claim="Supported fact 2", category="Observed", justification="Source 2"),
            ClaimEntry(claim="Inferred fact", category="Inference", justification="Source 3"),
            ClaimEntry(claim="Bad claim", category="Unsupported", justification="None"),
        ]  # 1/4 = 25% unsupported (<= 35%)
        findings = [{"type": "T1", "severity": "hard", "detail": "Single hard violation"}]  # 1 hard (< 2)

        decision, notes = guard_arbiter_decision("BLOCK", claims, findings)
        assert decision == "ALLOW_WITH_EDITS"
        assert any("overridden from block to allow_with_edits" in n.lower() for n in notes)

    def test_lightly_poisoned_keeps_block_when_no_truthful_claims(self):
        """BLOCK is preserved if there are no salvageable truthful claims in claim table."""
        claims = [
            ClaimEntry(claim="Hypothesis 1", category="Hypothesis", justification="Speculation"),
            ClaimEntry(claim="Unknown 1", category="Unknown", justification="Unknown"),
        ]  # 0% unsupported, 0 hard, but 0 truthful (no supported, observed, inference, user-provided)
        decision, notes = guard_arbiter_decision("BLOCK", claims, [])
        assert decision == "BLOCK"
        assert any("no salvageable" in n.lower() for n in notes)

    def test_allow_as_unknown_preserved_when_zero_hard_violations(self):
        """ALLOW_AS_UNKNOWN_ONLY is preserved when 0 hard violations exist and ratio <= 35%."""
        claims = [
            ClaimEntry(claim="Speculative point", category="Hypothesis", justification="Unknown context"),
        ]
        decision, notes = guard_arbiter_decision("ALLOW_AS_UNKNOWN_ONLY", claims, [])
        assert decision == "ALLOW_AS_UNKNOWN_ONLY"
        assert any("preserved" in n.lower() for n in notes)

    def test_allow_as_unknown_with_one_hard_violation_and_truthful_content(self):
        """ALLOW_AS_UNKNOWN_ONLY with 1 hard violation and truthful content transitions to ALLOW_WITH_EDITS."""
        claims = [
            ClaimEntry(claim="Supported point", category="Supported", justification="Source"),
        ]
        findings = [{"type": "T1", "severity": "hard", "detail": "Hard violation in non-essential part"}]
        decision, notes = guard_arbiter_decision("ALLOW_AS_UNKNOWN_ONLY", claims, findings)
        assert decision == "ALLOW_WITH_EDITS"
        assert any("allow_with_edits" in n.lower() for n in notes)

    def test_allow_with_edits_confirmed_when_lightly_poisoned(self):
        """ALLOW_WITH_EDITS is confirmed when draft is lightly poisoned."""
        claims = [
            ClaimEntry(claim="Clean fact", category="Observed", justification="Ref"),
        ]
        decision, notes = guard_arbiter_decision("ALLOW_WITH_EDITS", claims, [])
        assert decision == "ALLOW_WITH_EDITS"
        assert any("confirmed" in n.lower() for n in notes)

    def test_empty_decision_defaults_to_block_safely(self):
        """Empty string decision defaults safely."""
        claims = [{"claim": "Clean", "category": "supported"}]
        decision, notes = guard_arbiter_decision("", claims, [])
        # Default was BLOCK, converted to ALLOW_WITH_EDITS because of truthful claim
        assert decision == "ALLOW_WITH_EDITS"

