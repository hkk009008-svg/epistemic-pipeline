"""Tests for pipeline.arbiter -- parse_gpt3() and apply_edits().

All functions under test are fully deterministic (no LLM calls).
"""
from __future__ import annotations

import json

from pipeline.arbiter import parse_gpt3, apply_edits
from pipeline.models import EditEntry


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
