"""Tests for pipeline.verifier -- parse_gpt2().

All functions under test are fully deterministic (no LLM calls).
"""
from __future__ import annotations

import json

from pipeline.verifier import parse_gpt2, _all_soft


# ===================================================================
# _all_soft() helper
# ===================================================================


class TestAllSoft:
    """Internal helper that checks if all findings are severity=soft."""

    def test_all_soft_true(self):
        findings = [
            {"severity": "soft"},
            {"severity": "soft"},
        ]
        assert _all_soft(findings) is True

    def test_mixed_severities(self):
        findings = [
            {"severity": "soft"},
            {"severity": "hard"},
        ]
        assert _all_soft(findings) is False

    def test_all_hard(self):
        findings = [{"severity": "hard"}]
        assert _all_soft(findings) is False

    def test_empty_list(self):
        assert _all_soft([]) is False

    def test_missing_severity_key(self):
        findings = [{"type": "something"}]
        assert _all_soft(findings) is False


# ===================================================================
# parse_gpt2() -- basic parsing
# ===================================================================


class TestParseGpt2PassVerdict:
    """PASS verdicts with no findings."""

    def test_pass_verdict(self, gpt2_pass_json: str):
        claim_table, violations, verdict, findings, reasoning = parse_gpt2(gpt2_pass_json)
        assert verdict == "PASS"
        assert violations == []
        assert findings == []
        assert len(claim_table) == 1
        assert claim_table[0].claim == "Water boils at 100C at sea level."
        assert claim_table[0].category == "Supported"

    def test_pass_with_no_flags(self, gpt2_pass_json: str):
        claim_table, violations, verdict, findings, _ = parse_gpt2(gpt2_pass_json, flags=None)
        assert verdict == "PASS"


class TestParseGpt2HardFindings:
    """Hard findings always produce FAIL."""

    def test_single_hard_finding(self, gpt2_fail_hard_json: str):
        claim_table, violations, verdict, findings, _ = parse_gpt2(gpt2_fail_hard_json)
        assert verdict == "FAIL"
        assert len(findings) == 1
        assert findings[0]["severity"] == "hard"
        assert findings[0]["type"] == "Fabricated statistic"
        assert "Fabricated statistic" in violations

    def test_hard_finding_with_any_flags(self, gpt2_fail_hard_json: str, flags_advice_requested: dict):
        """Hard findings cause FAIL regardless of flags."""
        _, _, verdict, findings, _ = parse_gpt2(gpt2_fail_hard_json, flags=flags_advice_requested)
        assert verdict == "FAIL"
        assert len(findings) == 1


class TestParseGpt2SoftAccumulation:
    """Three or more soft findings trigger FAIL."""

    def test_three_soft_findings_fail(self, gpt2_fail_soft_accumulation_json: str):
        _, violations, verdict, findings, _ = parse_gpt2(gpt2_fail_soft_accumulation_json)
        assert verdict == "FAIL"
        assert len(findings) == 3
        assert all(f["severity"] == "soft" for f in findings)

    def test_two_soft_findings_pass(self):
        """Two soft findings should be PASS (threshold is >= 3)."""
        raw = json.dumps({
            "claim_table": [],
            "findings": [
                {"type": "Overconfidence", "severity": "soft", "detail": "X"},
                {"type": "Missing jurisdiction", "severity": "soft", "detail": "Y"},
            ],
            "verdict": "FAIL",  # GPT-2 may say FAIL, but we recompute
        })
        _, _, verdict, findings, _ = parse_gpt2(raw)
        assert verdict == "PASS"
        assert len(findings) == 2

    def test_one_soft_finding_pass(self):
        """One soft finding should be PASS."""
        raw = json.dumps({
            "claim_table": [],
            "findings": [
                {"type": "Overconfidence", "severity": "soft", "detail": "Z"},
            ],
            "verdict": "FAIL",
        })
        _, _, verdict, _, _ = parse_gpt2(raw)
        assert verdict == "PASS"


# ===================================================================
# parse_gpt2() -- verdict recomputation
# ===================================================================


class TestParseGpt2VerdictRecomputation:
    """parse_gpt2 always recomputes the verdict, ignoring GPT-2's own verdict."""

    def test_gpt2_says_fail_but_no_findings_gives_pass(self):
        raw = json.dumps({
            "claim_table": [],
            "findings": [],
            "verdict": "FAIL",
        })
        _, _, verdict, _, _ = parse_gpt2(raw)
        assert verdict == "PASS"

    def test_gpt2_says_pass_but_hard_finding_gives_fail(self):
        raw = json.dumps({
            "claim_table": [],
            "findings": [
                {"type": "Fabricated citation", "severity": "hard", "detail": "Fake source."}
            ],
            "verdict": "PASS",
        })
        _, _, verdict, findings, _ = parse_gpt2(raw)
        assert verdict == "FAIL"
        assert len(findings) == 1

    def test_mixed_hard_and_soft(self):
        raw = json.dumps({
            "claim_table": [],
            "findings": [
                {"type": "Fabricated statistic", "severity": "hard", "detail": "No source."},
                {"type": "Overconfidence", "severity": "soft", "detail": "Too confident."},
            ],
            "verdict": "PASS",
        })
        _, _, verdict, _, _ = parse_gpt2(raw)
        assert verdict == "FAIL"


# ===================================================================
# parse_gpt2() -- prescriptive creep filtering
# ===================================================================


class TestParseGpt2PrescriptiveCreepFiltering:
    """When advice_requested=True, soft Prescriptive creep without outcome language is dropped."""

    def test_prescriptive_creep_dropped_when_advice_requested(
        self, gpt2_prescriptive_creep_no_outcome_json: str, flags_advice_requested: dict
    ):
        _, violations, verdict, findings, _ = parse_gpt2(
            gpt2_prescriptive_creep_no_outcome_json, flags=flags_advice_requested
        )
        assert verdict == "PASS"
        assert findings == []
        assert violations == []

    def test_prescriptive_creep_kept_when_no_advice_flag(
        self, gpt2_prescriptive_creep_no_outcome_json: str, flags_all_false: dict
    ):
        _, violations, verdict, findings, _ = parse_gpt2(
            gpt2_prescriptive_creep_no_outcome_json, flags=flags_all_false
        )
        assert verdict == "PASS"  # 1 soft < 3 threshold
        assert len(findings) == 1
        assert findings[0]["type"] == "Prescriptive creep"

    def test_prescriptive_creep_kept_when_no_flags(
        self, gpt2_prescriptive_creep_no_outcome_json: str
    ):
        """No flags at all -- prescriptive creep is NOT filtered."""
        _, _, verdict, findings, _ = parse_gpt2(gpt2_prescriptive_creep_no_outcome_json)
        assert len(findings) == 1

    def test_prescriptive_creep_with_outcome_kept_even_with_advice_flag(
        self, gpt2_prescriptive_creep_with_outcome_json: str, flags_advice_requested: dict
    ):
        """Outcome-promise language means the finding is kept even when advice is requested."""
        _, violations, verdict, findings, _ = parse_gpt2(
            gpt2_prescriptive_creep_with_outcome_json, flags=flags_advice_requested
        )
        assert len(findings) == 1
        assert findings[0]["type"] == "Prescriptive creep"
        assert "will improve" in findings[0]["detail"]

    def test_hard_prescriptive_creep_not_filtered(self, flags_advice_requested: dict):
        """Hard-severity prescriptive creep should never be filtered (even with advice flag)."""
        raw = json.dumps({
            "claim_table": [],
            "findings": [
                {
                    "type": "Prescriptive creep",
                    "severity": "hard",
                    "detail": "GPT-1 told user to take specific action.",
                }
            ],
            "verdict": "FAIL",
        })
        _, _, verdict, findings, _ = parse_gpt2(raw, flags=flags_advice_requested)
        assert verdict == "FAIL"
        assert len(findings) == 1

    def test_filtering_changes_verdict_from_fail_to_pass(self, flags_advice_requested: dict):
        """Three soft findings, but two are filterable prescriptive creep -> PASS."""
        raw = json.dumps({
            "claim_table": [],
            "findings": [
                {"type": "Prescriptive creep", "severity": "soft", "detail": "Step list given."},
                {"type": "Prescriptive creep", "severity": "soft", "detail": "More steps given."},
                {"type": "Overconfidence", "severity": "soft", "detail": "Too confident."},
            ],
            "verdict": "FAIL",
        })
        _, _, verdict, findings, _ = parse_gpt2(raw, flags=flags_advice_requested)
        # Two prescriptive creep entries filtered, leaving 1 soft -> PASS
        assert verdict == "PASS"
        assert len(findings) == 1
        assert findings[0]["type"] == "Overconfidence"


# ===================================================================
# parse_gpt2() -- backward compatibility (old "violations" key)
# ===================================================================


class TestParseGpt2BackwardCompat:
    """Old GPT-2 schema used 'violations' list of strings instead of 'findings'."""

    def test_old_violations_converted_to_findings(self):
        raw = json.dumps({
            "claim_table": [
                {"claim": "Some claim.", "category": "Inference", "justification": "Because."}
            ],
            "violations": ["Unsupported evidence reference", "Overconfidence"],
            "verdict": "FAIL",
        })
        claim_table, violations, verdict, findings, _ = parse_gpt2(raw)
        assert len(findings) == 2
        assert all(f["severity"] == "soft" for f in findings)
        assert findings[0]["type"] == "Unsupported evidence reference"
        assert findings[1]["type"] == "Overconfidence"
        # 2 soft findings -> PASS
        assert verdict == "PASS"

    def test_old_violations_three_gives_fail(self):
        raw = json.dumps({
            "claim_table": [],
            "violations": ["A", "B", "C"],
            "verdict": "FAIL",
        })
        _, _, verdict, findings, _ = parse_gpt2(raw)
        assert verdict == "FAIL"
        assert len(findings) == 3


# ===================================================================
# parse_gpt2() -- malformed / unparseable JSON
# ===================================================================


class TestParseGpt2MalformedInput:
    """Malformed or unparseable input should produce a hard parse error."""

    def test_garbage_string(self):
        claim_table, violations, verdict, findings, _ = parse_gpt2("not json at all !!!")
        assert verdict == "FAIL"
        assert claim_table == []
        assert any("parse error" in v.lower() for v in violations)
        assert findings[0]["severity"] == "hard"

    def test_empty_string(self):
        _, violations, verdict, findings, _ = parse_gpt2("")
        assert verdict == "FAIL"
        assert len(findings) == 1
        assert findings[0]["severity"] == "hard"

    def test_valid_json_but_wrong_schema(self):
        raw = json.dumps({"foo": "bar"})
        claim_table, violations, verdict, findings, _ = parse_gpt2(raw)
        # No claim_table, no findings -> should PASS (no hard/soft findings)
        assert verdict == "PASS"
        assert claim_table == []
        assert findings == []

    def test_partial_json(self):
        """Truncated JSON that extract_json cannot fix."""
        raw = '{"claim_table": [{"claim": "test"'
        # extract_json might fix this or not -- either way parse_gpt2 should not crash
        claim_table, violations, verdict, findings, _ = parse_gpt2(raw)
        # Result depends on whether extract_json can recover; just check no exception
        assert verdict in ("PASS", "FAIL")


# ===================================================================
# parse_gpt2() -- claim_table parsing
# ===================================================================


class TestParseGpt2ClaimTable:
    """Claim table entries are correctly mapped to ClaimEntry models."""

    def test_multiple_claims(self):
        raw = json.dumps({
            "claim_table": [
                {"claim": "Claim A", "category": "Supported", "justification": "Source X."},
                {"claim": "Claim B", "category": "Inference", "justification": "Derived."},
                {"claim": "Claim C", "category": "Unsupported", "justification": "No source."},
            ],
            "findings": [],
            "verdict": "PASS",
        })
        claim_table, _, _, _, _ = parse_gpt2(raw)
        assert len(claim_table) == 3
        assert claim_table[0].claim == "Claim A"
        assert claim_table[1].category == "Inference"
        assert claim_table[2].justification == "No source."

    def test_missing_fields_default(self):
        """Missing fields in claim entries should get defaults."""
        raw = json.dumps({
            "claim_table": [
                {"claim": "Only claim field."}
            ],
            "findings": [],
            "verdict": "PASS",
        })
        claim_table, _, _, _, _ = parse_gpt2(raw)
        assert len(claim_table) == 1
        assert claim_table[0].claim == "Only claim field."
        assert claim_table[0].category == "Unknown"
        assert claim_table[0].justification == ""

    def test_empty_claim_table(self):
        raw = json.dumps({
            "claim_table": [],
            "findings": [],
            "verdict": "PASS",
        })
        claim_table, _, _, _, _ = parse_gpt2(raw)
        assert claim_table == []


# ===================================================================
# parse_gpt2() -- JSON wrapped in markdown fences
# ===================================================================


class TestParseGpt2MarkdownWrapped:
    """GPT-2 sometimes wraps JSON in markdown code fences."""

    def test_fenced_json(self):
        inner = json.dumps({
            "claim_table": [{"claim": "X", "category": "Supported", "justification": "Y"}],
            "findings": [],
            "verdict": "PASS",
        })
        raw = f"```json\n{inner}\n```"
        claim_table, violations, verdict, findings, _ = parse_gpt2(raw)
        assert verdict == "PASS"
        assert len(claim_table) == 1

    def test_fenced_json_with_prose_preamble(self):
        inner = json.dumps({
            "claim_table": [],
            "findings": [{"type": "Overconfidence", "severity": "soft", "detail": "Meh."}],
            "verdict": "FAIL",
        })
        raw = f"Here is the verification result:\n```json\n{inner}\n```"
        _, _, verdict, findings, _ = parse_gpt2(raw)
        assert len(findings) == 1
        assert verdict == "PASS"  # Only 1 soft -> PASS


# ===================================================================
# parse_gpt2() -- reasoning trace extraction
# ===================================================================


class TestParseGpt2ReasoningTrace:
    """parse_gpt2 extracts reasoning_trace from GPT-2 JSON output."""

    def test_reasoning_trace_extracted(self):
        raw = json.dumps({
            "reasoning_trace": [
                "Step 1: Checking claim about boiling point...",
                "Step 2: No findings detected.",
            ],
            "claim_table": [
                {"claim": "Water boils at 100C.", "category": "Observed", "justification": "Physics."}
            ],
            "findings": [],
            "verdict": "PASS",
        })
        _, _, verdict, _, reasoning = parse_gpt2(raw)
        assert verdict == "PASS"
        assert len(reasoning) == 2
        assert "boiling point" in reasoning[0]

    def test_reasoning_trace_missing_returns_empty_list(self):
        """If reasoning_trace is not in JSON, return empty list."""
        raw = json.dumps({
            "claim_table": [],
            "findings": [],
            "verdict": "PASS",
        })
        _, _, _, _, reasoning = parse_gpt2(raw)
        assert reasoning == []

    def test_reasoning_trace_non_list_returns_empty(self):
        """If reasoning_trace is not a list, return empty list."""
        raw = json.dumps({
            "reasoning_trace": "not a list",
            "claim_table": [],
            "findings": [],
            "verdict": "PASS",
        })
        _, _, _, _, reasoning = parse_gpt2(raw)
        assert reasoning == []

    def test_malformed_input_returns_empty_reasoning(self):
        """Malformed input should produce empty reasoning."""
        _, _, _, _, reasoning = parse_gpt2("garbage")
        assert reasoning == []


# ===================================================================
# parse_gpt2() -- arbiter fields no longer in GPT-2
# ===================================================================


class TestParseGpt2NoArbiterFields:
    """GPT-2 no longer returns arbiter fields (handled by GPT-3 separately).
    Verify that extra fields in GPT-2 output are silently ignored."""

    def test_extra_arbiter_fields_ignored(self):
        """GPT-2 output with leftover arbiter fields should still parse correctly."""
        raw = json.dumps({
            "claim_table": [],
            "findings": [{"type": "T1", "severity": "hard", "detail": "Fabricated stat."}],
            "verdict": "FAIL",
            "arbiter_decision": "ALLOW_WITH_EDITS",
            "rationale": ["Claim can be deleted."],
        })
        _, _, verdict, findings, _ = parse_gpt2(raw)
        assert verdict == "FAIL"
        assert len(findings) == 1

    def test_returns_5_values(self):
        """parse_gpt2 now returns exactly 5 values."""
        raw = json.dumps({
            "claim_table": [],
            "findings": [{"type": "T1", "severity": "hard", "detail": "Missing."}],
            "verdict": "FAIL",
        })
        result = parse_gpt2(raw)
        assert len(result) == 5

    def test_pass_returns_5_values(self):
        """PASS verdict also returns exactly 5 values."""
        raw = json.dumps({
            "claim_table": [],
            "findings": [],
            "verdict": "PASS",
        })
        result = parse_gpt2(raw)
        assert len(result) == 5


# ===================================================================
# parse_gpt2() -- tier-aware severity and thresholds
# ===================================================================


class TestParseGpt2TierSeverity:
    """Tier parameter changes severity classification and soft thresholds."""

    def test_t2_hard_in_strict(self):
        """T2 is hard in strict tier."""
        raw = json.dumps({
            "claim_table": [],
            "findings": [{"type": "T2", "severity": "soft", "detail": "Typicality."}],
            "verdict": "PASS",
        })
        _, _, verdict, findings, _ = parse_gpt2(raw, tier="strict")
        assert findings[0]["severity"] == "hard"
        assert verdict == "FAIL"

    def test_t2_soft_in_standard(self):
        """T2 is soft in standard tier."""
        raw = json.dumps({
            "claim_table": [],
            "findings": [{"type": "T2", "severity": "hard", "detail": "Typicality."}],
            "verdict": "FAIL",
        })
        _, _, verdict, findings, _ = parse_gpt2(raw, tier="standard")
        assert findings[0]["severity"] == "soft"
        assert verdict == "PASS"  # 1 soft < 4 threshold

    def test_t3_hard_in_strict(self):
        """T3 is hard in strict tier."""
        raw = json.dumps({
            "claim_table": [],
            "findings": [{"type": "T3", "severity": "soft", "detail": "Causal claim."}],
            "verdict": "PASS",
        })
        _, _, verdict, findings, _ = parse_gpt2(raw, tier="strict")
        assert findings[0]["severity"] == "hard"
        assert verdict == "FAIL"

    def test_t3_soft_in_standard(self):
        """T3 is soft in standard tier."""
        raw = json.dumps({
            "claim_table": [],
            "findings": [{"type": "T3", "severity": "hard", "detail": "Causal claim."}],
            "verdict": "FAIL",
        })
        _, _, verdict, findings, _ = parse_gpt2(raw, tier="standard")
        assert findings[0]["severity"] == "soft"
        assert verdict == "PASS"

    def test_t7_hard_in_strict(self):
        """T7 is hard in strict tier."""
        raw = json.dumps({
            "claim_table": [],
            "findings": [{"type": "T7", "severity": "soft", "detail": "Stale fact."}],
            "verdict": "PASS",
        })
        _, _, verdict, findings, _ = parse_gpt2(raw, tier="strict")
        assert findings[0]["severity"] == "hard"
        assert verdict == "FAIL"

    def test_t7_soft_in_light(self):
        """T7 is soft in light tier."""
        raw = json.dumps({
            "claim_table": [],
            "findings": [{"type": "T7", "severity": "hard", "detail": "Stale fact."}],
            "verdict": "FAIL",
        })
        _, _, verdict, findings, _ = parse_gpt2(raw, tier="light")
        assert findings[0]["severity"] == "soft"
        assert verdict == "PASS"  # 1 soft < 5 threshold

    def test_t1_always_hard(self):
        """T1 is hard in ALL tiers."""
        raw = json.dumps({
            "claim_table": [],
            "findings": [{"type": "T1", "severity": "hard", "detail": "Fabricated."}],
            "verdict": "FAIL",
        })
        for tier in ("strict", "standard", "light"):
            _, _, verdict, findings, _ = parse_gpt2(raw, tier=tier)
            assert findings[0]["severity"] == "hard", f"T1 should be hard in {tier}"
            assert verdict == "FAIL"

    def test_t5_skipped_in_light(self):
        """T5 findings are skipped entirely in light tier."""
        raw = json.dumps({
            "claim_table": [],
            "findings": [{"type": "T5", "severity": "soft", "detail": "Prescriptive."}],
            "verdict": "FAIL",
        })
        _, _, verdict, findings, _ = parse_gpt2(raw, tier="light")
        assert findings == []
        assert verdict == "PASS"

    def test_t6_skipped_in_light(self):
        """T6 (Reassurance framing) is skipped in light tier."""
        raw = json.dumps({
            "claim_table": [],
            "findings": [{"type": "Reassurance framing", "severity": "soft", "detail": "Praise."}],
            "verdict": "FAIL",
        })
        _, _, verdict, findings, _ = parse_gpt2(raw, tier="light")
        assert findings == []
        assert verdict == "PASS"

    def test_strict_threshold_3(self):
        """Strict tier: 3 soft findings = FAIL."""
        raw = json.dumps({
            "claim_table": [],
            "findings": [
                {"type": "Overconfidence", "severity": "soft", "detail": "A"},
                {"type": "Missing jurisdiction", "severity": "soft", "detail": "B"},
                {"type": "Unacknowledged conflict", "severity": "soft", "detail": "C"},
            ],
            "verdict": "FAIL",
        })
        _, _, verdict, _, _ = parse_gpt2(raw, tier="strict")
        assert verdict == "FAIL"

    def test_standard_threshold_4(self):
        """Standard tier: 3 soft findings = PASS (threshold is 4)."""
        raw = json.dumps({
            "claim_table": [],
            "findings": [
                {"type": "Overconfidence", "severity": "soft", "detail": "A"},
                {"type": "Missing jurisdiction", "severity": "soft", "detail": "B"},
                {"type": "Unacknowledged conflict", "severity": "soft", "detail": "C"},
            ],
            "verdict": "FAIL",
        })
        _, _, verdict, _, _ = parse_gpt2(raw, tier="standard")
        assert verdict == "PASS"

    def test_light_threshold_5(self):
        """Light tier: 4 soft findings = PASS (threshold is 5)."""
        raw = json.dumps({
            "claim_table": [],
            "findings": [
                {"type": "Overconfidence", "severity": "soft", "detail": "A"},
                {"type": "Missing jurisdiction", "severity": "soft", "detail": "B"},
                {"type": "Unacknowledged conflict", "severity": "soft", "detail": "C"},
                {"type": "T4", "severity": "soft", "detail": "D"},
            ],
            "verdict": "FAIL",
        })
        _, _, verdict, _, _ = parse_gpt2(raw, tier="light")
        assert verdict == "PASS"
