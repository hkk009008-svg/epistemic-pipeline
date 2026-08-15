"""Adversarial stress-testing suite authored by teamwork_preview_challenger_2_2.

Empirically challenges and stress-tests:
1. Control 3: Clause-Isolated Generation Schema & Output Sanitizer
   - Malformed grammar, unbalanced delimiters, heavily nested clauses.
   - Sentences with only banned phrases, paragraphs of consecutive banned phrases.
   - Multiple consecutive orphaned coordinators, whitespace, tabs, and multiline coordinators.
   - AST ID-based claim mutation (apply_edits_by_id) under adversarial and corrupt AST structures.
   - Sanitizer tier-gating, flag variations, regex DoS resilience, markdown/table non-interference.

2. Control 4: Closed-Loop Negative Constraints & Multi-Turn Repair Loops
   - Adversarial re-hallucinations across turns.
   - Oscillating finding patterns (e.g., T1 -> T3 -> T1).
   - Strict <= 2 iteration convergence guarantee and deterministic fail-closed fallback.
   - Monotonic constraint accumulation ensuring Turn 0 constraints are never lost.
   - Negative constraint extractor robustness against adversarial input strings and injection attempts.

3. Full-Pipeline Convergence:
   - Async stage execution (stage_verify -> stage_arbiter -> stage_rewrite_loop).
   - High-concurrency async repair loop stress testing.
"""
from __future__ import annotations

import asyncio
import json
import time
import pytest

from pipeline.models import (
    ClaimEntry,
    EditEntry,
    SearchSource,
    PipelineResponse,
    GPT2ResponseSchema,
    FindingSchema,
)
from pipeline.arbiter import (
    apply_edits_by_id,
    extract_negative_constraints,
    format_negative_constraints_block,
)
from pipeline.sanitizer import (
    _clean_grammar_and_punctuation,
    sanitize_output,
)
from pipeline.source_match import (
    build_source_keyword_sets,
    build_source_number_sets,
)
from pipeline.convergence import (
    compute_finding_delta,
    should_continue_rewrite,
)
from pipeline.prompts import (
    DEFAULT_GPT1_SYSTEM,
    DEFAULT_GPT2_SYSTEM,
    DEFAULT_GPT3_SYSTEM,
)
from pipeline.stages import (
    stage_rewrite_loop,
)
from pipeline.metrics import PipelineMetrics


# ===========================================================================
# 1. CONTROL 3: CLAUSE-ISOLATED SCHEMA & SANITIZER ADVERSARIAL TESTS
# ===========================================================================

class TestControl3AdversarialSanitizerGrammar:
    """Stress-test the Sanitizer and Grammar post-processor against adversarial inputs."""

    def test_malformed_grammar_unbalanced_parentheses_brackets(self):
        """Verify sanitizer handles broken bracket nesting, unbalanced delimiters without crashing."""
        malformed_inputs = [
            "Research shows ((( Smith 2022 that water is wet.",
            "Studies suggest [[[[1]] that outcome was 50% .",
            "According to [CDC (2023] the rate is 45% .",
            "((([Unverified generalization removed] ... ))",
            "Plan A [1][2][(3] is approved with .",
        ]
        flags = {"advice_requested": False, "percent_requested": False, "legal_mode": False}
        for text in malformed_inputs:
            cleaned = sanitize_output(text, flags, tier="strict")
            assert isinstance(cleaned, str)
            # Ensure no crash and grammar cleaner normalized trailing dangling prepositions/punctuation
            assert not cleaned.endswith("with .")
            assert not cleaned.startswith(",,")

    def test_heavily_nested_clauses_and_subordinates(self):
        """Sanitizer properly cleans heavily nested multi-clause structures with embedded typicality and outcome promises."""
        text = (
            "Although studies suggest that generally patients recover, "
            "whereas while the clinical dosage will improve symptoms in 80% of subjects, "
            "and furthermore data indicates that this strategy could help to reduce mortality, "
            "the ultimate outcome was approved for, and agreed with."
        )
        flags = {"advice_requested": False, "percent_requested": False, "legal_mode": False}
        cleaned = sanitize_output(text, flags, tier="strict")

        # Banned evidence removed
        assert "studies suggest" not in cleaned.lower()
        assert "data indicates" not in cleaned.lower()
        # Typicality language removed
        assert "generally" not in cleaned.lower()
        # Outcome promises neutralized
        assert "will improve" not in cleaned.lower()
        assert "could help to" not in cleaned.lower()
        # Bare percent without citation replaced
        assert "Unknown(Actionable)" in cleaned
        # Dangling prepositions cleaned
        assert "approved, and agreed." in cleaned or "approved for, and agreed." not in cleaned

    def test_sentence_with_only_banned_phrases(self):
        """A sentence composed entirely of banned phrases should sanitize cleanly without syntax errors."""
        text = "Generally, typically, commonly, usually, often."
        flags = {"advice_requested": False, "percent_requested": False, "legal_mode": False}
        cleaned = sanitize_output(text, flags, tier="strict")
        # Every word is replaced or removed
        for word in ["generally", "typically", "commonly", "usually", "often"]:
            assert word not in cleaned.lower()
        # Should contain sanitizer markers
        assert "[Typicality language removed]" in cleaned

    def test_paragraph_with_multiple_consecutive_banned_phrase_sentences(self):
        """Paragraph with multiple consecutive sentences of banned evidence phrases."""
        text = (
            "Research shows.\n"
            "Studies suggest.\n"
            "Data indicates.\n"
            "Evidence suggests.\n"
        )
        flags = {"advice_requested": False, "percent_requested": False, "legal_mode": False}
        cleaned = sanitize_output(text, flags, tier="strict")
        assert "research shows" not in cleaned.lower()
        assert "studies suggest" not in cleaned.lower()
        assert "data indicates" not in cleaned.lower()
        assert "evidence suggests" not in cleaned.lower()
        # Ensure multi-line structure preserved
        for line in cleaned.split("\n"):
            if line:
                assert line.startswith("[Unverified generalization removed]")

    def test_two_consecutive_orphaned_coordinators_and_punctuation(self):
        """Leading punctuation and up to two consecutive coordinators are stripped cleanly."""
        test_cases = [
            (",, whereas Plan A was approved with .", "Plan A was approved."),
            (";;; And whereas the trial concluded with .", "The trial concluded."),
            ("... But the second test succeeded for .", "The second test succeeded."),
            (",,, ,,, However the metric was 50% (CDC 2024).", "However the metric was 50% (CDC 2024)."),
            ("Whereas findings remain valid.", "Findings remain valid."),
        ]
        for raw, expected in test_cases:
            cleaned = _clean_grammar_and_punctuation(raw)
            assert cleaned == expected

    def test_multiple_consecutive_coordinators_residual_behavior(self):
        """Empirical challenge: when >= 3 coordinators appear in series, the sanitizer strips 2 but leaves the 3rd."""
        raw = ",, whereas and but while or nor Plan A was approved with ."
        cleaned = _clean_grammar_and_punctuation(raw)
        # Sanitizer strips leading commas, 'whereas', and 'and', leaving 'But while or nor Plan A was approved.'
        assert "Plan A was approved." in cleaned
        assert cleaned == "Plan A was approved." or cleaned.startswith("But while or nor Plan A was approved.")

    def test_interleaved_coordinators_and_punctuation_runs(self):
        """Weirdly interleaved punctuation and connector runs."""
        raw = "Clause 1. , whereas , but Clause 2; ; and while Clause 3."
        cleaned = _clean_grammar_and_punctuation(raw)
        assert ",," not in cleaned
        assert ";;" not in cleaned
        assert "Clause 1." in cleaned
        assert "Clause 2;" in cleaned or "Clause 2." in cleaned
        assert "Clause 3." in cleaned

    def test_banned_evidence_with_nested_citations_and_fake_brackets(self):
        """Verify that genuine citations protect banned phrases, but empty or fake citations do not."""
        # Valid citation protects
        text_valid = "Research shows [1] that the compound is stable."
        flags = {"advice_requested": False, "percent_requested": False, "legal_mode": False}
        res_valid = sanitize_output(text_valid, flags, tier="strict")
        assert "Research shows [1]" in res_valid

        # Parenthetical citation protects
        text_paren = "Studies suggest (FDA 2023) that the dosage is safe."
        res_paren = sanitize_output(text_paren, flags, tier="strict")
        assert "Studies suggest (FDA 2023)" in res_paren

        # Fake empty brackets do NOT protect
        text_fake = "Research shows [] that the compound is stable."
        res_fake = sanitize_output(text_fake, flags, tier="strict")
        assert "research shows" not in res_fake.lower()
        assert "[Unverified generalization removed]" in res_fake

    def test_sanitizer_regex_redos_resilience(self):
        """Test regex performance against pathological repetition (ReDoS attack stress test)."""
        pathological = "generally " * 500 + "usually " * 500 + "about 50% " * 200 + " ,,, ... " * 200
        flags = {"advice_requested": False, "percent_requested": True, "legal_mode": False}

        t0 = time.perf_counter()
        cleaned = sanitize_output(pathological, flags, tier="strict")
        elapsed = (time.perf_counter() - t0) * 1000.0

        assert elapsed < 100.0, f"Sanitization took {elapsed:.2f}ms on 1200-token pathological string, risk of ReDoS!"
        assert isinstance(cleaned, str)
        assert "generally" not in cleaned.lower()
        assert "usually" not in cleaned.lower()

    def test_sanitizer_with_code_blocks_markdown_and_tables(self):
        """Sanitizer preserves markdown structures (code blocks, headers, table pipes)."""
        markdown_text = (
            "### Analysis\n"
            "```python\n"
            "# Code comments should not be broken\n"
            "rate = 0.50 # 50%\n"
            "```\n"
            "| Column 1 | Column 2 |\n"
            "| --- | --- |\n"
            "| Value A | Value B |\n"
        )
        flags = {"advice_requested": False, "percent_requested": False, "legal_mode": False}
        cleaned = sanitize_output(markdown_text, flags, tier="strict")
        assert "### Analysis" in cleaned
        assert "```python" in cleaned
        assert "| Column 1 | Column 2 |" in cleaned
        assert "| Value A | Value B |" in cleaned


# ===========================================================================
# 2. CONTROL 3: AST ID-BASED EDITING ADVERSARIAL TESTS
# ===========================================================================

class TestControl3ClauseIsolatedASTEditing:
    """Stress-test AST ID-based deterministic editing (apply_edits_by_id)."""

    def test_apply_edits_by_id_missing_and_orphan_ids(self):
        """Applying edits with non-existent or orphan IDs does not mutate unrelated claims or crash."""
        claims = [
            {"claim_id": "c-101", "text": "Claim 101 is supported."},
            {"claim_id": "c-102", "text": "Claim 102 is supported."},
        ]
        edits = [
            EditEntry(action="DELETE", target="", replacement="", target_id="c-999"),
            EditEntry(action="REWRITE", target="", replacement="New text", target_id=""),
            EditEntry(action="MOVE_TO_UNKNOWN", target="", replacement="", target_id="c-888"),
        ]
        modified, summary = apply_edits_by_id(claims, edits)
        assert len(modified) == 2
        assert modified[0]["claim_id"] == "c-101"
        assert modified[1]["claim_id"] == "c-102"
        assert summary == "No ID-based edits applied"

    def test_apply_edits_by_id_duplicate_ids(self):
        """When multiple claims accidentally share the same ID, the mutation handles first occurrence cleanly."""
        claims = [
            {"claim_id": "c-dup", "text": "First instance."},
            {"claim_id": "c-dup", "text": "Second instance."},
            {"claim_id": "c-uniq", "text": "Third instance."},
        ]
        edits = [
            EditEntry(action="DELETE", target="", replacement="", target_id="c-dup"),
        ]
        modified, summary = apply_edits_by_id(claims, edits)
        assert len(modified) == 2
        assert "DELETED claim c-dup" in summary
        assert modified[0]["text"] == "Second instance."
        assert modified[1]["text"] == "Third instance."

    def test_apply_edits_by_id_all_actions_delete_rewrite_move_to_unknown(self):
        """Test DELETE, REWRITE, and MOVE_TO_UNKNOWN in a single batch on complex AST."""
        claims = [
            {"claim_id": "id-del", "text": "Toxic hallucinated figure $999M."},
            {"claim_id": "id-rew", "text": "Plan A will improve revenue."},
            {"claim_id": "id-unk", "text": "Statute XYZ is 100% applicable."},
            {"claim_id": "id-keep", "text": "Clean ground truth [1]."},
        ]
        edits = [
            EditEntry(action="DELETE", target="", replacement="", target_id="id-del"),
            EditEntry(action="REWRITE", target="", replacement="Plan A addresses revenue.", target_id="id-rew"),
            EditEntry(action="MOVE_TO_UNKNOWN", target="", replacement="Unknown(Actionable): Statute XYZ applicability is unverified.", target_id="id-unk"),
        ]
        modified, summary = apply_edits_by_id(claims, edits)
        assert len(modified) == 3
        # Verify id-del deleted
        assert not any(c.get("claim_id") == "id-del" for c in modified)
        # Verify id-rew rewritten
        rew_claim = next(c for c in modified if c.get("claim_id") == "id-rew")
        assert rew_claim["text"] == "Plan A addresses revenue."
        # Verify id-unk moved to unknown
        unk_claim = next(c for c in modified if c.get("claim_id") == "id-unk")
        assert unk_claim["is_unknown"] is True
        assert "Unknown(Actionable)" in unk_claim["text"]
        # Verify id-keep untouched
        keep_claim = next(c for c in modified if c.get("claim_id") == "id-keep")
        assert keep_claim["text"] == "Clean ground truth [1]."

    def test_apply_edits_by_id_preserves_unmodified_metadata(self):
        """Ensure all additional claim metadata fields (citations, confidence, line_number) are preserved during edits."""
        claims = [
            {
                "claim_id": "meta-1",
                "text": "Initial claim text.",
                "citations": [1, 2],
                "category": "Inference",
                "line_no": 42,
            }
        ]
        edits = [
            EditEntry(action="REWRITE", target="", replacement="Updated claim text.", target_id="meta-1"),
        ]
        modified, _ = apply_edits_by_id(claims, edits)
        assert len(modified) == 1
        assert modified[0]["text"] == "Updated claim text."
        assert modified[0]["citations"] == [1, 2]
        assert modified[0]["category"] == "Inference"
        assert modified[0]["line_no"] == 42


# ===========================================================================
# 3. CONTROL 4: NEGATIVE CONSTRAINTS EXTRACTION & FORMATTING ADVERSARIAL
# ===========================================================================

class TestControl4NegativeConstraintsExtractionAdversarial:
    """Stress-test negative constraints extraction against adversarial inputs."""

    def test_extract_negative_constraints_adversarial_details_injection(self):
        """Findings containing regex special characters, markdown formatting, or quote escaping."""
        findings = [
            {"type": "T1", "severity": "hard", "detail": "Fabricated citation: referenced non-existent source [999] (available sources: 1..2)."},
            {"type": "T1", "severity": "hard", "detail": "Unbacked numeric claim: [1] does not contain numeric value $1,234.56 from '$1,234.56 total'."},
            {"type": "T1", "severity": "hard", "detail": "Regex attack string: (.*?) [a-z]+ \\d+ $^ | [] {}"},
            {"type": "T3", "severity": "hard", "detail": 'Causal claim: "Smoking => Cancer" asserted without citation'},
        ]
        constraints = extract_negative_constraints(findings, max_source_count=2)
        assert len(constraints) == 4
        # Source constraint extracted
        assert "DO NOT cite non-existent source [999] (valid source indices are 1..2)." in constraints
        # Formatted currency extracted cleanly
        assert "DO NOT introduce the unbacked numeric figure $1,234.56." in constraints
        # Causal constraint extracted
        assert any("DO NOT assert causal relationships" in c for c in constraints)
        # Regex attack string safely wrapped
        assert any("Regex attack string" in c for c in constraints)

    def test_extract_negative_constraints_all_tripwires_t1_to_t7(self):
        """Comprehensive mapping of all Tripwires T1 through T7 to distinct imperative DO NOT directives."""
        findings = [
            {"type": "T1", "severity": "hard", "detail": "Fabricated law Section 99"},
            {"type": "T2", "severity": "hard", "detail": "Generally used without basis"},
            {"type": "T3", "severity": "hard", "detail": "X leads to Y as fact"},
            {"type": "T4", "severity": "soft", "detail": "Ranking A above B without discriminators"},
            {"type": "T5", "severity": "soft", "detail": "Guarantees 100% cure rate"},
            {"type": "T6", "severity": "soft", "detail": "Reassurance phrase 'do not worry'"},
            {"type": "T7", "severity": "hard", "detail": "Price in 2030 presented as current"},
        ]
        constraints = extract_negative_constraints(findings)
        assert len(constraints) == 7
        assert any("DO NOT introduce fabricated" in c for c in constraints)
        assert any("DO NOT use typicality words" in c for c in constraints)
        assert any("DO NOT assert causal relationships" in c for c in constraints)
        assert any("DO NOT rank, rate, or compare" in c for c in constraints)
        assert any("DO NOT include outcome promises" in c for c in constraints)
        assert any("DO NOT use reassurance framing" in c for c in constraints)
        assert any("DO NOT present time-sensitive facts" in c for c in constraints)

    def test_extract_negative_constraints_deduplication_and_ordering(self):
        """Duplicate findings across findings, edits, and claim tables are deduplicated while preserving order."""
        findings = [
            {"type": "T2", "severity": "hard", "detail": "Typicality language"},
            {"type": "T2", "severity": "hard", "detail": "Typicality language"},
        ]
        edits = [
            EditEntry(action="DELETE", target="Bad Claim X", replacement=""),
            EditEntry(action="DELETE", target="Bad Claim X", replacement=""),
        ]
        claim_table = [
            {"category": "Unsupported", "claim": "Bad Claim Y"},
            {"category": "Unsupported", "claim": "Bad Claim Y"},
        ]
        constraints = extract_negative_constraints(findings=findings, arbiter_edits=edits, claim_table=claim_table)
        # Should have exactly 3 unique constraints
        assert len(constraints) == 3
        assert 'DO NOT include the claim or text: "Bad Claim X"' in constraints
        assert any("DO NOT use typicality words" in c for c in constraints)
        assert 'DO NOT make the unbacked assertion: "Bad Claim Y"' in constraints

    def test_format_negative_constraints_block_special_characters_escaping(self):
        """Formatting markdown block with quotes, backticks, and newlines."""
        constraints = [
            'DO NOT include: "Quote with \'single\' quotes and `backticks`"',
            "DO NOT cite non-existent source [4].",
        ]
        block = format_negative_constraints_block(constraints)
        assert block.startswith("### Negative Constraints\n")
        assert '- DO NOT include: "Quote with \'single\' quotes and `backticks`"' in block
        assert "- DO NOT cite non-existent source [4]." in block

    def test_format_negative_constraints_empty_list_returns_empty_string(self):
        """Empty input list returns an empty string without empty markdown headers."""
        assert format_negative_constraints_block([]) == ""
        assert format_negative_constraints_block(None) == ""


# ===========================================================================
# 4. CONTROL 4: MULTI-TURN REPAIR CONVERGENCE & OSCILLATION HARNESS
# ===========================================================================

class TestControl4MultiTurnRepairConvergenceAdversarial:
    """Stress-test repair loop convergence, monotonic ledger, and oscillation handling."""

    def test_monotonic_constraint_accumulation_across_three_turns(self):
        """Turn 0, Turn 1, and Turn 2 findings accumulate monotonically with strict zero constraint loss."""
        # Turn 0: out-of-bounds citation [5]
        t0_findings = [{"type": "T1", "severity": "hard", "detail": "referenced non-existent source [5]"}]
        t0_constraints = extract_negative_constraints(t0_findings)
        assert len(t0_constraints) == 1

        # Turn 1: fixes [5], but introduces unbacked figure $500
        t1_findings = [{"type": "T1", "severity": "hard", "detail": "does not contain numeric value 500"}]
        t1_new = extract_negative_constraints(t1_findings)

        ledger_t1 = list(t0_constraints)
        for c in t1_new:
            if c not in ledger_t1:
                ledger_t1.append(c)

        assert len(ledger_t1) == 2
        assert any("[5]" in c for c in ledger_t1), "Turn 0 constraint for [5] was lost in Turn 1!"
        assert any("500" in c for c in ledger_t1)

        # Turn 2: fixes $500, but introduces typicality word
        t2_findings = [{"type": "T2", "severity": "hard", "detail": "used typically without citation"}]
        t2_new = extract_negative_constraints(t2_findings)

        ledger_t2 = list(ledger_t1)
        for c in t2_new:
            if c not in ledger_t2:
                ledger_t2.append(c)

        assert len(ledger_t2) == 3
        assert any("[5]" in c for c in ledger_t2), "Turn 0 constraint for [5] was lost in Turn 2!"
        assert any("500" in c for c in ledger_t2), "Turn 1 constraint for 500 was lost in Turn 2!"
        assert any("typicality" in c.lower() for c in ledger_t2)

    def test_oscillating_finding_patterns_detection_and_early_termination(self):
        """When findings oscillate between turns (T1 -> T3 -> T1), convergence detection halts loop."""
        turn0_findings = [{"type": "T1", "severity": "hard", "detail": "Fabricated figure"}]
        turn1_findings = [{"type": "T3", "severity": "hard", "detail": "Unverified causal assertion"}]
        turn2_findings = [{"type": "T1", "severity": "hard", "detail": "Fabricated figure"}]

        # Delta between Turn 0 and Turn 1
        delta_0_1 = compute_finding_delta(turn0_findings, turn1_findings)
        assert delta_0_1["oscillating"] is True
        assert delta_0_1["improved"] is False

        # should_continue_rewrite on oscillating history should be False
        history = [turn0_findings, turn1_findings]
        assert should_continue_rewrite(history, max_loops=3) is False

        # Delta between Turn 1 and Turn 2
        delta_1_2 = compute_finding_delta(turn1_findings, turn2_findings)
        assert delta_1_2["oscillating"] is True
        assert delta_1_2["improved"] is False

    def test_repair_loop_hard_cap_strictly_two_iterations(self):
        """Repair loop must never exceed 2 iterations under persistent adversarial failure."""
        findings_t0 = [{"type": "T1", "severity": "hard", "detail": "Stubborn violation"}]
        findings_t1 = [{"type": "T1", "severity": "hard", "detail": "Stubborn violation"}]
        findings_t2 = [{"type": "T1", "severity": "hard", "detail": "Stubborn violation"}]

        # History with initial + 2 rewrites (len == 3)
        history = [findings_t0, findings_t1, findings_t2]
        # With max_loops=2, history length >= 3 must halt immediately
        assert should_continue_rewrite(history, max_loops=2) is False

    def test_converged_identical_findings_halts_rewrite(self):
        """Identical findings across iterations indicates zero progress (converged on failure) -> halts rewrite."""
        f1 = [{"type": "T1", "severity": "hard", "detail": "Exact same violation"}]
        f2 = [{"type": "T1", "severity": "hard", "detail": "Exact same violation"}]

        delta = compute_finding_delta(f1, f2)
        assert delta["converged"] is True
        assert delta["improved"] is False
        assert should_continue_rewrite([f1, f2], max_loops=3) is False

    def test_strictly_improving_findings_permits_next_iteration(self):
        """Strictly decreasing hard findings is recognized as improvement."""
        f0 = [
            {"type": "T1", "severity": "hard", "detail": "Violation 1"},
            {"type": "T1", "severity": "hard", "detail": "Violation 2"},
        ]
        f1 = [
            {"type": "T1", "severity": "hard", "detail": "Violation 1"},
        ]
        delta = compute_finding_delta(f0, f1)
        assert delta["improved"] is True
        assert delta["hard_delta"] == -1
        assert delta["oscillating"] is False
        assert should_continue_rewrite([f0, f1], max_loops=2) is True


# ===========================================================================
# 5. FULL PIPELINE CONVERGENCE & STRESS HARNESS
# ===========================================================================

class TestFullPipelineConvergenceAdversarialHarness:
    """Simulate end-to-end async repair loop with adversarial LLM agents."""

    @pytest.mark.asyncio
    async def test_pipeline_stage_rewrite_loop_adversarial_rehallucination_to_fallback(self, monkeypatch):
        """Simulate an adversarial LLM that stubbornly re-hallucinates across 2 turns.

        Verifies:
        1. Turn 1 receives Turn 0 negative constraints.
        2. Turn 2 receives cumulative Turn 0 + Turn 1 negative constraints.
        3. After 2 failed turns, loop deterministically falls back to Unknown framing.
        4. Final response verdict is PASS with confidence Low.
        """
        prompts_received = []

        async def mock_call_llm_async(cfg, sys, usr, expect_json=False):
            if expect_json:
                return json.dumps({
                    "reasoning_trace": ["Evaluation"],
                    "claim_table": [{"claim": "Revenue claim", "category": "Unsupported", "justification": "Invalid"}],
                    "findings": [{"type": "T1", "severity": "hard", "detail": "Re-hallucinated violation"}],
                    "verdict": "FAIL",
                })
            prompts_received.append(usr)
            if len(prompts_received) == 1:
                # Turn 1: re-hallucinates $999M
                return "The revenue is $999M [1]."
            elif len(prompts_received) == 2:
                # Turn 2: re-hallucinates out-of-bounds citation [99]
                return "The revenue is verified [99]."
            else:
                # Fallback rewrite: frame as unknown
                return "Unknown(Actionable): Financial figures are unverified."

        monkeypatch.setattr("pipeline.stages.call_llm_async", mock_call_llm_async)

        async def mock_call_llm_structured(cfg, sys, usr, schema):
            return GPT2ResponseSchema(
                reasoning_trace=["Evaluation"],
                claim_table=[{"claim": "Revenue claim", "category": "Unsupported", "justification": "Invalid"}],
                findings=[FindingSchema(type="T1", severity="hard", detail="Re-hallucinated violation")],
                verdict="FAIL",
            )

        monkeypatch.setattr("pipeline.stages.call_llm_structured", mock_call_llm_structured)

        sources = [SearchSource(title="S1", url="http://s1.com", snippet="Revenue was $50M.")]
        src_kw = build_source_keyword_sets(sources)
        src_num = build_source_number_sets(sources)

        state = {
            "prompt": "What is the company revenue?",
            "gpt1_cfg": {"provider": "mock", "model": "mock"},
            "gpt2_cfg": {"provider": "mock", "model": "mock"},
            "gpt3_cfg": {"provider": "mock", "model": "mock"},
            "gpt1_system": DEFAULT_GPT1_SYSTEM,
            "gpt2_system": DEFAULT_GPT2_SYSTEM,
            "gpt3_system": DEFAULT_GPT3_SYSTEM,
            "sanitized_output": "The revenue is $500M [1].",
            "flags": {"percent_requested": False, "advice_requested": False, "legal_mode": False},
            "tier": "strict",
            "metrics": PipelineMetrics(request_id="adv-test-1", prompt_length=30),
            "search_sources": sources,
            "src_kw_sets": src_kw,
            "src_num_sets": src_num,
            "search_performed": True,
            "arbiter_decision": "ALLOW_WITH_EDITS",
            "arbiter_edits": [EditEntry(action="DELETE", target="The revenue is $500M [1].", replacement="")],
            "findings": [{"type": "T1", "severity": "hard", "detail": "does not contain numeric value 500"}],
            "claim_table": [ClaimEntry(claim="The revenue is $500M", category="Unsupported", justification="Unbacked figure")],
            "max_rewrite_loops": 2,
        }
        state["metrics"].start()

        result = await stage_rewrite_loop(state)
        early_return: PipelineResponse = result.get("early_return")

        assert early_return is not None
        assert early_return.final_verdict == "PASS"
        assert early_return.confidence.confidence_label == "Low"
        assert early_return.rewrite_occurred is True
        assert state["metrics"].rewrite_loops == 2
        assert state["metrics"].convergence_outcome == "fallback"

        # Verify negative constraints were injected in prompts
        assert len(prompts_received) == 3  # Turn 1, Turn 2, Fallback
        turn1_prompt = prompts_received[0]
        turn2_prompt = prompts_received[1]
        fallback_prompt = prompts_received[2]

        assert "### Negative Constraints" in turn1_prompt
        assert "500" in turn1_prompt
        assert "### Negative Constraints" in turn2_prompt
        assert "500" in turn2_prompt, "Turn 0 constraint (500) must be preserved in Turn 2!"
        assert "Unknown(Actionable)" in fallback_prompt

    @pytest.mark.asyncio
    async def test_pipeline_stage_rewrite_loop_successful_turn1_repair(self, monkeypatch):
        """Simulate an LLM that adheres to negative constraints on Turn 1, converging immediately."""
        prompts_received = []

        async def mock_call_llm_async(cfg, sys, usr, expect_json=False):
            prompts_received.append(usr)
            return "Revenue reached $50M in Q3 [1]."

        monkeypatch.setattr("pipeline.stages.call_llm_async", mock_call_llm_async)

        async def mock_call_llm_structured(cfg, sys, usr, schema):
            return GPT2ResponseSchema(
                reasoning_trace=["Verified against source"],
                claim_table=[{"claim": "Revenue reached $50M in Q3 [1]", "category": "Observed", "justification": "Matches S1"}],
                findings=[],
                verdict="PASS",
            )

        monkeypatch.setattr("pipeline.stages.call_llm_structured", mock_call_llm_structured)

        sources = [SearchSource(title="S1", url="http://s1.com", snippet="Revenue reached $50M in Q3.")]
        src_kw = build_source_keyword_sets(sources)
        src_num = build_source_number_sets(sources)

        state = {
            "prompt": "What is the revenue?",
            "gpt1_cfg": {"provider": "mock", "model": "mock"},
            "gpt2_cfg": {"provider": "mock", "model": "mock"},
            "gpt1_system": DEFAULT_GPT1_SYSTEM,
            "gpt2_system": DEFAULT_GPT2_SYSTEM,
            "sanitized_output": "Revenue reached $500M [1].",
            "flags": {},
            "tier": "strict",
            "metrics": PipelineMetrics(request_id="adv-test-2", prompt_length=20),
            "search_sources": sources,
            "src_kw_sets": src_kw,
            "src_num_sets": src_num,
            "search_performed": True,
            "arbiter_decision": "ALLOW_WITH_EDITS",
            "arbiter_edits": [],
            "findings": [{"type": "T1", "severity": "hard", "detail": "does not contain numeric value 500"}],
            "claim_table": [ClaimEntry(claim="Revenue reached $500M", category="Unsupported", justification="Unbacked figure")],
            "max_rewrite_loops": 2,
        }
        state["metrics"].start()

        result = await stage_rewrite_loop(state)
        early_return: PipelineResponse = result.get("early_return")

        assert early_return is not None
        assert early_return.final_verdict == "PASS"
        assert state["metrics"].rewrite_loops == 1
        assert state["metrics"].convergence_outcome == "pass"
        assert len(prompts_received) == 1  # Strictly 1 iteration needed

    @pytest.mark.asyncio
    async def test_pipeline_stage_rewrite_loop_block_mode_regeneration(self, monkeypatch):
        """Simulate BLOCK decision routing to complete fresh regeneration in Turn 1."""
        prompts_received = []

        async def mock_call_llm_async(cfg, sys, usr, expect_json=False):
            prompts_received.append(usr)
            return "Completely fresh clean answer with $50M [1]."

        monkeypatch.setattr("pipeline.stages.call_llm_async", mock_call_llm_async)

        async def mock_call_llm_structured(cfg, sys, usr, schema):
            return GPT2ResponseSchema(
                reasoning_trace=["Verified regenerated response"],
                claim_table=[{"claim": "Clean answer with $50M", "category": "Observed", "justification": "Matches S1"}],
                findings=[],
                verdict="PASS",
            )

        monkeypatch.setattr("pipeline.stages.call_llm_structured", mock_call_llm_structured)

        sources = [SearchSource(title="S1", url="http://s1.com", snippet="Completely fresh clean answer with $50M.")]
        src_kw = build_source_keyword_sets(sources)
        src_num = build_source_number_sets(sources)

        state = {
            "prompt": "Explain the situation.",
            "gpt1_cfg": {"provider": "mock", "model": "mock"},
            "gpt2_cfg": {"provider": "mock", "model": "mock"},
            "gpt1_system": DEFAULT_GPT1_SYSTEM,
            "gpt2_system": DEFAULT_GPT2_SYSTEM,
            "sanitized_output": "Heavily poisoned text with invented entities.",
            "flags": {},
            "tier": "strict",
            "metrics": PipelineMetrics(request_id="adv-test-3", prompt_length=20),
            "search_sources": sources,
            "src_kw_sets": src_kw,
            "src_num_sets": src_num,
            "search_performed": True,
            "arbiter_decision": "BLOCK",
            "arbiter_edits": [],
            "findings": [
                {"type": "T1", "severity": "hard", "detail": "Fabricated entity A"},
                {"type": "T1", "severity": "hard", "detail": "Fabricated entity B"},
            ],
            "claim_table": [
                ClaimEntry(claim="Entity A", category="Unsupported", justification="Fabricated"),
                ClaimEntry(claim="Entity B", category="Unsupported", justification="Fabricated"),
            ],
            "max_rewrite_loops": 2,
        }
        state["metrics"].start()

        result = await stage_rewrite_loop(state)
        early_return: PipelineResponse = result.get("early_return")

        assert early_return is not None
        assert early_return.final_verdict == "PASS"
        assert state["metrics"].rewrite_loops == 1
        assert state["metrics"].convergence_outcome == "pass"

        # Verify prompt instructed a completely fresh regeneration
        assert "Your previous response was rejected due to heavy poisoning" in prompts_received[0]
        assert "generate a completely fresh response" in prompts_received[0]
        assert "### Negative Constraints" in prompts_received[0]

    @pytest.mark.asyncio
    async def test_concurrent_multi_turn_repair_stress(self, monkeypatch):
        """Run 100 concurrent multi-turn repair executions under load to verify thread-safety and latency."""
        async def mock_call_llm_async(cfg, sys, usr, expect_json=False):
            await asyncio.sleep(0.001)  # small async tick
            return "Clean concurrent text with $100 [1]."

        monkeypatch.setattr("pipeline.stages.call_llm_async", mock_call_llm_async)

        async def mock_call_llm_structured(cfg, sys, usr, schema):
            return GPT2ResponseSchema(
                reasoning_trace=["Verified"],
                claim_table=[{"claim": "Clean concurrent text", "category": "Observed", "justification": "Valid"}],
                findings=[],
                verdict="PASS",
            )

        monkeypatch.setattr("pipeline.stages.call_llm_structured", mock_call_llm_structured)

        sources = [SearchSource(title="S1", url="http://s1.com", snippet="Clean concurrent text with $100.")]
        src_kw = build_source_keyword_sets(sources)
        src_num = build_source_number_sets(sources)

        async def run_single_task(idx: int):
            m = PipelineMetrics(request_id=f"conc-{idx}", prompt_length=15)
            m.start()
            state = {
                "prompt": f"Query {idx}",
                "gpt1_cfg": {"provider": "mock", "model": "mock"},
                "gpt2_cfg": {"provider": "mock", "model": "mock"},
                "gpt1_system": DEFAULT_GPT1_SYSTEM,
                "gpt2_system": DEFAULT_GPT2_SYSTEM,
                "sanitized_output": "Corrupted text [1].",
                "flags": {},
                "tier": "strict",
                "metrics": m,
                "search_sources": sources,
                "src_kw_sets": src_kw,
                "src_num_sets": src_num,
                "search_performed": True,
                "arbiter_decision": "ALLOW_WITH_EDITS",
                "arbiter_edits": [],
                "findings": [{"type": "T1", "severity": "hard", "detail": "Unbacked figure"}],
                "claim_table": [ClaimEntry(claim="Unbacked claim", category="Unsupported", justification="Unbacked")],
                "max_rewrite_loops": 2,
            }
            res = await stage_rewrite_loop(state)
            return res.get("early_return")

        t0 = time.perf_counter()
        tasks = [run_single_task(i) for i in range(100)]
        results = await asyncio.gather(*tasks)
        elapsed = (time.perf_counter() - t0)

        assert len(results) == 100
        assert all(r.final_verdict == "PASS" for r in results)
        assert elapsed < 5.0, f"100 concurrent repair runs took {elapsed:.2f}s, expected < 5s"
