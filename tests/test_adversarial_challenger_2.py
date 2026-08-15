"""Adversarial stress-testing suite authored by teamwork_preview_challenger_2_1.

Empirically tests and challenges:
1. Control 1: Adaptive Poisoning Threshold in Arbiter (exact 35% floating-point boundaries, 0 claims, massive claim tables, mixed category representations, conflicting findings, decision override edge cases).
2. Control 2: Deterministic Pre-Flight Scanner (nested citation brackets, extreme index ranges [0], [-1], [99999], formatted currencies, multipliers, zero retrieved sources, complex segmentation).
3. Latency Benchmark: Pre-flight scanner latency across 1,000 iterations to verify strict <10ms requirement.
4. Control 3 & 4: Clause isolation, negative constraint extraction robustness, monotonicity, deduplication.
"""
import time
import pytest
from typing import List, Dict, Any

from pipeline.models import ClaimEntry, EditEntry, SearchSource, GPT2ResponseSchema, GPT3ResponseSchema
from pipeline.arbiter import (
    check_poisoning_threshold,
    guard_arbiter_decision,
    extract_negative_constraints,
    format_negative_constraints_block,
    apply_edits,
    apply_edits_by_id,
)
from pipeline.source_match import (
    _extract_numbers,
    _extract_keywords,
    _get_citation_segments,
    verify_citation_grounding,
    run_preflight_scan,
    build_source_keyword_sets,
    build_source_number_sets,
    recategorize_with_sources,
    filter_findings_with_sources,
)
from pipeline.sanitizer import (
    _clean_grammar_and_punctuation,
    sanitize_output,
    _replace_bare_percents,
    _replace_outcome_promises,
)


# ===========================================================================
# 1. CONTROL 1 ADVERSARIAL CHALLENGES: ADAPTIVE POISONING THRESHOLD
# ===========================================================================

class TestAdversarialControl1AdaptivePoisoning:
    """Stress-test the Arbiter Adaptive Poisoning Guard."""

    def test_exact_float_boundary_35_0_percent_vs_35_001_percent(self):
        """Verify strict floating point behavior at the exact 35.0% threshold."""
        # 35.0% unsupported: exactly 35 / 100
        claims_35_0 = [
            {"category": "Unsupported", "claim": f"Bad claim {i}"} for i in range(35)
        ] + [
            {"category": "Supported", "claim": f"Good claim {i}"} for i in range(65)
        ]
        res_35_0 = check_poisoning_threshold(claims_35_0, findings=[], unsupported_threshold=0.35)
        assert res_35_0["unsupported_ratio"] == 0.35
        assert res_35_0["is_poisoned"] is False, "Exact 35.0% should NOT be poisoned (> 0.35 rule)"

        dec, notes = guard_arbiter_decision("ALLOW_WITH_EDITS", claims_35_0, findings=[])
        assert dec == "ALLOW_WITH_EDITS"

        # 35.001% unsupported: 35001 / 100000
        claims_35_001 = [
            {"category": "Unsupported", "claim": f"Bad {i}"} for i in range(35001)
        ] + [
            {"category": "Supported", "claim": f"Good {i}"} for i in range(64999)
        ]
        res_35_001 = check_poisoning_threshold(claims_35_001, findings=[], unsupported_threshold=0.35)
        assert res_35_001["unsupported_ratio"] > 0.35
        assert res_35_001["is_poisoned"] is True

        dec_p, notes_p = guard_arbiter_decision("ALLOW_WITH_EDITS", claims_35_001, findings=[])
        assert dec_p == "BLOCK"
        assert any("heavily poisoned" in n.lower() for n in notes_p)

    def test_float_epsilon_boundary_arithmetic(self):
        """Test floating-point representations that might suffer from IEEE 754 precision issues."""
        # 7 / 20 is exactly 0.35
        claims_7_20 = [
            {"category": "Unsupported", "claim": f"Bad {i}"} for i in range(7)
        ] + [
            {"category": "Supported", "claim": f"Good {i}"} for i in range(13)
        ]
        res = check_poisoning_threshold(claims_7_20, findings=[])
        assert res["unsupported_ratio"] == 0.35
        assert res["is_poisoned"] is False

        # 351 / 1000 is 0.351
        claims_351 = [
            {"category": "Unsupported", "claim": f"Bad {i}"} for i in range(351)
        ] + [
            {"category": "Supported", "claim": f"Good {i}"} for i in range(649)
        ]
        res_351 = check_poisoning_threshold(claims_351, findings=[])
        assert res_351["is_poisoned"] is True

    def test_massive_claim_table_performance_and_accuracy(self):
        """Stress-test with 50,000 mixed claims to check speed and stability."""
        claims = []
        for i in range(50000):
            if i % 4 == 0:
                claims.append(ClaimEntry(claim=f"Claim {i}", category="Unsupported", justification="test"))
            elif i % 4 == 1:
                claims.append(ClaimEntry(claim=f"Claim {i}", category="Observed", justification="test"))
            elif i % 4 == 2:
                claims.append(ClaimEntry(claim=f"Claim {i}", category="Supported", justification="test"))
            else:
                claims.append(ClaimEntry(claim=f"Claim {i}", category="Inference", justification="test"))

        t0 = time.perf_counter()
        res = check_poisoning_threshold(claims, findings=[])
        elapsed = time.perf_counter() - t0

        assert elapsed < 0.2, f"50k claims took {elapsed:.4f}s, expected <0.2s"
        assert res["total_claims"] == 50000
        assert res["unsupported_count"] == 12500
        assert abs(res["unsupported_ratio"] - 0.25) < 1e-6
        assert res["is_poisoned"] is False

    def test_mixed_category_representations_and_dirty_inputs(self):
        """Test heterogeneous, malformed, and dirty category entries."""
        dirty_claims = [
            {"category": "UNSUPPORTED ", "claim": "C1"},
            {"category": "unsupported_inferential", "claim": "C2"},
            {"category": " Contradicted\t", "claim": "C3"},
            {"category": "REFUTED\n", "claim": "C4"},
            {"category": "FABRICATED", "claim": "C5"},
            ClaimEntry(claim="C6", category="observed", justification="test"),
            ClaimEntry(claim="C7", category="SUPPORTED ", justification="test"),
            ClaimEntry(claim="C8", category="user-provided", justification="test"),
            ClaimEntry(claim="C9", category="inference", justification="test"),
            {"category": None, "claim": "C10"},
            {"no_category_key": "foo"},
            "raw string instead of dict or object",
            None,
        ]
        # Total claims = 13 entries
        # Unsupported: C1, C2, C3, C4, C5 -> 5
        res = check_poisoning_threshold(dirty_claims, findings=[])
        assert res["total_claims"] == 13
        assert res["unsupported_count"] == 5
        assert abs(res["unsupported_ratio"] - (5 / 13)) < 1e-5
        # 5/13 = 38.46% > 35% -> poisoned
        assert res["is_poisoned"] is True

    def test_zero_claims_and_empty_inputs(self):
        """Test boundary of zero claims, None lists, empty findings."""
        res_empty = check_poisoning_threshold([], [])
        assert res_empty["is_poisoned"] is False
        assert res_empty["unsupported_ratio"] == 0.0
        assert res_empty["hard_count"] == 0
        assert res_empty["total_claims"] == 0

        res_none = check_poisoning_threshold(None, None)
        assert res_none["is_poisoned"] is False
        assert res_none["unsupported_ratio"] == 0.0
        assert res_none["hard_count"] == 0
        assert res_none["total_claims"] == 0

        # Guard decision on empty claims
        dec, notes = guard_arbiter_decision("ALLOW_WITH_EDITS", [], [])
        assert dec == "ALLOW_WITH_EDITS"

        dec_block, notes_block = guard_arbiter_decision("BLOCK", [], [])
        # No salvageable truthful claims -> BLOCK maintained
        assert dec_block == "BLOCK"

    def test_conflicting_findings_and_severity_variations(self):
        """Test findings with dirty severity values, mixed soft/hard, and missing keys."""
        dirty_findings = [
            {"type": "T1", "severity": "HARD", "detail": "Hard 1"},
            {"type": "T1", "severity": " hard \n", "detail": "Hard 2"},
            {"type": "T2", "severity": "SOFT", "detail": "Soft 1"},
            {"type": "T3", "severity": "soft", "detail": "Soft 2"},
            {"type": "T4", "severity": None, "detail": "Missing severity"},
            {"type": "T5", "detail": "No severity key"},
            "corrupt string",
            None,
        ]
        res = check_poisoning_threshold([], dirty_findings)
        assert res["hard_count"] == 2
        assert res["is_poisoned"] is True  # 2 hard violations >= 2 triggers poisoning

    def test_allow_as_unknown_only_decision_guarding(self):
        """Test ALLOW_AS_UNKNOWN_ONLY edge cases with 0, 1, and 2 hard violations."""
        # 0 hard violations -> preserve ALLOW_AS_UNKNOWN_ONLY
        dec_0, _ = guard_arbiter_decision("ALLOW_AS_UNKNOWN_ONLY", [], [])
        assert dec_0 == "ALLOW_AS_UNKNOWN_ONLY"

        # 1 hard violation + salvageable truthful content -> adjust to ALLOW_WITH_EDITS
        claims_salvageable = [ClaimEntry(claim="Truth", category="Observed", justification="test")]
        findings_1_hard = [{"type": "T1", "severity": "hard", "detail": "Minor hard"}]
        dec_1, _ = guard_arbiter_decision("ALLOW_AS_UNKNOWN_ONLY", claims_salvageable, findings_1_hard)
        assert dec_1 == "ALLOW_WITH_EDITS"

        # 1 hard violation + NO salvageable content (e.g. Hypothesis claims, not poisoned) -> preserve ALLOW_AS_UNKNOWN_ONLY
        claims_hypothesis = [ClaimEntry(claim="Hypo", category="Hypothesis", justification="test")]
        dec_1_no_salvage, _ = guard_arbiter_decision("ALLOW_AS_UNKNOWN_ONLY", claims_hypothesis, findings_1_hard)
        assert dec_1_no_salvage == "ALLOW_AS_UNKNOWN_ONLY"

        # 1 hard violation + 100% unsupported (poisoned) -> BLOCK
        claims_unsupported = [ClaimEntry(claim="Untruth", category="Unsupported", justification="test")]
        dec_poisoned, _ = guard_arbiter_decision("ALLOW_AS_UNKNOWN_ONLY", claims_unsupported, findings_1_hard)
        assert dec_poisoned == "BLOCK"

        # 2 hard violations -> BLOCK
        findings_2_hard = [
            {"type": "T1", "severity": "hard", "detail": "Hard 1"},
            {"type": "T1", "severity": "hard", "detail": "Hard 2"},
        ]
        dec_2, notes_2 = guard_arbiter_decision("ALLOW_AS_UNKNOWN_ONLY", claims_salvageable, findings_2_hard)
        assert dec_2 == "BLOCK"
        assert any("heavily poisoned" in n.lower() for n in notes_2)


# ===========================================================================
# 2. CONTROL 2 ADVERSARIAL CHALLENGES: PRE-FLIGHT SCANNER
# ===========================================================================

class TestAdversarialControl2PreFlightScanner:
    """Stress-test the Pre-Flight Citation & Bounds Scanner with hostile inputs."""

    def test_nested_and_malformed_citation_brackets(self):
        """Test nested brackets, double brackets, and non-citation bracket annotations."""
        sources = [
            SearchSource(title="Alpha", snippet="Revenue was $50 million in 2024.", url="http://alpha.com"),
            SearchSource(title="Beta", snippet="Growth reached 15 percent.", url="http://beta.com"),
        ]
        src_kw = build_source_keyword_sets(sources)
        src_num = build_source_number_sets(sources)

        # Complex text with nested brackets, bracketed text annotations, and citations
        text = (
            "According to reports [[1]], the total revenue reached $50 million [verified]. "
            "However, nested citation [source: [2]] shows growth was 15% [beta]. "
            "Fabricated citation [[99]] claims $999M."
        )

        has_hard, findings = run_preflight_scan(text, sources, src_kw, src_num)
        assert has_hard is True

        # Verify citation 99 was caught
        found_99 = any("99" in f.get("detail", "") and "non-existent" in f.get("detail", "") for f in findings)
        assert found_99, f"Expected out-of-bounds citation 99 in findings: {findings}"

    def test_extreme_and_negative_citation_indices(self):
        """Test citation indices like [0], [-1], [999999999], [001]."""
        sources = [
            SearchSource(title="Doc 1", snippet="Federal statute applies to businesses.", url="http://doc1.com"),
        ]
        src_kw = build_source_keyword_sets(sources)
        src_num = build_source_number_sets(sources)

        # [0] is out of bounds (1-indexed)
        text_0 = "This rule applies universally [0]."
        has_hard_0, findings_0 = run_preflight_scan(text_0, sources, src_kw, src_num)
        assert has_hard_0 is True
        assert any("[0]" in f.get("detail", "") for f in findings_0)

        # [999999999]
        text_huge = "Extreme citation test [999999999]."
        has_hard_huge, findings_huge = run_preflight_scan(text_huge, sources, src_kw, src_num)
        assert has_hard_huge is True
        assert any("[999999999]" in f.get("detail", "") for f in findings_huge)

        # Valid [1]
        text_valid = "Federal statute applies to businesses [1]."
        has_hard_valid, findings_valid = run_preflight_scan(text_valid, sources, src_kw, src_num)
        assert has_hard_valid is False
        assert len(findings_valid) == 0

    def test_adversarial_formatted_currencies_multipliers_and_percentages(self):
        """Test complex currency formatting, commas, decimals, multipliers ($1.5M, 100k, €50B, 0.05%)."""
        sources = [
            SearchSource(
                title="Financials",
                snippet="Revenue was $1.5 million with $500,000 net income. Operating margin was 25.5%. Fee was €200.",
                url="http://finance.com"
            )
        ]
        src_kw = build_source_keyword_sets(sources)
        src_num = build_source_number_sets(sources)

        # Test valid variations
        valid_texts = [
            "Revenue reached $1.5M [1].",
            "Revenue was 1.5 million [1].",
            "Net income recorded at $500,000 [1].",
            "Operating margin came in at 25.5% [1].",
            "Fee amounted to €200 [1].",
        ]
        for vt in valid_texts:
            has_hard, findings = run_preflight_scan(vt, sources, src_kw, src_num)
            assert not has_hard, f"Valid text '{vt}' produced false positive findings: {findings}"

        # Test fabricated/unbacked variations
        invalid_texts = [
            "Revenue was $1.6M [1].",               # Unbacked $1.6M
            "Net income was $500,001 [1].",         # Off-by-one $500,001
            "Operating margin reached 30% [1].",     # Unbacked 30%
            "Fee was $300 [1].",                    # Unbacked 300
        ]
        for it in invalid_texts:
            has_hard, findings = run_preflight_scan(it, sources, src_kw, src_num)
            assert has_hard is True, f"Invalid text '{it}' was NOT caught by preflight scanner"
            assert any("unbacked numeric" in f.get("detail", "").lower() for f in findings)

    def test_zero_retrieved_sources_boundary(self):
        """Test preflight scanner behavior when zero sources are retrieved."""
        # Citation [1] with 0 sources -> must flag hard violation
        text = "The treaty was signed in Geneva [1]."
        has_hard, findings = run_preflight_scan(text, sources=[])
        assert has_hard is True
        assert len(findings) == 1
        assert "available sources: 0" in findings[0]["detail"]

        # Text with NO citations and 0 sources -> should pass preflight cleanly
        text_no_citations = "This is a general query with no search results."
        has_hard_clean, findings_clean = run_preflight_scan(text_no_citations, sources=[])
        assert has_hard_clean is False
        assert len(findings_clean) == 0

        # Null sources argument
        has_hard_null, findings_null = run_preflight_scan(text, sources=None)
        assert has_hard_null is True

    def test_complex_multi_sentence_multi_citation_segmentation(self):
        """Test sentences with multiple citations and numbers across conjunctions."""
        sources = [
            SearchSource(title="Dept A", snippet="Group A produced 100 units.", url="http://a.com"),
            SearchSource(title="Dept B", snippet="Group B produced 200 units.", url="http://b.com"),
        ]
        src_kw = build_source_keyword_sets(sources)
        src_num = build_source_number_sets(sources)

        # Correctly attributed numbers
        text_correct = "Group A produced 100 units [1], whereas Group B produced 200 units [2]."
        has_hard_c, findings_c = run_preflight_scan(text_correct, sources, src_kw, src_num)
        assert not has_hard_c, f"Unexpected findings for correct attribution: {findings_c}"

        # Cross-attributed numbers: Group A citing [2] for 100 (Dept B has 200, not 100)
        text_swapped = "Group A produced 100 units [2], whereas Group B produced 200 units [1]."
        has_hard_s, findings_s = run_preflight_scan(text_swapped, sources, src_kw, src_num)
        assert has_hard_s is True
        assert len(findings_s) >= 2


# ===========================================================================
# 3. LATENCY BENCHMARK: 1,000 ITERATIONS (<10ms STRICT REQUIREMENT)
# ===========================================================================

class TestLatencyBenchmarkPreFlightScanner:
    """Benchmark the deterministic pre-flight scanner across 1000 diverse iterations."""

    def test_1000_iterations_strict_latency_under_10ms(self):
        """Run 1000 iterations over varied payloads (short, medium, long, adversarial)."""
        sources = [
            SearchSource(title=f"Source {i}", snippet=f"Evidence snippet {i} with figures ${i * 100} and {i * 5}% in year 202{i}.", url=f"http://src{i}.com")
            for i in range(1, 6)
        ]
        src_kw = build_source_keyword_sets(sources)
        src_num = build_source_number_sets(sources)

        payloads = [
            # Payload 1: Clean multi-citation text
            "Evidence snippet 1 with figures $100 and 5% in year 2021 [1]. Evidence snippet 2 with figures $200 [2].",
            # Payload 2: Out of bounds citation [9]
            "Some general text claiming great progress in science [9] with $500 [5].",
            # Payload 3: Fabricated number
            "Evidence snippet 3 with figures $99999 and 77% [3].",
            # Payload 4: Long multi-paragraph document
            (
                "Paragraph 1 discusses evidence snippet 1 with $100 [1].\n"
                "Paragraph 2 discusses evidence snippet 2 with $200 and 10% [2].\n"
                "Paragraph 3 discusses evidence snippet 4 with $400 and 20% in year 2024 [4].\n"
                "Paragraph 4 discusses evidence snippet 5 with $500 [5].\n"
                "Summary statement with no citations."
            ),
            # Payload 5: Empty and whitespace text
            "   \n\t  ",
            # Payload 6: Dense table text
            "| Item | Val | Source |\n| Alpha | $100 | [1] |\n| Beta | $200 | [2] |\n| Gamma | $300 | [3] |",
        ]

        durations = []
        # Warmup 50 iterations
        for _ in range(50):
            p = payloads[_ % len(payloads)]
            run_preflight_scan(p, sources, src_kw, src_num)

        # Timed 1000 iterations
        for i in range(1000):
            p = payloads[i % len(payloads)]
            t0 = time.perf_counter()
            has_hard, findings = run_preflight_scan(p, sources, src_kw, src_num)
            t1 = time.perf_counter()
            durations.append((t1 - t0) * 1000.0)  # in milliseconds

        mean_ms = sum(durations) / len(durations)
        max_ms = max(durations)
        min_ms = min(durations)
        durations.sort()
        p50_ms = durations[500]
        p95_ms = durations[950]
        p99_ms = durations[990]

        print(f"\n[Pre-Flight Scanner 1000-Iteration Benchmark]")
        print(f"  Mean:   {mean_ms:.4f} ms")
        print(f"  P50:    {p50_ms:.4f} ms")
        print(f"  P95:    {p95_ms:.4f} ms")
        print(f"  P99:    {p99_ms:.4f} ms")
        print(f"  Max:    {max_ms:.4f} ms")
        print(f"  Min:    {min_ms:.4f} ms")

        assert max_ms < 10.0, f"Max latency {max_ms:.4f} ms exceeded strict 10ms threshold"
        assert p99_ms < 2.0, f"P99 latency {p99_ms:.4f} ms exceeded 2ms target"
        assert mean_ms < 0.5, f"Mean latency {mean_ms:.4f} ms exceeded 0.5ms target"


# ===========================================================================
# 4. CONTROL 3 & 4 ADVERSARIAL CHALLENGES: SCHEMAS & REPAIR CONSTRAINTS
# ===========================================================================

class TestAdversarialControl3And4SchemasAndRepair:
    """Stress-test clause isolation, AST edits, negative constraint extraction and formatting."""

    def test_ast_id_edits_with_missing_and_corrupted_ids(self):
        """Test apply_edits_by_id when target IDs are missing, mismatched, or duplicated."""
        atomic_claims = [
            {"claim_id": "c-1", "text": "First claim to keep."},
            {"claim_id": "c-2", "text": "Second claim to delete."},
            {"claim_id": "c-3", "text": "Third claim to rewrite."},
            {"claim_id": "c-4", "text": "Fourth claim to move to unknown."},
            {"claim_id": "c-5", "text": "Fifth claim intact."},
        ]

        edits = [
            EditEntry(action="DELETE", target="Second claim", replacement="", target_id="c-2"),
            EditEntry(action="REWRITE", target="Third claim", replacement="Rewritten third claim.", target_id="c-3"),
            EditEntry(action="MOVE_TO_UNKNOWN", target="Fourth claim", replacement="Unknown fourth.", target_id="c-4"),
            EditEntry(action="DELETE", target="Ghost", replacement="", target_id="non-existent-id"),
            EditEntry(action="REWRITE", target="No ID", replacement="No ID target", target_id=""),
        ]

        modified, summary = apply_edits_by_id(atomic_claims, edits)

        assert len(modified) == 4
        assert not any(c.get("claim_id") == "c-2" for c in modified)
        rewritten = next(c for c in modified if c.get("claim_id") == "c-3")
        assert rewritten["text"] == "Rewritten third claim."
        moved = next(c for c in modified if c.get("claim_id") == "c-4")
        assert moved["is_unknown"] is True
        assert moved["text"] == "Unknown fourth."
        assert "non-existent-id" not in summary

    def test_negative_constraint_extraction_monotonicity_and_deduplication(self):
        """Verify negative constraint extractor deduplicates and formats cleanly."""
        findings = [
            {"type": "T1", "severity": "hard", "detail": "Fabricated citation: referenced non-existent source [5] (available sources: 1..2)."},
            {"type": "T1", "severity": "hard", "detail": "Fabricated citation: referenced non-existent source [5] (available sources: 1..2)."}, # duplicate
            {"type": "T1", "severity": "hard", "detail": "Unbacked numeric claim: [1] does not contain numeric value $500 from 'Price is $500.'"},
            {"type": "T2", "severity": "soft", "detail": "typicality claim used without evidence"},
            {"type": "T5", "severity": "soft", "detail": "will improve efficiency"},
        ]
        edits = [
            EditEntry(action="DELETE", target="Unbacked clinical promise.", replacement=""),
            {"action": "REWRITE", "target": "Vague benefit.", "replacement": "Specific mechanism."},
        ]
        claim_table = [
            {"category": "Unsupported", "claim": "Unsupported claim Alpha"},
            {"category": "Supported", "claim": "Supported claim Beta"},
        ]

        constraints = extract_negative_constraints(
            findings=findings,
            edits=edits,
            claim_table=claim_table,
            max_source_count=2,
        )

        assert len(constraints) == len(set(constraints)), "Constraints must be deduplicated"
        assert any("DO NOT cite non-existent source [5]" in c for c in constraints)
        assert any("DO NOT introduce the unbacked numeric figure $500" in c for c in constraints)
        assert any("DO NOT use typicality words" in c for c in constraints)
        assert any("DO NOT include outcome promises" in c for c in constraints)
        assert any('DO NOT include the claim or text: "Unbacked clinical promise."' in c for c in constraints)
        assert any('DO NOT make the unbacked assertion: "Unsupported claim Alpha"' in c for c in constraints)
        assert not any("Supported claim Beta" in c for c in constraints)

        block = format_negative_constraints_block(constraints)
        assert "### Negative Constraints" in block
        assert block.count("- DO NOT") == len(constraints)

    def test_negative_constraints_with_special_characters_and_html(self):
        """Test negative constraints containing quotes, newlines, HTML tags, and markdown."""
        findings = [
            {"type": "T1", "severity": "hard", "detail": "Fabricated citation: referenced <script>alert(1)</script> [99]."},
            {"type": "T1", "severity": "hard", "detail": r"Unbacked numeric claim: [1] does not contain numeric value $1,234.56 from 'Total was *\$1,234.56*.\nNext line.'"},
        ]
        edits = [
            EditEntry(action="REWRITE", target="Text with \"quotes\" and 'single' quotes", replacement="Clean text.", target_id="c-1"),
        ]
        constraints = extract_negative_constraints(findings=findings, edits=edits)
        assert len(constraints) == 3
        block = format_negative_constraints_block(constraints)
        assert "### Negative Constraints" in block
        for c in constraints:
            assert f"- {c}" in block

    def test_apply_edits_by_id_multiple_edits_on_same_claim(self):
        """Test behavior when multiple sequential edits target the same claim_id."""
        atomic_claims = [
            {"claim_id": "c-100", "text": "Original text."},
        ]
        # First REWRITE, then MOVE_TO_UNKNOWN
        edits = [
            EditEntry(action="REWRITE", target="Original text.", replacement="Step 1 rewrite.", target_id="c-100"),
            EditEntry(action="MOVE_TO_UNKNOWN", target="Step 1 rewrite.", replacement="Step 2 unknown.", target_id="c-100"),
        ]
        modified, summary = apply_edits_by_id(atomic_claims, edits)
        assert len(modified) == 1
        assert modified[0]["text"] == "Step 2 unknown."
        assert modified[0]["is_unknown"] is True
        assert "REWROTE claim c-100" in summary
        assert "MOVED claim c-100 to Unknown" in summary

    def test_source_match_resilience_to_empty_and_corrupt_search_sources(self):
        """Test source_match module with empty, None, and edge-case SearchSource objects."""
        sources = [
            SearchSource(title="", snippet="", url="http://empty.com"),
            SearchSource(title="   ", snippet="   ", url="http://spaces.com"),
            SearchSource(title="Valid Doc", snippet="Revenue was $10 million in 2023.", url="http://valid.com"),
        ]
        src_kw = build_source_keyword_sets(sources)
        src_num = build_source_number_sets(sources)

        assert len(src_kw) == 3
        assert len(src_num) == 3
        assert len(src_kw[0]) == 0
        assert len(src_num[0]) == 0

        # Out-of-bounds citation against 3 sources ([4] is out of bounds)
        text_oob = "Some claim [4]."
        has_hard_oob, findings_oob = run_preflight_scan(text_oob, sources, src_kw, src_num)
        assert has_hard_oob is True
        assert any("[4]" in f.get("detail", "") for f in findings_oob)

        # Citing empty source [1] with a substantive statement -> unbacked citation finding
        text_empty_cite = "Substantive complex assertion about pharmaceutical research [1]."
        has_hard_emp, findings_emp = run_preflight_scan(text_empty_cite, sources, src_kw, src_num)
        assert has_hard_emp is True
        assert any("Unbacked citation: [1]" in f.get("detail", "") for f in findings_emp)

        # Citing valid source [3] with correct number -> passes
        text_valid = "Revenue was $10 million in 2023 [3]."
        has_hard_val, findings_val = run_preflight_scan(text_valid, sources, src_kw, src_num)
        assert not has_hard_val
        assert len(findings_val) == 0

    def test_e2e_two_turn_fail_closed_fallback_simulation(self):
        """Simulate a pathological scenario where GPT-1 fails Turn 1 and Turn 2, triggering Unknown fallback."""
        from pipeline.stages import stage_rewrite_loop
        from pipeline.metrics import PipelineMetrics

        # Construct a pipeline state with persistent hard violation
        state = {
            "prompt": "What is the revenue?",
            "gpt1_output": "Revenue is $999 billion [5].",
            "sanitized_output": "Revenue is $999 billion [5].",
            "gpt1_system": "You are a helpful assistant.",
            "gpt1_cfg": {"provider": "mock", "model": "mock"},
            "gpt2_cfg": {"provider": "mock", "model": "mock"},
            "gpt3_cfg": {"provider": "mock", "model": "mock"},
            "gpt2_system": "Verifier",
            "flags": {},
            "tier": "strict",
            "max_rewrite_loops": 2,
            "enable_repair": True,
            "metrics": PipelineMetrics(request_id="test-sim", prompt_length=20),
            "search_sources": [
                SearchSource(title="Report", snippet="Revenue was $50 million.", url="http://test.com")
            ],
            "src_kw_sets": [set(["revenue", "50", "million"])],
            "src_num_sets": [set([50.0, 50000000.0])],
            "findings": [
                {"type": "T1", "severity": "hard", "detail": "Fabricated citation: referenced non-existent source [5] (available sources: 1..1)."}
            ],
            "arbiter_decision": "BLOCK",
            "arbiter_edits": [],
            "claim_table": [
                {"claim": "Revenue is $999 billion [5].", "category": "Unsupported", "justification": "test"}
            ],
            "negative_constraints": [],
            "empty_arbiter": {},
            "search_kwargs": {},
            "decomp_kwargs": {},
        }
        state["metrics"].start()

        # Check negative constraint extraction before loop
        c_extracted = extract_negative_constraints(
            findings=state["findings"],
            claim_table=state["claim_table"],
            max_source_count=1,
        )
        assert any("DO NOT cite non-existent source [5]" in c for c in c_extracted)
        assert any("DO NOT make the unbacked assertion" in c for c in c_extracted)

    def test_sanitizer_grammar_cleanup_with_adversarial_conjunction_chains(self):
        """Test sanitizer on heavily degraded punctuation and coordinator chains."""
        corrupted_text = (
            ",,, ... Whereas, and, but the company increased revenue. \n\n"
            "While, however, approved with., ;\n"
            "and also the clinical outcomes were observed.\n"
        )
        cleaned = _clean_grammar_and_punctuation(corrupted_text)

        assert not cleaned.startswith(",")
        assert not cleaned.startswith(".")
        assert "Whereas, and, but" not in cleaned
        assert "with." in cleaned or "with" not in cleaned
        assert not any(line.strip().startswith(("and", "but", "whereas", "while")) for line in cleaned.splitlines() if line.strip())
