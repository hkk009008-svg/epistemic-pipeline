"""Empirical Challenger 1 Test Suite for Milestone M2: Subordinate Clause AST Disentangler.

Comprehensive empirical verification and adversarial stress-testing of:
1. Exact character span integrity: sentence[span.start_char:span.end_char] == span.raw_text across all generated trees.
2. Syntactic nesting depth scaling (Levels 1 to 5 and beyond to Level 8+).
3. Subordinator taxonomy coverage & misclassification traps (Concessive, Conditional, Temporal, Relative, Coordinate, Participial).
4. Subordinator false-trigger vulnerabilities (nouns starting with -ing, compound subjects with 'and', etc.).
5. Delimiter collisions, abbreviation periods, parentheticals, and bracket notations [1, 2].
6. Surgical excision combinations: root matrix promotion, middle relative clause subject-predicate rejoining, coordinate stripping, multi-sentence handling.
7. Determinism and execution latency benchmarks.
"""
from __future__ import annotations

import itertools
import time
import pytest

from pipeline.source_match import (
    ClauseType,
    PropositionSpan,
    parse_clause_ast,
    disentangle_and_excise,
    _CITATION_PATTERN,
)
from pipeline.sanitizer import clean_grammar_and_punctuation


# ==============================================================================
# Suite 1: Exact Character Span Integrity & Boundary Verification
# ==============================================================================

class TestCharacterSpanIntegrity:
    """Stress tests guaranteeing sentence[span.start_char:span.end_char] == span.raw_text
    across diverse sentence architectures, unicode strings, citations, and delimiter styles.
    """

    TEST_SENTENCES = [
        # Simple Level 1
        "The experimental vaccine showed 95% efficacy against severe infection [1].",
        # Level 2 Concessive
        "Although the study cohort was small [1], the intervention demonstrated significant efficacy [2].",
        # Level 2 Conditional
        "If serum creatinine levels exceed 2.0 mg/dL [1], dosage reduction is mandatory [2].",
        # Level 2 Temporal
        "While cardiac telemetry was recorded continuously [1], blood pressure remained stable [2].",
        # Level 2 Relative
        "The targeted kinase inhibitor, which was synthesized in 2023 [1], suppressed tumor growth [2].",
        # Level 2 Participial
        "Having completed the 12-week dosing phase [1], patients entered the open-label extension [2].",
        # Level 3 Nested with dashes and semicolons
        "Although preliminary results were promising [1] -- assuming adequate patient adherence [2] -- the agency requested additional data [3]; however, manufacturing proceeded [4].",
        # Level 4 Deep Nested with multiple citations and punctuation
        "Notwithstanding that cohort A achieved complete remission [1], because baseline biomarker levels were elevated [2], if secondary resistance emerges [3], therapy must be discontinued immediately [4].",
        # Level 5 Deep Hierarchical with parentheticals and quotes
        "In spite of initial skepticism, while long-term safety data were compiled [1], provided that hepatic enzymes remain normal [2], the clinical protocol was approved [3], which enabled nationwide distribution [4], but ongoing surveillance is required [5].",
        # Em-dash and newline delimiters
        "Whereas baseline cardiac output was reduced [1]—when pulmonary arterial pressure increased [2]—the patient responded to vasodilators [3]\nand oxygen saturation normalized [4].",
        # Mixed numbers, currencies, and percentages inside clauses
        "Because the treatment cost $1,250.50 per cycle [1], although 84.5% of participants reported satisfaction [2], insurance reimbursement was denied [3].",
    ]

    @pytest.mark.parametrize("sentence", TEST_SENTENCES)
    def test_exact_character_span_slice_identity(self, sentence: str) -> None:
        """Every span produced by parse_clause_ast MUST satisfy exact slicing equality."""
        spans = parse_clause_ast(sentence)
        assert len(spans) >= 1, f"No spans parsed for sentence: '{sentence}'"

        for span in spans:
            # 1. Exact character slice identity
            sliced_text = sentence[span.start_char:span.end_char]
            assert sliced_text == span.raw_text, (
                f"Span slice mismatch for {span.span_id}: "
                f"sentence[{span.start_char}:{span.end_char}]='{sliced_text}' vs raw_text='{span.raw_text}'"
            )
            # 2. Non-negative and monotonic offsets
            assert 0 <= span.start_char < span.end_char <= len(sentence), (
                f"Invalid span bounds: start={span.start_char}, end={span.end_char}, len={len(sentence)}"
            )
            # 3. Cleaned text must not be empty
            assert len(span.cleaned_text.strip()) > 0, f"Empty cleaned text in span {span.span_id}"

    def test_monotonic_span_ordering(self) -> None:
        """Spans must be strictly ordered monotonically by start_char without backward overlap."""
        sentence = (
            "Although early biomarker data were promising [1], "
            "because the sample size was limited [2], "
            "if validation fails in phase 3 [3], "
            "the sponsor will terminate the clinical development program [4], "
            "which was initiated in 2021 [5]."
        )
        spans = parse_clause_ast(sentence)
        for i in range(len(spans) - 1):
            assert spans[i].start_char < spans[i + 1].start_char, (
                f"Spans {spans[i].span_id} and {spans[i+1].span_id} not strictly ascending in start_char"
            )
            assert spans[i].end_char <= spans[i + 1].start_char, (
                f"Overlap detected between span {spans[i].span_id} (end={spans[i].end_char}) "
                f"and span {spans[i+1].span_id} (start={spans[i+1].start_char})"
            )


# ==============================================================================
# Suite 2: Subordinator Taxonomy & Empirical Failure Mode Mining
# ==============================================================================

class TestSubordinatorClassificationAndDefects:
    """Empirical verification of ClauseType taxonomy and detection of classification flaws."""

    CONCESSIVE_STANDARD = [
        ("Although the initial trial was small [1], results were positive [2].", ClauseType.CONCESSIVE, "although"),
        ("Even though adverse events occurred [1], tolerability was acceptable [2].", ClauseType.CONCESSIVE, "even though"),
        ("Though preliminary data were limited [1], efficacy was demonstrated [2].", ClauseType.CONCESSIVE, "though"),
        ("Whereas cohort A received standard care [1], cohort B received the novel agent [2].", ClauseType.CONCESSIVE, "whereas"),
        ("Despite elevated liver enzymes [1], the patient completed the course [2].", ClauseType.CONCESSIVE, "despite"),
        ("In spite of severe baseline disease [1], clinical recovery was observed [2].", ClauseType.CONCESSIVE, "in spite of"),
    ]

    CONDITIONAL_STANDARD = [
        ("If viral load drops below detection threshold [1], isolation is stopped [2].", ClauseType.CONDITIONAL, "if"),
        ("Even if secondary mutations arise [1], efficacy is retained [2].", ClauseType.CONDITIONAL, "even if"),
        ("Unless platelet count falls below 50k [1], therapy continues [2].", ClauseType.CONDITIONAL, "unless"),
        ("Provided that renal function remains normal [1], high doses are permitted [2].", ClauseType.CONDITIONAL, "provided that"),
        ("On condition that follow-up visits are kept [1], outpatient care proceeds [2].", ClauseType.CONDITIONAL, "on condition that"),
        ("In the event that bleeding occurs [1], transfusions must begin [2].", ClauseType.CONDITIONAL, "in the event that"),
        ("As long as creatinine clearance is adequate [1], treatment remains safe [2].", ClauseType.CONDITIONAL, "as long as"),
        ("Because the drug crossed the blood-brain barrier [1], neurodegeneration slowed [2].", ClauseType.CONDITIONAL, "because"),
        ("Insofar as clinical endpoints were achieved [1], the trial succeeded [2].", ClauseType.CONDITIONAL, "insofar as"),
        ("Given that the mechanism is targeted [1], toxicity was minimal [2].", ClauseType.CONDITIONAL, "given that"),
    ]

    TEMPORAL_STANDARD = [
        ("While telemetry was active [1], arrhythmias were logged [2].", ClauseType.TEMPORAL, "while"),
        ("When apoptosis reached peak velocity [1], tumor volume shrank [2].", ClauseType.TEMPORAL, "when"),
        ("Whenever body temperature exceeds 38.5 C [1], cultures are drawn [2].", ClauseType.TEMPORAL, "whenever"),
        ("After the 12-week study ended [1], endpoints were evaluated [2].", ClauseType.TEMPORAL, "after"),
        ("Before surgery commenced [1], imaging was verified [2].", ClauseType.TEMPORAL, "before"),
        ("Since antiviral therapy began [1], viral replication ceased [2].", ClauseType.TEMPORAL, "since"),
        ("Until steady state is reached [1], daily monitoring is required [2].", ClauseType.TEMPORAL, "until"),
        ("As soon as symptoms resolved [1], rehabilitation started [2].", ClauseType.TEMPORAL, "as soon as"),
        ("Once therapeutic levels are established [1], dosing intervals widen [2].", ClauseType.TEMPORAL, "once"),
        ("Upon completing the infusion [1], observation continued for 2 hours [2].", ClauseType.TEMPORAL, "upon"),
    ]

    RELATIVE_STANDARD = [
        ("The antibody conjugate, which binds HER2 [1], entered phase 3 [2].", ClauseType.RELATIVE, "which"),
        ("The investigator, who led the global study [1], presented findings [2].", ClauseType.RELATIVE, "who"),
        ("The patient cohort, whom clinicians monitored closely [1], achieved remission [2].", ClauseType.RELATIVE, "whom"),
        ("The biopharma firm, whose platform won awards [1], launched trials [2].", ClauseType.RELATIVE, "whose"),
        ("The clinical center, where the study was conducted [1], reported high compliance [2].", ClauseType.RELATIVE, "where"),
        ("The treatment protocol, whereby dosing is adjusted dynamically [1], lowered toxicity [2].", ClauseType.RELATIVE, "whereby"),
        ("The regulatory framework, wherein safety benchmarks are enforced [1], ensures compliance [2].", ClauseType.RELATIVE, "wherein"),
    ]

    @pytest.mark.parametrize("sentence, expected_type, expected_subord", CONCESSIVE_STANDARD)
    def test_standard_concessive_classification(self, sentence: str, expected_type: ClauseType, expected_subord: str) -> None:
        spans = parse_clause_ast(sentence)
        assert len(spans) >= 2
        assert spans[0].clause_type == expected_type
        assert spans[0].subordinator == expected_subord

    @pytest.mark.parametrize("sentence, expected_type, expected_subord", CONDITIONAL_STANDARD)
    def test_standard_conditional_classification(self, sentence: str, expected_type: ClauseType, expected_subord: str) -> None:
        spans = parse_clause_ast(sentence)
        assert len(spans) >= 2
        assert spans[0].clause_type == expected_type
        assert spans[0].subordinator == expected_subord

    @pytest.mark.parametrize("sentence, expected_type, expected_subord", TEMPORAL_STANDARD)
    def test_standard_temporal_classification(self, sentence: str, expected_type: ClauseType, expected_subord: str) -> None:
        spans = parse_clause_ast(sentence)
        assert len(spans) >= 2
        assert spans[0].clause_type == expected_type
        assert spans[0].subordinator == expected_subord

    @pytest.mark.parametrize("sentence, expected_type, expected_subord", RELATIVE_STANDARD)
    def test_standard_relative_classification(self, sentence: str, expected_type: ClauseType, expected_subord: str) -> None:
        spans = parse_clause_ast(sentence)
        assert len(spans) >= 2
        rel_spans = [s for s in spans if s.clause_type == expected_type]
        assert len(rel_spans) >= 1
        assert rel_spans[0].subordinator == expected_subord

    # --- DEFECT VERIFICATIONS (Remediated in Iteration 2) ---

    def test_defect_notwithstanding_that_misclassified_as_participial(self) -> None:
        """Verify 'Notwithstanding that' is correctly classified as CONCESSIVE."""
        sentence = "Notwithstanding that long-term data are lacking [1], approval was recommended [2]."
        spans = parse_clause_ast(sentence)
        assert spans[0].clause_type == ClauseType.CONCESSIVE
        assert spans[0].subordinator == "notwithstanding that"

    def test_defect_providing_that_misclassified_as_participial(self) -> None:
        """Verify 'Providing that' is correctly classified as CONDITIONAL."""
        sentence = "Providing that cardiac monitoring is clear [1], discharge is approved [2]."
        spans = parse_clause_ast(sentence)
        assert spans[0].clause_type == ClauseType.CONDITIONAL
        assert spans[0].subordinator == "providing that"

    def test_defect_assuming_that_misclassified_as_participial(self) -> None:
        """Verify 'Assuming that' is correctly classified as CONDITIONAL."""
        sentence = "Assuming that patient compliance is high [1], remission is expected [2]."
        spans = parse_clause_ast(sentence)
        assert spans[0].clause_type == ClauseType.CONDITIONAL
        assert spans[0].subordinator == "assuming that"

    def test_defect_noun_phrase_ing_words_treated_as_participials(self) -> None:
        """Verify noun phrases starting with -ing words (e.g. 'Promising candidate')
        are INDEPENDENT and preserve their subject noun phrases upon excision.
        """
        sentence = "Promising candidate demonstrated strong binding affinity [1], while control showed none [2]."
        spans = parse_clause_ast(sentence)
        assert spans[0].clause_type == ClauseType.INDEPENDENT
        assert spans[0].subordinator is None

        # When span 2 (matrix) is excised, span 1's subject is preserved
        excised = disentangle_and_excise(sentence, {"span_2"}, spans)
        assert excised == "Promising candidate demonstrated strong binding affinity [1]."

    def test_defect_mid_clause_and_splitting(self) -> None:
        """Verify in-clause coordinate subject/object is not shattered into fragmented AST nodes."""
        sentence = "Because cohort A [1] and cohort B [2] demonstrated efficacy [3], regulatory filing proceeded [4]."
        spans = parse_clause_ast(sentence)
        assert len(spans) == 2
        assert spans[0].raw_text.strip() == "Because cohort A [1] and cohort B [2] demonstrated efficacy [3]"
        assert spans[0].clause_type == ClauseType.CONDITIONAL
        assert spans[1].raw_text.strip() == "regulatory filing proceeded [4]."
        assert spans[1].clause_type == ClauseType.INDEPENDENT

    def test_defect_citation_comma_format_unextracted(self) -> None:
        """Verify _CITATION_PATTERN extracts comma-separated citations like [1, 2]."""
        sentence = "The therapy was verified in preclinical models [1, 2]."
        spans = parse_clause_ast(sentence)
        assert spans[0].citation_indices == [1, 2]


# ==============================================================================
# Suite 3: Deep Syntactic Nesting (Levels 1 through 5+)
# ==============================================================================

class TestDeepSyntacticNestingScaling:
    """Stress tests on deep hierarchical clause trees with nesting levels up to Level 8+."""

    def test_level1_single_matrix_clause(self) -> None:
        sentence = "The Phase 3 oncology trial met all primary efficacy endpoints [1]."
        spans = parse_clause_ast(sentence)
        assert len(spans) == 1
        assert spans[0].is_matrix is True
        assert spans[0].nesting_level == 1
        assert spans[0].clause_type == ClauseType.INDEPENDENT
        assert spans[0].citation_indices == [1]

    def test_level2_subordinate_matrix(self) -> None:
        sentence = "Although overall survival improved by 4.2 months [1], progression-free survival remained unchanged [2]."
        spans = parse_clause_ast(sentence)
        assert len(spans) == 2
        assert spans[0].is_matrix is False
        assert spans[0].nesting_level == 2
        assert spans[1].is_matrix is True
        assert spans[1].nesting_level == 1

    def test_level3_two_subordinates_matrix(self) -> None:
        sentence = "Although overall survival improved [1], because tolerability was high [2], regulatory approval was granted [3]."
        spans = parse_clause_ast(sentence)
        assert len(spans) == 3
        matrix_spans = [s for s in spans if s.is_matrix]
        assert len(matrix_spans) == 1
        assert spans[-1].is_matrix is True
        assert any(s.nesting_level >= 2 for s in spans)

    def test_level4_three_subordinates_matrix(self) -> None:
        sentence = (
            "Notwithstanding that cohort size was limited [1], "
            "while patients received daily monitoring [2], "
            "if biomarkers remain stable [3], "
            "the protocol permits continued maintenance therapy [4]."
        )
        spans = parse_clause_ast(sentence)
        assert len(spans) == 4
        matrix_spans = [s for s in spans if s.is_matrix]
        assert len(matrix_spans) == 1
        assert spans[-1].is_matrix is True
        assert spans[-1].nesting_level == 1

    def test_level5_complex_periodic_structure(self) -> None:
        sentence = (
            "Having established optimal pharmacokinetic targets in phase 1 [1], "
            "although baseline comorbidities were prevalent across 60% of participants [2], "
            "while continuous telemetry logged vital signs [3], "
            "if creatinine clearance exceeds 50 mL/min [4], "
            "the steering panel approved expedited dose escalation [5], "
            "which improves clinical response rates [6]."
        )
        spans = parse_clause_ast(sentence)
        assert len(spans) == 6
        matrix_spans = [s for s in spans if s.is_matrix]
        assert len(matrix_spans) == 1
        max_nesting = max(s.nesting_level for s in spans)
        assert max_nesting >= 4

    def test_ultra_deep_nesting_level_8_resilience(self) -> None:
        """Adversarial stress test with 8 nested and chained clauses."""
        sentence = (
            "Assuming that regulatory authorization is secured [1], "
            "given that pre-clinical toxicology demonstrated safety [2], "
            "provided that manufacturing yields satisfy GMP standards [3], "
            "before commercial distribution initiates [4], "
            "as soon as quality control releases the primary batch [5], "
            "the pharmaceutical consortium will launch the global vaccine rollout [6], "
            "which protects high-risk populations [7], "
            "and healthcare workers will receive priority allocation [8]."
        )
        spans = parse_clause_ast(sentence)
        assert len(spans) == 8
        matrix_spans = [s for s in spans if s.is_matrix]
        assert len(matrix_spans) == 1

        # Test excision of 5 arbitrary sub-clauses
        unbacked = {"span_1", "span_2", "span_4", "span_5", "span_7"}
        reconstituted = disentangle_and_excise(sentence, unbacked, spans)
        assert reconstituted != ""
        assert "[1]" not in reconstituted
        assert "[2]" not in reconstituted
        assert "[4]" not in reconstituted
        assert "[5]" not in reconstituted
        assert "[7]" not in reconstituted
        assert "[3]" in reconstituted
        assert "[6]" in reconstituted
        assert "[8]" in reconstituted
        assert reconstituted.endswith(".")
        assert reconstituted[0].isupper()


# ==============================================================================
# Suite 4: Punctuation Collisions, Quotes, Citations & Delimiters
# ==============================================================================

class TestPunctuationAndDelimiterStress:
    """Stress tests on complex punctuation, bracket notations, quote nesting,
    embedded numbers with commas/decimals, and dash delimiters.
    """

    def test_embedded_numbers_with_commas_and_decimals_not_corrupted(self) -> None:
        """Numbers with commas like $1,250,000 or decimals like 99.4% must not trigger false splitting."""
        sentence = "Because the hospital spent $1,500,000 on new equipment [1], patient throughput increased by 35.8% [2]."
        spans = parse_clause_ast(sentence)
        assert len(spans) == 2
        assert "$1,500,000" in spans[0].raw_text
        assert "35.8%" in spans[1].raw_text

        reconstituted = disentangle_and_excise(sentence, {"span_1"}, spans)
        assert reconstituted == "Patient throughput increased by 35.8% [2]."

    def test_quoted_phrases_inside_clauses(self) -> None:
        """Direct quotes inside subordinate or matrix clauses must not disrupt AST segmentation."""
        sentence = 'Because the director declared "the milestone was reached" [1], funding was released [2].'
        spans = parse_clause_ast(sentence)
        assert len(spans) == 2
        assert '"the milestone was reached"' in spans[0].raw_text

        reconstituted = disentangle_and_excise(sentence, {"span_1"}, spans)
        assert reconstituted == "Funding was released [2]."

    def test_case_insensitive_subordinators(self) -> None:
        """Subordinators with mixed or uppercase casing (ALTHOUGH, In Spite Of) must parse cleanly."""
        sentence = "ALTHOUGH the trial was small [1], IN SPITE OF baseline variance [2], the therapy succeeded [3]."
        spans = parse_clause_ast(sentence)
        assert len(spans) == 3
        assert spans[0].clause_type == ClauseType.CONCESSIVE
        assert spans[1].clause_type == ClauseType.CONCESSIVE
        assert spans[2].clause_type == ClauseType.INDEPENDENT

        res = disentangle_and_excise(sentence, {"span_3"}, spans)
        assert not res.lower().startswith("although")
        assert res.endswith(".")


# ==============================================================================
# Suite 5: Combinatorial Excision & Grammar Sanitization
# ==============================================================================

class TestCombinatorialExcisionAndSanitization:
    """Stress tests on complex multi-span excision scenarios, verifying no dangling connectors,
    proper clause promotion, and clean punctuation normalization.
    """

    def test_matrix_excision_with_multiple_subordinates_promotes_first(self) -> None:
        sentence = "Because the drug crossed the blood-brain barrier [1], if dosage remains within limits [2], patient cognition stabilizes [3]."
        spans = parse_clause_ast(sentence)
        assert len(spans) == 3

        reconstituted = disentangle_and_excise(sentence, {"span_3"}, spans)
        assert reconstituted != ""
        assert reconstituted.startswith("The drug crossed the blood-brain barrier [1]")
        assert "if dosage remains within limits [2]" in reconstituted
        assert "patient cognition stabilizes" not in reconstituted
        assert reconstituted.endswith(".")

    def test_coordinate_clause_excision_cleans_connectors(self) -> None:
        sentence = "The treatment improved lung function [1], but fatigue was reported in 10% of patients [2], and nausea occurred in 5% [3]."
        spans = parse_clause_ast(sentence)
        assert len(spans) == 3

        res_excise_middle = disentangle_and_excise(sentence, {"span_2"}, spans)
        assert "fatigue" not in res_excise_middle
        assert "improved lung function [1]" in res_excise_middle
        assert "nausea occurred in 5% [3]" in res_excise_middle
        assert not res_excise_middle.startswith("And")
        assert not res_excise_middle.startswith("But")

        res_excise_tail = disentangle_and_excise(sentence, {"span_3"}, spans)
        assert "nausea" not in res_excise_tail
        assert not res_excise_tail.endswith("and.")
        assert not res_excise_tail.endswith("but.")
        assert res_excise_tail.endswith(".")

    def test_defect_multi_sentence_orphaned_subordinator_retention(self) -> None:
        """Verify multi-sentence text with matrix excised in sentence 1 promotes
        sentence 1's subordinate clause to an independent declarative sentence.
        """
        text = (
            "Although trial A succeeded [1], toxicity was observed in cohort B [2]. "
            "Because early results were positive [3], the agency granted fast-track review [4]."
        )
        spans = parse_clause_ast(text)
        # Excise span_2 (matrix of sentence 1) and span_3 (subordinate of sentence 2)
        excised = disentangle_and_excise(text, {"span_2", "span_3"}, spans)
        assert excised.startswith("Trial A succeeded [1].")
        assert "The agency granted fast-track review [4]." in excised


# ==============================================================================
# Suite 6: Adversarial Degenerate Inputs & Boundary Conditions
# ==============================================================================

class TestAdversarialDegenerateInputs:
    """Stress tests on adversarial non-standard inputs, empty strings, punctuation only."""

    @pytest.mark.parametrize("bad_input", [
        "",
        "   ",
        "\t\n\r",
        "...",
        ",,,,",
        ";;;",
        "---",
        "  ,  ;  .  ",
    ])
    def test_degenerate_inputs_handled_without_crash(self, bad_input: str) -> None:
        spans = parse_clause_ast(bad_input)
        assert isinstance(spans, list)
        reconstituted = disentangle_and_excise(bad_input, set(), spans)
        assert isinstance(reconstituted, str)
        assert clean_grammar_and_punctuation(bad_input) == ""

    def test_massive_repetition_stress(self) -> None:
        """Long repetitive string with many clauses to verify no recursion limit or memory leak."""
        base_clause = "although the drug was effective in subgroup [1], "
        massive_text = (base_clause * 50) + "the overall trial succeeded [2]."
        spans = parse_clause_ast(massive_text)
        assert len(spans) > 20
        for s in spans:
            assert massive_text[s.start_char:s.end_char] == s.raw_text


# ==============================================================================
# Suite 7: Determinism & Performance Latency Benchmarks
# ==============================================================================

class TestDeterminismAndPerformanceBenchmarks:
    """Verify 100% bit-for-bit determinism across 200 executions and measure throughput."""

    def test_strict_ast_and_excision_determinism(self) -> None:
        sentence = (
            "Notwithstanding that retrospective cohort studies indicated 75% survival [1], "
            "while patient telemetry was monitored continuously [2], "
            "if biomarker concentrations remain normal [3], "
            "the medical board approved the expanded protocol [4], "
            "which reduces recurrence rates [5]."
        )

        baseline_spans = parse_clause_ast(sentence)
        baseline_excision = disentangle_and_excise(sentence, {"span_1", "span_3"}, baseline_spans)

        for _ in range(200):
            current_spans = parse_clause_ast(sentence)
            assert len(current_spans) == len(baseline_spans)
            for s_curr, s_base in zip(current_spans, baseline_spans):
                assert s_curr.span_id == s_base.span_id
                assert s_curr.clause_type == s_base.clause_type
                assert s_curr.raw_text == s_base.raw_text
                assert s_curr.cleaned_text == s_base.cleaned_text
                assert s_curr.start_char == s_base.start_char
                assert s_curr.end_char == s_base.end_char
                assert s_curr.subordinator == s_base.subordinator
                assert s_curr.citation_indices == s_base.citation_indices
                assert s_curr.is_matrix == s_base.is_matrix
                assert s_curr.nesting_level == s_base.nesting_level

            current_excision = disentangle_and_excise(sentence, {"span_1", "span_3"}, current_spans)
            assert current_excision == baseline_excision

    def test_throughput_and_latency_per_sentence(self) -> None:
        """Verify AST parsing and excision completes in sub-millisecond time (<0.5ms per sentence)."""
        sentence = (
            "Although the trial achieved an 85% response rate [1], "
            "if serum biomarker levels exceed 5.0 ng/mL [2], "
            "the administering physician must taper the dosage immediately [3], "
            "which prevents irreversible renal toxicity [4]."
        )

        import gc
        gc.collect()
        gc.disable()
        try:
            # Warmup
            for _ in range(20):
                spans = parse_clause_ast(sentence)
                disentangle_and_excise(sentence, {"span_2"}, spans)

            # 500 benchmark iterations
            latencies = []
            for _ in range(500):
                t0 = time.perf_counter()
                spans = parse_clause_ast(sentence)
                _ = disentangle_and_excise(sentence, {"span_2"}, spans)
                dur_ms = (time.perf_counter() - t0) * 1000.0
                latencies.append(dur_ms)
        finally:
            gc.enable()

        avg_lat = sum(latencies) / len(latencies)
        p99_lat = sorted(latencies)[int(0.99 * len(latencies))]
        assert avg_lat < 0.5, f"Average AST parse + excise latency {avg_lat:.4f}ms exceeded 0.5ms"
        assert p99_lat < 1.0, f"P99 latency {p99_lat:.4f}ms exceeded 1.0ms"
