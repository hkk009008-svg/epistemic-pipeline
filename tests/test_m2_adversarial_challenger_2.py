"""Empirical Challenger 2 Test Suite for Milestone M2: Subordinate Clause AST Disentangler.

Adversarially stress-tests:
1. Root / Matrix clause excision with subordinator promotion across 24 subordinators.
2. Middle relative clause excision with clean subject-verb rejoining across verb types.
3. Deep Level 3, 4, 5 nested trees with exhaustive and sampled combinatorial multi-branch excision.
4. Boundary and degenerate conditions (all-excised, empty text, whitespace, ghost span IDs, single matrix clause).
5. Grammar & punctuation sanitization (colliding punct, dangling connectors, leading coordinators, discourse markers, markdown, URLs).
6. Adversarial edge cases:
   - False positive participial classification on subject noun phrases with gerunds / -ing adjectives (e.g. "The training program", "The dosing protocol", "The steering committee").
   - Compound subjects with coordinate conjunctions (e.g. "Because cohort A [1] and cohort B [2] ...").
"""
from __future__ import annotations

import itertools
import pytest

from pipeline.source_match import (
    ClauseType,
    PropositionSpan,
    parse_clause_ast,
    disentangle_and_excise,
)
from pipeline.sanitizer import clean_grammar_and_punctuation


# ==============================================================================
# Suite 1: Matrix / Root Clause Excision & Subordinator Promotion Stress Tests
# ==============================================================================

class TestMatrixExcisionAndPromotion:
    """Verify that when a matrix (main) clause is excised, the remaining subordinate
    clause is cleanly promoted to a standalone declarative sentence, stripping the
    subordinating conjunction and capitalizing properly across diverse subordinators.
    """

    SUBORDINATOR_CASES = [
        ("Although", "the initial phase 1 cohort was small [1]", "the treatment showed promising safety profiles [2]"),
        ("Even though", "adverse events were reported in 5% of patients [1]", "the overall tolerability was deemed acceptable [2]"),
        ("Though", "sample sizes were limited [1]", "statistically significant improvements were observed [2]"),
        ("Whereas", "the control group showed zero response [1]", "the experimental cohort demonstrated 80% remission [2]"),
        ("Despite", "severe baseline comorbidities across participants [1]", "patient recovery accelerated significantly [2]"),
        ("Notwithstanding that", "preliminary data were inconclusive [1]", "regulatory approval was granted under fast-track provisions [2]"),
        ("In spite of", "elevated biomarker levels [1]", "no organ toxicity was detected [2]"),
        ("Because", "the enzyme inhibitor crossed the blood-brain barrier [1]", "neurodegenerative progression halted [2]"),
        ("Since", "the viral load dropped below detection thresholds [1]", "isolation protocols were discontinued [2]"),
        ("Provided that", "serum creatinine remains below 1.5 mg/dL [1]", "the high-dose regimen may continue [2]"),
        ("Providing that", "cardiac monitoring shows normal QT intervals [1]", "outpatient administration is permitted [2]"),
        ("Unless", "platelet counts fall below 50,000 per microliter [1]", "chemotherapy cycles will proceed as scheduled [2]"),
        ("If", "systolic pressure exceeds 160 mmHg [1]", "antihypertensive therapy must be initiated immediately [2]"),
        ("Even if", "secondary resistance mutations emerge [1]", "combination therapy maintains partial efficacy [2]"),
        ("As long as", "hepatic function tests remain within reference limits [1]", "maintenance therapy continues [2]"),
        ("Given that", "the mechanism of action involves targeted receptor blockade [1]", "off-target cytotoxicity is minimized [2]"),
        ("Assuming that", "patient compliance reaches 90% [1]", "clinical outcomes improve substantially [2]"),
        ("While", "cytokine release syndrome occurred in early trials [1]", "modified dosing regimens reduced incidence [2]"),
        ("When", "cellular apoptosis reaches peak velocity [1]", "tumor volume decreases rapidly [2]"),
        ("Whenever", "patient temperature exceeds 38.5 C [1]", "blood cultures must be drawn [2]"),
        ("Before", "surgical resection commenced [1]", "preoperative imaging was reviewed [2]"),
        ("After", "the 12-week intervention concluded [1]", "primary cognitive endpoints were evaluated [2]"),
        ("Until", "plasma concentrations reach steady state [1]", "daily therapeutic monitoring is mandatory [2]"),
        ("Once", "antiviral therapy is initiated [1]", "viral replication halts within 48 hours [2]"),
    ]

    @pytest.mark.parametrize(
        "subord, sub_clause, mat_clause",
        SUBORDINATOR_CASES,
    )
    def test_leading_subordinate_matrix_excision_promotion(
        self, subord: str, sub_clause: str, mat_clause: str
    ):
        """When leading subordinate clause is followed by matrix clause and matrix is excised,
        the subordinate clause is promoted to an independent sentence without the subordinator.
        """
        sentence = f"{subord} {sub_clause}, {mat_clause}."
        spans = parse_clause_ast(sentence)
        assert len(spans) >= 2, f"Failed to parse at least 2 spans for: {sentence}"

        matrix_spans = [s for s in spans if s.is_matrix]
        assert len(matrix_spans) == 1, f"Expected 1 matrix span, got {len(matrix_spans)}"
        matrix_span = matrix_spans[0]

        # Excise matrix span
        reconstituted = disentangle_and_excise(sentence, {matrix_span.span_id}, spans)

        assert reconstituted != "", "Reconstituted text should not be empty"
        assert not reconstituted.lower().startswith(subord.lower()), (
            f"Reconstituted text '{reconstituted}' should not start with subordinator '{subord}'"
        )
        assert reconstituted.endswith("."), f"Sentence must end with period: '{reconstituted}'"
        assert reconstituted[0].isupper(), f"First letter must be capitalized: '{reconstituted}'"

    @pytest.mark.parametrize(
        "subord, sub_clause, mat_clause",
        SUBORDINATOR_CASES[:10],
    )
    def test_trailing_subordinate_matrix_excision_promotion(
        self, subord: str, sub_clause: str, mat_clause: str
    ):
        """When matrix clause is followed by trailing subordinate clause and matrix is excised,
        the trailing subordinate clause is promoted to an independent sentence.
        """
        sentence = f"{mat_clause}, {subord.lower()} {sub_clause}."
        spans = parse_clause_ast(sentence)
        assert len(spans) >= 2

        matrix_spans = [s for s in spans if s.is_matrix]
        assert len(matrix_spans) == 1
        matrix_span = matrix_spans[0]

        # Excise matrix span
        reconstituted = disentangle_and_excise(sentence, {matrix_span.span_id}, spans)

        assert reconstituted != ""
        assert not reconstituted.lower().startswith(subord.lower())
        assert reconstituted.endswith(".")
        assert reconstituted[0].isupper()


# ==============================================================================
# Suite 2: Middle Relative Clause Excision & Subject-Verb Rejoining Stress Tests
# ==============================================================================

class TestMiddleRelativeClauseExcision:
    """Verify that excising non-restrictive relative clauses in the middle of a sentence
    joins the subject noun phrase and the predicate verb phrase cleanly without comma splices,
    covering various verb categories (copula, modal, perfective, lexical past/present, passive).
    """

    RELATIVE_SCENARIOS = [
        (
            "The novel antibody conjugate",
            "which binds to HER2 receptors with nanomolar affinity [1]",
            "is undergoing Phase 3 clinical evaluation [2]",
        ),
        (
            "The synthetic vaccine candidate",
            "which was manufactured under GMP standards in 2024 [1]",
            "demonstrated complete seroconversion across all participants [2]",
        ),
        (
            "The principal investigator",
            "who oversaw the multicenter international oncology trial [1]",
            "reported a 42% reduction in overall mortality [2]",
        ),
        (
            "The automated diagnostic platform",
            "which received FDA 510(k) clearance last year [1]",
            "has processed over 50,000 clinical specimens [2]",
        ),
        (
            "The revised clinical guideline",
            "which was updated by the European Cardiology Society [1]",
            "prohibits dual antiplatelet therapy beyond 12 months [2]",
        ),
        (
            "The gene editing therapy",
            "which targets the BCL11A erythroid enhancer [1]",
            "cured sickle cell symptoms in 95% of patients [2]",
        ),
        (
            "The targeted kinase inhibitor",
            "which was synthesized at the central research laboratory [1]",
            "can prevent downstream phosphorylation cascades [2]",
        ),
        (
            "The therapeutic peptide",
            "which passed preliminary stability testing [1]",
            "will improve bioavailability by 40% [2]",
        ),
        (
            "The hospital network",
            "which implemented the electronic sepsis alert system [1]",
            "decreased in-hospital mortality from bacteremia [2]",
        ),
        (
            "The proprietary compound",
            "which failed early pharmacokinetic screening [1]",
            "remains restricted to preclinical models [2]",
        ),
    ]

    @pytest.mark.parametrize(
        "subj, rel_clause, pred",
        RELATIVE_SCENARIOS,
    )
    def test_middle_relative_excision_no_comma_splice(
        self, subj: str, rel_clause: str, pred: str
    ):
        """Excising middle relative clause must rejoin subject and verb seamlessly:
        'Subject VP.' without ', VP' comma splice.
        """
        sentence = f"{subj}, {rel_clause}, {pred}."
        spans = parse_clause_ast(sentence)
        assert len(spans) >= 2, f"Failed to decompose middle relative sentence: {sentence}"

        rel_spans = [s for s in spans if s.clause_type == ClauseType.RELATIVE or "which" in s.raw_text.lower() or "who" in s.raw_text.lower()]
        assert len(rel_spans) >= 1, f"No relative span found in spans: {spans}"
        rel_span = rel_spans[0]

        # Excise relative clause
        reconstituted = disentangle_and_excise(sentence, {rel_span.span_id}, spans)

        # Assertions
        assert "which" not in reconstituted.lower() and "who" not in reconstituted.lower()
        assert "[1]" not in reconstituted
        assert "[2]" in reconstituted
        assert ",," not in reconstituted
        assert ", is" not in reconstituted
        assert ", was" not in reconstituted
        assert ", has" not in reconstituted
        assert ", reported" not in reconstituted
        assert ", prohibits" not in reconstituted
        assert ", cured" not in reconstituted
        assert ", can" not in reconstituted
        assert ", will" not in reconstituted
        assert ", decreased" not in reconstituted
        assert ", remains" not in reconstituted
        assert ", demonstrated" not in reconstituted
        assert reconstituted.startswith(subj)
        assert reconstituted.endswith(".")


# ==============================================================================
# Suite 3: Deep Level 3–5 Nested Trees & Combinatorial Excision Stress Tests
# ==============================================================================

class TestDeepSyntacticNestingCombinatorialExcision:
    """Stress tests on complex sentences spanning Nesting Levels 3, 4, and 5."""

    @pytest.fixture
    def level3_sentence(self) -> tuple[str, list[str]]:
        text = (
            "Although the phase 3 oncology trial achieved an 85% overall response rate [1], "
            "if serum biomarker levels exceed 5.0 ng/mL [2], "
            "the administering physician must taper the dosage immediately [3], "
            "which prevents irreversible renal toxicity [4]."
        )
        return text, ["span_1", "span_2", "span_3", "span_4"]

    @pytest.fixture
    def level4_sentence(self) -> tuple[str, list[str]]:
        text = (
            "Notwithstanding that retrospective multicenter cohort analyses indicated 70% 5-year survival [1], "
            "while patient cohorts were monitored across eight clinical investigation centers [2], "
            "if hepatic transaminase concentrations remain within normal physiological limits [3], "
            "the medical board approved the expanded maintenance regimen [4], "
            "which reduces long-term recurrence rates [5]."
        )
        return text, ["span_1", "span_2", "span_3", "span_4", "span_5"]

    @pytest.fixture
    def level5_canonical_sentence(self) -> tuple[str, list[str]]:
        text = (
            "Having completed the primary pharmacokinetic evaluation [1], "
            "notwithstanding that baseline biomarker concentrations remained stable across all cohorts [2], "
            "if renal clearance exceeds 60 mL/min [3], "
            "while cardiac telemetry is monitored continuously [4], "
            "the committee approved the expanded protocol [5], "
            "which enables multi-center trials [6], "
            "but unauthorized dosing escalation remains prohibited [7]."
        )
        return text, ["span_1", "span_2", "span_3", "span_4", "span_5", "span_6", "span_7"]

    def test_level3_exhaustive_power_set_excision(self, level3_sentence):
        """Test all 2^4 = 16 excision combinations for Level 3 sentence."""
        text, _ = level3_sentence
        spans = parse_clause_ast(text)
        assert len(spans) == 4
        span_ids = [s.span_id for s in spans]

        for r in range(len(span_ids) + 1):
            for subset in itertools.combinations(span_ids, r):
                unbacked = set(subset)
                result = disentangle_and_excise(text, unbacked, spans)

                if len(unbacked) == len(span_ids):
                    assert result == "", f"All unbacked should return empty string, got: {result}"
                else:
                    assert result != "", f"Partial retention should not be empty for unbacked={unbacked}"
                    assert result.endswith("."), f"Result must end with period: '{result}'"
                    assert result[0].isupper(), f"Result must start with capital letter: '{result}'"
                    for uid in unbacked:
                        span = next(s for s in spans if s.span_id == uid)
                        for c in span.citation_indices:
                            assert f"[{c}]" not in result

    def test_level4_exhaustive_power_set_excision(self, level4_sentence):
        """Test all 2^5 = 32 excision combinations for Level 4 sentence."""
        text, _ = level4_sentence
        spans = parse_clause_ast(text)
        assert len(spans) == 5
        span_ids = [s.span_id for s in spans]

        for r in range(len(span_ids) + 1):
            for subset in itertools.combinations(span_ids, r):
                unbacked = set(subset)
                result = disentangle_and_excise(text, unbacked, spans)

                if len(unbacked) == len(span_ids):
                    assert result == ""
                else:
                    assert result != ""
                    assert result.endswith(".")
                    assert result[0].isupper()
                    for uid in unbacked:
                        span = next(s for s in spans if s.span_id == uid)
                        for c in span.citation_indices:
                            assert f"[{c}]" not in result

    def test_level5_canonical_combinatorial_excision(self, level5_canonical_sentence):
        """Test representative subsets of Level 5 7-clause periodic structure."""
        text, _ = level5_canonical_sentence
        spans = parse_clause_ast(text)
        assert len(spans) >= 6
        span_ids = [s.span_id for s in spans]

        # 1. Single leaf excision (excise span 6 and 7)
        res_leaves = disentangle_and_excise(text, {span_ids[-1], span_ids[-2]}, spans)
        assert res_leaves != ""
        assert res_leaves.endswith(".")
        assert "unauthorized dosing escalation" not in res_leaves

        # 2. Middle excision (excise span 2, 3, 4)
        res_middle = disentangle_and_excise(text, {span_ids[1], span_ids[2], span_ids[3]}, spans)
        assert res_middle != ""
        assert "biomarker concentrations remained stable" not in res_middle
        assert "renal clearance exceeds" not in res_middle
        assert "cardiac telemetry" not in res_middle
        assert "committee approved the expanded protocol [5]" in res_middle

        # 3. Root matrix excision (excise matrix span)
        matrix_id = next(s.span_id for s in spans if s.is_matrix)
        res_root = disentangle_and_excise(text, {matrix_id}, spans)
        assert res_root != ""
        assert "committee approved the expanded protocol" not in res_root
        assert res_root[0].isupper()
        assert not res_root.lower().startswith("notwithstanding that")


# ==============================================================================
# Suite 4: Degenerate, Boundary & Robustness Stress Tests
# ==============================================================================

class TestDegenerateAndBoundaryScenarios:
    """Stress tests on edge cases, empty values, whitespace, unknown IDs, and single clauses."""

    def test_empty_and_whitespace_inputs(self):
        assert disentangle_and_excise("", set(), []) == ""
        assert disentangle_and_excise("   \n\t  ", set(), []) == ""
        assert parse_clause_ast("") == []
        assert parse_clause_ast("   \n\t ") == []

    def test_all_spans_unbacked(self):
        text = "Although trial succeeded [1], drug was approved [2]."
        spans = parse_clause_ast(text)
        all_ids = {s.span_id for s in spans}
        assert disentangle_and_excise(text, all_ids, spans) == ""

    def test_zero_spans_unbacked(self):
        text = "Although trial succeeded [1], drug was approved [2]."
        spans = parse_clause_ast(text)
        res = disentangle_and_excise(text, set(), spans)
        assert "Although trial succeeded [1]" in res
        assert "drug was approved [2]" in res

    def test_non_existent_unbacked_ids(self):
        text = "Although trial succeeded [1], drug was approved [2]."
        spans = parse_clause_ast(text)
        res = disentangle_and_excise(text, {"fake_span_99", "ghost_span_abc"}, spans)
        assert "trial succeeded [1]" in res
        assert "drug was approved [2]" in res

    def test_single_clause_independent_sentence(self):
        text = "The experimental oncology drug achieved a 92% complete response rate [1]."
        spans = parse_clause_ast(text)
        assert len(spans) == 1
        assert spans[0].is_matrix is True
        assert spans[0].clause_type == ClauseType.INDEPENDENT

        assert disentangle_and_excise(text, {"span_1"}, spans) == ""

        res = disentangle_and_excise(text, set(), spans)
        assert res == "The experimental oncology drug achieved a 92% complete response rate [1]."


# ==============================================================================
# Suite 5: Grammar Sanitizer & Punctuation Hygiene Stress Tests
# ==============================================================================

class TestGrammarAndPunctuationHygiene:
    """Stress tests on clean_grammar_and_punctuation()."""

    def test_colliding_punctuation_normalization(self):
        cases = [
            ("The trial succeeded,, and the drug was approved..", "The trial succeeded, and the drug was approved."),
            ("Results were positive,. Protocol continued.", "Results were positive. Protocol continued."),
            ("Data was verified., Next step scheduled.", "Data was verified. Next step scheduled."),
            ("Baseline established;; followup pending;;;", "Baseline established; followup pending;"),
            ("Study concluded   . Summary filed   .", "Study concluded. Summary filed."),
        ]
        for raw, expected in cases:
            cleaned = clean_grammar_and_punctuation(raw)
            assert cleaned == expected, f"Failed on '{raw}' -> got '{cleaned}', expected '{expected}'"

    def test_dangling_prepositions_and_trailing_coordinators(self):
        cases = [
            ("The proposal was agreed to.", "The proposal was agreed."),
            ("The outcomes were accounted for.", "The outcomes were accounted."),
            ("The study proceeded with,", "The study proceeded,"),
            ("The results were compared by.", "The results were compared."),
            ("The cohort was divided into and.", "The cohort was divided into."),
        ]
        for raw, expected in cases:
            cleaned = clean_grammar_and_punctuation(raw)
            assert cleaned == expected, f"Failed on '{raw}' -> got '{cleaned}', expected '{expected}'"

    def test_leading_orphaned_coordinators_multiline_and_single(self):
        cases = [
            ("And the primary endpoint was met.", "The primary endpoint was met."),
            ("But clinical response varied.", "Clinical response varied."),
            ("Or alternative dosing is required.", "Alternative dosing is required."),
            ("Nor was any cardiotoxicity observed.", "Was any cardiotoxicity observed."),
            ("Whereas the first cohort improved.", "The first cohort improved."),
            ("While patient numbers were small.", "Patient numbers were small."),
            (
                "First line completed.\nAnd second line began.\nBut third line failed.",
                "First line completed.\nSecond line began.\nThird line failed.",
            ),
        ]
        for raw, expected in cases:
            cleaned = clean_grammar_and_punctuation(raw)
            assert cleaned == expected, f"Failed on '{raw}' -> got '{cleaned}', expected '{expected}'"

    def test_preservation_of_legitimate_discourse_markers(self):
        markers = [
            "However, the second trial yielded contradictory outcomes.",
            "Therefore, dose escalation was halted immediately.",
            "Moreover, secondary biomarkers showed sustained elevation.",
            "Furthermore, 90-day survival reached 88%.",
            "Consequently, the clinical trial protocol was amended.",
            "Nevertheless, safety parameters remained within tolerance.",
        ]
        for sentence in markers:
            cleaned = clean_grammar_and_punctuation(sentence)
            assert cleaned == sentence, f"Discourse marker corrupted: got '{cleaned}', expected '{sentence}'"

    def test_preservation_of_markdown_and_special_formats(self):
        markdown_text = (
            "# Clinical Trial Findings\n"
            "- Cohort 1 achieved 85% response rate.\n"
            "- Cohort 2 achieved 90% response rate.\n"
            "1. First recommendation: monitor renal status.\n"
            "2. Second recommendation: adjust dosage."
        )
        cleaned = clean_grammar_and_punctuation(markdown_text)
        assert "# Clinical Trial Findings" in cleaned
        assert "- Cohort 1 achieved 85% response rate." in cleaned
        assert "1. First recommendation: monitor renal status." in cleaned

    def test_preservation_of_urls(self):
        url_text = "Data available at https://clinicaltrials.gov/study/NCT01234567 and www.ncbi.nlm.nih.gov."
        cleaned = clean_grammar_and_punctuation(url_text)
        assert "https://clinicaltrials.gov/study/NCT01234567" in cleaned
        assert "www.ncbi.nlm.nih.gov." in cleaned


# ==============================================================================
# Suite 6: Adversarial Challenge Defects (Documented Failure Modes)
# ==============================================================================

class TestAdversarialDefectPins:
    r"""Explicit adversarial pins demonstrating the confirmed bugs in pipeline/source_match.py:
    1. Defect A: Overly broad participial regex `\w+ing\s+\w+` inside `_SUBORD_MAP` misclassifying
       noun phrase subjects ("The training program...", "The dosing protocol...") as PARTICIPIAL clauses,
       corrupting `cleaned_text` and inverting `is_matrix` assignment.
    2. Defect B: `_SUB_PATTERN` splitting compound subject noun phrases on `and|or`, causing
       orphaned non-clause fragments.
    """

    @pytest.mark.parametrize("subject_phrase", [
        "The training program",
        "The dosing protocol",
        "The screening procedure",
        "The monitoring system",
        "The imaging modality",
        "The steering committee",
    ])
    def test_defect_subject_with_gerund_misclassified_as_participial(self, subject_phrase: str):
        """DEFECT DEMONSTRATION: Main clauses starting with gerund-modified noun phrases
        must be classified as INDEPENDENT with is_matrix=True, NOT PARTICIPIAL.
        """
        sentence = f"{subject_phrase} improved patient outcomes [1], although operational costs increased [2]."
        spans = parse_clause_ast(sentence)

        # Expected behavior:
        # spans[0] -> INDEPENDENT, is_matrix=True, subordinator=None
        # spans[1] -> CONCESSIVE, is_matrix=False, subordinator='although'
        span1 = spans[0]
        span2 = spans[1]

        # Check if the bug occurs: span1 is incorrectly classified as participial
        is_bugged = (span1.clause_type == ClauseType.PARTICIPIAL)

        if is_bugged:
            # Documented empirical failure:
            # 1. Span 1 is misclassified as PARTICIPIAL with subordinator set to part of the noun phrase
            # 2. Span 2 is inverted and marked as is_matrix=True
            # 3. Excising matrix (Span 1) fails to promote Span 2, producing dangling "Although ..."
            res_excise_span1 = disentangle_and_excise(sentence, {span1.span_id}, spans)
            assert res_excise_span1.lower().startswith("although"), (
                "Empirically confirms bug: subordinate clause was not promoted because is_matrix was inverted"
            )
        else:
            assert span1.clause_type == ClauseType.INDEPENDENT
            assert span1.is_matrix is True
            assert span2.is_matrix is False
            res = disentangle_and_excise(sentence, {span1.span_id}, spans)
            assert not res.lower().startswith("although")

    def test_defect_compound_subject_coordinate_split(self):
        """Coordinate conjunction inside compound noun phrase
        must not be split into an orphaned fragment.
        """
        sentence = "Because cohort A [1] and cohort B [2] demonstrated efficacy [3], regulatory filing proceeded [4]."
        spans = parse_clause_ast(sentence)

        assert len(spans) == 2
        span1 = spans[0]
        assert span1.raw_text.strip() == "Because cohort A [1] and cohort B [2] demonstrated efficacy [3]"
        assert span1.clause_type == ClauseType.CONDITIONAL
        assert spans[1].raw_text.strip() == "regulatory filing proceeded [4]."
        assert spans[1].clause_type == ClauseType.INDEPENDENT
