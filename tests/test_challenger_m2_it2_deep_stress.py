"""Deep Adversarial Stress Test Suite for Milestone M2 (Iteration 2).
Authored by Challenger 1 to exhaustively test boundary conditions, complex subordinators,
participial disambiguation, multi-citation patterns, and excision reconstitution.
"""
import pytest
from pipeline.source_match import (
    ClauseType,
    PropositionSpan,
    parse_clause_ast,
    disentangle_and_excise,
    _parse_citation_group,
)
from pipeline.sanitizer import clean_grammar_and_punctuation


class TestComplexSubordinatorsAndTaxonomy:
    """Stress testing all subordinator classes across upper, lower, and mixed cases."""

    ALL_SUBORDINATOR_CASES = [
        # Concessive
        ("Notwithstanding that the sample size was small [1], results were statistically robust [2].", ClauseType.CONCESSIVE, "notwithstanding that"),
        ("In spite of persistent adverse reactions [1], the treatment was deemed safe [2].", ClauseType.CONCESSIVE, "in spite of"),
        ("Even though genetic markers were absent [1], therapy remained effective [2].", ClauseType.CONCESSIVE, "even though"),
        ("Although baseline liver enzymes were elevated [1], no hepatotoxicity occurred [2].", ClauseType.CONCESSIVE, "although"),
        ("Though preliminary preclinical data were limited [1], clinical development proceeded [2].", ClauseType.CONCESSIVE, "though"),
        ("Whereas cohort 1 received 50mg daily [1], cohort 2 received 100mg daily [2].", ClauseType.CONCESSIVE, "whereas"),
        ("Despite significant baseline heterogeneity [1], remission was achieved [2].", ClauseType.CONCESSIVE, "despite"),

        # Conditional
        ("On condition that continuous monitoring is enabled [1], outpatient dosing is permitted [2].", ClauseType.CONDITIONAL, "on condition that"),
        ("In the event that severe neutropenia develops [1], administration must stop immediately [2].", ClauseType.CONDITIONAL, "in the event that"),
        ("Provided that renal clearance is above 60 mL/min [1], standard dosing applies [2].", ClauseType.CONDITIONAL, "provided that"),
        ("Providing that cardiac telemetry remains normal [1], discharge is scheduled [2].", ClauseType.CONDITIONAL, "providing that"),
        ("As long as viral titer remains undetectable [1], transmission risk is zero [2].", ClauseType.CONDITIONAL, "as long as"),
        ("Even if secondary resistance emerges [1], salvage therapy is available [2].", ClauseType.CONDITIONAL, "even if"),
        ("Insofar as regulatory guidelines require [1], safety auditing was conducted [2].", ClauseType.CONDITIONAL, "insofar as"),
        ("Given that the mechanism of action is targeted [1], off-target effects were minimal [2].", ClauseType.CONDITIONAL, "given that"),
        ("Assuming that patient compliance reaches 90% [1], complete remission is expected [2].", ClauseType.CONDITIONAL, "assuming that"),
        ("Unless platelet levels fall below 30,000 [1], full dose intensity is maintained [2].", ClauseType.CONDITIONAL, "unless"),
        ("If serum creatinine exceeds 2.5 mg/dL [1], dose reduction is mandatory [2].", ClauseType.CONDITIONAL, "if"),
        ("Because the drug crossed the blood-brain barrier [1], central nervous system efficacy was observed [2].", ClauseType.CONDITIONAL, "because"),

        # Temporal
        ("As soon as hemodynamic stability is confirmed [1], extubation may begin [2].", ClauseType.TEMPORAL, "as soon as"),
        ("Whenever body temperature exceeds 38.5 C [1], blood cultures are required [2].", ClauseType.TEMPORAL, "whenever"),
        ("Before surgical resection began [1], angiography was completed [2].", ClauseType.TEMPORAL, "before"),
        ("After the 24-week follow-up concluded [1], secondary endpoints were analyzed [2].", ClauseType.TEMPORAL, "after"),
        ("Since combination therapy was initiated [1], tumor progression halted [2].", ClauseType.TEMPORAL, "since"),
        ("Until steady-state concentration is reached [1], daily plasma sampling is required [2].", ClauseType.TEMPORAL, "until"),
        ("While telemetry remained active [1], zero ventricular arrhythmias occurred [2].", ClauseType.TEMPORAL, "while"),
        ("When peak plasma concentration was reached [1], blood pressure normalized [2].", ClauseType.TEMPORAL, "when"),
        ("Once therapeutic blood levels stabilize [1], monitoring frequency decreases [2].", ClauseType.TEMPORAL, "once"),
        ("Upon completion of the infusion protocol [1], vital signs were recorded [2].", ClauseType.TEMPORAL, "upon"),

        # Relative
        ("The antibody conjugate, which targets HER2 receptors [1], improved survival [2].", ClauseType.RELATIVE, "which"),
        ("The lead investigator, who supervised the phase 3 trial [1], reported findings [2].", ClauseType.RELATIVE, "who"),
        ("The clinical cohort, whom investigators tracked for five years [1], maintained remission [2].", ClauseType.RELATIVE, "whom"),
        ("The biotechnology firm, whose patent was granted in 2021 [1], expanded manufacturing [2].", ClauseType.RELATIVE, "whose"),
        ("The clinical center, where the multi-site trial was headquartered [1], enrolled 500 patients [2].", ClauseType.RELATIVE, "where"),
        ("The delivery mechanism, whereby nanoparticles cross the membrane [1], enhanced bioavailability [2].", ClauseType.RELATIVE, "whereby"),
        ("The protocol amendment, wherein safety checkpoints were added [1], reduced adverse events [2].", ClauseType.RELATIVE, "wherein"),
    ]

    @pytest.mark.parametrize("sentence, exp_type, exp_subord", ALL_SUBORDINATOR_CASES)
    def test_subordinator_parsing_and_classification(self, sentence: str, exp_type: ClauseType, exp_subord: str) -> None:
        spans = parse_clause_ast(sentence)
        assert len(spans) >= 2
        # Find matching span
        matching_spans = [s for s in spans if s.subordinator == exp_subord]
        assert len(matching_spans) == 1, f"Expected exactly 1 span with subordinator '{exp_subord}', got {len(matching_spans)}"
        assert matching_spans[0].clause_type == exp_type

    @pytest.mark.parametrize("sentence, exp_type, exp_subord", ALL_SUBORDINATOR_CASES)
    def test_character_slice_exactness(self, sentence: str, exp_type: ClauseType, exp_subord: str) -> None:
        spans = parse_clause_ast(sentence)
        for s in spans:
            assert sentence[s.start_char:s.end_char] == s.raw_text


class TestParticipialVsNounPhraseDisambiguation:
    """Stress tests verifying that participials are strictly recognized while noun phrases with -ing words are not."""

    TRUE_PARTICIPIALS = [
        ("Having completed the primary induction phase [1], patients entered maintenance therapy [2].", "having completed"),
        ("Having been evaluated in three multicenter trials [1], the therapeutic agent received FDA clearance [2].", "having been"),
        ("Being aware of potential hypersensitivity reactions [1], clinicians administered premedication [2].", "being aware"),
    ]

    NOUN_PHRASE_SUBJECTS = [
        "Promising candidate demonstrated potent inhibitory activity [1], while control showed baseline activity [2].",
        "The steering committee approved the adaptive clinical design [1], although logistical hurdles remained [2].",
        "Ongoing surveillance is strictly required [1], although short-term safety is demonstrated [2].",
        "Existing protocols prevent cross-contamination [1], while novel automation is developed [2].",
        "Leading researchers published their peer-reviewed findings [1], while secondary analyses commenced [2].",
        "Smoking cigarettes significantly increases cardiovascular risk [1], whereas smoking cessation reduces it [2].",
        "Rising inflation affects healthcare reimbursement [1], unless government subsidies intervene [2].",
        "Operating expenses escalated during the multi-site trial [1], although fundraising covered all costs [2].",
        "Targeting oncogenic drivers requires precise diagnostic assays [1], while broad therapies cause toxicity [2].",
        "Developing next-generation therapeutics requires substantial capital [1], although grant funding helps [2].",
    ]

    @pytest.mark.parametrize("sentence, exp_part_subord", TRUE_PARTICIPIALS)
    def test_true_participials_recognized(self, sentence: str, exp_part_subord: str) -> None:
        spans = parse_clause_ast(sentence)
        assert spans[0].clause_type == ClauseType.PARTICIPIAL
        assert exp_part_subord in spans[0].subordinator.lower()

    @pytest.mark.parametrize("sentence", NOUN_PHRASE_SUBJECTS)
    def test_noun_phrases_not_misclassified_as_participials(self, sentence: str) -> None:
        spans = parse_clause_ast(sentence)
        assert spans[0].clause_type == ClauseType.INDEPENDENT
        assert spans[0].subordinator is None

        # Excision of subordinate/matrix clause preserves full subject
        excised = disentangle_and_excise(sentence, {"span_2"}, spans)
        assert excised.endswith(".")
        first_word = sentence.split()[0]
        assert excised.startswith(first_word)


class TestMultiCitationBracketFormats:
    """Stress tests on bracket citation extraction across diverse delimiters and combinations."""

    CITATION_CASES = [
        ("Single citation [1].", [1]),
        ("Comma-separated pair [1, 2].", [1, 2]),
        ("Comma-separated triple [1, 2, 3].", [1, 2, 3]),
        ("Hyphen range [1-4].", [1, 2, 3, 4]),
        ("Mixed commas and ranges [1, 3-5, 9].", [1, 3, 4, 5, 9]),
        ("Spaced comma citations [ 1 , 2 , 3 ].", [1, 2, 3]),
        ("Adjacent brackets [1][2][3].", [1, 2, 3]),
        ("Multiple citation spots: Early [1, 2] and late [3, 4].", [1, 2, 3, 4]),
    ]

    @pytest.mark.parametrize("sentence, exp_indices", CITATION_CASES)
    def test_citation_extraction_fidelity(self, sentence: str, exp_indices: list[int]) -> None:
        spans = parse_clause_ast(sentence)
        extracted = []
        for s in spans:
            extracted.extend(s.citation_indices)
        assert extracted == exp_indices


class TestExcisionReconstitutionEdgeCases:
    """Stress testing edge cases for disentangle_and_excise."""

    def test_empty_unbacked_returns_cleaned_text(self) -> None:
        sentence = "Although data were sparse [1], the conclusion was verified [2]."
        spans = parse_clause_ast(sentence)
        result = disentangle_and_excise(sentence, set(), spans)
        assert result == "Although data were sparse [1], the conclusion was verified [2]."

    def test_all_unbacked_returns_empty_string(self) -> None:
        sentence = "Although data were sparse [1], the conclusion was verified [2]."
        spans = parse_clause_ast(sentence)
        result = disentangle_and_excise(sentence, {"span_1", "span_2"}, spans)
        assert result == ""

    def test_middle_relative_clause_excision_smooth_rejoining(self) -> None:
        sentence = "The kinase inhibitor, which was synthesized in 2021 [1], showed 80% binding affinity [2]."
        spans = parse_clause_ast(sentence)
        result = disentangle_and_excise(sentence, {"span_2"}, spans)
        assert result == "The kinase inhibitor showed 80% binding affinity [2]."

    def test_multi_sentence_complex_matrix_excision(self) -> None:
        text = (
            "Although trial A succeeded [1], toxicity occurred in cohort B [2]. "
            "Because early findings were robust [3], accelerated approval was granted [4]. "
            "If recurrence is detected [5], salvage therapy is initiated [6]."
        )
        spans = parse_clause_ast(text)
        # Excise span_2 (matrix of sent 1) and span_4 (matrix of sent 2) and span_5 (subordinate of sent 3)
        excised = disentangle_and_excise(text, {"span_2", "span_4", "span_5"}, spans)
        assert "Trial A succeeded [1]." in excised
        assert "Early findings were robust [3]." in excised
        assert "Salvage therapy is initiated [6]." in excised
        assert "toxicity occurred" not in excised
        assert "accelerated approval was granted" not in excised
        assert "If recurrence is detected" not in excised
