"""Comprehensive test suite for Syntactic Nesting Levels 1-5 and Surgical Clause AST Excision.

Validates Milestone M2:
- Syntactic Nesting Levels 1 through 5 AST decomposition.
- Exact character span offset integrity: sentence[s.start_char:s.end_char] == s.raw_text.
- Surgical excision across Root, Middle, Leaf, and Multi-Branch positions.
- Grammatical hygiene invariants (subordinator promotion, comma splice prevention, orphan connector stripping).
"""
import re
import pytest

from pipeline.source_match import (
    ClauseType,
    PropositionSpan,
    parse_clause_ast,
    disentangle_and_excise,
)
from pipeline.sanitizer import clean_grammar_and_punctuation


class TestNestingLevel1SingleSubordinate:
    """Nesting Level 1: Single subordinate clause with matrix clause."""

    def test_level1_leading_conditional_ast_and_leaf_excision(self):
        text = "Because the primary trial parameters were verified within range [1], the therapeutic protocol was approved [2]."
        spans = parse_clause_ast(text)
        assert len(spans) == 2, f"Expected 2 spans, got {len(spans)}"
        assert spans[0].clause_type == ClauseType.CONDITIONAL
        assert spans[0].subordinator == "because"
        assert spans[0].citation_indices == [1]
        assert spans[0].nesting_level >= 1
        assert spans[1].clause_type == ClauseType.INDEPENDENT
        assert spans[1].is_matrix is True
        assert spans[1].citation_indices == [2]

        # Verify span slice integrity
        for s in spans:
            assert text[s.start_char:s.end_char] == s.raw_text

        # Leaf excision (excise subordinate span_1)
        res_leaf = disentangle_and_excise(text, {"span_1"}, spans)
        assert res_leaf == "The therapeutic protocol was approved [2]."

        # Root excision (excise matrix span_2 -> promote subordinate span_1)
        res_root = disentangle_and_excise(text, {"span_2"}, spans)
        assert res_root == "The primary trial parameters were verified within range [1]."
        assert not res_root.lower().startswith("because")

    def test_level1_trailing_concessive_ast_and_root_excision(self):
        text = "The therapeutic protocol was approved [2], although Phase II trials demonstrated minor adverse reactions [1]."
        spans = parse_clause_ast(text)
        assert len(spans) == 2
        assert spans[0].is_matrix is True
        assert spans[0].citation_indices == [2]
        assert spans[1].clause_type == ClauseType.CONCESSIVE
        assert spans[1].subordinator == "although"
        assert spans[1].citation_indices == [1]

        # Leaf excision (excise trailing concessive)
        res_leaf = disentangle_and_excise(text, {"span_2"}, spans)
        assert res_leaf == "The therapeutic protocol was approved [2]."

        # Root excision (excise leading matrix -> promote trailing concessive)
        res_root = disentangle_and_excise(text, {"span_1"}, spans)
        assert "Phase II trials demonstrated minor adverse reactions [1]" in res_root
        assert not res_root.lower().startswith("although")
        assert res_root.endswith(".")

    def test_level1_boundary_all_or_none_excision(self):
        text = "Because parameters were verified [1], protocol was approved [2]."
        spans = parse_clause_ast(text)
        all_ids = {s.span_id for s in spans}

        # 100% unbacked excision yields empty string
        assert disentangle_and_excise(text, all_ids, spans) == ""

        # 0% unbacked excision returns sanitized text
        assert "parameters were verified [1]" in disentangle_and_excise(text, set(), spans)


class TestNestingLevel2NestedSubordinate:
    """Nesting Level 2: Nested subordinate clauses (Depth 2)."""

    def test_level2_concessive_plus_nested_conditional(self):
        text = (
            "Although the drug showed clinical efficacy [1] because secondary endpoints improved by 25% [2], "
            "the regulatory agency requested additional safety data [3]."
        )
        spans = parse_clause_ast(text)
        assert len(spans) >= 3
        types = [s.clause_type for s in spans]
        assert ClauseType.CONCESSIVE in types
        assert ClauseType.CONDITIONAL in types
        assert ClauseType.INDEPENDENT in types

        # Excising leaf conditional (Span 2)
        res_leaf = disentangle_and_excise(text, {"span_2"}, spans)
        assert "secondary endpoints improved" not in res_leaf
        assert "clinical efficacy [1]" in res_leaf
        assert "additional safety data [3]" in res_leaf

        # Excising middle concessive (Span 1)
        res_mid = disentangle_and_excise(text, {"span_1"}, spans)
        assert "clinical efficacy" not in res_mid
        assert "Secondary endpoints improved by 25% [2]" in res_mid or "secondary endpoints" in res_mid
        assert "additional safety data [3]" in res_mid

        # Excising root matrix (Span 3) -> promote subordinate clauses
        res_root = disentangle_and_excise(text, {"span_3"}, spans)
        assert "additional safety data" not in res_root
        assert not res_root.lower().startswith("although")
        assert "The drug showed clinical efficacy [1]" in res_root or "drug showed clinical efficacy [1]" in res_root

    def test_level2_matrix_temporal_relative(self):
        text = (
            "The financial institution expanded credit facilities [1], "
            "when liquidity ratios stabilized in Q3 [2], "
            "which satisfied regulatory capital requirements [3]."
        )
        spans = parse_clause_ast(text)
        assert len(spans) == 3
        assert spans[0].is_matrix is True
        assert spans[1].clause_type == ClauseType.TEMPORAL
        assert spans[2].clause_type == ClauseType.RELATIVE

        # Leaf excision (Span 3 relative)
        res_leaf = disentangle_and_excise(text, {"span_3"}, spans)
        assert "regulatory capital requirements" not in res_leaf
        assert "expanded credit facilities [1]" in res_leaf
        assert "liquidity ratios stabilized in Q3 [2]" in res_leaf

        # Middle excision (Span 2 temporal)
        res_mid = disentangle_and_excise(text, {"span_2"}, spans)
        assert "liquidity ratios stabilized" not in res_mid
        assert "expanded credit facilities [1]" in res_mid
        assert "satisfied regulatory capital requirements [3]" in res_mid


class TestNestingLevel3TripleNested:
    """Nesting Level 3: Triple nested clauses (Depth 3)."""

    def test_level3_concessive_conditional_matrix_relative(self):
        text = (
            "Although Phase 3 trials demonstrated an 85% response rate [1], "
            "if biomarker levels exceed 5.0 ng/mL [2], "
            "dosage must be tapered immediately [3], "
            "which prevents renal tubular necrosis [4]."
        )
        spans = parse_clause_ast(text)
        assert len(spans) == 4
        nesting_levels = [s.nesting_level for s in spans]
        assert max(nesting_levels) >= 3

        # Excise leaf relative (Span 4)
        res_leaf = disentangle_and_excise(text, {"span_4"}, spans)
        assert "renal tubular necrosis" not in res_leaf
        assert "85% response rate [1]" in res_leaf
        assert "biomarker levels exceed 5.0 ng/mL [2]" in res_leaf
        assert "dosage must be tapered immediately [3]" in res_leaf

        # Excise middle conditional (Span 2)
        res_mid = disentangle_and_excise(text, {"span_2"}, spans)
        assert "biomarker levels exceed" not in res_mid
        assert "85% response rate [1]" in res_mid
        assert "dosage must be tapered immediately [3]" in res_mid
        assert "prevents renal tubular necrosis [4]" in res_mid

        # Excise root matrix (Span 3)
        res_root = disentangle_and_excise(text, {"span_3"}, spans)
        assert "dosage must be tapered immediately" not in res_root
        assert "85% response rate [1]" in res_root
        assert not res_root.lower().startswith("although")

        # Multi-branch excision: excise lead (Span 1) and leaf (Span 4)
        res_multi = disentangle_and_excise(text, {"span_1", "span_4"}, spans)
        assert "85% response rate" not in res_multi
        assert "renal tubular necrosis" not in res_multi
        assert "If biomarker levels exceed 5.0 ng/mL [2]" in res_multi or "biomarker levels exceed 5.0 ng/mL [2]" in res_multi
        assert "dosage must be tapered immediately [3]" in res_multi


class TestNestingLevel4QuadrupleNested:
    """Nesting Level 4: Quadruple nested clauses & attribution hierarchies (Depth 4)."""

    def test_level4_concessive_temporal_conditional_matrix_relative(self):
        text = (
            "Notwithstanding that retrospective analyses indicated a 68% progression-free survival [1], "
            "while patient cohorts were monitored across three clinical sites [2], "
            "if hepatic enzyme levels remain within normal limits [3], "
            "clinicians may initiate maintenance therapy [4], "
            "which reduces 5-year recurrence rates [5]."
        )
        spans = parse_clause_ast(text)
        assert len(spans) == 5
        nesting_levels = [s.nesting_level for s in spans]
        assert max(nesting_levels) >= 4

        # Excise intermediate temporal & conditional (Spans 2 and 3)
        res_mid = disentangle_and_excise(text, {"span_2", "span_3"}, spans)
        assert "patient cohorts were monitored" not in res_mid
        assert "hepatic enzyme levels" not in res_mid
        assert "68% progression-free survival [1]" in res_mid
        assert "clinicians may initiate maintenance therapy [4]" in res_mid
        assert "reduces 5-year recurrence rates [5]" in res_mid

        # Excise root matrix (Span 4) -> promote lead concessive
        res_root = disentangle_and_excise(text, {"span_4"}, spans)
        assert "maintenance therapy" not in res_root
        assert not res_root.lower().startswith("notwithstanding that")
        assert "Retrospective analyses indicated a 68% progression-free survival [1]" in res_root

        # Deepest leaf retention: excise Spans 1, 2, 4, 5, retain only Span 3
        res_single = disentangle_and_excise(text, {"span_1", "span_2", "span_4", "span_5"}, spans)
        assert "progression-free survival" not in res_single
        assert "maintenance therapy" not in res_single
        assert not res_single.lower().startswith("if")
        assert "Hepatic enzyme levels remain within normal limits [3]." in res_single


class TestNestingLevel5QuintuplePeriodic:
    """Nesting Level 5: High-density periodic entanglement with 7 clauses (Depth 5)."""

    def test_level5_seven_clause_periodic_entanglement_surgical_repair(self):
        text = (
            "Having completed the primary pharmacokinetic evaluation [1], "
            "notwithstanding that baseline biomarker concentrations remained stable across all cohorts [2], "
            "if renal clearance exceeds 60 mL/min [3], "
            "while cardiac telemetry is monitored continuously [4], "
            "the committee approved the expanded protocol [5], "
            "which enables multi-center trials [6], "
            "but unauthorized dosing escalation remains prohibited [7]."
        )
        spans = parse_clause_ast(text)
        assert len(spans) >= 6
        nesting_levels = [s.nesting_level for s in spans]
        assert max(nesting_levels) >= 5

        # Excise 2 poison clauses (Span 3 conditional and Span 7 coordinate)
        unbacked = {"span_3", "span_7"}
        res = disentangle_and_excise(text, unbacked, spans)

        assert "renal clearance exceeds 60" not in res
        assert "unauthorized dosing escalation" not in res
        assert "pharmacokinetic evaluation [1]" in res
        assert "baseline biomarker concentrations remained stable across all cohorts [2]" in res
        assert "cardiac telemetry is monitored continuously [4]" in res
        assert "committee approved the expanded protocol [5]" in res
        assert "enables multi-center trials [6]" in res
        assert not res.startswith(("and", "but", "or", "which"))
        assert res.endswith(".")

        # Excise 4 spans (1, 2, 3, 7) retaining 4, 5, 6
        res_4spans = disentangle_and_excise(text, {"span_1", "span_2", "span_3", "span_7"}, spans)
        assert "pharmacokinetic" not in res_4spans
        assert "telemetry is monitored continuously [4]" in res_4spans
        assert "committee approved the expanded protocol [5]" in res_4spans
        assert "enables multi-center trials [6]" in res_4spans


class TestExcisionGrammarAndCommaSplices:
    """Detailed tests for punctuation normalization, comma splice prevention, and connector stripping."""

    def test_middle_relative_clause_subject_verb_rejoining(self):
        """Excising middle relative clause joins subject noun phrase and predicate verb phrase cleanly without comma."""
        text = "The experimental drug, which achieved 100% cure rates in animal models [1], is undergoing safety review [2]."
        spans = parse_clause_ast(text)
        unbacked = {"span_2"}
        cleaned = disentangle_and_excise(text, unbacked, spans)

        assert "100% cure rates" not in cleaned
        assert ",," not in cleaned
        assert ", is" not in cleaned
        assert "The experimental drug is undergoing safety review [2]." in cleaned

    def test_middle_relative_past_tense_verb_rejoining(self):
        """Excising middle relative clause joins past tense verb phrase without comma splice."""
        text = "The therapy, which was approved by the FDA in 2023 [1], reduced hospital readmission rates by 35% [2]."
        spans = parse_clause_ast(text)
        cleaned = disentangle_and_excise(text, {"span_2"}, spans)

        assert ", reduced" not in cleaned
        assert "The therapy reduced hospital readmission rates by 35% [2]." in cleaned

    def test_multi_sentence_ast_parsing_and_excision(self):
        """Multi-sentence inputs maintain correct boundaries and do not inject commas between sentences."""
        text = (
            "Although trial A succeeded [1], drug A was approved [1]. "
            "If trial B failed [2], drug B was rejected [2]."
        )
        spans = parse_clause_ast(text)
        assert len(spans) >= 4

        # Excise span 3 (If trial B failed [2])
        cleaned = disentangle_and_excise(text, {"span_3"}, spans)
        assert "If trial B failed" not in cleaned
        assert "trial A succeeded [1]" in cleaned
        assert "drug A was approved [1]." in cleaned
        assert "Drug B was rejected [2]." in cleaned or "drug B was rejected [2]." in cleaned
        assert not re.search(r"\.\s*,", cleaned)
