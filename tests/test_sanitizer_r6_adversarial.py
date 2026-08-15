"""Adversarial stress and verification tests for Requirement R6 in pipeline/sanitizer.py.

Covers:
- Boundary whitespace padding and token-gluing prevention
- Nested and adjacent punctuation handling around bare statistics
- Combinatorial Subject x Adverb x Verb authority vocabulary expansion
- Case-insensitivity stress testing
- Immediate citation preservation vs ungrounded assertion suppression
- Benign context preservation (false positive avoidance)
"""
from __future__ import annotations

import itertools
import re
import string
import pytest

from pipeline.sanitizer import (
    _BANNED_EVIDENCE_RE,
    _BARE_PERCENT_RE,
    _clean_grammar_and_punctuation,
    _replace_bare_percents,
    route_prompt,
    sanitize_output,
)

FLAGS_DEFAULT = {
    "advice_requested": False,
    "percent_requested": False,
    "legal_mode": False,
    "jurisdiction_present": False,
    "future_year": False,
    "current_events": False,
    "comparative": False,
}


# =====================================================================
# 1. WHITESPACE BOUNDARIES & TOKEN GLUING PREVENTION
# =====================================================================

class TestSanitizerWhitespaceBoundaries:
    """Stress-test whitespace handling and token gluing across various boundaries."""

    @pytest.mark.parametrize(
        "punct",
        [c for c in string.punctuation if c not in "([{\"'`_"],
    )
    def test_preceding_punctuation_no_token_gluing(self, punct: str):
        """Non-enclosure punctuation preceding bare percent inserts spacing and avoids gluing."""
        text = f"Measurement{punct}85% was recorded."
        res = sanitize_output(text, FLAGS_DEFAULT, tier="strict")
        # Ensure no token gluing between Measurement, punct, and Unknown
        assert "MeasurementUnknown" not in res
        assert "Unknown(Actionable)" in res

    def test_preceding_underscore_identifier_behavior(self):
        """In regex, underscore is a word character (\\w), so snake_case identifiers like 'metric_85%' are preserved."""
        text = "Variable metric_85% was recorded."
        res = sanitize_output(text, FLAGS_DEFAULT, tier="strict")
        assert "metric_85%" in res


    @pytest.mark.parametrize(
        "punct",
        [c for c in string.punctuation if c not in ".,;:!?)]}\"'`"],
    )
    def test_succeeding_punctuation_no_token_gluing(self, punct: str):
        """Non-terminal punctuation succeeding bare percent inserts spacing and avoids gluing."""
        text = f"The rate was 85%{punct}confirmed by none."
        res = sanitize_output(text, FLAGS_DEFAULT, tier="strict")
        assert "Unknown(Actionable)confirmed" not in res
        assert "figureconfirmed" not in res
        assert "Unknown(Actionable)" in res

    @pytest.mark.parametrize(
        "suffix_word",
        [
            "improvement", "increase", "reduction", "growth", "drop",
            "jump", "decline", "accuracy", "efficiency", "latency",
        ],
    )
    def test_bare_percent_with_attached_word_suffix(self, suffix_word: str):
        """Bare percent with immediately glued word suffix separates cleanly without token gluing."""
        text = f"We noticed a 45%{suffix_word} across all benchmarks."
        res = sanitize_output(text, FLAGS_DEFAULT, tier="strict")
        # Ensure 'figure' is not glued to suffix word
        assert f"figure{suffix_word}" not in res
        assert f"Actionable){suffix_word}" not in res
        assert suffix_word in res
        assert "Unknown(Actionable)" in res

    @pytest.mark.parametrize(
        "enclosure_pair",
        [
            ("(", ")"),
            ("[", "]"),
            ("{", "}"),
            ('"', '"'),
            ("'", "'"),
            ("`", "`"),
        ],
    )
    def test_bare_percent_inside_enclosures(self, enclosure_pair: tuple[str, str]):
        """Enclosed bare percents preserve enclosure structure cleanly."""
        open_c, close_c = enclosure_pair
        text = f"Results {open_c}roughly 30%{close_c} are preliminary."
        res = sanitize_output(text, FLAGS_DEFAULT, tier="strict")
        assert "30%" not in res
        assert "Unknown(Actionable)" in res
        assert open_c in res and close_c in res

    def test_nested_parentheses_and_brackets(self):
        """Deeply nested enclosures around bare statistics format cleanly."""
        text = "Findings ((75%)) were noted."
        res = sanitize_output(text, FLAGS_DEFAULT, tier="strict")
        assert "75%" not in res
        assert "((Unknown(Actionable): No authoritative dataset available for this figure))" in res

    def test_adjacent_multi_percentages(self):
        """Adjacent percentages with slashes or hyphens separate cleanly without malformed glue."""
        text = "Ratios ranged from 20%/40% to 60%-80% across trials."
        res = sanitize_output(text, FLAGS_DEFAULT, tier="strict")
        assert "20%" not in res
        assert "40%" not in res
        assert "60%" not in res
        assert "80%" not in res
        assert res.count("Unknown(Actionable)") == 4

    def test_whitespace_variations(self):
        """Tabs, multiple spaces, and newlines around bare percents normalize properly."""
        text = "Alpha:\t\t50%\nBeta:   60%   \nGamma: 70%."
        res = sanitize_output(text, FLAGS_DEFAULT, tier="strict")
        assert "  " not in res
        assert "50%" not in res
        assert "60%" not in res
        assert "70%" not in res
        assert res.count("Unknown(Actionable)") == 3


# =====================================================================
# 2. AUTHORITY VOCABULARY MATRIX & COMBINATORIAL EXPANSION
# =====================================================================

class TestSanitizerAuthorityMatrix:
    """Combinatorial testing for expanded authority phrases in _BANNED_EVIDENCE_RE."""

    SUBJECTS = [
        "clinical evidence", "scientific evidence", "empirical evidence",
        "experimental evidence", "medical evidence", "evidence",
        "medical consensus", "scientific consensus", "expert consensus",
        "market consensus", "consensus",
        "published papers", "published literature", "published studies",
        "published data", "published reports",
        "scientific studies", "clinical studies", "scientific trials", "clinical trials",
        "studies", "study", "research", "researchers", "data", "literature",
        "experts", "scientists", "clinicians", "doctors", "specialists",
    ]

    ADVERBS = [
        "", "clearly", "strongly", "consistently", "definitely",
        "unequivocally", "directly", "robustly",
    ]

    VERBS = [
        "demonstrates", "demonstrated", "shows", "showed", "shown",
        "indicates", "indicated", "suggests", "suggested",
        "confirms", "confirmed", "proves", "proved", "proven",
        "establishes", "established", "reveals", "revealed",
        "supports", "supported",
    ]

    def test_full_combinatorial_matrix_regex_matches(self):
        """All 4,900+ combinations of subjects x adverbs x verbs match _BANNED_EVIDENCE_RE."""
        missed = []
        for s in self.SUBJECTS:
            for adv in self.ADVERBS:
                for v in self.VERBS:
                    phrase = f"{s} {adv} {v}" if adv else f"{s} {v}"
                    if not _BANNED_EVIDENCE_RE.search(phrase):
                        missed.append(phrase)
        assert not missed, f"Missed {len(missed)} combinations in _BANNED_EVIDENCE_RE: {missed[:10]}"

    @pytest.mark.parametrize("subject", SUBJECTS)
    def test_representative_subject_sanitization(self, subject: str):
        """Representative subject with verb is stripped when uncited."""
        text = f"{subject.capitalize()} shows that the method is effective."
        res = sanitize_output(text, FLAGS_DEFAULT, tier="strict")
        assert "shows that the method" not in res.lower()
        assert "[Unverified generalization removed]" in res

    @pytest.mark.parametrize("adverb", [a for a in ADVERBS if a])
    def test_adverb_modifiers_sanitization(self, adverb: str):
        """Authority phrases with modifying adverbs are stripped cleanly."""
        text = f"Clinical evidence {adverb} demonstrates high efficacy."
        res = sanitize_output(text, FLAGS_DEFAULT, tier="strict")
        assert f"{adverb} demonstrates" not in res.lower()
        assert "[Unverified generalization removed]" in res


# =====================================================================
# 3. CASING & PUNCTUATION VARIANTS
# =====================================================================

class TestSanitizerCasingAndPunctuation:
    """Stress test casing variations and trailing punctuation on authority phrases."""

    @pytest.mark.parametrize(
        "case_variant",
        [
            "CLINICAL EVIDENCE DEMONSTRATES",
            "Clinical Evidence Demonstrates",
            "clinical evidence demonstrates",
            "cLiNiCaL eViDeNcE dEmOnStRaTeS",
            "MEDICAL CONSENSUS CLEARLY SHOWS",
            "Medical Consensus Clearly Shows",
            "PUBLISHED PAPERS CONSISTENTLY CONFIRM",
            "Published Papers Consistently Confirm",
        ],
    )
    def test_authority_case_insensitivity(self, case_variant: str):
        """Authority phrase detection is strictly case-insensitive."""
        text = f"{case_variant} the primary endpoint."
        res = sanitize_output(text, FLAGS_DEFAULT, tier="strict")
        assert case_variant.lower() not in res.lower()
        assert "[Unverified generalization removed]" in res

    @pytest.mark.parametrize(
        "punct",
        [",", ":", ";", "--", " —"],
    )
    def test_authority_phrase_followed_by_punctuation(self, punct: str):
        """Authority phrase followed by punctuation before clause is stripped without breaking grammar."""
        text = f"Clinical evidence demonstrates{punct} the drug works."
        res = sanitize_output(text, FLAGS_DEFAULT, tier="strict")
        assert "clinical evidence demonstrates" not in res.lower()
        assert "[Unverified generalization removed]" in res


# =====================================================================
# 4. CITATION IMMEDIATE QUALIFICATION VS UNGROUNDED ASSERTION
# =====================================================================

class TestSanitizerCitationPreservation:
    """Ensure properly cited authority phrases and statistics are preserved."""

    def test_parenthetical_citation_immediately_following_preserved(self):
        """Immediate parenthetical citation preserves authority statement."""
        text = "Clinical evidence demonstrates (Lancet 2023) high therapeutic value."
        res = sanitize_output(text, FLAGS_DEFAULT, tier="strict")
        assert "clinical evidence demonstrates" in res.lower()
        assert "Lancet 2023" in res

    def test_bracket_citation_immediately_following_preserved(self):
        """Immediate bracket citation preserves authority statement."""
        text = "Medical consensus shows [1] widespread adoption."
        res = sanitize_output(text, FLAGS_DEFAULT, tier="strict")
        assert "medical consensus shows" in res.lower()
        assert "[1]" in res

    def test_whitespace_before_citation_preserved(self):
        """Authority phrase followed by whitespace then citation is preserved."""
        text = "Published papers confirm   (Nature 2022)   the hypothesis."
        res = sanitize_output(text, FLAGS_DEFAULT, tier="strict")
        assert "published papers confirm" in res.lower()
        assert "Nature 2022" in res


# =====================================================================
# 5. BENIGN CONTEXT PRESERVATION (FALSE POSITIVE DEFENSE)
# =====================================================================

class TestSanitizerFalsePositiveDefense:
    """Verify that benign uses of vocabulary words without ungrounded claims are untouched."""

    @pytest.mark.parametrize(
        "benign_text",
        [
            "The international consensus conference was held in Geneva.",
            "We need empirical evidence before reaching any policy decision.",
            "A prospective clinical study enrolled 500 patients.",
            "Doctors and clinicians gathered at the annual symposium.",
            "The archived published reports are stored in the library.",
            "Market consensus remains cautious ahead of the earnings call.",
        ],
    )
    def test_benign_text_unmodified(self, benign_text: str):
        """Benign sentences lacking assertion verbs are preserved without unverified generalization markers."""
        res = sanitize_output(benign_text, FLAGS_DEFAULT, tier="strict")
        assert "[Unverified generalization removed]" not in res
        assert benign_text == res
