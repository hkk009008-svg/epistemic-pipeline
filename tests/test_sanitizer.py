"""Tests for pipeline.sanitizer -- route_prompt() and sanitize_output().

All functions under test are fully deterministic (no LLM calls).
"""
from __future__ import annotations

import pytest

from pipeline.sanitizer import route_prompt, sanitize_output


# ===================================================================
# route_prompt()
# ===================================================================


class TestRoutePromptAdvice:
    """Advice-detection flag."""

    @pytest.mark.parametrize(
        "prompt",
        [
            "What should I do about my taxes?",
            "Should I hire a lawyer?",
            "Can you recommend a good approach?",
            "What are the steps to file?",
            "What is the best way to appeal?",
            "Would it help to contact them first?",
            "What are my options here?",
            "How do I submit the form?",
            "How can I expedite this?",
            "Any tips for negotiating?",
            "Please help me understand this process.",
        ],
    )
    def test_advice_detected(self, prompt: str):
        flags = route_prompt(prompt)
        assert flags["advice_requested"] is True

    @pytest.mark.parametrize(
        "prompt",
        [
            "What is the capital of France?",
            "Explain quantum entanglement.",
            "Define photosynthesis.",
        ],
    )
    def test_advice_not_detected(self, prompt: str):
        flags = route_prompt(prompt)
        assert flags["advice_requested"] is False


class TestRoutePromptPercent:
    """Percent/statistical-request flag."""

    @pytest.mark.parametrize(
        "prompt",
        [
            "What percent of applications are approved?",
            "What is the approval rate?",
            "What are the odds of success?",
            "How many people file each year?",
            "How often does this happen?",
            "What is the probability of winning?",
            "What fraction are denied?",
            "What proportion is affected?",
            "What percentage of cases succeed?",
            "What typically happens after filing?",
        ],
    )
    def test_percent_detected(self, prompt: str):
        flags = route_prompt(prompt)
        assert flags["percent_requested"] is True

    def test_percent_not_detected(self):
        flags = route_prompt("Explain how gravity works.")
        assert flags["percent_requested"] is False


class TestRoutePromptLegal:
    """Legal-mode flag."""

    @pytest.mark.parametrize(
        "prompt",
        [
            "Is this legal in the US?",
            "What does the law say about importing goods?",
            "Is there a regulation on this?",
            "Is it illegal to do this?",
            "What IRS rules apply?",
            "SEC compliance requirements for startups.",
            "Is there a statute covering this?",
            "Any ordinance banning this?",
            "Is this prohibited under federal law?",
            "Export regulation for dual-use tech.",
        ],
    )
    def test_legal_detected(self, prompt: str):
        flags = route_prompt(prompt)
        assert flags["legal_mode"] is True

    def test_legal_not_detected(self):
        flags = route_prompt("What color is the sky?")
        assert flags["legal_mode"] is False


class TestRoutePromptJurisdiction:
    """Jurisdiction-present flag."""

    @pytest.mark.parametrize(
        "prompt",
        [
            "What are the tax rules in the US?",
            "UK regulations on data privacy.",
            "EU cookie law details.",
            "Federal tax code overview.",
            "State of California labor laws.",
            "United States import duties.",
            "Rules in Texas for home businesses.",
            "New York zoning requirements.",
            "Florida homestead exemption.",
            "Is this allowed in Germany?",
            "What about India's GST?",
            "Alaska fishing permits.",
        ],
    )
    def test_jurisdiction_detected(self, prompt: str):
        flags = route_prompt(prompt)
        assert flags["jurisdiction_present"] is True

    def test_jurisdiction_not_detected(self):
        flags = route_prompt("Explain thermodynamics.")
        assert flags["jurisdiction_present"] is False


class TestRoutePromptFutureYear:
    """Future-year flag."""

    @pytest.mark.parametrize(
        "prompt",
        [
            "What will taxes look like in 2035?",
            "Predict the economy for 2050.",
            "By the year 2100 what happens?",
            "Regulations expected in 2030.",
        ],
    )
    def test_future_year_detected(self, prompt: str):
        flags = route_prompt(prompt)
        assert flags["future_year"] is True

    @pytest.mark.parametrize(
        "prompt",
        [
            "What happened in 2020?",
            "Tell me about the year 2024.",
            "History of the 1990s.",
        ],
    )
    def test_future_year_not_detected(self, prompt: str):
        flags = route_prompt(prompt)
        assert flags["future_year"] is False


class TestRoutePromptCombined:
    """Multiple flags can fire simultaneously."""

    def test_legal_and_jurisdiction(self):
        flags = route_prompt("Is it legal to import goods into the US?")
        assert flags["legal_mode"] is True
        assert flags["jurisdiction_present"] is True

    def test_advice_and_percent(self):
        flags = route_prompt("What should I do if the approval rate is low?")
        assert flags["advice_requested"] is True
        assert flags["percent_requested"] is True

    def test_all_flags(self):
        flags = route_prompt(
            "What should I do about the legal compliance rate in the US by 2040?"
        )
        assert flags["advice_requested"] is True
        assert flags["percent_requested"] is True
        assert flags["legal_mode"] is True
        assert flags["jurisdiction_present"] is True
        assert flags["future_year"] is True

    def test_nothing_detected(self):
        flags = route_prompt("Explain how photosynthesis works.")
        assert flags == {
            "advice_requested": False,
            "percent_requested": False,
            "legal_mode": False,
            "jurisdiction_present": False,
            "future_year": False,
            "current_events": False,
            "comparative": False,
        }


# ===================================================================
# sanitize_output()
# ===================================================================


class TestSanitizeOutputBannedEvidence:
    """Banned evidence phrases (without citations) should be stripped."""

    @pytest.mark.parametrize(
        "phrase",
        [
            "studies suggest",
            "research shows",
            "data indicates",
            "generally",
            "often",
            "typically",
            "commonly",
            "usually",
        ],
    )
    def test_banned_phrase_removed(self, phrase: str, flags_all_false: dict):
        text = f"The answer is that {phrase} this is the case."
        result = sanitize_output(text, flags_all_false)
        assert phrase not in result.lower()

    def test_banned_phrase_with_parenthetical_citation_kept(self, flags_all_false: dict):
        text = "Research shows (Smith 2022) that water is wet."
        result = sanitize_output(text, flags_all_false)
        # The phrase "Research shows" should remain because it is followed by a citation
        assert "Smith 2022" in result

    def test_banned_phrase_with_bracket_citation_kept(self, flags_all_false: dict):
        text = "Studies suggest [CDC 2023] that vaccines are effective."
        result = sanitize_output(text, flags_all_false)
        assert "CDC 2023" in result


class TestSanitizeOutputBarePercent:
    """Bare percentage claims should be replaced with Unknown (Actionable)."""

    @pytest.mark.parametrize(
        "text",
        [
            "Roughly 60 percent of cases succeed.",
            "An estimated 70 percent of requests are granted.",
            "50 percent of claims are valid.",
        ],
    )
    def test_bare_percent_replaced(self, text: str, flags_all_false: dict):
        result = sanitize_output(text, flags_all_false)
        assert "Unknown(Actionable)" in result
        assert "percent" not in result.lower()

    def test_replacement_text_correct(self, flags_all_false: dict):
        text = "Roughly 60 percent of cases succeed."
        result = sanitize_output(text, flags_all_false)
        assert "No authoritative dataset available for this figure" in result

    @pytest.mark.parametrize(
        "text",
        [
            "About 45% of people agree.",
            "Approximately 12.5% is lost.",
            "Around 80% of applicants qualify.",
            "Nearly 90% pass the exam.",
            "Close to 33% are denied.",
            "50% of claims are valid.",
        ],
    )
    def test_percent_symbol_also_replaced(self, text: str, flags_all_false: dict):
        """Bare percentages using the % symbol are also replaced by the sanitizer."""
        result = sanitize_output(text, flags_all_false)
        assert "Unknown(Actionable)" in result


class TestSanitizeOutputOutcomePromise:
    """Outcome-promise phrases should be stripped."""

    @pytest.mark.parametrize(
        "phrase",
        [
            "will improve",
            "will reduce",
            "will increase",
            "could help",
            "could assist",
            "may improve",
            "may help",
            "could potentially",
        ],
    )
    def test_outcome_promise_removed(self, phrase: str, flags_all_false: dict):
        text = f"This action {phrase} your situation."
        result = sanitize_output(text, flags_all_false)
        assert phrase not in result.lower()


class TestSanitizeOutputWhitespace:
    """Double spaces and trailing whitespace should be cleaned up."""

    def test_double_spaces_collapsed(self, flags_all_false: dict):
        text = "Hello   world   today."
        result = sanitize_output(text, flags_all_false)
        assert "  " not in result

    def test_trailing_whitespace_per_line(self, flags_all_false: dict):
        text = "Line one   \nLine two   \n"
        result = sanitize_output(text, flags_all_false)
        for line in result.split("\n"):
            assert line == line.rstrip(" ")

    def test_result_is_stripped(self, flags_all_false: dict):
        text = "  some text  "
        result = sanitize_output(text, flags_all_false)
        assert result == result.strip()


class TestSanitizeOutputComposite:
    """Composite scenarios with multiple substitutions."""

    def test_multi_pattern_cleanup(self, flags_all_false: dict):
        text = (
            "Research shows that about 30 percent of cases succeed. "
            "This will improve your situation."
        )
        result = sanitize_output(text, flags_all_false)
        # Banned evidence phrase gone
        assert "research shows" not in result.lower()
        # Bare percent replaced (using word "percent" so regex matches)
        assert "Unknown(Actionable)" in result
        # Outcome promise removed
        assert "will improve" not in result.lower()


class TestSanitizeOutputCitedPercent:
    """Percentages with nearby citations should NOT be stripped (citation-aware sanitizer)."""

    def test_percent_with_parenthetical_citation_kept(self, flags_all_false: dict):
        text = "The approval rate is 73% (BLS 2024)."
        result = sanitize_output(text, flags_all_false)
        assert "Unknown(Actionable)" not in result
        assert "73%" in result

    def test_percent_with_bracket_citation_kept(self, flags_all_false: dict):
        text = "About 60% of cases succeed [1]."
        result = sanitize_output(text, flags_all_false)
        assert "Unknown(Actionable)" not in result
        assert "60%" in result

    def test_percent_with_distant_citation_kept(self, flags_all_false: dict):
        text = "The rate is approximately 45%, according to data from the Census Bureau (2023)."
        result = sanitize_output(text, flags_all_false)
        assert "Unknown(Actionable)" not in result

    def test_percent_without_any_citation_stripped(self, flags_all_false: dict):
        text = "About 80% of applicants qualify."
        result = sanitize_output(text, flags_all_false)
        assert "Unknown(Actionable)" in result

    def test_percent_word_with_citation_kept(self, flags_all_false: dict):
        text = "Roughly 60 percent of cases succeed (CDC 2023)."
        result = sanitize_output(text, flags_all_false)
        assert "Unknown(Actionable)" not in result

    def test_percent_word_without_citation_stripped(self, flags_all_false: dict):
        text = "Roughly 60 percent of cases succeed."
        result = sanitize_output(text, flags_all_false)
        assert "Unknown(Actionable)" in result

    def test_mixed_cited_and_bare_percents(self, flags_all_false: dict):
        """Only bare percents stripped; cited ones preserved."""
        text = (
            "The rate is 73% (BLS 2024). "
            "Meanwhile, about 50% of other claims are unverified."
        )
        result = sanitize_output(text, flags_all_false)
        # 73% has a citation — should survive
        assert "73%" in result
        # 50% has no citation — should be replaced
        assert "Unknown(Actionable)" in result


# ===================================================================
# route_prompt() -- comparative flag
# ===================================================================


class TestRoutePromptComparative:
    """Comparative-question detection flag."""

    @pytest.mark.parametrize(
        "prompt",
        [
            "Is chemotherapy or immunotherapy safer for lung cancer?",
            "Which is better, surgery or radiation?",
            "Is drug A more effective than drug B?",
            "Are generic drugs as effective as brand-name ones?",
            "Which has fewer side effects, aspirin or ibuprofen?",
            "Is living in the city versus the suburbs healthier?",
            "Does vaccine X cause more harm than vaccine Y?",
            "Which is the safest option for treatment?",
            "Is homeopathy as reliable as conventional medicine?",
            "Is option A or option B better for recovery?",
        ],
    )
    def test_comparative_detected(self, prompt: str):
        flags = route_prompt(prompt)
        assert flags["comparative"] is True

    @pytest.mark.parametrize(
        "prompt",
        [
            "What is the capital of France?",
            "Explain quantum entanglement.",
            "Define photosynthesis.",
            "How does gravity work?",
        ],
    )
    def test_comparative_not_detected(self, prompt: str):
        flags = route_prompt(prompt)
        assert flags["comparative"] is False


# ===================================================================
# sanitize_output() -- tier gating
# ===================================================================


class TestSanitizeTierGating:
    """Tier parameter controls which sanitizer rules fire."""

    def test_strict_strips_banned_evidence(self, flags_all_false: dict):
        text = "Research shows that water is wet."
        result = sanitize_output(text, flags_all_false, tier="strict")
        assert "research shows" not in result.lower()

    def test_standard_strips_banned_evidence(self, flags_all_false: dict):
        text = "Research shows that water is wet."
        result = sanitize_output(text, flags_all_false, tier="standard")
        assert "research shows" not in result.lower()

    def test_light_preserves_banned_evidence(self, flags_all_false: dict):
        text = "Research shows that water is wet."
        result = sanitize_output(text, flags_all_false, tier="light")
        assert "research shows" in result.lower()

    def test_strict_strips_typicality(self, flags_all_false: dict):
        text = "People generally prefer this approach."
        result = sanitize_output(text, flags_all_false, tier="strict")
        assert "generally" not in result.lower()

    def test_light_preserves_typicality(self, flags_all_false: dict):
        text = "People generally prefer this approach."
        result = sanitize_output(text, flags_all_false, tier="light")
        assert "generally" in result.lower()

    def test_strict_strips_bare_percent(self, flags_all_false: dict):
        text = "About 50% of cases succeed."
        result = sanitize_output(text, flags_all_false, tier="strict")
        assert "Unknown(Actionable)" in result

    def test_standard_preserves_bare_percent_without_flag(self, flags_all_false: dict):
        """Standard tier skips bare-percent stripping unless percent_requested."""
        text = "About 50% of cases succeed."
        result = sanitize_output(text, flags_all_false, tier="standard")
        assert "50%" in result

    def test_standard_strips_bare_percent_with_flag(self):
        """Standard tier strips bare percents when percent_requested is True."""
        flags = {
            "advice_requested": False, "percent_requested": True,
            "legal_mode": False, "jurisdiction_present": False,
            "future_year": False, "current_events": False, "comparative": False,
        }
        text = "About 50% of cases succeed."
        result = sanitize_output(text, flags, tier="standard")
        assert "Unknown(Actionable)" in result

    def test_light_preserves_bare_percent(self, flags_all_false: dict):
        text = "About 50% of cases succeed."
        result = sanitize_output(text, flags_all_false, tier="light")
        assert "50%" in result

    def test_strict_strips_outcome_promise(self, flags_all_false: dict):
        text = "This will improve your situation."
        result = sanitize_output(text, flags_all_false, tier="strict")
        assert "will improve" not in result.lower()

    def test_light_preserves_outcome_promise(self, flags_all_false: dict):
        text = "This will improve your situation."
        result = sanitize_output(text, flags_all_false, tier="light")
        assert "will improve" in result.lower()

    def test_light_still_cleans_whitespace(self, flags_all_false: dict):
        """Even light tier cleans up double spaces."""
        text = "Hello   world   today."
        result = sanitize_output(text, flags_all_false, tier="light")
        assert "  " not in result


# ===================================================================
# Regression: temporal detection (year regex hardening)
# ===================================================================

class TestFutureYearRegression:
    """Verify runtime year comparison catches all future years without gaps."""

    def test_year_2028_detected_as_future(self):
        """Regression: old regex had a gap for 2028-2029."""
        flags = route_prompt("What happens in 2028?")
        assert flags["future_year"] is True

    def test_year_2029_detected_as_future(self):
        """Regression: old regex had a gap for 2028-2029."""
        flags = route_prompt("Predictions for 2029.")
        assert flags["future_year"] is True

    def test_year_2100_detected_as_future(self):
        """Years in the 2100s should be detected as future."""
        flags = route_prompt("By the year 2100 what happens?")
        assert flags["future_year"] is True

    def test_zip_code_not_detected_as_future(self):
        """ZIP codes like 90210 should NOT be detected as a year."""
        flags = route_prompt("What's the weather in 90210?")
        assert flags["future_year"] is False

    def test_port_number_not_detected_as_future(self):
        """Port numbers like 8080 should NOT be detected as a year."""
        flags = route_prompt("Connect to port 8080 for the API.")
        assert flags["future_year"] is False

    def test_current_year_not_future(self):
        """The current year should not be flagged as future."""
        from datetime import date
        year = date.today().year
        flags = route_prompt(f"What is happening in {year}?")
        assert flags["future_year"] is False

    def test_past_year_not_future(self):
        flags = route_prompt("Events of 2020.")
        assert flags["future_year"] is False


class TestStaleDateRegression:
    """Verify stale date detection uses runtime comparison, not regex math."""

    def test_stale_date_detected(self):
        """A date before current year should be flagged as stale."""
        flags = {
            "advice_requested": False, "percent_requested": False,
            "legal_mode": False, "jurisdiction_present": False,
            "future_year": False, "current_events": True, "comparative": False,
        }
        text = "As of January 2020, the rate was 5%."
        result = sanitize_output(text, flags, tier="strict")
        assert "[Stale" in result

    def test_current_year_not_stale(self):
        """Current year should NOT be flagged as stale."""
        from datetime import date
        year = date.today().year
        flags = {
            "advice_requested": False, "percent_requested": False,
            "legal_mode": False, "jurisdiction_present": False,
            "future_year": False, "current_events": True, "comparative": False,
        }
        text = f"As of January {year}, the rate was 5%."
        result = sanitize_output(text, flags, tier="strict")
        assert "[Stale" not in result

    def test_stale_not_applied_without_current_events(self):
        """Stale detection only triggers when current_events flag is set."""
        flags = {
            "advice_requested": False, "percent_requested": False,
            "legal_mode": False, "jurisdiction_present": False,
            "future_year": False, "current_events": False, "comparative": False,
        }
        text = "As of January 2020, the rate was 5%."
        result = sanitize_output(text, flags, tier="strict")
        assert "[Stale" not in result


class TestRoutePromptCurrentEvents:
    """Current-events routing should not fire on everyday 'new'/'now'."""

    @pytest.mark.parametrize(
        "prompt",
        [
            "What is the latest WHO guidance?",
            "What is the current president of France?",
            "Who is the prime minister of Japan?",
            "Any breaking news on the treaty?",
        ],
    )
    def test_current_events_detected(self, prompt: str):
        assert route_prompt(prompt)["current_events"] is True

    @pytest.mark.parametrize(
        "prompt",
        [
            "Define a new data structure.",
            "Explain how a new LLC works.",
            "New York zoning requirements.",
            "Please help us understand photosynthesis.",
        ],
    )
    def test_current_events_not_detected(self, prompt: str):
        assert route_prompt(prompt)["current_events"] is False


class TestRoutePromptUsPronoun:
    """Case-insensitive \\bUS\\b must not treat the pronoun 'us' as a jurisdiction."""

    def test_pronoun_us_is_not_jurisdiction(self):
        flags = route_prompt("Tell us about gravity.")
        assert flags["jurisdiction_present"] is False

    def test_uppercase_us_is_jurisdiction(self):
        flags = route_prompt("What are the tax rules in the US?")
        assert flags["jurisdiction_present"] is True


class TestSanitizeTypicalityDoesNotShieldPercents:
    """Sanitizer markers must not count as citations that protect bare stats."""

    def test_percent_after_typicality_is_stripped(self, flags_all_false: dict):
        text = "About 50% typically succeed."
        result = sanitize_output(text, flags_all_false)
        assert "50%" not in result
        assert "Unknown(Actionable)" in result

    def test_percent_before_generally_is_stripped(self, flags_all_false: dict):
        text = "The approval rate is 73% generally."
        result = sanitize_output(text, flags_all_false)
        assert "73%" not in result
        assert "Unknown(Actionable)" in result
