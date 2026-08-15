"""Tests for pipeline/source_match.py — deterministic source-matching post-processor."""
from __future__ import annotations

from pipeline.models import ClaimEntry, SearchSource
from pipeline.source_match import (
    _extract_keywords,
    _extract_numbers,
    _find_nli_support,
    _get_citation_segments,
    build_source_number_sets,
    filter_findings_with_sources,
    recategorize_with_sources,
    run_preflight_scan,
    verify_citation_grounding,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _src(title: str, snippet: str, url: str = "https://example.com") -> SearchSource:
    return SearchSource(title=title, url=url, snippet=snippet, score=0.5)

def _claim(claim: str, category: str = "Unsupported", justification: str = "") -> ClaimEntry:
    return ClaimEntry(claim=claim, category=category, justification=justification)


# ---------------------------------------------------------------------------
# _extract_keywords
# ---------------------------------------------------------------------------

class TestExtractKeywords:
    def test_basic(self):
        kw = _extract_keywords("BTS is a South Korean boy band")
        assert "bts" in kw
        assert "south" in kw
        assert "korean" in kw
        assert "boy" in kw
        assert "band" in kw
        # stop words removed
        assert "is" not in kw
        assert "a" not in kw

    def test_empty(self):
        assert _extract_keywords("") == set()

    def test_single_char_filtered(self):
        kw = _extract_keywords("I am a b c")
        assert "b" not in kw
        assert "c" not in kw


# ---------------------------------------------------------------------------
# recategorize_with_sources
# ---------------------------------------------------------------------------

class TestRecategorizeWithSources:
    def test_unsupported_claim_matched_to_source(self):
        sources = [_src("BTS Wikipedia", "BTS is a South Korean boy band formed by Big Hit Entertainment")]
        claims = [_claim("BTS is a South Korean boy band")]
        result = recategorize_with_sources(claims, sources)
        assert result[0].category == "Observed"
        assert "Content-matched" in result[0].justification

    def test_observed_claims_unchanged(self):
        sources = [_src("BTS", "BTS is a boy band")]
        claims = [_claim("BTS is a boy band", category="Observed", justification="original")]
        result = recategorize_with_sources(claims, sources)
        assert result[0].category == "Observed"
        assert result[0].justification == "original"

    def test_inference_claims_unchanged(self):
        sources = [_src("BTS", "BTS is popular")]
        claims = [_claim("BTS is popular", category="Inference")]
        result = recategorize_with_sources(claims, sources)
        assert result[0].category == "Inference"

    def test_no_match_stays_unsupported(self):
        sources = [_src("Weather", "The weather in Paris is sunny today")]
        claims = [_claim("BTS debuted in 2013")]
        result = recategorize_with_sources(claims, sources)
        assert result[0].category == "Unsupported"

    def test_empty_sources(self):
        claims = [_claim("BTS is great")]
        result = recategorize_with_sources(claims, [])
        assert result[0].category == "Unsupported"

    def test_empty_claims(self):
        sources = [_src("BTS", "BTS info")]
        result = recategorize_with_sources([], sources)
        assert result == []

    def test_multiple_claims_mixed_match(self):
        sources = [
            _src("BTS Members", "BTS consists of seven members: RM, Jin, Suga, J-Hope, Jimin, V, Jungkook"),
        ]
        claims = [
            _claim("BTS has seven members including RM and Jin"),
            _claim("The weather is nice today"),
        ]
        result = recategorize_with_sources(claims, sources)
        assert result[0].category == "Observed"
        assert result[1].category == "Unsupported"

    def test_match_against_best_source(self):
        sources = [
            _src("Weather", "Sunny day in Paris"),
            _src("BTS Info", "BTS debuted on June 12 2013 with Big Hit Entertainment"),
        ]
        claims = [_claim("BTS debuted on June 12, 2013")]
        result = recategorize_with_sources(claims, sources)
        assert result[0].category == "Observed"
        assert "[2]" in result[0].justification

    def test_short_claim_skipped(self):
        """Claims with fewer than MIN_KEYWORDS are not matched."""
        sources = [_src("BTS", "BTS is great")]
        claims = [_claim("ok")]  # too short
        result = recategorize_with_sources(claims, sources)
        assert result[0].category == "Unsupported"

    def test_title_contributes_to_matching(self):
        """Source titles are included in keyword matching."""
        sources = [_src("BigHit Entertainment BTS", "kpop group info")]
        claims = [_claim("BTS was formed by BigHit Entertainment")]
        result = recategorize_with_sources(claims, sources)
        assert result[0].category == "Observed"


# ---------------------------------------------------------------------------
# filter_findings_with_sources
# ---------------------------------------------------------------------------

class TestFilterFindingsWithSources:
    def test_t1_finding_removed_when_source_backed(self):
        sources = [_src("BTS", "BTS debuted on June 12 2013")]
        findings = [{"type": "T1", "severity": "hard", "detail": "BTS debuted on June 12, 2013 has no citation"}]
        result = filter_findings_with_sources(findings, sources)
        assert len(result) == 0

    def test_t7_finding_removed_when_source_backed(self):
        sources = [_src("BTS", "BTS is a South Korean boy band from Big Hit")]
        findings = [{"type": "T7", "severity": "hard", "detail": "BTS is a South Korean boy band — unverified"}]
        result = filter_findings_with_sources(findings, sources)
        assert len(result) == 0

    def test_non_overridable_finding_kept(self):
        sources = [_src("BTS", "BTS is great")]
        findings = [{"type": "T5", "severity": "soft", "detail": "Prescriptive advice about BTS"}]
        result = filter_findings_with_sources(findings, sources)
        assert len(result) == 1

    def test_finding_kept_when_not_source_backed(self):
        sources = [_src("Weather", "Sunny day in Paris")]
        findings = [{"type": "T1", "severity": "hard", "detail": "BTS won 50 Grammy awards"}]
        result = filter_findings_with_sources(findings, sources)
        assert len(result) == 1

    def test_empty_findings(self):
        sources = [_src("BTS", "info")]
        result = filter_findings_with_sources([], sources)
        assert result == []

    def test_empty_sources(self):
        findings = [{"type": "T1", "severity": "hard", "detail": "test"}]
        result = filter_findings_with_sources(findings, [])
        assert len(result) == 1

    def test_mixed_findings(self):
        sources = [_src("BTS Members", "BTS has seven members RM Jin Suga J-Hope Jimin V Jungkook")]
        findings = [
            {"type": "T1", "severity": "hard", "detail": "BTS has seven members including RM"},
            {"type": "T5", "severity": "soft", "detail": "unsolicited advice"},
            {"type": "T7", "severity": "hard", "detail": "The price of gold is $2000"},
        ]
        result = filter_findings_with_sources(findings, sources)
        # T1 removed (source-backed), T5 kept (not overridable), T7 kept (not source-backed)
        assert len(result) == 2
        assert result[0]["type"] == "T5"
        assert result[1]["type"] == "T7"


# ---------------------------------------------------------------------------
# NLI-backed recategorization
# ---------------------------------------------------------------------------

class TestNLIBackedRecategorization:
    def test_nli_entailment_upgrades_claim(self):
        """Claims with strong NLI entailment should be upgraded."""
        sources = [_src("BTS", "BTS debuted on June 12 2013")]
        claims = [_claim("BTS debuted on June 12, 2013")]
        nli_claims = [{
            "text": "BTS debuted on June 12, 2013",
            "nli_result": {
                "best_entailment": 0.85,
                "worst_contradiction": 0.1,
                "supported": True,
                "best_source_idx": 0,
                "confidence_tier": "strong_support",
            },
        }]
        result = recategorize_with_sources(claims, sources, nli_claims=nli_claims)
        assert result[0].category == "Observed"
        assert "nli_entailment" in result[0].justification

    def test_nli_weak_entailment_not_upgraded(self):
        """Claims with weak NLI entailment and low keyword overlap should not be upgraded."""
        sources = [_src("Weather", "The weather in Tokyo is rainy and cold")]
        claims = [_claim("The stadium capacity holds 50000 spectators")]
        nli_claims = [{
            "text": "The stadium capacity holds 50000 spectators",
            "nli_result": {
                "best_entailment": 0.3,
                "worst_contradiction": 0.1,
                "supported": False,
                "best_source_idx": 0,
            },
        }]
        result = recategorize_with_sources(claims, sources, nli_claims=nli_claims)
        # Should stay unsupported — NLI too weak and no keyword overlap
        assert result[0].category == "Unsupported"

    def test_justification_records_match_method(self):
        """Justification should record whether match was keyword or NLI."""
        sources = [_src("BTS Wikipedia", "BTS is a South Korean boy band formed by Big Hit Entertainment")]
        claims = [_claim("BTS is a South Korean boy band")]
        # Without NLI, uses keyword match
        result = recategorize_with_sources(claims, sources)
        assert result[0].category == "Observed"
        assert "keyword_overlap" in result[0].justification


# ---------------------------------------------------------------------------
# _find_nli_support
# ---------------------------------------------------------------------------

class TestFindNLISupport:
    def test_exact_match_with_support(self):
        nli_claims = [{
            "text": "BTS has seven members",
            "nli_result": {"best_entailment": 0.9, "supported": True},
        }]
        result = _find_nli_support("BTS has seven members", nli_claims)
        assert result is not None
        assert result["best_entailment"] == 0.9

    def test_no_match_returns_none(self):
        nli_claims = [{
            "text": "Weather is sunny",
            "nli_result": {"best_entailment": 0.9},
        }]
        result = _find_nli_support("BTS has seven members", nli_claims)
        assert result is None

    def test_empty_nli_claims(self):
        assert _find_nli_support("any text", []) is None
        assert _find_nli_support("any text", None) is None

    def test_below_threshold_returns_none(self):
        nli_claims = [{
            "text": "BTS has seven members",
            "nli_result": {"best_entailment": 0.3},
        }]
        result = _find_nli_support("BTS has seven members", nli_claims)
        assert result is None


class TestSourceMatchThresholdAlignment:
    """Findings and claims must use the same keyword overlap cutoff."""

    def test_mid_overlap_does_not_drop_finding_when_claim_stays_unsupported(self):
        sources = [_src("Lexicon", "alpha beta gamma unused")]
        claim_text = "alpha beta gamma delta epsilon"
        claims = [_claim(claim_text)]
        findings = [{"type": "T1", "severity": "hard", "detail": claim_text}]
        nli_claims = [{
            "text": claim_text,
            "nli_result": {"best_entailment": 0.2, "worst_contradiction": 0.1},
        }]
        recat = recategorize_with_sources(claims, sources, nli_claims=nli_claims)
        filtered = filter_findings_with_sources(findings, sources, nli_claims=nli_claims)
        assert recat[0].category == "Unsupported"
        assert len(filtered) == 1

    def test_empty_nli_result_uses_keyword_threshold(self):
        sources = [_src("Lexicon", "alpha beta gamma unused")]
        claims = [_claim("alpha beta gamma delta epsilon")]
        nli_claims = [{"text": "alpha beta gamma delta epsilon"}]
        recat = recategorize_with_sources(claims, sources, nli_claims=nli_claims)
        assert recat[0].category == "Observed"


# ---------------------------------------------------------------------------
# verify_citation_grounding
# ---------------------------------------------------------------------------


class TestVerifyCitationGrounding:
    """Tests for deterministic citation verification against source snippets."""

    def test_valid_citation_matching_source(self):
        sources = [_src("Boiling Point", "Water boils at 100 degrees Celsius at sea level.")]
        text = "Under standard atmospheric pressure, water boils at 100 degrees Celsius [1]."
        findings = verify_citation_grounding(text, sources)
        assert findings == []

    def test_out_of_range_citation_flagged(self):
        sources = [_src("Physics", "Gravity pulls objects downward.")]
        text = "According to advanced experiments, gravity causes acceleration [5]."
        findings = verify_citation_grounding(text, sources)
        assert len(findings) == 1
        assert findings[0]["type"] == "T1"
        assert "non-existent source [5]" in findings[0]["detail"]

    def test_unbacked_citation_flagged(self):
        sources = [
            _src("Apples", "Apples are edible fruits produced by apple trees."),
            _src("Bananas", "Bananas are elongated yellow fruits rich in potassium."),
        ]
        # Text cites source [2] (bananas) for an apple claim with zero keyword overlap
        text = "Apples contain distinct red or green skin pigments and seeds [2]."
        findings = verify_citation_grounding(text, sources)
        assert len(findings) == 1
        assert findings[0]["type"] == "T1"
        assert "Unbacked citation: [2]" in findings[0]["detail"]

    def test_empty_text_or_sources_returns_empty(self):
        assert verify_citation_grounding("", [_src("Title", "Snippet")]) == []
        assert verify_citation_grounding("Some text without citations", []) == []
        assert verify_citation_grounding("   ", []) == []

    def test_zero_sources_with_citation_flagged(self):
        findings = verify_citation_grounding("Some text [1]", [])
        assert len(findings) == 1
        assert findings[0]["type"] == "T1"
        assert findings[0]["severity"] == "hard"
        assert "available sources: 0" in findings[0]["detail"]

    def test_zero_sources_multiple_citations_flagged(self):
        findings = verify_citation_grounding("First claim [1] and second claim [2].", [])
        assert len(findings) == 2
        assert all(f["type"] == "T1" and f["severity"] == "hard" for f in findings)
        assert any("[1]" in f["detail"] for f in findings)
        assert any("[2]" in f["detail"] for f in findings)


# ---------------------------------------------------------------------------
# _extract_numbers
# ---------------------------------------------------------------------------


class TestExtractNumbers:
    def test_extract_integers_and_decimals(self):
        nums = _extract_numbers("Items: 42, rate: 3.14, temp: -5, offset: .75")
        assert 42.0 in nums
        assert 3.14 in nums
        assert 0.75 in nums

    def test_extract_percentages(self):
        nums = _extract_numbers("Discount is 20%, tax is 8.5 percent, fee is 1 pct.")
        assert 20.0 in nums
        assert 8.5 in nums
        assert 1.0 in nums

    def test_extract_currency_symbols(self):
        nums = _extract_numbers("Costs: $50, €100, £75.50, ¥5000, $1,200.50, 50 USD, 50 dollars")
        assert 50.0 in nums
        assert 100.0 in nums
        assert 75.5 in nums
        assert 5000.0 in nums
        assert 1200.5 in nums

    def test_extract_scale_multipliers(self):
        nums = _extract_numbers("Revenues: $1.5M, 50k, 2 billion, 100 thousand")
        assert 1500000.0 in nums
        assert 1.5 in nums
        assert 50000.0 in nums
        assert 50.0 in nums
        assert 2000000000.0 in nums
        assert 2.0 in nums
        assert 100000.0 in nums
        assert 100.0 in nums

    def test_extract_word_numbers(self):
        nums = _extract_numbers("There are seven members, eight candidates, fifty participants, and thirty days.")
        assert 7.0 in nums
        assert 8.0 in nums
        assert 50.0 in nums
        assert 30.0 in nums

    def test_extract_compound_word_numbers(self):
        nums = _extract_numbers("Age is twenty-five, count is twenty five, distance is two million.")
        assert 25.0 in nums
        assert 2000000.0 in nums
        assert 2.0 in nums

    def test_extract_unit_one_quantifiers(self):
        nums = _extract_numbers("Plan includes one percent bonus, one dollar fee, and one year warranty.")
        assert 1.0 in nums

    def test_extract_pronoun_one_ignored(self):
        nums = _extract_numbers("This is one of the best methods and one can see the result.")
        assert 1.0 not in nums

    def test_strips_citation_and_internal_brackets(self):
        nums = _extract_numbers("Revenue was $50 [1] and [2] and [Typicality removed] and [1, 2].")
        assert 50.0 in nums
        assert 1.0 not in nums
        assert 2.0 not in nums

    def test_build_source_number_sets(self):
        sources = [
            _src("Pricing $100", "Includes 30-day trial and 20% discount"),
            _src("Specs", "Capacity is 500 units"),
        ]
        num_sets = build_source_number_sets(sources)
        assert len(num_sets) == 2
        assert {100.0, 30.0, 20.0}.issubset(num_sets[0])
        assert 500.0 in num_sets[1]


# ---------------------------------------------------------------------------
# _get_citation_segments
# ---------------------------------------------------------------------------


class TestCitationSegmentation:
    def test_single_citation_segment(self):
        segments = _get_citation_segments("Plan costs $50 [1].")
        assert len(segments) == 1
        assert segments[0][0] == [1]
        assert "Plan costs $50 [1]." in segments[0][1]

    def test_stacked_adjacent_citations(self):
        segments = _get_citation_segments("Plan costs $50 [1][2].")
        assert len(segments) == 1
        assert segments[0][0] == [1, 2]

    def test_stacked_comma_separated_citations(self):
        segments = _get_citation_segments("Plan costs $50 [1], [2].")
        assert len(segments) == 1
        assert segments[0][0] == [1, 2]

    def test_clause_delimited_multi_citations(self):
        segments = _get_citation_segments("Plan A costs $50 [1], whereas Plan B costs $100 [2].")
        assert len(segments) == 2
        assert segments[0][0] == [1]
        assert "$50" in segments[0][1]
        assert segments[1][0] == [2]
        assert "$100" in segments[1][1]

    def test_lead_in_citations(self):
        segments = _get_citation_segments("[1] found $50, while [2] found $100.")
        assert len(segments) == 2
        assert segments[0][0] == [1]
        assert "$50" in segments[0][1]
        assert segments[1][0] == [2]
        assert "$100" in segments[1][1]


# ---------------------------------------------------------------------------
# Quantitative Citation Verification
# ---------------------------------------------------------------------------


class TestNumericCitationVerification:
    """Tests for quantitative and numeric citation grounding against source snippets."""

    def test_valid_percentage_passes(self):
        sources = [_src("Tax Policy", "The standard deduction rate is 20% for eligible taxpayers.")]
        text = "Under the tax policy, the standard deduction is 20% [1]."
        findings = verify_citation_grounding(text, sources)
        assert findings == []

    def test_unbacked_percentage_flagged(self):
        sources = [_src("Tax Policy", "The standard deduction rate is 20% for eligible taxpayers.")]
        text = "Under the tax policy, the standard deduction is 50% [1]."
        findings = verify_citation_grounding(text, sources)
        assert len(findings) == 1
        assert findings[0]["type"] == "T1"
        assert findings[0]["severity"] == "hard"
        assert "[1]" in findings[0]["detail"]
        assert "50" in findings[0]["detail"]

    def test_valid_currency_passes(self):
        sources = [_src("SaaS Pricing", "Basic subscription costs $50 per month.")]
        text = "The basic subscription costs $50 per month [1]."
        findings = verify_citation_grounding(text, sources)
        assert findings == []

    def test_unbacked_currency_flagged(self):
        sources = [_src("SaaS Pricing", "Basic subscription costs $50 per month.")]
        text = "The basic subscription costs $150 per month [1]."
        findings = verify_citation_grounding(text, sources)
        assert len(findings) == 1
        assert findings[0]["type"] == "T1"
        assert findings[0]["severity"] == "hard"
        assert "[1]" in findings[0]["detail"]
        assert "150" in findings[0]["detail"]

    def test_valid_duration_days_passes(self):
        sources = [_src("Return Policy", "Customers enjoy a 30-day money-back guarantee.")]
        text = "The service provides a 30 days money-back guarantee [1]."
        findings = verify_citation_grounding(text, sources)
        assert findings == []

    def test_unbacked_duration_days_flagged(self):
        sources = [_src("Return Policy", "Customers enjoy a 14-day money-back guarantee.")]
        text = "The service provides a 30 days money-back guarantee [1]."
        findings = verify_citation_grounding(text, sources)
        assert len(findings) == 1
        assert findings[0]["type"] == "T1"
        assert findings[0]["severity"] == "hard"
        assert "[1]" in findings[0]["detail"]
        assert "30" in findings[0]["detail"]

    def test_word_form_number_passes(self):
        sources = [_src("Music Group", "BTS consists of seven members.")]
        text = "BTS has 7 members [1]."
        findings = verify_citation_grounding(text, sources)
        assert findings == []

    def test_word_form_number_unbacked_flagged(self):
        sources = [_src("Music Group", "BTS consists of seven members.")]
        text = "BTS has eight members [1]."
        findings = verify_citation_grounding(text, sources)
        assert len(findings) == 1
        assert findings[0]["type"] == "T1"
        assert findings[0]["severity"] == "hard"
        assert "8" in findings[0]["detail"]

    def test_compound_word_number_twenty_five(self):
        sources = [_src("Survey", "The average score was twenty-five points.")]
        text = "The survey reported an average of 25 points [1]."
        findings = verify_citation_grounding(text, sources)
        assert findings == []

    def test_scale_multiplier_million_passes(self):
        sources = [_src("Financial Report", "In 2023, net revenue was $1.5M.")]
        text = "In 2023, net revenue reached $1.5 million [1]."
        findings = verify_citation_grounding(text, sources)
        assert findings == []

    def test_scale_multiplier_unbacked_flagged(self):
        sources = [_src("Financial Report", "In 2023, net revenue was $1.5M.")]
        text = "In 2023, net revenue reached $5 million [1]."
        findings = verify_citation_grounding(text, sources)
        assert len(findings) >= 1
        assert findings[0]["type"] == "T1"
        assert findings[0]["severity"] == "hard"

    def test_formatted_comma_numbers_pass(self):
        sources = [_src("Enrollment", "Total enrollment exceeded 1000 participants.")]
        text = "Over 1,000 participants enrolled in the program [1]."
        findings = verify_citation_grounding(text, sources)
        assert findings == []

    def test_multi_citation_segmentation_valid(self):
        sources = [
            _src("Plan A", "Plan A is priced at $50 per month."),
            _src("Plan B", "Plan B is priced at $100 per month."),
        ]
        text = "Plan A costs $50 [1], whereas Plan B costs $100 [2]."
        findings = verify_citation_grounding(text, sources)
        assert findings == []

    def test_multi_citation_segmentation_partial_unbacked(self):
        sources = [
            _src("Plan A", "Plan A is priced at $50 per month."),
            _src("Plan B", "Plan B is priced at $100 per month."),
        ]
        text = "Plan A costs $50 [1], whereas Plan B costs $200 [2]."
        findings = verify_citation_grounding(text, sources)
        assert len(findings) == 1
        assert findings[0]["type"] == "T1"
        assert "[2]" in findings[0]["detail"]
        assert "200" in findings[0]["detail"]

    def test_stacked_citations_passes_if_in_any(self):
        sources = [
            _src("Study 1", "The growth rate was measured at 50% in the study."),
            _src("Study 2", "Methodology overview and discussion of growth rate."),
        ]
        text = "The growth rate was measured at 50% [1][2]."
        findings = verify_citation_grounding(text, sources)
        assert findings == []

    def test_stacked_citations_unbacked_flagged(self):
        sources = [
            _src("Study 1", "The growth rate was measured at 20% in the study."),
            _src("Study 2", "Methodology overview and discussion of growth rate."),
        ]
        text = "The growth rate was measured at 50% [1][2]."
        findings = verify_citation_grounding(text, sources)
        assert len(findings) == 1
        assert findings[0]["type"] == "T1"
        assert "[1,2]" in findings[0]["detail"]
        assert "50" in findings[0]["detail"]

    def test_pronoun_one_does_not_trigger_false_positive(self):
        sources = [_src("Solutions", "This is an effective method for pipeline validation.")]
        text = "This is one of the most effective methods for pipeline validation [1]."
        findings = verify_citation_grounding(text, sources)
        assert findings == []

    def test_multiple_missing_numbers_in_sentence(self):
        sources = [_src("Stats", "The team scored 10 points in 2 periods.")]
        text = "The team scored 50 points in 4 periods [1]."
        findings = verify_citation_grounding(text, sources)
        assert len(findings) == 2
        details = [f["detail"] for f in findings]
        assert any("50" in d for d in details)
        assert any("4" in d for d in details)

    def test_lead_in_citations_grounded_and_unbacked(self):
        sources = [
            _src("Option A", "Option A provides $50 credit."),
            _src("Option B", "Option B provides $100 credit."),
        ]
        valid_text = "[1] reported Option A at $50, while [2] reported Option B at $100."
        assert verify_citation_grounding(valid_text, sources) == []

        invalid_text = "[1] reported Option A at $50, while [2] reported Option B at $300."
        findings = verify_citation_grounding(invalid_text, sources)
        assert len(findings) == 1
        assert "[2]" in findings[0]["detail"]
        assert "300" in findings[0]["detail"]


# ---------------------------------------------------------------------------
# run_preflight_scan
# ---------------------------------------------------------------------------


class TestRunPreflightScan:
    """Tests for deterministic pre-flight token & bounds scanner."""

    def test_preflight_out_of_bounds_citation(self):
        sources = [_src("Source 1", "Information here."), _src("Source 2", "More info.")]
        has_hard, findings = run_preflight_scan("Claim citing non-existent [5].", sources)
        assert has_hard is True
        assert len(findings) == 1
        assert findings[0]["type"] == "T1"
        assert findings[0]["severity"] == "hard"
        assert "[5]" in findings[0]["detail"]

    def test_preflight_zero_sources_with_citation(self):
        has_hard, findings = run_preflight_scan("Claim with citation [1].", [])
        assert has_hard is True
        assert len(findings) == 1
        assert findings[0]["type"] == "T1"
        assert findings[0]["severity"] == "hard"
        assert "available sources: 0" in findings[0]["detail"]

    def test_preflight_unbacked_numeric_figure(self):
        sources = [_src("Pricing", "Basic subscription costs $50 per month.")]
        has_hard, findings = run_preflight_scan("The basic plan costs $150 per month [1].", sources)
        assert has_hard is True
        assert any("150" in f["detail"] for f in findings)

    def test_preflight_valid_citations_pass(self):
        sources = [
            _src("Revenue", "Revenue increased by 25% to $50M."),
            _src("Staff", "The team has 500 members."),
        ]
        has_hard, findings = run_preflight_scan(
            "Revenue increased by 25% to $50M [1]. The team has 500 members [2].",
            sources,
        )
        assert has_hard is False
        assert len(findings) == 0

    def test_preflight_empty_text(self):
        has_hard, findings = run_preflight_scan("", [_src("S1", "Text")])
        assert has_hard is False
        assert findings == []

    def test_preflight_latency_under_10ms(self):
        import time
        sources = [
            _src(f"Source {i}", f"Data point {i} has value {i * 10} percent.")
            for i in range(1, 6)
        ]
        text = "Data point 1 has value 10 percent [1]. Data point 2 has value 20 percent [2]."
        latencies = []
        for _ in range(100):
            t0 = time.perf_counter()
            run_preflight_scan(text, sources)
            latencies.append((time.perf_counter() - t0) * 1000.0)

        max_lat = max(latencies)
        assert max_lat < 10.0, f"Max latency {max_lat:.2f}ms exceeded 10ms budget"
