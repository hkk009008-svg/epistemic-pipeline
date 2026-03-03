"""Tests for pipeline/source_match.py — deterministic source-matching post-processor."""
from __future__ import annotations

from pipeline.models import ClaimEntry, SearchSource
from pipeline.source_match import (
    _extract_keywords,
    recategorize_with_sources,
    filter_findings_with_sources,
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
