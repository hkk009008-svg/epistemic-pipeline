"""Empirical Adversarial Stress Suite for Challenger Core 2 (M1-M5).

Focus Areas:
1. False Rejection Rate (FRR) on benign text with authority terms or harmless instructions
2. Polarity contradictions across medical, legal, and financial domains
3. Multi-source entity swapping with 3+ sources and numeric figures
4. Performance under large drafts (10KB - 100KB) and ReDoS resistance
"""
import base64
import time
import pytest

from pipeline.models import ClaimEntry, SearchSource
from pipeline.source_match import (
    ClauseType,
    PropositionSpan,
    _decode_b64,
    _extract_keywords,
    _extract_numbers,
    disentangle_and_excise,
    extract_polarity_state,
    filter_findings_with_sources,
    has_polarity_mismatch,
    normalize_preflight_text,
    parse_clause_ast,
    recategorize_with_sources,
    run_preflight_scan,
    scan_prompt_injection,
    verify_citation_grounding,
)


class TestFRRBenignText:
    """Stress-test False Rejection Rate (FRR) on benign prompts and drafts."""

    def test_benign_prompt_instructions_not_flagged(self):
        """Benign instructions mentioning past/prior or technical verbs must NOT trigger prompt injection."""
        benign_prompts = [
            "Please summarize the previous section regarding clinical trial outcomes.",
            "Can you explain how transformer models clear their key-value cache between requests?",
            "What happens if a database transaction drops a table that does not exist?",
            "In SQL, explain how to drop a temporary table after completing a query.",
            "Describe the standard protocol for handling previous protocol deviations in FDA submissions.",
            "Explain how the immune system resets its response after infection clearance.",
            "How do operating systems handle memory when an application overrides default configuration?",
            "The physician should review all prior medical history before prescribing medication.",
            "The compiler can ignore unused variables when optimization flags are enabled.",
            "Describe how to bypass network congestion using multipath routing protocols.",
            "We need to clear the temporary cache directory before running the benchmark.",
            "The policy requires developers to reset their passwords every 90 days.",
            "In legal proceedings, when can a judge dismiss an earlier motion without prejudice?",
            "How does the algorithm forget older observations in an exponential moving average?",
            "Explain the difference between override and overload in object-oriented programming.",
        ]

        false_positives = []
        for prompt in benign_prompts:
            injected, snip = scan_prompt_injection(prompt)
            if injected:
                false_positives.append((prompt, snip))

        assert len(false_positives) == 0, f"FRR failure on benign prompts: {false_positives}"

    def test_benign_base64_payloads_not_flagged(self):
        """Benign base64 strings and data URIs should decode cleanly without flagging as injection."""
        benign_b64_cases = [
            # "Hello world, this is a test report."
            "base64:SGVsbG8gd29ybGQsIHRoaXMgaXMgYSB0ZXN0IHJlcG9ydC4=",
            # "Clinical study results: 85% response rate."
            "[base64]Q2xpbmljYWwgc3R1ZHkgcmVzdWx0czogODUlIHJlc3BvbnNlIHJhdGUu[/base64]",
            # "The company reported Q3 revenue of $450 million."
            "atob('VGhlIGNvbXBhbnkgcmVwb3J0ZWQgUTMgcmV2ZW51ZSBvZiAkNDUwIG1pbGxpb24u')",
            # Non-injection image data wrapper
            "The image asset is stored at base64:iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAA=",
        ]

        for text in benign_b64_cases:
            injected, snip = scan_prompt_injection(text)
            assert not injected, f"Benign base64 falsely flagged: '{text}' (matched: {snip})"

    def test_benign_authority_in_cited_and_contextual_sentences(self):
        """Cited authority claims or structural non-assertions must not produce false T3 findings."""
        sources = [
            SearchSource(
                title="Consensus Guidelines",
                url="https://example.com/consensus",
                snippet="Major medical societies agree that early intervention improves cardiovascular outcomes by 25%.",
            )
        ]

        # Case 1: Bracket-cited sentence with authority wording — must be grounded, no T3 unbacked authority
        draft_cited = "Clinical evidence demonstrates that early intervention improves outcomes by 25% [1]."
        res_cited = run_preflight_scan(text=draft_cited, sources=sources)
        t3_findings = [f for f in res_cited.findings if f.get("type") == "T3"]
        assert len(t3_findings) == 0, f"Cited authority sentence should not have T3 findings: {t3_findings}"

        # Case 2: Uncited purely historical / structural statement without unbacked assertion
        draft_benign_prose = "The consensus meeting was held in Paris in 2022 to review historical clinical literature."
        res_prose = run_preflight_scan(text=draft_benign_prose, sources=sources)
        t3_auth = [f for f in res_prose.findings if f.get("type") == "T3" and "authority" in f.get("detail", "").lower()]
        assert len(t3_auth) == 0, f"Historical prose falsely flagged as unbacked authority: {t3_auth}"


class TestPolarityContradictionsCrossDomain:
    """Stress-test polarity and negation contradiction detection across medical, legal, and financial domains."""

    def test_medical_domain_polarity_contradictions_robust(self):
        """Verify standard contradictory medical claims against source ground truths."""
        test_cases = [
            (
                "Drug X prevents myocardial infarction.",
                "Clinical trials showed Drug X fails to prevent myocardial infarction in high-risk patients.",
                True,
            ),
            (
                "Compound B is non-toxic to hepatic cells.",
                "Compound B is toxic to hepatic cells at standard therapeutic doses.",
                True,
            ),
            (
                "The patient showed no recurrence of arrhythmia.",
                "The patient showed severe recurrence of arrhythmia within 30 days.",
                True,
            ),
            (
                "Vaccine induces neutralizing antibodies.",
                "Vaccine does not induce neutralizing antibodies against the target pathogen.",
                True,
            ),
            (
                "Drug Y reduces systolic blood pressure.",
                "Drug Y did not reduce systolic blood pressure compared to placebo.",
                True,
            ),
            # Affirmative agreement cases
            (
                "Drug Z reduces LDL cholesterol.",
                "In phase 3 trials, Drug Z reduces LDL cholesterol significantly.",
                False,
            ),
            (
                "The formulation is non-toxic.",
                "Laboratory analysis confirms the formulation is non-toxic and biocompatible.",
                False,
            ),
        ]

        for claim, source, expected_mismatch in test_cases:
            mismatch = has_polarity_mismatch(claim, source)
            assert mismatch == expected_mismatch, (
                f"Medical polarity mismatch failed for:\nClaim: '{claim}'\nSource: '{source}'\n"
                f"Expected {expected_mismatch}, got {mismatch}"
            )

    def test_medical_without_side_effects_vs_not_without_contradiction(self):
        """Assert that asserting side effects ('not without') contradicts a source asserting zero side effects ('without side effects')."""
        claim = "Therapy C was not without side effects."
        source = "Therapy C was completely without side effects during all phases."
        assert has_polarity_mismatch(claim, source) is True

    def test_legal_domain_polarity_contradictions_robust(self):
        """Verify standard contradictory legal claims against statutory and judicial sources."""
        test_cases = [
            (
                "The statute prohibits foreign corporate acquisitions.",
                "The amended statute permits foreign corporate acquisitions under license.",
                True,
            ),
            (
                "The district court denied the preliminary injunction.",
                "The district court granted the preliminary injunction against the defendant.",
                True,
            ),
            (
                "The contract was executed without binding arbitration clauses.",
                "The contract contains mandatory binding arbitration clauses.",
                True,
            ),
            (
                "The appellate ruling refutes earlier antitrust precedent.",
                "The appellate ruling upholds earlier antitrust precedent established in 2018.",
                True,
            ),
            # Agreement
            (
                "The court prohibited further asset transfers.",
                "The court order explicitly prohibits further asset transfers.",
                False,
            ),
        ]

        for claim, source, expected_mismatch in test_cases:
            mismatch = has_polarity_mismatch(claim, source)
            assert mismatch == expected_mismatch, (
                f"Legal polarity mismatch failed for:\nClaim: '{claim}'\nSource: '{source}'\n"
                f"Expected {expected_mismatch}, got {mismatch}"
            )

    def test_legal_and_medical_non_prefix_masking_contradictions(self):
        """Assert that grammatical negation ('does not') is not masked when source contains standard 'non-*' nouns."""
        cases = [
            ("The regulation does not apply to non-profit entities.", "The revised regulation applies to all entities including non-profit organizations."),
            ("The clinic does not administer non-steroidal anti-inflammatory drugs.", "The clinic administers non-steroidal anti-inflammatory drugs."),
            ("The company did not sign a non-disclosure agreement.", "The company signed a non-disclosure agreement."),
            ("The patient was not diagnosed with non-small cell lung cancer.", "The patient was diagnosed with non-small cell lung cancer."),
            ("The company does not report non-GAAP metrics.", "The company reports non-GAAP metrics."),
        ]
        for claim, source in cases:
            assert has_polarity_mismatch(claim, source) is True, f"Failed on: '{claim}' vs '{source}'"

    def test_financial_domain_polarity_contradictions(self):
        """Verify contradictory financial claims against earnings and market sources."""
        test_cases = [
            (
                "The company reported no net loss in Q4.",
                "The company reported a severe net loss of $250 million in Q4.",
                True,
            ),
            (
                "The investment fund does not hold fossil fuel assets.",
                "Portfolio disclosures show the investment fund holds $1.2B in fossil fuel assets.",
                True,
            ),
            (
                "The merger was completed without regulatory penalties.",
                "The merger was completed with $45 million in regulatory penalties.",
                True,
            ),
            (
                "Quarterly revenue failed to meet market consensus.",
                "Quarterly revenue met and exceeded market consensus by 8%.",
                True,
            ),
            # Agreement
            (
                "The corporation reported no outstanding debt.",
                "The corporation maintains zero debt and reported no outstanding debt obligations.",
                False,
            ),
        ]

        for claim, source, expected_mismatch in test_cases:
            mismatch = has_polarity_mismatch(claim, source)
            assert mismatch == expected_mismatch, (
                f"Financial polarity mismatch failed for:\nClaim: '{claim}'\nSource: '{source}'\n"
                f"Expected {expected_mismatch}, got {mismatch}"
            )

    def test_polarity_gating_in_recategorization_and_filtering(self):
        """Unsupported claims with polarity contradictions MUST NOT be upgraded to Observed or have T1 dropped."""
        sources = [
            SearchSource(
                title="Clinical Study",
                url="https://example.com/study",
                snippet="Drug Alpha failed to reduce arterial stiffness in hypertensive patients.",
            )
        ]

        claims = [
            ClaimEntry(
                claim="Drug Alpha reduces arterial stiffness in hypertensive patients.",
                category="Unsupported",
                justification="Initial GPT evaluation",
            )
        ]

        # High keyword overlap (Alpha, arterial, stiffness, hypertensive, patients)
        # BUT polarity contradiction (reduces vs failed to reduce)
        recat = recategorize_with_sources(claims, sources)
        assert recat[0].category == "Unsupported", (
            f"Contradicted claim was falsely upgraded to '{recat[0].category}' with justification '{recat[0].justification}'"
        )

        findings = [
            {
                "type": "T1",
                "severity": "hard",
                "detail": "Drug Alpha reduces arterial stiffness in hypertensive patients.",
            }
        ]
        filtered = filter_findings_with_sources(findings, sources)
        assert len(filtered) == 1, "T1 finding on contradicted claim was falsely dropped by filter_findings_with_sources!"


class TestMultiSourceEntitySwapping:
    """Stress-test cross-entity statistic swapping with 3, 4, and 5 sources."""

    def test_three_source_entity_statistic_swapping(self):
        """Verify that swapping statistics across 3 sources is intercepted."""
        sources = [
            SearchSource(title="Pfizer Vaccine Study", url="https://example.com/pfizer", snippet="Pfizer BNT162b2 demonstrated 95% efficacy against COVID-19."),
            SearchSource(title="Moderna Vaccine Study", url="https://example.com/moderna", snippet="Moderna mRNA-1273 demonstrated 94.1% efficacy against COVID-19."),
            SearchSource(title="J&J Vaccine Study", url="https://example.com/jnj", snippet="Johnson & Johnson Ad26.COV2.S demonstrated 66.3% efficacy against COVID-19."),
        ]

        # Swapped Case: Pfizer cited with Moderna's 94.1%
        draft_swapped_1 = "Pfizer BNT162b2 demonstrated 94.1% efficacy against COVID-19 [1, 2, 3]."
        findings_1 = verify_citation_grounding(draft_swapped_1, sources)
        t1_num_1 = [f for f in findings_1 if f.get("type") == "T1" and "numeric" in f.get("detail", "").lower()]
        assert len(t1_num_1) > 0, f"Expected unbacked numeric claim for swapped Pfizer figure, got: {findings_1}"
        assert "94.1" in t1_num_1[0]["detail"]

        # Swapped Case: J&J cited with Pfizer's 95%
        draft_swapped_2 = "Johnson & Johnson Ad26.COV2.S demonstrated 95% efficacy against COVID-19 [1, 2, 3]."
        findings_2 = verify_citation_grounding(draft_swapped_2, sources)
        t1_num_2 = [f for f in findings_2 if f.get("type") == "T1" and "numeric" in f.get("detail", "").lower()]
        assert len(t1_num_2) > 0, f"Expected unbacked numeric claim for swapped J&J figure, got: {findings_2}"
        assert "95" in t1_num_2[0]["detail"]

        # Multi-clause swapped sentence
        draft_multi_swapped = (
            "Pfizer BNT162b2 achieved 66.3% efficacy [1], Moderna achieved 95% efficacy [2], "
            "and Johnson & Johnson achieved 94.1% efficacy [3]."
        )
        findings_multi = verify_citation_grounding(draft_multi_swapped, sources)
        t1_nums = [f for f in findings_multi if f.get("type") == "T1" and "numeric" in f.get("detail", "").lower()]
        assert len(t1_nums) == 3, f"Expected 3 unbacked numeric findings for swapped multi-clause sentence, got {len(t1_nums)}: {findings_multi}"

        # Grounded correct case: all 3 match perfectly
        draft_correct = (
            "Pfizer BNT162b2 achieved 95% efficacy [1], Moderna achieved 94.1% efficacy [2], "
            "and Johnson & Johnson achieved 66.3% efficacy [3]."
        )
        findings_correct = verify_citation_grounding(draft_correct, sources)
        assert len(findings_correct) == 0, f"Correctly grounded multi-source sentence produced false findings: {findings_correct}"

    def test_four_source_financial_entity_swapping(self):
        """Verify statistic binding with 4 tech companies and multi-billion revenue numbers."""
        sources = [
            SearchSource(title="Alphabet 10-K", url="https://example.com/alphabet", snippet="Alphabet recorded annual revenue of $307.4 billion in fiscal year 2023."),
            SearchSource(title="Microsoft 10-K", url="https://example.com/msft", snippet="Microsoft recorded annual revenue of $211.9 billion in fiscal year 2023."),
            SearchSource(title="Amazon 10-K", url="https://example.com/amzn", snippet="Amazon recorded annual revenue of $574.8 billion in fiscal year 2023."),
            SearchSource(title="Apple 10-K", url="https://example.com/aapl", snippet="Apple recorded annual revenue of $383.3 billion in fiscal year 2023."),
        ]

        # Entity swap: Amazon revenue attributed to Alphabet
        draft_swap = "Alphabet recorded annual revenue of $574.8 billion in fiscal year 2023 [1, 2, 3, 4]."
        findings = verify_citation_grounding(draft_swap, sources)
        t1_num = [f for f in findings if f.get("type") == "T1" and "numeric" in f.get("detail", "").lower()]
        assert len(t1_num) > 0, f"Failed to catch swapped $574.8B on Alphabet: {findings}"
        assert "574.8" in t1_num[0]["detail"] or "574800000000" in t1_num[0]["detail"]

        # Inverted citation order [4, 3, 2, 1] with Apple stat
        draft_apple_ok = "Apple recorded annual revenue of $383.3 billion in fiscal year 2023 [4, 3, 2, 1]."
        findings_apple = verify_citation_grounding(draft_apple_ok, sources)
        assert len(findings_apple) == 0, f"Apple stat falsely flagged in reversed citation group: {findings_apple}"

    def test_five_source_international_currency_swapping(self):
        """Verify multi-source entity isolation with 5 distinct global entities and currencies (€, $, £, ¥, ₹)."""
        sources = [
            SearchSource(title="EU Bank", url="https://example.com/ecb", snippet="European Central Bank issued loans worth €150 million to renewable projects."),
            SearchSource(title="US Bank", url="https://example.com/fed", snippet="Federal Reserve tracked loans worth $250 million to commercial real estate."),
            SearchSource(title="UK Bank", url="https://example.com/boe", snippet="Bank of England tracked loans worth £180 million to fintech startups."),
            SearchSource(title="Japan Bank", url="https://example.com/boj", snippet="Bank of Japan allocated loans worth ¥900 million to green initiatives."),
            SearchSource(title="India Bank", url="https://example.com/rbi", snippet="Reserve Bank of India allocated loans worth ₹400 million to rural infrastructure."),
        ]

        # Swapped: UK Bank cited with Japanese ¥900 million
        draft_swap = "Bank of England tracked loans worth 900 million to fintech startups [1, 2, 3, 4, 5]."
        findings = verify_citation_grounding(draft_swap, sources)
        t1_num = [f for f in findings if f.get("type") == "T1" and "numeric" in f.get("detail", "").lower()]
        assert len(t1_num) > 0, f"Failed to flag swapped loan figure on Bank of England: {findings}"


class TestScaleAndPerformance:
    """Stress-test pipeline throughput, latency, and ReDoS resistance under 10KB to 100KB drafts."""

    def _generate_synthetic_corpus(self, target_bytes: int) -> tuple[str, list[SearchSource]]:
        """Generate realistic synthetic document with paragraphs, citations, and numbers."""
        sources = [
            SearchSource(title=f"Source {i}", url=f"https://example.com/{i}", snippet=f"Entity_{i} achieved benchmark score of {100 + i * 5}% in trial {i}.")
            for i in range(1, 21)
        ]

        paragraphs = []
        current_len = 0
        p_idx = 0
        while current_len < target_bytes:
            s_idx = (p_idx % 20) + 1
            score = 100 + s_idx * 5
            para = (
                f"Section: According to trial observations, Entity_{s_idx} achieved benchmark score of {score}% in trial {s_idx} [{s_idx}]. "
                f"Furthermore, secondary measurements indicated stable performance throughout the evaluation window. "
                f"Although initial baseline tests varied slightly, the overall system maintained high consistency.\n\n"
            )
            paragraphs.append(para)
            current_len += len(para.encode("utf-8"))
            p_idx += 1

        full_text = "".join(paragraphs)
        return full_text, sources

    @pytest.mark.parametrize("size_kb", [10, 25, 50, 100])
    def test_large_draft_preflight_and_grounding_performance(self, size_kb: int):
        """Verify sub-linear latency and zero crashes on drafts from 10KB up to 100KB."""
        text, sources = self._generate_synthetic_corpus(size_kb * 1024)
        actual_kb = len(text.encode("utf-8")) / 1024.0

        # Benchmark run_preflight_scan
        t0 = time.perf_counter()
        preflight_res = run_preflight_scan(text=text, sources=sources)
        preflight_ms = (time.perf_counter() - t0) * 1000.0

        # Benchmark verify_citation_grounding
        t0 = time.perf_counter()
        grounding_findings = verify_citation_grounding(text=text, sources=sources)
        grounding_ms = (time.perf_counter() - t0) * 1000.0

        # Benchmark parse_clause_ast
        t0 = time.perf_counter()
        first_50_sentences = text.split("\n\n")[:50]
        for sent in first_50_sentences:
            if sent.strip():
                _spans = parse_clause_ast(sent)
        ast_ms = (time.perf_counter() - t0) * 1000.0

        print(
            f"\n[Scale {actual_kb:.1f} KB] preflight={preflight_ms:.2f}ms, "
            f"grounding={grounding_ms:.2f}ms, ast_50_sent={ast_ms:.2f}ms, "
            f"findings={len(preflight_res.findings)}"
        )

        # Assertions on performance & correctness
        max_allowed_ms = 2500.0 if size_kb == 100 else 1000.0
        assert preflight_ms < max_allowed_ms, f"Preflight scan too slow on {size_kb}KB: {preflight_ms:.2f}ms"
        assert grounding_ms < max_allowed_ms, f"Grounding verification too slow on {size_kb}KB: {grounding_ms:.2f}ms"

        # Hard violations must be False since all synthetic data is well-formed and grounded
        assert not preflight_res.has_hard_preflight, f"Unexpected hard preflight findings: {preflight_res.findings}"

    def test_redos_adversarial_patterns(self):
        """Stress-test regexes against catastrophic backtracking attack vectors."""
        adversarial_payloads = [
            # Long chain of repeated action verbs
            "ignore " * 2000 + "instructions",
            # Long nested punctuation delimiters
            "clause 1, " * 1000 + "clause final.",
            # Long base64-like payload
            "base64:" + "A" * 10000,
            # Repetitive authority prefix tokens
            "studies show that " * 500 + "results are positive.",
            # Deeply nested quotes and brackets
            "[[[[[[[[[[" * 100 + "1" + "]]]]]]]]]]" * 100,
        ]

        for idx, payload in enumerate(adversarial_payloads):
            t0 = time.perf_counter()
            _inj, _ = scan_prompt_injection(payload)
            _nums = _extract_numbers(payload)
            _spans = parse_clause_ast(payload)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            assert elapsed_ms < 200.0, f"ReDoS vulnerability detected on payload {idx}! Latency: {elapsed_ms:.2f}ms"
