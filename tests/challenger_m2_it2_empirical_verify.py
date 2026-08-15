"""Empirical Challenger 2 Verification and Stress Test Script (Iteration 2).
Exercises all 5 remediated defects and deep adversarial edge cases.
"""
import time
import statistics
from pipeline.source_match import (
    ClauseType,
    PropositionSpan,
    parse_clause_ast,
    disentangle_and_excise,
    _CITATION_PATTERN,
    _parse_citation_group,
)
from pipeline.sanitizer import clean_grammar_and_punctuation


def test_defect_1_subordinator_precedence():
    """Verify that -ing subordinators are not shadowed by participials."""
    cases = [
        ("Notwithstanding that clinical trials were delayed [1], efficacy was high [2].", ClauseType.CONCESSIVE, "notwithstanding that"),
        ("Providing that cardiac monitoring is maintained [1], discharge is approved [2].", ClauseType.CONDITIONAL, "providing that"),
        ("Assuming that patient compliance is optimal [1], remission occurs [2].", ClauseType.CONDITIONAL, "assuming that"),
        ("In spite of severe symptoms [1], the patient recovered [2].", ClauseType.CONCESSIVE, "in spite of"),
        ("Even though mutations were found [1], the drug remained active [2].", ClauseType.CONCESSIVE, "even though"),
    ]
    results = []
    for sentence, exp_type, exp_sub in cases:
        spans = parse_clause_ast(sentence)
        match_type = spans[0].clause_type == exp_type
        match_sub = spans[0].subordinator == exp_sub
        results.append((sentence[:40], spans[0].clause_type, spans[0].subordinator, match_type and match_sub))
    return results


def test_defect_2_noun_phrases_with_ing():
    """Verify that noun subjects with -ing words are not classified as participials and subjects are preserved."""
    cases = [
        ("Promising candidate demonstrated strong binding affinity [1], while control showed none [2].", "Promising candidate demonstrated strong binding affinity [1]."),
        ("The steering committee approved the revised protocol [1], although budget constraints persisted [2].", "The steering committee approved the revised protocol [1]."),
        ("Ongoing surveillance is strictly required [1], although initial safety looks good [2].", "Ongoing surveillance is strictly required [1]."),
        ("Existing protocols prevent cross-contamination [1], while new methods are evaluated [2].", "Existing protocols prevent cross-contamination [1]."),
        ("Leading researchers presented novel findings [1], while attendees took notes [2].", "Leading researchers presented novel findings [1]."),
        ("Smoking cigarettes increases cardiovascular risk [1], whereas exercise mitigates it [2].", "Smoking cigarettes increases cardiovascular risk [1]."),
        ("Rising inflation impacts healthcare costs [1], unless price controls apply [2].", "Rising inflation impacts healthcare costs [1]."),
    ]
    results = []
    for sentence, exp_excised in cases:
        spans = parse_clause_ast(sentence)
        is_indep = spans[0].clause_type == ClauseType.INDEPENDENT
        sub_is_none = spans[0].subordinator is None
        excised = disentangle_and_excise(sentence, {"span_2"}, spans)
        excised_match = excised == exp_excised
        results.append((sentence[:35], spans[0].clause_type, spans[0].subordinator, excised, is_indep and sub_is_none and excised_match))
    return results


def test_defect_3_in_clause_conjunctions():
    """Verify compound subjects and objects with 'and' / 'or' within clauses are not split."""
    cases = [
        ("Because cohort A [1] and cohort B [2] demonstrated efficacy [3], regulatory filing proceeded [4].", 2),
        ("If doctors [1] or nurses [2] detect elevated blood pressure [3], medication is administered [4].", 2),
        ("Although the safety committee [1] and the scientific board [2] raised concerns [3], trials continued [4].", 2),
    ]
    results = []
    for sentence, exp_span_count in cases:
        spans = parse_clause_ast(sentence)
        count_match = len(spans) == exp_span_count
        results.append((sentence[:40], len(spans), exp_span_count, count_match, spans[0].raw_text))
    return results


def test_defect_4_multi_citation_formats():
    """Verify bracketed citation parsing for multi-citations, ranges, semicolons, and spaces."""
    cases = [
        ("The finding was confirmed in preclinical studies [1, 2].", [1, 2]),
        ("Multiple trials verified the outcome [1, 2, 3].", [1, 2, 3]),
        ("Extended cohort data supported the claim [1-4].", [1, 2, 3, 4]),
        ("Mixed format citations were referenced [1, 3-5, 8].", [1, 3, 4, 5, 8]),
        ("Spaced and semicolon citations [ 1 ; 2 ; 3 ].", [1, 2, 3]),
        ("Adjacent single citations [1][2] and [3].", [1, 2, 3]),
    ]
    results = []
    for sentence, exp_indices in cases:
        spans = parse_clause_ast(sentence)
        all_indices = []
        for s in spans:
            all_indices.extend(s.citation_indices)
        match = all_indices == exp_indices
        results.append((sentence[:40], all_indices, exp_indices, match))
    return results


def test_defect_5_multi_sentence_scoping():
    """Verify per-sentence matrix excision and promotion in multi-sentence paragraphs."""
    text = (
        "Although trial A succeeded [1], toxicity was observed in cohort B [2]. "
        "Because early results were positive [3], the agency granted fast-track review [4]."
    )
    spans = parse_clause_ast(text)
    
    # Excision Scenario 1: Excise span_2 (matrix of sent 1) and span_3 (subordinate of sent 2)
    # Expected: "Trial A succeeded [1]. The agency granted fast-track review [4]."
    excised_1 = disentangle_and_excise(text, {"span_2", "span_3"}, spans)
    match_1 = (
        excised_1 == "Trial A succeeded [1]. The agency granted fast-track review [4]."
    )

    # Excision Scenario 2: Excise span_1 (subordinate of sent 1) and span_4 (matrix of sent 2)
    # Expected: "Toxicity was observed in cohort B [2]. Early results were positive [3]."
    excised_2 = disentangle_and_excise(text, {"span_1", "span_4"}, spans)
    match_2 = (
        excised_2 == "Toxicity was observed in cohort B [2]. Early results were positive [3]."
    )

    # Excision Scenario 3: Excise matrix of both sentences (span_2 and span_4)
    # Expected: "Trial A succeeded [1]. Early results were positive [3]."
    excised_3 = disentangle_and_excise(text, {"span_2", "span_4"}, spans)
    match_3 = (
        excised_3 == "Trial A succeeded [1]. Early results were positive [3]."
    )

    return [
        ("Scenario 1 (excise 2, 3)", excised_1, match_1),
        ("Scenario 2 (excise 1, 4)", excised_2, match_2),
        ("Scenario 3 (excise 2, 4)", excised_3, match_3),
    ]


def test_stress_deep_nesting_and_middle_relatives():
    """Test deep nesting levels 1 to 8 and middle relative clause subject-verb rejoining."""
    # Middle relative clause rejoining
    sent_rel = "The monoclonal antibody, which was synthesized in 2022 [1], reduced tumor volume by 65% [2]."
    spans_rel = parse_clause_ast(sent_rel)
    excised_rel = disentangle_and_excise(sent_rel, {"span_2"}, spans_rel)
    # When relative clause (span_2) is excised, subject and verb phrase must re-join
    # spans_rel: span_1="The monoclonal antibody,", span_2="which was synthesized in 2022 [1],", span_3="reduced tumor volume by 65% [2]."
    # If span_2 is excised, span 1 + span 3 -> "The monoclonal antibody reduced tumor volume by 65% [2]."
    match_rel = excised_rel == "The monoclonal antibody reduced tumor volume by 65% [2]."

    # Level 5 complex nesting
    sent_l5 = (
        "Having established baseline safety in phase 1 [1], "
        "although comorbidities were frequent [2], "
        "while continuous telemetry monitored arrhythmias [3], "
        "if cardiac biomarkers remain stable [4], "
        "the hospital panel authorized patient discharge [5], "
        "which expedites bed turnover [6]."
    )
    spans_l5 = parse_clause_ast(sent_l5)
    l5_count = len(spans_l5) == 6
    l5_slices = all(sent_l5[s.start_char:s.end_char] == s.raw_text for s in spans_l5)
    
    # Excise unbacked spans 2, 3, 6
    excised_l5 = disentangle_and_excise(sent_l5, {"span_2", "span_3", "span_6"}, spans_l5)
    l5_excision_valid = (
        "[2]" not in excised_l5 and
        "[3]" not in excised_l5 and
        "[6]" not in excised_l5 and
        "[1]" in excised_l5 and
        "[4]" in excised_l5 and
        "[5]" in excised_l5 and
        excised_l5.endswith(".")
    )

    return {
        "middle_relative_match": match_rel,
        "middle_relative_output": excised_rel,
        "l5_span_count_6": l5_count,
        "l5_slice_identity": l5_slices,
        "l5_excision_valid": l5_excision_valid,
        "l5_excised_text": excised_l5,
    }


def benchmark_latency_and_determinism():
    """Benchmark 1,000 iterations for determinism and compute latency percentiles."""
    sentence = (
        "Notwithstanding that preliminary safety assays were encouraging [1], "
        "while clinical investigators monitored patient tolerance [2], "
        "if hematologic parameters stay within reference limits [3], "
        "the medical executive board authorized protocol continuation [4], "
        "which enables full trial completion [5]."
    )
    
    base_spans = parse_clause_ast(sentence)
    base_excised = disentangle_and_excise(sentence, {"span_1", "span_3", "span_5"}, base_spans)
    
    latencies_parse = []
    latencies_excise = []
    latencies_total = []

    for _ in range(1000):
        t0 = time.perf_counter()
        spans = parse_clause_ast(sentence)
        t1 = time.perf_counter()
        excised = disentangle_and_excise(sentence, {"span_1", "span_3", "span_5"}, spans)
        t2 = time.perf_counter()

        assert len(spans) == len(base_spans)
        assert excised == base_excised
        
        latencies_parse.append((t1 - t0) * 1000.0)
        latencies_excise.append((t2 - t1) * 1000.0)
        latencies_total.append((t2 - t0) * 1000.0)

    return {
        "iterations": 1000,
        "deterministic": True,
        "parse_p50_ms": statistics.median(latencies_parse),
        "parse_p99_ms": sorted(latencies_parse)[int(0.99 * len(latencies_parse))],
        "excise_p50_ms": statistics.median(latencies_excise),
        "excise_p99_ms": sorted(latencies_excise)[int(0.99 * len(latencies_excise))],
        "total_avg_ms": sum(latencies_total) / len(latencies_total),
        "total_p99_ms": sorted(latencies_total)[int(0.99 * len(latencies_total))],
    }


if __name__ == "__main__":
    print("=== DEFECT 1: -ing Subordinators ===")
    for r in test_defect_1_subordinator_precedence():
        print(f"  {r[0]} -> Type={r[1]}, Subord={r[2]}, Pass={r[3]}")

    print("\n=== DEFECT 2: Noun Phrases with -ing ===")
    for r in test_defect_2_noun_phrases_with_ing():
        print(f"  {r[0]} -> Type={r[1]}, Excised='{r[3]}', Pass={r[4]}")

    print("\n=== DEFECT 3: In-Clause Conjunctions ===")
    for r in test_defect_3_in_clause_conjunctions():
        print(f"  {r[0]} -> Spans={r[1]}, Exp={r[2]}, Pass={r[3]}")

    print("\n=== DEFECT 4: Multi-Citation Formats ===")
    for r in test_defect_4_multi_citation_formats():
        print(f"  {r[0]} -> Indices={r[1]}, Exp={r[2]}, Pass={r[3]}")

    print("\n=== DEFECT 5: Multi-Sentence Scoping ===")
    for r in test_defect_5_multi_sentence_scoping():
        print(f"  {r[0]} -> Output='{r[1]}', Pass={r[2]}")

    print("\n=== STRESS: Deep Nesting & Relatives ===")
    stress_res = test_stress_deep_nesting_and_middle_relatives()
    for k, v in stress_res.items():
        print(f"  {k}: {v}")

    print("\n=== BENCHMARK: Latency & Determinism ===")
    bench_res = benchmark_latency_and_determinism()
    for k, v in bench_res.items():
        print(f"  {k}: {v}")
