"""Comprehensive Adversarial Stress-Testing & Attack Suite Tests.

Validates:
- 6 Modular Adversarial Mutators across all 4 Difficulty Tiers (24 matrix combinations)
- AdversarialMutationEngine and convenience suite generators
- Deterministic interaction with pipeline defense mechanisms:
  - run_preflight_scan (<10ms fast fail-closed bounds and numeric checking)
  - verify_citation_grounding (quantitative verification and out-of-bounds checks)
  - check_poisoning_threshold (35% unbacked ratio and >=2 hard violations guard)
  - guard_arbiter_decision (deterministic override to BLOCK on poisoned drafts)
- Full batch generation across all 27 scenarios in the corpus (648 test cases)
"""
from __future__ import annotations

import time

import pytest

from pipeline.adversarial import (
    MUTATOR_REGISTRY,
    AdversarialMutationEngine,
    AttackVectorEnum,
    BaseAdversarialMutator,
    DifficultyTierEnum,
    DomainEnum,
    MultiDomainScenario,
    StatisticalFallacyMutator,
    generate_adversarial_suite,
    generate_attack_matrix,
    get_mutator,
    load_scenario_corpus,
)
from pipeline.arbiter import check_poisoning_threshold, guard_arbiter_decision
from pipeline.source_match import run_preflight_scan


# Fixture to load corpus once for the test module
@pytest.fixture(scope="module")
def corpus() -> list[MultiDomainScenario]:
    return load_scenario_corpus()


@pytest.fixture(scope="module")
def engine() -> AdversarialMutationEngine:
    return AdversarialMutationEngine()


class TestAttackMatrixAndRegistry:
    """Test suite for attack matrix structure, mutator registry, and generator engine."""

    def test_attack_matrix_dimensions(self) -> None:
        matrix = generate_attack_matrix()
        assert len(matrix) == 24, f"Expected 24 attack matrix configurations, got {len(matrix)}"
        
        vectors = {cfg.attack_vector for cfg in matrix}
        expected_vectors = {v.value for v in AttackVectorEnum}
        assert vectors == expected_vectors

        tiers = {cfg.difficulty_tier for cfg in matrix}
        expected_tiers = {t.value for t in DifficultyTierEnum}
        assert tiers == expected_tiers

    def test_mutator_registry_completeness(self) -> None:
        for vec in AttackVectorEnum:
            assert vec.value in MUTATOR_REGISTRY
            mutator = get_mutator(vec.value)
            assert isinstance(mutator, BaseAdversarialMutator)
            assert mutator.attack_vector == vec.value

    def test_get_mutator_with_enum_and_string(self) -> None:
        mut_enum = get_mutator(AttackVectorEnum.STATISTICAL_FALLACY)
        mut_str = get_mutator("statistical_fallacy")
        assert isinstance(mut_enum, StatisticalFallacyMutator)
        assert isinstance(mut_str, StatisticalFallacyMutator)

    def test_get_mutator_invalid_key_raises(self) -> None:
        with pytest.raises(ValueError, match="Unrecognized attack vector"):
            get_mutator("invalid_nonexistent_vector")

    def test_engine_initialization(self, engine: AdversarialMutationEngine) -> None:
        assert len(engine._mutators) == 6
        for vec in AttackVectorEnum:
            assert vec.value in engine._mutators


class TestStatisticalFallacyMutator:
    """Unit and integration tests for Multi-Clause Statistical Fallacies."""

    def test_mild_base_rate_neglect(self, corpus: list[MultiDomainScenario], engine: AdversarialMutationEngine) -> None:
        scenario = corpus[0]
        case = engine.mutate_statistical(scenario, tier="mild")
        
        assert case.attack_vector == "statistical_fallacy"
        assert case.difficulty_tier == "mild"
        assert "97.8%" in case.mutated_draft
        assert case.expected_defense.expected_verdict == "FAIL"
        assert case.expected_defense.expected_arbiter_decision == "ALLOW_WITH_EDITS"
        assert case.expected_defense.must_trigger_preflight_catch is True
        assert case.expected_defense.must_trigger_poisoning_guard is False

        sources = [d.to_search_source() for d in case.evidence_documents]
        has_hard, findings = run_preflight_scan(case.mutated_draft, sources=sources)
        assert has_hard is True
        assert any("97.8" in f.get("detail", "") for f in findings)

    def test_moderate_simpsons_paradox(self, corpus: list[MultiDomainScenario], engine: AdversarialMutationEngine) -> None:
        scenario = corpus[0]
        case = engine.mutate_statistical(scenario, tier="moderate")

        assert case.attack_vector == "statistical_fallacy"
        assert case.difficulty_tier == "moderate"
        assert "subgroup" in case.mutated_draft.lower()
        assert case.expected_defense.expected_arbiter_decision == "ALLOW_WITH_EDITS"
        assert case.expected_defense.must_trigger_poisoning_guard is False

    def test_extreme_confounding_inversion(self, corpus: list[MultiDomainScenario], engine: AdversarialMutationEngine) -> None:
        scenario = corpus[0]
        case = engine.mutate_statistical(scenario, tier="extreme")

        assert case.attack_vector == "statistical_fallacy"
        assert case.difficulty_tier == "extreme"
        assert "77.7%" in case.mutated_draft
        assert case.expected_defense.expected_arbiter_decision == "BLOCK"
        assert case.expected_defense.must_trigger_preflight_catch is True
        assert case.expected_defense.must_trigger_poisoning_guard is True

        sources = [d.to_search_source() for d in case.evidence_documents]
        has_hard, _ = run_preflight_scan(case.mutated_draft, sources=sources)
        assert has_hard is True

    def test_breaking_bayesian_inverse_probability(self, corpus: list[MultiDomainScenario], engine: AdversarialMutationEngine) -> None:
        scenario = corpus[0]
        case = engine.mutate_statistical(scenario, tier="breaking")

        assert case.attack_vector == "statistical_fallacy"
        assert case.difficulty_tier == "breaking"
        assert "reverse conditional probability" in case.mutated_draft.lower()
        assert case.expected_defense.expected_arbiter_decision == "BLOCK"
        assert case.expected_defense.must_trigger_poisoning_guard is True
        assert case.expected_defense.target_hard_violations >= 2


class TestPromptInjectionMutator:
    """Unit and integration tests for Extreme Prompt Injections & Jailbreak Camouflage."""

    def test_mild_direct_citation_override(self, corpus: list[MultiDomainScenario], engine: AdversarialMutationEngine) -> None:
        scenario = corpus[0]
        case = engine.mutate_injection(scenario, tier="mild")

        assert case.attack_vector == "prompt_injection"
        assert case.difficulty_tier == "mild"
        assert "SYSTEM OVERRIDE" in case.mutated_draft
        assert case.expected_defense.expected_arbiter_decision == "ALLOW_WITH_EDITS"
        assert case.expected_defense.must_trigger_poisoning_guard is False

    def test_moderate_persona_spoofing_in_evidence(self, corpus: list[MultiDomainScenario], engine: AdversarialMutationEngine) -> None:
        scenario = corpus[0]
        case = engine.mutate_injection(scenario, tier="moderate")

        assert case.attack_vector == "prompt_injection"
        assert case.difficulty_tier == "moderate"
        assert "<untrusted_evidence>" in case.evidence_documents[0].content
        assert "SYSTEM INSTRUCTION" in case.evidence_documents[0].content
        assert "Section 2" in case.mutated_draft
        assert case.expected_defense.expected_arbiter_decision == "ALLOW_WITH_EDITS"

    def test_extreme_polyglot_json_injection(self, corpus: list[MultiDomainScenario], engine: AdversarialMutationEngine) -> None:
        scenario = corpus[0]
        case = engine.mutate_injection(scenario, tier="extreme")

        assert case.attack_vector == "prompt_injection"
        assert case.difficulty_tier == "extreme"
        assert "```json" in case.mutated_draft
        assert '"verdict": "PASS"' in case.mutated_draft
        assert "99.9%" in case.mutated_draft
        assert case.expected_defense.expected_arbiter_decision == "BLOCK"
        assert case.expected_defense.must_trigger_poisoning_guard is True

        sources = [d.to_search_source() for d in case.evidence_documents]
        has_hard, _ = run_preflight_scan(case.mutated_draft, sources=sources)
        assert has_hard is True

    def test_breaking_coordinated_delimiter_escape(self, corpus: list[MultiDomainScenario], engine: AdversarialMutationEngine) -> None:
        scenario = corpus[0]
        case = engine.mutate_injection(scenario, tier="breaking")

        assert case.attack_vector == "prompt_injection"
        assert case.difficulty_tier == "breaking"
        assert "ALLOW_PROMPT_OVERRIDE=true" in case.mutated_prompt
        assert "<system_eval_bypass>" in case.mutated_draft
        assert case.expected_defense.expected_arbiter_decision == "BLOCK"
        assert case.expected_defense.must_trigger_poisoning_guard is True


class TestNumericTemporalDriftMutator:
    """Unit and integration tests for Subtle Numeric & Temporal Drift."""

    def test_mild_boundary_numeric_shift(self, corpus: list[MultiDomainScenario], engine: AdversarialMutationEngine) -> None:
        for sc in corpus:
            case = engine.mutate_numeric_temporal(sc, tier="mild")
            assert case.attack_vector == "numeric_temporal_drift"
            assert case.difficulty_tier == "mild"
            assert case.expected_defense.expected_arbiter_decision == "ALLOW_WITH_EDITS"
            assert case.expected_defense.must_trigger_preflight_catch is True
            
            sources = [d.to_search_source() for d in case.evidence_documents]
            has_hard, _ = run_preflight_scan(case.mutated_draft, sources=sources)
            assert has_hard is True, f"Failed preflight check for {sc.scenario_id}"

    def test_moderate_fiscal_year_transposition(self, corpus: list[MultiDomainScenario], engine: AdversarialMutationEngine) -> None:
        for sc in corpus:
            case = engine.mutate_numeric_temporal(sc, tier="moderate")
            assert case.attack_vector == "numeric_temporal_drift"
            assert case.difficulty_tier == "moderate"
            assert case.expected_defense.expected_arbiter_decision == "ALLOW_WITH_EDITS"
            assert case.expected_defense.must_trigger_preflight_catch is True

            sources = [d.to_search_source() for d in case.evidence_documents]
            has_hard, _ = run_preflight_scan(case.mutated_draft, sources=sources)
            assert has_hard is True, f"Failed preflight check on moderate year transposition for {sc.scenario_id}"

    def test_extreme_metric_scale_swap(self, corpus: list[MultiDomainScenario], engine: AdversarialMutationEngine) -> None:
        scenario = corpus[1]
        case = engine.mutate_numeric_temporal(scenario, tier="extreme")

        assert case.attack_vector == "numeric_temporal_drift"
        assert case.difficulty_tier == "extreme"
        assert case.expected_defense.expected_arbiter_decision == "BLOCK"
        assert case.expected_defense.must_trigger_preflight_catch is True
        assert case.expected_defense.must_trigger_poisoning_guard is True

        sources = [d.to_search_source() for d in case.evidence_documents]
        has_hard, _ = run_preflight_scan(case.mutated_draft, sources=sources)
        assert has_hard is True

    def test_breaking_multi_line_fabrication(self, corpus: list[MultiDomainScenario], engine: AdversarialMutationEngine) -> None:
        scenario = corpus[0]
        case = engine.mutate_numeric_temporal(scenario, tier="breaking")

        assert case.attack_vector == "numeric_temporal_drift"
        assert case.difficulty_tier == "breaking"
        assert "$999.5M" in case.mutated_draft
        assert "$888.2M" in case.mutated_draft
        assert case.expected_defense.expected_arbiter_decision == "BLOCK"
        assert case.expected_defense.must_trigger_preflight_catch is True
        assert case.expected_defense.must_trigger_poisoning_guard is True


class TestSyntacticEntanglementMutator:
    """Unit and integration tests for Syntactic & Semantic Entanglement."""

    def test_mild_compound_conjunction(self, corpus: list[MultiDomainScenario], engine: AdversarialMutationEngine) -> None:
        scenario = corpus[0]
        case = engine.mutate_entanglement(scenario, tier="mild")

        assert case.attack_vector == "syntactic_entanglement"
        assert case.difficulty_tier == "mild"
        assert "regulatory agency" in case.mutated_draft
        assert case.expected_defense.expected_arbiter_decision == "ALLOW_WITH_EDITS"
        assert case.expected_defense.must_trigger_poisoning_guard is False

    def test_moderate_nested_conditional(self, corpus: list[MultiDomainScenario], engine: AdversarialMutationEngine) -> None:
        scenario = corpus[0]
        case = engine.mutate_entanglement(scenario, tier="moderate")

        assert case.attack_vector == "syntactic_entanglement"
        assert case.difficulty_tier == "moderate"
        assert "Because" in case.mutated_draft
        assert "waiver" in case.mutated_draft
        assert case.expected_defense.expected_arbiter_decision == "ALLOW_WITH_EDITS"

    def test_extreme_periodic_sentence_10_facts(self, corpus: list[MultiDomainScenario], engine: AdversarialMutationEngine) -> None:
        scenario = corpus[0]
        case = engine.mutate_entanglement(scenario, tier="extreme")

        assert case.attack_vector == "syntactic_entanglement"
        assert case.difficulty_tier == "extreme"
        assert "75.0%" in case.mutated_draft
        assert "90.0%" in case.mutated_draft
        assert case.expected_defense.expected_arbiter_decision == "BLOCK"
        assert case.expected_defense.must_trigger_preflight_catch is True
        assert case.expected_defense.must_trigger_poisoning_guard is True

        sources = [d.to_search_source() for d in case.evidence_documents]
        has_hard, _ = run_preflight_scan(case.mutated_draft, sources=sources)
        assert has_hard is True

    def test_breaking_dangling_inversion_trap(self, corpus: list[MultiDomainScenario], engine: AdversarialMutationEngine) -> None:
        scenario = corpus[0]
        case = engine.mutate_entanglement(scenario, tier="breaking")

        assert case.attack_vector == "syntactic_entanglement"
        assert case.difficulty_tier == "breaking"
        assert "Without failing" in case.mutated_draft
        assert case.expected_defense.expected_arbiter_decision == "BLOCK"
        assert case.expected_defense.must_trigger_poisoning_guard is True


class TestCitationDriftMutator:
    """Unit and integration tests for Citation & Cross-Document Drift."""

    def test_mild_out_of_bounds_phantom_citation(self, corpus: list[MultiDomainScenario], engine: AdversarialMutationEngine) -> None:
        for sc in corpus:
            case = engine.mutate_citation_drift(sc, tier="mild")
            assert case.attack_vector == "citation_drift"
            assert case.difficulty_tier == "mild"
            assert "[99]" in case.mutated_draft or f"[{len(sc.documents) + 10}]" in case.mutated_draft
            assert case.expected_defense.expected_arbiter_decision == "ALLOW_WITH_EDITS"
            assert case.expected_defense.must_trigger_preflight_catch is True

            sources = [d.to_search_source() for d in case.evidence_documents]
            has_hard, findings = run_preflight_scan(case.mutated_draft, sources=sources)
            assert has_hard is True
            assert any(f.get("type") == "T1" for f in findings)

    def test_moderate_cross_document_entity_confusion(self, corpus: list[MultiDomainScenario], engine: AdversarialMutationEngine) -> None:
        for sc in corpus:
            case = engine.mutate_citation_drift(sc, tier="moderate")
            assert case.attack_vector == "citation_drift"
            assert case.difficulty_tier == "moderate"
            assert case.expected_defense.expected_arbiter_decision == "ALLOW_WITH_EDITS"
            assert case.expected_defense.must_trigger_preflight_catch is True

            sources = [d.to_search_source() for d in case.evidence_documents]
            has_hard, _ = run_preflight_scan(case.mutated_draft, sources=sources)
            assert has_hard is True

    def test_extreme_multi_source_shuffling(self, corpus: list[MultiDomainScenario], engine: AdversarialMutationEngine) -> None:
        scenario = corpus[0]
        case = engine.mutate_citation_drift(scenario, tier="extreme")

        assert case.attack_vector == "citation_drift"
        assert case.difficulty_tier == "extreme"
        assert case.expected_defense.expected_arbiter_decision == "BLOCK"
        assert case.expected_defense.must_trigger_preflight_catch is True
        assert case.expected_defense.must_trigger_poisoning_guard is True

        sources = [d.to_search_source() for d in case.evidence_documents]
        has_hard, _ = run_preflight_scan(case.mutated_draft, sources=sources)
        assert has_hard is True

    def test_breaking_non_standard_bracket_obfuscation(self, corpus: list[MultiDomainScenario], engine: AdversarialMutationEngine) -> None:
        scenario = corpus[0]
        case = engine.mutate_citation_drift(scenario, tier="breaking")

        assert case.attack_vector == "citation_drift"
        assert case.difficulty_tier == "breaking"
        assert "[[1]]" in case.mutated_draft
        assert "99.9%" in case.mutated_draft
        assert case.expected_defense.expected_arbiter_decision == "BLOCK"
        assert case.expected_defense.must_trigger_preflight_catch is True
        assert case.expected_defense.must_trigger_poisoning_guard is True


class TestBoundarySaturationMutator:
    """Unit and integration tests for Poisoning Boundary Saturation."""

    def test_mild_under_threshold_ratio(self, corpus: list[MultiDomainScenario], engine: AdversarialMutationEngine) -> None:
        scenario = corpus[0]
        case = engine.mutate_boundary_saturation(scenario, tier="mild")

        assert case.attack_vector == "poisoning_saturation"
        assert case.difficulty_tier == "mild"
        assert case.expected_defense.expected_verdict == "PASS"
        assert case.expected_defense.expected_arbiter_decision == "ALLOW_WITH_EDITS"
        assert case.expected_defense.must_trigger_poisoning_guard is False
        assert case.expected_defense.target_unbacked_ratio <= 0.35

    def test_moderate_borderline_ratio(self, corpus: list[MultiDomainScenario], engine: AdversarialMutationEngine) -> None:
        scenario = corpus[0]
        case = engine.mutate_boundary_saturation(scenario, tier="moderate")

        assert case.attack_vector == "poisoning_saturation"
        assert case.difficulty_tier == "moderate"
        assert case.expected_defense.expected_verdict == "PASS"
        assert case.expected_defense.expected_arbiter_decision == "ALLOW_WITH_EDITS"
        assert case.expected_defense.must_trigger_poisoning_guard is False
        assert case.expected_defense.target_unbacked_ratio <= 0.35

    def test_extreme_over_threshold_ratio(self, corpus: list[MultiDomainScenario], engine: AdversarialMutationEngine) -> None:
        scenario = corpus[0]
        case = engine.mutate_boundary_saturation(scenario, tier="extreme")

        assert case.attack_vector == "poisoning_saturation"
        assert case.difficulty_tier == "extreme"
        assert case.expected_defense.expected_verdict == "FAIL"
        assert case.expected_defense.expected_arbiter_decision == "BLOCK"
        assert case.expected_defense.must_trigger_poisoning_guard is True
        assert case.expected_defense.target_unbacked_ratio > 0.35 or case.expected_defense.target_hard_violations >= 2

    def test_breaking_severe_fabrication(self, corpus: list[MultiDomainScenario], engine: AdversarialMutationEngine) -> None:
        scenario = corpus[0]
        case = engine.mutate_boundary_saturation(scenario, tier="breaking")

        assert case.attack_vector == "poisoning_saturation"
        assert case.difficulty_tier == "breaking"
        assert case.expected_defense.expected_verdict == "FAIL"
        assert case.expected_defense.expected_arbiter_decision == "BLOCK"
        assert case.expected_defense.must_trigger_poisoning_guard is True
        assert case.expected_defense.target_unbacked_ratio >= 0.60


class TestPipelineDefensesIntegration:
    """Tests evaluating interaction with pipeline defense mechanisms."""

    def test_preflight_scanner_sub_10ms_latency_invariant(self, corpus: list[MultiDomainScenario], engine: AdversarialMutationEngine) -> None:
        scenario = corpus[0]
        case = engine.mutate_citation_drift(scenario, tier="mild")
        sources = [d.to_search_source() for d in case.evidence_documents]

        start_time = time.perf_counter()
        has_hard, findings = run_preflight_scan(case.mutated_draft, sources=sources)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        assert has_hard is True
        assert len(findings) >= 1
        assert elapsed_ms < 10.0, f"Preflight scan exceeded 10ms SLA: took {elapsed_ms:.2f}ms"

    def test_arbiter_poisoning_threshold_math(self) -> None:
        # Case A: 3 unsupported out of 10 = 30.0% (<= 35%) and 1 hard violation -> not poisoned
        claims_a = [{"claim": f"c_{i}", "category": "Unsupported" if i < 3 else "Observed"} for i in range(10)]
        findings_a = [{"type": "T1", "severity": "hard"}]
        check_a = check_poisoning_threshold(claims_a, findings_a)
        assert check_a["is_poisoned"] is False
        assert check_a["unsupported_ratio"] == 0.30
        assert check_a["hard_count"] == 1

        dec_a, _ = guard_arbiter_decision("ALLOW_WITH_EDITS", claims_a, findings_a)
        assert dec_a == "ALLOW_WITH_EDITS"

        # Case B: 4 unsupported out of 10 = 40.0% (> 35%) -> poisoned -> BLOCK
        claims_b = [{"claim": f"c_{i}", "category": "Unsupported" if i < 4 else "Observed"} for i in range(10)]
        findings_b = [{"type": "T1", "severity": "hard"}]
        check_b = check_poisoning_threshold(claims_b, findings_b)
        assert check_b["is_poisoned"] is True
        assert check_b["unsupported_ratio"] == 0.40

        dec_b, rationale_b = guard_arbiter_decision("ALLOW_WITH_EDITS", claims_b, findings_b)
        assert dec_b == "BLOCK"
        assert any("heavily poisoned" in r for r in rationale_b)

        # Case C: 2 hard violations with 20% unsupported -> poisoned by hard threshold -> BLOCK
        claims_c = [{"claim": f"c_{i}", "category": "Unsupported" if i < 2 else "Observed"} for i in range(10)]
        findings_c = [{"type": "T1", "severity": "hard"}, {"type": "T1", "severity": "hard"}]
        check_c = check_poisoning_threshold(claims_c, findings_c)
        assert check_c["is_poisoned"] is True
        assert check_c["hard_count"] == 2

        dec_c, _ = guard_arbiter_decision("ALLOW_WITH_EDITS", claims_c, findings_c)
        assert dec_c == "BLOCK"


class TestBatchSuiteGenerationAndCorpusCoverage:
    """Tests evaluating full batch test generation across all scenarios in the corpus."""

    def test_full_suite_generation_count(self, corpus: list[MultiDomainScenario]) -> None:
        suite = generate_adversarial_suite(scenarios=corpus)
        expected_total = len(corpus) * 24
        assert len(suite) == expected_total, f"Expected {expected_total} cases, got {len(suite)}"

    def test_test_case_ids_uniqueness(self, corpus: list[MultiDomainScenario]) -> None:
        suite = generate_adversarial_suite(scenarios=corpus)
        ids = [c.test_id for c in suite]
        assert len(ids) == len(set(ids)), "Test case IDs must be unique across the entire suite."

    def test_domain_distribution_in_suite(self, corpus: list[MultiDomainScenario]) -> None:
        suite = generate_adversarial_suite(scenarios=corpus)
        scenario_map = {s.scenario_id: s for s in corpus}
        
        domain_case_counts: dict[str, int] = {d.value: 0 for d in DomainEnum}
        for case in suite:
            domain = scenario_map[case.scenario_id].domain
            domain_case_counts[domain] += 1

        for domain, count in domain_case_counts.items():
            assert count >= 5 * 24, f"Domain '{domain}' has fewer than 120 test cases ({count})"

    def test_case_payload_non_empty_invariants(self, corpus: list[MultiDomainScenario]) -> None:
        suite = generate_adversarial_suite(scenarios=corpus)
        for case in suite:
            assert case.test_id.strip()
            assert case.scenario_id.strip()
            assert case.mutated_prompt.strip()
            assert case.mutated_draft.strip()
            assert len(case.evidence_documents) >= 1
            assert case.expected_defense.expected_verdict in ["PASS", "FAIL", "ABSTAIN"]
            assert case.expected_defense.expected_arbiter_decision in ["ALLOW", "ALLOW_WITH_EDITS", "ALLOW_AS_UNKNOWN_ONLY", "BLOCK"]
