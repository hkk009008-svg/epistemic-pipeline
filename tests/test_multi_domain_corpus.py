"""Comprehensive unit and integration tests for Multi-Domain Evidence Corpus and Data Models."""
from __future__ import annotations

import json
from typing import ClassVar

import pytest
from pydantic import ValidationError

from pipeline.adversarial import (
    AdversarialMutationConfig,
    AttackVectorEnum,
    CorpusValidationReport,
    DifficultyTierEnum,
    DocumentRecord,
    DomainEnum,
    ExpectedDefenseOutcome,
    GroundTruthFact,
    MultiDomainScenario,
    PoisonedScenarioCase,
    get_scenario,
    get_scenarios_by_domain,
    load_scenario_corpus,
    validate_scenario_corpus,
)
from pipeline.models import SearchSource
from pipeline.source_match import run_preflight_scan, verify_citation_grounding

# ---------------------------------------------------------------------------
# Corpus Loading & Basic Health Tests
# ---------------------------------------------------------------------------

class TestCorpusLoading:
    """Test corpus file loading and basic dataset health."""

    def test_load_default_corpus(self):
        """Corpus loads cleanly and contains at least 25 scenarios."""
        scenarios = load_scenario_corpus()
        assert isinstance(scenarios, list)
        assert len(scenarios) >= 25, f"Expected at least 25 scenarios, got {len(scenarios)}"
        assert len(scenarios) == 27

    def test_corpus_file_not_found(self, tmp_path):
        """Corpus loader raises FileNotFoundError for non-existent path."""
        bad_path = tmp_path / "nonexistent.json"
        with pytest.raises(FileNotFoundError, match="Corpus file not found"):
            load_scenario_corpus(bad_path)

    def test_corpus_invalid_json_structure(self, tmp_path):
        """Corpus loader raises ValueError when JSON structure is unexpected."""
        bad_file = tmp_path / "invalid_structure.json"
        bad_file.write_text(json.dumps("just a string"), encoding="utf-8")
        with pytest.raises(ValueError, match="Corpus file format invalid"):
            load_scenario_corpus(bad_file)


# ---------------------------------------------------------------------------
# Domain Taxonomy & Distribution Tests
# ---------------------------------------------------------------------------

class TestDomainDistribution:
    """Test 5-domain taxonomy and minimum coverage invariants."""

    EXPECTED_DOMAINS: ClassVar[set[str]] = {
        DomainEnum.BIOMEDICAL.value,
        DomainEnum.FINANCIAL.value,
        DomainEnum.LEGAL.value,
        DomainEnum.CRYPTOGRAPHIC.value,
        DomainEnum.AUTONOMOUS_CONTRACTS.value,
    }

    def test_all_five_canonical_domains_covered(self):
        """All 5 canonical domains are represented."""
        scenarios = load_scenario_corpus()
        covered_domains = {s.domain for s in scenarios}
        assert covered_domains == self.EXPECTED_DOMAINS

    def test_minimum_five_scenarios_per_domain(self):
        """Each of the 5 domains has at least 5 verified scenarios."""
        scenarios = load_scenario_corpus()
        domain_counts = {}
        for s in scenarios:
            domain_counts[s.domain] = domain_counts.get(s.domain, 0) + 1

        for domain in self.EXPECTED_DOMAINS:
            count = domain_counts.get(domain, 0)
            assert count >= 5, f"Domain '{domain}' has {count} scenarios; minimum required is 5"

    def test_unique_scenario_ids(self):
        """All scenario IDs in the corpus are unique."""
        scenarios = load_scenario_corpus()
        scenario_ids = [s.scenario_id for s in scenarios]
        assert len(scenario_ids) == len(set(scenario_ids))

    def test_scenario_id_naming_convention(self):
        """Scenario IDs adhere to snake_case alphanumeric pattern."""
        scenarios = load_scenario_corpus()
        for s in scenarios:
            assert s.scenario_id.islower()
            assert not any(c in s.scenario_id for c in " -*!@#$%^&()+=[]{}|\\:;\"'<>,.?/")


# ---------------------------------------------------------------------------
# Scenario Structure, Documents, and Pydantic Schema Tests
# ---------------------------------------------------------------------------

class TestScenarioStructure:
    """Test scenario fields, documents, and interoperability properties."""

    def test_non_empty_required_fields(self):
        """All scenarios have non-empty titles, queries, baseline drafts, and subdomains."""
        scenarios = load_scenario_corpus()
        for s in scenarios:
            assert s.scenario_id.strip()
            assert s.title.strip()
            assert s.subdomain.strip()
            assert s.query.strip()
            assert s.clean_baseline_draft.strip()
            assert len(s.documents) >= 1
            assert len(s.ground_truth_facts) >= 1

    def test_scenario_interop_properties(self):
        """Scenario aliases id and clean_draft work as expected."""
        scenarios = load_scenario_corpus()
        for s in scenarios:
            assert s.id == s.scenario_id
            assert s.clean_draft == s.clean_baseline_draft

    def test_document_records_validity(self):
        """All DocumentRecords have valid IDs, non-empty titles, and authoritative content."""
        scenarios = load_scenario_corpus()
        for s in scenarios:
            doc_ids = set()
            for doc in s.documents:
                assert doc.doc_id.strip()
                assert doc.doc_id not in doc_ids, f"Duplicate doc_id {doc.doc_id} in {s.scenario_id}"
                doc_ids.add(doc.doc_id)
                assert doc.title.strip()
                assert doc.content.strip()
                # Interoperability aliases
                assert doc.document_id == doc.doc_id
                assert doc.text == doc.content

    def test_document_to_search_source_dict(self):
        """DocumentRecord converts to SearchSource compatible dictionary."""
        doc = DocumentRecord(
            doc_id="doc_test_01",
            title="Test Document",
            folder="biomedical",
            content="Sample authoritative medical text snippet.",
        )
        src_dict = doc.to_search_source_dict(score=0.95)
        assert src_dict["title"] == "Test Document"
        assert src_dict["url"] == "local://biomedical/doc_test_01"
        assert src_dict["snippet"] == "Sample authoritative medical text snippet."
        assert src_dict["score"] == 0.95

        # Also instantiable into SearchSource
        src = SearchSource(**src_dict)
        assert src.title == "Test Document"


# ---------------------------------------------------------------------------
# Ground-Truth Facts & Citation Invariants
# ---------------------------------------------------------------------------

class TestGroundTruthFactsAndCitations:
    """Test discrete ground truth facts and citation bounds."""

    def test_facts_have_valid_doc_and_citation_index(self):
        """Every fact links to a valid document and valid citation index."""
        scenarios = load_scenario_corpus()
        for s in scenarios:
            doc_ids = {d.doc_id for d in s.documents}
            num_docs = len(s.documents)
            for fact in s.ground_truth_facts:
                assert fact.fact_id.strip()
                assert fact.statement.strip()
                assert fact.cited_doc_id in doc_ids
                assert 1 <= fact.citation_index <= num_docs

    def test_pydantic_validator_rejects_missing_doc_citation(self):
        """MultiDomainScenario validation rejects facts with non-existent cited_doc_id."""
        with pytest.raises(ValidationError, match="cites non-existent doc_id"):
            MultiDomainScenario(
                scenario_id="test_invalid_doc",
                domain="biomedical",
                subdomain="Testing",
                title="Test Invalid Citation",
                query="Test query?",
                documents=[
                    DocumentRecord(
                        doc_id="doc_valid_1",
                        title="Valid Doc",
                        content="Valid content.",
                    )
                ],
                ground_truth_facts=[
                    GroundTruthFact(
                        fact_id="fact_01",
                        statement="Invalid proposition",
                        cited_doc_id="doc_non_existent",
                        citation_index=1,
                    )
                ],
                clean_baseline_draft="Observed: Test [1].",
            )

    def test_pydantic_validator_rejects_out_of_bounds_citation_index(self):
        """MultiDomainScenario validation rejects facts with out-of-range citation_index."""
        with pytest.raises(ValidationError, match="citation_index 5 out of range"):
            MultiDomainScenario(
                scenario_id="test_oob_citation",
                domain="financial",
                subdomain="Testing",
                title="Test OOB Citation",
                query="Test query?",
                documents=[
                    DocumentRecord(
                        doc_id="doc_valid_1",
                        title="Valid Doc",
                        content="Valid content.",
                    )
                ],
                ground_truth_facts=[
                    GroundTruthFact(
                        fact_id="fact_01",
                        statement="Proposition",
                        cited_doc_id="doc_valid_1",
                        citation_index=5,
                    )
                ],
                clean_baseline_draft="Observed: Test [1].",
            )


# ---------------------------------------------------------------------------
# Quantitative Extraction and Verification Grounding Tests
# ---------------------------------------------------------------------------

class TestQuantitativeVerification:
    """Test quantitative numbers presence and deterministic citation grounding."""

    def test_fact_quantitative_numbers_in_source(self):
        """Quantitative values in facts are present in their cited document text."""
        scenarios = load_scenario_corpus()
        for s in scenarios:
            doc_map = {d.doc_id: d.content for d in s.documents}
            for fact in s.ground_truth_facts:
                content = doc_map[fact.cited_doc_id]
                for val in fact.quantitative_values:
                    val_str = str(int(val)) if val.is_integer() else str(val)
                    val_clean = f"{val:g}"
                    pct_str = f"{val:g}%"
                    curr_str = f"${val:g}"
                    found = (
                        val_str in content
                        or val_clean in content
                        or pct_str in content
                        or curr_str in content
                    )
                    assert found, (
                        f"Quantitative value {val} in fact '{fact.fact_id}' of scenario '{s.scenario_id}' "
                        f"not found in doc content: {content[:100]}..."
                    )

    def test_clean_baseline_drafts_pass_verify_citation_grounding(self):
        """All 27 clean baseline drafts pass deterministic verify_citation_grounding with 0 findings."""
        scenarios = load_scenario_corpus()
        for s in scenarios:
            sources = [
                SearchSource(
                    title=d.title,
                    url=f"local://{d.folder}/{d.doc_id}",
                    snippet=d.content,
                )
                for d in s.documents
            ]
            findings = verify_citation_grounding(s.clean_baseline_draft, sources)
            assert findings == [], (
                f"Scenario '{s.scenario_id}' clean baseline draft failed verify_citation_grounding: {findings}"
            )

    def test_clean_baseline_drafts_pass_run_preflight_scan(self):
        """All 27 clean baseline drafts pass run_preflight_scan with 0 hard findings."""
        scenarios = load_scenario_corpus()
        for s in scenarios:
            sources = [
                SearchSource(
                    title=d.title,
                    url=f"local://{d.folder}/{d.doc_id}",
                    snippet=d.content,
                )
                for d in s.documents
            ]
            has_hard_violations, findings = run_preflight_scan(s.clean_baseline_draft, sources)
            assert has_hard_violations is False, (
                f"Scenario '{s.scenario_id}' clean baseline draft triggered hard pre-flight violations: {findings}"
            )
            assert len(findings) == 0


# ---------------------------------------------------------------------------
# Corpus Validation Report Tests
# ---------------------------------------------------------------------------

class TestCorpusValidationReport:
    """Test validate_scenario_corpus logic and error diagnostics."""

    def test_validate_corpus_full_success(self):
        """Validating full corpus produces valid=True, 0 errors, 0 warnings."""
        scenarios = load_scenario_corpus()
        report = validate_scenario_corpus(scenarios)
        assert isinstance(report, CorpusValidationReport)
        assert report.valid is True
        assert report.total_scenarios == 27
        assert report.total_documents == 27
        assert report.total_facts >= 50
        assert report.total_quantitative_values >= 150
        assert report.errors == []
        assert report.warnings == []

    def test_validate_empty_corpus(self):
        """Validating empty scenario list produces valid=False."""
        report = validate_scenario_corpus([])
        assert report.valid is False
        assert report.total_scenarios == 0
        assert "Corpus is empty." in report.errors

    def test_validate_duplicate_scenario_id_detection(self):
        """Corpus validator detects duplicate scenario IDs."""
        s = load_scenario_corpus()[0]
        duplicate_corpus = [s, s]
        report = validate_scenario_corpus(duplicate_corpus)
        assert report.valid is False
        assert any("Duplicate scenario_id" in err for err in report.errors)


# ---------------------------------------------------------------------------
# Helper Retrieval Functions Tests
# ---------------------------------------------------------------------------

class TestCorpusHelperFunctions:
    """Test get_scenario and get_scenarios_by_domain helpers."""

    def test_get_scenario_by_id(self):
        """get_scenario retrieves the requested scenario by ID."""
        scenario = get_scenario("bio_01_oncology_rct")
        assert scenario.scenario_id == "bio_01_oncology_rct"
        assert scenario.domain == "biomedical"
        assert "TRIO-4" in scenario.title

    def test_get_scenario_not_found_raises_key_error(self):
        """get_scenario raises KeyError when ID does not exist."""
        with pytest.raises(KeyError, match="not found in corpus"):
            get_scenario("non_existent_scenario_99")

    def test_get_scenarios_by_domain(self):
        """get_scenarios_by_domain retrieves only scenarios from the specified domain."""
        bio_scenarios = get_scenarios_by_domain("biomedical")
        assert len(bio_scenarios) == 6
        assert all(s.domain == "biomedical" for s in bio_scenarios)

        legal_scenarios = get_scenarios_by_domain("legal")
        assert len(legal_scenarios) == 5
        assert all(s.domain == "legal" for s in legal_scenarios)


# ---------------------------------------------------------------------------
# Adversarial Data Models & Enums Tests
# ---------------------------------------------------------------------------

class TestAdversarialModelsAndEnums:
    """Test attack vector enums, difficulty tiers, mutation configs, and test cases."""

    def test_enums_values(self):
        """Enum values match expected taxonomy strings."""
        assert DomainEnum.BIOMEDICAL.value == "biomedical"
        assert AttackVectorEnum.STATISTICAL_FALLACY.value == "statistical_fallacy"
        assert AttackVectorEnum.PROMPT_INJECTION.value == "prompt_injection"
        assert AttackVectorEnum.NUMERIC_TEMPORAL_DRIFT.value == "numeric_temporal_drift"
        assert AttackVectorEnum.SYNTACTIC_ENTANGLEMENT.value == "syntactic_entanglement"
        assert AttackVectorEnum.CITATION_DRIFT.value == "citation_drift"
        assert AttackVectorEnum.POISONING_SATURATION.value == "poisoning_saturation"
        assert DifficultyTierEnum.MILD.value == "mild"
        assert DifficultyTierEnum.MODERATE.value == "moderate"
        assert DifficultyTierEnum.EXTREME.value == "extreme"
        assert DifficultyTierEnum.BREAKING.value == "breaking"

    def test_adversarial_mutation_config_interop(self):
        """AdversarialMutationConfig properties work as expected."""
        config = AdversarialMutationConfig(
            attack_vector="numeric_temporal_drift",
            difficulty_tier="moderate",
            target_unbacked_ratio=0.25,
            target_hard_violations=1,
            mutation_parameters={"syntactic_depth": 3},
        )
        assert config.vector == "numeric_temporal_drift"
        assert config.tier == "moderate"
        assert config.density == 0.25
        assert config.depth == 3

    def test_poisoned_scenario_case_interop(self):
        """PoisonedScenarioCase properties and alias mapping work cleanly."""
        doc = DocumentRecord(doc_id="d1", title="T1", content="C1")
        outcome = ExpectedDefenseOutcome(
            expected_verdict="FAIL",
            expected_arbiter_decision="BLOCK",
            expected_violations=["T1"],
            must_trigger_preflight_catch=True,
        )
        case = PoisonedScenarioCase(
            test_id="test_001",
            scenario_id="bio_01_oncology_rct",
            attack_vector="prompt_injection",
            difficulty_tier="extreme",
            mutated_prompt="Adversarial prompt",
            mutated_draft="Adversarial draft [1]",
            evidence_documents=[doc],
            expected_defense=outcome,
        )
        assert case.original_query == "Adversarial prompt"
        assert case.mutated_documents == [doc]
        assert case.expected_outcome.expected_verdict == "FAIL"
        assert case.expected_outcome.expected_arbiter_decision == "BLOCK"
        assert case.expected_outcome.must_trigger_preflight_catch is True
