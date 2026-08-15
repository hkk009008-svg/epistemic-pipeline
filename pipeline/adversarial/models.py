"""Data models and validation functions for Multi-Domain Adversarial Stress-Testing."""
from __future__ import annotations

import json
import re
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class DomainEnum(str, Enum):
    """Canonical 5 knowledge domains for epistemic stress testing."""
    BIOMEDICAL = "biomedical"
    FINANCIAL = "financial"
    LEGAL = "legal"
    CRYPTOGRAPHIC = "cryptographic"
    AUTONOMOUS_CONTRACTS = "autonomous_contracts"


DomainType = Literal[
    "biomedical",
    "financial",
    "legal",
    "cryptographic",
    "autonomous_contracts",
]


class AttackVectorEnum(str, Enum):
    """The 6 core attack vectors for adversarial evaluation."""
    STATISTICAL_FALLACY = "statistical_fallacy"
    PROMPT_INJECTION = "prompt_injection"
    NUMERIC_TEMPORAL_DRIFT = "numeric_temporal_drift"
    SYNTACTIC_ENTANGLEMENT = "syntactic_entanglement"
    CITATION_DRIFT = "citation_drift"
    POISONING_SATURATION = "poisoning_saturation"


AttackVectorType = Literal[
    "statistical_fallacy",
    "prompt_injection",
    "numeric_temporal_drift",
    "syntactic_entanglement",
    "citation_drift",
    "poisoning_saturation",
]


class DifficultyTierEnum(str, Enum):
    """The 4 difficulty tiers for adversarial attacks."""
    MILD = "mild"
    MODERATE = "moderate"
    EXTREME = "extreme"
    BREAKING = "breaking"


DifficultyTierType = Literal["mild", "moderate", "extreme", "breaking"]

ExpectedVerdictType = Literal["PASS", "FAIL", "ABSTAIN"]
ExpectedArbiterDecisionType = Literal[
    "ALLOW",
    "ALLOW_WITH_EDITS",
    "ALLOW_AS_UNKNOWN_ONLY",
    "BLOCK",
]


class GroundTruthFact(BaseModel):
    """An atomic, verifiable ground-truth proposition supported by source evidence."""
    fact_id: str = Field(..., description="Unique fact identifier (e.g. 'fact_01')")
    statement: str = Field(..., description="Canonical truthful atomic proposition")
    cited_doc_id: str = Field(..., description="ID of source document supporting this fact")
    citation_index: int = Field(..., ge=1, description="1-based citation index [N]")
    quantitative_values: list[float] = Field(default_factory=list, description="Extracted numbers (floats)")
    key_entities: list[str] = Field(default_factory=list, description="Core named entities / keywords")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata")


class DocumentRecord(BaseModel):
    """An authoritative source document or snippet in the evidence corpus."""
    doc_id: str = Field(..., description="Unique document ID (e.g. 'doc_bio_01')")
    title: str = Field(..., min_length=1, max_length=500)
    folder: str = Field(default="general", max_length=100)
    content: str = Field(..., min_length=1, description="Full authoritative text or snippet")
    source_sha256: str = Field(default="", description="Optional SHA256 digest of content")
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Interoperability properties
    @property
    def document_id(self) -> str:
        return self.doc_id

    @property
    def text(self) -> str:
        return self.content

    def to_search_source_dict(self, score: float = 1.0) -> dict:
        """Convert to SearchSource compatible dictionary."""
        return {
            "title": self.title,
            "url": f"local://{self.folder}/{self.doc_id}",
            "snippet": self.content,
            "score": score,
        }

    def to_search_source(self, score: float = 1.0) -> Any:
        """Convert to pipeline SearchSource instance."""
        from pipeline.models import SearchSource
        return SearchSource(
            title=self.title,
            url=f"local://{self.folder}/{self.doc_id}",
            snippet=self.content,
            score=score,
        )


class MultiDomainScenario(BaseModel):
    """A complete, verified scenario across one of the 5 target domains."""
    scenario_id: str = Field(..., pattern=r"^[a-z0-9_]+$", description="Unique scenario identifier")
    domain: DomainType = Field(..., description="One of the 5 canonical domains")
    subdomain: str = Field(..., description="Specialized topic area")
    title: str = Field(..., description="Descriptive scenario title")
    query: str = Field(..., description="Canonical user prompt or research question")
    documents: list[DocumentRecord] = Field(..., min_length=1, description="Immutable evidence documents")
    ground_truth_facts: list[GroundTruthFact] = Field(..., min_length=1, description="Discrete atomic ground-truth facts")
    clean_baseline_draft: str = Field(..., description="Golden verified clean answer draft")
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.scenario_id

    @property
    def clean_draft(self) -> str:
        return self.clean_baseline_draft

    @model_validator(mode="after")
    def validate_citation_integrity(self) -> MultiDomainScenario:
        """Verify citation indices within facts match available documents."""
        doc_ids = {d.doc_id for d in self.documents}
        num_docs = len(self.documents)
        for fact in self.ground_truth_facts:
            if fact.cited_doc_id not in doc_ids:
                raise ValueError(
                    f"Fact {fact.fact_id} cites non-existent doc_id '{fact.cited_doc_id}'. "
                    f"Available docs: {list(doc_ids)}"
                )
            if fact.citation_index < 1 or fact.citation_index > num_docs:
                raise ValueError(
                    f"Fact {fact.fact_id} citation_index {fact.citation_index} "
                    f"out of range [1, {num_docs}]"
                )
        return self


class AdversarialMutationConfig(BaseModel):
    """Configuration specifying how to mutate a clean scenario into an attack."""
    attack_vector: AttackVectorType
    difficulty_tier: DifficultyTierType
    target_unbacked_ratio: float = Field(0.0, ge=0.0, le=1.0)
    target_hard_violations: int = Field(0, ge=0)
    injection_payload: str | None = None
    mutation_parameters: dict[str, Any] = Field(default_factory=dict)

    # Aliases for interface contracts
    @property
    def vector(self) -> AttackVectorType:
        return self.attack_vector

    @property
    def tier(self) -> DifficultyTierType:
        return self.difficulty_tier

    @property
    def density(self) -> float:
        return self.target_unbacked_ratio

    @property
    def depth(self) -> int:
        return self.mutation_parameters.get("syntactic_depth", 1)


class ExpectedDefenseOutcome(BaseModel):
    """The precise expected defense behavior of the epistemic pipeline."""
    expected_verdict: ExpectedVerdictType
    expected_arbiter_decision: ExpectedArbiterDecisionType
    expected_violations: list[str] = Field(default_factory=list)
    must_trigger_preflight_catch: bool = False
    must_trigger_poisoning_guard: bool = False
    min_confidence_tier: str | None = None
    target_unbacked_ratio: float = Field(0.0, ge=0.0, le=1.0)
    target_hard_violations: int = Field(0, ge=0)


class PoisonedScenarioCase(BaseModel):
    """An evaluated test case consisting of mutated text, attack metadata, and expected defense."""
    test_id: str
    scenario_id: str
    attack_vector: AttackVectorType
    difficulty_tier: DifficultyTierType
    mutated_prompt: str
    mutated_draft: str
    evidence_documents: list[DocumentRecord]
    expected_defense: ExpectedDefenseOutcome
    mutation_notes: list[str] = Field(default_factory=list)
    mutation_parameters: dict[str, Any] = Field(default_factory=dict)

    # Interoperability aliases
    @property
    def original_query(self) -> str:
        return self.mutated_prompt

    @property
    def mutated_documents(self) -> list[DocumentRecord]:
        return self.evidence_documents

    @property
    def expected_outcome(self) -> ExpectedDefenseOutcome:
        return self.expected_defense


class CorpusValidationReport(BaseModel):
    """Validation report summarizing corpus health, integrity, and domain coverage."""
    valid: bool
    total_scenarios: int
    domains_covered: dict[str, int]
    total_documents: int = 0
    total_facts: int = 0
    total_quantitative_values: int = 0
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def _resolve_corpus_path(path: str | Path | None = None) -> Path:
    """Resolve the path to knowledge_data/corpus/scenarios.json."""
    if path is not None:
        p = Path(path)
        if p.is_file():
            return p
        if p.is_dir():
            return p / "scenarios.json"
        return p

    # Standard lookup locations
    candidates = [
        Path("knowledge_data/corpus/scenarios.json"),
        Path("/Users/hyungkoookkim/epistemic-pipeline/knowledge_data/corpus/scenarios.json"),
        Path(__file__).resolve().parent.parent.parent / "knowledge_data" / "corpus" / "scenarios.json",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return candidates[0]


def load_scenario_corpus(path: str | Path | None = None) -> list[MultiDomainScenario]:
    """Load and parse all scenarios from the corpus JSON file."""
    corpus_path = _resolve_corpus_path(path)
    if not corpus_path.is_file():
        raise FileNotFoundError(f"Corpus file not found at: {corpus_path}")

    with open(corpus_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "scenarios" in data:
        raw_scenarios = data["scenarios"]
    elif isinstance(data, list):
        raw_scenarios = data
    else:
        raise ValueError("Corpus file format invalid: expected list or object with 'scenarios' list.")

    scenarios = [MultiDomainScenario.model_validate(item) for item in raw_scenarios]
    return scenarios


def get_scenario(scenario_id: str, corpus_path: str | Path | None = None) -> MultiDomainScenario:
    """Retrieve a single scenario by its unique scenario_id."""
    scenarios = load_scenario_corpus(corpus_path)
    for s in scenarios:
        if s.scenario_id == scenario_id:
            return s
    raise KeyError(f"Scenario with ID '{scenario_id}' not found in corpus.")


def get_scenarios_by_domain(domain: str, corpus_path: str | Path | None = None) -> list[MultiDomainScenario]:
    """Retrieve all scenarios belonging to a specific domain."""
    scenarios = load_scenario_corpus(corpus_path)
    return [s for s in scenarios if s.domain == domain]


def validate_scenario_corpus(scenarios: list[MultiDomainScenario]) -> CorpusValidationReport:
    """Perform rigorous validation across all scenarios in the corpus."""
    errors: list[str] = []
    warnings: list[str] = []
    domains_covered: dict[str, int] = {
        DomainEnum.BIOMEDICAL.value: 0,
        DomainEnum.FINANCIAL.value: 0,
        DomainEnum.LEGAL.value: 0,
        DomainEnum.CRYPTOGRAPHIC.value: 0,
        DomainEnum.AUTONOMOUS_CONTRACTS.value: 0,
    }

    if not scenarios:
        return CorpusValidationReport(
            valid=False,
            total_scenarios=0,
            domains_covered=domains_covered,
            errors=["Corpus is empty."],
        )

    seen_ids = set()
    total_docs = 0
    total_facts = 0
    total_quant = 0

    for idx, s in enumerate(scenarios):
        # Unique ID check
        if s.scenario_id in seen_ids:
            errors.append(f"Duplicate scenario_id '{s.scenario_id}' at index {idx}.")
        seen_ids.add(s.scenario_id)

        # Domain tracking
        if s.domain in domains_covered:
            domains_covered[s.domain] += 1
        else:
            errors.append(f"Scenario '{s.scenario_id}' has unrecognized domain '{s.domain}'.")

        # Document checks
        if not s.documents:
            errors.append(f"Scenario '{s.scenario_id}' has no documents.")
        else:
            total_docs += len(s.documents)
            doc_ids = set()
            for d in s.documents:
                if d.doc_id in doc_ids:
                    errors.append(f"Scenario '{s.scenario_id}' has duplicate doc_id '{d.doc_id}'.")
                doc_ids.add(d.doc_id)
                if not d.content.strip():
                    errors.append(f"Scenario '{s.scenario_id}' document '{d.doc_id}' has empty content.")
                if not d.title.strip():
                    warnings.append(f"Scenario '{s.scenario_id}' document '{d.doc_id}' has empty title.")

        # Ground truth fact checks
        if not s.ground_truth_facts:
            errors.append(f"Scenario '{s.scenario_id}' has no ground truth facts.")
        else:
            total_facts += len(s.ground_truth_facts)
            fact_ids = set()
            for f in s.ground_truth_facts:
                if f.fact_id in fact_ids:
                    errors.append(f"Scenario '{s.scenario_id}' has duplicate fact_id '{f.fact_id}'.")
                fact_ids.add(f.fact_id)
                total_quant += len(f.quantitative_values)

                # Check citation validity
                doc_map = {d.doc_id: d for d in s.documents}
                if f.cited_doc_id not in doc_map:
                    errors.append(
                        f"Scenario '{s.scenario_id}' fact '{f.fact_id}' references missing doc_id '{f.cited_doc_id}'."
                    )
                else:
                    doc = doc_map[f.cited_doc_id]
                    # Verify numeric presence in doc content if quantitative values exist
                    for qv in f.quantitative_values:
                        # Check as integer string, float string, or formatted representation
                        qv_str = str(int(qv)) if qv.is_integer() else str(qv)
                        # Remove trailing zeros for floats like 0.68
                        qv_clean = f"{qv:g}"
                        if qv_str not in doc.content and qv_clean not in doc.content:
                            # Also check if it's formatted as percentage or dollar
                            pct_str = f"{qv:g}%"
                            curr_str = f"${qv:g}"
                            if pct_str not in doc.content and curr_str not in doc.content:
                                warnings.append(
                                    f"Scenario '{s.scenario_id}' fact '{f.fact_id}' quantitative value {qv} "
                                    f"not literally found in cited doc '{f.cited_doc_id}'."
                                )

        # Baseline draft check
        if not s.clean_baseline_draft.strip():
            errors.append(f"Scenario '{s.scenario_id}' has empty clean_baseline_draft.")
        else:
            # Check that citations in clean draft are valid indices
            citations = [int(m) for m in re.findall(r"\[(\d+)\]", s.clean_baseline_draft)]
            for cit in citations:
                if cit < 1 or cit > len(s.documents):
                    errors.append(
                        f"Scenario '{s.scenario_id}' clean_baseline_draft contains out-of-range citation [{cit}]. "
                        f"Document count is {len(s.documents)}."
                    )

    # Minimum domain counts check (must have >= 5 in each of the 5 domains)
    for domain_name, count in domains_covered.items():
        if count < 5:
            errors.append(f"Domain '{domain_name}' has only {count} scenarios; minimum required is 5.")

    valid = len(errors) == 0

    return CorpusValidationReport(
        valid=valid,
        total_scenarios=len(scenarios),
        domains_covered=domains_covered,
        total_documents=total_docs,
        total_facts=total_facts,
        total_quantitative_values=total_quant,
        errors=errors,
        warnings=warnings,
    )
