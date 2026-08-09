#!/usr/bin/env python3
"""Score recorded grounded-RAG observations without running the pipeline.

The input is JSON Lines using the closed ``grounded-rag-eval-v1`` schema.  A
record producer must independently label support and export actual retrieval
IDs. The bundled hand-authored fixture tests this schema and metric arithmetic;
it is not a measured system-performance baseline.

Usage::

    python scripts/evaluate_grounded_rag.py
    python scripts/evaluate_grounded_rag.py path/to/cases.jsonl --json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CASES = (
    REPOSITORY_ROOT / "tests" / "fixtures" / "grounded_rag_v1" / "cases.jsonl"
)


class ClosedModel(BaseModel):
    """Base for versioned fixture objects that reject undeclared fields."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class ExpectedOutcome(ClosedModel):
    must_abstain: bool
    relevant_evidence_ids: list[str] = Field(default_factory=list, max_length=24)

    @field_validator("relevant_evidence_ids")
    @classmethod
    def evidence_ids_are_unique(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("relevant_evidence_ids must be unique")
        return values

    @model_validator(mode="after")
    def answerable_cases_have_evidence(self) -> ExpectedOutcome:
        if not self.must_abstain and not self.relevant_evidence_ids:
            raise ValueError("an answerable case needs at least one relevant evidence ID")
        return self


class CitationObservation(ClosedModel):
    evidence_id: str = Field(..., min_length=1, max_length=80)
    supports_claim: bool


class ReleasedClaimObservation(ClosedModel):
    claim_id: str = Field(..., min_length=1, max_length=32)
    corpus_supported: bool
    citations: list[CitationObservation] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def citation_labels_are_consistent(self) -> ReleasedClaimObservation:
        evidence_ids = [citation.evidence_id for citation in self.citations]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("citation evidence IDs must be unique within a claim")
        if not self.corpus_supported and any(
            citation.supports_claim for citation in self.citations
        ):
            raise ValueError(
                "a corpus-unsupported claim cannot have a supporting citation"
            )
        return self


class RecordedOutcome(ClosedModel):
    status: Literal["ANSWER", "PARTIAL", "ABSTAIN"]
    retrieval_k: int = Field(..., ge=1, le=100)
    retrieved_evidence_ids: list[str] = Field(default_factory=list, max_length=100)
    released_claims: list[ReleasedClaimObservation] = Field(
        default_factory=list,
        max_length=24,
    )
    latency_ms: float | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)

    @field_validator("retrieved_evidence_ids")
    @classmethod
    def retrieved_ids_are_unique(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("retrieved_evidence_ids must be unique")
        return values

    @field_validator("released_claims")
    @classmethod
    def claim_ids_are_unique(
        cls, values: list[ReleasedClaimObservation]
    ) -> list[ReleasedClaimObservation]:
        claim_ids = [claim.claim_id for claim in values]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("released claim IDs must be unique")
        return values

    @model_validator(mode="after")
    def outcome_matches_release_contract(self) -> RecordedOutcome:
        if len(self.retrieved_evidence_ids) > self.retrieval_k:
            raise ValueError("retrieved_evidence_ids cannot exceed retrieval_k")
        if self.status == "ABSTAIN" and self.released_claims:
            raise ValueError("an abstention cannot release claims")
        if self.status != "ABSTAIN" and not self.released_claims:
            raise ValueError("ANSWER or PARTIAL must release at least one claim")
        retrieved_ids = set(self.retrieved_evidence_ids)
        for claim in self.released_claims:
            for citation in claim.citations:
                if (
                    citation.supports_claim
                    and citation.evidence_id not in retrieved_ids
                ):
                    raise ValueError(
                        "a supporting citation must resolve to retrieved evidence"
                    )
        return self


class EvaluationCase(ClosedModel):
    schema_version: Literal["grounded-rag-eval-v1"]
    case_id: str = Field(..., pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    category: Literal[
        "answerable",
        "unanswerable",
        "conflict",
        "distractor",
        "prompt_injection",
        "altered_fact",
        "altered_date",
        "polarity_reversal",
        "stale",
        "compound_claim",
        "citation_failure",
    ]
    query: str = Field(..., min_length=1, max_length=10_000)
    expected: ExpectedOutcome
    recorded: RecordedOutcome


class RateMetric(ClosedModel):
    numerator: int = Field(..., ge=0)
    denominator: int = Field(..., ge=0)
    rate: float | None = Field(..., ge=0, le=1)


class ValueMetric(ClosedModel):
    sample_count: int = Field(..., ge=0)
    total: float | None
    mean: float | None
    minimum: float | None
    maximum: float | None
    p50: float | None
    p95: float | None


class EvaluationSummary(ClosedModel):
    schema_version: Literal["grounded-rag-eval-summary-v1"] = (
        "grounded-rag-eval-summary-v1"
    )
    case_count: int = Field(..., ge=1)
    retrieval_k_values: list[int]
    retrieval_recall_at_k: RateMetric
    citation_validity: RateMetric
    citation_completeness: RateMetric
    unsupported_released_claim_rate: RateMetric
    correct_abstention_rate: RateMetric
    false_abstention_rate: RateMetric
    latency_ms: ValueMetric
    cost_usd: ValueMetric


def _rate(numerator: int, denominator: int) -> RateMetric:
    return RateMetric(
        numerator=numerator,
        denominator=denominator,
        rate=None if denominator == 0 else numerator / denominator,
    )


def _values(values: list[float]) -> ValueMetric:
    if not values:
        return ValueMetric(
            sample_count=0,
            total=None,
            mean=None,
            minimum=None,
            maximum=None,
            p50=None,
            p95=None,
        )
    ordered = sorted(values)

    def nearest_rank(percentile: float) -> float:
        """Return the 1-indexed nearest-rank percentile, ceil(p * n)."""
        return ordered[math.ceil(percentile * len(ordered)) - 1]

    total = sum(values)
    return ValueMetric(
        sample_count=len(values),
        total=total,
        mean=total / len(values),
        minimum=ordered[0],
        maximum=ordered[-1],
        p50=nearest_rank(0.50),
        p95=nearest_rank(0.95),
    )


def load_cases(path: Path) -> list[EvaluationCase]:
    """Load and validate JSONL cases, retaining line context on failure."""
    cases: list[EvaluationCase] = []
    seen_case_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line)
                case = EvaluationCase.model_validate(payload)
            except (json.JSONDecodeError, ValidationError) as exc:
                raise ValueError(f"invalid evaluation case at {path}:{line_number}: {exc}") from exc
            if case.case_id in seen_case_ids:
                raise ValueError(
                    f"duplicate case_id {case.case_id!r} at {path}:{line_number}"
                )
            seen_case_ids.add(case.case_id)
            cases.append(case)
    if not cases:
        raise ValueError(f"no evaluation cases found in {path}")
    return cases


def evaluate_cases(cases: Sequence[EvaluationCase]) -> EvaluationSummary:
    """Compute metrics over recorded, independently labeled observations."""
    if not cases:
        raise ValueError("at least one evaluation case is required")

    relevant_total = 0
    relevant_retrieved = 0
    citation_total = 0
    valid_citations = 0
    released_claims = 0
    claims_with_valid_citation = 0
    unsupported_claims = 0
    expected_abstentions = 0
    correct_abstentions = 0
    answerable_cases = 0
    false_abstentions = 0
    latencies: list[float] = []
    costs: list[float] = []

    for case in cases:
        expected_ids = set(case.expected.relevant_evidence_ids)
        retrieved_ids = set(case.recorded.retrieved_evidence_ids)
        relevant_total += len(expected_ids)
        relevant_retrieved += len(expected_ids & retrieved_ids)

        if case.expected.must_abstain:
            expected_abstentions += 1
            correct_abstentions += int(case.recorded.status == "ABSTAIN")
        else:
            answerable_cases += 1
            false_abstentions += int(case.recorded.status == "ABSTAIN")

        for claim in case.recorded.released_claims:
            released_claims += 1
            unsupported_claims += int(not claim.corpus_supported)
            claim_has_valid_citation = False
            for citation in claim.citations:
                citation_total += 1
                is_valid = (
                    citation.evidence_id in retrieved_ids
                    and citation.evidence_id in expected_ids
                    and citation.supports_claim
                )
                valid_citations += int(is_valid)
                claim_has_valid_citation = claim_has_valid_citation or is_valid
            claims_with_valid_citation += int(claim_has_valid_citation)

        if case.recorded.latency_ms is not None:
            latencies.append(case.recorded.latency_ms)
        if case.recorded.cost_usd is not None:
            costs.append(case.recorded.cost_usd)

    return EvaluationSummary(
        case_count=len(cases),
        retrieval_k_values=sorted({case.recorded.retrieval_k for case in cases}),
        retrieval_recall_at_k=_rate(relevant_retrieved, relevant_total),
        citation_validity=_rate(valid_citations, citation_total),
        citation_completeness=_rate(claims_with_valid_citation, released_claims),
        unsupported_released_claim_rate=_rate(unsupported_claims, released_claims),
        correct_abstention_rate=_rate(correct_abstentions, expected_abstentions),
        false_abstention_rate=_rate(false_abstentions, answerable_cases),
        latency_ms=_values(latencies),
        cost_usd=_values(costs),
    )


def _format_rate(metric: RateMetric) -> str:
    if metric.rate is None:
        return f"{metric.numerator}/{metric.denominator} (n/a)"
    return f"{metric.numerator}/{metric.denominator} ({metric.rate:.2%})"


def _print_human(summary: EvaluationSummary) -> None:
    print(f"Grounded RAG evaluation: {summary.case_count} cases")
    print(
        "Retrieval recall@k:          "
        f"{_format_rate(summary.retrieval_recall_at_k)} "
        f"(k values: {summary.retrieval_k_values})"
    )
    print(f"Citation validity:             {_format_rate(summary.citation_validity)}")
    print(f"Citation completeness:         {_format_rate(summary.citation_completeness)}")
    print(
        "Unsupported released claims: "
        f"{_format_rate(summary.unsupported_released_claim_rate)}"
    )
    print(f"Correct abstention:            {_format_rate(summary.correct_abstention_rate)}")
    print(f"False abstention:              {_format_rate(summary.false_abstention_rate)}")
    if summary.latency_ms.mean is not None:
        print(
            f"Latency: {summary.latency_ms.mean:.2f} ms mean "
            f"(p50 {summary.latency_ms.p50:.2f}, p95 {summary.latency_ms.p95:.2f}) "
            f"across {summary.latency_ms.sample_count} recorded cases"
        )
    else:
        print("Latency: not recorded")
    if summary.cost_usd.total is not None:
        print(
            f"Cost: ${summary.cost_usd.total:.6f} total "
            f"(p50 ${summary.cost_usd.p50:.6f}, p95 ${summary.cost_usd.p95:.6f}) "
            f"across {summary.cost_usd.sample_count} recorded cases"
        )
    else:
        print("Cost: not recorded")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score closed-schema recorded grounded-RAG cases (offline only)."
    )
    parser.add_argument(
        "cases_file",
        nargs="?",
        type=Path,
        default=DEFAULT_CASES,
        help=f"JSONL cases (default: {DEFAULT_CASES.relative_to(REPOSITORY_ROOT)})",
    )
    parser.add_argument("--json", action="store_true", help="emit the summary as JSON")
    args = parser.parse_args(argv)

    try:
        summary = evaluate_cases(load_cases(args.cases_file))
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(summary.model_dump(mode="json"), indent=2, sort_keys=True))
    else:
        _print_human(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
