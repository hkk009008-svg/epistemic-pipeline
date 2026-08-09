"""Deterministic tests for the offline grounded-RAG evaluation contract."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts.evaluate_grounded_rag import (
    DEFAULT_CASES,
    EvaluationCase,
    evaluate_cases,
    load_cases,
    main,
)


def test_canonical_fixture_metrics_are_reproducible():
    summary = evaluate_cases(load_cases(DEFAULT_CASES))

    assert summary.case_count == 12
    assert summary.retrieval_k_values == [1, 2]
    assert summary.retrieval_recall_at_k.numerator == 11
    assert summary.retrieval_recall_at_k.denominator == 13
    assert summary.retrieval_recall_at_k.rate == pytest.approx(11 / 13)

    assert summary.citation_validity.numerator == 3
    assert summary.citation_validity.denominator == 8
    assert summary.citation_validity.rate == pytest.approx(3 / 8)
    assert summary.citation_completeness.rate == pytest.approx(3 / 8)

    assert summary.unsupported_released_claim_rate.numerator == 4
    assert summary.unsupported_released_claim_rate.denominator == 8
    assert summary.unsupported_released_claim_rate.rate == pytest.approx(0.5)
    assert summary.correct_abstention_rate.rate == pytest.approx(1.0)
    assert summary.false_abstention_rate.numerator == 2
    assert summary.false_abstention_rate.denominator == 10

    assert summary.latency_ms.sample_count == 12
    assert summary.latency_ms.total == pytest.approx(1540.0)
    assert summary.latency_ms.mean == pytest.approx(1540.0 / 12)
    assert summary.latency_ms.p50 == pytest.approx(140.0)
    assert summary.latency_ms.p95 == pytest.approx(190.0)
    assert summary.cost_usd.sample_count == 11
    assert summary.cost_usd.total == pytest.approx(0.03)
    assert summary.cost_usd.p50 == pytest.approx(0.003)
    assert summary.cost_usd.p95 == pytest.approx(0.0045)


def test_fixture_schema_rejects_unknown_fields():
    payload = json.loads(DEFAULT_CASES.read_text(encoding="utf-8").splitlines()[0])
    payload["recorded"]["verifier_said_it_was_fine"] = True

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        EvaluationCase.model_validate(payload)


def test_fixture_schema_rejects_self_inconsistent_records():
    payload = json.loads(DEFAULT_CASES.read_text(encoding="utf-8").splitlines()[0])
    payload["recorded"]["status"] = "ABSTAIN"

    with pytest.raises(ValidationError, match="abstention cannot release claims"):
        EvaluationCase.model_validate(payload)


def test_supporting_citation_must_be_in_actual_retrieval_set():
    payload = json.loads(DEFAULT_CASES.read_text(encoding="utf-8").splitlines()[0])
    payload["recorded"]["released_claims"][0]["citations"][0][
        "evidence_id"
    ] = "E-not-retrieved"

    with pytest.raises(ValidationError, match="must resolve to retrieved evidence"):
        EvaluationCase.model_validate(payload)


def test_duplicate_citation_ids_are_rejected_within_a_claim():
    payload = json.loads(DEFAULT_CASES.read_text(encoding="utf-8").splitlines()[0])
    citation = payload["recorded"]["released_claims"][0]["citations"][0]
    payload["recorded"]["released_claims"][0]["citations"].append(dict(citation))

    with pytest.raises(ValidationError, match="citation evidence IDs must be unique"):
        EvaluationCase.model_validate(payload)


def test_unsupported_claim_cannot_label_a_citation_as_supporting():
    payload = json.loads(DEFAULT_CASES.read_text(encoding="utf-8").splitlines()[0])
    payload["recorded"]["released_claims"][0]["corpus_supported"] = False

    with pytest.raises(
        ValidationError,
        match="corpus-unsupported claim cannot have a supporting citation",
    ):
        EvaluationCase.model_validate(payload)


def test_retrieved_distractor_does_not_score_as_a_valid_citation():
    payload = json.loads(DEFAULT_CASES.read_text(encoding="utf-8").splitlines()[0])
    payload["recorded"]["released_claims"][0]["citations"][0][
        "evidence_id"
    ] = "E-distractor"
    case = EvaluationCase.model_validate(payload)

    summary = evaluate_cases([case])

    assert summary.citation_validity.numerator == 0
    assert summary.citation_validity.denominator == 1
    assert summary.citation_completeness.numerator == 0


def test_non_finite_measurements_are_rejected():
    payload = json.loads(DEFAULT_CASES.read_text(encoding="utf-8").splitlines()[0])
    payload["recorded"]["latency_ms"] = float("inf")

    with pytest.raises(ValidationError, match="finite number"):
        EvaluationCase.model_validate(payload)


def test_loader_reports_duplicate_case_id_with_line_context(tmp_path: Path):
    line = DEFAULT_CASES.read_text(encoding="utf-8").splitlines()[0]
    cases_file = tmp_path / "duplicates.jsonl"
    cases_file.write_text(f"{line}\n{line}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"duplicate case_id.*:2"):
        load_cases(cases_file)


def test_cli_json_output_is_machine_readable(capsys):
    assert main([str(DEFAULT_CASES), "--json"]) == 0
    output = json.loads(capsys.readouterr().out)

    assert output["schema_version"] == "grounded-rag-eval-summary-v1"
    assert output["unsupported_released_claim_rate"] == {
        "denominator": 8,
        "numerator": 4,
        "rate": pytest.approx(0.5),
    }
