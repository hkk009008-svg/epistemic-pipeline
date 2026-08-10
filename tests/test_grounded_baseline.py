"""Deterministic contract tests for the private measured-baseline harness."""
from __future__ import annotations

import hashlib
import json
import stat
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import pipeline.grounded_evaluation as evaluation
import pipeline.grounded_rag as grounded
import scripts.measure_grounded_rag as measure_grounded_rag
from pipeline.agy_evaluation_provider import AgyEvaluationProviderError
from pipeline.grounded_rag import (
    AnswererOutput,
    FinalizerOutput,
    GroundedQueryRequest,
    GroundedRecordedExecution,
    GroundedRecordedStageIdentity,
    VerifierOutput,
    run_grounded_rag,
    run_grounded_rag_recorded,
)
from pipeline.knowledge_store import EvidenceItem, EvidencePacket, KnowledgeStore


BLUE_QUOTE = "Alice's favorite color is blue."
YEAR_QUOTE = "Alice started using it in 2024."


def _packet(*, with_items: bool = True) -> EvidencePacket:
    items: list[EvidenceItem] = []
    if with_items:
        text = f"{BLUE_QUOTE} {YEAR_QUOTE}"
        items.append(
            EvidenceItem(
                evidence_id="E-source-1",
                rank=1,
                retrieval_score=-1.0,
                document_id="profile",
                document_revision_id="d" * 64,
                folder="personal",
                title="Profile",
                relative_path="sources/personal/profile/versions/abc.txt",
                source_sha256="a" * 64,
                chunk_sha256="b" * 64,
                start_char=0,
                end_char=len(text),
                start_line=1,
                end_line=1,
                text=text,
            )
        )
    return KnowledgeStore._build_packet(
        "what is alice favorite color",
        "c" * 64,
        items,
    )


class StubStore:
    def __init__(self, packet: EvidencePacket):
        self.packet = packet
        self.receipts: list[dict] = []

    def retrieve(self, query: str, top_k: int) -> EvidencePacket:
        return self.packet

    def append_run_receipt(self, receipt: dict) -> str:
        self.receipts.append(receipt)
        return hashlib.sha256(_json_bytes(receipt)).hexdigest()


def _task_from_prompt(user_content: str) -> dict:
    marker = "=== TASK_DATA ===\n"
    body = user_content.split(marker, 1)[1].split(
        "\n=== END TASK_DATA ===",
        1,
    )[0]
    return json.loads(body)


def _install_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        grounded.config,
        "get_stage_config",
        lambda stage: {
            "provider": "test-provider",
            "api_key": "secret-not-for-artifacts",
            "model": stage,
            "base_url": "https://private.invalid",
        },
    )


def _supported_call(
    packet: EvidencePacket,
    *,
    verifier_quote: str = BLUE_QUOTE,
    second_claim: bool = False,
    final_decision: str = "ANSWER",
    final_ids: list[str] | None = None,
) -> Callable:
    async def call(cfg, system, user_content, response_model):
        task = _task_from_prompt(user_content)
        if response_model is AnswererOutput:
            claims = [
                {
                    "text": BLUE_QUOTE,
                    "citations": [
                        {"evidence_id": "E-source-1", "quote": BLUE_QUOTE}
                    ],
                }
            ]
            if second_claim:
                claims.append(
                    {
                        "text": "Alice selected blue in 2023.",
                        "citations": [
                            {"evidence_id": "E-source-1", "quote": YEAR_QUOTE}
                        ],
                    }
                )
            return AnswererOutput(
                packet_id=packet.packet_id,
                answerability="ANSWERABLE",
                claims=claims,
            )
        if response_model is VerifierOutput:
            checks = [
                {
                    "claim_id": "C1",
                    "verdict": "SUPPORTED",
                    "support_spans": [
                        {"evidence_id": "E-source-1", "quote": verifier_quote}
                    ],
                }
            ]
            if second_claim:
                checks.append(
                    {
                        "claim_id": "C2",
                        "verdict": "CONTRADICTED",
                        "support_spans": [],
                    }
                )
            return VerifierOutput(
                packet_id=packet.packet_id,
                draft_hash=task["draft_hash"],
                checks=checks,
            )
        selected = final_ids if final_ids is not None else ["C1"]
        return FinalizerOutput(
            packet_id=packet.packet_id,
            draft_hash=task["verification"]["draft_hash"],
            verification_hash=task["verification_hash"],
            decision=final_decision,
            included_claim_ids=selected,
        )

    return call


def _freeze_execution_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(grounded.uuid, "uuid4", lambda: SimpleNamespace(hex="1" * 32))


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _write_json(path: Path, payload: dict) -> str:
    raw = _json_bytes(payload)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _agy_invocation(
    stage: str,
    captured: dict | None = None,
    *,
    worker_source_sha256: str = "c" * 64,
) -> dict:
    captured = captured or {
        "system_sha256": "e" * 64,
        "user_sha256": "f" * 64,
        "response_schema_sha256": "1" * 64,
        "response_sha256": "2" * 64,
    }
    return {
        "schema_version": "grounded-provider-invocation-v1",
        "stage": stage,
        "provider": "google-antigravity-sdk",
        "requested_model": "gemini-evaluation-test",
        "reported_model": None,
        "model_attestation": "REQUESTED_ONLY",
        "sdk_distribution": "google-antigravity",
        "sdk_version": "0.1.10",
        "sdk_artifact_sha256": "a" * 64,
        "localharness_sha256": "b" * 64,
        "worker_protocol_version": "agy-evaluation-worker-v1",
        "worker_source_sha256": worker_source_sha256,
        "invocation_policy_sha256": evaluation.AGY_INVOCATION_POLICY_SHA256,
        "system_sha256": captured["system_sha256"],
        "user_sha256": captured["user_sha256"],
        "response_schema_sha256": captured["response_schema_sha256"],
        "response_sha256": captured["response_sha256"],
        "duration_ms": 4.0,
        "outcome": "SUCCESS",
        "logical_model_calls": 1,
        "adapter_retries": 0,
        "observed_tool_names": ["finish"],
        "usage": {
            "status": "REPORTED",
            "input_tokens": 10,
            "output_tokens": 4,
            "total_tokens": 16,
            "cached_input_tokens": 2,
            "reasoning_tokens": 2,
            "service_tier": "standard",
        },
        "cost_usd": None,
    }


def _benchmark_payload() -> dict:
    return {
        "schema_version": "grounded-rag-benchmark-v1",
        "benchmark_id": "unit-grounded-baseline",
        "data_classification": "synthetic_non_sensitive",
        "label_authority": "human_owned_external",
        "top_k": 2,
        "documents": [
            {
                "document_id": "gold-document",
                "folder": "facts",
                "title": "Gold fact",
                "content": "The gold statement is correct. A distractor follows.",
            }
        ],
        "cases": [
            {
                "case_id": "answerable-case",
                "category": "answerable",
                "query": "What is the gold statement?",
                "must_abstain": False,
                "relevant_spans": [
                    {
                        "span_id": "gold-span",
                        "document_id": "gold-document",
                        "quote": "The gold statement is correct.",
                    }
                ],
            },
            {
                "case_id": "must-abstain-case",
                "category": "unanswerable",
                "query": "What is not present?",
                "must_abstain": True,
                "relevant_spans": [],
            },
        ],
        "thresholds": {
            "retrieval_recall_at_k_min": 1.0,
            "unsupported_released_claim_rate_max": 0.0,
            "citation_validity_min": 1.0,
            "citation_completeness_min": 1.0,
            "correct_abstention_rate_min": 1.0,
            "false_abstention_rate_max": 0.0,
            "answer_coverage_min": 1.0,
            "verifier_supported_precision_min": 1.0,
            "verifier_supported_recall_min": 1.0,
        },
        "execution_policy": {
            "all_cases_required": True,
            "provider_error_state": "INCOMPLETE",
            "raw_provider_transcripts": "FORBIDDEN",
            "prompts_and_reasoning": "FORBIDDEN",
            "usage_unavailable_is_null": True,
            "live_execution_requires_separate_authorization": True,
            "maximum_wall_time_seconds": 60,
            "maximum_cost_usd": 1.0,
        },
    }


def _load_unit_benchmark(tmp_path: Path) -> evaluation.ResolvedBenchmark:
    path = tmp_path / "benchmark.json"
    digest = _write_json(path, _benchmark_payload())
    return evaluation.load_benchmark(path, digest)


def _private_span(
    benchmark: evaluation.ResolvedBenchmark,
) -> evaluation.PrivateSpanObservation:
    gold = benchmark.gold_spans_by_case["answerable-case"][0]
    expected_documents, _ = evaluation._expected_corpus(benchmark)
    expected_document = expected_documents[gold.document_id]
    evidence_id = next(iter(expected_document.chunks_by_evidence_id))
    expected_chunk = expected_document.chunks_by_evidence_id[evidence_id]
    local_start = expected_chunk.text.find(gold.quote)
    return evaluation.PrivateSpanObservation(
        evidence_id=evidence_id,
        quote=gold.quote,
        valid=True,
        document_id=gold.document_id,
        document_revision_id=expected_document.revision_id,
        title=expected_document.definition.title,
        relative_path=expected_document.relative_path,
        source_sha256=gold.source_sha256,
        source_start_char=gold.source_start_char,
        source_end_char=gold.source_end_char,
        start_line=expected_chunk.start_line + expected_chunk.text.count(
            "\n", 0, local_start
        ),
        end_line=expected_chunk.start_line + expected_chunk.text.count(
            "\n", 0, local_start + len(gold.quote)
        ),
    )


def _execution_identity(
    *,
    response_run_id: str,
    packet_id: str,
    corpus_revision: str,
    stage_fingerprints: evaluation.StageFingerprintsObservation,
    stages_completed: list[str],
    draft_hash: str | None,
    verification_hash: str | None,
    selected_claim_ids: list[str],
    citation_evidence_ids: list[str],
    status: str,
    reason_code: str,
    draft_claim_count: int,
    supported_claim_count: int,
    latency_ms: int = 10,
    contradicted_claim_count: int = 0,
    conflict_claim_count: int = 0,
    insufficient_claim_count: int = 0,
) -> evaluation.ExecutionIdentityObservation:
    prompt_versions = evaluation.PromptVersionsObservation(
        answerer="grounded-answerer-v1",
        verifier="grounded-verifier-v1",
        finalizer="grounded-finalizer-v1",
    )
    receipt = evaluation.ProductionReceiptObservation(
        schema_version="grounded-run-receipt-v1",
        contract_version="grounded-rag-v1",
        run_id=response_run_id,
        created_at="2026-08-11T00:00:00+00:00",
        latency_ms=latency_ms,
        corpus_revision=corpus_revision,
        packet_id=packet_id,
        packet_schema_version=2,
        retrieval_version="sqlite-fts5-v2",
        chunker_version="words-180-overlap-30-chars-4000-v2",
        coverage_limited=False,
        coverage_reasons=[],
        prompt_versions=prompt_versions,
        stage_fingerprints=stage_fingerprints,
        stages_completed=stages_completed,
        draft_hash=draft_hash,
        verification_hash=verification_hash,
        selected_claim_ids=selected_claim_ids,
        citation_evidence_ids=citation_evidence_ids,
        status=status,
        reason_code=reason_code,
        draft_claim_count=draft_claim_count,
        supported_claim_count=supported_claim_count,
        contradicted_claim_count=contradicted_claim_count,
        conflict_claim_count=conflict_claim_count,
        insufficient_claim_count=insufficient_claim_count,
    )
    receipt_sha256 = hashlib.sha256(
        _json_bytes(receipt.model_dump(mode="json"))
    ).hexdigest()
    return evaluation.ExecutionIdentityObservation(
        response_run_id=response_run_id,
        contract_version="grounded-rag-v1",
        packet_schema_version=2,
        retrieval_version="sqlite-fts5-v2",
        chunker_version="words-180-overlap-30-chars-4000-v2",
        prompt_versions=prompt_versions,
        stage_fingerprints=stage_fingerprints,
        draft_hash=draft_hash,
        verification_hash=verification_hash,
        receipt_sha256=receipt_sha256,
        receipt=receipt,
    )


def _latency_observation(
    identity: evaluation.ExecutionIdentityObservation,
    *,
    retrieval_ms: float,
    answerer_ms: float | None,
    verifier_ms: float | None,
    finalizer_ms: float | None,
    total_ms: float,
) -> evaluation.StageLatencyObservation:
    return evaluation.StageLatencyObservation(
        retrieval_ms=retrieval_ms,
        answerer_ms=answerer_ms,
        verifier_ms=verifier_ms,
        finalizer_ms=finalizer_ms,
        total_ms=total_ms,
        capture_sha256=evaluation.latency_capture_sha256(
            identity.receipt_sha256,
            retrieval_ms=retrieval_ms,
            answerer_ms=answerer_ms,
            verifier_ms=verifier_ms,
            finalizer_ms=finalizer_ms,
            total_ms=total_ms,
        ),
    )


def _complete_observations(
    benchmark: evaluation.ResolvedBenchmark,
) -> evaluation.ObservationBundle:
    definition = benchmark.definition
    gold = benchmark.gold_spans_by_case["answerable-case"][0]
    document = definition.documents[0]
    expected_documents, corpus_revision = evaluation._expected_corpus(benchmark)
    expected_document = expected_documents[document.document_id]
    expected_chunk = next(iter(expected_document.chunks_by_evidence_id.values()))
    expected_packet = benchmark.expected_packets_by_case["answerable-case"]
    private_span = _private_span(benchmark)
    retrieval = evaluation.RetrievedEvidenceObservation(
        evidence_id=expected_chunk.evidence_id,
        rank=1,
        retrieval_score=expected_packet.items[0].retrieval_score,
        document_id=document.document_id,
        document_revision_id=expected_document.revision_id,
        folder=document.folder,
        title=document.title,
        relative_path=expected_document.relative_path,
        source_sha256=gold.source_sha256,
        chunk_sha256=expected_chunk.chunk_sha256,
        start_char=expected_chunk.start_char,
        end_char=expected_chunk.end_char,
        start_line=expected_chunk.start_line,
        end_line=expected_chunk.end_line,
    )
    claims = [
        evaluation.DraftClaimObservation(
            claim_id="C1",
            text="The gold statement is correct.",
            render_safe=True,
            proposed_citations=[private_span],
        ),
        evaluation.DraftClaimObservation(
            claim_id="C2",
            text="The unsupported statement is also correct.",
            render_safe=True,
            proposed_citations=[private_span],
        ),
    ]
    checks = [
        evaluation.VerifierCheckObservation(
            claim_id=claim.claim_id,
            raw_verdict="SUPPORTED",
            effective_verdict="SUPPORTED",
            support_spans=[private_span],
        )
        for claim in claims
    ]
    citations = [
        evaluation.ResponseCitationObservation(
            citation_id=f"R{index}",
            evidence_id=expected_chunk.evidence_id,
            document_id=document.document_id,
            document_revision_id=expected_document.revision_id,
            title=document.title,
            relative_path=retrieval.relative_path,
            source_sha256=gold.source_sha256,
            quote=gold.quote,
            source_start_char=gold.source_start_char,
            source_end_char=gold.source_end_char,
            start_line=1,
            end_line=1,
        )
        for index in (1, 2)
    ]
    packet = KnowledgeStore._build_packet(
        evaluation.canonicalize_query(definition.cases[0].query),
        corpus_revision,
        [EvidenceItem(
            evidence_id=retrieval.evidence_id,
            rank=retrieval.rank,
            retrieval_score=retrieval.retrieval_score,
            document_id=retrieval.document_id,
            document_revision_id=retrieval.document_revision_id,
            folder=retrieval.folder,
            title=retrieval.title,
            relative_path=retrieval.relative_path,
            source_sha256=retrieval.source_sha256,
            chunk_sha256=retrieval.chunk_sha256,
            start_char=retrieval.start_char,
            end_char=retrieval.end_char,
            start_line=retrieval.start_line,
            end_line=retrieval.end_line,
            text=expected_chunk.text,
        )],
    )
    draft_hash = hashlib.sha256(_json_bytes({
        "packet_id": packet.packet_id,
        "claims": [
            {
                "claim_id": claim.claim_id,
                "text": claim.text,
                "proposed_citations": [{
                    "evidence_id": citation.evidence_id,
                    "quote": citation.quote,
                } for citation in claim.proposed_citations],
            }
            for claim in claims
        ],
    })).hexdigest()
    resolved_span = {
        "evidence_id": private_span.evidence_id,
        "document_id": private_span.document_id,
        "document_revision_id": private_span.document_revision_id,
        "title": private_span.title,
        "relative_path": private_span.relative_path,
        "source_sha256": private_span.source_sha256,
        "quote": private_span.quote,
        "source_start_char": private_span.source_start_char,
        "source_end_char": private_span.source_end_char,
        "start_line": private_span.start_line,
        "end_line": private_span.end_line,
    }
    verification_hash = hashlib.sha256(_json_bytes({
        "packet_id": packet.packet_id,
        "draft_hash": draft_hash,
        "checks": [
            {
                "claim_id": check.claim_id,
                "verdict": check.effective_verdict,
                "support_spans": [resolved_span],
            }
            for check in checks
        ],
    })).hexdigest()
    configured_stage_invocations = [
        evaluation.StageInvocationObservation(
            stage=stage,
            provider="openai",
            model=f"configured-test-model-{index}",
            system_sha256=str(index) * 64,
            user_sha256=str(index + 3) * 64,
            response_schema_sha256=str(index + 6) * 64,
            response_sha256=chr(ord("a") + index - 1) * 64,
        )
        for index, stage in enumerate(("gpt1", "gpt2", "gpt3"), start=1)
    ]
    configured_fingerprints = {
        invocation.stage: hashlib.sha256(_json_bytes({
            "model": invocation.model,
            "provider": invocation.provider,
        })).hexdigest()
        for invocation in configured_stage_invocations
    }
    called_identity = _execution_identity(
        response_run_id="a" * 32,
        packet_id=packet.packet_id,
        corpus_revision=corpus_revision,
        stage_fingerprints=evaluation.StageFingerprintsObservation(
            **configured_fingerprints,
        ),
        stages_completed=["retrieval", "answerer", "verifier", "finalizer"],
        draft_hash=draft_hash,
        verification_hash=verification_hash,
        selected_claim_ids=["C1", "C2"],
        citation_evidence_ids=[expected_chunk.evidence_id],
        status="ANSWER",
        reason_code="answered",
        draft_claim_count=2,
        supported_claim_count=2,
    )
    answer_case = evaluation.CaseObservation(
        case_id="answerable-case",
        packet_id=packet.packet_id,
        corpus_revision=corpus_revision,
        retrieval_k=definition.top_k,
        execution_identity=called_identity,
        retrieval=[retrieval],
        answerer=evaluation.AnswererStageObservation(
            called=True,
            protocol_valid=True,
            answerability="ANSWERABLE",
            claims=claims,
        ),
        verifier=evaluation.VerifierStageObservation(
            called=True,
            protocol_valid=True,
            checks=checks,
            eligible_claim_ids=["C1", "C2"],
        ),
        finalizer=evaluation.FinalizerStageObservation(
            called=True,
            protocol_valid=True,
            decision="ANSWER",
            requested_claim_ids=["C1", "C2"],
            accepted_claim_ids=["C1", "C2"],
        ),
        response=evaluation.FinalResponseObservation(
            status="ANSWER",
            reason_code="answered",
            packet_id=packet.packet_id,
            corpus_revision=corpus_revision,
            retrieval_count=1,
            retrieval_truncated=False,
            coverage_limited=False,
            coverage_reasons=[],
            stages_completed=["retrieval", "answerer", "verifier", "finalizer"],
            draft_claim_count=2,
            supported_claim_count=2,
            contradicted_claim_count=0,
            conflict_claim_count=0,
            insufficient_claim_count=0,
            selected_claim_ids=["C1", "C2"],
            released_claims=[
                evaluation.ReleasedClaimObservation(
                    claim_id="C1",
                    citation_ids=["R1"],
                ),
                evaluation.ReleasedClaimObservation(
                    claim_id="C2",
                    citation_ids=["R2"],
                ),
            ],
            citations=citations,
        ),
        latency=_latency_observation(
            called_identity,
            retrieval_ms=1.0,
            answerer_ms=2.0,
            verifier_ms=3.0,
            finalizer_ms=4.0,
            total_ms=10.0,
        ),
        stage_invocations=configured_stage_invocations,
    )
    empty_packet = KnowledgeStore._build_packet(
        evaluation.canonicalize_query(definition.cases[1].query),
        corpus_revision,
        [],
    )
    abstain_identity = _execution_identity(
        response_run_id="b" * 32,
        stage_fingerprints=evaluation.StageFingerprintsObservation(),
        stages_completed=["retrieval"],
        packet_id=empty_packet.packet_id,
        corpus_revision=corpus_revision,
        draft_hash=None,
        verification_hash=None,
        selected_claim_ids=[],
        citation_evidence_ids=[],
        status="ABSTAIN",
        reason_code="no_lexical_match",
        draft_claim_count=0,
        supported_claim_count=0,
        latency_ms=5,
    )
    abstain_case = evaluation.CaseObservation(
        case_id="must-abstain-case",
        packet_id=empty_packet.packet_id,
        corpus_revision=corpus_revision,
        retrieval_k=definition.top_k,
        execution_identity=abstain_identity,
        retrieval=[],
        answerer=evaluation.AnswererStageObservation(
            called=False,
            protocol_valid=None,
            answerability=None,
            claims=[],
        ),
        verifier=evaluation.VerifierStageObservation(
            called=False,
            protocol_valid=None,
            checks=[],
            eligible_claim_ids=[],
        ),
        finalizer=evaluation.FinalizerStageObservation(
            called=False,
            protocol_valid=None,
            decision=None,
            requested_claim_ids=[],
            accepted_claim_ids=[],
        ),
        response=evaluation.FinalResponseObservation(
            status="ABSTAIN",
            reason_code="no_lexical_match",
            packet_id=empty_packet.packet_id,
            corpus_revision=corpus_revision,
            retrieval_count=0,
            retrieval_truncated=False,
            coverage_limited=False,
            coverage_reasons=[],
            stages_completed=["retrieval"],
            draft_claim_count=0,
            supported_claim_count=0,
            contradicted_claim_count=0,
            conflict_claim_count=0,
            insufficient_claim_count=0,
            selected_claim_ids=[],
            released_claims=[],
            citations=[],
        ),
        latency=_latency_observation(
            abstain_identity,
            retrieval_ms=5.0,
            answerer_ms=None,
            verifier_ms=None,
            finalizer_ms=None,
            total_ms=5.0,
        ),
    )
    return evaluation.ObservationBundle(
        benchmark_id=definition.benchmark_id,
        benchmark_sha256=benchmark.raw_sha256,
        implementation=evaluation.implementation_binding(),
        run_id="6" * 32,
        status="COMPLETE",
        cases=[answer_case, abstain_case],
        failure=None,
    )


def _complete_adjudication(
    benchmark: evaluation.ResolvedBenchmark,
    observations: evaluation.ObservationBundle,
) -> evaluation.AdjudicationBundle:
    return evaluation.AdjudicationBundle(
        benchmark_id=benchmark.definition.benchmark_id,
        benchmark_sha256=benchmark.raw_sha256,
        observation_run_id=observations.run_id,
        observation_sha256=evaluation.canonical_model_sha256(observations),
        label_authority="independent_human",
        cases=[
            evaluation.CaseAdjudication(
                case_id="answerable-case",
                claims=[
                    evaluation.ClaimAdjudication(
                        claim_id="C1",
                        corpus_supported=True,
                    ),
                    evaluation.ClaimAdjudication(
                        claim_id="C2",
                        corpus_supported=False,
                    ),
                ],
                released_citations=[
                    evaluation.CitationAdjudication(
                        citation_id="R1",
                        supported_claim_ids=["C1"],
                    ),
                    evaluation.CitationAdjudication(
                        citation_id="R2",
                        supported_claim_ids=[],
                    ),
                ],
            ),
            evaluation.CaseAdjudication(
                case_id="must-abstain-case",
                claims=[],
                released_citations=[],
            ),
        ],
    )


@pytest.mark.asyncio
async def test_recorded_execution_is_response_equivalent_and_captures_full_trace(
    monkeypatch: pytest.MonkeyPatch,
):
    packet = _packet()
    _install_config(monkeypatch)
    _freeze_execution_identity(monkeypatch)
    monkeypatch.setattr(grounded, "call_llm_structured", _supported_call(packet))

    normal = await run_grounded_rag(
        GroundedQueryRequest(prompt="What is Alice's favorite color?"),
        store=StubStore(packet),
    )
    recorded, artifacts = await run_grounded_rag_recorded(
        GroundedQueryRequest(prompt="What is Alice's favorite color?"),
        store=StubStore(packet),
    )

    recorded_payload = recorded.model_dump(mode="json", exclude={"receipt_sha256"})
    normal_payload = normal.model_dump(mode="json", exclude={"receipt_sha256"})
    assert recorded_payload == normal_payload
    assert recorded.status == "ANSWER"
    assert artifacts.schema_version == "grounded-run-artifacts-v2"
    assert artifacts.contract_version == "grounded-rag-v1"
    assert artifacts.packet_schema_version == 2
    assert artifacts.retrieval_version == "sqlite-fts5-v2"
    assert artifacts.chunker_version == "words-180-overlap-30-chars-4000-v2"
    assert artifacts.prompt_versions.answerer == "grounded-answerer-v1"
    assert artifacts.stage_fingerprints.gpt1 is not None
    assert artifacts.draft_hash == artifacts.receipt.draft_hash
    assert artifacts.verification_hash == artifacts.receipt.verification_hash
    assert artifacts.response.receipt_sha256 == hashlib.sha256(
        _json_bytes(artifacts.receipt.model_dump(mode="json"))
    ).hexdigest()
    assert [item.evidence_id for item in artifacts.retrieval] == ["E-source-1"]
    assert artifacts.retrieval[0].source_sha256 == "a" * 64
    assert artifacts.answerer.called is True
    assert artifacts.answerer.protocol_valid is True
    assert artifacts.answerer.claims[0].claim_id == "C1"
    assert artifacts.answerer.claims[0].text == BLUE_QUOTE
    assert artifacts.answerer.claims[0].proposed_citations[0].valid is True
    assert artifacts.verifier.protocol_valid is True
    assert artifacts.verifier.checks[0].raw_verdict == "SUPPORTED"
    assert artifacts.verifier.checks[0].effective_verdict == "SUPPORTED"
    assert artifacts.verifier.eligible_claim_ids == ("C1",)
    assert artifacts.finalizer.protocol_valid is True
    assert artifacts.finalizer.requested_claim_ids == ("C1",)
    assert artifacts.finalizer.accepted_claim_ids == ("C1",)
    assert artifacts.response.selected_claim_ids == ("C1",)
    assert artifacts.response.citations[0].quote == BLUE_QUOTE
    assert artifacts.latency_ms.retrieval >= 0
    assert artifacts.latency_ms.answerer is not None
    assert artifacts.latency_ms.verifier is not None
    assert artifacts.latency_ms.finalizer is not None
    assert artifacts.latency_ms.total >= 0
    assert artifacts.provider_usage == "UNAVAILABLE"
    assert artifacts.cost_usd is None


@pytest.mark.asyncio
async def test_recorded_execution_can_inject_a_closed_evaluation_caller(
    monkeypatch: pytest.MonkeyPatch,
):
    packet = _packet()
    caller = _supported_call(packet)
    stages: list[str] = []

    async def recorded_caller(stage, system, user_content, response_model):
        stages.append(stage)
        return await caller({}, system, user_content, response_model)

    monkeypatch.setattr(
        grounded.config,
        "get_stage_config",
        lambda _stage: (_ for _ in ()).throw(
            AssertionError("recorded execution must not read production config")
        ),
    )
    monkeypatch.setattr(
        grounded,
        "call_llm_structured",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("recorded execution must not use production dispatch")
        ),
    )
    execution = GroundedRecordedExecution(
        stage_identities={
            stage: GroundedRecordedStageIdentity(
                provider="google-antigravity-sdk",
                model="gemini-evaluation-model",
            )
            for stage in ("gpt1", "gpt2", "gpt3")
        },
        caller=recorded_caller,
    )

    response, artifacts = await run_grounded_rag_recorded(
        GroundedQueryRequest(prompt="What is Alice's favorite color?"),
        store=StubStore(packet),
        recorded_execution=execution,
    )

    assert response.status == "ANSWER"
    assert stages == ["gpt1", "gpt2", "gpt3"]
    expected_fingerprint = hashlib.sha256(_json_bytes({
        "model": "gemini-evaluation-model",
        "provider": "google-antigravity-sdk",
    })).hexdigest()
    assert artifacts.stage_fingerprints.gpt1 == expected_fingerprint
    assert artifacts.stage_fingerprints.gpt2 == expected_fingerprint
    assert artifacts.stage_fingerprints.gpt3 == expected_fingerprint


def test_observation_v2_binds_provider_receipts_to_called_stage_fingerprints(
    tmp_path: Path,
):
    benchmark = _load_unit_benchmark(tmp_path)
    observations = _complete_observations(benchmark)
    payload = observations.model_dump(mode="json")
    case = payload["cases"][0]
    fingerprint = hashlib.sha256(_json_bytes({
        "model": "gemini-evaluation-test",
        "provider": "google-antigravity-sdk",
    })).hexdigest()
    fingerprints = {"gpt1": fingerprint, "gpt2": fingerprint, "gpt3": fingerprint}
    identity = case["execution_identity"]
    identity["stage_fingerprints"] = fingerprints
    identity["receipt"]["stage_fingerprints"] = fingerprints
    identity["receipt_sha256"] = hashlib.sha256(
        _json_bytes(identity["receipt"])
    ).hexdigest()
    latency = case["latency"]
    latency["capture_sha256"] = evaluation.latency_capture_sha256(
        identity["receipt_sha256"],
        retrieval_ms=latency["retrieval_ms"],
        answerer_ms=latency["answerer_ms"],
        verifier_ms=latency["verifier_ms"],
        finalizer_ms=latency["finalizer_ms"],
        total_ms=latency["total_ms"],
    )
    case["stage_invocations"] = [
        {
            "stage": stage,
            "provider": "google-antigravity-sdk",
            "model": "gemini-evaluation-test",
            "system_sha256": character * 64,
            "user_sha256": str(index) * 64,
            "response_schema_sha256": chr(ord("a") + index) * 64,
            "response_sha256": str(index + 5) * 64,
        }
        for index, (stage, character) in enumerate(
            (("gpt1", "3"), ("gpt2", "4"), ("gpt3", "5")),
            start=1,
        )
    ]
    for captured in case["stage_invocations"]:
        assert captured["provider"] == "google-antigravity-sdk"
    case["provider_invocations"] = [
        _agy_invocation(stage, captured)
        for stage, captured in zip(
            ("gpt1", "gpt2", "gpt3"),
            case["stage_invocations"],
            strict=True,
        )
    ]
    case["provider_observation_mode"] = "AGY_SDK"
    payload["provider_observation_mode"] = "AGY_SDK"
    for other_case in payload["cases"][1:]:
        other_case["provider_observation_mode"] = "AGY_SDK"

    rebound = evaluation.ObservationBundle.model_validate(payload)
    assert rebound.schema_version == "grounded-rag-observations-v2"
    assert [item.stage for item in rebound.cases[0].provider_invocations] == [
        "gpt1",
        "gpt2",
        "gpt3",
    ]
    assert rebound.cases[0].provider_invocations[0].usage is not None
    assert rebound.cases[0].provider_invocations[0].usage.total_tokens == 16
    assert rebound.usage_cost.status == "UNAVAILABLE"

    reversed_payload = rebound.model_dump(mode="json")
    reversed_payload["cases"][0]["provider_invocations"].reverse()
    with pytest.raises(ValidationError, match="called stage order"):
        evaluation.ObservationBundle.model_validate(reversed_payload)

    wrong_model = rebound.model_dump(mode="json")
    wrong_model["cases"][0]["provider_invocations"][0][
        "requested_model"
    ] = "different-model"
    with pytest.raises(ValidationError, match="does not match engine capture"):
        evaluation.ObservationBundle.model_validate(wrong_model)

    stripped = rebound.model_dump(mode="json")
    stripped["cases"][0]["provider_invocations"] = []
    with pytest.raises(ValidationError, match="called stage order"):
        evaluation.ObservationBundle.model_validate(stripped)

    mislabeled = rebound.model_dump(mode="json")
    mislabeled["cases"][0]["provider_observation_mode"] = "CONFIGURED_UNOBSERVED"
    with pytest.raises(ValidationError, match="cannot carry provider invocations"):
        evaluation.ObservationBundle.model_validate(mislabeled)

    stripped_and_mislabeled = rebound.model_dump(mode="json")
    stripped_and_mislabeled["cases"][0]["provider_invocations"] = []
    stripped_and_mislabeled["cases"][0][
        "provider_observation_mode"
    ] = "CONFIGURED_UNOBSERVED"
    with pytest.raises(ValidationError, match="requires AGY receipt mode"):
        evaluation.ObservationBundle.model_validate(stripped_and_mislabeled)

    fully_stripped = rebound.model_dump(mode="json")
    fully_stripped["provider_observation_mode"] = "CONFIGURED_UNOBSERVED"
    for stripped_case in fully_stripped["cases"]:
        stripped_case["provider_observation_mode"] = "CONFIGURED_UNOBSERVED"
        stripped_case["provider_invocations"] = []
    fully_stripped["cases"][0]["stage_invocations"] = []
    with pytest.raises(ValidationError, match="stage invocation captures"):
        evaluation.ObservationBundle.model_validate(fully_stripped)

    forged_request_hash = rebound.model_dump(mode="json")
    forged_request_hash["cases"][0]["provider_invocations"][0][
        "user_sha256"
    ] = "9" * 64
    with pytest.raises(ValidationError, match="does not match engine capture"):
        evaluation.ObservationBundle.model_validate(forged_request_hash)

    forged_worker_policy = rebound.model_dump(mode="json")
    for invocation in forged_worker_policy["cases"][0]["provider_invocations"]:
        invocation["invocation_policy_sha256"] = "9" * 64
    with pytest.raises(ValidationError, match="bound worker policy"):
        evaluation.ObservationBundle.model_validate(forged_worker_policy)

    runtime_drift = rebound.model_dump(mode="json")
    runtime_drift["cases"][0]["provider_invocations"][1][
        "sdk_artifact_sha256"
    ] = "9" * 64
    with pytest.raises(ValidationError, match="runtime identity changed"):
        evaluation.ObservationBundle.model_validate(runtime_drift)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("packet_matches", "answerability", "expected_reason"),
    [
        (False, "ANSWERABLE", "answerer_packet_mismatch"),
        (True, "UNANSWERABLE", "answerer_abstained"),
    ],
)
async def test_fail_closed_answerer_outputs_with_claims_remain_complete_observations(
    monkeypatch: pytest.MonkeyPatch,
    packet_matches: bool,
    answerability: str,
    expected_reason: str,
):
    packet = _packet()
    _install_config(monkeypatch)

    async def stopped_answerer(cfg, system, user_content, response_model):
        assert response_model is AnswererOutput
        return AnswererOutput(
            packet_id=packet.packet_id if packet_matches else "0" * 64,
            answerability=answerability,
            claims=[{
                "text": BLUE_QUOTE,
                "citations": [{"evidence_id": "E-source-1", "quote": BLUE_QUOTE}],
            }],
        )

    monkeypatch.setattr(grounded, "call_llm_structured", stopped_answerer)
    response, artifacts = await run_grounded_rag_recorded(
        GroundedQueryRequest(prompt="What is Alice's favorite color?"),
        store=StubStore(packet),
    )
    observation = evaluation.case_observation_from_artifacts("answerer-stop", artifacts)

    assert response.reason_code == expected_reason
    assert response.draft_claim_count == 0
    assert len(observation.answerer.claims) == 1
    assert observation.response.draft_claim_count == 0
    assert observation.verifier.called is False
    assert observation.finalizer.called is False


@pytest.mark.asyncio
async def test_normalized_empty_claim_is_captured_without_selection_bias(
    monkeypatch: pytest.MonkeyPatch,
):
    packet = _packet()
    _install_config(monkeypatch)

    async def whitespace_claim(cfg, system, user_content, response_model):
        task = _task_from_prompt(user_content)
        if response_model is AnswererOutput:
            return AnswererOutput(
                packet_id=packet.packet_id,
                answerability="ANSWERABLE",
                claims=[{
                    "text": "   ",
                    "citations": [{"evidence_id": "E-source-1", "quote": BLUE_QUOTE}],
                }],
            )
        if response_model is VerifierOutput:
            return VerifierOutput(
                packet_id=packet.packet_id,
                draft_hash=task["draft_hash"],
                checks=[{
                    "claim_id": "C1",
                    "verdict": "SUPPORTED",
                    "support_spans": [{
                        "evidence_id": "E-source-1",
                        "quote": BLUE_QUOTE,
                    }],
                }],
            )
        return FinalizerOutput(
            packet_id=packet.packet_id,
            draft_hash=task["verification"]["draft_hash"],
            verification_hash=task["verification_hash"],
            decision="ANSWER",
            included_claim_ids=["C1"],
        )

    monkeypatch.setattr(grounded, "call_llm_structured", whitespace_claim)
    response, artifacts = await run_grounded_rag_recorded(
        GroundedQueryRequest(prompt="What is Alice's favorite color?"),
        store=StubStore(packet),
    )
    observation = evaluation.case_observation_from_artifacts("empty-normalized", artifacts)

    assert response.status == "ANSWER"
    assert artifacts.answerer.claims[0].text == ""
    assert observation.answerer.claims[0].text == ""


@pytest.mark.asyncio
async def test_recorded_no_match_abstention_has_only_retrieval_trace(
    monkeypatch: pytest.MonkeyPatch,
):
    packet = _packet(with_items=False)

    async def should_not_call(*args, **kwargs):
        raise AssertionError("a lexical miss must not configure or call models")

    monkeypatch.setattr(grounded, "call_llm_structured", should_not_call)
    response, artifacts = await run_grounded_rag_recorded(
        GroundedQueryRequest(prompt="What is missing?"),
        store=StubStore(packet),
    )

    assert response.status == "ABSTAIN"
    assert response.reason_code == "no_lexical_match"
    assert artifacts.retrieval == ()
    assert artifacts.answerer.called is False
    assert artifacts.answerer.protocol_valid is None
    assert artifacts.verifier.called is False
    assert artifacts.finalizer.called is False
    assert artifacts.response.stages_completed == ("retrieval",)
    assert artifacts.latency_ms.answerer is None
    assert artifacts.latency_ms.verifier is None
    assert artifacts.latency_ms.finalizer is None


@pytest.mark.asyncio
async def test_recorded_invalid_verifier_quote_preserves_raw_and_effective_verdicts(
    monkeypatch: pytest.MonkeyPatch,
):
    packet = _packet()
    _install_config(monkeypatch)
    monkeypatch.setattr(
        grounded,
        "call_llm_structured",
        _supported_call(packet, verifier_quote="invented purple quote"),
    )

    response, artifacts = await run_grounded_rag_recorded(
        GroundedQueryRequest(prompt="What is Alice's favorite color?"),
        store=StubStore(packet),
    )

    check = artifacts.verifier.checks[0]
    assert response.status == "ABSTAIN"
    assert response.reason_code == "no_supported_claims"
    assert check.raw_verdict == "SUPPORTED"
    assert check.effective_verdict == "INSUFFICIENT"
    assert check.support_spans[0].quote == "invented purple quote"
    assert check.support_spans[0].valid is False
    assert check.support_spans[0].resolved is None
    assert artifacts.verifier.eligible_claim_ids == ()
    assert artifacts.finalizer.called is False


@pytest.mark.asyncio
async def test_recorded_verifier_protocol_failure_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
):
    packet = _packet()

    async def missing_check(cfg, system, user_content, response_model):
        task = _task_from_prompt(user_content)
        if response_model is AnswererOutput:
            return AnswererOutput(
                packet_id=packet.packet_id,
                answerability="ANSWERABLE",
                claims=[
                    {
                        "text": BLUE_QUOTE,
                        "citations": [
                            {"evidence_id": "E-source-1", "quote": BLUE_QUOTE}
                        ],
                    }
                ],
            )
        return VerifierOutput(
            packet_id=packet.packet_id,
            draft_hash=task["draft_hash"],
            checks=[],
        )

    _install_config(monkeypatch)
    monkeypatch.setattr(grounded, "call_llm_structured", missing_check)
    response, artifacts = await run_grounded_rag_recorded(
        GroundedQueryRequest(prompt="What is Alice's favorite color?"),
        store=StubStore(packet),
    )

    assert response.reason_code == "verifier_protocol_error"
    assert artifacts.verifier.called is True
    assert artifacts.verifier.protocol_valid is False
    assert artifacts.verifier.checks == ()
    assert artifacts.verifier.eligible_claim_ids == ()
    assert artifacts.finalizer.called is False


@pytest.mark.asyncio
async def test_recorded_finalizer_protocol_failure_retains_request_not_acceptance(
    monkeypatch: pytest.MonkeyPatch,
):
    packet = _packet()
    _install_config(monkeypatch)
    monkeypatch.setattr(
        grounded,
        "call_llm_structured",
        _supported_call(packet, final_ids=["C999"]),
    )

    response, artifacts = await run_grounded_rag_recorded(
        GroundedQueryRequest(prompt="What is Alice's favorite color?"),
        store=StubStore(packet),
    )

    assert response.reason_code == "finalizer_protocol_error"
    assert artifacts.finalizer.called is True
    assert artifacts.finalizer.protocol_valid is False
    assert artifacts.finalizer.requested_claim_ids == ("C999",)
    assert artifacts.finalizer.accepted_claim_ids == ()
    assert artifacts.response.selected_claim_ids == ()


@pytest.mark.asyncio
async def test_recorded_partial_trace_contains_only_accepted_supported_subset(
    monkeypatch: pytest.MonkeyPatch,
):
    packet = _packet()
    _install_config(monkeypatch)
    monkeypatch.setattr(
        grounded,
        "call_llm_structured",
        _supported_call(
            packet,
            second_claim=True,
            final_decision="PARTIAL",
            final_ids=["C1"],
        ),
    )

    response, artifacts = await run_grounded_rag_recorded(
        GroundedQueryRequest(prompt="Tell me Alice's color history."),
        store=StubStore(packet),
    )

    assert response.status == "PARTIAL"
    assert response.reason_code == "partial_evidence"
    assert [claim.claim_id for claim in artifacts.answerer.claims] == ["C1", "C2"]
    assert [check.effective_verdict for check in artifacts.verifier.checks] == [
        "SUPPORTED",
        "CONTRADICTED",
    ]
    assert artifacts.verifier.eligible_claim_ids == ("C1",)
    assert artifacts.finalizer.accepted_claim_ids == ("C1",)
    assert artifacts.response.selected_claim_ids == ("C1",)
    assert "2023" not in response.answer


@pytest.mark.asyncio
async def test_private_capture_serialization_excludes_prompts_packet_and_secrets(
    monkeypatch: pytest.MonkeyPatch,
):
    packet = _packet()
    _install_config(monkeypatch)
    monkeypatch.setattr(grounded, "call_llm_structured", _supported_call(packet))

    _, artifacts = await run_grounded_rag_recorded(
        GroundedQueryRequest(prompt="PRIVATE QUESTION TOKEN"),
        store=StubStore(packet),
    )
    serialized = artifacts.model_dump_json()

    assert BLUE_QUOTE in serialized  # explicitly private claim/span data is retained
    assert YEAR_QUOTE not in serialized  # unselected packet text is not retained
    for forbidden in (
        "PRIVATE QUESTION TOKEN",
        "secret-not-for-artifacts",
        "https://private.invalid",
        "EVIDENCE_PACKET",
        "TASK_DATA",
        "raw_provider",
        "reasoning",
        "api_key",
        "base_url",
    ):
        assert forbidden not in serialized


def test_benchmark_hash_schema_path_and_span_fail_before_any_model_hook(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    model_touches = 0

    def model_hook(*args, **kwargs):
        nonlocal model_touches
        model_touches += 1
        raise AssertionError("benchmark validation must precede model setup")

    monkeypatch.setattr(grounded.config, "get_stage_config", model_hook)
    monkeypatch.setattr(grounded, "call_llm_structured", model_hook)

    valid_path = tmp_path / "valid.json"
    valid_digest = _write_json(valid_path, _benchmark_payload())
    with pytest.raises(evaluation.EvaluationDataError, match="SHA-256"):
        evaluation.load_benchmark(valid_path, "0" * 64)

    invalid_payloads: list[tuple[str, dict]] = []

    extra = _benchmark_payload()
    extra["model_should_ignore_this"] = "forbidden"
    invalid_payloads.append(("closed schema", extra))

    unsafe = _benchmark_payload()
    unsafe["documents"][0]["folder"] = "../private"
    invalid_payloads.append(("path unsafe", unsafe))

    unknown_document = _benchmark_payload()
    unknown_document["cases"][0]["relevant_spans"][0][
        "document_id"
    ] = "missing-document"
    invalid_payloads.append(("unknown source", unknown_document))

    duplicate_quote = _benchmark_payload()
    duplicate_quote["documents"][0]["content"] = (
        "The gold statement is correct. The gold statement is correct."
    )
    invalid_payloads.append(("non-unique quote", duplicate_quote))

    for index, (label, payload) in enumerate(invalid_payloads):
        path = tmp_path / f"invalid-{index}.json"
        digest = _write_json(path, payload)
        with pytest.raises(evaluation.EvaluationDataError, match="benchmark"):
            evaluation.load_benchmark(path, digest)

    symlink = tmp_path / "benchmark-link.json"
    symlink.symlink_to(valid_path)
    with pytest.raises(evaluation.EvaluationDataError, match="non-symlink"):
        evaluation.load_benchmark(symlink, valid_digest)

    assert model_touches == 0


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("source_sha256", "0" * 64, "retrieval provenance"),
        ("relative_path", "sources/../outside.txt", "retrieval provenance"),
    ],
)
def test_observation_source_binding_and_paths_fail_closed(
    tmp_path: Path,
    field: str,
    replacement: str,
    message: str,
):
    benchmark = _load_unit_benchmark(tmp_path)
    observations = _complete_observations(benchmark)
    first_case = observations.cases[0]
    bad_item = first_case.retrieval[0].model_copy(update={field: replacement})
    bad_case = first_case.model_copy(update={"retrieval": [bad_item]})
    bad_bundle = observations.model_copy(
        update={"cases": [bad_case, observations.cases[1]]}
    )

    with pytest.raises(evaluation.EvaluationDataError, match=message):
        evaluation.validate_observation_bundle(benchmark, bad_bundle)


@pytest.mark.parametrize(
    "mutation",
    ["missing_case", "missing_claim", "duplicate_claim", "missing_citation"],
)
def test_independent_annotations_require_exact_claim_and_citation_coverage(
    tmp_path: Path,
    mutation: str,
):
    benchmark = _load_unit_benchmark(tmp_path)
    observations = _complete_observations(benchmark)
    adjudication = _complete_adjudication(benchmark, observations)
    first = adjudication.cases[0]
    if mutation == "missing_case":
        cases = [first]
    elif mutation == "missing_claim":
        cases = [
            first.model_copy(update={"claims": first.claims[:1]}),
            adjudication.cases[1],
        ]
    elif mutation == "duplicate_claim":
        cases = [
            first.model_copy(update={"claims": [first.claims[0], first.claims[0]]}),
            adjudication.cases[1],
        ]
    else:
        cases = [
            first.model_copy(
                update={"released_citations": first.released_citations[:1]}
            ),
            adjudication.cases[1],
        ]
    incomplete_labels = adjudication.model_copy(update={"cases": cases})

    with pytest.raises(evaluation.EvaluationDataError, match="cover every"):
        evaluation.validate_adjudication_bundle(
            benchmark,
            observations,
            incomplete_labels,
        )


def test_incomplete_and_selectively_omitted_observations_cannot_be_scored(
    tmp_path: Path,
):
    benchmark = _load_unit_benchmark(tmp_path)
    complete = _complete_observations(benchmark)
    first_case = complete.cases[0]
    incomplete = evaluation.ObservationBundle(
        benchmark_id=complete.benchmark_id,
        benchmark_sha256=complete.benchmark_sha256,
        implementation=complete.implementation,
        run_id=complete.run_id,
        status="INCOMPLETE",
        cases=[first_case],
        failure=evaluation.ObservationFailure(
            kind="PROVIDER_OR_PIPELINE_ERROR",
            failed_case_id="must-abstain-case",
        ),
    )
    labels = _complete_adjudication(benchmark, complete).model_copy(
        update={"cases": [_complete_adjudication(benchmark, complete).cases[0]]}
    )

    with pytest.raises(evaluation.EvaluationDataError, match="incomplete"):
        evaluation.score_baseline(benchmark, incomplete, labels)

    selective = complete.model_copy(update={"cases": [first_case]})
    with pytest.raises(evaluation.EvaluationDataError, match="every benchmark case"):
        evaluation.validate_observation_bundle(benchmark, selective)


def test_baseline_scorer_uses_human_labels_and_exact_metric_arithmetic(
    tmp_path: Path,
):
    benchmark = _load_unit_benchmark(tmp_path)
    observations = _complete_observations(benchmark)
    adjudication = _complete_adjudication(benchmark, observations)

    summary = evaluation.score_baseline(benchmark, observations, adjudication)

    assert summary.case_count == 2
    assert summary.retrieval_recall_at_k.model_dump() == {
        "numerator": 1,
        "denominator": 1,
        "rate": 1.0,
    }
    assert summary.unsupported_claim_rates.answerer.rate == pytest.approx(0.5)
    assert summary.unsupported_claim_rates.verifier_supported.rate == pytest.approx(
        0.5
    )
    assert summary.unsupported_claim_rates.released.rate == pytest.approx(0.5)
    assert summary.verifier_supported_precision.rate == pytest.approx(0.5)
    assert summary.verifier_supported_recall.rate == pytest.approx(1.0)
    assert summary.citation_validity.rate == pytest.approx(0.5)
    assert summary.citation_completeness.rate == pytest.approx(0.5)
    assert summary.correct_abstention_rate.rate == pytest.approx(1.0)
    assert summary.false_abstention_rate.rate == pytest.approx(0.0)
    assert summary.answer_coverage.rate == pytest.approx(1.0)
    assert summary.coverage_reason_counts == {
        "query_term_limit": 0,
        "top_k": 0,
        "byte_budget": 0,
    }
    assert summary.latency.retrieval_ms.sample_count == 2
    assert summary.latency.retrieval_ms.total == pytest.approx(6.0)
    assert summary.latency.retrieval_ms.p50 == pytest.approx(1.0)
    assert summary.latency.retrieval_ms.p95 == pytest.approx(5.0)
    assert summary.latency.answerer_ms.sample_count == 1
    assert summary.latency.total_ms.total == pytest.approx(15.0)
    assert summary.usage_cost.status == "UNAVAILABLE"
    assert summary.usage_cost.reason == (
        "current_adapter_does_not_expose_provider_usage_or_cost"
    )
    assert summary.usage_cost.input_tokens is None
    assert summary.usage_cost.output_tokens is None
    assert summary.usage_cost.total_tokens is None
    assert summary.usage_cost.cost_usd is None
    assert summary.thresholds_passed is False


def test_runtime_and_metric_contracts_reject_forged_aggregates(tmp_path: Path):
    benchmark = _load_unit_benchmark(tmp_path)
    observations = _complete_observations(benchmark)
    summary = evaluation.score_baseline(
        benchmark,
        observations,
        _complete_adjudication(benchmark, observations),
    )

    implementation_payload = observations.implementation.model_dump(mode="json")
    implementation_payload["runtime"]["python_version"] = "0.0-forged"
    with pytest.raises(ValidationError, match="aggregate"):
        evaluation.ImplementationBinding.model_validate(implementation_payload)

    rate_payload = summary.retrieval_recall_at_k.model_dump(mode="json")
    rate_payload["rate"] = 0.5
    with pytest.raises(ValidationError, match="not derived"):
        evaluation.RateMetric.model_validate(rate_payload)

    value_payload = summary.latency.total_ms.model_dump(mode="json")
    value_payload["p50"] = value_payload["p50"] + 0.25
    with pytest.raises(ValidationError, match="derived from samples"):
        evaluation.ValueMetric.model_validate(value_payload)


def test_summary_thresholds_are_bound_to_metrics_and_frozen_contract(tmp_path: Path):
    benchmark = _load_unit_benchmark(tmp_path)
    observations = _complete_observations(benchmark)
    summary = evaluation.score_baseline(
        benchmark,
        observations,
        _complete_adjudication(benchmark, observations),
    )
    payload = summary.model_dump(mode="json")
    for result in payload["threshold_results"].values():
        result["observed"] = result["threshold"]
        result["passed"] = True
    payload["thresholds_passed"] = True
    payload["result"] = "PASS"

    with pytest.raises(ValidationError, match="metric contract"):
        evaluation.BaselineSummary.model_validate(payload)

    forged_payload = summary.model_dump(mode="json")
    forged_payload["retrieval_recall_at_k"] = {
        "numerator": 0,
        "denominator": 1,
        "rate": 0.0,
    }
    forged_payload["threshold_results"]["retrieval_recall_at_k_min"] = {
        "observed": 0.0,
        "operator": ">=",
        "threshold": 1.0,
        "passed": False,
    }
    forged = evaluation.BaselineSummary.model_validate(forged_payload)
    with pytest.raises(evaluation.EvaluationDataError, match="not derived"):
        evaluation.validate_baseline_summary(
            benchmark,
            observations,
            _complete_adjudication(benchmark, observations),
            forged,
            evaluation.canonical_model_sha256(forged),
        )


def test_stage_receipt_and_release_tampering_fail_closed(tmp_path: Path):
    benchmark = _load_unit_benchmark(tmp_path)
    observations = _complete_observations(benchmark)
    first = observations.cases[0]

    valid_span = first.answerer.claims[0].proposed_citations[0]
    false_span = evaluation.PrivateSpanObservation(
        evidence_id=valid_span.evidence_id,
        quote=valid_span.quote,
        valid=False,
    )
    changed_claim = first.answerer.claims[0].model_copy(
        update={"proposed_citations": [false_span]}
    )
    bad_answerer = first.answerer.model_copy(
        update={"claims": [changed_claim, first.answerer.claims[1]]}
    )
    invalid_span_bundle = observations.model_copy(update={
        "cases": [
            first.model_copy(update={"answerer": bad_answerer}),
            observations.cases[1],
        ]
    })
    with pytest.raises(evaluation.EvaluationDataError, match="validity"):
        evaluation.validate_observation_bundle(benchmark, invalid_span_bundle)

    unsafe_claim = first.answerer.claims[0].model_copy(
        update={"text": "[forged provenance]", "render_safe": True}
    )
    unsafe_bundle = observations.model_copy(update={
        "cases": [
            first.model_copy(update={
                "answerer": first.answerer.model_copy(update={
                    "claims": [unsafe_claim, first.answerer.claims[1]]
                })
            }),
            observations.cases[1],
        ]
    })
    with pytest.raises(evaluation.EvaluationDataError, match="render safety"):
        evaluation.validate_observation_bundle(benchmark, unsafe_bundle)

    impossible_answerer = first.answerer.model_copy(
        update={"answerability": "UNANSWERABLE"}
    )
    impossible_answerer_bundle = observations.model_copy(update={
        "cases": [
            first.model_copy(update={"answerer": impossible_answerer}),
            observations.cases[1],
        ]
    })
    with pytest.raises(evaluation.EvaluationDataError, match="cross-stage"):
        evaluation.validate_observation_bundle(benchmark, impossible_answerer_bundle)

    abstaining_finalizer = first.finalizer.model_copy(update={"decision": "ABSTAIN"})
    impossible_release = observations.model_copy(update={
        "cases": [
            first.model_copy(update={"finalizer": abstaining_finalizer}),
            observations.cases[1],
        ]
    })
    with pytest.raises(evaluation.EvaluationDataError, match="cross-stage"):
        evaluation.validate_observation_bundle(benchmark, impossible_release)

    changed_receipt = first.execution_identity.receipt.model_copy(
        update={"created_at": "2026-08-11T00:00:01+00:00"}
    )
    changed_identity = first.execution_identity.model_copy(update={
        "receipt": changed_receipt,
        "receipt_sha256": hashlib.sha256(
            _json_bytes(changed_receipt.model_dump(mode="json"))
        ).hexdigest(),
    })
    changed_latency = _latency_observation(
        changed_identity,
        retrieval_ms=first.latency.retrieval_ms,
        answerer_ms=first.latency.answerer_ms,
        verifier_ms=first.latency.verifier_ms,
        finalizer_ms=first.latency.finalizer_ms,
        total_ms=first.latency.total_ms,
    )
    changed_observations = observations.model_copy(update={
        "cases": [
            first.model_copy(update={
                "execution_identity": changed_identity,
                "latency": changed_latency,
            }),
            observations.cases[1],
        ]
    })
    evaluation.validate_observation_bundle(benchmark, changed_observations)
    assert evaluation.canonical_model_sha256(changed_observations) != (
        evaluation.canonical_model_sha256(observations)
    )
    with pytest.raises(evaluation.EvaluationDataError, match="exact observations"):
        evaluation.validate_adjudication_bundle(
            benchmark,
            changed_observations,
            _complete_adjudication(benchmark, observations),
        )


def test_private_observation_schema_forbids_prompts_reasoning_and_unknown_fields(
    tmp_path: Path,
):
    benchmark = _load_unit_benchmark(tmp_path)
    observations = _complete_observations(benchmark)
    payload = observations.model_dump(mode="json")
    payload["cases"][0]["prompt"] = "must not be recorded"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        evaluation.ObservationBundle.model_validate(payload)

    serialized = observations.model_dump_json()
    assert "The gold statement is correct." in serialized
    for forbidden in (
        "What is the gold statement?",
        '"prompt":',
        "reasoning",
        "api_key",
        "base_url",
        "provider_payload",
        "evidence_packet",
    ):
        assert forbidden not in serialized


def test_observation_rejects_forged_chunks_spans_counts_and_latency(
    tmp_path: Path,
):
    benchmark = _load_unit_benchmark(tmp_path)
    observations = _complete_observations(benchmark)
    first = observations.cases[0]

    forged_retrieval = first.retrieval[0].model_copy(
        update={"end_char": 10_000_000}
    )
    forged_retrieval_case = first.model_copy(
        update={"retrieval": [forged_retrieval]}
    )

    claim = first.answerer.claims[0]
    forged_proposed = claim.proposed_citations[0].model_copy(
        update={
            "source_end_char": claim.proposed_citations[0].source_end_char + 1,
        }
    )
    forged_claim = claim.model_copy(update={"proposed_citations": [forged_proposed]})
    forged_answerer = first.answerer.model_copy(
        update={"claims": [forged_claim, first.answerer.claims[1]]}
    )
    forged_answerer_case = first.model_copy(update={"answerer": forged_answerer})

    check = first.verifier.checks[0]
    forged_support = check.support_spans[0].model_copy(
        update={"source_sha256": "0" * 64}
    )
    forged_check = check.model_copy(update={"support_spans": [forged_support]})
    forged_verifier = first.verifier.model_copy(
        update={"checks": [forged_check, first.verifier.checks[1]]}
    )
    forged_verifier_case = first.model_copy(update={"verifier": forged_verifier})

    forged_citation = first.response.citations[0].model_copy(
        update={"relative_path": "sources/../outside.txt"}
    )
    forged_response = first.response.model_copy(
        update={"citations": [forged_citation, first.response.citations[1]]}
    )
    forged_response_case = first.model_copy(update={"response": forged_response})

    wrong_count_case = first.model_copy(update={
        "response": first.response.model_copy(update={"retrieval_count": 12})
    })
    wrong_latency_case = first.model_copy(update={
        "latency": first.latency.model_copy(update={"total_ms": 1.0})
    })
    zeroed_latency = first.latency.model_copy(update={
        "retrieval_ms": 0.0,
        "answerer_ms": 0.0,
        "verifier_ms": 0.0,
        "finalizer_ms": 0.0,
        "total_ms": 0.0,
    })
    zeroed_latency_case = first.model_copy(update={"latency": zeroed_latency})
    rehashed_zeroed_latency = zeroed_latency.model_copy(update={
        "capture_sha256": evaluation.latency_capture_sha256(
            first.execution_identity.receipt_sha256,
            retrieval_ms=0.0,
            answerer_ms=0.0,
            verifier_ms=0.0,
            finalizer_ms=0.0,
            total_ms=0.0,
        )
    })
    receipt_contradiction_case = first.model_copy(
        update={"latency": rehashed_zeroed_latency}
    )

    for bad_case in (
        forged_retrieval_case,
        forged_answerer_case,
        forged_verifier_case,
        forged_response_case,
        wrong_count_case,
        wrong_latency_case,
        zeroed_latency_case,
        receipt_contradiction_case,
    ):
        bad_bundle = observations.model_copy(
            update={"cases": [bad_case, observations.cases[1]]}
        )
        with pytest.raises(evaluation.EvaluationDataError):
            evaluation.validate_observation_bundle(benchmark, bad_bundle)


def test_adjudication_and_summary_bind_exact_canonical_artifacts(tmp_path: Path):
    benchmark = _load_unit_benchmark(tmp_path)
    observations = _complete_observations(benchmark)
    adjudication = _complete_adjudication(benchmark, observations)

    first = observations.cases[0]
    changed_claim = first.answerer.claims[0].model_copy(
        update={"text": "The moon is made of cheese."}
    )
    changed_answerer = first.answerer.model_copy(
        update={"claims": [changed_claim, first.answerer.claims[1]]}
    )
    changed_claim_case = first.model_copy(update={"answerer": changed_answerer})
    changed_claim_observations = observations.model_copy(
        update={"cases": [changed_claim_case, observations.cases[1]]}
    )
    with pytest.raises(evaluation.EvaluationDataError, match="draft hash"):
        evaluation.validate_observation_bundle(benchmark, changed_claim_observations)

    changed_case = first.model_copy(update={
        "latency": _latency_observation(
            first.execution_identity,
            retrieval_ms=1.1,
            answerer_ms=first.latency.answerer_ms,
            verifier_ms=first.latency.verifier_ms,
            finalizer_ms=first.latency.finalizer_ms,
            total_ms=first.latency.total_ms,
        )
    })
    changed_observations = observations.model_copy(
        update={"cases": [changed_case, observations.cases[1]]}
    )
    evaluation.validate_observation_bundle(benchmark, changed_observations)

    with pytest.raises(evaluation.EvaluationDataError, match="exact observations"):
        evaluation.validate_adjudication_bundle(
            benchmark,
            changed_observations,
            adjudication,
        )

    summary = evaluation.score_baseline(benchmark, observations, adjudication)
    assert summary.observation_sha256 == hashlib.sha256(
        evaluation.canonical_artifact_bytes(observations)
    ).hexdigest()
    assert summary.adjudication_sha256 == hashlib.sha256(
        evaluation.canonical_artifact_bytes(adjudication)
    ).hexdigest()
    assert summary.implementation_aggregate_sha256 == (
        observations.implementation.aggregate_sha256
    )
    assert summary.result == "FAIL"


def test_score_cli_publishes_explicit_failure_and_returns_nonzero(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_digest = _write_json(benchmark_path, _benchmark_payload())
    benchmark = evaluation.load_benchmark(benchmark_path, benchmark_digest)
    observations = _complete_observations(benchmark)
    adjudication = _complete_adjudication(benchmark, observations)
    observations_path = tmp_path / "observations.json"
    adjudication_path = tmp_path / "adjudication.json"
    observations_path.write_bytes(evaluation.canonical_artifact_bytes(observations))
    adjudication_path.write_bytes(evaluation.canonical_artifact_bytes(adjudication))
    observations_sha256 = hashlib.sha256(observations_path.read_bytes()).hexdigest()
    adjudication_sha256 = hashlib.sha256(adjudication_path.read_bytes()).hexdigest()
    output_dir = tmp_path / "summary-output"

    return_code = measure_grounded_rag.main([
        "score",
        "--benchmark",
        str(benchmark_path),
        "--benchmark-sha256",
        benchmark_digest,
        "--observations",
        str(observations_path),
        "--observations-sha256",
        observations_sha256,
        "--adjudication",
        str(adjudication_path),
        "--adjudication-sha256",
        adjudication_sha256,
        "--output-dir",
        str(output_dir),
    ])

    assert return_code == 4
    score_output = capsys.readouterr().out
    assert score_output.startswith("FAIL ")
    summary = evaluation.BaselineSummary.model_validate_json(
        (output_dir / "summary.json").read_bytes()
    )
    assert summary.result == "FAIL"
    assert summary.thresholds_passed is False
    assert stat.S_IMODE(output_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE((output_dir / "summary.json").stat().st_mode) == 0o600
    summary_digest = evaluation.canonical_model_sha256(summary)
    assert f"summary_sha256={summary_digest}" in score_output

    verify_code = measure_grounded_rag.main([
        "verify-summary",
        "--benchmark",
        str(benchmark_path),
        "--benchmark-sha256",
        benchmark_digest,
        "--observations",
        str(observations_path),
        "--observations-sha256",
        observations_sha256,
        "--adjudication",
        str(adjudication_path),
        "--adjudication-sha256",
        adjudication_sha256,
        "--summary",
        str(output_dir / "summary.json"),
        "--summary-sha256",
        summary_digest,
    ])
    assert verify_code == 0
    assert capsys.readouterr().out.startswith("VERIFIED ")


def test_private_publication_never_replaces_an_existing_directory(tmp_path: Path):
    benchmark = _load_unit_benchmark(tmp_path)
    observations = _complete_observations(benchmark)
    destination = tmp_path / "private-output"
    destination.mkdir(mode=0o700)
    sentinel = destination / "sentinel"
    sentinel.write_text("retain me", encoding="utf-8")

    with pytest.raises(evaluation.EvaluationDataError, match="already exists"):
        evaluation.publish_private_json_directory(
            destination,
            "observations.json",
            observations,
        )

    assert sentinel.read_text(encoding="utf-8") == "retain me"
    assert not (destination / "observations.json").exists()


def test_hash_bound_loaders_reject_rewritten_observations_and_adjudication(
    tmp_path: Path,
):
    benchmark = _load_unit_benchmark(tmp_path)
    observations = _complete_observations(benchmark)
    adjudication = _complete_adjudication(benchmark, observations)

    observations_path = tmp_path / "observations.json"
    original_observation_bytes = evaluation.canonical_artifact_bytes(observations)
    original_observation_sha256 = hashlib.sha256(
        original_observation_bytes
    ).hexdigest()
    observations_path.write_bytes(original_observation_bytes)

    first = observations.cases[0]
    rewritten_latency = _latency_observation(
        first.execution_identity,
        retrieval_ms=0.0,
        answerer_ms=0.0,
        verifier_ms=0.0,
        finalizer_ms=0.0,
        total_ms=first.latency.total_ms,
    )
    rewritten_observations = observations.model_copy(update={
        "cases": [
            first.model_copy(update={"latency": rewritten_latency}),
            observations.cases[1],
        ]
    })
    evaluation.validate_observation_bundle(benchmark, rewritten_observations)
    observations_path.write_bytes(
        evaluation.canonical_artifact_bytes(rewritten_observations)
    )
    with pytest.raises(evaluation.EvaluationDataError, match="raw-byte SHA-256"):
        evaluation.load_observations(
            observations_path,
            original_observation_sha256,
        )

    adjudication_path = tmp_path / "adjudication.json"
    original_adjudication_bytes = evaluation.canonical_artifact_bytes(adjudication)
    original_adjudication_sha256 = hashlib.sha256(
        original_adjudication_bytes
    ).hexdigest()
    adjudication_path.write_bytes(original_adjudication_bytes)
    changed_label = adjudication.cases[0].claims[0].model_copy(
        update={"corpus_supported": False}
    )
    changed_case = adjudication.cases[0].model_copy(update={
        "claims": [changed_label, *adjudication.cases[0].claims[1:]]
    })
    changed_adjudication = adjudication.model_copy(update={
        "cases": [changed_case, *adjudication.cases[1:]]
    })
    adjudication_path.write_bytes(
        evaluation.canonical_artifact_bytes(changed_adjudication)
    )
    with pytest.raises(evaluation.EvaluationDataError, match="raw-byte SHA-256"):
        evaluation.load_adjudication(
            adjudication_path,
            original_adjudication_sha256,
        )


def test_publication_failure_never_deletes_a_raced_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    benchmark = _load_unit_benchmark(tmp_path)
    observations = _complete_observations(benchmark)
    destination = tmp_path / "private-output"
    final_path = destination / "observations.json"
    original_fsync = evaluation.os.fsync
    fsync_calls = 0

    def replace_then_fail(file_descriptor: int) -> None:
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls == 2:
            final_path.unlink()
            final_path.write_bytes(b"raced replacement")
            raise OSError("injected directory fsync failure")
        original_fsync(file_descriptor)

    monkeypatch.setattr(evaluation.os, "fsync", replace_then_fail)
    with pytest.raises(evaluation.EvaluationDataError, match="could not be published"):
        evaluation.publish_private_json_directory(
            destination,
            "observations.json",
            observations,
        )

    assert final_path.read_bytes() == b"raced replacement"


def test_publication_rejects_staging_and_destination_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    benchmark = _load_unit_benchmark(tmp_path)
    observations = _complete_observations(benchmark)
    original_link = evaluation.os.link

    def substitute_staging(source, destination, **kwargs):
        source_fd = kwargs["src_dir_fd"]
        evaluation.os.unlink(source, dir_fd=source_fd)
        attacker_fd = evaluation.os.open(
            source,
            evaluation.os.O_WRONLY | evaluation.os.O_CREAT | evaluation.os.O_EXCL,
            0o600,
            dir_fd=source_fd,
        )
        try:
            evaluation.os.write(attacker_fd, b'{"attacker":true}\n')
        finally:
            evaluation.os.close(attacker_fd)
        return original_link(source, destination, **kwargs)

    monkeypatch.setattr(evaluation.os, "link", substitute_staging)
    with pytest.raises(evaluation.EvaluationDataError, match="staged bytes"):
        evaluation.publish_private_json_directory(
            tmp_path / "staging-substitution",
            "observations.json",
            observations,
        )

    monkeypatch.setattr(evaluation.os, "link", original_link)
    destination = tmp_path / "directory-substitution"
    displaced = tmp_path / "displaced-canonical"
    original_fsync = evaluation.os.fsync
    fsync_calls = 0

    def substitute_directory(file_descriptor: int) -> None:
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls == 2:
            destination.rename(displaced)
            destination.mkdir(mode=0o700)
            (destination / "observations.json").write_bytes(b"competitor")
        original_fsync(file_descriptor)

    monkeypatch.setattr(evaluation.os, "fsync", substitute_directory)
    with pytest.raises(evaluation.EvaluationDataError, match="directory changed"):
        evaluation.publish_private_json_directory(
            destination,
            "observations.json",
            observations,
        )
    assert (destination / "observations.json").read_bytes() == b"competitor"
    assert (displaced / "observations.json").read_bytes() == (
        evaluation.canonical_artifact_bytes(observations)
    )


def test_publication_rechecks_directory_after_final_entry_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    benchmark = _load_unit_benchmark(tmp_path)
    observations = _complete_observations(benchmark)
    destination = tmp_path / "late-directory-substitution"
    displaced = tmp_path / "late-displaced-canonical"
    original_stat = evaluation.os.stat
    final_entry_checks = 0

    def substitute_after_named_directory_check(path, *args, **kwargs):
        nonlocal final_entry_checks
        if path == "observations.json":
            final_entry_checks += 1
            if final_entry_checks == 2:
                destination.rename(displaced)
                destination.mkdir(mode=0o700)
                (destination / "observations.json").write_bytes(b"competitor")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(evaluation.os, "stat", substitute_after_named_directory_check)
    with pytest.raises(
        evaluation.EvaluationDataError,
        match="output directory changed before confirmation",
    ):
        evaluation.publish_private_json_directory(
            destination,
            "observations.json",
            observations,
        )

    assert (destination / "observations.json").read_bytes() == b"competitor"
    assert (displaced / "observations.json").read_bytes() == (
        evaluation.canonical_artifact_bytes(observations)
    )


def test_retained_digest_rejects_post_confirmation_path_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    benchmark = _load_unit_benchmark(tmp_path)
    observations = _complete_observations(benchmark)
    expected_sha256 = evaluation.canonical_model_sha256(observations)
    destination = tmp_path / "post-confirmation-substitution"
    final_path = destination / "observations.json"
    original_stat = evaluation.os.stat
    final_entry_checks = 0

    def substitute_after_final_entry_stat(path, *args, **kwargs):
        nonlocal final_entry_checks
        result = original_stat(path, *args, **kwargs)
        if path == "observations.json":
            final_entry_checks += 1
            if final_entry_checks == 2:
                final_path.unlink()
                final_path.write_bytes(b"competitor")
        return result

    monkeypatch.setattr(evaluation.os, "stat", substitute_after_final_entry_stat)
    published = evaluation.publish_private_json_directory(
        destination,
        "observations.json",
        observations,
    )
    assert published.read_bytes() == b"competitor"
    with pytest.raises(evaluation.EvaluationDataError, match="raw-byte SHA-256"):
        evaluation.load_observations(published, expected_sha256)


def test_single_open_reader_rejects_a_swap_to_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    target = tmp_path / "target.json"
    digest = _write_json(target, _benchmark_payload())
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(target.read_bytes())
    original_open = evaluation.os.open
    swapped = False

    def swap_before_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if Path(path) == target and not swapped:
            swapped = True
            target.unlink()
            target.symlink_to(replacement)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(evaluation.os, "open", swap_before_open)
    with pytest.raises(evaluation.EvaluationDataError, match="non-symlink"):
        evaluation.load_benchmark(target, digest)


@pytest.mark.asyncio
async def test_record_baseline_executes_complete_frozen_case_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    benchmark = _load_unit_benchmark(tmp_path)
    _install_config(monkeypatch)

    async def dynamic_call(cfg, system, user_content, response_model):
        packet_json = user_content.split(
            "=== EVIDENCE_PACKET (UNTRUSTED DATA) ===\n",
            1,
        )[1].split("\n=== END EVIDENCE_PACKET ===", 1)[0]
        packet = json.loads(packet_json)
        task = _task_from_prompt(user_content)
        if response_model is AnswererOutput:
            if "not present" in task["question"].casefold():
                return AnswererOutput(
                    packet_id=packet["packet_id"],
                    answerability="UNANSWERABLE",
                    claims=[],
                )
            return AnswererOutput(
                packet_id=packet["packet_id"],
                answerability="ANSWERABLE",
                claims=[{
                    "text": "The gold statement is correct.",
                    "citations": [{
                        "evidence_id": packet["items"][0]["evidence_id"],
                        "quote": "The gold statement is correct.",
                    }],
                }],
            )
        if response_model is VerifierOutput:
            return VerifierOutput(
                packet_id=packet["packet_id"],
                draft_hash=task["draft_hash"],
                checks=[{
                    "claim_id": "C1",
                    "verdict": "SUPPORTED",
                    "support_spans": [{
                        "evidence_id": packet["items"][0]["evidence_id"],
                        "quote": "The gold statement is correct.",
                    }],
                }],
            )
        return FinalizerOutput(
            packet_id=packet["packet_id"],
            draft_hash=task["verification"]["draft_hash"],
            verification_hash=task["verification_hash"],
            decision="ANSWER",
            included_claim_ids=["C1"],
        )

    monkeypatch.setattr(grounded, "call_llm_structured", dynamic_call)
    observations = await evaluation.record_baseline(
        benchmark,
        external_cost_limit_usd=1.0,
    )

    assert observations.status == "COMPLETE"
    assert [case.case_id for case in observations.cases] == [
        "answerable-case",
        "must-abstain-case",
    ]
    assert observations.cases[0].response.status == "ANSWER"
    assert observations.cases[1].response.status == "ABSTAIN"
    assert observations.implementation == evaluation.implementation_binding()
    evaluation.validate_observation_bundle(benchmark, observations)


@pytest.mark.asyncio
async def test_record_baseline_preserves_agy_mode_for_complete_and_failed_runs(
    tmp_path: Path,
):
    payload = _benchmark_payload()
    for index, case in enumerate(payload["cases"], start=1):
        case["query"] = f"xqzvnonexistent{index}"
    benchmark_path = tmp_path / "no-call-benchmark.json"
    benchmark = evaluation.load_benchmark(
        benchmark_path,
        _write_json(benchmark_path, payload),
    )

    async def no_call_agy_runner(request, *, store):
        response, artifacts = await run_grounded_rag_recorded(
            request,
            store=store,
        )
        return response, artifacts, ()

    complete = await evaluation.record_baseline(
        benchmark,
        recorded_runner=no_call_agy_runner,
        external_cost_limit_usd=1.0,
        provider_observation_mode="AGY_SDK",
    )

    assert complete.status == "COMPLETE"
    assert complete.provider_observation_mode == "AGY_SDK"
    assert len(complete.cases) == len(benchmark.definition.cases)
    assert {
        case.provider_observation_mode for case in complete.cases
    } == {"AGY_SDK"}
    assert all(not case.stage_invocations for case in complete.cases)
    assert all(not case.provider_invocations for case in complete.cases)

    async def failed_agy_runner(*_args, **_kwargs):
        raise AgyEvaluationProviderError("WORKER_TIMEOUT")

    incomplete = await evaluation.record_baseline(
        benchmark,
        recorded_runner=failed_agy_runner,
        external_cost_limit_usd=1.0,
        provider_observation_mode="AGY_SDK",
    )

    assert incomplete.status == "INCOMPLETE"
    assert incomplete.provider_observation_mode == "AGY_SDK"
    assert incomplete.cases == []
    assert incomplete.failure is not None
    assert incomplete.failure.kind == "PROVIDER_OR_PIPELINE_ERROR"
    assert incomplete.failure.failed_case_id == benchmark.definition.cases[0].case_id


@pytest.mark.asyncio
async def test_recording_requires_external_cost_control_before_runner(
    tmp_path: Path,
):
    benchmark = _load_unit_benchmark(tmp_path)
    calls = 0

    async def runner(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("cost guard must run before execution")

    with pytest.raises(evaluation.EvaluationDataError, match="cost limit"):
        await evaluation.record_baseline(benchmark, recorded_runner=runner)
    with pytest.raises(evaluation.EvaluationDataError, match="cost limit"):
        await evaluation.record_baseline(
            benchmark,
            recorded_runner=runner,
            external_cost_limit_usd=2.0,
        )
    assert calls == 0
