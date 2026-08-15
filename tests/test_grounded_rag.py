"""Fail-closed contract tests for the isolated three-role grounded lane."""
from __future__ import annotations

import hashlib
import json

import pytest
from pydantic import ValidationError

import pipeline.grounded_rag as grounded
from pipeline.grounded_rag import (
    AnswererOutput,
    FinalizerOutput,
    GroundedQueryRequest,
    VerifierOutput,
    run_grounded_rag,
)
from pipeline.helpers import PipelineError
from pipeline.knowledge_store import (
    EvidenceItem,
    EvidencePacket,
    KnowledgeStore,
    KnowledgeStoreError,
)


def _packet(with_items: bool = True) -> EvidencePacket:
    items = []
    if with_items:
        text = "Alice's favorite color is blue. Alice started using it in 2024."
        items.append(EvidenceItem(
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
        ))
    return KnowledgeStore._build_packet("what is alice favorite color", "c" * 64, items)


class StubStore:
    def __init__(self, packet: EvidencePacket):
        self.packet = packet
        self.receipts: dict[str, dict] = {}

    def retrieve(self, query: str, top_k: int) -> EvidencePacket:
        return self.packet

    def append_run_receipt(self, receipt: dict) -> str:
        payload = json.dumps(
            receipt,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.receipts[receipt["run_id"]] = receipt
        return hashlib.sha256(payload).hexdigest()


def _task_from_prompt(user_content: str) -> dict:
    marker = "=== TASK_DATA ===\n"
    body = user_content.split(marker, 1)[1].split("\n=== END TASK_DATA ===", 1)[0]
    return json.loads(body)


def _packet_bytes_from_prompt(user_content: str) -> str:
    start = "=== EVIDENCE_PACKET (UNTRUSTED DATA) ===\n"
    end = "\n=== END EVIDENCE_PACKET ==="
    return user_content.split(start, 1)[1].split(end, 1)[0]


def test_claim_renderer_rejects_spoofed_provenance_and_direction_controls():
    assert grounded._render_claim_text("Claim [999](https://evil.example)") is None
    assert grounded._render_claim_text("Claim \u202e[1]") is None
    assert grounded._render_claim_text("Claim [999\u200b]") is None
    assert grounded._render_claim_text("Claim ［1］") is None
    assert grounded._render_claim_text("Claim \x1b[31m") is None
    assert grounded._render_claim_text("Claim [ordinary brackets]") is None
    assert grounded._render_claim_text("<b>claim</b> *bold*") == (
        "&lt;b&gt;claim&lt;/b&gt; \\*bold\\*"
    )


def _install_config(monkeypatch):
    monkeypatch.setattr(
        grounded.config,
        "get_stage_config",
        lambda stage: {"provider": "openai", "api_key": "test", "model": stage, "base_url": ""},
    )


@pytest.mark.asyncio
async def test_supported_answer_uses_same_packet_and_blind_verifier(monkeypatch):
    packet = _packet()
    store = StubStore(packet)
    calls = []

    async def fake_call(cfg, system, user_content, response_model):
        calls.append((response_model, user_content))
        task = _task_from_prompt(user_content)
        if response_model is AnswererOutput:
            return AnswererOutput(
                packet_id=packet.packet_id,
                answerability="ANSWERABLE",
                claims=[{
                    "text": "Alice's favorite color is blue.",
                    "citations": [{
                        "evidence_id": "E-source-1",
                        "quote": "Alice's favorite color is blue.",
                    }],
                }],
            )
        if response_model is VerifierOutput:
            assert "proposed_citations" not in user_content
            return VerifierOutput(
                packet_id=packet.packet_id,
                draft_hash=task["draft_hash"],
                checks=[{
                    "claim_id": "C1",
                    "verdict": "SUPPORTED",
                    "support_spans": [{
                        "evidence_id": "E-source-1",
                        "quote": "Alice's favorite color is blue.",
                    }],
                }],
            )
        return FinalizerOutput(
            packet_id=packet.packet_id,
            draft_hash=task["draft"]["claims"] and task["verification"]["draft_hash"],
            verification_hash=task["verification_hash"],
            decision="ANSWER",
            included_claim_ids=["C1"],
        )

    _install_config(monkeypatch)
    monkeypatch.setattr(grounded, "call_llm_structured", fake_call)
    response = await run_grounded_rag(
        GroundedQueryRequest(prompt="What is Alice's favorite color?"),
        store=store,
    )

    assert response.status == "ANSWER"
    assert response.answer == "- Alice's favorite color is blue. [1]"
    assert response.citations[0].quote == "Alice's favorite color is blue."
    assert response.stages_completed == ["retrieval", "answerer", "verifier", "finalizer"]
    assert response.contract_version == "grounded-rag-v1"
    assert response.packet_schema_version == 2
    assert response.receipt_sha256 is not None
    assert all(response.stage_fingerprints.model_dump().values())
    receipt = store.receipts[response.run_id]
    serialized_receipt = json.dumps(receipt, sort_keys=True)
    assert receipt["selected_claim_ids"] == ["C1"]
    assert receipt["citation_evidence_ids"] == ["E-source-1"]
    assert "What is Alice" not in serialized_receipt
    assert "Alice's favorite color" not in serialized_receipt
    assert '"api_key"' not in serialized_receipt
    invalid_payload = response.model_dump(mode="json")
    invalid_payload["reason_code"] = "models_agreed_so_probably_true"
    with pytest.raises(ValidationError):
        grounded.GroundedRAGResponse.model_validate(invalid_payload)
    impossible_payload = response.model_dump(mode="json")
    impossible_payload.update({
        "status": "ABSTAIN",
        "coverage_limited": False,
        "coverage_reasons": ["top_k"],
        "stages_completed": ["finalizer", "retrieval"],
    })
    with pytest.raises(ValidationError):
        grounded.GroundedRAGResponse.model_validate(impossible_payload)
    assert len(calls) == 3
    assert len({_packet_bytes_from_prompt(call[1]) for call in calls}) == 1


@pytest.mark.asyncio
async def test_receipt_write_failure_prevents_answer_release(monkeypatch):
    packet = _packet(with_items=False)

    class FailingReceiptStore(StubStore):
        def append_run_receipt(self, receipt: dict) -> str:
            raise KnowledgeStoreError("receipt unavailable")

    with pytest.raises(KnowledgeStoreError, match="receipt unavailable"):
        await run_grounded_rag(
            GroundedQueryRequest(prompt="What is missing?"),
            store=FailingReceiptStore(packet),
        )


@pytest.mark.asyncio
async def test_no_evidence_abstains_without_model_calls(monkeypatch):
    packet = _packet(with_items=False)

    async def should_not_call(*args, **kwargs):
        raise AssertionError("models must not run without evidence")

    monkeypatch.setattr(grounded, "call_llm_structured", should_not_call)
    response = await run_grounded_rag(
        GroundedQueryRequest(prompt="What is missing?"),
        store=StubStore(packet),
    )
    assert response.status == "ABSTAIN"
    assert response.reason_code == "no_lexical_match"
    assert response.stages_completed == ["retrieval"]


@pytest.mark.asyncio
async def test_matching_evidence_omitted_by_packet_budget_is_not_a_lexical_miss(
    monkeypatch,
):
    packet = KnowledgeStore._build_packet(
        "oversized query",
        "c" * 64,
        [],
        truncated=True,
        coverage_reasons=("byte_budget",),
    )

    async def should_not_call(*args, **kwargs):
        raise AssertionError("models must not run without a packet item")

    monkeypatch.setattr(grounded, "call_llm_structured", should_not_call)
    response = await run_grounded_rag(
        GroundedQueryRequest(prompt="oversized query"),
        store=StubStore(packet),
    )

    assert response.status == "ABSTAIN"
    assert response.reason_code == "evidence_packet_budget_exceeded"
    assert response.coverage_reasons == ["byte_budget"]


@pytest.mark.asyncio
async def test_query_term_limited_empty_packet_is_not_a_lexical_miss(monkeypatch):
    packet = KnowledgeStore._build_packet(
        "limited query",
        "c" * 64,
        [],
        coverage_reasons=("query_term_limit",),
    )

    async def should_not_call(*args, **kwargs):
        raise AssertionError("models must not run without a packet item")

    monkeypatch.setattr(grounded, "call_llm_structured", should_not_call)
    response = await run_grounded_rag(
        GroundedQueryRequest(prompt="limited query"),
        store=StubStore(packet),
    )

    assert response.status == "ABSTAIN"
    assert response.reason_code == "retrieval_query_term_limit_exceeded"
    assert response.coverage_reasons == ["query_term_limit"]


@pytest.mark.asyncio
async def test_invalid_verifier_quote_cannot_reach_final_output(monkeypatch):
    packet = _packet()
    store = StubStore(packet)

    async def fake_call(cfg, system, user_content, response_model):
        task = _task_from_prompt(user_content)
        if response_model is AnswererOutput:
            return AnswererOutput(
                packet_id=packet.packet_id,
                answerability="ANSWERABLE",
                claims=[{
                    "text": "Alice's favorite color is blue.",
                    "citations": [{
                        "evidence_id": "E-source-1",
                        "quote": "Alice's favorite color is blue.",
                    }],
                }],
            )
        if response_model is VerifierOutput:
            return VerifierOutput(
                packet_id=packet.packet_id,
                draft_hash=task["draft_hash"],
                checks=[{
                    "claim_id": "C1",
                    "verdict": "SUPPORTED",
                    "support_spans": [{"evidence_id": "E-source-1", "quote": "purple"}],
                }],
            )
        return FinalizerOutput(
            packet_id=packet.packet_id,
            draft_hash=task["verification"]["draft_hash"],
            verification_hash=task["verification_hash"],
            decision="ANSWER",
            included_claim_ids=["C1"],
        )

    _install_config(monkeypatch)
    monkeypatch.setattr(grounded, "call_llm_structured", fake_call)
    response = await run_grounded_rag(
        GroundedQueryRequest(prompt="What is Alice's favorite color?"),
        store=store,
    )
    assert response.status == "ABSTAIN"
    assert response.reason_code == "no_supported_claims"
    assert response.stages_completed == ["retrieval", "answerer", "verifier"]
    assert "blue" not in response.answer
    assert store.receipts[response.run_id]["verification_hash"] is not None


@pytest.mark.asyncio
async def test_verifier_must_cover_every_claim_exactly_once(monkeypatch):
    packet = _packet()

    async def fake_call(cfg, system, user_content, response_model):
        task = _task_from_prompt(user_content)
        if response_model is AnswererOutput:
            return AnswererOutput(
                packet_id=packet.packet_id,
                answerability="ANSWERABLE",
                claims=[{
                    "text": "Alice's favorite color is blue.",
                    "citations": [{
                        "evidence_id": "E-source-1",
                        "quote": "Alice's favorite color is blue.",
                    }],
                }],
            )
        return VerifierOutput(
            packet_id=packet.packet_id,
            draft_hash=task["draft_hash"],
            checks=[],
        )

    _install_config(monkeypatch)
    monkeypatch.setattr(grounded, "call_llm_structured", fake_call)
    response = await run_grounded_rag(
        GroundedQueryRequest(prompt="What is Alice's favorite color?"),
        store=StubStore(packet),
    )
    assert response.status == "ABSTAIN"
    assert response.reason_code == "verifier_protocol_error"
    assert response.stages_completed == ["retrieval", "answerer", "verifier"]


@pytest.mark.asyncio
async def test_supported_subset_is_rendered_as_partial(monkeypatch):
    packet = _packet()

    async def fake_call(cfg, system, user_content, response_model):
        task = _task_from_prompt(user_content)
        if response_model is AnswererOutput:
            return AnswererOutput(
                packet_id=packet.packet_id,
                answerability="ANSWERABLE",
                claims=[
                    {
                        "text": "Alice's favorite color is blue.",
                        "citations": [{
                            "evidence_id": "E-source-1",
                            "quote": "Alice's favorite color is blue.",
                        }],
                    },
                    {
                        "text": "Alice chose blue in 2023.",
                        "citations": [{
                            "evidence_id": "E-source-1",
                            "quote": "Alice started using it in 2024.",
                        }],
                    },
                ],
            )
        if response_model is VerifierOutput:
            return VerifierOutput(
                packet_id=packet.packet_id,
                draft_hash=task["draft_hash"],
                checks=[
                    {
                        "claim_id": "C1",
                        "verdict": "SUPPORTED",
                        "support_spans": [{
                            "evidence_id": "E-source-1",
                            "quote": "Alice's favorite color is blue.",
                        }],
                    },
                    {"claim_id": "C2", "verdict": "CONFLICT", "support_spans": []},
                ],
            )
        return FinalizerOutput(
            packet_id=packet.packet_id,
            draft_hash=task["verification"]["draft_hash"],
            verification_hash=task["verification_hash"],
            decision="PARTIAL",
            included_claim_ids=["C1"],
        )

    _install_config(monkeypatch)
    monkeypatch.setattr(grounded, "call_llm_structured", fake_call)
    response = await run_grounded_rag(
        GroundedQueryRequest(prompt="Tell me Alice's color history."),
        store=StubStore(packet),
    )
    assert response.status == "PARTIAL"
    assert response.reason_code == "partial_with_conflict"
    assert "2023" not in response.answer
    assert response.contradicted_claim_count == 0
    assert response.conflict_claim_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("missing_stage", "configured_stages"),
    [
        ("gpt1", {"gpt2": "key2", "gpt3": "key3"}),
        ("gpt2", {"gpt1": "key1", "gpt3": "key3"}),
        ("gpt3", {"gpt1": "key1", "gpt2": "key2"}),
    ],
)
async def test_grounded_rag_missing_single_stage_api_key(
    monkeypatch, missing_stage: str, configured_stages: dict[str, str]
):
    """R5: Test granular stage error message when a single stage API key is missing."""
    packet = _packet(with_items=True)
    store = StubStore(packet)

    configs = {
        "gpt1": {"provider": "openai", "api_key": configured_stages.get("gpt1", ""), "model": "gpt1", "base_url": ""},
        "gpt2": {"provider": "openai", "api_key": configured_stages.get("gpt2", ""), "model": "gpt2", "base_url": ""},
        "gpt3": {"provider": "openai", "api_key": configured_stages.get("gpt3", ""), "model": "gpt3", "base_url": ""},
    }
    monkeypatch.setattr(grounded.config, "get_stage_config", lambda stage: configs[stage])

    with pytest.raises(PipelineError) as exc_info:
        await run_grounded_rag(
            GroundedQueryRequest(prompt="What is Alice's favorite color?"),
            store=store,
        )

    assert exc_info.value.status_code == 400
    assert str(exc_info.value) == f"Configure an API key for grounded stage '{missing_stage}' first."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("missing_stages", "expected_msg"),
    [
        (
            ["gpt1", "gpt3"],
            "Configure an API key for grounded stages 'gpt1, gpt3' first.",
        ),
        (
            ["gpt2", "gpt3"],
            "Configure an API key for grounded stages 'gpt2, gpt3' first.",
        ),
        (
            ["gpt1", "gpt2", "gpt3"],
            "Configure an API key for grounded stages 'gpt1, gpt2, gpt3' first.",
        ),
    ],
)
async def test_grounded_rag_missing_multiple_stages_api_key(
    monkeypatch, missing_stages: list[str], expected_msg: str
):
    """R5: Test granular stage error message when multiple stage API keys are missing."""
    packet = _packet(with_items=True)
    store = StubStore(packet)

    configs = {
        stage: {
            "provider": "openai",
            "api_key": "" if stage in missing_stages else "valid_key",
            "model": stage,
            "base_url": "",
        }
        for stage in ("gpt1", "gpt2", "gpt3")
    }
    monkeypatch.setattr(grounded.config, "get_stage_config", lambda stage: configs[stage])

    with pytest.raises(PipelineError) as exc_info:
        await run_grounded_rag(
            GroundedQueryRequest(prompt="What is Alice's favorite color?"),
            store=store,
        )

    assert exc_info.value.status_code == 400
    assert str(exc_info.value) == expected_msg
