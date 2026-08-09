"""Fail-closed contract tests for the isolated three-role grounded lane."""
from __future__ import annotations

import json

import pytest

import pipeline.grounded_rag as grounded
from pipeline.grounded_rag import (
    AnswererOutput,
    FinalizerOutput,
    GroundedQueryRequest,
    VerifierOutput,
    run_grounded_rag,
)
from pipeline.knowledge_store import EvidenceItem, EvidencePacket, KnowledgeStore


def _packet(with_items: bool = True) -> EvidencePacket:
    items = []
    if with_items:
        text = "Alice's favorite color is blue. Alice started using it in 2024."
        items.append(EvidenceItem(
            evidence_id="E-source-1",
            rank=1,
            retrieval_score=-1.0,
            document_id="profile",
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

    def retrieve(self, query: str, top_k: int) -> EvidencePacket:
        return self.packet


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
        store=StubStore(packet),
    )

    assert response.status == "ANSWER"
    assert response.answer == "- Alice's favorite color is blue. [1]"
    assert response.citations[0].quote == "Alice's favorite color is blue."
    assert response.stages_completed == ["retrieval", "answerer", "verifier", "finalizer"]
    assert len(calls) == 3
    assert len({_packet_bytes_from_prompt(call[1]) for call in calls}) == 1


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
async def test_invalid_verifier_quote_cannot_reach_final_output(monkeypatch):
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
        store=StubStore(packet),
    )
    assert response.status == "ABSTAIN"
    assert response.reason_code == "no_supported_claims"
    assert response.stages_completed == ["retrieval", "answerer", "verifier"]
    assert "blue" not in response.answer


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
