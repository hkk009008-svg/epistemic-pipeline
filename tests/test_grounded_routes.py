"""API boundary tests for the private grounded corpus."""
from __future__ import annotations

import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

import config
from app import app
from pipeline.knowledge_store import KnowledgeStore, KnowledgeStoreError

client = TestClient(app, raise_server_exceptions=False)


def test_grounded_endpoints_are_disabled_without_token(monkeypatch):
    monkeypatch.setattr(config, "KNOWLEDGE_API_TOKEN", "")
    response = client.post("/api/grounded/query", json={"prompt": "test"})
    assert response.status_code == 503


def test_grounded_endpoints_reject_wrong_token(monkeypatch):
    monkeypatch.setattr(config, "KNOWLEDGE_API_TOKEN", "secret")
    response = client.post(
        "/api/grounded/query",
        headers={"Authorization": "Bearer wrong"},
        json={"prompt": "test"},
    )
    assert response.status_code == 401


def test_grounded_endpoints_reject_non_ascii_token_without_server_error(monkeypatch):
    monkeypatch.setattr(config, "KNOWLEDGE_API_TOKEN", "secret")
    response = client.post(
        "/api/grounded/query",
        headers={b"Authorization": b"Bearer wrong-\xff"},
        json={"prompt": "test"},
    )
    assert response.status_code == 401


def test_grounded_token_also_closes_shared_model_control_plane(monkeypatch):
    monkeypatch.setattr(config, "ADMIN_TOKEN", "")
    monkeypatch.setattr(config, "KNOWLEDGE_API_TOKEN", "secret")
    response = client.post(
        "/api/stage/config",
        json={
            "stage": "gpt1",
            "provider": "openai",
            "api_key": "attacker-key",
            "model": "attacker-model",
        },
    )
    assert response.status_code == 401


def test_document_upsert_and_empty_query_round_trip(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "KNOWLEDGE_API_TOKEN", "secret")
    monkeypatch.setattr(config, "KNOWLEDGE_ROOT", str(tmp_path / "knowledge"))
    headers = {"Authorization": "Bearer secret"}

    upsert = client.post(
        "/api/grounded/documents/profile",
        headers=headers,
        json={
            "folder": "personal/preferences",
            "title": "Profile",
            "content": "Alice's preferred editor is Neovim.",
        },
    )
    assert upsert.status_code == 200
    assert upsert.json()["document_id"] == "profile"
    assert upsert.json()["folder"] == "personal/preferences"
    assert upsert.json()["revision_reason"] == "initial_create"

    no_match = client.post(
        "/api/grounded/query",
        headers=headers,
        json={"prompt": "orbital mechanics"},
    )
    assert no_match.status_code == 200
    assert no_match.json()["status"] == "ABSTAIN"
    assert no_match.json()["reason_code"] == "no_lexical_match"
    assert no_match.json()["contract_version"] == "grounded-rag-v1"
    assert no_match.json()["coverage_limited"] is False
    assert no_match.json()["receipt_sha256"]

    store = KnowledgeStore(config.KNOWLEDGE_ROOT)
    receipt = store.load_run_receipt(no_match.json()["run_id"])
    serialized = json.dumps(receipt, sort_keys=True)
    assert receipt["reason_code"] == "no_lexical_match"
    assert "orbital mechanics" not in serialized
    assert "preferred editor" not in serialized
    injected_receipt = dict(receipt)
    injected_receipt["private_payload"] = "preferred editor"
    injected_receipt["run_id"] = "f" * 32
    with pytest.raises(KnowledgeStoreError, match="fields do not match"):
        store.append_run_receipt(injected_receipt)
    with sqlite3.connect(store.index_path) as conn:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute(
                "UPDATE grounded_run_receipts SET created_at = 'changed' WHERE run_id = ?",
                (no_match.json()["run_id"],),
            )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute(
                """
                INSERT OR REPLACE INTO grounded_run_receipts (
                    run_id, receipt_json, receipt_sha256, created_at
                )
                SELECT run_id, '{}', receipt_sha256, created_at
                FROM grounded_run_receipts WHERE run_id = ?
                """,
                (no_match.json()["run_id"],),
            )


def test_nfkc_expanded_query_is_rejected_before_retrieval(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "KNOWLEDGE_API_TOKEN", "secret")
    monkeypatch.setattr(config, "KNOWLEDGE_ROOT", str(tmp_path / "knowledge"))

    response = client.post(
        "/api/grounded/query",
        headers={"Authorization": "Bearer secret"},
        json={"prompt": "\ufdfa" * 10_000},
    )

    assert response.status_code == 422
    assert not (tmp_path / "knowledge").exists()


def test_document_update_uses_revision_compare_and_swap(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "KNOWLEDGE_API_TOKEN", "secret")
    monkeypatch.setattr(config, "KNOWLEDGE_ROOT", str(tmp_path / "knowledge"))
    headers = {"Authorization": "Bearer secret"}
    endpoint = "/api/grounded/documents/profile"

    first = client.post(
        endpoint,
        headers=headers,
        json={"folder": "personal", "title": "Profile", "content": "Alice likes blue."},
    )
    assert first.status_code == 200
    first_revision = first.json()["revision_id"]

    missing_head = client.post(
        endpoint,
        headers=headers,
        json={"folder": "personal", "title": "Profile", "content": "Alice likes green."},
    )
    assert missing_head.status_code == 409

    second = client.post(
        endpoint,
        headers=headers,
        json={
            "folder": "personal",
            "title": "Profile",
            "content": "Alice likes green.",
            "expected_revision_id": first_revision,
            "revision_reason": "correction",
        },
    )
    assert second.status_code == 200
    assert second.json()["supersedes_revision_id"] == first_revision

    stale = client.post(
        endpoint,
        headers=headers,
        json={
            "folder": "personal",
            "title": "Profile",
            "content": "Alice likes red.",
            "expected_revision_id": first_revision,
            "revision_reason": "correction",
        },
    )
    assert stale.status_code == 409


def test_document_path_traversal_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "KNOWLEDGE_API_TOKEN", "secret")
    monkeypatch.setattr(config, "KNOWLEDGE_ROOT", str(tmp_path / "knowledge"))
    response = client.post(
        "/api/grounded/documents/..",
        headers={"Authorization": "Bearer secret"},
        json={"folder": "personal", "title": "Bad", "content": "data"},
    )
    # The ASGI router may normalize '..' before the handler, but it must never write.
    assert response.status_code in (400, 404, 405)
    assert not (tmp_path / "escape").exists()
