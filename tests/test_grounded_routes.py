"""API boundary tests for the private grounded corpus."""
from __future__ import annotations

from fastapi.testclient import TestClient

import config
from app import app

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

    no_match = client.post(
        "/api/grounded/query",
        headers=headers,
        json={"prompt": "orbital mechanics"},
    )
    assert no_match.status_code == 200
    assert no_match.json()["status"] == "ABSTAIN"
    assert no_match.json()["reason_code"] == "no_lexical_match"


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
