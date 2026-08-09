"""Deterministic storage, isolation, provenance, and retrieval tests."""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from pipeline.knowledge_store import (
    MAX_CHUNK_CHARS,
    MAX_PACKET_BYTES,
    KnowledgeStore,
    KnowledgeStoreError,
    StaleKnowledgeIndexError,
    chunk_text,
    normalize_folder,
)


def test_chunk_text_preserves_exact_source_spans_and_overlap():
    content = " ".join(f"word{i}" for i in range(240)) + "\nfinal line"
    chunks = chunk_text(content, max_words=50, overlap_words=10)

    assert len(chunks) > 1
    for chunk in chunks:
        assert content[chunk["start_char"]:chunk["end_char"]] == chunk["text"]
        assert chunk["start_line"] >= 1
        assert chunk["end_line"] >= chunk["start_line"]
    assert set(chunks[0]["text"].split()) & set(chunks[1]["text"].split())


def test_chunk_text_bounds_whitespace_free_spans():
    content = "x" * (MAX_CHUNK_CHARS * 3)
    chunks = chunk_text(content)

    assert len(chunks) > 1
    assert max(len(chunk["text"]) for chunk in chunks) <= MAX_CHUNK_CHARS
    assert chunks[0]["end_char"] - chunks[1]["start_char"] > 0
    for chunk in chunks:
        assert content[chunk["start_char"]:chunk["end_char"]] == chunk["text"]


@pytest.mark.parametrize(
    "folder",
    ["../private", "/absolute", "a/../../b", "a\\b", "a//b", ".", ".."],
)
def test_folder_traversal_and_ambiguous_paths_are_rejected(folder):
    with pytest.raises(ValueError):
        normalize_folder(folder)


def test_upsert_retrieve_and_packet_are_stable(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge")
    record = store.upsert_document(
        "profile",
        "personal/preferences",
        "User preferences",
        "Alice's preferred editor is Neovim. Alice uses a dark color theme.",
    )

    packet_a = store.retrieve("What editor does Alice prefer?", top_k=4)
    packet_b = store.retrieve("What editor does Alice prefer?", top_k=4)

    assert record.chunk_count == 1
    assert packet_a == packet_b
    assert packet_a.packet_id == packet_b.packet_id
    assert packet_a.corpus_revision == record.corpus_revision
    assert [item.rank for item in packet_a.items] == [1]
    item = packet_a.items[0]
    assert item.document_id == "profile"
    assert item.folder == "personal/preferences"
    assert item.text == "Alice's preferred editor is Neovim. Alice uses a dark color theme."
    assert item.relative_path.endswith(f"/{record.source_sha256}.txt")
    assert (store.root / item.relative_path).read_text(encoding="utf-8") == item.text


def test_concurrent_identical_upserts_publish_only_complete_source(monkeypatch, tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge")
    real_link = os.link
    first_at_publish = threading.Event()
    release_first = threading.Event()
    calls = 0
    calls_lock = threading.Lock()

    def controlled_link(source, destination):
        nonlocal calls
        with calls_lock:
            calls += 1
            call_number = calls
        if call_number == 1:
            first_at_publish.set()
            assert release_first.wait(timeout=5)
        else:
            assert first_at_publish.wait(timeout=5)
            release_first.set()
        return real_link(source, destination)

    monkeypatch.setattr(os, "link", controlled_link)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                store.upsert_document,
                "profile",
                "personal",
                "Profile",
                "Alice likes immutable data.",
            )
            for _ in range(2)
        ]
        records = [future.result(timeout=10) for future in futures]

    assert len(records) == 2
    packet = store.retrieve("Alice immutable data")
    assert packet.items[0].text == "Alice likes immutable data."

def test_update_creates_new_immutable_version_and_changes_revision(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge")
    first = store.upsert_document(
        "profile", "personal", "Profile", "Alice's favorite color is blue."
    )
    first_path = store.root / first.relative_path
    second = store.upsert_document(
        "profile", "personal", "Profile", "Alice's favorite color is green."
    )

    assert first.source_sha256 != second.source_sha256
    assert first.corpus_revision != second.corpus_revision
    assert first_path.read_text(encoding="utf-8") == "Alice's favorite color is blue."
    assert (store.root / second.relative_path).read_text(encoding="utf-8").endswith("green.")
    packet = store.retrieve("Alice favorite color", top_k=4)
    assert packet.items
    assert all("green" in item.text for item in packet.items)


def test_title_change_is_part_of_corpus_revision(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge")
    first = store.upsert_document("notes", "work", "Old title", "release codename comet")
    second = store.upsert_document("notes", "work", "New title", "release codename comet")
    assert first.source_sha256 == second.source_sha256
    assert first.corpus_revision != second.corpus_revision


def test_no_index_and_no_match_return_empty_packets(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge")
    before = store.retrieve("nothing indexed")
    assert before.items == ()
    assert not store.root.exists()

    store.upsert_document("profile", "personal", "Profile", "Alice likes tea.")
    after = store.retrieve("orbital mechanics")
    assert after.items == ()
    assert after.corpus_revision != before.corpus_revision


def test_query_punctuation_is_escaped_for_fts(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge")
    store.upsert_document("profile", "personal", "Profile", "Alice uses C++ for simulations.")
    packet = store.retrieve('Alice AND (C++) "simulations"?', top_k=3)
    assert packet.items


def test_document_identifier_is_not_a_path(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge")
    with pytest.raises(ValueError):
        store.upsert_document("../escape", "personal", "Bad", "content")


def test_source_tampering_fails_closed(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge")
    record = store.upsert_document(
        "profile", "personal", "Profile", "Alice's favorite color is blue."
    )
    source_path = Path(store.root / record.relative_path)
    source_path.write_text("Alice's favorite color is red.", encoding="utf-8")

    with pytest.raises(StaleKnowledgeIndexError):
        store.retrieve("Alice favorite color")


def test_index_directory_symlink_cannot_escape_root(tmp_path):
    root = tmp_path / "knowledge"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / ".rag").symlink_to(outside, target_is_directory=True)
    store = KnowledgeStore(root)

    with pytest.raises(KnowledgeStoreError):
        store.retrieve("anything")
    assert not (outside / "index.sqlite3").exists()


def test_sources_symlink_cannot_escape_root(tmp_path):
    root = tmp_path / "knowledge"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "sources").symlink_to(outside, target_is_directory=True)
    store = KnowledgeStore(root)

    with pytest.raises(KnowledgeStoreError):
        store.upsert_document("profile", "personal", "Profile", "private data")
    assert list(outside.iterdir()) == []


def test_same_document_id_moves_instead_of_leaving_two_active_entries(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge")
    store.upsert_document("profile", "personal", "Profile", "Alice likes blue.")
    store.upsert_document("profile", "archive", "Profile", "Alice likes green.")

    packet = store.retrieve("profile Alice likes", top_k=12)
    assert packet.items
    assert {item.folder for item in packet.items} == {"archive"}
    with sqlite3.connect(store.index_path) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM documents WHERE document_id = 'profile'"
        ).fetchone()[0] == 1


def test_folder_and_document_identity_participate_in_retrieval(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge")
    store.upsert_document("frey_mesh", "projects/kurogane", "Notes", "hidden scalp work")

    by_folder = store.retrieve("kurogane")
    by_document = store.retrieve("frey_mesh")
    assert by_folder.items[0].folder == "projects/kurogane"
    assert by_document.items[0].document_id == "frey_mesh"


def test_evidence_packet_has_a_deterministic_byte_budget(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge")
    for index in range(12):
        store.upsert_document(
            f"document_{index}",
            "bounded",
            f"Bounded {index}",
            "needle " + ("U0001f600" * (MAX_CHUNK_CHARS - 20)),
        )

    packet = store.retrieve("needle", top_k=12)
    encoded = json.dumps(
        packet.prompt_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    assert packet.items
    assert packet.truncated is True
    assert len(encoded) <= MAX_PACKET_BYTES


def test_stale_inactive_chunk_cannot_enter_active_packet(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge")
    store.upsert_document("profile", "personal", "Profile", "Alice likes blue.")
    old = store.retrieve("Alice likes").items[0]
    store.upsert_document("profile", "personal", "Profile", "Alice likes green.")

    with sqlite3.connect(store.index_path) as conn:
        conn.execute(
            "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                old.evidence_id + "-stale",
                old.document_id,
                old.folder,
                old.title,
                old.relative_path,
                old.source_sha256,
                old.chunk_sha256,
                old.start_char,
                old.end_char,
                old.start_line,
                old.end_line,
                old.text,
            ),
        )
        conn.commit()

    packet = store.retrieve("Alice likes", top_k=12)
    assert packet.items
    assert all("green" in item.text for item in packet.items)
    assert all(item.source_sha256 != old.source_sha256 for item in packet.items)


def test_duplicate_evidence_ids_in_retrieved_packet_fail_closed(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge")
    store.upsert_document("profile", "personal", "Profile", "Alice likes green.")

    with sqlite3.connect(store.index_path) as conn:
        row = conn.execute("SELECT * FROM chunks LIMIT 1").fetchone()
        conn.execute(
            "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            row,
        )
        conn.commit()

    with pytest.raises(StaleKnowledgeIndexError):
        store.retrieve("Alice likes", top_k=12)
