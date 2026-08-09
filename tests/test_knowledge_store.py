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
    DocumentVersionConflictError,
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
    assert len({record.revision_id for record in records}) == 1
    packet = store.retrieve("Alice immutable data")
    assert packet.items[0].text == "Alice likes immutable data."
    with sqlite3.connect(store.index_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM document_versions").fetchone()[0] == 1


def test_update_creates_new_immutable_version_and_changes_revision(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge")
    first = store.upsert_document(
        "profile", "personal", "Profile", "Alice's favorite color is blue."
    )
    first_path = store.root / first.relative_path
    second = store.upsert_document(
        "profile",
        "personal",
        "Profile",
        "Alice's favorite color is green.",
        expected_revision_id=first.revision_id,
        revision_reason="content_update",
    )

    assert first.source_sha256 != second.source_sha256
    assert second.supersedes_revision_id == first.revision_id
    assert second.revision_reason == "content_update"
    assert first.corpus_revision != second.corpus_revision
    assert first_path.read_text(encoding="utf-8") == "Alice's favorite color is blue."
    assert (store.root / second.relative_path).read_text(encoding="utf-8").endswith("green.")
    packet = store.retrieve("Alice favorite color", top_k=4)
    assert packet.items
    assert all("green" in item.text for item in packet.items)


def test_title_change_is_part_of_corpus_revision(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge")
    first = store.upsert_document("notes", "work", "Old title", "release codename comet")
    second = store.upsert_document(
        "notes",
        "work",
        "New title",
        "release codename comet",
        expected_revision_id=first.revision_id,
        revision_reason="metadata_update",
    )
    assert first.source_sha256 == second.source_sha256
    assert first.corpus_revision != second.corpus_revision


def test_no_index_and_no_match_return_empty_packets(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge")
    before = store.retrieve("nothing indexed")
    assert before.items == ()
    assert before.coverage_limited is False
    assert before.coverage_reasons == ()
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


def test_single_character_and_stopword_only_queries_are_not_silently_dropped(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge")
    store.upsert_document(
        "launch",
        "work",
        "Launch",
        "Project X launches Friday. The approved level is 7. To be or not to be.",
    )

    assert store.retrieve("What is X?").items
    assert store.retrieve("Is it 7?").items
    assert store.retrieve("to be").items


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


def test_retrieval_rejects_forged_multiline_chunk_lines(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge")
    store.upsert_document(
        "profile",
        "personal",
        "Profile",
        "\nAlice likes blue.\nDetails follow.",
    )
    with sqlite3.connect(store.index_path) as conn:
        conn.execute("UPDATE chunks SET start_line = 1, end_line = 2")

    with pytest.raises(StaleKnowledgeIndexError, match="source lines"):
        store.retrieve("Alice")


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
    first = store.upsert_document("profile", "personal", "Profile", "Alice likes blue.")
    store.upsert_document(
        "profile",
        "archive",
        "Profile",
        "Alice likes green.",
        expected_revision_id=first.revision_id,
        revision_reason="content_update",
    )

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
    assert packet.coverage_limited is True
    assert "byte_budget" in packet.coverage_reasons
    assert len(encoded) <= MAX_PACKET_BYTES


def test_first_lexical_match_can_be_explicitly_omitted_by_packet_budget(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge")
    store.upsert_document(
        "oversized",
        "bounded",
        "\U0001f600" * 300,
        "needle " + ("\U0001f600" * 3_900),
    )

    packet = store.retrieve("needle " + ("\U0001f600" * 9_900))

    assert packet.items == ()
    assert packet.truncated is True
    assert packet.coverage_limited is True
    assert "byte_budget" in packet.coverage_reasons


def test_top_k_omission_is_explicit_and_distinct_from_byte_truncation(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge")
    for index in range(3):
        store.upsert_document(
            f"document_{index}",
            "coverage",
            f"Coverage {index}",
            f"needle evidence number {index}",
        )

    packet = store.retrieve("needle evidence", top_k=1)

    assert len(packet.items) == 1
    assert packet.truncated is False
    assert packet.coverage_limited is True
    assert packet.coverage_reasons == ("top_k",)
    assert packet.prompt_dict()["coverage_reasons"] == ["top_k"]


def test_query_term_limit_is_explicit_when_a_later_term_is_omitted(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge")
    store.upsert_document("profile", "personal", "Profile", "needle")
    query = " ".join([f"noise{index}" for index in range(24)] + ["needle"])

    packet = store.retrieve(query)

    assert packet.items == ()
    assert packet.truncated is False
    assert packet.coverage_limited is True
    assert packet.coverage_reasons == ("query_term_limit",)


@pytest.mark.parametrize("seed_corpus", [False, True])
def test_nfkc_expansion_cannot_exceed_packet_budget(tmp_path, seed_corpus):
    store = KnowledgeStore(tmp_path / "knowledge")
    if seed_corpus:
        store.upsert_document("profile", "personal", "Profile", "ordinary evidence")

    with pytest.raises(ValueError, match="canonical query exceeds"):
        store.retrieve("\ufdfa" * 10_000)


def test_retrieval_rejects_forged_active_revision_provenance(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge")
    store.upsert_document("profile", "personal", "Profile", "Alice likes blue.")
    with sqlite3.connect(store.index_path) as conn:
        conn.execute(
            "UPDATE documents SET active_revision_id = ? WHERE document_id = ?",
            ("f" * 64, "profile"),
        )

    with pytest.raises(StaleKnowledgeIndexError, match="immutable revision"):
        store.retrieve("Alice")


def test_retrieval_rejects_hash_invalid_revision_ledger_row(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge")
    record = store.upsert_document(
        "profile",
        "personal",
        "Profile",
        "Alice likes blue.",
    )
    with sqlite3.connect(store.index_path) as conn:
        row = conn.execute(
            """
            SELECT document_id, source_sha256, supersedes_revision_id, folder,
                   title, relative_path, chunk_count, revision_reason, created_at
            FROM document_versions WHERE revision_id = ?
            """,
            (record.revision_id,),
        ).fetchone()
        conn.execute(
            """
            INSERT INTO document_versions (
                revision_id, document_id, source_sha256, supersedes_revision_id,
                folder, title, relative_path, chunk_count, revision_reason, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("f" * 64, *row),
        )
        conn.execute(
            "UPDATE documents SET active_revision_id = ? WHERE document_id = ?",
            ("f" * 64, "profile"),
        )

    with pytest.raises(StaleKnowledgeIndexError, match="immutable revision"):
        store.retrieve("Alice")


def test_document_update_requires_exact_head_and_preserves_lineage(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge")
    first = store.upsert_document("profile", "personal", "Profile", "Alice likes blue.")

    with pytest.raises(DocumentVersionConflictError, match="expected_revision_id"):
        store.upsert_document("profile", "personal", "Profile", "Alice likes green.")

    second = store.upsert_document(
        "profile",
        "personal",
        "Profile",
        "Alice likes green.",
        expected_revision_id=first.revision_id,
        revision_reason="correction",
    )
    with pytest.raises(DocumentVersionConflictError, match="changed"):
        store.upsert_document(
            "profile",
            "personal",
            "Profile",
            "Alice likes red.",
            expected_revision_id=first.revision_id,
            revision_reason="correction",
        )

    assert second.supersedes_revision_id == first.revision_id
    assert {item.document_revision_id for item in store.retrieve("Alice likes").items} == {
        second.revision_id
    }
    with sqlite3.connect(store.index_path) as conn:
        rows = conn.execute(
            """
            SELECT revision_id, supersedes_revision_id, revision_reason
            FROM document_versions WHERE document_id = ? ORDER BY created_at
            """,
            ("profile",),
        ).fetchall()
        assert rows == [
            (first.revision_id, None, "initial_create"),
            (second.revision_id, first.revision_id, "correction"),
        ]
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute(
                "UPDATE document_versions SET revision_reason = 'restore' WHERE revision_id = ?",
                (first.revision_id,),
            )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute(
                """
                INSERT OR REPLACE INTO document_versions (
                    revision_id, document_id, source_sha256,
                    supersedes_revision_id, folder, title, relative_path,
                    chunk_count, revision_reason, created_at
                )
                SELECT revision_id, document_id, source_sha256,
                       supersedes_revision_id, folder, title, relative_path,
                       chunk_count, 'restore', created_at
                FROM document_versions WHERE revision_id = ?
                """,
                (first.revision_id,),
            )


def test_revision_reasons_match_the_actual_change(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge")
    first = store.upsert_document("profile", "personal", "Profile", "blue")

    with pytest.raises(ValueError, match="metadata_update cannot change"):
        store.upsert_document(
            "profile",
            "personal",
            "Profile",
            "green",
            expected_revision_id=first.revision_id,
            revision_reason="metadata_update",
        )
    with pytest.raises(ValueError, match="restore must target"):
        store.upsert_document(
            "profile",
            "personal",
            "Profile",
            "never active",
            expected_revision_id=first.revision_id,
            revision_reason="restore",
        )

    second = store.upsert_document(
        "profile",
        "personal",
        "Profile",
        "green",
        expected_revision_id=first.revision_id,
        revision_reason="correction",
    )
    restored = store.upsert_document(
        "profile",
        "personal",
        "Profile",
        "blue",
        expected_revision_id=second.revision_id,
        revision_reason="restore",
    )

    assert restored.source_sha256 == first.source_sha256
    assert restored.supersedes_revision_id == second.revision_id
    assert restored.revision_reason == "restore"


def test_concurrent_updates_from_one_head_allow_exactly_one_winner(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge")
    first = store.upsert_document("profile", "personal", "Profile", "Alice likes blue.")
    barrier = threading.Barrier(2)

    def update(color: str):
        barrier.wait(timeout=5)
        try:
            return store.upsert_document(
                "profile",
                "personal",
                "Profile",
                f"Alice likes {color}.",
                expected_revision_id=first.revision_id,
                revision_reason="correction",
            )
        except DocumentVersionConflictError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(update, ("green", "red")))

    winners = [result for result in results if not isinstance(result, Exception)]
    losers = [result for result in results if isinstance(result, DocumentVersionConflictError)]
    assert len(winners) == 1
    assert len(losers) == 1
    packet = store.retrieve("Alice likes")
    assert {item.document_revision_id for item in packet.items} == {winners[0].revision_id}
    assert any(color in packet.items[0].text for color in ("green", "red"))
    with sqlite3.connect(store.index_path) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM document_versions WHERE document_id = 'profile'"
        ).fetchone()[0] == 2


def test_schema_v1_active_documents_are_backfilled_into_revision_history(tmp_path):
    root = tmp_path / "knowledge"
    internal = root / ".rag"
    internal.mkdir(parents=True)
    index_path = internal / "index.sqlite3"
    with sqlite3.connect(index_path) as conn:
        conn.execute(
            """
            CREATE TABLE documents (
                document_id TEXT PRIMARY KEY,
                folder TEXT NOT NULL,
                title TEXT NOT NULL,
                source_sha256 TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                chunk_count INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                "legacy",
                "archive",
                "Legacy",
                "a" * 64,
                f"sources/archive/legacy/versions/{'a' * 64}.txt",
                1,
                "2026-01-01T00:00:00+00:00",
            ),
        )

    store = KnowledgeStore(root)
    migrated = store._connect(create=False)
    assert migrated is not None
    try:
        row = migrated.execute(
            "SELECT active_revision_id FROM documents WHERE document_id = 'legacy'"
        ).fetchone()
        version = migrated.execute(
            "SELECT revision_reason FROM document_versions WHERE document_id = 'legacy'"
        ).fetchone()
    finally:
        migrated.close()

    assert row["active_revision_id"]
    assert version["revision_reason"] == "legacy_import"


def test_stale_inactive_chunk_cannot_enter_active_packet(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge")
    first = store.upsert_document("profile", "personal", "Profile", "Alice likes blue.")
    old = store.retrieve("Alice likes").items[0]
    store.upsert_document(
        "profile",
        "personal",
        "Profile",
        "Alice likes green.",
        expected_revision_id=first.revision_id,
        revision_reason="content_update",
    )

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
