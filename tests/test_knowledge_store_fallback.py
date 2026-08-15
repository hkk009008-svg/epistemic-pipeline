"""Unit tests for KnowledgeStore cross-device and unsupported filesystem link fallbacks."""
from __future__ import annotations

import errno
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from pipeline.knowledge_store import (
    KnowledgeStore,
)


def test_upsert_document_os_link_exdev_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verifies that KnowledgeStore falls back to os.replace when os.link raises EXDEV (cross-device link)."""
    store = KnowledgeStore(tmp_path / "knowledge")
    link_called = False

    def mock_link(src: Path | str, dst: Path | str):
        nonlocal link_called
        link_called = True
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    monkeypatch.setattr(os, "link", mock_link)

    record = store.upsert_document(
        "doc_exdev",
        "engineering/storage",
        "Cross Mount Document",
        "Content stored across different volume mounts.",
    )

    assert link_called is True
    assert record.document_id == "doc_exdev"
    assert record.chunk_count == 1

    # Verify source file was placed properly and matches hash
    source_file = store.root / record.relative_path
    assert source_file.exists()
    assert source_file.read_text(encoding="utf-8") == "Content stored across different volume mounts."

    # Verify retrieval works seamlessly
    packet = store.retrieve("volume mounts", top_k=5)
    assert len(packet.items) == 1
    assert packet.items[0].document_id == "doc_exdev"
    assert packet.items[0].text == "Content stored across different volume mounts."


def test_upsert_document_os_link_eperm_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verifies that KnowledgeStore falls back to os.replace when os.link raises EPERM (operation not permitted)."""
    store = KnowledgeStore(tmp_path / "knowledge")
    link_called = False

    def mock_link(src: Path | str, dst: Path | str):
        nonlocal link_called
        link_called = True
        raise OSError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(os, "link", mock_link)

    record = store.upsert_document(
        "doc_eperm",
        "security/sandboxes",
        "Hardlink Restricted Sandbox",
        "Content stored in an unprivileged container with protected hardlinks.",
    )

    assert link_called is True
    assert record.document_id == "doc_eperm"
    source_file = store.root / record.relative_path
    assert source_file.exists()

    packet = store.retrieve("protected hardlinks", top_k=5)
    assert len(packet.items) == 1
    assert packet.items[0].document_id == "doc_eperm"


def test_upsert_document_os_link_enosys_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verifies fallback when os.link raises ENOSYS (function not implemented)."""
    store = KnowledgeStore(tmp_path / "knowledge")

    def mock_link(src: Path | str, dst: Path | str):
        raise OSError(errno.ENOSYS, "Function not implemented")

    monkeypatch.setattr(os, "link", mock_link)

    record = store.upsert_document(
        "doc_enosys",
        "system/syscalls",
        "No Syscall Support",
        "System does not implement the link syscall.",
    )

    assert record.document_id == "doc_enosys"
    source_file = store.root / record.relative_path
    assert source_file.exists()


def test_upsert_document_os_link_enotsup_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verifies fallback when os.link raises ENOTSUP (operation not supported)."""
    store = KnowledgeStore(tmp_path / "knowledge")

    enotsup = getattr(errno, "ENOTSUP", getattr(errno, "EOPNOTSUPP", errno.EPERM))

    def mock_link(src: Path | str, dst: Path | str):
        raise OSError(enotsup, "Operation not supported")

    monkeypatch.setattr(os, "link", mock_link)

    record = store.upsert_document(
        "doc_enotsup",
        "system/fuse",
        "FUSE Mount",
        "FUSE filesystem does not support hard link creation.",
    )

    assert record.document_id == "doc_enotsup"
    source_file = store.root / record.relative_path
    assert source_file.exists()


def test_upsert_document_os_link_emlink_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verifies fallback when os.link raises EMLINK (too many links)."""
    store = KnowledgeStore(tmp_path / "knowledge")

    def mock_link(src: Path | str, dst: Path | str):
        raise OSError(errno.EMLINK, "Too many links")

    monkeypatch.setattr(os, "link", mock_link)

    record = store.upsert_document(
        "doc_emlink",
        "system/inodes",
        "Max Inode Links",
        "Underlying inode reached link limit.",
    )

    assert record.document_id == "doc_emlink"
    source_file = store.root / record.relative_path
    assert source_file.exists()


def test_upsert_document_os_link_unhandled_oserror_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verifies that non-link-related OSErrors (e.g. EACCES on parent directory) are not masked."""
    store = KnowledgeStore(tmp_path / "knowledge")

    def mock_link(src: Path | str, dst: Path | str):
        raise OSError(errno.EACCES, "Permission denied")

    monkeypatch.setattr(os, "link", mock_link)

    with pytest.raises(OSError) as exc_info:
        store.upsert_document(
            "doc_eacces",
            "system/perms",
            "Permission Denied Document",
            "Should raise OSError EACCES.",
        )
    assert exc_info.value.errno == errno.EACCES


def test_upsert_document_concurrent_fallback_identical(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verifies that concurrent threads encountering EXDEV on the same content publish safely."""
    store = KnowledgeStore(tmp_path / "knowledge")

    def mock_link(src: Path | str, dst: Path | str):
        raise OSError(errno.EXDEV, "Cross-device link")

    monkeypatch.setattr(os, "link", mock_link)

    doc_text = "Highly concurrent content published simultaneously under EXDEV."

    def run_upsert(thread_id: int):
        return store.upsert_document(
            "concurrent_doc",
            "concurrency/race",
            "Concurrent Document Title",
            doc_text,
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_upsert, i) for i in range(4)]
        results = [f.result() for f in futures]

    assert len(results) == 4
    for r in results:
        assert r.document_id == "concurrent_doc"

    # Verify packet retrieval
    packet = store.retrieve("concurrent content", top_k=5)
    assert len(packet.items) == 1
    assert packet.items[0].text == doc_text
