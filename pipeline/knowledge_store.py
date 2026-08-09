"""Versioned local knowledge storage and deterministic SQLite FTS5 retrieval.

Content-addressed filesystem blobs preserve immutable source bytes. SQLite is
the authoritative active-document map and retrieval index for this first
slice. A query materializes one immutable EvidencePacket before any model is
called; models never receive filesystem access or choose their own corpus.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile
import threading
import time
import unicodedata
from bisect import bisect_right
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1
RETRIEVER_VERSION = "sqlite-fts5-v1"
CHUNKER_VERSION = "words-180-overlap-30-chars-4000-v2"
MAX_DOCUMENT_CHARS = 2_000_000
MAX_FOLDER_DEPTH = 8
MAX_CHUNK_CHARS = 4_000
CHUNK_CHAR_OVERLAP = 256
MAX_PACKET_BYTES = 48_000

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_WORD_RE = re.compile(r"\S+")
_QUERY_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_STOP_WORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "can",
    "did", "do", "does", "for", "from", "had", "has", "have", "how", "i",
    "in", "into", "is", "it", "its", "may", "of", "on", "or", "our",
    "should", "that", "the", "their", "them", "they", "this", "to", "was",
    "were", "what", "when", "where", "which", "who", "why", "will", "with",
    "would", "you", "your",
})
_SCHEMA_LOCK = threading.Lock()
_SCHEMA_INIT_RETRIES = 20


class KnowledgeStoreError(RuntimeError):
    """Base error for local knowledge storage failures."""


class StaleKnowledgeIndexError(KnowledgeStoreError):
    """Raised when the derived index no longer matches immutable source bytes."""


@dataclass(frozen=True)
class DocumentRecord:
    document_id: str
    folder: str
    title: str
    source_sha256: str
    relative_path: str
    chunk_count: int
    corpus_revision: str


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    rank: int
    retrieval_score: float
    document_id: str
    folder: str
    title: str
    relative_path: str
    source_sha256: str
    chunk_sha256: str
    start_char: int
    end_char: int
    start_line: int
    end_line: int
    text: str

    def prompt_dict(self) -> dict:
        """Return the data sent to models, excluding no provenance fields."""
        return asdict(self)


@dataclass(frozen=True)
class EvidencePacket:
    packet_id: str
    corpus_revision: str
    retrieval_version: str
    canonical_query: str
    truncated: bool
    items: tuple[EvidenceItem, ...]

    def prompt_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "packet_id": self.packet_id,
            "corpus_revision": self.corpus_revision,
            "retrieval_version": self.retrieval_version,
            "canonical_query": self.canonical_query,
            "truncated": self.truncated,
            "items": [item.prompt_dict() for item in self.items],
        }


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonicalize_query(query: str) -> str:
    """Normalize a query without allowing an LLM to rewrite its meaning."""
    return " ".join(unicodedata.normalize("NFKC", query).split())


def _query_terms(query: str) -> list[str]:
    seen: set[str] = set()
    terms: list[str] = []
    for token in _QUERY_TOKEN_RE.findall(query.casefold()):
        if token in _STOP_WORDS or len(token) < 2 or token in seen:
            continue
        seen.add(token)
        terms.append(token)
        if len(terms) == 24:
            break
    return terms


def _validate_identifier(value: str, label: str) -> str:
    if not _ID_RE.fullmatch(value or ""):
        raise ValueError(
            f"{label} must start with an alphanumeric character and contain only "
            "letters, numbers, '_' or '-' (maximum 64 characters)."
        )
    return value


def normalize_folder(folder: str) -> str:
    """Validate and normalize a relative logical folder path."""
    raw = (folder or "general").strip()
    if not raw:
        raw = "general"
    if "\\" in raw or raw.startswith("/"):
        raise ValueError("folder must be a relative POSIX-style path")
    parts = raw.split("/")
    if len(parts) > MAX_FOLDER_DEPTH:
        raise ValueError(f"folder may contain at most {MAX_FOLDER_DEPTH} levels")
    for part in parts:
        _validate_identifier(part, "folder segment")
    return "/".join(parts)


def chunk_text(content: str, max_words: int = 180, overlap_words: int = 30) -> list[dict]:
    """Split text into deterministic, overlapping exact-source spans."""
    if max_words < 1 or overlap_words < 0 or overlap_words >= max_words:
        raise ValueError("invalid chunk sizing")
    words = list(_WORD_RE.finditer(content))
    if not words:
        return []
    newline_offsets = [index for index, character in enumerate(content) if character == "\n"]

    chunks: list[dict] = []

    def append_span(span_start: int, span_end: int) -> None:
        while span_start < span_end:
            bounded_end = min(span_start + MAX_CHUNK_CHARS, span_end)
            text = content[span_start:bounded_end]
            chunks.append({
                "ordinal": len(chunks),
                "start_char": span_start,
                "end_char": bounded_end,
                "start_line": bisect_right(newline_offsets, span_start - 1) + 1,
                "end_line": bisect_right(newline_offsets, bounded_end - 1) + 1,
                "text": text,
            })
            if bounded_end == span_end:
                break
            span_start = bounded_end - CHUNK_CHAR_OVERLAP

    step = max_words - overlap_words
    for word_start in range(0, len(words), step):
        word_end = min(word_start + max_words, len(words))
        start_char = words[word_start].start()
        end_char = words[word_end - 1].end()
        append_span(start_char, end_char)
        if word_end == len(words):
            break
    return chunks


class KnowledgeStore:
    """Single authenticated corpus backed by immutable source versions + FTS5."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.sources_dir = self.root / "sources"
        self.internal_dir = self.root / ".rag"
        self.index_path = self.internal_dir / "index.sqlite3"

    def _safe_child(self, parent: Path, *parts: str) -> Path:
        candidate = parent.joinpath(*parts).resolve(strict=False)
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise KnowledgeStoreError("resolved knowledge path escapes configured root") from exc
        return candidate

    def _connect(self, *, create: bool) -> sqlite3.Connection | None:
        safe_index_path = self._safe_child(self.root, ".rag", "index.sqlite3")
        if self.internal_dir.is_symlink():
            raise KnowledgeStoreError("knowledge index directory must not be a symbolic link")
        if not create and not safe_index_path.exists():
            return None
        if safe_index_path.is_symlink():
            raise KnowledgeStoreError("knowledge index must not be a symbolic link")
        if create:
            self.internal_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            try:
                os.chmod(self.internal_dir, 0o700)
            except OSError:
                pass
        conn = sqlite3.connect(safe_index_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            if create:
                # Serialize same-process first use, and retry for concurrent
                # service processes contending on the initial WAL transition.
                with _SCHEMA_LOCK:
                    for attempt in range(_SCHEMA_INIT_RETRIES):
                        try:
                            self._ensure_schema(conn)
                            break
                        except sqlite3.OperationalError as exc:
                            conn.rollback()
                            message = str(exc).lower()
                            busy = "locked" in message or "busy" in message
                            if not busy or attempt == _SCHEMA_INIT_RETRIES - 1:
                                raise
                            time.sleep(0.05)
        except Exception:
            conn.close()
            raise
        return conn

    @staticmethod
    def _ensure_schema(conn: sqlite3.Connection) -> None:
        conn.execute("PRAGMA busy_timeout=10000")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
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
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
                evidence_id UNINDEXED,
                document_id,
                folder,
                title,
                relative_path UNINDEXED,
                source_sha256 UNINDEXED,
                chunk_sha256 UNINDEXED,
                start_char UNINDEXED,
                end_char UNINDEXED,
                start_line UNINDEXED,
                end_line UNINDEXED,
                body,
                tokenize='unicode61 remove_diacritics 2'
            )
            """
        )
        conn.commit()

    @staticmethod
    def _corpus_revision(conn: sqlite3.Connection | None) -> str:
        if conn is None:
            return _sha256_bytes(_canonical_json({
                "chunker_version": CHUNKER_VERSION,
                "retriever_version": RETRIEVER_VERSION,
                "documents": [],
            }))
        rows = conn.execute(
            """
            SELECT document_id, source_sha256, title, folder, relative_path, chunk_count
            FROM documents
            ORDER BY document_id
            """
        ).fetchall()
        return _sha256_bytes(_canonical_json({
            "chunker_version": CHUNKER_VERSION,
            "retriever_version": RETRIEVER_VERSION,
            "documents": [list(row) for row in rows],
        }))

    def upsert_document(
        self,
        document_id: str,
        folder: str,
        title: str,
        content: str,
    ) -> DocumentRecord:
        """Create an immutable source version and make it the active indexed version."""
        document_id = _validate_identifier(document_id, "document_id")
        folder = normalize_folder(folder)
        title = " ".join((title or "").split())
        if not title or len(title) > 300:
            raise ValueError("title must contain between 1 and 300 characters")
        if not content.strip():
            raise ValueError("content must not be empty")
        if len(content) > MAX_DOCUMENT_CHARS:
            raise ValueError(f"content exceeds {MAX_DOCUMENT_CHARS} characters")

        source_bytes = content.encode("utf-8")
        source_sha = _sha256_bytes(source_bytes)
        folder_parts = folder.split("/")
        document_dir = self._safe_child(self.sources_dir, *folder_parts, document_id)
        versions_dir = self._safe_child(document_dir, "versions")
        version_path = self._safe_child(versions_dir, f"{source_sha}.txt")
        relative_path = version_path.relative_to(self.root).as_posix()

        for directory in (self.root, self.sources_dir, document_dir, versions_dir):
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            try:
                os.chmod(directory, 0o700)
            except OSError:
                pass
        if version_path.is_symlink():
            raise KnowledgeStoreError("source version must not be a symbolic link")
        if not version_path.exists():
            temporary_path: Path | None = None
            try:
                fd, temporary_name = tempfile.mkstemp(
                    dir=versions_dir,
                    prefix=f".{source_sha}.",
                    suffix=".tmp",
                )
                temporary_path = Path(temporary_name)
                os.fchmod(fd, 0o600)
                with os.fdopen(fd, "wb") as handle:
                    handle.write(source_bytes)
                    handle.flush()
                    os.fsync(handle.fileno())
                try:
                    # The hard link is an atomic no-clobber publication. A
                    # concurrent writer can observe only the complete blob.
                    os.link(temporary_path, version_path)
                except FileExistsError:
                    pass
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
        if _sha256_bytes(version_path.read_bytes()) != source_sha:
            raise KnowledgeStoreError("content-addressed source file has been modified")

        chunks = chunk_text(content)
        if not chunks:
            raise ValueError("content did not produce any searchable chunks")
        now = datetime.now(timezone.utc).isoformat()
        conn = self._connect(create=True)
        assert conn is not None
        try:
            with conn:
                conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
                for chunk in chunks:
                    chunk_sha = _sha256_bytes(chunk["text"].encode("utf-8"))
                    evidence_seed = (
                        f"{document_id}:{source_sha}:{chunk['ordinal']}:".encode()
                    )
                    evidence_id = f"E-{_sha256_bytes(evidence_seed)[:32]}"
                    conn.execute(
                        """
                        INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            evidence_id,
                            document_id,
                            folder,
                            title,
                            relative_path,
                            source_sha,
                            chunk_sha,
                            chunk["start_char"],
                            chunk["end_char"],
                            chunk["start_line"],
                            chunk["end_line"],
                            chunk["text"],
                        ),
                    )
                conn.execute(
                    """
                    INSERT INTO documents (
                        document_id, folder, title, source_sha256,
                        relative_path, chunk_count, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(document_id) DO UPDATE SET
                        folder=excluded.folder,
                        title=excluded.title,
                        source_sha256=excluded.source_sha256,
                        relative_path=excluded.relative_path,
                        chunk_count=excluded.chunk_count,
                        updated_at=excluded.updated_at
                    """,
                    (
                        document_id,
                        folder,
                        title,
                        source_sha,
                        relative_path,
                        len(chunks),
                        now,
                    ),
                )

            corpus_revision = self._corpus_revision(conn)
        finally:
            conn.close()
        return DocumentRecord(
            document_id=document_id,
            folder=folder,
            title=title,
            source_sha256=source_sha,
            relative_path=relative_path,
            chunk_count=len(chunks),
            corpus_revision=corpus_revision,
        )

    def _verify_sources(self, rows: Iterable[sqlite3.Row]) -> None:
        source_cache: dict[tuple[str, str], str] = {}
        for row in rows:
            key = (row["relative_path"], row["source_sha256"])
            if key not in source_cache:
                source_path = self._safe_child(self.root, *Path(row["relative_path"]).parts)
                if source_path.is_symlink() or not source_path.is_file():
                    raise StaleKnowledgeIndexError("indexed source is missing or unsafe")
                source_bytes = source_path.read_bytes()
                if _sha256_bytes(source_bytes) != row["source_sha256"]:
                    raise StaleKnowledgeIndexError("indexed source hash does not match source bytes")
                try:
                    source_cache[key] = source_bytes.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise StaleKnowledgeIndexError("indexed source is no longer valid UTF-8") from exc

            source_text = source_cache[key]
            start = int(row["start_char"])
            end = int(row["end_char"])
            indexed_text = row["body"]
            if not (0 <= start < end <= len(source_text)):
                raise StaleKnowledgeIndexError("indexed source span is invalid")
            if source_text[start:end] != indexed_text:
                raise StaleKnowledgeIndexError("indexed chunk does not match source span")
            if _sha256_bytes(indexed_text.encode("utf-8")) != row["chunk_sha256"]:
                raise StaleKnowledgeIndexError("indexed chunk hash does not match chunk text")

    def retrieve(self, query: str, top_k: int = 6) -> EvidencePacket:
        """Retrieve a stable, hash-bound evidence packet from the active corpus."""
        if not 1 <= top_k <= 12:
            raise ValueError("top_k must be between 1 and 12")
        canonical_query = canonicalize_query(query)
        if not canonical_query:
            raise ValueError("query must not be empty")

        conn = self._connect(create=False)
        if conn is None:
            corpus_revision = self._corpus_revision(None)
            return self._build_packet(canonical_query, corpus_revision, [])

        terms = _query_terms(canonical_query)
        if not terms:
            corpus_revision = self._corpus_revision(conn)
            conn.close()
            return self._build_packet(canonical_query, corpus_revision, [])
        match_query = " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)

        try:
            conn.execute("BEGIN")
            corpus_revision = self._corpus_revision(conn)
            rows = conn.execute(
                """
                SELECT chunks.evidence_id, chunks.document_id, chunks.folder,
                       chunks.title, chunks.relative_path, chunks.source_sha256,
                       chunks.chunk_sha256, chunks.start_char, chunks.end_char,
                       chunks.start_line, chunks.end_line, chunks.body,
                       bm25(chunks, 0.0, 1.5, 1.5, 2.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0, 0.0, 0.0, 1.0) AS score
                FROM chunks
                JOIN documents ON documents.document_id = chunks.document_id
                    AND documents.folder = chunks.folder
                    AND documents.title = chunks.title
                    AND documents.source_sha256 = chunks.source_sha256
                    AND documents.relative_path = chunks.relative_path
                WHERE chunks MATCH ?
                ORDER BY score ASC, chunks.evidence_id ASC
                LIMIT ?
                """,
                (match_query, top_k),
            ).fetchall()
            evidence_ids = [row["evidence_id"] for row in rows]
            if len(evidence_ids) != len(set(evidence_ids)):
                raise StaleKnowledgeIndexError(
                    "retrieved packet contains duplicate evidence identifiers"
                )
            self._verify_sources(rows)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

        retrieved_items = [
            EvidenceItem(
                evidence_id=row["evidence_id"],
                rank=index,
                retrieval_score=round(float(row["score"]), 6),
                document_id=row["document_id"],
                folder=row["folder"],
                title=row["title"],
                relative_path=row["relative_path"],
                source_sha256=row["source_sha256"],
                chunk_sha256=row["chunk_sha256"],
                start_char=int(row["start_char"]),
                end_char=int(row["end_char"]),
                start_line=int(row["start_line"]),
                end_line=int(row["end_line"]),
                text=row["body"],
            )
            for index, row in enumerate(rows, start=1)
        ]
        bounded_items: list[EvidenceItem] = []
        truncated = False
        for item in retrieved_items:
            candidate = self._build_packet(
                canonical_query,
                corpus_revision,
                [*bounded_items, item],
            )
            if len(_canonical_json(candidate.prompt_dict())) > MAX_PACKET_BYTES:
                truncated = True
                break
            bounded_items.append(item)
        return self._build_packet(
            canonical_query,
            corpus_revision,
            bounded_items,
            truncated=truncated,
        )

    @staticmethod
    def _build_packet(
        canonical_query: str,
        corpus_revision: str,
        items: list[EvidenceItem],
        *,
        truncated: bool = False,
    ) -> EvidencePacket:
        packet_body = {
            "schema_version": SCHEMA_VERSION,
            "corpus_revision": corpus_revision,
            "retrieval_version": RETRIEVER_VERSION,
            "canonical_query": canonical_query,
            "truncated": truncated,
            "items": [item.prompt_dict() for item in items],
        }
        packet_id = _sha256_bytes(_canonical_json(packet_body))
        return EvidencePacket(
            packet_id=packet_id,
            corpus_revision=corpus_revision,
            retrieval_version=RETRIEVER_VERSION,
            canonical_query=canonical_query,
            truncated=truncated,
            items=tuple(items),
        )
