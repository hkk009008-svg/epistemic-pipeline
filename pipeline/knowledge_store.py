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

SCHEMA_VERSION = 2
DATABASE_SCHEMA_VERSION = 3
RETRIEVER_VERSION = "sqlite-fts5-v2"
CHUNKER_VERSION = "words-180-overlap-30-chars-4000-v2"
DOCUMENT_REVISION_VERSION = "document-revision-v1"
RUN_RECEIPT_VERSION = "grounded-run-receipt-v1"
MAX_DOCUMENT_CHARS = 2_000_000
MAX_FOLDER_DEPTH = 8
MAX_CHUNK_CHARS = 4_000
CHUNK_CHAR_OVERLAP = 256
MAX_PACKET_BYTES = 48_000
MAX_CANONICAL_QUERY_JSON_BYTES = 40_000
MAX_RECEIPT_BYTES = 32_000

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_WORD_RE = re.compile(r"\S+")
_QUERY_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_STOP_WORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "can",
    "did", "do", "does", "for", "from", "had", "has", "have", "how", "i",
    "in", "into", "is", "it", "its", "may", "of", "on", "or", "our", "s",
    "should", "that", "the", "their", "them", "they", "this", "to", "was",
    "were", "what", "when", "where", "which", "who", "why", "will", "with",
    "would", "you", "your",
})
_TABLE_CELL_DELIMITER_RE = re.compile(r"^\s*:?-{1,}:?\s*$")
_SCHEMA_LOCK = threading.Lock()
_SCHEMA_INIT_RETRIES = 20
_UPDATE_REVISION_REASONS = frozenset({
    "content_update",
    "metadata_update",
    "correction",
    "restore",
})
_FORBIDDEN_RECEIPT_KEYS = frozenset({
    "api_key",
    "answer",
    "base_url",
    "content",
    "evidence_packet",
    "items",
    "prompt",
    "question",
    "quote",
    "rationale",
    "text",
})
_RUN_RECEIPT_KEYS = frozenset({
    "schema_version",
    "contract_version",
    "run_id",
    "created_at",
    "latency_ms",
    "corpus_revision",
    "packet_id",
    "packet_schema_version",
    "retrieval_version",
    "chunker_version",
    "coverage_limited",
    "coverage_reasons",
    "prompt_versions",
    "stage_fingerprints",
    "stages_completed",
    "draft_hash",
    "verification_hash",
    "selected_claim_ids",
    "citation_evidence_ids",
    "status",
    "reason_code",
    "draft_claim_count",
    "supported_claim_count",
    "contradicted_claim_count",
    "conflict_claim_count",
    "insufficient_claim_count",
})


class KnowledgeStoreError(RuntimeError):
    """Base error for local knowledge storage failures."""


class StaleKnowledgeIndexError(KnowledgeStoreError):
    """Raised when the derived index no longer matches immutable source bytes."""


class DocumentVersionConflictError(KnowledgeStoreError):
    """Raised when a document mutation does not target the active revision."""


@dataclass(frozen=True)
class DocumentRecord:
    document_id: str
    revision_id: str
    supersedes_revision_id: str | None
    revision_reason: str
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
    document_revision_id: str
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
    coverage_limited: bool
    coverage_reasons: tuple[str, ...]
    items: tuple[EvidenceItem, ...]

    def prompt_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "packet_id": self.packet_id,
            "corpus_revision": self.corpus_revision,
            "retrieval_version": self.retrieval_version,
            "canonical_query": self.canonical_query,
            "truncated": self.truncated,
            "coverage_limited": self.coverage_limited,
            "coverage_reasons": list(self.coverage_reasons),
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


def _document_revision_id(
    *,
    document_id: str,
    source_sha256: str,
    folder: str,
    title: str,
    relative_path: str,
    chunk_count: int,
    supersedes_revision_id: str | None,
    revision_reason: str,
) -> str:
    """Derive a stable ID for one immutable logical document revision."""
    return _sha256_bytes(_canonical_json({
        "schema_version": DOCUMENT_REVISION_VERSION,
        "document_id": document_id,
        "source_sha256": source_sha256,
        "folder": folder,
        "title": title,
        "relative_path": relative_path,
        "chunk_count": chunk_count,
        "supersedes_revision_id": supersedes_revision_id,
        "revision_reason": revision_reason,
    }))


def _assert_receipt_is_metadata_only(value: object) -> None:
    """Reject fields that could persist private corpus or model prose."""
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).casefold() in _FORBIDDEN_RECEIPT_KEYS:
                raise KnowledgeStoreError(
                    f"run receipt field {key!r} may contain private content"
                )
            _assert_receipt_is_metadata_only(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_receipt_is_metadata_only(child)


def _assert_receipt_has_closed_shape(receipt: dict) -> None:
    if set(receipt) != _RUN_RECEIPT_KEYS:
        raise KnowledgeStoreError("grounded run receipt fields do not match its version")
    nested_shapes = {
        "prompt_versions": {"answerer", "verifier", "finalizer"},
        "stage_fingerprints": {"gpt1", "gpt2", "gpt3"},
    }
    for field, expected_keys in nested_shapes.items():
        value = receipt.get(field)
        if not isinstance(value, dict) or set(value) != expected_keys:
            raise KnowledgeStoreError(
                f"grounded run receipt {field} fields do not match its version"
            )


def canonicalize_query(query: str) -> str:
    """Normalize a query without allowing an LLM to rewrite its meaning."""
    canonical_query = " ".join(unicodedata.normalize("NFKC", query).split())
    if len(_canonical_json(canonical_query)) > MAX_CANONICAL_QUERY_JSON_BYTES:
        raise ValueError(
            "canonical query exceeds the fixed evidence-packet query budget"
        )
    return canonical_query


def _query_terms(query: str) -> tuple[list[str], bool]:
    raw_tokens = _QUERY_TOKEN_RE.findall(query.casefold())

    def capped_unique(tokens: Iterable[str]) -> tuple[list[str], bool]:
        seen: set[str] = set()
        terms: list[str] = []
        for token in tokens:
            if token in seen:
                continue
            if len(terms) == 24:
                return terms, True
            seen.add(token)
            terms.append(token)
        return terms, False

    preferred, limited = capped_unique(
        token for token in raw_tokens if token not in _STOP_WORDS
    )
    if preferred:
        return preferred, limited

    # FTS5 indexes stopwords too. A stopword-only query must not silently
    # become an empty query, and one-character identifiers/numerals are kept
    # by the preferred path above.
    return capped_unique(raw_tokens)

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


def _is_table_delimiter_row(line: str) -> bool:
    """Return True if line is a valid Markdown table delimiter row."""
    stripped = line.strip()
    if not stripped or "|" not in stripped:
        return False
    content = stripped.removeprefix("|").removesuffix("|")
    cells = content.split("|")
    if not cells:
        return False
    return all(_TABLE_CELL_DELIMITER_RE.match(cell) is not None for cell in cells)


@dataclass(frozen=True)
class _MarkdownTable:
    start_char: int
    end_char: int
    header_start_char: int
    header_end_char: int
    header_text: str
    first_data_row_start: int


def _find_markdown_tables(content: str) -> list[_MarkdownTable]:
    """Identify valid multi-row Markdown tables and their header blocks."""
    if "|" not in content:
        return []
    lines = content.splitlines(keepends=True)
    if len(lines) < 3:
        return []

    line_starts: list[int] = []
    current_pos = 0
    for line in lines:
        line_starts.append(current_pos)
        current_pos += len(line)

    tables: list[_MarkdownTable] = []
    i = 0
    while i < len(lines) - 1:
        line = lines[i]
        next_line = lines[i + 1]
        if (
            "|" in line
            and line.strip()
            and not _is_table_delimiter_row(line)
            and _is_table_delimiter_row(next_line)
        ):
            header_start = line_starts[i]
            header_end = line_starts[i + 1] + len(next_line)
            header_text = line.rstrip("\r\n") + "\n" + next_line.rstrip("\r\n")
            first_data_start = header_end

            j = i + 2
            while j < len(lines) and "|" in lines[j] and lines[j].strip():
                j += 1

            if j > i + 2:
                table_end = line_starts[j - 1] + len(lines[j - 1])
                tables.append(
                    _MarkdownTable(
                        start_char=header_start,
                        end_char=table_end,
                        header_start_char=header_start,
                        header_end_char=header_end,
                        header_text=header_text,
                        first_data_row_start=first_data_start,
                    )
                )
                i = j
                continue
        i += 1

    return tables


def chunk_text(content: str, max_words: int = 180, overlap_words: int = 30) -> list[dict]:
    """Split text into deterministic, overlapping exact-source spans."""
    if max_words < 1 or overlap_words < 0 or overlap_words >= max_words:
        raise ValueError("invalid chunk sizing")
    words = list(_WORD_RE.finditer(content))
    if not words:
        return []
    newline_offsets = [index for index, character in enumerate(content) if character == "\n"]
    tables = _find_markdown_tables(content)

    chunks: list[dict] = []

    def append_span(span_start: int, span_end: int) -> None:
        while span_start < span_end:
            bounded_end = min(span_start + MAX_CHUNK_CHARS, span_end)
            raw_text = content[span_start:bounded_end]
            chunk_text_value = raw_text
            for table in tables:
                if table.start_char < span_start < table.end_char:
                    if not raw_text.startswith(table.header_text):
                        if span_start < table.first_data_row_start:
                            data_part = content[table.first_data_row_start:bounded_end]
                            if data_part:
                                chunk_text_value = f"{table.header_text}\n{data_part}"
                        else:
                            chunk_text_value = f"{table.header_text}\n{raw_text}"
                    break
            chunks.append({
                "ordinal": len(chunks),
                "start_char": span_start,
                "end_char": bounded_end,
                "start_line": bisect_right(newline_offsets, span_start - 1) + 1,
                "end_line": bisect_right(newline_offsets, bounded_end - 1) + 1,
                "text": chunk_text_value,
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
            conn.execute("PRAGMA busy_timeout=10000")
            conn.execute("PRAGMA synchronous=FULL")
            current_schema = int(conn.execute("PRAGMA user_version").fetchone()[0])
            if current_schema > DATABASE_SCHEMA_VERSION:
                raise KnowledgeStoreError(
                    "knowledge index schema is newer than this service version"
                )
            if current_schema < DATABASE_SCHEMA_VERSION:
                # Serialize same-process migration and retry for concurrent
                # service processes contending on the WAL/schema transition.
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
                active_revision_id TEXT,
                folder TEXT NOT NULL,
                title TEXT NOT NULL,
                source_sha256 TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                chunk_count INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        document_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(documents)").fetchall()
        }
        if "active_revision_id" not in document_columns:
            conn.execute("ALTER TABLE documents ADD COLUMN active_revision_id TEXT")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS document_versions (
                revision_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                source_sha256 TEXT NOT NULL,
                supersedes_revision_id TEXT,
                folder TEXT NOT NULL,
                title TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                chunk_count INTEGER NOT NULL,
                revision_reason TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS document_versions_document_id
            ON document_versions(document_id, created_at)
            """
        )

        # Backfill the active row of an index created by schema v1. The legacy
        # active state becomes an immutable root revision; source bytes and
        # chunks are not rewritten.
        legacy_rows = conn.execute(
            """
            SELECT document_id, folder, title, source_sha256, relative_path,
                   chunk_count, updated_at
            FROM documents
            WHERE active_revision_id IS NULL
            ORDER BY document_id
            """
        ).fetchall()
        for row in legacy_rows:
            revision_id = _document_revision_id(
                document_id=row["document_id"],
                source_sha256=row["source_sha256"],
                folder=row["folder"],
                title=row["title"],
                relative_path=row["relative_path"],
                chunk_count=int(row["chunk_count"]),
                supersedes_revision_id=None,
                revision_reason="legacy_import",
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO document_versions (
                    revision_id, document_id, source_sha256,
                    supersedes_revision_id, folder, title, relative_path,
                    chunk_count, revision_reason, created_at
                ) VALUES (?, ?, ?, NULL, ?, ?, ?, ?, ?, ?)
                """,
                (
                    revision_id,
                    row["document_id"],
                    row["source_sha256"],
                    row["folder"],
                    row["title"],
                    row["relative_path"],
                    int(row["chunk_count"]),
                    "legacy_import",
                    row["updated_at"],
                ),
            )
            conn.execute(
                "UPDATE documents SET active_revision_id = ? WHERE document_id = ?",
                (revision_id, row["document_id"]),
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
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS grounded_run_receipts (
                run_id TEXT PRIMARY KEY,
                receipt_json TEXT NOT NULL,
                receipt_sha256 TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS document_versions_no_reinsert
            BEFORE INSERT ON document_versions
            WHEN EXISTS (
                SELECT 1 FROM document_versions
                WHERE revision_id = NEW.revision_id
            )
            BEGIN
                SELECT RAISE(ABORT, 'document versions are append-only');
            END
            """
        )
        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS document_versions_no_update
            BEFORE UPDATE ON document_versions
            BEGIN
                SELECT RAISE(ABORT, 'document versions are append-only');
            END
            """
        )
        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS document_versions_no_delete
            BEFORE DELETE ON document_versions
            BEGIN
                SELECT RAISE(ABORT, 'document versions are append-only');
            END
            """
        )
        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS grounded_run_receipts_no_reinsert
            BEFORE INSERT ON grounded_run_receipts
            WHEN EXISTS (
                SELECT 1 FROM grounded_run_receipts
                WHERE run_id = NEW.run_id
            )
            BEGIN
                SELECT RAISE(ABORT, 'grounded run receipts are append-only');
            END
            """
        )
        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS grounded_run_receipts_no_update
            BEFORE UPDATE ON grounded_run_receipts
            BEGIN
                SELECT RAISE(ABORT, 'grounded run receipts are append-only');
            END
            """
        )
        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS grounded_run_receipts_no_delete
            BEFORE DELETE ON grounded_run_receipts
            BEGIN
                SELECT RAISE(ABORT, 'grounded run receipts are append-only');
            END
            """
        )
        conn.execute(f"PRAGMA user_version={DATABASE_SCHEMA_VERSION}")
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
            SELECT document_id, active_revision_id, source_sha256, title, folder,
                   relative_path, chunk_count
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
        *,
        expected_revision_id: str | None = None,
        revision_reason: str | None = None,
    ) -> DocumentRecord:
        """Create an immutable revision and atomically advance its active head.

        The first insert has no expected revision. A non-idempotent update must
        name the exact active revision it supersedes, preventing concurrent
        content *and* metadata updates from silently overwriting each other.
        """
        document_id = _validate_identifier(document_id, "document_id")
        if expected_revision_id is not None and not re.fullmatch(
            r"[0-9a-f]{64}", expected_revision_id
        ):
            raise ValueError("expected_revision_id must be a lowercase SHA-256 value")
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

        chunks = chunk_text(content)
        if not chunks:
            raise ValueError("content did not produce any searchable chunks")

        # Publish immutable bytes before taking the SQLite head lock. A stale
        # writer may leave an unreferenced content-addressed blob, but it cannot
        # change the active corpus; avoiding file I/O under BEGIN IMMEDIATE also
        # keeps concurrent, identical publications from blocking each other.
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
                    # Atomic no-clobber publication: another process can
                    # observe only the complete content-addressed blob.
                    os.link(temporary_path, version_path)
                except FileExistsError:
                    pass
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
        if _sha256_bytes(version_path.read_bytes()) != source_sha:
            raise KnowledgeStoreError("content-addressed source file has been modified")

        now = datetime.now(timezone.utc).isoformat()
        conn = self._connect(create=True)
        assert conn is not None
        try:
            conn.execute("BEGIN IMMEDIATE")
            current = conn.execute(
                """
                SELECT document_id, active_revision_id, folder, title,
                       source_sha256, relative_path, chunk_count, updated_at
                FROM documents
                WHERE document_id = ?
                """,
                (document_id,),
            ).fetchone()

            if current is None:
                if expected_revision_id is not None:
                    raise DocumentVersionConflictError(
                        "document does not exist at the expected revision"
                    )
                if revision_reason not in (None, "initial_create"):
                    raise ValueError(
                        "revision_reason must be omitted when creating a document"
                    )
                normalized_reason = "initial_create"
                supersedes_revision_id = None
            else:
                current_revision_id = current["active_revision_id"]
                if not current_revision_id:
                    raise StaleKnowledgeIndexError(
                        "active document is missing immutable revision lineage"
                    )
                unchanged = (
                    current["folder"] == folder
                    and current["title"] == title
                    and current["source_sha256"] == source_sha
                    and current["relative_path"] == relative_path
                    and int(current["chunk_count"]) == len(chunks)
                )
                if unchanged:
                    version = conn.execute(
                        """
                        SELECT supersedes_revision_id, revision_reason
                        FROM document_versions
                        WHERE revision_id = ?
                        """,
                        (current_revision_id,),
                    ).fetchone()
                    if version is None:
                        raise StaleKnowledgeIndexError(
                            "active document revision is missing from history"
                        )
                    corpus_revision = self._corpus_revision(conn)
                    conn.commit()
                    return DocumentRecord(
                        document_id=document_id,
                        revision_id=current_revision_id,
                        supersedes_revision_id=version["supersedes_revision_id"],
                        revision_reason=version["revision_reason"],
                        folder=current["folder"],
                        title=current["title"],
                        source_sha256=current["source_sha256"],
                        relative_path=current["relative_path"],
                        chunk_count=int(current["chunk_count"]),
                        corpus_revision=corpus_revision,
                    )
                if expected_revision_id is None:
                    raise DocumentVersionConflictError(
                        "expected_revision_id is required when updating a document"
                    )
                if expected_revision_id != current_revision_id:
                    raise DocumentVersionConflictError(
                        "document changed since the supplied expected_revision_id"
                    )
                if revision_reason not in _UPDATE_REVISION_REASONS:
                    raise ValueError(
                        "revision_reason is required for updates and must be one of: "
                        "content_update, metadata_update, correction, restore"
                    )
                if (
                    revision_reason == "metadata_update"
                    and current["source_sha256"] != source_sha
                ):
                    raise ValueError("metadata_update cannot change document content")
                if (
                    revision_reason == "content_update"
                    and current["source_sha256"] == source_sha
                ):
                    raise ValueError("content_update must change document content")
                if revision_reason == "restore":
                    prior_representation = conn.execute(
                        """
                        SELECT 1 FROM document_versions
                        WHERE document_id = ?
                          AND source_sha256 = ?
                          AND folder = ?
                          AND title = ?
                          AND relative_path = ?
                          AND chunk_count = ?
                        LIMIT 1
                        """,
                        (
                            document_id,
                            source_sha,
                            folder,
                            title,
                            relative_path,
                            len(chunks),
                        ),
                    ).fetchone()
                    if prior_representation is None:
                        raise ValueError(
                            "restore must target a prior representation of this document"
                        )
                normalized_reason = revision_reason
                supersedes_revision_id = current_revision_id

            revision_id = _document_revision_id(
                document_id=document_id,
                source_sha256=source_sha,
                folder=folder,
                title=title,
                relative_path=relative_path,
                chunk_count=len(chunks),
                supersedes_revision_id=supersedes_revision_id,
                revision_reason=normalized_reason,
            )

            try:
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
                        document_id, active_revision_id, folder, title,
                        source_sha256, relative_path, chunk_count, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(document_id) DO UPDATE SET
                        active_revision_id=excluded.active_revision_id,
                        folder=excluded.folder,
                        title=excluded.title,
                        source_sha256=excluded.source_sha256,
                        relative_path=excluded.relative_path,
                        chunk_count=excluded.chunk_count,
                        updated_at=excluded.updated_at
                    """,
                    (
                        document_id,
                        revision_id,
                        folder,
                        title,
                        source_sha,
                        relative_path,
                        len(chunks),
                        now,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO document_versions (
                        revision_id, document_id, source_sha256,
                        supersedes_revision_id, folder, title, relative_path,
                        chunk_count, revision_reason, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        revision_id,
                        document_id,
                        source_sha,
                        supersedes_revision_id,
                        folder,
                        title,
                        relative_path,
                        len(chunks),
                        normalized_reason,
                        now,
                    ),
                )
                corpus_revision = self._corpus_revision(conn)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return DocumentRecord(
            document_id=document_id,
            revision_id=revision_id,
            supersedes_revision_id=supersedes_revision_id,
            revision_reason=normalized_reason,
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
            expected_revision_id = _document_revision_id(
                document_id=row["revision_document_id"],
                source_sha256=row["revision_source_sha256"],
                folder=row["revision_folder"],
                title=row["revision_title"],
                relative_path=row["revision_relative_path"],
                chunk_count=row["revision_chunk_count"],
                supersedes_revision_id=row["revision_parent_id"],
                revision_reason=row["revision_reason"],
            )
            active_revision_matches = (
                row["document_revision_id"]
                and row["revision_record_id"] == row["document_revision_id"]
                and row["revision_record_id"] == expected_revision_id
                and row["revision_document_id"] == row["document_id"]
                and row["revision_source_sha256"] == row["source_sha256"]
                and row["revision_folder"] == row["folder"]
                and row["revision_title"] == row["title"]
                and row["revision_relative_path"] == row["relative_path"]
                and row["revision_chunk_count"] == row["document_chunk_count"]
            )
            if not active_revision_matches:
                raise StaleKnowledgeIndexError(
                    "active document does not match its immutable revision"
                )
            key = (row["relative_path"], row["source_sha256"])
            if key not in source_cache:
                source_path = self._safe_child(self.root, *Path(row["relative_path"]).parts)
                if source_path.is_symlink() or not source_path.is_file():
                    raise StaleKnowledgeIndexError("indexed source is missing or unsafe")
                source_bytes = source_path.read_bytes()
                if _sha256_bytes(source_bytes) != row["source_sha256"]:
                    raise StaleKnowledgeIndexError(
                        "indexed source hash does not match source bytes"
                    )
                try:
                    source_cache[key] = source_bytes.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise StaleKnowledgeIndexError(
                        "indexed source is no longer valid UTF-8"
                    ) from exc

            source_text = source_cache[key]
            start = int(row["start_char"])
            end = int(row["end_char"])
            indexed_text = row["body"]
            if not (0 <= start < end <= len(source_text)):
                raise StaleKnowledgeIndexError("indexed source span is invalid")
            start_line = row["start_line"]
            end_line = row["end_line"]
            if type(start_line) is not int or type(end_line) is not int:
                raise StaleKnowledgeIndexError("indexed source lines are invalid")
            expected_start_line = source_text.count("\n", 0, start) + 1
            expected_end_line = source_text.count("\n", 0, end) + 1
            if start_line != expected_start_line or end_line != expected_end_line:
                raise StaleKnowledgeIndexError("indexed source lines do not match source span")
            if source_text[start:end] != indexed_text:
                tables = _find_markdown_tables(source_text)
                expected_with_header = None
                for table in tables:
                    if table.start_char < start < table.end_char:
                        if start < table.first_data_row_start:
                            data_part = source_text[table.first_data_row_start:end]
                            if data_part:
                                expected_with_header = f"{table.header_text}\n{data_part}"
                        else:
                            expected_with_header = f"{table.header_text}\n{source_text[start:end]}"
                        break
                if expected_with_header != indexed_text:
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
        terms, query_term_limited = _query_terms(canonical_query)
        initial_coverage = ("query_term_limit",) if query_term_limited else ()

        conn = self._connect(create=False)
        if conn is None:
            corpus_revision = self._corpus_revision(None)
            return self._build_packet(
                canonical_query,
                corpus_revision,
                [],
                coverage_reasons=initial_coverage,
            )

        if not terms:
            corpus_revision = self._corpus_revision(conn)
            conn.close()
            return self._build_packet(
                canonical_query,
                corpus_revision,
                [],
                coverage_reasons=initial_coverage,
            )
        match_query = " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)

        try:
            conn.execute("BEGIN")
            corpus_revision = self._corpus_revision(conn)
            candidate_rows = conn.execute(
                """
                SELECT chunks.evidence_id, chunks.document_id,
                       documents.active_revision_id AS document_revision_id,
                       documents.chunk_count AS document_chunk_count,
                       chunks.folder,
                       chunks.title, chunks.relative_path, chunks.source_sha256,
                       chunks.chunk_sha256, chunks.start_char, chunks.end_char,
                       chunks.start_line, chunks.end_line, chunks.body,
                       active_version.revision_id AS revision_record_id,
                       active_version.document_id AS revision_document_id,
                       active_version.source_sha256 AS revision_source_sha256,
                       active_version.folder AS revision_folder,
                       active_version.title AS revision_title,
                       active_version.relative_path AS revision_relative_path,
                       active_version.chunk_count AS revision_chunk_count,
                       active_version.supersedes_revision_id AS revision_parent_id,
                       active_version.revision_reason AS revision_reason,
                       bm25(chunks, 0.0, 1.5, 1.5, 2.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0, 0.0, 0.0, 1.0) AS score
                FROM chunks
                JOIN documents ON documents.document_id = chunks.document_id
                    AND documents.folder = chunks.folder
                    AND documents.title = chunks.title
                    AND documents.source_sha256 = chunks.source_sha256
                    AND documents.relative_path = chunks.relative_path
                LEFT JOIN document_versions AS active_version
                    ON active_version.revision_id = documents.active_revision_id
                WHERE chunks MATCH ?
                ORDER BY score ASC, chunks.evidence_id ASC
                LIMIT ?
                """,
                (match_query, max((top_k + 1) * 4, 64)),
            ).fetchall()
            min_matched_terms = min(2, len(terms))
            meaningful_rows = []
            for row in candidate_rows:
                searchable = (
                    f"{row['folder']} {row['title']} {row['document_id']} {row['body']}"
                ).casefold()
                chunk_tokens = set(_QUERY_TOKEN_RE.findall(searchable))
                matched_count = sum(1 for term in terms if term in chunk_tokens)
                if matched_count >= min_matched_terms:
                    meaningful_rows.append(row)

            top_k_limited = len(meaningful_rows) > top_k
            rows = meaningful_rows[:top_k]
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
                document_revision_id=row["document_revision_id"],
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
        coverage_reasons: list[str] = list(initial_coverage)
        if top_k_limited:
            coverage_reasons.append("top_k")
        bounded_items: list[EvidenceItem] = []
        truncated = False
        for item in retrieved_items:
            candidate = self._build_packet(
                canonical_query,
                corpus_revision,
                [*bounded_items, item],
                coverage_reasons=tuple(coverage_reasons),
            )
            if len(_canonical_json(candidate.prompt_dict())) > MAX_PACKET_BYTES:
                truncated = True
                break
            bounded_items.append(item)
        if truncated:
            coverage_reasons.append("byte_budget")

        # Adding the explicit byte-budget reason itself consumes a few bytes.
        # If a packet sat exactly on the boundary, remove the lowest-ranked
        # item until the final canonical packet, including its reason, fits.
        while True:
            packet = self._build_packet(
                canonical_query,
                corpus_revision,
                bounded_items,
                truncated=truncated,
                coverage_reasons=tuple(coverage_reasons),
            )
            if len(_canonical_json(packet.prompt_dict())) <= MAX_PACKET_BYTES:
                return packet
            if not bounded_items:
                raise KnowledgeStoreError("evidence packet metadata exceeds byte budget")
            bounded_items.pop()
            if not truncated:
                truncated = True
                coverage_reasons.append("byte_budget")

    @staticmethod
    def _build_packet(
        canonical_query: str,
        corpus_revision: str,
        items: list[EvidenceItem],
        *,
        truncated: bool = False,
        coverage_reasons: tuple[str, ...] = (),
    ) -> EvidencePacket:
        if any(
            reason not in {"query_term_limit", "top_k", "byte_budget"}
            for reason in coverage_reasons
        ):
            raise ValueError("unknown evidence coverage reason")
        ordered_reasons = tuple(
            reason
            for reason in ("query_term_limit", "top_k", "byte_budget")
            if reason in coverage_reasons
        )
        if truncated != ("byte_budget" in ordered_reasons):
            raise ValueError("truncated must exactly reflect byte_budget coverage")
        packet_body = {
            "schema_version": SCHEMA_VERSION,
            "corpus_revision": corpus_revision,
            "retrieval_version": RETRIEVER_VERSION,
            "canonical_query": canonical_query,
            "truncated": truncated,
            "coverage_limited": bool(ordered_reasons),
            "coverage_reasons": list(ordered_reasons),
            "items": [item.prompt_dict() for item in items],
        }
        packet_id = _sha256_bytes(_canonical_json(packet_body))
        return EvidencePacket(
            packet_id=packet_id,
            corpus_revision=corpus_revision,
            retrieval_version=RETRIEVER_VERSION,
            canonical_query=canonical_query,
            truncated=truncated,
            coverage_limited=bool(ordered_reasons),
            coverage_reasons=ordered_reasons,
            items=tuple(items),
        )

    def append_run_receipt(self, receipt: dict) -> str:
        """Persist one append-only, metadata-only grounded run receipt."""
        if receipt.get("schema_version") != RUN_RECEIPT_VERSION:
            raise KnowledgeStoreError("unsupported grounded run receipt version")
        run_id = _validate_identifier(str(receipt.get("run_id", "")), "run_id")
        created_at = receipt.get("created_at")
        if not isinstance(created_at, str) or not created_at:
            raise KnowledgeStoreError("grounded run receipt is missing created_at")
        _assert_receipt_has_closed_shape(receipt)
        _assert_receipt_is_metadata_only(receipt)
        payload = _canonical_json(receipt)
        if len(payload) > MAX_RECEIPT_BYTES:
            raise KnowledgeStoreError("grounded run receipt exceeds metadata byte budget")
        receipt_sha256 = _sha256_bytes(payload)

        conn = self._connect(create=True)
        assert conn is not None
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO grounded_run_receipts (
                        run_id, receipt_json, receipt_sha256, created_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        payload.decode("utf-8"),
                        receipt_sha256,
                        created_at,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise KnowledgeStoreError("grounded run receipt ID already exists") from exc
        finally:
            conn.close()
        return receipt_sha256

    def load_run_receipt(self, run_id: str) -> dict:
        """Load and hash-check a receipt for local diagnostics and tests."""
        run_id = _validate_identifier(run_id, "run_id")
        conn = self._connect(create=False)
        if conn is None:
            raise KnowledgeStoreError("grounded run receipt does not exist")
        try:
            row = conn.execute(
                """
                SELECT receipt_json, receipt_sha256
                FROM grounded_run_receipts
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            raise KnowledgeStoreError("grounded run receipt does not exist")
        payload = row["receipt_json"].encode("utf-8")
        if _sha256_bytes(payload) != row["receipt_sha256"]:
            raise KnowledgeStoreError("grounded run receipt hash mismatch")
        parsed = json.loads(payload)
        if not isinstance(parsed, dict):
            raise KnowledgeStoreError("grounded run receipt is not a JSON object")
        _assert_receipt_has_closed_shape(parsed)
        _assert_receipt_is_metadata_only(parsed)
        return parsed

    def list_documents(self, folder: str | None = None) -> list[DocumentRecord]:
        """List active documents in the knowledge store, optionally filtered by folder."""
        conn = self._connect(create=False)
        if conn is None:
            return []
        try:
            corpus_rev = self._corpus_revision(conn)
            query = """
                SELECT d.document_id, d.active_revision_id, d.source_sha256,
                       d.folder, d.title, d.relative_path, d.chunk_count,
                       v.supersedes_revision_id, v.revision_reason
                FROM documents d
                LEFT JOIN document_versions v ON d.active_revision_id = v.revision_id
            """
            params: list = []
            if folder:
                query += " WHERE d.folder = ?"
                params.append(folder)
            query += " ORDER BY d.folder, d.title, d.document_id"
            rows = conn.execute(query, params).fetchall()
            records = []
            for r in rows:
                records.append(DocumentRecord(
                    document_id=r["document_id"],
                    revision_id=r["active_revision_id"] or "",
                    supersedes_revision_id=r["supersedes_revision_id"],
                    revision_reason=r["revision_reason"] or "initial_create",
                    folder=r["folder"],
                    title=r["title"],
                    source_sha256=r["source_sha256"],
                    relative_path=r["relative_path"],
                    chunk_count=int(r["chunk_count"]),
                    corpus_revision=corpus_rev,
                ))
            return records
        finally:
            conn.close()

    def sync_folder(self, folder_path: str | Path, target_folder: str = "general") -> list[DocumentRecord]:
        """Ingest all readable text and markdown documents from a filesystem directory."""
        path = Path(folder_path).resolve()
        if not path.is_dir():
            raise KnowledgeStoreError(f"Directory not found: {folder_path}")

        supported_exts = {".txt", ".md", ".markdown", ".json", ".csv", ".py", ".html"}
        records = []
        for file in sorted(path.rglob("*")):
            if file.is_file() and file.suffix.lower() in supported_exts and not file.name.startswith("."):
                try:
                    content = file.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError):
                    continue
                if not content.strip():
                    continue
                rel = file.relative_to(path).as_posix()
                doc_id = re.sub(r"[^A-Za-z0-9_-]", "_", rel)
                title = file.stem.replace("_", " ").title()
                subfolder = target_folder
                if file.parent != path:
                    subfolder = f"{target_folder}/{file.parent.relative_to(path).as_posix()}"
                rec = self.upsert_document(
                    document_id=doc_id,
                    folder=subfolder,
                    title=title,
                    content=content,
                    revision_reason=None,
                )
                records.append(rec)
        return records
