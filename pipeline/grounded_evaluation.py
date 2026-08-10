"""Closed, private evaluation contracts for the grounded-RAG baseline.

This module deliberately keeps human gold data, recorded model prose, and
claim-level adjudication outside the production response and receipt schemas.
It never configures or calls a provider while loading or scoring data.
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import sqlite3
import stat
import tempfile
import unicodedata
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from pipeline.knowledge_store import (
    CHUNKER_VERSION,
    DOCUMENT_REVISION_VERSION,
    EvidencePacket,
    KnowledgeStore,
    MAX_DOCUMENT_CHARS,
    RETRIEVER_VERSION,
    SCHEMA_VERSION,
    canonicalize_query,
    chunk_text,
    normalize_folder,
)


BENCHMARK_SCHEMA_VERSION = "grounded-rag-benchmark-v1"
OBSERVATION_SCHEMA_VERSION = "grounded-rag-observations-v2"
ADJUDICATION_SCHEMA_VERSION = "grounded-rag-adjudication-v1"
SUMMARY_SCHEMA_VERSION = "grounded-rag-baseline-summary-v1"
EXPECTED_BENCHMARK_SHA256 = (
    "178e2398a526c3f5e37ecbb57c88ec173ffbdab58cf6ce7741855f9e7edbd2e6"
)
# Persist the published v1 reason literal unchanged. Observation v2 can carry
# per-invocation SDK token counts, but the v1 aggregate remains unavailable.
USAGE_UNAVAILABLE_REASON = "current_adapter_does_not_expose_provider_usage_or_cost"
MAX_EVALUATION_FILE_BYTES = 256 * 1024 * 1024

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_CASE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_CLAIM_ID_RE = re.compile(r"^C[1-9][0-9]{0,2}$")
_CITATION_ID_RE = re.compile(r"^R[1-9][0-9]{0,2}$")
_STAGES = ("retrieval", "answerer", "verifier", "finalizer")
_COVERAGE_REASONS = ("query_term_limit", "top_k", "byte_budget")
_STATUS_REASONS = {
    "ANSWER": frozenset({"answered"}),
    "PARTIAL": frozenset({"partial_with_conflict", "partial_evidence"}),
    "ABSTAIN": frozenset({
        "no_lexical_match",
        "retrieval_query_term_limit_exceeded",
        "evidence_packet_budget_exceeded",
        "model_input_budget_exceeded",
        "answerer_packet_mismatch",
        "answerer_abstained",
        "verifier_protocol_error",
        "conflict_in_evidence",
        "no_supported_claims",
        "finalizer_protocol_error",
        "finalizer_abstained",
    }),
}
_THRESHOLD_KEYS = (
    "retrieval_recall_at_k_min",
    "unsupported_released_claim_rate_max",
    "citation_validity_min",
    "citation_completeness_min",
    "correct_abstention_rate_min",
    "false_abstention_rate_max",
    "answer_coverage_min",
    "verifier_supported_precision_min",
    "verifier_supported_recall_min",
)
_IMPLEMENTATION_FILES = (
    "config.py",
    "pipeline/helpers.py",
    "pipeline/knowledge_store.py",
    "pipeline/grounded_rag.py",
    "pipeline/grounded_evaluation.py",
    "scripts/measure_grounded_rag.py",
    "requirements.txt",
)
_RUNTIME_PACKAGES = (
    "pydantic",
    "pydantic-core",
    "openai",
    "anthropic",
    "httpx",
)
_AGY_INVOCATION_POLICY = {
    "adapter_retries": 0,
    "api_retries": 0,
    "custom_tools": [],
    "enabled_builtin_tools": ["finish"],
    "hooks": [],
    "mcp_servers": [],
    "skills": [],
    "subagents": [],
    "triggers": [],
    "workspaces": [],
}
AGY_INVOCATION_POLICY_SHA256 = hashlib.sha256(
    json.dumps(
        _AGY_INVOCATION_POLICY,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


class EvaluationDataError(ValueError):
    """A closed evaluation artifact failed validation without echoing its data."""


class ClosedModel(BaseModel):
    """Strict, finite, versioned evaluation data with no undeclared fields."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


BenchmarkCategory = Literal[
    "answerable",
    "unanswerable",
    "conflict",
    "distractor",
    "prompt_injection",
    "altered_fact",
    "altered_date",
    "polarity_reversal",
    "stale",
    "compound_claim",
    "citation_failure",
]


class BenchmarkDocument(ClosedModel):
    document_id: str = Field(..., min_length=1, max_length=64)
    folder: str = Field(..., min_length=1, max_length=520)
    title: str = Field(..., min_length=1, max_length=300)
    content: str = Field(..., min_length=1, max_length=MAX_DOCUMENT_CHARS)

    @field_validator("document_id")
    @classmethod
    def identifier_is_safe(cls, value: str) -> str:
        if not _IDENTIFIER_RE.fullmatch(value):
            raise ValueError("document_id is not a safe identifier")
        return value

    @field_validator("folder")
    @classmethod
    def folder_is_canonical_and_safe(cls, value: str) -> str:
        try:
            normalized = normalize_folder(value)
        except ValueError as exc:
            raise ValueError("folder is path-unsafe") from exc
        if normalized != value:
            raise ValueError("folder must already be canonical")
        return value

    @field_validator("title")
    @classmethod
    def title_is_canonical(cls, value: str) -> str:
        if " ".join(value.split()) != value:
            raise ValueError("title must already be whitespace-normalized")
        return value

    @field_validator("content")
    @classmethod
    def content_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content must not be blank")
        return value


class BenchmarkRelevantSpan(ClosedModel):
    span_id: str = Field(..., pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    document_id: str = Field(..., min_length=1, max_length=64)
    quote: str = Field(..., min_length=1, max_length=1_200)


class BenchmarkCase(ClosedModel):
    case_id: str = Field(..., pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    category: BenchmarkCategory
    query: str = Field(..., min_length=1, max_length=10_000)
    must_abstain: bool
    relevant_spans: list[BenchmarkRelevantSpan] = Field(
        default_factory=list,
        max_length=24,
    )

    @model_validator(mode="after")
    def span_ids_are_unique(self) -> BenchmarkCase:
        span_ids = [span.span_id for span in self.relevant_spans]
        if len(span_ids) != len(set(span_ids)):
            raise ValueError("span_id values must be unique within a case")
        if not self.must_abstain and not self.relevant_spans:
            raise ValueError("an answerable case requires relevant spans")
        return self


class BenchmarkThresholds(ClosedModel):
    retrieval_recall_at_k_min: float = Field(..., ge=0, le=1)
    unsupported_released_claim_rate_max: float = Field(..., ge=0, le=1)
    citation_validity_min: float = Field(..., ge=0, le=1)
    citation_completeness_min: float = Field(..., ge=0, le=1)
    correct_abstention_rate_min: float = Field(..., ge=0, le=1)
    false_abstention_rate_max: float = Field(..., ge=0, le=1)
    answer_coverage_min: float = Field(..., ge=0, le=1)
    verifier_supported_precision_min: float = Field(..., ge=0, le=1)
    verifier_supported_recall_min: float = Field(..., ge=0, le=1)


class BenchmarkExecutionPolicy(ClosedModel):
    all_cases_required: Literal[True]
    provider_error_state: Literal["INCOMPLETE"]
    raw_provider_transcripts: Literal["FORBIDDEN"]
    prompts_and_reasoning: Literal["FORBIDDEN"]
    usage_unavailable_is_null: Literal[True]
    live_execution_requires_separate_authorization: Literal[True]
    maximum_wall_time_seconds: int = Field(..., ge=1, le=86_400)
    # This is an external authorization ceiling. Invocation receipts may retain
    # provider-reported token counts, but no adapter exposes trustworthy dollar
    # cost, so a caller must separately enforce and attest the provider cap.
    external_authorization_maximum_cost_usd: float = Field(
        ...,
        alias="maximum_cost_usd",
        serialization_alias="maximum_cost_usd",
        ge=0,
    )


class GroundedRAGBenchmark(ClosedModel):
    schema_version: Literal["grounded-rag-benchmark-v1"]
    benchmark_id: str = Field(..., pattern=r"^[a-z0-9][a-z0-9_-]{0,79}$")
    data_classification: Literal["synthetic_non_sensitive"]
    label_authority: Literal["human_owned_external"]
    top_k: int = Field(..., ge=1, le=12)
    documents: list[BenchmarkDocument] = Field(..., min_length=1, max_length=100)
    cases: list[BenchmarkCase] = Field(..., min_length=1, max_length=100)
    thresholds: BenchmarkThresholds
    execution_policy: BenchmarkExecutionPolicy

    @model_validator(mode="after")
    def top_level_ids_are_unique(self) -> GroundedRAGBenchmark:
        document_ids = [document.document_id for document in self.documents]
        case_ids = [case.case_id for case in self.cases]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("document_id values must be unique")
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("case_id values must be unique")
        return self


class ResolvedGoldSpan(ClosedModel):
    span_id: str
    document_id: str
    source_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    quote: str
    source_start_char: int = Field(..., ge=0)
    source_end_char: int = Field(..., gt=0)

    @model_validator(mode="after")
    def span_is_ordered(self) -> ResolvedGoldSpan:
        if self.source_end_char <= self.source_start_char:
            raise ValueError("resolved span must be non-empty")
        return self


@dataclass(frozen=True)
class ResolvedBenchmark:
    definition: GroundedRAGBenchmark
    raw_sha256: str
    document_source_sha256: Mapping[str, str]
    gold_spans_by_case: Mapping[str, tuple[ResolvedGoldSpan, ...]]
    expected_packets_by_case: Mapping[str, EvidencePacket]


@dataclass(frozen=True)
class _ExpectedChunk:
    evidence_id: str
    chunk_sha256: str
    start_char: int
    end_char: int
    start_line: int
    end_line: int
    text: str


@dataclass(frozen=True)
class _ExpectedDocument:
    definition: BenchmarkDocument
    source_sha256: str
    relative_path: str
    revision_id: str
    chunks_by_evidence_id: Mapping[str, _ExpectedChunk]


def _expected_corpus(
    benchmark: ResolvedBenchmark,
) -> tuple[Mapping[str, _ExpectedDocument], str]:
    documents: dict[str, _ExpectedDocument] = {}
    corpus_rows: list[list[Any]] = []
    for document in sorted(
        benchmark.definition.documents,
        key=lambda item: item.document_id,
    ):
        source_sha256 = benchmark.document_source_sha256[document.document_id]
        relative_path = (
            f"sources/{document.folder}/{document.document_id}/versions/"
            f"{source_sha256}.txt"
        )
        source_chunks = chunk_text(document.content)
        revision_id = hashlib.sha256(_canonical_json_bytes({
            "schema_version": DOCUMENT_REVISION_VERSION,
            "document_id": document.document_id,
            "source_sha256": source_sha256,
            "folder": document.folder,
            "title": document.title,
            "relative_path": relative_path,
            "chunk_count": len(source_chunks),
            "supersedes_revision_id": None,
            "revision_reason": "initial_create",
        })).hexdigest()
        chunks: dict[str, _ExpectedChunk] = {}
        for chunk in source_chunks:
            evidence_seed = (
                f"{document.document_id}:{source_sha256}:{chunk['ordinal']}:"
            ).encode()
            evidence_id = f"E-{hashlib.sha256(evidence_seed).hexdigest()[:32]}"
            chunks[evidence_id] = _ExpectedChunk(
                evidence_id=evidence_id,
                chunk_sha256=hashlib.sha256(
                    chunk["text"].encode("utf-8")
                ).hexdigest(),
                start_char=chunk["start_char"],
                end_char=chunk["end_char"],
                start_line=chunk["start_line"],
                end_line=chunk["end_line"],
                text=chunk["text"],
            )
        documents[document.document_id] = _ExpectedDocument(
            definition=document,
            source_sha256=source_sha256,
            relative_path=relative_path,
            revision_id=revision_id,
            chunks_by_evidence_id=chunks,
        )
        corpus_rows.append([
            document.document_id,
            revision_id,
            source_sha256,
            document.title,
            document.folder,
            relative_path,
            len(source_chunks),
        ])
    corpus_revision = hashlib.sha256(_canonical_json_bytes({
        "chunker_version": CHUNKER_VERSION,
        "retriever_version": RETRIEVER_VERSION,
        "documents": corpus_rows,
    })).hexdigest()
    return documents, corpus_revision


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_model_sha256(value: BaseModel) -> str:
    """Hash one closed model using the harness's canonical JSON contract."""
    return hashlib.sha256(canonical_artifact_bytes(value)).hexdigest()


def canonical_artifact_bytes(value: BaseModel) -> bytes:
    """Return the exact bytes used for private artifact publication."""
    return _canonical_json_bytes(value.model_dump(mode="json")) + b"\n"


def _read_regular_file(path: Path, *, label: str) -> bytes:
    """Open once without following the final symlink, then bounded-read that fd."""
    flags = os.O_RDONLY
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = -1
    try:
        fd = os.open(path, flags)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise EvaluationDataError(
                f"{label} must be a regular non-symlink file"
            )
        if before.st_size > MAX_EVALUATION_FILE_BYTES:
            raise EvaluationDataError(f"{label} exceeds the evaluation byte limit")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(fd, min(1024 * 1024, MAX_EVALUATION_FILE_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > MAX_EVALUATION_FILE_BYTES:
                raise EvaluationDataError(f"{label} exceeds the evaluation byte limit")
        after = os.fstat(fd)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise EvaluationDataError(f"{label} changed while it was being read")
        return b"".join(chunks)
    except EvaluationDataError:
        raise
    except OSError as exc:
        raise EvaluationDataError(
            f"{label} must be a readable regular non-symlink file"
        ) from exc
    finally:
        if fd >= 0:
            os.close(fd)


def load_benchmark(path: Path, expected_sha256: str) -> ResolvedBenchmark:
    """Hash and resolve the protected benchmark before any provider setup.

    The caller must provide the externally frozen raw-byte hash. Source hashes
    and exact character spans are then derived from the validated inline source
    bytes; no model-produced label participates in this process.
    """
    if not _SHA256_RE.fullmatch(expected_sha256):
        raise EvaluationDataError("expected benchmark SHA-256 is malformed")
    raw = _read_regular_file(path, label="benchmark")
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != expected_sha256:
        raise EvaluationDataError("benchmark raw-byte SHA-256 does not match")
    try:
        payload = json.loads(raw)
        definition = GroundedRAGBenchmark.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise EvaluationDataError("benchmark does not match its closed schema") from exc

    documents = {document.document_id: document for document in definition.documents}
    source_hashes = {
        document_id: hashlib.sha256(document.content.encode("utf-8")).hexdigest()
        for document_id, document in documents.items()
    }
    resolved: dict[str, tuple[ResolvedGoldSpan, ...]] = {}
    for case in definition.cases:
        try:
            canonicalize_query(case.query)
        except ValueError as exc:
            raise EvaluationDataError(
                "benchmark query is not valid for grounded retrieval"
            ) from exc
        case_spans: list[ResolvedGoldSpan] = []
        for span in case.relevant_spans:
            document = documents.get(span.document_id)
            if document is None:
                raise EvaluationDataError("benchmark span names an unknown document")
            if document.content.count(span.quote) != 1:
                raise EvaluationDataError(
                    "benchmark quote must resolve exactly once in its source"
                )
            start = document.content.find(span.quote)
            case_spans.append(ResolvedGoldSpan(
                span_id=span.span_id,
                document_id=span.document_id,
                source_sha256=source_hashes[span.document_id],
                quote=span.quote,
                source_start_char=start,
                source_end_char=start + len(span.quote),
            ))
        resolved[case.case_id] = tuple(case_spans)
    expected_packets: dict[str, EvidencePacket] = {}
    try:
        with tempfile.TemporaryDirectory(
            prefix="grounded-rag-benchmark-validation-v1-"
        ) as root:
            store = KnowledgeStore(Path(root) / "knowledge")
            for document in sorted(
                definition.documents,
                key=lambda item: item.document_id,
            ):
                record = store.upsert_document(
                    document.document_id,
                    document.folder,
                    document.title,
                    document.content,
                )
                if record.source_sha256 != source_hashes[document.document_id]:
                    raise EvaluationDataError(
                        "validated corpus source hash does not match benchmark"
                    )
            for case in definition.cases:
                expected_packets[case.case_id] = store.retrieve(
                    case.query,
                    definition.top_k,
                )
    except EvaluationDataError:
        raise
    except Exception as exc:
        raise EvaluationDataError(
            "benchmark could not materialize deterministic retrieval"
        ) from exc
    return ResolvedBenchmark(
        definition=definition,
        raw_sha256=actual_sha256,
        document_source_sha256=source_hashes,
        gold_spans_by_case=resolved,
        expected_packets_by_case=expected_packets,
    )


class UnavailableUsageCost(ClosedModel):
    status: Literal["UNAVAILABLE"] = "UNAVAILABLE"
    reason: Literal[
        "current_adapter_does_not_expose_provider_usage_or_cost"
    ] = USAGE_UNAVAILABLE_REASON
    input_tokens: None = None
    output_tokens: None = None
    total_tokens: None = None
    cost_usd: None = None


class ReportedProviderUsage(ClosedModel):
    """Token counts reported by a provider SDK; never estimated by the harness."""

    status: Literal["REPORTED"] = "REPORTED"
    input_tokens: int = Field(..., ge=0)
    output_tokens: int = Field(..., ge=0)
    total_tokens: int = Field(..., ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    service_tier: str | None = Field(default=None, min_length=1, max_length=80)

    @model_validator(mode="after")
    def reported_totals_are_possible(self) -> ReportedProviderUsage:
        if self.total_tokens < max(self.input_tokens, self.output_tokens):
            raise ValueError("reported total tokens are smaller than a component")
        if self.service_tier is not None and (
            self.service_tier != self.service_tier.strip()
            or any(ord(character) < 0x20 for character in self.service_tier)
        ):
            raise ValueError("reported service tier is malformed")
        return self


class ProviderInvocationObservation(ClosedModel):
    """Sanitized identity and usage for one evaluation-only structured call."""

    schema_version: Literal["grounded-provider-invocation-v1"] = (
        "grounded-provider-invocation-v1"
    )
    stage: Literal["gpt1", "gpt2", "gpt3"]
    provider: Literal["google-antigravity-sdk"]
    requested_model: str = Field(..., min_length=1, max_length=160)
    reported_model: str | None = Field(default=None, min_length=1, max_length=160)
    model_attestation: Literal["REQUESTED_ONLY", "PROVIDER_REPORTED"]
    sdk_distribution: Literal["google-antigravity"]
    sdk_version: Literal["0.1.10"]
    sdk_artifact_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    localharness_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    worker_protocol_version: Literal["agy-evaluation-worker-v1"]
    worker_source_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    invocation_policy_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    system_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    user_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    response_schema_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    response_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    duration_ms: float = Field(..., ge=0, allow_inf_nan=False)
    outcome: Literal["SUCCESS"] = "SUCCESS"
    logical_model_calls: Literal[1] = 1
    adapter_retries: Literal[0] = 0
    observed_tool_names: list[Literal["finish"]] = Field(
        default_factory=list,
        max_length=1,
    )
    usage: ReportedProviderUsage | None = None
    cost_usd: None = None

    @model_validator(mode="after")
    def identity_and_attestation_are_consistent(
        self,
    ) -> ProviderInvocationObservation:
        if len(self.observed_tool_names) != len(set(self.observed_tool_names)):
            raise ValueError("observed tool names must be unique")
        if self.model_attestation == "REQUESTED_ONLY" and self.reported_model is not None:
            raise ValueError("requested-only identity cannot carry a reported model")
        if self.model_attestation == "PROVIDER_REPORTED" and self.reported_model is None:
            raise ValueError("provider-reported identity requires a reported model")
        for value in (self.requested_model, self.reported_model):
            if value is not None and (
                value != value.strip()
                or any(ord(character) < 0x20 for character in value)
            ):
                raise ValueError("provider model identity is malformed")
        return self


class StageInvocationObservation(ClosedModel):
    """Engine-captured identity and hashes independent of provider receipts."""

    stage: Literal["gpt1", "gpt2", "gpt3"]
    provider: str = Field(..., min_length=1, max_length=80)
    model: str = Field(..., min_length=1, max_length=160)
    system_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    user_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    response_schema_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    response_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")

    @field_validator("provider", "model")
    @classmethod
    def invocation_identity_is_log_safe(cls, value: str) -> str:
        if value != value.strip() or any(ord(character) < 0x20 for character in value):
            raise ValueError("stage invocation identity is malformed")
        return value


class RuntimeBinding(ClosedModel):
    python_implementation: str = Field(..., min_length=1, max_length=40)
    python_version: str = Field(..., min_length=1, max_length=40)
    platform_system: str = Field(..., min_length=1, max_length=40)
    platform_machine: str = Field(..., min_length=1, max_length=80)
    sqlite_version: str = Field(..., min_length=1, max_length=40)
    package_versions: dict[str, str]

    @model_validator(mode="after")
    def package_set_is_closed(self) -> RuntimeBinding:
        if set(self.package_versions) != set(_RUNTIME_PACKAGES):
            raise ValueError("runtime package version set is not closed")
        if any(
            not value or len(value) > 80 or any(ord(char) < 0x20 for char in value)
            for value in self.package_versions.values()
        ):
            raise ValueError("runtime package version is malformed")
        return self


class ImplementationBinding(ClosedModel):
    file_sha256: dict[str, str]
    runtime: RuntimeBinding
    aggregate_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def files_and_aggregate_are_closed(self) -> ImplementationBinding:
        if set(self.file_sha256) != set(_IMPLEMENTATION_FILES):
            raise ValueError("implementation file set is not closed")
        if any(not _SHA256_RE.fullmatch(value) for value in self.file_sha256.values()):
            raise ValueError("implementation file digest is malformed")
        expected = hashlib.sha256(_canonical_json_bytes({
            "file_sha256": self.file_sha256,
            "runtime": self.runtime.model_dump(mode="json"),
        })).hexdigest()
        if self.aggregate_sha256 != expected:
            raise ValueError("implementation aggregate digest does not match files")
        return self


def implementation_binding() -> ImplementationBinding:
    """Bind exact recorder sources and resolved retrieval/model SDK runtime."""
    repository_root = Path(__file__).resolve().parent.parent
    digests = {
        relative_path: hashlib.sha256(_read_regular_file(
            repository_root / relative_path,
            label=f"implementation file {relative_path}",
        )).hexdigest()
        for relative_path in _IMPLEMENTATION_FILES
    }
    runtime = RuntimeBinding(
        python_implementation=platform.python_implementation(),
        python_version=platform.python_version(),
        platform_system=platform.system(),
        platform_machine=platform.machine(),
        sqlite_version=sqlite3.sqlite_version,
        package_versions={
            package: importlib.metadata.version(package)
            for package in _RUNTIME_PACKAGES
        },
    )
    aggregate_body = {
        "file_sha256": digests,
        "runtime": runtime.model_dump(mode="json"),
    }
    return ImplementationBinding(
        file_sha256=digests,
        runtime=runtime,
        aggregate_sha256=hashlib.sha256(
            _canonical_json_bytes(aggregate_body)
        ).hexdigest(),
    )

class RetrievedEvidenceObservation(ClosedModel):
    evidence_id: str = Field(..., min_length=1, max_length=80)
    rank: int = Field(..., ge=1, le=100)
    retrieval_score: float
    document_id: str = Field(..., min_length=1, max_length=64)
    document_revision_id: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    folder: str = Field(..., min_length=1, max_length=520)
    title: str = Field(..., min_length=1, max_length=300)
    relative_path: str = Field(..., min_length=1, max_length=1_000)
    source_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    chunk_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    start_char: int = Field(..., ge=0)
    end_char: int = Field(..., gt=0)
    start_line: int = Field(..., ge=1)
    end_line: int = Field(..., ge=1)

    @model_validator(mode="after")
    def spans_are_ordered(self) -> RetrievedEvidenceObservation:
        if self.end_char <= self.start_char or self.end_line < self.start_line:
            raise ValueError("retrieved evidence span is not ordered")
        return self


class PrivateSpanObservation(ClosedModel):
    evidence_id: str = Field(..., min_length=1, max_length=80)
    quote: str = Field(..., min_length=1, max_length=1_200)
    valid: bool
    document_id: str | None = Field(default=None, min_length=1, max_length=64)
    document_revision_id: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    title: str | None = Field(default=None, min_length=1, max_length=300)
    relative_path: str | None = Field(default=None, min_length=1, max_length=1_000)
    source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_start_char: int | None = Field(default=None, ge=0)
    source_end_char: int | None = Field(default=None, gt=0)
    start_line: int | None = Field(default=None, ge=1)
    end_line: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validity_matches_resolved_provenance(self) -> PrivateSpanObservation:
        resolved = (
            self.document_id,
            self.document_revision_id,
            self.title,
            self.relative_path,
            self.source_sha256,
            self.source_start_char,
            self.source_end_char,
            self.start_line,
            self.end_line,
        )
        if self.valid and any(value is None for value in resolved):
            raise ValueError("a valid span requires resolved provenance")
        if not self.valid and any(value is not None for value in resolved):
            raise ValueError("an invalid span cannot claim resolved provenance")
        if (
            self.valid
            and self.source_start_char is not None
            and self.source_end_char is not None
            and self.source_end_char <= self.source_start_char
        ):
            raise ValueError("resolved span is not ordered")
        if (
            self.valid
            and self.start_line is not None
            and self.end_line is not None
            and self.end_line < self.start_line
        ):
            raise ValueError("resolved line span is not ordered")
        return self


class DraftClaimObservation(ClosedModel):
    claim_id: str = Field(..., pattern=r"^C[1-9][0-9]{0,2}$")
    # Production records the NFKC/whitespace-normalized form. A non-empty raw
    # ProposedClaim may normalize to empty or expand beyond its 1,500-codepoint
    # input limit, and the evaluation capture must preserve that exact outcome.
    text: str = Field(..., max_length=30_000)
    render_safe: bool
    proposed_citations: list[PrivateSpanObservation] = Field(
        default_factory=list,
        max_length=4,
    )


class AnswererStageObservation(ClosedModel):
    called: bool
    protocol_valid: bool | None
    answerability: Literal["ANSWERABLE", "UNANSWERABLE"] | None
    claims: list[DraftClaimObservation] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def call_state_is_consistent(self) -> AnswererStageObservation:
        if not self.called and (
            self.protocol_valid is not None
            or self.answerability is not None
            or self.claims
        ):
            raise ValueError("an uncalled answerer cannot have output")
        if self.called and (
            self.protocol_valid is None or self.answerability is None
        ):
            raise ValueError("a called answerer requires a closed outcome")
        claim_ids = [claim.claim_id for claim in self.claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("answerer claim IDs must be unique")
        return self


class VerifierCheckObservation(ClosedModel):
    claim_id: str = Field(..., min_length=1, max_length=20)
    raw_verdict: Literal["SUPPORTED", "CONTRADICTED", "INSUFFICIENT", "CONFLICT"]
    effective_verdict: Literal[
        "SUPPORTED", "CONTRADICTED", "INSUFFICIENT", "CONFLICT"
    ] | None
    support_spans: list[PrivateSpanObservation] = Field(default_factory=list, max_length=6)

    @model_validator(mode="after")
    def supported_effective_verdict_has_valid_spans(self) -> VerifierCheckObservation:
        if self.effective_verdict == "SUPPORTED" and (
            not self.support_spans or not all(span.valid for span in self.support_spans)
        ):
            raise ValueError("effective SUPPORTED requires only validated spans")
        if self.raw_verdict == "SUPPORTED" and any(
            not span.valid for span in self.support_spans
        ) and self.effective_verdict != "INSUFFICIENT":
            raise ValueError("an invalid SUPPORTED span must downgrade to INSUFFICIENT")
        return self


class VerifierStageObservation(ClosedModel):
    called: bool
    protocol_valid: bool | None
    checks: list[VerifierCheckObservation] = Field(default_factory=list, max_length=12)
    eligible_claim_ids: list[str] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def call_state_is_consistent(self) -> VerifierStageObservation:
        if not self.called and (
            self.protocol_valid is not None or self.checks or self.eligible_claim_ids
        ):
            raise ValueError("an uncalled verifier cannot have output")
        if self.called and (
            self.protocol_valid is None
            or any(check.effective_verdict is None for check in self.checks)
        ):
            raise ValueError("a called verifier requires a closed outcome")
        if len(self.eligible_claim_ids) != len(set(self.eligible_claim_ids)):
            raise ValueError("eligible claim IDs must be unique")
        return self


class FinalizerStageObservation(ClosedModel):
    called: bool
    protocol_valid: bool | None
    decision: Literal["ANSWER", "PARTIAL", "ABSTAIN"] | None
    requested_claim_ids: list[str] = Field(default_factory=list, max_length=12)
    accepted_claim_ids: list[str] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def call_state_is_consistent(self) -> FinalizerStageObservation:
        if not self.called and (
            self.protocol_valid is not None
            or self.decision is not None
            or self.requested_claim_ids
            or self.accepted_claim_ids
        ):
            raise ValueError("an uncalled finalizer cannot have output")
        if self.called and (
            self.protocol_valid is None or self.decision is None
        ):
            raise ValueError("a called finalizer requires a closed outcome")
        if len(self.requested_claim_ids) != len(set(self.requested_claim_ids)):
            # Raw duplicates are represented by protocol_valid=false, but are
            # still retained. Do not reject them here.
            if self.protocol_valid is not False:
                raise ValueError("duplicate requested IDs require protocol failure")
        if len(self.accepted_claim_ids) != len(set(self.accepted_claim_ids)):
            raise ValueError("accepted claim IDs must be unique")
        if self.protocol_valid is not True and self.accepted_claim_ids:
            raise ValueError("only a valid finalizer can accept claims")
        return self


class StageLatencyObservation(ClosedModel):
    retrieval_ms: float = Field(..., ge=0)
    answerer_ms: float | None = Field(default=None, ge=0)
    verifier_ms: float | None = Field(default=None, ge=0)
    finalizer_ms: float | None = Field(default=None, ge=0)
    total_ms: float = Field(..., ge=0)
    capture_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")


def latency_capture_sha256(
    receipt_sha256: str,
    *,
    retrieval_ms: float,
    answerer_ms: float | None,
    verifier_ms: float | None,
    finalizer_ms: float | None,
    total_ms: float,
) -> str:
    """Bind private stage timing to the exact durable production receipt."""
    return hashlib.sha256(_canonical_json_bytes({
        "schema_version": "grounded-latency-capture-v1",
        "receipt_sha256": receipt_sha256,
        "retrieval_ms": retrieval_ms,
        "answerer_ms": answerer_ms,
        "verifier_ms": verifier_ms,
        "finalizer_ms": finalizer_ms,
        "total_ms": total_ms,
    })).hexdigest()


class ResponseCitationObservation(ClosedModel):
    citation_id: str = Field(..., pattern=r"^R[1-9][0-9]{0,2}$")
    evidence_id: str = Field(..., min_length=1, max_length=80)
    document_id: str = Field(..., min_length=1, max_length=64)
    document_revision_id: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    title: str = Field(..., min_length=1, max_length=300)
    relative_path: str = Field(..., min_length=1, max_length=1_000)
    source_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    quote: str = Field(..., min_length=1, max_length=1_200)
    source_start_char: int = Field(..., ge=0)
    source_end_char: int = Field(..., gt=0)
    start_line: int = Field(..., ge=1)
    end_line: int = Field(..., ge=1)

    @model_validator(mode="after")
    def source_span_is_ordered(self) -> ResponseCitationObservation:
        if (
            self.source_end_char <= self.source_start_char
            or self.end_line < self.start_line
        ):
            raise ValueError("response citation span is not ordered")
        return self


class ReleasedClaimObservation(ClosedModel):
    claim_id: str = Field(..., pattern=r"^C[1-9][0-9]{0,2}$")
    citation_ids: list[str] = Field(..., min_length=1, max_length=6)

    @field_validator("citation_ids")
    @classmethod
    def citations_are_unique(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("released claim citation IDs must be unique")
        if any(not _CITATION_ID_RE.fullmatch(value) for value in values):
            raise ValueError("released claim citation ID is malformed")
        return values


class FinalResponseObservation(ClosedModel):
    status: Literal["ANSWER", "PARTIAL", "ABSTAIN"]
    reason_code: str = Field(..., min_length=1, max_length=80)
    packet_id: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    corpus_revision: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    retrieval_count: int = Field(..., ge=0, le=12)
    retrieval_truncated: bool
    coverage_limited: bool
    coverage_reasons: list[
        Literal["query_term_limit", "top_k", "byte_budget"]
    ] = Field(default_factory=list, max_length=3)
    stages_completed: list[
        Literal["retrieval", "answerer", "verifier", "finalizer"]
    ] = Field(..., min_length=1, max_length=4)
    draft_claim_count: int = Field(..., ge=0, le=12)
    supported_claim_count: int = Field(..., ge=0, le=12)
    contradicted_claim_count: int = Field(..., ge=0, le=12)
    conflict_claim_count: int = Field(..., ge=0, le=12)
    insufficient_claim_count: int = Field(..., ge=0, le=12)
    selected_claim_ids: list[str] = Field(default_factory=list, max_length=12)
    released_claims: list[ReleasedClaimObservation] = Field(
        default_factory=list,
        max_length=12,
    )
    citations: list[ResponseCitationObservation] = Field(
        default_factory=list,
        max_length=48,
    )

    @model_validator(mode="after")
    def response_shape_is_closed(self) -> FinalResponseObservation:
        if self.reason_code not in _STATUS_REASONS[self.status]:
            raise ValueError("reason_code is incompatible with status")
        ordered_stages = list(_STAGES[: len(self.stages_completed)])
        if self.stages_completed != ordered_stages:
            raise ValueError("stages_completed must be a pipeline prefix")
        ordered_reasons = [
            reason for reason in _COVERAGE_REASONS if reason in self.coverage_reasons
        ]
        if self.coverage_reasons != ordered_reasons:
            raise ValueError("coverage reasons must be unique and ordered")
        if self.coverage_limited != bool(self.coverage_reasons):
            raise ValueError("coverage_limited must match coverage_reasons")
        if self.retrieval_truncated != ("byte_budget" in self.coverage_reasons):
            raise ValueError("retrieval_truncated must represent byte budget")
        if len(self.selected_claim_ids) != len(set(self.selected_claim_ids)):
            raise ValueError("selected claim IDs must be unique")
        released_ids = [claim.claim_id for claim in self.released_claims]
        if released_ids != self.selected_claim_ids:
            raise ValueError("released claims must exactly follow selected claim IDs")
        citation_ids = [citation.citation_id for citation in self.citations]
        expected_citation_ids = [f"R{index}" for index in range(1, len(citation_ids) + 1)]
        if citation_ids != expected_citation_ids:
            raise ValueError("response citation IDs must be sequential")
        citation_set = set(citation_ids)
        if any(
            citation_id not in citation_set
            for claim in self.released_claims
            for citation_id in claim.citation_ids
        ):
            raise ValueError("released claim names an unknown response citation")
        if self.status == "ABSTAIN" and (
            self.selected_claim_ids or self.released_claims or self.citations
        ):
            raise ValueError("an abstention cannot release claims or citations")
        if self.status != "ABSTAIN" and (
            not self.selected_claim_ids or not self.citations
        ):
            raise ValueError("an answer must release claims and citations")
        return self


class PromptVersionsObservation(ClosedModel):
    answerer: Literal["grounded-answerer-v1"]
    verifier: Literal["grounded-verifier-v1"]
    finalizer: Literal["grounded-finalizer-v1"]


class StageFingerprintsObservation(ClosedModel):
    gpt1: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    gpt2: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    gpt3: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class ProductionReceiptObservation(ClosedModel):
    schema_version: Literal["grounded-run-receipt-v1"]
    contract_version: Literal["grounded-rag-v1"]
    run_id: str = Field(..., pattern=r"^[0-9a-f]{32}$")
    created_at: str = Field(..., min_length=1, max_length=80)
    latency_ms: int = Field(..., ge=0)
    corpus_revision: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    packet_id: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    packet_schema_version: Literal[2]
    retrieval_version: Literal["sqlite-fts5-v2"]
    chunker_version: Literal["words-180-overlap-30-chars-4000-v2"]
    coverage_limited: bool
    coverage_reasons: list[
        Literal["query_term_limit", "top_k", "byte_budget"]
    ] = Field(default_factory=list, max_length=3)
    prompt_versions: PromptVersionsObservation
    stage_fingerprints: StageFingerprintsObservation
    stages_completed: list[
        Literal["retrieval", "answerer", "verifier", "finalizer"]
    ] = Field(..., min_length=1, max_length=4)
    draft_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    verification_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    selected_claim_ids: list[str] = Field(default_factory=list, max_length=12)
    citation_evidence_ids: list[str] = Field(default_factory=list, max_length=48)
    status: Literal["ANSWER", "PARTIAL", "ABSTAIN"]
    reason_code: str = Field(..., min_length=1, max_length=80)
    draft_claim_count: int = Field(..., ge=0, le=12)
    supported_claim_count: int = Field(..., ge=0, le=12)
    contradicted_claim_count: int = Field(..., ge=0, le=12)
    conflict_claim_count: int = Field(..., ge=0, le=12)
    insufficient_claim_count: int = Field(..., ge=0, le=12)


class ExecutionIdentityObservation(ClosedModel):
    response_run_id: str = Field(..., pattern=r"^[0-9a-f]{32}$")
    contract_version: Literal["grounded-rag-v1"]
    packet_schema_version: Literal[2]
    retrieval_version: Literal["sqlite-fts5-v2"]
    chunker_version: Literal["words-180-overlap-30-chars-4000-v2"]
    prompt_versions: PromptVersionsObservation
    stage_fingerprints: StageFingerprintsObservation
    draft_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    verification_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    receipt_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    receipt: ProductionReceiptObservation

    @model_validator(mode="after")
    def receipt_hash_is_exact(self) -> ExecutionIdentityObservation:
        expected = hashlib.sha256(
            _canonical_json_bytes(self.receipt.model_dump(mode="json"))
        ).hexdigest()
        if self.receipt_sha256 != expected:
            raise ValueError("production receipt hash does not match receipt bytes")
        return self


class CaseObservation(ClosedModel):
    case_id: str = Field(..., pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    packet_id: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    corpus_revision: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    retrieval_k: int = Field(..., ge=1, le=12)
    execution_identity: ExecutionIdentityObservation
    retrieval: list[RetrievedEvidenceObservation] = Field(
        default_factory=list,
        max_length=12,
    )
    answerer: AnswererStageObservation
    verifier: VerifierStageObservation
    finalizer: FinalizerStageObservation
    response: FinalResponseObservation
    latency: StageLatencyObservation
    provider_observation_mode: Literal[
        "CONFIGURED_UNOBSERVED",
        "AGY_SDK",
    ] = "CONFIGURED_UNOBSERVED"
    stage_invocations: list[StageInvocationObservation] = Field(
        default_factory=list,
        max_length=3,
    )
    provider_invocations: list[ProviderInvocationObservation] = Field(
        default_factory=list,
        max_length=3,
    )
    usage_cost: UnavailableUsageCost = Field(default_factory=UnavailableUsageCost)

    @model_validator(mode="after")
    def cross_stage_ids_are_consistent(self) -> CaseObservation:
        if self.packet_id != self.response.packet_id:
            raise ValueError("case and response packet IDs differ")
        if self.corpus_revision != self.response.corpus_revision:
            raise ValueError("case and response corpus revisions differ")
        if self.response.retrieval_count != len(self.retrieval):
            raise ValueError("response retrieval_count differs from captured retrieval")
        expected_provider_stages = [
            stage
            for stage, called in (
                ("gpt1", self.answerer.called),
                ("gpt2", self.verifier.called),
                ("gpt3", self.finalizer.called),
            )
            if called
        ]
        if (
            [item.stage for item in self.stage_invocations]
            != expected_provider_stages
        ):
            raise ValueError("stage invocation captures do not match called stage order")
        fingerprints = self.execution_identity.stage_fingerprints
        for invocation in self.stage_invocations:
            expected_fingerprint = hashlib.sha256(_canonical_json_bytes({
                "model": invocation.model,
                "provider": invocation.provider,
            })).hexdigest()
            if getattr(fingerprints, invocation.stage) != expected_fingerprint:
                raise ValueError(
                    "stage invocation identity does not match its fingerprint"
                )

        if self.provider_observation_mode == "CONFIGURED_UNOBSERVED":
            if self.provider_invocations:
                raise ValueError(
                    "configured-unobserved mode cannot carry provider invocations"
                )
            if any(
                item.provider == "google-antigravity-sdk"
                for item in self.stage_invocations
            ):
                raise ValueError("AGY stage invocation requires AGY receipt mode")
        else:
            actual_provider_stages = [
                invocation.stage for invocation in self.provider_invocations
            ]
            if actual_provider_stages != expected_provider_stages:
                raise ValueError("provider invocations do not match called stage order")
            for captured, invocation in zip(
                self.stage_invocations,
                self.provider_invocations,
                strict=True,
            ):
                if (
                    captured.provider != invocation.provider
                    or captured.model != invocation.requested_model
                    or captured.system_sha256 != invocation.system_sha256
                    or captured.user_sha256 != invocation.user_sha256
                    or captured.response_schema_sha256
                    != invocation.response_schema_sha256
                    or captured.response_sha256 != invocation.response_sha256
                ):
                    raise ValueError(
                        "provider invocation does not match engine capture"
                    )
        ranks = [item.rank for item in self.retrieval]
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError("retrieval ranks must be sequential")
        evidence_ids = [item.evidence_id for item in self.retrieval]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("retrieved evidence IDs must be unique")
        claim_ids = [claim.claim_id for claim in self.answerer.claims]
        if self.answerer.protocol_valid is True:
            expected = [f"C{index}" for index in range(1, len(claim_ids) + 1)]
            if claim_ids != expected:
                raise ValueError("valid answerer claim IDs must be sequential")
        claim_set = set(claim_ids)
        if self.verifier.called and (
            not self.answerer.called
            or self.answerer.protocol_valid is not True
            or self.answerer.answerability != "ANSWERABLE"
            or not self.answerer.claims
        ):
            raise ValueError("verifier call lacks a valid answerable draft")
        if any(claim_id not in claim_set for claim_id in self.verifier.eligible_claim_ids):
            raise ValueError("eligible claim is absent from the draft")
        if self.verifier.protocol_valid is True:
            returned_ids = [check.claim_id for check in self.verifier.checks]
            if len(returned_ids) != len(set(returned_ids)) or set(returned_ids) != claim_set:
                raise ValueError("valid verifier checks must cover every draft claim")
            if any(check.effective_verdict is None for check in self.verifier.checks):
                raise ValueError("valid verifier checks require effective verdicts")
        if any(claim_id not in claim_set for claim_id in self.finalizer.accepted_claim_ids):
            raise ValueError("accepted claim is absent from the draft")
        if self.finalizer.called and (
            not self.verifier.called
            or self.verifier.protocol_valid is not True
            or not self.verifier.eligible_claim_ids
        ):
            raise ValueError("finalizer call lacks eligible verified claims")
        if any(
            claim_id not in self.verifier.eligible_claim_ids
            for claim_id in self.finalizer.accepted_claim_ids
        ):
            raise ValueError("accepted claim did not clear verifier eligibility")
        expected_accepted = (
            self.finalizer.requested_claim_ids
            if self.finalizer.protocol_valid is True
            and self.finalizer.decision != "ABSTAIN"
            and self.finalizer.requested_claim_ids
            else []
        )
        if self.finalizer.accepted_claim_ids != expected_accepted:
            raise ValueError("accepted claims are not derived from finalizer output")
        if self.finalizer.accepted_claim_ids != self.response.selected_claim_ids:
            raise ValueError("accepted and released claim IDs differ")
        if self.finalizer.called:
            if self.finalizer.protocol_valid is not True or not expected_accepted:
                if self.response.status != "ABSTAIN":
                    raise ValueError("failed or abstaining finalizer requires abstention")
            elif self.finalizer.decision == "PARTIAL" and self.response.status != "PARTIAL":
                raise ValueError("partial finalizer decision requires partial response")
        elif self.response.status != "ABSTAIN":
            raise ValueError("an uncalled finalizer cannot release an answer")
        stage_called = {
            "answerer": self.answerer.called,
            "verifier": self.verifier.called,
            "finalizer": self.finalizer.called,
        }
        for stage, called in stage_called.items():
            if called != (stage in self.response.stages_completed):
                raise ValueError("stage call state differs from stages_completed")
        stage_latencies = {
            "answerer": self.latency.answerer_ms,
            "verifier": self.latency.verifier_ms,
            "finalizer": self.latency.finalizer_ms,
        }
        for stage, latency in stage_latencies.items():
            if (latency is not None) != stage_called[stage]:
                raise ValueError("stage latency presence differs from call state")
        measured_stage_total = self.latency.retrieval_ms + sum(
            latency for latency in stage_latencies.values() if latency is not None
        )
        # The receipt rounds the pre-persistence elapsed time to an integer.
        # Stage timers are floats, so allow only that bounded rounding gap.
        if self.latency.total_ms + 0.500001 < measured_stage_total:
            raise ValueError("total latency is below measured stage latency")
        processed_draft_count = (
            len(self.answerer.claims)
            if self.answerer.protocol_valid is True
            and self.answerer.answerability == "ANSWERABLE"
            else 0
        )
        if self.response.draft_claim_count != processed_draft_count:
            raise ValueError("response draft count differs from captured claims")
        if self.verifier.protocol_valid is True:
            verdict_counts = Counter(
                check.effective_verdict for check in self.verifier.checks
            )
            response_counts = {
                "SUPPORTED": self.response.supported_claim_count,
                "CONTRADICTED": self.response.contradicted_claim_count,
                "CONFLICT": self.response.conflict_claim_count,
                "INSUFFICIENT": self.response.insufficient_claim_count,
            }
            if any(
                response_counts[verdict] != verdict_counts[verdict]
                for verdict in response_counts
            ):
                raise ValueError("response verdict counts differ from verifier checks")
        stage_fingerprint_values = tuple(
            self.execution_identity.stage_fingerprints.model_dump().values()
        )
        if self.retrieval and any(value is None for value in stage_fingerprint_values):
            raise ValueError("non-empty retrieval requires configured stage fingerprints")
        if not self.retrieval and any(
            value is not None for value in stage_fingerprint_values
        ):
            raise ValueError("empty retrieval cannot have configured stage fingerprints")
        receipt = self.execution_identity.receipt
        if self.latency.total_ms != float(receipt.latency_ms):
            raise ValueError("total latency does not match production receipt")
        expected_latency_hash = latency_capture_sha256(
            self.execution_identity.receipt_sha256,
            retrieval_ms=self.latency.retrieval_ms,
            answerer_ms=self.latency.answerer_ms,
            verifier_ms=self.latency.verifier_ms,
            finalizer_ms=self.latency.finalizer_ms,
            total_ms=self.latency.total_ms,
        )
        if self.latency.capture_sha256 != expected_latency_hash:
            raise ValueError("latency capture hash does not match captured timing")
        expected_citation_evidence = list(dict.fromkeys(
            citation.evidence_id for citation in self.response.citations
        ))
        if (
            receipt.run_id != self.execution_identity.response_run_id
            or receipt.packet_id != self.packet_id
            or receipt.corpus_revision != self.corpus_revision
            or receipt.contract_version != self.execution_identity.contract_version
            or receipt.packet_schema_version
            != self.execution_identity.packet_schema_version
            or receipt.retrieval_version != self.execution_identity.retrieval_version
            or receipt.chunker_version != self.execution_identity.chunker_version
            or receipt.prompt_versions != self.execution_identity.prompt_versions
            or receipt.stage_fingerprints
            != self.execution_identity.stage_fingerprints
            or receipt.draft_hash != self.execution_identity.draft_hash
            or receipt.verification_hash != self.execution_identity.verification_hash
            or receipt.coverage_limited != self.response.coverage_limited
            or receipt.coverage_reasons != self.response.coverage_reasons
            or receipt.stages_completed != self.response.stages_completed
            or receipt.selected_claim_ids != self.response.selected_claim_ids
            or receipt.citation_evidence_ids != expected_citation_evidence
            or receipt.status != self.response.status
            or receipt.reason_code != self.response.reason_code
            or receipt.draft_claim_count != self.response.draft_claim_count
            or receipt.supported_claim_count != self.response.supported_claim_count
            or receipt.contradicted_claim_count
            != self.response.contradicted_claim_count
            or receipt.conflict_claim_count != self.response.conflict_claim_count
            or receipt.insufficient_claim_count
            != self.response.insufficient_claim_count
        ):
            raise ValueError("production receipt does not match captured response")
        return self


class ObservationFailure(ClosedModel):
    kind: Literal["PROVIDER_OR_PIPELINE_ERROR", "INTERRUPTED"]
    failed_case_id: str | None = Field(
        default=None,
        pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$",
    )


class ObservationBundle(ClosedModel):
    schema_version: Literal["grounded-rag-observations-v2"] = (
        OBSERVATION_SCHEMA_VERSION
    )
    benchmark_id: str = Field(..., pattern=r"^[a-z0-9][a-z0-9_-]{0,79}$")
    benchmark_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    implementation: ImplementationBinding
    run_id: str = Field(..., pattern=r"^[0-9a-f]{32}$")
    provider_observation_mode: Literal[
        "CONFIGURED_UNOBSERVED",
        "AGY_SDK",
    ] = "CONFIGURED_UNOBSERVED"
    status: Literal["COMPLETE", "INCOMPLETE"]
    cases: list[CaseObservation] = Field(default_factory=list, max_length=100)
    failure: ObservationFailure | None = None
    usage_cost: UnavailableUsageCost = Field(default_factory=UnavailableUsageCost)

    @model_validator(mode="after")
    def completion_state_is_consistent(self) -> ObservationBundle:
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("observation case IDs must be unique")
        if self.status == "COMPLETE" and self.failure is not None:
            raise ValueError("a complete run cannot have a failure")
        if self.status == "INCOMPLETE" and self.failure is None:
            raise ValueError("an incomplete run requires a closed failure code")
        if any(
            case.provider_observation_mode != self.provider_observation_mode
            for case in self.cases
        ):
            raise ValueError("case provider mode differs from the recording runner")
        runtime_bindings = {
            (
                invocation.provider,
                invocation.requested_model,
                invocation.sdk_distribution,
                invocation.sdk_version,
                invocation.sdk_artifact_sha256,
                invocation.localharness_sha256,
                invocation.worker_protocol_version,
                invocation.worker_source_sha256,
                invocation.invocation_policy_sha256,
            )
            for case in self.cases
            for invocation in case.provider_invocations
        }
        if len(runtime_bindings) > 1:
            raise ValueError("provider runtime identity changed during recording")
        if self.provider_observation_mode == "AGY_SDK":
            if any(
                invocation.invocation_policy_sha256
                != AGY_INVOCATION_POLICY_SHA256
                for case in self.cases
                for invocation in case.provider_invocations
            ):
                raise ValueError(
                    "provider receipt does not match the bound worker policy"
                )
        return self


class ClaimAdjudication(ClosedModel):
    claim_id: str = Field(..., pattern=r"^C[1-9][0-9]{0,2}$")
    corpus_supported: bool


class CitationAdjudication(ClosedModel):
    citation_id: str = Field(..., pattern=r"^R[1-9][0-9]{0,2}$")
    supported_claim_ids: list[str] = Field(default_factory=list, max_length=12)

    @field_validator("supported_claim_ids")
    @classmethod
    def supported_claims_are_unique(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("supported claim IDs must be unique")
        if any(not _CLAIM_ID_RE.fullmatch(value) for value in values):
            raise ValueError("supported claim ID is malformed")
        return values


class CaseAdjudication(ClosedModel):
    case_id: str = Field(..., pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    claims: list[ClaimAdjudication] = Field(default_factory=list, max_length=12)
    released_citations: list[CitationAdjudication] = Field(
        default_factory=list,
        max_length=48,
    )


class AdjudicationBundle(ClosedModel):
    schema_version: Literal["grounded-rag-adjudication-v1"] = (
        ADJUDICATION_SCHEMA_VERSION
    )
    benchmark_id: str = Field(..., pattern=r"^[a-z0-9][a-z0-9_-]{0,79}$")
    benchmark_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    observation_run_id: str = Field(..., pattern=r"^[0-9a-f]{32}$")
    observation_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    label_authority: Literal["independent_human"]
    cases: list[CaseAdjudication] = Field(..., min_length=1, max_length=100)


def validate_observation_bundle(
    benchmark: ResolvedBenchmark,
    observations: ObservationBundle,
) -> None:
    """Reject wrong binding, selective omission, or source-hash substitution."""
    try:
        observations = ObservationBundle.model_validate(
            observations.model_dump(mode="json")
        )
    except ValidationError as exc:
        raise EvaluationDataError(
            "observations violate closed cross-stage invariants"
        ) from exc
    definition = benchmark.definition
    if observations.benchmark_id != definition.benchmark_id:
        raise EvaluationDataError("observation benchmark_id does not match")
    if observations.benchmark_sha256 != benchmark.raw_sha256:
        raise EvaluationDataError("observation benchmark SHA-256 does not match")
    expected_case_ids = [case.case_id for case in definition.cases]
    actual_case_ids = [case.case_id for case in observations.cases]
    if observations.status == "COMPLETE":
        if actual_case_ids != expected_case_ids:
            raise EvaluationDataError(
                "complete observations must contain every benchmark case in order"
            )
    elif actual_case_ids != expected_case_ids[: len(actual_case_ids)]:
        raise EvaluationDataError(
            "incomplete observations may contain only the executed case prefix"
        )
    if observations.failure is not None:
        next_case = (
            expected_case_ids[len(actual_case_ids)]
            if len(actual_case_ids) < len(expected_case_ids)
            else None
        )
        if observations.failure.failed_case_id != next_case:
            raise EvaluationDataError("failure must identify the next unrecorded case")

    expected_documents, expected_corpus_revision = _expected_corpus(benchmark)
    cases_by_id = {case.case_id: case for case in definition.cases}
    for case in observations.cases:
        benchmark_case = cases_by_id[case.case_id]
        expected_packet = benchmark.expected_packets_by_case[case.case_id]
        if case.retrieval_k != definition.top_k:
            raise EvaluationDataError("observation retrieval_k does not match benchmark")
        if len(case.retrieval) > case.retrieval_k:
            raise EvaluationDataError("observation retrieval exceeds its declared k")
        if case.corpus_revision != expected_corpus_revision or (
            case.corpus_revision != expected_packet.corpus_revision
        ):
            raise EvaluationDataError("observation corpus revision does not match benchmark")
        if (
            case.packet_id != expected_packet.packet_id
            or case.response.retrieval_truncated != expected_packet.truncated
            or case.response.coverage_limited != expected_packet.coverage_limited
            or tuple(case.response.coverage_reasons)
            != expected_packet.coverage_reasons
            or len(case.retrieval) != len(expected_packet.items)
        ):
            raise EvaluationDataError(
                "observation does not match deterministic benchmark retrieval"
            )
        identity = case.execution_identity
        if (
            identity.contract_version != "grounded-rag-v1"
            or identity.packet_schema_version != SCHEMA_VERSION
            or identity.retrieval_version != RETRIEVER_VERSION
            or identity.chunker_version != CHUNKER_VERSION
        ):
            raise EvaluationDataError("observation execution contract is unsupported")

        retrieval_by_id: dict[str, RetrievedEvidenceObservation] = {}
        packet_items: list[dict[str, Any]] = []
        for item, packet_item in zip(
            case.retrieval,
            expected_packet.items,
            strict=True,
        ):
            expected_document = expected_documents.get(item.document_id)
            if expected_document is None:
                raise EvaluationDataError("retrieval names an unknown benchmark document")
            expected_chunk = expected_document.chunks_by_evidence_id.get(
                item.evidence_id
            )
            if expected_chunk is None:
                raise EvaluationDataError("retrieval evidence ID is not a benchmark chunk")
            expected_values = (
                expected_document.revision_id,
                expected_document.definition.folder,
                expected_document.definition.title,
                expected_document.relative_path,
                expected_document.source_sha256,
                expected_chunk.chunk_sha256,
                expected_chunk.start_char,
                expected_chunk.end_char,
                expected_chunk.start_line,
                expected_chunk.end_line,
            )
            actual_values = (
                item.document_revision_id,
                item.folder,
                item.title,
                item.relative_path,
                item.source_sha256,
                item.chunk_sha256,
                item.start_char,
                item.end_char,
                item.start_line,
                item.end_line,
            )
            if actual_values != expected_values:
                raise EvaluationDataError(
                    "retrieval provenance does not match the frozen benchmark chunk"
                )
            expected_packet_item = packet_item.prompt_dict()
            expected_packet_item.pop("text")
            if item.model_dump(mode="json") != expected_packet_item:
                raise EvaluationDataError(
                    "observation differs from deterministic ranked retrieval"
                )
            retrieval_by_id[item.evidence_id] = item
            packet_items.append({
                **item.model_dump(mode="json"),
                "text": expected_chunk.text,
            })

        packet_body = {
            "schema_version": SCHEMA_VERSION,
            "corpus_revision": expected_corpus_revision,
            "retrieval_version": RETRIEVER_VERSION,
            "canonical_query": canonicalize_query(benchmark_case.query),
            "truncated": case.response.retrieval_truncated,
            "coverage_limited": case.response.coverage_limited,
            "coverage_reasons": list(case.response.coverage_reasons),
            "items": packet_items,
        }
        expected_packet_id = hashlib.sha256(
            _canonical_json_bytes(packet_body)
        ).hexdigest()
        if case.packet_id != expected_packet_id or expected_packet_id != expected_packet.packet_id:
            raise EvaluationDataError("observation packet ID does not match retrieval")

        def validate_private_span(span: PrivateSpanObservation) -> None:
            item = retrieval_by_id.get(span.evidence_id)
            expected_document = (
                None if item is None else expected_documents[item.document_id]
            )
            expected_chunk = (
                None
                if item is None or expected_document is None
                else expected_document.chunks_by_evidence_id[item.evidence_id]
            )
            local_start = (
                -1 if expected_chunk is None else expected_chunk.text.find(span.quote)
            )
            expected_valid = item is not None and local_start >= 0
            if span.valid != expected_valid:
                raise EvaluationDataError(
                    "captured model span validity does not match retrieval"
                )
            if not expected_valid:
                return
            if item is None or expected_document is None or expected_chunk is None:
                raise EvaluationDataError(
                    "validated model span names evidence outside retrieval"
                )
            expected_start = item.start_char + local_start
            expected_end = expected_start + len(span.quote)
            expected_start_line = item.start_line + expected_chunk.text.count(
                "\n", 0, local_start
            )
            expected_end_line = item.start_line + expected_chunk.text.count(
                "\n", 0, local_start + len(span.quote)
            )
            if (
                span.document_id != item.document_id
                or span.document_revision_id != item.document_revision_id
                or span.title != item.title
                or span.relative_path != item.relative_path
                or span.source_sha256 != item.source_sha256
                or span.source_start_char is None
                or span.source_end_char is None
                or span.source_start_char != expected_start
                or span.source_end_char != expected_end
                or span.start_line != expected_start_line
                or span.end_line != expected_end_line
                or not (
                    item.start_char
                    <= span.source_start_char
                    < span.source_end_char
                    <= item.end_char
                )
                or expected_document.definition.content[
                    span.source_start_char:span.source_end_char
                ]
                != span.quote
            ):
                raise EvaluationDataError(
                    "validated model span does not match frozen source bytes"
                )

        for claim in case.answerer.claims:
            normalized_text = unicodedata.normalize(
                "NFKC",
                " ".join(claim.text.split()),
            )
            expected_render_safe = (
                normalized_text == claim.text
                and not any(
                    unicodedata.category(character) in {"Cc", "Cf"}
                    for character in claim.text
                )
                and "[" not in claim.text
                and "]" not in claim.text
            )
            if claim.render_safe != expected_render_safe:
                raise EvaluationDataError(
                    "render safety is not derived from captured claim text"
                )
            for citation in claim.proposed_citations:
                validate_private_span(citation)
        valid_verifier_spans_by_claim: dict[
            str,
            set[tuple[str, int, int, str]],
        ] = {}
        for check in case.verifier.checks:
            claim_spans = valid_verifier_spans_by_claim.setdefault(
                check.claim_id,
                set(),
            )
            for span in check.support_spans:
                validate_private_span(span)
                if (
                    span.valid
                    and span.source_start_char is not None
                    and span.source_end_char is not None
                ):
                    claim_spans.add((
                        span.evidence_id,
                        span.source_start_char,
                        span.source_end_char,
                        span.quote,
                    ))

        expected_draft_hash: str | None = None
        if (
            case.answerer.protocol_valid is True
            and case.answerer.answerability == "ANSWERABLE"
            and case.answerer.claims
        ):
            expected_draft_hash = hashlib.sha256(_canonical_json_bytes({
                "packet_id": case.packet_id,
                "claims": [
                    {
                        "claim_id": claim.claim_id,
                        "text": claim.text,
                        "proposed_citations": [
                            {
                                "evidence_id": citation.evidence_id,
                                "quote": citation.quote,
                            }
                            for citation in claim.proposed_citations
                        ],
                    }
                    for claim in case.answerer.claims
                ],
            })).hexdigest()
        if identity.draft_hash != expected_draft_hash:
            raise EvaluationDataError("draft hash does not bind captured answerer data")

        expected_verification_hash: str | None = None
        checks_by_id = {check.claim_id: check for check in case.verifier.checks}
        if case.verifier.protocol_valid is True:
            normalized_checks: list[dict[str, Any]] = []
            for claim in case.answerer.claims:
                check = checks_by_id[claim.claim_id]
                expected_effective = check.raw_verdict
                if check.raw_verdict == "SUPPORTED" and (
                    not check.support_spans
                    or not all(span.valid for span in check.support_spans)
                ):
                    expected_effective = "INSUFFICIENT"
                if check.effective_verdict != expected_effective:
                    raise EvaluationDataError(
                        "effective verdict is not derived from captured verifier data"
                    )
                normalized_checks.append({
                    "claim_id": check.claim_id,
                    "verdict": expected_effective,
                    "support_spans": [
                        {
                            "evidence_id": span.evidence_id,
                            "document_id": span.document_id,
                            "document_revision_id": span.document_revision_id,
                            "title": span.title,
                            "relative_path": span.relative_path,
                            "source_sha256": span.source_sha256,
                            "quote": span.quote,
                            "source_start_char": span.source_start_char,
                            "source_end_char": span.source_end_char,
                            "start_line": span.start_line,
                            "end_line": span.end_line,
                        }
                        for span in check.support_spans
                        if span.valid
                    ],
                })
            expected_verification_hash = hashlib.sha256(
                _canonical_json_bytes({
                    "packet_id": case.packet_id,
                    "draft_hash": expected_draft_hash,
                    "checks": normalized_checks,
                })
            ).hexdigest()
        if identity.verification_hash != expected_verification_hash:
            raise EvaluationDataError(
                "verification hash does not bind captured verifier data"
            )

        expected_eligible_ids: list[str] = []
        if case.verifier.protocol_valid is True:
            for claim in case.answerer.claims:
                check = checks_by_id[claim.claim_id]
                answerer_citations_valid = bool(claim.proposed_citations) and all(
                    citation.valid for citation in claim.proposed_citations
                )
                verifier_citations_valid = bool(check.support_spans) and all(
                    span.valid for span in check.support_spans
                )
                if (
                    answerer_citations_valid
                    and verifier_citations_valid
                    and claim.render_safe
                    and check.effective_verdict == "SUPPORTED"
                ):
                    expected_eligible_ids.append(claim.claim_id)
        if case.verifier.eligible_claim_ids != expected_eligible_ids:
            raise EvaluationDataError(
                "eligible claims are not derived from answerer and verifier gates"
            )

        released_links = {
            released.claim_id: set(released.citation_ids)
            for released in case.response.released_claims
        }
        claims_by_citation = {
            citation.citation_id: {
                claim_id
                for claim_id, citation_ids in released_links.items()
                if citation.citation_id in citation_ids
            }
            for citation in case.response.citations
        }

        for citation in case.response.citations:
            item = retrieval_by_id.get(citation.evidence_id)
            if item is None:
                raise EvaluationDataError(
                    "released citation names evidence outside retrieval"
                )
            expected_document = expected_documents[item.document_id]
            expected_chunk = expected_document.chunks_by_evidence_id[item.evidence_id]
            local_start = expected_chunk.text.find(citation.quote)
            expected_start_line = (
                expected_document.definition.content.count(
                    "\n", 0, citation.source_start_char
                )
                + 1
            )
            expected_end_line = (
                expected_document.definition.content.count(
                    "\n", 0, citation.source_end_char
                )
                + 1
            )
            if (
                citation.document_id != item.document_id
                or citation.document_revision_id != item.document_revision_id
                or citation.title != item.title
                or citation.relative_path != item.relative_path
                or citation.source_sha256 != item.source_sha256
                or local_start < 0
                or citation.source_start_char != item.start_char + local_start
                or citation.source_end_char
                != citation.source_start_char + len(citation.quote)
                or not (
                    item.start_char
                    <= citation.source_start_char
                    < citation.source_end_char
                    <= item.end_char
                )
                or expected_document.definition.content[
                    citation.source_start_char:citation.source_end_char
                ]
                != citation.quote
                or citation.start_line != expected_start_line
                or citation.end_line != expected_end_line
                or not claims_by_citation[citation.citation_id]
                or any(
                    (
                        citation.evidence_id,
                        citation.source_start_char,
                        citation.source_end_char,
                        citation.quote,
                    )
                    not in valid_verifier_spans_by_claim.get(claim_id, set())
                    for claim_id in claims_by_citation[citation.citation_id]
                )
            ):
                raise EvaluationDataError(
                    "released citation does not match verified frozen source bytes"
                )


def validate_adjudication_bundle(
    benchmark: ResolvedBenchmark,
    observations: ObservationBundle,
    adjudication: AdjudicationBundle,
) -> None:
    """Require one independent label for every draft claim and released citation."""
    try:
        adjudication = AdjudicationBundle.model_validate(
            adjudication.model_dump(mode="json")
        )
    except ValidationError as exc:
        raise EvaluationDataError("adjudication violates its closed schema") from exc
    validate_observation_bundle(benchmark, observations)
    if observations.status != "COMPLETE":
        raise EvaluationDataError("incomplete observations cannot be adjudicated or scored")
    if adjudication.benchmark_id != benchmark.definition.benchmark_id:
        raise EvaluationDataError("adjudication benchmark_id does not match")
    if adjudication.benchmark_sha256 != benchmark.raw_sha256:
        raise EvaluationDataError("adjudication benchmark SHA-256 does not match")
    if adjudication.observation_run_id != observations.run_id:
        raise EvaluationDataError("adjudication names a different observation run")
    if adjudication.observation_sha256 != canonical_model_sha256(observations):
        raise EvaluationDataError("adjudication does not bind the exact observations")

    observation_ids = [case.case_id for case in observations.cases]
    adjudication_ids = [case.case_id for case in adjudication.cases]
    if adjudication_ids != observation_ids:
        raise EvaluationDataError("adjudication must cover every case exactly once in order")

    for observation, labels in zip(observations.cases, adjudication.cases, strict=True):
        expected_claim_ids = [claim.claim_id for claim in observation.answerer.claims]
        actual_claim_ids = [claim.claim_id for claim in labels.claims]
        if actual_claim_ids != expected_claim_ids:
            raise EvaluationDataError(
                "adjudication must cover every draft claim exactly once in order"
            )
        expected_citation_ids = [
            citation.citation_id for citation in observation.response.citations
        ]
        actual_citation_ids = [
            citation.citation_id for citation in labels.released_citations
        ]
        if actual_citation_ids != expected_citation_ids:
            raise EvaluationDataError(
                "adjudication must cover every released citation exactly once in order"
            )
        released_links = {
            claim.claim_id: set(claim.citation_ids)
            for claim in observation.response.released_claims
        }
        supported_claims = {
            label.claim_id for label in labels.claims if label.corpus_supported
        }
        for citation_label in labels.released_citations:
            for claim_id in citation_label.supported_claim_ids:
                if claim_id not in released_links:
                    raise EvaluationDataError(
                        "citation adjudication names an unreleased claim"
                    )
                if citation_label.citation_id not in released_links[claim_id]:
                    raise EvaluationDataError(
                        "citation adjudication names an unlinked claim"
                    )
                if claim_id not in supported_claims:
                    raise EvaluationDataError(
                        "a citation cannot support a corpus-unsupported claim"
                    )


class RateMetric(ClosedModel):
    numerator: int = Field(..., ge=0)
    denominator: int = Field(..., ge=0)
    rate: float | None = Field(..., ge=0, le=1)

    @model_validator(mode="after")
    def rate_is_derived(self) -> RateMetric:
        if self.numerator > self.denominator:
            raise ValueError("rate numerator cannot exceed denominator")
        expected = (
            None if self.denominator == 0 else self.numerator / self.denominator
        )
        if expected is None:
            if self.rate is not None or self.numerator != 0:
                raise ValueError("empty rate must be 0/0 with a null value")
        elif self.rate is None or not math.isclose(
            self.rate,
            expected,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("rate is not derived from numerator and denominator")
        return self


class ValueMetric(ClosedModel):
    samples: list[float] = Field(..., max_length=100)
    sample_count: int = Field(..., ge=0)
    total: float | None = Field(..., ge=0)
    mean: float | None = Field(..., ge=0)
    minimum: float | None = Field(..., ge=0)
    maximum: float | None = Field(..., ge=0)
    p50: float | None = Field(..., ge=0)
    p95: float | None = Field(..., ge=0)

    @model_validator(mode="after")
    def aggregates_are_coherent(self) -> ValueMetric:
        if any(value < 0 for value in self.samples):
            raise ValueError("value metric samples cannot be negative")
        if self.samples != sorted(self.samples):
            raise ValueError("value metric samples must be sorted")
        if self.sample_count != len(self.samples):
            raise ValueError("value metric sample_count differs from samples")
        values = (
            self.total,
            self.mean,
            self.minimum,
            self.maximum,
            self.p50,
            self.p95,
        )
        if self.sample_count == 0:
            if any(value is not None for value in values):
                raise ValueError("empty value metric must contain only null aggregates")
            return self
        if any(value is None for value in values):
            raise ValueError("non-empty value metric requires every aggregate")
        assert self.total is not None
        assert self.mean is not None
        assert self.minimum is not None
        assert self.maximum is not None
        assert self.p50 is not None
        assert self.p95 is not None
        expected_total = sum(self.samples)
        expected_p50 = self.samples[math.ceil(0.50 * self.sample_count) - 1]
        expected_p95 = self.samples[math.ceil(0.95 * self.sample_count) - 1]
        expected_values = (
            (self.total, expected_total),
            (self.mean, expected_total / self.sample_count),
            (self.minimum, self.samples[0]),
            (self.maximum, self.samples[-1]),
            (self.p50, expected_p50),
            (self.p95, expected_p95),
        )
        if any(
            not math.isclose(
                actual,
                expected,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            for actual, expected in expected_values
        ):
            raise ValueError("value metric aggregates are not derived from samples")
        if not self.minimum <= self.p50 <= self.p95 <= self.maximum:
            raise ValueError("value metric percentiles are not ordered")
        if not math.isclose(
            self.mean,
            self.total / self.sample_count,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("value metric mean is not derived from total")
        tolerance = max(1e-12, abs(self.total) * 1e-12)
        if not (
            self.minimum * self.sample_count - tolerance
            <= self.total
            <= self.maximum * self.sample_count + tolerance
        ):
            raise ValueError("value metric total is outside its sample bounds")
        return self


class LatencySummary(ClosedModel):
    retrieval_ms: ValueMetric
    answerer_ms: ValueMetric
    verifier_ms: ValueMetric
    finalizer_ms: ValueMetric
    total_ms: ValueMetric


class UnsupportedClaimRates(ClosedModel):
    answerer: RateMetric
    verifier_supported: RateMetric
    released: RateMetric


class ThresholdResult(ClosedModel):
    observed: float | None = Field(..., ge=0, le=1)
    operator: Literal[">=", "<="]
    threshold: float = Field(..., ge=0, le=1)
    passed: bool

    @model_validator(mode="after")
    def pass_state_is_derived(self) -> ThresholdResult:
        expected = self.observed is not None and (
            self.observed >= self.threshold
            if self.operator == ">="
            else self.observed <= self.threshold
        )
        if self.passed != expected:
            raise ValueError("threshold pass state is not derived from values")
        return self


class BaselineSummary(ClosedModel):
    schema_version: Literal["grounded-rag-baseline-summary-v1"] = (
        SUMMARY_SCHEMA_VERSION
    )
    benchmark_id: str
    benchmark_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    observation_run_id: str = Field(..., pattern=r"^[0-9a-f]{32}$")
    observation_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    adjudication_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    implementation_aggregate_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    execution_identity_sha256_by_case: dict[str, str]
    case_count: int = Field(..., ge=1)
    retrieval_k: int = Field(..., ge=1, le=12)
    retrieval_recall_at_k: RateMetric
    unsupported_claim_rates: UnsupportedClaimRates
    verifier_supported_precision: RateMetric
    verifier_supported_recall: RateMetric
    citation_validity: RateMetric
    citation_completeness: RateMetric
    correct_abstention_rate: RateMetric
    false_abstention_rate: RateMetric
    answer_coverage: RateMetric
    coverage_reason_counts: dict[
        Literal["query_term_limit", "top_k", "byte_budget"],
        int,
    ]
    latency: LatencySummary
    usage_cost: UnavailableUsageCost = Field(default_factory=UnavailableUsageCost)
    threshold_contract: BenchmarkThresholds
    threshold_results: dict[str, ThresholdResult]
    thresholds_passed: bool
    result: Literal["PASS", "FAIL"]

    @model_validator(mode="after")
    def result_matches_thresholds(self) -> BaselineSummary:
        if set(self.threshold_results) != set(_THRESHOLD_KEYS):
            raise ValueError("threshold result set is not closed")
        derived_pass = all(
            threshold.passed for threshold in self.threshold_results.values()
        )
        if self.thresholds_passed != derived_pass:
            raise ValueError("thresholds_passed is not derived from results")
        expected = "PASS" if derived_pass else "FAIL"
        if self.result != expected:
            raise ValueError("baseline result must match threshold outcome")
        expected_results = {
            "retrieval_recall_at_k_min": (
                self.retrieval_recall_at_k.rate,
                ">=",
                self.threshold_contract.retrieval_recall_at_k_min,
            ),
            "unsupported_released_claim_rate_max": (
                self.unsupported_claim_rates.released.rate,
                "<=",
                self.threshold_contract.unsupported_released_claim_rate_max,
            ),
            "citation_validity_min": (
                self.citation_validity.rate,
                ">=",
                self.threshold_contract.citation_validity_min,
            ),
            "citation_completeness_min": (
                self.citation_completeness.rate,
                ">=",
                self.threshold_contract.citation_completeness_min,
            ),
            "correct_abstention_rate_min": (
                self.correct_abstention_rate.rate,
                ">=",
                self.threshold_contract.correct_abstention_rate_min,
            ),
            "false_abstention_rate_max": (
                self.false_abstention_rate.rate,
                "<=",
                self.threshold_contract.false_abstention_rate_max,
            ),
            "answer_coverage_min": (
                self.answer_coverage.rate,
                ">=",
                self.threshold_contract.answer_coverage_min,
            ),
            "verifier_supported_precision_min": (
                self.verifier_supported_precision.rate,
                ">=",
                self.threshold_contract.verifier_supported_precision_min,
            ),
            "verifier_supported_recall_min": (
                self.verifier_supported_recall.rate,
                ">=",
                self.threshold_contract.verifier_supported_recall_min,
            ),
        }
        for key, (observed, operator, threshold) in expected_results.items():
            result = self.threshold_results[key]
            same_observed = (
                observed is None and result.observed is None
            ) or (
                observed is not None
                and result.observed is not None
                and math.isclose(
                    observed,
                    result.observed,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            )
            if (
                not same_observed
                or result.operator != operator
                or not math.isclose(
                    result.threshold,
                    threshold,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError("threshold result is not bound to its metric contract")
        if not self.execution_identity_sha256_by_case or any(
            not _CASE_ID_RE.fullmatch(case_id)
            or not _SHA256_RE.fullmatch(digest)
            for case_id, digest in self.execution_identity_sha256_by_case.items()
        ):
            raise ValueError("execution identity digests are malformed")
        if len(self.execution_identity_sha256_by_case) != self.case_count:
            raise ValueError("execution identity count differs from case_count")
        if set(self.coverage_reason_counts) != set(_COVERAGE_REASONS) or any(
            value < 0 for value in self.coverage_reason_counts.values()
        ):
            raise ValueError("coverage reason counts are not closed")
        return self


def _rate(numerator: int, denominator: int) -> RateMetric:
    return RateMetric(
        numerator=numerator,
        denominator=denominator,
        rate=None if denominator == 0 else numerator / denominator,
    )


def _values(values: Sequence[float]) -> ValueMetric:
    if not values:
        return ValueMetric(
            samples=[],
            sample_count=0,
            total=None,
            mean=None,
            minimum=None,
            maximum=None,
            p50=None,
            p95=None,
        )
    ordered = sorted(values)

    def nearest_rank(percentile: float) -> float:
        return ordered[math.ceil(percentile * len(ordered)) - 1]

    total = sum(values)
    return ValueMetric(
        samples=ordered,
        sample_count=len(values),
        total=total,
        mean=total / len(values),
        minimum=ordered[0],
        maximum=ordered[-1],
        p50=nearest_rank(0.50),
        p95=nearest_rank(0.95),
    )


def _threshold(
    observed: float | None,
    operator: Literal[">=", "<="],
    expected: float,
) -> ThresholdResult:
    passed = observed is not None and (
        observed >= expected if operator == ">=" else observed <= expected
    )
    return ThresholdResult(
        observed=observed,
        operator=operator,
        threshold=expected,
        passed=passed,
    )


def score_baseline(
    benchmark: ResolvedBenchmark,
    observations: ObservationBundle,
    adjudication: AdjudicationBundle,
) -> BaselineSummary:
    """Join frozen gold, recorded stages, and independent human labels."""
    validate_adjudication_bundle(benchmark, observations, adjudication)

    relevant_total = 0
    relevant_retrieved = 0
    answerer_total = 0
    answerer_unsupported = 0
    verifier_supported_total = 0
    verifier_supported_true = 0
    released_total = 0
    released_unsupported = 0
    citation_total = 0
    valid_citations = 0
    released_with_supporting_citation = 0
    expected_abstentions = 0
    correct_abstentions = 0
    answerable = 0
    false_abstentions = 0
    answered = 0
    coverage_counts: Counter[str] = Counter()
    stage_latency: dict[str, list[float]] = {stage: [] for stage in _STAGES}
    total_latency: list[float] = []

    benchmark_cases = {
        case.case_id: case for case in benchmark.definition.cases
    }
    label_cases = {case.case_id: case for case in adjudication.cases}

    for observation in observations.cases:
        benchmark_case = benchmark_cases[observation.case_id]
        labels = label_cases[observation.case_id]
        claim_support = {
            label.claim_id: label.corpus_supported for label in labels.claims
        }
        citation_support = {
            label.citation_id: set(label.supported_claim_ids)
            for label in labels.released_citations
        }

        gold_spans = benchmark.gold_spans_by_case[observation.case_id]
        relevant_total += len(gold_spans)
        for gold in gold_spans:
            retrieved = any(
                item.document_id == gold.document_id
                and item.source_sha256 == gold.source_sha256
                and item.start_char <= gold.source_start_char
                and item.end_char >= gold.source_end_char
                for item in observation.retrieval
            )
            relevant_retrieved += int(retrieved)

        answerer_total += len(claim_support)
        answerer_unsupported += sum(not supported for supported in claim_support.values())
        effective_by_claim = (
            {
                check.claim_id: check.effective_verdict
                for check in observation.verifier.checks
                if check.claim_id in claim_support
            }
            if observation.verifier.protocol_valid is True
            else {}
        )
        predicted_supported = {
            claim_id
            for claim_id, verdict in effective_by_claim.items()
            if verdict == "SUPPORTED"
        }
        truly_supported = {
            claim_id for claim_id, supported in claim_support.items() if supported
        }
        verifier_supported_total += len(predicted_supported)
        verifier_supported_true += len(predicted_supported & truly_supported)

        released_ids = set(observation.response.selected_claim_ids)
        released_total += len(released_ids)
        released_unsupported += sum(not claim_support[claim_id] for claim_id in released_ids)
        for citation in observation.response.citations:
            linked_claim_ids = {
                released.claim_id
                for released in observation.response.released_claims
                if citation.citation_id in released.citation_ids
            }
            citation_total += len(linked_claim_ids)
            valid_citations += len(
                linked_claim_ids & citation_support[citation.citation_id]
            )
        for released in observation.response.released_claims:
            released_with_supporting_citation += int(any(
                released.claim_id in citation_support[citation_id]
                for citation_id in released.citation_ids
            ))

        if benchmark_case.must_abstain:
            expected_abstentions += 1
            correct_abstentions += int(observation.response.status == "ABSTAIN")
        else:
            answerable += 1
            is_abstain = observation.response.status == "ABSTAIN"
            false_abstentions += int(is_abstain)
            answered += int(not is_abstain)

        coverage_counts.update(observation.response.coverage_reasons)
        stage_latency["retrieval"].append(observation.latency.retrieval_ms)
        if observation.latency.answerer_ms is not None:
            stage_latency["answerer"].append(observation.latency.answerer_ms)
        if observation.latency.verifier_ms is not None:
            stage_latency["verifier"].append(observation.latency.verifier_ms)
        if observation.latency.finalizer_ms is not None:
            stage_latency["finalizer"].append(observation.latency.finalizer_ms)
        total_latency.append(observation.latency.total_ms)

    retrieval_recall = _rate(relevant_retrieved, relevant_total)
    unsupported_rates = UnsupportedClaimRates(
        answerer=_rate(answerer_unsupported, answerer_total),
        verifier_supported=_rate(
            verifier_supported_total - verifier_supported_true,
            verifier_supported_total,
        ),
        released=_rate(released_unsupported, released_total),
    )
    verifier_precision = _rate(verifier_supported_true, verifier_supported_total)
    total_human_supported = sum(
        label.corpus_supported
        for case in adjudication.cases
        for label in case.claims
    )
    verifier_recall = _rate(verifier_supported_true, total_human_supported)
    citation_validity = _rate(valid_citations, citation_total)
    citation_completeness = _rate(released_with_supporting_citation, released_total)
    correct_abstention = _rate(correct_abstentions, expected_abstentions)
    false_abstention = _rate(false_abstentions, answerable)
    answer_coverage = _rate(answered, answerable)

    thresholds = benchmark.definition.thresholds
    threshold_results = {
        "retrieval_recall_at_k_min": _threshold(
            retrieval_recall.rate, ">=", thresholds.retrieval_recall_at_k_min
        ),
        "unsupported_released_claim_rate_max": _threshold(
            unsupported_rates.released.rate,
            "<=",
            thresholds.unsupported_released_claim_rate_max,
        ),
        "citation_validity_min": _threshold(
            citation_validity.rate, ">=", thresholds.citation_validity_min
        ),
        "citation_completeness_min": _threshold(
            citation_completeness.rate,
            ">=",
            thresholds.citation_completeness_min,
        ),
        "correct_abstention_rate_min": _threshold(
            correct_abstention.rate,
            ">=",
            thresholds.correct_abstention_rate_min,
        ),
        "false_abstention_rate_max": _threshold(
            false_abstention.rate,
            "<=",
            thresholds.false_abstention_rate_max,
        ),
        "answer_coverage_min": _threshold(
            answer_coverage.rate, ">=", thresholds.answer_coverage_min
        ),
        "verifier_supported_precision_min": _threshold(
            verifier_precision.rate,
            ">=",
            thresholds.verifier_supported_precision_min,
        ),
        "verifier_supported_recall_min": _threshold(
            verifier_recall.rate,
            ">=",
            thresholds.verifier_supported_recall_min,
        ),
    }
    return BaselineSummary(
        benchmark_id=benchmark.definition.benchmark_id,
        benchmark_sha256=benchmark.raw_sha256,
        observation_run_id=observations.run_id,
        observation_sha256=canonical_model_sha256(observations),
        adjudication_sha256=canonical_model_sha256(adjudication),
        implementation_aggregate_sha256=observations.implementation.aggregate_sha256,
        execution_identity_sha256_by_case={
            case.case_id: canonical_model_sha256(case.execution_identity)
            for case in observations.cases
        },
        case_count=len(observations.cases),
        retrieval_k=benchmark.definition.top_k,
        retrieval_recall_at_k=retrieval_recall,
        unsupported_claim_rates=unsupported_rates,
        verifier_supported_precision=verifier_precision,
        verifier_supported_recall=verifier_recall,
        citation_validity=citation_validity,
        citation_completeness=citation_completeness,
        correct_abstention_rate=correct_abstention,
        false_abstention_rate=false_abstention,
        answer_coverage=answer_coverage,
        coverage_reason_counts={
            reason: coverage_counts[reason] for reason in _COVERAGE_REASONS
        },
        latency=LatencySummary(
            retrieval_ms=_values(stage_latency["retrieval"]),
            answerer_ms=_values(stage_latency["answerer"]),
            verifier_ms=_values(stage_latency["verifier"]),
            finalizer_ms=_values(stage_latency["finalizer"]),
            total_ms=_values(total_latency),
        ),
        threshold_contract=thresholds,
        threshold_results=threshold_results,
        thresholds_passed=all(result.passed for result in threshold_results.values()),
        result=(
            "PASS"
            if all(result.passed for result in threshold_results.values())
            else "FAIL"
        ),
    )


def validate_baseline_summary(
    benchmark: ResolvedBenchmark,
    observations: ObservationBundle,
    adjudication: AdjudicationBundle,
    summary: BaselineSummary,
    expected_summary_sha256: str,
) -> None:
    """Recompute a published summary from its exact bound private inputs."""
    if not _SHA256_RE.fullmatch(expected_summary_sha256):
        raise EvaluationDataError("expected summary SHA-256 is malformed")
    actual_sha256 = canonical_model_sha256(summary)
    if actual_sha256 != expected_summary_sha256:
        raise EvaluationDataError("summary raw bytes do not match expected SHA-256")
    expected = score_baseline(benchmark, observations, adjudication)
    if canonical_artifact_bytes(summary) != canonical_artifact_bytes(expected):
        raise EvaluationDataError(
            "summary metrics are not derived from bound observations and adjudication"
        )


def _read_hash_bound_artifact(
    path: Path,
    expected_sha256: str,
    *,
    label: str,
) -> bytes:
    if not _SHA256_RE.fullmatch(expected_sha256):
        raise EvaluationDataError(f"expected {label} SHA-256 is malformed")
    raw = _read_regular_file(path, label=label)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise EvaluationDataError(f"{label} raw-byte SHA-256 does not match")
    return raw


def load_observations(
    path: Path,
    expected_sha256: str,
) -> ObservationBundle:
    """Load only the exact observation bytes whose digest record retained."""
    raw = _read_hash_bound_artifact(
        path,
        expected_sha256,
        label="observations",
    )
    try:
        return ObservationBundle.model_validate_json(raw)
    except ValidationError as exc:
        raise EvaluationDataError("observations do not match their closed schema") from exc


def load_adjudication(
    path: Path,
    expected_sha256: str,
) -> AdjudicationBundle:
    """Load only the exact independently owned adjudication bytes."""
    raw = _read_hash_bound_artifact(
        path,
        expected_sha256,
        label="adjudication",
    )
    try:
        return AdjudicationBundle.model_validate_json(raw)
    except ValidationError as exc:
        raise EvaluationDataError("adjudication does not match its closed schema") from exc


def load_summary(path: Path, expected_sha256: str) -> BaselineSummary:
    raw = _read_hash_bound_artifact(
        path,
        expected_sha256,
        label="summary",
    )
    try:
        return BaselineSummary.model_validate_json(raw)
    except ValidationError as exc:
        raise EvaluationDataError("summary does not match its closed schema") from exc


def _private_span_from_artifact(value: Any) -> PrivateSpanObservation:
    resolved = value.resolved
    return PrivateSpanObservation(
        evidence_id=value.evidence_id,
        quote=value.quote,
        valid=value.valid,
        document_id=None if resolved is None else resolved.document_id,
        document_revision_id=(
            None if resolved is None else resolved.document_revision_id
        ),
        title=None if resolved is None else resolved.title,
        relative_path=None if resolved is None else resolved.relative_path,
        source_sha256=None if resolved is None else resolved.source_sha256,
        source_start_char=None if resolved is None else resolved.source_start_char,
        source_end_char=None if resolved is None else resolved.source_end_char,
        start_line=None if resolved is None else resolved.start_line,
        end_line=None if resolved is None else resolved.end_line,
    )


def case_observation_from_artifacts(
    case_id: str,
    artifacts: Any,
    *,
    provider_invocations: Sequence[Any] | None = None,
) -> CaseObservation:
    """Project private engine artifacts into the closed observation schema."""
    try:
        retrieval = [
            RetrievedEvidenceObservation(**item.model_dump(mode="json"))
            for item in artifacts.retrieval
        ]
        answerer = AnswererStageObservation(
            called=artifacts.answerer.called,
            protocol_valid=artifacts.answerer.protocol_valid,
            answerability=artifacts.answerer.answerability,
            claims=[
                DraftClaimObservation(
                    claim_id=claim.claim_id,
                    text=claim.text,
                    render_safe=claim.render_safe,
                    proposed_citations=[
                        _private_span_from_artifact(citation)
                        for citation in claim.proposed_citations
                    ],
                )
                for claim in artifacts.answerer.claims
            ],
        )
        verifier = VerifierStageObservation(
            called=artifacts.verifier.called,
            protocol_valid=artifacts.verifier.protocol_valid,
            checks=[
                VerifierCheckObservation(
                    claim_id=check.claim_id,
                    raw_verdict=check.raw_verdict,
                    effective_verdict=check.effective_verdict,
                    support_spans=[
                        _private_span_from_artifact(span)
                        for span in check.support_spans
                    ],
                )
                for check in artifacts.verifier.checks
            ],
            eligible_claim_ids=list(artifacts.verifier.eligible_claim_ids),
        )
        finalizer = FinalizerStageObservation(
            called=artifacts.finalizer.called,
            protocol_valid=artifacts.finalizer.protocol_valid,
            decision=artifacts.finalizer.decision,
            requested_claim_ids=list(artifacts.finalizer.requested_claim_ids),
            accepted_claim_ids=list(artifacts.finalizer.accepted_claim_ids),
        )

        response_citations = [
            ResponseCitationObservation(
                citation_id=f"R{index}",
                **citation.model_dump(mode="json"),
            )
            for index, citation in enumerate(artifacts.response.citations, start=1)
        ]
        citation_id_by_span = {
            (
                citation.evidence_id,
                citation.source_start_char,
                citation.source_end_char,
            ): citation.citation_id
            for citation in response_citations
        }
        checks_by_id = {
            check.claim_id: check for check in artifacts.verifier.checks
        }
        released_claims: list[ReleasedClaimObservation] = []
        for claim_id in artifacts.response.selected_claim_ids:
            check = checks_by_id.get(claim_id)
            if check is None:
                raise EvaluationDataError(
                    "released claim is absent from captured verifier checks"
                )
            citation_ids: list[str] = []
            for span in check.support_spans:
                if not span.valid or span.resolved is None:
                    continue
                key = (
                    span.resolved.evidence_id,
                    span.resolved.source_start_char,
                    span.resolved.source_end_char,
                )
                citation_id = citation_id_by_span.get(key)
                if citation_id is None:
                    raise EvaluationDataError(
                        "released verifier span is absent from response citations"
                    )
                if citation_id not in citation_ids:
                    citation_ids.append(citation_id)
            released_claims.append(ReleasedClaimObservation(
                claim_id=claim_id,
                citation_ids=citation_ids,
            ))

        response = FinalResponseObservation(
            status=artifacts.response.status,
            reason_code=artifacts.response.reason_code,
            packet_id=artifacts.packet_id,
            corpus_revision=artifacts.corpus_revision,
            retrieval_count=len(artifacts.retrieval),
            retrieval_truncated=artifacts.retrieval_truncated,
            coverage_limited=artifacts.coverage_limited,
            coverage_reasons=list(artifacts.coverage_reasons),
            stages_completed=list(artifacts.response.stages_completed),
            draft_claim_count=artifacts.response.draft_claim_count,
            supported_claim_count=artifacts.response.supported_claim_count,
            contradicted_claim_count=artifacts.response.contradicted_claim_count,
            conflict_claim_count=artifacts.response.conflict_claim_count,
            insufficient_claim_count=artifacts.response.insufficient_claim_count,
            selected_claim_ids=list(artifacts.response.selected_claim_ids),
            released_claims=released_claims,
            citations=response_citations,
        )
        return CaseObservation(
            case_id=case_id,
            packet_id=artifacts.packet_id,
            corpus_revision=artifacts.corpus_revision,
            retrieval_k=artifacts.retrieval_k,
            execution_identity=ExecutionIdentityObservation(
                response_run_id=artifacts.response.run_id,
                contract_version=artifacts.contract_version,
                packet_schema_version=artifacts.packet_schema_version,
                retrieval_version=artifacts.retrieval_version,
                chunker_version=artifacts.chunker_version,
                prompt_versions=PromptVersionsObservation.model_validate(
                    artifacts.prompt_versions.model_dump(mode="json")
                ),
                stage_fingerprints=StageFingerprintsObservation.model_validate(
                    artifacts.stage_fingerprints.model_dump(mode="json")
                ),
                draft_hash=artifacts.draft_hash,
                verification_hash=artifacts.verification_hash,
                receipt_sha256=artifacts.response.receipt_sha256,
                receipt=ProductionReceiptObservation.model_validate(
                    artifacts.receipt.model_dump(mode="json")
                ),
            ),
            retrieval=retrieval,
            answerer=answerer,
            verifier=verifier,
            finalizer=finalizer,
            response=response,
            latency=StageLatencyObservation(
                retrieval_ms=artifacts.latency_ms.retrieval,
                answerer_ms=artifacts.latency_ms.answerer,
                verifier_ms=artifacts.latency_ms.verifier,
                finalizer_ms=artifacts.latency_ms.finalizer,
                total_ms=artifacts.latency_ms.total,
                capture_sha256=artifacts.latency_capture_sha256,
            ),
            provider_observation_mode=(
                "CONFIGURED_UNOBSERVED"
                if provider_invocations is None
                else "AGY_SDK"
            ),
            stage_invocations=[
                StageInvocationObservation.model_validate(
                    invocation.model_dump(mode="json")
                    if isinstance(invocation, BaseModel)
                    else invocation
                )
                for invocation in artifacts.stage_invocations
            ],
            provider_invocations=[
                ProviderInvocationObservation.model_validate(
                    invocation.model_dump(mode="json")
                    if isinstance(invocation, BaseModel)
                    else invocation
                )
                for invocation in (provider_invocations or ())
            ],
        )
    except EvaluationDataError:
        raise
    except (AttributeError, TypeError, ValidationError, ValueError) as exc:
        raise EvaluationDataError(
            "recorded artifacts do not match the closed observation contract"
        ) from exc


async def record_baseline(
    benchmark: ResolvedBenchmark,
    *,
    recorded_runner: Any | None = None,
    external_cost_limit_usd: float | None = None,
    provider_observation_mode: Literal[
        "CONFIGURED_UNOBSERVED",
        "AGY_SDK",
    ] = "CONFIGURED_UNOBSERVED",
) -> ObservationBundle:
    """Execute each frozen case once against a fresh temporary corpus.

    The benchmark must already have passed :func:`load_benchmark`; only then is
    the provider-aware grounded module imported. Provider and interruption
    failures retain the completed prefix and return a closed INCOMPLETE bundle.
    """
    policy_limit = (
        benchmark.definition.execution_policy.external_authorization_maximum_cost_usd
    )
    if (
        external_cost_limit_usd is None
        or not math.isfinite(external_cost_limit_usd)
        or external_cost_limit_usd < 0
        or external_cost_limit_usd > policy_limit
    ):
        raise EvaluationDataError(
            "recording requires an externally enforced cost limit within benchmark policy"
        )
    frozen_implementation = implementation_binding()

    from pipeline.grounded_rag import GroundedQueryRequest, run_grounded_rag_recorded
    from pipeline.knowledge_store import KnowledgeStore

    runner = recorded_runner or run_grounded_rag_recorded
    run_id = uuid.uuid4().hex
    completed: list[CaseObservation] = []
    failed_case_id: str | None = None
    failure_kind: Literal["PROVIDER_OR_PIPELINE_ERROR", "INTERRUPTED"] | None = None

    with tempfile.TemporaryDirectory(prefix="grounded-rag-baseline-v1-") as root:
        store = KnowledgeStore(Path(root) / "knowledge")
        for document in sorted(
            benchmark.definition.documents,
            key=lambda item: item.document_id,
        ):
            record = store.upsert_document(
                document.document_id,
                document.folder,
                document.title,
                document.content,
            )
            if (
                record.source_sha256
                != benchmark.document_source_sha256[document.document_id]
            ):
                raise EvaluationDataError(
                    "temporary corpus source hash does not match benchmark"
                )

        current_case_id: str | None = None
        try:
            async with asyncio.timeout(
                benchmark.definition.execution_policy.maximum_wall_time_seconds
            ):
                for benchmark_case in benchmark.definition.cases:
                    current_case_id = benchmark_case.case_id
                    runner_result = await runner(
                        GroundedQueryRequest(
                            prompt=benchmark_case.query,
                            top_k=benchmark.definition.top_k,
                        ),
                        store=store,
                    )
                    if not isinstance(runner_result, tuple) or len(runner_result) not in {
                        2,
                        3,
                    }:
                        raise EvaluationDataError(
                            "recorded runner returned an invalid closed result"
                        )
                    _response, artifacts = runner_result[:2]
                    provider_invocations = (
                        None if len(runner_result) == 2 else runner_result[2]
                    )
                    if provider_invocations is not None and not isinstance(
                        provider_invocations,
                        (list, tuple),
                    ):
                        raise EvaluationDataError(
                            "recorded provider receipts are not an ordered sequence"
                        )
                    completed.append(case_observation_from_artifacts(
                        benchmark_case.case_id,
                        artifacts,
                        provider_invocations=provider_invocations,
                    ))
        except (asyncio.CancelledError, TimeoutError, KeyboardInterrupt):
            failed_case_id = current_case_id
            failure_kind = "INTERRUPTED"
        except Exception:
            # Error text is intentionally excluded: provider exceptions can
            # contain prompts, endpoints, headers, or raw response fragments.
            failed_case_id = current_case_id
            failure_kind = "PROVIDER_OR_PIPELINE_ERROR"

    if failure_kind is not None:
        if implementation_binding() != frozen_implementation:
            raise EvaluationDataError("implementation changed during baseline recording")
        incomplete_bundle = ObservationBundle(
            benchmark_id=benchmark.definition.benchmark_id,
            benchmark_sha256=benchmark.raw_sha256,
            implementation=frozen_implementation,
            run_id=run_id,
            provider_observation_mode=provider_observation_mode,
            status="INCOMPLETE",
            cases=completed,
            failure=ObservationFailure(
                kind=failure_kind,
                failed_case_id=failed_case_id,
            ),
        )
        validate_observation_bundle(benchmark, incomplete_bundle)
        return incomplete_bundle
    bundle = ObservationBundle(
        benchmark_id=benchmark.definition.benchmark_id,
        benchmark_sha256=benchmark.raw_sha256,
        implementation=frozen_implementation,
        run_id=run_id,
        provider_observation_mode=provider_observation_mode,
        status="COMPLETE",
        cases=completed,
    )
    if implementation_binding() != frozen_implementation:
        raise EvaluationDataError("implementation changed during baseline recording")
    validate_observation_bundle(benchmark, bundle)
    return bundle


def publish_private_json_directory(
    destination: Path,
    filename: str,
    value: BaseModel,
) -> Path:
    """Publish one private JSON artifact without replacing an existing path.

    The destination directory is created with the filesystem's exclusive mkdir
    primitive. The final file name then appears through an atomic no-clobber
    hard link from a fully written private temporary file in that directory.
    A pathname can change after any final inode check, so callers must retain
    and verify the artifact digest before consuming the returned path.
    """
    if not filename or filename != Path(filename).name or not filename.endswith(".json"):
        raise EvaluationDataError("output filename is unsafe")
    destination = destination.absolute()
    parent = destination.parent
    if destination.name in {"", ".", ".."}:
        raise EvaluationDataError("output directory name is unsafe")
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    parent_fd = -1
    destination_fd = -1
    confirmed_destination_fd = -1
    output_fd = -1
    output_identity: tuple[int, int] | None = None
    temporary_name = f".{filename}.{uuid.uuid4().hex}.tmp"
    try:
        parent_flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            parent_flags |= os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            parent_flags |= os.O_NOFOLLOW
        parent_fd = os.open(parent, parent_flags)
        if not stat.S_ISDIR(os.fstat(parent_fd).st_mode):
            raise EvaluationDataError("output parent must be a non-symlink directory")
        os.mkdir(destination.name, 0o700, dir_fd=parent_fd)
        created_destination = os.stat(
            destination.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if not stat.S_ISDIR(created_destination.st_mode):
            raise EvaluationDataError("created output path is not a directory")
        destination_fd = os.open(destination.name, parent_flags, dir_fd=parent_fd)
        opened_destination = os.fstat(destination_fd)
        if (
            opened_destination.st_dev,
            opened_destination.st_ino,
        ) != (
            created_destination.st_dev,
            created_destination.st_ino,
        ):
            raise EvaluationDataError("output directory changed before publication")

        output_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_CLOEXEC"):
            output_flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            output_flags |= os.O_NOFOLLOW
        output_fd = os.open(
            temporary_name,
            output_flags,
            0o600,
            dir_fd=destination_fd,
        )
        payload = canonical_artifact_bytes(value)
        with os.fdopen(output_fd, "wb") as handle:
            output_fd = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            written_output = os.fstat(handle.fileno())
            if not stat.S_ISREG(written_output.st_mode):
                raise EvaluationDataError("private output staging file is not regular")
            output_identity = (written_output.st_dev, written_output.st_ino)
        staged_output = os.stat(
            temporary_name,
            dir_fd=destination_fd,
            follow_symlinks=False,
        )
        if output_identity != (staged_output.st_dev, staged_output.st_ino):
            raise EvaluationDataError("private output staging file was replaced")
        os.link(
            temporary_name,
            filename,
            src_dir_fd=destination_fd,
            dst_dir_fd=destination_fd,
            follow_symlinks=False,
        )
        linked_output = os.stat(
            filename,
            dir_fd=destination_fd,
            follow_symlinks=False,
        )
        if output_identity != (linked_output.st_dev, linked_output.st_ino):
            raise EvaluationDataError("published output does not match staged bytes")
        os.unlink(temporary_name, dir_fd=destination_fd)
        os.fsync(destination_fd)
        os.fsync(parent_fd)
        confirmed_destination_fd = os.open(
            destination.name,
            parent_flags,
            dir_fd=parent_fd,
        )
        named_destination = os.fstat(confirmed_destination_fd)
        if (
            named_destination.st_dev,
            named_destination.st_ino,
        ) != (
            created_destination.st_dev,
            created_destination.st_ino,
        ):
            raise EvaluationDataError("output directory changed during publication")
        final_output = os.stat(
            filename,
            dir_fd=confirmed_destination_fd,
            follow_symlinks=False,
        )
        if output_identity != (final_output.st_dev, final_output.st_ino):
            raise EvaluationDataError("published output changed before confirmation")
        final_named_destination = os.stat(
            destination.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if (
            final_named_destination.st_dev,
            final_named_destination.st_ino,
        ) != (
            created_destination.st_dev,
            created_destination.st_ino,
        ):
            raise EvaluationDataError("output directory changed before confirmation")
        return destination / filename
    except FileExistsError as exc:
        raise EvaluationDataError("output directory already exists") from exc
    except EvaluationDataError:
        raise
    except OSError as exc:
        raise EvaluationDataError("private output could not be published") from exc
    finally:
        if output_fd >= 0:
            os.close(output_fd)
        if destination_fd >= 0:
            os.close(destination_fd)
        if confirmed_destination_fd >= 0:
            os.close(confirmed_destination_fd)
        if parent_fd >= 0:
            os.close(parent_fd)


def model_from_mapping(model: type[BaseModel], value: Any) -> BaseModel:
    """Validate a capture fragment while avoiding payload-bearing exceptions."""
    try:
        if isinstance(value, BaseModel):
            value = value.model_dump(mode="json")
        elif not isinstance(value, Mapping):
            value = vars(value)
        return model.model_validate(value)
    except (TypeError, ValidationError) as exc:
        raise EvaluationDataError("recorded capture does not match evaluation schema") from exc
