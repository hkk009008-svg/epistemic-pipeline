"""Evaluation-only Google Antigravity SDK structured-call adapter.

This module is intentionally absent from the production provider registry.  It
is imported only by the measured-baseline CLI after benchmark, authorization,
and external-cost gates pass.  Importing it never imports the optional SDK and
never starts a process or network call.
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
import os
import shutil
import signal
import stat
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator


SDK_DISTRIBUTION = "google-antigravity"
SDK_VERSION = "0.1.10"
WORKER_PROTOCOL_VERSION = "agy-evaluation-worker-v1"
PROVIDER_NAME = "google-antigravity-sdk"
MAX_REQUEST_BYTES = 256 * 1024
MAX_STDOUT_BYTES = 256 * 1024
MAX_STDERR_BYTES = 32 * 1024
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_STAGES = ("gpt1", "gpt2", "gpt3")
_INVOCATION_POLICY = {
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


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


INVOCATION_POLICY_SHA256 = _sha256(_canonical_bytes(_INVOCATION_POLICY))


class AgyEvaluationProviderError(RuntimeError):
    """Sanitized adapter failure that never includes provider-controlled text."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(f"AGY evaluation provider failed closed: {code}")


class AgyReportedUsage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["REPORTED"] = "REPORTED"
    input_tokens: int = Field(..., ge=0)
    output_tokens: int = Field(..., ge=0)
    total_tokens: int = Field(..., ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    service_tier: str | None = Field(default=None, min_length=1, max_length=80)

    @model_validator(mode="after")
    def totals_are_possible(self) -> AgyReportedUsage:
        if self.total_tokens < max(self.input_tokens, self.output_tokens):
            raise ValueError("reported total tokens are smaller than a component")
        return self


class AgyRuntimeIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sdk_distribution: Literal["google-antigravity"] = SDK_DISTRIBUTION
    sdk_version: Literal["0.1.10"] = SDK_VERSION
    sdk_artifact_sha256: str = Field(..., pattern=_SHA256_PATTERN)
    localharness_sha256: str = Field(..., pattern=_SHA256_PATTERN)


class AgyEvaluationConfig(BaseModel):
    """Closed live parameters; construction alone remains inert."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: Literal[True]
    requested_model: str = Field(..., min_length=1, max_length=160)
    api_key: SecretStr
    expected_sdk_artifact_sha256: str = Field(..., pattern=_SHA256_PATTERN)
    expected_localharness_sha256: str = Field(..., pattern=_SHA256_PATTERN)
    call_timeout_seconds: float = Field(
        default=120.0,
        gt=0,
        le=300,
        allow_inf_nan=False,
    )

    @model_validator(mode="after")
    def secret_and_model_are_well_formed(self) -> AgyEvaluationConfig:
        if not self.api_key.get_secret_value():
            raise ValueError("AGY SDK evaluation requires an explicit API key")
        if self.requested_model != self.requested_model.strip() or any(
            ord(character) < 0x20 for character in self.requested_model
        ):
            raise ValueError("requested model is malformed")
        return self


class AgyInvocationReceipt(BaseModel):
    """Content-free receipt for one successful logical structured call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["grounded-provider-invocation-v1"] = (
        "grounded-provider-invocation-v1"
    )
    stage: Literal["gpt1", "gpt2", "gpt3"]
    provider: Literal["google-antigravity-sdk"] = PROVIDER_NAME
    requested_model: str = Field(..., min_length=1, max_length=160)
    reported_model: str | None = Field(default=None, min_length=1, max_length=160)
    model_attestation: Literal["REQUESTED_ONLY", "PROVIDER_REPORTED"]
    sdk_distribution: Literal["google-antigravity"] = SDK_DISTRIBUTION
    sdk_version: Literal["0.1.10"] = SDK_VERSION
    sdk_artifact_sha256: str = Field(..., pattern=_SHA256_PATTERN)
    localharness_sha256: str = Field(..., pattern=_SHA256_PATTERN)
    worker_protocol_version: Literal["agy-evaluation-worker-v1"] = (
        WORKER_PROTOCOL_VERSION
    )
    worker_source_sha256: str = Field(..., pattern=_SHA256_PATTERN)
    invocation_policy_sha256: str = Field(..., pattern=_SHA256_PATTERN)
    system_sha256: str = Field(..., pattern=_SHA256_PATTERN)
    user_sha256: str = Field(..., pattern=_SHA256_PATTERN)
    response_schema_sha256: str = Field(..., pattern=_SHA256_PATTERN)
    response_sha256: str = Field(..., pattern=_SHA256_PATTERN)
    duration_ms: float = Field(..., ge=0, allow_inf_nan=False)
    outcome: Literal["SUCCESS"] = "SUCCESS"
    logical_model_calls: Literal[1] = 1
    adapter_retries: Literal[0] = 0
    observed_tool_names: list[Literal["finish"]] = Field(
        default_factory=list,
        max_length=1,
    )
    usage: AgyReportedUsage | None = None
    cost_usd: None = None


class _WorkerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol_version: Literal["agy-evaluation-worker-v1"] = WORKER_PROTOCOL_VERSION
    stage: Literal["gpt1", "gpt2", "gpt3"]
    requested_model: str
    system: str
    user_content: str
    response_schema: dict[str, Any]
    expected_runtime: AgyRuntimeIdentity
    expected_worker_source_sha256: str = Field(..., pattern=_SHA256_PATTERN)
    expected_invocation_policy_sha256: str = Field(..., pattern=_SHA256_PATTERN)
    private_save_dir: str
    private_app_data_dir: str


class _WorkerSuccess(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol_version: Literal["agy-evaluation-worker-v1"]
    status: Literal["SUCCESS"]
    response: dict[str, Any]
    reported_model: str | None = Field(default=None, min_length=1, max_length=160)
    observed_tool_names: list[str] = Field(default_factory=list, max_length=4)
    usage: AgyReportedUsage | None = None
    runtime: AgyRuntimeIdentity
    worker_source_sha256: str = Field(..., pattern=_SHA256_PATTERN)
    invocation_policy_sha256: str = Field(..., pattern=_SHA256_PATTERN)


def _regular_file_bytes(path: Path, *, maximum_bytes: int = 512 * 1024 * 1024) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        raise AgyEvaluationProviderError("RUNTIME_FILE_UNAVAILABLE") from exc
    if not stat.S_ISREG(before.st_mode) or path.is_symlink():
        raise AgyEvaluationProviderError("RUNTIME_FILE_UNSAFE")
    if before.st_size < 0 or before.st_size > maximum_bytes:
        raise AgyEvaluationProviderError("RUNTIME_FILE_TOO_LARGE")
    try:
        with path.open("rb") as handle:
            after = os.fstat(handle.fileno())
            if (before.st_dev, before.st_ino, before.st_size) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
            ):
                raise AgyEvaluationProviderError("RUNTIME_FILE_CHANGED")
            data = handle.read(maximum_bytes + 1)
    except AgyEvaluationProviderError:
        raise
    except OSError as exc:
        raise AgyEvaluationProviderError("RUNTIME_FILE_UNAVAILABLE") from exc
    if len(data) != before.st_size or len(data) > maximum_bytes:
        raise AgyEvaluationProviderError("RUNTIME_FILE_CHANGED")
    return data


def _distribution_runtime_identity() -> AgyRuntimeIdentity:
    """Hash the exact installed SDK files without importing the SDK."""
    try:
        distribution = importlib.metadata.distribution(SDK_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError as exc:
        raise AgyEvaluationProviderError("SDK_NOT_INSTALLED") from exc
    if distribution.version != SDK_VERSION:
        raise AgyEvaluationProviderError("SDK_VERSION_MISMATCH")
    files = sorted(distribution.files or (), key=lambda item: str(item))
    if not files:
        raise AgyEvaluationProviderError("SDK_ARTIFACT_EMPTY")
    aggregate = hashlib.sha256()
    localharness: list[tuple[Path, bytes]] = []
    for relative in files:
        relative_text = str(relative).replace(os.sep, "/")
        path = Path(distribution.locate_file(relative)).resolve()
        data = _regular_file_bytes(path)
        aggregate.update(len(relative_text).to_bytes(4, "big"))
        aggregate.update(relative_text.encode("utf-8"))
        aggregate.update(len(data).to_bytes(8, "big"))
        aggregate.update(hashlib.sha256(data).digest())
        if path.name in {"localharness", "localharness.exe"}:
            localharness.append((path, data))
    if len(localharness) != 1:
        raise AgyEvaluationProviderError("LOCALHARNESS_AMBIGUOUS")
    harness_path, harness_bytes = localharness[0]
    mode = harness_path.stat().st_mode
    if mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise AgyEvaluationProviderError("LOCALHARNESS_WRITABLE")
    return AgyRuntimeIdentity(
        sdk_artifact_sha256=aggregate.hexdigest(),
        localharness_sha256=_sha256(harness_bytes),
    )


def inspect_runtime() -> AgyRuntimeIdentity:
    """Offline-only SDK identity inspection; it never imports or runs the SDK."""
    return _distribution_runtime_identity()


async def _read_bounded(
    stream: asyncio.StreamReader,
    limit: int,
) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await stream.read(min(64 * 1024, limit + 1 - total))
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)
        total += len(chunk)
        if total > limit:
            raise AgyEvaluationProviderError("WORKER_OUTPUT_LIMIT")


async def _terminate_process_group(process: asyncio.subprocess.Process) -> None:
    def group_exists() -> bool:
        try:
            os.killpg(process.pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        await asyncio.wait_for(process.wait(), timeout=0.5)
    except TimeoutError:
        pass
    for _ in range(5):
        if not group_exists():
            return
        await asyncio.sleep(0.02)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    await process.wait()
    for _ in range(25):
        if not group_exists():
            return
        await asyncio.sleep(0.02)
    raise AgyEvaluationProviderError("WORKER_GROUP_RETAINED")


class AgyEvaluationProvider:
    """One-call-at-a-time provider whose SDK work occurs only in a child."""

    def __init__(
        self,
        config: AgyEvaluationConfig,
        *,
        worker_path: Path | None = None,
        runtime_resolver: Callable[[], AgyRuntimeIdentity] | None = None,
    ):
        self._config = config
        self._worker_path = (
            Path(__file__).with_name("agy_evaluation_worker.py")
            if worker_path is None
            else worker_path
        ).resolve()
        self._runtime_resolver = runtime_resolver or _distribution_runtime_identity
        self._receipts: list[AgyInvocationReceipt] = []
        self._call_lock = asyncio.Lock()

    def recorded_execution(self):
        """Return the private injection object; no SDK inspection occurs here."""
        from pipeline.grounded_rag import (
            GroundedRecordedExecution,
            GroundedRecordedStageIdentity,
        )

        return GroundedRecordedExecution(
            stage_identities={
                stage: GroundedRecordedStageIdentity(
                    provider=PROVIDER_NAME,
                    model=self._config.requested_model,
                )
                for stage in _STAGES
            },
            caller=self.call,
        )

    def take_receipts(self) -> tuple[AgyInvocationReceipt, ...]:
        receipts = tuple(self._receipts)
        self._receipts.clear()
        return receipts

    def _verify_runtime(self) -> AgyRuntimeIdentity:
        runtime = self._runtime_resolver()
        if (
            runtime.sdk_artifact_sha256
            != self._config.expected_sdk_artifact_sha256
            or runtime.localharness_sha256
            != self._config.expected_localharness_sha256
        ):
            raise AgyEvaluationProviderError("SDK_RUNTIME_MISMATCH")
        return runtime

    async def call(
        self,
        stage: Literal["gpt1", "gpt2", "gpt3"],
        system: str,
        user_content: str,
        response_model: type[BaseModel],
    ) -> BaseModel:
        if stage not in _STAGES:
            raise AgyEvaluationProviderError("STAGE_INVALID")
        async with self._call_lock:
            return await self._call_once(stage, system, user_content, response_model)

    async def _call_once(
        self,
        stage: Literal["gpt1", "gpt2", "gpt3"],
        system: str,
        user_content: str,
        response_model: type[BaseModel],
    ) -> BaseModel:
        runtime = self._verify_runtime()
        worker_bytes = _regular_file_bytes(self._worker_path, maximum_bytes=2 * 1024 * 1024)
        worker_sha256 = _sha256(worker_bytes)
        response_schema = response_model.model_json_schema()
        started = time.perf_counter()
        scratch = Path(tempfile.mkdtemp(prefix="agy-evaluation-stage-"))
        os.chmod(scratch, 0o700)
        directories = {
            name: scratch / name for name in ("home", "cwd", "tmp", "app", "save")
        }
        for directory in directories.values():
            directory.mkdir(mode=0o700)
        request = _WorkerRequest(
            stage=stage,
            requested_model=self._config.requested_model,
            system=system,
            user_content=user_content,
            response_schema=response_schema,
            expected_runtime=runtime,
            expected_worker_source_sha256=worker_sha256,
            expected_invocation_policy_sha256=INVOCATION_POLICY_SHA256,
            private_save_dir=str(directories["save"]),
            private_app_data_dir=str(directories["app"]),
        )
        request_bytes = _canonical_bytes(request.model_dump(mode="json"))
        if len(request_bytes) > MAX_REQUEST_BYTES:
            shutil.rmtree(scratch)
            raise AgyEvaluationProviderError("WORKER_INPUT_LIMIT")
        process: asyncio.subprocess.Process | None = None
        stdout_task: asyncio.Task[bytes] | None = None
        stderr_task: asyncio.Task[bytes] | None = None
        wait_task: asyncio.Task[int] | None = None
        result: BaseModel | None = None
        receipt: AgyInvocationReceipt | None = None
        try:
            env = {
                "GEMINI_API_KEY": self._config.api_key.get_secret_value(),
                "HOME": str(directories["home"]),
                "TMPDIR": str(directories["tmp"]),
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
                "PYTHONNOUSERSITE": "1",
            }
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-I",
                str(self._worker_path),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=directories["cwd"],
                env=env,
                start_new_session=True,
            )
            if process.stdin is None or process.stdout is None or process.stderr is None:
                raise AgyEvaluationProviderError("WORKER_PIPE_SETUP")
            stdout_task = asyncio.create_task(
                _read_bounded(process.stdout, MAX_STDOUT_BYTES)
            )
            stderr_task = asyncio.create_task(
                _read_bounded(process.stderr, MAX_STDERR_BYTES)
            )
            wait_task = asyncio.create_task(process.wait())
            async with asyncio.timeout(self._config.call_timeout_seconds):
                # The bound covers request delivery too. A worker that never
                # reads stdin must not be able to block before the timeout.
                process.stdin.write(request_bytes)
                await process.stdin.drain()
                process.stdin.close()
                await process.stdin.wait_closed()
                stdout, stderr, returncode = await asyncio.gather(
                    stdout_task,
                    stderr_task,
                    wait_task,
                )
            if returncode != 0:
                raise AgyEvaluationProviderError("WORKER_NONZERO")
            if stderr:
                raise AgyEvaluationProviderError("WORKER_STDERR")
            if not stdout or stdout.strip() != stdout:
                raise AgyEvaluationProviderError("WORKER_OUTPUT_FRAMING")
            try:
                payload = json.loads(stdout.decode("utf-8", errors="strict"))
                output = _WorkerSuccess.model_validate(payload)
            except Exception as exc:
                raise AgyEvaluationProviderError("WORKER_OUTPUT_INVALID") from exc
            if (
                output.runtime != runtime
                or output.worker_source_sha256 != worker_sha256
                or output.invocation_policy_sha256 != INVOCATION_POLICY_SHA256
            ):
                raise AgyEvaluationProviderError("WORKER_IDENTITY_MISMATCH")
            if (
                self._verify_runtime() != runtime
                or _sha256(_regular_file_bytes(
                    self._worker_path,
                    maximum_bytes=2 * 1024 * 1024,
                ))
                != worker_sha256
            ):
                raise AgyEvaluationProviderError("RUNTIME_CHANGED_DURING_CALL")
            if any(name != "finish" for name in output.observed_tool_names):
                raise AgyEvaluationProviderError("WORKER_TOOL_VIOLATION")
            if output.reported_model is not None and (
                output.reported_model != self._config.requested_model
            ):
                raise AgyEvaluationProviderError("MODEL_MISMATCH")
            try:
                result = response_model.model_validate(output.response)
            except Exception as exc:
                raise AgyEvaluationProviderError("RESPONSE_SCHEMA_INVALID") from exc
            receipt = AgyInvocationReceipt(
                stage=stage,
                requested_model=self._config.requested_model,
                reported_model=output.reported_model,
                model_attestation=(
                    "PROVIDER_REPORTED"
                    if output.reported_model is not None
                    else "REQUESTED_ONLY"
                ),
                sdk_artifact_sha256=runtime.sdk_artifact_sha256,
                localharness_sha256=runtime.localharness_sha256,
                worker_source_sha256=worker_sha256,
                invocation_policy_sha256=INVOCATION_POLICY_SHA256,
                system_sha256=_sha256(system.encode("utf-8")),
                user_sha256=_sha256(user_content.encode("utf-8")),
                response_schema_sha256=_sha256(_canonical_bytes(response_schema)),
                response_sha256=_sha256(_canonical_bytes(
                    result.model_dump(mode="json")
                )),
                duration_ms=(time.perf_counter() - started) * 1000,
                observed_tool_names=output.observed_tool_names,
                usage=output.usage,
            )
        except TimeoutError as exc:
            raise AgyEvaluationProviderError("WORKER_TIMEOUT") from exc
        except asyncio.CancelledError:
            raise
        except AgyEvaluationProviderError:
            raise
        except (OSError, ValueError) as exc:
            raise AgyEvaluationProviderError("WORKER_SETUP") from exc
        finally:
            try:
                if process is not None:
                    await asyncio.shield(_terminate_process_group(process))
            finally:
                # Containment failure is itself fail-closed, but it must never
                # skip reader cleanup or private-state removal.
                pending_tasks = tuple(
                    task
                    for task in (stdout_task, stderr_task, wait_task)
                    if task is not None
                )
                for task in pending_tasks:
                    if not task.done():
                        task.cancel()
                if pending_tasks:
                    await asyncio.gather(*pending_tasks, return_exceptions=True)
                try:
                    shutil.rmtree(scratch)
                except OSError as exc:
                    raise AgyEvaluationProviderError("SCRATCH_CLEANUP") from exc
                if scratch.exists():
                    raise AgyEvaluationProviderError("SCRATCH_RETAINED")
        if result is None or receipt is None:  # pragma: no cover - construction guard
            raise AgyEvaluationProviderError("WORKER_RESULT_MISSING")
        self._receipts.append(receipt)
        return result


def main() -> int:
    """Print sanitized installed runtime identity without starting the SDK."""
    try:
        print(_canonical_bytes(inspect_runtime().model_dump(mode="json")).decode("utf-8"))
        return 0
    except AgyEvaluationProviderError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
