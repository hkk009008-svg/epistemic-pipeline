"""Private subprocess entry point for the evaluation-only Antigravity SDK.

The parent sends one closed request on stdin.  This worker emits exactly one
closed JSON object on success and no provider-controlled error text on failure.
It is not imported by the production application.
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
import logging
import os
import stat
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


SDK_DISTRIBUTION = "google-antigravity"
SDK_VERSION = "0.1.10"
PROTOCOL_VERSION = "agy-evaluation-worker-v1"
MAX_REQUEST_BYTES = 256 * 1024
MAX_RESPONSE_BYTES = 256 * 1024
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
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


class RuntimeIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sdk_distribution: Literal["google-antigravity"] = SDK_DISTRIBUTION
    sdk_version: Literal["0.1.10"] = SDK_VERSION
    sdk_artifact_sha256: str = Field(..., pattern=_SHA256_PATTERN)
    localharness_sha256: str = Field(..., pattern=_SHA256_PATTERN)


class WorkerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol_version: Literal["agy-evaluation-worker-v1"]
    stage: Literal["gpt1", "gpt2", "gpt3"]
    requested_model: str = Field(..., min_length=1, max_length=160)
    system: str = Field(..., min_length=1, max_length=100_000)
    user_content: str = Field(..., min_length=1, max_length=100_000)
    response_schema: dict[str, Any]
    expected_runtime: RuntimeIdentity
    expected_worker_source_sha256: str = Field(..., pattern=_SHA256_PATTERN)
    expected_invocation_policy_sha256: str = Field(..., pattern=_SHA256_PATTERN)
    private_save_dir: str = Field(..., min_length=1, max_length=2_000)
    private_app_data_dir: str = Field(..., min_length=1, max_length=2_000)


def _regular_bytes(path: Path, maximum_bytes: int = 512 * 1024 * 1024) -> bytes:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or path.is_symlink():
        raise RuntimeError("unsafe runtime file")
    if before.st_size < 0 or before.st_size > maximum_bytes:
        raise RuntimeError("runtime file size")
    with path.open("rb") as handle:
        after = os.fstat(handle.fileno())
        if (before.st_dev, before.st_ino, before.st_size) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
        ):
            raise RuntimeError("runtime file changed")
        data = handle.read(maximum_bytes + 1)
    if len(data) != before.st_size or len(data) > maximum_bytes:
        raise RuntimeError("runtime file changed")
    return data


def _runtime_identity() -> RuntimeIdentity:
    distribution = importlib.metadata.distribution(SDK_DISTRIBUTION)
    if distribution.version != SDK_VERSION:
        raise RuntimeError("SDK version")
    files = sorted(distribution.files or (), key=lambda item: str(item))
    if not files:
        raise RuntimeError("SDK files")
    aggregate = hashlib.sha256()
    harnesses: list[tuple[Path, bytes]] = []
    for relative in files:
        relative_text = str(relative).replace(os.sep, "/")
        path = Path(distribution.locate_file(relative)).resolve()
        data = _regular_bytes(path)
        aggregate.update(len(relative_text).to_bytes(4, "big"))
        aggregate.update(relative_text.encode("utf-8"))
        aggregate.update(len(data).to_bytes(8, "big"))
        aggregate.update(hashlib.sha256(data).digest())
        if path.name in {"localharness", "localharness.exe"}:
            harnesses.append((path, data))
    if len(harnesses) != 1:
        raise RuntimeError("localharness")
    harness_path, harness_bytes = harnesses[0]
    if harness_path.stat().st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise RuntimeError("localharness mode")
    return RuntimeIdentity(
        sdk_artifact_sha256=aggregate.hexdigest(),
        localharness_sha256=_sha256(harness_bytes),
    )


def _read_request() -> WorkerRequest:
    raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
    if not raw or len(raw) > MAX_REQUEST_BYTES:
        raise RuntimeError("input")
    value = json.loads(raw.decode("utf-8", errors="strict"))
    return WorkerRequest.model_validate(value)


def _private_directory(value: str) -> str:
    path = Path(value)
    before = path.lstat()
    if not path.is_absolute() or path.is_symlink() or not stat.S_ISDIR(before.st_mode):
        raise RuntimeError("private directory")
    if stat.S_IMODE(before.st_mode) & 0o077:
        raise RuntimeError("private directory mode")
    return str(path)


def _usage_payload(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    required = (
        usage.prompt_token_count,
        usage.candidates_token_count,
        usage.total_token_count,
    )
    if any(value is None for value in required):
        raise RuntimeError("usage")
    service_tier = usage.service_tier
    if service_tier is not None:
        service_tier = getattr(service_tier, "value", str(service_tier))
    return {
        "status": "REPORTED",
        "input_tokens": usage.prompt_token_count,
        "output_tokens": usage.candidates_token_count,
        "total_tokens": usage.total_token_count,
        "cached_input_tokens": usage.cached_content_token_count,
        "reasoning_tokens": usage.thoughts_token_count,
        "service_tier": service_tier,
    }


async def _run(request: WorkerRequest) -> dict[str, Any]:
    worker_sha256 = _sha256(_regular_bytes(Path(__file__).resolve(), 2 * 1024 * 1024))
    runtime = _runtime_identity()
    if (
        runtime != request.expected_runtime
        or worker_sha256 != request.expected_worker_source_sha256
        or INVOCATION_POLICY_SHA256
        != request.expected_invocation_policy_sha256
    ):
        raise RuntimeError("identity")

    # This is the only point where the optional SDK is imported. The parent
    # reaches it only after every external authorization and runtime gate.
    from google.antigravity import (  # type: ignore[import-not-found]
        Agent,
        BuiltinTools,
        CapabilitiesConfig,
        CustomSystemInstructions,
        LocalAgentConfig,
        ModelAPIRetryConfig,
        ModelOutputRetryConfig,
        RetryConfig,
    )

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("credential")
    config = LocalAgentConfig(
        system_instructions=CustomSystemInstructions(text=request.system),
        capabilities=CapabilitiesConfig(
            enabled_tools=[BuiltinTools.FINISH],
            enable_subagents=False,
        ),
        tools=[],
        policies=[],
        hooks=[],
        triggers=[],
        mcp_servers=[],
        subagents=[],
        workspaces=[],
        env={
            "HOME": os.environ["HOME"],
            "TMPDIR": os.environ["TMPDIR"],
            "LANG": os.environ["LANG"],
            "LC_ALL": os.environ["LC_ALL"],
        },
        save_dir=_private_directory(request.private_save_dir),
        app_data_dir=_private_directory(request.private_app_data_dir),
        response_schema=request.response_schema,
        skills_paths=[],
        retry_config=RetryConfig(
            api_retry=ModelAPIRetryConfig(max_retries=0),
            model_output_retry=ModelOutputRetryConfig(max_retries=0),
        ),
        model=request.requested_model,
        api_key=api_key,
    )
    async with Agent(config) as agent:
        response = await agent.chat(request.user_content)
        structured = await response.structured_output()
        tool_names: list[str] = []
        async for call in response.tool_calls:
            name = getattr(call.name, "value", str(call.name))
            if name not in tool_names:
                tool_names.append(name)
        usage = _usage_payload(response.usage_metadata)
    if structured is None or not isinstance(structured, dict):
        raise RuntimeError("structured output")
    if any(name != "finish" for name in tool_names):
        raise RuntimeError("tool violation")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": "SUCCESS",
        "response": structured,
        "reported_model": None,
        "observed_tool_names": tool_names,
        "usage": usage,
        "runtime": runtime.model_dump(mode="json"),
        "worker_source_sha256": worker_sha256,
        "invocation_policy_sha256": INVOCATION_POLICY_SHA256,
    }


def main() -> int:
    logging.disable(logging.CRITICAL)
    try:
        request = _read_request()
        output = asyncio.run(_run(request))
        raw = _canonical_bytes(output)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise RuntimeError("output")
        sys.stdout.buffer.write(raw)
        sys.stdout.buffer.flush()
        return 0
    except (Exception, KeyboardInterrupt, ValidationError):
        # No provider-controlled error text crosses this boundary.
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
