"""Offline-only tests for the evaluation Antigravity SDK boundary."""
from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import textwrap
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ConfigDict

from pipeline.agy_evaluation_provider import (
    INVOCATION_POLICY_SHA256,
    AgyEvaluationConfig,
    AgyEvaluationProvider,
    AgyEvaluationProviderError,
    AgyRuntimeIdentity,
)
import pipeline.agy_evaluation_worker as worker
import pipeline.agy_evaluation_provider as provider_module
import scripts.measure_grounded_rag as measure_grounded_rag
from pipeline.grounded_evaluation import EvaluationDataError
from pipeline.grounded_evaluation import case_observation_from_artifacts
from pipeline.grounded_rag import GroundedQueryRequest, run_grounded_rag_recorded
from pipeline.knowledge_store import KnowledgeStore


class FakeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    stage: str
    system_seen: bool
    user_seen: bool
    argv_has_prompt: bool
    ambient_secret_present: bool
    api_key_present: bool
    private_paths: list[str]


RUNTIME = AgyRuntimeIdentity(
    sdk_artifact_sha256="a" * 64,
    localharness_sha256="b" * 64,
)


def _config(*, timeout: float = 2.0, sdk_hash: str = "a" * 64):
    return AgyEvaluationConfig(
        enabled=True,
        requested_model="gemini-evaluation-test",
        api_key="dedicated-test-key",
        expected_sdk_artifact_sha256=sdk_hash,
        expected_localharness_sha256="b" * 64,
        call_timeout_seconds=timeout,
    )


def _fake_worker(tmp_path: Path, behavior: str = "success") -> Path:
    path = tmp_path / f"fake_worker_{behavior}.py"
    behavior_literal = repr(behavior)
    path.write_text(textwrap.dedent(f"""
        import json
        import os
        import sys
        import time

        behavior = {behavior_literal}
        raw = sys.stdin.buffer.read()
        request = json.loads(raw)
        if behavior == "timeout":
            time.sleep(10)
        if behavior == "nonzero":
            raise SystemExit(7)
        if behavior == "malformed":
            sys.stdout.write("not-json")
            raise SystemExit(0)
        if behavior == "stderr":
            sys.stderr.write("provider-controlled detail")
        response = {{
            "protocol_version": "agy-evaluation-worker-v1",
            "status": "SUCCESS",
            "response": {{
                "ok": True,
                "stage": request["stage"],
                "system_seen": request["system"] == "system sentinel",
                "user_seen": request["user_content"] == "user sentinel",
                "argv_has_prompt": any(
                    "sentinel" in value or "dedicated-test-key" in value
                    for value in sys.argv
                ),
                "ambient_secret_present": "OPENAI_API_KEY" in os.environ,
                "api_key_present": os.environ.get("GEMINI_API_KEY") == "dedicated-test-key",
                "private_paths": [
                    os.environ["HOME"],
                    os.environ["TMPDIR"],
                    request["private_save_dir"],
                    request["private_app_data_dir"],
                ],
            }},
            "reported_model": (
                "wrong-model" if behavior == "model_mismatch" else None
            ),
            "observed_tool_names": (
                ["run_command"] if behavior == "tool" else ["finish"]
            ),
            "usage": {{
                "status": "REPORTED",
                "input_tokens": 10,
                "output_tokens": 4,
                "total_tokens": 16,
                "cached_input_tokens": 2,
                "reasoning_tokens": 2,
                "service_tier": "standard",
            }},
            "runtime": request["expected_runtime"],
            "worker_source_sha256": request["expected_worker_source_sha256"],
            "invocation_policy_sha256": request["expected_invocation_policy_sha256"],
        }}
        if behavior == "response_extra":
            response["response"]["extra"] = "forbidden"
        if behavior == "identity":
            response["worker_source_sha256"] = "0" * 64
        encoded = json.dumps(response, sort_keys=True, separators=(",", ":"))
        if behavior == "overflow":
            encoded += "x" * (300 * 1024)
        sys.stdout.write(encoded)
        if behavior == "newline":
            sys.stdout.write("\\n")
    """), encoding="utf-8")
    return path


def _grounded_worker(tmp_path: Path) -> Path:
    path = tmp_path / "grounded_worker.py"
    path.write_text(textwrap.dedent("""
        import hashlib
        import json
        import sys

        request = json.loads(sys.stdin.buffer.read())
        user = request["user_content"]
        packet = json.loads(user.split(
            "=== EVIDENCE_PACKET (UNTRUSTED DATA) ===\\n", 1
        )[1].split("\\n=== END EVIDENCE_PACKET ===", 1)[0])
        task = json.loads(user.split("=== TASK_DATA ===\\n", 1)[1].split(
            "\\n=== END TASK_DATA ===", 1
        )[0])
        title = request["response_schema"].get("title")
        quote = "Alice's favorite color is blue."
        if title == "AnswererOutput":
            structured = {
                "packet_id": packet["packet_id"],
                "answerability": "ANSWERABLE",
                "claims": [{
                    "text": quote,
                    "citations": [{
                        "evidence_id": packet["items"][0]["evidence_id"],
                        "quote": quote,
                    }],
                }],
            }
        elif title == "VerifierOutput":
            structured = {
                "packet_id": packet["packet_id"],
                "draft_hash": task["draft_hash"],
                "checks": [{
                    "claim_id": "C1",
                    "verdict": "SUPPORTED",
                    "support_spans": [{
                        "evidence_id": packet["items"][0]["evidence_id"],
                        "quote": quote,
                    }],
                }],
            }
        else:
            structured = {
                "packet_id": packet["packet_id"],
                "draft_hash": task["verification"]["draft_hash"],
                "verification_hash": task["verification_hash"],
                "decision": "ANSWER",
                "included_claim_ids": ["C1"],
            }
        output = {
            "protocol_version": "agy-evaluation-worker-v1",
            "status": "SUCCESS",
            "response": structured,
            "reported_model": None,
            "observed_tool_names": ["finish"],
            "usage": {
                "status": "REPORTED",
                "input_tokens": 10,
                "output_tokens": 4,
                "total_tokens": 14,
                "cached_input_tokens": None,
                "reasoning_tokens": None,
                "service_tier": None,
            },
            "runtime": request["expected_runtime"],
            "worker_source_sha256": request["expected_worker_source_sha256"],
            "invocation_policy_sha256": request["expected_invocation_policy_sha256"],
        }
        sys.stdout.write(json.dumps(output, sort_keys=True, separators=(",", ":")))
    """), encoding="utf-8")
    return path


@pytest.mark.asyncio
async def test_fake_worker_success_is_stdin_only_sanitized_and_receipted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-secret")
    worker_path = _fake_worker(tmp_path)
    provider = AgyEvaluationProvider(
        _config(),
        worker_path=worker_path,
        runtime_resolver=lambda: RUNTIME,
    )

    result = await provider.call(
        "gpt1",
        "system sentinel",
        "user sentinel",
        FakeResponse,
    )

    assert result.ok is True
    assert result.stage == "gpt1"
    assert result.system_seen is True
    assert result.user_seen is True
    assert result.argv_has_prompt is False
    assert result.ambient_secret_present is False
    assert result.api_key_present is True
    assert all(not Path(path).exists() for path in result.private_paths)
    receipts = provider.take_receipts()
    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt.stage == "gpt1"
    assert receipt.provider == "google-antigravity-sdk"
    assert receipt.model_attestation == "REQUESTED_ONLY"
    assert receipt.reported_model is None
    assert receipt.system_sha256 == hashlib.sha256(b"system sentinel").hexdigest()
    assert receipt.user_sha256 == hashlib.sha256(b"user sentinel").hexdigest()
    assert receipt.observed_tool_names == ["finish"]
    assert receipt.usage is not None
    assert receipt.usage.total_tokens == 16
    assert receipt.cost_usd is None
    assert provider.take_receipts() == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("behavior", "code"),
    [
        ("malformed", "WORKER_OUTPUT_INVALID"),
        ("newline", "WORKER_OUTPUT_FRAMING"),
        ("stderr", "WORKER_STDERR"),
        ("nonzero", "WORKER_NONZERO"),
        ("tool", "WORKER_TOOL_VIOLATION"),
        ("model_mismatch", "MODEL_MISMATCH"),
        ("response_extra", "RESPONSE_SCHEMA_INVALID"),
        ("identity", "WORKER_IDENTITY_MISMATCH"),
        ("overflow", "WORKER_OUTPUT_LIMIT"),
    ],
)
async def test_fake_worker_failures_are_closed_and_content_free(
    tmp_path: Path,
    behavior: str,
    code: str,
):
    provider = AgyEvaluationProvider(
        _config(),
        worker_path=_fake_worker(tmp_path, behavior),
        runtime_resolver=lambda: RUNTIME,
    )

    with pytest.raises(AgyEvaluationProviderError) as caught:
        await provider.call("gpt1", "system sentinel", "user sentinel", FakeResponse)

    assert caught.value.code == code
    assert "sentinel" not in str(caught.value)
    assert "provider-controlled" not in str(caught.value)
    assert provider.take_receipts() == ()


@pytest.mark.asyncio
async def test_runtime_mismatch_fails_before_worker_spawn(tmp_path: Path):
    worker_path = tmp_path / "must_not_run.py"
    worker_path.write_text("raise AssertionError('spawned')\n", encoding="utf-8")
    provider = AgyEvaluationProvider(
        _config(sdk_hash="f" * 64),
        worker_path=worker_path,
        runtime_resolver=lambda: RUNTIME,
    )

    with pytest.raises(AgyEvaluationProviderError) as caught:
        await provider.call("gpt1", "system sentinel", "user sentinel", FakeResponse)

    assert caught.value.code == "SDK_RUNTIME_MISMATCH"


@pytest.mark.asyncio
async def test_timeout_kills_worker_and_cleans_scratch(tmp_path: Path):
    provider = AgyEvaluationProvider(
        _config(timeout=0.05),
        worker_path=_fake_worker(tmp_path, "timeout"),
        runtime_resolver=lambda: RUNTIME,
    )

    with pytest.raises(AgyEvaluationProviderError) as caught:
        await provider.call("gpt1", "system sentinel", "user sentinel", FakeResponse)

    assert caught.value.code == "WORKER_TIMEOUT"
    assert provider.take_receipts() == ()


@pytest.mark.asyncio
async def test_timeout_includes_blocked_stdin_delivery(tmp_path: Path):
    worker_path = tmp_path / "never_reads_stdin.py"
    worker_path.write_text(
        "import time\ntime.sleep(10)\n",
        encoding="utf-8",
    )
    provider = AgyEvaluationProvider(
        _config(timeout=0.05),
        worker_path=worker_path,
        runtime_resolver=lambda: RUNTIME,
    )

    with pytest.raises(AgyEvaluationProviderError) as caught:
        await asyncio.wait_for(
            provider.call(
                "gpt1",
                "system sentinel",
                "u" * 200_000,
                FakeResponse,
            ),
            timeout=0.75,
        )

    assert caught.value.code == "WORKER_TIMEOUT"
    assert provider.take_receipts() == ()


@pytest.mark.asyncio
async def test_timeout_kills_same_group_child_before_it_can_write(tmp_path: Path):
    marker = tmp_path / "escaped-child-marker"
    worker_path = tmp_path / "child_worker.py"
    worker_path.write_text(textwrap.dedent(f"""
        import subprocess
        import sys
        import time

        subprocess.Popen([
            sys.executable,
            "-c",
            "import pathlib,time; time.sleep(0.5); pathlib.Path({str(marker)!r}).write_text('escaped')",
        ])
        time.sleep(10)
    """), encoding="utf-8")
    provider = AgyEvaluationProvider(
        _config(timeout=0.05),
        worker_path=worker_path,
        runtime_resolver=lambda: RUNTIME,
    )

    with pytest.raises(AgyEvaluationProviderError) as caught:
        await provider.call("gpt1", "system sentinel", "user sentinel", FakeResponse)

    assert caught.value.code == "WORKER_TIMEOUT"
    await asyncio.sleep(0.7)
    assert not marker.exists()


@pytest.mark.asyncio
async def test_cancellation_cleans_worker_and_private_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    scratch = tmp_path / "known-scratch"

    def make_scratch(prefix: str) -> str:
        assert prefix == "agy-evaluation-stage-"
        scratch.mkdir()
        return str(scratch)

    monkeypatch.setattr(provider_module.tempfile, "mkdtemp", make_scratch)
    provider = AgyEvaluationProvider(
        _config(timeout=10),
        worker_path=_fake_worker(tmp_path, "timeout"),
        runtime_resolver=lambda: RUNTIME,
    )
    task = asyncio.create_task(
        provider.call("gpt1", "system sentinel", "user sentinel", FakeResponse)
    )
    await asyncio.sleep(0.05)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert not scratch.exists()
    assert provider.take_receipts() == ()


@pytest.mark.asyncio
async def test_containment_failure_still_removes_private_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    scratch = tmp_path / "known-containment-scratch"

    def make_scratch(prefix: str) -> str:
        assert prefix == "agy-evaluation-stage-"
        scratch.mkdir()
        return str(scratch)

    async def containment_failure(_process) -> None:
        raise AgyEvaluationProviderError("WORKER_GROUP_RETAINED")

    monkeypatch.setattr(provider_module.tempfile, "mkdtemp", make_scratch)
    monkeypatch.setattr(
        provider_module,
        "_terminate_process_group",
        containment_failure,
    )
    provider = AgyEvaluationProvider(
        _config(),
        worker_path=_fake_worker(tmp_path),
        runtime_resolver=lambda: RUNTIME,
    )

    with pytest.raises(AgyEvaluationProviderError) as caught:
        await provider.call("gpt1", "system sentinel", "user sentinel", FakeResponse)

    assert caught.value.code == "WORKER_GROUP_RETAINED"
    assert not scratch.exists()
    assert provider.take_receipts() == ()


def test_recorded_execution_is_evaluation_only_and_binds_all_stages(tmp_path: Path):
    provider = AgyEvaluationProvider(
        _config(),
        worker_path=_fake_worker(tmp_path),
        runtime_resolver=lambda: RUNTIME,
    )
    execution = provider.recorded_execution()

    assert set(execution.stage_identities) == {"gpt1", "gpt2", "gpt3"}
    assert all(
        identity.provider == "google-antigravity-sdk"
        and identity.model == "gemini-evaluation-test"
        for identity in execution.stage_identities.values()
    )


@pytest.mark.asyncio
async def test_fake_sdk_provider_runs_only_the_recorded_grounded_lane(tmp_path: Path):
    store = KnowledgeStore(tmp_path / "knowledge")
    store.upsert_document(
        "profile",
        "personal",
        "Profile",
        "Alice's favorite color is blue.",
    )
    provider = AgyEvaluationProvider(
        _config(),
        worker_path=_grounded_worker(tmp_path),
        runtime_resolver=lambda: RUNTIME,
    )

    response, artifacts = await run_grounded_rag_recorded(
        GroundedQueryRequest(prompt="What is Alice's favorite color?"),
        store=store,
        recorded_execution=provider.recorded_execution(),
    )
    receipts = provider.take_receipts()
    observation = case_observation_from_artifacts(
        "agy-fake-case",
        artifacts,
        provider_invocations=receipts,
    )

    assert response.status == "ANSWER"
    assert [receipt.stage for receipt in receipts] == ["gpt1", "gpt2", "gpt3"]
    assert [receipt.usage.total_tokens for receipt in receipts if receipt.usage] == [
        14,
        14,
        14,
    ]
    assert [item.stage for item in observation.provider_invocations] == [
        "gpt1",
        "gpt2",
        "gpt3",
    ]
    assert observation.usage_cost.status == "UNAVAILABLE"


@pytest.mark.asyncio
async def test_worker_builds_finish_only_agent_with_zero_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    app_dir = tmp_path / "app"
    save_dir = tmp_path / "save"
    app_dir.mkdir(mode=0o700)
    save_dir.mkdir(mode=0o700)
    worker_hash = hashlib.sha256(Path(worker.__file__).read_bytes()).hexdigest()
    request = worker.WorkerRequest(
        protocol_version="agy-evaluation-worker-v1",
        stage="gpt1",
        requested_model="gemini-evaluation-test",
        system="system sentinel",
        user_content="user sentinel",
        response_schema=FakeResponse.model_json_schema(),
        expected_runtime=worker.RuntimeIdentity(
            sdk_artifact_sha256="a" * 64,
            localharness_sha256="b" * 64,
        ),
        expected_worker_source_sha256=worker_hash,
        expected_invocation_policy_sha256=INVOCATION_POLICY_SHA256,
        private_save_dir=str(save_dir),
        private_app_data_dir=str(app_dir),
    )
    captured: dict = {}

    class BuiltinTools:
        FINISH = "finish"

    class CapabilitiesConfig:
        def __init__(self, **kwargs):
            captured["capabilities"] = kwargs

    class CustomSystemInstructions:
        def __init__(self, **kwargs):
            captured["instructions"] = kwargs

    class ModelAPIRetryConfig:
        def __init__(self, **kwargs):
            captured["api_retry"] = kwargs

    class ModelOutputRetryConfig:
        def __init__(self, **kwargs):
            captured["output_retry"] = kwargs

    class RetryConfig:
        def __init__(self, **kwargs):
            captured["retry"] = kwargs

    class LocalAgentConfig:
        def __init__(self, **kwargs):
            captured["config"] = kwargs

    class FakeToolCalls:
        def __aiter__(self):
            return self

        async def __anext__(self):
            if getattr(self, "done", False):
                raise StopAsyncIteration
            self.done = True
            return SimpleNamespace(name=SimpleNamespace(value="finish"))

    class Response:
        tool_calls = FakeToolCalls()
        usage_metadata = SimpleNamespace(
            prompt_token_count=3,
            candidates_token_count=2,
            total_token_count=5,
            cached_content_token_count=None,
            thoughts_token_count=None,
            service_tier=None,
        )

        async def structured_output(self):
            return {"ok": True}

    class Agent:
        def __init__(self, config):
            captured["agent_config"] = config

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def chat(self, prompt):
            captured["prompt"] = prompt
            return Response()

    module = types.ModuleType("google.antigravity")
    for name, value in {
        "Agent": Agent,
        "BuiltinTools": BuiltinTools,
        "CapabilitiesConfig": CapabilitiesConfig,
        "CustomSystemInstructions": CustomSystemInstructions,
        "LocalAgentConfig": LocalAgentConfig,
        "ModelAPIRetryConfig": ModelAPIRetryConfig,
        "ModelOutputRetryConfig": ModelOutputRetryConfig,
        "RetryConfig": RetryConfig,
    }.items():
        setattr(module, name, value)
    monkeypatch.setitem(sys.modules, "google.antigravity", module)
    monkeypatch.setattr(
        worker,
        "_runtime_identity",
        lambda: worker.RuntimeIdentity(
            sdk_artifact_sha256="a" * 64,
            localharness_sha256="b" * 64,
        ),
    )
    monkeypatch.setenv("GEMINI_API_KEY", "dedicated-test-key")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setenv("LANG", "C.UTF-8")
    monkeypatch.setenv("LC_ALL", "C.UTF-8")

    result = await worker._run(request)

    assert result["status"] == "SUCCESS"
    assert captured["prompt"] == "user sentinel"
    assert captured["capabilities"] == {
        "enabled_tools": ["finish"],
        "enable_subagents": False,
    }
    config = captured["config"]
    assert config["tools"] == []
    assert config["policies"] == []
    assert config["hooks"] == []
    assert config["triggers"] == []
    assert config["mcp_servers"] == []
    assert config["subagents"] == []
    assert config["workspaces"] == []
    assert config["skills_paths"] == []
    assert captured["api_retry"] == {"max_retries": 0}
    assert captured["output_retry"] == {"max_retries": 0}


def test_import_and_runtime_inspection_do_not_import_sdk(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delitem(sys.modules, "google.antigravity", raising=False)
    assert "google.antigravity" not in sys.modules
    assert INVOCATION_POLICY_SHA256 == hashlib.sha256(json.dumps(
        {
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
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()).hexdigest()


def test_cli_selection_is_explicit_and_live_execution_remains_disabled(
    monkeypatch: pytest.MonkeyPatch,
):
    parser = measure_grounded_rag.build_parser()
    configured = parser.parse_args([
        "record",
        "--output-dir",
        "/private/fresh-output",
        "--external-cost-limit-usd",
        "1",
        "--allow-live-execution",
    ])
    assert measure_grounded_rag._recorded_runner(configured) is None

    agy = parser.parse_args([
        "record",
        "--output-dir",
        "/private/fresh-output",
        "--external-cost-limit-usd",
        "1",
        "--evaluation-provider",
        "agy-sdk",
        "--agy-model",
        "gemini-evaluation-test",
        "--agy-sdk-artifact-sha256",
        "a" * 64,
        "--agy-localharness-sha256",
        "b" * 64,
        "--agy-call-timeout-seconds",
        "120",
        "--allow-live-execution",
    ])
    monkeypatch.setenv("GEMINI_API_KEY", "must-not-be-read")
    with pytest.raises(EvaluationDataError, match="AGY_LIVE_EXECUTION_DISABLED"):
        measure_grounded_rag._recorded_runner(agy)
    assert "google.antigravity" not in sys.modules


@pytest.mark.parametrize(
    ("evaluation_provider", "expected_mode"),
    [
        ("configured", "CONFIGURED_UNOBSERVED"),
        ("agy-sdk", "AGY_SDK"),
    ],
)
def test_record_command_forwards_the_selected_provider_observation_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    evaluation_provider: str,
    expected_mode: str,
):
    parser = measure_grounded_rag.build_parser()
    args = parser.parse_args([
        "record",
        "--output-dir",
        str(tmp_path / "output"),
        "--external-cost-limit-usd",
        "1",
        "--allow-live-execution",
        "--evaluation-provider",
        evaluation_provider,
    ])
    benchmark = SimpleNamespace(
        definition=SimpleNamespace(
            execution_policy=SimpleNamespace(
                external_authorization_maximum_cost_usd=10.0,
            ),
        ),
    )
    captured: dict[str, object] = {}
    bundle = SimpleNamespace(status="INCOMPLETE", run_id="1" * 32, cases=[])

    async def fake_record_baseline(_benchmark, **kwargs):
        captured.update(kwargs)
        return bundle

    monkeypatch.setattr(
        measure_grounded_rag,
        "load_benchmark",
        lambda *_args, **_kwargs: benchmark,
    )
    monkeypatch.setattr(
        measure_grounded_rag,
        "_recorded_runner",
        lambda _args: object(),
    )
    monkeypatch.setattr(
        measure_grounded_rag,
        "record_baseline",
        fake_record_baseline,
    )
    monkeypatch.setattr(
        measure_grounded_rag,
        "publish_private_json_directory",
        lambda *_args, **_kwargs: tmp_path / "output" / "observations.json",
    )
    monkeypatch.setattr(
        measure_grounded_rag,
        "canonical_model_sha256",
        lambda _value: "a" * 64,
    )
    monkeypatch.setattr(
        measure_grounded_rag,
        "load_observations",
        lambda *_args, **_kwargs: bundle,
    )

    assert measure_grounded_rag._record_command(args) == 3
    assert captured["provider_observation_mode"] == expected_mode
