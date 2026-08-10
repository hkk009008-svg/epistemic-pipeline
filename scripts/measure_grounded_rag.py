#!/usr/bin/env python3
"""Validate, privately record, and independently score Grounded-RAG Baseline v1.

``record`` is intentionally inert unless ``--allow-live-execution`` is supplied.
Providing that flag is a local safety acknowledgement, not authorization by
itself; organizational authorization and provider controls remain external.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from pipeline.grounded_evaluation import (  # noqa: E402
    EXPECTED_BENCHMARK_SHA256,
    EvaluationDataError,
    load_adjudication,
    load_benchmark,
    load_observations,
    load_summary,
    canonical_model_sha256,
    publish_private_json_directory,
    record_baseline,
    score_baseline,
    validate_baseline_summary,
    validate_observation_bundle,
)


DEFAULT_BENCHMARK = (
    REPOSITORY_ROOT
    / "knowledge_data"
    / "benchmarks"
    / "grounded-rag-baseline-v1"
    / "benchmark.json"
)


def _add_benchmark_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=DEFAULT_BENCHMARK,
        help="protected benchmark JSON",
    )
    parser.add_argument(
        "--benchmark-sha256",
        default=EXPECTED_BENCHMARK_SHA256,
        help="externally frozen raw-byte SHA-256",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measured Grounded-RAG Baseline v1 evaluation harness."
    )
    subparsers = parser.add_subparsers(dest="operation", required=True)

    validate_parser = subparsers.add_parser(
        "validate",
        help="validate and resolve only the protected benchmark",
    )
    _add_benchmark_arguments(validate_parser)

    record_parser = subparsers.add_parser(
        "record",
        help="execute the protected benchmark into a private observation bundle",
    )
    _add_benchmark_arguments(record_parser)
    record_parser.add_argument("--output-dir", type=Path, required=True)
    record_parser.add_argument(
        "--allow-live-execution",
        action="store_true",
        help="required safety acknowledgement; external authorization still applies",
    )
    record_parser.add_argument(
        "--external-cost-limit-usd",
        type=float,
        required=True,
        help=(
            "attested provider-side hard cap; must not exceed the protected "
            "benchmark's external authorization ceiling"
        ),
    )

    score_parser = subparsers.add_parser(
        "score",
        help="join complete observations with independent human adjudication",
    )
    _add_benchmark_arguments(score_parser)
    score_parser.add_argument("--observations", type=Path, required=True)
    score_parser.add_argument(
        "--observations-sha256",
        required=True,
        help="raw-byte SHA-256 retained from record",
    )
    score_parser.add_argument("--adjudication", type=Path, required=True)
    score_parser.add_argument(
        "--adjudication-sha256",
        required=True,
        help="raw-byte SHA-256 retained by the human label authority",
    )
    score_parser.add_argument("--output-dir", type=Path, required=True)

    verify_parser = subparsers.add_parser(
        "verify-summary",
        help="recompute and verify a published summary against its private inputs",
    )
    _add_benchmark_arguments(verify_parser)
    verify_parser.add_argument("--observations", type=Path, required=True)
    verify_parser.add_argument("--observations-sha256", required=True)
    verify_parser.add_argument("--adjudication", type=Path, required=True)
    verify_parser.add_argument("--adjudication-sha256", required=True)
    verify_parser.add_argument("--summary", type=Path, required=True)
    verify_parser.add_argument("--summary-sha256", required=True)
    return parser


def _validate_command(args: argparse.Namespace) -> int:
    benchmark = load_benchmark(args.benchmark, args.benchmark_sha256)
    definition = benchmark.definition
    span_count = sum(len(spans) for spans in benchmark.gold_spans_by_case.values())
    # Only metadata is printed; source data, queries, and labels remain private.
    print(
        "VALID "
        f"benchmark_id={definition.benchmark_id} "
        f"sha256={benchmark.raw_sha256} "
        f"documents={len(definition.documents)} "
        f"cases={len(definition.cases)} "
        f"gold_spans={span_count}"
    )
    return 0


def _record_command(args: argparse.Namespace) -> int:
    # Hash and fully resolve the benchmark before importing/configuring/calling
    # any provider-aware execution path in record_baseline.
    benchmark = load_benchmark(args.benchmark, args.benchmark_sha256)
    if not args.allow_live_execution:
        raise EvaluationDataError(
            "record requires --allow-live-execution and separate external authorization"
        )
    bundle = asyncio.run(record_baseline(
        benchmark,
        external_cost_limit_usd=args.external_cost_limit_usd,
    ))
    # Re-read and re-hash the protected file after execution before publication.
    load_benchmark(args.benchmark, args.benchmark_sha256)
    published = publish_private_json_directory(
        args.output_dir,
        "observations.json",
        bundle,
    )
    observation_sha256 = canonical_model_sha256(bundle)
    # A pathname can be renamed after any inode check. Reopen the returned path
    # under the digest retained from the in-memory capture before reporting it.
    load_observations(published, observation_sha256)
    print(
        f"{bundle.status} run_id={bundle.run_id} "
        f"observation_sha256={observation_sha256} "
        f"recorded_cases={len(bundle.cases)} output={published}"
    )
    return 0 if bundle.status == "COMPLETE" else 3


def _score_command(args: argparse.Namespace) -> int:
    benchmark = load_benchmark(args.benchmark, args.benchmark_sha256)
    observations = load_observations(
        args.observations,
        args.observations_sha256,
    )
    validate_observation_bundle(benchmark, observations)
    if observations.status != "COMPLETE":
        raise EvaluationDataError("incomplete observations cannot be scored")
    adjudication = load_adjudication(
        args.adjudication,
        args.adjudication_sha256,
    )
    summary = score_baseline(benchmark, observations, adjudication)
    # Confirm the human-owned benchmark is still byte-identical before publish.
    load_benchmark(args.benchmark, args.benchmark_sha256)
    published = publish_private_json_directory(
        args.output_dir,
        "summary.json",
        summary,
    )
    summary_sha256 = canonical_model_sha256(summary)
    load_summary(published, summary_sha256)
    print(
        f"{summary.result} run_id={summary.observation_run_id} "
        f"cases={summary.case_count} thresholds_passed={summary.thresholds_passed} "
        f"summary_sha256={summary_sha256} "
        f"output={published}"
    )
    return 0 if summary.result == "PASS" else 4


def _verify_summary_command(args: argparse.Namespace) -> int:
    benchmark = load_benchmark(args.benchmark, args.benchmark_sha256)
    observations = load_observations(
        args.observations,
        args.observations_sha256,
    )
    adjudication = load_adjudication(
        args.adjudication,
        args.adjudication_sha256,
    )
    summary = load_summary(args.summary, args.summary_sha256)
    validate_baseline_summary(
        benchmark,
        observations,
        adjudication,
        summary,
        args.summary_sha256,
    )
    print(
        f"VERIFIED run_id={summary.observation_run_id} "
        f"summary_sha256={args.summary_sha256} result={summary.result}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.operation == "validate":
            return _validate_command(args)
        if args.operation == "record":
            return _record_command(args)
        if args.operation == "verify-summary":
            return _verify_summary_command(args)
        return _score_command(args)
    except EvaluationDataError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        # A real signal before an observation bundle exists cannot be safely
        # represented as a published run. No provider exception text is shown.
        print("ERROR: recording interrupted before publication", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
