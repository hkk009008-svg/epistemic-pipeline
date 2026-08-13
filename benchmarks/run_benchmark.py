#!/usr/bin/env python3
"""Epistemic Pipeline benchmark harness.

Runs test prompts through the pipeline (and optionally a raw LLM baseline),
computes claim-level and pipeline-level metrics, and outputs structured results.

Usage:
    python benchmarks/run_benchmark.py
    python benchmarks/run_benchmark.py --category statistical_percentage_trap
    python benchmarks/run_benchmark.py --baseline --output results.json
    python benchmarks/run_benchmark.py --tier standard --count 5
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from pipeline.models import PipelineRequest
from pipeline.orchestrator import run_pipeline
from pipeline.helpers import call_llm
from pipeline.stress import compute_pss_metrics, has_leaked_stats


def load_tests(tests_path: str, category: str | None = None, count: int | None = None) -> list:
    """Load test cases from tests.json, optionally filtering."""
    with open(tests_path, "r", encoding="utf-8") as f:
        tests = json.load(f)

    if category:
        tests = [t for t in tests if t["category"] == category]

    if count:
        by_cat: dict[str, list] = defaultdict(list)
        for t in tests:
            by_cat[t["category"]].append(t)
        tests = []
        for cat in sorted(by_cat):
            tests.extend(by_cat[cat][:count])

    return tests


def run_pipeline_test(test: dict, tier: str = "strict") -> dict:
    """Run a single test through the pipeline, capturing full results."""
    start = time.time()
    try:
        req = PipelineRequest(prompt=test["prompt"], tier=tier)
        resp = run_pipeline(req)
        data = resp.model_dump()
        duration = time.time() - start

        rewrite_occurred = data.get("rewrite_occurred", False)
        final_violations = (
            data.get("rewrite_violations", []) if rewrite_occurred
            else data.get("violations", [])
        )

        return {
            "id": test["id"],
            "category": test["category"],
            "prompt": test["prompt"],
            "final_verdict": data.get("final_verdict", "FAIL"),
            "final_result": data.get("final_result", ""),
            "gpt2_verdict": data.get("gpt2_verdict", "FAIL"),
            "violations": data.get("violations", []),
            "final_violations": final_violations,
            "rewrite_occurred": rewrite_occurred,
            "rewrite_cycles": 1 if rewrite_occurred else 0,
            "arbiter_invoked": data.get("arbiter_invoked", False),
            "arbiter_decision": data.get("arbiter_decision", ""),
            "bypassed": data.get("bypassed", False),
            "labels": test.get("labels", {}),
            "duration_s": round(duration, 2),
            "confidence": data.get("confidence", {}),
            "claim_table": data.get("claim_table", []),
            "search_performed": data.get("search_performed", False),
            "atomic_claims": data.get("atomic_claims", []),
            "error": "",
        }
    except Exception as e:
        duration = time.time() - start
        return {
            "id": test["id"],
            "category": test["category"],
            "prompt": test["prompt"],
            "final_verdict": "ERROR",
            "final_result": "",
            "gpt2_verdict": "ERROR",
            "violations": [],
            "final_violations": [],
            "rewrite_occurred": False,
            "rewrite_cycles": 0,
            "arbiter_invoked": False,
            "arbiter_decision": "",
            "bypassed": False,
            "labels": test.get("labels", {}),
            "duration_s": round(duration, 2),
            "confidence": {},
            "claim_table": [],
            "search_performed": False,
            "atomic_claims": [],
            "error": str(e),
        }


def run_baseline_test(test: dict) -> dict:
    """Run a prompt directly through the LLM without pipeline verification."""
    start = time.time()
    try:
        response = call_llm(
            config.get_stage_config("gpt1"),
            "You are a helpful assistant. Answer the user's question accurately and concisely.",
            test["prompt"],
        )
        duration = time.time() - start
        return {
            "id": test["id"],
            "prompt": test["prompt"],
            "response": response,
            "has_bare_stats": has_leaked_stats(response),
            "duration_s": round(duration, 2),
            "error": "",
        }
    except Exception as e:
        duration = time.time() - start
        return {
            "id": test["id"],
            "prompt": test["prompt"],
            "response": "",
            "has_bare_stats": False,
            "duration_s": round(duration, 2),
            "error": str(e),
        }


_HEDGING_RE = re.compile(
    r"\b(may|might|could|approximately|roughly|around|estimated|varies|"
    r"it depends|generally|typically|often|usually|some sources)\b",
    re.IGNORECASE,
)


def compute_claim_level_metrics(results: list) -> dict:
    """Compute claim-level metrics across all results."""
    total_claims = 0
    tier_counts = defaultdict(int)
    violation_counts = defaultdict(int)

    for r in results:
        if r["final_verdict"] == "ERROR":
            continue
        claims = r.get("claim_table", [])
        total_claims += len(claims)
        for claim in claims:
            cat = claim.get("category", "Unknown")
            tier_counts[cat] += 1

        for v in r.get("final_violations", []):
            violation_counts[v] += 1

    tier_pcts = {}
    if total_claims > 0:
        for tier_name, count in tier_counts.items():
            tier_pcts[tier_name] = round(count / total_claims * 100, 1)

    return {
        "total_claims": total_claims,
        "tier_distribution": dict(tier_counts),
        "tier_percentages": tier_pcts,
        "violation_density": (
            round(sum(violation_counts.values()) / max(total_claims, 1) * 100, 2)
        ),
        "top_violations": dict(
            sorted(violation_counts.items(), key=lambda x: -x[1])[:10]
        ),
    }


def compute_baseline_comparison(pipeline_results: list, baseline_results: list) -> dict:
    """Compare pipeline output with raw baseline."""
    baseline_by_id = {b["id"]: b for b in baseline_results}

    pipeline_bare_stats = 0
    baseline_bare_stats = 0
    pipeline_hedging = 0
    baseline_hedging = 0
    compared = 0

    for pr in pipeline_results:
        br = baseline_by_id.get(pr["id"])
        if not br or pr["final_verdict"] == "ERROR" or br.get("error"):
            continue
        compared += 1

        if has_leaked_stats(pr.get("final_result", "")):
            pipeline_bare_stats += 1
        if br["has_bare_stats"]:
            baseline_bare_stats += 1

        if _HEDGING_RE.search(pr.get("final_result", "")):
            pipeline_hedging += 1
        if _HEDGING_RE.search(br.get("response", "")):
            baseline_hedging += 1

    return {
        "compared": compared,
        "pipeline_bare_stat_count": pipeline_bare_stats,
        "baseline_bare_stat_count": baseline_bare_stats,
        "pipeline_bare_stat_rate": round(pipeline_bare_stats / max(compared, 1) * 100, 1),
        "baseline_bare_stat_rate": round(baseline_bare_stats / max(compared, 1) * 100, 1),
        "pipeline_hedging_rate": round(pipeline_hedging / max(compared, 1) * 100, 1),
        "baseline_hedging_rate": round(baseline_hedging / max(compared, 1) * 100, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Epistemic Pipeline Benchmark")
    parser.add_argument("--category", type=str, help="Filter to a specific category")
    parser.add_argument("--count", type=int, help="Max tests per category")
    parser.add_argument("--tier", type=str, default="strict", choices=["strict", "standard", "light"])
    parser.add_argument("--baseline", action="store_true", help="Also run raw LLM baseline for comparison")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    args = parser.parse_args()

    if not config.has_api_key():
        print("ERROR: OPENAI_API_KEY not set. Export it or set via .env file.")
        sys.exit(1)

    # Locate tests.json
    project_root = Path(__file__).resolve().parent.parent
    tests_path = project_root / "tests.json"
    if not tests_path.exists():
        print(f"ERROR: {tests_path} not found.")
        sys.exit(1)

    tests = load_tests(str(tests_path), category=args.category, count=args.count)
    if not tests:
        print("ERROR: No test cases found.")
        sys.exit(1)

    print(f"Running {len(tests)} tests (tier={args.tier})")
    print("=" * 60)

    # Run pipeline tests
    pipeline_results = []
    start_all = time.time()
    for i, t in enumerate(tests):
        print(f"  [{i+1}/{len(tests)}] {t['id']}: {t['prompt'][:60]}...", end=" ", flush=True)
        result = run_pipeline_test(t, tier=args.tier)
        pipeline_results.append(result)
        status = result["final_verdict"]
        print(f"→ {status} ({result['duration_s']}s)")

    total_duration = time.time() - start_all

    # Run baseline if requested
    baseline_results = []
    baseline_comparison = None
    if args.baseline:
        print("\nRunning baseline (raw LLM, no pipeline)...")
        print("=" * 60)
        for i, t in enumerate(tests):
            print(f"  [{i+1}/{len(tests)}] {t['id']}...", end=" ", flush=True)
            br = run_baseline_test(t)
            baseline_results.append(br)
            print(f"done ({br['duration_s']}s)")

        baseline_comparison = compute_baseline_comparison(pipeline_results, baseline_results)

    # Compute metrics
    valid = [r for r in pipeline_results if r["final_verdict"] != "ERROR"]
    pss = compute_pss_metrics(valid) if valid else {"score": 0, "metrics": {}, "penalties": {}}
    claim_level = compute_claim_level_metrics(valid)

    # Category breakdown
    by_cat: dict[str, list] = defaultdict(list)
    for r in pipeline_results:
        by_cat[r["category"]].append(r)

    categories = {}
    for cat in sorted(by_cat):
        rs = by_cat[cat]
        categories[cat] = {
            "total": len(rs),
            "pass": sum(1 for r in rs if r["final_verdict"] == "PASS"),
            "fail": sum(1 for r in rs if r["final_verdict"] == "FAIL"),
            "error": sum(1 for r in rs if r["final_verdict"] == "ERROR"),
        }

    # Summary
    output = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": config.get_model(),
            "tier": args.tier,
            "pipeline_version": "3.0.0",
            "total_prompts": len(tests),
            "total_duration_s": round(total_duration, 2),
        },
        "pss": pss,
        "claim_level": claim_level,
        "categories": categories,
        "results": pipeline_results,
    }

    if baseline_comparison:
        output["baseline_comparison"] = baseline_comparison
        output["baseline_results"] = baseline_results

    # Print summary
    print("\n" + "=" * 60)
    print(f"PSS Score: {pss['score']}")
    print(f"Total: {len(pipeline_results)} | "
          f"PASS: {sum(1 for r in pipeline_results if r['final_verdict'] == 'PASS')} | "
          f"FAIL: {sum(1 for r in pipeline_results if r['final_verdict'] == 'FAIL')} | "
          f"ERROR: {sum(1 for r in pipeline_results if r['final_verdict'] == 'ERROR')}")
    print(f"Claims analyzed: {claim_level['total_claims']}")
    print(f"Duration: {round(total_duration, 1)}s")

    if baseline_comparison:
        print(f"\nBaseline comparison ({baseline_comparison['compared']} prompts):")
        print(f"  Bare stats — pipeline: {baseline_comparison['pipeline_bare_stat_rate']}% | "
              f"baseline: {baseline_comparison['baseline_bare_stat_rate']}%")
        print(f"  Hedging    — pipeline: {baseline_comparison['pipeline_hedging_rate']}% | "
              f"baseline: {baseline_comparison['baseline_hedging_rate']}%")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {output_path}")

    return output


if __name__ == "__main__":
    main()
