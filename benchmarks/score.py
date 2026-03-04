#!/usr/bin/env python3
"""Score a previously saved benchmark run.

Usage:
    python benchmarks/score.py results.json
    python benchmarks/score.py results.json --verbose
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.stress import compute_pss_metrics


def score_results(data: dict, verbose: bool = False) -> None:
    """Print a formatted score report from benchmark output."""
    meta = data.get("meta", {})
    pss = data.get("pss", {})
    claim_level = data.get("claim_level", {})
    categories = data.get("categories", {})
    results = data.get("results", [])
    baseline = data.get("baseline_comparison")

    print(f"Benchmark Report — {meta.get('timestamp', 'unknown')}")
    print(f"Model: {meta.get('model')} | Tier: {meta.get('tier')} | Version: {meta.get('pipeline_version')}")
    print("=" * 60)

    # PSS
    print(f"\nPipeline Stability Score (PSS): {pss.get('score', 0)}")
    metrics = pss.get("metrics", {})
    penalties = pss.get("penalties", {})
    print(f"  HLR (Hallucination Leakage):   {metrics.get('HLR', 0):.4f}  (penalty: {penalties.get('P1', 0):.2f})")
    print(f"  FPF (False Positive FAIL):     {metrics.get('FPF', 0):.4f}  (penalty: {penalties.get('P2', 0):.2f})")
    print(f"  MCP (Mundane Correct Pass):    {metrics.get('MCP', 0):.4f}  (penalty: {penalties.get('P3', 0):.2f})")
    print(f"  RLS (Rewrite Loop Stress):     {metrics.get('RLS', 0):.4f}  (penalty: {penalties.get('P4', 0):.2f})")
    print(f"  EOI (Enforcement Overreach):   {metrics.get('EOI', 0):.4f}  (penalty: {penalties.get('P5', 0):.2f})")

    # Verdicts
    total = len(results)
    pass_n = sum(1 for r in results if r["final_verdict"] == "PASS")
    fail_n = sum(1 for r in results if r["final_verdict"] == "FAIL")
    err_n = sum(1 for r in results if r["final_verdict"] == "ERROR")
    print(f"\nVerdicts: {pass_n} PASS / {fail_n} FAIL / {err_n} ERROR (of {total})")

    # Claims
    print("\nClaim Analysis:")
    print(f"  Total claims: {claim_level.get('total_claims', 0)}")
    print(f"  Violation density: {claim_level.get('violation_density', 0):.2f} per 100 claims")
    tier_pcts = claim_level.get("tier_percentages", {})
    for tier_name, pct in sorted(tier_pcts.items(), key=lambda x: -x[1]):
        print(f"  {tier_name}: {pct}%")

    # Categories
    print("\nCategory Breakdown:")
    for cat, info in sorted(categories.items()):
        t = info["total"]
        p = info.get("pass", 0)
        f = info.get("fail", 0)
        e = info.get("error", 0)
        print(f"  {cat:40s}  {p}/{t} PASS  {f}/{t} FAIL  {e}/{t} ERR")

    # Baseline
    if baseline:
        print(f"\nBaseline Comparison ({baseline['compared']} prompts):")
        print(f"  Bare stats — pipeline: {baseline['pipeline_bare_stat_rate']}% vs baseline: {baseline['baseline_bare_stat_rate']}%")
        print(f"  Hedging    — pipeline: {baseline['pipeline_hedging_rate']}% vs baseline: {baseline['baseline_hedging_rate']}%")

    # Verbose: show failures
    if verbose:
        failures = [r for r in results if r["final_verdict"] == "FAIL"]
        if failures:
            print(f"\n{'=' * 60}")
            print(f"Failed prompts ({len(failures)}):")
            for r in failures:
                print(f"\n  [{r['id']}] {r['prompt'][:80]}")
                print(f"    Violations: {', '.join(r.get('final_violations', []))}")
                if r.get("arbiter_decision"):
                    print(f"    Arbiter: {r['arbiter_decision']}")

    # Duration
    print(f"\nTotal duration: {meta.get('total_duration_s', 0)}s")


def main():
    parser = argparse.ArgumentParser(description="Score a benchmark run")
    parser.add_argument("results_file", type=str, help="Path to benchmark results JSON")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show individual failures")
    args = parser.parse_args()

    path = Path(args.results_file)
    if not path.exists():
        print(f"ERROR: {path} not found.")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Optionally recompute PSS from raw results
    results = data.get("results", [])
    if results:
        valid = [r for r in results if r["final_verdict"] != "ERROR"]
        if valid:
            data["pss"] = compute_pss_metrics(valid)

    score_results(data, verbose=args.verbose)


if __name__ == "__main__":
    main()
