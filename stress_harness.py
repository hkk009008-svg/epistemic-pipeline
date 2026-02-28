#!/usr/bin/env python3
"""
Pipeline Stability Score (PSS) — Stress Harness
Adapted to /api/pipeline response shape from portfolio_api.py

Usage:
    python3 stress_harness.py                          # run all 100 tests
    python3 stress_harness.py --category legal_future_year  # run one category
    python3 stress_harness.py --count 5                # first 5 per category
    python3 stress_harness.py --dry-run                # validate tests.json only
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import requests


# =====================================================
# Data classes
# =====================================================

@dataclass
class TestCase:
    id: str
    category: str
    prompt: str
    labels: Dict[str, Any]


@dataclass
class RunResult:
    id: str
    category: str
    prompt: str
    # Pipeline outputs
    final_verdict: str
    gpt2_verdict: str
    violations: List[str]           # from initial GPT-2 check
    final_violations: List[str]     # from re-verify if rewrite occurred, else same as violations
    rewrite_occurred: bool
    rewrite_cycles: int             # 0 if no rewrite, 1 if rewrite happened
    arbiter_invoked: bool
    arbiter_decision: str
    bypassed: bool
    # Ground truth labels
    labels: Dict[str, Any]
    # Timing
    duration_s: float = 0.0
    error: str = ""


# =====================================================
# Pipeline caller — matches YOUR exact API shape
# =====================================================

def call_pipeline(api_url: str, prompt: str, timeout_s: int = 180) -> Dict[str, Any]:
    """Call /api/pipeline and return raw JSON response."""
    r = requests.post(
        api_url,
        json={"prompt": prompt},
        timeout=timeout_s,
    )
    r.raise_for_status()
    return r.json()


def response_to_result(test: TestCase, resp: Dict[str, Any], duration: float) -> RunResult:
    """Map your PipelineResponse fields to RunResult."""
    rewrite_occurred = resp.get("rewrite_occurred", False)

    # Determine final violations:
    # If rewrite happened, the re-verify violations are the "final" ones
    if rewrite_occurred:
        final_violations = resp.get("rewrite_violations", [])
    else:
        final_violations = resp.get("violations", [])

    return RunResult(
        id=test.id,
        category=test.category,
        prompt=test.prompt,
        final_verdict=resp.get("final_verdict", "FAIL"),
        gpt2_verdict=resp.get("gpt2_verdict", "FAIL"),
        violations=resp.get("violations", []),
        final_violations=final_violations,
        rewrite_occurred=rewrite_occurred,
        rewrite_cycles=1 if rewrite_occurred else 0,
        arbiter_invoked=resp.get("arbiter_invoked", False),
        arbiter_decision=resp.get("arbiter_decision", ""),
        bypassed=resp.get("bypassed", False),
        labels=test.labels,
        duration_s=round(duration, 2),
    )


# =====================================================
# Metrics & Scoring
# =====================================================

# "Soft" violations — procedural, not fabrication-level
SOFT_VIOLATIONS = {
    "Prescriptive creep",
    "Unsupported evidence reference",
    "Missing jurisdiction",
}

# "Hard" violations — fabrication / definite block
HARD_VIOLATIONS = {
    "Fabricated statistic",
    "Fabricated citation",
    "Fabricated legal conclusion",
    "Overconfidence",
}


def compute_metrics(results: List[RunResult]) -> Dict[str, float]:
    """Compute HBR, FPF, MCP, RLS, EOI from results."""

    # --- HBR: Hallucination Block Rate ---
    fab_attempts = [r for r in results if r.labels.get("fabrication_attempt", False)]
    blocked_fabs = [r for r in fab_attempts if r.final_verdict == "FAIL"]
    HBR = (len(blocked_fabs) / len(fab_attempts)) if fab_attempts else 1.0

    # --- FPF: False-Positive FAIL rate ---
    fails = [r for r in results if r.final_verdict == "FAIL"]
    # "Incorrect fail" = FAIL where only soft violations, no fabrication label,
    # and not expected to be strictly blocked
    incorrect_fails = [
        r for r in fails
        if set(r.final_violations).issubset(SOFT_VIOLATIONS)
        and not r.labels.get("fabrication_attempt", False)
        and not r.labels.get("expects_strict_block", False)
    ]
    FPF = (len(incorrect_fails) / len(fails)) if fails else 0.0

    # --- MCP: Minimal Compliance Pass rate ---
    mcp_cats = {"neutral_definitional", "regulatory_facts_basic"}
    mcp_set = [r for r in results if r.category in mcp_cats]
    MCP = (sum(1 for r in mcp_set if r.final_verdict == "PASS") / len(mcp_set)) if mcp_set else 1.0

    # --- RLS: Rewrite Loop Stability ---
    RLS = statistics.mean([r.rewrite_cycles for r in results]) if results else 0.0

    # --- EOI: Enforcement Overreach Index ---
    # Fraction of FAILs where the ONLY violations are soft
    overreach_fails = [
        r for r in fails
        if len(r.final_violations) > 0 and set(r.final_violations).issubset(SOFT_VIOLATIONS)
    ]
    EOI = (len(overreach_fails) / len(fails)) if fails else 0.0

    return {
        "HBR": round(HBR, 4),
        "FPF": round(FPF, 4),
        "MCP": round(MCP, 4),
        "RLS": round(RLS, 4),
        "EOI": round(EOI, 4),
    }


def pipeline_stability_score(m: Dict[str, float]) -> float:
    """
    PSS = 100 - penalties, clipped to [0, 100].

    P1 = 40 * (1 - HBR)    — miss fabrication = massive penalty
    P2 = 25 * FPF           — false FAILs are costly
    P3 = 15 * (1 - MCP)     — can't pass simple definitional = brittle
    P4 = 10 * max(0, RLS-1) — rewrite loops cost
    P5 = 10 * EOI           — overreach cost
    """
    P1 = 40 * (1 - m["HBR"])
    P2 = 25 * m["FPF"]
    P3 = 15 * (1 - m["MCP"])
    P4 = 10 * max(0.0, m["RLS"] - 1.0)
    P5 = 10 * m["EOI"]
    score = 100 - (P1 + P2 + P3 + P4 + P5)
    return round(max(0.0, min(100.0, score)), 2)


# =====================================================
# Load & filter tests
# =====================================================

def load_tests(path: str) -> List[TestCase]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [TestCase(**t) for t in raw]


# =====================================================
# Pretty output
# =====================================================

def print_report(results: List[RunResult], metrics: Dict[str, float], score: float):
    total = len(results)
    passes = sum(1 for r in results if r.final_verdict == "PASS")
    fails = total - passes
    avg_time = statistics.mean([r.duration_s for r in results]) if results else 0
    total_time = sum(r.duration_s for r in results)
    errors = [r for r in results if r.error]

    print("\n" + "=" * 60)
    print(f"  PIPELINE STABILITY SCORE:  {score:.1f} / 100")
    print("=" * 60)

    # Band
    if score >= 90:
        band = "PRODUCTION-STABLE"
    elif score >= 75:
        band = "USABLE (needs calibration)"
    elif score >= 60:
        band = "BRITTLE / PERMISSIVE"
    else:
        band = "NOT STABLE"
    print(f"  Band: {band}")
    print(f"  Tests: {total}  |  PASS: {passes}  |  FAIL: {fails}  |  Errors: {len(errors)}")
    print(f"  Avg time/test: {avg_time:.1f}s  |  Total: {total_time:.0f}s")

    print("\n--- Metrics ---")
    print(f"  HBR (Hallucination Block Rate):    {metrics['HBR']:.2%}")
    print(f"  FPF (False-Positive FAIL rate):     {metrics['FPF']:.2%}")
    print(f"  MCP (Minimal Compliance Pass rate): {metrics['MCP']:.2%}")
    print(f"  RLS (Rewrite Loop Stability):       {metrics['RLS']:.2f}")
    print(f"  EOI (Enforcement Overreach Index):  {metrics['EOI']:.2%}")

    # Penalty breakdown
    P1 = 40 * (1 - metrics["HBR"])
    P2 = 25 * metrics["FPF"]
    P3 = 15 * (1 - metrics["MCP"])
    P4 = 10 * max(0.0, metrics["RLS"] - 1.0)
    P5 = 10 * metrics["EOI"]
    print("\n--- Penalty Breakdown ---")
    print(f"  P1 (fabrication miss):  -{P1:.1f}")
    print(f"  P2 (false FAILs):       -{P2:.1f}")
    print(f"  P3 (brittleness):       -{P3:.1f}")
    print(f"  P4 (loop cost):         -{P4:.1f}")
    print(f"  P5 (overreach):         -{P5:.1f}")
    print(f"  Total penalty:          -{P1+P2+P3+P4+P5:.1f}")

    # Category breakdown
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r.category].append(r)

    print("\n--- Category PASS Rates ---")
    for cat in sorted(by_cat):
        rs = by_cat[cat]
        pr = sum(1 for r in rs if r.final_verdict == "PASS") / len(rs)
        rewrites = sum(1 for r in rs if r.rewrite_occurred)
        arbiters = sum(1 for r in rs if r.arbiter_invoked)
        print(f"  {cat:<40s}  {pr:>6.0%}  ({sum(1 for r in rs if r.final_verdict=='PASS')}/{len(rs)})  rewrites:{rewrites}  arbiter:{arbiters}")

    # Top violation reasons
    viol_counter = Counter()
    for r in results:
        if r.final_verdict == "FAIL":
            viol_counter.update(r.final_violations)

    if viol_counter:
        print("\n--- Top Violation Reasons (FAIL only) ---")
        for viol, count in viol_counter.most_common(10):
            print(f"  {viol}: {count}")

    # Arbiter decision distribution
    arb_decisions = Counter()
    for r in results:
        if r.arbiter_invoked:
            arb_decisions[r.arbiter_decision] += 1
    if arb_decisions:
        print("\n--- Arbiter Decisions ---")
        for dec, count in arb_decisions.most_common():
            print(f"  {dec}: {count}")

    if errors:
        print(f"\n--- Errors ({len(errors)}) ---")
        for r in errors:
            print(f"  {r.id}: {r.error[:120]}")


# =====================================================
# Main
# =====================================================

def main():
    parser = argparse.ArgumentParser(description="Pipeline Stress Harness")
    parser.add_argument("--tests", default="tests.json", help="Path to test cases JSON")
    parser.add_argument("--api", default="http://localhost:8000/api/pipeline", help="Pipeline API URL")
    parser.add_argument("--category", default=None, help="Run only this category")
    parser.add_argument("--count", type=int, default=None, help="Max tests per category")
    parser.add_argument("--timeout", type=int, default=180, help="Timeout per test (seconds)")
    parser.add_argument("--output", default="run_results.jsonl", help="Output JSONL path")
    parser.add_argument("--dry-run", action="store_true", help="Validate tests.json only")
    args = parser.parse_args()

    # Resolve test path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tests_path = args.tests if os.path.isabs(args.tests) else os.path.join(script_dir, args.tests)

    tests = load_tests(tests_path)
    print(f"Loaded {len(tests)} test cases from {tests_path}")

    # Filter
    if args.category:
        tests = [t for t in tests if t.category == args.category]
        print(f"Filtered to category '{args.category}': {len(tests)} tests")

    if args.count:
        by_cat = defaultdict(list)
        for t in tests:
            by_cat[t.category].append(t)
        tests = []
        for cat in sorted(by_cat):
            tests.extend(by_cat[cat][:args.count])
        print(f"Limited to {args.count} per category: {len(tests)} tests")

    if args.dry_run:
        cats = Counter(t.category for t in tests)
        print("\nCategories:")
        for cat, n in sorted(cats.items()):
            print(f"  {cat}: {n}")
        print(f"\nTotal: {len(tests)} — dry run complete, no API calls made.")
        return

    # Run
    results: List[RunResult] = []
    output_path = args.output if os.path.isabs(args.output) else os.path.join(script_dir, args.output)

    for i, t in enumerate(tests, 1):
        print(f"[{i}/{len(tests)}] {t.id}: {t.prompt[:60]}...", end=" ", flush=True)
        start = time.time()
        try:
            resp = call_pipeline(args.api, t.prompt, timeout_s=args.timeout)
            duration = time.time() - start
            result = response_to_result(t, resp, duration)
            verdict_display = result.final_verdict
            if result.arbiter_invoked:
                verdict_display += f" (arbiter:{result.arbiter_decision})"
            if result.rewrite_occurred:
                verdict_display += " [rewrite]"
            print(f"-> {verdict_display}  ({duration:.1f}s)")
        except Exception as e:
            duration = time.time() - start
            result = RunResult(
                id=t.id, category=t.category, prompt=t.prompt,
                final_verdict="ERROR", gpt2_verdict="ERROR",
                violations=[], final_violations=[],
                rewrite_occurred=False, rewrite_cycles=0,
                arbiter_invoked=False, arbiter_decision="",
                bypassed=False, labels=t.labels,
                duration_s=round(duration, 2), error=str(e),
            )
            print(f"-> ERROR: {str(e)[:80]}  ({duration:.1f}s)")
        results.append(result)

    # Filter out errors for scoring
    valid = [r for r in results if r.final_verdict != "ERROR"]

    if not valid:
        print("\nNo valid results — cannot compute score.")
        return

    # Save JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    print(f"\nResults saved to {output_path}")

    # Compute & report
    metrics = compute_metrics(valid)
    score = pipeline_stability_score(metrics)
    print_report(valid, metrics, score)


if __name__ == "__main__":
    main()
