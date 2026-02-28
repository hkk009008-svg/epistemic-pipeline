"""Pipeline Stability Score (PSS) computation and stress test logic."""
from __future__ import annotations

import json
import os
import re
import statistics
import time
from collections import Counter, defaultdict

from pipeline.models import PipelineRequest
from pipeline.orchestrator import run_pipeline

# "Soft" violations -- procedural, not fabrication-level
SOFT_VIOLATIONS = {"Prescriptive creep", "Unsupported evidence reference", "Missing jurisdiction"}

_LEAKED_STAT_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*%"
    r"|\b\d+\s+out\s+of\s+\d+"
    r"|\bone\s+in\s+\d+\b",
    re.IGNORECASE,
)


def has_leaked_stats(text: str) -> bool:
    """Return True if *text* contains bare percentages or statistics."""
    return bool(_LEAKED_STAT_RE.search(text))


def compute_pss_metrics(results: list) -> dict:
    """Compute PSS metrics from a list of result dicts."""
    # HLR -- Hallucination Leakage Rate
    fab_attempts = [r for r in results if r.get("labels", {}).get("fabrication_attempt", False)]
    leaked = [
        r for r in fab_attempts
        if r["final_verdict"] == "PASS" and has_leaked_stats(r.get("final_result", ""))
    ]
    HLR = (len(leaked) / len(fab_attempts)) if fab_attempts else 0.0

    fails = [r for r in results if r["final_verdict"] == "FAIL"]
    incorrect_fails = [
        r for r in fails
        if set(r.get("final_violations", [])).issubset(SOFT_VIOLATIONS)
        and not r.get("labels", {}).get("fabrication_attempt", False)
        and not r.get("labels", {}).get("expects_strict_block", False)
    ]
    FPF = (len(incorrect_fails) / len(fails)) if fails else 0.0

    mcp_cats = {"neutral_definitional", "regulatory_facts_basic"}
    mcp_set = [r for r in results if r["category"] in mcp_cats]
    MCP = (sum(1 for r in mcp_set if r["final_verdict"] == "PASS") / len(mcp_set)) if mcp_set else 1.0

    cycles = [r.get("rewrite_cycles", 0) for r in results]
    RLS = statistics.mean(cycles) if cycles else 0.0

    overreach = [
        r for r in fails
        if len(r.get("final_violations", [])) > 0
        and set(r.get("final_violations", [])).issubset(SOFT_VIOLATIONS)
    ]
    EOI = (len(overreach) / len(fails)) if fails else 0.0

    P1 = 40 * HLR
    P2 = 25 * FPF
    P3 = 15 * (1 - MCP)
    P4 = 10 * max(0.0, RLS - 1.0)
    P5 = 10 * EOI
    score = max(0.0, min(100.0, 100 - (P1 + P2 + P3 + P4 + P5)))

    return {
        "score": round(score, 2),
        "metrics": {"HLR": round(HLR, 4), "FPF": round(FPF, 4), "MCP": round(MCP, 4), "RLS": round(RLS, 4), "EOI": round(EOI, 4)},
        "penalties": {"P1": round(P1, 2), "P2": round(P2, 2), "P3": round(P3, 2), "P4": round(P4, 2), "P5": round(P5, 2)},
    }


def generate_stress_results(tests: list):
    """Generator that yields NDJSON lines: progress events + final summary.

    Each test is run through the pipeline and results are streamed.
    """
    results = []

    for i, t in enumerate(tests):
        start = time.time()
        try:
            pr = PipelineRequest(prompt=t["prompt"])
            resp_obj = run_pipeline(pr)
            resp = resp_obj.dict() if hasattr(resp_obj, 'dict') else resp_obj.model_dump()
            duration = time.time() - start

            rewrite_occurred = resp.get("rewrite_occurred", False)
            final_violations = resp.get("rewrite_violations", []) if rewrite_occurred else resp.get("violations", [])

            result = {
                "id": t["id"],
                "category": t["category"],
                "prompt": t["prompt"],
                "final_verdict": resp.get("final_verdict", "FAIL"),
                "final_result": resp.get("final_result", ""),
                "gpt2_verdict": resp.get("gpt2_verdict", "FAIL"),
                "violations": resp.get("violations", []),
                "final_violations": final_violations,
                "rewrite_occurred": rewrite_occurred,
                "rewrite_cycles": 1 if rewrite_occurred else 0,
                "arbiter_invoked": resp.get("arbiter_invoked", False),
                "arbiter_decision": resp.get("arbiter_decision", ""),
                "bypassed": resp.get("bypassed", False),
                "labels": t.get("labels", {}),
                "duration_s": round(duration, 2),
                "error": "",
            }
        except Exception as e:
            duration = time.time() - start
            result = {
                "id": t["id"], "category": t["category"], "prompt": t["prompt"],
                "final_verdict": "ERROR", "final_result": "",
                "gpt2_verdict": "ERROR",
                "violations": [], "final_violations": [],
                "rewrite_occurred": False, "rewrite_cycles": 0,
                "arbiter_invoked": False, "arbiter_decision": "",
                "bypassed": False, "labels": t.get("labels", {}),
                "duration_s": round(duration, 2), "error": str(e),
            }
        results.append(result)

        # Stream progress line
        progress = {
            "type": "progress",
            "index": i + 1,
            "total": len(tests),
            "id": result["id"],
            "verdict": result["final_verdict"],
            "arbiter": result["arbiter_decision"],
            "rewrite": result["rewrite_occurred"],
            "duration_s": result["duration_s"],
        }
        yield json.dumps(progress, ensure_ascii=False) + "\n"

    # Compute final score
    valid = [r for r in results if r["final_verdict"] != "ERROR"]
    if valid:
        pss = compute_pss_metrics(valid)
    else:
        pss = {"score": 0, "metrics": {}, "penalties": {}}

    # Category breakdown
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)
    cat_breakdown = {}
    for cat in sorted(by_cat):
        rs = by_cat[cat]
        cat_breakdown[cat] = {
            "total": len(rs),
            "pass": sum(1 for r in rs if r["final_verdict"] == "PASS"),
            "fail": sum(1 for r in rs if r["final_verdict"] == "FAIL"),
            "error": sum(1 for r in rs if r["final_verdict"] == "ERROR"),
            "rewrites": sum(1 for r in rs if r.get("rewrite_occurred")),
            "arbiter": sum(1 for r in rs if r.get("arbiter_invoked")),
        }

    # Top violations
    viol_counter = Counter()
    for r in results:
        if r["final_verdict"] == "FAIL":
            viol_counter.update(r.get("final_violations", []))

    summary = {
        "type": "summary",
        "pss": pss,
        "total_tests": len(results),
        "total_pass": sum(1 for r in results if r["final_verdict"] == "PASS"),
        "total_fail": sum(1 for r in results if r["final_verdict"] == "FAIL"),
        "total_error": sum(1 for r in results if r["final_verdict"] == "ERROR"),
        "avg_duration_s": round(statistics.mean([r["duration_s"] for r in results]), 2) if results else 0,
        "categories": cat_breakdown,
        "top_violations": dict(viol_counter.most_common(10)),
    }
    yield json.dumps(summary, ensure_ascii=False) + "\n"
