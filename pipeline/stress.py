"""Pipeline Stability Score (PSS) computation and stress test logic."""
from __future__ import annotations

import json
import re
import statistics
import time
from collections import Counter, defaultdict

from pipeline.models import PipelineRequest
from pipeline.orchestrator import run_pipeline

# "Soft" violations -- procedural, not fabrication-level (Audit v7 T4-T6 + legacy)
SOFT_VIOLATIONS = {
    "Prescriptive creep", "Unsupported evidence reference", "Missing jurisdiction",
    "T4", "T5", "T6",
    "Ranking violation", "Prescriptive violation", "Reassurance framing",
    "Overconfidence", "Unacknowledged conflict",
}

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


_HEARTBEAT_INTERVAL = 15  # seconds between keepalive heartbeats


_TEST_MAX_RETRIES = 2  # Retry a failed test once before recording ERROR
_TEST_RETRY_DELAY = 3  # Seconds between test retries


def _run_single_test(t: dict) -> dict:
    """Run a single test case through the pipeline with retry on transient errors.

    Returns a result dict suitable for PSS computation and progress reporting.
    Retries up to _TEST_MAX_RETRIES times on transient API errors before
    recording the test as ERROR.
    """
    from pipeline.helpers import PipelineError

    last_error = ""
    for attempt in range(_TEST_MAX_RETRIES + 1):
        start = time.time()
        try:
            pr = PipelineRequest(prompt=t["prompt"])
            resp_obj = run_pipeline(pr)
            resp = resp_obj.dict() if hasattr(resp_obj, 'dict') else resp_obj.model_dump()
            duration = time.time() - start

            rewrite_occurred = resp.get("rewrite_occurred", False)
            final_violations = resp.get("rewrite_violations", []) if rewrite_occurred else resp.get("violations", [])

            return {
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
        except PipelineError as e:
            duration = time.time() - start
            last_error = str(e)
            # Retry on transient server/provider errors
            if e.status_code in (429, 502, 503, 504) and attempt < _TEST_MAX_RETRIES:
                time.sleep(_TEST_RETRY_DELAY * (attempt + 1))
                continue
        except Exception as e:
            duration = time.time() - start
            last_error = str(e)
            # Retry on connection-style errors
            if attempt < _TEST_MAX_RETRIES and _is_transient_test_error(e):
                time.sleep(_TEST_RETRY_DELAY * (attempt + 1))
                continue

    # All retries exhausted — record as ERROR
    return {
        "id": t["id"], "category": t["category"], "prompt": t["prompt"],
        "final_verdict": "ERROR", "final_result": "",
        "gpt2_verdict": "ERROR",
        "violations": [], "final_violations": [],
        "rewrite_occurred": False, "rewrite_cycles": 0,
        "arbiter_invoked": False, "arbiter_decision": "",
        "bypassed": False, "labels": t.get("labels", {}),
        "duration_s": round(duration, 2), "error": last_error,
    }


def _is_transient_test_error(exc: Exception) -> bool:
    """Return True if the exception looks transient and worth retrying."""
    msg = str(exc).lower()
    return any(kw in msg for kw in ("timeout", "connection", "rate limit", "503", "502", "504"))


def generate_stress_results(tests: list):
    """Generator that yields NDJSON lines: progress events + final summary.

    Each test is run through the pipeline and results are streamed.
    Emits periodic heartbeat lines to keep the connection alive during
    long-running LLM calls (prevents proxy idle-timeout disconnects).
    """
    import threading

    results = []

    for i, t in enumerate(tests):
        # Run each test in a background thread so we can emit heartbeats
        result_box: list = []

        def _worker(test_case=t):
            result_box.append(_run_single_test(test_case))

        worker = threading.Thread(target=_worker, daemon=True)
        worker.start()

        while worker.is_alive():
            worker.join(timeout=_HEARTBEAT_INTERVAL)
            if worker.is_alive():
                yield json.dumps({"type": "heartbeat"}) + "\n"

        result = result_box[0]
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
            "error": result.get("error", ""),
        }
        yield json.dumps(progress, ensure_ascii=False) + "\n"

    # Compute final score
    valid = [r for r in results if r["final_verdict"] != "ERROR"]
    if valid:
        pss = compute_pss_metrics(valid)
    else:
        pss = {
            "score": 0,
            "metrics": {"HLR": 0, "FPF": 0, "MCP": 0, "RLS": 0, "EOI": 0},
            "penalties": {"P1": 0, "P2": 0, "P3": 0, "P4": 0, "P5": 0},
        }

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
