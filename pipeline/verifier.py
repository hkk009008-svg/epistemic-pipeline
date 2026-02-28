"""GPT-2 Verifier: parses verification output and recomputes verdicts."""
from __future__ import annotations

import re
from typing import List, Optional

from pipeline.models import ClaimEntry
from pipeline.helpers import extract_json


def _all_soft(findings: List[dict]) -> bool:
    """Return True if every finding has severity == 'soft'."""
    return len(findings) > 0 and all(f.get("severity") == "soft" for f in findings)


def parse_gpt2(raw: str, flags: Optional[dict] = None):
    """Parse GPT-2 JSON output into claim_table, findings, violations, verdict.

    When *flags* is provided and ``advice_requested`` is True, soft
    "Prescriptive creep" findings that do NOT contain outcome-promise
    language are filtered out before the verdict is recalculated.
    """
    try:
        parsed = extract_json(raw)
        claim_table = [
            ClaimEntry(
                claim=c.get("claim", ""),
                category=c.get("category", "Unknown"),
                justification=c.get("justification", ""),
            )
            for c in parsed.get("claim_table", [])
        ]

        # ---------- findings (new schema) ----------
        raw_findings = parsed.get("findings", [])

        # Backward compat: if GPT-2 returned old "violations" list instead
        if not raw_findings and parsed.get("violations"):
            raw_findings = [
                {"type": v, "severity": "soft", "detail": v}
                for v in parsed["violations"]
            ]

        findings = []  # type: List[dict]
        _OUTCOME_KW = re.compile(
            r"(?i)\b(?:will improve|will reduce|will increase|improve your|"
            r"could help|could assist|may improve|may help|could potentially|"
            r"guarantee|ensure|succeed)\b"
        )
        for f in raw_findings:
            ftype = f.get("type", "")
            severity = f.get("severity", "soft").lower()
            detail = f.get("detail", "")

            # Context-aware filter: if advice was requested, drop soft
            # "Prescriptive creep" unless it contains outcome promises.
            if (
                flags
                and flags.get("advice_requested")
                and ftype == "Prescriptive creep"
                and severity == "soft"
                and not _OUTCOME_KW.search(detail)
            ):
                continue

            findings.append({"type": ftype, "severity": severity, "detail": detail})

        # Derive violations list (backward compat)
        violations = [f["type"] for f in findings]

        # Recompute verdict based on severity-tier rule
        hard_count = sum(1 for f in findings if f["severity"] == "hard")
        soft_count = sum(1 for f in findings if f["severity"] == "soft")
        if hard_count > 0 or soft_count >= 3:
            verdict = "FAIL"
        else:
            verdict = "PASS"

        return claim_table, violations, verdict, findings
    except Exception:
        return (
            [],
            ["GPT-2 parse error: could not extract valid JSON from response"],
            "FAIL",
            [{"type": "GPT-2 parse error", "severity": "hard", "detail": "could not extract valid JSON"}],
        )
