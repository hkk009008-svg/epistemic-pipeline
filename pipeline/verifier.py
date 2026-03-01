"""GPT-2 Verifier: parses verification output and recomputes verdicts.

Supports Audit v7 tripwire types (T1-T7) and legacy finding type names.
"""
from __future__ import annotations

import re
from typing import List, Optional

from pipeline.models import ClaimEntry
from pipeline.helpers import extract_json

# Module-level compiled regex for outcome-promise keywords
_OUTCOME_KW = re.compile(
    r"(?i)\b(?:will improve|will reduce|will increase|improve your|"
    r"could help|could assist|may improve|may help|could potentially|"
    r"guarantee|ensure|succeed)\b"
)

# Map legacy finding type names to correct hard/soft severity
_LEGACY_SEVERITY = {
    "Fabricated statistic": "hard",
    "Fabricated citation": "hard",
    "False legal conclusion": "hard",
    "Evidence instantiation": "hard",
    "Causal claim as fact": "hard",
    "Unverified current fact": "hard",
}

# Map legacy finding type names to Audit v7 tripwire codes
_LEGACY_TO_TRIPWIRE = {
    "Fabricated statistic": "T1",
    "Fabricated citation": "T1",
    "False legal conclusion": "T1",
    "Unsupported evidence reference": "T2",
    "Prescriptive creep": "T5",
    "Overconfidence": "Overconfidence",
    "Missing jurisdiction": "Missing jurisdiction",
}

# All finding types that are context-filterable when advice is requested
_PRESCRIPTIVE_TYPES = {"Prescriptive creep", "T5", "Prescriptive violation"}


def _all_soft(findings: List[dict]) -> bool:
    """Return True if every finding has severity == 'soft'."""
    return len(findings) > 0 and all(f.get("severity") == "soft" for f in findings)


def parse_gpt2(raw: str, flags: Optional[dict] = None):
    """Parse GPT-2 JSON output into claim_table, findings, violations, verdict, reasoning.

    When *flags* is provided:
    - advice_requested=True: soft prescriptive findings without outcome-promise
      language are filtered out.
    - jurisdiction_present=True: "Missing jurisdiction" findings are filtered out.

    Returns 5 values: (claim_table, violations, verdict, findings, reasoning_trace)
    """
    try:
        parsed = extract_json(raw)

        # Extract reasoning trace (chain-of-thought)
        reasoning_trace = parsed.get("reasoning_trace", [])
        if not isinstance(reasoning_trace, list):
            reasoning_trace = []
        reasoning_trace = [str(s) for s in reasoning_trace]

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
                {"type": v, "severity": _LEGACY_SEVERITY.get(v, "soft"), "detail": v}
                for v in parsed["violations"]
            ]

        findings = []  # type: List[dict]
        for f in raw_findings:
            ftype = f.get("type", "")
            severity = f.get("severity", "soft").lower()
            detail = f.get("detail", "")

            # Override severity for known hard types (in case GPT-2 misclassified)
            if ftype in _LEGACY_SEVERITY:
                severity = _LEGACY_SEVERITY[ftype]

            # Context-aware filter: if advice was requested, drop soft
            # prescriptive findings unless they contain outcome promises.
            if (
                flags
                and flags.get("advice_requested")
                and ftype in _PRESCRIPTIVE_TYPES
                and severity == "soft"
                and not _OUTCOME_KW.search(detail)
            ):
                continue

            # Context-aware filter: if jurisdiction is present, drop
            # "Missing jurisdiction" findings.
            if (
                flags
                and flags.get("jurisdiction_present")
                and ftype in ("Missing jurisdiction",)
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

        return claim_table, violations, verdict, findings, reasoning_trace
    except Exception:
        return (
            [],
            ["GPT-2 parse error: could not extract valid JSON from response"],
            "FAIL",
            [{"type": "GPT-2 parse error", "severity": "hard", "detail": "could not extract valid JSON"}],
            [],
        )
