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

# Tier-specific severity maps — keys absent from a tier's map default to "soft".
_TIER_SEVERITY = {
    "strict": {
        "Fabricated statistic": "hard",
        "Fabricated citation": "hard",
        "False legal conclusion": "hard",
        "Evidence instantiation": "hard",
        "Causal claim as fact": "hard",
        "Unverified current fact": "hard",
        "T1": "hard", "T2": "hard", "T3": "hard", "T7": "hard",
    },
    "standard": {
        "Fabricated statistic": "hard",
        "Fabricated citation": "hard",
        "False legal conclusion": "hard",
        "Evidence instantiation": "hard",
        "Unverified current fact": "hard",
        "Causal claim as fact": "soft",
        "T1": "hard", "T2": "soft", "T3": "soft", "T7": "hard",
    },
    "light": {
        "Fabricated statistic": "hard",
        "Fabricated citation": "hard",
        "False legal conclusion": "hard",
        "Evidence instantiation": "hard",
        "Causal claim as fact": "soft",
        "Unverified current fact": "soft",
        "T1": "hard", "T2": "soft", "T3": "soft", "T7": "soft",
    },
}

# Soft-finding threshold per tier: soft_count >= threshold → FAIL
_SOFT_THRESHOLD = {"strict": 3, "standard": 4, "light": 5}

# Finding types to skip entirely per tier
_SKIP_TYPES = {
    "strict": set(),
    "standard": set(),
    "light": {"T5", "T6", "Prescriptive creep", "Prescriptive violation", "Reassurance framing"},
}

# All finding types that are context-filterable when advice is requested
_PRESCRIPTIVE_TYPES = {"Prescriptive creep", "T5", "Prescriptive violation"}


def _all_soft(findings: List[dict]) -> bool:
    """Return True if every finding has severity == 'soft'."""
    return len(findings) > 0 and all(f.get("severity") == "soft" for f in findings)


def parse_gpt2(raw: str, flags: Optional[dict] = None, tier: str = "strict"):
    """Parse GPT-2 JSON output into claim_table, findings, violations, verdict, reasoning, arbiter.

    When *flags* is provided:
    - advice_requested=True: soft prescriptive findings without outcome-promise
      language are filtered out.
    - jurisdiction_present=True: "Missing jurisdiction" findings are filtered out.

    The *tier* parameter gates severity classification and soft-finding thresholds:
    - strict: current behavior (T1/T2/T3/T7 hard, 3+ soft = FAIL)
    - standard: T3 soft, T2 soft always, threshold 4+
    - light: only T1 hard, T5/T6 skipped, threshold 5+

    Returns 6 values: (claim_table, violations, verdict, findings, reasoning_trace, arbiter_result)

    arbiter_result is None when verdict=PASS, or a dict with keys:
      decision, rationale, edits, policy_notes
    when verdict=FAIL and GPT-2 included arbiter fields.
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

        # Resolve severity map and skip-set for the active tier
        severity_map = _TIER_SEVERITY.get(tier, _TIER_SEVERITY["strict"])
        skip_types = _SKIP_TYPES.get(tier, set())

        findings = []  # type: List[dict]
        for f in raw_findings:
            ftype = f.get("type", "")
            severity = f.get("severity", "soft").lower()
            detail = f.get("detail", "")

            # Skip finding types not applicable in this tier
            if ftype in skip_types:
                continue

            # Override severity for known types in this tier's map;
            # for unknown types, keep GPT-2's self-reported severity
            if ftype in severity_map:
                severity = severity_map[ftype]

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

        # Recompute verdict based on tier-specific severity rules
        hard_count = sum(1 for f in findings if f["severity"] == "hard")
        soft_count = sum(1 for f in findings if f["severity"] == "soft")
        soft_threshold = _SOFT_THRESHOLD.get(tier, 3)
        if hard_count > 0 or soft_count >= soft_threshold:
            verdict = "FAIL"
        else:
            verdict = "PASS"

        # ---------- arbiter fields (merged into GPT-2 output) ----------
        arbiter_result = None
        if verdict == "FAIL":
            arbiter_decision = parsed.get("arbiter_decision", "").upper()
            if arbiter_decision in ("BLOCK", "ALLOW_WITH_EDITS", "ALLOW_AS_UNKNOWN_ONLY"):
                from pipeline.models import EditEntry
                edits_raw = parsed.get("edits_for_gpt1", [])
                edits = [
                    EditEntry(
                        action=e.get("action", ""),
                        target=e.get("target", ""),
                        replacement=e.get("replacement", ""),
                    )
                    for e in edits_raw
                ]
                arbiter_result = {
                    "decision": arbiter_decision,
                    "rationale": parsed.get("rationale", []),
                    "edits": edits,
                    "policy_notes": parsed.get("final_policy_notes", []),
                }
            else:
                # GPT-2 didn't include arbiter fields — default to ALLOW_WITH_EDITS
                arbiter_result = {
                    "decision": "ALLOW_WITH_EDITS",
                    "rationale": ["GPT-2 found violations; attempting auto-repair."],
                    "edits": [],
                    "policy_notes": [],
                }

        return claim_table, violations, verdict, findings, reasoning_trace, arbiter_result
    except Exception:
        return (
            [],
            ["GPT-2 parse error: could not extract valid JSON from response"],
            "FAIL",
            [{"type": "GPT-2 parse error", "severity": "hard", "detail": "could not extract valid JSON"}],
            [],
            {"decision": "ALLOW_WITH_EDITS", "rationale": ["GPT-2 output could not be parsed; attempting auto-repair."], "edits": [], "policy_notes": []},
        )
