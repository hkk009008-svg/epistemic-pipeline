"""GPT-2 Verifier: parses verification output and recomputes verdicts.

Supports Audit v7 tripwire types (T1-T7) and legacy finding type names.
Provides both legacy text parsing (parse_gpt2) and structured output
parsing (parse_gpt2_structured) for the V4 async pipeline.
"""
from __future__ import annotations

import json
import re
from typing import List, Optional

from pipeline.helpers import extract_json
from pipeline.models import ClaimEntry, GPT2ResponseSchema
from pipeline.prompts import GPT2_TRIPWIRE_REFERENCE

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

# Map legacy finding type names to Audit v7 tripwire codes.
# Used for severity, skip-sets, and confidence weights. Output type strings
# stay as the model reported them so API consumers and tests keep working.
# "Unsupported evidence reference" is intentionally unmapped: it is a legacy
# soft label, not T2 (typicality).
_FINDING_TYPE_ALIASES = {
    "Fabricated statistic": "T1",
    "Fabricated citation": "T1",
    "False legal conclusion": "T1",
    "Evidence instantiation": "T1",
    "Typicality violation": "T2",
    "Causal claim as fact": "T3",
    "Ranking violation": "T4",
    "Prescriptive creep": "T5",
    "Prescriptive violation": "T5",
    "Reassurance framing": "T6",
    "Unverified current fact": "T7",
}

# Tier-specific severity maps. Unknown types keep GPT-2's self-reported
# severity. T4/T5/T6 are always soft under Audit v7 even if the model
# marks them hard — otherwise a single stylistic finding fails the run.
_SOFT_TRIPWIRES = {
    "T4": "soft", "T5": "soft", "T6": "soft",
    "Ranking violation": "soft",
    "Prescriptive creep": "soft",
    "Prescriptive violation": "soft",
    "Reassurance framing": "soft",
}
_TIER_SEVERITY = {
    "strict": {
        "Fabricated statistic": "hard",
        "Fabricated citation": "hard",
        "False legal conclusion": "hard",
        "Evidence instantiation": "hard",
        "Causal claim as fact": "hard",
        "Unverified current fact": "hard",
        "Typicality violation": "hard",
        "T1": "hard", "T2": "hard", "T3": "hard", "T7": "hard",
        **_SOFT_TRIPWIRES,
    },
    "standard": {
        "Fabricated statistic": "hard",
        "Fabricated citation": "hard",
        "False legal conclusion": "hard",
        "Evidence instantiation": "hard",
        "Unverified current fact": "hard",
        "Causal claim as fact": "soft",
        "Typicality violation": "soft",
        "T1": "hard", "T2": "soft", "T3": "soft", "T7": "hard",
        **_SOFT_TRIPWIRES,
    },
    "light": {
        "Fabricated statistic": "hard",
        "Fabricated citation": "hard",
        "False legal conclusion": "hard",
        "Evidence instantiation": "hard",
        "Causal claim as fact": "soft",
        "Unverified current fact": "soft",
        "Typicality violation": "soft",
        "T1": "hard", "T2": "soft", "T3": "soft", "T7": "soft",
        **_SOFT_TRIPWIRES,
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


def canonical_finding_type(ftype: str) -> str:
    """Return the Audit v7 T-code for a finding type, or the original label."""
    return _FINDING_TYPE_ALIASES.get(ftype, ftype)


def _all_soft(findings: List[dict]) -> bool:
    """Return True if every finding has severity == 'soft'."""
    return len(findings) > 0 and all(f.get("severity") == "soft" for f in findings)


def recompute_verdict(findings: List[dict], tier: str = "strict") -> str:
    """Recompute PASS/FAIL verdict from findings using tier-specific thresholds."""
    hard_count = sum(1 for f in findings if f.get("severity") == "hard")
    soft_count = sum(1 for f in findings if f.get("severity") == "soft")
    soft_threshold = _SOFT_THRESHOLD.get(tier, 3)
    if hard_count > 0 or soft_count >= soft_threshold:
        return "FAIL"
    return "PASS"


def _process_findings(
    raw_findings: list,
    flags: Optional[dict] = None,
    tier: str = "strict",
) -> List[dict]:
    """Normalize, filter, and assign tier-aware severity for GPT-2 findings.

    Shared by parse_gpt2 and parse_gpt2_structured so the two parsers cannot
    drift. Reported type strings are preserved; aliases are used only for
    skip/severity decisions.
    """
    severity_map = _TIER_SEVERITY.get(tier, _TIER_SEVERITY["strict"])
    skip_types = _SKIP_TYPES.get(tier, set())

    findings: List[dict] = []
    for f in raw_findings:
        ftype = f.get("type", "")
        canonical = canonical_finding_type(ftype)
        severity = f.get("severity", "soft").lower()
        detail = f.get("detail", "")

        if ftype in skip_types or canonical in skip_types:
            continue

        if canonical in severity_map:
            severity = severity_map[canonical]
        elif ftype in severity_map:
            severity = severity_map[ftype]

        if (
            flags
            and flags.get("advice_requested")
            and (ftype in _PRESCRIPTIVE_TYPES or canonical in _PRESCRIPTIVE_TYPES)
            and severity == "soft"
            and not _OUTCOME_KW.search(detail)
        ):
            continue

        if (
            flags
            and flags.get("jurisdiction_present")
            and ftype in ("Missing jurisdiction",)
        ):
            continue

        findings.append({"type": ftype, "severity": severity, "detail": detail})
    return findings


def build_gpt2_user_content(
    prompt: str,
    text_to_verify: str,
    atomic_claims: list | None = None,
    nli_grounding: dict | None = None,
) -> str:
    """Build GPT-2 user content with the tripwire reference first.

    Placing the tripwire block at the start of user content avoids
    lost-in-the-middle attention drop on long generator outputs.
    Atomic claims / NLI signals are included only when provided — rewrite
    re-verification must not attach claims decomposed from a prior draft.
    """
    claims_block = ""
    if atomic_claims:
        claims_json = json.dumps(atomic_claims, indent=2)
        nli_lines = []
        for c in atomic_claims:
            nli = c.get("nli_result") or {}
            nli_tier = nli.get("confidence_tier", "")
            text = str(c.get("text", ""))[:80]
            if nli_tier == "strong_support":
                nli_lines.append(
                    f'  NLI-STRONG-SUPPORT (ent={nli["best_entailment"]:.2f}): "{text}"'
                )
            elif nli_tier == "weak_support":
                nli_lines.append(
                    f'  NLI-WEAK-SUPPORT (ent={nli["best_entailment"]:.2f}): "{text}"'
                )
            elif nli_tier == "strong_contradiction":
                nli_lines.append(
                    f'  NLI-CONTRADICTED (con={nli["worst_contradiction"]:.2f}): "{text}"'
                )
            elif nli_tier == "weak_contradiction":
                nli_lines.append(
                    f'  NLI-WEAK-CONTRADICTION (con={nli["worst_contradiction"]:.2f}): "{text}"'
                )
        nli_block = ""
        if nli_lines:
            grounding_str = ""
            if nli_grounding:
                grounding_str = (
                    f"\nGrounding Rate: {nli_grounding['grounding_rate']:.1%} "
                    f"({nli_grounding['grounded_count']}/{nli_grounding['total_evaluated']} claims grounded)"
                )
            nli_block = (
                "\n\nNLI PRE-VERIFICATION SIGNALS:\n" + "\n".join(nli_lines) + grounding_str
            )
        claims_block = (
            f"\n\nPRE-DECOMPOSED ATOMIC CLAIMS (verify each independently):\n{claims_json}"
            f"{nli_block}"
        )

    return (
        f"{GPT2_TRIPWIRE_REFERENCE}\n\n"
        f"=== TASK ===\n"
        f"ORIGINAL PROMPT:\n{prompt}\n\n"
        f"GPT-1 RESPONSE TO VERIFY:\n{text_to_verify}"
        f"{claims_block}"
    )


def parse_gpt2(raw: str, flags: Optional[dict] = None, tier: str = "strict"):
    """Parse GPT-2 JSON output into claim_table, findings, violations, verdict, reasoning.

    When *flags* is provided:
    - advice_requested=True: soft prescriptive findings without outcome-promise
      language are filtered out.
    - jurisdiction_present=True: "Missing jurisdiction" findings are filtered out.

    The *tier* parameter gates severity classification and soft-finding thresholds:
    - strict: current behavior (T1/T2/T3/T7 hard, 3+ soft = FAIL)
    - standard: T3 soft, T2 soft always, threshold 4+
    - light: only T1 hard, T5/T6 skipped, threshold 5+

    Returns 5 values: (claim_table, violations, verdict, findings, reasoning_trace)

    Arbiter logic is handled separately by GPT-3 via pipeline.arbiter.parse_gpt3().
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

        findings = _process_findings(raw_findings, flags=flags, tier=tier)
        violations = [f["type"] for f in findings]
        verdict = recompute_verdict(findings, tier=tier)

        return claim_table, violations, verdict, findings, reasoning_trace
    except Exception:
        return (
            [],
            ["GPT-2 parse error: could not extract valid JSON from response"],
            "FAIL",
            [{"type": "GPT-2 parse error", "severity": "hard", "detail": "could not extract valid JSON"}],
            [],
        )


def parse_gpt2_structured(
    parsed: GPT2ResponseSchema,
    flags: Optional[dict] = None,
    tier: str = "strict",
):
    """Parse a structured GPT-2 response (Pydantic model) into the same 5-tuple.

    This is the V4 equivalent of parse_gpt2() — it skips JSON extraction
    since the response is already validated by the structured output API.

    Returns 5 values: (claim_table, violations, verdict, findings, reasoning_trace)
    """
    try:
        reasoning_trace = [str(s) for s in parsed.reasoning_trace]

        claim_table = [
            ClaimEntry(
                claim=c.get("claim", ""),
                category=c.get("category", "Unknown"),
                justification=c.get("justification", ""),
            )
            for c in parsed.claim_table
        ]

        raw_findings = [
            {"type": f.type, "severity": f.severity, "detail": f.detail}
            for f in parsed.findings
        ]

        findings = _process_findings(raw_findings, flags=flags, tier=tier)
        violations = [f["type"] for f in findings]
        verdict = recompute_verdict(findings, tier=tier)

        return claim_table, violations, verdict, findings, reasoning_trace
    except Exception:
        return (
            [],
            ["GPT-2 structured parse error"],
            "FAIL",
            [{"type": "GPT-2 parse error", "severity": "hard", "detail": "structured parse failed"}],
            [],
        )
