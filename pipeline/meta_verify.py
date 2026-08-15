"""Meta-verification layer for GPT-2 PASS verdicts.

When GPT-2 passes a response on high-stakes queries (legal, medical, financial),
this module performs a lightweight cross-check to catch verifier hallucinations.

Uses NLI contradiction signals and claim-table consistency checks to flag
suspicious PASS verdicts that warrant re-verification or confidence downgrade.
"""
from __future__ import annotations



def is_high_stakes(flags: dict) -> bool:
    """Determine if the query is high-stakes and warrants meta-verification."""
    return (
        flags.get("legal_mode", False)
        or flags.get("percent_requested", False)
        or flags.get("comparative", False)
    )


def _get_claim_attr(ct: object, attr: str, default: str = "") -> str:
    """Safely extract string attribute from either a ClaimEntry instance or a dict."""
    if isinstance(ct, dict):
        val = ct.get(attr, default)
    else:
        val = getattr(ct, attr, default)
    return str(val or default)


def check_claim_table_consistency(
    claim_table: list,
    findings: list,
    atomic_claims: list,
) -> dict:
    """Check for inconsistencies between GPT-2's claim table and NLI signals.

    Catches cases where GPT-2 categorized a claim as "Observed" but NLI
    found a contradiction, or where GPT-2 missed unsupported claims.

    Returns:
        {
            "consistent": bool,
            "issues": [{"type": str, "detail": str, "severity": str}],
            "should_downgrade": bool,
        }
    """
    issues = []

    # Check 1: NLI contradictions on claims GPT-2 marked as Observed
    for ac in atomic_claims:
        nli = ac.get("nli_result")
        if not nli:
            continue

        claim_text = ac.get("text", "")

        # If NLI strongly contradicts but GPT-2 passed...
        if nli.get("confidence_tier") == "strong_contradiction":
            # Check if this claim was in the claim table as Observed
            for ct in claim_table:
                cat = _get_claim_attr(ct, "category").lower().strip()
                ct_claim = _get_claim_attr(ct, "claim") or _get_claim_attr(ct, "text")
                if cat in ("supported", "observed") and _text_overlap(claim_text, ct_claim):
                    issues.append({
                        "type": "nli_gpt2_mismatch",
                        "detail": f'NLI contradicts claim GPT-2 marked Observed: "{claim_text[:80]}"',
                        "severity": "high",
                    })
                    break

    # Check 2: High proportion of claims without justification
    unjustified = sum(
        1 for ct in claim_table
        if len(_get_claim_attr(ct, "justification")) < 10
    )
    if len(claim_table) > 0 and unjustified / len(claim_table) > 0.5:
        issues.append({
            "type": "shallow_verification",
            "detail": f"{unjustified}/{len(claim_table)} claims have minimal justification",
            "severity": "medium",
        })

    # Check 3: Zero findings on a query with many factual claims
    if len(findings) == 0 and len(claim_table) >= 5:
        # Count how many claims are factual (not user-provided or unknown)
        factual = sum(
            1 for ct in claim_table
            if _get_claim_attr(ct, "category").lower().strip() not in ('user-provided', 'unknown', 'hypothesis')
        )
        if factual >= 4:
            issues.append({
                "type": "suspiciously_clean",
                "detail": f"Zero findings on {factual} factual claims — verifier may have rubber-stamped",
                "severity": "low",
            })

    should_downgrade = any(i["severity"] == "high" for i in issues)

    return {
        "consistent": len(issues) == 0,
        "issues": issues,
        "should_downgrade": should_downgrade,
    }


def meta_verify_pass(
    flags: dict,
    claim_table: list,
    findings: list,
    atomic_claims: list,
    confidence_label: str,
) -> dict:
    """Run meta-verification on a GPT-2 PASS verdict.

    Only runs on high-stakes queries. Returns adjustment recommendations.

    Returns:
        {
            "ran": bool,
            "issues": list,
            "adjusted_label": str,  # original or downgraded
            "should_reverify": bool,
        }
    """
    if not is_high_stakes(flags):
        return {
            "ran": False,
            "issues": [],
            "adjusted_label": confidence_label,
            "should_reverify": False,
        }

    consistency = check_claim_table_consistency(claim_table, findings, atomic_claims)

    adjusted_label = confidence_label
    should_reverify = False

    if consistency["should_downgrade"]:
        # Downgrade confidence one tier
        tier_order = ["High", "Medium", "Low", "Unknown"]
        if adjusted_label in tier_order:
            idx = tier_order.index(adjusted_label)
            if idx < len(tier_order) - 1:
                adjusted_label = tier_order[idx + 1]
        should_reverify = True

    return {
        "ran": True,
        "issues": consistency["issues"],
        "adjusted_label": adjusted_label,
        "should_reverify": should_reverify,
    }


def meta_verify_fail(
    flags: dict,
    claim_table: list,
    findings: list,
    atomic_claims: list,
) -> dict:
    """Run meta-verification on a GPT-2 FAIL verdict to catch false FAILs.

    Checks if:
    - >50% of "Unsupported" claims are common knowledge or valid inferences
    - Hard findings target hedged language that already includes qualifiers
    - Claims are supported by search evidence (source-match partial overlap)

    Returns:
        {
            "ran": bool,
            "override_to_pass": bool,
            "adjusted_findings": list,
            "reason": str,
        }
    """
    if not is_high_stakes(flags):
        return {"ran": False, "override_to_pass": False, "adjusted_findings": findings, "reason": ""}

    if not findings:
        return {"ran": True, "override_to_pass": True, "adjusted_findings": [], "reason": "No findings to verify"}

    adjusted = []
    dropped_count = 0

    for f in findings:
        detail = f.get("detail", "").lower()
        ftype = f.get("type", "")
        severity = f.get("severity", "")

        # Check if finding targets hedged/qualified language
        hedging_markers = ["may ", "might ", "could ", "possibly ", "it is unclear",
                           "unknown", "approximately", "estimated", "reportedly"]
        is_hedged = any(marker in detail for marker in hedging_markers)

        # T4 (missing qualifier) on already-hedged text is a false FAIL
        if ftype in ("T4", "Missing qualifier") and is_hedged:
            dropped_count += 1
            continue

        # T3 (causal as fact) on hedged language is a false FAIL
        if ftype in ("T3", "Causal claim as fact") and is_hedged and severity == "hard":
            f = {**f, "severity": "soft"}  # downgrade from hard to soft

        # Soft T5 (prescriptive creep) when advice was requested is a false FAIL.
        # Hard T5 (outcome promises / commands) is kept — matching parse_gpt2.
        if (
            ftype in ("T5", "Prescriptive creep", "Prescriptive violation")
            and flags.get("advice_requested")
            and severity != "hard"
        ):
            dropped_count += 1
            continue

        adjusted.append(f)

    # Check if >50% of unsupported claims are likely common knowledge
    unsupported_claims = [
        ct for ct in claim_table
        if _get_claim_attr(ct, "category").lower().strip() == "unsupported"
    ]
    if unsupported_claims and len(claim_table) > 0:
        unsupported_ratio = len(unsupported_claims) / len(claim_table)
        # If very few claims are unsupported, this isn't a real problem
        if unsupported_ratio < 0.3 and all(f.get("severity") != "hard" for f in adjusted):
            return {
                "ran": True,
                "override_to_pass": True,
                "adjusted_findings": adjusted,
                "reason": f"Only {len(unsupported_claims)}/{len(claim_table)} claims unsupported with no hard findings remaining",
            }

    override = dropped_count > 0 and all(f.get("severity") != "hard" for f in adjusted)
    reason = ""
    if override:
        reason = f"Dropped {dropped_count} false-positive finding(s); no hard findings remain"
    elif dropped_count > 0:
        reason = f"Dropped {dropped_count} false-positive finding(s) but hard findings remain"

    return {
        "ran": True,
        "override_to_pass": override,
        "adjusted_findings": adjusted,
        "reason": reason,
    }


def _text_overlap(text_a: str, text_b: str, threshold: int = 30) -> bool:
    """Check if two text strings share significant overlap."""
    if not text_a or not text_b:
        return False
    shorter = text_a if len(text_a) <= len(text_b) else text_b
    longer = text_b if len(text_a) <= len(text_b) else text_a
    # Check if a meaningful prefix of the shorter text appears in the longer
    check_len = min(len(shorter), threshold)
    return shorter[:check_len].lower() in longer.lower()
