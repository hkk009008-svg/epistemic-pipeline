"""GPT-3 Arbiter: parses arbiter decisions and applies edits.

Provides both legacy text parsing (parse_gpt3) and structured output
parsing (parse_gpt3_structured) for the V4 async pipeline.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from pipeline.models import EditEntry, GPT3ResponseSchema
from pipeline.helpers import extract_json

VALID_ARBITER_DECISIONS = frozenset({
    "BLOCK",
    "ALLOW_WITH_EDITS",
    "ALLOW_AS_UNKNOWN_ONLY",
})


def _as_str_list(value) -> list:
    """Coerce arbiter list fields; a bare string becomes a one-item list."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return value
    return []


def _coerce_decision(raw_decision, rationale: list) -> tuple[str, list]:
    """Fail closed: unknown arbiter decisions become BLOCK."""
    decision = str(raw_decision or "BLOCK").upper()
    if decision in VALID_ARBITER_DECISIONS:
        return decision, rationale
    note = f"Invalid arbiter_decision {raw_decision!r}; failing closed to BLOCK"
    return "BLOCK", [note] + list(rationale)


def parse_gpt3(raw: str):
    """Parse GPT-3 Arbiter JSON output."""
    try:
        parsed = extract_json(raw)
        rationale = _as_str_list(parsed.get("rationale", []))
        decision, rationale = _coerce_decision(
            parsed.get("arbiter_decision", "BLOCK"), rationale,
        )
        edits_raw = parsed.get("edits_for_gpt1", [])
        if not isinstance(edits_raw, list):
            edits_raw = []
        edits = [
            EditEntry(
                action=str(e.get("action", "")).upper(),
                target=e.get("target", ""),
                replacement=e.get("replacement", ""),
                target_id=e.get("target_id", ""),
            )
            for e in edits_raw
        ]
        policy_notes = _as_str_list(parsed.get("final_policy_notes", []))
        return decision, rationale, edits, policy_notes
    except Exception:
        return "BLOCK", ["GPT-3 parse error: could not extract valid JSON"], [], []


def apply_edits(gpt1_output: str, edits: List[EditEntry]) -> str:
    """Apply GPT-3 edit instructions to GPT-1 output for rewrite prompt.

    Supports both legacy text-based targeting and ID-based targeting.
    When target_id is present, includes the claim ID in the instruction
    for more precise targeting.
    """
    instructions = []
    for e in edits:
        action = (e.action or "").upper()
        id_tag = f" [claim_id={e.target_id}]" if getattr(e, "target_id", "") else ""
        if action == "DELETE":
            instructions.append(f'DELETE the following text{id_tag}: "{e.target}"')
        elif action == "REWRITE":
            instructions.append(f'REWRITE{id_tag} "{e.target}" to: "{e.replacement}"')
        elif action == "MOVE_TO_UNKNOWN":
            instructions.append(
                f'MOVE the following to the Unknowns section{id_tag}: "{e.target}" '
                f'\u2014 reframe as: "{e.replacement}"'
            )
    edit_block = "\n".join(f"- {inst}" for inst in instructions)
    return (
        f"You previously produced this response:\n\n"
        f"---\n{gpt1_output}\n---\n\n"
        f"Apply ONLY these edits (do not add new claims, do not change structure beyond what is required):\n"
        f"{edit_block}\n\n"
        f"Output the corrected response in full."
    )


def apply_edits_by_id(
    atomic_claims: List[dict],
    edits: List[EditEntry],
) -> tuple[List[dict], str]:
    """Deterministically apply arbiter edits to atomic claims by UUID.

    This is the V5 ID-based approach: instead of relying on GPT-1 to find
    and replace text, we modify the claim JSON directly. This eliminates
    linguistic fragility from the rewrite loop.

    Args:
        atomic_claims: List of claim dicts with "claim_id" and "text" fields.
        edits: List of EditEntry with target_id set.

    Returns:
        (modified_claims, summary): The edited claim list and a human-readable
        summary of what changed (for logging/metrics).
    """
    # Build a lookup by claim_id
    claims_by_id = {c.get("claim_id", ""): c for c in atomic_claims if c.get("claim_id")}
    applied = []
    modified_claims = list(atomic_claims)

    for e in edits:
        target_id = getattr(e, "target_id", "")
        if not target_id or target_id not in claims_by_id:
            # Fall back to text-based matching if no ID or ID not found
            continue

        claim = claims_by_id[target_id]
        idx = next((i for i, c in enumerate(modified_claims) if c.get("claim_id") == target_id), None)
        if idx is None:
            continue

        action = (e.action or "").upper()
        claim_text = claim.get("text", claim.get("claim", ""))
        if action == "DELETE":
            modified_claims.pop(idx)
            applied.append(f"DELETED claim {target_id}: {claim_text[:60]}...")
        elif action == "REWRITE":
            modified_claims[idx] = {**claim, "text": e.replacement}
            applied.append(f"REWROTE claim {target_id}")
        elif action == "MOVE_TO_UNKNOWN":
            modified_claims[idx] = {
                **claim,
                "text": e.replacement or f"Unknown(Actionable): {claim_text}",
                "is_unknown": True,
            }
            applied.append(f"MOVED claim {target_id} to Unknown")

    summary = "; ".join(applied) if applied else "No ID-based edits applied"
    return modified_claims, summary


def parse_gpt3_structured(parsed: GPT3ResponseSchema):
    """Parse a structured GPT-3 response (Pydantic model) into the same 4-tuple.

    This is the V4 equivalent of parse_gpt3() — it skips JSON extraction
    since the response is already validated by the structured output API.

    Returns: (decision, rationale, edits, policy_notes)
    """
    try:
        rationale = _as_str_list(parsed.rationale)
        decision, rationale = _coerce_decision(parsed.arbiter_decision, rationale)
        edits = [
            EditEntry(
                action=str(e.action).upper(),
                target=e.target,
                replacement=e.replacement,
                target_id=getattr(e, "target_id", ""),
            )
            for e in parsed.edits_for_gpt1
        ]
        policy_notes = _as_str_list(parsed.final_policy_notes)
        return decision, rationale, edits, policy_notes
    except Exception:
        return "BLOCK", ["GPT-3 structured parse error"], [], []


def check_poisoning_threshold(
    claim_table: List[Any],
    findings: List[Any],
    unsupported_threshold: float = 0.35,
    hard_threshold: int = 2,
) -> Dict[str, Any]:
    """Calculate unsupported claims ratio and hard violations to detect poisoning.

    Args:
        claim_table: List of claim entries (dict or ClaimEntry).
        findings: List of finding entries (dict or FindingSchema).
        unsupported_threshold: Maximum allowed ratio of unsupported claims (default 0.35).
        hard_threshold: Hard violation count threshold that triggers poisoning (default 2).

    Returns:
        Dict containing:
            - is_poisoned: bool, True if unsupported_ratio > unsupported_threshold or hard_count >= hard_threshold
            - unsupported_ratio: float, ratio of unsupported claims to total claims
            - hard_count: int, number of hard severity findings
            - unsupported_count: int, count of unsupported claims
            - total_claims: int, total number of claims evaluated
    """
    unsupported_cats = {"unsupported", "unsupported_inferential", "contradicted", "refuted", "fabricated"}
    total_claims = len(claim_table) if claim_table else 0
    unsupported_count = 0
    if claim_table:
        for c in claim_table:
            cat = getattr(c, "category", None)
            if cat is None and isinstance(c, dict):
                cat = c.get("category", "")
            if isinstance(cat, str) and cat.lower().strip() in unsupported_cats:
                unsupported_count += 1

    unsupported_ratio = (unsupported_count / total_claims) if total_claims > 0 else 0.0

    hard_count = 0
    if findings:
        for f in findings:
            sev = getattr(f, "severity", None)
            if sev is None and isinstance(f, dict):
                sev = f.get("severity", "")
            if isinstance(sev, str) and sev.lower().strip() == "hard":
                hard_count += 1

    is_poisoned = bool((unsupported_ratio > unsupported_threshold) or (hard_count >= hard_threshold))

    return {
        "is_poisoned": is_poisoned,
        "unsupported_ratio": unsupported_ratio,
        "hard_count": hard_count,
        "unsupported_count": unsupported_count,
        "total_claims": total_claims,
    }


def guard_arbiter_decision(
    decision: str,
    claim_table: List[Any],
    findings: List[Any],
    unsupported_threshold: float = 0.35,
    hard_threshold: int = 2,
) -> Tuple[str, List[str]]:
    """Guard and adjust arbiter decision based on adaptive poisoning thresholds.

    - If is_poisoned: decision must be BLOCK with explanatory rationale notes.
    - If not poisoned (<= 35% unsupported and < 2 hard violations) and truthful claims exist:
      allow ALLOW_WITH_EDITS.
    - Preserve ALLOW_AS_UNKNOWN_ONLY if no hard violations exist.

    Args:
        decision: Raw arbiter decision ("BLOCK", "ALLOW_WITH_EDITS", "ALLOW_AS_UNKNOWN_ONLY").
        claim_table: List of claim entries.
        findings: List of verification findings.
        unsupported_threshold: Unsupported claims ratio threshold (default 0.35).
        hard_threshold: Hard findings threshold (default 2).

    Returns:
        (final_decision, rationale_notes)
    """
    check = check_poisoning_threshold(
        claim_table,
        findings,
        unsupported_threshold=unsupported_threshold,
        hard_threshold=hard_threshold,
    )
    raw_dec = (decision or "BLOCK").upper().strip()
    notes: List[str] = []

    salvageable_cats = {"supported", "observed", "inference", "user-provided"}
    has_truthful = False
    if claim_table:
        for c in claim_table:
            cat = getattr(c, "category", None)
            if cat is None and isinstance(c, dict):
                cat = c.get("category", "")
            if isinstance(cat, str) and cat.lower().strip() in salvageable_cats:
                has_truthful = True
                break

    if check["is_poisoned"]:
        diag_parts = []
        if check["unsupported_ratio"] > unsupported_threshold:
            diag_parts.append(
                f"unsupported ratio {check['unsupported_ratio']:.1%} > {unsupported_threshold:.1%}"
            )
        if check["hard_count"] >= hard_threshold:
            diag_parts.append(
                f"hard violations {check['hard_count']} >= {hard_threshold}"
            )
        reason = "; ".join(diag_parts) if diag_parts else "exceeded threshold"

        if raw_dec != "BLOCK":
            notes.append(
                f"Decision overridden from {raw_dec} to BLOCK: draft is heavily poisoned ({reason})."
            )
        else:
            notes.append(
                f"BLOCK decision confirmed by poisoning guard: draft is heavily poisoned ({reason})."
            )
        return "BLOCK", notes

    # Not poisoned:
    if raw_dec == "ALLOW_AS_UNKNOWN_ONLY":
        if check["hard_count"] == 0:
            notes.append("ALLOW_AS_UNKNOWN_ONLY preserved: zero hard violations detected.")
            return "ALLOW_AS_UNKNOWN_ONLY", notes
        else:
            # 1 hard violation present with low unsupported ratio
            if has_truthful:
                notes.append(
                    "Decision adjusted to ALLOW_WITH_EDITS: salvageable truthful content exists to resolve hard violation."
                )
                return "ALLOW_WITH_EDITS", notes
            else:
                return "ALLOW_AS_UNKNOWN_ONLY", notes

    if raw_dec == "BLOCK":
        if has_truthful:
            notes.append(
                f"Decision overridden from BLOCK to ALLOW_WITH_EDITS: draft is lightly poisoned "
                f"({check['unsupported_ratio']:.1%} unsupported, {check['hard_count']} hard violations) with salvageable content."
            )
            return "ALLOW_WITH_EDITS", notes
        else:
            notes.append(
                "BLOCK decision maintained: no salvageable truthful claims found in claim table."
            )
            return "BLOCK", notes

    # raw_dec is ALLOW_WITH_EDITS (or other)
    notes.append(
        f"ALLOW_WITH_EDITS confirmed: draft is lightly poisoned "
        f"({check['unsupported_ratio']:.1%} unsupported, {check['hard_count']} hard violations)."
    )
    return "ALLOW_WITH_EDITS", notes


def extract_negative_constraints(
    findings: Optional[List[Any]] = None,
    edits: Optional[List[Any]] = None,
    claim_table: Optional[List[Any]] = None,
    max_source_count: Optional[int] = None,
    arbiter_edits: Optional[List[Any]] = None,
) -> List[str]:
    """Extract actionable, imperative negative constraints from findings, edits, and claim tables.

    Translates tripwire violations, unbacked numeric claims, out-of-bounds citations,
    arbiter edit instructions (DELETE, REWRITE, MOVE_TO_UNKNOWN), and unsupported claims
    into explicit 'DO NOT ...' directives to prevent re-hallucination during repair loops.
    """
    constraints: List[str] = []
    seen: Set[str] = set()

    def _add(directive: str):
        cleaned = directive.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            constraints.append(cleaned)

    # 1. Process Arbiter Edits
    all_edits = arbiter_edits if arbiter_edits is not None else (edits or [])
    for edit in all_edits:
        action = (getattr(edit, "action", "") or (edit.get("action", "") if isinstance(edit, dict) else "")).upper().strip()
        target = (getattr(edit, "target", "") or (edit.get("target", "") if isinstance(edit, dict) else "")).strip()
        replacement = (getattr(edit, "replacement", "") or (edit.get("replacement", "") if isinstance(edit, dict) else "")).strip()
        if not target:
            continue
        if action == "DELETE":
            _add(f'DO NOT include the claim or text: "{target}"')
        elif action == "REWRITE":
            _add(f'DO NOT use the unverified phrasing: "{target}" (replace with: "{replacement}")')
        elif action == "MOVE_TO_UNKNOWN":
            _add(f'DO NOT state as an established fact: "{target}" (frame strictly as Unknown/Unverified)')

    # 2. Process Findings (Verifier & Pre-Flight)
    if findings:
        for f in findings:
            ftype = getattr(f, "type", "") or (f.get("type", "") if isinstance(f, dict) else "")
            detail = getattr(f, "detail", "") or (f.get("detail", "") if isinstance(f, dict) else "")
            detail_lower = detail.lower()

            # Out-of-bounds or fabricated citation
            if "referenced non-existent source" in detail_lower or "out_of_range" in detail_lower or "available sources: 0" in detail_lower:
                m = re.search(r"source\s*\[(\d+)\]", detail, re.IGNORECASE)
                if m:
                    idx = m.group(1)
                    limit_str = f" (valid source indices are 1..{max_source_count})" if (max_source_count is not None and max_source_count > 0) else ""
                    _add(f"DO NOT cite non-existent source [{idx}]{limit_str}.")
                elif "available sources: 0" in detail_lower:
                    _add("DO NOT cite non-existent sources (available sources: 0).")
                else:
                    _add(f"DO NOT cite non-existent source numbers: {detail}")
            # Unbacked numeric claim
            elif "does not contain numeric value" in detail_lower or "unbacked numeric" in detail_lower:
                m_num = re.search(r"numeric value\s*([\$0-9\.\,]+[a-zA-Z]*)", detail, re.IGNORECASE)
                if not m_num:
                    m_num = re.search(r"numeric figure\s*([\$0-9\.\,]+[a-zA-Z]*)", detail, re.IGNORECASE)
                if not m_num:
                    m_num = re.search(r"(?:value|figure|number|claim:)\s*([\$0-9\.\,]+[a-zA-Z]*)", detail, re.IGNORECASE)
                num_str = m_num.group(1).strip() if m_num else "unbacked numbers"
                num_str = num_str.rstrip(".,\"'")
                _add(f"DO NOT introduce the unbacked numeric figure {num_str}.")
            # Unbacked citation / keyword mismatch
            elif "does not contain facts supporting statement" in detail_lower:
                m_snip = re.search(r"supporting statement '([^']+)'", detail)
                snip = m_snip.group(1) if m_snip else detail
                _add(f"DO NOT attribute statement '{snip}' to source without direct evidence.")
            # Tripwire T1
            elif ftype == "T1":
                _add(f"DO NOT introduce fabricated entities, unverified legal conclusions, or invented statistics ({detail}).")
            # Tripwire T2
            elif ftype == "T2" or "typicality" in detail_lower:
                _add("DO NOT use typicality words ('usually', 'often', 'typically', 'generally', 'commonly') to justify claims without citation.")
            # Tripwire T3
            elif ftype == "T3" or "causal" in detail_lower:
                _add(f"DO NOT assert causal relationships as established facts without explicit source citation ({detail}).")
            # Tripwire T4
            elif ftype == "T4" or "ranking" in detail_lower:
                _add("DO NOT rank, rate, or compare options without evidence-backed discriminators.")
            # Tripwire T5
            elif ftype == "T5" or "outcome promise" in detail_lower or "guarantee" in detail_lower:
                _add("DO NOT include outcome promises ('will improve', 'guarantees') or unsolicited advice.")
            # Tripwire T6
            elif ftype == "T6" or "reassurance" in detail_lower:
                _add("DO NOT use reassurance framing, praise, or conversational filler.")
            # Tripwire T7
            elif ftype == "T7" or "time-sensitive" in detail_lower or "future" in detail_lower:
                _add("DO NOT present time-sensitive facts or future predictions as current/certain without verification; frame as Unknown(Actionable).")
            else:
                _add(f"DO NOT repeat violation: {detail}")

    # 3. Process Unsupported Claims from Claim Table
    if claim_table:
        for ct in claim_table:
            cat = (getattr(ct, "category", "") or (ct.get("category", "") if isinstance(ct, dict) else ""))
            cat_norm = str(cat).lower().strip()
            if cat_norm in ("unsupported", "unsupported_inferential", "contradicted", "refuted", "fabricated"):
                txt = (getattr(ct, "claim", "") or (ct.get("claim", "") if isinstance(ct, dict) else "")).strip()
                if txt:
                    _add(f'DO NOT make the unbacked assertion: "{txt}"')

    return constraints


def format_negative_constraints_block(constraints: List[str]) -> str:
    """Format negative constraints into a markdown block for retry prompts.

    Returns empty string if constraints list is empty.
    """
    if not constraints:
        return ""
    lines = [f"- {c}" for c in constraints]
    return (
        "### Negative Constraints\n"
        "The following claims, figures, citations, and rhetorical patterns were rejected during verification and MUST NOT appear anywhere in your rewritten response:\n"
        + "\n".join(lines)
    )


