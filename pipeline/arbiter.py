"""GPT-3 Arbiter: parses arbiter decisions and applies edits.

Provides both legacy text parsing (parse_gpt3) and structured output
parsing (parse_gpt3_structured) for the V4 async pipeline.
"""
from __future__ import annotations

from typing import List

from pipeline.models import EditEntry, GPT3ResponseSchema
from pipeline.helpers import extract_json


def parse_gpt3(raw: str):
    """Parse GPT-3 Arbiter JSON output."""
    try:
        parsed = extract_json(raw)
        decision = parsed.get("arbiter_decision", "BLOCK").upper()
        rationale = parsed.get("rationale", [])
        edits_raw = parsed.get("edits_for_gpt1", [])
        edits = [
            EditEntry(
                action=str(e.get("action", "")).upper(),
                target=e.get("target", ""),
                replacement=e.get("replacement", ""),
                target_id=e.get("target_id", ""),
            )
            for e in edits_raw
        ]
        policy_notes = parsed.get("final_policy_notes", [])
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
        if action == "DELETE":
            modified_claims.pop(idx)
            applied.append(f"DELETED claim {target_id}: {claim['text'][:60]}...")
        elif action == "REWRITE":
            modified_claims[idx] = {**claim, "text": e.replacement}
            applied.append(f"REWROTE claim {target_id}")
        elif action == "MOVE_TO_UNKNOWN":
            modified_claims[idx] = {
                **claim,
                "text": e.replacement or f"Unknown(Actionable): {claim['text']}",
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
        decision = parsed.arbiter_decision.upper()
        rationale = parsed.rationale
        edits = [
            EditEntry(
                action=str(e.action).upper(),
                target=e.target,
                replacement=e.replacement,
                target_id=getattr(e, "target_id", ""),
            )
            for e in parsed.edits_for_gpt1
        ]
        policy_notes = parsed.final_policy_notes
        return decision, rationale, edits, policy_notes
    except Exception:
        return "BLOCK", ["GPT-3 structured parse error"], [], []
