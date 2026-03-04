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
                action=e.get("action", ""),
                target=e.get("target", ""),
                replacement=e.get("replacement", ""),
            )
            for e in edits_raw
        ]
        policy_notes = parsed.get("final_policy_notes", [])
        return decision, rationale, edits, policy_notes
    except Exception:
        return "BLOCK", ["GPT-3 parse error: could not extract valid JSON"], [], []


def apply_edits(gpt1_output: str, edits: List[EditEntry]) -> str:
    """Apply GPT-3 edit instructions to GPT-1 output for rewrite prompt."""
    instructions = []
    for e in edits:
        if e.action == "DELETE":
            instructions.append(f'DELETE the following text: "{e.target}"')
        elif e.action == "REWRITE":
            instructions.append(f'REWRITE "{e.target}" to: "{e.replacement}"')
        elif e.action == "MOVE_TO_UNKNOWN":
            instructions.append(
                f'MOVE the following to the Unknowns section: "{e.target}" '
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
                action=e.action,
                target=e.target,
                replacement=e.replacement,
            )
            for e in parsed.edits_for_gpt1
        ]
        policy_notes = parsed.final_policy_notes
        return decision, rationale, edits, policy_notes
    except Exception:
        return "BLOCK", ["GPT-3 structured parse error"], [], []
