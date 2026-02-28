"""Core pipeline orchestrator: GPT-1 -> GPT-2 -> GPT-3 verification flow."""
from __future__ import annotations

import json

import openai

import config
from pipeline.models import PipelineRequest, PipelineResponse
from pipeline.prompts import DEFAULT_GPT1_SYSTEM, DEFAULT_GPT2_SYSTEM, DEFAULT_GPT3_SYSTEM
from pipeline.sanitizer import route_prompt, sanitize_output
from pipeline.helpers import PipelineError, call_openai, is_activation_phrase
from pipeline.verifier import parse_gpt2, _all_soft
from pipeline.arbiter import parse_gpt3, apply_edits


def run_pipeline(req: PipelineRequest) -> PipelineResponse:
    """Execute the full epistemic verification pipeline.

    Flow:
        1. Route prompt (deterministic flags)
        2. GPT-1 generates response
        3. Check activation bypass
        4. Sanitize output
        5. GPT-2 verifies
        6. If FAIL (soft-only): auto-repair + re-verify
        7. If still FAIL: GPT-3 Arbiter
        8. Arbiter BLOCK / ALLOW_WITH_EDITS / ALLOW_AS_UNKNOWN_ONLY
    """
    if not config.has_api_key():
        raise PipelineError(400, "Set your OpenAI API key first.")

    client = openai.OpenAI(api_key=config.get_api_key())
    model = config.get_model()

    # ---- Deterministic prompt routing ----
    flags = route_prompt(req.prompt)

    gpt1_system = req.gpt1_system or DEFAULT_GPT1_SYSTEM
    gpt2_system = req.gpt2_system or DEFAULT_GPT2_SYSTEM
    gpt3_system = req.gpt3_system or DEFAULT_GPT3_SYSTEM

    # If user explicitly requested advice, augment GPT-1 system prompt
    if flags.get("advice_requested"):
        gpt1_system += (
            "\n\nNOTE: The user is explicitly requesting advice/options. "
            "You may provide conditional process guidance."
        )

    # Empty defaults for response
    empty_response = dict(
        arbiter_invoked=False, arbiter_decision="", arbiter_rationale=[],
        arbiter_edits=[], arbiter_policy_notes=[], arbiter_raw="",
        rewrite_occurred=False, rewrite_output="", rewrite_gpt2_raw="",
        rewrite_claim_table=[], rewrite_violations=[], rewrite_verdict="",
    )

    # ---- Step 1: GPT-1 Generate ----
    gpt1_output = call_openai(client, model, gpt1_system, req.prompt)

    # ---- Activation bypass ----
    if is_activation_phrase(gpt1_output):
        return PipelineResponse(
            gpt1_input=req.prompt, gpt1_output=gpt1_output, bypassed=True,
            gpt2_raw="(bypassed)", claim_table=[], violations=[], gpt2_verdict="PASS",
            final_verdict="PASS", final_result=gpt1_output,
            prompt_flags=flags, sanitizer_applied=False,
            **empty_response,
        )

    # ---- Deterministic sanitizer (pre-clean before GPT-2) ----
    sanitized_output = sanitize_output(gpt1_output, flags)
    sanitizer_applied = (sanitized_output != gpt1_output)

    # ---- Step 2: GPT-2 Verify (on sanitized output) ----
    gpt2_user = f"ORIGINAL PROMPT:\n{req.prompt}\n\nGPT-1 RESPONSE TO VERIFY:\n{sanitized_output}"
    gpt2_raw = call_openai(client, model, gpt2_system, gpt2_user, expect_json=True)
    claim_table, violations, gpt2_verdict, findings = parse_gpt2(gpt2_raw, flags=flags)

    # ---- If GPT-2 PASS: done ----
    if gpt2_verdict == "PASS":
        return PipelineResponse(
            gpt1_input=req.prompt, gpt1_output=gpt1_output, bypassed=False,
            gpt2_raw=gpt2_raw, claim_table=claim_table, violations=violations,
            gpt2_verdict="PASS",
            final_verdict="PASS", final_result=sanitized_output,
            prompt_flags=flags, sanitizer_applied=sanitizer_applied,
            **empty_response,
        )

    # ---- GPT-2 FAIL: soft-only auto-repair path ----
    if _all_soft(findings):
        # Try auto-repair via sanitize_output + re-verify (no arbiter yet)
        repaired = sanitize_output(sanitized_output, flags)
        re_gpt2_user = f"ORIGINAL PROMPT:\n{req.prompt}\n\nGPT-1 RESPONSE TO VERIFY:\n{repaired}"
        re_gpt2_raw = call_openai(client, model, gpt2_system, re_gpt2_user, expect_json=True)
        re_ct, re_viol, re_verdict, re_findings = parse_gpt2(re_gpt2_raw, flags=flags)

        if re_verdict == "PASS":
            return PipelineResponse(
                gpt1_input=req.prompt, gpt1_output=gpt1_output, bypassed=False,
                gpt2_raw=gpt2_raw, claim_table=claim_table, violations=violations,
                gpt2_verdict=gpt2_verdict,
                rewrite_occurred=True, rewrite_output=repaired,
                rewrite_gpt2_raw=re_gpt2_raw, rewrite_claim_table=re_ct,
                rewrite_violations=re_viol, rewrite_verdict=re_verdict,
                arbiter_invoked=False, arbiter_decision="", arbiter_rationale=[],
                arbiter_edits=[], arbiter_policy_notes=[], arbiter_raw="",
                final_verdict="PASS", final_result=repaired,
                prompt_flags=flags, sanitizer_applied=True,
            )
        # Auto-repair didn't clear it -- fall through to arbiter below
        sanitized_output = repaired

    # ---- Step 3: GPT-2 FAIL -> invoke GPT-3 Arbiter ----
    gpt2_json_for_arbiter = json.dumps({
        "claim_table": [
            {"claim": c.claim, "category": c.category, "justification": c.justification}
            for c in claim_table
        ],
        "violations": violations,
        "verdict": gpt2_verdict,
    }, indent=2)

    gpt3_user = (
        f"user_prompt:\n{req.prompt}\n\n"
        f"gpt1_output:\n{sanitized_output}\n\n"
        f"gpt2_result_json:\n{gpt2_json_for_arbiter}"
    )

    gpt3_raw = call_openai(client, model, gpt3_system, gpt3_user, expect_json=True)
    arbiter_decision, arbiter_rationale, arbiter_edits, arbiter_policy_notes = parse_gpt3(gpt3_raw)

    # ---- Decision: BLOCK ----
    if arbiter_decision == "BLOCK":
        return PipelineResponse(
            gpt1_input=req.prompt, gpt1_output=gpt1_output, bypassed=False,
            gpt2_raw=gpt2_raw, claim_table=claim_table, violations=violations,
            gpt2_verdict=gpt2_verdict,
            arbiter_invoked=True, arbiter_decision="BLOCK",
            arbiter_rationale=arbiter_rationale, arbiter_edits=arbiter_edits,
            arbiter_policy_notes=arbiter_policy_notes, arbiter_raw=gpt3_raw,
            rewrite_occurred=False, rewrite_output="", rewrite_gpt2_raw="",
            rewrite_claim_table=[], rewrite_violations=[], rewrite_verdict="",
            final_verdict="FAIL", final_result="NO PASS",
            prompt_flags=flags, sanitizer_applied=sanitizer_applied,
        )

    # ---- Decision: ALLOW_AS_UNKNOWN_ONLY ----
    if arbiter_decision == "ALLOW_AS_UNKNOWN_ONLY":
        rewrite_prompt = (
            f"You previously produced this response:\n\n---\n{sanitized_output}\n---\n\n"
            f"The arbiter has determined this question is inherently indeterminate.\n"
            f"Rewrite your response so that ALL claims are framed as Unknown(Actionable) or Unknown(Structural).\n"
            f"Do NOT make conclusions. Do NOT add new facts. Preserve the structure but move all substance to Unknowns.\n"
            f"Output the corrected response in full."
        )
        rewrite_output = call_openai(client, model, gpt1_system, rewrite_prompt)

        # Re-verify with GPT-2
        re_gpt2_user = f"ORIGINAL PROMPT:\n{req.prompt}\n\nGPT-1 RESPONSE TO VERIFY:\n{rewrite_output}"
        re_gpt2_raw = call_openai(client, model, gpt2_system, re_gpt2_user, expect_json=True)
        re_ct, re_viol, re_verdict, re_findings = parse_gpt2(re_gpt2_raw, flags=flags)

        return PipelineResponse(
            gpt1_input=req.prompt, gpt1_output=gpt1_output, bypassed=False,
            gpt2_raw=gpt2_raw, claim_table=claim_table, violations=violations,
            gpt2_verdict=gpt2_verdict,
            arbiter_invoked=True, arbiter_decision="ALLOW_AS_UNKNOWN_ONLY",
            arbiter_rationale=arbiter_rationale, arbiter_edits=arbiter_edits,
            arbiter_policy_notes=arbiter_policy_notes, arbiter_raw=gpt3_raw,
            rewrite_occurred=True, rewrite_output=rewrite_output,
            rewrite_gpt2_raw=re_gpt2_raw, rewrite_claim_table=re_ct,
            rewrite_violations=re_viol, rewrite_verdict=re_verdict,
            final_verdict=re_verdict,
            final_result=rewrite_output if re_verdict == "PASS" else "NO PASS",
            prompt_flags=flags, sanitizer_applied=sanitizer_applied,
        )

    # ---- Decision: ALLOW_WITH_EDITS ----
    rewrite_prompt = apply_edits(sanitized_output, arbiter_edits)
    rewrite_output = call_openai(client, model, gpt1_system, rewrite_prompt)

    # Re-verify the rewritten output with GPT-2
    re_gpt2_user = f"ORIGINAL PROMPT:\n{req.prompt}\n\nGPT-1 RESPONSE TO VERIFY:\n{rewrite_output}"
    re_gpt2_raw = call_openai(client, model, gpt2_system, re_gpt2_user, expect_json=True)
    re_ct, re_viol, re_verdict, re_findings = parse_gpt2(re_gpt2_raw, flags=flags)

    # If still failing on soft-only violations after arbiter rewrite, force Unknown-only
    if re_verdict == "FAIL" and _all_soft(re_findings):
        unknown_prompt = (
            f"You previously produced this response:\n\n---\n{rewrite_output}\n---\n\n"
            f"Remaining soft violations could not be resolved. "
            f"Rewrite your response so that ALL claims are framed as Unknown(Actionable) or Unknown(Structural).\n"
            f"Do NOT make conclusions. Do NOT add new facts. Output the corrected response in full."
        )
        rewrite_output = call_openai(client, model, gpt1_system, unknown_prompt)
        re_verdict = "PASS"

    return PipelineResponse(
        gpt1_input=req.prompt, gpt1_output=gpt1_output, bypassed=False,
        gpt2_raw=gpt2_raw, claim_table=claim_table, violations=violations,
        gpt2_verdict=gpt2_verdict,
        arbiter_invoked=True, arbiter_decision="ALLOW_WITH_EDITS",
        arbiter_rationale=arbiter_rationale, arbiter_edits=arbiter_edits,
        arbiter_policy_notes=arbiter_policy_notes, arbiter_raw=gpt3_raw,
        rewrite_occurred=True, rewrite_output=rewrite_output,
        rewrite_gpt2_raw=re_gpt2_raw, rewrite_claim_table=re_ct,
        rewrite_violations=re_viol, rewrite_verdict=re_verdict,
        final_verdict=re_verdict,
        final_result=rewrite_output if re_verdict == "PASS" else "NO PASS",
        prompt_flags=flags, sanitizer_applied=sanitizer_applied,
    )
