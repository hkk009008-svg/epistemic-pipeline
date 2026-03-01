"""Core pipeline orchestrator: GPT-1 -> GPT-2 -> GPT-3 verification flow."""
from __future__ import annotations

import json

import openai

import config
from pipeline.models import PipelineRequest, PipelineResponse, ConfidenceBreakdown, SearchSource
from pipeline.prompts import DEFAULT_GPT1_SYSTEM, DEFAULT_GPT2_SYSTEM, DEFAULT_GPT3_SYSTEM, PROMPT_VERSION, build_augmentation
from pipeline.sanitizer import route_prompt, sanitize_output
from pipeline.helpers import PipelineError, call_openai, is_activation_phrase
from pipeline.verifier import parse_gpt2, _all_soft
from pipeline.arbiter import parse_gpt3, apply_edits
from pipeline.search import should_search, perform_web_search


def compute_confidence(claim_table: list) -> ConfidenceBreakdown:
    """Compute a confidence breakdown from a list of ClaimEntry objects."""
    total = len(claim_table)
    if total == 0:
        return ConfidenceBreakdown()

    observed = inference = hypothesis = unsupported = user_provided = 0
    for entry in claim_table:
        cat = (entry.category if isinstance(entry.category, str) else str(entry.category)).lower().strip()
        if cat in ("supported", "observed"):
            observed += 1
        elif cat == "inference":
            inference += 1
        elif cat == "hypothesis":
            hypothesis += 1
        elif cat == "unsupported":
            unsupported += 1
        elif cat == "user-provided":
            user_provided += 1

    observed_pct = round((observed / total) * 100, 1)
    if observed_pct >= 70:
        label = "High"
    elif observed_pct >= 40:
        label = "Medium"
    elif observed_pct >= 20:
        label = "Low"
    else:
        label = "Unknown"

    return ConfidenceBreakdown(
        observed_pct=observed_pct,
        inference_pct=round((inference / total) * 100, 1),
        hypothesis_pct=round((hypothesis / total) * 100, 1),
        unsupported_pct=round((unsupported / total) * 100, 1),
        user_provided_pct=round((user_provided / total) * 100, 1),
        total_claims=total,
        confidence_label=label,
    )


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

    # Flag-driven augmentation for all 3 stages
    gpt1_aug, gpt2_aug, gpt3_aug = build_augmentation(flags)
    gpt1_system += gpt1_aug
    gpt2_system += gpt2_aug
    gpt3_system += gpt3_aug

    # ---- Web Search Enrichment (before GPT-1) ----
    search_sources: list[SearchSource] = []
    search_context = ""
    search_performed = False

    if should_search(flags):
        search_sources, search_context = perform_web_search(req.prompt)
        search_performed = len(search_sources) > 0

    gpt1_user_content = req.prompt
    if search_performed and search_context:
        gpt1_system += (
            "\n\nWEB SEARCH RESULTS are provided below the user's question. "
            "You MUST ground your response in these sources. "
            "When citing a fact from a source, reference it as [1], [2], etc. "
            "If a source provides a specific statistic, you may quote it with the citation. "
            "Do NOT fabricate additional sources beyond what is provided. "
            "If the sources do not contain the answer, state Unknown (Actionable)."
        )
        gpt1_user_content = (
            f"{req.prompt}\n\n"
            f"--- WEB SEARCH RESULTS ---\n"
            f"{search_context}\n"
            f"--- END SEARCH RESULTS ---"
        )

        # Augment GPT-2 to recognize the provided sources
        source_summary = "; ".join(
            f'[{i}] "{s.title}" ({s.url})' for i, s in enumerate(search_sources, 1)
        )
        gpt2_system += (
            "\n\nIMPORTANT: GPT-1 was given web search results from the following sources:\n"
            f"{source_summary}\n\n"
            "When evaluating claims:\n"
            "- If a claim cites one of these sources (e.g., [1], [2]) and the source snippet "
            "supports the claim, categorize it as 'Supported' (not 'Unsupported').\n"
            "- A statistic that is attributed to a provided source is NOT 'Fabricated' or "
            "'Unverified' -- categorize it as 'Supported'.\n"
            "- If GPT-1 cites a source number that does not exist in the list above, "
            "flag it as 'Fabricated citation'.\n"
            "- Claims NOT backed by any provided source should still be evaluated normally."
        )

    search_kwargs = dict(
        search_performed=search_performed,
        search_query=req.prompt if search_performed else "",
        search_sources=search_sources,
    )

    # Empty defaults for response
    empty_response = dict(
        arbiter_invoked=False, arbiter_decision="", arbiter_rationale=[],
        arbiter_edits=[], arbiter_policy_notes=[], arbiter_raw="",
        rewrite_occurred=False, rewrite_output="", rewrite_gpt2_raw="",
        rewrite_claim_table=[], rewrite_violations=[], rewrite_verdict="",
    )

    # ---- Step 1: GPT-1 Generate ----
    gpt1_output = call_openai(client, model, gpt1_system, gpt1_user_content)

    # ---- Activation bypass ----
    if is_activation_phrase(gpt1_output):
        return PipelineResponse(
            prompt_version=PROMPT_VERSION,
            gpt1_input=req.prompt, gpt1_output=gpt1_output, bypassed=True,
            gpt2_raw="(bypassed)", claim_table=[], violations=[], gpt2_verdict="PASS",
            final_verdict="PASS", final_result=gpt1_output,
            prompt_flags=flags, sanitizer_applied=False,
            confidence=compute_confidence([]),
            **empty_response, **search_kwargs,
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
            prompt_version=PROMPT_VERSION,
            gpt1_input=req.prompt, gpt1_output=gpt1_output, bypassed=False,
            gpt2_raw=gpt2_raw, claim_table=claim_table, violations=violations,
            gpt2_verdict="PASS",
            final_verdict="PASS", final_result=sanitized_output,
            prompt_flags=flags, sanitizer_applied=sanitizer_applied,
            confidence=compute_confidence(claim_table),
            **empty_response, **search_kwargs,
        )

    # ---- GPT-2 FAIL: soft-only auto-repair path ----
    max_rewrite_loops = getattr(config, "MAX_REWRITE_LOOPS", 1)
    if _all_soft(findings):
        # Re-verify with GPT-2 directly (sanitizer already ran on sanitized_output)
        re_gpt2_user = f"ORIGINAL PROMPT:\n{req.prompt}\n\nGPT-1 RESPONSE TO VERIFY:\n{sanitized_output}"
        re_gpt2_raw = call_openai(client, model, gpt2_system, re_gpt2_user, expect_json=True)
        re_ct, re_viol, re_verdict, re_findings = parse_gpt2(re_gpt2_raw, flags=flags)

        if re_verdict == "PASS":
            return PipelineResponse(
                prompt_version=PROMPT_VERSION,
                gpt1_input=req.prompt, gpt1_output=gpt1_output, bypassed=False,
                gpt2_raw=gpt2_raw, claim_table=claim_table, violations=violations,
                gpt2_verdict=gpt2_verdict,
                rewrite_occurred=True, rewrite_output=sanitized_output,
                rewrite_gpt2_raw=re_gpt2_raw, rewrite_claim_table=re_ct,
                rewrite_violations=re_viol, rewrite_verdict=re_verdict,
                arbiter_invoked=False, arbiter_decision="", arbiter_rationale=[],
                arbiter_edits=[], arbiter_policy_notes=[], arbiter_raw="",
                final_verdict="PASS", final_result=sanitized_output,
                prompt_flags=flags, sanitizer_applied=True,
                confidence=compute_confidence(re_ct),
                **search_kwargs,
            )
        # Auto-repair didn't clear it -- fall through to arbiter below

    # ---- Step 3: GPT-2 FAIL -> invoke GPT-3 Arbiter ----
    gpt2_json_for_arbiter = json.dumps({
        "claim_table": [
            {"claim": c.claim, "category": c.category, "justification": c.justification}
            for c in claim_table
        ],
        "violations": violations,
        "verdict": gpt2_verdict,
    }, indent=2)

    flags_json = json.dumps(flags, indent=2)
    gpt3_user = (
        f"user_prompt:\n{req.prompt}\n\n"
        f"gpt1_output:\n{sanitized_output}\n\n"
        f"gpt2_result_json:\n{gpt2_json_for_arbiter}\n\n"
        f"prompt_flags:\n{flags_json}"
    )

    gpt3_raw = call_openai(client, model, gpt3_system, gpt3_user, expect_json=True)
    arbiter_decision, arbiter_rationale, arbiter_edits, arbiter_policy_notes = parse_gpt3(gpt3_raw)

    # ---- Decision: BLOCK ----
    if arbiter_decision == "BLOCK":
        return PipelineResponse(
            prompt_version=PROMPT_VERSION,
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
            confidence=compute_confidence(claim_table),
            **search_kwargs,
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
            prompt_version=PROMPT_VERSION,
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
            confidence=compute_confidence(re_ct),
            **search_kwargs,
        )

    # ---- Decision: ALLOW_WITH_EDITS ----
    rewrite_prompt = apply_edits(sanitized_output, arbiter_edits)
    rewrite_output = call_openai(client, model, gpt1_system, rewrite_prompt)

    # Re-verify the rewritten output with GPT-2
    re_gpt2_user = f"ORIGINAL PROMPT:\n{req.prompt}\n\nGPT-1 RESPONSE TO VERIFY:\n{rewrite_output}"
    re_gpt2_raw = call_openai(client, model, gpt2_system, re_gpt2_user, expect_json=True)
    re_ct, re_viol, re_verdict, re_findings = parse_gpt2(re_gpt2_raw, flags=flags)

    # If still failing on soft-only violations after arbiter rewrite, force Unknown-only
    # and re-verify (up to max_rewrite_loops)
    rewrite_count = 0
    while re_verdict == "FAIL" and _all_soft(re_findings) and rewrite_count < max_rewrite_loops:
        rewrite_count += 1
        unknown_prompt = (
            f"You previously produced this response:\n\n---\n{rewrite_output}\n---\n\n"
            f"Remaining soft violations could not be resolved. "
            f"Rewrite your response so that ALL claims are framed as Unknown(Actionable) or Unknown(Structural).\n"
            f"Do NOT make conclusions. Do NOT add new facts. Output the corrected response in full."
        )
        rewrite_output = call_openai(client, model, gpt1_system, unknown_prompt)
        # Re-verify instead of force-passing
        re_gpt2_user = f"ORIGINAL PROMPT:\n{req.prompt}\n\nGPT-1 RESPONSE TO VERIFY:\n{rewrite_output}"
        re_gpt2_raw = call_openai(client, model, gpt2_system, re_gpt2_user, expect_json=True)
        re_ct, re_viol, re_verdict, re_findings = parse_gpt2(re_gpt2_raw, flags=flags)

    return PipelineResponse(
        prompt_version=PROMPT_VERSION,
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
        confidence=compute_confidence(re_ct),
        **search_kwargs,
    )
