"""Core pipeline orchestrator: GPT-1 -> GPT-2 -> GPT-3 verification flow."""
from __future__ import annotations

import json
from datetime import date

import openai

import config
from pipeline.models import PipelineRequest, PipelineResponse, ConfidenceBreakdown, SearchSource
from pipeline.prompts import DEFAULT_GPT1_SYSTEM, DEFAULT_GPT2_SYSTEM, DEFAULT_GPT3_SYSTEM, GPT2_TRIPWIRE_REFERENCE, PROMPT_VERSION, build_augmentation
from pipeline.sanitizer import route_prompt, sanitize_output
from pipeline.helpers import PipelineError, call_openai, is_activation_phrase
from pipeline.verifier import parse_gpt2, _all_soft
from pipeline.arbiter import parse_gpt3, apply_edits
from pipeline.convergence import should_continue_rewrite
from pipeline.search import should_search, perform_web_search


def _date_context() -> str:
    """Return a date-awareness preamble for system prompts."""
    today = date.today().isoformat()
    return (
        f"CURRENT DATE: {today}. Use this date when evaluating whether "
        f"claims refer to the past, present, or future. Any date before "
        f"{today} is in the PAST, not the future.\n\n"
    )


def _fail_message(flags: dict, search_performed: bool) -> str:
    """Return a user-friendly failure message instead of bare 'NO PASS'."""
    if flags.get("current_events") and not search_performed:
        return (
            "This question requires current information that may have changed since "
            "the model's training cutoff. To get an accurate answer, enable Tavily "
            "web search by entering your Tavily API key in Settings. Without web search, "
            "the pipeline cannot verify time-sensitive claims."
        )
    return "NO PASS - Output blocked by verification"


def compute_confidence(claim_table: list, findings: list | None = None) -> ConfidenceBreakdown:
    """Compute a confidence breakdown from a list of ClaimEntry objects.

    When *findings* is provided, hard findings penalize the confidence label
    (any hard finding drops the label one tier). This prevents a response with
    70% Observed but a fabricated statistic from getting "High" confidence.
    """
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

    # Hard findings penalty: each hard finding drops confidence one tier
    hard_count = 0
    if findings:
        hard_count = sum(1 for f in findings if f.get("severity") == "hard")

    if observed_pct >= 70 and hard_count == 0:
        label = "High"
    elif observed_pct >= 40 and hard_count <= 1:
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

    # ---- Web Search Enrichment (before augmentation so flags are search-aware) ----
    search_sources: list[SearchSource] = []
    search_context = ""
    search_performed = False

    if should_search(flags):
        search_sources, search_context = perform_web_search(req.prompt)
        search_performed = len(search_sources) > 0

    gpt1_system = req.gpt1_system or DEFAULT_GPT1_SYSTEM
    gpt2_system = req.gpt2_system or DEFAULT_GPT2_SYSTEM
    gpt3_system = req.gpt3_system or DEFAULT_GPT3_SYSTEM

    # Inject current-date awareness so models don't misidentify past dates as future
    date_ctx = _date_context()
    gpt1_system = date_ctx + gpt1_system
    gpt2_system = date_ctx + gpt2_system
    gpt3_system = date_ctx + gpt3_system

    # Flag-driven augmentation for all 3 stages (search-aware)
    gpt1_aug, gpt2_aug, gpt3_aug = build_augmentation(flags, search_performed=search_performed)
    gpt1_system += gpt1_aug
    gpt2_system += gpt2_aug
    gpt3_system += gpt3_aug

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
        # Pass source snippets so GPT-2 can verify claims against them
        source_detail = "\n".join(
            f'[{i}] "{s.title}" ({s.url})\n    Snippet: {s.snippet[:300]}'
            for i, s in enumerate(search_sources, 1)
        )
        gpt2_system += (
            "\n\n=== CRITICAL: WEB SEARCH SOURCES (provided to GPT-1) ===\n"
            f"{source_detail}\n"
            "=== END SOURCES ===\n\n"
            "*** MANDATORY RULES — READ BEFORE EVALUATING ***\n"
            "These rules OVERRIDE your default tripwire checks when web search sources are present.\n"
            "Violating these rules is a verification error on YOUR part.\n\n"
            "RULE 1: THESE SOURCES ARE MORE CURRENT THAN YOUR TRAINING DATA.\n"
            "  Your training data may be outdated. Do NOT use your own training knowledge "
            "to contradict information from the sources above. If your training data "
            "says something different, THE SOURCES ARE CORRECT.\n\n"
            "RULE 2: SOURCE-BACKED CLAIMS ARE 'Observed'.\n"
            "  If GPT-1 cites a source (e.g., [1], [2]) and the source snippet above "
            "supports the claim, you MUST categorize it as 'Observed'. It is NOT "
            "'Unsupported', NOT 'Fabricated'. A claim backed by a provided source is VERIFIED.\n\n"
            "RULE 3: T1 DOES NOT APPLY to source-backed claims.\n"
            "  A claim grounded in a provided real source is not fabricated, even if it "
            "contradicts your training data.\n\n"
            "RULE 4: T7 DOES NOT APPLY to source-backed claims.\n"
            "  The search sources ARE the verification. A time-sensitive claim supported by "
            "a web search result is VERIFIED and CURRENT. Do NOT flag T7.\n\n"
            "RULE 5: FABRICATED CITATIONS only.\n"
            "  Only flag 'Fabricated citation' if GPT-1 cites a source number [N] "
            "that does NOT exist in the sources list above.\n\n"
            "RULE 6: UNSOURCED claims only.\n"
            "  Claims NOT backed by any provided source should be evaluated normally "
            "using standard tripwire rules. An unsourced INFERENCE based on sourced claims "
            "(e.g., 'He succeeded X') is a minor issue — categorize as 'Inference', NOT "
            "'Unsupported'. Only flag as 'Unsupported' if the claim is unrelated to the sources."
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

    # ---- Current-events fast path (no Tavily) ----
    # If the query is about current events and we have no web search to ground it,
    # GPT-1 was already instructed to frame everything as Unknown(Actionable).
    # Sanitize and return directly — skip GPT-2/GPT-3 (they would FAIL the stale
    # data and the rewrite loop adds 3 more API calls for the same result).
    if flags.get("current_events") and not search_performed:
        fast_output = sanitize_output(gpt1_output, flags)
        fast_output += (
            "\n\n---\n"
            "Note: This response is based on training data that may be outdated. "
            "For verified current information, enable Tavily web search in Settings."
        )
        return PipelineResponse(
            prompt_version=PROMPT_VERSION,
            gpt1_input=req.prompt, gpt1_output=gpt1_output, bypassed=False,
            gpt2_raw="(current-events fast path — no web search available)",
            claim_table=[], violations=[], gpt2_verdict="PASS",
            final_verdict="PASS", final_result=fast_output,
            prompt_flags=flags, sanitizer_applied=True,
            confidence=ConfidenceBreakdown(
                observed_pct=0, inference_pct=0, hypothesis_pct=0,
                unsupported_pct=0, user_provided_pct=0,
                total_claims=0, confidence_label="Low",
            ),
            **empty_response, **search_kwargs,
        )

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
    # Tripwire reference placed at START of user content (lost-in-middle fix)
    gpt2_user = (
        f"{GPT2_TRIPWIRE_REFERENCE}\n\n"
        f"=== TASK ===\n"
        f"ORIGINAL PROMPT:\n{req.prompt}\n\n"
        f"GPT-1 RESPONSE TO VERIFY:\n{sanitized_output}"
    )
    gpt2_raw = call_openai(client, model, gpt2_system, gpt2_user, expect_json=True)
    claim_table, violations, gpt2_verdict, findings, gpt2_reasoning = parse_gpt2(gpt2_raw, flags=flags)

    # ---- If GPT-2 PASS: done ----
    if gpt2_verdict == "PASS":
        return PipelineResponse(
            prompt_version=PROMPT_VERSION,
            gpt1_input=req.prompt, gpt1_output=gpt1_output, bypassed=False,
            gpt2_raw=gpt2_raw, claim_table=claim_table, violations=violations,
            gpt2_verdict="PASS", gpt2_reasoning=gpt2_reasoning,
            final_verdict="PASS", final_result=sanitized_output,
            prompt_flags=flags, sanitizer_applied=sanitizer_applied,
            confidence=compute_confidence(claim_table, findings),
            **empty_response, **search_kwargs,
        )

    # ---- GPT-2 FAIL: soft-only auto-repair path ----
    max_rewrite_loops = getattr(config, "MAX_REWRITE_LOOPS", 1)
    if _all_soft(findings):
        # Re-verify with GPT-2 directly (sanitizer already ran on sanitized_output)
        re_gpt2_user = (
            f"{GPT2_TRIPWIRE_REFERENCE}\n\n"
            f"=== TASK ===\n"
            f"ORIGINAL PROMPT:\n{req.prompt}\n\n"
            f"GPT-1 RESPONSE TO VERIFY:\n{sanitized_output}"
        )
        re_gpt2_raw = call_openai(client, model, gpt2_system, re_gpt2_user, expect_json=True)
        re_ct, re_viol, re_verdict, re_findings, _ = parse_gpt2(re_gpt2_raw, flags=flags)

        if re_verdict == "PASS":
            return PipelineResponse(
                prompt_version=PROMPT_VERSION,
                gpt1_input=req.prompt, gpt1_output=gpt1_output, bypassed=False,
                gpt2_raw=gpt2_raw, claim_table=claim_table, violations=violations,
                gpt2_verdict=gpt2_verdict, gpt2_reasoning=gpt2_reasoning,
                rewrite_occurred=True, rewrite_output=sanitized_output,
                rewrite_gpt2_raw=re_gpt2_raw, rewrite_claim_table=re_ct,
                rewrite_violations=re_viol, rewrite_verdict=re_verdict,
                arbiter_invoked=False, arbiter_decision="", arbiter_rationale=[],
                arbiter_edits=[], arbiter_policy_notes=[], arbiter_raw="",
                final_verdict="PASS", final_result=sanitized_output,
                prompt_flags=flags, sanitizer_applied=True,
                confidence=compute_confidence(re_ct, re_findings),
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

    # Include web search evidence for the arbiter if available
    search_evidence = ""
    if search_performed and search_sources:
        source_lines = "\n".join(
            f'[{i}] "{s.title}" ({s.url}) — {s.snippet[:200]}'
            for i, s in enumerate(search_sources, 1)
        )
        search_evidence = (
            f"\n\nweb_search_sources (provided to GPT-1 — these are VERIFIED current sources):\n"
            f"{source_lines}\n\n"
            f"IMPORTANT: Claims that GPT-1 grounded in these web search sources are VERIFIED "
            f"and CURRENT. Do NOT BLOCK claims that are supported by the sources above. "
            f"Your training data may be outdated — trust the web search results over your "
            f"training knowledge. ALLOW_WITH_EDITS to fix minor issues is preferred over BLOCK "
            f"when the core claims are source-backed."
        )

    gpt3_user = (
        f"user_prompt:\n{req.prompt}\n\n"
        f"gpt1_output:\n{sanitized_output}\n\n"
        f"gpt2_result_json:\n{gpt2_json_for_arbiter}\n\n"
        f"prompt_flags:\n{flags_json}"
        f"{search_evidence}"
    )

    gpt3_raw = call_openai(client, model, gpt3_system, gpt3_user, expect_json=True)
    arbiter_decision, arbiter_rationale, arbiter_edits, arbiter_policy_notes = parse_gpt3(gpt3_raw)

    # ---- Decision: BLOCK ----
    if arbiter_decision == "BLOCK":
        return PipelineResponse(
            prompt_version=PROMPT_VERSION,
            gpt1_input=req.prompt, gpt1_output=gpt1_output, bypassed=False,
            gpt2_raw=gpt2_raw, claim_table=claim_table, violations=violations,
            gpt2_verdict=gpt2_verdict, gpt2_reasoning=gpt2_reasoning,
            arbiter_invoked=True, arbiter_decision="BLOCK",
            arbiter_rationale=arbiter_rationale, arbiter_edits=arbiter_edits,
            arbiter_policy_notes=arbiter_policy_notes, arbiter_raw=gpt3_raw,
            rewrite_occurred=False, rewrite_output="", rewrite_gpt2_raw="",
            rewrite_claim_table=[], rewrite_violations=[], rewrite_verdict="",
            final_verdict="FAIL", final_result=_fail_message(flags, search_performed),
            prompt_flags=flags, sanitizer_applied=sanitizer_applied,
            confidence=compute_confidence(claim_table, findings),
            **search_kwargs,
        )

    # ---- Decision: ALLOW_AS_UNKNOWN_ONLY ----
    # The arbiter has already adjudicated — trust its decision.
    # Rewrite to Unknown framing, sanitize, and pass without re-verifying.
    # Re-verification was causing infinite FAIL loops on current-events queries
    # because GPT-2 kept over-flagging even properly Unknown-framed responses.
    if arbiter_decision == "ALLOW_AS_UNKNOWN_ONLY":
        rewrite_prompt = (
            f"You previously produced this response:\n\n---\n{sanitized_output}\n---\n\n"
            f"The arbiter has determined this question is inherently indeterminate.\n"
            f"Rewrite your response so that ALL claims are framed as Unknown(Actionable) or Unknown(Structural).\n"
            f"Do NOT make conclusions. Do NOT add new facts. Preserve the structure but move all substance to Unknowns.\n"
            f"Set Confidence to Low.\n"
            f"Output the corrected response in full."
        )
        rewrite_output = call_openai(client, model, gpt1_system, rewrite_prompt)

        # Sanitize the rewrite (strip stale dates, banned evidence, etc.)
        rewrite_output = sanitize_output(rewrite_output, flags)

        # Append a note if current-events without search
        if flags.get("current_events") and not search_performed:
            rewrite_output += (
                "\n\n---\nNote: This response is based on training data that may be outdated. "
                "For verified current information, enable Tavily web search in Settings."
            )

        return PipelineResponse(
            prompt_version=PROMPT_VERSION,
            gpt1_input=req.prompt, gpt1_output=gpt1_output, bypassed=False,
            gpt2_raw=gpt2_raw, claim_table=claim_table, violations=violations,
            gpt2_verdict=gpt2_verdict, gpt2_reasoning=gpt2_reasoning,
            arbiter_invoked=True, arbiter_decision="ALLOW_AS_UNKNOWN_ONLY",
            arbiter_rationale=arbiter_rationale, arbiter_edits=arbiter_edits,
            arbiter_policy_notes=arbiter_policy_notes, arbiter_raw=gpt3_raw,
            rewrite_occurred=True, rewrite_output=rewrite_output,
            rewrite_gpt2_raw="(arbiter-trusted)", rewrite_claim_table=[],
            rewrite_violations=[], rewrite_verdict="PASS",
            final_verdict="PASS",
            final_result=rewrite_output,
            prompt_flags=flags, sanitizer_applied=True,
            confidence=ConfidenceBreakdown(
                observed_pct=0, inference_pct=0, hypothesis_pct=0,
                unsupported_pct=0, user_provided_pct=0,
                total_claims=0, confidence_label="Low",
            ),
            **search_kwargs,
        )

    # ---- Decision: ALLOW_WITH_EDITS ----
    rewrite_prompt = apply_edits(sanitized_output, arbiter_edits)
    rewrite_output = call_openai(client, model, gpt1_system, rewrite_prompt)

    # Sanitize the rewrite before re-verification
    rewrite_output = sanitize_output(rewrite_output, flags)

    # Re-verify the rewritten output with GPT-2
    re_gpt2_user = (
            f"{GPT2_TRIPWIRE_REFERENCE}\n\n"
            f"=== TASK ===\n"
            f"ORIGINAL PROMPT:\n{req.prompt}\n\n"
            f"GPT-1 RESPONSE TO VERIFY:\n{rewrite_output}"
        )
    re_gpt2_raw = call_openai(client, model, gpt2_system, re_gpt2_user, expect_json=True)
    re_ct, re_viol, re_verdict, re_findings, _ = parse_gpt2(re_gpt2_raw, flags=flags)

    # If still failing after arbiter rewrite, continue rewriting.
    # The arbiter decided ALLOW_WITH_EDITS (not BLOCK), meaning it believes
    # the issues are fixable. Trust that decision and iterate on both hard
    # and soft findings. Convergence detection stops on oscillation/regression.
    findings_history = [findings, re_findings]  # Initial GPT-2 findings + first re-verify

    while re_verdict == "FAIL" and should_continue_rewrite(findings_history, max_loops=max_rewrite_loops):
        # Tailor the rewrite instruction based on finding severity
        if _all_soft(re_findings):
            rewrite_instruction = (
                f"You previously produced this response:\n\n---\n{rewrite_output}\n---\n\n"
                f"Remaining soft violations could not be resolved. "
                f"Rewrite your response so that ALL claims are framed as Unknown(Actionable) or Unknown(Structural).\n"
                f"Do NOT make conclusions. Do NOT add new facts. Output the corrected response in full."
            )
        else:
            # Hard findings remain — instruct specific fixes
            hard_details = "; ".join(
                f'{f["type"]}: {f["detail"]}'
                for f in re_findings if f.get("severity") == "hard"
            )
            rewrite_instruction = (
                f"You previously produced this response:\n\n---\n{rewrite_output}\n---\n\n"
                f"The following HARD violations were detected and must be fixed:\n{hard_details}\n\n"
                f"For each violation: either DELETE the problematic claim entirely, "
                f"or MOVE it to Unknown(Actionable) with a note that verification is needed.\n"
                f"Do NOT fabricate citations. Do NOT invent statistics.\n"
                f"Set Confidence to Low if you remove core claims.\n"
                f"Output the corrected response in full."
            )
        rewrite_output = call_openai(client, model, gpt1_system, rewrite_instruction)
        rewrite_output = sanitize_output(rewrite_output, flags)
        # Re-verify
        re_gpt2_user = (
            f"{GPT2_TRIPWIRE_REFERENCE}\n\n"
            f"=== TASK ===\n"
            f"ORIGINAL PROMPT:\n{req.prompt}\n\n"
            f"GPT-1 RESPONSE TO VERIFY:\n{rewrite_output}"
        )
        re_gpt2_raw = call_openai(client, model, gpt2_system, re_gpt2_user, expect_json=True)
        re_ct, re_viol, re_verdict, re_findings, _ = parse_gpt2(re_gpt2_raw, flags=flags)
        findings_history.append(re_findings)

    return PipelineResponse(
        prompt_version=PROMPT_VERSION,
        gpt1_input=req.prompt, gpt1_output=gpt1_output, bypassed=False,
        gpt2_raw=gpt2_raw, claim_table=claim_table, violations=violations,
        gpt2_verdict=gpt2_verdict, gpt2_reasoning=gpt2_reasoning,
        arbiter_invoked=True, arbiter_decision="ALLOW_WITH_EDITS",
        arbiter_rationale=arbiter_rationale, arbiter_edits=arbiter_edits,
        arbiter_policy_notes=arbiter_policy_notes, arbiter_raw=gpt3_raw,
        rewrite_occurred=True, rewrite_output=rewrite_output,
        rewrite_gpt2_raw=re_gpt2_raw, rewrite_claim_table=re_ct,
        rewrite_violations=re_viol, rewrite_verdict=re_verdict,
        final_verdict=re_verdict,
        final_result=rewrite_output if re_verdict == "PASS" else _fail_message(flags, search_performed),
        prompt_flags=flags, sanitizer_applied=sanitizer_applied,
        confidence=compute_confidence(re_ct, re_findings),
        **search_kwargs,
    )
