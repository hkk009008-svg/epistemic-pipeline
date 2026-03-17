"""Async pipeline stage functions — decomposed from run_pipeline_async.

Each stage takes a PipelineState dict, reads what it needs, performs one
logical step, and returns a dict of updated fields. The orchestrator
merges updates via state.update(await stage_fn(state)).

Stages that need to short-circuit the pipeline return an "early_return"
key containing the final PipelineResponse.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from datetime import date
from typing import Optional

import config
from pipeline.pipeline_state import PipelineState
from pipeline.models import (
    PipelineRequest, PipelineResponse, ConfidenceBreakdown,
    SearchSource, GPT2ResponseSchema, GPT3ResponseSchema,
)
from pipeline.prompts import (
    DEFAULT_GPT1_SYSTEM, DEFAULT_GPT2_SYSTEM, DEFAULT_GPT3_SYSTEM,
    GPT2_TRIPWIRE_REFERENCE, PROMPT_VERSION, build_augmentation,
)
from pipeline.sanitizer import route_prompt, sanitize_output
from pipeline.helpers import PipelineError, call_llm_async, call_llm_structured, is_activation_phrase
from pipeline.verifier import parse_gpt2, parse_gpt2_structured, _all_soft, recompute_verdict
from pipeline.arbiter import parse_gpt3, parse_gpt3_structured, apply_edits, apply_edits_by_id
from pipeline.convergence import should_continue_rewrite
from pipeline.search import should_search, perform_web_search, refine_search_query, fetch_claim_evidence
from pipeline.source_match import recategorize_with_sources, filter_findings_with_sources, build_source_keyword_sets
from pipeline.decomposer import decompose_claims
from pipeline.nli import verify_claims_with_nli, is_nli_available, compute_grounding_rate, detect_unsupported_spans
from pipeline.meta_verify import meta_verify_pass, meta_verify_fail, is_high_stakes
from pipeline.metrics import PipelineMetrics, record_run
from pipeline.best_of_n import generate_best_of_n_async

# Re-import orchestrator utilities (these stay in orchestrator.py)
from pipeline.orchestrator import (
    compute_confidence, clean_for_display, _fail_message,
    _date_context, _resolve_output_format, _emit_stage_start, _emit_stage_complete, _noop_emit,
    StageEventEmitter,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

async def _verify_text(state: PipelineState, text_to_verify: str) -> dict:
    """Run GPT-2 verification on a text, with structured output fallback + source-match.

    Deduplicates the identical verify-and-correct pattern that appears 4 times
    in the original orchestrator (initial verify, soft-retry, first rewrite, loop rewrites).

    Returns dict: {gpt2_raw, claim_table, violations, verdict, findings, reasoning}.
    """
    gpt2_user = (
        f"{GPT2_TRIPWIRE_REFERENCE}\n\n"
        f"=== TASK ===\n"
        f"ORIGINAL PROMPT:\n{state['prompt']}\n\n"
        f"GPT-1 RESPONSE TO VERIFY:\n{text_to_verify}"
    )
    gpt2_cfg = state["gpt2_cfg"]
    gpt2_system = state["gpt2_system"]
    flags = state["flags"]
    tier = state["tier"]

    try:
        parsed = await call_llm_structured(gpt2_cfg, gpt2_system, gpt2_user, GPT2ResponseSchema)
        gpt2_raw = parsed.model_dump_json()
        ct, viol, verdict, findings, reasoning = parse_gpt2_structured(parsed, flags=flags, tier=tier)
    except Exception:
        gpt2_raw = await call_llm_async(gpt2_cfg, gpt2_system, gpt2_user, expect_json=True)
        ct, viol, verdict, findings, reasoning = parse_gpt2(gpt2_raw, flags=flags, tier=tier)

    # Source-match correction
    search_sources = state.get("search_sources", [])
    if search_sources:
        src_kw = state.get("src_kw_sets")
        ct = recategorize_with_sources(ct, search_sources, src_kw)
        findings = filter_findings_with_sources(findings, search_sources, src_kw)
        viol = [f["type"] for f in findings]
        verdict = recompute_verdict(findings, tier=tier)

    return {
        "gpt2_raw": gpt2_raw,
        "claim_table": ct,
        "violations": viol,
        "verdict": verdict,
        "findings": findings,
        "reasoning": reasoning,
    }


def _base_response(state: PipelineState, **overrides) -> PipelineResponse:
    """Build a PipelineResponse from state with sensible defaults.

    Merges empty_arbiter defaults, search_kwargs, decomp_kwargs,
    then applies any overrides. This replaces the 9 copy-pasted
    PipelineResponse(...) blocks in the original orchestrator.
    """
    base = dict(
        prompt_version=PROMPT_VERSION,
        tier=state["tier"],
        output_format=state["output_format"],
        gpt1_input=state["prompt"],
        gpt1_output=state.get("gpt1_output", ""),
        prompt_flags=state.get("flags", {}),
        bypassed=state.get("is_bypassed", False),
        gpt2_raw=state.get("gpt2_raw", ""),
        claim_table=state.get("claim_table", []),
        violations=state.get("violations", []),
        gpt2_verdict=state.get("gpt2_verdict", "PASS"),
        final_verdict=state.get("final_verdict", "PASS"),
        final_result=state.get("final_result", state.get("gpt1_output", "")),
    )
    base.update(state.get("empty_arbiter", {}))
    base.update(state.get("search_kwargs", {}))
    base.update(state.get("decomp_kwargs", {}))
    base.update(overrides)

    # Strip internal sanitizer/epistemic markers before returning to users
    if "final_result" in base and base["final_result"]:
        base["final_result"] = clean_for_display(base["final_result"])

    return PipelineResponse(**base)


# ---------------------------------------------------------------------------
# Stage: init
# ---------------------------------------------------------------------------

async def stage_init(state: PipelineState) -> dict:
    """Validate config, resolve tier/format, create metrics, load stage configs."""
    if not config.has_api_key():
        raise PipelineError(400, "Set your OpenAI API key first.")

    req = state["request"]
    metrics = PipelineMetrics(request_id=uuid.uuid4().hex[:12], prompt_length=len(req.prompt))
    metrics.start()

    tier = getattr(req, "tier", "strict") or "strict"
    output_format = _resolve_output_format(tier, getattr(req, "output_format", "auto") or "auto")

    return {
        "prompt": req.prompt,
        "metrics": metrics,
        "gpt1_cfg": config.get_stage_config("gpt1"),
        "gpt2_cfg": config.get_stage_config("gpt2"),
        "gpt3_cfg": config.get_stage_config("gpt3"),
        "tier": tier,
        "output_format": output_format,
        "empty_arbiter": {
            "arbiter_invoked": False, "arbiter_decision": "", "arbiter_rationale": [],
            "arbiter_edits": [], "arbiter_policy_notes": [], "arbiter_raw": "",
            "rewrite_occurred": False, "rewrite_output": "", "rewrite_gpt2_raw": "",
            "rewrite_claim_table": [], "rewrite_violations": [], "rewrite_verdict": "",
        }
    }


# ---------------------------------------------------------------------------
# Stage: route
# ---------------------------------------------------------------------------

async def stage_route(state: PipelineState) -> dict:
    """Deterministic prompt routing — extract flags."""
    emit = state["emit"]
    _emit_stage_start(emit, "routing")
    flags = route_prompt(state["prompt"])
    state["metrics"].flags = flags
    _emit_stage_complete(emit, "routing", data={"flags": flags})
    return {"flags": flags}


# ---------------------------------------------------------------------------
# Stage: search
# ---------------------------------------------------------------------------

async def stage_search(state: PipelineState) -> dict:
    """Tavily web search enrichment (if flags warrant it)."""
    flags = state["flags"]
    emit = state["emit"]
    metrics = state["metrics"]
    result: dict = {
        "search_sources": [],
        "search_context": "",
        "search_performed": False,
        "search_attempted": False,
        "search_note": "",
    }

    if should_search(flags):
        result["search_attempted"] = True
        _emit_stage_start(emit, "search", data={"query": state["prompt"]})
        sm = metrics.start_stage("search")
        sources, context = await asyncio.to_thread(perform_web_search, state["prompt"])
        performed = len(sources) > 0
        metrics.end_stage(sm)
        _emit_stage_complete(emit, "search", data={"source_count": len(sources), "performed": performed})
        metrics.search_performed = performed
        metrics.search_sources_count = len(sources)
        result["search_sources"] = sources
        result["search_context"] = context
        result["search_performed"] = performed
        if not performed:
            result["search_note"] = "Web search was enabled but returned no relevant sources for this query."

    return result


# ---------------------------------------------------------------------------
# Stage: build_prompts
# ---------------------------------------------------------------------------

async def stage_build_prompts(state: PipelineState) -> dict:
    """Assemble system prompts with date context, augmentation, and search rules."""
    req = state["request"]
    flags = state["flags"]
    tier = state["tier"]
    output_format = state["output_format"]
    search_performed = state["search_performed"]
    search_sources = state.get("search_sources", [])
    search_context = state.get("search_context", "")

    gpt1_system = req.gpt1_system or DEFAULT_GPT1_SYSTEM
    gpt2_system = req.gpt2_system or DEFAULT_GPT2_SYSTEM
    date_ctx = _date_context()
    gpt1_system = date_ctx + gpt1_system
    gpt2_system = date_ctx + gpt2_system

    gpt1_aug, gpt2_aug, gpt3_aug = build_augmentation(
        flags, search_performed=search_performed, tier=tier, output_format=output_format,
    )
    gpt1_system += gpt1_aug
    gpt2_system += gpt2_aug
    gpt3_system = date_ctx + (
        req.gpt3_system if hasattr(req, "gpt3_system") and req.gpt3_system
        else DEFAULT_GPT3_SYSTEM
    ) + gpt3_aug

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
            "  A claim is 'source-backed' if EITHER: (a) GPT-1 explicitly cites it with [1], [2], etc., "
            "OR (b) the claim is factually consistent with ANY source snippet listed above. "
            "You MUST check each claim's content against ALL source snippets. "
            "If the information appears in any snippet, categorize the claim as 'Observed'. "
            "It is NOT 'Unsupported', NOT 'Fabricated'. Do NOT require explicit [N] markers — "
            "content match is sufficient.\n\n"
            "RULE 3: T1 DOES NOT APPLY to source-backed claims.\n"
            "  A claim grounded in a provided real source (by citation OR content match) "
            "is not fabricated, even if it contradicts your training data.\n\n"
            "RULE 4: T7 DOES NOT APPLY to source-backed claims.\n"
            "  The search sources ARE the verification. A time-sensitive claim supported by "
            "a web search result (by citation OR content match) is VERIFIED and CURRENT. Do NOT flag T7.\n\n"
            "RULE 5: FABRICATED CITATIONS only.\n"
            "  Only flag 'Fabricated citation' if GPT-1 cites a source number [N] "
            "that does NOT exist in the sources list above.\n\n"
            "RULE 6: UNSOURCED claims only.\n"
            "  A claim is 'unsourced' ONLY if its content does NOT appear in ANY source snippet above "
            "AND GPT-1 does not cite it. An unsourced INFERENCE based on sourced claims "
            "(e.g., 'He succeeded X') is a minor issue — categorize as 'Inference', NOT "
            "'Unsupported'. Only flag as 'Unsupported' if the claim is completely unrelated to all sources."
        )

    search_kwargs = dict(
        search_performed=search_performed,
        search_attempted=state.get("search_attempted", False),
        search_note=state.get("search_note", ""),
        search_query=req.prompt if search_performed else "",
        search_sources=search_sources,
    )
    src_kw_sets = build_source_keyword_sets(search_sources) if search_sources else None
    empty_arbiter = dict(
        arbiter_invoked=False, arbiter_decision="", arbiter_rationale=[],
        arbiter_edits=[], arbiter_policy_notes=[], arbiter_raw="",
        rewrite_occurred=False, rewrite_output="", rewrite_gpt2_raw="",
        rewrite_claim_table=[], rewrite_violations=[], rewrite_verdict="",
    )

    return {
        "gpt1_system": gpt1_system,
        "gpt2_system": gpt2_system,
        "gpt3_system": gpt3_system,
        "gpt1_user_content": gpt1_user_content,
        "src_kw_sets": src_kw_sets,
        "search_kwargs": search_kwargs,
        "empty_arbiter": empty_arbiter,
    }


# ---------------------------------------------------------------------------
# Stage: generate
# ---------------------------------------------------------------------------

async def stage_generate(state: PipelineState) -> dict:
    """GPT-1 generation with optional best-of-N."""
    emit = state["emit"]
    gpt1_cfg = state["gpt1_cfg"]
    metrics = state["metrics"]

    _emit_stage_start(emit, "gpt1", data={
        "provider": gpt1_cfg.get("provider", ""),
        "model": gpt1_cfg.get("model", ""),
    })
    gpt1_sm = metrics.start_stage("gpt1", gpt1_cfg.get("provider", ""), gpt1_cfg.get("model", ""))

    best_of_n_count = getattr(config, "BEST_OF_N", 1)
    if best_of_n_count >= 2:
        gpt1_output, _bon_info = await generate_best_of_n_async(
            gpt1_cfg, state["gpt1_system"], state["gpt1_user_content"],
            state["flags"], n=best_of_n_count,
        )
    else:
        gpt1_output = await call_llm_async(gpt1_cfg, state["gpt1_system"], state["gpt1_user_content"])

    metrics.end_stage(gpt1_sm)
    _emit_stage_complete(emit, "gpt1")
    return {"gpt1_output": gpt1_output}


# ---------------------------------------------------------------------------
# Stage: check_fast_paths (current-events bypass + activation bypass)
# ---------------------------------------------------------------------------

async def stage_check_fast_paths(state: PipelineState) -> dict:
    """Check for early exits before verification: current-events and activation bypass."""
    flags = state["flags"]
    gpt1_output = state["gpt1_output"]
    metrics = state["metrics"]

    # Current-events fast path: no web search available
    if flags.get("current_events") and not state["search_performed"]:
        fast_output = sanitize_output(gpt1_output, flags, tier=state["tier"])
        fast_output += (
            "\n\n---\n"
            "Note: This response is based on training data that may be outdated. "
            "For verified current information, enable Tavily web search in Settings."
        )
        metrics.final_verdict = "PASS"
        metrics.confidence_label = "Low"
        metrics.bypassed = True
        metrics.finish()
        record_run(metrics)
        return {
            "is_bypassed": True,
            "final_verdict": "PASS",
            "final_result": fast_output,
            "sanitizer_applied": True,
            "gpt2_raw": "(current-events fast path — no web search available)",
            "gpt2_verdict": "PASS",
            "claim_table": [],
            "violations": [],
            "confidence": ConfidenceBreakdown(
                observed_pct=0, inference_pct=0, hypothesis_pct=0,
                unsupported_pct=0, user_provided_pct=0, total_claims=0, confidence_label="Low",
            )
        }

    # Activation bypass
    if is_activation_phrase(gpt1_output):
        metrics.bypassed = True
        metrics.final_verdict = "PASS"
        metrics.finish()
        record_run(metrics)
        return {
            "is_bypassed": True,
            "final_verdict": "PASS",
            "final_result": gpt1_output,
            "sanitizer_applied": False,
            "gpt2_raw": "(bypassed)",
            "gpt2_verdict": "PASS",
            "claim_table": [],
            "violations": [],
            "confidence": compute_confidence([]),
        }

    return {}


# ---------------------------------------------------------------------------
# Stage: sanitize
# ---------------------------------------------------------------------------

async def stage_sanitize(state: PipelineState) -> dict:
    """Apply deterministic citation-aware sanitizer to GPT-1 output."""
    sanitized = sanitize_output(state["gpt1_output"], state["flags"], tier=state["tier"])
    return {
        "sanitized_output": sanitized,
        "sanitizer_applied": sanitized != state["gpt1_output"],
    }


# ---------------------------------------------------------------------------
# Stage: decompose
# ---------------------------------------------------------------------------

async def stage_decompose(state: PipelineState) -> dict:
    """Atomic claim decomposition (optional, pre-GPT-2)."""
    flags = state["flags"]
    emit = state["emit"]
    metrics = state["metrics"]
    do_decompose = is_nli_available() or is_high_stakes(flags)

    if not do_decompose:
        return {"atomic_claims": [], "decomposition_ran": False}

    _emit_stage_start(emit, "decomposition")
    sm = metrics.start_stage("decomposition")
    claims = await asyncio.to_thread(
        decompose_claims, state["gpt2_cfg"], state["sanitized_output"], state["prompt"],
    )
    metrics.end_stage(sm)
    _emit_stage_complete(emit, "decomposition", data={"claim_count": len(claims)})
    metrics.decomposition_ran = len(claims) > 0
    metrics.atomic_claims_count = len(claims)
    return {"atomic_claims": claims, "decomposition_ran": len(claims) > 0}


# ---------------------------------------------------------------------------
# Stage: nli
# ---------------------------------------------------------------------------

async def stage_nli(state: PipelineState) -> dict:
    """NLI pre-verification of atomic claims against evidence."""
    atomic_claims = state.get("atomic_claims", [])
    if not atomic_claims or not is_nli_available():
        return {"nli_grounding": {}, "nli_unsupported_spans": []}

    search_sources = state.get("search_sources", [])
    evidence = [s.snippet for s in search_sources] if search_sources else []
    if not evidence:
        return {"nli_grounding": {}, "nli_unsupported_spans": []}

    emit = state["emit"]
    metrics = state["metrics"]
    _emit_stage_start(emit, "nli")
    sm = metrics.start_stage("nli")

    enriched_claims = await asyncio.to_thread(verify_claims_with_nli, atomic_claims, evidence)

    metrics.end_stage(sm)
    metrics.nli_ran = True
    metrics.nli_supported_count = sum(
        1 for c in enriched_claims if c.get("nli_result", {}).get("supported")
    )
    metrics.nli_contradicted_count = sum(
        1 for c in enriched_claims if c.get("nli_result", {}).get("contradicted")
    )
    grounding = compute_grounding_rate(enriched_claims)
    metrics.grounding_rate = grounding.get("grounding_rate", 0.0)
    unsupported_spans = detect_unsupported_spans(state["sanitized_output"], enriched_claims)

    _emit_stage_complete(emit, "nli", data={
        "grounding_rate": grounding.get("grounding_rate", 0.0),
        "supported": metrics.nli_supported_count,
        "contradicted": metrics.nli_contradicted_count,
    })

    return {
        "atomic_claims": enriched_claims,
        "nli_grounding": grounding,
        "nli_unsupported_spans": unsupported_spans,
        "decomp_kwargs": dict(atomic_claims=enriched_claims, decomposition_ran=True),
    }


# ---------------------------------------------------------------------------
# Stage: verify (GPT-2)
# ---------------------------------------------------------------------------

async def stage_verify(state: PipelineState) -> dict:
    """GPT-2 verification with structured outputs, source-match, claim retrieval.

    Sets early_return if verdict is PASS (with meta-verify adjustments).
    """
    emit = state["emit"]
    metrics = state["metrics"]
    gpt2_cfg = state["gpt2_cfg"]
    atomic_claims = state.get("atomic_claims", [])
    nli_grounding = state.get("nli_grounding", {})
    flags = state["flags"]
    tier = state["tier"]

    # Build GPT-2 user content (with NLI signals if available)
    if atomic_claims:
        claims_json = json.dumps(atomic_claims, indent=2)
        nli_block = ""
        nli_lines = []
        for c in atomic_claims:
            nli = c.get("nli_result", {})
            nli_tier = nli.get("confidence_tier", "")
            if nli_tier == "strong_support":
                nli_lines.append(f'  NLI-STRONG-SUPPORT (ent={nli["best_entailment"]:.2f}): "{c["text"][:80]}"')
            elif nli_tier == "weak_support":
                nli_lines.append(f'  NLI-WEAK-SUPPORT (ent={nli["best_entailment"]:.2f}): "{c["text"][:80]}"')
            elif nli_tier == "strong_contradiction":
                nli_lines.append(f'  NLI-CONTRADICTED (con={nli["worst_contradiction"]:.2f}): "{c["text"][:80]}"')
            elif nli_tier == "weak_contradiction":
                nli_lines.append(f'  NLI-WEAK-CONTRADICTION (con={nli["worst_contradiction"]:.2f}): "{c["text"][:80]}"')
        if nli_lines:
            grounding_str = ""
            if nli_grounding:
                grounding_str = (
                    f"\nGrounding Rate: {nli_grounding['grounding_rate']:.1%} "
                    f"({nli_grounding['grounded_count']}/{nli_grounding['total_evaluated']} claims grounded)"
                )
            nli_block = "\n\nNLI PRE-VERIFICATION SIGNALS:\n" + "\n".join(nli_lines) + grounding_str

        gpt2_user = (
            f"{GPT2_TRIPWIRE_REFERENCE}\n\n"
            f"=== TASK ===\n"
            f"ORIGINAL PROMPT:\n{state['prompt']}\n\n"
            f"GPT-1 RESPONSE TO VERIFY:\n{state['sanitized_output']}\n\n"
            f"PRE-DECOMPOSED ATOMIC CLAIMS (verify each independently):\n{claims_json}"
            f"{nli_block}"
        )
    else:
        gpt2_user = (
            f"{GPT2_TRIPWIRE_REFERENCE}\n\n"
            f"=== TASK ===\n"
            f"ORIGINAL PROMPT:\n{state['prompt']}\n\n"
            f"GPT-1 RESPONSE TO VERIFY:\n{state['sanitized_output']}"
        )

    _emit_stage_start(emit, "gpt2", data={
        "provider": gpt2_cfg.get("provider", ""), "model": gpt2_cfg.get("model", ""),
    })
    gpt2_sm = metrics.start_stage("gpt2", gpt2_cfg.get("provider", ""), gpt2_cfg.get("model", ""))

    try:
        gpt2_parsed = await call_llm_structured(gpt2_cfg, state["gpt2_system"], gpt2_user, GPT2ResponseSchema)
        gpt2_raw = gpt2_parsed.model_dump_json()
        claim_table, violations, gpt2_verdict, findings, gpt2_reasoning = parse_gpt2_structured(
            gpt2_parsed, flags=flags, tier=tier,
        )
    except Exception:
        gpt2_raw = await call_llm_async(gpt2_cfg, state["gpt2_system"], gpt2_user, expect_json=True)
        claim_table, violations, gpt2_verdict, findings, gpt2_reasoning = parse_gpt2(
            gpt2_raw, flags=flags, tier=tier,
        )

    metrics.end_stage(gpt2_sm)
    _emit_stage_complete(emit, "gpt2", data={
        "verdict": gpt2_verdict, "claim_count": len(claim_table), "violations": violations,
    })

    # Source-match correction
    search_sources = state.get("search_sources", [])
    if search_sources:
        src_kw = state.get("src_kw_sets")
        claim_table = recategorize_with_sources(claim_table, search_sources, src_kw, nli_claims=atomic_claims or None)
        findings = filter_findings_with_sources(findings, search_sources, src_kw)
        violations = [f["type"] for f in findings]
        gpt2_verdict = recompute_verdict(findings, tier=tier)

    metrics.gpt2_verdict = gpt2_verdict
    metrics.total_claims = len(claim_table)
    metrics.hard_findings = sum(1 for f in findings if f.get("severity") == "hard")
    metrics.soft_findings = sum(1 for f in findings if f.get("severity") == "soft")

    # Claim-conditional retrieval
    if state.get("search_performed") and claim_table:
        unsupported_ct = sum(
            1 for ct in claim_table
            if (ct.category if isinstance(ct.category, str) else "").lower().strip() == "unsupported"
        )
        unsupported_ratio = unsupported_ct / len(claim_table)
        if unsupported_ratio > 0.3:
            _emit_stage_start(emit, "claim_retrieval", data={"unsupported_count": unsupported_ct})
            if atomic_claims:
                new_sources = await asyncio.to_thread(fetch_claim_evidence, atomic_claims, search_sources)
            else:
                unsupported_texts = [
                    ct.claim for ct in claim_table
                    if (ct.category if isinstance(ct.category, str) else "").lower().strip() == "unsupported"
                ]
                refined_query = refine_search_query(state["prompt"], unsupported_texts)
                retry_sources, _ = await asyncio.to_thread(perform_web_search, refined_query, 3)
                existing_urls = {s.url for s in search_sources}
                new_sources = [s for s in retry_sources if s.url not in existing_urls]

            if new_sources:
                search_sources = search_sources + new_sources
                src_kw_sets = build_source_keyword_sets(search_sources)
                claim_table = recategorize_with_sources(claim_table, search_sources, src_kw_sets, nli_claims=atomic_claims or None)
                findings = filter_findings_with_sources(findings, search_sources, src_kw_sets)
                violations = [f["type"] for f in findings]
                gpt2_verdict = recompute_verdict(findings, tier=tier)
                # Update search state
                state["search_sources"] = search_sources
                state["src_kw_sets"] = src_kw_sets
                state["search_kwargs"]["search_sources"] = search_sources
            _emit_stage_complete(emit, "claim_retrieval", data={
                "new_sources": len(new_sources) if new_sources else 0,
            })

    # Update decomp_kwargs (may have been set by stage_nli, needs update here)
    if "decomp_kwargs" not in state:
        state["decomp_kwargs"] = dict(atomic_claims=atomic_claims, decomposition_ran=len(atomic_claims) > 0)

    updates: dict = {
        "gpt2_raw": gpt2_raw,
        "claim_table": claim_table,
        "violations": violations,
        "gpt2_verdict": gpt2_verdict,
        "findings": findings,
        "gpt2_reasoning": gpt2_reasoning,
        "max_rewrite_loops": getattr(config, "MAX_REWRITE_LOOPS", 1),
    }

    # If PASS: compute confidence and return
    nli_unsupported_spans = state.get("nli_unsupported_spans", [])
    if gpt2_verdict == "PASS":
        conf = compute_confidence(claim_table, findings, nli_grounding or None, nli_unsupported_spans or None, search_sources or None)
        meta_result = meta_verify_pass(flags, claim_table, findings, atomic_claims, conf.confidence_label)
        if meta_result["ran"] and meta_result["adjusted_label"] != conf.confidence_label:
            conf.confidence_label = meta_result["adjusted_label"]

        metrics.final_verdict = "PASS"
        metrics.confidence_label = conf.confidence_label
        metrics.finish()
        record_run(metrics)
        updates.update({
            "is_pass": True,
            "final_verdict": "PASS",
            "final_result": state.get("sanitized_output", ""),
            "confidence": conf,
            "meta_verification": meta_result if meta_result["ran"] else None,
        })
        return updates

    # Meta-verify FAIL override
    fail_meta = meta_verify_fail(flags, claim_table, findings, atomic_claims)
    if fail_meta["ran"] and fail_meta["override_to_pass"]:
        findings = fail_meta["adjusted_findings"]
        violations = [f["type"] for f in findings]
        gpt2_verdict = recompute_verdict(findings, tier=tier)
        updates["findings"] = findings
        updates["violations"] = violations
        updates["gpt2_verdict"] = gpt2_verdict
        if gpt2_verdict == "PASS":
            conf = compute_confidence(claim_table, findings, nli_grounding or None, nli_unsupported_spans or None, search_sources or None)
            metrics.final_verdict = "PASS"
            metrics.confidence_label = conf.confidence_label
            metrics.finish()
            record_run(metrics)
            updates.update({
                "is_pass": True,
                "final_verdict": "PASS",
                "final_result": state.get("sanitized_output", ""),
                "confidence": conf,
                "meta_verification": {"type": "false_fail_override", "reason": fail_meta["reason"]},
            })
            return updates

    return updates


# ---------------------------------------------------------------------------
# Stage: soft_retry
# ---------------------------------------------------------------------------

async def stage_soft_retry(state: PipelineState) -> dict:
    """If all findings are soft-only, re-verify (GPT-2 may change its mind)."""
    if not _all_soft(state.get("findings", [])):
        return {}  # Not applicable — fall through to arbiter

    result = await _verify_text(state, state["sanitized_output"])

    if result["verdict"] == "PASS":
        nli_grounding = state.get("nli_grounding", {})
        nli_unsupported_spans = state.get("nli_unsupported_spans", [])
        search_sources = state.get("search_sources", [])
        conf = compute_confidence(
            result["claim_table"], result["findings"],
            nli_grounding or None, nli_unsupported_spans or None, search_sources or None,
        )
        metrics = state["metrics"]
        metrics.final_verdict = "PASS"
        metrics.confidence_label = conf.confidence_label
        metrics.rewrite_loops = 1
        metrics.convergence_outcome = "pass"
        metrics.finish()
        record_run(metrics)
        return {
            "is_pass": True, # Triggers early return in Graph
            "rewrite_occurred": True,
            "rewrite_output": state.get("sanitized_output", ""),
            "rewrite_gpt2_raw": result["gpt2_raw"],
            "rewrite_claim_table": result["claim_table"],
            "rewrite_violations": result["violations"],
            "rewrite_verdict": result["verdict"],
            "rewrite_reasoning": result["reasoning"],
            "arbiter_invoked": False,
            "arbiter_decision": "",
            "arbiter_rationale": [],
            "arbiter_edits": [],
            "final_verdict": "PASS",
            "final_result": state.get("sanitized_output", ""),
            "sanitizer_applied": True,
            "confidence": conf,
        }

    return {}


# ---------------------------------------------------------------------------
# Stage: arbiter (GPT-3)
# ---------------------------------------------------------------------------

async def stage_arbiter(state: PipelineState) -> dict:
    """GPT-3 Arbiter: BLOCK / ALLOW_WITH_EDITS / ALLOW_AS_UNKNOWN_ONLY.

    Sets early_return for BLOCK and ALLOW_AS_UNKNOWN_ONLY decisions.
    ALLOW_WITH_EDITS falls through to stage_rewrite_loop.
    """
    emit = state["emit"]
    metrics = state["metrics"]
    gpt3_cfg = state["gpt3_cfg"]
    flags = state["flags"]

    gpt3_user = (
        f"user_prompt:\n{state['prompt']}\n\n"
        f"gpt1_output:\n{state['sanitized_output']}\n\n"
        f"gpt2_result_json:\n{state['gpt2_raw']}\n\n"
        f"prompt_flags:\n{json.dumps(flags)}"
    )
    _emit_stage_start(emit, "gpt3", data={
        "provider": gpt3_cfg.get("provider", ""), "model": gpt3_cfg.get("model", ""),
    })
    gpt3_sm = metrics.start_stage("gpt3", gpt3_cfg.get("provider", ""), gpt3_cfg.get("model", ""))

    try:
        gpt3_parsed = await call_llm_structured(gpt3_cfg, state["gpt3_system"], gpt3_user, GPT3ResponseSchema)
        gpt3_raw = gpt3_parsed.model_dump_json()
        arbiter_decision, arbiter_rationale, arbiter_edits, arbiter_policy_notes = parse_gpt3_structured(gpt3_parsed)
    except Exception:
        gpt3_raw = await call_llm_async(gpt3_cfg, state["gpt3_system"], gpt3_user, expect_json=True)
        arbiter_decision, arbiter_rationale, arbiter_edits, arbiter_policy_notes = parse_gpt3(gpt3_raw)

    metrics.end_stage(gpt3_sm)
    _emit_stage_complete(emit, "gpt3", data={"decision": arbiter_decision})

    # Safety net: override BLOCK to ALLOW_WITH_EDITS when truthful content exists
    claim_table = state.get("claim_table", [])
    if arbiter_decision == "BLOCK" and claim_table:
        salvageable_cats = {"supported", "observed", "inference", "user-provided"}
        has_truthful = any(
            (ct.category if isinstance(ct.category, str) else "").lower().strip() in salvageable_cats
            for ct in claim_table
        )
        if has_truthful:
            arbiter_decision = "ALLOW_WITH_EDITS"
            arbiter_rationale = [
                "Overridden from BLOCK: claim table contains truthful content that can be preserved with edits."
            ] + arbiter_rationale

    metrics.arbiter_decision = arbiter_decision

    updates: dict = {
        "arbiter_invoked": True,
        "arbiter_decision": arbiter_decision,
        "arbiter_rationale": arbiter_rationale,
        "arbiter_edits": arbiter_edits,
        "arbiter_policy_notes": arbiter_policy_notes,
        "gpt3_raw": gpt3_raw,
    }

    # ---- BLOCK ----
    if arbiter_decision == "BLOCK":
        nli_grounding = state.get("nli_grounding", {})
        nli_unsupported_spans = state.get("nli_unsupported_spans", [])
        search_sources = state.get("search_sources", [])
        block_conf = compute_confidence(
            claim_table, state.get("findings", []),
            nli_grounding or None, nli_unsupported_spans or None, search_sources or None,
        )
        metrics.final_verdict = "FAIL"
        metrics.confidence_label = block_conf.confidence_label
        metrics.finish()
        record_run(metrics)
        updates.update({
            "is_pass": False,
            "final_verdict": "FAIL",
            "final_result": _fail_message(flags, state.get("search_performed", False)),
            "confidence": block_conf,
        })
        return updates

    # ---- ALLOW_AS_UNKNOWN_ONLY ----
    if arbiter_decision == "ALLOW_AS_UNKNOWN_ONLY":
        rewrite_prompt = (
            f"You previously produced this response:\n\n---\n{state['sanitized_output']}\n---\n\n"
            f"The arbiter has determined this question is inherently indeterminate.\n"
            f"Rewrite your response so that ALL claims are framed as Unknown(Actionable) or Unknown(Structural).\n"
            f"Do NOT make conclusions. Do NOT add new facts. Preserve the structure but move all substance to Unknowns.\n"
            f"Set Confidence to Low.\n"
            f"Output the corrected response in full."
        )
        rw_sm = metrics.start_stage("rewrite_unknown")
        rewrite_output = await call_llm_async(state["gpt1_cfg"], state["gpt1_system"], rewrite_prompt)
        metrics.end_stage(rw_sm)
        rewrite_output = sanitize_output(rewrite_output, flags, tier=state["tier"])

        if flags.get("current_events") and not state.get("search_performed"):
            rewrite_output += (
                "\n\n---\nNote: This response is based on training data that may be outdated. "
                "For verified current information, enable Tavily web search in Settings."
            )

        metrics.final_verdict = "PASS"
        metrics.confidence_label = "Low"
        metrics.rewrite_loops = 1
        metrics.convergence_outcome = "arbiter_unknown"
        metrics.finish()
        record_run(metrics)
        updates["early_return"] = _base_response(
            state, bypassed=False,
            gpt2_raw=state["gpt2_raw"], claim_table=state.get("claim_table", []),
            violations=state["violations"],
            gpt2_verdict=state["gpt2_verdict"], gpt2_reasoning=state["gpt2_reasoning"],
            arbiter_invoked=True, arbiter_decision="ALLOW_AS_UNKNOWN_ONLY",
            arbiter_rationale=arbiter_rationale, arbiter_edits=arbiter_edits,
            arbiter_policy_notes=arbiter_policy_notes, arbiter_raw=gpt3_raw,
            rewrite_occurred=True, rewrite_output=rewrite_output,
            rewrite_gpt2_raw="(arbiter-trusted)", rewrite_claim_table=[],
            rewrite_violations=[], rewrite_verdict="PASS",
            final_verdict="PASS", final_result=rewrite_output,
            sanitizer_applied=True,
            confidence=ConfidenceBreakdown(
                observed_pct=0, inference_pct=0, hypothesis_pct=0,
                unsupported_pct=0, user_provided_pct=0, total_claims=0, confidence_label="Low",
            ),
        )
        return updates

    # ALLOW_WITH_EDITS — fall through to stage_rewrite_loop
    return updates


# ---------------------------------------------------------------------------
# Stage: rewrite_loop (ALLOW_WITH_EDITS)
# ---------------------------------------------------------------------------

async def stage_rewrite_loop(state: PipelineState) -> dict:
    """Convergence-aware rewrite loop with AST edit support.

    When atomic claims have claim_id fields and arbiter edits have target_id,
    uses apply_edits_by_id for deterministic JSON edits alongside the
    text-based GPT-1 rewrite. Otherwise falls back to text-only rewrite.
    """
    gpt1_cfg = state["gpt1_cfg"]
    gpt1_system = state["gpt1_system"]
    flags = state["flags"]
    tier = state["tier"]
    metrics = state["metrics"]
    findings = state.get("findings", [])
    arbiter_edits = state.get("arbiter_edits", [])
    sanitized = state["sanitized_output"]
    atomic_claims = state.get("atomic_claims", [])
    max_loops = state.get("max_rewrite_loops", 3)

    # --- Build rewrite prompt (with optional AST edit info) ---
    rewrite_prompt = apply_edits(sanitized, arbiter_edits)
    if findings:
        finding_lines = "\n".join(
            f"- {f['type']}: {f.get('detail', 'no detail')} (severity: {f.get('severity', '?')})"
            for f in findings
        )
        rewrite_prompt += (
            f"\n\nPrevious verification found these specific issues:\n{finding_lines}\n"
            f"Please address each finding in your rewrite."
        )

    # Wire in apply_edits_by_id for ID-based deterministic edits
    has_id_edits = any(getattr(e, "target_id", "") for e in arbiter_edits)
    if has_id_edits and atomic_claims:
        modified_claims, edit_summary = apply_edits_by_id(atomic_claims, arbiter_edits)
        rewrite_prompt += f"\n\nDeterministic edits applied to claim metadata: {edit_summary}"
        # Update atomic claims for downstream confidence scoring
        atomic_claims = modified_claims

    rewrite_output = await call_llm_async(gpt1_cfg, gpt1_system, rewrite_prompt)
    rewrite_output = sanitize_output(rewrite_output, flags, tier=tier)

    # Re-verify
    re = await _verify_text(state, rewrite_output)
    findings_history = [findings, re["findings"]]

    # Convergence-aware loop
    while re["verdict"] == "FAIL" and should_continue_rewrite(findings_history, max_loops=max_loops):
        if _all_soft(re["findings"]):
            instruction = (
                f"You previously produced this response:\n\n---\n{rewrite_output}\n---\n\n"
                f"Remaining soft violations could not be resolved. "
                f"Rewrite your response so that ALL claims are framed as Unknown(Actionable) or Unknown(Structural).\n"
                f"Do NOT make conclusions. Do NOT add new facts. Output the corrected response in full."
            )
        else:
            hard_details = "; ".join(
                f'{f["type"]}: {f["detail"]}'
                for f in re["findings"] if f.get("severity") == "hard"
            )
            instruction = (
                f"You previously produced this response:\n\n---\n{rewrite_output}\n---\n\n"
                f"The following HARD violations were detected and must be fixed:\n{hard_details}\n\n"
                f"For each violation: either DELETE the problematic claim entirely, "
                f"or MOVE it to Unknown(Actionable) with a note that verification is needed.\n"
                f"Do NOT fabricate citations. Do NOT invent statistics.\n"
                f"Set Confidence to Low if you remove core claims.\n"
                f"Output the corrected response in full."
            )
        rewrite_output = await call_llm_async(gpt1_cfg, gpt1_system, instruction)
        rewrite_output = sanitize_output(rewrite_output, flags, tier=tier)
        re = await _verify_text(state, rewrite_output)
        findings_history.append(re["findings"])

    metrics.rewrite_loops = len(findings_history) - 1

    # Build common arbiter fields for response
    arbiter_fields = dict(
        arbiter_invoked=True, arbiter_decision="ALLOW_WITH_EDITS",
        arbiter_rationale=state.get("arbiter_rationale", []),
        arbiter_edits=state.get("arbiter_edits", []),
        arbiter_policy_notes=state.get("arbiter_policy_notes", []),
        arbiter_raw=state.get("gpt3_raw", ""),
    )

    nli_grounding = state.get("nli_grounding", {})
    nli_unsupported_spans = state.get("nli_unsupported_spans", [])
    search_sources = state.get("search_sources", [])

    if re["verdict"] == "PASS":
        conf = compute_confidence(
            re["claim_table"], re["findings"],
            nli_grounding or None, nli_unsupported_spans or None, search_sources or None,
        )
        metrics.final_verdict = "PASS"
        metrics.confidence_label = conf.confidence_label
        metrics.convergence_outcome = "pass"
        metrics.finish()
        record_run(metrics)
        return {"early_return": _base_response(
            state, bypassed=False,
            gpt2_raw=state["gpt2_raw"], claim_table=state["claim_table"],
            violations=state["violations"],
            gpt2_verdict=state["gpt2_verdict"], gpt2_reasoning=state["gpt2_reasoning"],
            rewrite_occurred=True, rewrite_output=rewrite_output,
            rewrite_gpt2_raw=re["gpt2_raw"], rewrite_claim_table=re["claim_table"],
            rewrite_violations=re["violations"], rewrite_verdict=re["verdict"],
            rewrite_reasoning=re["reasoning"],
            final_verdict="PASS", final_result=rewrite_output,
            sanitizer_applied=True, confidence=conf,
            **arbiter_fields,
        )}

    # ---- Fallback: rewrite failed, frame as Unknown ----
    fallback_prompt = (
        f"You previously produced this response:\n\n---\n{rewrite_output}\n---\n\n"
        f"The verification system could not clear all violations after multiple attempts.\n"
        f"Rewrite your response so that ALL factual claims are framed as "
        f"Unknown(Actionable) or Unknown(Structural).\n"
        f"Preserve the structure and topic coverage, but present everything as unverified.\n"
        f"List authoritative sources where the user can verify each claim.\n"
        f"Set Confidence to Low.\n"
        f"Output the corrected response in full."
    )
    fallback_output = await call_llm_async(gpt1_cfg, gpt1_system, fallback_prompt)
    fallback_output = sanitize_output(fallback_output, flags, tier=tier)

    metrics.final_verdict = "PASS"
    metrics.confidence_label = "Low"
    metrics.convergence_outcome = "fallback"
    metrics.finish()
    record_run(metrics)
    return {"early_return": _base_response(
        state, bypassed=False,
        gpt2_raw=state["gpt2_raw"], claim_table=state["claim_table"],
        violations=state["violations"],
        gpt2_verdict=state["gpt2_verdict"], gpt2_reasoning=state["gpt2_reasoning"],
        rewrite_occurred=True, rewrite_output=fallback_output,
        rewrite_gpt2_raw=re["gpt2_raw"], rewrite_claim_table=re["claim_table"],
        rewrite_violations=re["violations"], rewrite_verdict="PASS",
        rewrite_reasoning=re["reasoning"],
        final_verdict="PASS", final_result=fallback_output,
        sanitizer_applied=True,
        confidence=ConfidenceBreakdown(
            observed_pct=0, inference_pct=0, hypothesis_pct=0,
            unsupported_pct=0, user_provided_pct=0, total_claims=0, confidence_label="Low",
        ),
        **arbiter_fields,
    )}
