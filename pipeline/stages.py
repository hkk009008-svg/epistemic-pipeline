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

import config
from pipeline.arbiter import (
    apply_edits,
    apply_edits_by_id,
    extract_negative_constraints,
    format_negative_constraints_block,
    guard_arbiter_decision,
    parse_gpt3,
    parse_gpt3_structured,
)
from pipeline.best_of_n import generate_best_of_n_async
from pipeline.decomposer import decompose_claims
from pipeline.helpers import (
    PipelineError,
    call_llm_async,
    call_llm_structured,
    is_activation_phrase,
)
from pipeline.meta_verify import is_high_stakes, meta_verify_fail, meta_verify_pass
from pipeline.metrics import PipelineMetrics, record_run
from pipeline.models import (
    ClaimEntry,
    ConfidenceBreakdown,
    GPT2ResponseSchema,
    GPT3ResponseSchema,
    PipelineResponse,
)
from pipeline.nli import (
    compute_grounding_rate,
    detect_unsupported_spans,
    is_nli_available,
    verify_claims_concurrently,
)

# Re-import orchestrator utilities (these stay in orchestrator.py)
from pipeline.orchestrator import (
    _date_context,
    _emit_stage_complete,
    _emit_stage_start,
    _fail_message,
    _resolve_output_format,
    clean_for_display,
    compute_confidence,
)
from pipeline.pipeline_state import PipelineState
from pipeline.prompts import (
    DEFAULT_GPT1_SYSTEM,
    DEFAULT_GPT2_SYSTEM,
    DEFAULT_GPT3_SYSTEM,
    PROMPT_VERSION,
    build_augmentation,
)
from pipeline.sanitizer import route_prompt, sanitize_output
from pipeline.search import (
    fetch_claim_evidence,
    perform_web_search,
    refine_search_query,
    should_search,
)
from pipeline.source_match import (
    build_source_keyword_sets,
    build_source_number_sets,
    filter_findings_with_sources,
    recategorize_with_sources,
    run_preflight_scan,
    verify_citation_grounding,
)
from pipeline.verifier import (
    _all_soft,
    build_gpt2_user_content,
    parse_gpt2,
    parse_gpt2_structured,
    recompute_verdict,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

async def _verify_text(state: PipelineState, text_to_verify: str) -> dict:
    """Run GPT-2 verification on a text, with structured output fallback + source-match.

    Deduplicates the identical verify-and-correct pattern that appears 4 times
    in the original orchestrator (initial verify, soft-retry, first rewrite, loop rewrites).

    Atomic claims are attached only when re-verifying the same sanitized draft
    they were decomposed from. Rewritten text must not be scored against a
    stale claim list.

    Returns dict: {gpt2_raw, claim_table, violations, verdict, findings, reasoning}.
    """
    search_sources = state.get("search_sources", [])
    src_kw = state.get("src_kw_sets")
    src_nums = state.get("src_num_sets")
    prompt_text = state.get("prompt", "")

    # Fast deterministic pre-flight check (<0.5ms)
    has_hard_preflight, preflight_findings = run_preflight_scan(
        text_to_verify, search_sources, src_kw, src_nums, prompt=prompt_text
    )
    if has_hard_preflight:
        viol = [f["type"] for f in preflight_findings]
        verdict = "FAIL"
        reasoning = ["Pre-flight citation/bounds check failed with hard findings"]
        ct = [
            ClaimEntry(claim=f.get("detail", ""), category="Unsupported", justification="Pre-flight violation")
            for f in preflight_findings
        ]
        gpt2_raw = json.dumps({
            "reasoning_trace": reasoning,
            "claim_table": [c.model_dump() if hasattr(c, "model_dump") else dict(c) for c in ct],
            "findings": preflight_findings,
            "verdict": "FAIL",
        })
        return {
            "gpt2_raw": gpt2_raw,
            "claim_table": ct,
            "violations": viol,
            "verdict": verdict,
            "findings": preflight_findings,
            "reasoning": reasoning,
        }

    include_claims = text_to_verify == state.get("sanitized_output")
    gpt2_user = build_gpt2_user_content(
        state["prompt"],
        text_to_verify,
        atomic_claims=(state.get("atomic_claims") or None) if include_claims else None,
        nli_grounding=(state.get("nli_grounding") or None) if include_claims else None,
    )
    gpt2_cfg = state["gpt2_cfg"]
    gpt2_system = state["gpt2_system"]
    flags = state["flags"]
    tier = state["tier"]

    try:
        parsed = await call_llm_structured(gpt2_cfg, gpt2_system, gpt2_user, GPT2ResponseSchema)
        gpt2_raw = parsed.model_dump_json()
        ct, viol, verdict, findings, reasoning = parse_gpt2_structured(parsed, flags=flags, tier=tier)
    except PipelineError:
        raise
    except Exception:
        gpt2_raw = await call_llm_async(gpt2_cfg, gpt2_system, gpt2_user, expect_json=True)
        ct, viol, verdict, findings, reasoning = parse_gpt2(gpt2_raw, flags=flags, tier=tier)

    # Source-match correction
    if search_sources:
        nli_claims = state.get("atomic_claims") or None
        ct = recategorize_with_sources(ct, search_sources, src_kw, nli_claims=nli_claims)
        findings = filter_findings_with_sources(findings, search_sources, src_kw, nli_claims=nli_claims)
        citation_findings = verify_citation_grounding(text_to_verify, search_sources, src_kw, src_nums)
        if citation_findings:
            findings.extend(citation_findings)
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


_DEFAULT_EMPTY_ARBITER = dict(
    arbiter_invoked=False, arbiter_decision="", arbiter_rationale=[],
    arbiter_edits=[], arbiter_policy_notes=[], arbiter_raw="",
    rewrite_occurred=False, rewrite_output="", rewrite_gpt2_raw="",
    rewrite_claim_table=[], rewrite_violations=[], rewrite_verdict="",
)
_DEFAULT_SEARCH_KWARGS = dict(
    search_performed=False,
    search_attempted=False,
    search_note="",
    search_query="",
    search_sources=[],
)
_DEFAULT_DECOMP_KWARGS = dict(
    atomic_claims=[],
    decomposition_ran=False,
)


def _base_response(state: PipelineState, **overrides) -> PipelineResponse:
    """Build a PipelineResponse from state with sensible defaults.

    Merges empty_arbiter defaults, search_kwargs, decomp_kwargs,
    then applies any overrides. This replaces the 9 copy-pasted
    PipelineResponse(...) blocks in the original orchestrator.
    """
    base = dict(
        prompt_version=PROMPT_VERSION,
        tier=state.get("tier", "strict"),
        output_format=state.get("output_format", "structured"),
        gpt1_input=state.get("prompt", ""),
        gpt1_output=state.get("gpt1_output", ""),
        prompt_flags=state.get("flags", {}),
    )
    base.update(_DEFAULT_EMPTY_ARBITER)
    base.update(_DEFAULT_SEARCH_KWARGS)
    base.update(_DEFAULT_DECOMP_KWARGS)
    base.update(state.get("empty_arbiter", {}))
    base.update(state.get("search_kwargs", {}))
    base.update(state.get("decomp_kwargs", {}))
    base.update(overrides)

    # Strip internal sanitizer/epistemic markers before returning to users
    if base.get("final_result"):
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
            f'<untrusted_evidence id="{i}" url="{s.url}">\n'
            f'Title: "{s.title}"\n'
            f'Snippet: {s.snippet[:300]}\n'
            f'</untrusted_evidence>'
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
    src_num_sets = build_source_number_sets(search_sources) if search_sources else None
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
        "src_num_sets": src_num_sets,
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
        metrics.bypassed = False
        metrics.finish()
        record_run(metrics)
        return {"early_return": _base_response(
            state, bypassed=False, gpt2_raw="(current-events fast path — no web search available)",
            claim_table=[], violations=[], gpt2_verdict="PASS",
            final_verdict="PASS", final_result=fast_output, sanitizer_applied=True,
            confidence=ConfidenceBreakdown(
                observed_pct=0, inference_pct=0, hypothesis_pct=0,
                unsupported_pct=0, user_provided_pct=0, total_claims=0, confidence_label="Low",
            ),
        )}

    # Activation bypass
    if is_activation_phrase(gpt1_output):
        metrics.bypassed = True
        metrics.final_verdict = "PASS"
        metrics.finish()
        record_run(metrics)
        return {"early_return": _base_response(
            state, bypassed=True, gpt2_raw="(bypassed)",
            claim_table=[], violations=[], gpt2_verdict="PASS",
            final_verdict="PASS", final_result=gpt1_output, sanitizer_applied=False,
            confidence=compute_confidence([]),
        )}

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
        return {
            "atomic_claims": [],
            "decomposition_ran": False,
            "decomp_kwargs": dict(atomic_claims=[], decomposition_ran=False),
        }

    _emit_stage_start(emit, "decomposition")
    sm = metrics.start_stage("decomposition")
    claims = await asyncio.to_thread(
        decompose_claims, state["gpt2_cfg"], state["sanitized_output"], state["prompt"],
    )
    metrics.end_stage(sm)
    _emit_stage_complete(emit, "decomposition", data={"claim_count": len(claims)})
    metrics.decomposition_ran = len(claims) > 0
    metrics.atomic_claims_count = len(claims)
    return {
        "atomic_claims": claims,
        "decomposition_ran": len(claims) > 0,
        "decomp_kwargs": dict(atomic_claims=claims, decomposition_ran=len(claims) > 0),
    }


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

    enriched_claims = await verify_claims_concurrently(atomic_claims, evidence)

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

    # --- CONTROL 2: Fast Deterministic Pre-Flight Citation & Bounds Scanner ---
    search_sources = state.get("search_sources", [])
    src_kw = state.get("src_kw_sets")
    src_nums = state.get("src_num_sets")
    draft_text = state.get("sanitized_output", "") or state.get("gpt1_output", "")
    prompt_text = state.get("prompt", "")

    has_hard_preflight, preflight_findings = run_preflight_scan(
        draft_text, search_sources, src_kw, src_nums, prompt=prompt_text
    )

    if has_hard_preflight:
        # Deterministic short-circuit: skip LLM 2 Verifier call completely (0 LLM tokens, <10ms)
        violations = [f["type"] for f in preflight_findings]
        gpt2_verdict = "FAIL"
        gpt2_reasoning = ["Pre-flight citation/bounds check failed with hard findings"]

        if atomic_claims:
            claim_table = [
                ClaimEntry(
                    claim=c.get("text", "") if isinstance(c, dict) else getattr(c, "text", str(c)),
                    category="Unsupported",
                    justification="Pre-flight citation/bounds check failed with hard findings",
                )
                for c in atomic_claims
            ]
        else:
            claim_table = [
                ClaimEntry(
                    claim=f.get("detail", ""),
                    category="Unsupported",
                    justification="Pre-flight violation",
                )
                for f in preflight_findings
            ]

        gpt2_raw = json.dumps({
            "reasoning_trace": gpt2_reasoning,
            "claim_table": [c.model_dump() if hasattr(c, "model_dump") else dict(c) for c in claim_table],
            "findings": preflight_findings,
            "verdict": "FAIL",
        })

        metrics.gpt2_verdict = "FAIL"
        metrics.hard_findings = sum(1 for f in preflight_findings if f.get("severity") == "hard")
        metrics.soft_findings = sum(1 for f in preflight_findings if f.get("severity") == "soft")
        metrics.total_claims = len(claim_table)

        _emit_stage_start(emit, "gpt2", data={
            "provider": "preflight", "model": "deterministic_bounds_scanner",
        })
        _emit_stage_complete(emit, "gpt2", data={
            "verdict": "FAIL", "claim_count": len(claim_table), "violations": violations,
            "preflight_short_circuit": True,
        })

        if "decomp_kwargs" not in state:
            state["decomp_kwargs"] = dict(atomic_claims=atomic_claims, decomposition_ran=len(atomic_claims) > 0)

        return {
            "gpt2_raw": gpt2_raw,
            "claim_table": claim_table,
            "violations": violations,
            "gpt2_verdict": "FAIL",
            "findings": preflight_findings,
            "verification_findings": preflight_findings,
            "gpt2_reasoning": gpt2_reasoning,
            "max_rewrite_loops": getattr(config, "MAX_REWRITE_LOOPS", 1),
        }

    # Build GPT-2 user content (with NLI signals if available)
    gpt2_user = build_gpt2_user_content(
        state["prompt"],
        state["sanitized_output"],
        atomic_claims=atomic_claims or None,
        nli_grounding=nli_grounding or None,
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
    except PipelineError:
        raise
    except Exception:
        gpt2_raw = await call_llm_async(gpt2_cfg, state["gpt2_system"], gpt2_user, expect_json=True)
        claim_table, violations, gpt2_verdict, findings, gpt2_reasoning = parse_gpt2(
            gpt2_raw, flags=flags, tier=tier,
        )

    metrics.end_stage(gpt2_sm)
    _emit_stage_complete(emit, "gpt2", data={
        "verdict": gpt2_verdict, "claim_count": len(claim_table), "violations": violations,
    })

    # Source-match correction & citation verification
    if search_sources:
        claim_table = recategorize_with_sources(claim_table, search_sources, src_kw, nli_claims=atomic_claims or None)
        findings = filter_findings_with_sources(findings, search_sources, src_kw, nli_claims=atomic_claims or None)
        citation_findings = verify_citation_grounding(state.get("sanitized_output", ""), search_sources, src_kw, src_nums)
        if citation_findings:
            findings.extend(citation_findings)
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
                findings = filter_findings_with_sources(findings, search_sources, src_kw_sets, nli_claims=atomic_claims or None)
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
        "max_rewrite_loops": config.MAX_REWRITE_LOOPS,
        "decomp_kwargs": state["decomp_kwargs"],
        "search_sources": search_sources,
        "src_kw_sets": state.get("src_kw_sets"),
        "search_kwargs": state.get("search_kwargs", {}),
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
        updates["early_return"] = _base_response(
            state, bypassed=False,
            gpt2_raw=gpt2_raw, claim_table=claim_table, violations=violations,
            gpt2_verdict="PASS", gpt2_reasoning=gpt2_reasoning,
            final_verdict="PASS", final_result=state["sanitized_output"],
            sanitizer_applied=state.get("sanitizer_applied", False),
            confidence=conf,
            meta_verification=meta_result if meta_result["ran"] else None,
        )
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
            updates["early_return"] = _base_response(
                state, bypassed=False,
                gpt2_raw=gpt2_raw, claim_table=claim_table, violations=violations,
                gpt2_verdict="PASS", gpt2_reasoning=gpt2_reasoning,
                final_verdict="PASS", final_result=state["sanitized_output"],
                sanitizer_applied=state.get("sanitizer_applied", False),
                confidence=conf,
                meta_verification={"type": "false_fail_override", "reason": fail_meta["reason"]},
            )

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
        return {"early_return": _base_response(
            state, bypassed=False,
            gpt2_raw=state["gpt2_raw"], claim_table=state["claim_table"],
            violations=state["violations"],
            gpt2_verdict=state["gpt2_verdict"], gpt2_reasoning=state["gpt2_reasoning"],
            rewrite_occurred=True, rewrite_output=state["sanitized_output"],
            rewrite_gpt2_raw=result["gpt2_raw"], rewrite_claim_table=result["claim_table"],
            rewrite_violations=result["violations"], rewrite_verdict=result["verdict"],
            rewrite_reasoning=result["reasoning"],
            arbiter_invoked=False, arbiter_decision="", arbiter_rationale=[],
            arbiter_edits=[], arbiter_policy_notes=[], arbiter_raw="",
            final_verdict="PASS", final_result=state["sanitized_output"],
            sanitizer_applied=True, confidence=conf,
        )}

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
    except PipelineError:
        raise
    except Exception:
        gpt3_raw = await call_llm_async(gpt3_cfg, state["gpt3_system"], gpt3_user, expect_json=True)
        arbiter_decision, arbiter_rationale, arbiter_edits, arbiter_policy_notes = parse_gpt3(gpt3_raw)

    metrics.end_stage(gpt3_sm)
    _emit_stage_complete(emit, "gpt3", data={"decision": arbiter_decision})

    # Adaptive Poisoning Guard
    claim_table = state.get("claim_table", [])
    findings = state.get("findings", [])
    guarded_decision, guard_notes = guard_arbiter_decision(arbiter_decision, claim_table, findings)
    if guard_notes:
        arbiter_rationale = guard_notes + arbiter_rationale
    arbiter_decision = guarded_decision

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
        enable_repair = state.get("enable_repair", True) and (state.get("max_rewrite_loops", 3) > 0)
        if not enable_repair:
            nli_grounding = state.get("nli_grounding", {})
            nli_unsupported_spans = state.get("nli_unsupported_spans", [])
            search_sources = state.get("search_sources", [])
            block_conf = compute_confidence(
                claim_table, findings,
                nli_grounding or None, nli_unsupported_spans or None, search_sources or None,
            )
            metrics.final_verdict = "FAIL"
            metrics.confidence_label = block_conf.confidence_label
            metrics.finish()
            record_run(metrics)
            updates["early_return"] = _base_response(
                state, bypassed=False,
                gpt2_raw=state["gpt2_raw"], claim_table=claim_table,
                violations=state["violations"],
                gpt2_verdict=state["gpt2_verdict"], gpt2_reasoning=state["gpt2_reasoning"],
                arbiter_invoked=True, arbiter_decision="BLOCK",
                arbiter_rationale=arbiter_rationale, arbiter_edits=arbiter_edits,
                arbiter_policy_notes=arbiter_policy_notes, arbiter_raw=gpt3_raw,
                rewrite_occurred=False, rewrite_output="", rewrite_gpt2_raw="",
                rewrite_claim_table=[], rewrite_violations=[], rewrite_verdict="",
                final_verdict="FAIL", final_result=_fail_message(flags, state.get("search_performed", False)),
                sanitizer_applied=state.get("sanitizer_applied", False),
                confidence=block_conf,
            )
            return updates
        # Heavily poisoned drafts (BLOCK) are flagged for repair rather than fail-closed early returns
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
    """Multi-turn repair loop with Closed-Loop Negative-Constraint Feedback.

    Maintains a monotonically accumulating negative constraints ledger across repair
    turns to prevent re-hallucination. Enforces a hard cap of at most 2 repair turns,
    backed by deterministic fail-closed Unknown fallback.
    """
    gpt1_cfg = state["gpt1_cfg"]
    gpt1_system = state["gpt1_system"]
    flags = state["flags"]
    tier = state["tier"]
    metrics = state["metrics"]
    findings = state.get("findings", [])
    arbiter_edits = state.get("arbiter_edits", [])
    arbiter_decision = state.get("arbiter_decision", "ALLOW_WITH_EDITS")
    sanitized = state.get("sanitized_output", "") or state.get("gpt1_output", "")
    atomic_claims = state.get("atomic_claims", [])
    claim_table = state.get("claim_table", [])
    search_sources = state.get("search_sources", [])
    max_loops = min(state.get("max_rewrite_loops", 2), 2)

    # 1. Initialize and monotonically accumulate negative constraints ledger
    cumulative_constraints: list[str] = list(state.get("negative_constraints", []))
    initial_constraints = extract_negative_constraints(
        findings=findings,
        arbiter_edits=arbiter_edits,
        claim_table=claim_table,
        max_source_count=len(search_sources) if search_sources else None,
    )
    for c in initial_constraints:
        if c not in cumulative_constraints:
            cumulative_constraints.append(c)

    state["negative_constraints"] = cumulative_constraints
    nc_block = format_negative_constraints_block(cumulative_constraints)

    # 2. Build Turn 1 Prompt (Handling both BLOCK/REGENERATE and ALLOW_WITH_EDITS)
    if arbiter_decision in ("BLOCK", "REGENERATE"):
        prompt_text = state.get("prompt", "")
        rewrite_prompt = (
            f"Your previous response was rejected due to heavy poisoning or multiple critical violations.\n\n"
            f"Original Question:\n{prompt_text}\n\n"
            f"{nc_block}\n\n"
            f"Please generate a completely fresh response that strictly satisfies all epistemic rules "
            f"and contains ZERO claims or patterns listed in the Negative Constraints above.\n"
            f"Output your fresh response in full."
        )
    else:
        rewrite_prompt = apply_edits(sanitized, arbiter_edits)
        if findings:
            finding_lines = "\n".join(
                f"- {f.get('type', '?')}: {f.get('detail', 'no detail')} (severity: {f.get('severity', '?')})"
                for f in findings
            )
            rewrite_prompt += (
                f"\n\nPrevious verification found these specific issues:\n{finding_lines}\n"
                f"Please address each finding in your rewrite."
            )
        if nc_block:
            rewrite_prompt += f"\n\n{nc_block}\n\nYou MUST strictly adhere to all negative constraints above."

    # Wire in apply_edits_by_id for ID-based deterministic edits
    has_id_edits = any(getattr(e, "target_id", "") for e in arbiter_edits)
    if has_id_edits and atomic_claims:
        modified_claims, edit_summary = apply_edits_by_id(atomic_claims, arbiter_edits)
        rewrite_prompt += f"\n\nDeterministic edits applied to claim metadata: {edit_summary}"
        # Update atomic claims for downstream confidence scoring
        atomic_claims = modified_claims

    # 3. Generate Turn 1 Rewrite
    rw_sm = metrics.start_stage("rewrite_turn_1")
    rewrite_output = await call_llm_async(gpt1_cfg, gpt1_system, rewrite_prompt)
    metrics.end_stage(rw_sm)
    rewrite_output = sanitize_output(rewrite_output, flags, tier=tier)

    # 4. Re-verify Turn 1
    re = await _verify_text(state, rewrite_output)
    findings_history = [findings, re["findings"]]
    iteration_count = 1

    # 5. Multi-turn repair loop (Turn 2 if Turn 1 FAILS)
    while re["verdict"] == "FAIL" and iteration_count < max_loops:
        iteration_count += 1

        # Monotonically accumulate new constraints from Turn 1 re-verification
        turn_new_constraints = extract_negative_constraints(
            findings=re.get("findings", []),
            claim_table=re.get("claim_table", []),
            max_source_count=len(search_sources) if search_sources else None,
        )
        for c in turn_new_constraints:
            if c not in cumulative_constraints:
                cumulative_constraints.append(c)

        state["negative_constraints"] = cumulative_constraints
        nc_block_turn = format_negative_constraints_block(cumulative_constraints)

        if _all_soft(re["findings"]):
            instruction = (
                f"You previously produced this response:\n\n---\n{rewrite_output}\n---\n\n"
                f"{nc_block_turn}\n\n"
                f"Remaining soft violations could not be resolved. "
                f"Rewrite your response so that ALL claims are framed as Unknown(Actionable) or Unknown(Structural).\n"
                f"Do NOT make conclusions. Do NOT add new facts. Output the corrected response in full."
            )
        else:
            hard_details = "; ".join(
                f'{f.get("type", "?")}: {f.get("detail", "no detail")}'
                for f in re["findings"] if f.get("severity") == "hard"
            )
            instruction = (
                f"You previously produced this response:\n\n---\n{rewrite_output}\n---\n\n"
                f"{nc_block_turn}\n\n"
                f"The following critical violations persist and MUST be resolved immediately:\n{hard_details}\n\n"
                f"STRICT REQUIREMENT: For each violation, either DELETE the problematic claim entirely, "
                f"or MOVE it to Unknown(Actionable) with a note that verification is needed.\n"
                f"Do NOT fabricate citations. Do NOT invent statistics.\n"
                f"Set Confidence to Low if you remove core claims.\n"
                f"Output the corrected response in full."
            )

        rw_sm = metrics.start_stage(f"rewrite_turn_{iteration_count}")
        rewrite_output = await call_llm_async(gpt1_cfg, gpt1_system, instruction)
        metrics.end_stage(rw_sm)
        rewrite_output = sanitize_output(rewrite_output, flags, tier=tier)
        re = await _verify_text(state, rewrite_output)
        findings_history.append(re["findings"])

    metrics.rewrite_loops = iteration_count

    # Build common arbiter fields for response
    arbiter_fields = dict(
        arbiter_invoked=True,
        arbiter_decision=arbiter_decision,
        arbiter_rationale=state.get("arbiter_rationale", []),
        arbiter_edits=state.get("arbiter_edits", []),
        arbiter_policy_notes=state.get("arbiter_policy_notes", []),
        arbiter_raw=state.get("gpt3_raw", ""),
    )

    nli_grounding = state.get("nli_grounding", {})
    nli_unsupported_spans = state.get("nli_unsupported_spans", [])

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
            gpt2_raw=state.get("gpt2_raw", "{}"), claim_table=state.get("claim_table", []),
            violations=state.get("violations", []),
            gpt2_verdict=state.get("gpt2_verdict", "FAIL"), gpt2_reasoning=state.get("gpt2_reasoning", []),
            rewrite_occurred=True, rewrite_output=rewrite_output,
            rewrite_gpt2_raw=re.get("gpt2_raw", "{}"), rewrite_claim_table=re.get("claim_table", []),
            rewrite_violations=re.get("violations", []), rewrite_verdict=re.get("verdict", "PASS"),
            rewrite_reasoning=re.get("reasoning", []),
            final_verdict="PASS", final_result=rewrite_output,
            sanitizer_applied=True, confidence=conf,
            **arbiter_fields,
        )}

    # ---- Fallback: rewrite failed after <=2 turns, frame as Unknown ----
    fallback_prompt = (
        f"You previously produced this response:\n\n---\n{rewrite_output}\n---\n\n"
        f"The verification system could not clear all violations after {iteration_count} attempts.\n"
        f"Rewrite your response so that ALL factual claims are framed as "
        f"Unknown(Actionable) or Unknown(Structural).\n"
        f"Preserve the structure and topic coverage, but present everything as unverified.\n"
        f"List authoritative sources where the user can verify each claim.\n"
        f"Set Confidence to Low.\n"
        f"Output the corrected response in full."
    )
    fallback_sm = metrics.start_stage("rewrite_fallback")
    fallback_output = await call_llm_async(gpt1_cfg, gpt1_system, fallback_prompt)
    metrics.end_stage(fallback_sm)
    fallback_output = sanitize_output(fallback_output, flags, tier=tier)

    metrics.final_verdict = "PASS"
    metrics.confidence_label = "Low"
    metrics.convergence_outcome = "fallback"
    metrics.finish()
    record_run(metrics)
    return {"early_return": _base_response(
        state, bypassed=False,
        gpt2_raw=state.get("gpt2_raw", "{}"), claim_table=state.get("claim_table", []),
        violations=state.get("violations", []),
        gpt2_verdict=state.get("gpt2_verdict", "FAIL"), gpt2_reasoning=state.get("gpt2_reasoning", []),
        rewrite_occurred=True, rewrite_output=fallback_output,
        rewrite_gpt2_raw=re.get("gpt2_raw", "{}"), rewrite_claim_table=re.get("claim_table", []),
        rewrite_violations=re.get("violations", []), rewrite_verdict=re.get("verdict", "FAIL"),
        rewrite_reasoning=re.get("reasoning", []),
        final_verdict="PASS", final_result=fallback_output,
        sanitizer_applied=True,
        confidence=ConfidenceBreakdown(
            observed_pct=0, inference_pct=0, hypothesis_pct=0,
            unsupported_pct=0, user_provided_pct=0, total_claims=0, confidence_label="Low",
        ),
        **arbiter_fields,
    )}

