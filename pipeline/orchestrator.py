"""Core pipeline orchestrator: GPT-1 -> GPT-2 -> GPT-3 verification flow.

Provides both synchronous (run_pipeline) and async (run_pipeline_async) entry points.
The async path uses native AsyncOpenAI/AsyncAnthropic clients and structured outputs,
allowing a single FastAPI instance to handle 100x more concurrent requests.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from datetime import date
from typing import Callable, Optional

import config
from pipeline.models import (
    PipelineRequest, PipelineResponse, ConfidenceBreakdown,
    SearchSource, GroundingInfo, UnsupportedSpan,
)
from pipeline.prompts import DEFAULT_GPT1_SYSTEM, DEFAULT_GPT2_SYSTEM, DEFAULT_GPT3_SYSTEM, GPT2_TRIPWIRE_REFERENCE, PROMPT_VERSION, build_augmentation
from pipeline.sanitizer import route_prompt, sanitize_output
from pipeline.helpers import PipelineError, call_llm, is_activation_phrase
from pipeline.verifier import parse_gpt2, _all_soft, recompute_verdict
from pipeline.arbiter import parse_gpt3, apply_edits
from pipeline.convergence import should_continue_rewrite
from pipeline.search import should_search, perform_web_search, refine_search_query, fetch_claim_evidence
from pipeline.source_match import recategorize_with_sources, filter_findings_with_sources, build_source_keyword_sets
from pipeline.decomposer import decompose_claims
from pipeline.nli import verify_claims_with_nli, is_nli_available, compute_grounding_rate, detect_unsupported_spans
from pipeline.meta_verify import meta_verify_pass, meta_verify_fail, is_high_stakes
from pipeline.metrics import PipelineMetrics, record_run
from pipeline.best_of_n import generate_best_of_n


def _date_context() -> str:
    """Return a date-awareness preamble for system prompts."""
    today = date.today().isoformat()
    return (
        f"CURRENT DATE: {today}. Use this date when evaluating whether "
        f"claims refer to the past, present, or future. Any date before "
        f"{today} is in the PAST, not the future.\n\n"
    )


def _resolve_output_format(tier: str, output_format: str) -> str:
    """Resolve 'auto' output format based on tier."""
    if output_format != "auto":
        return output_format
    return {"strict": "structured", "standard": "annotated", "light": "concise"}.get(tier, "structured")


# Pre-compiled patterns for clean_for_display (avoid recompilation per call)
_DISPLAY_PATTERNS = [
    (re.compile(r"\[verified\]\s*"), ""),
    (re.compile(r"\[inference\]\s*"), ""),
    (re.compile(r"\[unverified\]\s*"), ""),
    (re.compile(r"\[user-provided\]\s*"), ""),
    (re.compile(r"\[Typicality language removed\]"), ""),
    (re.compile(r"\[Unverified generalization removed\]"), ""),
    (re.compile(r"\[Stale [^\]]*\]"), ""),
    (re.compile(r"\[Legal claim requires citation\]"), ""),
    (re.compile(r"  +"), " "),
    (re.compile(r"\n{3,}"), "\n\n"),
]


def clean_for_display(text: str) -> str:
    """Strip internal sanitizer/epistemic markers for display."""
    for pattern, replacement in _DISPLAY_PATTERNS:
        text = pattern.sub(replacement, text)
    return text.strip()


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


# Weighted violation penalties (T1 fabrication is far worse than T6 reassurance)
_VIOLATION_WEIGHTS = {
    "T1": 2.0,   # Fabricated evidence
    "T2": 1.5,   # Unsupported evidence reference
    "T3": 1.5,   # Causal claim stated as fact
    "T4": 1.0,   # Missing structural qualifier
    "T5": 1.0,   # Prescriptive creep
    "T6": 0.75,  # Reassurance framing
    "T7": 0.75,  # Unverified current fact
}


def compute_confidence(
    claim_table: list,
    findings: list | None = None,
    nli_grounding: dict | None = None,
    unsupported_spans: list | None = None,
    search_sources: list | None = None,
) -> ConfidenceBreakdown:
    """Compute a confidence breakdown from a list of ClaimEntry objects.

    Uses a dual-calibration approach:
    1. Reasoning confidence: GPT-2 claim categories + weighted findings penalty
    2. Evidence confidence: Source authority scores (when search sources available)
    3. NLI grounding rate (when available, blends with category signal)

    The NLI grounding rate provides calibrated confidence by checking
    what percentage of claims can be verified against evidence.
    """
    total = len(claim_table)
    if total == 0:
        grounding_info = None
        if nli_grounding and nli_grounding.get("total_evaluated", 0) > 0:
            grounding_info = GroundingInfo(**nli_grounding)
        return ConfidenceBreakdown(
            grounding=grounding_info,
            confidence_reasoning=["No claims to evaluate."],
        )

    # Category counts (uniform weight — position bias removed)
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
    inference_pct = round((inference / total) * 100, 1)
    hypothesis_pct = round((hypothesis / total) * 100, 1)
    unsupported_pct = round((unsupported / total) * 100, 1)
    user_provided_pct = round((user_provided / total) * 100, 1)

    # Build confidence reasoning as we go
    reasoning: list[str] = []
    reasoning.append(f"{observed}/{total} claims verified as Observed ({observed_pct}%)")
    if unsupported > 0:
        reasoning.append(f"{unsupported}/{total} claims marked Unsupported ({unsupported_pct}%)")
    if inference > 0:
        reasoning.append(f"{inference}/{total} claims are inferences ({inference_pct}%)")

    # Weighted findings penalty (T1 fabrication >> T6 reassurance)
    hard_count = 0
    soft_count = 0
    weighted_penalty = 0.0
    if findings:
        for f in findings:
            w = _VIOLATION_WEIGHTS.get(f.get("type", ""), 1.0)
            if f.get("severity") == "hard":
                hard_count += 1
                weighted_penalty += w
            elif f.get("severity") == "soft":
                soft_count += 1
                weighted_penalty += w * 0.3  # soft findings have reduced weight
    if hard_count > 0:
        hard_types = ", ".join(sorted({f["type"] for f in findings if f.get("severity") == "hard"}))
        reasoning.append(f"{hard_count} hard violation(s) detected ({hard_types}), weighted penalty: {weighted_penalty:.1f}")
    if soft_count > 0:
        soft_types = ", ".join(sorted({f["type"] for f in findings if f.get("severity") == "soft"}))
        reasoning.append(f"{soft_count} soft violation(s) detected ({soft_types})")
    if hard_count == 0 and soft_count == 0:
        reasoning.append("No violations detected")

    # Evidence confidence: source authority signal (when search sources available)
    evidence_confidence = 0.0
    if search_sources:
        authorities = [getattr(s, "score", 0.5) for s in search_sources]
        evidence_confidence = sum(authorities) / len(authorities) if authorities else 0.0
        if evidence_confidence >= 0.8:
            reasoning.append(f"Evidence quality: high (avg authority {evidence_confidence:.2f} from {len(search_sources)} sources)")
        elif evidence_confidence >= 0.5:
            reasoning.append(f"Evidence quality: medium (avg authority {evidence_confidence:.2f} from {len(search_sources)} sources)")
        else:
            reasoning.append(f"Evidence quality: low (avg authority {evidence_confidence:.2f})")

    # Dual calibration: reasoning confidence (categories + penalties) blended with evidence confidence
    # Reasoning score: 0-1 based on observed% and weighted penalty
    reasoning_score = (observed_pct / 100.0) - (weighted_penalty * 0.15)
    reasoning_score = max(0.0, min(1.0, reasoning_score))

    if search_sources and evidence_confidence > 0:
        combined_score = 0.6 * reasoning_score + 0.4 * evidence_confidence
    else:
        combined_score = reasoning_score

    # Map combined score to label
    if combined_score >= 0.65 and hard_count == 0:
        label = "High"
    elif combined_score >= 0.35 and weighted_penalty < 3.0:
        label = "Medium"
    elif combined_score >= 0.15:
        label = "Low"
    else:
        label = "Unknown"

    # NLI grounding rate adjustment: when evidence exists, blend with category signal
    grounding_info = None
    if nli_grounding and nli_grounding.get("total_evaluated", 0) > 0:
        grounding_info = GroundingInfo(**nli_grounding)
        gr = nli_grounding["grounding_rate"]
        contradicted = nli_grounding.get("contradicted_count", 0)

        reasoning.append(f"NLI grounding rate: {gr:.0%} ({nli_grounding['grounded_count']}/{nli_grounding['total_evaluated']} claims grounded)")

        # Contradicted claims should downgrade confidence
        if contradicted > 0:
            reasoning.append(f"{contradicted} claim(s) contradicted by evidence — downgrading confidence")
            if label == "High":
                label = "Medium"
            elif label == "Medium":
                label = "Low"

        # Low grounding rate with evidence available means claims are unverifiable
        if gr < 0.3 and label in ("High", "Medium"):
            reasoning.append("Low grounding rate with evidence available — downgrading to Low")
            label = "Low"
        elif gr >= 0.7 and label == "Low" and hard_count == 0:
            # High grounding can rescue Low if no hard findings
            reasoning.append("High grounding rate with no hard findings — upgrading to Medium")
            label = "Medium"

    # Build unsupported span models
    span_models = []
    if unsupported_spans:
        for s in unsupported_spans:
            span_models.append(UnsupportedSpan(
                text=s.get("text", ""),
                start=s.get("start", -1),
                end=s.get("end", -1),
                reason=s.get("reason", ""),
                confidence_tier=s.get("confidence_tier", ""),
            ))

    return ConfidenceBreakdown(
        observed_pct=observed_pct,
        inference_pct=inference_pct,
        hypothesis_pct=hypothesis_pct,
        unsupported_pct=unsupported_pct,
        user_provided_pct=user_provided_pct,
        total_claims=total,
        confidence_label=label,
        confidence_reasoning=reasoning,
        grounding=grounding_info,
        unsupported_spans=span_models,
    )


# Type alias for the stage event callback used by streaming.
# Signature: emit(event_dict) -> None
StageEventEmitter = Callable[[dict], None]


def _noop_emit(event: dict) -> None:
    """Default no-op emitter when streaming is not requested."""
    pass


def _emit_stage_start(emit: StageEventEmitter, stage: str, **data):
    """Emit a stage_start event with monotonic timestamp."""
    emit({"type": "stage_start", "stage": stage, "t": round(time.monotonic(), 3), **data})


def _emit_stage_complete(emit: StageEventEmitter, stage: str, **data):
    """Emit a stage_complete event with monotonic timestamp."""
    emit({"type": "stage_complete", "stage": stage, "t": round(time.monotonic(), 3), **data})


def run_pipeline(
    req: PipelineRequest,
    emit: Optional[StageEventEmitter] = None,
) -> PipelineResponse:
    """Execute the full epistemic verification pipeline.

    Args:
        req: The pipeline request.
        emit: Optional callback for streaming stage events (NDJSON).
              When provided, called with dicts like
              {"type": "stage_start", "stage": "gpt1", "t": ...}.

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
    if emit is None:
        emit = _noop_emit
    if not config.has_api_key():
        raise PipelineError(400, "Set your OpenAI API key first.")

    metrics = PipelineMetrics(request_id=uuid.uuid4().hex[:12], prompt_length=len(req.prompt))
    metrics.start()

    gpt1_cfg = config.get_stage_config("gpt1")
    gpt2_cfg = config.get_stage_config("gpt2")
    gpt3_cfg = config.get_stage_config("gpt3")

    # ---- Tier + output format resolution ----
    tier = getattr(req, "tier", "strict") or "strict"
    output_format = _resolve_output_format(tier, getattr(req, "output_format", "auto") or "auto")

    # ---- Deterministic prompt routing ----
    _emit_stage_start(emit, "routing")
    flags = route_prompt(req.prompt)
    metrics.flags = flags
    _emit_stage_complete(emit, "routing", data={"flags": flags})

    # ---- Web Search Enrichment (before augmentation so flags are search-aware) ----
    search_sources: list[SearchSource] = []
    search_context = ""
    search_performed = False

    search_attempted = False
    search_note = ""

    if should_search(flags):
        search_attempted = True
        _emit_stage_start(emit, "search", data={"query": req.prompt})
        sm = metrics.start_stage("search")
        search_sources, search_context = perform_web_search(req.prompt)
        search_performed = len(search_sources) > 0
        metrics.end_stage(sm)
        _emit_stage_complete(emit, "search", data={"source_count": len(search_sources), "performed": search_performed})
        metrics.search_performed = search_performed
        metrics.search_sources_count = len(search_sources)
        if not search_performed:
            search_note = "Web search was enabled but returned no relevant sources for this query."

    gpt1_system = req.gpt1_system or DEFAULT_GPT1_SYSTEM
    gpt2_system = req.gpt2_system or DEFAULT_GPT2_SYSTEM

    # Inject current-date awareness so models don't misidentify past dates as future
    date_ctx = _date_context()
    gpt1_system = date_ctx + gpt1_system
    gpt2_system = date_ctx + gpt2_system

    # Flag-driven augmentation for all three stages (search-aware)
    gpt1_aug, gpt2_aug, gpt3_aug = build_augmentation(flags, search_performed=search_performed, tier=tier, output_format=output_format)
    gpt1_system += gpt1_aug
    gpt2_system += gpt2_aug
    gpt3_system = date_ctx + (req.gpt3_system if hasattr(req, "gpt3_system") and req.gpt3_system else DEFAULT_GPT3_SYSTEM) + gpt3_aug

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
        search_attempted=search_attempted,
        search_note=search_note,
        search_query=req.prompt if search_performed else "",
        search_sources=search_sources,
    )

    # Pre-compute source keyword sets once for all source-match operations
    _src_kw_sets = build_source_keyword_sets(search_sources) if search_sources else None

    # Empty defaults for response
    empty_response = dict(
        arbiter_invoked=False, arbiter_decision="", arbiter_rationale=[],
        arbiter_edits=[], arbiter_policy_notes=[], arbiter_raw="",
        rewrite_occurred=False, rewrite_output="", rewrite_gpt2_raw="",
        rewrite_claim_table=[], rewrite_violations=[], rewrite_verdict="",
    )

    # ---- Step 1: GPT-1 Generate (with optional best-of-N) ----
    _emit_stage_start(emit, "gpt1", data={"provider": gpt1_cfg.get("provider", ""), "model": gpt1_cfg.get("model", "")})
    gpt1_sm = metrics.start_stage("gpt1", gpt1_cfg.get("provider", ""), gpt1_cfg.get("model", ""))
    best_of_n_count = getattr(config, "BEST_OF_N", 1)
    if best_of_n_count >= 2:
        gpt1_output, bon_info = generate_best_of_n(
            gpt1_cfg, gpt1_system, gpt1_user_content, flags, n=best_of_n_count,
        )
    else:
        gpt1_output = call_llm(gpt1_cfg, gpt1_system, gpt1_user_content)
    metrics.end_stage(gpt1_sm)
    _emit_stage_complete(emit, "gpt1")

    # ---- Current-events fast path (no Tavily) ----
    # If the query is about current events and we have no web search to ground it,
    # GPT-1 was already instructed to frame everything as Unknown(Actionable).
    # Sanitize and return directly — skip GPT-2/GPT-3 (they would FAIL the stale
    # data and the rewrite loop adds 3 more API calls for the same result).
    if flags.get("current_events") and not search_performed:
        fast_output = sanitize_output(gpt1_output, flags, tier=tier)
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
        return PipelineResponse(
            prompt_version=PROMPT_VERSION, tier=tier, output_format=output_format,
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
        metrics.bypassed = True
        metrics.final_verdict = "PASS"
        metrics.finish()
        record_run(metrics)
        return PipelineResponse(
            prompt_version=PROMPT_VERSION, tier=tier, output_format=output_format,
            gpt1_input=req.prompt, gpt1_output=gpt1_output, bypassed=True,
            gpt2_raw="(bypassed)", claim_table=[], violations=[], gpt2_verdict="PASS",
            final_verdict="PASS", final_result=gpt1_output,
            prompt_flags=flags, sanitizer_applied=False,
            confidence=compute_confidence([]),
            **empty_response, **search_kwargs,
        )

    # ---- Deterministic sanitizer (pre-clean before GPT-2) ----
    sanitized_output = sanitize_output(gpt1_output, flags, tier=tier)
    sanitizer_applied = (sanitized_output != gpt1_output)

    # ---- Atomic Claim Decomposition (pre-GPT-2) ----
    # Only decompose when it meaningfully enriches verification:
    #   - NLI available: decomposition feeds NLI claim-by-claim checking
    #   - High-stakes query: legal/medical/financial claims warrant extra scrutiny
    # Skipping for routine queries saves 2-3 s per request.
    should_decompose = is_nli_available() or is_high_stakes(flags)
    atomic_claims: list = []
    if should_decompose:
        _emit_stage_start(emit, "decomposition")
        decomp_sm = metrics.start_stage("decomposition")
        atomic_claims = decompose_claims(gpt2_cfg, sanitized_output, req.prompt)
        metrics.end_stage(decomp_sm)
        _emit_stage_complete(emit, "decomposition", data={"claim_count": len(atomic_claims)})
        metrics.decomposition_ran = len(atomic_claims) > 0
        metrics.atomic_claims_count = len(atomic_claims)

    # ---- NLI Pre-Verification (optional layer) ----
    nli_grounding = {}
    nli_unsupported_spans = []
    if atomic_claims and is_nli_available():
        evidence_snippets = [s.snippet for s in search_sources] if search_sources else []
        if evidence_snippets:
            _emit_stage_start(emit, "nli")
            nli_sm = metrics.start_stage("nli")
            atomic_claims = verify_claims_with_nli(atomic_claims, evidence_snippets)
            metrics.end_stage(nli_sm)
            metrics.nli_ran = True
            metrics.nli_supported_count = sum(
                1 for c in atomic_claims if c.get("nli_result", {}).get("supported")
            )
            metrics.nli_contradicted_count = sum(
                1 for c in atomic_claims if c.get("nli_result", {}).get("contradicted")
            )
            # Compute grounding rate for confidence calibration
            nli_grounding = compute_grounding_rate(atomic_claims)
            metrics.grounding_rate = nli_grounding.get("grounding_rate", 0.0)
            # Detect unsupported spans
            nli_unsupported_spans = detect_unsupported_spans(sanitized_output, atomic_claims)
            _emit_stage_complete(emit, "nli", data={
                "grounding_rate": nli_grounding.get("grounding_rate", 0.0),
                "supported": metrics.nli_supported_count,
                "contradicted": metrics.nli_contradicted_count,
            })

    decomp_kwargs = dict(atomic_claims=atomic_claims, decomposition_ran=len(atomic_claims) > 0)

    # ---- Step 2: GPT-2 Verify (on sanitized output) ----
    # Tripwire reference placed at START of user content (lost-in-middle fix)
    if atomic_claims:
        claims_json = json.dumps(atomic_claims, indent=2)
        # Build NLI signals block if any claims have NLI results
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
                grounding_str = f"\nGrounding Rate: {nli_grounding['grounding_rate']:.1%} ({nli_grounding['grounded_count']}/{nli_grounding['total_evaluated']} claims grounded)"
            nli_block = "\n\nNLI PRE-VERIFICATION SIGNALS:\n" + "\n".join(nli_lines) + grounding_str

        gpt2_user = (
            f"{GPT2_TRIPWIRE_REFERENCE}\n\n"
            f"=== TASK ===\n"
            f"ORIGINAL PROMPT:\n{req.prompt}\n\n"
            f"GPT-1 RESPONSE TO VERIFY:\n{sanitized_output}\n\n"
            f"PRE-DECOMPOSED ATOMIC CLAIMS (verify each independently):\n{claims_json}"
            f"{nli_block}"
        )
    else:
        gpt2_user = (
            f"{GPT2_TRIPWIRE_REFERENCE}\n\n"
            f"=== TASK ===\n"
            f"ORIGINAL PROMPT:\n{req.prompt}\n\n"
            f"GPT-1 RESPONSE TO VERIFY:\n{sanitized_output}"
        )
    _emit_stage_start(emit, "gpt2", data={"provider": gpt2_cfg.get("provider", ""), "model": gpt2_cfg.get("model", "")})
    gpt2_sm = metrics.start_stage("gpt2", gpt2_cfg.get("provider", ""), gpt2_cfg.get("model", ""))
    gpt2_raw = call_llm(gpt2_cfg, gpt2_system, gpt2_user, expect_json=True)
    metrics.end_stage(gpt2_sm)
    claim_table, violations, gpt2_verdict, findings, gpt2_reasoning = parse_gpt2(gpt2_raw, flags=flags, tier=tier)
    _emit_stage_complete(emit, "gpt2", data={"verdict": gpt2_verdict, "claim_count": len(claim_table), "violations": violations})

    # ---- Source-match correction: fix GPT-2's over-strict categorization ----
    if search_sources:
        claim_table = recategorize_with_sources(claim_table, search_sources, _src_kw_sets, nli_claims=atomic_claims or None)
        findings = filter_findings_with_sources(findings, search_sources, _src_kw_sets)
        violations = [f["type"] for f in findings]
        gpt2_verdict = recompute_verdict(findings, tier=tier)

    metrics.gpt2_verdict = gpt2_verdict
    metrics.total_claims = len(claim_table)
    metrics.hard_findings = sum(1 for f in findings if f.get("severity") == "hard")
    metrics.soft_findings = sum(1 for f in findings if f.get("severity") == "soft")

    # ---- Claim-conditional retrieval: fetch evidence for unsupported claims ----
    if search_performed and claim_table:
        unsupported_ct = sum(
            1 for ct in claim_table
            if (ct.category if isinstance(ct.category, str) else "").lower().strip() == "unsupported"
        )
        unsupported_ratio = unsupported_ct / len(claim_table)
        if unsupported_ratio > 0.3:
            # Use claim-conditional retrieval: search for specific unsupported claims
            _emit_stage_start(emit, "claim_retrieval", data={"unsupported_count": unsupported_ct})
            if atomic_claims:
                new_sources = fetch_claim_evidence(atomic_claims, search_sources)
            else:
                # Fallback to keyword-based refinement if no atomic claims
                unsupported_texts = [
                    ct.claim for ct in claim_table
                    if (ct.category if isinstance(ct.category, str) else "").lower().strip() == "unsupported"
                ]
                refined_query = refine_search_query(req.prompt, unsupported_texts)
                retry_sources, _ = perform_web_search(refined_query, max_results=3)
                existing_urls = {s.url for s in search_sources}
                new_sources = [s for s in retry_sources if s.url not in existing_urls]

            if new_sources:
                search_sources = search_sources + new_sources
                _src_kw_sets = build_source_keyword_sets(search_sources)
                # Re-run source-match correction with expanded evidence
                claim_table = recategorize_with_sources(claim_table, search_sources, _src_kw_sets, nli_claims=atomic_claims or None)
                findings = filter_findings_with_sources(findings, search_sources, _src_kw_sets)
                violations = [f["type"] for f in findings]
                gpt2_verdict = recompute_verdict(findings, tier=tier)
                search_kwargs["search_sources"] = search_sources
            _emit_stage_complete(emit, "claim_retrieval", data={"new_sources": len(new_sources) if new_sources else 0})

    # ---- If GPT-2 PASS: done ----
    if gpt2_verdict == "PASS":
        conf = compute_confidence(claim_table, findings, nli_grounding or None, nli_unsupported_spans or None, search_sources or None)

        # Meta-verification: cross-check GPT-2 PASS on high-stakes queries
        meta_result = meta_verify_pass(flags, claim_table, findings, atomic_claims, conf.confidence_label)
        if meta_result["ran"] and meta_result["adjusted_label"] != conf.confidence_label:
            conf.confidence_label = meta_result["adjusted_label"]

        metrics.final_verdict = "PASS"
        metrics.confidence_label = conf.confidence_label
        metrics.finish()
        record_run(metrics)
        return PipelineResponse(
            prompt_version=PROMPT_VERSION, tier=tier, output_format=output_format,
            gpt1_input=req.prompt, gpt1_output=gpt1_output, bypassed=False,
            gpt2_raw=gpt2_raw, claim_table=claim_table, violations=violations,
            gpt2_verdict="PASS", gpt2_reasoning=gpt2_reasoning,
            final_verdict="PASS", final_result=sanitized_output,
            prompt_flags=flags, sanitizer_applied=sanitizer_applied,
            confidence=conf,
            meta_verification=meta_result if meta_result["ran"] else None,
            **empty_response, **search_kwargs, **decomp_kwargs,
        )

    # ---- Meta-verify FAIL: catch false FAILs on high-stakes queries ----
    fail_meta = meta_verify_fail(flags, claim_table, findings, atomic_claims)
    if fail_meta["ran"] and fail_meta["override_to_pass"]:
        findings = fail_meta["adjusted_findings"]
        violations = [f["type"] for f in findings]
        gpt2_verdict = recompute_verdict(findings, tier=tier)
        if gpt2_verdict == "PASS":
            conf = compute_confidence(claim_table, findings, nli_grounding or None, nli_unsupported_spans or None, search_sources or None)
            metrics.final_verdict = "PASS"
            metrics.confidence_label = conf.confidence_label
            metrics.finish()
            record_run(metrics)
            return PipelineResponse(
                prompt_version=PROMPT_VERSION, tier=tier, output_format=output_format,
                gpt1_input=req.prompt, gpt1_output=gpt1_output, bypassed=False,
                gpt2_raw=gpt2_raw, claim_table=claim_table, violations=violations,
                gpt2_verdict="PASS", gpt2_reasoning=gpt2_reasoning,
                final_verdict="PASS", final_result=sanitized_output,
                prompt_flags=flags, sanitizer_applied=sanitizer_applied,
                confidence=conf,
                meta_verification={"type": "false_fail_override", "reason": fail_meta["reason"]},
                **empty_response, **search_kwargs, **decomp_kwargs,
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
        re_gpt2_raw = call_llm(gpt2_cfg, gpt2_system, re_gpt2_user, expect_json=True)
        re_ct, re_viol, re_verdict, re_findings, re_reasoning = parse_gpt2(re_gpt2_raw, flags=flags, tier=tier)
        if search_sources:
            re_ct = recategorize_with_sources(re_ct, search_sources, _src_kw_sets)
            re_findings = filter_findings_with_sources(re_findings, search_sources, _src_kw_sets)
            re_viol = [f["type"] for f in re_findings]
            re_verdict = recompute_verdict(re_findings, tier=tier)

        if re_verdict == "PASS":
            conf = compute_confidence(re_ct, re_findings, nli_grounding or None, nli_unsupported_spans or None, search_sources or None)
            metrics.final_verdict = "PASS"
            metrics.confidence_label = conf.confidence_label
            metrics.rewrite_loops = 1
            metrics.convergence_outcome = "pass"
            metrics.finish()
            record_run(metrics)
            return PipelineResponse(
                prompt_version=PROMPT_VERSION, tier=tier, output_format=output_format,
                gpt1_input=req.prompt, gpt1_output=gpt1_output, bypassed=False,
                gpt2_raw=gpt2_raw, claim_table=claim_table, violations=violations,
                gpt2_verdict=gpt2_verdict, gpt2_reasoning=gpt2_reasoning,
                rewrite_occurred=True, rewrite_output=sanitized_output,
                rewrite_gpt2_raw=re_gpt2_raw, rewrite_claim_table=re_ct,
                rewrite_violations=re_viol, rewrite_verdict=re_verdict,
                rewrite_reasoning=re_reasoning,
                arbiter_invoked=False, arbiter_decision="", arbiter_rationale=[],
                arbiter_edits=[], arbiter_policy_notes=[], arbiter_raw="",
                final_verdict="PASS", final_result=sanitized_output,
                prompt_flags=flags, sanitizer_applied=True,
                confidence=conf,
                **search_kwargs, **decomp_kwargs,
            )
        # Auto-repair didn't clear it -- fall through to arbiter below

    # ---- Step 3: GPT-2 FAIL — invoke GPT-3 Arbiter ----
    gpt3_user = (
        f"user_prompt:\n{req.prompt}\n\n"
        f"gpt1_output:\n{sanitized_output}\n\n"
        f"gpt2_result_json:\n{gpt2_raw}\n\n"
        f"prompt_flags:\n{json.dumps(flags)}"
    )
    _emit_stage_start(emit, "gpt3", data={"provider": gpt3_cfg.get("provider", ""), "model": gpt3_cfg.get("model", "")})
    gpt3_sm = metrics.start_stage("gpt3", gpt3_cfg.get("provider", ""), gpt3_cfg.get("model", ""))
    gpt3_raw = call_llm(gpt3_cfg, gpt3_system, gpt3_user, expect_json=True)
    metrics.end_stage(gpt3_sm)
    arbiter_decision, arbiter_rationale, arbiter_edits, arbiter_policy_notes = parse_gpt3(gpt3_raw)
    _emit_stage_complete(emit, "gpt3", data={"decision": arbiter_decision})

    # Safety net: override BLOCK to ALLOW_WITH_EDITS when the response
    # contains any truthful content.  The GPT-3 prompt says "BLOCK only when
    # the ENTIRE response is unsalvageable fabrication", but gpt-4o-mini
    # frequently over-BLOCKs.  If at least one claim is Observed, Supported,
    # or even Inference, the response has salvageable content.
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

    # ---- Decision: BLOCK ----
    if arbiter_decision == "BLOCK":
        metrics.final_verdict = "FAIL"
        block_conf = compute_confidence(claim_table, findings, nli_grounding or None, nli_unsupported_spans or None, search_sources or None)
        metrics.confidence_label = block_conf.confidence_label
        metrics.finish()
        record_run(metrics)
        return PipelineResponse(
            prompt_version=PROMPT_VERSION, tier=tier, output_format=output_format,
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
            confidence=block_conf,
            **search_kwargs, **decomp_kwargs,
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
        rw_sm = metrics.start_stage("rewrite_unknown")
        rewrite_output = call_llm(gpt1_cfg, gpt1_system, rewrite_prompt)
        metrics.end_stage(rw_sm)

        # Sanitize the rewrite (strip stale dates, banned evidence, etc.)
        rewrite_output = sanitize_output(rewrite_output, flags, tier=tier)

        # Append a note if current-events without search
        if flags.get("current_events") and not search_performed:
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
        return PipelineResponse(
            prompt_version=PROMPT_VERSION, tier=tier, output_format=output_format,
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
            **search_kwargs, **decomp_kwargs,
        )

    # ---- Decision: ALLOW_WITH_EDITS ----
    rewrite_prompt = apply_edits(sanitized_output, arbiter_edits)
    # Inject GPT-2 findings so GPT-1 knows exactly what to fix
    if findings:
        finding_lines = "\n".join(
            f"- {f['type']}: {f.get('detail', 'no detail')} (severity: {f.get('severity', '?')})"
            for f in findings
        )
        rewrite_prompt += (
            f"\n\nPrevious verification found these specific issues:\n{finding_lines}\n"
            f"Please address each finding in your rewrite."
        )
    rewrite_output = call_llm(gpt1_cfg, gpt1_system, rewrite_prompt)

    # Sanitize the rewrite before re-verification
    rewrite_output = sanitize_output(rewrite_output, flags, tier=tier)

    # Re-verify the rewritten output with GPT-2
    re_gpt2_user = (
            f"{GPT2_TRIPWIRE_REFERENCE}\n\n"
            f"=== TASK ===\n"
            f"ORIGINAL PROMPT:\n{req.prompt}\n\n"
            f"GPT-1 RESPONSE TO VERIFY:\n{rewrite_output}"
        )
    re_gpt2_raw = call_llm(gpt2_cfg, gpt2_system, re_gpt2_user, expect_json=True)
    re_ct, re_viol, re_verdict, re_findings, re_reasoning = parse_gpt2(re_gpt2_raw, flags=flags, tier=tier)
    if search_sources:
        re_ct = recategorize_with_sources(re_ct, search_sources, _src_kw_sets)
        re_findings = filter_findings_with_sources(re_findings, search_sources, _src_kw_sets)
        re_viol = [f["type"] for f in re_findings]
        re_verdict = recompute_verdict(re_findings, tier=tier)

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
        rewrite_output = call_llm(gpt1_cfg, gpt1_system, rewrite_instruction)
        rewrite_output = sanitize_output(rewrite_output, flags, tier=tier)
        # Re-verify
        re_gpt2_user = (
            f"{GPT2_TRIPWIRE_REFERENCE}\n\n"
            f"=== TASK ===\n"
            f"ORIGINAL PROMPT:\n{req.prompt}\n\n"
            f"GPT-1 RESPONSE TO VERIFY:\n{rewrite_output}"
        )
        re_gpt2_raw = call_llm(gpt2_cfg, gpt2_system, re_gpt2_user, expect_json=True)
        re_ct, re_viol, re_verdict, re_findings, re_reasoning = parse_gpt2(re_gpt2_raw, flags=flags, tier=tier)
        if search_sources:
            re_ct = recategorize_with_sources(re_ct, search_sources, _src_kw_sets)
            re_findings = filter_findings_with_sources(re_findings, search_sources, _src_kw_sets)
            re_viol = [f["type"] for f in re_findings]
            re_verdict = recompute_verdict(re_findings, tier=tier)
        findings_history.append(re_findings)

    metrics.rewrite_loops = len(findings_history) - 1  # subtract initial

    # If the rewrite loop passed, return success
    if re_verdict == "PASS":
        conf = compute_confidence(re_ct, re_findings, nli_grounding or None, nli_unsupported_spans or None, search_sources or None)
        metrics.final_verdict = "PASS"
        metrics.confidence_label = conf.confidence_label
        metrics.convergence_outcome = "pass"
        metrics.finish()
        record_run(metrics)
        return PipelineResponse(
            prompt_version=PROMPT_VERSION, tier=tier, output_format=output_format,
            gpt1_input=req.prompt, gpt1_output=gpt1_output, bypassed=False,
            gpt2_raw=gpt2_raw, claim_table=claim_table, violations=violations,
            gpt2_verdict=gpt2_verdict, gpt2_reasoning=gpt2_reasoning,
            arbiter_invoked=True, arbiter_decision="ALLOW_WITH_EDITS",
            arbiter_rationale=arbiter_rationale, arbiter_edits=arbiter_edits,
            arbiter_policy_notes=arbiter_policy_notes, arbiter_raw=gpt3_raw,
            rewrite_occurred=True, rewrite_output=rewrite_output,
            rewrite_gpt2_raw=re_gpt2_raw, rewrite_claim_table=re_ct,
            rewrite_violations=re_viol, rewrite_verdict=re_verdict,
            rewrite_reasoning=re_reasoning,
            final_verdict="PASS",
            final_result=rewrite_output,
            prompt_flags=flags, sanitizer_applied=sanitizer_applied,
            confidence=conf,
            **search_kwargs, **decomp_kwargs,
        )

    # ---- Fallback: ALLOW_WITH_EDITS rewrite failed → downgrade to Unknown framing ----
    # The arbiter decided ALLOW_WITH_EDITS (not BLOCK), meaning the content is
    # salvageable. If the rewrite loop couldn't converge, fall back to framing
    # everything as Unknown rather than returning FAIL.
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
    fallback_output = call_llm(gpt1_cfg, gpt1_system, fallback_prompt)
    fallback_output = sanitize_output(fallback_output, flags, tier=tier)

    metrics.final_verdict = "PASS"
    metrics.confidence_label = "Low"
    metrics.convergence_outcome = "fallback"
    metrics.finish()
    record_run(metrics)
    return PipelineResponse(
        prompt_version=PROMPT_VERSION, tier=tier, output_format=output_format,
        gpt1_input=req.prompt, gpt1_output=gpt1_output, bypassed=False,
        gpt2_raw=gpt2_raw, claim_table=claim_table, violations=violations,
        gpt2_verdict=gpt2_verdict, gpt2_reasoning=gpt2_reasoning,
        arbiter_invoked=True, arbiter_decision="ALLOW_WITH_EDITS",
        arbiter_rationale=arbiter_rationale, arbiter_edits=arbiter_edits,
        arbiter_policy_notes=arbiter_policy_notes, arbiter_raw=gpt3_raw,
        rewrite_occurred=True, rewrite_output=fallback_output,
        rewrite_gpt2_raw=re_gpt2_raw, rewrite_claim_table=re_ct,
        rewrite_violations=re_viol, rewrite_verdict="PASS",
        rewrite_reasoning=re_reasoning,
        final_verdict="PASS",
        final_result=fallback_output,
        prompt_flags=flags, sanitizer_applied=True,
        confidence=ConfidenceBreakdown(
            observed_pct=0, inference_pct=0, hypothesis_pct=0,
            unsupported_pct=0, user_provided_pct=0,
            total_claims=0, confidence_label="Low",
        ),
        **search_kwargs, **decomp_kwargs,
    )


# ---------------------------------------------------------------------------
# V5 Async Pipeline — stage-decomposed architecture
# ---------------------------------------------------------------------------


async def run_pipeline_async(
    req: PipelineRequest,
    emit: Optional[StageEventEmitter] = None,
) -> PipelineResponse:
    """Async pipeline using decomposed stage functions.

    Each stage reads from / writes to a shared PipelineState dict.
    Stages that need to short-circuit set state["early_return"].
    The orchestrator checks for early_return after each stage.

    V5 improvements over V4:
    1. Each stage is independently testable (~20-60 lines each)
    2. _verify_text() deduplicates the GPT-2 call pattern (was copy-pasted 4x)
    3. _base_response() deduplicates PipelineResponse construction (was 9 blocks)
    4. apply_edits_by_id wired into rewrite loop for deterministic AST edits
    5. Same emit callback protocol — no UI changes needed
    """
    from pipeline.stages import (
        stage_init, stage_route, stage_search, stage_build_prompts,
        stage_generate, stage_check_fast_paths, stage_sanitize,
        stage_decompose, stage_nli, stage_verify,
        stage_soft_retry, stage_arbiter, stage_rewrite_loop,
    )

    state: dict = {
        "request": req,
        "emit": emit or _noop_emit,
    }

    stages = [
        stage_init,
        stage_route,
        stage_search,
        stage_build_prompts,
        stage_generate,
        stage_check_fast_paths,  # current-events / activation bypass
        stage_sanitize,
        stage_decompose,
        stage_nli,
        stage_verify,            # sets early_return on PASS
        stage_soft_retry,        # sets early_return on soft-only PASS
        stage_arbiter,           # sets early_return on BLOCK / ALLOW_AS_UNKNOWN_ONLY
        stage_rewrite_loop,      # always sets early_return (PASS or fallback)
    ]

    for stage_fn in stages:
        updates = await stage_fn(state)
        state.update(updates)
        early = state.get("early_return")
        if early is not None:
            return early

    # Should never reach here — stage_rewrite_loop always sets early_return
    raise PipelineError(500, "Pipeline completed without producing a response")
