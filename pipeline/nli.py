"""NLI (Natural Language Inference) verification layer using DeBERTa-v3.

Classifies (premise, hypothesis) pairs as entailment/contradiction/neutral.
Used to pre-verify atomic claims against evidence before GPT-2.

Features:
- Continuous confidence scores (not just binary supported/contradicted)
- Span-level unsupported segment detection
- Configurable thresholds
- Grounding rate computation for confidence calibration

This module is OPTIONAL — if torch/transformers are not installed,
all functions return None and the pipeline proceeds without NLI.
"""
from __future__ import annotations

import concurrent.futures
import logging
import os
import re
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Lazy-loaded model (heavy imports)
_nli_pipeline = None
_nli_available: Optional[bool] = None

# Model choice: cross-encoder/nli-deberta-v3-xsmall is 22M params, fast inference
NLI_MODEL = "cross-encoder/nli-deberta-v3-xsmall"
LABEL_MAP = ["contradiction", "entailment", "neutral"]

# Optional remote NLI service URL
NLI_SERVICE_URL = os.getenv("NLI_SERVICE_URL", "")

# Configurable thresholds (calibrate against labeled data)
ENTAILMENT_THRESHOLD = 0.7
CONTRADICTION_THRESHOLD = 0.7
# Weaker threshold for "likely supported" (continuous signal)
WEAK_SUPPORT_THRESHOLD = 0.4


def is_nli_available() -> bool:
    """Check if NLI dependencies are installed or remote service is configured."""
    global _nli_available
    if _nli_available is None:
        if NLI_SERVICE_URL:
            _nli_available = True
        else:
            try:
                import torch  # noqa: F401
                from transformers import AutoTokenizer  # noqa: F401
                _nli_available = True
            except ImportError:
                _nli_available = False
                logger.info("NLI layer disabled: torch/transformers not installed")
    return _nli_available


def _get_nli_pipeline():
    """Lazy-load the NLI model on first use."""
    global _nli_pipeline
    if _nli_pipeline is None and is_nli_available() and not NLI_SERVICE_URL:
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
            model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
            model.eval()
            _nli_pipeline = (tokenizer, model)
            logger.info(f"NLI model loaded: {NLI_MODEL}")
        except Exception as e:
            logger.warning(f"Failed to load NLI model: {e}")
            _nli_pipeline = None
    return _nli_pipeline


def classify_nli(premise: str, hypothesis: str) -> Optional[dict]:
    """Classify a single (premise, hypothesis) pair.

    Returns: {"label": "entailment"|"contradiction"|"neutral",
              "scores": {"entailment": float, "contradiction": float, "neutral": float}}
    Returns None if NLI is not available.
    """
    if NLI_SERVICE_URL:
        return _classify_nli_remote(premise, hypothesis)

    pipeline = _get_nli_pipeline()
    if pipeline is None:
        return None

    tokenizer, model = pipeline
    import torch

    try:
        features = tokenizer(
            [premise], [hypothesis],
            padding=True, truncation=True, max_length=512,
            return_tensors="pt"
        )
        with torch.no_grad():
            logits = model(**features).logits
            probs = torch.softmax(logits, dim=1)[0].tolist()

        scores = {LABEL_MAP[i]: round(probs[i], 4) for i in range(3)}
        label = LABEL_MAP[probs.index(max(probs))]
        return {"label": label, "scores": scores}
    except Exception as e:
        logger.warning(f"NLI classification failed: {e}")
        return None


def batch_classify_nli(pairs: List[Tuple[str, str]]) -> List[Optional[dict]]:
    """Classify multiple (premise, hypothesis) pairs efficiently.

    Returns list of results (same length as pairs). None for any failures.
    """
    if NLI_SERVICE_URL:
        return _batch_classify_nli_remote(pairs)

    pipeline = _get_nli_pipeline()
    if pipeline is None:
        return [None] * len(pairs)

    tokenizer, model = pipeline
    import torch

    try:
        premises = [p[0] for p in pairs]
        hypotheses = [p[1] for p in pairs]

        features = tokenizer(
            premises, hypotheses,
            padding=True, truncation=True, max_length=512,
            return_tensors="pt"
        )

        with torch.no_grad():
            logits = model(**features).logits
            all_probs = torch.softmax(logits, dim=1).tolist()

        results = []
        for probs in all_probs:
            scores = {LABEL_MAP[i]: round(probs[i], 4) for i in range(3)}
            label = LABEL_MAP[probs.index(max(probs))]
            results.append({"label": label, "scores": scores})
        return results
    except Exception as e:
        logger.warning(f"Batch NLI classification failed: {e}")
        return [None] * len(pairs)


def _compute_confidence_tier(entailment: float, contradiction: float) -> str:
    """Map continuous NLI scores to a confidence tier.

    Contradiction outranks support: mixed evidence is a conflict, not verification.

    Returns: "strong_support", "weak_support", "neutral",
             "weak_contradiction", "strong_contradiction"
    """
    if contradiction >= CONTRADICTION_THRESHOLD:
        return "strong_contradiction"
    if entailment >= ENTAILMENT_THRESHOLD:
        return "strong_support"
    if contradiction >= WEAK_SUPPORT_THRESHOLD and contradiction >= entailment:
        return "weak_contradiction"
    if entailment >= WEAK_SUPPORT_THRESHOLD:
        return "weak_support"
    if contradiction >= WEAK_SUPPORT_THRESHOLD:
        return "weak_contradiction"
    return "neutral"


def _nli_result_from_classifications(results: List[Optional[dict]]) -> Optional[dict]:
    """Aggregate per-snippet NLI scores into one claim-level result.

    Returns None if every classification failed so callers omit nli_result
    instead of treating a dead NLI layer as 'all ungrounded'.
    """
    best_entailment = 0.0
    worst_contradiction = 0.0
    best_source_idx = -1
    all_scores = []

    for i, r in enumerate(results):
        if r is None:
            continue
        scores = r.get("scores") or {}
        ent_score = scores.get("entailment", 0.0)
        con_score = scores.get("contradiction", 0.0)
        all_scores.append({
            "source_idx": i,
            "entailment": ent_score,
            "contradiction": con_score,
        })
        if ent_score > best_entailment:
            best_entailment = ent_score
            best_source_idx = i
        if con_score > worst_contradiction:
            worst_contradiction = con_score

    if not all_scores:
        return None

    contradicted = worst_contradiction >= CONTRADICTION_THRESHOLD
    supported = (best_entailment >= ENTAILMENT_THRESHOLD) and not contradicted
    return {
        "best_entailment": round(best_entailment, 4),
        "worst_contradiction": round(worst_contradiction, 4),
        "best_source_idx": best_source_idx,
        "supported": supported,
        "contradicted": contradicted,
        "confidence_tier": _compute_confidence_tier(best_entailment, worst_contradiction),
        "per_source_scores": all_scores,
    }


def verify_claims_with_nli(
    claims: List[dict],
    evidence_snippets: List[str],
) -> List[dict]:
    """Run NLI verification on atomic claims against evidence snippets.

    For each claim, checks entailment against each evidence snippet.
    Returns claims enriched with nli_result field containing continuous scores.

    claims: List of {"text": "...", ...} from decomposer
    evidence_snippets: List of text strings (from Tavily or other sources)
    """
    if not claims or not evidence_snippets or not is_nli_available():
        return claims  # Return unmodified

    enriched = []
    for claim in claims:
        claim_text = claim.get("text", "")
        if not claim_text:
            enriched.append(claim)
            continue

        # Check claim against each evidence snippet
        pairs = [(snippet, claim_text) for snippet in evidence_snippets]
        results = batch_classify_nli(pairs)
        nli_result = _nli_result_from_classifications(results)
        updated = dict(claim)
        if nli_result is not None:
            updated["nli_result"] = nli_result
        enriched.append(updated)

    return enriched


def compute_grounding_rate(claims: List[dict]) -> dict:
    """Compute the Claim Grounding Rate (CGR) from NLI-enriched claims.

    Returns:
        {
            "grounding_rate": float (0.0-1.0),
            "grounded_count": int,
            "ungrounded_count": int,
            "contradicted_count": int,
            "neutral_count": int,
            "total_evaluated": int,
        }
    """
    if not claims:
        return {
            "grounding_rate": 0.0,
            "grounded_count": 0,
            "ungrounded_count": 0,
            "contradicted_count": 0,
            "neutral_count": 0,
            "total_evaluated": 0,
        }

    grounded = 0
    contradicted = 0
    neutral = 0
    evaluated = 0

    for claim in claims:
        nli = claim.get("nli_result")
        if not nli:
            continue
        evaluated += 1
        tier = nli.get("confidence_tier", "neutral")
        if tier in ("strong_support", "weak_support"):
            grounded += 1
        elif tier in ("strong_contradiction", "weak_contradiction"):
            contradicted += 1
        else:
            neutral += 1

    rate = grounded / evaluated if evaluated > 0 else 0.0

    return {
        "grounding_rate": round(rate, 4),
        "grounded_count": grounded,
        "ungrounded_count": neutral + contradicted,
        "contradicted_count": contradicted,
        "neutral_count": neutral,
        "total_evaluated": evaluated,
    }


def detect_unsupported_spans(
    text: str,
    claims: List[dict],
) -> List[dict]:
    """Identify spans of text that are unsupported by evidence.

    Uses NLI results from verified claims to find unsupported segments.
    Returns a list of span descriptors:
    [{"text": "...", "start": int, "end": int, "reason": "..."}]
    """
    spans = []
    for claim in claims:
        nli = claim.get("nli_result")
        if not nli:
            continue

        claim_text = claim.get("text", "")
        if not claim_text:
            continue

        tier = nli.get("confidence_tier", "neutral")
        if tier in ("strong_contradiction", "weak_contradiction"):
            # Find this claim text in the original output
            match = re.search(re.escape(claim_text[:50]), text)
            start = match.start() if match else -1
            end = match.end() if match else -1
            spans.append({
                "text": claim_text,
                "start": start,
                "end": end,
                "reason": "contradicted_by_evidence",
                "confidence_tier": tier,
                "contradiction_score": nli.get("worst_contradiction", 0.0),
            })
        elif tier == "neutral" and nli.get("best_entailment", 0.0) < 0.2:
            # Very low support — likely unsupported
            match = re.search(re.escape(claim_text[:50]), text)
            start = match.start() if match else -1
            end = match.end() if match else -1
            spans.append({
                "text": claim_text,
                "start": start,
                "end": end,
                "reason": "no_evidence_found",
                "confidence_tier": tier,
                "best_entailment": nli.get("best_entailment", 0.0),
            })

    return spans


def _classify_nli_remote(premise: str, hypothesis: str) -> Optional[dict]:
    """Call remote NLI service if local model is not available."""
    if not NLI_SERVICE_URL:
        return None
    try:
        import httpx
        resp = httpx.post(
            f"{NLI_SERVICE_URL}/classify",
            json={"premise": premise, "hypothesis": hypothesis},
            timeout=10,
        )
        return resp.json()
    except Exception:
        return None


def _batch_classify_nli_remote(pairs: List[Tuple[str, str]]) -> List[Optional[dict]]:
    """Call remote NLI batch service."""
    if not NLI_SERVICE_URL:
        return [None] * len(pairs)
    try:
        import httpx
        req_pairs = [{"premise": p[0], "hypothesis": p[1]} for p in pairs]
        resp = httpx.post(
            f"{NLI_SERVICE_URL}/batch",
            json={"pairs": req_pairs},
            timeout=30,
        )
        return resp.json()
    except Exception:
        return [None] * len(pairs)


# ---------------------------------------------------------------------------
# V4: Concurrent NLI verification via asyncio + ProcessPoolExecutor
# ---------------------------------------------------------------------------

# ProcessPoolExecutor for CPU-bound torch inference.
# Avoids the GIL trap: Python threads share the GIL, so multiple
# concurrent asyncio.to_thread(batch_classify_nli) calls would serialize
# under the GIL when PyTorch runs CPU-bound C++ kernels. A process pool
# gives each inference its own GIL and CPU cache, eliminating thread thrashing.
_nli_process_pool: Optional[concurrent.futures.ProcessPoolExecutor] = None


# Configurable worker count: memory-constrained deployments may need 1.
_NLI_WORKERS = int(os.getenv("NLI_WORKERS", "2"))


def _init_nli_worker():
    """Pre-load the NLI model in each worker process at spawn time.

    Without this initializer, the 22M-param DeBERTa model is loaded lazily
    on the first inference request, causing a large latency spike.

    Wrapped in try/except so a failed model load (OOM, download failure)
    doesn't silently kill the worker process. The worker remains alive
    and _get_nli_pipeline() will retry lazily on the first real request.
    """
    try:
        _get_nli_pipeline()
    except Exception as e:
        # Log but don't crash — worker stays alive for lazy retry
        logger.warning(f"NLI worker pre-warm failed (will retry lazily): {e}")


def _get_nli_process_pool() -> concurrent.futures.ProcessPoolExecutor:
    """Lazily create a ProcessPoolExecutor for NLI inference."""
    global _nli_process_pool
    if _nli_process_pool is None:
        # initializer pre-loads the model so the first request isn't slow.
        # Worker count configurable via NLI_WORKERS env var (default 2).
        _nli_process_pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=_NLI_WORKERS,
            initializer=_init_nli_worker,
        )
    return _nli_process_pool


def _nli_worker(pairs: List[Tuple[str, str]]) -> List[Optional[dict]]:
    """Process-pool worker: run batch NLI classification in an isolated process.

    This function is invoked in a child process, avoiding GIL contention
    with the main event loop and other inference tasks.
    """
    return batch_classify_nli(pairs)


async def verify_claims_concurrently(
    claims: List[dict],
    evidence_snippets: List[str],
) -> List[dict]:
    """Run NLI verification on atomic claims concurrently using asyncio.

    Uses ProcessPoolExecutor for local torch inference to avoid the GIL trap.
    For remote NLI services (NLI_SERVICE_URL set), uses asyncio.to_thread()
    since the work is I/O-bound (HTTP), not CPU-bound.

    Falls back to the sequential verify_claims_with_nli() on any error.
    """
    import asyncio

    if not claims or not evidence_snippets or not is_nli_available():
        return claims

    # Choose executor based on whether NLI is local (CPU-bound) or remote (I/O-bound)
    use_process_pool = not NLI_SERVICE_URL

    async def _verify_single_claim(claim: dict) -> dict:
        claim_text = claim.get("text", "")
        if not claim_text:
            return claim

        pairs = [(snippet, claim_text) for snippet in evidence_snippets]

        try:
            if use_process_pool:
                # CPU-bound: use ProcessPoolExecutor to bypass GIL
                loop = asyncio.get_running_loop()
                pool = _get_nli_process_pool()
                results = await loop.run_in_executor(pool, _nli_worker, pairs)
            else:
                # I/O-bound (remote service): thread pool is fine
                results = await asyncio.to_thread(batch_classify_nli, pairs)
        except Exception:
            # Fallback to sync in-process if process pool fails
            results = batch_classify_nli(pairs)

        nli_result = _nli_result_from_classifications(results)
        if nli_result is not None:
            claim["nli_result"] = nli_result
        return claim

    # Run all claims concurrently
    enriched = await asyncio.gather(*[_verify_single_claim(c) for c in claims])
    return list(enriched)
