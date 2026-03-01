"""NLI (Natural Language Inference) verification layer using DeBERTa-v3.

Classifies (premise, hypothesis) pairs as entailment/contradiction/neutral.
Used to pre-verify atomic claims against evidence before GPT-2.

This module is OPTIONAL — if torch/transformers are not installed,
all functions return None and the pipeline proceeds without NLI.
"""
from __future__ import annotations

import logging
import os
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


def verify_claims_with_nli(
    claims: List[dict],
    evidence_snippets: List[str],
) -> List[dict]:
    """Run NLI verification on atomic claims against evidence snippets.

    For each claim, checks entailment against each evidence snippet.
    Returns claims enriched with nli_result field.

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

        # Aggregate: best entailment score, worst contradiction score
        best_entailment = 0.0
        worst_contradiction = 0.0
        best_source_idx = -1

        for i, r in enumerate(results):
            if r is None:
                continue
            ent_score = r["scores"].get("entailment", 0.0)
            con_score = r["scores"].get("contradiction", 0.0)
            if ent_score > best_entailment:
                best_entailment = ent_score
                best_source_idx = i
            if con_score > worst_contradiction:
                worst_contradiction = con_score

        claim["nli_result"] = {
            "best_entailment": round(best_entailment, 4),
            "worst_contradiction": round(worst_contradiction, 4),
            "best_source_idx": best_source_idx,
            "supported": best_entailment > 0.7,
            "contradicted": worst_contradiction > 0.7,
        }
        enriched.append(claim)

    return enriched


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
