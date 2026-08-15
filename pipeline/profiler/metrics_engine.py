"""Quantitative metrics computation engine for Epistemic Pipeline profiling.

Provides formal calculations for:
- Confusion Matrix (TP, FP, TN, FN, multi-class mapping)
- Binary classification metrics: Precision, Recall, F1, FAR, FRR, Specificity, Accuracy
- Computational performance: Latency per token, percentile distributions (p50, p90, p95, p99)
- Resource footprints: Cross-platform Peak RSS (Darwin bytes vs Linux KB) and Tracemalloc heap delta
- Granular aggregation by domain, attack vector, and difficulty tier.
"""
from __future__ import annotations

import platform
import resource
import statistics
import sys
import tracemalloc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Cross-Platform Memory Helpers
# ---------------------------------------------------------------------------

def get_peak_rss_mb() -> float:
    """Return process peak resident set size in megabytes.

    Handles platform differences:
    - macOS (Darwin): ru_maxrss is reported in bytes.
    - Linux / BSD / Solaris: ru_maxrss is reported in kilobytes.
    """
    raw_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return round(raw_rss / (1024.0 * 1024.0), 2)
    return round(raw_rss / 1024.0, 2)


def get_peak_rss_bytes() -> int:
    """Return process peak resident set size in bytes across platforms."""
    raw_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(raw_rss)
    return int(raw_rss * 1024)


# ---------------------------------------------------------------------------
# Data Models & Metric Containers
# ---------------------------------------------------------------------------

@dataclass
class ConfusionMatrix:
    """Binary and Multi-Class Confusion Matrix representation."""
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    labels: list[str] = field(default_factory=lambda: ["ATTACK", "CLEAN"])
    matrix: list[list[int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.matrix:
            self.matrix = [
                [self.tp, self.fn],
                [self.fp, self.tn],
            ]

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    def get(self, true_label: str, pred_label: str) -> int:
        """Get cell count for a given true and predicted label."""
        if true_label in self.labels and pred_label in self.labels:
            t_idx = self.labels.index(true_label)
            p_idx = self.labels.index(pred_label)
            return self.matrix[t_idx][p_idx]
        return 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "total": self.total,
            "labels": self.labels,
            "matrix": self.matrix,
        }


class ClassificationMetrics(BaseModel):
    """Core binary classification and defense security metrics."""
    tp: int = Field(0, description="True Positives (Attacks correctly caught/blocked)")
    fp: int = Field(0, description="False Positives (Clean inputs incorrectly flagged/blocked)")
    tn: int = Field(0, description="True Negatives (Clean inputs correctly passed)")
    fn: int = Field(0, description="False Negatives (Attacks that bypassed defenses)")
    total_cases: int = Field(0, description="Total evaluated test cases")
    precision: float = Field(0.0, description="Precision = TP / (TP + FP)")
    recall: float = Field(0.0, description="Recall (TPR / Detection Rate) = TP / (TP + FN)")
    far: float = Field(0.0, description="False Accept Rate (Adversarial Bypass Rate) = FN / (TP + FN) or FP / (FP + TN)")
    frr: float = Field(0.0, description="False Reject Rate (False Rejection Rate) = FP / (TN + FP)")
    specificity: float = Field(0.0, description="Specificity (TNR) = TN / (TN + FP)")
    f1_score: float = Field(0.0, description="Harmonic Mean of Precision and Recall")
    accuracy: float = Field(0.0, description="Accuracy = (TP + TN) / Total")


class LatencyProfile(BaseModel):
    """Detailed stage-by-stage and percentile latency profile."""
    total_duration_ms: float = 0.0
    mean_duration_ms: float = 0.0
    p50_ms: float = 0.0
    p90_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    latency_per_token_ms: float = 0.0
    preflight_mean_ms: float = 0.0
    preflight_speedup_factor: float = 1.0
    stage_latencies_ms: dict[str, float] = Field(default_factory=dict)


class MemoryProfile(BaseModel):
    """Operating system and Python runtime memory footprint."""
    initial_rss_mb: float = 0.0
    peak_rss_mb: float = 0.0
    rss_delta_mb: float = 0.0
    tracemalloc_peak_mb: float = 0.0
    tracemalloc_current_mb: float = 0.0
    platform_name: str = Field(default_factory=lambda: f"{platform.system()} ({sys.platform})")


class DomainMetrics(BaseModel):
    """Aggregated evaluation metrics for a specific knowledge domain."""
    domain: str
    total_cases: int
    clean_cases: int = 0
    adversarial_cases: int = 0
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    far: float = 0.0
    frr: float = 0.0
    f1_score: float = 0.0
    accuracy: float = 0.0
    pass_count: int = 0
    block_count: int = 0
    edit_count: int = 0
    preflight_intercept_count: int = 0
    mean_duration_ms: float = 0.0


class AttackVectorMetrics(BaseModel):
    """Aggregated evaluation metrics for a specific attack vector."""
    attack_vector: str
    difficulty_tier: Optional[str] = None
    total_cases: int
    tp: int = 0
    fn: int = 0
    detection_rate: float = 0.0  # Recall
    bypass_rate: float = 0.0     # FAR
    block_count: int = 0
    edit_count: int = 0
    preflight_intercept_count: int = 0
    preflight_intercept_rate: float = 0.0
    mean_duration_ms: float = 0.0


class EvaluationMetrics(BaseModel):
    """Comprehensive evaluation metrics bundle from a profiler execution."""
    total_cases_evaluated: int = 0
    clean_cases_count: int = 0
    adversarial_cases_count: int = 0
    classification: ClassificationMetrics = Field(default_factory=ClassificationMetrics)
    confusion_matrix: dict[str, Any] = Field(default_factory=dict)
    latency: LatencyProfile = Field(default_factory=LatencyProfile)
    memory: MemoryProfile = Field(default_factory=MemoryProfile)
    domain_breakdowns: dict[str, DomainMetrics] = Field(default_factory=dict)
    attack_vector_breakdowns: dict[str, AttackVectorMetrics] = Field(default_factory=dict)
    tier_breakdowns: dict[str, dict[str, Any]] = Field(default_factory=dict)
    pipeline_stability_score: float = 100.0


# ---------------------------------------------------------------------------
# Metrics Computation Engine
# ---------------------------------------------------------------------------

class MetricsEngine:
    """Quantitative evaluation and metrics computation engine for adversarial stress tests."""

    @staticmethod
    def compute_classification(
        y_true_is_attack: Sequence[bool],
        y_pred_is_defense_triggered: Sequence[bool],
    ) -> tuple[ClassificationMetrics, ConfusionMatrix]:
        """Compute binary classification metrics for adversarial defense.

        Definitions:
        - True (is_attack=True): Input is an adversarial / corrupted / defective prompt.
        - Negative (is_attack=False): Input is clean / valid.
        - Predicted Defense Triggered (True): Pipeline caught attack, performed edit, blocked, or aborted.
        - Predicted Defense Not Triggered (False): Pipeline passed draft unmodified as clean.

        Formulas:
        - TP: Attack caught (y_true=1, y_pred=1)
        - FP: Clean input falsely triggered (y_true=0, y_pred=1) -> False Alarm
        - TN: Clean input passed clean (y_true=0, y_pred=0)
        - FN: Attack slipped through undetected (y_true=1, y_pred=0) -> Adversarial Bypass
        - Precision = TP / (TP + FP)
        - Recall (Detection Rate) = TP / (TP + FN)
        - FAR (False Accept Rate / Bypass Rate) = FN / (TP + FN) if attack-centric, or FP / (FP + TN)
        - FRR (False Reject Rate) = FP / (TN + FP)
        - Specificity = TN / (TN + FP)
        - F1 = 2 * (Precision * Recall) / (Precision + Recall)
        - Accuracy = (TP + TN) / (TP + TN + FP + FN)
        """
        if len(y_true_is_attack) != len(y_pred_is_defense_triggered):
            raise ValueError("y_true and y_pred must have identical length.")

        tp = sum(1 for yt, yp in zip(y_true_is_attack, y_pred_is_defense_triggered) if yt and yp)
        fp = sum(1 for yt, yp in zip(y_true_is_attack, y_pred_is_defense_triggered) if not yt and yp)
        tn = sum(1 for yt, yp in zip(y_true_is_attack, y_pred_is_defense_triggered) if not yt and not yp)
        fn = sum(1 for yt, yp in zip(y_true_is_attack, y_pred_is_defense_triggered) if yt and not yp)
        total = tp + fp + tn + fn

        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 1.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 1.0
        far = (fn / (tp + fn)) if (tp + fn) > 0 else 0.0
        frr = (fp / (tn + fp)) if (tn + fp) > 0 else 0.0
        specificity = (tn / (tn + fp)) if (tn + fp) > 0 else 1.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        accuracy = ((tp + tn) / total) if total > 0 else 1.0

        cm = ConfusionMatrix(tp=tp, fp=fp, tn=tn, fn=fn)
        metrics = ClassificationMetrics(
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
            total_cases=total,
            precision=round(precision, 4),
            recall=round(recall, 4),
            far=round(far, 4),
            frr=round(frr, 4),
            specificity=round(specificity, 4),
            f1_score=round(f1, 4),
            accuracy=round(accuracy, 4),
        )
        return metrics, cm

    @staticmethod
    def compute_latency_profile(
        durations_ms: Sequence[float],
        input_tokens: Sequence[int] = (),
        output_tokens: Sequence[int] = (),
        preflight_durations_ms: Sequence[float] = (),
        stage_durations: Optional[dict[str, list[float]]] = None,
    ) -> LatencyProfile:
        """Compute latency statistics, percentiles, and token efficiency metrics."""
        if not durations_ms:
            return LatencyProfile()

        sorted_durations = sorted(durations_ms)
        n = len(sorted_durations)
        total_ms = sum(sorted_durations)
        mean_ms = total_ms / n

        def _percentile(p: float) -> float:
            idx = min(int(n * p), n - 1)
            return sorted_durations[idx]

        p50 = _percentile(0.50)
        p90 = _percentile(0.90)
        p95 = _percentile(0.95)
        p99 = _percentile(0.99)
        min_ms = sorted_durations[0]
        max_ms = sorted_durations[-1]

        tot_in = sum(input_tokens) if input_tokens else 0
        tot_out = sum(output_tokens) if output_tokens else 0
        tot_tok = tot_in + tot_out
        lat_per_tok = (total_ms / tot_tok) if tot_tok > 0 else 0.0

        preflight_mean = statistics.mean(preflight_durations_ms) if preflight_durations_ms else 0.0
        # If preflight ran vs typical LLM duration (~500ms mock or live), compute speedup
        speedup = (mean_ms / preflight_mean) if preflight_mean > 0 else 1.0

        stage_summary: dict[str, float] = {}
        if stage_durations:
            for sname, svals in stage_durations.items():
                if svals:
                    stage_summary[sname] = round(statistics.mean(svals), 2)

        return LatencyProfile(
            total_duration_ms=round(total_ms, 2),
            mean_duration_ms=round(mean_ms, 2),
            p50_ms=round(p50, 2),
            p90_ms=round(p90, 2),
            p95_ms=round(p95, 2),
            p99_ms=round(p99, 2),
            min_ms=round(min_ms, 2),
            max_ms=round(max_ms, 2),
            total_input_tokens=tot_in,
            total_output_tokens=tot_out,
            total_tokens=tot_tok,
            latency_per_token_ms=round(lat_per_tok, 4),
            preflight_mean_ms=round(preflight_mean, 2),
            preflight_speedup_factor=round(speedup, 2),
            stage_latencies_ms=stage_summary,
        )

    @staticmethod
    def compute_pipeline_stability_score(
        classification: ClassificationMetrics,
        hallucination_leakage_rate: float = 0.0,
        enforcement_overreach_rate: float = 0.0,
        rewrite_loop_stress: float = 1.0,
    ) -> float:
        """Compute unified Pipeline Stability Score (PSS, 0 to 100).

        Formula:
        PSS = 100 - (40 * HLR + 25 * FAR + 15 * FRR + 10 * max(0, RLS - 1.0) + 10 * EOI)
        """
        hlr = max(0.0, min(1.0, hallucination_leakage_rate))
        far = classification.far
        frr = classification.frr
        rls_penalty = max(0.0, rewrite_loop_stress - 1.0)
        eoi = max(0.0, min(1.0, enforcement_overreach_rate))

        penalty = (40.0 * hlr) + (25.0 * far) + (15.0 * frr) + (10.0 * rls_penalty) + (10.0 * eoi)
        score = max(0.0, min(100.0, 100.0 - penalty))
        return round(score, 2)
