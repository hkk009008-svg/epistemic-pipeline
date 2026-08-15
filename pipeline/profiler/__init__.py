"""Automated Empirical Profiler & Breaking Limit Reporting Module."""
from __future__ import annotations

from pipeline.profiler.breaking_analyzer import (
    BoundaryState,
    BreakingAnalysisResult,
    BreakingPointAnalyzer,
    DensityProbePoint,
    SyntacticDepthProbePoint,
    VectorBreakingSummary,
    classify_defense_state,
)
from pipeline.profiler.harness import (
    AdversarialHarness,
    CaseEvaluationResult,
    ProfilerConfig,
)
from pipeline.profiler.metrics_engine import (
    AttackVectorMetrics,
    ClassificationMetrics,
    ConfusionMatrix,
    DomainMetrics,
    EvaluationMetrics,
    LatencyProfile,
    MemoryProfile,
    MetricsEngine,
    get_peak_rss_bytes,
    get_peak_rss_mb,
)
from pipeline.profiler.report_generator import (
    LimitReportData,
    LimitReportGenerator,
    ThresholdComparison,
)

__all__ = [
    "AdversarialHarness",
    "AttackVectorMetrics",
    "BoundaryState",
    "BreakingAnalysisResult",
    "BreakingPointAnalyzer",
    "CaseEvaluationResult",
    "ClassificationMetrics",
    "ConfusionMatrix",
    "DensityProbePoint",
    "DomainMetrics",
    "EvaluationMetrics",
    "LatencyProfile",
    "LimitReportData",
    "LimitReportGenerator",
    "MemoryProfile",
    "MetricsEngine",
    "ProfilerConfig",
    "SyntacticDepthProbePoint",
    "ThresholdComparison",
    "VectorBreakingSummary",
    "classify_defense_state",
    "get_peak_rss_bytes",
    "get_peak_rss_mb",
]
