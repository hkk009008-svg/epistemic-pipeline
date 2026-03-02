"""Structured logging and pipeline metrics collection.

Provides a PipelineMetrics collector that tracks timing, token usage,
verdicts, and stage-level details for each pipeline run. Emits structured
JSON log entries for observability.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict

logger = logging.getLogger("pipeline.metrics")


@dataclass
class StageMetric:
    """Timing and metadata for a single pipeline stage."""
    stage: str
    started_at: float = 0.0
    ended_at: float = 0.0
    duration_ms: float = 0.0
    provider: str = ""
    model: str = ""
    error: str = ""


@dataclass
class PipelineMetrics:
    """Collects metrics across a single pipeline run."""
    request_id: str = ""
    prompt_length: int = 0
    # Stage timings
    stages: list[StageMetric] = field(default_factory=list)
    # Verdict chain
    gpt2_verdict: str = ""
    arbiter_decision: str = ""
    final_verdict: str = ""
    # Findings
    hard_findings: int = 0
    soft_findings: int = 0
    total_claims: int = 0
    # Confidence
    confidence_label: str = ""
    grounding_rate: float = 0.0
    # Search
    search_performed: bool = False
    search_sources_count: int = 0
    # Decomposition
    decomposition_ran: bool = False
    atomic_claims_count: int = 0
    # NLI
    nli_ran: bool = False
    nli_supported_count: int = 0
    nli_contradicted_count: int = 0
    # Rewrite loop
    rewrite_loops: int = 0
    convergence_outcome: str = ""  # "pass", "converged", "oscillating", "max_loops"
    # Overall
    total_duration_ms: float = 0.0
    bypassed: bool = False
    flags: dict = field(default_factory=dict)
    _start_time: float = field(default=0.0, repr=False)

    def start(self) -> None:
        """Mark the start of the pipeline run."""
        self._start_time = time.monotonic()

    def finish(self) -> None:
        """Mark the end of the pipeline run and compute total duration."""
        if self._start_time > 0:
            self.total_duration_ms = round((time.monotonic() - self._start_time) * 1000, 1)

    def start_stage(self, stage: str, provider: str = "", model: str = "") -> StageMetric:
        """Begin timing a pipeline stage."""
        metric = StageMetric(
            stage=stage, started_at=time.monotonic(),
            provider=provider, model=model,
        )
        self.stages.append(metric)
        return metric

    def end_stage(self, metric: StageMetric, error: str = "") -> None:
        """End timing a pipeline stage."""
        metric.ended_at = time.monotonic()
        metric.duration_ms = round((metric.ended_at - metric.started_at) * 1000, 1)
        metric.error = error

    def to_dict(self) -> dict:
        """Convert to a serializable dictionary."""
        d = {
            "request_id": self.request_id,
            "prompt_length": self.prompt_length,
            "stages": [asdict(s) for s in self.stages],
            "gpt2_verdict": self.gpt2_verdict,
            "arbiter_decision": self.arbiter_decision,
            "final_verdict": self.final_verdict,
            "hard_findings": self.hard_findings,
            "soft_findings": self.soft_findings,
            "total_claims": self.total_claims,
            "confidence_label": self.confidence_label,
            "grounding_rate": self.grounding_rate,
            "search_performed": self.search_performed,
            "search_sources_count": self.search_sources_count,
            "decomposition_ran": self.decomposition_ran,
            "atomic_claims_count": self.atomic_claims_count,
            "nli_ran": self.nli_ran,
            "nli_supported_count": self.nli_supported_count,
            "nli_contradicted_count": self.nli_contradicted_count,
            "rewrite_loops": self.rewrite_loops,
            "convergence_outcome": self.convergence_outcome,
            "total_duration_ms": self.total_duration_ms,
            "bypassed": self.bypassed,
            "flags": self.flags,
        }
        # Strip internal timing fields from stage dicts
        for s in d["stages"]:
            s.pop("started_at", None)
            s.pop("ended_at", None)
        return d

    def emit(self) -> None:
        """Emit the metrics as a structured JSON log entry."""
        try:
            logger.info(json.dumps(self.to_dict(), default=str))
        except Exception:
            logger.warning("Failed to emit pipeline metrics")


# ---------------------------------------------------------------------------
# Aggregate counters (in-memory, reset on restart)
# ---------------------------------------------------------------------------

@dataclass
class AggregateMetrics:
    """Running totals across all pipeline runs since startup."""
    total_requests: int = 0
    total_pass: int = 0
    total_fail: int = 0
    total_bypass: int = 0
    total_arbiter_invoked: int = 0
    total_rewrites: int = 0
    total_search_performed: int = 0
    avg_duration_ms: float = 0.0
    _duration_sum: float = field(default=0.0, repr=False)

    def record(self, m: PipelineMetrics) -> None:
        """Record a completed pipeline run."""
        self.total_requests += 1
        if m.bypassed:
            self.total_bypass += 1
        elif m.final_verdict == "PASS":
            self.total_pass += 1
        else:
            self.total_fail += 1
        if m.arbiter_decision:
            self.total_arbiter_invoked += 1
        if m.rewrite_loops > 0:
            self.total_rewrites += 1
        if m.search_performed:
            self.total_search_performed += 1
        self._duration_sum += m.total_duration_ms
        self.avg_duration_ms = round(self._duration_sum / self.total_requests, 1)

    def to_dict(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "total_pass": self.total_pass,
            "total_fail": self.total_fail,
            "total_bypass": self.total_bypass,
            "total_arbiter_invoked": self.total_arbiter_invoked,
            "total_rewrites": self.total_rewrites,
            "total_search_performed": self.total_search_performed,
            "avg_duration_ms": self.avg_duration_ms,
        }


# Singleton aggregate tracker
_aggregate = AggregateMetrics()


def get_aggregate() -> AggregateMetrics:
    """Return the global aggregate metrics instance."""
    return _aggregate


def record_run(m: PipelineMetrics) -> None:
    """Record a pipeline run in both structured log and aggregates."""
    m.emit()
    _aggregate.record(m)
