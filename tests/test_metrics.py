"""Tests for pipeline metrics collection."""
from __future__ import annotations

import time

from pipeline.metrics import PipelineMetrics, AggregateMetrics, get_aggregate


class TestPipelineMetrics:
    """Test individual pipeline run metrics."""

    def test_start_and_finish_sets_duration(self):
        m = PipelineMetrics()
        m.start()
        time.sleep(0.01)
        m.finish()
        assert m.total_duration_ms > 0

    def test_stage_timing(self):
        m = PipelineMetrics()
        sm = m.start_stage("gpt1", provider="openai", model="gpt-4o-mini")
        time.sleep(0.01)
        m.end_stage(sm)
        assert len(m.stages) == 1
        assert m.stages[0].stage == "gpt1"
        assert m.stages[0].duration_ms > 0

    def test_stage_error(self):
        m = PipelineMetrics()
        sm = m.start_stage("gpt2")
        m.end_stage(sm, error="timeout")
        assert m.stages[0].error == "timeout"

    def test_to_dict_excludes_internal_timing(self):
        m = PipelineMetrics(request_id="abc123")
        sm = m.start_stage("gpt1")
        m.end_stage(sm)
        d = m.to_dict()
        assert d["request_id"] == "abc123"
        assert "started_at" not in d["stages"][0]
        assert "ended_at" not in d["stages"][0]
        assert "duration_ms" in d["stages"][0]

    def test_to_dict_all_fields_present(self):
        m = PipelineMetrics()
        d = m.to_dict()
        assert "gpt2_verdict" in d
        assert "hard_findings" in d
        assert "confidence_label" in d
        assert "nli_ran" in d
        assert "rewrite_loops" in d

    def test_emit_does_not_raise(self):
        m = PipelineMetrics(request_id="test")
        m.start()
        m.finish()
        m.emit()  # should not raise

    def test_defaults(self):
        m = PipelineMetrics()
        assert m.request_id == ""
        assert m.total_duration_ms == 0.0
        assert m.bypassed is False
        assert m.rewrite_loops == 0


class TestAggregateMetrics:
    """Test aggregate metrics tracking."""

    def test_record_pass(self):
        agg = AggregateMetrics()
        m = PipelineMetrics(final_verdict="PASS", total_duration_ms=100.0)
        agg.record(m)
        assert agg.total_requests == 1
        assert agg.total_pass == 1
        assert agg.total_fail == 0

    def test_record_fail(self):
        agg = AggregateMetrics()
        m = PipelineMetrics(final_verdict="FAIL", total_duration_ms=200.0)
        agg.record(m)
        assert agg.total_fail == 1

    def test_record_bypass(self):
        agg = AggregateMetrics()
        m = PipelineMetrics(bypassed=True, total_duration_ms=10.0)
        agg.record(m)
        assert agg.total_bypass == 1

    def test_record_arbiter(self):
        agg = AggregateMetrics()
        m = PipelineMetrics(arbiter_decision="BLOCK", total_duration_ms=300.0)
        agg.record(m)
        assert agg.total_arbiter_invoked == 1

    def test_record_rewrite(self):
        agg = AggregateMetrics()
        m = PipelineMetrics(rewrite_loops=2, total_duration_ms=500.0)
        agg.record(m)
        assert agg.total_rewrites == 1

    def test_record_search(self):
        agg = AggregateMetrics()
        m = PipelineMetrics(search_performed=True, total_duration_ms=150.0)
        agg.record(m)
        assert agg.total_search_performed == 1

    def test_avg_duration(self):
        agg = AggregateMetrics()
        agg.record(PipelineMetrics(final_verdict="PASS", total_duration_ms=100.0))
        agg.record(PipelineMetrics(final_verdict="PASS", total_duration_ms=200.0))
        assert agg.avg_duration_ms == 150.0

    def test_to_dict(self):
        agg = AggregateMetrics()
        d = agg.to_dict()
        assert "total_requests" in d
        assert "avg_duration_ms" in d

    def test_get_aggregate_returns_singleton(self):
        a1 = get_aggregate()
        a2 = get_aggregate()
        assert a1 is a2
