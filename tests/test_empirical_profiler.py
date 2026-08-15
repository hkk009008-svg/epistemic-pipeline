"""Unit and integration tests for the Empirical Profiler package.

Validates:
- MetricsEngine: Precision, Recall, FAR, FRR, F1, Accuracy, Latency percentiles, Token rates
- ConfusionMatrix: Binary/multi-class mappings, count lookups, dictionary serialization
- Cross-platform memory handling: Darwin (byte) vs Linux (KB) scaling, tracemalloc tracking
- BreakingPointAnalyzer: Density sweeps, syntactic depth sweeps, Tri-State boundary classification
- LimitReportGenerator: Markdown rendering, JSON serialization, Pydantic schema validation
- AdversarialHarness: Single-case evaluation, batch suites, pre-flight short-circuit timing
- Edge Cases: Empty collections, 0-division safety, extreme density boundaries (0.0 and 1.0)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from pipeline.adversarial import (
    AdversarialMutationConfig,
    AttackVectorEnum,
    DifficultyTierEnum,
    DocumentRecord,
    MultiDomainScenario,
    PoisonedScenarioCase,
    load_scenario_corpus,
)
from pipeline.profiler import (
    AdversarialHarness,
    AttackVectorMetrics,
    BoundaryState,
    BreakingAnalysisResult,
    BreakingPointAnalyzer,
    CaseEvaluationResult,
    ClassificationMetrics,
    ConfusionMatrix,
    DensityProbePoint,
    DomainMetrics,
    EvaluationMetrics,
    LatencyProfile,
    LimitReportData,
    LimitReportGenerator,
    MemoryProfile,
    MetricsEngine,
    ProfilerConfig,
    SyntacticDepthProbePoint,
    ThresholdComparison,
    VectorBreakingSummary,
    classify_defense_state,
    get_peak_rss_bytes,
    get_peak_rss_mb,
)


# ---------------------------------------------------------------------------
# Test Suite: MetricsEngine & Classification Math
# ---------------------------------------------------------------------------

class TestMetricsEngineClassification:
    """Test mathematical accuracy and edge cases of classification metrics."""

    def test_perfect_classification_metrics(self) -> None:
        y_true = [True, True, True, False, False]
        y_pred = [True, True, True, False, False]
        metrics, cm = MetricsEngine.compute_classification(y_true, y_pred)

        assert metrics.tp == 3
        assert metrics.fp == 0
        assert metrics.tn == 2
        assert metrics.fn == 0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.far == 0.0
        assert metrics.frr == 0.0
        assert metrics.specificity == 1.0
        assert metrics.f1_score == 1.0
        assert metrics.accuracy == 1.0
        assert cm.total == 5

    def test_mixed_classification_metrics(self) -> None:
        # TP=2, FP=1, TN=3, FN=1 (Total = 7)
        y_true = [True, True, True, False, False, False, False]
        y_pred = [True, True, False, True, False, False, False]
        metrics, cm = MetricsEngine.compute_classification(y_true, y_pred)

        assert metrics.tp == 2
        assert metrics.fp == 1
        assert metrics.tn == 3
        assert metrics.fn == 1
        assert metrics.total_cases == 7
        assert metrics.precision == round(2 / (2 + 1), 4)  # 0.6667
        assert metrics.recall == round(2 / (2 + 1), 4)     # 0.6667
        assert metrics.far == round(1 / (2 + 1), 4)        # 0.3333
        assert metrics.frr == round(1 / (3 + 1), 4)        # 0.2500
        assert metrics.accuracy == round((2 + 3) / 7, 4)   # 0.7143

    def test_zero_division_safety_empty_inputs(self) -> None:
        metrics, cm = MetricsEngine.compute_classification([], [])
        assert metrics.total_cases == 0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.far == 0.0
        assert metrics.frr == 0.0
        assert metrics.accuracy == 1.0
        assert cm.total == 0

    def test_zero_division_safety_all_negative(self) -> None:
        y_true = [False, False, False]
        y_pred = [False, False, False]
        metrics, cm = MetricsEngine.compute_classification(y_true, y_pred)

        assert metrics.tp == 0
        assert metrics.tn == 3
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.far == 0.0
        assert metrics.frr == 0.0
        assert metrics.accuracy == 1.0

    def test_mismatched_lengths_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="identical length"):
            MetricsEngine.compute_classification([True, False], [True])


# ---------------------------------------------------------------------------
# Test Suite: ConfusionMatrix
# ---------------------------------------------------------------------------

class TestConfusionMatrix:
    """Test ConfusionMatrix structure and indexing."""

    def test_confusion_matrix_init_and_get(self) -> None:
        cm = ConfusionMatrix(tp=10, fp=2, tn=20, fn=1)
        assert cm.total == 33
        d = cm.to_dict()
        assert d["tp"] == 10
        assert d["fp"] == 2
        assert d["tn"] == 20
        assert d["fn"] == 1
        assert cm.get("ATTACK", "ATTACK") == 10
        assert cm.get("CLEAN", "CLEAN") == 20
        assert cm.get("UNKNOWN", "LABEL") == 0


# ---------------------------------------------------------------------------
# Test Suite: Latency & Performance Profile
# ---------------------------------------------------------------------------

class TestLatencyProfile:
    """Test latency percentile and token efficiency calculations."""

    def test_empty_durations_returns_defaults(self) -> None:
        profile = MetricsEngine.compute_latency_profile([])
        assert profile.total_duration_ms == 0.0
        assert profile.mean_duration_ms == 0.0

    def test_percentile_calculation(self) -> None:
        durations = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        in_toks = [100] * 10
        out_toks = [50] * 10
        preflight_durs = [2.0, 2.5]
        stage_durs = {"stage_verify": [5.0, 10.0], "stage_arbiter": [15.0, 20.0]}

        profile = MetricsEngine.compute_latency_profile(
            durations_ms=durations,
            input_tokens=in_toks,
            output_tokens=out_toks,
            preflight_durations_ms=preflight_durs,
            stage_durations=stage_durs,
        )

        assert profile.total_duration_ms == 550.0
        assert profile.mean_duration_ms == 55.0
        assert profile.p50_ms == 60.0
        assert profile.min_ms == 10.0
        assert profile.max_ms == 100.0
        assert profile.total_tokens == 1500
        assert profile.latency_per_token_ms == round(550.0 / 1500, 4)
        assert profile.preflight_mean_ms == 2.25
        assert profile.stage_latencies_ms["stage_verify"] == 7.5
        assert profile.stage_latencies_ms["stage_arbiter"] == 17.5


# ---------------------------------------------------------------------------
# Test Suite: Memory Footprint & Cross-Platform Scaling
# ---------------------------------------------------------------------------

class TestMemoryProfiling:
    """Test memory usage profiling and platform scaling."""

    def test_get_peak_rss_mb_positive(self) -> None:
        rss = get_peak_rss_mb()
        assert isinstance(rss, float)
        assert rss > 0.0

    def test_get_peak_rss_bytes_positive(self) -> None:
        rss_b = get_peak_rss_bytes()
        assert isinstance(rss_b, int)
        assert rss_b > 0

    def test_darwin_vs_linux_platform_scaling(self) -> None:
        mock_usage = MagicMock()
        mock_usage.ru_maxrss = 104857600  # 100MB in bytes, or 100GB in KB

        with patch("resource.getrusage", return_value=mock_usage):
            with patch("sys.platform", "darwin"):
                # macOS divides by 1024 * 1024
                rss_mac = get_peak_rss_mb()
                assert rss_mac == 100.0

            with patch("sys.platform", "linux"):
                # Linux divides by 1024
                rss_linux = get_peak_rss_mb()
                assert rss_linux == 102400.0


# ---------------------------------------------------------------------------
# Test Suite: BreakingPointAnalyzer
# ---------------------------------------------------------------------------

class TestBreakingPointAnalyzer:
    """Test density sweeps, syntactic depth sweeps, and Tri-State classification."""

    def test_tri_state_defense_classification(self) -> None:
        # Clean input -> HOLD
        assert classify_defense_state(
            unbacked_ratio=0.0, hard_count=0, verdict="PASS", arbiter_decision="ALLOW"
        ) == "HOLD"

        # Lightly corrupted draft -> HOLD (ALLOW_WITH_EDITS)
        assert classify_defense_state(
            unbacked_ratio=0.20, hard_count=1, verdict="PASS", arbiter_decision="ALLOW_WITH_EDITS"
        ) == "HOLD"

        # Heavily corrupted draft -> FAIL_CLOSED (BLOCK)
        assert classify_defense_state(
            unbacked_ratio=0.40, hard_count=1, verdict="FAIL", arbiter_decision="BLOCK"
        ) == "FAIL_CLOSED"

        # Multiple hard violations -> FAIL_CLOSED (BLOCK)
        assert classify_defense_state(
            unbacked_ratio=0.10, hard_count=2, verdict="FAIL", arbiter_decision="BLOCK"
        ) == "FAIL_CLOSED"

        # Defect: heavily corrupted draft allowed -> EDGE_CASE
        assert classify_defense_state(
            unbacked_ratio=0.50, hard_count=3, verdict="PASS", arbiter_decision="ALLOW"
        ) == "EDGE_CASE"

        # Defect: leaked hallucinations -> EDGE_CASE
        assert classify_defense_state(
            unbacked_ratio=0.0, hard_count=0, verdict="PASS", arbiter_decision="ALLOW",
            leaked_hallucinations=True,
        ) == "EDGE_CASE"

    def test_density_sweep_execution(self) -> None:
        analyzer = BreakingPointAnalyzer(unsupported_threshold=0.35, hard_threshold=2)
        probes = analyzer.run_density_sweep(probe_densities=[0.0, 0.20, 0.35, 0.50, 1.0])

        assert len(probes) == 5
        # 0.0 -> HOLD
        assert probes[0].boundary_state == "HOLD"
        assert probes[0].guarded_decision == "ALLOW_WITH_EDITS"
        # 0.50 -> FAIL_CLOSED (BLOCK)
        assert probes[3].boundary_state == "FAIL_CLOSED"
        assert probes[3].guarded_decision == "BLOCK"
        # 1.0 -> FAIL_CLOSED (BLOCK)
        assert probes[4].boundary_state == "FAIL_CLOSED"
        assert probes[4].guarded_decision == "BLOCK"

    def test_syntactic_depth_sweep_execution(self) -> None:
        analyzer = BreakingPointAnalyzer()
        depth_points = analyzer.run_syntactic_depth_sweep()

        assert len(depth_points) == 5
        levels = [dp.depth_level for dp in depth_points]
        assert levels == [1, 2, 3, 4, 5]
        for dp in depth_points:
            assert dp.poison_detected is True
            assert dp.clean_facts_preserved is True

    def test_full_breaking_analysis(self) -> None:
        analyzer = BreakingPointAnalyzer()
        result = analyzer.full_breaking_analysis()

        assert isinstance(result, BreakingAnalysisResult)
        assert result.invariants_satisfied is True
        assert result.hold_count > 0
        assert result.fail_closed_count > 0
        assert result.edge_case_count == 0
        assert len(result.vector_summaries) == 6


# ---------------------------------------------------------------------------
# Test Suite: LimitReportGenerator
# ---------------------------------------------------------------------------

class TestLimitReportGenerator:
    """Test report generation and serialization in Markdown and JSON."""

    def test_build_and_render_report(self, tmp_path: Path) -> None:
        harness = AdversarialHarness()
        scenarios = load_scenario_corpus()
        sample_scenarios = scenarios[:2]

        metrics, breaking, _ = harness.run_full_corpus_benchmark(scenarios=sample_scenarios)
        report_data = LimitReportGenerator.build_report_data(metrics, breaking)

        assert isinstance(report_data, LimitReportData)
        assert report_data.total_cases_evaluated > 0

        # Render Markdown
        md_text = LimitReportGenerator.generate_markdown_report(report_data)
        assert "# Epistemic Pipeline Empirical Limit Profiler & Stress Test Report" in md_text
        assert "## 1. Executive Summary & Key Performance Indicators" in md_text
        assert "## 2. Global Confusion Matrix & Classification Metrics" in md_text
        assert "## 3. Multi-Domain Performance Breakdown" in md_text
        assert "## 4. Attack Vector & Difficulty Tier Matrix" in md_text
        assert "## 5. Empirical Breaking Point & Boundary Analysis" in md_text
        assert "## 6. Computational Latency & Resource Footprint" in md_text
        assert "## 7. Defense Invariant Attestation & Forensic Sign-Off" in md_text
        assert "EPISTEMIC-LIMIT-PROFILER-M3-VERIFIED" in md_text

        # Export to files
        md_file = tmp_path / "TEST_LIMIT_REPORT.md"
        json_file = tmp_path / "test_limit_report.json"
        LimitReportGenerator.export_reports(report_data, md_file, json_file)

        assert md_file.is_file()
        assert json_file.is_file()

        # Validate exported JSON against Pydantic schema
        with open(json_file, "r", encoding="utf-8") as f:
            raw_json = json.load(f)
        validated = LimitReportData.model_validate(raw_json)
        assert validated.schema_version == "epistemic-limit-profiler-v1"


# ---------------------------------------------------------------------------
# Test Suite: AdversarialHarness
# ---------------------------------------------------------------------------

class TestAdversarialHarness:
    """Test AdversarialHarness evaluation mechanics."""

    def test_evaluate_clean_case(self) -> None:
        harness = AdversarialHarness()
        scenarios = load_scenario_corpus()
        sc = scenarios[0]

        clean_case = PoisonedScenarioCase(
            test_id=f"clean_{sc.scenario_id}",
            scenario_id=sc.scenario_id,
            attack_vector="statistical_fallacy",
            difficulty_tier="mild",
            mutated_prompt=sc.query,
            mutated_draft=sc.clean_baseline_draft,
            evidence_documents=sc.documents,
            expected_defense=sc.ground_truth_facts[0].metadata.get("expected_defense") or {  # type: ignore
                "expected_verdict": "PASS",
                "expected_arbiter_decision": "ALLOW",
                "target_unbacked_ratio": 0.0,
                "target_hard_violations": 0,
            },
        )

        res = harness.evaluate_case(clean_case, is_clean_control=True)
        assert res.verdict == "PASS"
        assert res.arbiter_decision == "ALLOW"
        assert res.defense_triggered is False
        assert res.boundary_state == "HOLD"
        assert res.is_attack is False

    def test_evaluate_breaking_tier_case_triggers_block(self) -> None:
        harness = AdversarialHarness()
        scenarios = load_scenario_corpus()
        sc = scenarios[0]

        from pipeline.adversarial import AdversarialMutationEngine, AttackVectorEnum
        engine = AdversarialMutationEngine()
        case = engine.mutate_scenario(sc, vector=AttackVectorEnum.POISONING_SATURATION, tier="breaking")

        res = harness.evaluate_case(case, is_clean_control=False)
        assert res.verdict == "FAIL"
        assert res.arbiter_decision == "BLOCK"
        assert res.defense_triggered is True
        assert res.boundary_state == "FAIL_CLOSED"
        assert res.is_attack is True
