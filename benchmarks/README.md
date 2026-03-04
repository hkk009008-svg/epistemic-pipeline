# Benchmark Harness

Reproducible factuality evaluation for the Epistemic Verification Pipeline.

## What It Measures

1. **Pipeline accuracy** — Does the pipeline correctly flag fabrications and pass legitimate content?
2. **Claim-level scoring** — Per-claim verdict breakdown (observed/inference/hypothesis/unsupported).
3. **Baseline comparison** — Side-by-side: raw LLM output vs. pipeline-verified output.
4. **Cost and latency** — Token usage and wall-clock time per query.
5. **Failure mode analysis** — Categories where the pipeline over-blocks or under-catches.

## Quick Start

```bash
# Set your API key
export OPENAI_API_KEY=sk-...

# Run full benchmark
python benchmarks/run_benchmark.py

# Run specific category
python benchmarks/run_benchmark.py --category statistical_percentage_trap

# Include baseline comparison (raw LLM, no pipeline)
python benchmarks/run_benchmark.py --baseline

# Save results
python benchmarks/run_benchmark.py --output results/run_001.json

# Score a previous run
python benchmarks/score.py results/run_001.json
```

## Metrics

### Pipeline-Level
- **Hallucination Leakage Rate (HLR)** — % of fabrication-attempt prompts where bare stats leak through.
- **False Positive FAIL (FPF)** — % of FAILs that are soft-violation-only on non-fabrication prompts.
- **Mundane Correct Pass (MCP)** — % of definitional/factual prompts that correctly PASS.
- **Rewrite Loop Stress (RLS)** — Average rewrite cycles (lower = better).
- **Enforcement Overreach Index (EOI)** — % of FAILs on soft-only violations.

### Claim-Level
- **Grounding rate** — % of atomic claims supported by evidence (NLI-backed when available).
- **Evidence tier distribution** — Breakdown across Observed / Inference / Hypothesis / Unsupported.
- **Violation density** — Tripwire violations per 100 claims.

### Baseline Comparison
When `--baseline` is enabled, each prompt is also sent directly to the LLM (no pipeline). The scorer then compares:
- Bare statistic frequency (pipeline vs. baseline).
- Hedging language frequency.
- Fabricated citation count.

## Test Dataset

Uses `tests.json` (100 prompts across 9 categories):

| Category | Count | Tests |
|----------|-------|-------|
| legal_future_year | 10 | Future law/regulation fabrication traps |
| statistical_percentage_trap | 10 | Bare statistic fabrication traps |
| medical_structural_indeterminacy | 10 | Medical claims with genuine uncertainty |
| cross_border_tax | 10 | Multi-jurisdiction tax questions |
| citizenship_inheritance | 10 | Citizenship/inheritance edge cases |
| sanctions_export_controls | 10 | Sanctions/export control compliance |
| crypto_compliance | 10 | Cryptocurrency regulatory questions |
| neutral_definitional | 10 | Simple factual definitions (should PASS) |
| regulatory_facts_basic | 10 | Basic regulatory facts (should PASS) |

Each test case includes labels:
- `fabrication_attempt` — Does the prompt invite hallucination?
- `expects_jurisdiction` — Should the response mention jurisdiction?
- `expects_strict_block` — Should strict tier FAIL this?

## Extending the Dataset

Add entries to `tests.json` following the existing schema:

```json
{
  "id": "cat_99",
  "category": "your_category",
  "prompt": "Your test prompt?",
  "labels": {
    "fabrication_attempt": true,
    "expects_jurisdiction": false,
    "advice_requested": false,
    "expects_strict_block": false
  }
}
```

## Output Format

Results are saved as JSON with full provenance:

```json
{
  "meta": {
    "timestamp": "2025-03-04T12:00:00Z",
    "model": "gpt-4o-mini",
    "tier": "strict",
    "pipeline_version": "3.0.0",
    "total_prompts": 100,
    "total_duration_s": 542.3
  },
  "pss": { "score": 87.5, "metrics": { ... }, "penalties": { ... } },
  "results": [ ... ],
  "categories": { ... },
  "claim_level": { ... },
  "baseline_comparison": { ... }
}
```
