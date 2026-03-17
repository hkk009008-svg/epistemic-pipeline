# Epistemic Verification Pipeline

A 3-stage LLM verification pipeline that checks factual accuracy and epistemic integrity of AI-generated content. Uses a **Generator → Verifier → Arbiter** architecture with the Audit v8 epistemic framework to catch hallucinations, fabricated statistics, unsupported causal claims, and prescriptive creep before they reach users.

## The Problem

Large language models hallucinate. They fabricate statistics, state causal claims as fact, give prescriptive advice as if it were established truth, and present future speculation as current reality. Standard guardrails (content filters, RLHF) don't catch these epistemic failures because they aren't designed to.

This pipeline adds a **verification layer** between the LLM and the user:

```
User Question
  → GPT-1 (Generator)     — produces an answer
  → GPT-2 (Verifier)      — checks every claim against 7 tripwire violation types
  → GPT-3 (Arbiter)       — rewrites or blocks if violations are found
  → Verified Response      — with confidence score and evidence breakdown
```

## What It Catches

| Code | Violation | Example |
|------|-----------|---------|
| T1 | Fabricated Statistic | "73% of startups fail" (no source) |
| T2 | False Certainty | "This will definitely work" |
| T3 | Causal Claim as Fact | "X causes Y" without evidence |
| T4 | Ranking Violation | "Country A is best" without criteria |
| T5 | Prescriptive Creep | "You should do X" framed as fact |
| T6 | Unsupported Evidence | Citing a study that doesn't exist |
| T7 | Unverified Current Fact | Stating future laws/prices as known |

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env — set OPENAI_API_KEY (required)

# Run
uvicorn app:app --host 0.0.0.0 --port 8000

# Test
python -m pytest tests/ -v
```

Then open http://localhost:8000 for the web UI, or call the API directly:

```bash
curl -X POST http://localhost:8000/api/pipeline \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What percentage of startups fail in the first 5 years?"}'
```

## API

### `POST /api/pipeline`

Main verification endpoint. Accepts a prompt, runs it through the 3-stage pipeline, returns a verified response with confidence scoring.

**Request:**
```json
{
  "prompt": "Is staking income taxable in the US?",
  "tier": "strict",
  "output_format": "auto",
  "stream": false
}
```

**Response:**
```json
{
  "final_verdict": "PASS",
  "final_result": "Staking income is generally treated as taxable income by the IRS...",
  "verdict_label": "Verified with evidence",
  "confidence": {
    "observed_pct": 60.0,
    "inference_pct": 30.0,
    "hypothesis_pct": 10.0,
    "unsupported_pct": 0.0,
    "confidence_label": "Medium",
    "confidence_reasoning": ["Claims grounded in IRS guidance..."]
  },
  "claim_table": [
    {"claim": "Staking rewards are taxable", "category": "Observed", "justification": "IRS Rev. Rul. 2023-14"}
  ],
  "violations": [],
  "gpt2_verdict": "PASS",
  "search_performed": true,
  "search_sources": [
    {"title": "IRS Guidance on Staking", "url": "https://...", "snippet": "..."}
  ]
}
```

**Streaming mode** (`stream: true`): Returns NDJSON with real-time stage events:
```
{"type": "stage_start", "stage": "gpt1", "elapsed": 0.0}
{"type": "stage_complete", "stage": "gpt1", "elapsed": 2.3}
{"type": "stage_start", "stage": "gpt2", "elapsed": 2.3}
...
{"type": "result", "data": { ... full response ... }}
```

### Other Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Web chat UI |
| `GET` | `/health` | Health check |
| `POST` | `/api/stress` | Stress test (streaming NDJSON) |
| `POST` | `/api/openai/config` | Set OpenAI key/model (admin) |
| `POST` | `/api/tavily/config` | Set Tavily key (admin) |
| `POST` | `/api/stage/config` | Per-stage model config (admin) |
| `GET` | `/api/metrics` | Aggregate pipeline metrics |
| `POST` | `/api/feedback` | Submit accuracy feedback |

Admin endpoints require `Authorization: Bearer <ADMIN_TOKEN>` when `ADMIN_TOKEN` is set.

## Verification Tiers

| Tier | Description |
|------|-------------|
| **strict** | Full Audit v8 — all tripwire checks, arbiter rewrite loops, maximum scrutiny |
| **standard** | Balanced — core fact-checking, relaxed on stylistic violations |
| **light** | Fast fact-check only — catches fabrications, skips prescriptive/ranking checks |

## Architecture

```
pipeline/
  orchestrator.py     # Core pipeline flow with stage event emission
  prompts.py          # Audit v8 system prompts and flag-aware augmentation
  sanitizer.py        # Deterministic routing (flags) and citation-aware output cleaning
  verifier.py         # GPT-2 JSON parsing, verdict computation, reasoning trace
  arbiter.py          # GPT-3 decision parsing, edit application
  convergence.py      # Rewrite loop convergence/oscillation detection
  decomposer.py       # Atomic claim decomposition (pre-verification)
  nli.py              # Optional NLI verification (DeBERTa-v3 or remote)
  search.py           # Tavily web search integration
  helpers.py          # LLM client management, JSON extraction, error types
```

### Key Design Decisions

- **Split GPT-2 prompt**: Core instructions in system prompt, detailed tripwire reference injected at start of user content (avoids lost-in-middle attention degradation).
- **Citation-aware sanitizer**: Statistics with nearby citations are preserved; bare stats without sources are flagged.
- **Convergence detection**: Rewrite loops track finding deltas — stops on convergence, oscillation, or regression (max 3 loops).
- **Deterministic routing**: All flag extraction and sanitization is regex-based. No randomness in the verification logic.
- **SSRF protection**: Stage `base_url` values are validated against an allowlist.

## Benchmarking

Run the built-in benchmark harness to evaluate pipeline accuracy:

```bash
# Run full benchmark (requires OPENAI_API_KEY)
python benchmarks/run_benchmark.py

# Run specific category
python benchmarks/run_benchmark.py --category statistical_percentage_trap

# Compare with baseline (no pipeline)
python benchmarks/run_benchmark.py --baseline

# Output results to file
python benchmarks/run_benchmark.py --output results.json
```

See [`benchmarks/README.md`](benchmarks/README.md) for methodology, metrics, and interpretation.

## Stress Testing

The built-in stress test runs 100 prompts across 9 categories and computes a **Pipeline Stability Score (PSS)**:

```bash
curl -X POST http://localhost:8000/api/stress \
  -H "Content-Type: application/json" \
  -d '{"tier": "strict"}' \
  --no-buffer
```

PSS metrics: Hallucination Leakage Rate (HLR), False Positive FAIL rate (FPF), Mundane Correct Pass rate (MCP), Rewrite Loop Stress (RLS), Enforcement Overreach Index (EOI).

## Deployment

### Railway (recommended)
```bash
# Configure railway.toml (included), set env vars, deploy
railway up
```

### Docker
```bash
docker build -f Dockerfile.local -t epistemic-pipeline .
docker run -p 8000:8000 --env-file .env epistemic-pipeline
```

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | — | OpenAI API key |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | Model identifier |
| `TAVILY_API_KEY` | No | — | Tavily web search key |
| `ADMIN_TOKEN` | No | — | Protects config endpoints |
| `ALLOWED_ORIGINS` | No | — | CORS allowed origins |
| `FORCE_HTTPS` | No | `false` | Enable HSTS header |

See [`.env.example`](.env.example) for the full list.

## Testing

```bash
python -m pytest tests/ -v          # All 279+ tests
python -m pytest tests/ -v -x       # Stop on first failure
python -m pytest tests/test_sanitizer.py -v   # Specific module
```

All tests are deterministic — no LLM calls needed.

## Security

See [SECURITY.md](SECURITY.md) for vulnerability reporting, security design, and production hardening checklist.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style, and PR guidelines.

## License

See [LICENSE](LICENSE) for details.
