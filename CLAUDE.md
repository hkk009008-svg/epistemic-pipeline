# CLAUDE.md — Epistemic Verification Pipeline

## Project Overview

A FastAPI-based 3-stage LLM verification pipeline that checks factual accuracy and epistemic integrity of AI-generated content. Uses a Generator → Verifier → Arbiter architecture with the **Audit v7 epistemic framework** (priority stack V1-V7, global rules G1-G12, tripwire violations T1-T7). Includes Tavily web search integration, deterministic sanitization, convergence-aware rewrite loops, multi-provider LLM support, atomic claim decomposition, optional NLI verification, and a comprehensive stress-testing framework.

**Version:** 3.0.0 | **Python:** 3.11.11 | **Prompt Version:** 7.1.0 (Audit v7)

## Repository Structure

```
app.py                  # FastAPI entry point, global exception handler
config.py               # Centralized config, env vars, thread-safe runtime overrides
requirements.txt        # Dependencies: fastapi, uvicorn, openai, pydantic, tavily-python
runtime.txt             # Python version pinned to 3.11.11
Procfile                # Heroku/Railway start command
railway.toml            # Railway deployment config (Nixpacks builder)
Dockerfile.local        # Local Docker build (python:3.11-slim, non-root user)
tests.json              # 100 real-world test cases across 9 categories
.env.example            # Template for environment variables

api/
  routes.py             # All API route definitions
  rate_limit.py         # Per-IP sliding-window rate limiter (20 req/min)
  ui.py                 # Embedded HTML chat UI (served at GET /)

pipeline/
  orchestrator.py       # Main pipeline logic: run_pipeline(), confidence scoring
  models.py             # Pydantic request/response models
  prompts.py            # GPT-1/2/3 system prompts, Audit v7 rules, build_augmentation()
  sanitizer.py          # Deterministic routing (route_prompt) and citation-aware output cleaning
  verifier.py           # GPT-2 JSON parsing (parse_gpt2), verdict computation, reasoning trace
  arbiter.py            # GPT-3 decision parsing (parse_gpt3), edit application
  helpers.py            # Shared utilities: extract_json(), call_openai(), call_llm(), PipelineError
  search.py             # Tavily web search (should_search, perform_web_search)
  convergence.py        # Rewrite loop convergence detection (finding deltas, oscillation)
  decomposer.py         # Atomic claim decomposition (pre-GPT-2 claim splitting)
  nli.py                # Optional NLI verification layer (DeBERTa-v3 or remote service)
  knowledge_store.py    # Versioned private sources + deterministic SQLite FTS5 evidence packets
  grounded_rag.py       # Isolated claim-first/verifier/constrained-finalizer RAG lane
  stress.py             # Stress testing framework, PSS (Pipeline Stability Score)

tests/
  conftest.py           # Pytest fixtures for GPT-2/3 JSON payloads, flags, edits
  test_helpers.py       # extract_json(), is_activation_phrase() tests (~51 tests)
  test_sanitizer.py     # route_prompt(), sanitize_output() tests (~70+ tests)
  test_verifier.py      # parse_gpt2() tests (~65+ tests)
  test_arbiter.py       # parse_gpt3(), apply_edits() tests
  test_convergence.py   # compute_finding_delta(), should_continue_rewrite() tests
  test_decomposer.py    # decompose_claims() tests
  test_multi_provider.py # Per-stage config, _make_client(), call_llm() dispatch tests
  test_nli.py           # NLI verification layer tests

n8n-workflows/          # n8n automation templates (verify, batch stress, health check)

.github/workflows/
  docker-image.yml      # CI: Docker image build on push to main / PRs
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and fill environment variables
cp .env.example .env
# Edit .env with your OPENAI_API_KEY (required) and optionally TAVILY_API_KEY

# Run the server
uvicorn app:app --host 0.0.0.0 --port 8000

# Run tests
pip install pytest
python -m pytest tests/ -v
```

## Pipeline Data Flow

```
User Prompt
  → route_prompt()          # Deterministic flag extraction (advice, legal, current_events, etc.)
  → perform_web_search()    # Tavily search if flags warrant it (optional)
  → GPT-1 (Generator)       # Produces response using Audit v7 + flag augmentation + search context
  → Activation bypass check  # Skip verification if output matches activation patterns
  → sanitize_output()       # Citation-aware: strip bare %, banned evidence, typicality; preserve cited stats
  → decompose_claims()      # Atomic claim decomposition (pre-GPT-2, best-effort)
  → verify_claims_with_nli() # Optional NLI pre-verification against evidence (if available)
  → GPT-2 (Verifier)        # Tripwire reference + task in user content (lost-in-middle fix)
                             # Chain-of-thought reasoning trace → claim table → T1-T7 findings → verdict
  → Decision tree:
      ├ PASS → return result
      ├ FAIL (soft-only) → auto-retry verification
      └ FAIL (hard/persistent) → GPT-3 (Arbiter)
          ├ BLOCK → return failure
          ├ ALLOW_WITH_EDITS → rewrite, re-verify (convergence-aware, max 3 loops)
          └ ALLOW_AS_UNKNOWN_ONLY → reframe claims, Low confidence
  → Final response with confidence scoring (hard-findings penalty)
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Embedded web chat UI |
| GET | `/health` | Health check (`{"status": "ok", "key_set": bool}`) |
| POST | `/api/pipeline` | Main verification pipeline (rate-limited) |
| POST | `/api/stress` | Stress testing (streaming NDJSON) |
| POST | `/api/openai/config` | Set OpenAI API key/model at runtime |
| GET | `/api/openai/config` | Get current OpenAI config (key redacted) |
| POST | `/api/tavily/config` | Set Tavily API key at runtime |
| GET | `/api/tavily/config` | Get current Tavily config |
| POST | `/api/tavily/toggle` | Enable/disable web search |
| POST | `/api/stage/config` | Set per-stage model config (provider/model/key) |
| GET | `/api/stage/config/{stage}` | Get stage config (gpt1/gpt2/gpt3) |
| POST | `/api/grounded/documents/{document_id}` | Version and index a private UTF-8 document |
| POST | `/api/grounded/query` | Run the fail-closed folder-grounded lane |

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | — | OpenAI API key (`sk-...`) |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | OpenAI model identifier |
| `TAVILY_API_KEY` | No | — | Tavily web search key (`tvly-...`) |
| `PORT` | No | `8000` | Server port |
| `RATE_LIMIT_PER_MINUTE` | No | `20` | Per-IP request rate limit |
| `PIPELINE_URL` | No | `http://localhost:8000` | Base URL for n8n workflows |
| `NLI_SERVICE_URL` | No | — | Optional remote NLI service URL |
| `ANTHROPIC_API_KEY` | No | — | Anthropic API key (if using Claude for any stage) |
| `KNOWLEDGE_ROOT` | No | `knowledge_data` | Fixed private corpus root for grounded mode |
| `KNOWLEDGE_API_TOKEN` | Grounded mode | — | Required bearer token for all grounded reads/writes |

## Running Tests

```bash
python -m pytest tests/ -v          # Run all 279 tests
python -m pytest tests/ -v -x       # Stop on first failure
python -m pytest tests/test_sanitizer.py -v   # Run specific module
```

All tests are deterministic (no LLM calls, no mocking needed). They verify business logic: routing, sanitization, JSON parsing, verdict computation, convergence detection, and arbiter decisions.

**Test fixtures** are centralized in `tests/conftest.py`. Tests use `@pytest.mark.parametrize` extensively.

## Key Conventions

### Code Style
- **Type hints** throughout — use `from __future__ import annotations` at top of every module
- **Pydantic models** for all request/response schemas (in `pipeline/models.py`)
- **snake_case** for functions and variables, **PascalCase** for classes
- **JSON responses** use snake_case keys
- Custom `PipelineError` exception (in `helpers.py`) with `status_code` and `detail`

### Architecture Patterns
- **Thread-safe config**: Runtime API key/model overrides use `threading.Lock()` in `config.py`
- **Deterministic routing**: All flag extraction and sanitization is regex-based, no randomness
- **Hardened JSON parsing**: `extract_json()` in `helpers.py` handles markdown fences, prose preambles, and truncation recovery
- **Flag-aware augmentation**: `build_augmentation()` in `prompts.py` modifies all 3 GPT system prompts based on routing flags
- **Citation-aware sanitizer**: `sanitize_output()` checks for nearby citations before stripping percentages — cited statistics are preserved
- **Split GPT-2 prompt**: Core instructions in system prompt, detailed tripwire reference injected at start of user content (avoids lost-in-middle attention problem)
- **Reasoning trace**: GPT-2 outputs a `reasoning_trace` array showing which text triggered each finding (chain-of-thought for verification accuracy)
- **Convergence detection**: `pipeline/convergence.py` monitors finding deltas across rewrite iterations, stopping on convergence/oscillation/regression
- **Streaming stress tests**: `/api/stress` returns NDJSON for real-time progress

### When Adding Pipeline Logic
- Routing flags go in `pipeline/sanitizer.py` → `route_prompt()`
- Prompt augmentations go in `pipeline/prompts.py` → `build_augmentation()`
- New tripwire violations must follow the T-code pattern (T1-T7) with HARD/SOFT severity
- Sanitizer rules are flag-aware — check `flags` dict before applying transforms
- Search integration touches GPT-1 (source context) and GPT-2 (claim grounding) prompts
- `parse_gpt2()` returns 5 values: `(claim_table, violations, verdict, findings, reasoning_trace)`
- `compute_confidence()` accepts optional `findings` parameter for hard-findings penalty

### When Adding Tests
- Add fixtures to `tests/conftest.py`
- Use `@pytest.mark.parametrize` for data-driven tests
- Test deterministic logic only — no LLM calls in unit tests
- Stress/integration tests use the `/api/stress` endpoint with `tests.json` test cases

### Hard-Coded Limits
- `MAX_REWRITE_LOOPS = 3` — arbiter rewrite re-verification cap (convergence detection handles early stopping)
- `MAX_PROMPT_LENGTH = 10000` — input prompt character limit
- Rate limit window: 60 seconds sliding window per IP

## Deployment

### Railway (Production)
- **Builder:** Nixpacks (not Docker)
- **Start:** `/opt/venv/bin/uvicorn app:app --host 0.0.0.0 --port $PORT`
- **Health check:** `GET /health` (120s timeout)
- **Config:** `railway.toml`

### Docker (Local)
```bash
docker build -f Dockerfile.local -t epistemic-pipeline .
docker run -p 8000:8000 --env-file .env epistemic-pipeline
```

### CI
GitHub Actions workflow (`.github/workflows/docker-image.yml`) builds a Docker image on pushes to `main` and on pull requests.

## Important Notes

- Never commit `.env` files — they contain API keys. Use `.env.example` as a template.
- The `ui.py` file contains a large embedded HTML string (~62KB) for the chat UI — avoid reformatting it.
- The Audit v7 framework prompts in `prompts.py` are carefully tuned — changes to system prompts should be validated with stress tests.
- Sanitizer rules and GPT-2 augmentation must stay in sync — if you change what the sanitizer strips, update GPT-2's awareness of those patterns.
- The GPT-2 prompt is split: core in `DEFAULT_GPT2_SYSTEM`, detailed tripwire reference in `GPT2_TRIPWIRE_REFERENCE` (injected into user content). Keep both in sync.
