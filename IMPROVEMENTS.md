# Epistemic Pipeline: Comprehensive Improvement Roadmap

> Research-backed analysis of usability, performance, and functionality improvements.
> Compiled from deep codebase exploration + state-of-the-art technology research (March 2026).

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current System Assessment](#2-current-system-assessment)
3. [UX & Usability Improvements](#3-ux--usability-improvements)
4. [Pipeline Architecture Improvements](#4-pipeline-architecture-improvements)
5. [Verification Logic Improvements](#5-verification-logic-improvements)
6. [Search & Evidence Grounding](#6-search--evidence-grounding)
7. [Performance & Cost Optimization](#7-performance--cost-optimization)
8. [Confidence Calibration](#8-confidence-calibration)
9. [Testing & Observability](#9-testing--observability)
10. [Deployment & Infrastructure](#10-deployment--infrastructure)
11. [Implementation Roadmap](#11-implementation-roadmap)

---

## 1. Executive Summary

The Epistemic Verification Pipeline is a technically sophisticated 3-stage LLM fact-checking system (Generator -> Verifier -> Arbiter) with 523 passing tests, 100 real-world stress test cases, multi-provider LLM support, NLI verification, and convergence-aware rewrite loops. The system works well internally but has a significant gap between **what the pipeline actually does** (7+ decision points, convergence detection, NLI scoring, meta-verification, source matching) and **what users understand it's doing**.

**Key findings:**

| Area | Strengths | Gaps |
|------|-----------|------|
| **UI/UX** | Professional dark-mode design, collapsible details, stress test UI | No streaming, time-based fake progress, confidence not explained, no conversation history |
| **Pipeline** | Robust 3-stage architecture, convergence detection, source matching | Sequential LLM calls (2-7 per request), no async execution, decomposer always runs |
| **Verification** | Audit v8 framework, T1-T7 tripwires, reasoning traces | GPT-2 conflicts with source rules, NLI underutilized, meta-verify only on PASS |
| **Search** | Tavily integration, authority scoring, flag-aware triggering | No fallback provider, authority ignores recency, zero-results silent fallback |
| **Performance** | Client caching, pre-compiled regex, connection pooling | No semantic caching, no response dedup, sequential best-of-N |
| **Testing** | 523 unit tests, 100 stress cases, 10 categories | No CI test execution, no integration tests, no latency/cost tracking in PSS |
| **Deployment** | Railway + Docker + Heroku, health checks, auto-restart | Single replica, no observability export, no secrets management |

---

## 2. Current System Assessment

### 2.1 What Users Experience Today

1. **Submit prompt** -> see time-based fake progress messages (1.5s -> 5s -> 10s -> 16s) that don't correspond to actual pipeline stages
2. **Wait 10-55 seconds** with no streaming, no cancellation option
3. **Receive final answer** with a confidence badge (High/Medium/Low) but no explanation of why
4. **Optionally expand** "Pipeline Details" to see raw claim tables, violations, arbiter decisions
5. **Hit rate limit** (20 req/min) with no warning beforehand

### 2.2 What the Pipeline Actually Does (Hidden from Users)

- **Routing flag extraction**: Deterministic regex analysis of prompt (advice, legal, current_events, etc.)
- **Web search with authority scoring**: Tavily search with domain-based authority ranking
- **Atomic claim decomposition**: Pre-GPT-2 splitting of response into verifiable claims
- **NLI pre-verification**: Evidence-grounding via DeBERTa-v3 (when available)
- **Source-match correction**: Post-hoc fix of GPT-2 verdicts using search evidence
- **Convergence detection**: Intelligent stopping of rewrite loops (oscillation, regression)
- **Meta-verification**: Cross-checking on high-stakes queries (legal, medical)
- **Best-of-N sampling**: Generation diversity (env var only, no UI)
- **Sanitization**: Deterministic output cleaning (bare stats, stale dates, banned patterns)

### 2.3 Response Data Utilization

The `PipelineResponse` has **39 fields** but only ~10 are displayed in the UI:

| Displayed | Not Displayed (but computed) |
|-----------|------------------------------|
| final_verdict, final_result, tier, confidence | atomic_claims, decomposition_ran |
| gpt1_output, gpt2_raw, claim_table, violations | meta_verification, sanitizer_applied |
| arbiter_decision, arbiter_rationale, arbiter_edits | prompt_flags, nli_grounding_rate |
| search_sources, search_performed | rewrite_convergence_reason |

---

## 3. UX & Usability Improvements

### 3.1 Real-Time Pipeline Progress (Critical)

**Problem**: Users wait 10-55 seconds seeing fake time-based messages. The current loading implementation in `ui.py` uses hardcoded timeouts that don't correspond to actual pipeline stages.

**Solution**: Server-Sent Events (SSE) for real-time progress.

**Backend change** (`api/routes.py` + `pipeline/orchestrator.py`):
- Add `?stream=true` parameter to `POST /api/pipeline`
- Emit structured NDJSON events as each stage completes (reuse the pattern already working in `/api/stress`)
- Event types: `stage_start`, `stage_complete`, `search_results`, `token` (for GPT-1 streaming), `verdict`, `result`

**Frontend change** (`api/ui.py`):
- Replace hardcoded timeout steps with SSE event listener
- Show a vertical stepper that lights up as stages complete:
  ```
  [done]    Analyzing prompt... (advice, current_events detected)
  [done]    Searching web... (3 sources found)
  [active]  Generating response...
  [ ]       Verifying claims...
  [ ]       Final review...
  ```
- Add elapsed time counter ("Verifying... 4.2s")
- Add cancel/abort button using `AbortController`

**Research backing**: SSE is the de facto standard for streaming LLM responses (2025-2026). Time-to-first-token < 300-700ms feels snappy. Nielsen Norman Group: "Visibility of system status" is the #1 usability heuristic.

### 3.2 Confidence Score Transparency

**Problem**: Users see "Confidence: High" but don't know why. The confidence computation in `orchestrator.py:77-180` uses multiple signals (claim categories, hard findings penalty, NLI grounding rate, meta-verification) but none are surfaced.

**Solution**: Add a confidence breakdown tooltip/expandable.

**Display format**:
```
Confidence: High
  - 92% of claims verified against evidence
  - 0 hard violations, 1 soft violation (T5: prescriptive creep)
  - Web-grounded: 3 authoritative sources (gov, edu)
  - NLI agreement: 87%
```

**Implementation**: Return `confidence_reasoning` as a structured field in `PipelineResponse` containing the signals that contributed to the score.

### 3.3 Progressive Disclosure (3-Layer Architecture)

**Problem**: Raw claim tables and JSON reasoning traces overwhelm non-technical users. Hiding everything loses trust.

**Solution**: Three disclosure layers:

| Layer | Content | Default State |
|-------|---------|---------------|
| **L1: Summary** | Final response + confidence badge + verdict + violation count | Always visible |
| **L2: Verification** | Which stages ran, claim count, violation summaries in plain English, source list | Expandable (auto-expand on FAIL) |
| **L3: Full Trace** | Raw claim table, reasoning trace JSON, convergence history, finding deltas | Hidden, accessible via "Advanced" link |

**Research backing**: Progressive disclosure is established as essential for AI interfaces. ScienceDirect confirms on-demand disclosure helps users follow AI reasoning without overload. Maximum 2-3 layers recommended.

### 3.4 Conversation History

**Problem**: The UI is single-session. Users can't reference previous verifications or compare results.

**Solution**: Client-side `localStorage` persistence (no backend changes needed).

- Store conversations as `{id, title, created_at, messages[]}`
- Auto-generate titles from first 60 chars of user prompt
- Add collapsible left sidebar with conversation list (hidden on mobile)
- Search/filter via `Fuse.js` for fuzzy matching
- Limit to 100 conversations with LRU eviction
- Export as Markdown or JSON

### 3.5 Accessibility Improvements

**Current gaps**: No ARIA labels on spinners, no `aria-live` on message container, color-only status indicators, no skip-to-content link, no `prefers-reduced-motion` respect.

**Priority fixes**:
```html
<div id="messages" role="log" aria-live="polite" aria-relevant="additions">
```
- Add `aria-label` to all interactive elements
- Pair color indicators with icons + text labels (not color alone)
- Add `@media (prefers-reduced-motion: reduce)` CSS
- Add skip link: "Skip to chat input"
- Ensure 4.5:1 contrast ratio (current dark theme variables should be tested)

### 3.6 Mobile Optimization

**Current**: Single breakpoint at 640px, 2-col grid on mobile.

**Improvements**:
- Add `env(safe-area-inset-bottom)` padding for iPhone home bar
- Use bottom-sheet pattern for settings on mobile (instead of side drawer)
- Make verification details a full-screen modal on mobile (not inline accordion)
- Ensure all touch targets are minimum 44x44px
- Test with iOS Safari keyboard behavior (fixed input bar)

### 3.7 Light/Dark Mode Toggle

**Current**: Dark-only by design.

**Addition**: Define light-mode CSS variables under `[data-theme="light"]`, add toggle in top bar, respect `prefers-color-scheme` as default, store preference in `localStorage`.

### 3.8 Error Handling & Rate Limiting UX

**Current issues**:
- Rate limit (20 req/min) hits with no warning; bare "HTTP 429" shown
- Pipeline timeout shows "HTTP 504" with no guidance on which stage failed
- No retry mechanism for main pipeline (stress test has auto-resume)

**Improvements**:
- Show "Requests remaining: 15/20" counter in UI
- On timeout, include which stage was running when it timed out
- Add auto-retry with exponential backoff for transient failures
- Provide actionable recovery suggestions ("Try a simpler query" / "Check your API key")

### 3.9 User Feedback Loop

**Current**: `/api/feedback` endpoint exists but no UI affordance to trigger it.

**Addition**: Add "Was this verification correct?" thumbs-up/down button below each response. On click, `POST /api/feedback` with verdict accuracy rating. This enables quality tracking and pipeline improvement over time.

---

## 4. Pipeline Architecture Improvements

### 4.1 Async/Concurrent Execution

**Problem**: All LLM calls are sequential via `call_llm()`. A single failing request can hit 7+ API calls, each waiting 3-30 seconds. Total latency can exceed 2-3 minutes.

**Solution**: Use `asyncio` for parallelizable operations:

| Operation | Current | Improved |
|-----------|---------|----------|
| Best-of-N generation | Sequential N calls | `asyncio.gather()` N calls |
| Decomposition + NLI | Sequential | Parallel after GPT-1 output |
| Multi-source search | Single Tavily call | Parallel Tavily + Brave |
| Rewrite + re-verify | Sequential per loop | GPT-1 rewrite can overlap with convergence check |

**Estimated impact**: 30-40% latency reduction on multi-call requests.

### 4.2 Streaming Pipeline Response

**Problem**: API waits for entire pipeline before returning. Users see nothing for 10-55 seconds.

**Solution**: Extend the existing NDJSON streaming pattern from `/api/stress` to `/api/pipeline`.

```python
# In routes.py
@router.post("/api/pipeline")
async def pipeline_endpoint(request: PipelineRequest):
    if request.stream:
        return StreamingResponse(
            run_pipeline_streaming(request),
            media_type="application/x-ndjson"
        )
    return run_pipeline(request)  # existing behavior
```

Emit events for: routing complete, search results, GPT-1 output (token-by-token), GPT-2 verdict, arbiter decision, rewrite progress.

### 4.3 Lazy Decomposition

**Problem**: `decompose_claims()` runs on every request even when NLI is disabled, adding 3-5 seconds.

**Solution**: Only decompose when:
1. NLI service is available, OR
2. GPT-2 returns ambiguous findings (soft violations only), OR
3. Tier is "strict"

**Estimated impact**: 10-15% faster on standard/light tier without NLI.

### 4.4 Rewrite Loop Feedback Injection

**Problem**: Rewrite loops re-verify with GPT-2 but don't tell GPT-1 what specifically to fix. GPT-2's findings from the previous iteration are discarded.

**Solution**: Inject previous GPT-2 findings into the GPT-1 rewrite prompt:
```
Previous verification found these issues:
- T1: "73% of startups fail" — fabricated statistic, no source
- T3: "caffeine causes cancer" — causal claim stated as fact
Please rewrite to address these specific findings.
```

**Estimated impact**: Reduce average rewrite loops by ~20% (1.5 -> 1.2 iterations).

### 4.5 Early Exit for High-Confidence Responses

**Problem**: Every response goes through full GPT-2 verification, even when GPT-1 output is obviously factual with 80%+ citations.

**Solution**: Staged verification:
1. **Fast path**: If GPT-1 output has >80% cited claims and no routing flags for high-stakes topics, skip full GPT-2 and use lightweight heuristic check
2. **Standard path**: Full GPT-2 verification (current behavior)
3. **Strict path**: Full GPT-2 + meta-verification (already implemented for high-stakes)

**Estimated savings**: 30-40% faster on high-confidence responses.

### 4.6 Replace call_llm() with LiteLLM

**Problem**: Custom provider dispatch code in `helpers.py` handles OpenAI and Anthropic. Adding new providers requires code changes.

**Solution**: Replace `call_llm()` with [LiteLLM](https://github.com/BerriAI/litellm) (`pip install litellm`):
- Unified API for 100+ LLM providers
- OpenAI-compatible format (drop-in replacement)
- Automatic retries and fallbacks
- Built-in cost tracking per call
- 8ms P95 latency overhead at 1k RPS

**Research backing**: Organizations using single-provider LLMs overpay 40-85% vs. intelligent routing. LiteLLM is the leading open-source solution (2025-2026).

---

## 5. Verification Logic Improvements

### 5.1 Fix GPT-2 Source-Match Conflict

**Problem**: GPT-2 receives competing instructions: "sources are correct" AND "detect fabricated stats (T1)". Result: `gpt-4o-mini` frequently flags source-backed claims as T1, requiring post-hoc correction via `source_match.py`.

**Solution**: Split verification into two phases:
1. **Content matching**: Check if claims align with provided sources (factual grounding)
2. **Tripwire checking**: Check for epistemic violations (fabrication, prescriptive creep, etc.)

This eliminates the instruction conflict and reduces arbiter invocations by an estimated 40%.

### 5.2 Make NLI Binding for High-Stakes Queries

**Problem**: NLI signals are collected but GPT-2 can override them. An NLI "strong_contradiction" should hard-fail the claim, but GPT-2 might still categorize it as "Observed".

**Solution**: For high-stakes queries (`legal_mode=true`, `advice_requested=true`):
- NLI `strong_contradiction` -> force claim to "Unsupported" regardless of GPT-2
- NLI `strong_entailment` from `.gov`/`.edu` source -> force claim to "Observed"
- Non-high-stakes: keep current behavior (NLI as advisory signal)

**Research backing**: Fine-tuned DeBERTa-v3 achieves 88% F1 on fact verification (VerifAI), outperforming GPT-4 zero-shot by 7% for domain-specific claims.

### 5.3 Add Meta-Verify for False FAILs

**Problem**: `meta_verify_pass()` catches false PASSes, but there's no `meta_verify_fail()` for false FAILs. GPT-2 can over-flag valid inferences as T1/T3.

**Solution**: Add `meta_verify_fail()` that checks:
- If >50% of "Unsupported" claims are common knowledge or valid inferences
- If hard findings target hedged language that already includes qualifiers
- If the claim is supported by the search evidence (source-match already does this partially)

### 5.4 Validate Decomposition Quality

**Problem**: `check_decomposition_quality()` in `decomposer.py` computes metrics but is never called in the orchestrator. Bad decompositions (compound claims, missing coverage) silently pass to GPT-2.

**Solution**: Call `check_decomposition_quality()` in orchestrator after decomposition. Require `quality_tier >= "acceptable"` or re-decompose with refined prompt.

### 5.5 Add Decontextualization Before Decomposition

**Problem**: Atomic claims may contain unresolved pronouns ("They reported..." — who is "they"?).

**Solution**: Add an explicit pronoun/reference resolution step before decomposition so each claim is self-contained. Research (EMNLP 2025) shows decontextualization before decomposition prevents information loss and improves verification accuracy.

### 5.6 Add Checkworthiness Filtering

**Problem**: Every atomic claim goes through full GPT-2 verification, including trivial ones ("The sky is blue").

**Solution**: Add a lightweight checkworthiness filter (as in the Loki fact-checking system) that skips verification for:
- Tautologies and definitions
- Claims the user themselves provided (user-provided category)
- Well-known common knowledge

**Estimated savings**: 10-20% fewer claims sent to GPT-2.

### 5.7 Improve Convergence Detection

**Problem**: Convergence detection is too conservative. If findings oscillate between {T1, T2} and {T1, T3}, the loop stops even though the hard count might be decreasing.

**Solution**: Allow oscillation up to 2 iterations if hard finding count is strictly decreasing. Only stop on:
- Hard count unchanged after 2 iterations
- Hard count increased (regression)
- Max iterations reached (3)

---

## 6. Search & Evidence Grounding

### 6.1 Add Brave Search API as Fallback Provider

**Problem**: Single search provider (Tavily). If Tavily is down or returns 0 results, the pipeline silently falls back to ungrounded generation.

**Solution**: Add [Brave LLM Context API](https://brave.com/blog/most-powerful-search-api-for-ai/) as secondary provider:
- Pre-chunked, LLM-optimized results
- $5/1k requests (potentially cheaper than Tavily)
- 30 billion page index, 100M daily updates
- P90 latency under 600ms

Configure in `config.py` with `BRAVE_API_KEY` env var. In `search.py`, try Tavily first, fall back to Brave, combine results for high-stakes queries.

### 6.2 Authority Scoring Should Include Recency

**Problem**: `compute_source_authority()` in `search.py` uses domain-based scoring only (gov/edu = 1.0). A 10-year-old CDC page scores identically to a current one.

**Solution**: Add temporal signal:
```python
authority = domain_score * recency_factor
# recency_factor: 1.0 for <1 year, 0.8 for 1-3 years, 0.6 for 3-5 years, 0.4 for >5 years
```
Use `publication_date` from search results when available (Tavily includes this).

### 6.3 Validate Search Results for Query Relevance

**Problem**: Top 5 Tavily results are accepted without checking if they actually answer the query. A news article mentioning the topic tangentially gets treated as evidence.

**Solution**: Use NLI to validate entailment: "Does this source support the query intent?" before including in evidence. Filter out sources with NLI score < 0.3.

### 6.4 Fix Zero-Results Silent Fallback

**Problem**: If Tavily returns 0 results, `search_performed` is set to `False` and the pipeline proceeds without search. Users who enabled search get no indication it failed.

**Solution**: Set `search_performed = True` with `search_note = "No relevant sources found"`. Display this in the UI. Optionally suggest: "Would you like me to try with different search terms?"

### 6.5 Iterative Retrieval for Insufficient Evidence

**Problem**: Single-shot search. If initial results don't contain enough evidence, GPT-2 flags claims as "Unsupported" when better evidence might exist.

**Solution**: Implement adaptive retrieval (FIRE/PCC pattern):
1. Run initial search
2. If GPT-2 finds >30% "Unsupported" claims, trigger follow-up search with refined queries
3. Re-verify with additional evidence before proceeding to arbiter

---

## 7. Performance & Cost Optimization

### 7.1 Semantic Caching for LLM Calls

**Problem**: No deduplication of identical/similar prompts. Repeated queries (stress tests, retries, similar questions) make full LLM calls every time.

**Solution**: Implement semantic caching using [GPTCache](https://github.com/zilliztech/GPTCache) or Redis with vector search:

| Cache Layer | Target | TTL | Similarity Threshold |
|-------------|--------|-----|---------------------|
| **Exact match** (in-memory) | Identical prompt + flags | 5 min | 1.0 |
| **Semantic match** (vector) | Similar prompts | 15 min | 0.92 |
| **Claim verification** | Same atomic claim | 1 hour | 0.95 |

**Important**: Use shorter TTLs for Verifier and Arbiter caches since facts change. Never cache with search results included (search is inherently time-sensitive).

**Research backing**: Semantic caching delivers 40-80% cost reduction and 250x latency improvement (25ms vs 7s) with 80-90% cache hit rates in production.

### 7.2 Per-Stage Caching Policy

- **GPT-1 (Generator)**: Cache by `hash(prompt + flags + search_context)`. Longer TTL (15 min). Safe to reuse.
- **GPT-2 (Verifier)**: Cache by `hash(gpt1_output + tier + search_context)`. Short TTL (5 min). Sensitive to context changes.
- **GPT-3 (Arbiter)**: Minimal caching. Only exact-match on identical inputs. Short TTL (2 min).

### 7.3 Cost Tracking per Request

**Problem**: No visibility into API cost per pipeline request. Can't answer: "Is the rewrite loop worth the extra cost?"

**Solution**: Track token usage and estimated cost per stage. Return in response metadata and integrate into stress test PSS scoring.

### 7.4 Connection Pooling Enhancement

**Current**: Client caching by `(provider, api_key, base_url)` in `helpers.py` (recently added).

**Additional**: Configure HTTP/2 connection pooling with `httpx` for better multiplexing. Set `max_connections=20` and `max_keepalive_connections=10` to handle concurrent requests efficiently.

---

## 8. Confidence Calibration

### 8.1 Dual Calibration (Evidence + Reasoning)

**Problem**: `compute_confidence()` blends all signals equally. A response grounded by CDC data (authority 1.0) scores the same as one grounded by a random blog (authority 0.3).

**Solution**: Implement DoublyCal-style dual calibration:
1. **Evidence confidence**: `sum(source_authorities * source_relevance) / len(sources)`
2. **Reasoning confidence**: GPT-2 claim category distribution + hard findings penalty
3. **Combined**: `0.4 * evidence_confidence + 0.6 * reasoning_confidence`

### 8.2 Weight Hard Findings by Violation Type

**Problem**: Any single hard finding drops confidence one tier. But T1 (fabrication) is far more serious than T7 (unverified current fact).

**Solution**: Weighted penalties:
```
T1 (fabrication) = 2.0x penalty
T2 (unsupported reference) = 1.5x
T3 (causal as fact) = 1.5x
T4 (missing qualifier) = 1.0x
T5 (prescriptive creep) = 1.0x
T6 (reassurance framing) = 0.75x
T7 (unverified current) = 0.75x
```

### 8.3 Post-Hoc Calibration via Stress Tests

**Problem**: Confidence labels may be systematically over- or under-confident.

**Solution**: Apply isotonic regression calibration trained on stress test results:
1. Run stress tests, record (predicted_confidence, actual_accuracy) pairs
2. Fit isotonic regression model
3. Apply as post-hoc correction in `compute_confidence()`

**Research backing**: Reinforcement learning with proper scoring rules (Yaldiz 2026) reduces Expected Calibration Error by up to 9 points. SteerConf provides training-free calibration via multi-direction prompting.

### 8.4 Add Flex-ECE to PSS

**Problem**: PSS (Pipeline Stability Score) doesn't measure calibration quality.

**Solution**: Add Flex-ECE (flexible Expected Calibration Error) metric to stress testing. This captures partial correctness better than standard ECE.

---

## 9. Testing & Observability

### 9.1 Add Tests to CI

**Problem**: GitHub Actions only builds Docker image; doesn't run pytest.

**Solution**:
```yaml
steps:
  - uses: actions/checkout@v4
  - uses: actions/setup-python@v5
    with:
      python-version: '3.11'
  - run: pip install -r requirements.txt
  - run: python -m pytest tests/ -v --tb=short
  - run: ruff check .
  - run: docker build -f Dockerfile.local -t epistemic-pipeline .
```

### 9.2 Integration Tests with FastAPI TestClient

**Problem**: All 523 tests are unit-level. No tests for API routes, middleware, error handling, or rate limiting.

**Solution**: Add `tests/test_routes.py` using FastAPI `TestClient`:
- Test rate limiting enforcement (hit limit, verify 429)
- Test ADMIN_TOKEN validation
- Test malformed request handling
- Test CORS enforcement
- Test streaming stress endpoint

### 9.3 Per-Stage Metrics in Stress Testing

**Problem**: PSS tracks final results but not per-stage quality.

**Solution**: Add to stress output:
- `gpt1_hallucination_rate`: How often does GPT-1 introduce ungrounded claims?
- `gpt2_precision`: Of claims GPT-2 flags, how many are actually wrong?
- `gpt2_recall`: Of actual issues, how many does GPT-2 catch?
- `gpt3_override_rate`: How often does arbiter override GPT-2?
- `latency_p95`, `latency_p99`: Per-stage timing percentiles
- `cost_per_request`: Estimated API cost

### 9.4 Deterministic Stress Test Runs

**Problem**: LLM outputs aren't deterministic. Running the same test twice may yield different verdicts.

**Solution**: Run K >= 3 iterations per test case and compute failure rates with confidence intervals. Report variance, not just pass/fail.

### 9.5 Observability Export

**Problem**: Metrics collected in `pipeline/metrics.py` but not exported to monitoring systems.

**Solution**: Add [Prometheus](https://github.com/prometheus/client_python) metrics export:
```python
from prometheus_client import Histogram, Counter, Gauge

pipeline_latency = Histogram('pipeline_latency_seconds', 'Pipeline latency', ['stage'])
verdict_count = Counter('verdict_total', 'Verdict counts', ['verdict', 'tier'])
confidence_gauge = Gauge('confidence_score', 'Confidence distribution', ['label'])
```

Expose at `GET /metrics` in Prometheus format.

---

## 10. Deployment & Infrastructure

### 10.1 High Availability

**Problem**: Single Railway replica. Single point of failure.

**Solution**: `numReplicas = 2` in `railway.toml`. Ensure stateless design (already the case — no persistent state).

### 10.2 Secrets Management

**Problem**: API keys in plaintext `.env`.

**Solution**: For production, use Railway environment variables (already supported) or a secrets manager. Add documentation for secure key rotation.

### 10.3 Database Layer for Persistence

**Problem**: Feedback, metrics, and conversation history are all in-memory. Lost on restart.

**Solution**: Add optional SQLite persistence (zero-dependency for local) or PostgreSQL for production:
- Feedback entries
- Aggregate metrics over time
- Stress test history
- (Optional) Semantic cache store

### 10.4 Request Correlation IDs

**Problem**: No way to trace a request across pipeline stages in logs.

**Solution**: Generate UUID per request, pass through all stages, include in logs and response headers (`X-Request-ID`).

---

## 11. Implementation Roadmap

### Phase 1: Quick Wins (1-2 weeks)

| # | Improvement | Effort | Impact | Section |
|---|------------|--------|--------|---------|
| 1 | Add tests to CI (pytest + ruff) | Low | High | 9.1 |
| 2 | Fix zero-results silent fallback | Low | Medium | 6.4 |
| 3 | Add confidence reasoning to response | Low | High | 3.2 |
| 4 | Add user feedback button in UI | Low | Medium | 3.9 |
| 5 | Add ARIA labels and live regions | Low | Medium | 3.5 |
| 6 | Rate limit counter in UI | Low | Medium | 3.8 |
| 7 | Call decomposition quality check | Low | Medium | 5.4 |

### Phase 2: Core UX Improvements (2-4 weeks)

| # | Improvement | Effort | Impact | Section |
|---|------------|--------|--------|---------|
| 8 | SSE streaming for pipeline progress | Medium | Very High | 3.1, 4.2 |
| 9 | Progressive disclosure (3-layer) | Medium | High | 3.3 |
| 10 | Conversation history (localStorage) | Medium | High | 3.4 |
| 11 | Lazy decomposition | Low | Medium | 4.3 |
| 12 | Rewrite loop feedback injection | Medium | Medium | 4.4 |
| 13 | Light/dark mode toggle | Low | Medium | 3.7 |

### Phase 3: Pipeline Intelligence (3-6 weeks)

| # | Improvement | Effort | Impact | Section |
|---|------------|--------|--------|---------|
| 14 | Async/concurrent LLM execution | High | High | 4.1 |
| 15 | Semantic caching (GPTCache/Redis) | Medium | Very High | 7.1 |
| 16 | Brave Search fallback provider | Medium | High | 6.1 |
| 17 | Split GPT-2 verification phases | High | High | 5.1 |
| 18 | NLI binding for high-stakes | Medium | High | 5.2 |
| 19 | Dual confidence calibration | Medium | High | 8.1 |
| 20 | Replace call_llm with LiteLLM | Medium | High | 4.6 |

### Phase 4: Advanced Features (1-3 months)

| # | Improvement | Effort | Impact | Section |
|---|------------|--------|--------|---------|
| 21 | Integration tests (TestClient) | Medium | High | 9.2 |
| 22 | Per-stage stress test metrics | Medium | High | 9.3 |
| 23 | Prometheus observability export | Medium | Medium | 9.5 |
| 24 | Post-hoc confidence calibration | High | High | 8.3 |
| 25 | Iterative retrieval for evidence | High | High | 6.5 |
| 26 | Meta-verify for false FAILs | Medium | Medium | 5.3 |
| 27 | Checkworthiness filtering | Medium | Medium | 5.6 |
| 28 | Decontextualization step | Medium | Medium | 5.5 |
| 29 | Database persistence layer | High | Medium | 10.3 |
| 30 | Mobile bottom-sheet + a11y audit | Medium | Medium | 3.6 |

---

## Research Sources

### Fact-Checking Pipelines
- ClaimCheck (2025): 76.4% accuracy with Qwen3-4B via modular design
- FIRE (NAACL 2025): Iterative retrieval with adaptive confidence gating
- PCC (2026): Joint internal certainty + reasoning consistency
- OpenFactCheck: Unified Python framework for modular fact-checking
- Loki: Open-source tool with 5-step parallelized pipeline

### NLI & Verification
- VerifAI: 88% F1 with fine-tuned DeBERTa, outperforming GPT-4 zero-shot by 7%
- Atomic-SNLI (2026): Atomic fact decomposition improves NLI classification
- DYDECOMP (ACL 2025): RL-learned decomposition policies for verifier-aware splitting

### Streaming & UX
- SSE is the de facto standard for LLM response streaming (2025-2026)
- Progressive disclosure: Max 2-3 layers for AI interfaces
- Trust indicators: Confidence visualization, inline citations, "I Don't Know" pattern
- Accessibility: ARIA live regions, keyboard navigation, prefers-reduced-motion

### Caching & Performance
- Semantic caching: 40-80% cost reduction, 250x latency improvement
- GPTCache: Open-source semantic cache for LLM calls
- LiteLLM: Unified multi-provider routing with 8ms overhead

### Confidence Calibration
- DoublyCal: Dual evidence/reasoning calibration
- SteerConf: Training-free multi-direction confidence steering
- Flex-ECE: More nuanced calibration metric than standard ECE

### Search & RAG
- Brave LLM Context API: Pre-chunked LLM-optimized results, $5/1k requests
- Perplexity Sonar: Fastest grounded answers for latency-sensitive paths
- Hybrid retrieval (dense + BM25): Standard for RAG in 2025-2026
- Google Check Grounding API: 0-to-1 support score at <500ms latency
