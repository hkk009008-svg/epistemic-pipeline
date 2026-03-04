# Changelog

All notable changes to the Epistemic Verification Pipeline are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [3.0.0] - 2025-03-04

### Added
- **Benchmark harness** (`benchmarks/`) — reproducible factuality evaluation with claim-level scoring, baseline comparison, and cost/latency tracking.
- **SECURITY.md** — vulnerability reporting policy and security design documentation.
- **CONTRIBUTING.md** — contributor guide with development setup, testing, and architecture notes.
- **CODE_OF_CONDUCT.md** — Contributor Covenant v2.1.
- **GitHub issue/PR templates** — structured bug reports, feature requests, and PR checklists.
- **CodeQL scanning** — automated code security analysis on push/PR.
- **OpenSSF Scorecard** — automated supply chain security checks.
- **Dependabot** — automated dependency and GitHub Actions updates.
- **CHANGELOG.md** — this file.

### Changed
- Bumped version to 3.0.0 in `app.py`.

## [2.0.0] - 2025-03-04

### Added
- **True streaming** — NDJSON stage events from orchestrator via thread-safe queue (replaces fake progress ticks).
- **V2 API** — `/v2/pipeline` with verdict labels, `/v2/admin/config`, `/v2/public/capabilities`.
- **Evidence semantics** — NLI verification layer, source-aware claim recategorization, unsupported span detection.
- **Security hardening** — SSRF protection on `base_url`, admin token auth, CORS/HSTS/security headers, input sanitization.
- **Confidence breakdown** — evidence-tier scoring with grounding rate and confidence reasoning.
- **Convergence detection** — finding-delta tracking, oscillation/regression detection for rewrite loops.
- **Atomic claim decomposition** — pre-GPT-2 claim splitting for granular verification.
- **Multi-provider support** — per-stage provider/model/key config (OpenAI, Anthropic, OpenRouter, Ollama).
- **Pipeline metrics** — aggregate run statistics via `/api/metrics`.
- **User feedback** — `/api/feedback` endpoint for rating pipeline accuracy.
- **Meta-verification** — high-stakes cross-check for PASS/FAIL verdicts.
- **Stress test improvements** — heartbeat keepalives, resume from index, retry on transient errors.
- **Rate limit info** — `/api/rate-limit` endpoint.

## [1.0.0] - 2024-12-01

### Added
- Initial 3-stage pipeline: Generator (GPT-1) → Verifier (GPT-2) → Arbiter (GPT-3).
- Audit v7 epistemic framework (V1-V7 priority stack, G1-G12 global rules, T1-T7 tripwire violations).
- Tavily web search integration.
- Deterministic routing and sanitization.
- Embedded web chat UI.
- 100 real-world test cases across 9 categories.
- Pipeline Stability Score (PSS) computation.
- Docker and Railway deployment support.
