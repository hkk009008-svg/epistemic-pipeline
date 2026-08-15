# Project: Epistemic Pipeline Architectural Controls

## Architecture
The Epistemic Pipeline is a multi-stage grounded RAG verification system with 4 new architectural controls:
- `pipeline/arbiter.py`: Adaptive Poisoning Guard (`check_poisoning_threshold`, `guard_arbiter_decision`) to prevent string surgery on heavily corrupted text.
- `pipeline/stages.py`: 
  - `stage_verify`: Deterministic Pre-Flight Token & Citation Bounds Scanner running in <10ms, short-circuiting LLM 2.
  - `stage_arbiter`: Routes heavily poisoned drafts (`BLOCK`) directly to iterative repair.
  - `stage_rewrite_loop`: Closed-Loop Negative-Constraint Feedback with monotonic constraint accumulation across turns and $\le 2$-turn convergence.
- `pipeline/source_match.py`: Deterministic citation verification with out-of-bounds and zero-source handling.
- `pipeline/prompts.py`: Clause-isolated generation schema for atomic sentence generation.
- `pipeline/sanitizer.py`: Grammatical post-processing to clean orphaned coordinators (*whereas, and, but*).

## Feature Inventory
| # | Feature | Description | Milestone | Source | Status |
|---|---------|-------------|-----------|--------|--------|
| 1 | Arbiter Poisoning Ratio & Hard Violation Calculation | Compute $R_{\text{unsupported}} = N_{\text{unsupported}} / N_{\text{total}}$ and $N_{\text{hard}}$ from findings | M1 | survey | DONE |
| 2 | Adaptive Decision Guard | Route $>35\%$ unsupported or $\ge 2$ hard to `BLOCK`; $\le 35\%$ and $< 2$ hard to `ALLOW_WITH_EDITS` | M1 | survey | DONE |
| 3 | Fast Pre-Flight Citation & Number Scanner | Deterministic scan (<10ms) at `stage_verify` entrypoint | M2 | survey | DONE |
| 4 | LLM 2 Short-Circuiting on Pre-Flight Violations | Short-circuit Blind Verifier LLM call on hard pre-flight violations | M2 | survey | DONE |
| 5 | Zero-Source Grounding Edge Case | Catch `[N]` citations when `len(sources) == 0` | M2 | survey | DONE |
| 6 | Clause-Isolated Generation Prompt Schema | Prompt GPT-1 with atomic proposition constraints | M2 | survey | DONE |
| 7 | Conjunction & Punctuation Cleanup | Sanitize orphaned coordinators and trailing conjunctions | M2 | survey | DONE |
| 8 | Negative Constraint Extractor | Transform findings and edits into explicit `DO NOT ...` constraints | M3 | survey | DONE |
| 9 | Monotonic Constraint Accumulator | Accumulate constraints in `PipelineState["negative_constraints"]` across repair turns | M3 | survey | DONE |
| 10 | Rewrite Prompt Negative Constraints Injection | Inject `### Negative Constraints` in Turn 1 and Turn 2 retry prompts | M3 | survey | DONE |
| 11 | Two-Turn Convergence Guarantee | Fast repair convergence in $\le 2$ turns with fallback | M3 | survey | DONE |
| 12 | Comprehensive E2E Test Suite | 4-Tier requirement-driven opaque-box test suite for all 4 controls | M_E2E | survey | DONE |
| 13 | Final Integration & Adversarial Hardening | Pass 100% E2E tests + white-box adversarial verification | M4 | survey | DONE |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M_E2E | E2E Testing Track | `TEST_INFRA.md`, 4-tier E2E test suite (`tests/test_architectural_controls_e2e.py`), publish `TEST_READY.md` | none | DONE |
| M1 | Adaptive Poisoning Threshold in Arbiter | `pipeline/arbiter.py`, `pipeline/stages.py:stage_arbiter`, `tests/test_arbiter.py` | none | DONE |
| M2 | Deterministic Pre-Flight Scanner & Clause Schema | `pipeline/stages.py:stage_verify`, `pipeline/source_match.py`, `pipeline/prompts.py`, `pipeline/sanitizer.py`, `tests/test_source_match.py`, `tests/test_prompts.py`, `tests/test_sanitizer.py` | none | DONE |
| M3 | Closed-Loop Negative Constraints & Repair Loop | `pipeline/stages.py:stage_rewrite_loop`, `pipeline/arbiter.py`, `pipeline/convergence.py`, `tests/test_stages.py`, `tests/test_convergence.py` | M1, M2 | DONE |
| M4 | Final Milestone: 100% E2E Pass & Adversarial Hardening | Full pipeline E2E verification, adversarial coverage testing, all 1,101 tests clean | M1, M2, M3, M_E2E | DONE |

## Gate Result
**PASS** — Unanimously approved by Reviewer 1 (APPROVE), Reviewer 2 (APPROVE), Challenger 1 (APPROVE), Challenger 2 (APPROVE), and Forensic Auditor (CLEAN). Total 1,101 / 1,101 tests pass cleanly in 2.68s.
