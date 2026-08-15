# Epistemic Pipeline Hardening: E2E Test Suite Ready (TEST_READY)

## Overview
The comprehensive, requirement-driven E2E Hardening Test Suite has been created in:
`tests/test_e2e_hardening.py`

This suite adheres strictly to opaque-box specification testing derived from `ORIGINAL_REQUEST.md`, `PROJECT.md`, and `TEST_INFRA.md`, covering Tiers 1 through 4 across all 6 hardening requirements (R1–R6, 9 distinct features).

## Test Suite Structure & Distribution

| Tier | Scope | Feature Coverage | Test Count | Status |
|---|---|---|:---:|:---:|
| **Tier 1** | Feature Coverage Isolation | F1: Polarity & Negation Grounding (R1)<br>F2: Prompt Injection Synonym Matrix (R2)<br>F3: Base64 Wrapper Decoder (R2)<br>F4: AST Proposition Span Grounding (R3)<br>F5: Multi-Citation Entity Isolation (R4)<br>F6: Uncited Quantitative Scanner (R5)<br>F7: Unbacked Authority Scanner (R5)<br>F8: Sanitizer Whitespace Boundary (R6)<br>F9: Sanitizer Authority Vocabulary (R6) | **45** (5/feat) | Collected & Validated |
| **Tier 2** | Boundary & Corner Cases | Subtle negations, prefix words, double negations, casing/whitespace evasion, corrupt base64, deep AST nesting, excision promotion, reversed citation groups `[2, 1]`, zero deltas, unbracketed currency/scale formats, adverb-split authority, punctuation cleanup | **45** (5/feat) | Collected & Validated |
| **Tier 3** | Cross-Feature Pairwise Interactions | Combinations across R1–R6 (e.g. R1+R3, R1+R4, R2+R3, R2+R5, R2+R6, R3+R4, R3+R5, R3+R6, R4+R5, R4+R6, R5+R6, R1+R5, R2+R4, R1+R6, R3+R5+R6, R2+R3+R4) | **16** | Collected & Validated |
| **Tier 4** | Real-World Application Scenarios | 1. Medical Claims with Negation & Dual-Entity Clinical Trials<br>2. Prompt Injection Obfuscation via Base64 in Regulatory Reports<br>3. Multi-Clause Scientific Paper with Ungrounded Authority & Swapped Stats<br>4. Financial Earnings Report with Uncited Figures & Bare Percentages<br>5. Healthcare Policy Synthesis with Complete Pipeline Flow | **5** | Collected & Validated |
| **Total** | Full Hardening Suite | Comprehensive coverage across R1–R6 | **111** | **100% Collectable** |

## Execution Commands
- **Collection Verification**:
  ```bash
  .venv/bin/pytest tests/test_e2e_hardening.py --collect-only
  ```
  *Result*: 111 tests collected in 0.07s.

- **Suite Execution**:
  ```bash
  .venv/bin/pytest tests/test_e2e_hardening.py -v
  ```

- **All Tests Verification**:
  ```bash
  .venv/bin/pytest tests/ -v
  ```

## Initial TDD / Red Phase Baseline
- **Collected**: 111 tests
- **Passing**: 39 tests (baseline legacy capabilities)
- **Failing**: 72 tests (expected red phase prior to M1–M6 hardening feature implementation)

All test cases are fully self-contained, isolated, and ready to guide implementation and regression testing.
