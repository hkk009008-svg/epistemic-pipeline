# E2E Test Infra: Epistemic Pipeline Hardening

## Test Philosophy
- Opaque-box, requirement-driven. No dependency on implementation design.
- Methodology: Category-Partition + Boundary Value Analysis (BVA) + Pairwise Combinatorial + Real-World Workload Testing.

## Feature Inventory
| # | Feature | Source (requirement) | Tier 1 | Tier 2 | Tier 3 | Tier 4 |
|---|---------|---------------------|:------:|:------:|:------:|:------:|
| 1 | F1: Pre-Flight Injection Interception | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ | ✓ |
| 2 | F2: Dual-Target Preflight (Prompt & Draft) | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ | ✓ |
| 3 | F4: Subordinate Clause AST Disentangling | ORIGINAL_REQUEST §R2 | 5 | 5 | ✓ | ✓ |
| 4 | F5: AST-Aware Grammar Reconstruction & Excision | ORIGINAL_REQUEST §R2 | 5 | 5 | ✓ | ✓ |
| 5 | F6: Worktree Sandbox Event Locking & 4-Tier Fallback | ORIGINAL_REQUEST §R3 | 5 | 5 | ✓ | ✓ |
| 6 | F7: Cross-Mount Storage Resilience (`os.link` fallback) | ORIGINAL_REQUEST §R3 | 5 | 5 | ✓ | ✓ |

## Test Architecture
- Test Runner: `.venv/bin/pytest tests/ -v`
- Test Files Layout:
  - `tests/e2e/test_e2e_tier1_features.py`: Happy-path feature isolation tests (≥5 per feature, 30+ tests).
  - `tests/e2e/test_e2e_tier2_boundaries.py`: Boundary, corner case, and stress tests (≥5 per feature, 30+ tests).
  - `tests/e2e/test_e2e_tier3_interactions.py`: Cross-feature pairwise interaction tests (≥10 tests).
  - `tests/e2e/test_e2e_tier4_scenarios.py`: Realistic multi-domain application scenarios (≥6 complex workloads).
- Pass/Fail Semantics:
  - Exit code 0, 0 test failures, 0 errors.
  - Pre-flight scan latency < 0.5ms (and strictly < 1.0ms).
  - 0.0% False Rejection Rate on clean domain texts (medicine, finance, law, science).
  - No grammatical fragmentation / dangling subordinators on Level 3–5 syntactic excision.
  - 100% lock acquisition and cleanup in read-only / linked worktree environments.

## Real-World Application Scenarios (Tier 4)
| # | Scenario | Features Exercised | Complexity |
|---|----------|--------------------|------------|
| 1 | Medical Oncology Report with Nested Concessions & Unbacked Dosing | F4, F5, F1 | High (Nesting Level 4) |
| 2 | Financial Earnings Advisory with Polyglot Injection & Footnotes | F1, F2, F3 | High (Polyglot + XML injection) |
| 3 | Multi-Process Git Worktree Evaluation under Read-Only Sandbox | F6, F7 | High (Concurrency + Worktree `.git` file) |
| 4 | Legal Contract Concessive Clauses with Deep Conditionals | F4, F5 | High (Nesting Level 5) |
| 5 | Coordinated Injection in Prompt + Concessive Hallucination in Draft | F1, F2, F4, F5 | High (Multi-stage threat) |
| 6 | High-Concurrency Worktree Lock Contention with Cross-Mount Publication | F6, F7 | High (Parallel process contention) |

## Coverage Thresholds
- Tier 1: ≥5 test cases per feature (30+ total)
- Tier 2: ≥5 test cases per feature (30+ total)
- Tier 3: Pairwise coverage across features (10+ total)
- Tier 4: ≥6 realistic application scenarios
- Total E2E Tests: ≥76 comprehensive tests
