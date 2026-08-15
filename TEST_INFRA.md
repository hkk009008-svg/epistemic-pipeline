# E2E Test Infra: Epistemic Pipeline Hardening

## Test Philosophy
- Opaque-box, requirement-driven. Derives strictly from `ORIGINAL_REQUEST.md`.
- Methodology: Category-Partition + Boundary Value Analysis + Pairwise Combinatorial + Real-World Workload Testing.

## Feature Inventory
| # | Feature | Source | Tier 1 (Coverage) | Tier 2 (Boundary/Adversarial) | Tier 3 (Cross-Feature) |
|---|---------|--------|:-----------------:|:-----------------------------:|:----------------------:|
| 1 | Polarity & Negation Grounding (R1) | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ |
| 2 | Prompt Injection Combinatorial Matrix (R2) | ORIGINAL_REQUEST §R2 | 5 | 5 | ✓ |
| 3 | Base64 Wrapper Decoder (R2) | ORIGINAL_REQUEST §R2 | 5 | 5 | ✓ |
| 4 | AST Proposition Span Grounding (R3) | ORIGINAL_REQUEST §R3 | 5 | 5 | ✓ |
| 5 | Multi-Citation Entity Isolation (R4) | ORIGINAL_REQUEST §R4 | 5 | 5 | ✓ |
| 6 | Uncited Quantitative Scanner (R5) | ORIGINAL_REQUEST §R5 | 5 | 5 | ✓ |
| 7 | Unbacked Authority Scanner (R5) | ORIGINAL_REQUEST §R5 | 5 | 5 | ✓ |
| 8 | Sanitizer Boundary Whitespace (R6) | ORIGINAL_REQUEST §R6 | 5 | 5 | ✓ |
| 9 | Sanitizer Authority Vocabulary (R6) | ORIGINAL_REQUEST §R6 | 5 | 5 | ✓ |

## Test Architecture
- Test runner: `.venv/bin/pytest tests/test_e2e_hardening.py -v` and `.venv/bin/pytest tests/ -v`
- Pass/Fail Semantics: 100% test pass rate, exit code 0.
- All existing 1,921+ baseline tests MUST remain 100% passing.

## Real-World Application Scenarios (Tier 4)
| # | Scenario | Features Exercised | Complexity |
|---|----------|--------------------|------------|
| 1 | Medical Claims with Negation & Dual-Entity Clinical Trials | F1, F4, F5 | High |
| 2 | Prompt Injection Obfuscation via Base64 in Regulatory Reports | F2, F3, F7 | High |
| 3 | Multi-Clause Scientific Paper with Ungrounded Authority & Swapped Stats | F4, F5, F6, F9 | High |
| 4 | Financial Report with Uncited Numbers and Bare Percentages | F6, F8, F9 | High |
| 5 | Complex Multi-Source Synthesis with AST Sub-Clause Grounding | F1, F4, F5, F8 | High |

## Coverage Thresholds
- Tier 1: ≥5 per feature (≥45 test cases)
- Tier 2: ≥5 per feature (≥45 test cases)
- Tier 3: Pairwise combinations across all 9 features (≥15 test cases)
- Tier 4: ≥5 realistic multi-feature application workloads
- Total: ≥110 dedicated hardening test cases in `tests/test_e2e_hardening.py`
