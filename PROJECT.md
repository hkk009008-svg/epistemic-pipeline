# Project: Epistemic Pipeline Hardening (R1–R6)

## Architecture
- **Epistemic Pipeline**: Grounding, preflight verification, citation validation, and output sanitization system.
- **Core Modules**:
  - `pipeline/source_match.py`: Claim-to-source matching, polarity checking, injection detection, citation grounding, AST proposition span parsing, entity-attribute binding, and unbracketed quantitative scanning.
  - `pipeline/sanitizer.py`: Output sanitization, bare percent placeholder replacement, banned authority claim suppression, whitespace/token boundary preservation.
- **Data Flow**:
  1. Input Draft -> `run_preflight_scan` (Injection scanning, Unbracketed quantitative & authority scanning) -> Finding List
  2. Input Draft + Sources -> `verify_citation_grounding` (AST Proposition Span keyword grounding, multi-citation entity-attribute isolation, number validation) -> Finding List
  3. Claims + Sources -> `recategorize_with_sources` & `filter_findings_with_sources` (Polarity & negation-aware grounding overrides) -> Updated Claims & Filtered Findings
  4. Final Text -> `sanitize_output` (Banned authority phrase filtering, bare percent replacement with clean whitespace boundaries) -> Sanitized Text

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Polarity & Negation Detection | Prohibit lexical keyword overlap from overriding Unsupported to Observed or dropping T1 violations when polarity tokens (not, no, never, fails to, without, prevents, prohibits) disagree | M1 | ORIGINAL_REQUEST §R1 |
| 2 | Prompt Injection Combinatorial Matrix | Expand `_INJECTION_PATTERN` into combinatorial $(ActionVerb \times Modifier \times Target)$ matrix covering natural synonyms | M2 | ORIGINAL_REQUEST §R2 |
| 3 | Base64 Wrapper Decoder | Fast (<0.05ms) Base64 encoded payload wrapper detection and decoding for prompt injection preflight | M2 | ORIGINAL_REQUEST §R2 |
| 4 | AST Proposition Span Grounding | Integrate `parse_clause_ast` into `verify_citation_grounding` to evaluate keyword overlap per PropositionSpan AST node | M3 | ORIGINAL_REQUEST §R3 |
| 5 | Multi-Citation Entity Isolation | Bind quantitative figures in multi-citations `[1, 2]` to specific sources by co-occurring entity keywords in proposition spans | M4 | ORIGINAL_REQUEST §R4 |
| 6 | Uncited Quantitative Scanner | Detect unbracketed sentences containing quantitative figures (numbers, percentages, currencies) and flag as T3 uncited claims | M5 | ORIGINAL_REQUEST §R5 |
| 7 | Unbacked Authority Scanner | Detect unbracketed sentences containing unbacked authority assertions and flag as T3 uncited claims | M5 | ORIGINAL_REQUEST §R5 |
| 8 | Sanitizer Boundary Whitespace Padding | Ensure `_replace_bare_percents` preserves word boundaries and avoids token gluing (e.g. `showsUnknown(...)`) | M6 | ORIGINAL_REQUEST §R6 |
| 9 | Sanitizer Authority Vocabulary Expansion | Expand `_BANNED_EVIDENCE_RE` to filter ungrounded authority claims ("clinical evidence demonstrates", "medical consensus shows", "published papers confirm") | M6 | ORIGINAL_REQUEST §R6 |
| 10| E2E Test Suite & Adversarial Hardening | Comprehensive test suite covering Tiers 1-5, verifying 100% pass rate across 1,921+ existing tests and new unit/adversarial tests | M7 | ORIGINAL_REQUEST §Acceptance Criteria |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Polarity & Negation Grounding | `pipeline/source_match.py`: polarity detection, negation tokens, override gating | none | **DONE** |
| M2 | Prompt Injection Matrix & Base64 | `pipeline/source_match.py`: combinatorial synonym matrix, Base64 wrapper decoder | none | **DONE** |
| M3 | AST Proposition Span Grounding | `pipeline/source_match.py`: integrate `parse_clause_ast` in `verify_citation_grounding` | none | **DONE** |
| M4 | Multi-Citation Entity Isolation | `pipeline/source_match.py`: span-level entity keyword matching and number binding | M3 | **DONE** |
| M5 | Uncited Quantitative & Authority Scanner | `pipeline/source_match.py`: scan unbracketed sentences for numbers/authority phrases | none | **DONE** |
| M6 | Output Sanitizer Whitespace & Vocabulary | `pipeline/sanitizer.py`: boundary padding, `_BANNED_EVIDENCE_RE` expansion | none | **DONE** |
| M7 | Full E2E & Adversarial Hardening Pass | Phase 1: 100% E2E pass (Tiers 1-4); Phase 2: Adversarial hardening (Tier 5) | M1, M2, M3, M4, M5, M6 | **DONE** |

## Interface Contracts
### Polarity Grounding Contract (`pipeline/source_match.py`)
- `extract_polarity_state(text: str) -> bool` (or polarity representation): Identifies positive vs negative polarity using negation words (`not`, `no`, `never`, `without`, `fails to`, `prevents`, `prohibits`, `refutes`, etc.).
- `has_polarity_mismatch(claim_text: str, source_text: str) -> bool`: Returns `True` if claim and source express contradictory polarity.
- `recategorize_with_sources` and `filter_findings_with_sources` MUST NOT upgrade `Unsupported` to `Observed` or drop `T1` findings when `has_polarity_mismatch` is `True`.

### Prompt Injection Contract (`pipeline/source_match.py`)
- `scan_prompt_injection(text: str) -> list[GroundingFinding]`: Emits findings for prompt injection matches.
- Regex matrix combines `(ignore|disregard|forget|bypass|override|drop|clear|reset|disable|dismiss)` $\times$ `(previous|prior|earlier|past|preceding|existing|above)` $\times$ `(instructions|directives|rules|constraints|policies|safety|guidelines|protocols|requirements|safeguards)`.
- Base64 wrapper decoding handles `base64:`, `[base64]`, and `atob()` wrappers in <0.05ms.

### AST Grounding Contract (`pipeline/source_match.py`)
- `verify_citation_grounding(text: str, sources: list[str]) -> list[GroundingFinding]`:
  - Parses clauses into `PropositionSpan` nodes using `parse_clause_ast(sent)`.
  - Evaluates keyword overlap and relevance per individual `PropositionSpan` AST node.

### Multi-Citation Entity Isolation Contract (`pipeline/source_match.py`)
- When citation groups `[1, 2]` are encountered:
  - Extract entity keywords for each `PropositionSpan`.
  - Map each proposition span to the best-matching source in the citation group.
  - Numbers in the span are checked strictly against the bound source's numbers.

### Uncited Quantitative Scanner Contract (`pipeline/source_match.py`)
- `verify_citation_grounding` and `run_preflight_scan`:
  - For unbracketed sentences containing quantitative numbers (percentages, currencies, counts, multipliers) or unbacked authority claims, emit `T3` uncited factual claim findings.

### Output Sanitizer Contract (`pipeline/sanitizer.py`)
- `_replace_bare_percents(text: str) -> str`: Inserts boundary whitespace/padding around `"Unknown(Actionable)..."` to prevent token gluing (e.g. `showsUnknown(...)`).
- `_BANNED_EVIDENCE_RE`: Extended with phrases: `clinical evidence demonstrates`, `medical consensus shows`, `published papers confirm`, etc.

## Code Layout
- Implementation:
  - `/Users/hyungkoookkim/epistemic-pipeline/pipeline/source_match.py`
  - `/Users/hyungkoookkim/epistemic-pipeline/pipeline/sanitizer.py`
- Tests:
  - `/Users/hyungkoookkim/epistemic-pipeline/tests/test_source_match.py`
  - `/Users/hyungkoookkim/epistemic-pipeline/tests/test_sanitizer.py`
  - `/Users/hyungkoookkim/epistemic-pipeline/tests/test_e2e_hardening.py` (New Comprehensive E2E Suite)
  - `/Users/hyungkoookkim/epistemic-pipeline/tests/test_tier5_preflight_ast_adversarial.py`
