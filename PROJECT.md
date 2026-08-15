# Project: Epistemic Pipeline Hardening

## Architecture
The Epistemic Pipeline is a multi-stage epistemic verification engine for LLM outputs. It consists of:
- **Pre-Flight Scanner & Source Matcher (`pipeline/source_match.py`)**: High-speed, deterministic token and regex scanner performing pre-flight threat detection (prompt injections, delimiter breakout, system overrides) in <0.5ms before LLM 2 invocation, as well as citation-to-source proposition grounding.
- **AST Clause Disentangler & Grammar Sanitizer (`pipeline/source_match.py`, `pipeline/sanitizer.py`)**: Deterministic clause parser and grammar cleaner parsing complex syntactic nesting (Levels 3–5) into discrete `PropositionSpan` objects, enabling surgical excision of unbacked sub-clauses while preserving verified independent clauses without grammatical fragmentation or dangling connectors.
- **Worktree Coordination & Robust Event Locking (`pipeline/event_lock.py`, `pipeline/knowledge_store.py`)**: Resilient multi-tier file locking and worktree resolution supporting linked worktrees (`.git` pointer files), read-only sandboxes, and cross-mount storage with comprehensive diagnostics and graceful 4-tier fallback.
- **Multi-Stage Verifier & Arbiter (`pipeline/stages.py`, `pipeline/verifier.py`, `pipeline/arbiter.py`)**: End-to-end pipeline coordination validating claims against retrieved sources, executing LLM 2 verification with pre-flight bypass, and arbitrating final verdicts.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| F1 | Pre-Flight Injection & Delimiter Scanner | Pre-compiled regex and token scanner with Unicode zero-width and ANSI stripping, intercepting 100% of prompt injections, system overrides, XML tags, polyglot JSON codeblocks, and template breakouts in <0.5ms with 0.0% FRR | M1 | Survey (R1) |
| F2 | Dual-Target Preflight Verification | Multi-target preflight scan in `stage_verify` and `_verify_text` scanning both prompt and draft text before LLM 2 invocation | M1 | Survey (R1) |
| F3 | Source Match Baseline Fix | Resolve `num_sources` uninitialized variable / NameError in `pipeline/source_match.py` | M1 | Survey (R1/R2/R3) |
| F4 | Subordinate Clause AST Parser | Deterministic syntax parser extracting typed `PropositionSpan` nodes (Levels 1–5: concessive, conditional, temporal, relative, coordinate, participial) with character spans, subordinator tokens, and citation mappings | M2 | Survey (R2) |
| F5 | AST-Aware Grammar Cleaner & Excision | Surgical excision of unbacked sub-clauses with subordinator promotion, orphan connector stripping, and punctuation normalization in `pipeline/sanitizer.py` | M2 | Survey (R2) |
| F6 | Worktree Sandbox Event Lock | 4-tier fallback locking mechanism (`WorktreeEventLock`) handling linked worktrees (`.git` pointer files), read-only sandboxes, NFS/overlayfs, with atomic file creation, stale lock recovery, and telemetry | M3 | Survey (R3) |
| F7 | Cross-Mount Storage Resilience | `os.link` fallback to `os.replace` on `EXDEV`/`EPERM` in `pipeline/knowledge_store.py` | M3 | Survey (R3) |
| F8 | E2E Test Suite (Tiers 1–4) | Requirement-driven opaque-box E2E test suite covering feature isolation, boundaries, cross-feature interactions, and real-world multi-domain scenarios | E2E-Track | Survey (Test Infra) |
| F9 | Full Integration & Adversarial Hardening (Tier 5) | 100% pass of all 1,240+ existing tests and new E2E tests, followed by adversarial challenger coverage hardening | Final-Milestone | Survey (Acceptance) |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| E2E | E2E Testing Track | Design and implement comprehensive opaque-box E2E test suite (Tiers 1–4) and publish `TEST_READY.md` | none | DONE (105 tests passed, Gate PASS, Audit CLEAN) |
| M1 | Pre-Flight Prompt Injection Interceptor | Implement F1, F2, F3 in `pipeline/source_match.py` and `pipeline/stages.py` | none | DONE (83 tests passed, Audit CLEAN) |
| M2 | Subordinate Clause AST Disentangler | Implement F4, F5 in `pipeline/source_match.py` and `pipeline/sanitizer.py` | M1 | DONE (394 tests passed, Audit CLEAN) |
| M3 | Worktree Sandbox & Event Lock Fallback | Implement F6, F7 in `pipeline/event_lock.py` and `pipeline/knowledge_store.py` | none | DONE (123 tests passed, Audit CLEAN) |
| Final | Final Integration & Adversarial Hardening | Phase 1: Pass 100% E2E test suite and existing 1,240+ tests. Phase 2: Tier 5 Adversarial challenger hardening | E2E, M1, M2, M3 | DONE (1,921 tests passed, 140 Tier 5 tests added, Gate PASS, Audit CLEAN) |

## Interface Contracts

### 1. `pipeline/source_match.py` — Pre-Flight & AST Disentangler
```python
class ClauseType(str, Enum):
    INDEPENDENT = "independent"
    CONCESSIVE = "concessive"
    CONDITIONAL = "conditional"
    TEMPORAL = "temporal"
    RELATIVE = "relative"
    COORDINATE = "coordinate"
    PARTICIPIAL = "participial"

@dataclass
class PropositionSpan:
    span_id: str
    clause_type: ClauseType
    raw_text: str
    cleaned_text: str
    start_char: int
    end_char: int
    subordinator: Optional[str] = None
    citation_indices: list[int] = field(default_factory=list)
    parent_span_id: Optional[str] = None
    is_matrix: bool = False
    nesting_level: int = 1

def run_preflight_scan(
    text: str,
    sources: Optional[list[dict]] = None,
    source_keywords: Optional[list[str]] = None,
    source_numbers: Optional[list[int]] = None,
    prompt: Optional[str] = None,
) -> dict:
    """Returns {'has_hard_preflight': bool, 'findings': list[dict], 'preflight_latency_ms': float}"""

def parse_clause_ast(sentence: str) -> list[PropositionSpan]:
    """Deterministically decomposes complex sentence into hierarchical PropositionSpan AST nodes."""

def disentangle_and_excise(
    text: str,
    unbacked_span_ids: set[str],
    spans: list[PropositionSpan],
) -> str:
    """Surgically excises unbacked sub-clauses and reconstitutes grammatical sentences."""
```

### 2. `pipeline/sanitizer.py` — Grammar Cleaning
```python
def clean_grammar_and_punctuation(text: str) -> str:
    """Cleans leading/trailing coordinators, orphaned subordinators, double punctuation, and fixes capitalization."""
```

### 3. `pipeline/event_lock.py` — Worktree Event Lock
```python
class LockTier(str, Enum):
    KERNEL_FLOCK = "kernel_flock"
    TEMP_FLOCK = "temp_flock"
    USER_SPACE_ATOMIC = "user_space_atomic"
    IN_MEMORY_MUTEX = "in_memory_mutex"

@dataclass
class LockDiagnostic:
    target_path: str
    resolved_lock_path: str
    active_tier: LockTier
    fallback_reasons: list[str]
    is_locked: bool
    holder_pid: Optional[int] = None
    acquired_at: Optional[float] = None

class WorktreeEventLock:
    def __init__(self, target_path: str, timeout_seconds: float = 30.0, stale_timeout_seconds: float = 60.0): ...
    def acquire(self, blocking: bool = True) -> bool: ...
    def release(self) -> None: ...
    def __enter__(self) -> "WorktreeEventLock": ...
    def __exit__(self, exc_type, exc_val, exc_tb) -> None: ...
    def get_diagnostic(self) -> LockDiagnostic: ...

def resolve_git_dir(path: Union[str, Path]) -> Path:
    """Resolves true git directory handling linked worktree '.git' pointer files."""
```

## Code Layout
- `pipeline/source_match.py`: Pre-flight regex scanner, Unicode normalizer, AST clause parser, proposition span extractor.
- `pipeline/sanitizer.py`: AST-aware grammar reconstruction, subordinator cleaner, punctuation normalizer.
- `pipeline/stages.py`: Pipeline stage orchestration and dual-target preflight hooks.
- `pipeline/event_lock.py`: 4-tier fallback WorktreeEventLock, linked worktree resolver, and diagnostic reporting.
- `pipeline/knowledge_store.py`: Knowledge store SQLite integration with cross-mount fallback.
- `tests/`: Existing 1,240 unit/integration tests and new E2E/adversarial test suites.
