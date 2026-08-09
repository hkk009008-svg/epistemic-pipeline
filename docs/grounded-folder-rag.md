# Grounded folder RAG: minimal executable design

This repository now has a separate grounded lane for questions that must be
answered only from a user's managed local corpus. It reuses the repository's
three configurable model stages and structured-output client, but it does not
reuse the legacy web-search, keyword source-matching, sanitizer bypass, rewrite
loop, or heuristic confidence paths.

The safety boundary is not "three models agree." It is:

```text
fixed authenticated corpus
  -> deterministic SQLite FTS5 retrieval
  -> one immutable evidence packet
  -> LLM 1 proposes atomic cited claims
  -> LLM 2 blindly verifies every claim against the same packet
  -> LLM 3 selects only verifier-supported claim IDs
  -> Python validates references and renders the cited answer
```

Three model calls are procedurally different interpretations of the same
evidence, not three independent sources of truth.

## Why this is a separate lane

The original `/api/pipeline` is a broad epistemic-policy pipeline. Its GPT-3
stage is conditional, its evidence is web-search snippets, and several legacy
fallbacks can produce a `PASS` without all three roles checking the same source
packet. Those behaviors are useful compatibility surfaces but are the wrong
trust contract for private folder-backed RAG.

The grounded lane therefore has no activation bypass, web fallback, keyword
entailment upgrade, NLI authority, rewrite loop, best-of-N, or numeric
"hallucination probability." It returns a verified answer, a verified partial
answer, or an abstention.

## Storage layout

`KNOWLEDGE_ROOT` points to one authenticated corpus per deployment:

```text
knowledge_data/
  sources/
    <folder>/
      <document_id>/
        versions/
          <sha256>.txt
  .rag/
    index.sqlite3
```

- Source versions are immutable and addressed by the SHA-256 of their UTF-8
  bytes. Every logical change also appends a `document_versions` row with its
  revision ID, parent revision, closed reason code, folder/title snapshot, and
  source location. Old source bytes and revision metadata remain available for
  audit.
- `index.sqlite3` is the authoritative active document map and FTS5 chunk index
  for this slice. Immutable blobs preserve evidence bytes, but there is no
  automatic database rebuild yet; back up the database with the source tree.
- `document_id` is globally unique inside the corpus. Reusing it in a different
  folder moves the active logical document; the old content-addressed blob
  remains inactive for audit. A non-idempotent update must supply the current
  `expected_revision_id` plus `revision_reason`; SQLite `BEGIN IMMEDIATE`
  compare-and-swap allows only one concurrent writer to advance that head.
  Reasons are checked against the mutation: `metadata_update` cannot change
  source bytes, `content_update` must change them, and `restore` must reproduce
  a representation already present in that document's history.
- Chunks preserve exact source character and line spans. Retrieval order is
  deterministic (`BM25`, then evidence ID).
- Whitespace-free or unusually long spans are split at a fixed 4,000-character
  ceiling. Retrieval uses at most 24 unique non-stopword query terms, retaining
  one-character identifiers and numerals; when filtering would leave no terms,
  it falls back to the raw lexical tokens. A further eligible term sets
  `query_term_limit`. A query whose normalized canonical JSON would consume
  more than the fixed 40,000-byte query allocation is rejected before
  retrieval; it is never silently truncated or allowed to exceed the packet
  ceiling. Retrieval
  asks SQLite for `top_k + 1`, so omission at the caller's rank limit sets
  `top_k` without requiring a full index count. Canonical evidence packets also
  have a 48,000-byte ceiling; omitted lower-ranked items set `byte_budget`.
  `coverage_limited` and the closed `coverage_reasons` values
  `query_term_limit`, `top_k`, and `byte_budget` disclose these distinct limits.
  `retrieval_truncated` remains the byte-budget-only compatibility field.
- Each retrieved source and chunk is hash-checked before it can enter a model
  prompt. A stale or modified index fails closed.

The first version accepts UTF-8 text through JSON. PDF/OCR, spreadsheet, email,
multimodal, file-watching, and auto-taxonomy ingestion are deliberately not in
this slice because each adds a separate extraction-correctness problem.
There is no deletion/retention API yet, so old immutable versions must be
managed at the private-volume level when policy requires erasure.

## Immutable evidence packet

Retrieval runs once. The resulting canonical packet contains:

- corpus revision;
- canonical query;
- pinned retriever version;
- ordered evidence IDs;
- document revision ID, folder/title, and versioned relative path;
- source and chunk SHA-256 hashes;
- exact character and line locations;
- chunk text and BM25 ranking value.

The packet ID is the SHA-256 of its canonical JSON. The exact same serialized
packet is included in all three model requests, and each model must echo its
ID. LLM 2 also echoes the canonical draft hash; LLM 3 echoes the draft and
verification hashes. Any mismatch abstains.

Evidence text is explicitly marked as untrusted data. Instructions embedded in
a source document are not authorized instructions to any model.

This proves support relative to the retrieved packet, not completeness against
the entire folder map. The first version intentionally accepts false abstention
from lexical misses. A zero-match vocabulary mismatch produces no claim that
could drive claim-specific retrieval; only a real labeled benchmark can justify
a bounded deterministic query-expansion, hybrid, or embedding experiment for
that failure. Claim-specific FTS is a different future mechanism for a draft
whose supporting or conflicting evidence was omitted from the first packet. It
must use the same corpus revision, produce a superseding shared packet, and
rerun all three roles—not become an open-ended agent retrieval loop.

## Three constrained roles

### 1. Claim-first answerer

LLM 1 returns standalone atomic claims, not trusted final prose. Every claim
must cite an evidence ID and an exact quote. Python assigns canonical `C1`,
`C2`, ... claim IDs and rejects nonexistent IDs or non-verbatim quotes.

### 2. Blind verifier

LLM 2 receives the question, claim IDs/text, and evidence packet. It does not
receive LLM 1's proposed citations, rationale, or confidence. It must return
exactly one verdict per claim:

- `SUPPORTED`
- `CONTRADICTED`
- `INSUFFICIENT`
- `CONFLICT`

A supported claim must include an exact support quote. Missing/duplicate claim
IDs, invented evidence IDs, and non-verbatim quotes invalidate or downgrade the
result; keyword overlap can never create support.

### 3. Constrained final adjudicator

LLM 3 runs when at least one claim is eligible for release and its complete
input fits the fixed model-input budget. An oversized finalizer input abstains
before the call. Otherwise LLM 3 sees the packet, canonical draft, verification
ledger, and the IDs eligible for release. It may only order a subset of those
IDs or abstain. It cannot write or rewrite factual prose.

Python then renders the verified claim text with numbered citation spans.
An unsupported, contradicted, insufficient, conflicting, duplicate, or unknown
claim ID cannot reach the final answer. Model-authored Markdown and HTML are
escaped, while square brackets and Unicode control/format characters make a
claim ineligible, so only Python can create citation markers.

## API setup

Grounded endpoints are disabled until a dedicated token is configured:

```bash
KNOWLEDGE_ROOT=/absolute/path/to/private-knowledge
KNOWLEDGE_API_TOKEN=replace-with-a-long-random-secret
```

If `ADMIN_TOKEN` is unset, this knowledge token also protects configuration
mutation endpoints because their stage/provider settings control where private
evidence packets are sent. Set a separate `ADMIN_TOKEN` to split those roles.

The same existing GPT stage configuration is used for the answerer, verifier,
and final adjudicator. Different providers may reduce some correlated model
errors, but they do not create independent evidence.

Ingest or update one logical document:

```bash
curl -X POST http://localhost:8000/api/grounded/documents/profile \
  -H "Authorization: Bearer $KNOWLEDGE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "folder": "personal/preferences",
    "title": "Profile",
    "content": "Alice prefers Neovim and uses a dark color theme."
  }'
```

The create response contains `revision_id`. A later update is compare-and-swap:

```bash
curl -X POST http://localhost:8000/api/grounded/documents/profile \
  -H "Authorization: Bearer $KNOWLEDGE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "folder": "personal/preferences",
    "title": "Profile",
    "content": "Alice prefers Helix and uses a dark color theme.",
    "expected_revision_id": "<revision_id from the current head>",
    "revision_reason": "correction"
  }'
```

Omitting or supplying a stale head for a real update returns HTTP 409. Repeating
the exact active representation is idempotent and does not append a duplicate
revision.

Query the corpus:

```bash
curl -X POST http://localhost:8000/api/grounded/query \
  -H "Authorization: Bearer $KNOWLEDGE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Which editor does Alice prefer?", "top_k": 6}'
```

The versioned `grounded-rag-v1` response exposes the status, rendered answer,
run/packet/corpus IDs, retrieval/chunker/prompt versions, coverage reasons,
aggregate verification counts, and exact source citations. Once execution
reaches model configuration, it also exposes credential-free SHA-256
fingerprints of each provider/model pair. They are identifiers, not secrets;
the no-lexical-match path runs before model configuration and returns `null`
fingerprints. `reason_code` is a closed vocabulary. The response does not expose
raw model outputs, provider credentials, endpoint URLs, or hidden reasoning.

Before releasing a normal `ANSWER`, `PARTIAL`, or `ABSTAIN`, the service appends
a hash-bound `grounded-run-receipt-v1` row. Receipts contain only versions,
hashes, IDs, stages, counts, status/reason, coverage, and latency. They
deliberately exclude the query, answer, claim text, quotations, source paths,
evidence text, API keys, endpoint URLs, and rationale. Receipt UPDATE and DELETE
are blocked; failure to persist the receipt prevents release of the response.
Input, retrieval, store, or model-configuration failures, provider exceptions,
and route timeouts remain HTTP errors and do not currently receive a run
receipt.

## Status contract

- `ANSWER`: every emitted claim cleared all gates.
- `PARTIAL`: only a supported subset is safe to emit.
- `ABSTAIN`: no evidence, matching evidence that cannot fit the fixed packet
  budget, an answerer abstention, a protocol/hash/citation failure, no supported
  claim, or a final adjudicator abstention. Packet-budget omission is reported
  as `evidence_packet_budget_exceeded`; an empty result after query-term capping
  is `retrieval_query_term_limit_exceeded`. Neither is labeled a lexical miss.

`reason_code=partial_with_conflict` or `conflict_in_evidence`, together with
`conflict_claim_count`, keeps conflicting corpus evidence distinct from a claim
that is simply contradicted or insufficiently supported.

Availability failures return an HTTP error and never fall back to draft text.

## Security and deployment boundary

This is a single-corpus local/deployment slice. Clients cannot supply a user ID
or filesystem root. A real multi-user service must derive an opaque corpus root
from an authenticated server-side principal and enforce per-document access
before retrieval; a client-provided `user_id` is not an authorization control.

Both immutable source files and the SQLite index contain user data. Directories
are restricted to the service account, but application-level encryption is not
implemented. Use encrypted storage where required. Retrieved chunks are sent to
all configured LLM providers, so provider data-handling policy is part of the
trust boundary. On Railway/container deployments, mount `KNOWLEDGE_ROOT` on a
persistent private volume rather than the ephemeral application filesystem.

Do not commit or bake `knowledge_data/` into an image; it is excluded by the
repository ignore files.

## Offline evaluation schema self-test

The repository includes a closed offline scoring contract and a synthetic
schema fixture at `tests/fixtures/grounded_rag_v1/cases.jsonl`. It makes no model
or network calls:

```bash
python scripts/evaluate_grounded_rag.py
python scripts/evaluate_grounded_rag.py --json
```

The fixture contains hand-authored expected and recorded observations so CI can
test schema validation and metric arithmetic. Its recall, citation, unsupported
claim, abstention, latency, and cost values are not measurements of this
pipeline. Citation support and corpus support are recorded labels; a citation
counts as valid only when its labeled evidence ID is both retrieved and present
in the expected relevant-evidence set. The scorer does not execute retrieval or
validate source quotations itself.

There is no live recording runner, independently human-labeled benchmark, stage
ablation runner, or verifier precision/recall evaluator yet. Do not describe the
synthetic fixture as a quality baseline or use its numbers to select retrieval
or model changes. A future explicitly initiated runner must freeze a real corpus
and independent labels before comparing answerer-only, answerer-plus-verifier,
and the complete constrained path. Only those measurements may justify
embeddings, hybrid retrieval, reranking, or bounded claim-specific retrieval.
False abstention remains preferable to unmeasured recall machinery in the first
evidence-enforcement slice.

Do not report the existing hand-weighted confidence label as a probability of
hallucination. A numerical risk estimate requires a held-out labeled set and
calibration/risk-coverage evaluation.

Mechanisms deferred until a qualifying measured, product, legal, privacy,
security, or operational trigger exists are recorded in
`docs/grounded-rag-adopt-later.md`.

## Research basis

- [RAGTruth (ACL 2024)](https://aclanthology.org/2024.acl-long.585/) shows that
  retrieval-augmented systems still produce unsupported and contradictory
  claims.
- [RAGChecker (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/27245589131d17368cccdfa990cbf16e-Abstract-Datasets_and_Benchmarks_Track.html)
  motivates claim-level, separate retriever/generator diagnostics.
- [ALCE (EMNLP 2023)](https://aclanthology.org/2023.emnlp-main.398/) motivates
  measuring citation correctness and completeness rather than trusting the
  presence of citation markers.
- [Chain-of-Verification (ACL 2024)](https://aclanthology.org/2024.findings-acl.212/)
  supports procedurally independent fact-checking instead of conversational
  self-agreement.
- [SQLite FTS5](https://www.sqlite.org/fts5.html) supplies the dependency-free
  BM25 baseline used here.
- [Risk-controlled RAG (EMNLP 2024)](https://aclanthology.org/2024.findings-emnlp.133/)
  illustrates why a statistical reliability claim requires calibration data,
  not three verbal confidence scores.
