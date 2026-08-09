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
  bytes. Updating a document creates a new version; old source bytes remain
  available for audit.
- `index.sqlite3` is the authoritative active document map and FTS5 chunk index
  for this slice. Immutable blobs preserve evidence bytes, but there is no
  automatic database rebuild yet; back up the database with the source tree.
- `document_id` is globally unique inside the corpus. Reusing it in a different
  folder moves the active logical document; the old content-addressed blob
  remains inactive for audit.
- Chunks preserve exact source character and line spans. Retrieval order is
  deterministic (`BM25`, then evidence ID).
- Whitespace-free or unusually long spans are split at a fixed 4,000-character
  ceiling. Canonical evidence packets have a 48,000-byte ceiling; lower-ranked
  items that do not fit are omitted and `retrieval_truncated` is reported.
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
- document/folder/title and versioned relative path;
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
from lexical misses. If a labeled retrieval benchmark shows that misses or
omitted conflicts dominate, the next bounded step is claim-specific FTS against
the same corpus revision followed by a superseding shared packet—not an
open-ended agent retrieval loop.

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

LLM 3 runs whenever at least one claim is eligible for release. It sees the packet, canonical draft,
verification ledger, and the IDs eligible for release. It may only order a
subset of those IDs or abstain. It cannot write or rewrite factual prose.

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

Query the corpus:

```bash
curl -X POST http://localhost:8000/api/grounded/query \
  -H "Authorization: Bearer $KNOWLEDGE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Which editor does Alice prefer?", "top_k": 6}'
```

The response exposes the status, rendered answer, packet/corpus IDs, aggregate
verification counts (including conflict separately from contradiction), and
exact source citations. `retrieval_truncated` discloses when the fixed packet
byte budget omitted lower-ranked matches. It does not expose raw model outputs
or hidden reasoning traces.

## Status contract

- `ANSWER`: every emitted claim cleared all gates.
- `PARTIAL`: only a supported subset is safe to emit.
- `ABSTAIN`: no evidence, an answerer abstention, a protocol/hash/citation
  failure, no supported claim, or a final adjudicator abstention.

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

## What to measure before adding sophistication

Build a small human-reviewed corpus with answerable, unanswerable, conflicting,
stale, distractor, prompt-injection, number/date mutation, modality reversal,
and compound-claim cases. Measure separately:

- retrieval recall@k and MRR for gold evidence;
- citation span validity and citation completeness;
- supported/unsupported/contradiction verdict precision and recall;
- unsupported claims reaching the final answer (primary safety metric);
- correct abstention and false-abstention rates;
- p50/p95 latency and cost.

Compare answerer-only, answerer+verifier, and the complete constrained path on
the same frozen corpus. Add embeddings, hybrid retrieval, reranking, or bounded
claim-specific retrieval only if that evaluation shows lexical retrieval recall
is the bottleneck. False abstention is preferable to opaque recall machinery in
the first evidence-enforcement slice.

Do not report the existing hand-weighted confidence label as a probability of
hallucination. A numerical risk estimate requires a held-out labeled set and
calibration/risk-coverage evaluation.

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
