# Grounded RAG: adopt later

This file is a parking lot for mechanisms that may become useful after the
current grounded lane has evidence of need. It is not an implementation roadmap.
The active design remains the minimal UTF-8, single-corpus, SQLite-FTS5 pipeline
in `docs/grounded-folder-rag.md`.

Promote an item only when its named trigger is satisfied by either:

- a frozen, human-reviewed benchmark or production incident that demonstrates
  the failure mode; or
- an explicit product, legal, privacy, security, or operational requirement.

For a performance or quality mechanism, real measurements must identify the
bottleneck and define an acceptance threshold. The synthetic offline scorer
fixture is a schema self-test, not a measured baseline. For every mechanism,
the owner must define retention, provider-disclosure, and deletion rules, and
the smallest slice must fail closed, preserve corpus/source binding, and have a
rollback or disable path.

## Format-specific ingestion

**Trigger:** users need PDF, OCR, spreadsheet, email, or multimodal sources that
cannot be represented faithfully as the current UTF-8 document input.

**Smallest future slice:** add one extractor for one format. Produce a
hash-bound ingestion manifest containing source hash, extractor/version,
records or pages scanned, chunks emitted, records dropped, kinded anomalies,
and exact source spans. Keep extraction separate from indexing and commit the
new active revision only when validation succeeds.

**Evidence gate:** fixtures cover malformed, empty, encrypted, reordered,
partially readable, and adversarial files. Tests prove that unsupported content
and silent loss are reported, exact locations survive extraction, and a failed
import leaves the active corpus unchanged.

**Privacy and retention:** retain original files and extracted text only under
an explicit policy. Derived text can be as sensitive as the source. Manifests
and logs should prefer hashes, counts, and reason codes over raw content.

**Do not build:** a universal ingestion framework, file watcher, automatic
taxonomy, or several extractors in parallel before one format passes its gold
fixtures.

## Deterministic structured and numeric answers

**Trigger:** a recurring high-stakes question requires calculations or exact
entity, period, unit, and source-revision matching, and evaluation shows that
quotation-level verification is insufficient.

**Smallest future slice:** define one domain schema and one deterministic query
or calculation. Bind each fact to `{entity, period, value, unit,
source_revision}`; represent missing and unknown explicitly. Models may explain
verified results but may not perform authoritative arithmetic.

**Evidence gate:** golden calculations and reconciliation fixtures cover unit,
date, entity, missing-value, stale-revision, and conflicting-source cases. The
structured path must improve the target error rate without increasing
unsupported claims released.

**Privacy and retention:** store only fields required for the approved query.
Keep private values out of operational receipts and provider prompts when a
deterministic local calculation is enough.

**Do not build:** a general query language, knowledge graph, or catalog-only RAG
system for ordinary prose questions.

## Source lineage, deduplication, and conflict handling

**Trigger:** duplicate uploads, overlapping extracts, or stale sources are being
mistaken for independent corroboration or are hiding meaningful conflicts.

**Smallest future slice:** add stable source identity plus `source_kind`,
`observed_at`, validity interval, authority label, and `supersedes` metadata.
Collapse byte-identical evidence for corroboration counts and surface conflicts
without automatically choosing a winner.

**Evidence gate:** labeled duplicate, supersession, stale, and disagreement
cases demonstrate improved conflict and citation behavior. Cross-source
authority rules must be explicit and deterministic.

**Privacy and retention:** origin metadata may reveal organizations, accounts,
or communication history. Expose only the citation labels the caller is
authorized to see and preserve erasure behavior across lineage links.

**Do not build:** probabilistic truth selection, automatic entity merging, or a
general provenance graph before observed conflicts require them.

## Reviewed activation for authoritative changes

**Trigger:** extracted facts, user corrections, or retrieval-policy changes can
alter high-authority answers and require an explicit human acceptance point.

**Smallest future slice:** use immutable `draft -> reviewed -> active`
revisions bound by a review digest and expected active head. Restore copies an
old revision into a new draft; it never rewrites history or silently
reactivates prior content.

**Evidence gate:** stale-head, digest-mismatch, concurrent-review, failed
activation, and restore tests prove that unreviewed data cannot become active.
The owner must define which folders or change types require review.

**Privacy and retention:** receipts contain actor, hashes, states, and times—not
raw corrections. Immutability does not override a deletion obligation; define
how tombstones and physical erasure affect prior proof before activation ships.

**Do not build:** approval workflow for every ordinary text upsert, multiple
organizational roles, or a generic governance engine.

## Portable proof bundles

**Trigger:** an answer must be independently reviewed, reproduced, or retained
outside the running service.

**Smallest future slice:** export a versioned JSON bundle containing corpus and
packet IDs, retrieval-coverage state, claim-gate decisions, selected claim IDs,
citations and hashes, contract versions, and credential-free model/provider
fingerprints. Treat those fingerprints as guessable identifiers, not secrets.
Include source text only through an explicit privileged option.

**Evidence gate:** an offline validator detects altered fields, missing
evidence, contract mismatch, and non-verbatim citations. Redaction and replay
tests cover both complete and abstained answers.

**Privacy and retention:** proof bundles are private user data. Encrypt at rest
where required, define expiry and revocation behavior, and avoid hidden
reasoning or unrestricted raw model transcripts.

**Do not build:** a global hash chain, blockchain, external anchoring, or public
audit service until a concrete tamper or regulatory threat model requires it.

## Calibrated risk and outcome learning

**Trigger:** before collecting results, the owner defines the maximum acceptable
unsupported-release risk, minimum useful answer coverage, required strata, and
confidence level. Promotion requires enough independent held-out released claims
for the one-sided confidence bound on risk to meet that target at the required
coverage. Outcome learning has a separate trigger: the product begins making
recommendations with observable outcomes that the owner explicitly chooses to
track.

**Smallest future slice:** first publish an offline risk-coverage calibration
report. If outcome learning is later justified, record an explicit prediction,
its evidence revision, observed outcome, and an owner-ratified versioned lesson.

**Evidence gate:** calibration is evaluated on an untouched temporal holdout and
reports sample count, answer coverage, calibration error, risk-coverage curve,
and the predeclared confidence bound overall and for required strata. Drift has
an owner-set alert threshold. A lesson cannot affect prompts or retrieval until
the owner accepts it and regression tests pass.

**Privacy and retention:** labels, feedback, and outcomes can disclose sensitive
preferences or decisions. Separate evaluation data from operational logs and
define consent, access, and deletion before collection.

**Do not build:** a heuristic "hallucination probability," silent self-learning,
automatic promotion of model output into corpus truth, or an autonomous
recommendation loop.

## Retention, retirement, and erasure

**Trigger:** an owner policy, consent withdrawal, legal obligation, source
withdrawal, or owner-set storage ceiling requires data to stop being retrieved
or to be physically removed.

**Smallest future slice:** atomically retire one document from the active map and
FTS index, advance the corpus revision, and record a privacy-minimal tombstone.
Add privileged garbage collection for source versions, lineage metadata,
receipts, exports, and backups only after their separate retention rules are
defined.

**Evidence gate:** retired content cannot enter new packets; concurrent queries
remain bound to their original corpus revision; physical erasure covers every
declared storage location; shared or still-active evidence is not removed; and
restore/export behavior reports the tombstone honestly.

**Privacy and retention:** a tombstone must not retain the erased content or a
revealing title. Append-only history is not an excuse to violate an erasure
obligation.

**Do not build:** a generic records-management system, legal-hold workflow, or
silent filesystem deletion that leaves searchable index or backup copies.

## Backup, restore, and index recovery

**Trigger:** the deployment has a durability or restore-time objective, or a
corruption/loss incident demonstrates that manual volume backup is insufficient.

**Smallest future slice:** create one transactionally consistent private backup
of `index.sqlite3` and the source tree plus a hash manifest. Add an offline
restore validator. Attempt deterministic index rebuild only after all active
heads, titles, revision reasons, and chunker/retriever versions can be recovered
without guessing.

**Evidence gate:** destructive restore drills reproduce the exact corpus
revision and representative packet IDs; missing, altered, stale, and partial
backups fail closed; version upgrades have an explicit compatibility test.

**Privacy and retention:** backups inherit the corpus's access, encryption,
retention, location, and erasure requirements. Validation output contains hashes
and counts rather than source text.

**Do not build:** distributed replication, event sourcing, or automatic repair
that silently selects an active version.

## Retry-safe and corpus-pinned queries

**Trigger:** client retries or timeouts cause duplicate three-model cost, or a
caller must answer against a previously observed corpus revision.

**Smallest future slice:** accept `request_id`, a canonical request hash, and an
optional `expected_corpus_revision`. Under one lock, identical requests execute
once and replay a bounded private result; reusing an ID with a different payload
fails, and a corpus mismatch fails before any model call. If result retention is
not allowed, return an explicit already-completed conflict rather than rerunning.

**Evidence gate:** concurrent duplicate requests make at most one provider call
per stage; replay bytes and receipt binding are exact; payload conflicts and
stale corpus revisions fail closed; expiry behavior is deterministic.

**Privacy and retention:** a replayable response contains private answer and
citation data. Define encryption, access, maximum lifetime, and erasure before
storing it.

**Do not build:** a general job queue, workflow engine, or indefinitely retained
response cache.

## Provider eligibility and data classification

**Trigger:** any folder or document class is not approved for one or more
configured model providers, or policy requires local-only inference.

**Smallest future slice:** assign classification server-side and require every
stage provider to be eligible for every source in the packet. Fail before the
first model call when no compliant three-stage route exists. Use a local model
route rather than pretending that lossy redaction preserves exact evidence.

**Evidence gate:** mixed-classification packets, configuration changes, fallback
attempts, and provider outages cannot send evidence to an ineligible endpoint.
Receipts identify the policy/routing version without recording classifications
that disclose private context.

**Privacy and retention:** classification and provider policy are themselves
sensitive authorization data. Clients cannot downgrade them or choose a more
permissive route.

**Do not build:** content inspection by another unapproved model, silent
cross-provider fallback, or a generic policy language before a concrete class
requires it.

## Failure receipts

**Trigger:** incident response, cost reconciliation, or an audit requirement
needs durable accounting for attempts that currently end before a normal
response receipt.

**Smallest future slice:** append a metadata-only start event after authentication
and validation, then an immutable terminal event with a closed, redacted failure
class for configuration, provider, timeout, cancellation, retrieval, or store
failure. Never persist raw exception strings.

**Evidence gate:** fault injection covers every terminal path, duplicate events,
timeouts, cancellation, and receipt-store failure. Receipt collection must not
cause private response text to be released or turn a provider failure into a
false successful run.

**Privacy and retention:** even timing, provider fingerprints, and failure classes
can reveal usage patterns. Define access and expiry independently from source
retention.

**Do not build:** full distributed tracing, raw prompt logging, or hidden-reasoning
capture solely to obtain failure counts.

## Multi-user identity and access control

**Trigger:** more than one security principal or corpus is served by the same
deployment.

**Smallest future slice:** derive an opaque corpus identifier from authenticated
server-side identity, enforce document access before retrieval, and isolate
storage, receipts, configuration, and provider routing by principal.

**Evidence gate:** adversarial tenant-isolation tests cover ingestion, retrieval,
caches, errors, exports, configuration mutation, and concurrent requests.

**Privacy and retention:** each principal needs explicit export, retention, and
erasure behavior. A client-provided `user_id` or folder path is never an
authorization boundary.

**Do not build:** Supabase/Postgres, row-level security, a session framework, or
a PWA solely in anticipation of future users. Choose infrastructure only after
the actual deployment and threat model are known.

## Permanent constraints

Later sophistication must not weaken the current invariants: one immutable
packet per run, identical evidence for all three roles, exact quote validation,
closed claim release gates, Python-only rendering, explicit abstention, and no
private evidence in unapproved providers or logs. A mechanism that cannot keep
those invariants should not be adopted.
