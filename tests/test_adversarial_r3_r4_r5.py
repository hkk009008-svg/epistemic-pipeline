"""Adversarial stress-testing and empirical verification for R3, R4, and R5.

R3: Lexical Specificity & Retrieval Cutoff (pipeline/knowledge_store.py)
R4: Natural Sentence Structure in Output Sanitizer (pipeline/sanitizer.py)
R5: Granular Stage-Level Credential Diagnostics (pipeline/grounded_rag.py)
"""
from __future__ import annotations

import pytest

from pipeline.helpers import PipelineError
from pipeline.knowledge_store import KnowledgeStore
from pipeline.sanitizer import sanitize_output
import pipeline.grounded_rag as grounded
from pipeline.grounded_rag import (
    GroundedQueryRequest,
    run_grounded_rag,
)


# ==============================================================================
# R3: Lexical Specificity & Retrieval Cutoff Tests
# ==============================================================================

class TestR3RetrievalSpecificityAdversarial:
    """Stress tests for multi-term cutoff, distractor rejection, and query edge cases."""

    def test_multi_term_query_with_single_distractor_matches_abstains(self, tmp_path):
        """Multi-term query (N=4) with chunks matching only 1 term each must return empty packet."""
        store = KnowledgeStore(tmp_path / "knowledge")
        store.upsert_document("doc1", "infra", "Kubernetes 101", "Kubernetes cluster operations manual.")
        store.upsert_document("doc2", "infra", "Deploy Guide", "Application deployment strategies in production.")
        store.upsert_document("doc3", "infra", "Autoscaling", "Database autoscaling policies and metrics.")
        store.upsert_document("doc4", "infra", "Cloud Design", "High performance architecture design patterns.")

        # Query has 4 terms: [kubernetes, deployment, autoscaling, architecture]
        # Each document contains exactly ONE of those terms.
        query = "kubernetes deployment autoscaling architecture"
        packet = store.retrieve(query, top_k=6)

        # min_matched_terms is min(2, 4) = 2. Each chunk has matched_count == 1 < 2.
        assert len(packet.items) == 0
        assert packet.truncated is False
        assert packet.coverage_limited is False

    @pytest.mark.asyncio
    async def test_multi_term_distractor_query_fail_closed_in_grounded_rag(self, tmp_path):
        """End-to-end grounded RAG with single-term distractors must abstain with no_lexical_match."""
        store = KnowledgeStore(tmp_path / "knowledge")
        store.upsert_document("alpha", "general", "Alpha Doc", "The alpha project uses python.")
        store.upsert_document("beta", "general", "Beta Doc", "The beta project uses rust.")

        # Query: "alpha beta integration test" (terms: alpha, beta, integration, test)
        # alpha matches 1, beta matches 1. Both < 2.
        req = GroundedQueryRequest(prompt="alpha beta integration test", top_k=4)
        resp = await run_grounded_rag(req, store=store)

        assert resp.status == "ABSTAIN"
        assert resp.reason_code == "no_lexical_match"
        assert resp.retrieval_count == 0
        assert resp.stages_completed == ["retrieval"]
        assert resp.citations == []

    def test_multi_term_query_with_ge_2_matches_retrieves_successfully(self, tmp_path):
        """Multi-term query matching >= 2 terms in a chunk must retrieve it."""
        store = KnowledgeStore(tmp_path / "knowledge")
        store.upsert_document(
            "sec1",
            "security",
            "Security Audit",
            "Quarterly security audit report for cloud infrastructure access control.",
        )
        store.upsert_document(
            "gen1",
            "general",
            "General Notice",
            "General office policy and parking access rules.",
        )

        # Query: "security audit report access"
        # sec1 matches 4 terms: security, audit, report, access (>= 2) -> retrieved
        # gen1 matches 1 term: access (< 2) -> filtered out
        packet = store.retrieve("security audit report access", top_k=4)
        assert len(packet.items) == 1
        assert packet.items[0].document_id == "sec1"

    def test_match_across_folder_title_and_body_counts_towards_threshold(self, tmp_path):
        """Terms matching in folder name, title, or body all contribute to specificity count."""
        store = KnowledgeStore(tmp_path / "knowledge")
        store.upsert_document(
            "doc_xyz",
            "finance",
            "Quarterly Ledger",
            "All expenses are audited annually by external partners.",
        )

        # Query: "finance ledger" (2 terms)
        # Folder is "finance" (term 1), Title is "Quarterly Ledger" (term 2).
        # Body contains neither, but combined searchable tokens contain both -> 2 matches >= 2.
        packet = store.retrieve("finance ledger", top_k=4)
        assert len(packet.items) == 1
        assert packet.items[0].document_id == "doc_xyz"

    @pytest.mark.parametrize(
        ("term", "content"),
        [
            ("kubernetes", "Kubernetes cluster operations manual."),
            ("7", "Project release version 7 is scheduled for June."),
            ("404", "HTTP error 404 indicates resource not found."),
            ("X", "Variable X denotes the unknown parameter."),
        ],
    )
    def test_single_term_queries_require_only_1_match(self, tmp_path, term: str, content: str):
        """Single-term queries (N=1) must successfully retrieve chunks matching that single term."""
        store = KnowledgeStore(tmp_path / "knowledge")
        store.upsert_document("doc1", "general", "Single Term Test", content)

        packet = store.retrieve(term, top_k=4)
        assert len(packet.items) == 1
        assert packet.items[0].document_id == "doc1"

    def test_duplicate_terms_in_query_deduplicated_for_threshold(self, tmp_path):
        """Repeated terms in query should be deduplicated, sizing N by unique terms."""
        store = KnowledgeStore(tmp_path / "knowledge")
        store.upsert_document("doc1", "general", "Python Guide", "Python is an interpreted language.")

        # Query with repeated term "python python python" -> unique term count = 1.
        # min_matched_terms = min(2, 1) = 1.
        packet = store.retrieve("python python python", top_k=4)
        assert len(packet.items) == 1
        assert packet.items[0].document_id == "doc1"

    def test_stopword_only_query_fallback(self, tmp_path):
        """Queries consisting solely of stopwords fallback to indexing stopwords."""
        store = KnowledgeStore(tmp_path / "knowledge")
        store.upsert_document("quotes", "literature", "Hamlet", "To be or not to be, that is the question.")

        packet = store.retrieve("to be", top_k=4)
        assert len(packet.items) == 1
        assert packet.items[0].document_id == "quotes"

    def test_stopword_single_term_query_fallback(self, tmp_path):
        """Single stopword query fallback (N=1) requires 1 match."""
        store = KnowledgeStore(tmp_path / "knowledge")
        store.upsert_document("doc1", "general", "Article", "The quick brown fox jumps over the lazy dog.")

        packet = store.retrieve("the", top_k=4)
        assert len(packet.items) == 1

    def test_mixed_stopword_and_content_terms_filters_stopwords(self, tmp_path):
        """Mixed query 'the quantum algorithm' strips 'the', requiring 2 matches between quantum and algorithm."""
        store = KnowledgeStore(tmp_path / "knowledge")
        store.upsert_document("doc1", "physics", "Quantum Physics", "The study of quantum mechanics and particles.")
        store.upsert_document("doc2", "cs", "Quantum Algorithms", "Quantum algorithm design for Grover search.")

        # Query: "the quantum algorithm" -> terms: ["quantum", "algorithm"] (N=2, min=2)
        # doc1 matches only "quantum" (1 match < 2) -> dropped
        # doc2 matches "quantum" and "algorithm" (2 matches >= 2) -> retrieved
        packet = store.retrieve("the quantum algorithm", top_k=4)
        assert len(packet.items) == 1
        assert packet.items[0].document_id == "doc2"

    def test_query_term_limit_budget_exceeded(self, tmp_path):
        """Queries exceeding 24 unique terms trigger query_term_limit coverage flag."""
        store = KnowledgeStore(tmp_path / "knowledge")
        store.upsert_document("doc1", "general", "Vocabulary", "word01 word02 word03 word04 word05.")

        terms = [f"term{i:02d}" for i in range(30)]
        long_query = " ".join(terms)
        packet = store.retrieve(long_query, top_k=4)

        assert "query_term_limit" in packet.coverage_reasons
        assert packet.coverage_limited is True
        assert len(packet.items) == 0

    def test_fts5_special_characters_escaping(self, tmp_path):
        """Adversarial query strings containing FTS5 operators or quotes do not cause syntax errors."""
        store = KnowledgeStore(tmp_path / "knowledge")
        store.upsert_document("doc1", "general", "Code Notes", "Testing AND OR NOT NEAR operators in SQL queries.")

        adversarial_queries = [
            '"""hello"""',
            'AND OR NOT NEAR * ()',
            'SELECT * FROM chunks WHERE "body" MATCH',
            'foo:bar AND body:test',
            "c++ programming in .net",
        ]
        for q in adversarial_queries:
            # Should execute cleanly without raising sqlite3.OperationalError
            packet = store.retrieve(q, top_k=4)
            assert isinstance(packet.items, tuple)


# ==============================================================================
# R4: Natural Sentence Structure in Sanitizer Tests
# ==============================================================================

class TestR4NaturalSanitizerPhrasing:
    """Stress tests for outcome promise replacements, dangling prepositions, spacing, and markdown."""

    @pytest.mark.parametrize(
        ("input_text", "expected_output"),
        [
            (
                "This updated software could help with database performance.",
                "This updated software addresses database performance.",
            ),
            (
                "Regular exercise may help with cardiovascular health.",
                "Regular exercise addresses cardiovascular health.",
            ),
            (
                "A specialist could assist in diagnosing rare conditions.",
                "A specialist addresses diagnosing rare conditions.",
            ),
            (
                "The team could assist with infrastructure deployment.",
                "The team addresses infrastructure deployment.",
            ),
            (
                "Our consultant may assist with regulatory compliance.",
                "Our consultant addresses regulatory compliance.",
            ),
            (
                "The tool may assist in identifying bottlenecks.",
                "The tool addresses identifying bottlenecks.",
            ),
            (
                "This configuration could help to prevent memory leaks.",
                "This configuration is intended to prevent memory leaks.",
            ),
            (
                "Refactoring may help to clarify modular boundaries.",
                "Refactoring is intended to clarify modular boundaries.",
            ),
            (
                "Automated linting could assist to catch syntax errors.",
                "Automated linting is intended to catch syntax errors.",
            ),
            (
                "Upgrading the cluster will improve request latency.",
                "Upgrading the cluster addresses request latency.",
            ),
            (
                "Tuning parameters may improve overall throughput.",
                "Tuning parameters addresses overall throughput.",
            ),
            (
                "Caching will reduce server load substantially.",
                "Caching addresses server load substantially.",
            ),
            (
                "Scaling replicas will increase concurrent capacity.",
                "Scaling replicas addresses concurrent capacity.",
            ),
            (
                "This unstable release could potentially crash the node.",
                "This unstable release may crash the node.",
            ),
        ],
    )
    def test_all_outcome_promise_patterns(self, input_text: str, expected_output: str):
        flags = {"advice_requested": False, "percent_requested": False, "legal_mode": False, "current_events": False}
        result = sanitize_output(input_text, flags, tier="strict")
        assert result == expected_output

    def test_compound_outcome_promises_in_single_sentence(self):
        flags = {"advice_requested": False}
        text = "This patch will reduce memory usage and will improve system stability."
        result = sanitize_output(text, flags, tier="strict")
        assert result == "This patch addresses memory usage and addresses system stability."

    def test_case_insensitive_outcome_promises(self):
        flags = {"advice_requested": False}
        text = "THE NEW POLICY WILL IMPROVE COMPLIANCE AND COULD HELP WITH AUDITS."
        result = sanitize_output(text, flags, tier="strict")
        assert "WILL IMPROVE" not in result
        assert "COULD HELP WITH" not in result
        assert "addresses" in result.lower()

    @pytest.mark.parametrize(
        ("input_text", "expected_output"),
        [
            ("The policy was agreed to .", "The policy was agreed."),
            ("A problem to deal with ,", "A problem to deal,"),
            ("An outcome to hope for ;", "An outcome to hope;"),
            ("A detail to inquire about :", "A detail to inquire:"),
            ("A place to live in !", "A place to live!"),
            ("Something to look at ?", "Something to look?"),
            ("The car he was picked up by .", "The car he was picked up."),
            ("The station they arrived at .", "The station they arrived."),
            ("The airport we departed from .", "The airport we departed."),
        ],
    )
    def test_dangling_prepositions_before_punctuation(self, input_text: str, expected_output: str):
        flags = {"advice_requested": False}
        result = sanitize_output(input_text, flags, tier="strict")
        assert result == expected_output

    def test_punctuation_spacing_and_colliding_cleanup(self):
        flags = {"advice_requested": False}
        text = "Item one  .  Item two  ,  item three  ;  item four  :  item five  !  End  ? "
        result = sanitize_output(text, flags, tier="strict")
        assert result == "Item one. Item two, item three; item four: item five! End?"

    def test_colliding_punctuation_combinations(self):
        flags = {"advice_requested": False}
        text = "Clause one ,. Clause two ... Clause three ,,, Clause four . ."
        result = sanitize_output(text, flags, tier="strict")
        assert result == "Clause one. Clause two. Clause three, Clause four."

    def test_markdown_tables_preserved(self):
        flags = {"advice_requested": False}
        table = (
            "| Service | Status | Latency |\n"
            "| --- | --- | --- |\n"
            "| Auth | Healthy | 12ms |\n"
            "| Storage | Degraded | 145ms |"
        )
        result = sanitize_output(table, flags, tier="strict")
        assert result == table

    def test_markdown_tables_with_colons_preserved_as_valid_markdown(self):
        flags = {"advice_requested": False}
        table = (
            "| Service | Status | Latency |\n"
            "| :--- | :---: | ---: |\n"
            "| Auth | Healthy | 12ms |\n"
            "| Storage | Degraded | 145ms |"
        )
        result = sanitize_output(table, flags, tier="strict")
        # Table structure, headers, cells and delimiter lines remain intact
        assert "| Service | Status | Latency |" in result
        assert "| Auth | Healthy | 12ms |" in result
        assert "| Storage | Degraded | 145ms |" in result

    def test_markdown_headings_and_lists_preserved(self):
        flags = {"advice_requested": False}
        doc = (
            "# System Architecture\n\n"
            "## Key Components\n\n"
            "- Ingestion pipeline\n"
            "- Validation stage\n"
            "- Evaluation engine\n\n"
            "1. First step\n"
            "2. Second step"
        )
        result = sanitize_output(doc, flags, tier="strict")
        assert "# System Architecture" in result
        assert "## Key Components" in result
        assert "- Ingestion pipeline" in result
        assert "1. First step" in result

    def test_bare_percent_replacement_leaves_clean_punctuation(self):
        flags = {"advice_requested": False, "percent_requested": True}
        text = "The measured error rate was roughly 45% ."
        result = sanitize_output(text, flags, tier="strict")
        assert "45%" not in result
        assert "Unknown(Actionable): No authoritative dataset available for this figure." in result
        assert " ." not in result
        assert ".." not in result

    def test_advice_requested_flag_preserves_outcome_phrases(self):
        """When user asks for advice, outcome phrases are kept for downstream contextual evaluation."""
        flags = {"advice_requested": True}
        text = "You should exercise because it could help with stress and will improve mood."
        result = sanitize_output(text, flags, tier="strict")
        assert "could help with" in result
        assert "will improve" in result

    def test_tier_light_skips_outcome_stripping_but_cleans_punctuation(self):
        """Tier 'light' skips rule replacements but normalizes spacing and punctuation."""
        flags = {"advice_requested": False}
        text = "This tool will improve speed  . "
        result = sanitize_output(text, flags, tier="light")
        assert "will improve" in result
        assert result == "This tool will improve speed."


# ==============================================================================
# R5: Granular Stage-Level Credential Diagnostics Tests
# ==============================================================================

class TestR5GranularCredentialDiagnostics:
    """Stress tests for identifying exact missing stage credentials."""

    class StubPacketStore:
        def __init__(self, packet):
            self.packet = packet

        def retrieve(self, query: str, top_k: int = 6):
            return self.packet

    def _make_populated_packet(self):
        from pipeline.knowledge_store import EvidenceItem, EvidencePacket
        item = EvidenceItem(
            evidence_id="ev_001",
            rank=1,
            retrieval_score=0.1,
            document_id="doc1",
            document_revision_id="a" * 64,
            folder="general",
            title="Title",
            relative_path="sources/doc1/versions/a.txt",
            source_sha256="a" * 64,
            chunk_sha256="b" * 64,
            start_char=0,
            end_char=20,
            start_line=1,
            end_line=1,
            text="Evidence body text.",
        )
        return EvidencePacket(
            packet_id="c" * 64,
            corpus_revision="d" * 64,
            retrieval_version="sqlite-fts5-v2",
            canonical_query="test query",
            truncated=False,
            coverage_limited=False,
            coverage_reasons=(),
            items=(item,),
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("configured_stages", "expected_error_msg"),
        [
            # Single missing stage
            ({"gpt2": "k2", "gpt3": "k3"}, "Configure an API key for grounded stage 'gpt1' first."),
            ({"gpt1": "k1", "gpt3": "k3"}, "Configure an API key for grounded stage 'gpt2' first."),
            ({"gpt1": "k1", "gpt2": "k2"}, "Configure an API key for grounded stage 'gpt3' first."),
            # Two missing stages
            ({"gpt3": "k3"}, "Configure an API key for grounded stages 'gpt1, gpt2' first."),
            ({"gpt2": "k2"}, "Configure an API key for grounded stages 'gpt1, gpt3' first."),
            ({"gpt1": "k1"}, "Configure an API key for grounded stages 'gpt2, gpt3' first."),
            # All three missing
            ({}, "Configure an API key for grounded stages 'gpt1, gpt2, gpt3' first."),
        ],
    )
    async def test_all_missing_credential_permutations(
        self, monkeypatch, configured_stages: dict[str, str], expected_error_msg: str
    ):
        packet = self._make_populated_packet()
        store = self.StubPacketStore(packet)

        configs = {
            stage: {
                "provider": "openai",
                "api_key": configured_stages.get(stage, ""),
                "model": stage,
                "base_url": "",
            }
            for stage in ("gpt1", "gpt2", "gpt3")
        }
        monkeypatch.setattr(grounded.config, "get_stage_config", lambda stage: configs[stage])

        with pytest.raises(PipelineError) as exc_info:
            await run_grounded_rag(
                GroundedQueryRequest(prompt="Valid query with matching evidence"),
                store=store,
            )

        assert exc_info.value.status_code == 400
        assert str(exc_info.value) == expected_error_msg

    @pytest.mark.asyncio
    async def test_none_or_empty_api_key_detected_as_missing(self, monkeypatch):
        """None value or empty string for api_key triggers stage diagnostic."""
        packet = self._make_populated_packet()
        store = self.StubPacketStore(packet)

        configs = {
            "gpt1": {"provider": "openai", "api_key": None, "model": "gpt1", "base_url": ""},
            "gpt2": {"provider": "openai", "api_key": "", "model": "gpt2", "base_url": ""},
            "gpt3": {"provider": "openai", "api_key": "valid_key", "model": "gpt3", "base_url": ""},
        }
        monkeypatch.setattr(grounded.config, "get_stage_config", lambda stage: configs[stage])

        with pytest.raises(PipelineError) as exc_info:
            await run_grounded_rag(
                GroundedQueryRequest(prompt="Valid query with matching evidence"),
                store=store,
            )

        assert exc_info.value.status_code == 400
        assert str(exc_info.value) == "Configure an API key for grounded stages 'gpt1, gpt2' first."

    @pytest.mark.asyncio
    async def test_empty_retrieval_bypasses_credential_check_cleanly(self, monkeypatch, tmp_path):
        """When retrieval has 0 items, pipeline abstains immediately without checking API keys."""
        store = KnowledgeStore(tmp_path / "knowledge")
        # Empty knowledge store -> retrieval yields 0 items
        configs = {
            stage: {"provider": "openai", "api_key": "", "model": stage, "base_url": ""}
            for stage in ("gpt1", "gpt2", "gpt3")
        }
        monkeypatch.setattr(grounded.config, "get_stage_config", lambda stage: configs[stage])

        # Must NOT raise PipelineError(400) because it abstains at retrieval stage
        resp = await run_grounded_rag(
            GroundedQueryRequest(prompt="Query with empty corpus"),
            store=store,
        )
        assert resp.status == "ABSTAIN"
        assert resp.reason_code == "no_lexical_match"
        assert resp.retrieval_count == 0

    @pytest.mark.asyncio
    async def test_missing_stage_dict_keys_triggers_diagnostic(self, monkeypatch):
        """If stage_config dict completely lacks 'api_key' key, it is detected as missing."""
        packet = self._make_populated_packet()
        store = self.StubPacketStore(packet)

        configs = {
            "gpt1": {"provider": "openai", "model": "gpt1"},  # missing "api_key" key
            "gpt2": {"provider": "openai", "api_key": "valid_key", "model": "gpt2"},
            "gpt3": {"provider": "openai", "api_key": "valid_key", "model": "gpt3"},
        }
        monkeypatch.setattr(grounded.config, "get_stage_config", lambda stage: configs[stage])

        with pytest.raises(PipelineError) as exc_info:
            await run_grounded_rag(
                GroundedQueryRequest(prompt="Valid query with matching evidence"),
                store=store,
            )

        assert exc_info.value.status_code == 400
        assert str(exc_info.value) == "Configure an API key for grounded stage 'gpt1' first."


class TestR3ExtendedEdgeCases:
    """Extended edge cases for query tokenization, unicode, and multi-threading."""

    def test_query_punctuation_glued_tokens_extracted(self, tmp_path):
        """Punctuation glued to tokens (e.g. 'python, rust; golang!') is cleaned into separate tokens."""
        store = KnowledgeStore(tmp_path / "knowledge")
        store.upsert_document("doc1", "dev", "Languages", "We use python and rust in production services.")

        packet = store.retrieve("python, rust; golang!", top_k=4)
        # terms are ['python', 'rust', 'golang']. doc1 has 'python' and 'rust' -> 2 matches >= 2.
        assert len(packet.items) == 1
        assert packet.items[0].document_id == "doc1"

    def test_cjk_query_retrieval(self, tmp_path):
        """CJK and non-Latin query tokens participate properly in specificity cutoff."""
        store = KnowledgeStore(tmp_path / "knowledge")
        store.upsert_document("ja1", "i18n", "Japanese Guide", "システム アーキテクチャ 設計 書")
        store.upsert_document("ja2", "i18n", "Japanese Notes", "システム 概要")

        # Query: "システム アーキテクチャ" (2 terms)
        # ja1 has both terms (2 matches >= 2) -> retrieved
        # ja2 has only "システム" (1 match < 2) -> dropped
        packet = store.retrieve("システム アーキテクチャ", top_k=4)
        assert len(packet.items) == 1
        assert packet.items[0].document_id == "ja1"

    def test_whitespace_and_empty_queries_raise_value_error(self, tmp_path):
        """Empty or whitespace queries must raise ValueError."""
        store = KnowledgeStore(tmp_path / "knowledge")
        with pytest.raises(ValueError):
            store.retrieve("")
        with pytest.raises(ValueError):
            store.retrieve("   \t\n  ")

    def test_concurrent_retrieval_stress(self, tmp_path):
        """Concurrent retrieval across multiple threads maintains deterministic results."""
        import concurrent.futures

        store = KnowledgeStore(tmp_path / "knowledge")
        store.upsert_document("doc1", "general", "Concurrent Test", "Alpha beta gamma delta epsilon.")
        store.upsert_document("doc2", "general", "Concurrent Test 2", "Zeta eta theta iota kappa.")

        def run_search(query: str):
            return store.retrieve(query, top_k=4)

        queries = [
            "alpha beta",
            "zeta eta",
            "gamma delta",
            "theta iota",
            "alpha zeta",  # 1 in doc1, 1 in doc2 -> 0 items
        ] * 10

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(run_search, queries))

        assert len(results) == 50
        for i, packet in enumerate(results):
            q = queries[i]
            if q == "alpha zeta":
                assert len(packet.items) == 0
            else:
                assert len(packet.items) == 1


class TestR4ExtendedSanitizerEdgeCases:
    """Extended edge cases for sanitizer token boundaries and formatting."""

    def test_substring_safety_does_not_corrupt_unrelated_words(self):
        """Words containing outcome substrings (e.g. improvement, unhelpful) are not corrupted."""
        flags = {"advice_requested": False}
        text = "The improvement was noticeable. An unhelpful error occurred. Do not diminish our efforts."
        result = sanitize_output(text, flags, tier="strict")
        assert "improvement" in result
        assert "unhelpful" in result
        assert "diminish" in result

    def test_html_and_markdown_links_preserved(self):
        """HTML formatting and Markdown links are preserved without breaking."""
        flags = {"advice_requested": False}
        text = (
            "<p>Check the documentation at <a href='https://example.com'>Docs</a>.</p>\n"
            "[Official Guide](https://example.com/guide) will improve onboarding."
        )
        result = sanitize_output(text, flags, tier="strict")
        assert "<p>Check the documentation" in result
        assert "[Official Guide](https://example.com/guide) addresses onboarding." in result

    def test_multi_sentence_complex_flow(self):
        """A full multi-paragraph response with mixed rules cleans naturally."""
        flags = {"advice_requested": False, "percent_requested": True}
        text = (
            "First, this tool could help with performance .\n\n"
            "Second, the estimated success rate is 90% .\n\n"
            "Finally, the team was assisted by .\n"
        )
        result = sanitize_output(text, flags, tier="strict")
        assert "could help with" not in result
        assert "90%" not in result
        assert " ." not in result
        assert "First, this tool addresses performance." in result
        assert "Finally, the team was assisted." in result

