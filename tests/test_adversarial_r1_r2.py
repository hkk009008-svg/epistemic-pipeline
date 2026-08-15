"""Adversarial stress-testing suite for Requirement R1 and Requirement R2.

Covers edge cases, boundary conditions, malformed inputs, multi-citation isolation,
table continuation chunk propagation across multiple chunks, alignment colons,
multiple tables per document, tamper detection, scale multipliers, and currencies.
"""
from __future__ import annotations

import sqlite3

import pytest

from pipeline.knowledge_store import (
    KnowledgeStore,
    StaleKnowledgeIndexError,
    chunk_text,
)
from pipeline.models import SearchSource
from pipeline.source_match import (
    _extract_numbers,
    verify_citation_grounding,
)


def _src(title: str, snippet: str, url: str = "https://example.com") -> SearchSource:
    return SearchSource(title=title, url=url, snippet=snippet, score=0.5)


# ===========================================================================
# R1: Quantitative & Numeric Citation Verification Adversarial Tests
# ===========================================================================


class TestAdversarialR1UnbackedPercentages:
    """Stress-testing percentage extraction and verification."""

    def test_unbacked_percentage_single_citation(self):
        sources = [_src("Finance", "The company reported a 15% increase in annual recurring revenue.")]
        text = "The company reported a 45% increase in annual recurring revenue [1]."
        findings = verify_citation_grounding(text, sources)
        assert len(findings) == 1
        assert findings[0]["type"] == "T1"
        assert findings[0]["severity"] == "hard"
        assert "45" in findings[0]["detail"]

    def test_backed_percentage_various_notations(self):
        sources = [_src("Health", "The recovery rate reached 87.5 percent among patients.")]
        # Test % symbol matching 'percent'
        text1 = "Patients experienced an 87.5% recovery rate [1]."
        assert verify_citation_grounding(text1, sources) == []

        # Test 'pct' notation matching 'percent'
        text2 = "Patients experienced an 87.5 pct recovery rate [1]."
        assert verify_citation_grounding(text2, sources) == []

    def test_zero_and_hundred_percentages(self):
        sources = [_src("Performance", "System experienced 0% packet loss and 100% uptime.")]
        valid_text = "The network had 0% packet loss and 100% uptime [1]."
        assert verify_citation_grounding(valid_text, sources) == []

        invalid_text = "The network had 5% packet loss and 99% uptime [1]."
        findings = verify_citation_grounding(invalid_text, sources)
        assert len(findings) == 2
        details = [f["detail"] for f in findings]
        assert any("5" in d for d in details)
        assert any("99" in d for d in details)

    def test_sub_one_percent_decimals(self):
        sources = [_src("Chemistry", "Impurity levels were measured at 0.05% in the batch.")]
        valid_text = "Batch impurities remained below 0.05% [1]."
        assert verify_citation_grounding(valid_text, sources) == []

        invalid_text = "Batch impurities remained below 0.5% [1]."
        findings = verify_citation_grounding(invalid_text, sources)
        assert len(findings) == 1
        assert "0.5" in findings[0]["detail"]


class TestAdversarialR1CurrenciesAndSymbols:
    """Stress-testing global currency symbols, commas, and word forms."""

    def test_multi_currency_symbols(self):
        sources = [
            _src(
                "Global Pricing",
                "Regional costs: US $50, Europe €100, UK £75.50, Japan ¥5000, India ₹2500, Korea ₩10000.",
            )
        ]
        valid_text = "Global rates are $50, €100, £75.50, ¥5000, ₹2500, and ₩10000 [1]."
        assert verify_citation_grounding(valid_text, sources) == []

    def test_unbacked_currency_symbol_mismatch(self):
        sources = [_src("Pricing", "Standard membership is €100 per quarter.")]
        invalid_text = "Standard membership is €120 per quarter [1]."
        findings = verify_citation_grounding(invalid_text, sources)
        assert len(findings) == 1
        assert "120" in findings[0]["detail"]

    def test_currency_with_commas_and_cents(self):
        sources = [_src("Real Estate", "Median house price reached $1,250,000.75 in the metro area.")]
        valid_text = "The metro median house price was $1,250,000.75 [1]."
        assert verify_citation_grounding(valid_text, sources) == []

        invalid_text = "The metro median house price was $1,250,000.50 [1]."
        findings = verify_citation_grounding(invalid_text, sources)
        assert len(findings) == 1
        assert "1250000.5" in findings[0]["detail"]

    def test_currency_word_forms(self):
        sources = [_src("Finance", "The entry fee is fifty dollars.")]
        valid_text = "The entry fee is $50 [1]."
        assert verify_citation_grounding(valid_text, sources) == []

        valid_word_text = "The entry fee is fifty dollars [1]."
        assert verify_citation_grounding(valid_word_text, sources) == []

        invalid_word_text = "The entry fee is sixty dollars [1]."
        findings = verify_citation_grounding(invalid_word_text, sources)
        assert len(findings) == 1
        assert "60" in findings[0]["detail"]


class TestAdversarialR1ScaleMultipliers:
    """Stress-testing scale multipliers (k, M, B, T, thousand, million, billion)."""

    def test_million_scale_equivalence(self):
        sources = [_src("Valuation", "Startup valuation was estimated at $1.5M in Series A.")]
        valid_text1 = "Startup was valued at $1.5 million in Series A [1]."
        assert verify_citation_grounding(valid_text1, sources) == []

        valid_text2 = "Startup valuation reached $1500000 [1]."
        assert verify_citation_grounding(valid_text2, sources) == []

        valid_text3 = "Startup valuation reached 1.5M [1]."
        assert verify_citation_grounding(valid_text3, sources) == []

    def test_billion_and_trillion_scales(self):
        sources = [_src("Macroeconomics", "National debt surpassed $2 trillion, with $500 billion allocated.")]
        valid_text = "The national debt reached 2T with 500B allocated [1]."
        assert verify_citation_grounding(valid_text, sources) == []

        invalid_text = "The national debt reached 3 trillion with 500 billion allocated [1]."
        findings = verify_citation_grounding(invalid_text, sources)
        assert len(findings) >= 1
        assert any("3" in f["detail"] for f in findings)

    def test_unbacked_scale_order_of_magnitude(self):
        sources = [_src("Funding", "The project received $50k in seed grants.")]
        invalid_text = "The project received $50M in seed grants [1]."
        findings = verify_citation_grounding(invalid_text, sources)
        assert len(findings) >= 1
        # 50,000,000 is unbacked
        assert any("50000000" in f["detail"] for f in findings)


class TestAdversarialR1Durations:
    """Stress-testing duration timeframes (days, months, years)."""

    def test_hyphenated_vs_spaced_durations(self):
        sources = [_src("Warranty", "Product includes a 30-day return window.")]
        valid_text = "Customers are entitled to a 30 days return window [1]."
        assert verify_citation_grounding(valid_text, sources) == []

        invalid_text = "Customers are entitled to a 60-day return window [1]."
        findings = verify_citation_grounding(invalid_text, sources)
        assert len(findings) == 1
        assert "60" in findings[0]["detail"]

    def test_multi_month_durations(self):
        sources = [_src("Contract", "The initial agreement spans 12 months with 6 months extension.")]
        valid_text = "The contract duration is 12 months plus 6 months [1]."
        assert verify_citation_grounding(valid_text, sources) == []

        invalid_text = "The contract duration is 24 months [1]."
        findings = verify_citation_grounding(invalid_text, sources)
        assert len(findings) == 1
        assert "24" in findings[0]["detail"]


class TestAdversarialR1MixedMultiCitations:
    """Stress-testing sentences containing multiple citations with isolated numbers."""

    def test_two_citations_with_distinct_numbers_both_valid(self):
        sources = [
            _src("Plan Alpha", "Plan Alpha charges $50 per month."),
            _src("Plan Beta", "Plan Beta charges $100 per month."),
        ]
        text = "Plan Alpha is priced at $50 [1], whereas Plan Beta is priced at $100 [2]."
        assert verify_citation_grounding(text, sources) == []

    def test_two_citations_first_valid_second_unbacked(self):
        sources = [
            _src("Plan Alpha", "Plan Alpha charges $50 per month."),
            _src("Plan Beta", "Plan Beta charges $100 per month."),
        ]
        text = "Plan Alpha is priced at $50 [1], whereas Plan Beta is priced at $250 [2]."
        findings = verify_citation_grounding(text, sources)
        assert len(findings) == 1
        assert findings[0]["type"] == "T1"
        assert findings[0]["detail"].startswith("Unbacked numeric claim: [2]")
        assert "250" in findings[0]["detail"]

    def test_two_citations_first_unbacked_second_valid(self):
        sources = [
            _src("Plan Alpha", "Plan Alpha charges $50 per month."),
            _src("Plan Beta", "Plan Beta charges $100 per month."),
        ]
        text = "Plan Alpha is priced at $80 [1], whereas Plan Beta is priced at $100 [2]."
        findings = verify_citation_grounding(text, sources)
        assert len(findings) == 1
        assert findings[0]["type"] == "T1"
        assert findings[0]["detail"].startswith("Unbacked numeric claim: [1]")
        assert "80" in findings[0]["detail"]

    def test_conjunction_clause_boundaries(self):
        sources = [
            _src("Sensor A", "Sensor A recorded 40 degrees."),
            _src("Sensor B", "Sensor B recorded 80 degrees."),
        ]
        # Test 'while'
        text_while = "Sensor A reached 40 degrees [1], while Sensor B peaked at 95 degrees [2]."
        findings_while = verify_citation_grounding(text_while, sources)
        assert len(findings_while) == 1
        assert findings_while[0]["detail"].startswith("Unbacked numeric claim: [2]")
        assert "95" in findings_while[0]["detail"]

        # Test 'but'
        text_but = "Sensor A registered 40 degrees [1], but Sensor B showed 80 degrees [2]."
        assert verify_citation_grounding(text_but, sources) == []


class TestAdversarialR1ThreeWayCitations:
    """Stress-testing 3 or more citations in a complex sentence."""

    def test_three_citations_one_unbacked(self):
        sources = [
            _src("Source 1", "Metric A was 10."),
            _src("Source 2", "Metric B was 25."),
            _src("Source 3", "Metric C was 30."),
        ]
        # [1] has 10 (valid), [2] claims 20 (invalid: source has 25), [3] has 30 (valid)
        text = "Metric A was 10 [1], Metric B was 20 [2], whereas Metric C was 30 [3]."
        findings = verify_citation_grounding(text, sources)
        assert len(findings) == 1
        assert findings[0]["detail"].startswith("Unbacked numeric claim: [2]")
        assert "20" in findings[0]["detail"]


class TestAdversarialR1NegativeAndZeroNumbers:
    """Stress-testing zero, decimals, and negative values."""

    def test_zero_amount(self):
        sources = [_src("Account", "Transaction fee is zero dollars ($0) for members.")]
        valid_text = "The transaction fee is $0 [1]."
        assert verify_citation_grounding(valid_text, sources) == []

        invalid_text = "The transaction fee is $5 [1]."
        findings = verify_citation_grounding(invalid_text, sources)
        assert len(findings) == 1
        assert "5" in findings[0]["detail"]

    def test_decimal_precision(self):
        sources = [_src("Physics", "The physical constant was calibrated to 3.14159.")]
        valid_text = "The constant is 3.14159 [1]."
        assert verify_citation_grounding(valid_text, sources) == []

        invalid_text = "The constant is 3.14150 [1]."
        findings = verify_citation_grounding(invalid_text, sources)
        assert len(findings) == 1
        assert "3.1415" in findings[0]["detail"]


class TestAdversarialR1FalseCitationNumberRejection:
    """Ensure citation bracket indices [1], [2], etc. are never treated as claimed numbers."""

    def test_bracket_citations_do_not_fabricate_number_claims(self):
        sources = [_src("Overview", "The algorithm uses heuristic search with pruning.")]
        # Source has NO numbers. Text cites [1] with no numbers in prose.
        text = "The algorithm uses heuristic search with pruning [1]."
        findings = verify_citation_grounding(text, sources)
        assert findings == []

    def test_multiple_bracket_citations_no_numbers_in_source(self):
        sources = [
            _src("Part 1", "Alpha phase initiated."),
            _src("Part 2", "Beta phase concluded."),
        ]
        text = "Alpha phase was initiated [1], and beta phase concluded [2]."
        findings = verify_citation_grounding(text, sources)
        assert findings == []

    def test_extract_numbers_ignores_all_bracket_contents(self):
        nums = _extract_numbers("Citation [1] and [2] and [10] and [verified] and [note 5].")
        assert nums == set()


# ===========================================================================
# R2: Markdown Table Header Propagation in Chunking Adversarial Tests
# ===========================================================================


class TestAdversarialR2MultiChunkTables:
    """Stress-testing Markdown table chunking across 2, 3, 4, or more chunks."""

    def test_large_table_spanning_four_chunks(self):
        header = "| Item ID | Description | Category | Unit Price | In Stock |"
        delimiter = "| :--- | :--- | :---: | ---: | :---: |"
        rows = [
            f"| ITEM-{i:03d} | Product Description for item {i} | Category-{i % 5} | ${10 + i * 2}.00 | {100 - i} |"
            for i in range(80)
        ]
        content = f"# Inventory Catalog\n\n{header}\n{delimiter}\n" + "\n".join(rows)

        chunks = chunk_text(content, max_words=30, overlap_words=5)
        assert len(chunks) >= 4

        header_block = f"{header}\n{delimiter}"

        # Chunk 0 contains table header and first batch of rows
        assert header in chunks[0]["text"]
        assert delimiter in chunks[0]["text"]

        # Every subsequent continuation chunk containing table rows must start with the header block
        for idx, chunk in enumerate(chunks[1:], start=1):
            if "ITEM-" in chunk["text"]:
                assert chunk["text"].startswith(header_block), f"Chunk {idx} missing prepended header"

    def test_table_with_various_alignment_colons(self):
        header = "| LeftCol | RightCol | CenterCol | DefaultCol | Narrow |"
        delimiter = "| :--- | ---: | :---: | --- | :-: |"
        rows = [f"| L-{i} | R-{i} | C-{i} | D-{i} | N-{i} |" for i in range(40)]
        content = f"{header}\n{delimiter}\n" + "\n".join(rows)

        chunks = chunk_text(content, max_words=25, overlap_words=5)
        assert len(chunks) > 1

        header_block = f"{header}\n{delimiter}"
        for chunk in chunks:
            assert chunk["text"].startswith(header_block)

    def test_single_column_table(self):
        header = "| Server Node Name |"
        delimiter = "| :--- |"
        rows = [f"| production-worker-node-{i:02d}.infra.corp |" for i in range(35)]
        content = f"{header}\n{delimiter}\n" + "\n".join(rows)

        chunks = chunk_text(content, max_words=20, overlap_words=4)
        assert len(chunks) > 1

        header_block = f"{header}\n{delimiter}"
        for chunk in chunks:
            assert chunk["text"].startswith(header_block)

    def test_table_followed_by_regular_paragraphs(self):
        header = "| Metric | Baseline | Target |"
        delimiter = "| --- | --- | --- |"
        table_rows = [f"| KPI-{i} | {i * 10}ms | {i * 5}ms |" for i in range(30)]
        table_content = f"{header}\n{delimiter}\n" + "\n".join(table_rows)

        paragraphs = (
            "\n\n## Discussion Section\n\n"
            "This is a regular analytical paragraph discussing the performance characteristics.\n\n"
            "Another detailed paragraph explaining operational procedures and escalation paths without any table rows.\n\n"
            "Final conclusion paragraph summarizing key performance recommendations."
        )

        full_document = table_content + paragraphs

        chunks = chunk_text(full_document, max_words=30, overlap_words=5)
        assert len(chunks) > 2

        header_block = f"{header}\n{delimiter}"
        for chunk in chunks:
            if "KPI-" in chunk["text"]:
                assert header_block in chunk["text"]
            elif "Discussion Section" in chunk["text"] or "operational procedures" in chunk["text"]:
                # Non-table paragraph chunks must NOT have the table header prepended
                assert not chunk["text"].startswith(header_block)
                assert header not in chunk["text"]

    def test_multiple_tables_in_single_document(self):
        table1_header = "| User ID | Name | Role |"
        table1_delim = "| :--- | :--- | :--- |"
        table1_rows = [f"| U-{i:02d} | User Name {i} | Admin |" for i in range(25)]
        table1_text = f"## Users\n\n{table1_header}\n{table1_delim}\n" + "\n".join(table1_rows)

        intermission_paras = [
            f"Explanatory paragraph {k} providing background context and architectural descriptions "
            f"for system operations and security policies."
            for k in range(10)
        ]
        section_text = "\n\n## Intermission Section\n\n" + "\n\n".join(intermission_paras) + "\n\n"

        table2_header = "| Server ID | Hostname | IP Address | Status |"
        table2_delim = "| ---: | :--- | :---: | :--- |"
        table2_rows = [f"| S-{i:02d} | host-{i}.net | 192.168.1.{i} | Active |" for i in range(25)]
        table2_text = f"## Servers\n\n{table2_header}\n{table2_delim}\n" + "\n".join(table2_rows)

        document = table1_text + section_text + table2_text

        chunks = chunk_text(document, max_words=25, overlap_words=5)
        assert len(chunks) > 5

        for chunk in chunks:
            text = chunk["text"]
            # If chunk contains table 1 data rows and no table 2 data rows
            if "U-15" in text:
                assert text.startswith(f"{table1_header}\n{table1_delim}")
                assert table2_header not in text
            # If chunk contains table 2 data rows and no table 1 data rows
            elif "S-15" in text:
                assert text.startswith(f"{table2_header}\n{table2_delim}")
                assert table1_header not in text
            # If chunk is purely within the intermission text
            elif "Explanatory paragraph" in text and "U-" not in text and "S-" not in text and "Admin" not in text:
                assert not text.startswith(table1_header)
                assert not text.startswith(table2_header)


class TestAdversarialR2TamperVerification:
    """Stress-testing cryptographic and structural tamper resistance in _verify_sources."""

    def test_disk_source_byte_corruption_detected(self, tmp_path):
        store = KnowledgeStore(tmp_path / "knowledge")
        header = "| Code | Status | Message |"
        delimiter = "| --- | --- | --- |"
        rows = [f"| ERR_{i:03d} | Failure | Error detail for {i} |" for i in range(40)]
        content = f"{header}\n{delimiter}\n" + "\n".join(rows)

        record = store.upsert_document("errors", "system", "Error Codes", content)
        assert record.chunk_count > 1

        # Retrieval works initially
        packet = store.retrieve("ERR_035 Error detail", top_k=3)
        assert packet.items

        # Corrupt the immutable source file on disk
        source_file = store.root / record.relative_path
        source_file.write_text(content.replace("Failure", "Success"), encoding="utf-8")

        with pytest.raises(StaleKnowledgeIndexError):
            store.retrieve("ERR_035 Error detail", top_k=3)

    def test_sqlite_chunk_body_tampering_detected(self, tmp_path):
        store = KnowledgeStore(tmp_path / "knowledge")
        header = "| Code | Status | Message |"
        delimiter = "| --- | --- | --- |"
        rows = [f"| ERR_{i:03d} | Failure | Error detail for {i} |" for i in range(40)]
        content = f"{header}\n{delimiter}\n" + "\n".join(rows)

        store.upsert_document("errors", "system", "Error Codes", content)
        packet = store.retrieve("ERR_035 Error detail", top_k=3)
        assert packet.items
        target_item = packet.items[0]

        # Tamper chunk text directly in SQLite database while keeping the search keyword
        with sqlite3.connect(store.index_path) as conn:
            conn.execute(
                "UPDATE chunks SET body = ? WHERE evidence_id = ?",
                (f"{header}\n{delimiter}\n| ERR_035 | Tampered | Forged message |", target_item.evidence_id),
            )
            conn.commit()

        with pytest.raises(StaleKnowledgeIndexError):
            store.retrieve("ERR_035 Error detail", top_k=3)

    def test_sqlite_chunk_sha_tampering_detected(self, tmp_path):
        store = KnowledgeStore(tmp_path / "knowledge")
        header = "| Code | Status | Message |"
        delimiter = "| --- | --- | --- |"
        rows = [f"| ERR_{i:03d} | Failure | Error detail for {i} |" for i in range(40)]
        content = f"{header}\n{delimiter}\n" + "\n".join(rows)

        store.upsert_document("errors", "system", "Error Codes", content)
        packet = store.retrieve("ERR_035 Error detail", top_k=3)
        target_item = packet.items[0]

        # Tamper chunk_sha256 in SQLite database
        with sqlite3.connect(store.index_path) as conn:
            conn.execute(
                "UPDATE chunks SET chunk_sha256 = ? WHERE evidence_id = ?",
                ("0" * 64, target_item.evidence_id),
            )
            conn.commit()

        with pytest.raises(StaleKnowledgeIndexError):
            store.retrieve("ERR_035 Error detail", top_k=3)

    def test_sqlite_chunk_offsets_tampering_detected(self, tmp_path):
        store = KnowledgeStore(tmp_path / "knowledge")
        header = "| Code | Status | Message |"
        delimiter = "| --- | --- | --- |"
        rows = [f"| ERR_{i:03d} | Failure | Error detail for {i} |" for i in range(40)]
        content = f"{header}\n{delimiter}\n" + "\n".join(rows)

        store.upsert_document("errors", "system", "Error Codes", content)
        packet = store.retrieve("ERR_035 Error detail", top_k=3)
        target_item = packet.items[0]

        # Tamper start_char and end_char in SQLite database
        with sqlite3.connect(store.index_path) as conn:
            conn.execute(
                "UPDATE chunks SET start_char = start_char + 10 WHERE evidence_id = ?",
                (target_item.evidence_id,),
            )
            conn.commit()

        with pytest.raises(StaleKnowledgeIndexError):
            store.retrieve("ERR_035 Error detail", top_k=3)

    def test_sqlite_chunk_line_numbers_tampering_detected(self, tmp_path):
        store = KnowledgeStore(tmp_path / "knowledge")
        header = "| Code | Status | Message |"
        delimiter = "| --- | --- | --- |"
        rows = [f"| ERR_{i:03d} | Failure | Error detail for {i} |" for i in range(40)]
        content = f"{header}\n{delimiter}\n" + "\n".join(rows)

        store.upsert_document("errors", "system", "Error Codes", content)
        packet = store.retrieve("ERR_035 Error detail", top_k=3)
        target_item = packet.items[0]

        # Tamper start_line in SQLite database
        with sqlite3.connect(store.index_path) as conn:
            conn.execute(
                "UPDATE chunks SET start_line = start_line + 5 WHERE evidence_id = ?",
                (target_item.evidence_id,),
            )
            conn.commit()

        with pytest.raises(StaleKnowledgeIndexError):
            store.retrieve("ERR_035 Error detail", top_k=3)
