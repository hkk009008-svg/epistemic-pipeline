"""Tests for user feedback collection."""
from __future__ import annotations

from pipeline.feedback import FeedbackEntry, FeedbackStore


class TestFeedbackEntry:
    def test_to_dict(self):
        entry = FeedbackEntry(feedback_id="abc123", rating="accurate", comment="Good")
        d = entry.to_dict()
        assert d["feedback_id"] == "abc123"
        assert d["rating"] == "accurate"
        assert d["comment"] == "Good"

    def test_defaults(self):
        entry = FeedbackEntry(feedback_id="x")
        assert entry.request_id == ""
        assert entry.verdict_correct is None
        assert entry.timestamp == 0.0


class TestFeedbackStore:
    def test_add_and_count(self):
        store = FeedbackStore()
        store.add(FeedbackEntry(feedback_id="1", rating="accurate"))
        store.add(FeedbackEntry(feedback_id="2", rating="inaccurate"))
        assert store.count() == 2

    def test_get_all(self):
        store = FeedbackStore()
        store.add(FeedbackEntry(feedback_id="1", rating="accurate"))
        entries = store.get_all()
        assert len(entries) == 1
        assert entries[0]["feedback_id"] == "1"

    def test_timestamp_auto_set(self):
        store = FeedbackStore()
        entry = FeedbackEntry(feedback_id="1", rating="accurate")
        store.add(entry)
        assert entry.timestamp > 0

    def test_max_entries_eviction(self):
        store = FeedbackStore(max_entries=3)
        for i in range(5):
            store.add(FeedbackEntry(feedback_id=str(i), rating="accurate"))
        assert store.count() == 3
        entries = store.get_all()
        # Oldest entries should have been evicted
        assert entries[0]["feedback_id"] == "2"
        assert entries[-1]["feedback_id"] == "4"

    def test_summary_empty(self):
        store = FeedbackStore()
        summary = store.get_summary()
        assert summary["total_feedback"] == 0
        assert summary["accurate_count"] == 0

    def test_summary_counts(self):
        store = FeedbackStore()
        store.add(FeedbackEntry(feedback_id="1", rating="accurate", verdict_correct=True))
        store.add(FeedbackEntry(feedback_id="2", rating="inaccurate", verdict_correct=False))
        store.add(FeedbackEntry(feedback_id="3", rating="partially_accurate", verdict_correct=True))
        summary = store.get_summary()
        assert summary["total_feedback"] == 3
        assert summary["accurate_count"] == 1
        assert summary["inaccurate_count"] == 1
        assert summary["partially_accurate_count"] == 1
        assert summary["verdict_correct_count"] == 2
        assert summary["verdict_incorrect_count"] == 1

    def test_summary_none_verdicts(self):
        store = FeedbackStore()
        store.add(FeedbackEntry(feedback_id="1", rating="accurate"))
        summary = store.get_summary()
        assert summary["verdict_correct_count"] == 0
        assert summary["verdict_incorrect_count"] == 0
