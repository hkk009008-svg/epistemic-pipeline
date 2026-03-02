"""User feedback collection for pipeline verification results.

Stores feedback in memory (resets on restart). Each feedback entry
links a user's assessment to a pipeline run for evaluation purposes.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class FeedbackEntry:
    """A single user feedback submission."""
    feedback_id: str
    request_id: str = ""
    prompt: str = ""
    rating: str = ""  # "accurate", "inaccurate", "partially_accurate"
    verdict_correct: Optional[bool] = None  # Was the PASS/FAIL correct?
    confidence_correct: Optional[bool] = None  # Was the confidence label right?
    comment: str = ""
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


class FeedbackStore:
    """Thread-safe in-memory feedback storage."""

    def __init__(self, max_entries: int = 1000):
        self._entries: list[FeedbackEntry] = []
        self._lock = threading.Lock()
        self._max_entries = max_entries

    def add(self, entry: FeedbackEntry) -> None:
        """Add a feedback entry."""
        with self._lock:
            if len(self._entries) >= self._max_entries:
                self._entries.pop(0)  # FIFO eviction
            if entry.timestamp == 0.0:
                entry.timestamp = time.time()
            self._entries.append(entry)

    def get_all(self) -> list[dict]:
        """Return all feedback entries as dicts."""
        with self._lock:
            return [e.to_dict() for e in self._entries]

    def get_summary(self) -> dict:
        """Return aggregate feedback statistics."""
        with self._lock:
            total = len(self._entries)
            if total == 0:
                return {
                    "total_feedback": 0,
                    "accurate_count": 0,
                    "inaccurate_count": 0,
                    "partially_accurate_count": 0,
                    "verdict_correct_count": 0,
                    "verdict_incorrect_count": 0,
                }

            accurate = sum(1 for e in self._entries if e.rating == "accurate")
            inaccurate = sum(1 for e in self._entries if e.rating == "inaccurate")
            partial = sum(1 for e in self._entries if e.rating == "partially_accurate")
            verdict_correct = sum(1 for e in self._entries if e.verdict_correct is True)
            verdict_incorrect = sum(1 for e in self._entries if e.verdict_correct is False)

            return {
                "total_feedback": total,
                "accurate_count": accurate,
                "inaccurate_count": inaccurate,
                "partially_accurate_count": partial,
                "verdict_correct_count": verdict_correct,
                "verdict_incorrect_count": verdict_incorrect,
            }

    def count(self) -> int:
        with self._lock:
            return len(self._entries)


# Global singleton
_store = FeedbackStore()


def get_feedback_store() -> FeedbackStore:
    """Return the global feedback store."""
    return _store
