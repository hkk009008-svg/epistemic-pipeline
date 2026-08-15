"""Tests for pipeline/prompts.py — build_augmentation tier and format behavior.

All functions under test are fully deterministic (no LLM calls).
"""
from __future__ import annotations

from pipeline.prompts import DEFAULT_GPT1_SYSTEM, build_augmentation

_ALL_FALSE_FLAGS = {
    "advice_requested": False,
    "percent_requested": False,
    "legal_mode": False,
    "jurisdiction_present": False,
    "future_year": False,
    "current_events": False,
    "comparative": False,
}


# ---------------------------------------------------------------------------
# Clause-Isolated Generation Schema
# ---------------------------------------------------------------------------


class TestClauseIsolatedSchema:
    """Verify DEFAULT_GPT1_SYSTEM includes explicit clause-isolation instructions."""

    def test_default_gpt1_system_contains_clause_isolation_rules(self):
        assert "Clause-Isolated Generation Rules" in DEFAULT_GPT1_SYSTEM
        assert "Atomic Propositions" in DEFAULT_GPT1_SYSTEM
        assert "Prohibit Compound Conjunction Chaining" in DEFAULT_GPT1_SYSTEM
        assert "whereas" in DEFAULT_GPT1_SYSTEM.lower()
        assert "while" in DEFAULT_GPT1_SYSTEM.lower()
        assert "and" in DEFAULT_GPT1_SYSTEM.lower()
        assert "but" in DEFAULT_GPT1_SYSTEM.lower()

    def test_default_gpt1_system_preserves_priority_stack(self):
        assert "Priority Stack" in DEFAULT_GPT1_SYSTEM
        assert "V1 Abstention" in DEFAULT_GPT1_SYSTEM
        assert "V2 Evidence-Boundedness" in DEFAULT_GPT1_SYSTEM


# ---------------------------------------------------------------------------
# build_augmentation — Tier-specific augmentation
# ---------------------------------------------------------------------------


class TestBuildAugmentationTier:
    """Verify tier-specific augmentation text in build_augmentation."""

    def test_strict_no_tier_augmentation(self):
        """Strict tier adds no TIER block (it's the default)."""
        gpt1, gpt2, gpt3 = build_augmentation(
            _ALL_FALSE_FLAGS, tier="strict", output_format="structured"
        )
        assert "TIER" not in gpt1
        assert "TIER" not in gpt2
        assert "TIER" not in gpt3

    def test_standard_adds_tier_context(self):
        gpt1, gpt2, gpt3 = build_augmentation(
            _ALL_FALSE_FLAGS, tier="standard", output_format="structured"
        )
        assert "TIER" in gpt2
        assert "soft" in gpt2.lower()
        assert "TIER" in gpt3

    def test_light_adds_tier_context(self):
        gpt1, gpt2, gpt3 = build_augmentation(
            _ALL_FALSE_FLAGS, tier="light", output_format="structured"
        )
        assert "TIER" in gpt1
        assert "TIER" in gpt2
        assert "TIER" in gpt3
        assert "fact-check" in gpt1.lower()

    def test_default_tier_is_strict(self):
        """Calling without tier matches strict behavior (no TIER block)."""
        gpt1_default, gpt2_default, gpt3_default = build_augmentation(_ALL_FALSE_FLAGS)
        gpt1_strict, gpt2_strict, gpt3_strict = build_augmentation(
            _ALL_FALSE_FLAGS, tier="strict"
        )
        assert gpt1_default == gpt1_strict
        assert gpt2_default == gpt2_strict
        assert gpt3_default == gpt3_strict


# ---------------------------------------------------------------------------
# build_augmentation — Output format instructions
# ---------------------------------------------------------------------------


class TestBuildAugmentationFormat:
    """Verify output format instructions in build_augmentation."""

    def test_structured_format_no_override(self):
        """Structured format adds no extra format instructions."""
        gpt1, gpt2, gpt3 = build_augmentation(
            _ALL_FALSE_FLAGS, tier="strict", output_format="structured"
        )
        assert "Format Override" not in gpt1
        assert "FORMAT NOTE" not in gpt2

    def test_annotated_format_instruction(self):
        gpt1, gpt2, gpt3 = build_augmentation(
            _ALL_FALSE_FLAGS, tier="standard", output_format="annotated"
        )
        assert "[verified]" in gpt1
        assert "ANNOTATED" in gpt2

    def test_concise_format_instruction(self):
        gpt1, gpt2, gpt3 = build_augmentation(
            _ALL_FALSE_FLAGS, tier="light", output_format="concise"
        )
        assert "CONCISE" in gpt1
        assert "CONCISE" in gpt2

    def test_format_independent_of_tier(self):
        """Format can be set independently of tier."""
        gpt1, gpt2, gpt3 = build_augmentation(
            _ALL_FALSE_FLAGS, tier="strict", output_format="concise"
        )
        assert "CONCISE" in gpt1
        assert "CONCISE" in gpt2

    def test_annotated_with_light_tier(self):
        """annotated format + light tier should include both."""
        gpt1, gpt2, gpt3 = build_augmentation(
            _ALL_FALSE_FLAGS, tier="light", output_format="annotated"
        )
        assert "[verified]" in gpt1
        assert "TIER" in gpt2
