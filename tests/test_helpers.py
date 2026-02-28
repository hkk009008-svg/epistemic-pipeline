"""Tests for pipeline.helpers -- extract_json() and is_activation_phrase().

All functions under test are fully deterministic (no LLM calls).
"""
from __future__ import annotations

import json

import pytest

from pipeline.helpers import extract_json, is_activation_phrase


# ===================================================================
# extract_json() -- clean JSON
# ===================================================================


class TestExtractJsonClean:
    """Properly formatted JSON should parse directly."""

    def test_simple_object(self):
        raw = '{"key": "value"}'
        assert extract_json(raw) == {"key": "value"}

    def test_nested_object(self):
        raw = json.dumps({"a": {"b": [1, 2, 3]}, "c": True})
        result = extract_json(raw)
        assert result["a"]["b"] == [1, 2, 3]
        assert result["c"] is True

    def test_object_with_whitespace(self):
        raw = '  \n  {"key": "value"}  \n  '
        assert extract_json(raw) == {"key": "value"}

    def test_empty_object(self):
        assert extract_json("{}") == {}

    def test_complex_values(self):
        data = {
            "claim_table": [
                {"claim": "X", "category": "Supported", "justification": "Y"}
            ],
            "findings": [],
            "verdict": "PASS",
        }
        raw = json.dumps(data)
        assert extract_json(raw) == data


# ===================================================================
# extract_json() -- markdown code fences
# ===================================================================


class TestExtractJsonMarkdownFences:
    """JSON wrapped in markdown code fences."""

    def test_json_fence(self):
        inner = '{"key": "value"}'
        raw = f"```json\n{inner}\n```"
        assert extract_json(raw) == {"key": "value"}

    def test_plain_fence(self):
        inner = '{"key": "value"}'
        raw = f"```\n{inner}\n```"
        assert extract_json(raw) == {"key": "value"}

    def test_fence_with_extra_whitespace(self):
        inner = '{"key": "value"}'
        raw = f"```json\n  {inner}  \n```"
        assert extract_json(raw) == {"key": "value"}

    def test_fence_with_prose_before(self):
        inner = '{"key": "value"}'
        raw = f"Here is the JSON output:\n```json\n{inner}\n```"
        assert extract_json(raw) == {"key": "value"}

    def test_fence_with_prose_after(self):
        inner = '{"key": "value"}'
        raw = f"```json\n{inner}\n```\nHope that helps!"
        assert extract_json(raw) == {"key": "value"}

    def test_nested_json_in_fence(self):
        data = {"a": {"b": [1, 2]}, "c": "d"}
        inner = json.dumps(data)
        raw = f"```json\n{inner}\n```"
        assert extract_json(raw) == data


# ===================================================================
# extract_json() -- prose wrapping
# ===================================================================


class TestExtractJsonProseWrapping:
    """JSON embedded in prose text (no fences)."""

    def test_prose_before_json(self):
        raw = 'Here is my analysis: {"key": "value"}'
        assert extract_json(raw) == {"key": "value"}

    def test_prose_after_json_not_recovered(self):
        """JSON followed by prose cannot be parsed: it starts with '{' so the
        prose-extraction branch is skipped, and json.loads fails on the trailing text."""
        with pytest.raises(ValueError, match="Could not parse JSON"):
            extract_json('{"key": "value"} That is my output.')

    def test_prose_surrounding_json(self):
        raw = 'Analysis: {"verdict": "PASS"} End of response.'
        assert extract_json(raw) == {"verdict": "PASS"}

    def test_multiline_prose_and_json(self):
        raw = (
            "I have analyzed the claims.\n"
            'Here is the result:\n'
            '{"claim_table": [], "findings": [], "verdict": "PASS"}\n'
            "Let me know if you need more."
        )
        result = extract_json(raw)
        assert result["verdict"] == "PASS"


# ===================================================================
# extract_json() -- truncated JSON recovery
# ===================================================================


class TestExtractJsonTruncated:
    """Truncated JSON that needs bracket-closing recovery."""

    def test_missing_closing_brace(self):
        raw = '{"key": "value"'
        result = extract_json(raw)
        assert result == {"key": "value"}

    def test_missing_closing_bracket_and_brace(self):
        raw = '{"items": [1, 2, 3]'
        result = extract_json(raw)
        assert result == {"items": [1, 2, 3]}

    def test_missing_value_and_brackets(self):
        """Truncated after a value -- closing with '}' recovers."""
        raw = '{"items": [1, 2, 3], "status": "ok"'
        result = extract_json(raw)
        assert result["items"] == [1, 2, 3]
        assert result["status"] == "ok"

    def test_missing_bracket_and_brace(self):
        """Truncated after a complete inner object -- ']}' recovers."""
        raw = '{"items": [{"a": "b"}'
        result = extract_json(raw)
        assert result["items"][0]["a"] == "b"

    def test_deeply_truncated_unrepairable(self):
        """JSON so badly truncated that none of the suffix fixes work."""
        raw = '{"a": {"b": {"c": [1, 2, '
        with pytest.raises(ValueError, match="Could not parse JSON"):
            extract_json(raw)


# ===================================================================
# extract_json() -- error cases
# ===================================================================


class TestExtractJsonErrors:
    """Inputs that cannot be parsed as JSON at all."""

    def test_plain_text(self):
        with pytest.raises(ValueError, match="Could not parse JSON"):
            extract_json("This is just plain text with no JSON at all.")

    def test_empty_string(self):
        with pytest.raises(ValueError, match="Could not parse JSON"):
            extract_json("")

    def test_array_parses_as_list(self):
        """A bare JSON array is valid JSON. extract_json does not enforce
        dict-only output; json.loads('[1,2,3]') succeeds and returns a list."""
        result = extract_json("[1, 2, 3]")
        assert result == [1, 2, 3]

    def test_xml_not_json(self):
        with pytest.raises(ValueError, match="Could not parse JSON"):
            extract_json("<root><item>value</item></root>")


# ===================================================================
# is_activation_phrase()
# ===================================================================


class TestIsActivationPhrase:
    """Detect activation/init phrases that should bypass GPT-2 verification."""

    @pytest.mark.parametrize(
        "text",
        [
            "Epistemic Pipeline active.",
            "Production active.",
            "Audit v1",
            "Audit v2",
            "Audit v12",
            "System initialized",
            "System initialized and ready.",
            "Ready.",
        ],
    )
    def test_activation_phrases_detected(self, text: str):
        assert is_activation_phrase(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "This is a normal response about the economy.",
            "The answer to your question involves several factors.",
            "Let me analyze the legal implications of this scenario.",
            "",
        ],
    )
    def test_non_activation_phrases_not_detected(self, text: str):
        assert is_activation_phrase(text) is False


class TestIsActivationPhraseLengthGuard:
    """Responses >= 100 characters are never activation phrases."""

    def test_long_text_with_activation_keyword(self):
        # Contains "active." but is too long
        text = "This is a very long response. " * 5 + "Production active."
        assert len(text) >= 100
        assert is_activation_phrase(text) is False

    def test_exactly_100_chars_with_activation_keyword(self):
        # Build a string of exactly 100 chars that contains an activation pattern
        base = "Production active."
        padding = "x" * (100 - len(base))
        text = padding + base
        assert len(text.strip()) == 100
        assert is_activation_phrase(text) is False

    def test_99_chars_with_activation_keyword(self):
        # Build a string of exactly 99 chars that contains an activation pattern
        base = "Production active."
        padding = "x" * (99 - len(base))
        text = padding + base
        assert len(text.strip()) == 99
        assert is_activation_phrase(text) is True


class TestIsActivationPhrasePatternDetails:
    """Verify each ACTIVATION_PATTERN individually."""

    def test_active_dot_at_end(self):
        """Pattern: r'active\\.$' -- must end with 'active.'"""
        assert is_activation_phrase("System active.") is True
        assert is_activation_phrase("active.") is True
        # Does not match without trailing dot
        assert is_activation_phrase("System active") is False

    def test_production_active(self):
        """Pattern: r'Production active\\.'"""
        assert is_activation_phrase("Production active.") is True

    def test_audit_version(self):
        """Pattern: r'Audit v\\d+'"""
        assert is_activation_phrase("Audit v1") is True
        assert is_activation_phrase("Audit v99") is True
        assert is_activation_phrase("Audit v") is False

    def test_system_initialized(self):
        """Pattern: r'^System initialized'"""
        assert is_activation_phrase("System initialized") is True
        assert is_activation_phrase("System initialized.") is True
        # Anchor ^ means it must start with it
        assert is_activation_phrase("The System initialized") is False

    def test_ready_dot(self):
        """Pattern: r'^Ready\\.$'"""
        assert is_activation_phrase("Ready.") is True
        assert is_activation_phrase("Ready") is False
        assert is_activation_phrase("Not Ready.") is False


class TestIsActivationPhraseCaseInsensitive:
    """Patterns are matched case-insensitively (re.IGNORECASE)."""

    def test_lowercase_active(self):
        assert is_activation_phrase("production active.") is True

    def test_uppercase_ready(self):
        assert is_activation_phrase("READY.") is True

    def test_mixed_case_audit(self):
        assert is_activation_phrase("audit V3") is True

    def test_mixed_case_system(self):
        assert is_activation_phrase("system initialized") is True
