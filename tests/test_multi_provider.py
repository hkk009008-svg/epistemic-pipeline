"""Tests for per-stage model configuration and multi-provider support."""
from __future__ import annotations

from unittest.mock import patch

import pytest

import config
from pipeline.helpers import _make_client, call_llm


# ---------------------------------------------------------------------------
# config.set_stage_config / get_stage_config
# ---------------------------------------------------------------------------

class TestStageConfig:
    """Test per-stage configuration with fallback to global OpenAI config."""

    def setup_method(self):
        """Reset stage configs before each test."""
        with config._stage_lock:
            config._stage_configs.clear()

    def test_set_and_get_stage_config(self):
        config.set_stage_config("gpt1", "anthropic", "sk-ant-test", "claude-3-opus")
        cfg = config.get_stage_config("gpt1")
        assert cfg["provider"] == "anthropic"
        assert cfg["api_key"] == "sk-ant-test"
        assert cfg["model"] == "claude-3-opus"

    def test_fallback_to_global_when_no_stage_config(self):
        cfg = config.get_stage_config("gpt2")
        assert cfg["provider"] == "openai"
        assert cfg["model"] == config.get_model()

    def test_merge_when_stage_has_empty_key(self):
        config.set_stage_config("gpt1", "anthropic", "", "claude-3-opus")
        cfg = config.get_stage_config("gpt1")
        # Provider and model are merged from stage config (v2 merge semantics).
        # api_key falls back to global since stage key is empty.
        assert cfg["provider"] == "anthropic"
        assert cfg["model"] == "claude-3-opus"
        assert cfg["api_key"] == config.get_api_key()  # inherits global key

    def test_invalid_stage_raises(self):
        with pytest.raises(AssertionError, match="Invalid stage"):
            config.set_stage_config("gpt4", "openai", "key", "model")

    def test_invalid_provider_raises(self):
        with pytest.raises(AssertionError, match="Invalid provider"):
            config.set_stage_config("gpt1", "gemini", "key", "model")

    def test_all_three_stages_independent(self):
        config.set_stage_config("gpt1", "openai", "key1", "gpt-4o")
        config.set_stage_config("gpt2", "anthropic", "key2", "claude-3")
        config.set_stage_config("gpt3", "openrouter", "key3", "mix")

        cfg1 = config.get_stage_config("gpt1")
        cfg2 = config.get_stage_config("gpt2")
        cfg3 = config.get_stage_config("gpt3")

        assert cfg1["provider"] == "openai"
        assert cfg2["provider"] == "anthropic"
        assert cfg3["provider"] == "openrouter"

    def test_overwrite_stage_config(self):
        config.set_stage_config("gpt1", "openai", "key1", "gpt-4o")
        config.set_stage_config("gpt1", "anthropic", "key2", "claude-3")
        cfg = config.get_stage_config("gpt1")
        assert cfg["provider"] == "anthropic"
        assert cfg["api_key"] == "key2"

    def test_base_url_stored(self):
        config.set_stage_config("gpt1", "ollama", "ollama", "llama3",
                                base_url="http://localhost:11434/v1")
        cfg = config.get_stage_config("gpt1")
        assert cfg["base_url"] == "http://localhost:11434/v1"


# ---------------------------------------------------------------------------
# _make_client
# ---------------------------------------------------------------------------

class TestMakeClient:
    """Test client factory creates correct client types."""

    def test_openai_provider(self):
        provider_type, client = _make_client({
            "provider": "openai",
            "api_key": "sk-test",
            "model": "gpt-4o",
            "base_url": "",
        })
        assert provider_type == "openai"

    def test_anthropic_provider(self):
        provider_type, client = _make_client({
            "provider": "anthropic",
            "api_key": "sk-ant-test",
            "model": "claude-3",
            "base_url": "",
        })
        assert provider_type == "anthropic"

    def test_openrouter_uses_openai_type(self):
        provider_type, client = _make_client({
            "provider": "openrouter",
            "api_key": "or-test",
            "model": "mix",
            "base_url": "https://openrouter.ai/api/v1",
        })
        assert provider_type == "openai"

    def test_ollama_uses_openai_type(self):
        provider_type, client = _make_client({
            "provider": "ollama",
            "api_key": "ollama",
            "model": "llama3",
            "base_url": "http://localhost:11434/v1",
        })
        assert provider_type == "openai"


# ---------------------------------------------------------------------------
# call_llm dispatch
# ---------------------------------------------------------------------------

class TestCallLlmDispatch:
    """Test that call_llm dispatches to the correct provider."""

    @patch("pipeline.helpers.call_openai")
    def test_dispatches_to_openai(self, mock_call):
        mock_call.return_value = "response"
        cfg = {"provider": "openai", "api_key": "sk-test", "model": "gpt-4o", "base_url": ""}
        result = call_llm(cfg, "system", "user")
        assert result == "response"
        mock_call.assert_called_once()

    @patch("pipeline.helpers._call_anthropic")
    def test_dispatches_to_anthropic(self, mock_call):
        mock_call.return_value = "claude response"
        cfg = {"provider": "anthropic", "api_key": "sk-ant-test", "model": "claude-3", "base_url": ""}
        result = call_llm(cfg, "system", "user")
        assert result == "claude response"
        mock_call.assert_called_once()

    @patch("pipeline.helpers.call_openai")
    def test_openrouter_dispatches_to_openai(self, mock_call):
        mock_call.return_value = "or response"
        cfg = {"provider": "openrouter", "api_key": "or-test", "model": "mix",
               "base_url": "https://openrouter.ai/api/v1"}
        result = call_llm(cfg, "system", "user")
        assert result == "or response"


# ---------------------------------------------------------------------------
# StageConfig model
# ---------------------------------------------------------------------------

class TestStageConfigModel:
    """Test the StageConfig Pydantic model."""

    def test_defaults(self):
        from pipeline.models import StageConfig
        cfg = StageConfig(stage="gpt1")
        assert cfg.provider == "openai"
        assert cfg.api_key == ""
        assert cfg.model == ""
        assert cfg.base_url == ""

    def test_full_config(self):
        from pipeline.models import StageConfig
        cfg = StageConfig(
            stage="gpt2", provider="anthropic",
            api_key="sk-ant-test", model="claude-3",
        )
        assert cfg.stage == "gpt2"
        assert cfg.provider == "anthropic"
