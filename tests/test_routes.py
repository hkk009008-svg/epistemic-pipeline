"""Integration tests for API routes using FastAPI TestClient.

Tests API endpoints, middleware, rate limiting, error handling,
and security headers without making any LLM calls.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from app import app

client = TestClient(app, raise_server_exceptions=False)


# ===================================================================
# Health endpoint
# ===================================================================

class TestHealth:
    def test_health_returns_ok(self):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"


# ===================================================================
# UI endpoint
# ===================================================================

class TestUI:
    def test_root_returns_html(self):
        r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
        assert "Epistemic" in r.text


# ===================================================================
# Security headers
# ===================================================================

class TestSecurityHeaders:
    def test_security_headers_present(self):
        r = client.get("/health")
        assert r.headers.get("x-content-type-options") == "nosniff"
        assert r.headers.get("x-frame-options") == "DENY"
        assert r.headers.get("x-xss-protection") == "1; mode=block"
        assert r.headers.get("referrer-policy") == "strict-origin-when-cross-origin"


# ===================================================================
# Rate limit info endpoint
# ===================================================================

class TestRateLimit:
    def test_rate_limit_returns_info(self):
        r = client.get("/api/rate-limit")
        assert r.status_code == 200
        data = r.json()
        assert "limit" in data
        assert "remaining" in data
        assert "used" in data
        assert data["limit"] > 0

    def test_rate_limit_remaining_is_bounded(self):
        r = client.get("/api/rate-limit")
        data = r.json()
        assert 0 <= data["remaining"] <= data["limit"]


# ===================================================================
# OpenAI config endpoints
# ===================================================================

class TestOpenAIConfig:
    def test_get_config_no_key(self):
        r = client.get("/api/openai/config")
        assert r.status_code == 200
        data = r.json()
        # In test env, key may or may not be set
        assert "key_set" in data

    def test_set_config_empty_key_rejected(self):
        r = client.post("/api/openai/config", json={"api_key": "", "model": "gpt-4o-mini"})
        assert r.status_code == 400

    def test_set_config_whitespace_key_rejected(self):
        r = client.post("/api/openai/config", json={"api_key": "   ", "model": "gpt-4o-mini"})
        assert r.status_code == 400


# ===================================================================
# Tavily config endpoints
# ===================================================================

class TestTavilyConfig:
    def test_get_tavily_config(self):
        r = client.get("/api/tavily/config")
        assert r.status_code == 200
        data = r.json()
        assert "key_set" in data

    def test_toggle_without_key_fails(self):
        r = client.post("/api/tavily/toggle?enabled=true")
        # May fail if no key set
        assert r.status_code in (200, 400)


# ===================================================================
# Pipeline endpoint - error cases (no LLM calls)
# ===================================================================

class TestPipelineErrors:
    def test_missing_api_key(self):
        """Pipeline should fail gracefully when no API key is set."""
        # This depends on test env state, but should not crash
        r = client.post("/api/pipeline", json={"prompt": "test"})
        assert r.status_code in (200, 400, 429)

    def test_empty_prompt_rejected(self):
        """Empty prompt should be rejected by Pydantic validation."""
        r = client.post("/api/pipeline", json={})
        assert r.status_code == 422  # Pydantic validation error

    def test_prompt_too_long_rejected(self):
        """Prompt exceeding MAX_PROMPT_LENGTH should be rejected."""
        r = client.post("/api/pipeline", json={"prompt": "x" * 10001})
        assert r.status_code == 422


# ===================================================================
# Feedback endpoint
# ===================================================================

class TestFeedback:
    def test_submit_valid_feedback(self):
        r = client.post("/api/feedback", json={
            "rating": "accurate",
            "prompt": "test prompt",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "feedback_id" in data

    def test_invalid_rating_rejected(self):
        r = client.post("/api/feedback", json={
            "rating": "invalid_value",
            "prompt": "test",
        })
        assert r.status_code == 400

    def test_feedback_summary(self):
        r = client.get("/api/feedback/summary")
        assert r.status_code == 200
        data = r.json()
        assert "total_feedback" in data


# ===================================================================
# Metrics endpoint
# ===================================================================

class TestMetrics:
    def test_metrics_returns_data(self):
        r = client.get("/api/metrics")
        assert r.status_code == 200
        data = r.json()
        assert "total_requests" in data


# ===================================================================
# Stage config endpoints
# ===================================================================

class TestStageConfig:
    def test_get_stage_config(self):
        for stage in ("gpt1", "gpt2", "gpt3"):
            r = client.get(f"/api/stage/config/{stage}")
            assert r.status_code == 200
            data = r.json()
            assert data["stage"] == stage
            assert "provider" in data

    def test_set_stage_config(self):
        r = client.post("/api/stage/config", json={
            "stage": "gpt1",
            "provider": "openai",
            "api_key": "sk-test123",
            "model": "gpt-4o-mini",
        })
        assert r.status_code == 200


# ===================================================================
# NLI status endpoint
# ===================================================================

class TestNLIStatus:
    def test_nli_status_returns_structure(self):
        r = client.get("/api/nli/status")
        assert r.status_code == 200
        data = r.json()
        assert "available" in data
        assert "mode" in data
        assert "thresholds" in data
        assert "entailment" in data["thresholds"]

    def test_health_includes_nli(self):
        r = client.get("/health")
        data = r.json()
        assert "nli_available" in data
        assert "nli_mode" in data


# ===================================================================
# V2 pipeline endpoint
# ===================================================================

class TestV2Pipeline:
    def test_v2_pipeline_validation(self):
        """V2 pipeline rejects empty prompt same as v1."""
        r = client.post("/v2/pipeline", json={})
        assert r.status_code == 422

    def test_v2_pipeline_exists(self):
        """V2 pipeline is reachable and returns error (no API key)."""
        r = client.post("/v2/pipeline", json={"prompt": "test"})
        assert r.status_code in (200, 400, 429)


# ===================================================================
# V2 admin and public capabilities
# ===================================================================

class TestV2Admin:
    def test_v2_admin_config_accepted_no_auth(self):
        """Without ADMIN_TOKEN set, admin endpoints are open."""
        r = client.post("/v2/admin/config", json={
            "stage": "gpt1",
            "provider": "openai",
            "api_key": "sk-test",
            "model": "gpt-4o",
        })
        assert r.status_code == 200

    def test_v2_admin_rejects_bad_base_url(self):
        """base_url not in allowlist should be rejected."""
        r = client.post("/v2/admin/config", json={
            "stage": "gpt1",
            "provider": "ollama",
            "api_key": "ollama",
            "model": "llama3",
            "base_url": "http://evil.example.com/v1",
        })
        assert r.status_code == 400

    def test_v2_admin_allows_localhost_base_url(self):
        """Localhost base_url should be allowed."""
        r = client.post("/v2/admin/config", json={
            "stage": "gpt1",
            "provider": "ollama",
            "api_key": "ollama",
            "model": "llama3",
            "base_url": "http://localhost:11434/v1",
        })
        assert r.status_code == 200


class TestV2PublicCapabilities:
    def test_capabilities_returns_structure(self):
        r = client.get("/v2/public/capabilities")
        assert r.status_code == 200
        data = r.json()
        assert "providers" in data
        assert "stages" in data
        assert "tiers" in data
        assert "nli_available" in data
        assert "tavily_enabled" in data


# ===================================================================
# Stage config base_url validation
# ===================================================================

class TestBaseUrlValidation:
    def test_stage_config_rejects_bad_base_url(self):
        """base_url not in allowlist should be rejected."""
        r = client.post("/api/stage/config", json={
            "stage": "gpt1",
            "provider": "ollama",
            "api_key": "ollama",
            "model": "llama3",
            "base_url": "http://internal-server.local/v1",
        })
        assert r.status_code == 400

    def test_stage_config_accepts_openrouter_url(self):
        r = client.post("/api/stage/config", json={
            "stage": "gpt1",
            "provider": "openrouter",
            "api_key": "or-test",
            "model": "mix",
            "base_url": "https://openrouter.ai/api/v1",
        })
        assert r.status_code == 200


# ===================================================================
# Stress endpoint - validation only
# ===================================================================

class TestStressValidation:
    def test_stress_no_key(self):
        """Stress should fail gracefully without API key."""
        r = client.post("/api/stress", json={})
        assert r.status_code in (200, 400, 429)
