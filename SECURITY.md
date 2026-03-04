# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 3.x     | Yes                |
| < 3.0   | No                 |

## Reporting a Vulnerability

If you discover a security vulnerability, **do not open a public issue**. Instead:

1. **Email:** Send a detailed report to the maintainers via GitHub's [private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability) feature on this repository.
2. **Include:** A description of the vulnerability, steps to reproduce, and (if possible) a suggested fix.
3. **Response time:** We aim to acknowledge reports within 48 hours and provide a fix or mitigation within 7 days for critical issues.

## Security Design

### Authentication
- Config-mutation endpoints (`POST /api/openai/config`, `/api/tavily/config`, `/api/stage/config`, `/v2/admin/config`) require `Authorization: Bearer <ADMIN_TOKEN>` when `ADMIN_TOKEN` is set.
- When `ADMIN_TOKEN` is unset (local development), config endpoints are open.

### Input Validation
- All API inputs are validated through Pydantic models with strict type constraints.
- Prompt length is capped at 10,000 characters (`MAX_PROMPT_LENGTH`).
- System prompt overrides are silently ignored unless `ALLOW_PROMPT_OVERRIDE=true` (dev only).
- Stage `base_url` values are validated against an allowlist to prevent SSRF.

### Output Safety
- Raw exception messages are never exposed to clients — the global exception handler returns generic "Internal server error" for unhandled exceptions.
- API keys are redacted in all GET config responses (only previews like `sk-...xxxx` are returned).

### Transport Security
- CORS is disabled by default (same-origin only); configurable via `ALLOWED_ORIGINS`.
- Security headers are set on all responses: `X-Content-Type-Options`, `X-Frame-Options`, `X-XSS-Protection`, `Referrer-Policy`.
- HSTS is enabled when `FORCE_HTTPS=true`.

### Rate Limiting
- Per-IP sliding-window rate limiter (default: 20 requests/minute) on pipeline and stress endpoints.

### Dependencies
- Dependabot is enabled for automated security updates.
- CodeQL scanning runs on every push and pull request.

## Hardening Checklist for Production

- [ ] Set `ADMIN_TOKEN` to a strong random value
- [ ] Set `FORCE_HTTPS=true`
- [ ] Set `ALLOWED_ORIGINS` to your frontend domain(s)
- [ ] Review `ALLOWED_BASE_URLS` if using custom providers
- [ ] Ensure `ALLOW_PROMPT_OVERRIDE` is unset or `false`
- [ ] Run behind a reverse proxy (Railway, nginx, Cloudflare) for TLS termination
