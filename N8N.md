# n8n Workflow Integration

## Overview

The `n8n-workflows/` directory contains n8n workflow definitions that orchestrate the Epistemic Verification Pipeline. There are two generations:

**Legacy workflows** (`epistemic-*`) provide basic webhook proxying, health checks, and batch stress testing.

**ECA workflows** (`eca-*`) add a deterministic pre-check guardrail, proper NDJSON summary extraction, and nightly regression with alert thresholds.

## Workflow Reference

### Legacy Workflows

| File | Trigger | Webhook Path | Purpose |
|------|---------|-------------|---------|
| `epistemic-verify-pipeline.json` | Webhook POST | `/webhook/epistemic-verify` | Direct pipeline proxy |
| `epistemic-health-check.json` | Schedule (6h) | — | Health check on known-good prompt |
| `epistemic-batch-stress.json` | Webhook POST | `/webhook/epistemic-stress` | Stress test with API key setup |

### ECA Workflows

| File | Trigger | Webhook Path | Purpose |
|------|---------|-------------|---------|
| `eca-verify-proxy.json` | Webhook POST | `/webhook/eca-verify` | Pipeline proxy **with deterministic pre-check** |
| `eca-stress-webhook.json` | Webhook POST | `/webhook/eca-stress` | Stress test with NDJSON summary extraction |
| `eca-nightly-stress.json` | Schedule (daily 02:00) | — | Nightly regression + alert threshold (score < 75) |

## Prerequisites

- **n8n** installed and running (default: `http://localhost:5678`)
- **FastAPI pipeline** running (default: `http://localhost:8000`)
- Both can be started with `./start.sh`

## Environment Variable

All ECA workflows reference `PIPELINE_BASE_URL` for the FastAPI host.

**Set in n8n UI**: Settings → Variables → Add `PIPELINE_BASE_URL`

**Set via environment**:
```bash
export PIPELINE_BASE_URL=http://localhost:8000
```

**Default fallback**: `http://localhost:8000` (used if the variable is not set)

## Importing Workflows

1. Open your n8n instance (e.g. `http://localhost:5678`)
2. Go to **Workflows → Add Workflow → Import from File**
3. Select the desired `.json` file from `n8n-workflows/`
4. Set `PIPELINE_BASE_URL` if not already configured
5. Toggle the workflow **Active**

## Workflow Details

### eca-verify-proxy.json — Verify with Guardrail

**Key feature: Deterministic pre-check**

Before calling the GPT pipeline, a Code node performs a regex check on the incoming prompt. If the prompt requests percentage data, statistics, rates, odds, or probability figures, the workflow returns an immediate ABSTAIN response **without invoking any GPT calls**:

```json
{"status": "ABSTAIN", "result": "Unknown (Actionable): No authoritative dataset available."}
```

The regex mirrors `_PERCENT_RE` from `portfolio_api.py` and catches prompts containing: `percent`, `percentage`, `rate`, `odds`, `how many`, `how often`, `probability`, `fraction`, `proportion`, `typically`, or the `%` symbol.

**Non-matching prompts** are forwarded to `POST /api/pipeline` and the response is routed:

- PASS → `{"status": "PASS", "final_result": "..."}`
- FAIL → `{"status": "NO_PASS", "violations": [...], "arbiter_decision": "..."}`

### eca-stress-webhook.json — On-Demand Stress Test

Accepts optional `category` and `count` parameters in the POST body. Calls `POST /api/stress`, parses the streaming NDJSON response, extracts the summary line (`{"type": "summary", ...}`), and returns it as clean JSON.

**Example request**:
```bash
curl -X POST http://localhost:5678/webhook/eca-stress \
  -H "Content-Type: application/json" \
  -d '{"category": "legal_future_year", "count": 5}'
```

### eca-nightly-stress.json — Nightly Regression

Runs the full stress suite daily at 02:00 UTC. Parses the NDJSON summary and checks the PSS score against a threshold of **75**.

| Condition | Output |
|-----------|--------|
| Score < 75 | `{"alert": true, "score": ..., "timestamp": ..., "summary": ...}` |
| Score >= 75 | `{"alert": false, "score": ..., "timestamp": ..., "summary": ...}` |

The Alert / OK terminal nodes have **no downstream connections** by default. To receive notifications, append a Slack, Email, or GitHub Issues node after the "Alert Payload" node.

## NDJSON Parsing

The `/api/stress` endpoint returns streaming NDJSON (`application/x-ndjson`). Each line is a JSON object:

- **Progress lines**: `{"type": "progress", "index": 1, "total": 100, ...}`
- **Summary line** (final): `{"type": "summary", "pss": {"score": 87.5, ...}, ...}`

The ECA stress workflows set the HTTP Request node to `responseFormat: "text"` and use a Code node to split by newlines, parse each line, and extract the summary object.

## Extending — Adding Alerts

To receive Slack/email/GitHub notifications when the nightly score drops:

1. Open `eca-nightly-stress.json` in n8n
2. Add a **Slack** (or Email / GitHub) node after "Alert Payload"
3. Connect "Alert Payload" → your notification node
4. Configure credentials and message template using `{{ $json.score }}` and `{{ $json.summary }}`
