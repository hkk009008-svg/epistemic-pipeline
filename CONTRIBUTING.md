# Contributing to Epistemic Pipeline

Thank you for considering contributing! This document explains how to get started.

## Development Setup

```bash
# Clone the repo
git clone https://github.com/hkk009008-svg/epistemic-pipeline.git
cd epistemic-pipeline

# Install dependencies
pip install -r requirements.txt

# Copy env template
cp .env.example .env
# Edit .env with your OPENAI_API_KEY

# Run the server
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Run tests
python -m pytest tests/ -v
```

## How to Contribute

### Reporting Bugs
- Use [GitHub Issues](https://github.com/hkk009008-svg/epistemic-pipeline/issues) with the **bug** label.
- Include: steps to reproduce, expected vs. actual behavior, and your environment (Python version, OS).

### Suggesting Features
- Open an issue with the **enhancement** label.
- Describe the use case and how it fits the pipeline's epistemic verification goals.

### Submitting Pull Requests
1. Fork the repo and create a feature branch from `main`.
2. Write tests for any new logic (see [Testing](#testing) below).
3. Ensure all tests pass: `python -m pytest tests/ -v`
4. Lint your code: `ruff check .`
5. Open a PR against `main` with a clear description.

## Testing

- All unit tests are deterministic — no LLM calls, no mocking needed.
- Tests verify business logic: routing, sanitization, JSON parsing, verdict computation, convergence detection, and arbiter decisions.
- Add fixtures to `tests/conftest.py`.
- Use `@pytest.mark.parametrize` for data-driven tests.
- Run the full suite: `python -m pytest tests/ -v`

## Code Style

- **Type hints** throughout — use `from __future__ import annotations`.
- **snake_case** for functions/variables, **PascalCase** for classes.
- **Pydantic models** for request/response schemas.
- Lint with `ruff check .` before submitting.

## Architecture Notes

- **Routing flags** → `pipeline/sanitizer.py` → `route_prompt()`
- **Prompt augmentations** → `pipeline/prompts.py` → `build_augmentation()`
- **Tripwire violations** follow the T-code pattern (T1–T7) with HARD/SOFT severity.
- **Sanitizer and GPT-2 must stay in sync** — if you change what the sanitizer strips, update GPT-2's awareness.
- **GPT-2 prompt is split**: core in `DEFAULT_GPT2_SYSTEM`, tripwire reference in `GPT2_TRIPWIRE_REFERENCE` (user content injection).

## Security

- Never commit `.env` files or API keys.
- Report security vulnerabilities privately — see [SECURITY.md](SECURITY.md).

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.
