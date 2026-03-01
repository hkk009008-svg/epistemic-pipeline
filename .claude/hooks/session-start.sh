#!/bin/bash
set -euo pipefail

# Only run in remote (Claude Code on the web) environments
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

# Install project dependencies
pip install -r "$CLAUDE_PROJECT_DIR/requirements.txt"

# Install test and lint tools
pip install pytest ruff
