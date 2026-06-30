#!/usr/bin/env bash
# Start the LiteLLM gateway with fleet env. Used manually, by `make verify`, and by the launchd service.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -f "$HOME/.hermes/.env" ]; then
  export OPENROUTER_API_KEY="$(grep '^OPENROUTER_API_KEY=' "$HOME/.hermes/.env" | cut -d= -f2-)"
fi
# z.ai GLM Coding Plan subscription key (for the glm-coding model). Last match wins.
if [ -z "${Z_AI_API_KEY:-}" ] && [ -f "$HOME/.hermes/.env" ]; then
  export Z_AI_API_KEY="$(grep '^Z_AI_API_KEY=' "$HOME/.hermes/.env" | tail -1 | cut -d= -f2-)"
fi
export HERMES_LOG_PATH="${HERMES_LOG_PATH:-$HOME/.hermes/litellm-logs/traffic.jsonl}"
LITELLM_BIN="${LITELLM_BIN:-$HOME/.local/bin/litellm}"
exec "$LITELLM_BIN" --config "$HERE/config.yaml" --port "${LITELLM_PORT:-4010}"
