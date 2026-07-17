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
# NVIDIA NIM key (for the opt-in `nemotron` model — build.nvidia.com).
if [ -z "${NVIDIA_API_KEY:-}" ] && [ -f "$HOME/.hermes/.env" ]; then
  export NVIDIA_API_KEY="$(grep '^NVIDIA_API_KEY=' "$HOME/.hermes/.env" | tail -1 | cut -d= -f2-)"
fi
if grep -q '^META_MODEL_API_KEY=' "$HOME/.hermes/.env" 2>/dev/null; then
  export META_MODEL_API_KEY="$(grep '^META_MODEL_API_KEY=' "$HOME/.hermes/.env" | tail -1 | cut -d= -f2-)"
fi
# Moonshot/Kimi platform key (opt-in `kimi-coding` = kimi-k2.7-code, PER-TOKEN).
if grep -q '^MOONSHOT_API_KEY=' "$HOME/.hermes/.env" 2>/dev/null; then
  export MOONSHOT_API_KEY="$(grep '^MOONSHOT_API_KEY=' "$HOME/.hermes/.env" | tail -1 | cut -d= -f2-)"
fi
# Kimi Code membership key (sk-kimi-*, flat-rate `kimi-code` route on api.kimi.com).
if grep -q '^KIMI_CODE_API_KEY=' "$HOME/.hermes/.env" 2>/dev/null; then
  export KIMI_CODE_API_KEY="$(grep '^KIMI_CODE_API_KEY=' "$HOME/.hermes/.env" | tail -1 | cut -d= -f2-)"
fi
# OpenCode Zen/Go key (opencode.ai gateway; free Zen models need no billing).
if grep -q '^OPENCODE_API_KEY=' "$HOME/.hermes/.env" 2>/dev/null; then
  export OPENCODE_API_KEY="$(grep '^OPENCODE_API_KEY=' "$HOME/.hermes/.env" | tail -1 | cut -d= -f2-)"
fi
export HERMES_LOG_PATH="${HERMES_LOG_PATH:-$HOME/.hermes/litellm-logs/traffic.jsonl}"
LITELLM_BIN="${LITELLM_BIN:-$HOME/.local/bin/litellm}"
exec "$LITELLM_BIN" --config "$HERE/config.yaml" --port "${LITELLM_PORT:-4010}"
