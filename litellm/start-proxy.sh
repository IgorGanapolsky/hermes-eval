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
# Together.ai serverless key (per-token; opt-in-by-name routes together-inkling/together-glm).
if grep -q '^GEMINI_API_KEY=' "$HOME/.hermes/.env" 2>/dev/null; then
  export GEMINI_API_KEY="$(grep '^GEMINI_API_KEY=' "$HOME/.hermes/.env" | tail -1 | cut -d= -f2-)"
fi
if grep -q '^TOGETHER_API_KEY=' "$HOME/.hermes/.env" 2>/dev/null; then
  export TOGETHER_API_KEY="$(grep '^TOGETHER_API_KEY=' "$HOME/.hermes/.env" | tail -1 | cut -d= -f2-)"
fi
# GPT-6 Astra is opt-in ($10/mo hard cap). Prefer OPENAI_API_KEY; else reuse the
# existing project key already used for voice tools. Never default the fleet here.
if [ -z "${OPENAI_API_KEY:-}" ] && [ -f "$HOME/.hermes/.env" ]; then
  if grep -q '^OPENAI_API_KEY=' "$HOME/.hermes/.env"; then
    export OPENAI_API_KEY="$(grep '^OPENAI_API_KEY=' "$HOME/.hermes/.env" | tail -1 | cut -d= -f2-)"
  elif grep -q '^VOICE_TOOLS_OPENAI_KEY=' "$HOME/.hermes/.env"; then
    export OPENAI_API_KEY="$(grep '^VOICE_TOOLS_OPENAI_KEY=' "$HOME/.hermes/.env" | tail -1 | cut -d= -f2-)"
  fi
fi
export HERMES_LOG_PATH="${HERMES_LOG_PATH:-$HOME/.hermes/litellm-logs/traffic.jsonl}"
LITELLM_BIN="${LITELLM_BIN:-$HOME/.local/bin/litellm}"
python3 "$HERE/merge_astra_route.py" "$HERE/config.yaml" "$HERE/config.runtime.yaml"
exec "$LITELLM_BIN" --config "$HERE/config.runtime.yaml" --port "${LITELLM_PORT:-4010}"
