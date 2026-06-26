#!/usr/bin/env bash
# One command: ensure the proxy is up (spawn if needed) -> run the local gate -> tear down what we spawned.
# Exit code == the gate's exit code, so this is CI-ready.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="${1:-promptfooconfig.local.yaml}"
PORT="${LITELLM_PORT:-4010}"
export LITELLM_MASTER_KEY="${LITELLM_MASTER_KEY:-sk-hermes-local-dev}"

SPAWNED=0
if curl -s --max-time 3 "http://127.0.0.1:${PORT}/health/liveliness" >/dev/null 2>&1; then
  echo "▶ proxy already up on :${PORT}"
else
  echo "▶ starting proxy..."
  bash "$ROOT/litellm/start-proxy.sh" >/tmp/hermes-eval-proxy.log 2>&1 &
  SPAWNED=$!
  curl -s --retry 40 --retry-delay 2 --retry-connrefused --max-time 100 \
    "http://127.0.0.1:${PORT}/health/liveliness" >/dev/null \
    || { echo "✖ proxy failed to start"; tail -20 /tmp/hermes-eval-proxy.log; kill "$SPAWNED" 2>/dev/null; exit 1; }
fi
cleanup(){ [ "$SPAWNED" != 0 ] && kill "$SPAWNED" 2>/dev/null || true; }
trap cleanup EXIT

echo "▶ running gate ($CONFIG)"
EVAL_SUBSET="${EVAL_SUBSET:-ci-smoke}" bash "$ROOT/eval/run_gate.sh" "$CONFIG"
