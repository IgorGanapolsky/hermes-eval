#!/bin/zsh
# Hourly gateway competence check. 0 = a capable model is serving; 1 = degraded; 2 = probe failed.
# Non-zero => append the line to the log and raise a macOS notification.
#
# Why this exists: on 2026-07-09 the z.ai quota capped, cloud-fallback 402'd, and a local 8B
# served HTTP 200 with degraded output for hours. Liveness checks stayed green. See hermes-eval #3.
REPO="${HERMES_EVAL_REPO:-$HOME/workspace/git/igor/hermes-eval}"
LOG="$HOME/.hermes/litellm-logs/competence.jsonl"
mkdir -p "$(dirname "$LOG")"

# Read ONLY the key we need. Sourcing ~/.hermes/.env executes malformed lines (unquoted paths
# with spaces) and pollutes the environment.
ENVFILE="$HOME/.hermes/.env"
if [ -f "$ENVFILE" ]; then
  export OPENROUTER_API_KEY="$(grep -m1 '^OPENROUTER_API_KEY=' "$ENVFILE" | cut -d= -f2-)"
fi

cd "$REPO" || exit 2

# ---- QUOTA GATE (added 2026-07-28): skip the probe when the fleet has been idle. ----
# The probe sends a real prompt to glm-coding, and z.ai bills the Coding Plan in PROMPTS,
# not tokens (docs.z.ai/devpack/faq). Hourly => 24 prompts/day => ~168/week spent purely on
# monitoring, which on a ~400-prompt weekly cap is ~40% of the plan. Igor's cap blew on
# 2026-07-28 (reset 2026-08-01 21:07) and took the fleet down for 4 days. A degraded gateway
# only matters if something is USING it, so probe only when real traffic has appeared since
# the last probe. Watermark = traffic.jsonl size; compared with != because the log is rotated
# daily (03:24), which SHRINKS it — a > comparison would skip forever after every rotation.
TRAFFIC="${HERMES_LOG_PATH:-$HOME/.hermes/litellm-logs/traffic.jsonl}"
MARK="$HOME/.hermes/litellm-logs/.competence-probe.watermark"
# The watermark is written AFTER the probe runs (below), never here: the probe's OWN call is
# logged to traffic.jsonl, so stamping the size up-front means the file has always grown by the
# next run and the gate never fires. Verified empirically 2026-07-28 — the first version of this
# gate skipped nothing.
if [ -z "$HERMES_PROBE_FORCE" ] && [ -f "$TRAFFIC" ]; then
  SIZE="$(/usr/bin/stat -f %z "$TRAFFIC" 2>/dev/null)"
  PREV="$(cat "$MARK" 2>/dev/null)"
  if [ -n "$SIZE" ] && [ "$SIZE" = "$PREV" ]; then
    exit 0   # no gateway traffic since the last probe — nothing to verify, spend nothing
  fi
fi

# Capture the probe's status BEFORE any pipe. `x="$(cmd | tail -1)"; rc=$?` yields tail's
# status (always 0), which would make this monitor silently never alert.
# zsh does not word-split parameter expansions, so build argv as an array.
# (HERMES_PROBE_* are test hooks; unset in normal operation.)
args=(--json)
[ -n "$HERMES_PROBE_GATEWAY" ] && args+=(--gateway "$HERMES_PROBE_GATEWAY")
[ -n "$HERMES_PROBE_TIMEOUT" ] && args+=(--timeout "$HERMES_PROBE_TIMEOUT")
OUT="$(uv run python litellm/competence_probe.py "${args[@]}" 2>&1)"
RC=$?
LAST="${OUT##*$'\n'}"

TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
HOST="$(scutil --get ComputerName 2>/dev/null)"
printf '{"ts":"%s","host":"%s","rc":%d,"out":%s}\n' "$TS" "$HOST" "$RC" \
  "$(printf '%s' "$LAST" | python3 -c 'import sys,json;print(json.dumps(sys.stdin.read().strip()))')" >> "$LOG"

# Stamp the watermark with the size INCLUDING this probe's own traffic, so the next run fires
# only if something OTHER than the probe used the gateway.
if [ -z "$HERMES_PROBE_FORCE" ] && [ -f "$TRAFFIC" ]; then
  SIZE_AFTER="$(/usr/bin/stat -f %z "$TRAFFIC" 2>/dev/null)"
  [ -n "$SIZE_AFTER" ] && printf '%s' "$SIZE_AFTER" > "$MARK"
fi

if [ $RC -ne 0 ]; then
  osascript -e "display notification \"gateway degraded (rc=$RC) — agents will loop\" with title \"Hermes competence probe\"" 2>/dev/null
fi
exit $RC
