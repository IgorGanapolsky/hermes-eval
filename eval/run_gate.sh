#!/usr/bin/env bash
# Local model-drift eval gate. Runs promptfoo against the LiteLLM proxy, computes the
# pass-rate, runs an optional baseline regression check, posts a commit status if in CI,
# and exits non-zero when the gate fails (so it blocks a merge/release).
set -uo pipefail
cd "$(dirname "$0")"

CONFIG="${1:-promptfooconfig.yaml}"
THRESHOLD="${EVAL_THRESHOLD:-0.85}"
TOLERANCE="${EVAL_TOLERANCE:-0.05}"
OUT="results.json"

: "${LITELLM_MASTER_KEY:?set LITELLM_MASTER_KEY (the proxy Bearer key)}"

echo "▶ eval config=$CONFIG threshold=$THRESHOLD subset=${EVAL_SUBSET:-<all>}"
npx --yes promptfoo@latest eval -c "$CONFIG" -o "$OUT" --no-table || true

if [ ! -f "$OUT" ]; then echo "✖ no $OUT produced"; exit 1; fi

PASS=$(jq '.results.stats.successes // 0' "$OUT")
FAIL=$(jq '.results.stats.failures // 0' "$OUT")
ERR=$(jq '.results.stats.errors // 0' "$OUT")           # eval errors must NOT count as a pass
DEN=$((PASS + FAIL + ERR))
RATE=$(python3 -c "p=$PASS; d=$DEN; print(round(p/d,4) if d else 0.0)")
echo "▶ pass-rate=$RATE (pass=$PASS fail=$FAIL errors=$ERR)"

REG_OK=1
if [ -f baseline.json ]; then
  python3 compare_baseline.py --current "$OUT" --baseline baseline.json --tolerance "$TOLERANCE" || REG_OK=0
fi

GATE_OK=$(python3 -c "print(1 if ($RATE >= $THRESHOLD and $ERR == 0) else 0)")
STATE="failure"; { [ "$GATE_OK" = 1 ] && [ "$REG_OK" = 1 ]; } && STATE="success"

if [ -n "${GITHUB_SHA:-}" ] && [ -n "${GITHUB_REPOSITORY:-}" ] && command -v gh >/dev/null 2>&1; then
  gh api "repos/${GITHUB_REPOSITORY}/statuses/${GITHUB_SHA}" \
    -f state="$STATE" -f context="eval/golden-gate" \
    -f description="pass-rate ${RATE} (min ${THRESHOLD})" >/dev/null 2>&1 \
    && echo "▶ posted commit status eval/golden-gate=$STATE"
fi

if [ "$STATE" = success ]; then echo "✔ GATE PASSED"; exit 0; else echo "✖ GATE FAILED"; exit 1; fi
