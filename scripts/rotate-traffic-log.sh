#!/bin/bash
# Rotate the LiteLLM traffic log when it exceeds MAX_MB (default 100MB).
# Safe because hermes_logger.py opens/appends/closes per call — after a rename,
# the next call simply recreates the file. Keeps the newest KEEP archives.
# Installed as launchd com.igor.hermes-traffic-rotate (daily 03:30).
set -euo pipefail

LOG="${HERMES_LOG_PATH:-$HOME/.hermes/litellm-logs/traffic.jsonl}"
MAX_MB="${ROTATE_MAX_MB:-100}"
KEEP="${ROTATE_KEEP:-5}"

[ -f "$LOG" ] || exit 0
size_mb=$(( $(stat -f %z "$LOG") / 1024 / 1024 ))
[ "$size_mb" -ge "$MAX_MB" ] || exit 0

ts=$(date +%Y%m%d-%H%M%S)
mv "$LOG" "$LOG.$ts"
gzip "$LOG.$ts"
# prune oldest beyond KEEP
ls -t "$LOG".*.gz 2>/dev/null | tail -n +$((KEEP + 1)) | xargs rm -f 2>/dev/null || true
echo "rotated ${size_mb}MB -> $LOG.$ts.gz (kept newest $KEEP)"
