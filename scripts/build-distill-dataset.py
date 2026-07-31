#!/usr/bin/env python3
"""Build a distillation dataset from the Hermes LiteLLM traffic log.

The gateway already logs every call to ~/.hermes/litellm-logs/traffic.jsonl with the
full input `messages`, the `response`, and `has_tool_calls` / `finish_reason` flags.
This is exactly the raw material for trajectory distillation (2505.17612): teach a
small local model (qwen3:8b) the tool-use behavior of the strong teacher (GLM-5.2).

This is the "instrument first" no-regret step from the July-2026 fine-tuning research:
it costs nothing and accumulates real agentic traces so that IF distillation is ever
justified, the dataset already exists — instead of the 194 thumbs events, which are the
wrong shape (use KTO on those, not this).

Output: JSONL in chat-messages format (input turns + assistant target), consumable by
MLX-LM LoRA / mlx-tune SFT.

Usage:
  python3 build-distill-dataset.py [--in TRAFFIC] [--out DATASET] [--teacher glm,nemotron]
"""

import argparse
import glob
import gzip
import hashlib
import json
import os
import sys

LOG_DIR = os.path.expanduser("~/.hermes/litellm-logs")
DEFAULT_IN = os.path.join(LOG_DIR, "traffic.jsonl")
GOOD_FINISH = {"stop", "tool_calls"}


def _open(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def load(path, include_archives=False):
    """Load the live log, and (for --accumulate) the rotated .gz archives too, so trajectory
    history survives log rotation (rotate-traffic-log.sh gzips traffic.jsonl.<date>.gz)."""
    paths = [path]
    if include_archives:
        paths += sorted(glob.glob(os.path.join(os.path.dirname(path), "traffic.jsonl.*.gz")))
    rows = []
    for p in paths:
        if not os.path.exists(p):
            continue
        with _open(p) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def example_key(ex):
    """Stable content hash for dedup across re-runs and overlapping log windows."""
    m = ex["messages"]
    payload = json.dumps([m[0] if m else {}, m[-1]], sort_keys=True) + str(
        ex.get("meta", {}).get("ts")
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def usable(row, teachers):
    if row.get("finish_reason") not in GOOD_FINISH:
        return False
    model = str(row.get("model", "")).lower()
    if teachers and not any(t in model for t in teachers):
        return False
    if not row.get("messages"):
        return False
    # A turn is a valid training target if it produced EITHER response text OR a tool call.
    # Pure tool-call turns (empty response text) are the crux of tool-use distillation and
    # require the logger's tool_calls field (added 2026-07-11); older rows lack it and are
    # skipped rather than distilling a blank target.
    return bool(row.get("response")) or bool(row.get("tool_calls"))


def to_example(row):
    """Chat-messages SFT format: the logged input turns + the teacher's answer as target.
    The target is the assistant turn, carrying response text and/or the tool_calls payload."""
    messages = list(row["messages"])
    assistant = {"role": "assistant", "content": row.get("response") or ""}
    if row.get("tool_calls"):
        assistant["tool_calls"] = row["tool_calls"]
    messages.append(assistant)
    return {"messages": messages, "meta": {"model": row.get("model"), "ts": row.get("ts_end")}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=DEFAULT_IN)
    ap.add_argument("--out", default=os.path.expanduser("~/.hermes/distill/teacher-tooluse.jsonl"))
    ap.add_argument(
        "--teacher",
        default="glm,nemotron,claude",
        help="comma-separated teacher substrings to keep (empty = all)",
    )
    ap.add_argument(
        "--accumulate",
        action="store_true",
        help="read rotated .gz archives too and merge/dedupe into the existing "
        "dataset instead of overwriting (for the scheduled job)",
    )
    args = ap.parse_args()

    teachers = [t.strip().lower() for t in args.teacher.split(",") if t.strip()]
    if not os.path.exists(args.inp):
        sys.exit(f"traffic log not found: {args.inp}")

    rows = load(args.inp, include_archives=args.accumulate)
    fresh = [to_example(r) for r in rows if usable(r, teachers)]

    # Merge with any existing dataset, deduping by content hash so re-runs and overlapping
    # log windows never double-count.
    existing = []
    if args.accumulate and os.path.exists(args.out):
        existing = load(args.out)
    by_key = {example_key(ex): ex for ex in existing}
    before = len(by_key)
    for ex in fresh:
        by_key.setdefault(example_key(ex), ex)
    merged = list(by_key.values())
    added = len(merged) - before

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w") as fh:
        for ex in merged:
            fh.write(json.dumps(ex) + "\n")
    os.replace(tmp, args.out)

    src = "live+archives" if args.accumulate else "live log"
    print(f"read      {len(rows)} trajectories from {src}")
    print(f"usable    {len(fresh)} teacher tool-use trajectories (teachers={teachers or 'all'})")
    if args.accumulate:
        print(f"added     {added} new (deduped against {before} existing)")
    print(f"total     {len(merged)} in dataset -> {args.out}")
    print("note      distillation becomes worthwhile around a few hundred+ clean traces.")


if __name__ == "__main__":
    main()
