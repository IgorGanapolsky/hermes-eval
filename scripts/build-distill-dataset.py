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
import argparse, json, os, sys

DEFAULT_IN = os.path.expanduser("~/.hermes/litellm-logs/traffic.jsonl")
GOOD_FINISH = {"stop", "tool_calls"}


def load(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


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
    ap.add_argument("--teacher", default="glm,nemotron,claude",
                    help="comma-separated teacher substrings to keep (empty = all)")
    args = ap.parse_args()

    teachers = [t.strip().lower() for t in args.teacher.split(",") if t.strip()]
    if not os.path.exists(args.inp):
        sys.exit(f"traffic log not found: {args.inp}")

    rows = load(args.inp)
    kept = [to_example(r) for r in rows if usable(r, teachers)]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        for ex in kept:
            fh.write(json.dumps(ex) + "\n")

    print(f"read      {len(rows)} trajectories from {args.inp}")
    print(f"kept      {len(kept)} teacher tool-use trajectories (teachers={teachers or 'all'})")
    print(f"wrote     {args.out}")
    print(f"note      distillation becomes worthwhile around a few hundred+; run periodically to accumulate.")


if __name__ == "__main__":
    main()
