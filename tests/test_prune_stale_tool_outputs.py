#!/usr/bin/env python3
"""Unit tests for the stale-tool-output pruning middleware (litellm/hermes_logger.py).

Standalone: `python3 tests/test_prune_stale_tool_outputs.py` (no litellm needed — the
logger guards that import).
"""
import copy
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "litellm"))

from hermes_logger import PRUNE_STUB_PREFIX, stub_stale_tool_outputs  # noqa: E402


def call(name, args, cid):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": args}}


def convo(n_pad=0):
    """A conversation with: superseded read_file, aged big grep, small ok result."""
    msgs = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "do the thing"},
        {"role": "assistant", "tool_calls": [call("read_file", '{"path":"a.ts"}', "c1")]},
        {"role": "tool", "tool_call_id": "c1", "content": "OLD CONTENT " * 10},  # superseded
        {"role": "assistant", "tool_calls": [call("grep", '{"q":"x"}', "c2")]},
        {"role": "tool", "tool_call_id": "c2", "content": "G" * 5000},  # aged (big)
        {"role": "assistant", "tool_calls": [call("run", "{}", "c3")]},
        {"role": "tool", "tool_call_id": "c3", "content": "ok"},  # small: kept
        {"role": "assistant", "tool_calls": [call("read_file", '{"path":"a.ts"}', "c4")]},
        {"role": "tool", "tool_call_id": "c4", "content": "NEW CONTENT " * 10},  # newest same-key
    ]
    msgs += [{"role": "user", "content": f"pad {i}"} for i in range(n_pad)]
    return msgs


def kw(msgs, model="glm-coding", dep="openai/glm-5.2"):
    return {"model": model, "litellm_params": {"model": dep}, "messages": msgs}


# 1. superseded + aged stubbed; small + newest kept (window 4 => cutoff protects tail)
k = kw(convo())
stub_stale_tool_outputs(k, protect_last_n=4, min_chars=600)
m = k["messages"]
assert m[3]["content"].startswith(PRUNE_STUB_PREFIX), "superseded read_file must be stubbed"
assert "superseded" in m[3]["content"]
assert m[5]["content"].startswith(PRUNE_STUB_PREFIX), "aged 5000-char grep must be stubbed"
assert "aged out" in m[5]["content"]
assert m[7]["content"] == "ok", "small tool result must be kept"
assert m[9]["content"].startswith("NEW CONTENT"), "newest same-key result must be kept"

# 2. protected window untouched
k = kw(convo())
stub_stale_tool_outputs(k, protect_last_n=len(k["messages"]), min_chars=1)
assert all(not str(x.get("content", "")).startswith(PRUNE_STUB_PREFIX) for x in k["messages"])

# 3. idempotent: prune(prune(x)) == prune(x)
k = kw(convo())
stub_stale_tool_outputs(k, protect_last_n=4, min_chars=600)
once = copy.deepcopy(k["messages"])
stub_stale_tool_outputs(k, protect_last_n=4, min_chars=600)
assert k["messages"] == once, "pruning must be idempotent"

# 4. non-GLM deployments untouched
k = kw(convo(), model="hermes-local", dep="ollama_chat/qwen3:8b-64k")
stub_stale_tool_outputs(k, protect_last_n=4, min_chars=600)
assert not any(str(x.get("content", "")).startswith(PRUNE_STUB_PREFIX) for x in k["messages"])

# 5. openrouter GLM fallback untouched (clamped separately, never pruned here)
k = kw(convo(), model="cloud-fallback", dep="openrouter/z-ai/glm-5.2")
stub_stale_tool_outputs(k, protect_last_n=4, min_chars=600)
assert not any(str(x.get("content", "")).startswith(PRUNE_STUB_PREFIX) for x in k["messages"])

# 6. schema preserved: same message count, roles, tool_call_ids
k = kw(convo())
before = [(x.get("role"), x.get("tool_call_id")) for x in k["messages"]]
stub_stale_tool_outputs(k, protect_last_n=4, min_chars=600)
after = [(x.get("role"), x.get("tool_call_id")) for x in k["messages"]]
assert before == after, "pruning must never add/remove/reorder messages"

# 7. non-string content (vision parts) skipped without error
k = kw([{"role": "tool", "tool_call_id": "z", "content": [{"type": "text", "text": "x" * 9000}]}] * 6)
stub_stale_tool_outputs(k, protect_last_n=1, min_chars=1)
assert isinstance(k["messages"][0]["content"], list)

print("PASS: stub_stale_tool_outputs (7 cases)")
