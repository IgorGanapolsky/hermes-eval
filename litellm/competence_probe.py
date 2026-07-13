"""Assert that the model actually serving the gateway is competent — not merely that it returns 200.

On 2026-07-09 the z.ai Coding Plan hit its rolling 5-hour cap, `cloud-fallback` 402'd on a
drained OpenRouter balance, and LiteLLM fell through to a local 8B. The 8B answered HTTP 200
with degraded output, so the router treated every request as served. A tool-using agent
re-emitted one identical tool call for hours. Nothing crashed. Every health check was green.

A liveness check cannot see this. Only an identity check can: ask *who* answered.

Two properties make this probe work, and both are load-bearing:

1. It sends a PRODUCTION-SIZED prompt. When OpenRouter's balance drains it shrinks the allowed
   prompt (observed 17043 -> 13342 -> 5567 tokens in one hour). A small probe fits under the
   degraded cap and passes while real ~36k-token agent traffic 402s into the local model. A
   probe that can't reproduce the failure is decoration.

2. It asserts on the RESPONSE's `model` field, not the requested one. LiteLLM echoes the served
   deployment, so a silent demotion to `ollama_chat/qwen3:8b-64k` is visible there and nowhere else.

Exit codes: 0 competent, 1 degraded (a real alert), 2 probe could not run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

DEFAULT_GATEWAY = "http://127.0.0.1:4010/v1/chat/completions"
DEFAULT_MODEL = "glm-coding"

# Deliberately small. A 36k-token probe (production prompt size) run on a schedule would burn
# ~3M tokens/day of the z.ai Coding Plan's rolling 5-hour quota — the very quota it exists to
# protect. A monitor that causes the outage is worse than no monitor.
#
# A small prompt still detects the common degradation (primary capped -> a local model answers),
# because it asserts on WHO replied, not on prompt size. Its one blind spot is the OpenRouter-402
# case, where a drained balance shrinks the allowed prompt so small probes squeak through while
# real ~36k-token traffic 402s into the local model. That blind spot is closed for free by
# check_openrouter_balance() below — asserting the balance directly, rather than paying tokens
# to rediscover it. Use --prompt-tokens 36000 for a deliberate, occasional deep probe.
DEFAULT_PROMPT_TOKENS = 2_000
_FILLER = "the quick brown fox jumps over the lazy dog. "

# OpenRouter starts shrinking the allowed prompt as credit drains (observed 2026-07-09:
# 17043 -> 13342 -> 5567 tokens within one hour, at a balance near zero). Below this many
# dollars, cloud-fallback can no longer serve an agent-sized prompt and is effectively dead.
MIN_FALLBACK_BALANCE_USD = 2.0
OPENROUTER_CREDITS_URL = "https://openrouter.ai/api/v1/credits"

# Models fit to drive a tool-using agent. Matched as substrings against the served model id,
# which carries provider prefixes and date suffixes (e.g. "z-ai/glm-5.2-20260616").
# nemotron-super-49b: NIM fallback, tool-calling verified 2026-07-09 (config note) and
# re-proven live 2026-07-12/13 (this probe's own tool_calls=["ping"] passing + real agent
# runs) — without this marker the probe reported a healthy 49B-served fleet as degraded.
# nemotron-3-ultra: OpenRouter free tier, tool-calling verified 2026-07-10 before wiring.
COMPETENT_MARKERS = (
    "glm-coding",
    "glm-5.2",
    "glm-5-turbo",
    "glm-4.6v",
    "nemotron-super-49b",
    "nemotron-3-ultra",
)

# Known degradation targets. Listed explicitly so a NEW unknown model is reported as unknown
# rather than silently assumed competent — the exact assumption that caused the incident.
DEGRADED_MARKERS = ("qwen", "llama-3.2", "gemma", "qwen2.5", "ollama_chat")

PING_TOOL = {
    "type": "function",
    "function": {
        "name": "ping",
        "description": "Reply by calling this tool.",
        "parameters": {"type": "object", "properties": {}},
    },
}


class ProbeError(RuntimeError):
    """The probe could not reach a verdict (network, auth, malformed response)."""


def build_prompt(approx_tokens: int) -> str:
    """Filler sized to approximate a production agent prompt."""
    approx_chars = max(1, approx_tokens) * 4
    repeats = max(1, approx_chars // len(_FILLER))
    return _FILLER * repeats + "\nNow call the ping tool."


def classify(served_model: str) -> str:
    """Return 'competent', 'degraded', or 'unknown' for a served model id."""
    m = (served_model or "").lower()
    if not m:
        return "unknown"
    if any(marker in m for marker in DEGRADED_MARKERS):
        return "degraded"
    if any(marker in m for marker in COMPETENT_MARKERS):
        return "competent"
    return "unknown"


def evaluate(payload: dict) -> dict:
    """Reduce a chat-completion response to a verdict.

    A competent answer must satisfy BOTH: served by a capable model, and able to emit a tool
    call. A model that returns prose where a tool call was demanded will stall an agent just
    as surely as one that loops.
    """
    try:
        choice = payload["choices"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise ProbeError(f"malformed response: {payload!r:.200}") from exc

    served = str(payload.get("model") or "")
    tool_calls = choice.get("message", {}).get("tool_calls") or []
    tier = classify(served)
    usage = payload.get("usage") or {}

    return {
        "served_model": served,
        "tier": tier,
        "tool_calls": [t.get("function", {}).get("name") for t in tool_calls],
        "prompt_tokens": usage.get("prompt_tokens"),
        "competent": tier == "competent" and bool(tool_calls),
    }


def evaluate_balance(payload: dict, minimum_usd: float = MIN_FALLBACK_BALANCE_USD) -> dict:
    """Reduce an OpenRouter /credits response to a fallback-readiness verdict."""
    try:
        data = payload["data"]
        remaining = float(data["total_credits"]) - float(data["total_usage"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProbeError(f"malformed credits response: {payload!r:.200}") from exc
    return {"remaining_usd": round(remaining, 4), "fallback_ready": remaining >= minimum_usd}


def check_openrouter_balance(api_key: str, timeout: float = 15.0) -> dict:
    """Free (no token spend) assertion that cloud-fallback can still serve a real prompt."""
    req = urllib.request.Request(
        OPENROUTER_CREDITS_URL, headers={"Authorization": f"Bearer {api_key}"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise ProbeError(f"openrouter credits unreachable: {exc}") from exc
    return evaluate_balance(payload)


def probe(gateway: str, model: str, approx_tokens: int, timeout: float) -> dict:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": build_prompt(approx_tokens)}],
            "tools": [PING_TOOL],
            "max_tokens": 4096,
        }
    ).encode()

    req = urllib.request.Request(gateway, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise ProbeError(f"gateway HTTP {exc.code}: {exc.read()[:300]!r}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise ProbeError(f"gateway unreachable: {exc}") from exc

    if "error" in payload:
        raise ProbeError(f"gateway error: {str(payload['error'])[:300]}")
    return evaluate(payload)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gateway", default=DEFAULT_GATEWAY)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prompt-tokens", type=int, default=DEFAULT_PROMPT_TOKENS)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--json", action="store_true", help="emit the verdict as JSON")
    ap.add_argument(
        "--skip-balance-check",
        action="store_true",
        help="don't assert the cloud-fallback balance (offline / no OPENROUTER_API_KEY)",
    )
    args = ap.parse_args(argv)

    try:
        verdict = probe(args.gateway, args.model, args.prompt_tokens, args.timeout)
    except ProbeError as exc:
        print(f"PROBE-ERROR {exc}", file=sys.stderr)
        return 2

    # Free, token-free assertion that the rescue route can still serve a real agent prompt.
    # Without this, a drained balance stays invisible until the primary caps and agents loop.
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not args.skip_balance_check and api_key:
        try:
            balance = check_openrouter_balance(api_key)
            verdict |= balance
        except ProbeError as exc:
            print(f"BALANCE-CHECK-FAILED {exc}", file=sys.stderr)
            verdict["fallback_ready"] = None

    if args.json:
        print(json.dumps(verdict))
    else:
        print(
            f"{'COMPETENT' if verdict['competent'] else 'DEGRADED'} "
            f"served={verdict['served_model']} tier={verdict['tier']} "
            f"tools={verdict['tool_calls']} prompt_tokens={verdict['prompt_tokens']} "
            f"fallback_ready={verdict.get('fallback_ready')} "
            f"remaining_usd={verdict.get('remaining_usd')}"
        )

    if verdict.get("fallback_ready") is False:
        print(
            f"ALERT: cloud-fallback has ${verdict['remaining_usd']} remaining "
            f"(< ${MIN_FALLBACK_BALANCE_USD}). OpenRouter shrinks the allowed prompt as credit "
            "drains, so the rescue route will 402 on agent-sized prompts and traffic will fall "
            "to a local model. Top up before the primary's next quota cap.",
            file=sys.stderr,
        )
        if verdict["competent"]:
            return 1

    if not verdict["competent"]:
        print(
            f"ALERT: '{args.model}' is being served by '{verdict['served_model']}' "
            f"(tier={verdict['tier']}, tool_calls={verdict['tool_calls']}). "
            "Agents will degrade or loop. Check the primary's quota and the "
            "cloud-fallback balance.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
