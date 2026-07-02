# FLEET.md — agent-maintained fleet wiki

Synthesized state of the Hermes fleet, in the [wiki-memory](https://www.langchain.com/blog/wiki-memory)
pattern: instead of every agent re-deriving fleet topology from raw sources (traffic logs,
launchd plists, shell archaeology), this file IS the precomputed synthesis. Any agent
(Claude Code, Codex, Cursor, Gemini, Hermes itself) working on this fleet reads this first
and **updates it in the same commit as any routing/topology change**.

> Maintenance contract: if you change `litellm/config.yaml`, a launchd service, or a
> machine's role, update this file in the same PR/commit. Stamp the verified date.
> Raw evidence lives in `~/.hermes/litellm-logs/traffic.jsonl` (every served call).

## Topology (verified 2026-07-02)

| Node | Tailscale IP | Role |
|---|---|---|
| Mac Pro | `100.87.85.85` | Runs the LiteLLM proxy `*:4010` (launchd `com.igor.hermes-litellm` → `litellm/start-proxy.sh`). Primary Hermes agent (v0.17.0). Local Ollama `:11434`. |
| Mac mini | `100.94.135.78` | Fallback/secondary. Ollama exposed to tailnet via forwarder `:11436 → 127.0.0.1:11434` (launchd `com.igor.ollama-tailnet`). 24GB — chronically RAM-pressured; don't co-locate heavy work. |

Both machines' Hermes use `provider: custom:litellm-gateway` and send a model NAME;
the proxy decides where it runs. LAN IPs are NOT stable — always Tailscale.
Proxy auth: `LITELLM_MASTER_KEY` (local-dev default documented in SECURITY.md; tailnet-only exposure).

## Model map (verified 2026-07-02)

| Proxy name | Actual model | Where | Cost | Notes |
|---|---|---|---|---|
| `glm-coding` | GLM-5.2 | z.ai Coding Plan `api.z.ai/api/coding/paas/v4` | flat monthly (already paid) | **Fleet default.** 1M ctx / 128K out. Reasons heavily — give ≥300 max_tokens. |
| `glm-turbo` | GLM-5-Turbo | same subscription/endpoint/key | flat monthly | Mid-tier: faster, lighter on quota. 200K ctx / 128K out. Also reasons heavily. |
| `hermes-local` | qwen3:8b-64k | Ollama, load-balanced Pro+mini | $0 | `think:false` required or content comes back empty. |
| `hermes-local-fast` | qwen2.5:3b-64k | Mac Pro Ollama | $0 | Smallest/fastest local. |
| `hermes-coder` | qwen2.5-coder:14b-64k | mini Ollama | $0 | Opt-in only; avoid under mini memory pressure. |
| `hermes-gemma` | gemma4-12b (LiteRT `:9379`) | Mac Pro | $0 | Cross-family eval JUDGE. |
| `cloud-fallback` | GLM-5.2 via OpenRouter | cloud | **per-token** | Last resort only — double-charges on top of the subscription. |
| `escalation` | sakana/fugu-ultra via OpenRouter | cloud | $5/$30 per 1M | OFF by default, deliberate invocation only. |

**Fallback chains:** `glm-coding → hermes-local → cloud-fallback`;
`glm-turbo → hermes-local → cloud-fallback` (NOT via glm-coding — a z.ai 429 caps both
subscription routes); `hermes-local → hermes-local-fast → glm-coding → cloud-fallback`.
Context overflow: `hermes-local → glm-coding`, `glm-turbo → glm-coding` (>200K).
Chains are config-declared; behaviorally fire-drilled only for local-node death, not z.ai 429.

## Gotchas that cost real debugging time

1. **z.ai ≠ OpenRouter.** "Use GLM" means the subscription (`glm-coding`/`glm-turbo`).
   A missing `Z_AI_API_KEY` in `~/.hermes/.env` silently falls back to per-token OpenRouter.
2. **GLM models reason.** Tiny `max_tokens` returns empty content (reasoning ate the budget).
3. **Proxy curl needs `Content-Type: application/json`** or LiteLLM sees `model=None` (400).
4. **`launchctl setenv` doesn't reach already-open shells** — test routing with inline env vars.
5. **Verify with the traffic log, not vibes:** snapshot `wc -l traffic.jsonl` before, read only
   new lines after. `ollama ps` empty during a run = cloud was used.
6. New env key ⇒ add an `export` line in `start-proxy.sh` + `launchctl kickstart -k
   gui/$UID/com.igor.hermes-litellm`.

## Quota notes (2026-07-02)

- z.ai campaign through 2026-07-31: 0.67x metering ("1.5x quota") — **confirmed only inside
  the ZCode desktop client**; whether plain coding-endpoint API traffic gets it is UNVERIFIED.
- ZCode itself: GUI-only (no CLI/headless) → not integrable into this headless fleet; skipped.

## Version log

| Date | Change |
|---|---|
| 2026-07-02 | Added `glm-turbo`; registered true ctx windows; proven end-to-end from both nodes (`9c25c48`). Mini Hermes update v0.16.0→current in progress (canary before Pro). |
| 2026-06-30 | z.ai Coding Plan subscription made cloud primary over OpenRouter. |
| 2026-06-26 | Fleet default routed through this proxy; every call logs to traffic.jsonl. |
