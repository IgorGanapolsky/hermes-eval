# Hermes fleet setup — deep research, July 2026

Multi-agent deep research run 2026-07-07 (101 agents; 19 sources fetched; 95 claims extracted;
top 25 adversarially verified 3-vote: 23 confirmed, 2 refuted). Triggered by the 2026-07-07
z.ai weekly-quota exhaustion incident (27.8M tokens burned 2026-07-06 → silent fallback to a
3B model that confabulated tool calls).

## Verdict in one paragraph

The architecture (LiteLLM gateway, GLM subscription primary, local fallback, OpenRouter last
resort) is directionally correct but hand-rolls what LiteLLM v1.89 provides natively. Keep GLM
as cloud primary; never let a <8B model be a silent fallback; alert on burn rate BEFORE the
weekly quota dies; schedule heavy autonomous runs off-peak.

## Prioritized recommendations

### P0 — done 2026-07-07
1. **Remove Llama-3.2-3B from silent fallback paths.** BFCL V4: rank 98/109, 21.95% overall,
   **4.00% multi-turn** — effectively no tool-calling in agent loops. This was the verified
   mechanism of the garbage-JSON incident. Done: hermes-eval `f629796` (fallbacks now
   glm-coding → hermes-local qwen3:8b → cloud-fallback) + loud degradation warnings
   (mac-yolo-safeguards T-108). Source: https://gorilla.cs.berkeley.edu/leaderboard.html
2. **Keep z.ai GLM as cloud primary.** GLM-4.6 is the top open-weight tool-caller on BFCL V4:
   #4 overall, 72.38% (68% multi-turn) — ahead of Kimi-K2 (59.06%), DeepSeek-V3.2-Exp (56.73%),
   GPT-5.2 (55.87%). Caveat: board benchmarks GLM-4.6; the plan serves GLM-5.2 (unbenchmarked).
3. **Burn + degradation alerting.** Done as mac-yolo-safeguards T-109
   (`tools/hermes-burn-alert.js`, ntfy, 30-min LaunchAgent): >8M tokens/day and
   GLM-failing-degraded alerts. LiteLLM-native path (below) is the eventual upgrade.

### P1 — worth doing next (LiteLLM-native, needs setup)
4. **provider_budget_config** (7d window) as a calibrated USD proxy for the weekly prompt
   quota; router skips over-budget providers; loud 429 when all exhausted; remaining budget on
   `GET /provider/budgets` + Prometheus gauge. Requires per-token costs configured on the
   flat-rate plan (else spend registers $0); Redis only if both Macs' proxies must share
   budgets. https://docs.litellm.ai/docs/proxy/provider_budget_routing
5. **Cooldown/order hygiene** (verified in installed 1.89.4 source): defaults are 3 fails/min,
   5s cooldown, and **single-deployment model groups are EXEMPT from cooldown** — a lone
   quota-dead z.ai deployment never leaves rotation (each request pays a ~1–2s failing GLM
   attempt first; also what makes recovery automatic). Our cooldown_time is already 45s.
   Priority chains can be expressed natively via deployment `order`.
   https://docs.litellm.ai/docs/routing
6. **Alerting/monitoring built-ins** when we outgrow the T-109 script: webhook alerting
   (llm_exceptions, cooldown_deployment, budget threshold_crossed 85/95%,
   projected_limit_exceeded on soft_budget), Prometheus /metrics (OSS again since v1.80.0;
   alert on `litellm_deployment_successful_fallbacks` = the exact silent-degradation detector).
   /metrics is UNAUTHENTICATED by default — enable require_auth_for_metrics_endpoint on a
   Tailscale-exposed box. Budget features need Postgres; enforcement had bypass regressions
   (#26672, #27381, #20324) — smoke-test, don't trust docs.
   https://docs.litellm.ai/docs/proxy/prometheus https://docs.litellm.ai/docs/proxy/alerting
7. **Per-session caps** via A2A gateway (max_iterations 429s a runaway session;
   max_budget_per_session) — only for A2A-registered agents, presence in 1.89 unverified;
   complements HERMES_MAX_TURN_SECONDS. https://docs.litellm.ai/docs/a2a_iteration_budgets

### P2 — quota economics & local tier
8. **GLM Coding Plan quota mechanics** (docs.z.ai, verified 2026-07-07): two windows — 5-hour
   AND weekly (Lite ~80/5h + ~400/wk; Pro ~400/5h + ~2,000/wk; Max ~1,600/5h + ~8,000/wk).
   Weekly = 5× the 5h cap → ~25 maxed hours exhausts a week. GLM-5.2/GLM-5-Turbo deduct **3×
   during peak (14:00–18:00 UTC+8 = 2–6 AM ET) and 2× off-peak (1× off-peak promo through
   Sept 2026)**. US daytime is off-peak — schedule heavy cron/autonomous runs accordingly;
   GLM-4.7/4.5-Air stay 1× all day (cheap-quota lever for background jobs). Lite $18/mo
   ($12.60 promo). https://docs.z.ai/devpack/overview
9. **Local tier refresh:** Qwen3.5-9B is the best ≤14B tool-caller in llm-stats' BFCL-V4
   coverage (0.661; steep cliff below 9B: 4B=0.503, 2B=0.436) — medium confidence (sparse
   13-model coverage). Piloting: pulled qwen3.5:9b into Ollama 2026-07-07; smoke-test native
   tool-calls before swapping into hermes-local. The claim that qwen3:8b beats qwen3:14b on
   BFCL was REFUTED 0-3 — don't cite it. https://llm-stats.com/benchmarks/bfcl-v4

### DeepSeek-V4 (evaluated separately, same day — Igor's unsloth link)
- **Local: skip.** V4-Flash GGUF needs 92GB RAM at 1-bit (162GB at 4-bit); no fit on any
  fleet Mac; llama.cpp support still WIP. No V4 distills exist; old R1-distills lack reliable
  tool-calling. https://unsloth.ai/docs/models/deepseek-v4
- **API: pilot as a quota-escape valve.** V4-Flash $0.14/M in / $0.28/M out, 1M ctx — the
  27.8M-token burn day ≈ **$2–6/day** on it. Red flag to gate on: open issue #1244 (V4-Pro
  intermittently emits tool calls as plain text — the exact incident failure mode); Flash
  unverified. Needs: prepay $5–10, add `deepseek/deepseek-v4-flash` as a fallback tier, run a
  20-call tool-calling smoke test. https://api-docs.deepseek.com/quick_start/pricing
- Note: `deepseek-chat`/`deepseek-reasoner` API names retire 2026-07-24.

## Refuted claims (do not reuse)
- "qwen3:8b (42.57%) outperforms qwen3:14b; xLAM-2-8b-fc-r (46.68%) is the best ≤14B" — 0-3.
- "Redis is a hard requirement for provider budget tracking" — 1-2 (single instance uses
  in-memory cache).

## Open questions
- GLM-5.2's actual BFCL score (plan serves it; board has 4.6 only).
- Subscription head-to-head per usable agent-prompt (GLM vs Claude/Kimi/MiniMax) — one blog
  (jia.je, July 2026 update) rates the restructured GLM plan's value "dropped from S to C
  level vs Kimi/MiniMax"; not adversarially verified, watch this.
- Qwen3.5-9B Ollama tool-template reliability in real agent loops (pilot in flight).
- USD-cost calibration that faithfully mirrors prompt-count quotas incl. peak multipliers.
