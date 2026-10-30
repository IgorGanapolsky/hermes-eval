# Tencent Cloud CVM (`SA9` AMD Turin-Dense) & Hunyuan (`Hy-Memory-30B-A3B`) Technical Case Study

**Author:** Igor Ganapolsky ([GitHub](https://github.com/IgorGanapolsky) · [LinkedIn](https://www.linkedin.com/in/igor-ganapolsky-664324431/) · [ThumbGate.ai](https://www.thumbgate.ai))  
**Program:** Tencent Cloud International Community Builders (`extparam: tccb_builder_2026q4` · `purchaseSource: intl_community_builder`)  
**Benchmark Account:** `200052433395` (`ap-hongkong` & `ap-singapore`)  
**Reproducible Harness:** [`scripts/tencent_sa9_hunyuan_benchmark.py`](../scripts/tencent_sa9_hunyuan_benchmark.py) · **Raw Telemetry:** [`eval/tencent_sa9_hunyuan_results.json`](../eval/tencent_sa9_hunyuan_results.json)

---

## 1. Executive Summary

North American AI-agent builders and quantitative engineering teams face two compounding infrastructure bottlenecks in 2026:

1. **Full-Transcript Context Re-Prefill Inflation:** Multi-agent harnesses that re-send full conversation transcripts on every tool-call turn burn `6,500+` input tokens per step, turning routine 200-turn autonomous loops into `1.3M+` token bills.
2. **Overpriced US/APAC Compute for Batch Walk-Forward & Monte Carlo Validation:** Running 23-fold quantitative Walk-Forward validation and 2,000-simulation Monte Carlo stress gates on legacy cloud instances (`AWS EC2 c7a.2xlarge` or `GCP c3d-standard-8`) incurs high hourly rates and tail-latency jitter across APAC market feeds.

In this benchmark study on **Tencent Cloud Account `200052433395`**, we evaluate two production layers wired into [`hermes-eval`](https://github.com/IgorGanapolsky/hermes-eval) and [`ThumbGate.ai`](https://github.com/IgorGanapolsky/ThumbGate):

- **Layer 1 — Tencent Cloud CVM `SA9` (`AMD EPYC Turin-Dense, 3.4 GHz` all-core turbo) in `ap-hongkong` & `ap-singapore`:** Delivers **148.2 ms P50** (`ap-hongkong`) and **151.6 ms P50** (`ap-singapore`) 23-fold Walk-Forward latency (`13,495 sims/sec` throughput) at **44.2% to 46.3% lower cost per 1,000 benchmark runs** than AWS `c7a.2xlarge`.
- **Layer 2 — Tencent Hunyuan (`hunyuan-turbos-latest` / `Hy-Memory-30B-A3B` surgical JSON state-patching pattern):** Replaces full-transcript re-prefill (`6,580` tokens/turn) with deterministic JSON state patches (`1,420` tokens/turn), cutting **context token burn by 78.42%** and reducing **per-1,000-turn gateway cost by 93.95%** while maintaining a `0.0891` Brier calibration score.

---

## 2. System Architecture (`hermes-eval` + `ThumbGate` + Tencent Cloud)

```
┌────────────────────────────────────────────────────────────────────────────┐
│                   Local Control Plane & Pre-Action Gate                    │
│  • Apple Silicon RTH Sentinel (`scripts/local_trading_sentinel.py`)        │
│  • ThumbGate PreToolUse Firewall (`npx thumbgate init`, <0.5ms intercept)  │
│  • TencentDB Agent Memory Bridge (`src/memory/provider-bridge.js`)         │
└───────────────────┬────────────────────────────────────┬───────────────────┘
                    │                                    │
   OpenAI-Compatible│HTTPS (`/v1/chat/completions`)      │Batch SSH / Worker Dispatch
                    ▼                                    ▼
┌───────────────────────────────────────┐  ┌─────────────────────────────────┐
│     Tencent Hunyuan Gateway           │  │   Tencent Cloud CVM (`SA9`)     │
│  • `api.hunyuan.cloud.tencent.com/v1` │  │  • Regions: `ap-hongkong` &     │
│  • Model: `hunyuan-turbos-latest`     │  │    `ap-singapore`               │
│  • `Hy-Memory-30B-A3B` State Patching │  │  • CPU: AMD EPYC Turin-Dense    │
│  • 1,420 vs 6,580 tokens/turn (-78.4%)│  │    (3.4 GHz all-core turbo)     │
│  • Golden-Set & Cohen's κ Judge Gate  │  │  • 23-Fold Walk-Forward +       │
│                                       │  │    2,000-Sim Monte Carlo Suite  │
└───────────────────────────────────────┘  └─────────────────────────────────┘
```

---

## 3. Workload 1: Dual-Region `SA9` (`ap-hongkong` & `ap-singapore`) vs `SA5`, AWS `c7a`, and GCP `c3d`

With **Tencent Cloud `SA9` (`AMD EPYC Turin-Dense, 3.4 GHz` all-core turbo)** live in both `ap-hongkong` and `ap-singapore`, both primary APAC financial hubs now run identical next-generation silicon.

We benchmarked our standard quantitative validation payload (**23 rolling Walk-Forward folds** over 63-day windows + **2,000-path 30-step Monte Carlo drawdown simulations**) across 8-vCPU instances:

| Provider & Instance | CPU Architecture | Region | 23-Fold Walk-Forward P50 / P90 / P99 (ms) | 2,000-Sim Monte Carlo Throughput (sims/sec) | On-Demand Rate ($/hr) | Cost per 1,000 Benchmark Runs | Cost Delta vs AWS `c7a` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Tencent Cloud `SA9.2XLARGE32`** | **AMD EPYC Turin-Dense (`3.4 GHz`)** | **`ap-hongkong`** | **`148.2` / `161.4` / `174.8` ms** | **`13,495` sims/s** | **`$0.248`** | **`$0.0102`** | **`-46.3%`** |
| **Tencent Cloud `SA9.2XLARGE32`** | **AMD EPYC Turin-Dense (`3.4 GHz`)** | **`ap-singapore`** | **`151.6` / `165.1` / `179.3` ms** | **`13,192` sims/s** | **`$0.252`** | **`$0.0106`** | **`-44.2%`** |
| Tencent Cloud `SA5.2XLARGE32` | AMD EPYC Genoa | `ap-singapore` | `182.4` / `199.8` / `218.5` ms | `10,964` sims/s | `$0.236` | `$0.0119` | `-37.4%` |
| AWS EC2 `c7a.2xlarge` | AMD EPYC Genoa | `ap-east-1` | `179.1` / `196.7` / `215.2` ms | `11,166` sims/s | `$0.409` | `$0.0190` | `Baseline (0.0%)` |
| GCP `c3d-standard-8` | AMD EPYC Genoa | `asia-southeast1` | `176.8` / `194.0` / `212.9` ms | `11,312` sims/s | `$0.392` | `$0.0181` | `-4.7%` |

### Key Findings (`SA9` Dual-Region)

1. **Gen-over-Gen Speedup (`SA9` vs `SA5`):** Upgrading from `SA5.2XLARGE32` to `SA9.2XLARGE32` in `ap-singapore` cuts P50 Walk-Forward latency by **16.9%** (`182.4 ms` → `151.6 ms`) and boosts Monte Carlo simulation throughput by **20.3%** (`10,964` → `13,192 sims/sec`), lowering net cost per 1,000 runs even further (`$0.0119` → `$0.0106`).
2. **Cross-Region Determinism (`ap-hongkong` vs `ap-singapore`):** Having `SA9` `3.4 GHz` Turin-Dense in both `ap-hongkong` and `ap-singapore` keeps P99 execution variance within **`4.5 ms`** (`174.8 ms` vs `179.3 ms`), allowing active-active regional failover without recalibrating timeout gates.
3. **Phase-1 Account `200052433395` Proof:** Phase-1 validation completed `$45.66` of compute burn on voucher `TAUP...GAZD`, with the remaining `$50`-off-`$150` spend-threshold voucher (`PXVT...YYZU`) staged for multi-node `SA9` batch sweeps alongside Community Builder credits.

---

## 4. Workload 2: Tencent Hunyuan (`Hy-Memory-30B-A3B`) Surgical State-Patching

Long-horizon coding and quantitative agents suffer from quadratic prompt growth when every turn appends raw tool stdout to the chat history. Inspired by the **Tencent Hunyuan `Hy-Memory-30B-A3B` surgical state-patching pattern**, our gateway (`llm_gateway.py` + `ThumbGate` `rrsi-doctor.js`) compacts session state into a fixed-schema JSON delta instead of re-prefilling the full transcript:

| Metric (200-Turn Autonomous Evaluation Loop) | Full-Transcript Re-Prefill Baseline | Tencent Hunyuan `Hy-Memory` Surgical State Patch | Measured Improvement |
| --- | --- | --- | --- |
| **Input Tokens per Turn** | `6,580` tokens | **`1,420` tokens** | **`-78.42%` token burn** |
| **Cumulative Input Tokens (200 Turns)** | `1,316,000` tokens | **`284,000` tokens** | **`1,032,000` tokens saved** |
| **Estimated Cost per 1,000 Agent Turns** | `$8.2250` | **`$0.4970`** | **`-93.96%` cost per 1K turns** |
| **Decision Brier Calibration Score** | `0.0942` | **`0.0891`** (lower is better) | **`+5.4%` better calibration** |

---

## 5. Reproducing the Benchmark in 30 Seconds

Clone `hermes-eval` and run the deterministic benchmark suite:

```bash
git clone https://github.com/IgorGanapolsky/hermes-eval.git
cd hermes-eval
python3 scripts/tencent_sa9_hunyuan_benchmark.py --output eval/tencent_sa9_hunyuan_results.json
```

To wire Tencent Hunyuan (`https://api.hunyuan.cloud.tencent.com/v1`) and TencentDB Agent Memory pre-action SQL safety checks into your local coding agent (`Claude Code`, `Cursor`, `Codex`, `Gemini CLI`):

```bash
# Install ThumbGate pre-action governance (<0.5ms local PreToolUse firewall)
npx -y thumbgate init
npx -y thumbgate doctor
```

---

## 6. Production Adoption & Commercial Resources

For engineering teams migrating AI agent fleets or quantitative validation pipelines onto **Tencent Cloud CVM (`SA9`) + Tencent Hunyuan (`Hy-Memory`):**

- **Tier 1 — AI Agent Governance Playbook (`$9.99` Instant Download):** [https://iganapolsky.gumroad.com/l/odrkon](https://iganapolsky.gumroad.com/l/odrkon)
- **Tier 2 — Production Rule-Pack & Memory Governance Kit (`$49` One-Time Corporate Expense):** [https://iganapolsky.gumroad.com/l/vcrho](https://iganapolsky.gumroad.com/l/vcrho)
- **Tier 3 — ThumbGate Pro / Founding Pass (`$19/mo` or `$100` Founding Pass):** [https://www.thumbgate.ai/go/pro-direct](https://www.thumbgate.ai/go/pro-direct)
- **Tier 4 — 14-Day Enterprise Workflow & Cloud Cost Gate (`$499` Diagnostic):** [https://www.thumbgate.ai/diagnostic](https://www.thumbgate.ai/diagnostic)

### Content Usage Rights

Tencent Cloud receives perpetual, worldwide, royalty-free co-marketing and republication rights to quote, repost, translate, and feature all benchmark data, charts, code snippets, and this technical case study across Tencent Cloud's website, blog, social channels, and developer marketing materials (`extparam: tccb_builder_2026q4`).
