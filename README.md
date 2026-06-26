# hermes-eval

[![CI](https://github.com/IgorGanapolsky/hermes-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/IgorGanapolsky/hermes-eval/actions/workflows/ci.yml)
[![CodeQL](https://github.com/IgorGanapolsky/hermes-eval/actions/workflows/codeql.yml/badge.svg)](https://github.com/IgorGanapolsky/hermes-eval/actions/workflows/codeql.yml)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/IgorGanapolsky/hermes-eval/badge)](https://scorecard.dev/viewer/?uri=github.com/IgorGanapolsky/hermes-eval)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A LiteLLM gateway + LLM evaluation pipeline for a self-hosted agent fleet ("Hermes")
running local Ollama models across multiple machines (Mac Pro + Mac mini), with a
cloud fallback.

Two things live here:

1. **`litellm/`** — a LiteLLM proxy that fronts the whole fleet: one OpenAI-compatible
   endpoint, cross-host load-balancing, health-checked failover, and (when you add
   Postgres) full prompt/response logging.
2. **`eval/`** — an evaluation pipeline that turns those logs into a **pre-deploy quality
   gate**: golden datasets, an LLM-as-judge with a rubric, RAG faithfulness + semantic
   similarity, a regression check against a baseline, and a CI gate that blocks a
   merge/release when quality drops.

## The thesis

> I tracked model drift operationally (streaming metrics). But stream detection is
> *reactive* — by the time it fires, users already got the bad output. The next step is
> moving that same distributional thinking **left**, into a pre-deploy gate with a
> labeled golden set, so a regression is **blocked before it ships** instead of alerted
> after. The two are complementary, and they close a loop: failures the stream surfaces
> become new golden cases that harden the gate.

The single line that connects the gateway and the eval pipeline is in `litellm/config.yaml`:

```yaml
# store_prompts_in_spend_logs: true   # every fleet call's prompt+response -> your golden-set / drift feed
```

## Architecture

```
        Hermes (Mac Pro / Mac mini)
                  │
                  ▼
         LiteLLM proxy :4000 ──► local Ollama fleet (qwen3, qwen2.5, gemma) + OpenRouter fallback
                  │
                  ▼
      Postgres spend_logs (prompt + response + tokens, every call)   [optional, off by default]
                  │
       curate ────┴──── nightly drift sample
          │                     │
          ▼                     ▼
   golden.jsonl  ─────►  eval harness (promptfoo): LLM-judge + RAG faithfulness + similarity
                                │
                                ▼
              commit status `eval/golden-gate` (pass/fail @ SHA)  ──►  blocks merge/release
```

## Quickstart

```bash
# 0. deps: an Ollama with a chat model + an embedding model, Node (for promptfoo), Python 3.9+
ollama pull qwen3:8b-64k          # or your model; match litellm/config.yaml
ollama pull nomic-embed-text      # for the `similar` assertion
uv tool install 'litellm[proxy]'  # or: pip install 'litellm[proxy]'

# 1. start the gateway
export LITELLM_MASTER_KEY=sk-hermes-local-dev
export OPENROUTER_API_KEY=...     # for the cloud fallback (optional)
litellm --config litellm/config.yaml --port 4000        # `make proxy`

# 2. sanity-check it routes to a local model
curl -s http://127.0.0.1:4000/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"hermes-local","messages":[{"role":"user","content":"ping"}]}' | jq .

# 3. run the eval gate (exits non-zero if pass-rate < threshold)
make gate            # or: EVAL_THRESHOLD=0.85 bash eval/run_gate.sh
```

## The five eval components (and where each lives)

| # | Component | What it does | Files |
|---|---|---|---|
| 1 | **Golden set** | versioned JSONL of answerable + refusal cases with reference answers, contexts, strata, canary | `eval/golden.jsonl`, `eval/corpus/` |
| 2 | **LLM-as-judge** | rubric-graded pass/fail; judge is a *different model family* than the SUT (anti self-preference); validated against human labels | `eval/rubric.txt`, `eval/load_golden.py`, `eval/validate_judge.py` |
| 3 | **RAG / similarity** | faithfulness (hallucination guard) + cosine similarity to the reference | `eval/promptfooconfig.yaml` (`context-faithfulness`, `similar`) |
| 4 | **Regression vs drift** | diff aggregate + per-critical-row vs a committed baseline; tolerance band for nondeterminism | `eval/compare_baseline.py`, `eval/run_gate.sh` |
| 5 | **CI gate** | local runner posts `eval/golden-gate` commit status; Actions runs a cheap cloud smoke | `eval/run_gate.sh`, `.github/workflows/eval-gate.yml` |

Bootstrapping when you have no data yet (cold start): `eval/synth_golden.py` generates
synthetic candidates from `eval/corpus/`, **for human review** — they are not golden until
curated. Replace them with mined production traces as the LiteLLM logs fill up.

## LLM-as-judge: the failure modes this guards against

| Bias | Mitigation in this repo |
|---|---|
| **Self-preference** (judge favors its own family) | judge = `hermes-gemma`, SUT = `qwen3` — different families |
| **Verbosity** (favors longer answers) | rubric explicitly says "ignore length, tone, formatting" |
| **Score clustering** (everything gets 4/5) | binary PASS/FAIL, not a 1–5 scale |
| **Untrusted judge** | `validate_judge.py` reports Cohen's κ + TPR/TNR vs human labels before you trust it |

## The CI gate, two tiers

- **Tier 1 (authoritative): local runner.** The Ollama fleet isn't reachable from
  GitHub-hosted runners, so the real model-drift gate runs on the Mac Pro against the
  proxy and posts a `eval/golden-gate` commit status (`eval/run_gate.sh`). Add that status
  to branch protection and nothing ships under threshold.
- **Tier 2 (cheap PR smoke): GitHub Actions.** `.github/workflows/eval-gate.yml` runs the
  `ci-smoke` subset against a cloud model via OpenRouter — catches prompt-logic
  regressions fast, without the local fleet.

## Honest status / limitations

- The golden set here is a small **seed** (8 rows) over a toy `corpus/pricing.md`. Real use
  needs 20–50+ cases mined from real failures; size by per-slice confidence interval, not a
  magic global number.
- **No judge κ is claimed until measured** — run `make validate-judge` to get a real number.
  A low κ means fix the rubric/judge before trusting the gate.
- Spend-log capture (the golden-set feed) is **off by default** — it needs a Postgres URL.
- `compare_baseline.py` is defensive about promptfoo's `results.json` shape, which changes
  across versions; verify against your installed version.
- The Mac mini deployment is wired in `config.yaml` but is only live when the mini is on the
  LAN; LiteLLM benches it automatically when it's offline.

See `EVIDENCE.md` for a real, dated test run (raw command output).

## Layout

```
litellm/config.yaml         the gateway (fleet + routing + fallback)
eval/golden.jsonl           the golden set
eval/corpus/pricing.md      toy RAG corpus the goldens are grounded in
eval/rubric.txt             the judge rubric
eval/promptfooconfig.yaml   the eval (SUT + judge + embedding, all via the proxy)
eval/load_golden.py         golden.jsonl -> promptfoo tests (subset/embedding flags)
eval/run_gate.sh            the gate (pass-rate + regression + commit status)
eval/compare_baseline.py    regression check vs baseline
eval/validate_judge.py      judge calibration vs human labels (Cohen's kappa)
eval/synth_golden.py        cold-start synthetic candidate generation
.github/workflows/          Tier-2 cloud smoke gate
Makefile                    proxy / eval / gate / synth / validate-judge / baseline
```
