# EVIDENCE — tested 2026-06-25 on the Mac Pro

Raw proof that this repo runs, not a "should work." Every claim below has a command and
its actual output. Honest negatives are included.

Environment: LiteLLM **1.89.4** (uv tool), Node v22.22.3, Python 3.14 (system) / 3.13 (litellm's uv venv),
Ollama on 127.0.0.1:11434, OpenRouter key from `~/.hermes/.env`.

## Summary

| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | Python scripts compile | ✅ PASS | `py_compile` clean |
| 2 | `run_gate.sh` shell syntax | ✅ PASS | `bash -n` clean |
| 3 | `golden.jsonl` valid JSON | ✅ PASS | 8/8 lines parse |
| 4 | Adapter → promptfoo tests | ✅ PASS | 8 full / 3 ci-smoke, correct asserts |
| 5 | All YAML parses | ✅ PASS | litellm + 2 promptfoo + workflow |
| 6 | LiteLLM proxy boots + serves | ✅ PASS | `/v1/models` lists 5 groups, liveliness OK |
| 7 | Port-conflict handling | ✅ PASS | :4000 held by `node pricing-proxy.js`; moved to :4010 |
| 8 | Cloud path + cost tracking | ✅ PASS | GLM-5.2 → `content:"pong"`, `cost:$0.00033` |
| 9 | Judge calibration script | ✅ PASS | n=4, agreement 1.0, **Cohen's κ=1.0** (small-n caveat) |
| 10 | Eval gate PASSES legitimately | ✅ PASS | 3/3, 0 errors, exit 0 |
| 11 | Eval gate BLOCKS on threshold | ✅ PASS | threshold 1.01 → exit 1 |
| 13 | Local model end-to-end (SUT=qwen2.5:3b-64k via proxy) | ✅ PASS | after `brew services restart ollama`: gate 3/3, all asserts ok, **refusal correct** |
| 12 | Gate discriminates (catches a hallucination) | ✅ PASS | faithful→1.00 PASS, hallucinated→0.00 FAIL |
| — | Mac mini in the fleet | ❌ OFFLINE | `192.168.1.172` unreachable (ping fail) |

## Detailed runs

### Proxy serves the fleet (port 4010)
```
$ curl .../health/liveliness            -> "I'm alive!"
$ curl .../v1/models                    -> hermes-local, hermes-local-fast, hermes-gemma, cloud-fallback, text-embedding
```

### HONEST NEGATIVE — local inference is wedged right now
```
$ time curl 127.0.0.1:11434/v1/chat/completions  (qwen3:8b-64k, DIRECT, bypassing proxy)   -> 2:00 TIMEOUT
$ time curl .../api/generate qwen2.5:3b                                                      -> 1:00 TIMEOUT
$ time curl .../api/generate phi4-mini                                                       -> 1:00 TIMEOUT
$ ollama ps                                                                                  -> (empty, nothing loaded)
$ time curl .../v1/chat/completions hermes-gemma (LiteRT 9379)                               -> 1:30 TIMEOUT
$ memory_pressure                                                                            -> free 12%
```
Conclusion: **every local model times out** (Ollama *and* LiteRT), so the Mac Pro cannot serve
inference at the moment — which also means the Hermes fleet's default model is currently down.

**Correction (re-check 2026-06-26):** I first blamed memory pressure (12% free). Re-tested with a 200s
timeout: `qwen2.5:3b` STILL returns nothing and `ollama ps` is empty, while free RAM is now **55%**.
So memory was NOT the cause — the Ollama daemon itself is wedged and needs a restart. My 12%-RAM
attribution was wrong.

**RESOLVED 2026-06-26:** `brew services restart ollama` (it's the brew launchd service
`homebrew.mxcl.ollama`, not the desktop app) brought it back — `qwen2.5:3b-64k` generates again
(~45s cold, fast warm); `nomic-embed-text` returns 768-dim vectors. The **local end-to-end gate then
PASSED**: SUT=qwen2.5:3b-64k (local, via proxy), judge=gpt-4o-mini (cloud, cross-family),
embeddings=local nomic → **3/3, 0 errors**. rag-0001/0002 grounded + correct (rubric+faithfulness+similar
all ok); **rag-0006 correctly REFUSED** the unanswerable question. Local-model E2E gap CLOSED.

### Gate discrimination (the proof that it catches bad answers, not just arithmetic)
```
$ promptfoo eval -c trap.yaml   (same judge gpt-4o-mini, same context-faithfulness assertion)
  [PASS] FAITHFUL answer     -> Faithfulness 1.00 >= 0.6
  [FAIL] HALLUCINATED answer -> Faithfulness 0.00 <  0.6   (unlimited keys / 24-7 phone support: not in context)
  stats: {successes:1, failures:1, errors:0}
```
Note: GLM's chain-of-thought ("Thinking:...") leaked into the graded output; the judge scored
faithfulness correctly anyway, but production should strip `reasoning_content` before grading.
The LiteLLM proxy correctly surfaced this as connect-errors + retries + a 120s timeout in its log.

### Cloud path works (proves routing/auth/cost, and is the failover target)
```
$ curl .../v1/chat/completions  model=cloud-fallback (openrouter/z-ai/glm-5.2)
  -> { "finish":"stop", "content":"pong", "usage":{... "cost":0.00033459 }}
```

### Judge calibration
```
$ python3 validate_judge.py --labels judge_labels.example.jsonl   (judge via cloud)
  jl-1 human=PASS judge=PASS ok | jl-2 human=FAIL judge=FAIL ok | jl-3..4 ok
  n=4 agreement=1.000 cohen_kappa=1.000 TPR=1.000 TNR=1.000  -> judge usable (kappa>=0.6)
```
Caveat: n=4 on easy cases proves the *mechanism*, not a production κ. Real calibration needs
30+ human-labeled rows including hard/ambiguous ones.

### The gate — both directions (SUT=GLM-5.2, judge=gpt-4o-mini, cross-family)
```
$ EVAL_THRESHOLD=0.8  bash eval/run_gate.sh promptfooconfig.ci.yaml
  Results: 3 passed (100%), 0 failed, 0 errors
  pass-rate=1.0 (pass=3 fail=0 errors=0)  ->  GATE PASSED   exit 0

$ EVAL_THRESHOLD=1.01 bash eval/run_gate.sh promptfooconfig.ci.yaml
  pass-rate=1.0  ->  GATE FAILED   exit 1   (blocks the merge)
```

## Bugs found AND fixed during testing (this is why we test)

1. **`context-faithfulness` requires a `query` var** — first gate run threw 2 errors; the assertion
   needs `vars.query`, not `question`. Fixed in `eval/load_golden.py` (set both).
2. **The gate counted errors as a pass** — first run reported `pass-rate=1.0 GATE PASSED` despite 2
   eval *errors* (it divided successes by successes+failures, ignoring `errors`). A broken eval would
   have shipped. Fixed in `eval/run_gate.sh`: errors are now in the denominator and force a hard fail.

## Not yet proven (honest)
- Local-model end-to-end (SUT on qwen/gemma): blocked by the Ollama/LiteRT wedge above.
- The mini as a load-balanced peer: it's offline; config wires it, LiteLLM benches it.
- The GitHub Actions workflow: YAML validated; not executed (needs a PR + the `OPENROUTER_API_KEY` secret).
- A real judge κ and a real golden set sized by per-slice CI.

## High-ROI improvements — tested 2026-06-26

| Improvement | Result | Evidence |
|---|---|---|
| JSONL logging callback (gateway→golden-set feed) | ✅ | cache-miss probe + Hermes calls land in `traffic.jsonl`; 0 health-check pings leaked |
| `escalation` model (fugu-ultra, OFF by default) | ✅ | `/v1/models` lists `escalation`; in no fallback chain; not invoked (cost/latency by design) |
| Reasoning-strip transform | ✅ | unit-tested: strips qwen `<think>…</think>` and GLM `Thinking:…`; clean text unchanged |
| `make verify` (one command) | ✅ | boot→gate→teardown → GATE PASSED, exit 0 |
| Persistent proxy (launchd) | ✅ | `com.igor.hermes-litellm` loaded (KeepAlive); liveliness OK after `kickstart -k` |
| **Live Hermes repointed at the proxy** | ✅ | default=`custom:litellm-gateway`; `hermes -z` (no flag) → "ROUTED"; logged `qwen2.5:3b-64k` 11158 tok success; direct-Ollama fallback kept; backup `~/.hermes/config.yaml.bak.hermes-eval-20260626` |
| GitHub repo (public) | ✅ | https://github.com/IgorGanapolsky/hermes-eval — pushed after a clean secret scan |

Known minor: LiteRT gemma judge still down (separate server, unrelated to Ollama); background health checks disabled to stop wasteful 28s probes + log noise. To revert the Hermes repoint: `cp ~/.hermes/config.yaml.bak.hermes-eval-20260626 ~/.hermes/config.yaml`.

## Tinker tool-use training and promotion gate — 2026-07-21

This work fixes the evidence path before another paid fine-tuning run. It does **not** claim a new
Tinker training run, a better candidate, or candidate adoption.

### Private dataset audit

The local Tinker source contained 4,408 rows. After exact-content deduplication it contained 4,239
unique rows: 169 duplicate rows across 14 duplicate groups, with one conversation repeated 152
times. A deterministic SHA-256 split now produces 3,824 training rows and 415 held-out rows. The
held-out set contains 278 assistant targets with tool calls and is written with owner-only file
permissions.

The original conversion path reduced messages to `{role, content}` and silently discarded tool-call
structure. In the audited source, 3,020 final assistant targets contain tool calls and 1,481 of those
targets contain only tool calls. The normalizer now preserves both the source's flat
`{name, arguments}` form and OpenAI-style nested function calls.

### Offline tokenizer reproduction

Using the cached Qwen3 tokenizer on a real tool-call-only trajectory:

- old conversion: 1 trainable token, no `<tool_call>` marker, no function payload;
- normalized conversion: 16 trainable tokens, marker and function payload both present.

On 64 real examples, the context limiter rendered every example within Qwen3-8B's 32,768-token
catalog limit. Six examples required structural history pruning; the largest rendered example was
31,927 tokens. All 60 tool-call targets in that sample retained the marker and payload.

### Promotion contract

`eval/wrap_profile_run.py` binds each Promptfoo result to the exact private holdout digest and stable
case IDs. `eval/compare_profiles.py` rejects candidate adoption unless both profiles use the same
held-out cases, run at least three repeats without errors, avoid aggregate and per-case regressions,
reach at least an 85% candidate pass rate, and improve by at least one percentage point. Its
`tinker-yolo/profile-comparison-v1` receipt is the format consumed by the installed `tinker-yolo`
doctor.

### Verification

```text
uv run ruff check --output-format=concise .  -> pass
uv run ruff format --check .                 -> 26 files already formatted
uv run pytest                                -> 80 passed
```

No raw private validation row is logged by the training path. No candidate alias is promoted by
these scripts; promotion still requires real baseline and candidate evaluation receipts. Deployment
also fails unless at least 95% of selected rows render and the Ollama smoke response exactly equals
`TINKER-DEPLOY-OK`; extra text, an empty response, or a non-zero process exit cannot print `DONE`.
