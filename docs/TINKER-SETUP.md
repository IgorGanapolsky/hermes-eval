# Tinker for the Hermes fleet

Verified 2026-07-21: `tinker-yolo`, Tinker SDK 0.23.0, and Tinker Cookbook 0.5.2 are
already installed. Tinker authentication succeeds and Qwen/Qwen3-8B is available.
Installing the Cookbook again is not an improvement; using it to train and evaluate
without leakage is.

## Current role

- Tinker LoRA-trains `Qwen/Qwen3-8B`, the upstream model for the local Hermes candidate.
- Sampler weights are exported, merged into Hugging Face weights, and imported into
  Ollama. Normal inference stays local.
- `tinker-yolo` blocks paid training unless both paid spend and data upload are
  explicitly approved with a bounded cost estimate.
- Inkling remains a metered remote evaluation candidate. It cannot run locally on this
  host and never replaces the local baseline automatically.

The official [Tinker quickstart](https://tinker-docs.thinkingmachines.ai/tinker/quickstart/)
and [Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) describe
the underlying supervised, preference, reinforcement-learning, and evaluation APIs.

## Private dataset and leakage prevention

The current private dataset contains 4,408 rows and 4,239 unique usable conversations;
169 duplicate rows are excluded from both training and evaluation. Of the raw rows,
3,020 final targets contain tool calls and 1,481 contain tool calls without prose. The
full message history contains 237,179 valid function calls.

Training scripts now:

1. Deduplicate identical conversations before selection or split accounting.
2. Assign every conversation to a stable SHA-256 90/10 train/holdout split.
3. Train only on the 90% partition.
4. Normalize both logger tool-call schemas into Cookbook `ToolCall` objects.
5. Preserve tool calls and tool-result correlation fields during rendering.
6. Keep every rendered example at or below Tinker's verified 32,768-token Qwen3-8B
   limit by dropping complete old turns or assistant/tool cycles at safe boundaries.
7. Refuse paid training if fewer than 95% of selected rows render.
8. Export sampler weights rather than an optimizer-state path.
9. Require the Ollama smoke response to equal `TINKER-DEPLOY-OK` exactly before
   reporting deployment success.

Materialize the private holdout and its digest-bound manifest without uploading data:

```sh
uv run python scripts/prepare-tinker-holdout.py \
  --in ~/.hermes/tinker/datasets/conversations.jsonl \
  --out ~/.hermes/tinker/evals/holdout.jsonl \
  --manifest ~/.hermes/tinker/evals/holdout-manifest.json
```

The output and manifest are written atomically with mode `0600`; their directory is
mode `0700`. Stable case IDs let evaluation receipts prove that every scored example
came from the deterministic holdout.

## Evidence-gated candidate promotion

Each baseline and candidate evaluation repeat must be wrapped as
`hermes-eval/profile-run-v1`, bound to the real holdout file and manifest:

```sh
uv run python eval/wrap_profile_run.py \
  --profile hermes-local-baseline \
  --manifest ~/.hermes/tinker/evals/holdout-manifest.json \
  --holdout ~/.hermes/tinker/evals/holdout.jsonl \
  --results /path/to/baseline-results-1.json \
  --out ~/.hermes/tinker/evals/baseline-1.json
```

After at least three repeats per profile, create the receipt consumed by
`tinker-yolo --doctor`:

```sh
uv run python eval/compare_profiles.py \
  --manifest ~/.hermes/tinker/evals/holdout-manifest.json \
  --baseline ~/.hermes/tinker/evals/baseline-1.json \
  --baseline ~/.hermes/tinker/evals/baseline-2.json \
  --baseline ~/.hermes/tinker/evals/baseline-3.json \
  --candidate ~/.hermes/tinker/evals/candidate-1.json \
  --candidate ~/.hermes/tinker/evals/candidate-2.json \
  --candidate ~/.hermes/tinker/evals/candidate-3.json \
  --out ~/.hermes/tinker/evals/inkling-vs-baseline.json
```

Promotion is rejected unless all of these are true:

- at least three baseline and candidate repeats;
- identical case sets from the manifest-bound holdout;
- zero provider/evaluator errors;
- no aggregate holdout regression;
- no per-case regression;
- candidate pass rate at least 85%; and
- candidate improves by at least one percentage point.

The baseline replacement gate remains false. A passing receipt approves only the
isolated remote candidate role.

## Paid execution

Training and deployment remain explicit, receipt-producing operations:

```sh
tinker-yolo proof --approve-paid --approve-data-upload --max-cost-usd N
tinker-yolo train 64 8 --approve-paid --approve-data-upload --max-cost-usd N
tinker-yolo deploy 64 8 --approve-paid --approve-data-upload --max-cost-usd N
```

No paid training or Inkling evaluation is implied by local tests, a saved checkpoint,
or an Ollama smoke response. Candidate quality is established only by the repeated
held-out comparison above.

## Why DPO and RL are not next

The Cookbook supports DPO, RLHF, tool-use RL, and multi-agent training. They are not the
highest-ROI next step until held-out SFT evaluation is trustworthy and ThumbGate feedback
has been curated into unambiguous preference pairs. Vague thumbs or auto-promoted noise
must not become a reward function.
