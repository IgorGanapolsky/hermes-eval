# Tinker (Thinking Machines) — fine-tune the fleet's local model on our own traffic

Status 2026-07-17: everything below is prepared and verified locally except the two
Igor-only steps (account + API key). Tinker went GA Dec 2025 (no waitlist); pricing
rose ~10% on train rates 2026-07-17 — still single-digit dollars at our scale.

## Why

- Tinker LoRA-tunes **`Qwen/Qwen3-8B` — the exact upstream of our local `qwen3:8b`**
  (also Qwen3.5/3.6, Kimi K2.x, gpt-oss, Nemotron-3, DeepSeek — see
  https://tinker-docs.thinkingmachines.ai/tinker/models/).
- Training runs on their GPUs (nothing local — our 24GB no-GPU Macs can't tune 8B).
- **Weights come back out** (`tinker checkpoint download`, `build_hf_model` → merged
  safetensors) → llama.cpp `convert_hf_to_gguf.py` + `llama-quantize` q4_K_M →
  `ollama create qwen3-hermes` → new `hermes-local` deployment in the LiteLLM gateway.
  Inference does NOT stay on their platform.
- Fit per docs/FINE-TUNING-RESEARCH-2026-07.md: distillation of strong-teacher
  trajectories is the one tuning path that pays; RAG covers knowledge.

## Dataset (READY — verified 2026-07-17)

```sh
python3 scripts/build-distill-dataset.py --accumulate --out /tmp/tinker-dataset.jsonl
# read 6,260 trajectories from live+archives → 1,628 usable teacher tool-use traces
jq -c 'del(.meta)' /tmp/tinker-dataset.jsonl > /tmp/tinker-conversations.jsonl  # Tinker wants {"messages":[...]} only
```

1,628 traces ≳ the few-hundred threshold where distillation becomes worthwhile.
The traffic log keeps growing this dataset automatically (gateway logger).

## Igor-only steps (5 minutes)

1. Sign up: https://auth.thinkingmachines.ai/sign-up (open signup; $150 intro credits
   were reported — unverified whether still live).
2. Create key: https://tinker-console.thinkingmachines.ai → `TINKER_API_KEY=...` into
   `~/.hermes/.env` (never paste into chat).

## Train + export (agent-runnable once the key exists)

```sh
python3 -m venv ~/.hermes/tinker-venv && ~/.hermes/tinker-venv/bin/pip install tinker tinker-cookbook
export TINKER_API_KEY="$(grep '^TINKER_API_KEY=' ~/.hermes/.env | cut -d= -f2-)"
# sl_basic-style supervised run: base_model="Qwen/Qwen3-8B", rank 32, lr 2e-4,
# FromConversationFileBuilder(file_path="/tmp/tinker-conversations.jsonl")
# Cost estimate: ~1.6k traces × ~2k tok ≈ 3M train tokens ≈ $1.30 @ $0.44/M.
tinker checkpoint download 'tinker://<run>/sampler_weights/final' -o ~/models/qwen3-hermes
# merge → GGUF → ollama create qwen3-hermes-64k → add gateway deployment; A/B with
# litellm/competence_probe.py before ever making it a default (never silently degrade).
```

Notes: Inkling (their 975B open-weights MoE, July 15 2026) is fine-tunable too — 50%
launch discount — but it's cloud-inference-class, not a local candidate here.
