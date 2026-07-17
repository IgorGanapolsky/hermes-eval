# Fine-tuning the fleet's open-weight models — July 2026 research + decision

Adversarially-verified deep research (112 agents, 3-vote-per-claim). Question: can we
fine-tune our local open-weight models (qwen3:8b, qwen2.5, Llama-3.2-3B, gemma-12B) to be
more business-savvy and more autonomous, on Apple Silicon, inside the Hermes/LiteLLM fleet.

## Verdict

1. **"Business-savvy" is NOT a fine-tuning problem.** For domain/business knowledge, RAG
   beats fine-tuning (Ovadia et al., EMNLP 2024, arXiv 2312.05934: "RAG consistently
   outperforms" unsupervised FT; "LLMs struggle to learn new factual information through
   unsupervised fine-tuning"). Lever = RAG + strong base model + system prompts. `high`

2. **Autonomy/tool-use is the one place tuning pays — via distillation, not RL/DPO.**
   - Trajectory distillation from a strong teacher buys ~one size-tier and transfers
     out-of-domain to 3B-class models (NeurIPS 2025, arXiv 2505.17612). `high`
   - On-policy distillation is 7–10× fewer steps than RL, works with tiny prompt sets
     (Thinking Machines; Qwen3-8B hit 70% AIME'24 in ~150 steps). `high (headline 2-1)`
   - Step-wise on-policy distillation beats SFT/GRPO for tiny agents, but only shown at
     0.6–1.7B in a single preprint (arXiv 2605.07725). `medium`

3. **On the 194 thumbs events:** below threshold for standalone preference tuning. If used,
   **KTO not DPO** — KTO takes unpaired binary labels and handles the 41-pos/153-neg
   imbalance (arXiv 2402.01306). Better: use them as distillation prompts; leverage is data
   *selection* not volume (ICLR 2026, arXiv 2511.10985). `high`

4. **Toolchain works on our Macs:** MLX-LM (LoRA/QLoRA/DoRA for Qwen2/Gemma/Llama, ~250 tok/s
   on M1 Max 32GB) + mlx-tune (KTO/DPO/GRPO). **Serving gotcha:** MLX→GGUF export is
   Llama/Mistral only — Qwen/Gemma adapters need the llama.cpp path or serve via MLX into
   LiteLLM. This is the real integration cost for our Qwen-based fleet. `high`

Thin evidence: strongest small-agent result is unreplicated at 1.7B not 8B; on-policy
headline was 2-1 with vendor numbers; no study tested the *business/sales* axis.

## Decision for a solo operator at $0 revenue

**Do not fine-tune yet.** The business-savvy gain you want isn't a tuning problem, and the
routing is already hardened (GLM primary + free Nemotron fallback). Fine-tuning is a "later,"
when autonomy on a specific local model is worth the Qwen-GGUF serving tax.

**The one no-regret step, done 2026-07-11:** instrument trajectory capture so the dataset
exists if distillation is ever justified.
- Fixed the traffic logger to record the `tool_calls` payload (name+arguments), not just a
  boolean — previously the crux of tool-use distillation was discarded (125/212 tool turns
  had empty response text). Live on both gateways.
- `scripts/build-distill-dataset.py` filters the log to successful teacher (GLM/Nemotron)
  tool-use trajectories → chat-messages JSONL for MLX-LM/mlx-tune. Run periodically to
  accumulate; distillation becomes worthwhile around a few hundred+ clean traces.

Sources: 2312.05934, 2412.14964, 2505.17612, thinkingmachines.ai/blog/on-policy-distillation,
2605.07725, 2402.01306, 2511.10985, 2508.04149, github.com/ml-explore/mlx-lm, github.com/ARahim3/mlx-tune
