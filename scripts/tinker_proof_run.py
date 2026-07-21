#!/usr/bin/env python3
"""Cheap end-to-end PROOF that the Tinker distill loop works for our fleet:
auth -> LoRA client on Qwen3-8B (our local model's upstream) -> a few real
training steps on our own traffic traces -> a downloadable checkpoint path.

Runs a handful of steps on a small slice (cents, not dollars). Full training is
the same code with more data/steps. Reads TINKER_API_KEY from env.
"""

import contextlib
import logging
import os
import sys
import warnings

from tinker_training_data import (
    DEFAULT_HOLDOUT_RATIO,
    DEFAULT_SPLIT_SEED,
    load_split_conversations,
    render_with_context_limit,
    to_renderer_messages,
)

# Silence the cosmetic "Please set a HF_TOKEN …" warning huggingface_hub emits on
# every unauthenticated tokenizer fetch (it looks like an error but is only about
# rate limits). Must run BEFORE importing tinker (which pulls in huggingface_hub).
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
for _n in ("huggingface_hub", "huggingface_hub.utils._auth", "huggingface_hub.file_download"):
    logging.getLogger(_n).setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")

import tinker  # noqa: E402 — deliberately after the HF env/logger setup above

DATA = os.environ.get("TINKER_PROOF_DATA", "/tmp/tinker-conversations.jsonl")
BASE = "Qwen/Qwen3-8B"
N = int(os.environ.get("TINKER_PROOF_N", "16"))
STEPS = int(os.environ.get("TINKER_PROOF_STEPS", "3"))
HOLDOUT_RATIO = float(os.environ.get("TINKER_HOLDOUT_RATIO", str(DEFAULT_HOLDOUT_RATIO)))
SPLIT_SEED = os.environ.get("TINKER_SPLIT_SEED", DEFAULT_SPLIT_SEED)
MAX_CONTEXT_TOKENS = int(os.environ.get("TINKER_MAX_CONTEXT_TOKENS", "32768"))
MIN_RENDER_FRACTION = float(os.environ.get("TINKER_MIN_RENDER_FRACTION", "0.95"))


def main():
    print(f"[proof] tinker {getattr(tinker, '__version__', '?')} | data={DATA}", flush=True)
    sc = tinker.ServiceClient()
    caps = sc.get_server_capabilities()
    models = [m.model_name for m in caps.supported_models]
    assert any(BASE in m for m in models), f"{BASE} not in server catalog"
    print(f"[proof] auth OK — {len(models)} models, {BASE} available", flush=True)

    selection = load_split_conversations(
        DATA,
        split="train",
        limit=N,
        holdout_ratio=HOLDOUT_RATIO,
        seed=SPLIT_SEED,
    )
    exs = selection.conversations
    assert exs, f"no usable conversations in {DATA}"
    print(
        f"[proof] selected {len(exs)} train conversations "
        f"(reserved holdout={selection.holdout_rows}/{selection.usable_rows}; "
        f"tool targets={selection.selected_tool_targets})",
        flush=True,
    )

    tc = sc.create_lora_training_client(base_model=BASE, rank=16)
    print(f"[proof] LoRA training client created (rank=16) on {BASE}", flush=True)

    tokenizer = tc.get_tokenizer()
    from tinker_cookbook import renderers
    from tinker_cookbook.renderers.base import ToolCall

    renderer_name = tc.get_renderer_name() if hasattr(tc, "get_renderer_name") else "qwen3"
    try:
        renderer = renderers.get_renderer(renderer_name, tokenizer)
    except Exception:
        renderer = renderers.get_renderer("qwen3", tokenizer)

    def to_datum(messages):
        # Preserve structured tool calls and tool-result correlation fields. The logger's
        # flat final-target calls are normalized into Cookbook ToolCall objects here.
        conv = to_renderer_messages(messages, ToolCall.model_validate)
        # LAST_ASSISTANT_MESSAGE satisfies the renderer extension property (warning-free)
        # and limits loss to the teacher target. Shift for next-token prediction.
        rendered = render_with_context_limit(
            renderer,
            conv,
            train_on_what=renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE,
            max_tokens=MAX_CONTEXT_TOKENS,
        )
        model_input, weights = rendered.model_input, rendered.weights
        tokens = model_input.to_ints()
        w = weights.tolist() if hasattr(weights, "tolist") else list(weights)
        if len(tokens) < 2 or sum(w[1:]) == 0:
            raise ValueError("no trainable assistant tokens")
        return (
            tinker.Datum(
                model_input=tinker.ModelInput.from_ints(tokens[:-1]),
                loss_fn_inputs={"target_tokens": tokens[1:], "weights": w[1:]},
            ),
            rendered.dropped_messages,
        )

    data = []
    context_pruned = 0
    for m in exs:
        try:
            datum, dropped_messages = to_datum(m)
            data.append(datum)
            context_pruned += int(dropped_messages > 0)
        except Exception:
            # Do not print validation payloads: conversations can contain private traffic.
            print("[proof]   skip one (render validation)")
    if not 0 < MIN_RENDER_FRACTION <= 1:
        raise ValueError("TINKER_MIN_RENDER_FRACTION must be in (0, 1]")
    if len(data) / len(exs) < MIN_RENDER_FRACTION:
        raise RuntimeError(
            f"render coverage {len(data)}/{len(exs)} is below {MIN_RENDER_FRACTION:.0%}"
        )
    print(
        f"[proof] rendered {len(data)} training examples "
        f"(context-pruned={context_pruned}; max_tokens={MAX_CONTEXT_TOKENS})",
        flush=True,
    )

    for step in range(STEPS):
        # The forward_backward + optim_step run on Tinker's cloud GPUs and take
        # ~20-60s each — print BEFORE so a slow step never looks like a hang.
        print(f"[proof] step {step + 1}/{STEPS} training on Tinker cloud (~20-60s)…", flush=True)
        fut = tc.forward_backward(data, loss_fn="cross_entropy")
        opt = tc.optim_step(tinker.AdamParams(learning_rate=1e-4))
        fb = fut.result()
        opt.result()
        loss = None
        with contextlib.suppress(Exception):
            loss = float(fb.metrics.get("loss:sum", 0)) / max(
                1, float(fb.metrics.get("loss:count", 1))
            )
        print(f"[proof] step {step + 1}/{STEPS} done | loss~{loss}", flush=True)

    weights = tc.save_weights_for_sampler(name="hermes-distill-proof")
    path = weights.result().path
    print(f"[proof] SAMPLER WEIGHTS SAVED: {path}")
    print(
        f"[proof] weights export cmd: tinker checkpoint download '{path}' -o ~/models/qwen3-hermes"
    )
    print("[proof] PIPELINE OK — auth+train+downloadable sampler weights verified end to end")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[proof] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
