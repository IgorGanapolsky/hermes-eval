#!/usr/bin/env python3
"""Cheap end-to-end PROOF that the Tinker distill loop works for our fleet:
auth -> LoRA client on Qwen3-8B (our local model's upstream) -> a few real
training steps on our own traffic traces -> a downloadable checkpoint path.

Runs a handful of steps on a small slice (cents, not dollars). Full training is
the same code with more data/steps. Reads TINKER_API_KEY from env.
"""
import json
import os
import sys

import tinker

DATA = os.environ.get("TINKER_PROOF_DATA", "/tmp/tinker-conversations.jsonl")
BASE = "Qwen/Qwen3-8B"
N = int(os.environ.get("TINKER_PROOF_N", "16"))
STEPS = int(os.environ.get("TINKER_PROOF_STEPS", "3"))


def load_examples(path, n):
    rows = []
    with open(path) as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            msgs = obj.get("messages")
            if msgs and len(msgs) >= 2:
                rows.append(msgs)
            if len(rows) >= n:
                break
    return rows


def main():
    print(f"[proof] tinker {getattr(tinker, '__version__', '?')} | data={DATA}")
    sc = tinker.ServiceClient()
    caps = sc.get_server_capabilities()
    models = [m.model_name for m in caps.supported_models]
    assert any(BASE in m for m in models), f"{BASE} not in server catalog"
    print(f"[proof] auth OK — {len(models)} models, {BASE} available")

    exs = load_examples(DATA, N)
    assert exs, f"no usable conversations in {DATA}"
    print(f"[proof] loaded {len(exs)} real traffic conversations")

    tc = sc.create_lora_training_client(base_model=BASE, rank=16)
    print(f"[proof] LoRA training client created (rank=16) on {BASE}")

    tokenizer = tc.get_tokenizer()
    from tinker_cookbook import renderers
    renderer_name = tc.get_renderer_name() if hasattr(tc, "get_renderer_name") else "qwen3"
    try:
        renderer = renderers.get_renderer(renderer_name, tokenizer)
    except Exception:
        renderer = renderers.get_renderer("qwen3", tokenizer)

    def to_datum(messages):
        conv = [{"role": m["role"], "content": m.get("content") or ""} for m in messages]
        tokens, weights = renderer.build_supervised_example(conv)
        return tinker.Datum(
            model_input=tinker.ModelInput.from_ints(tokens[:-1]),
            loss_fn_inputs={"target_tokens": tokens[1:], "weights": weights[1:]},
        )

    data = []
    for m in exs:
        try:
            data.append(to_datum(m))
        except Exception as e:
            print(f"[proof]   skip one (render): {e}")
    assert data, "no rendered data"
    print(f"[proof] rendered {len(data)} training examples")

    for step in range(STEPS):
        fut = tc.forward_backward(data, loss_fn="cross_entropy")
        opt = tc.optim_step(tinker.AdamParams(learning_rate=1e-4))
        fb = fut.result()
        opt.result()
        loss = None
        try:
            loss = float(fb.metrics.get("loss:sum", 0)) / max(1, float(fb.metrics.get("loss:count", 1)))
        except Exception:
            pass
        print(f"[proof] step {step+1}/{STEPS} done | loss~{loss}")

    state = tc.save_state(name="hermes-distill-proof")
    path = state.result().path
    print(f"[proof] CHECKPOINT SAVED: {path}")
    print(f"[proof] weights export cmd: tinker checkpoint download '{path}' -o ~/models/qwen3-hermes")
    print("[proof] PIPELINE OK — auth+train+checkpoint verified end to end")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[proof] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
