#!/usr/bin/env python3
"""tinker-yolo deploy — the REAL end-to-end: fine-tune the fleet's local model and
produce a RUNNABLE Ollama model (not just a proof checkpoint).

  1. train a LoRA on Qwen/Qwen3-8B from our traffic  (or reuse TINKER_CHECKPOINT)
  2. download the checkpoint                          (tinker weights.download)
  3. merge LoRA into the base -> HF safetensors        (weights.build_hf_model)
  4. ollama create <name> --experimental [--quantize]  (imports safetensors, no llama.cpp)
  5. smoke-test the new model answers

Every stage streams progress (no silent hangs). Heavy: step 3 downloads the ~16GB
base model from HF and merges (needs ~16GB RAM) — free memory first on a 24GB box.

Env: TINKER_API_KEY (req), TINKER_DEPLOY_DATA, TINKER_DEPLOY_N/STEPS,
TINKER_DEPLOY_MODEL (ollama name), TINKER_DEPLOY_QUANT (e.g. q4_K_M), TINKER_CHECKPOINT
(reuse instead of training), TINKER_DEPLOY_WORK (scratch dir).
"""
import logging
import os
import subprocess
import sys
import time

os.environ.setdefault("HF_HUB_VERBOSITY", "error")
for _n in ("huggingface_hub", "huggingface_hub.utils._auth", "huggingface_hub.file_download"):
    logging.getLogger(_n).setLevel(logging.ERROR)

import tinker  # noqa: E402 — after HF env setup

BASE = "Qwen/Qwen3-8B"
DATA = os.environ.get("TINKER_DEPLOY_DATA", "/tmp/tinker-conversations.jsonl")
N = int(os.environ.get("TINKER_DEPLOY_N", "64"))
STEPS = int(os.environ.get("TINKER_DEPLOY_STEPS", "8"))
OLLAMA_NAME = os.environ.get("TINKER_DEPLOY_MODEL", "qwen3-hermes-tinker")
QUANT = os.environ.get("TINKER_DEPLOY_QUANT", "q4_K_M")
WORK = os.environ.get("TINKER_DEPLOY_WORK", os.path.expanduser("~/models/tinker-deploy"))


def log(msg):
    print(f"[deploy] {msg}", flush=True)


def train_checkpoint():
    from tinker_cookbook import renderers

    sc = tinker.ServiceClient()
    caps = sc.get_server_capabilities()
    assert any(BASE in m.model_name for m in caps.supported_models), f"{BASE} unavailable"
    log(f"auth OK; training LoRA on {BASE} (N={N}, steps={STEPS})")

    rows = []
    with open(DATA) as f:
        import json

        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("messages") and len(obj["messages"]) >= 2:
                rows.append(obj["messages"])
            if len(rows) >= N:
                break
    assert rows, f"no data in {DATA}"

    tc = sc.create_lora_training_client(base_model=BASE, rank=32)
    tok = tc.get_tokenizer()
    rname = tc.get_renderer_name() if hasattr(tc, "get_renderer_name") else "qwen3"
    try:
        rend = renderers.get_renderer(rname, tok)
    except Exception:
        rend = renderers.get_renderer("qwen3", tok)

    data = []
    for msgs in rows:
        try:
            conv = [{"role": m["role"], "content": m.get("content") or ""} for m in msgs]
            mi, w = rend.build_supervised_example(
                conv, train_on_what=renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE
            )
            toks = mi.to_ints()
            wl = w.tolist() if hasattr(w, "tolist") else list(w)
            if len(toks) >= 2 and sum(wl[1:]) > 0:
                data.append(
                    tinker.Datum(
                        model_input=tinker.ModelInput.from_ints(toks[:-1]),
                        loss_fn_inputs={"target_tokens": toks[1:], "weights": wl[1:]},
                    )
                )
        except Exception:
            continue
    assert data, "no rendered data"
    log(f"rendered {len(data)} examples; training…")
    for s in range(STEPS):
        log(f"step {s+1}/{STEPS} on Tinker cloud (~20-60s)…")
        fut = tc.forward_backward(data, loss_fn="cross_entropy")
        tc.optim_step(tinker.AdamParams(learning_rate=1e-4)).result()
        fut.result()
    # save_weights_for_sampler → downloadable INFERENCE weights (save_state is the
    # optimizer/training state and its path is NOT accepted by weights.download).
    path = tc.save_weights_for_sampler(name="hermes-distill-deploy").result().path
    log(f"sampler-weights checkpoint: {path}")
    return path


def main():
    os.makedirs(WORK, exist_ok=True)
    ckpt = os.environ.get("TINKER_CHECKPOINT") or train_checkpoint()

    from tinker_cookbook import weights

    adapter_dir = os.path.join(WORK, "adapter")
    merged_dir = os.path.join(WORK, "merged")
    log(f"downloading checkpoint -> {adapter_dir} …")
    weights.download(tinker_path=ckpt, output_dir=adapter_dir)
    log(f"merging LoRA into {BASE} -> {merged_dir} (downloads ~16GB base on first run)…")
    t0 = time.time()
    weights.build_hf_model(base_model=BASE, adapter_path=adapter_dir, output_path=merged_dir)
    log(f"merged HF model ready ({time.time()-t0:.0f}s)")

    modelfile = os.path.join(WORK, "Modelfile")
    with open(modelfile, "w") as f:
        f.write(f"FROM {merged_dir}\n")
    cmd = ["ollama", "create", OLLAMA_NAME, "--experimental"]
    # ollama create supports -q/--quantize on import; try quantized, fall back to f16.
    log(f"ollama create {OLLAMA_NAME} (quantize {QUANT})…")
    r = subprocess.run([*cmd, "-q", QUANT, "-f", modelfile], capture_output=True, text=True)
    if r.returncode != 0:
        log(f"quantized create failed ({r.stderr.strip()[:120]}); retrying f16…")
        r = subprocess.run([*cmd, "-f", modelfile], capture_output=True, text=True)
        if r.returncode != 0:
            log(f"FAILED: ollama create: {r.stderr.strip()[:200]}")
            sys.exit(1)
    log(f"ollama model created: {OLLAMA_NAME}")

    log("smoke-testing the new model…")
    sm = subprocess.run(
        ["ollama", "run", OLLAMA_NAME, "Reply with exactly: TINKER-DEPLOY-OK"],
        capture_output=True, text=True, timeout=180,
    )
    out = (sm.stdout or "").strip()
    log(f"model replied: {out[:80]!r}")
    print(f"[deploy] DONE — runnable model '{OLLAMA_NAME}' in Ollama.", flush=True)
    print(f"[deploy] wire into the gateway: add a model_name block pointing at "
          f"ollama_chat/{OLLAMA_NAME}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[deploy] FAILED: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        sys.exit(1)
