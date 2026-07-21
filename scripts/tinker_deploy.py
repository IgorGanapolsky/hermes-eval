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
import pathlib
import shutil
import subprocess
import sys
import time

from tinker_training_data import (
    DEFAULT_HOLDOUT_RATIO,
    DEFAULT_SPLIT_SEED,
    load_split_conversations,
    render_with_context_limit,
    to_renderer_messages,
)

os.environ.setdefault("HF_HUB_VERBOSITY", "error")
for _n in ("huggingface_hub", "huggingface_hub.utils._auth", "huggingface_hub.file_download"):
    logging.getLogger(_n).setLevel(logging.ERROR)

BASE = "Qwen/Qwen3-8B"
SMOKE_SENTINEL = "TINKER-DEPLOY-OK"
DATA = os.environ.get("TINKER_DEPLOY_DATA", "/tmp/tinker-conversations.jsonl")
N = int(os.environ.get("TINKER_DEPLOY_N", "64"))
STEPS = int(os.environ.get("TINKER_DEPLOY_STEPS", "8"))
OLLAMA_NAME = os.environ.get("TINKER_DEPLOY_MODEL", "qwen3-hermes-tinker")
QUANT = os.environ.get("TINKER_DEPLOY_QUANT", "q4_K_M")
WORK = os.environ.get("TINKER_DEPLOY_WORK", os.path.expanduser("~/models/tinker-deploy"))
OLLAMA_BIN = os.environ.get("TINKER_OLLAMA_BIN", "ollama")
LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp.git"
# Match the pinned Homebrew llama.cpp toolchain used by the Hermes fleet. The
# converter is checked out by immutable revision so a deploy cannot silently run
# arbitrary new conversion code after training has already completed.
LLAMA_CPP_REV = "b15ca938ad00aa6b3ee6c2edda7363fd02826b18"
LLAMA_CPP_TAG = "b10050"
HOLDOUT_RATIO = float(os.environ.get("TINKER_HOLDOUT_RATIO", str(DEFAULT_HOLDOUT_RATIO)))
SPLIT_SEED = os.environ.get("TINKER_SPLIT_SEED", DEFAULT_SPLIT_SEED)
MAX_CONTEXT_TOKENS = int(os.environ.get("TINKER_MAX_CONTEXT_TOKENS", "32768"))
MIN_RENDER_FRACTION = float(os.environ.get("TINKER_MIN_RENDER_FRACTION", "0.95"))


def log(msg):
    print(f"[deploy] {msg}", flush=True)


def command_failure(result):
    """Return a useful bounded error without losing the final panic/root cause."""
    output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    return output[-2000:] or f"exit {result.returncode} with no output"


def run_checked(args, *, env=None, label):
    result = subprocess.run(args, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"{label}: {command_failure(result)}")
    return result


def verify_llama_cpp_checkout(source, converter, git):
    revision = run_checked(
        [git, "-C", str(source), "rev-parse", "HEAD"],
        label="llama.cpp revision check failed",
    ).stdout.strip()
    if revision != LLAMA_CPP_REV:
        raise RuntimeError(
            f"llama.cpp revision mismatch: expected {LLAMA_CPP_REV}, received {revision}"
        )
    if not converter.is_file():
        raise RuntimeError(f"pinned llama.cpp converter missing after revision check: {converter}")
    return converter


def ensure_llama_cpp_converter():
    override = os.environ.get("TINKER_HF_TO_GGUF")
    if override:
        converter = pathlib.Path(override).expanduser()
        if not converter.is_file():
            raise RuntimeError(f"TINKER_HF_TO_GGUF is not a file: {converter}")
        return converter

    source = pathlib.Path(WORK) / f"llama.cpp-{LLAMA_CPP_TAG}"
    converter = source / "convert_hf_to_gguf.py"
    git = shutil.which("git")
    if not git:
        raise RuntimeError("git is required to fetch the pinned llama.cpp converter")
    if converter.is_file():
        return verify_llama_cpp_checkout(source, converter, git)
    if source.exists():
        raise RuntimeError(f"incomplete pinned llama.cpp checkout: {source}")

    log(f"fetching pinned llama.cpp converter {LLAMA_CPP_TAG} ({LLAMA_CPP_REV[:12]})…")
    run_checked(
        [
            git,
            "clone",
            "--depth",
            "1",
            "--branch",
            LLAMA_CPP_TAG,
            LLAMA_CPP_REPO,
            str(source),
        ],
        label="llama.cpp clone failed",
    )
    return verify_llama_cpp_checkout(source, converter, git)


def create_via_gguf(merged_dir):
    """Recover from Ollama's experimental MLX safetensors quantizer via GGUF."""
    converter = ensure_llama_cpp_converter()
    quantize = os.environ.get("TINKER_LLAMA_QUANTIZE") or shutil.which("llama-quantize")
    if not quantize:
        raise RuntimeError("llama-quantize is required for the GGUF fallback")

    gguf_dir = pathlib.Path(WORK) / "gguf-current"
    gguf_dir.mkdir(parents=True, exist_ok=True)
    bf16 = gguf_dir / "qwen3-hermes-tinker-bf16.gguf"
    quantized = gguf_dir / f"qwen3-hermes-tinker-{QUANT.lower()}.gguf"
    gguf_modelfile = gguf_dir / "Modelfile"
    for generated in (bf16, quantized, gguf_modelfile):
        if generated.exists():
            generated.unlink()

    converter_root = converter.parent
    env = os.environ.copy()
    gguf_python = str(converter_root / "gguf-py")
    env["PYTHONPATH"] = (
        f"{gguf_python}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else gguf_python
    )
    log(f"converting merged model to BF16 GGUF -> {bf16} …")
    run_checked(
        [
            sys.executable,
            str(converter),
            str(merged_dir),
            "--outfile",
            str(bf16),
            "--outtype",
            "bf16",
        ],
        env=env,
        label="HF to GGUF conversion failed",
    )
    log(f"quantizing GGUF to {QUANT} -> {quantized} …")
    run_checked(
        [quantize, str(bf16), str(quantized), QUANT],
        label="llama.cpp quantization failed",
    )
    gguf_modelfile.write_text(f"FROM {quantized}\n", encoding="utf-8")
    log(f"ollama create {OLLAMA_NAME} from GGUF…")
    run_checked(
        [OLLAMA_BIN, "create", OLLAMA_NAME, "--file", str(gguf_modelfile)],
        label="Ollama GGUF import failed",
    )


def create_ollama_model(merged_dir, modelfile):
    log(f"ollama create {OLLAMA_NAME} (quantize {QUANT})…")
    result = subprocess.run(
        [OLLAMA_BIN, "create", OLLAMA_NAME, "-q", QUANT, "--experimental", "-f", modelfile],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return
    log(
        "experimental safetensors quantization failed; "
        f"using pinned llama.cpp GGUF fallback: {command_failure(result)}"
    )
    create_via_gguf(merged_dir)


def smoke_and_alias_model():
    log("smoke-testing the new model…")
    smoke_env = os.environ.copy()
    smoke_env["OLLAMA_KEEP_ALIVE"] = "0"
    smoke_timeout = int(os.environ.get("TINKER_DEPLOY_SMOKE_TIMEOUT", "180"))
    result = subprocess.run(
        [OLLAMA_BIN, "run", OLLAMA_NAME, f"Reply with exactly: {SMOKE_SENTINEL}"],
        capture_output=True,
        text=True,
        timeout=smoke_timeout,
        env=smoke_env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Ollama smoke failed: {command_failure(result)}")
    output = require_exact_smoke_output(result.returncode, result.stdout or "")
    log(f"model replied: {output[:80]!r}")
    if ":" not in OLLAMA_NAME:
        alias = f"{OLLAMA_NAME}:q4"
        run_checked(
            [OLLAMA_BIN, "cp", OLLAMA_NAME, alias],
            label=f"Ollama alias creation failed ({alias})",
        )
        log(f"validated Q4 alias updated: {alias}")


def require_exact_smoke_output(returncode: int, stdout: str) -> str:
    """Reject command failures and prompt-echo/extra-text false positives."""

    output = (stdout or "").strip()
    if returncode != 0:
        raise RuntimeError(f"Ollama smoke process exited {returncode}")
    if output != SMOKE_SENTINEL:
        raise RuntimeError("Ollama smoke response did not exactly match the sentinel")
    return output


def require_render_coverage(rendered: int, selected: int) -> None:
    if not 0 < MIN_RENDER_FRACTION <= 1:
        raise RuntimeError("TINKER_MIN_RENDER_FRACTION must be in (0, 1]")
    if selected < 1 or rendered / selected < MIN_RENDER_FRACTION:
        raise RuntimeError(
            f"render coverage {rendered}/{selected} is below {MIN_RENDER_FRACTION:.0%}"
        )


def train_checkpoint():
    import tinker
    from tinker_cookbook import renderers
    from tinker_cookbook.renderers.base import ToolCall

    sc = tinker.ServiceClient()
    caps = sc.get_server_capabilities()
    assert any(BASE in m.model_name for m in caps.supported_models), f"{BASE} unavailable"
    log(f"auth OK; training LoRA on {BASE} (N={N}, steps={STEPS})")

    selection = load_split_conversations(
        DATA,
        split="train",
        limit=N,
        holdout_ratio=HOLDOUT_RATIO,
        seed=SPLIT_SEED,
    )
    rows = selection.conversations
    assert rows, f"no data in {DATA}"
    log(
        f"selected {len(rows)} train conversations "
        f"(reserved holdout={selection.holdout_rows}/{selection.usable_rows}; "
        f"tool targets={selection.selected_tool_targets})"
    )

    tc = sc.create_lora_training_client(base_model=BASE, rank=32)
    tok = tc.get_tokenizer()
    rname = tc.get_renderer_name() if hasattr(tc, "get_renderer_name") else "qwen3"
    try:
        rend = renderers.get_renderer(rname, tok)
    except Exception:
        rend = renderers.get_renderer("qwen3", tok)

    data = []
    context_pruned = 0
    for msgs in rows:
        try:
            conv = to_renderer_messages(msgs, ToolCall.model_validate)
            rendered = render_with_context_limit(
                rend,
                conv,
                train_on_what=renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE,
                max_tokens=MAX_CONTEXT_TOKENS,
            )
            mi, w = rendered.model_input, rendered.weights
            toks = mi.to_ints()
            wl = w.tolist() if hasattr(w, "tolist") else list(w)
            if len(toks) >= 2 and sum(wl[1:]) > 0:
                data.append(
                    tinker.Datum(
                        model_input=tinker.ModelInput.from_ints(toks[:-1]),
                        loss_fn_inputs={"target_tokens": toks[1:], "weights": wl[1:]},
                    )
                )
                context_pruned += int(rendered.dropped_messages > 0)
        except Exception:
            continue
    require_render_coverage(len(data), len(rows))
    log(
        f"rendered {len(data)} examples "
        f"(context-pruned={context_pruned}; max_tokens={MAX_CONTEXT_TOKENS}); training…"
    )
    for s in range(STEPS):
        log(f"step {s + 1}/{STEPS} on Tinker cloud (~20-60s)…")
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
    log(f"merged HF model ready ({time.time() - t0:.0f}s)")

    modelfile = os.path.join(WORK, "Modelfile")
    with open(modelfile, "w") as f:
        f.write(f"FROM {merged_dir}\n")
    # Ollama's experimental MLX safetensors quantizer can panic on Apple Silicon
    # (0.30.10: "There is no Stream(gpu, 1) in current thread"). Never silently
    # fall back to a 16GB BF16 Ollama model on a 24GB host: recover through the
    # documented, pinned llama.cpp GGUF conversion path instead.
    create_ollama_model(merged_dir, modelfile)
    log(f"ollama model created: {OLLAMA_NAME}")
    smoke_and_alias_model()
    print(f"[deploy] DONE — runnable model '{OLLAMA_NAME}' in Ollama.", flush=True)
    print(
        f"[deploy] wire into the gateway: add a model_name block pointing at "
        f"ollama_chat/{OLLAMA_NAME}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[deploy] FAILED: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        sys.exit(1)
