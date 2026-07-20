import importlib.util
import pathlib
import subprocess
import sys
import types

SCRIPT = pathlib.Path(__file__).parents[1] / "scripts" / "tinker_deploy.py"


def load_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "tinker", types.SimpleNamespace())
    spec = importlib.util.spec_from_file_location("tinker_deploy_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def completed(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


def test_safetensors_quantizer_success_skips_gguf(monkeypatch):
    module = load_module(monkeypatch)
    calls = []
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: completed())
    monkeypatch.setattr(module, "create_via_gguf", lambda merged: calls.append(merged))

    module.create_ollama_model("/merged", "/merged/Modelfile")

    assert calls == []


def test_mlx_quantizer_panic_routes_to_pinned_gguf_fallback(monkeypatch):
    module = load_module(monkeypatch)
    calls = []
    panic = "panic: mlx: There is no Stream(gpu, 1) in current thread"
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: completed(returncode=2, stderr=panic),
    )
    monkeypatch.setattr(module, "create_via_gguf", lambda merged: calls.append(merged))

    module.create_ollama_model("/merged", "/merged/Modelfile")

    assert calls == ["/merged"]


def test_error_keeps_tail_where_ollama_prints_the_panic(monkeypatch):
    module = load_module(monkeypatch)
    result = completed(returncode=2, stderr="prefix" * 600 + "ROOT_CAUSE_AT_END")

    message = module.command_failure(result)

    assert len(message) <= 2000
    assert message.endswith("ROOT_CAUSE_AT_END")


def test_q4_alias_moves_only_after_successful_smoke(monkeypatch):
    module = load_module(monkeypatch)
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        return completed(stdout="TINKER-DEPLOY-OK\n")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module, "run_checked", lambda args, **kwargs: calls.append(args))

    module.smoke_and_alias_model()

    assert calls[0][:3] == [module.OLLAMA_BIN, "run", module.OLLAMA_NAME]
    assert calls[1] == [module.OLLAMA_BIN, "cp", module.OLLAMA_NAME, f"{module.OLLAMA_NAME}:q4"]


def test_failed_smoke_preserves_existing_q4_alias(monkeypatch):
    module = load_module(monkeypatch)
    aliases = []
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: completed(returncode=1, stderr="load failed"),
    )
    monkeypatch.setattr(module, "run_checked", lambda args, **kwargs: aliases.append(args))

    try:
        module.smoke_and_alias_model()
    except RuntimeError as exc:
        assert "load failed" in str(exc)
    else:
        raise AssertionError("smoke failure must raise")

    assert aliases == []
