"""TDD: GPT-6 Astra is opt-in and hard-capped at $10/month."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "litellm"))
import yaml

import astra_budget
from merge_astra_route import merge


def test_estimate_uses_standard_not_fast_rates() -> None:
    # $10 / 1M in + $50 / 1M out. Fast (2x) must never be the estimator.
    usd = astra_budget.estimate_cost(1_000_000, 1_000_000)
    assert abs(usd - 60.0) < 1e-9


def test_over_cap_rewrites_to_glm_coding(tmp_path: Path) -> None:
    ledger = tmp_path / "astra-spend.json"
    ledger.write_text(
        json.dumps(
            {
                "month": astra_budget.month_key(),
                "usd": 9.60,
                "calls": 12,
            }
        )
        + "\n"
    )
    data = {"model": "gpt-6-astra", "max_tokens": 4000}
    out = astra_budget.guard_request(data, ledger_path=ledger)
    assert out["model"] == "glm-coding"
    assert out["metadata"]["astra_budget_rewrite"]["reason"] == "monthly_cap"


def test_under_cap_clamps_tokens_and_effort(tmp_path: Path) -> None:
    ledger = tmp_path / "astra-spend.json"
    data = {
        "model": "astra",
        "max_tokens": 20000,
        "reasoning_effort": "max",
    }
    out = astra_budget.guard_request(data, ledger_path=ledger)
    assert out["model"] == "astra"
    assert out["max_completion_tokens"] == astra_budget.MAX_OUT_TOKENS
    assert "max_tokens" not in out
    assert out["reasoning_effort"] == "low"
    assert out["disable_fallbacks"] is True


def test_non_astra_untouched(tmp_path: Path) -> None:
    data = {"model": "hermes-local", "max_tokens": 50}
    out = astra_budget.guard_request(data, ledger_path=tmp_path / "x.json")
    assert out == {"model": "hermes-local", "max_tokens": 50}


def test_record_usage_accumulates(tmp_path: Path) -> None:
    ledger = tmp_path / "astra-spend.json"
    astra_budget.record_usage(
        "gpt-6-astra",
        prompt_tokens=1000,
        completion_tokens=200,
        ledger_path=ledger,
    )
    body = json.loads(ledger.read_text())
    assert body["calls"] == 1
    assert body["usd"] > 0
    assert body["month"] == astra_budget.month_key()


def test_fast_or_pro_alias_is_rejected_to_standard(tmp_path: Path) -> None:
    data = {"model": "gpt-6-astra-fast"}
    out = astra_budget.guard_request(data, ledger_path=tmp_path / "x.json")
    assert out["model"] == "gpt-6-astra"
    assert out["metadata"]["astra_alias"] == "standard_only"


def test_merge_inserts_fragment_once(tmp_path: Path) -> None:
    src = tmp_path / "config.yaml"
    src.write_text("model_list:\n- model_name: hermes-local\n\nrouter_settings:\n  timeout: 1\n")
    dest = tmp_path / "runtime.yaml"
    frag = Path(__file__).resolve().parents[1] / "litellm" / "astra_models.yaml"
    # merge() reads fragment from sibling; copy into tmp layout
    (tmp_path / "astra_models.yaml").write_text(frag.read_text())
    # Use the module helper against explicit paths
    assert merge(src, tmp_path / "astra_models.yaml", dest) == "merged"
    text = dest.read_text()
    assert text.count("model_name: gpt-6-astra") == 1
    assert "router_settings:" in text
    assert merge(dest, tmp_path / "astra_models.yaml", dest) == "already"


def test_merge_aligns_to_column0_list(tmp_path: Path) -> None:
    src = tmp_path / "config.yaml"
    src.write_text("model_list:\n- model_name: nim-nv-embed\n\nrouter_settings:\n  timeout: 1\n")
    dest = tmp_path / "runtime.yaml"
    frag = Path(__file__).resolve().parents[1] / "litellm" / "astra_models.yaml"
    assert merge(src, frag, dest) == "merged"
    text = dest.read_text()
    assert "\n- model_name: gpt-6-astra\n" in text
    parsed = yaml.safe_load(text)
    names = [row["model_name"] for row in parsed["model_list"]]
    assert "gpt-6-astra" in names
    assert "astra" in names


def test_fallback_model_honors_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HERMES_ASTRA_FALLBACK", "hermes-main")
    ledger = tmp_path / "astra-spend.json"
    ledger.write_text(
        json.dumps({"month": astra_budget.month_key(), "usd": 9.60, "calls": 1}) + "\n"
    )
    out = astra_budget.guard_request({"model": "gpt-6-astra"}, ledger_path=ledger)
    assert out["model"] == "hermes-main"


def test_merge_injects_callback_when_logger_absent(tmp_path: Path) -> None:
    src = tmp_path / "config.yaml"
    src.write_text(
        "model_list:\n  - model_name: hermes-main\n\n"
        "router_settings:\n  timeout: 1\n"
        "litellm_settings:\n  drop_params: true\n"
    )
    dest = tmp_path / "runtime.yaml"
    frag = Path(__file__).resolve().parents[1] / "litellm" / "astra_models.yaml"
    assert merge(src, frag, dest) == "merged"
    text = dest.read_text()
    assert "callbacks: astra_budget.proxy_handler_instance" in text
    assert text.count("astra_budget.proxy_handler_instance") == 1


def test_merge_skips_callback_when_hermes_logger_present(tmp_path: Path) -> None:
    src = tmp_path / "config.yaml"
    src.write_text(
        "model_list:\n  - model_name: hermes-local\n\n"
        "router_settings:\n  timeout: 1\n"
        "litellm_settings:\n  callbacks: hermes_logger.proxy_handler_instance\n"
    )
    dest = tmp_path / "runtime.yaml"
    frag = Path(__file__).resolve().parents[1] / "litellm" / "astra_models.yaml"
    assert merge(src, frag, dest) == "merged"
    assert "astra_budget.proxy_handler_instance" not in dest.read_text()


def test_merge_aligns_to_two_space_list(tmp_path: Path) -> None:
    src = tmp_path / "config.yaml"
    src.write_text("model_list:\n  - model_name: hermes-local\n\nrouter_settings:\n  timeout: 1\n")
    dest = tmp_path / "runtime.yaml"
    frag = Path(__file__).resolve().parents[1] / "litellm" / "astra_models.yaml"
    assert merge(src, frag, dest) == "merged"
    text = dest.read_text()
    assert "\n  - model_name: gpt-6-astra\n" in text
    parsed = yaml.safe_load(text)
    names = [row["model_name"] for row in parsed["model_list"]]
    assert "gpt-6-astra" in names
    assert "astra" in names
