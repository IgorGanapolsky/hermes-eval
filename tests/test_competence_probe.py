"""Tests for the gateway competence probe.

The bug these guard against: a degraded model returns HTTP 200, so liveness checks pass while
agents loop. Every assertion below encodes a real observation from the 2026-07-09 incident.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "litellm"))

from competence_probe import (
    DEFAULT_PROMPT_TOKENS,
    ProbeError,
    build_prompt,
    classify,
    evaluate,
    evaluate_balance,
)


def _response(model: str, *, tool_calls: bool = True, prompt_tokens: int = 36140) -> dict:
    message: dict = {"role": "assistant", "content": ""}
    if tool_calls:
        message["tool_calls"] = [{"id": "c1", "type": "function", "function": {"name": "ping"}}]
    return {
        "model": model,
        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": message}],
        "usage": {"prompt_tokens": prompt_tokens},
    }


# --- classify -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "served",
    [
        "glm-coding",
        "z-ai/glm-5.2-20260616",  # what OpenRouter returned on a healthy failover
        "glm-5-turbo",
    ],
)
def test_competent_models(served):
    assert classify(served) == "competent"


@pytest.mark.parametrize(
    "served",
    [
        "ollama_chat/qwen3:8b-64k",  # the model that looped for hours
        "qwen2.5:3b-64k",
        "openai/Llama-3.2-3B-Instruct-4bit",
    ],
)
def test_degraded_models(served):
    assert classify(served) == "degraded"


def test_unknown_model_is_not_assumed_competent():
    # An unrecognised model must NOT pass. Assuming competence is what let the incident hide.
    assert classify("some-new-model-v9") == "unknown"


def test_empty_served_model_is_unknown():
    assert classify("") == "unknown"


# --- evaluate -------------------------------------------------------------------------------


def test_healthy_primary_is_competent():
    v = evaluate(_response("glm-coding"))
    assert v["competent"] is True
    assert v["tier"] == "competent"
    assert v["tool_calls"] == ["ping"]


def test_cloud_failover_is_competent():
    v = evaluate(_response("z-ai/glm-5.2-20260616"))
    assert v["competent"] is True


def test_silent_demotion_to_local_8b_is_flagged():
    # The exact shape of the incident: HTTP 200, a valid tool call, wrong model.
    v = evaluate(_response("ollama_chat/qwen3:8b-64k"))
    assert v["competent"] is False
    assert v["tier"] == "degraded"


def test_competent_model_without_tool_call_is_flagged():
    # Prose where a tool call was demanded stalls an agent just as surely as a loop.
    v = evaluate(_response("glm-coding", tool_calls=False))
    assert v["competent"] is False
    assert v["tool_calls"] == []


def test_prompt_tokens_are_surfaced():
    v = evaluate(_response("glm-coding", prompt_tokens=36140))
    assert v["prompt_tokens"] == 36140


def test_malformed_response_raises():
    with pytest.raises(ProbeError):
        evaluate({"model": "glm-coding"})  # no choices


# --- build_prompt ---------------------------------------------------------------------------


def test_default_prompt_is_cheap():
    # A 36k-token probe on a schedule would burn ~3M tokens/day of the z.ai rolling quota it
    # exists to protect. The default must stay small; balance is asserted for free instead.
    assert DEFAULT_PROMPT_TOKENS <= 4_000


def test_deep_probe_can_exceed_the_degraded_cap():
    # The opt-in deep probe must be able to reproduce the OpenRouter 402 (smallest observed
    # degraded cap was 5567 tokens).
    assert len(build_prompt(36_000)) > 4 * 5567


# --- evaluate_balance -----------------------------------------------------------------------


def test_healthy_balance_is_fallback_ready():
    v = evaluate_balance({"data": {"total_credits": 175, "total_usage": 165.116}})
    assert v["fallback_ready"] is True
    assert v["remaining_usd"] == pytest.approx(9.884, abs=1e-3)


def test_overdrawn_balance_is_not_fallback_ready():
    # The exact 2026-07-09 state: negative balance, so cloud-fallback 402s on real prompts.
    v = evaluate_balance({"data": {"total_credits": 165, "total_usage": 165.116486344}})
    assert v["fallback_ready"] is False
    assert v["remaining_usd"] < 0


def test_thin_balance_is_flagged_before_it_bites():
    # Alert while there's still credit, not after the fallback has already started 402ing.
    v = evaluate_balance({"data": {"total_credits": 100, "total_usage": 99.0}})
    assert v["fallback_ready"] is False


def test_malformed_credits_response_raises():
    with pytest.raises(ProbeError):
        evaluate_balance({"data": {"total_credits": 10}})


def test_prompt_demands_a_tool_call():
    assert "ping" in build_prompt(1000)


def test_tiny_prompt_request_still_builds():
    assert build_prompt(0)
