import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "litellm"))
import hermes_logger


def test_glm_min_max_tokens_raises_small_budget():
    # a tiny budget on a GLM route -> raised to the floor (reasoning would eat it -> empty)
    k = hermes_logger.raise_glm_min_max_tokens({"model": "glm-coding", "max_tokens": 20}, floor=1024)
    assert k["max_tokens"] == 1024
    k = hermes_logger.raise_glm_min_max_tokens(
        {"model": "grp", "litellm_params": {"model": "openai/glm-5.2"}, "max_tokens": 200}, floor=1024
    )
    assert k["max_tokens"] == 1024


def test_glm_min_max_tokens_never_lowers_or_touches_non_glm():
    # already-generous budget is left alone
    k = hermes_logger.raise_glm_min_max_tokens({"model": "glm-coding", "max_tokens": 8000}, floor=1024)
    assert k["max_tokens"] == 8000
    # non-GLM routes untouched
    assert "max_tokens" not in hermes_logger.raise_glm_min_max_tokens(
        {"model": "hermes-local", "max_tokens": 50}, floor=1024
    ) or hermes_logger.raise_glm_min_max_tokens(
        {"model": "hermes-local", "max_tokens": 50}, floor=1024
    )["max_tokens"] == 50
    # the openrouter glm fallback is excluded (clamped down elsewhere, not floored up)
    k = hermes_logger.raise_glm_min_max_tokens(
        {"model": "openrouter/z-ai/glm-5.2", "max_tokens": 200}, floor=1024
    )
    assert k["max_tokens"] == 200
    # no explicit budget -> nothing to raise (leave None; deployment/default applies)
    k = hermes_logger.raise_glm_min_max_tokens({"model": "glm-coding"}, floor=1024)
    assert "max_tokens" not in k


def test_health_check_pings_are_filtered():
    assert hermes_logger.is_health_check([{"role": "user", "content": "Hey, how's it going?"}])
    assert hermes_logger.is_health_check([{"role": "user", "content": ""}])


def test_real_traffic_is_not_filtered():
    assert not hermes_logger.is_health_check([{"role": "user", "content": "How many API keys?"}])
    assert not hermes_logger.is_health_check(
        [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}]
    )
    assert not hermes_logger.is_health_check([])


def test_extract_content_handles_shapes():
    obj = {"choices": [{"message": {"content": "hello"}}]}
    assert hermes_logger.extract_content(obj, {}) == "hello"
    assert hermes_logger.extract_content({}, {"response": "fallback"}) == "fallback"
    # non-string (e.g. a failure returning {}) -> None
    assert hermes_logger.extract_content({}, {"response": {}}) is None


def test_build_record_shape():
    rec = hermes_logger.build_record(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "q"}],
            "standard_logging_object": {"total_tokens": 5},
        },
        {"choices": [{"message": {"content": "a"}}]},
        1.5,
        "success",
    )
    assert rec["model"] == "m" and rec["response"] == "a"
    assert rec["total_tokens"] == 5 and rec["latency_s"] == 1.5 and rec["status"] == "success"
    # normal answer -> classified as non-empty
    assert rec["empty_kind"] is None and rec["has_tool_calls"] is False


def test_empty_content_kind_separates_toolcall_from_truncation():
    # content present -> not empty
    assert hermes_logger.empty_content_kind("hi", "stop", False) is None
    # empty with tool_calls -> legitimate (the qwen3:8b 62% case)
    assert hermes_logger.empty_content_kind("", "tool_calls", True) == "tool_call"
    assert hermes_logger.empty_content_kind("", None, True) == "tool_call"
    # empty with finish_reason=length -> the truncation bug the GLM floor targets
    assert hermes_logger.empty_content_kind("", "length", False) == "truncated"
    # empty, no tool_calls, not length -> unexplained
    assert hermes_logger.empty_content_kind("", "stop", False) == "empty"


def test_finish_reason_and_tool_calls_extraction():
    obj = {"choices": [{"finish_reason": "length",
                        "message": {"content": "", "tool_calls": [{"id": "1"}]}}]}
    assert hermes_logger.extract_finish_reason(obj, {}) == "length"
    assert hermes_logger.has_tool_calls(obj) is True
    # no tool_calls / malformed -> safe defaults
    assert hermes_logger.has_tool_calls({"choices": [{"message": {"content": "x"}}]}) is False
    assert hermes_logger.extract_finish_reason({}, {"finish_reason": "stop"}) == "stop"
    assert hermes_logger.extract_finish_reason({}, {}) is None


# ---- Alerting helpers ------------------------------------------------------------


def test_classify_failure_pages_only_on_quota_exhaustion():
    # code 1310 / quota / exhausted -> the one instant-page failure
    assert "quota" in hermes_logger.classify_failure("Error code: 1310 quota exhausted").lower()
    assert hermes_logger.classify_failure("insufficient balance / quota") is not None


def test_classify_failure_ignores_the_33pct_noise():
    # generic GLM failures, rate-limit bursts, auth, local blips: NOT paged in real time
    # (the 30-min poller's 6h degraded alert owns sustained degradation)
    assert hermes_logger.classify_failure("429 Too Many Requests") is None
    assert hermes_logger.classify_failure("401 Unauthorized") is None
    assert hermes_logger.classify_failure("connection reset by peer") is None
    assert hermes_logger.classify_failure("") is None
    assert hermes_logger.classify_failure(None) is None


def test_update_burn_accumulates_crosses_once_then_resets():
    st, crossed = hermes_logger.update_burn(None, 600, now=1000.0, window_sec=3600, threshold=1000)
    assert not crossed and st["tokens"] == 600
    st, crossed = hermes_logger.update_burn(st, 600, now=1100.0, window_sec=3600, threshold=1000)
    assert crossed and st["tokens"] == 1200 and st["alerted"]
    # same window, already alerted -> does not re-fire
    st, crossed = hermes_logger.update_burn(st, 600, now=1200.0, window_sec=3600, threshold=1000)
    assert not crossed
    # window elapsed -> resets and can fire again
    st, crossed = hermes_logger.update_burn(st, 100, now=9999.0, window_sec=3600, threshold=1000)
    assert not crossed and st["tokens"] == 100 and not st["alerted"]


def test_should_alert_cooldown():
    assert hermes_logger.should_alert(now=1000.0, last_alert_ts=None, cooldown_sec=900)
    assert not hermes_logger.should_alert(now=1000.0, last_alert_ts=500.0, cooldown_sec=900)
    assert hermes_logger.should_alert(now=1500.0, last_alert_ts=500.0, cooldown_sec=900)


def test_extract_error_text_shapes():
    assert hermes_logger.extract_error_text({"exception": ValueError("boom")}, None) == "boom"
    assert "1310" in hermes_logger.extract_error_text({}, "Error 1310")
    assert hermes_logger.extract_error_text(
        {"standard_logging_object": {"error_str": "quota"}}, None
    ) == "quota"


def test_send_ntfy_disabled_is_noop(monkeypatch):
    monkeypatch.setattr(hermes_logger, "ALERTS_ENABLED", False)
    assert hermes_logger.send_ntfy("t", "m") is False
