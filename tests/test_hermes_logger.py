import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "litellm"))
import hermes_logger


def test_glm_min_max_tokens_raises_small_budget():
    # a tiny budget on a GLM route -> raised to the floor (reasoning would eat it -> empty)
    k = hermes_logger.raise_glm_min_max_tokens(
        {"model": "glm-coding", "max_tokens": 20}, floor=1024
    )
    assert k["max_tokens"] == 1024
    k = hermes_logger.raise_glm_min_max_tokens(
        {"model": "grp", "litellm_params": {"model": "openai/glm-5.2"}, "max_tokens": 200},
        floor=1024,
    )
    assert k["max_tokens"] == 1024


def test_glm_min_max_tokens_never_lowers_or_touches_non_glm():
    # already-generous budget is left alone
    k = hermes_logger.raise_glm_min_max_tokens(
        {"model": "glm-coding", "max_tokens": 8000}, floor=1024
    )
    assert k["max_tokens"] == 8000
    # non-GLM routes untouched
    assert (
        "max_tokens"
        not in hermes_logger.raise_glm_min_max_tokens(
            {"model": "hermes-local", "max_tokens": 50}, floor=1024
        )
        or hermes_logger.raise_glm_min_max_tokens(
            {"model": "hermes-local", "max_tokens": 50}, floor=1024
        )["max_tokens"]
        == 50
    )
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
    obj = {
        "choices": [
            {"finish_reason": "length", "message": {"content": "", "tool_calls": [{"id": "1"}]}}
        ]
    }
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
    assert (
        hermes_logger.extract_error_text({"standard_logging_object": {"error_str": "quota"}}, None)
        == "quota"
    )


def test_send_ntfy_disabled_is_noop(monkeypatch):
    monkeypatch.setattr(hermes_logger, "ALERTS_ENABLED", False)
    assert hermes_logger.send_ntfy("t", "m") is False


def test_module_imports_cleanly_under_gateway_semantics():
    # Regression: a module-level re.compile() was added while 're' was unimported.
    # py_compile PASSED (syntax was fine) and only a real import surfaced the NameError.
    # This logger runs inside the proxy's logging path, so an import-time failure is a
    # gateway outage. Import the module fresh rather than trusting the cached one.
    import importlib

    importlib.reload(hermes_logger)
    assert hermes_logger.redact_secrets("plain text") == "plain text"


def test_redact_secrets_scrubs_credential_shapes():
    fake_openai = "sk-" + "A" * 24
    fake_gh = "ghp_" + "B" * 20
    fake_bearer = "Bearer " + "C" * 24
    for secret, blob in (
        (fake_openai, f"401 from provider: key {fake_openai} rejected"),
        (fake_gh, f"header token {fake_gh} denied"),
        (fake_bearer, f"Authorization: {fake_bearer}"),
    ):
        out = hermes_logger.redact_secrets(blob)
        assert secret.split()[-1] not in out, f"leaked: {out}"
        assert "[REDACTED]" in out


def test_redact_secrets_never_raises_on_odd_input():
    assert hermes_logger.redact_secrets(None) == ""
    assert hermes_logger.redact_secrets("") == ""
    assert hermes_logger.redact_secrets(12345) == "12345"


def test_failure_error_fields_empty_on_success():
    # extract_error_text() falls through to response_obj, so without the status gate a
    # SUCCESSFUL call would serialize its whole response into 'error'.
    big = {"choices": [{"message": {"content": "x" * 5000}}]}
    got = hermes_logger.failure_error_fields({}, big, "success")
    assert got == {"error": None, "error_class": None}


def test_failure_error_fields_captures_and_caps():
    class UpstreamBoom(Exception):
        pass

    exc = UpstreamBoom("429 rate limit exceeded " + "z" * 9000)
    got = hermes_logger.failure_error_fields({"exception": exc}, None, "failure")
    assert got["error_class"] == "UpstreamBoom"
    assert "429 rate limit exceeded" in got["error"]
    assert len(got["error"]) <= hermes_logger.ERROR_TEXT_MAX


def test_build_record_records_why_a_failure_failed():
    # The bug this fixes: 787 failures in one day logged status='failure', response=None,
    # finish_reason=None and NO error field -> the log could count them, never explain them.
    class Boom(Exception):
        pass

    rec = hermes_logger.build_record(
        {"model": "deepseek-v4-flash", "exception": Boom("upstream 429")}, None, 0.13, "failure"
    )
    assert rec["status"] == "failure"
    assert rec["error_class"] == "Boom"
    assert "upstream 429" in rec["error"]


def test_build_record_success_still_has_no_error():
    rec = hermes_logger.build_record(
        {"model": "m"}, {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]}, 1.0, "success"
    )
    assert rec["error"] is None and rec["error_class"] is None
    assert rec["response"] == "hi"


def test_parse_and_record_zai_weekly_quota():
    err = (
        "RateLimitError: OpenAIException - Weekly/Monthly Limit Exhausted. "
        "Your limit will reset at 2026-08-29 21:07:02"
    )
    until = hermes_logger.parse_quota_until(err)
    assert until.year == 2026 and until.month == 8 and until.day == 29
    assert hermes_logger.is_glm_quota_error(err)
    assert not hermes_logger.is_glm_quota_error("ordinary 429 try again in 45 seconds")


def test_route_exhausted_glm_rewrites_only_while_marker_live(tmp_path):
    marker = str(tmp_path / "zai.json")
    err = "Weekly/Monthly Limit Exhausted. Your limit will reset at 2099-01-01 00:00:00"
    hermes_logger.record_quota_exhaustion(err, marker_path=marker, rewrite_to="together-glm")
    data = {"model": "glm-5.3", "messages": [{"role": "user", "content": "hi"}]}
    out = hermes_logger.route_exhausted_glm(data, marker_path=marker, rewrite_to="together-glm")
    assert out["model"] == "together-glm"
    assert out["metadata"]["zai_quota_rewrite"]["from"] == "glm-5.3"
    grouped = {"model": "glm-coding", "model_group": "glm-coding"}
    grouped_out = hermes_logger.route_exhausted_glm(
        grouped, marker_path=marker, rewrite_to="together-glm"
    )
    assert grouped_out["model"] == "together-glm"
    assert grouped_out["model_group"] == "together-glm"
    # vision / non-glm stay
    vis = {"model": "glm-vision", "messages": []}
    assert hermes_logger.route_exhausted_glm(vis, marker_path=marker)["model"] == "glm-vision"
    local = {"model": "hermes-local"}
    assert hermes_logger.route_exhausted_glm(local, marker_path=marker)["model"] == "hermes-local"
    # expired marker is a no-op
    hermes_logger.write_quota_marker(datetime(2020, 1, 1, 0, 0, 0), marker_path=marker)
    again = {"model": "glm-coding"}
    assert hermes_logger.route_exhausted_glm(again, marker_path=marker)["model"] == "glm-coding"
