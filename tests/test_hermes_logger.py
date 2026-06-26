import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "litellm"))
import hermes_logger


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
