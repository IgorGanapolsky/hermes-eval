import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))
import mine_failures


def rec(**kw):
    base = {"status": "success", "response": "", "empty_kind": None}
    base.update(kw)
    return base


def test_status_failure_is_mined():
    assert mine_failures.classify(rec(status="failure")) == "status_failure"


def test_content_present_is_healthy():
    assert mine_failures.classify(rec(response="the answer is 42")) is None


def test_tool_call_empty_is_not_a_failure():
    # the qwen3:8b 62%-empty case: empty content by design -> must NOT be mined
    assert mine_failures.classify(rec(response="", empty_kind="tool_call")) is None


def test_truncated_empty_is_its_own_class():
    # reasoning ate the budget (finish_reason=length) -> the real defect
    assert mine_failures.classify(rec(response="", empty_kind="truncated")) == "truncated_empty"


def test_unexplained_empty_and_legacy_records():
    # empty_kind == "empty" -> unexplained
    assert mine_failures.classify(rec(response="", empty_kind="empty")) == "empty_response"
    # legacy record logged before empty_kind existed -> conservative, still surfaced
    legacy = {"status": "success", "response": ""}
    assert mine_failures.classify(legacy) == "empty_response"
