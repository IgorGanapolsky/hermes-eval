import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))

from summarize_errors import error_summaries, redact


def test_error_summaries_reads_promptfoo_rows_and_deduplicates():
    payload = {
        "results": {
            "results": [
                {"error": "provider unavailable"},
                {"response": {"error": "provider unavailable"}},
                {"response": {"error": "quota exceeded"}},
            ]
        }
    }
    assert error_summaries(payload) == ["provider unavailable", "quota exceeded"]


def test_redact_removes_provider_keys_and_bearer_tokens():
    provider_key = "sk-or-v1-" + "example"
    bearer_token = "bearer-" + "example"
    message = f"api_key={provider_key} Authorization: Bearer {bearer_token}"
    redacted = redact(message)
    assert provider_key not in redacted
    assert bearer_token not in redacted
    assert redacted.count("[REDACTED]") == 2


def test_redact_bounds_multiline_errors():
    assert len(redact("line one\n" + "x" * 1000)) == 500
