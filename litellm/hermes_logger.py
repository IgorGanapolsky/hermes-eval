"""LiteLLM custom callback: append every call to a JSONL file.

This is the gateway -> golden-set / drift feed (no Postgres needed). Wire it in config.yaml:

    litellm_settings:
      callbacks: hermes_logger.proxy_handler_instance

Each line: {ts_end, model, messages, response, prompt_tokens, completion_tokens, total_tokens,
latency_s, status}. Path via HERMES_LOG_PATH (default ~/.hermes/litellm-logs/traffic.jsonl).
Curate these into golden.jsonl with error analysis; mine drift from the token/latency fields.
"""

import contextlib
import json
import os

try:  # guarded so the pure helpers can be unit-tested without litellm installed
    from litellm.integrations.custom_logger import CustomLogger
except Exception:  # pragma: no cover
    CustomLogger = object

LOG_PATH = os.environ.get(
    "HERMES_LOG_PATH", os.path.expanduser("~/.hermes/litellm-logs/traffic.jsonl")
)

# LiteLLM background health-check ping prompts — not real traffic for the golden set.
_HEALTH_CHECK_PROMPTS = {"", "hey, how's it going?"}

# OpenRouter (per-token, low balance) 402s any request whose max_tokens exceeds what the
# remaining credits can afford — Hermes asks for 65536, killing the LAST-RESORT fallback
# exactly when it's needed (z.ai quota exhaustion, 2026-07-07). Deployment-level clamp:
# config litellm_params.max_tokens does NOT override a request's own value.
OPENROUTER_MAX_TOKENS = int(os.environ.get("HERMES_OPENROUTER_MAX_TOKENS", "4096"))


def clamp_openrouter_max_tokens(kwargs, cap=None):
    """Clamp max_tokens for openrouter/* deployments so a low credit balance can't 402
    the emergency route. Pure helper (unit-testable); mutates and returns kwargs."""
    cap = OPENROUTER_MAX_TOKENS if cap is None else cap
    model = str(kwargs.get("model") or "")
    lp = kwargs.get("litellm_params") or {}
    dep_model = str(lp.get("model") or "") if isinstance(lp, dict) else ""
    if not (model.startswith("openrouter/") or dep_model.startswith("openrouter/")):
        return kwargs
    mt = kwargs.get("max_tokens")
    if not isinstance(mt, int) or mt > cap:
        kwargs["max_tokens"] = cap
    return kwargs


def is_health_check(messages):
    """True for a LiteLLM background health-check ping (single short canned message)."""
    if isinstance(messages, list) and len(messages) == 1 and isinstance(messages[0], dict):
        return str(messages[0].get("content", "")).strip().lower() in _HEALTH_CHECK_PROMPTS
    return False


def extract_content(response_obj, slo):
    """Best-effort response text across object/dict shapes; None if not a string."""
    content = None
    try:
        content = response_obj["choices"][0]["message"]["content"]
    except Exception:
        content = slo.get("response")
    return content if isinstance(content, str) else None


def build_record(kwargs, response_obj, latency_s, status):
    """Pure record builder (unit-testable)."""
    slo = kwargs.get("standard_logging_object") or {}
    return {
        "model": kwargs.get("model") or slo.get("model"),
        "messages": kwargs.get("messages") or slo.get("messages"),
        "response": extract_content(response_obj, slo),
        "prompt_tokens": slo.get("prompt_tokens"),
        "completion_tokens": slo.get("completion_tokens"),
        "total_tokens": slo.get("total_tokens"),
        "latency_s": latency_s,
        "status": status,
    }


class HermesJSONLLogger(CustomLogger):
    async def async_pre_call_deployment_hook(self, kwargs, call_type):
        with contextlib.suppress(Exception):  # never break a request because of the clamp
            return clamp_openrouter_max_tokens(kwargs)
        return kwargs

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._write(kwargs, response_obj, start_time, end_time, "success")

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._write(kwargs, response_obj, start_time, end_time, "success")

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._write(kwargs, response_obj, start_time, end_time, "failure")

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._write(kwargs, response_obj, start_time, end_time, "failure")

    def _write(self, kwargs, response_obj, start_time, end_time, status):
        try:
            slo = kwargs.get("standard_logging_object") or {}
            msgs = kwargs.get("messages") or slo.get("messages") or []
            if is_health_check(msgs):
                return
            latency = None
            with contextlib.suppress(Exception):
                latency = (end_time - start_time).total_seconds()
            rec = build_record(kwargs, response_obj, latency, status)
            rec["ts_end"] = str(end_time)
            os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str, ensure_ascii=False) + "\n")
        except Exception as e:  # never break a request because logging failed
            print(f"[hermes_logger] log error: {e}")


proxy_handler_instance = HermesJSONLLogger()
