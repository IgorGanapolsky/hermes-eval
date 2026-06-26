"""LiteLLM custom callback: append every call to a JSONL file.

This is the gateway -> golden-set / drift feed (no Postgres needed). Wire it in config.yaml:

    litellm_settings:
      callbacks: hermes_logger.proxy_handler_instance

Each line: {ts_end, model, messages, response, prompt_tokens, completion_tokens, total_tokens,
latency_s, status}. Path via HERMES_LOG_PATH (default ~/.hermes/litellm-logs/traffic.jsonl).
Curate these into golden.jsonl with error analysis; mine drift from the token/latency fields.
"""
import json
import os

from litellm.integrations.custom_logger import CustomLogger

LOG_PATH = os.environ.get(
    "HERMES_LOG_PATH", os.path.expanduser("~/.hermes/litellm-logs/traffic.jsonl")
)


class HermesJSONLLogger(CustomLogger):
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
            os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
            slo = kwargs.get("standard_logging_object") or {}
            msgs = kwargs.get("messages") or slo.get("messages") or []
            # skip LiteLLM background health-check pings — not real traffic for the golden set
            if isinstance(msgs, list) and len(msgs) == 1 and isinstance(msgs[0], dict):
                c0 = str(msgs[0].get("content", "")).strip()
                if c0 in ("", "Hey, how's it going?"):
                    return
            content = None
            try:
                content = response_obj["choices"][0]["message"]["content"]
            except Exception:
                content = slo.get("response")
            if not isinstance(content, str):
                content = None
            latency = None
            try:
                latency = (end_time - start_time).total_seconds()
            except Exception:
                pass
            rec = {
                "ts_end": str(end_time),
                "model": kwargs.get("model") or slo.get("model"),
                "messages": kwargs.get("messages") or slo.get("messages"),
                "response": content,
                "prompt_tokens": slo.get("prompt_tokens"),
                "completion_tokens": slo.get("completion_tokens"),
                "total_tokens": slo.get("total_tokens"),
                "latency_s": latency,
                "status": status,
            }
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str, ensure_ascii=False) + "\n")
        except Exception as e:  # never break a request because logging failed
            print(f"[hermes_logger] log error: {e}")


proxy_handler_instance = HermesJSONLLogger()
