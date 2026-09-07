"""GPT-6 Astra spend governor — $10/month, opt-in only, fail over to GLM.

Standard API rates (2026-09): $10 / 1M input, $50 / 1M output.
Fast / Pro / Flex aliases are 2x and are rewritten to standard.
Never put Astra in default fallbacks. Agents must name gpt-6-astra or astra.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

MONTHLY_CAP_USD = 10.0
SOFT_STOP_USD = 9.50
INPUT_USD_PER_M = 10.0
OUTPUT_USD_PER_M = 50.0
MAX_OUT_TOKENS = 800
FALLBACK_MODEL = "glm-coding"


def fallback_model() -> str:
    return os.environ.get("HERMES_ASTRA_FALLBACK") or FALLBACK_MODEL


ASTRA_CANONICAL = "gpt-6-astra"
ASTRA_ALIASES = frozenset(
    {
        "gpt-6-astra",
        "astra",
        "openai/gpt-6-astra",
        "gpt-6-astra-fast",
        "gpt-6-astra-flex",
        "gpt-6-astra-pro",
        "gpt-6-astra-pro-fast",
        "gpt-6-astra-pro-flex",
        "openai/gpt-6-astra-fast",
        "openai/gpt-6-astra-flex",
        "openai/gpt-6-astra-pro",
    }
)
DEFAULT_LEDGER = Path.home() / ".hermes" / "astra-spend.json"


def month_key(now: Optional[datetime] = None) -> str:
    stamp = now or datetime.now(tz=timezone.utc)
    return stamp.strftime("%Y-%m")


def estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (
        (max(0, int(prompt_tokens)) / 1_000_000.0) * INPUT_USD_PER_M
        + (max(0, int(completion_tokens)) / 1_000_000.0) * OUTPUT_USD_PER_M
    )


def is_astra(model: str) -> bool:
    name = str(model or "").split("/")[-1].lower()
    full = str(model or "").lower()
    return full in ASTRA_ALIASES or name in ASTRA_ALIASES or "gpt-6-astra" in full


def _empty_ledger(month: str) -> dict[str, Any]:
    return {"month": month, "usd": 0.0, "calls": 0}


def load_ledger(ledger_path: Optional[Path] = None) -> dict[str, Any]:
    path = Path(ledger_path or os.environ.get("HERMES_ASTRA_LEDGER") or DEFAULT_LEDGER)
    month = month_key()
    if not path.is_file():
        return _empty_ledger(month)
    try:
        body = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, TypeError):
        return _empty_ledger(month)
    if not isinstance(body, dict) or body.get("month") != month:
        return _empty_ledger(month)
    return {
        "month": month,
        "usd": float(body.get("usd") or 0),
        "calls": int(body.get("calls") or 0),
    }


def save_ledger(payload: dict[str, Any], ledger_path: Optional[Path] = None) -> Path:
    path = Path(ledger_path or os.environ.get("HERMES_ASTRA_LEDGER") or DEFAULT_LEDGER)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def remaining_usd(ledger: Optional[dict[str, Any]] = None, ledger_path: Optional[Path] = None) -> float:
    body = ledger if ledger is not None else load_ledger(ledger_path)
    return max(0.0, MONTHLY_CAP_USD - float(body.get("usd") or 0))


def guard_request(data: dict[str, Any], ledger_path: Optional[Path] = None) -> dict[str, Any]:
    """Clamp / rewrite an Astra request. No-op for every other model."""
    if not isinstance(data, dict):
        return data
    requested = str(data.get("model") or "")
    if not is_astra(requested):
        return data
    meta = data.get("metadata")
    if not isinstance(meta, dict):
        meta = {}
        data["metadata"] = meta
    leaf = requested.split("/")[-1].lower()
    if leaf not in {"gpt-6-astra", "astra"}:
        data["model"] = ASTRA_CANONICAL
        meta["astra_alias"] = "standard_only"
    body = load_ledger(ledger_path)
    if float(body.get("usd") or 0) >= SOFT_STOP_USD:
        dest = fallback_model()
        data["model"] = dest
        if str(data.get("model_group") or ""):
            data["model_group"] = dest
        meta["astra_budget_rewrite"] = {
            "from": requested,
            "to": dest,
            "reason": "monthly_cap",
            "usd": body.get("usd"),
            "cap": MONTHLY_CAP_USD,
        }
        return data
    asked = data.get("max_completion_tokens") or data.get("max_tokens") or MAX_OUT_TOKENS
    data["max_completion_tokens"] = min(int(asked), MAX_OUT_TOKENS)
    data.pop("max_tokens", None)
    data["reasoning_effort"] = "low"
    data["disable_fallbacks"] = True
    return data


def record_usage(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    ledger_path: Optional[Path] = None,
) -> dict[str, Any]:
    if not is_astra(model):
        return load_ledger(ledger_path)
    body = load_ledger(ledger_path)
    body["usd"] = round(float(body.get("usd") or 0) + estimate_cost(prompt_tokens, completion_tokens), 6)
    body["calls"] = int(body.get("calls") or 0) + 1
    body["month"] = month_key()
    save_ledger(body, ledger_path)
    return body


try:  # optional: LiteLLM custom callback when hermes_logger is not loaded
    from litellm.integrations.custom_logger import CustomLogger
except Exception:  # pragma: no cover
    CustomLogger = object


class AstraBudgetLogger(CustomLogger):
    """Budget-only hook for configs that do not load hermes_logger."""

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        return guard_request(data)

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._record(kwargs)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._record(kwargs)

    def _record(self, kwargs: dict[str, Any]) -> None:
        slo = kwargs.get("standard_logging_object") or {}
        model = str(kwargs.get("model") or slo.get("model") or "")
        record_usage(
            model,
            int(slo.get("prompt_tokens") or 0),
            int(slo.get("completion_tokens") or 0),
        )


proxy_handler_instance = AstraBudgetLogger()
