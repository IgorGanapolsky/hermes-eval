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
import time

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

# GLM-5.2/turbo are REASONING models: hidden reasoning_content is billed against the
# request's max_tokens, so a small budget (e.g. 20-256) is fully consumed thinking and
# the call returns HTTP 200 with EMPTY content (finish_reason=length). Data science on
# the live log (2026-07-08) found this in 31% of GLM "successes" (111/360) — a SILENT
# quality bug invisible to failure-based monitoring. Floor the budget so reasoning can't
# starve the answer. max_tokens is a CEILING (GLM stops at finish_reason=stop when done),
# so flooring adds ~0 cost on short answers but eliminates the empty-response failures.
GLM_MIN_MAX_TOKENS = int(os.environ.get("HERMES_GLM_MIN_MAX_TOKENS", "1024"))


def _model_strings(kwargs):
    """(request model, deployment model) as lowercased strings for route matching."""
    model = str(kwargs.get("model") or "")
    lp = kwargs.get("litellm_params") or {}
    dep_model = str(lp.get("model") or "") if isinstance(lp, dict) else ""
    return model, dep_model


def clamp_openrouter_max_tokens(kwargs, cap=None):
    """Clamp max_tokens for openrouter/* deployments so a low credit balance can't 402
    the emergency route. Pure helper (unit-testable); mutates and returns kwargs."""
    cap = OPENROUTER_MAX_TOKENS if cap is None else cap
    model, dep_model = _model_strings(kwargs)
    if not (model.startswith("openrouter/") or dep_model.startswith("openrouter/")):
        return kwargs
    mt = kwargs.get("max_tokens")
    if not isinstance(mt, int) or mt > cap:
        kwargs["max_tokens"] = cap
    return kwargs


def raise_glm_min_max_tokens(kwargs, floor=None):
    """Raise max_tokens to a floor for GLM (z.ai reasoning) deployments so hidden
    reasoning can't consume the whole budget and return empty content. Pure helper;
    only RAISES a too-small explicit budget, never lowers one. Excludes the
    openrouter/* glm fallback (that route is clamped down separately)."""
    floor = GLM_MIN_MAX_TOKENS if floor is None else floor
    model, dep_model = _model_strings(kwargs)
    if model.startswith("openrouter/") or dep_model.startswith("openrouter/"):
        return kwargs
    if "glm" not in model.lower() and "glm" not in dep_model.lower():
        return kwargs
    mt = kwargs.get("max_tokens")
    if isinstance(mt, int) and mt < floor:
        kwargs["max_tokens"] = floor
    return kwargs


# ---- Alerting -------------------------------------------------------------------
# The callback already sees every served call; make it also PAGE on the two RARE,
# HARD, ACTIONABLE events that used to be silent (found out only via garbage output,
# per the zai-quota-exhaustion / never-headline-fixed-while-degraded incidents):
#   1. GLM quota EXHAUSTION (429 code 1310, weekly/monthly cap) -> every agent
#      silently drops to a local fallback until the reset. A state, not a blip.
#   2. runaway token burn (the 27.8M-token/day incident) with no signal.
# Deliberately NOT paged here: generic GLM failures. Data science on the live log
# (2026-07-08) showed a 33% GLM failure rate in sustained clusters -> routine
# rate-limiting/blips that recover. Real-time paging on those is pure noise; the
# 30-min `hermes-burn-alert.js` poller's 6h "degraded" alert owns sustained
# degradation. This layer stays quiet unless something is genuinely wrong.
# ntfy is the existing phone-alert channel (same as the yolo-guard on this fleet).
# All best-effort; a failed alert never breaks a request.
ALERTS_ENABLED = os.environ.get("HERMES_ALERT_ENABLED", "1") != "0"
NTFY_TOPIC = os.environ.get("HERMES_ALERT_NTFY_TOPIC", "yolo-guard-fdh8ktuw1vtxb5sb")
ALERT_STATE_PATH = os.environ.get(
    "HERMES_ALERT_STATE_PATH", os.path.join(os.path.dirname(LOG_PATH), "alert-state.json")
)
ALERT_COOLDOWN_SEC = int(os.environ.get("HERMES_ALERT_COOLDOWN_SEC", "3600"))
BURN_WINDOW_SEC = int(os.environ.get("HERMES_BURN_WINDOW_SEC", "3600"))
# 5M/1h = 2.3x the busiest legit hour ever observed in traffic.jsonl (2.15M), so a
# trip means a runaway loop, not a heavy-but-real hour. Tune via env if history shifts.
BURN_TOKENS_THRESHOLD = int(os.environ.get("HERMES_BURN_TOKENS_THRESHOLD", "5000000"))


def extract_error_text(kwargs, response_obj):
    """Best-effort error string from a failure event (exception / dict / str shapes)."""
    for cand in (kwargs.get("exception"), kwargs.get("traceback_exception"), response_obj):
        if cand:
            return str(cand)
    slo = kwargs.get("standard_logging_object") or {}
    return str(slo.get("error_str") or "")


def classify_failure(error_text):
    """Return an alert reason ONLY for GLM quota EXHAUSTION, else None.

    Quota exhaustion (HTTP 429 code 1310 "weekly/monthly limit exhausted") is the one
    failure worth an instant page: it persists until the reset and silently routes every
    agent to a weaker local model. Generic GLM failures/rate-limit bursts are the 33%
    background-noise case (see module comment) and are intentionally NOT paged here."""
    et = (error_text or "").lower()
    if any(s in et for s in ("1310", "quota", "exhaust", "insufficient")):
        return "GLM quota exhausted (429/1310) -> every agent now on local fallback"
    return None


def update_burn(state, total_tokens, now, window_sec, threshold):
    """Rolling-window token accumulator. Pure: returns (new_state, crossed_now).

    state: {"window_start": float, "tokens": int, "alerted": bool}. The window resets
    once window_sec elapses; crossed_now is True exactly on the call that first trips
    the threshold within a window (so the alert fires once, not on every later call)."""
    st = dict(state or {})
    start = st.get("window_start")
    if start is None or (now - start) >= window_sec:
        st = {"window_start": now, "tokens": 0, "alerted": False}
    st["tokens"] = int(st.get("tokens", 0)) + int(total_tokens or 0)
    crossed = st["tokens"] >= threshold and not st.get("alerted")
    if crossed:
        st["alerted"] = True
    return st, crossed


def should_alert(now, last_alert_ts, cooldown_sec):
    """Rate-limit: True if enough time has passed since the last alert of this kind."""
    return last_alert_ts is None or (now - last_alert_ts) >= cooldown_sec


def send_ntfy(title, message, topic=None, priority="high", tags="warning"):
    """Best-effort phone alert via ntfy.sh. Never raises; short timeout so it can't
    stall the logging path."""
    topic = topic or NTFY_TOPIC
    if not (ALERTS_ENABLED and topic):
        return False
    with contextlib.suppress(Exception):
        import urllib.request

        req = urllib.request.Request(
            f"https://ntfy.sh/{topic}",
            data=message.encode("utf-8"),
            headers={"Title": title, "Priority": priority, "Tags": tags},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=3)
        return True
    return False


def _load_alert_state():
    with contextlib.suppress(Exception), open(ALERT_STATE_PATH, encoding="utf-8") as f:
        return json.load(f)
    return {}


def _save_alert_state(state):
    with contextlib.suppress(Exception):
        os.makedirs(os.path.dirname(ALERT_STATE_PATH), exist_ok=True)
        with open(ALERT_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f)


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


def extract_finish_reason(response_obj, slo):
    """Best-effort finish_reason ('stop' | 'length' | 'tool_calls' | ...); None if absent."""
    fr = None
    try:
        fr = response_obj["choices"][0]["finish_reason"]
    except Exception:
        fr = slo.get("finish_reason") if isinstance(slo, dict) else None
    return fr if isinstance(fr, str) else None


def tools_offered(kwargs):
    """Whether the REQUEST supplied tools at all.

    Without this you cannot tell a stuck agent from an ordinary chat completion:
    both show has_tool_calls=False. A vision or embedding call legitimately never
    calls a tool. Any spin/no-progress detector must only judge calls where tools
    were actually available."""
    with contextlib.suppress(Exception):
        tools = kwargs.get("tools")
        if tools:
            return True
        funcs = kwargs.get("functions")
        if funcs:
            return True
    return False


def has_tool_calls(response_obj):
    """True if the response carried tool_calls (empty content is then legitimate, not a
    truncation bug — the distinction the raw 'empty response' metric couldn't make)."""
    with contextlib.suppress(Exception):
        msg = response_obj["choices"][0]["message"]
        tc = msg.get("tool_calls") if isinstance(msg, dict) else getattr(msg, "tool_calls", None)
        return bool(tc)
    return False


def extract_tool_calls(response_obj):
    """The actual tool-call payload (name + arguments), serialized to plain dicts, or None.
    The boolean has_tool_calls flags THAT a tool was called; this captures WHAT — the crux
    of any tool-use distillation dataset, which was previously discarded (payload lived only
    in the response object, and `response` text is empty on a pure tool-call turn)."""
    with contextlib.suppress(Exception):
        msg = response_obj["choices"][0]["message"]
        tc = msg.get("tool_calls") if isinstance(msg, dict) else getattr(msg, "tool_calls", None)
        if not tc:
            return None
        out = []
        for c in tc:
            fn = c.get("function") if isinstance(c, dict) else getattr(c, "function", None)
            name = (
                (fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", None))
                if fn
                else None
            )
            args = (
                (fn.get("arguments") if isinstance(fn, dict) else getattr(fn, "arguments", None))
                if fn
                else None
            )
            out.append({"name": name, "arguments": args})
        return out or None
    return None


def empty_content_kind(response, finish_reason, tool_calls):
    """Classify an empty-content success so drift analysis can separate the real defect
    from normal tool use. Returns None when content is present.
      'tool_call'  -> empty by design (payload in tool_calls); healthy
      'truncated'  -> reasoning/length ate the budget (finish_reason=length); the bug
                      the GLM max_tokens floor targets
      'empty'      -> empty for some other reason; worth a look"""
    if (response or "").strip():
        return None
    if tool_calls:
        return "tool_call"
    if finish_reason == "length":
        return "truncated"
    return "empty"


def build_record(kwargs, response_obj, latency_s, status):
    """Pure record builder (unit-testable)."""
    slo = kwargs.get("standard_logging_object") or {}
    content = extract_content(response_obj, slo)
    finish_reason = extract_finish_reason(response_obj, slo)
    tool_calls = has_tool_calls(response_obj)
    return {
        "model": kwargs.get("model") or slo.get("model"),
        "messages": kwargs.get("messages") or slo.get("messages"),
        "response": content,
        "finish_reason": finish_reason,
        "has_tool_calls": tool_calls,
        "tool_calls": extract_tool_calls(response_obj),
        "tools_offered": tools_offered(kwargs),
        "empty_kind": empty_content_kind(content, finish_reason, tool_calls),
        "prompt_tokens": slo.get("prompt_tokens"),
        "completion_tokens": slo.get("completion_tokens"),
        "total_tokens": slo.get("total_tokens"),
        "latency_s": latency_s,
        "status": status,
    }


class HermesJSONLLogger(CustomLogger):
    async def async_pre_call_deployment_hook(self, kwargs, call_type):
        with contextlib.suppress(Exception):  # never break a request because of the guards
            raise_glm_min_max_tokens(kwargs)  # GLM reasoning floor (empty-content fix)
            return clamp_openrouter_max_tokens(kwargs)  # openrouter cap (last-resort 402 fix)
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
            with contextlib.suppress(Exception):  # alerting must never break a request
                self._maybe_alert(
                    status, rec.get("model"), kwargs, response_obj, rec.get("total_tokens")
                )
        except Exception as e:  # never break a request because logging failed
            print(f"[hermes_logger] log error: {e}")

    def _maybe_alert(self, status, model, kwargs, response_obj, total_tokens):
        """Fire phone alerts on the two previously-silent failure modes. Best-effort:
        reads/writes a tiny state file for cooldown + rolling burn window."""
        if not ALERTS_ENABLED:
            return
        now = time.time()
        state = _load_alert_state()
        changed = False
        if status == "failure":
            reason = classify_failure(extract_error_text(kwargs, response_obj))
            if (
                reason
                and should_alert(now, state.get("last_quota"), ALERT_COOLDOWN_SEC)
                and send_ntfy("Hermes GLM quota exhausted", f"{reason} (model={model})")
            ):
                state["last_quota"] = now
                changed = True
        elif status == "success":
            burn, crossed = update_burn(
                state.get("burn"), total_tokens, now, BURN_WINDOW_SEC, BURN_TOKENS_THRESHOLD
            )
            state["burn"] = burn
            changed = True
            if crossed:
                send_ntfy(
                    "Hermes token-burn runaway",
                    f"{burn['tokens']:,} tokens in {BURN_WINDOW_SEC // 60}m "
                    f"(>{BURN_TOKENS_THRESHOLD:,}) - check for a stuck loop",
                )
        if changed:
            _save_alert_state(state)


proxy_handler_instance = HermesJSONLLogger()
