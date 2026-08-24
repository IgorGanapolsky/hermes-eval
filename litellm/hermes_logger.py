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
import re
import time
from datetime import datetime

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

# Only glm-vision (GLM-4.6V) can SEE on this fleet. z.ai's coding endpoint hard-rejects
# image content on the text routes — verified 2026-07-25: glm-coding returns HTTP 400
# "messages.content.type is invalid, allowed values: ['text']", then burns ~36s walking a
# text-only fallback chain that also can't see. Agents attach screenshots against whatever
# model the session happens to be on (opencode-yolo TUI, hermes-yolo), so route by request
# CONTENT rather than trusting each client to pick the vision model. glm-vision tool-calls
# correctly (verified same day), so an agentic turn survives the reroute.
VISION_MODEL = os.environ.get("HERMES_VISION_MODEL", "glm-vision")
VISION_CAPABLE_MODELS = {
    m.strip()
    for m in os.environ.get(
        "HERMES_VISION_CAPABLE", "glm-vision,vision-gemini,vision-free,vision-local,muse-spark"
    ).split(",")
    if m.strip()
}
# OpenAI-compatible multimodal content blocks, across the spellings clients emit.
IMAGE_PART_TYPES = {"image_url", "image", "input_image"}

# z.ai Coding Plan weekly/monthly 429 is a STATE until the reset timestamp in the
# error body (desktop streaming then surfaces "No deployments available" because
# pre_call_checks + 45s cooldown refuse the glm-* group before fallbacks). Remap
# text GLM groups onto a different quota pool before the router runs.
QUOTA_MARKER_PATH = os.environ.get(
    "HERMES_ZAI_QUOTA_MARKER",
    os.path.expanduser("~/.hermes/quota/zai-coding-exhausted-until.json"),
)
QUOTA_REWRITE_MODEL = os.environ.get("HERMES_ZAI_QUOTA_REWRITE_MODEL", "together-glm")
GLM_TEXT_GROUPS = {
    "glm-5.3",
    "glm-coding",
    "glm-5.2",
    "glm-turbo",
    "glm-47",
    "glm-4.7",
    "glm-4.7-flash",
}
RESET_AT_RE = re.compile(r"reset at (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", re.I)
QUOTA_EXHAUST_NEEDLES = (
    "weekly/monthly limit exhausted",
    "limit will reset at",
    "code 1310",
)


def has_image_parts(messages):
    """True if any message carries an image content block."""
    for msg in messages or []:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and str(part.get("type") or "") in IMAGE_PART_TYPES:
                return True
    return False


def route_image_request_to_vision(data, vision_model=None, vision_capable=None):
    """Send image-bearing requests to the vision route instead of 400ing on a text model.

    Pure helper (unit-testable); mutates and returns the proxy request `data`. No-op when
    the request has no image or the requested group can already see."""
    if not isinstance(data, dict):
        return data
    vision_model = VISION_MODEL if vision_model is None else vision_model
    capable = VISION_CAPABLE_MODELS if vision_capable is None else vision_capable
    requested = str(data.get("model") or "")
    # Clients may address a group bare ("glm-coding") or prefixed ("hermes/glm-coding").
    if not requested or requested.split("/")[-1] in capable:
        return data
    if not has_image_parts(data.get("messages")):
        return data
    data["model"] = vision_model
    # Never silent: the reroute changes which model answers, so say so in the proxy log
    # (the JSONL record already lands under the model that actually ran).
    print(f"[hermes_logger] image input detected: routing {requested} -> {vision_model}")
    return data


def parse_quota_until(error_text):
    """Parse z.ai 'reset at YYYY-MM-DD HH:MM:SS' from a 429 body. None if absent."""
    if not error_text:
        return None
    m = RESET_AT_RE.search(str(error_text))
    if not m:
        return None
    with contextlib.suppress(ValueError):
        return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
    return None


def is_glm_quota_error(error_text):
    et = str(error_text or "").lower()
    return any(n in et for n in QUOTA_EXHAUST_NEEDLES)


def write_quota_marker(until, marker_path=None, rewrite_to=None):
    """Persist exhaustion until `until` (datetime). Returns the path written."""
    path = marker_path or QUOTA_MARKER_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "until": until.strftime("%Y-%m-%d %H:%M:%S") if until else None,
        "rewrite_to": rewrite_to or QUOTA_REWRITE_MODEL,
        "source": "z.ai coding-plan 429 weekly/monthly",
        "written_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
        f.write("\n")
    return path


def quota_exhausted(now=None, marker_path=None):
    """True while the on-disk marker's until is in the future."""
    path = marker_path or QUOTA_MARKER_PATH
    if not os.path.isfile(path):
        return False
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError, TypeError):
        return False
    raw = payload.get("until") if isinstance(payload, dict) else None
    if not raw:
        return False
    try:
        until = datetime.strptime(str(raw), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return False
    return (now or datetime.now()) < until


def route_exhausted_glm(data, now=None, marker_path=None, rewrite_to=None):
    """Before routing: send text GLM groups off the dead z.ai plan.

    Pure helper. No-op for vision, non-GLM, or when the marker is absent/expired.
    Desktop streaming 429s are this miss — cooldown + pre_call_checks never reach
    fallbacks. Changing the model group here makes the first hop a live pool.
    """
    if not isinstance(data, dict):
        return data
    requested = str(data.get("model") or "")
    group = requested.split("/")[-1]
    if group not in GLM_TEXT_GROUPS:
        return data
    if not quota_exhausted(now=now, marker_path=marker_path):
        return data
    dest = rewrite_to or QUOTA_REWRITE_MODEL
    if group == dest:
        return data
    data["model"] = dest
    # LiteLLM sometimes routes on model_group even after model is rewritten.
    if str(data.get("model_group") or "").split("/")[-1] in GLM_TEXT_GROUPS:
        data["model_group"] = dest
    meta = data.get("metadata")
    if not isinstance(meta, dict):
        meta = {}
        data["metadata"] = meta
    meta["zai_quota_rewrite"] = {"from": requested, "to": dest}
    print(f"[hermes_logger] z.ai coding quota exhausted: routing {requested} -> {dest}")
    return data


def fallback_attempt_count(kwargs, slo=None):
    """How many prior models LiteLLM already tried (Ramp-style receipt field)."""
    slo = slo or {}
    meta = kwargs.get("metadata") if isinstance(kwargs.get("metadata"), dict) else {}
    lp = kwargs.get("litellm_params") if isinstance(kwargs.get("litellm_params"), dict) else {}
    lp_meta = lp.get("metadata") if isinstance(lp.get("metadata"), dict) else {}
    prev = (
        meta.get("previous_models") or lp_meta.get("previous_models") or slo.get("previous_models")
    )
    if isinstance(prev, list):
        return len(prev)
    n = kwargs.get("attempted_fallbacks")
    if n is None:
        n = slo.get("attempted_fallbacks")
    if isinstance(n, int) and n >= 0:
        return n
    return 0


def record_quota_exhaustion(error_text, marker_path=None, rewrite_to=None):
    """If this failure is a z.ai weekly/monthly cap, write/refresh the marker."""
    if not is_glm_quota_error(error_text):
        return None
    until = parse_quota_until(error_text)
    if until is None:
        until = datetime.now().replace(hour=23, minute=59, second=59, microsecond=0)
    return write_quota_marker(until, marker_path=marker_path, rewrite_to=rewrite_to)


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


# ---- Stale-tool-output pruning (deterministic, idempotent) ------------------------
# Data science on the live log (2026-07-17, rotation 07-10→07-17: 1,911 calls, 23.17M
# prompt tokens): on the biggest GLM calls (140K+ tokens, 420+ messages) tool outputs
# are 57% of prompt chars, and only the newest same-key tool result is still
# trustworthy. Deterministic middleware (safe-prompt-pruning-layer pattern): OUTSIDE a
# protected recency window, replace — never delete, the assistant tool_calls ↔ tool
# tool_call_id pairing is schema-required — stale tool-message CONTENT with a one-line
# stub. Two rules, both idempotent and pure functions of the message list:
#   1. superseded: a LATER assistant tool_call re-issues the same (name, args) key
#   2. aged: the message is outside the window and its content is large
# Subscription-GLM deployments only (same targeting as the max_tokens floor); the
# in-place mutation intentionally carries to a subsequent local-fallback attempt,
# where a smaller context window benefits even more. Kill switch:
# HERMES_PRUNE_TOOL_OUTPUTS=0.
PRUNE_ENABLED = os.environ.get("HERMES_PRUNE_TOOL_OUTPUTS", "1") != "0"
PRUNE_PROTECT_LAST_N = int(os.environ.get("HERMES_PRUNE_PROTECT_LAST_N", "60"))
PRUNE_MIN_CHARS = int(os.environ.get("HERMES_PRUNE_MIN_CHARS", "600"))
PRUNE_STUB_PREFIX = "[pruned stale tool output"


def _tool_call_keys(messages):
    """tool_call_id -> (name, whitespace-normalized arguments) across all assistant
    tool_calls, plus the LAST message index at which each key was issued."""
    by_id, last_index_by_key = {}, {}
    for i, m in enumerate(messages):
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        for c in m.get("tool_calls") or []:
            if not isinstance(c, dict):
                continue
            fn = c.get("function") or {}
            if not isinstance(fn, dict):
                continue
            key = (str(fn.get("name")), " ".join(str(fn.get("arguments") or "").split()))
            cid = c.get("id")
            if cid:
                by_id[cid] = key
            last_index_by_key[key] = i
    return by_id, last_index_by_key


def stub_stale_tool_outputs(kwargs, protect_last_n=None, min_chars=None):
    """Stub stale role:"tool" message content on subscription-GLM deployments. Pure
    helper (unit-testable); mutates messages in place and returns kwargs."""
    if not PRUNE_ENABLED:
        return kwargs
    protect_last_n = PRUNE_PROTECT_LAST_N if protect_last_n is None else protect_last_n
    min_chars = PRUNE_MIN_CHARS if min_chars is None else min_chars
    model, dep_model = _model_strings(kwargs)
    if model.startswith("openrouter/") or dep_model.startswith("openrouter/"):
        return kwargs
    if "glm" not in model.lower() and "glm" not in dep_model.lower():
        return kwargs
    msgs = kwargs.get("messages")
    if not isinstance(msgs, list) or len(msgs) <= protect_last_n:
        return kwargs
    by_id, last_index_by_key = _tool_call_keys(msgs)
    cutoff = len(msgs) - protect_last_n
    for i, m in enumerate(msgs[:cutoff]):
        if not isinstance(m, dict) or m.get("role") != "tool":
            continue
        content = m.get("content")
        if not isinstance(content, str) or content.startswith(PRUNE_STUB_PREFIX):
            continue
        key = by_id.get(m.get("tool_call_id"))
        superseded = key is not None and last_index_by_key.get(key, -1) > i
        if superseded or len(content) >= min_chars:
            reason = "superseded by a newer identical call" if superseded else "aged out"
            m["content"] = (
                f"{PRUNE_STUB_PREFIX}: {reason}; {len(content)} chars elided — "
                "re-run the tool if this result is needed]"
            )
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


# Provider error bodies echo the request back often enough that an unredacted error
# field would turn this log into a credential store. traffic.jsonl is world-readable
# and ships to the distill dataset, so redaction happens here, not at read time.
_SECRET_RE = re.compile(
    r"(sk-[A-Za-z0-9_\-]{8,}"
    r"|Bearer\s+[A-Za-z0-9_\-\.=]{8,}"
    r"|gh[pousr]_[A-Za-z0-9]{16,}"
    r"|(?i:api[-_]?key\"?\s*[:=]\s*\"?)[A-Za-z0-9_\-]{8,})"
)
ERROR_TEXT_MAX = int(os.environ.get("HERMES_ERROR_TEXT_MAX", "2000"))


def redact_secrets(text):
    """Replace credential-shaped substrings. Pure; never raises on odd input."""
    if not text:
        return ""
    with contextlib.suppress(Exception):
        return _SECRET_RE.sub("[REDACTED]", str(text))
    return "[unredactable]"


def failure_error_fields(kwargs, response_obj, status):
    """Redacted, length-capped error text + exception class for failure records.

    Returns {} on success: extract_error_text() falls through to response_obj, so on a
    successful call it would serialize the entire response into an 'error' field. The
    whole point of this pair is that build_record previously recorded status='failure'
    with response=None and no error anywhere, which made 787 failures in one day
    unattributable — the log could count them but never say why."""
    if status != "failure":
        return {"error": None, "error_class": None}
    exc = kwargs.get("exception")
    return {
        "error": redact_secrets(extract_error_text(kwargs, response_obj))[:ERROR_TEXT_MAX] or None,
        "error_class": type(exc).__name__ if exc is not None else None,
    }


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
    # LiteLLM's standard_logging_object carries 39 fields; we were recording 13 and
    # dropping the ones that answer the questions we actually ask of this log.
    #
    # model_group vs model is the important pair. model_group is what the CALLER asked
    # for (e.g. "glm-coding"); model is what was SERVED. On 2026-08-05 glm-coding
    # requests were being served by deepseek-v4-flash-free after a quota exhaustion —
    # silent substitution that took a manual investigation to find, because the log only
    # ever recorded one of the two names. With both, it is a one-line query.
    #
    # response_cost turns this from a token log into a dollar log, which is what
    # "cost per accepted task" needs. api_base identifies WHICH box served a request:
    # hermes-local round-robins across this Mac and a Tailscale mini, and a 90s stall
    # traced to the mini lacking the model — invisible in a log without api_base.
    #
    # All reads are .get() with no fallback logic: this runs inside the gateway's
    # logging path and must never raise on an unexpected payload shape.
    metadata = slo.get("metadata") or {}
    return {
        # Why a failure failed. Without these two, a failure record carries status
        # ='failure', response=None and finish_reason=None — countable but never
        # explainable (787 such records in a single day on 2026-08-13).
        **failure_error_fields(kwargs, response_obj, status),
        "model": kwargs.get("model") or slo.get("model"),
        "model_group": slo.get("model_group"),
        "model_id": slo.get("model_id"),
        "api_base": slo.get("api_base"),
        "custom_llm_provider": slo.get("custom_llm_provider"),
        "response_cost": slo.get("response_cost"),
        "cache_hit": slo.get("cache_hit"),
        "trace_id": slo.get("trace_id"),
        "litellm_call_id": slo.get("litellm_call_id"),
        # Where a caller-supplied workflow/task id rides, when one is passed. Null today;
        # recording it now means callers can start attributing without a logger change.
        "requester_metadata": metadata.get("requester_metadata"),
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
        "fallback_attempts": fallback_attempt_count(kwargs, slo),
        "status": status,
    }


class HermesJSONLLogger(CustomLogger):
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        """Proxy-level, BEFORE routing — the only place a different model group can still
        be chosen (the deployment hook below runs after that decision)."""
        with contextlib.suppress(Exception):  # never break a request because of the guard
            data = route_image_request_to_vision(data)
        with contextlib.suppress(Exception):
            data = route_exhausted_glm(data)
        return data

    async def async_pre_call_deployment_hook(self, kwargs, call_type):
        with contextlib.suppress(
            Exception
        ):  # each guard isolated: one failing must not skip others
            stub_stale_tool_outputs(kwargs)  # stale tool-output prune (GLM quota headroom)
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
            if status == "failure":
                with contextlib.suppress(Exception):
                    record_quota_exhaustion(
                        rec.get("error") or extract_error_text(kwargs, response_obj)
                    )
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
