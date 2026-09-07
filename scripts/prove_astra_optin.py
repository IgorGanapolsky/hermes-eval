#!/usr/bin/env python3
"""Prove Astra is listed and a 16-token ping succeeds. Never prints secrets."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = os.environ.get("LITELLM_BASE", "http://127.0.0.1:4010")
KEY = os.environ.get("LITELLM_MASTER_KEY") or "sk-hermes-local-dev"
LEDGER = Path.home() / ".hermes" / "astra-spend.json"


def _req(path: str, payload: dict | None = None, timeout: float = 60.0) -> tuple[int, dict]:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=data,
        headers={
            "Authorization": f"Bearer {KEY}",
            "Content-Type": "application/json",
        },
        method="GET" if payload is None else "POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
            return resp.status, body
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode(errors="replace")
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = {"error": raw[:240]}
        return exc.code, body


def main() -> int:
    code, models = _req("/v1/models")
    ids = sorted(str(m.get("id") or "") for m in (models.get("data") or []) if isinstance(m, dict))
    print(
        f"models_http={code} count={len(ids)} gpt-6-astra={('gpt-6-astra' in ids)} astra={('astra' in ids)}"
    )
    if code != 200 or "gpt-6-astra" not in ids:
        err = models.get("error") or models.get("message") or ""
        print(f"models_error={str(err)[:200]}")
        return 1

    code, ping = _req(
        "/v1/chat/completions",
        {
            "model": "gpt-6-astra",
            "messages": [{"role": "user", "content": "reply PING"}],
            "max_completion_tokens": 16,
        },
    )
    content = ""
    choices = ping.get("choices") or []
    if choices:
        content = str((choices[0].get("message") or {}).get("content") or "")
    usage = ping.get("usage") or {}
    print(f"ping_http={code} model={ping.get('model')} content={content!r} usage={usage}")
    if ping.get("error"):
        print(f"ping_error={str(ping.get('error'))[:240]}")
    time.sleep(0.4)
    if LEDGER.exists():
        ledger = json.loads(LEDGER.read_text())
        print(
            f"ledger month={ledger.get('month')} usd={ledger.get('usd')} "
            f"calls={ledger.get('calls')}"
        )
    else:
        print("ledger_exists=false")
    return 0 if code == 200 and content else 2


if __name__ == "__main__":
    raise SystemExit(main())
