#!/usr/bin/env python3
"""Print bounded, secret-redacted provider errors from a Promptfoo result."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

SECRET_PATTERNS = (
    re.compile(r"sk-or-v1-[A-Za-z0-9_-]+"),
    re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{16,}"),
    re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;}]+"),
    re.compile(r"(?i)((?:api[_-]?key|token)\s*[:=]\s*)[^\s,;}]+"),
)


def redact(value: str) -> str:
    text = value.replace("\r", " ").replace("\n", " ")
    for pattern in SECRET_PATTERNS:
        text = pattern.sub(
            lambda match: f"{match.group(1) if match.lastindex else ''}[REDACTED]", text
        )
    return " ".join(text.split())[:500]


def result_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    root = payload.get("results", payload)
    rows = root.get("results") or root.get("table", {}).get("body") or []
    return [row for row in rows if isinstance(row, dict)]


def error_summaries(payload: dict[str, Any]) -> list[str]:
    summaries = []
    for row in result_rows(payload):
        response = row.get("response") if isinstance(row.get("response"), dict) else {}
        error = row.get("error") or response.get("error")
        if error:
            summaries.append(redact(str(error)))
    return list(dict.fromkeys(summaries))[:10]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: summarize_errors.py RESULTS.json")
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    summaries = error_summaries(payload)
    print("▶ eval error summaries (redacted):")
    if not summaries:
        print("  - result schema contained no printable row-level error")
    for summary in summaries:
        print(f"  - {summary}")


if __name__ == "__main__":
    main()
