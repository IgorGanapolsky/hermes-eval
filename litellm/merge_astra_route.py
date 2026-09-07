#!/usr/bin/env python3
"""Insert Astra model list into a LiteLLM config copy (no secret rewrite)."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MARKER = "model_name: gpt-6-astra"


def _list_indent(text: str) -> str:
    for line in text.splitlines():
        if line.lstrip().startswith("- model_name:"):
            return line[: len(line) - len(line.lstrip())]
    return ""


def _align_fragment(frag: str, indent: str) -> str:
    lines = [ln.rstrip() for ln in frag.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    base = None
    for ln in lines:
        if ln.lstrip().startswith("- model_name:"):
            base = ln[: len(ln) - len(ln.lstrip())]
            break
    if base is None:
        base = ""
    aligned: list[str] = []
    for ln in lines:
        if not ln.strip():
            aligned.append("")
            continue
        body = ln[len(base) :] if ln.startswith(base) else ln.lstrip()
        aligned.append(f"{indent}{body}")
    return "\n".join(aligned).rstrip() + "\n\n"


CALLBACK = "astra_budget.proxy_handler_instance"
HERMES_LOGGER = "hermes_logger.proxy_handler_instance"


def _ensure_callback(text: str) -> str:
    if HERMES_LOGGER in text or CALLBACK in text:
        return text
    needle = "litellm_settings:"
    idx = text.find(needle)
    if idx < 0:
        return text
    nl = text.find("\n", idx)
    if nl < 0:
        return text + f"\n  callbacks: {CALLBACK}\n"
    return text[: nl + 1] + f"  callbacks: {CALLBACK}\n" + text[nl + 1 :]


def merge(src: Path, fragment: Path, dest: Path) -> str:
    text = src.read_text()
    frag = _align_fragment(fragment.read_text(), _list_indent(text))
    if MARKER in text:
        dest.write_text(_ensure_callback(text))
        return "already"
    needle = "router_settings:"
    idx = text.find(needle)
    if idx < 0:
        raise SystemExit(f"no {needle} in {src}")
    dest.write_text(_ensure_callback(text[:idx] + frag + text[idx:]))
    return "merged"


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    src = Path(args[0]) if args else HERE / "config.yaml"
    dest = Path(args[1]) if len(args) > 1 else HERE / "config.runtime.yaml"
    frag = HERE / "astra_models.yaml"
    print(merge(src, frag, dest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
