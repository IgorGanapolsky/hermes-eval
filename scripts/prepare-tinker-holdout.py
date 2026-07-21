#!/usr/bin/env python3
"""Materialize the deterministic private holdout used for Tinker candidate evaluation."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

from tinker_training_data import (
    DEFAULT_HOLDOUT_RATIO,
    DEFAULT_SPLIT_SEED,
    conversation_digest,
    load_split_conversations,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)


def write_private_jsonl(
    path: Path,
    conversations: list[list[dict]],
    *,
    seed: str,
) -> list[str]:
    ensure_private_directory(path.parent)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    case_ids = []
    with temporary.open("w", encoding="utf-8") as handle:
        for messages in conversations:
            case_id = conversation_digest(messages, seed)
            case_ids.append(case_id)
            handle.write(
                json.dumps({"id": case_id, "messages": messages}, ensure_ascii=False) + "\n"
            )
    temporary.chmod(0o600)
    temporary.replace(path)
    path.chmod(0o600)
    return case_ids


def write_private_json(path: Path, payload: dict) -> None:
    ensure_private_directory(path.parent)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)
    path.chmod(0o600)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--holdout-ratio", type=float, default=DEFAULT_HOLDOUT_RATIO)
    parser.add_argument("--seed", default=DEFAULT_SPLIT_SEED)
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    output = Path(args.out).expanduser().resolve()
    manifest = Path(args.manifest).expanduser().resolve()
    if len({source, output, manifest}) != 3:
        parser.error("source, holdout output, and manifest must be different paths")

    selection = load_split_conversations(
        source,
        split="holdout",
        holdout_ratio=args.holdout_ratio,
        seed=args.seed,
    )
    if not selection.conversations:
        parser.error("deterministic split produced no holdout conversations")
    case_ids = write_private_jsonl(output, selection.conversations, seed=args.seed)
    case_ids_sha256 = hashlib.sha256("\n".join(sorted(case_ids)).encode()).hexdigest()

    payload = {
        "schema": "hermes-tinker/split-manifest-v1",
        "generatedAt": dt.datetime.now(dt.UTC).isoformat(),
        "split": {
            "algorithm": "sha256-threshold-v1",
            "seed": args.seed,
            "holdoutRatio": args.holdout_ratio,
        },
        "source": {
            "sha256": file_sha256(source),
            "rows": selection.scanned_rows,
            "usableRows": selection.usable_rows,
            "duplicateRows": selection.duplicate_rows,
            "trainRows": selection.train_rows,
            "holdoutRows": selection.holdout_rows,
        },
        "holdout": {
            "sha256": file_sha256(output),
            "caseIdsSha256": case_ids_sha256,
            "rows": len(selection.conversations),
            "toolCallTargets": selection.selected_tool_targets,
        },
    }
    write_private_json(manifest, payload)
    print(
        "holdout ready: "
        f"rows={payload['holdout']['rows']} "
        f"tool_targets={payload['holdout']['toolCallTargets']} "
        f"sha256={payload['holdout']['sha256'][:12]}"
    )


if __name__ == "__main__":
    main()
