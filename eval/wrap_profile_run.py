#!/usr/bin/env python3
"""Bind a promptfoo result to a profile and deterministic holdout digest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from compare_profiles import write_private


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_holdout_ids(path: Path, manifest: dict) -> set[str]:
    if not path.exists() or not path.is_file() or path.is_symlink():
        raise ValueError("holdout must be a regular file")
    holdout = manifest.get("holdout") or {}
    if file_sha256(path) != holdout.get("sha256"):
        raise ValueError("holdout file does not match the split-manifest digest")
    case_ids = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            case_id = row.get("id") if isinstance(row, dict) else None
            if not isinstance(case_id, str) or not case_id:
                raise ValueError(f"holdout line {line_number} is missing its stable id")
            if case_id in case_ids:
                raise ValueError(f"holdout contains duplicate id on line {line_number}")
            case_ids.add(case_id)
    expected_rows = int(holdout.get("rows", 0))
    if len(case_ids) != expected_rows:
        raise ValueError("holdout row count does not match the split manifest")
    ids_digest = hashlib.sha256("\n".join(sorted(case_ids)).encode()).hexdigest()
    if ids_digest != holdout.get("caseIdsSha256"):
        raise ValueError("holdout case ids do not match the split manifest")
    return case_ids


def extract_results(payload: dict) -> tuple[list[dict], int]:
    result_root = payload.get("results", payload)
    rows = result_root.get("results") or result_root.get("table", {}).get("body") or []
    if not isinstance(rows, list) or not rows:
        raise ValueError("promptfoo result contains no identifiable cases")
    normalized = []
    detected_errors = 0
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"promptfoo result row {index} is not an object")
        case_id = (row.get("testCase") or {}).get("description") or row.get("description")
        if not case_id and isinstance(row.get("vars"), dict):
            case_id = row["vars"].get("id") or row["vars"].get("question")
        if not case_id:
            raise ValueError(f"promptfoo result row {index} has no identifiable case id")
        success = row.get("success")
        if not isinstance(success, bool):
            raise ValueError(f"promptfoo result row {index} has no boolean success value")
        response = row.get("response") if isinstance(row.get("response"), dict) else {}
        detected_errors += int(bool(row.get("error") or response.get("error")))
        normalized.append({"id": str(case_id), "success": success})
    stats = result_root.get("stats") or {}
    try:
        errors = int(stats.get("errors", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("promptfoo result has an invalid error count") from exc
    if errors < 0:
        raise ValueError("promptfoo result has a negative error count")
    errors = max(errors, detected_errors)
    return normalized, errors


def validate_result_ids(result_ids: list[str], holdout_ids: set[str], min_cases: int) -> None:
    if len(set(result_ids)) != len(result_ids):
        raise ValueError("profile result contains duplicate case ids")
    if set(result_ids) != holdout_ids:
        raise ValueError("profile result must cover every deterministic holdout case exactly once")
    if min_cases < 1 or len(result_ids) < min_cases:
        raise ValueError(f"profile result must contain at least {max(1, min_cases)} cases")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--holdout", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-cases", type=int, default=20)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).expanduser().read_text(encoding="utf-8"))
    if manifest.get("schema") != "hermes-tinker/split-manifest-v1":
        parser.error("unsupported split manifest")
    holdout = manifest.get("holdout") or {}
    holdout_ids = load_holdout_ids(Path(args.holdout).expanduser().resolve(), manifest)
    results, errors = extract_results(
        json.loads(Path(args.results).expanduser().read_text(encoding="utf-8"))
    )
    result_ids = [row["id"] for row in results]
    try:
        validate_result_ids(result_ids, holdout_ids, args.min_cases)
    except ValueError as exc:
        parser.error(str(exc))
    payload = {
        "schema": "hermes-eval/profile-run-v1",
        "profile": args.profile,
        "dataset": {
            "split": "holdout",
            "sha256": holdout.get("sha256"),
            "evaluatedCases": len(result_ids),
        },
        "results": results,
        "errors": errors,
    }
    write_private(Path(args.out).expanduser().resolve(), payload)
    print(f"profile run wrapped: profile={args.profile} cases={len(results)} errors={errors}")


if __name__ == "__main__":
    main()
