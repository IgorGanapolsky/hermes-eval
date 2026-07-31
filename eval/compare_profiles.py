#!/usr/bin/env python3
"""Create the evidence receipt consumed by ``tinker-yolo doctor``.

Inputs are repeated, held-out profile-run artifacts produced from the same split
manifest. The comparison rejects missing cases, provider errors, aggregate regression,
any per-case regression, weak absolute quality, or a candidate that did not improve.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BASELINE_PROFILE = "hermes-local-baseline"
CANDIDATE_PROFILE = "inkling-tinker-candidate"


class ComparisonError(ValueError):
    """Raised when evaluation evidence is incomplete or internally inconsistent."""


@dataclass(frozen=True)
class ProfileRun:
    path: Path
    profile: str
    dataset_sha256: str
    outcomes: dict[str, bool]
    errors: int


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "hermes-tinker/split-manifest-v1":
        raise ComparisonError("unsupported split manifest schema")
    holdout = payload.get("holdout") or {}
    if not isinstance(holdout.get("sha256"), str) or not holdout.get("sha256"):
        raise ComparisonError("split manifest is missing the holdout digest")
    if int(holdout.get("rows", 0)) < 1:
        raise ComparisonError("split manifest contains no holdout rows")
    return payload


def load_profile_run(path: Path) -> ProfileRun:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "hermes-eval/profile-run-v1":
        raise ComparisonError(f"{path.name}: unsupported profile-run schema")
    profile = payload.get("profile")
    dataset = payload.get("dataset") or {}
    if not isinstance(profile, str) or not profile:
        raise ComparisonError(f"{path.name}: profile is required")
    if dataset.get("split") != "holdout" or not isinstance(dataset.get("sha256"), str):
        raise ComparisonError(f"{path.name}: a held-out dataset digest is required")

    outcomes: dict[str, bool] = {}
    rows = payload.get("results")
    if not isinstance(rows, list) or not rows:
        raise ComparisonError(f"{path.name}: results must be a non-empty list")
    for row in rows:
        if not isinstance(row, dict):
            raise ComparisonError(f"{path.name}: result rows must be objects")
        case_id = row.get("id")
        success = row.get("success")
        if not isinstance(case_id, str) or not case_id or not isinstance(success, bool):
            raise ComparisonError(f"{path.name}: each result needs id and boolean success")
        if case_id in outcomes:
            raise ComparisonError(f"{path.name}: duplicate result id {case_id}")
        outcomes[case_id] = success
    errors = payload.get("errors", 0)
    if not isinstance(errors, int) or errors < 0:
        raise ComparisonError(f"{path.name}: errors must be a non-negative integer")
    return ProfileRun(path, profile, dataset["sha256"], outcomes, errors)


def aggregate(runs: list[ProfileRun]) -> tuple[float, dict[str, float], int]:
    case_ids = set(runs[0].outcomes)
    if any(set(run.outcomes) != case_ids for run in runs[1:]):
        raise ComparisonError("profile repeats do not contain identical case sets")
    per_case = {
        case_id: sum(run.outcomes[case_id] for run in runs) / len(runs)
        for case_id in sorted(case_ids)
    }
    pass_rate = sum(per_case.values()) / len(per_case)
    return pass_rate, per_case, sum(run.errors for run in runs)


def compare(
    baseline_runs: list[ProfileRun],
    candidate_runs: list[ProfileRun],
    manifest: dict[str, Any],
    *,
    min_repeats: int,
    tolerance: float,
    min_candidate_rate: float,
    min_improvement: float,
) -> dict[str, Any]:
    if not baseline_runs or not candidate_runs:
        raise ComparisonError("baseline and candidate runs are required")
    baseline_profile = baseline_runs[0].profile
    candidate_profile = candidate_runs[0].profile
    if any(run.profile != baseline_profile for run in baseline_runs):
        raise ComparisonError("baseline run profiles do not match")
    if any(run.profile != candidate_profile for run in candidate_runs):
        raise ComparisonError("candidate run profiles do not match")
    if baseline_profile != BASELINE_PROFILE or candidate_profile != CANDIDATE_PROFILE:
        raise ComparisonError("profile names do not match the tinker-yolo adoption contract")

    holdout = manifest["holdout"]
    expected_digest = holdout["sha256"]
    all_runs = [*baseline_runs, *candidate_runs]
    if any(run.dataset_sha256 != expected_digest for run in all_runs):
        raise ComparisonError("profile runs do not match the split-manifest holdout digest")

    baseline_rate, baseline_cases, baseline_errors = aggregate(baseline_runs)
    candidate_rate, candidate_cases, candidate_errors = aggregate(candidate_runs)
    case_sets_match = set(baseline_cases) == set(candidate_cases)
    per_case_regressions = []
    if case_sets_match:
        per_case_regressions = [
            case_id
            for case_id in sorted(baseline_cases)
            if candidate_cases[case_id] + tolerance < baseline_cases[case_id]
        ]

    enough_repeats = len(baseline_runs) >= min_repeats and len(candidate_runs) >= min_repeats
    no_provider_errors = baseline_errors == 0 and candidate_errors == 0
    holdout_no_regression = (
        case_sets_match and no_provider_errors and candidate_rate + tolerance >= baseline_rate
    )
    no_regressions = case_sets_match and no_provider_errors and not per_case_regressions
    candidate_quality = candidate_rate >= min_candidate_rate
    candidate_improved = candidate_rate >= baseline_rate + min_improvement
    gates = {
        "enoughRepeats": enough_repeats,
        "holdoutNoRegression": holdout_no_regression,
        "noRegressions": no_regressions,
        "caseSetsMatch": case_sets_match,
        "noProviderErrors": no_provider_errors,
        "candidateQuality": candidate_quality,
        "candidateImproved": candidate_improved,
    }
    status = "adopt" if all(gates.values()) else "reject"

    return {
        "schema": "tinker-yolo/profile-comparison-v1",
        "generatedAt": dt.datetime.now(dt.UTC).isoformat(),
        "profileComparison": {
            "baselineProfile": baseline_profile,
            "candidateProfile": candidate_profile,
            "status": status,
            "dataset": {
                "split": "holdout",
                "sha256": expected_digest,
                "rows": holdout["rows"],
                "evaluatedCases": len(baseline_cases) if case_sets_match else 0,
            },
            "metrics": {
                "baselinePassRate": round(baseline_rate, 6),
                "candidatePassRate": round(candidate_rate, 6),
                "delta": round(candidate_rate - baseline_rate, 6),
                "baselineErrors": baseline_errors,
                "candidateErrors": candidate_errors,
                "baselineRepeats": len(baseline_runs),
                "candidateRepeats": len(candidate_runs),
                "regressedCases": per_case_regressions,
            },
            "thresholds": {
                "minRepeats": min_repeats,
                "tolerance": tolerance,
                "minCandidateRate": min_candidate_rate,
                "minImprovement": min_improvement,
            },
            "gates": gates,
        },
        "evidence": {
            "manifestSha256": sha256(Path(manifest["_path"])),
            "baselineRuns": [
                {"file": run.path.name, "sha256": sha256(run.path)} for run in baseline_runs
            ],
            "candidateRuns": [
                {"file": run.path.name, "sha256": sha256(run.path)} for run in candidate_runs
            ],
        },
    }


def write_private(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)
    path.chmod(0o600)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--baseline", action="append", required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-repeats", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=0.0)
    parser.add_argument("--min-candidate-rate", type=float, default=0.85)
    parser.add_argument("--min-improvement", type=float, default=0.01)
    args = parser.parse_args()
    if args.min_repeats < 1:
        parser.error("min repeats must be positive")
    for value, name in (
        (args.tolerance, "tolerance"),
        (args.min_candidate_rate, "min candidate rate"),
        (args.min_improvement, "min improvement"),
    ):
        if not 0 <= value <= 1:
            parser.error(f"{name} must be between 0 and 1")

    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = load_manifest(manifest_path)
    manifest["_path"] = str(manifest_path)
    receipt = compare(
        [load_profile_run(Path(path).expanduser().resolve()) for path in args.baseline],
        [load_profile_run(Path(path).expanduser().resolve()) for path in args.candidate],
        manifest,
        min_repeats=args.min_repeats,
        tolerance=args.tolerance,
        min_candidate_rate=args.min_candidate_rate,
        min_improvement=args.min_improvement,
    )
    output = Path(args.out).expanduser().resolve()
    write_private(output, receipt)
    comparison = receipt["profileComparison"]
    print(
        f"profile comparison: status={comparison['status']} "
        f"baseline={comparison['metrics']['baselinePassRate']:.3f} "
        f"candidate={comparison['metrics']['candidatePassRate']:.3f}"
    )
    raise SystemExit(0 if comparison["status"] == "adopt" else 1)


if __name__ == "__main__":
    main()
