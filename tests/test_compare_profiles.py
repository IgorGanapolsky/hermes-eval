import json
import os
import stat
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))

from compare_profiles import (
    BASELINE_PROFILE,
    CANDIDATE_PROFILE,
    ComparisonError,
    compare,
    load_manifest,
    load_profile_run,
    write_private,
)


def write_manifest(tmp_path, digest="holdout-digest"):
    path = tmp_path / "manifest.json"
    payload = {
        "schema": "hermes-tinker/split-manifest-v1",
        "holdout": {"sha256": digest, "rows": 20},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_manifest(path)
    loaded["_path"] = str(path)
    return loaded


def write_run(tmp_path, name, profile, outcomes, *, digest="holdout-digest", errors=0):
    path = tmp_path / f"{name}.json"
    path.write_text(
        json.dumps(
            {
                "schema": "hermes-eval/profile-run-v1",
                "profile": profile,
                "dataset": {"split": "holdout", "sha256": digest},
                "results": [
                    {"id": case_id, "success": success} for case_id, success in outcomes.items()
                ],
                "errors": errors,
            }
        ),
        encoding="utf-8",
    )
    return load_profile_run(path)


def run_comparison(tmp_path, baseline_outcomes, candidate_outcomes, *, repeats=3, **kwargs):
    manifest = write_manifest(tmp_path)
    baseline = [
        write_run(tmp_path, f"baseline-{index}", BASELINE_PROFILE, baseline_outcomes)
        for index in range(repeats)
    ]
    candidate = [
        write_run(tmp_path, f"candidate-{index}", CANDIDATE_PROFILE, candidate_outcomes)
        for index in range(repeats)
    ]
    return compare(
        baseline,
        candidate,
        manifest,
        min_repeats=3,
        tolerance=0.0,
        min_candidate_rate=0.75,
        min_improvement=0.01,
        **kwargs,
    )


def test_adopts_only_repeated_improving_candidate(tmp_path):
    receipt = run_comparison(
        tmp_path,
        {"a": True, "b": True, "c": True, "d": False},
        {"a": True, "b": True, "c": True, "d": True},
    )
    comparison = receipt["profileComparison"]
    assert comparison["status"] == "adopt"
    assert all(comparison["gates"].values())
    assert comparison["metrics"]["delta"] == 0.25


def test_rejects_per_case_regression_even_when_aggregate_improves(tmp_path):
    receipt = run_comparison(
        tmp_path,
        {"a": True, "b": False, "c": False},
        {"a": False, "b": True, "c": True},
    )
    comparison = receipt["profileComparison"]
    assert comparison["status"] == "reject"
    assert comparison["gates"]["holdoutNoRegression"] is True
    assert comparison["gates"]["noRegressions"] is False
    assert comparison["metrics"]["regressedCases"] == ["a"]


def test_rejects_insufficient_repeats(tmp_path):
    receipt = run_comparison(
        tmp_path,
        {"a": False, "b": True},
        {"a": True, "b": True},
        repeats=2,
    )
    assert receipt["profileComparison"]["status"] == "reject"
    assert receipt["profileComparison"]["gates"]["enoughRepeats"] is False


def test_rejects_provider_errors(tmp_path):
    manifest = write_manifest(tmp_path)
    baseline = [
        write_run(tmp_path, f"baseline-{index}", BASELINE_PROFILE, {"a": True})
        for index in range(3)
    ]
    candidate = [
        write_run(
            tmp_path,
            f"candidate-{index}",
            CANDIDATE_PROFILE,
            {"a": True},
            errors=1 if index == 0 else 0,
        )
        for index in range(3)
    ]
    receipt = compare(
        baseline,
        candidate,
        manifest,
        min_repeats=3,
        tolerance=0,
        min_candidate_rate=0.5,
        min_improvement=0,
    )
    assert receipt["profileComparison"]["status"] == "reject"
    assert receipt["profileComparison"]["gates"]["noProviderErrors"] is False


def test_requires_manifest_digest_match(tmp_path):
    manifest = write_manifest(tmp_path)
    baseline = [write_run(tmp_path, "baseline", BASELINE_PROFILE, {"a": True})]
    candidate = [
        write_run(
            tmp_path,
            "candidate",
            CANDIDATE_PROFILE,
            {"a": True},
            digest="different",
        )
    ]
    with pytest.raises(ComparisonError, match="holdout digest"):
        compare(
            baseline,
            candidate,
            manifest,
            min_repeats=1,
            tolerance=0,
            min_candidate_rate=0,
            min_improvement=0,
        )


def test_private_receipt_permissions(tmp_path):
    path = tmp_path / "private" / "receipt.json"
    write_private(path, {"ok": True})
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
