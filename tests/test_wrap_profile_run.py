import hashlib
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))

from wrap_profile_run import extract_results, load_holdout_ids, validate_result_ids


def test_extract_results_normalizes_promptfoo_rows_and_errors():
    rows, errors = extract_results(
        {
            "results": {
                "results": [
                    {"testCase": {"description": "case-a"}, "success": True},
                    {"testCase": {"description": "case-b"}, "success": False},
                ],
                "stats": {"errors": 2},
            }
        }
    )
    assert rows == [
        {"id": "case-a", "success": True},
        {"id": "case-b", "success": False},
    ]
    assert errors == 2


def test_extract_results_rejects_empty_artifact():
    with pytest.raises(ValueError, match="no identifiable cases"):
        extract_results({"results": {"results": [], "stats": {}}})


@pytest.mark.parametrize(
    "row, message",
    [
        ({"success": False}, "no identifiable case id"),
        ({"description": "case-a", "success": "false"}, "no boolean success value"),
    ],
)
def test_extract_results_rejects_rows_that_could_hide_failures(row, message):
    with pytest.raises(ValueError, match=message):
        extract_results({"results": {"results": [row], "stats": {}}})


def test_extract_results_counts_row_level_provider_errors():
    rows, errors = extract_results(
        {
            "results": {
                "results": [
                    {
                        "description": "case-a",
                        "success": False,
                        "response": {"error": "provider timeout"},
                    }
                ],
                "stats": {"errors": 0},
            }
        }
    )
    assert rows == [{"id": "case-a", "success": False}]
    assert errors == 1


def test_load_holdout_ids_requires_manifest_bound_file(tmp_path):
    holdout = tmp_path / "holdout.jsonl"
    rows = [
        {"id": "case-a", "messages": []},
        {"id": "case-b", "messages": []},
    ]
    holdout.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    digest = hashlib.sha256(holdout.read_bytes()).hexdigest()
    ids_digest = hashlib.sha256(b"case-a\ncase-b").hexdigest()
    manifest = {"holdout": {"sha256": digest, "caseIdsSha256": ids_digest, "rows": 2}}
    assert load_holdout_ids(holdout, manifest) == {"case-a", "case-b"}


def test_load_holdout_ids_rejects_tampering(tmp_path):
    holdout = tmp_path / "holdout.jsonl"
    holdout.write_text(json.dumps({"id": "case-a", "messages": []}) + "\n")
    manifest = {"holdout": {"sha256": "wrong", "caseIdsSha256": "wrong", "rows": 1}}
    with pytest.raises(ValueError, match="manifest digest"):
        load_holdout_ids(holdout, manifest)


def test_validate_result_ids_requires_the_complete_holdout():
    holdout_ids = {f"case-{index:02d}" for index in range(30)}
    subset = sorted(holdout_ids)[:20]

    with pytest.raises(ValueError, match="every deterministic holdout case"):
        validate_result_ids(subset, holdout_ids, min_cases=20)


def test_validate_result_ids_accepts_each_holdout_case_exactly_once():
    holdout_ids = {"case-a", "case-b"}

    validate_result_ids(["case-b", "case-a"], holdout_ids, min_cases=2)


def test_validate_result_ids_rejects_duplicates_before_coverage_check():
    with pytest.raises(ValueError, match="duplicate case ids"):
        validate_result_ids(["case-a", "case-a"], {"case-a"}, min_cases=1)
