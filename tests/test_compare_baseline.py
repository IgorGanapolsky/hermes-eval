import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))
import compare_baseline


def _write_results(tmp_path, rows, successes, failures):
    data = {"results": {"results": rows, "stats": {"successes": successes, "failures": failures}}}
    p = tmp_path / "r.json"
    p.write_text(json.dumps(data))
    return str(p)


def test_load_results_parses_rate_and_map(tmp_path):
    rows = [
        {"testCase": {"description": "rag-0001"}, "success": True},
        {"testCase": {"description": "rag-0006"}, "success": False},
    ]
    rate, m = compare_baseline.load_results(_write_results(tmp_path, rows, 1, 1))
    assert rate == 0.5
    assert m["rag-0001"] is True and m["rag-0006"] is False


def test_critical_ids_includes_refusals_and_hard():
    ids = compare_baseline.critical_ids()
    # refusal rows in golden.jsonl are critical
    assert "rag-0006" in ids and "rag-0007" in ids and "rag-0008" in ids
