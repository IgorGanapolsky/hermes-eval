"""Regression check against a baseline run.

Fails (exit 1) if the aggregate pass-rate drops more than --tolerance, OR if any
critical golden row (difficulty=hard or a refusal case) flips pass -> fail.

Defensive about promptfoo's results.json shape, which varies across versions.
"""

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def load_results(path):
    with open(path) as fh:
        data = json.load(fh)
    res = data.get("results", data)
    rows = res.get("results") or res.get("table", {}).get("body") or []
    out = {}
    for r in rows:
        desc = (r.get("testCase", {}) or {}).get("description") or r.get("description")
        if not desc and isinstance(r.get("vars"), dict):
            desc = r["vars"].get("question")
        if desc is not None:
            out[desc] = bool(r.get("success", r.get("pass", False)))
    stats = res.get("stats", {})
    p, f = stats.get("successes", 0), stats.get("failures", 0)
    rate = p / (p + f) if (p + f) else 0.0
    return rate, out


def critical_ids():
    ids = set()
    gp = os.path.join(HERE, "golden.jsonl")
    if not os.path.exists(gp):
        return ids
    with open(gp) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            g = json.loads(line)
            m = g.get("metadata", {})
            if m.get("difficulty") == "hard" or not g.get("expected_answerable", True):
                ids.add(g["id"])
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--tolerance", type=float, default=0.05)
    a = ap.parse_args()

    cur_rate, cur = load_results(a.current)
    base_rate, base = load_results(a.baseline)
    print(f"  baseline={base_rate:.4f} current={cur_rate:.4f} tolerance={a.tolerance}")

    ok = True
    if base_rate - cur_rate > a.tolerance:
        print(f"  x aggregate regression: dropped {base_rate - cur_rate:.4f} > {a.tolerance}")
        ok = False
    crit = critical_ids()
    for d, passed_before in base.items():
        if passed_before and d in cur and not cur[d] and d in crit:
            print(f"  x critical regression: '{d}' pass->fail")
            ok = False
    print("  regression:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
