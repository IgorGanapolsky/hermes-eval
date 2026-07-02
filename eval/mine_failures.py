"""Mine production traffic for failure-shaped calls and emit golden-set CANDIDATES.

This closes the flywheel loop the README promises ("failures the stream surfaces become
new golden cases"): the LiteLLM proxy logs every served call to traffic.jsonl; this script
extracts the ones that went wrong and shapes them for curation into golden.jsonl.
Like synth_golden.py, output is NOT golden until a human reviews it
(created_by=traffic-miner, needs_review=true). The miner never grades — grading stays
with the judge (optimizer/evaluator decoupling).

Failure classes mined:
  - status_failure : the proxy recorded status != "success" (provider error, timeout)
  - empty_response : call "succeeded" but returned no content (e.g. reasoning ate the
                     max_tokens budget, or qwen3 `think` regression)

Usage: python3 mine_failures.py [--traffic ~/.hermes/litellm-logs/traffic.jsonl]
                                [--out golden.candidates.jsonl] [--max 200]
"""

import argparse
import hashlib
import json
import os


def last_user_message(messages):
    for m in reversed(messages or []):
        if m.get("role") == "user":
            content = m.get("content")
            if isinstance(content, list):  # multimodal: keep text parts
                content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
            return (content or "").strip()
    return ""


def classify(rec):
    if rec.get("status") != "success":
        return "status_failure"
    if not (rec.get("response") or "").strip():
        return "empty_response"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traffic", default=os.path.expanduser("~/.hermes/litellm-logs/traffic.jsonl"))
    ap.add_argument("--out", default="golden.candidates.jsonl")
    ap.add_argument("--max", type=int, default=200)
    args = ap.parse_args()

    seen, candidates, totals = set(), [], {"status_failure": 0, "empty_response": 0, "scanned": 0}
    with open(args.traffic) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            totals["scanned"] += 1
            failure_class = classify(rec)
            if not failure_class:
                continue
            totals[failure_class] += 1
            user_input = last_user_message(rec.get("messages"))
            if not user_input:
                continue
            key = hashlib.sha256(f"{failure_class}:{user_input}".encode()).hexdigest()[:16]
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "id": f"mine-{key}",
                    "input": user_input,
                    "expected_output": "",  # curator fills in the CORRECT answer
                    "observed_output": (rec.get("response") or "")[:2000],
                    "expected_answerable": True,
                    "rationale": f"Mined from production traffic: {failure_class} "
                    f"(model={rec.get('model')}, ts_end={rec.get('ts_end')})",
                    "metadata": {
                        "intent": "regression",
                        "topic": "mined",
                        "difficulty": "unknown",
                        "split": "candidate",
                        "created_by": "traffic-miner",
                        "needs_review": True,
                        "failure_class": failure_class,
                        "source_model": rec.get("model"),
                        "tags": ["mined", failure_class],
                    },
                }
            )

    candidates = candidates[: args.max]
    with open(args.out, "w") as f:
        for c in candidates:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(
        f"scanned={totals['scanned']} status_failure={totals['status_failure']} "
        f"empty_response={totals['empty_response']} unique_candidates={len(candidates)} -> {args.out}"
    )
    print("Candidates need human review before promotion to golden.jsonl (needs_review=true).")


if __name__ == "__main__":
    main()
