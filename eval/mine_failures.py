"""Mine production traffic for failure-shaped calls and emit golden-set CANDIDATES.

This closes the flywheel loop the README promises ("failures the stream surfaces become
new golden cases"): the LiteLLM proxy logs every served call to traffic.jsonl; this script
extracts the ones that went wrong and shapes them for curation into golden.jsonl.
Like synth_golden.py, output is NOT golden until a human reviews it
(created_by=traffic-miner, needs_review=true). The miner never grades — grading stays
with the judge (optimizer/evaluator decoupling).

Failure classes mined:
  - status_failure  : the proxy recorded status != "success" (provider error, timeout)
  - truncated_empty : succeeded but empty because reasoning hit the length limit
                      (empty_kind="truncated") — the real defect the GLM max_tokens floor targets
  - empty_response  : succeeded but empty for some other/unknown reason (empty_kind="empty",
                      or a legacy record logged before empty_kind existed)

Legitimate tool-call responses (empty content by design, empty_kind="tool_call") are
NOT mined — they are healthy, and flagging them polluted the candidate set with the
tool-calling workhorse's normal output.

Usage: python3 mine_failures.py [--traffic ~/.hermes/litellm-logs/traffic.jsonl]
                                [--out golden.candidates.jsonl] [--max 200]
"""

import argparse
import collections
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
    """Failure class for a traffic record, or None if it's healthy.

    Uses the empty_kind field (emitted since 2026-07-08) to separate a real empty-content
    defect from a legitimate tool-call (empty by design). Records logged before empty_kind
    existed fall back to the conservative 'empty_response' so nothing is silently dropped."""
    if rec.get("status") != "success":
        return "status_failure"
    if (rec.get("response") or "").strip():
        return None  # has content — healthy
    kind = rec.get("empty_kind")
    if kind == "tool_call":
        return None  # empty by design (payload in tool_calls) — NOT a failure
    if kind == "truncated":
        return "truncated_empty"  # reasoning ate the budget — the real defect
    return "empty_response"  # empty_kind == "empty", or a legacy record without the field


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traffic", default=os.path.expanduser("~/.hermes/litellm-logs/traffic.jsonl"))
    ap.add_argument("--out", default="golden.candidates.jsonl")
    ap.add_argument("--max", type=int, default=200)
    args = ap.parse_args()

    seen, candidates, totals = set(), [], collections.defaultdict(int)
    with open(args.traffic) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            totals["scanned"] += 1
            failure_class = classify(rec)
            if not failure_class:
                # count healthy tool-call empties that we correctly no longer flag
                if not (rec.get("response") or "").strip() and rec.get("empty_kind") == "tool_call":
                    totals["tool_call_excluded"] += 1
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
        f"truncated_empty={totals['truncated_empty']} empty_response={totals['empty_response']} "
        f"| tool_call_excluded={totals['tool_call_excluded']} (healthy, not mined) "
        f"unique_candidates={len(candidates)} -> {args.out}"
    )
    print("Candidates need human review before promotion to golden.jsonl (needs_review=true).")


if __name__ == "__main__":
    main()
