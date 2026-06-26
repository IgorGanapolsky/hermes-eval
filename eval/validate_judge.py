"""Validate the LLM-as-judge against human labels BEFORE trusting the gate.

Computes raw agreement, Cohen's kappa, and TPR/TNR. The judge is itself a model;
if kappa is low, fix the rubric or judge model before relying on eval scores.
(Who validates the validators? — Shankar et al., UIST 2024.)

Usage: python3 validate_judge.py --labels judge_labels.example.jsonl
Each line: {"id","question","context","candidate_answer","human_verdict":"PASS"|"FAIL"}
Env: LITELLM_BASE_URL (default http://127.0.0.1:4000/v1), LITELLM_MASTER_KEY, JUDGE_MODEL (default hermes-gemma)
"""

import argparse
import json
import os
import re
import urllib.request
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("LITELLM_BASE_URL", "http://127.0.0.1:4010/v1")
KEY = os.environ.get("LITELLM_MASTER_KEY", "sk-hermes-local-dev")
MODEL = os.environ.get("JUDGE_MODEL", "hermes-gemma")
RUBRIC = Path(HERE, "rubric.txt").read_text(encoding="utf-8")


def judge(q, ctx, ans):
    prompt = (
        f"{RUBRIC}\n\nQUESTION:\n{q}\n\nCONTEXT:\n{ctx}\n\nCANDIDATE_ANSWER:\n{ans}\n\n"
        "Respond with exactly one word on the last line: PASS or FAIL."
    )
    body = json.dumps(
        {"model": MODEL, "temperature": 0, "messages": [{"role": "user", "content": prompt}]}
    ).encode()
    req = urllib.request.Request(
        BASE + "/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        txt = json.load(r)["choices"][0]["message"]["content"]
    found = re.findall(r"\b(PASS|FAIL)\b", txt.upper())
    return found[-1] if found else "FAIL"


def cohen_kappa(a, b):
    n = len(a)
    po = sum(1 for x, y in zip(a, b, strict=False) if x == y) / n
    pe = sum((a.count(lbl) / n) * (b.count(lbl) / n) for lbl in ("PASS", "FAIL"))
    k = (po - pe) / (1 - pe) if (1 - pe) else 1.0
    return k, po


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    a = ap.parse_args()
    human, model = [], []
    with open(a.labels, encoding="utf-8") as fh:
        lines = fh.readlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        hv = r["human_verdict"].upper()
        mv = judge(r["question"], r.get("context", ""), r["candidate_answer"])
        human.append(hv)
        model.append(mv)
        print(f"  {r['id']}: human={hv} judge={mv} {'ok' if hv == mv else 'MISMATCH'}")
    k, po = cohen_kappa(human, model)
    tp = sum(1 for h, m in zip(human, model, strict=False) if h == "PASS" and m == "PASS")
    fn = sum(1 for h, m in zip(human, model, strict=False) if h == "PASS" and m == "FAIL")
    tn = sum(1 for h, m in zip(human, model, strict=False) if h == "FAIL" and m == "FAIL")
    fp = sum(1 for h, m in zip(human, model, strict=False) if h == "FAIL" and m == "PASS")
    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    tnr = tn / (tn + fp) if (tn + fp) else float("nan")
    print(f"\n  n={len(human)} agreement={po:.3f} cohen_kappa={k:.3f} TPR={tpr:.3f} TNR={tnr:.3f}")
    print(
        "  "
        + (
            "OK: judge usable (kappa>=0.6)"
            if k >= 0.6
            else "WARN: judge NOT trustworthy (kappa<0.6) — fix rubric/model"
        )
    )


if __name__ == "__main__":
    main()
