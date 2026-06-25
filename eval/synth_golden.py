"""Bootstrap synthetic golden CANDIDATES from a docs corpus (cold-start, before real
traffic exists). Output requires human review (created_by=llm, needs_review=true) — it is
NOT golden until a human curates it. DIY (Hamel-style) so the generation logic is
transparent; swap for RAGAS TestsetGenerator / DeepEval Synthesizer at scale.

Usage: python3 synth_golden.py --docs corpus --out golden.candidates.jsonl
Env: LITELLM_BASE_URL, LITELLM_MASTER_KEY, SYNTH_MODEL (default hermes-local)
"""
import argparse
import glob
import json
import os
import re
import urllib.request

BASE = os.environ.get("LITELLM_BASE_URL", "http://127.0.0.1:4010/v1")
KEY = os.environ.get("LITELLM_MASTER_KEY", "sk-hermes-local-dev")
MODEL = os.environ.get("SYNTH_MODEL", "hermes-local")


def call(prompt):
    body = json.dumps({"model": MODEL, "temperature": 0.3,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(BASE + "/chat/completions", data=body,
                                 headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def paragraphs(text):
    return [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip()) > 40]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    n = 0
    with open(a.out, "w", encoding="utf-8") as out:
        for path in sorted(glob.glob(os.path.join(a.docs, "**", "*"), recursive=True)):
            if not os.path.isfile(path) or not path.endswith((".md", ".txt")):
                continue
            for i, ch in enumerate(paragraphs(open(path, encoding="utf-8").read())):
                prompt = ('From the CONTEXT below, write ONE factual question a user might ask and the '
                          'exact answer grounded in the context. Return strict JSON only: '
                          '{"question": "...", "answer": "..."}\n\nCONTEXT:\n' + ch)
                try:
                    raw = call(prompt)
                    m = re.search(r"\{.*\}", raw, re.S)
                    obj = json.loads(m.group(0)) if m else None
                except Exception as e:
                    print("  skip", path, i, ":", e)
                    continue
                if not obj or "question" not in obj:
                    continue
                row = {"id": f"synth-{os.path.basename(path)}-{i}",
                       "input": obj["question"], "expected_output": obj.get("answer", ""),
                       "expected_contexts": [{"doc_id": f"{os.path.basename(path)}#{i}", "text": ch}],
                       "expected_answerable": True,
                       "metadata": {"intent": "factual", "source": "synthetic_unreviewed",
                                    "created_by": "llm", "needs_review": True, "tags": []}}
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
                print("  +", row["id"], "::", obj["question"][:70])
    print(f"\nwrote {n} candidates -> {a.out}  (REVIEW before promoting to golden.jsonl)")


if __name__ == "__main__":
    main()
