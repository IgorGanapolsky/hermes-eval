"""Map golden.jsonl -> promptfoo test cases.

Referenced by promptfooconfig*.yaml as:  tests: file://load_golden.py:make_tests

Env:
  EVAL_SUBSET    if set, only rows whose metadata.tags contains this value (e.g. "ci-smoke")
  EVAL_NO_EMBED  if "1", drop the `similar` (embedding) assertion (for CI with no embed model)
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def _rubric():
    with open(os.path.join(HERE, "rubric.txt"), encoding="utf-8") as f:
        return f.read()


def make_tests():
    subset = os.environ.get("EVAL_SUBSET")
    no_embed = os.environ.get("EVAL_NO_EMBED") == "1"
    rubric = _rubric()
    tests = []
    with open(os.path.join(HERE, "golden.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            g = json.loads(line)
            meta = g.get("metadata", {})
            if subset and subset not in meta.get("tags", []):
                continue
            ctx = "\n\n".join(c["text"] for c in g.get("expected_contexts", []))
            asserts = [{"type": "llm-rubric", "value": rubric}]
            if g.get("expected_answerable", True):
                asserts.append({"type": "context-faithfulness", "threshold": 0.6})
                if g.get("expected_output") and not no_embed:
                    asserts.append(
                        {"type": "similar", "value": g["expected_output"], "threshold": 0.55}
                    )
            else:
                asserts.append(
                    {
                        "type": "llm-rubric",
                        "value": (
                            "PASS only if the answer declines or says it lacks the information. "
                            "FAIL if it states any specific fact as the answer."
                        ),
                    }
                )
            tests.append(
                {
                    "description": g["id"],
                    # `query` is required by promptfoo's RAG assertions (context-faithfulness);
                    # `question` is what the prompt template references. Keep both.
                    "vars": {
                        "question": g["input"],
                        "query": g["input"],
                        "context": ctx,
                        "reference": g.get("expected_output", ""),
                        # the "why" behind the correct answer — calibrates the judge
                        # ("teach your AI how you make decisions", HBR 2026-06-25)
                        "rationale": g.get("rationale", ""),
                    },
                    "assert": asserts,
                }
            )
    return tests


if __name__ == "__main__":
    t = make_tests()
    print(
        f"{len(t)} tests (EVAL_SUBSET={os.environ.get('EVAL_SUBSET')}, EVAL_NO_EMBED={os.environ.get('EVAL_NO_EMBED')})"
    )
    for x in t:
        print(" -", x["description"], "| asserts:", [a["type"] for a in x["assert"]])
