import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))
import load_golden


def _reload(**env):
    for k in ("EVAL_SUBSET", "EVAL_NO_EMBED"):
        os.environ.pop(k, None)
    for k, v in env.items():
        os.environ[k] = v
    return load_golden.make_tests()


def test_full_set_is_eight():
    tests = _reload()
    assert len(tests) == 8
    # every test carries both `question` (for the prompt) and `query` (for RAG asserts)
    for t in tests:
        assert "question" in t["vars"] and "query" in t["vars"]


def test_answerable_have_faithfulness_and_similar():
    tests = {t["description"]: t for t in _reload()}
    kinds = [a["type"] for a in tests["rag-0001"]["assert"]]
    assert "llm-rubric" in kinds and "context-faithfulness" in kinds and "similar" in kinds


def test_refusal_rows_have_no_similar():
    tests = {t["description"]: t for t in _reload()}
    kinds = [a["type"] for a in tests["rag-0006"]["assert"]]
    assert "similar" not in kinds and "context-faithfulness" not in kinds


def test_ci_smoke_subset_is_three():
    assert len(_reload(EVAL_SUBSET="ci-smoke")) == 3


def test_no_embed_drops_similar():
    tests = _reload(EVAL_NO_EMBED="1")
    for t in tests:
        assert all(a["type"] != "similar" for a in t["assert"])
