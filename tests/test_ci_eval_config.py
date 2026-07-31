import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = (ROOT / "eval" / "promptfooconfig.ci.yaml").read_text(encoding="utf-8")


def test_ci_smoke_uses_pinned_zero_cost_cross_family_models():
    model_ids = re.findall(r"id: openrouter:([^\s]+)", CONFIG)

    assert len(model_ids) == 2
    assert all(model_id.endswith(":free") for model_id in model_ids)
    assert model_ids[0].split("/", 1)[0] != model_ids[1].split("/", 1)[0]
    assert "openrouter/free" not in model_ids
