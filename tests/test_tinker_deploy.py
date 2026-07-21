import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from tinker_deploy import SMOKE_SENTINEL, require_exact_smoke_output


def test_deploy_smoke_requires_exact_sentinel():
    assert require_exact_smoke_output(0, f" {SMOKE_SENTINEL}\n") == SMOKE_SENTINEL


@pytest.mark.parametrize(
    "returncode, output, message",
    [
        (1, SMOKE_SENTINEL, "exited 1"),
        (0, f"thinking...\n{SMOKE_SENTINEL}", "exactly match"),
        (0, f"{SMOKE_SENTINEL} extra", "exactly match"),
        (0, "", "exactly match"),
    ],
)
def test_deploy_smoke_rejects_false_positive_output(returncode, output, message):
    with pytest.raises(RuntimeError, match=message):
        require_exact_smoke_output(returncode, output)
