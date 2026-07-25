#!/usr/bin/env python3
"""Unit tests for image -> vision-route auto-routing (litellm/hermes_logger.py).

Standalone: `python3 tests/test_route_image_to_vision.py` (no litellm needed — the
logger guards that import).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "litellm"))

from hermes_logger import has_image_parts, route_image_request_to_vision

IMG = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}


def req(model, content):
    return {"model": model, "messages": [{"role": "user", "content": content}]}


def test_image_request_on_text_model_is_rerouted():
    data = route_image_request_to_vision(req("glm-coding", [{"type": "text", "text": "hi"}, IMG]))
    assert data["model"] == "glm-vision", data["model"]


def test_text_only_request_is_untouched():
    data = route_image_request_to_vision(req("glm-coding", [{"type": "text", "text": "hi"}]))
    assert data["model"] == "glm-coding", data["model"]
    # plain-string content is the common shape and must not be scanned as a list
    data = route_image_request_to_vision(req("glm-coding", "just a string"))
    assert data["model"] == "glm-coding", data["model"]


def test_vision_capable_model_is_not_rerouted():
    """A client that already picked a seeing model keeps it — including muse-spark, whose
    1M context is the reason to address it directly."""
    for model in ("glm-vision", "muse-spark", "hermes/glm-vision"):
        data = route_image_request_to_vision(req(model, [IMG]))
        assert data["model"] == model, data["model"]


def test_prefixed_text_group_still_reroutes():
    data = route_image_request_to_vision(req("hermes/glm-coding", [IMG]))
    assert data["model"] == "glm-vision", data["model"]


def test_image_anywhere_in_the_conversation_counts():
    """opencode attaches the screenshot on an earlier turn and keeps chatting; every later
    request in that session still carries the image block."""
    data = {
        "model": "glm-coding",
        "messages": [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": [{"type": "text", "text": "look"}, IMG]},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": [{"type": "text", "text": "now fix it"}]},
        ],
    }
    assert route_image_request_to_vision(data)["model"] == "glm-vision"


def test_alternate_image_part_spellings():
    for part_type in ("image_url", "image", "input_image"):
        assert has_image_parts([{"role": "user", "content": [{"type": part_type}]}]), part_type
    assert not has_image_parts([{"role": "user", "content": [{"type": "text", "text": "x"}]}])
    assert not has_image_parts([])
    assert not has_image_parts(None)


def test_malformed_input_never_raises():
    """The hook wraps this in suppress(), but a guard that throws would still cost the
    request its routing — keep the helper total."""
    assert route_image_request_to_vision(None) is None
    assert route_image_request_to_vision({}) == {}  # no model, no messages
    assert route_image_request_to_vision({"model": "glm-coding"})["model"] == "glm-coding"
    weird = {"model": "glm-coding", "messages": ["not-a-dict", None, {"content": [None, "x"]}]}
    assert route_image_request_to_vision(weird)["model"] == "glm-coding"


def test_override_targets_are_honored():
    data = route_image_request_to_vision(
        req("glm-coding", [IMG]), vision_model="muse-spark", vision_capable={"muse-spark"}
    )
    assert data["model"] == "muse-spark", data["model"]


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if failures else 0)
