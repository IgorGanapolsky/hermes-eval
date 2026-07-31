import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from tinker_training_data import (
    TrainingDataError,
    is_usable_conversation,
    load_split_conversations,
    normalize_messages,
    normalize_tool_call,
    render_with_context_limit,
    split_for_messages,
    to_renderer_messages,
)


def conversation(index, *, tool_call=False):
    target = {"role": "assistant", "content": f"answer {index}"}
    if tool_call:
        target = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"name": "lookup", "arguments": '{"id": 7}'}],
        }
    return [{"role": "user", "content": f"question {index}"}, target]


def test_normalize_tool_call_accepts_flat_and_openai_shapes():
    flat = normalize_tool_call({"name": "lookup", "arguments": '{"b":2,"a":1}'})
    nested = normalize_tool_call(
        {
            "id": "call-1",
            "type": "function",
            "function": {"name": "lookup", "arguments": {"a": 1, "b": 2}},
        }
    )
    assert flat["function"] == nested["function"]
    assert flat["function"]["arguments"] == '{"a":1,"b":2}'
    assert flat["id"] is None
    assert nested["id"] == "call-1"


def test_normalize_tool_call_rejects_non_object_arguments():
    with pytest.raises(TrainingDataError, match="decode to an object"):
        normalize_tool_call({"name": "lookup", "arguments": "[1, 2]"})


def test_renderer_conversion_preserves_tool_metadata():
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"name": "lookup", "arguments": "{}"}],
        },
        {"role": "tool", "content": "ok", "tool_call_id": "call-1", "name": "lookup"},
        {"role": "assistant", "content": "done"},
    ]
    converted = to_renderer_messages(messages, lambda payload: ("validated", payload))
    assert converted[0]["tool_calls"][0][0] == "validated"
    assert converted[0]["tool_calls"][0][1]["function"]["name"] == "lookup"
    assert converted[1]["tool_call_id"] == "call-1"
    assert converted[1]["name"] == "lookup"


def test_normalize_messages_does_not_mutate_source():
    source = conversation(1, tool_call=True)
    normalized = normalize_messages(source)
    normalized[-1]["tool_calls"][0]["function"]["name"] = "changed"
    assert source[-1]["tool_calls"][0]["name"] == "lookup"


def test_tool_only_target_is_usable():
    assert is_usable_conversation(conversation(1, tool_call=True))


def test_split_is_stable_and_keeps_holdout_out_of_training(tmp_path):
    rows = [conversation(index, tool_call=index % 3 == 0) for index in range(200)]
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        "".join(json.dumps({"messages": messages}) + "\n" for messages in rows),
        encoding="utf-8",
    )
    train = load_split_conversations(dataset, split="train", holdout_ratio=0.1)
    holdout = load_split_conversations(dataset, split="holdout", holdout_ratio=0.1)

    assert train.usable_rows == holdout.usable_rows == 200
    assert train.train_rows == len(train.conversations)
    assert holdout.holdout_rows == len(holdout.conversations)
    assert train.train_rows + holdout.holdout_rows == 200
    assert train.duplicate_rows == holdout.duplicate_rows == 0
    assert 5 <= holdout.holdout_rows <= 40

    train_keys = {json.dumps(messages, sort_keys=True) for messages in train.conversations}
    holdout_keys = {json.dumps(messages, sort_keys=True) for messages in holdout.conversations}
    assert train_keys.isdisjoint(holdout_keys)
    assert all(split_for_messages(messages, 0.1) == "holdout" for messages in holdout.conversations)


def test_limit_does_not_change_complete_split_counts(tmp_path):
    rows = [conversation(index) for index in range(50)]
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        "".join(json.dumps({"messages": messages}) + "\n" for messages in rows),
        encoding="utf-8",
    )
    selection = load_split_conversations(dataset, split="train", limit=3)
    assert len(selection.conversations) == 3
    assert selection.train_rows + selection.holdout_rows == 50


def test_duplicate_conversations_are_not_overweighted_or_leaked(tmp_path):
    repeated = conversation(1, tool_call=True)
    rows = [repeated, conversation(2), repeated, repeated]
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        "".join(json.dumps({"messages": messages}) + "\n" for messages in rows),
        encoding="utf-8",
    )
    train = load_split_conversations(dataset, split="train")
    holdout = load_split_conversations(dataset, split="holdout")
    assert train.usable_rows == holdout.usable_rows == 2
    assert train.duplicate_rows == holdout.duplicate_rows == 2
    assert train.train_rows + holdout.holdout_rows == 2


class FakeModelInput:
    def __init__(self, size):
        self.size = size

    def to_ints(self):
        return list(range(self.size))


class FakeRenderer:
    def build_supervised_example(self, messages, *, train_on_what):
        size = sum(len(message.get("content", "")) for message in messages)
        return FakeModelInput(size), [1] * size


def test_context_limit_prunes_only_at_user_turn_boundary():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "old-user"},
        {"role": "assistant", "content": "", "tool_calls": [{"validated": True}]},
        {"role": "tool", "content": "old-tool-result"},
        {"role": "user", "content": "latest-user"},
        {"role": "assistant", "content": "target"},
    ]
    rendered = render_with_context_limit(
        FakeRenderer(),
        messages,
        train_on_what="last-assistant",
        max_tokens=20,
    )
    assert len(rendered.model_input.to_ints()) == len("syslatest-usertarget")
    assert rendered.dropped_messages == 3


def test_context_limit_rejects_oversized_latest_turn():
    messages = [
        {"role": "user", "content": "x" * 20},
        {"role": "assistant", "content": "y" * 20},
    ]
    with pytest.raises(TrainingDataError, match="latest user-to-assistant"):
        render_with_context_limit(
            FakeRenderer(),
            messages,
            train_on_what="last-assistant",
            max_tokens=10,
        )


def test_context_limit_drops_complete_old_tool_cycles_inside_one_user_turn():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "user"},
        {"role": "assistant", "content": "old-call"},
        {"role": "tool", "content": "old-result"},
        {"role": "assistant", "content": "new-call"},
        {"role": "tool", "content": "new-result"},
        {"role": "assistant", "content": "target"},
    ]
    expected_tokens = len("sysusernew-callnew-resulttarget")
    rendered = render_with_context_limit(
        FakeRenderer(),
        messages,
        train_on_what="last-assistant",
        max_tokens=expected_tokens,
    )
    assert len(rendered.model_input.to_ints()) == expected_tokens
    assert rendered.dropped_messages == 2
