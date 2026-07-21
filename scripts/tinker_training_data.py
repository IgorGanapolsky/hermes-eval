"""Shared, provider-free data preparation for Hermes Tinker training.

The traffic logger stores two tool-call shapes:

* historical OpenAI calls: ``{type, id, function: {name, arguments}}``
* final teacher targets: ``{name, arguments}``

Tinker Cookbook renderers require the first shape with a validated ``ToolCall``.
This module normalizes both shapes without importing Tinker, so the safety and split
logic remains cheap to test in CI.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_HOLDOUT_RATIO = 0.10
DEFAULT_SPLIT_SEED = "hermes-tinker-holdout-v1"


class TrainingDataError(ValueError):
    """Raised when a conversation cannot be represented safely for training."""


@dataclass(frozen=True)
class SplitSelection:
    """Selected conversations plus complete split statistics for the source file."""

    conversations: list[list[dict[str, Any]]]
    scanned_rows: int
    usable_rows: int
    duplicate_rows: int
    train_rows: int
    holdout_rows: int
    selected_tool_targets: int


@dataclass(frozen=True)
class RenderedTrainingExample:
    """A renderer result plus auditable context-pruning metadata."""

    model_input: Any
    weights: Any
    dropped_messages: int


def validate_holdout_ratio(value: float) -> float:
    """Require a meaningful holdout while keeping most examples available for SFT."""

    ratio = float(value)
    if not 0.01 <= ratio <= 0.50:
        raise TrainingDataError("holdout ratio must be between 0.01 and 0.50")
    return ratio


def conversation_digest(messages: list[dict[str, Any]], seed: str = DEFAULT_SPLIT_SEED) -> str:
    """Return a stable split key without exposing conversation text."""

    if not isinstance(messages, list) or len(messages) < 2:
        raise TrainingDataError("conversation must contain at least two messages")
    encoded = json.dumps(
        messages,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(seed.encode() + b"\0" + encoded).hexdigest()


def split_for_messages(
    messages: list[dict[str, Any]],
    holdout_ratio: float = DEFAULT_HOLDOUT_RATIO,
    seed: str = DEFAULT_SPLIT_SEED,
) -> str:
    """Assign a conversation deterministically to ``train`` or ``holdout``."""

    ratio = validate_holdout_ratio(holdout_ratio)
    bucket = int(conversation_digest(messages, seed)[:16], 16) / float(1 << 64)
    return "holdout" if bucket < ratio else "train"


def normalize_tool_call(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize logger tool calls into the Cookbook/OpenAI function-call schema."""

    if not isinstance(raw, dict):
        raise TrainingDataError("tool call must be an object")
    function = raw.get("function") if isinstance(raw.get("function"), dict) else raw
    name = function.get("name")
    if not isinstance(name, str) or not name:
        raise TrainingDataError("tool call requires a non-empty function name")

    arguments = function.get("arguments")
    if isinstance(arguments, dict):
        parsed_arguments = arguments
    elif isinstance(arguments, str):
        try:
            parsed_arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise TrainingDataError("tool-call arguments must be valid JSON") from exc
    else:
        raise TrainingDataError("tool-call arguments must be a JSON object or string")
    if not isinstance(parsed_arguments, dict):
        raise TrainingDataError("tool-call arguments must decode to an object")

    call_type = raw.get("type", "function")
    if call_type != "function":
        raise TrainingDataError("only function tool calls are supported")
    call_id = raw.get("id")
    if call_id is not None and not isinstance(call_id, str):
        raise TrainingDataError("tool-call id must be a string when present")

    return {
        "type": "function",
        "id": call_id,
        "function": {
            "name": name,
            "arguments": json.dumps(
                parsed_arguments,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        },
    }


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Preserve text, tool calls, and tool-result correlation fields for rendering."""

    if not isinstance(messages, list) or len(messages) < 2:
        raise TrainingDataError("conversation must contain at least two messages")
    normalized = []
    for message in messages:
        if not isinstance(message, dict):
            raise TrainingDataError("message must be an object")
        role = message.get("role")
        if not isinstance(role, str) or not role:
            raise TrainingDataError("message requires a role")
        content = message.get("content")
        if content is None:
            content = ""
        if not isinstance(content, (str, list)):
            raise TrainingDataError("message content must be text or structured parts")
        item: dict[str, Any] = {"role": role, "content": content}
        if message.get("tool_calls"):
            item["tool_calls"] = [normalize_tool_call(call) for call in message["tool_calls"]]
        for field in ("tool_call_id", "name"):
            value = message.get(field)
            if value is not None:
                if not isinstance(value, str):
                    raise TrainingDataError(f"{field} must be a string when present")
                item[field] = value
        if "trainable" in message:
            if not isinstance(message["trainable"], bool):
                raise TrainingDataError("trainable must be boolean when present")
            item["trainable"] = message["trainable"]
        normalized.append(item)
    return normalized


def to_renderer_messages(
    messages: list[dict[str, Any]],
    tool_call_factory: Callable[[dict[str, Any]], Any],
) -> list[dict[str, Any]]:
    """Convert canonical tool-call dictionaries into renderer-specific objects."""

    normalized = normalize_messages(messages)
    for message in normalized:
        if "tool_calls" in message:
            message["tool_calls"] = [tool_call_factory(call) for call in message["tool_calls"]]
    return normalized


def render_with_context_limit(
    renderer: Any,
    messages: list[dict[str, Any]],
    *,
    train_on_what: Any,
    max_tokens: int,
) -> RenderedTrainingExample:
    """Render the largest recent, turn-aligned context that fits the model limit.

    The full conversation is attempted first. If it is too long, pruning begins at a
    user-message boundary or at an assistant boundary within one long tool-use turn.
    The initial system prefix, relevant user prompt, and final assistant target remain;
    a candidate never starts with an orphaned tool result.
    """

    if max_tokens < 2:
        raise TrainingDataError("max context tokens must be at least two")
    if not messages or messages[-1].get("role") != "assistant":
        raise TrainingDataError("training conversation must end with an assistant target")

    system_end = 0
    while system_end < len(messages) - 1 and messages[system_end].get("role") == "system":
        system_end += 1
    user_starts = [
        index
        for index in range(system_end, len(messages) - 1)
        if messages[index].get("role") == "user"
    ]
    candidates: list[tuple[list[dict[str, Any]], int]] = []
    seen: set[tuple[int, ...]] = set()

    def add_candidate(candidate: list[dict[str, Any]]) -> None:
        signature = tuple(id(message) for message in candidate)
        if signature not in seen:
            seen.add(signature)
            candidates.append((candidate, len(messages) - len(candidate)))

    add_candidate(messages)
    for user_index in user_starts:
        # First try dropping complete older user turns.
        add_candidate([*messages[:system_end], *messages[user_index:]])
        # A single agentic turn can contain dozens of assistant/tool cycles. Keep the
        # user prompt, then begin at an assistant boundary so no tool result is orphaned.
        for assistant_index in range(user_index + 1, len(messages) - 1):
            if messages[assistant_index].get("role") == "user":
                break
            if messages[assistant_index].get("role") == "assistant":
                add_candidate(
                    [
                        *messages[:system_end],
                        messages[user_index],
                        *messages[assistant_index:],
                    ]
                )
    candidates.sort(key=lambda item: item[1])

    for candidate, dropped in candidates:
        model_input, weights = renderer.build_supervised_example(
            candidate,
            train_on_what=train_on_what,
        )
        if len(model_input.to_ints()) <= max_tokens:
            return RenderedTrainingExample(model_input, weights, dropped)
    raise TrainingDataError("latest user-to-assistant target exceeds the model context limit")


def is_usable_conversation(messages: Any) -> bool:
    """Require a next-assistant target containing text, one or more tool calls, or both."""

    if not isinstance(messages, list) or len(messages) < 2:
        return False
    target = messages[-1]
    return (
        isinstance(target, dict)
        and target.get("role") == "assistant"
        and (bool(target.get("content")) or bool(target.get("tool_calls")))
    )


def load_split_conversations(
    path: str | Path,
    *,
    split: str,
    limit: int = 0,
    holdout_ratio: float = DEFAULT_HOLDOUT_RATIO,
    seed: str = DEFAULT_SPLIT_SEED,
) -> SplitSelection:
    """Stream a JSONL source, select one split, and count the complete source."""

    if split not in {"train", "holdout"}:
        raise TrainingDataError("split must be train or holdout")
    if limit < 0:
        raise TrainingDataError("limit cannot be negative")
    ratio = validate_holdout_ratio(holdout_ratio)
    source = Path(path)
    if not source.exists() or not source.is_file() or source.is_symlink():
        raise TrainingDataError("dataset path must be a regular file")

    selected: list[list[dict[str, Any]]] = []
    scanned_rows = usable_rows = duplicate_rows = 0
    train_rows = holdout_rows = selected_tool_targets = 0
    seen_digests: set[str] = set()
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            scanned_rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TrainingDataError(f"invalid JSON on dataset line {line_number}") from exc
            messages = row.get("messages") if isinstance(row, dict) else None
            if not is_usable_conversation(messages):
                continue
            digest = conversation_digest(messages, seed)
            if digest in seen_digests:
                duplicate_rows += 1
                continue
            seen_digests.add(digest)
            usable_rows += 1
            bucket = int(digest[:16], 16) / float(1 << 64)
            assigned = "holdout" if bucket < ratio else "train"
            if assigned == "train":
                train_rows += 1
            else:
                holdout_rows += 1
            if assigned != split or (limit and len(selected) >= limit):
                continue
            selected.append(messages)
            if messages[-1].get("tool_calls"):
                selected_tool_targets += 1

    return SplitSelection(
        conversations=selected,
        scanned_rows=scanned_rows,
        usable_rows=usable_rows,
        duplicate_rows=duplicate_rows,
        train_rows=train_rows,
        holdout_rows=holdout_rows,
        selected_tool_targets=selected_tool_targets,
    )
