import pytest

from src.domain.conversation import ConversationStore
from src.domain.messages import (
    Message,
    MessageRole,
    extract_say_text,
    extract_think_topic,
    is_context_overflow_error,
    to_chat_messages,
)


def test_extract_say_text_basic() -> None:
    assert extract_say_text("say hello world") == "hello world"
    assert extract_say_text("SAY: hi") == "hi"
    assert extract_say_text("  say\nok") == "ok"


def test_extract_say_text_rejects_non_say() -> None:
    assert extract_say_text("saying hello") is None
    assert extract_say_text("hello") is None
    assert extract_say_text("") is None


def test_extract_think_topic_basic() -> None:
    assert extract_think_topic("think how to answer") == "how to answer"
    assert extract_think_topic("THINK: план") == "план"
    assert extract_think_topic("thinking aloud") is None


def test_to_chat_messages_skips_inner_role() -> None:
    messages = [
        Message(role=MessageRole.SYSTEM, content="sys"),
        Message(role=MessageRole.USER, content="hi"),
        Message(role=MessageRole.INNER, content="focus"),
        Message(role=MessageRole.ASSISTANT, content="ok"),
    ]
    chat = to_chat_messages(messages)
    assert chat == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ok"},
    ]


@pytest.mark.asyncio
async def test_store_inner_slot_overwrite_and_snapshot_chat() -> None:
    store = ConversationStore(system_prompt="sys")
    await store.append(MessageRole.USER, "hi")
    await store.set_inner_context("first")
    chat = await store.snapshot_chat()
    assert chat[0] == {"role": "system", "content": "sys"}
    assert chat[1] == {"role": "system", "content": "[INNER]\nfirst"}
    assert chat[2] == {"role": "user", "content": "hi"}

    await store.set_inner_context("second")
    chat2 = await store.snapshot_chat()
    inner_blocks = [m for m in chat2 if m["content"].startswith("[INNER]\n")]
    assert len(inner_blocks) == 1
    assert inner_blocks[0]["content"] == "[INNER]\nsecond"
    assert MessageRole.INNER not in {m.role for m in await store.snapshot()}


@pytest.mark.asyncio
async def test_drop_oldest_preserves_inner_slot() -> None:
    store = ConversationStore(system_prompt="sys")
    await store.append(MessageRole.USER, "old")
    await store.set_inner_context("keep me")
    await store.drop_oldest(1)
    assert await store.get_inner_context() == "keep me"


def test_is_context_overflow_error() -> None:
    assert is_context_overflow_error(RuntimeError("maximum context length exceeded"))
    assert not is_context_overflow_error(RuntimeError("connection reset"))
