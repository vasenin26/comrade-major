from src.domain.messages import (
    Message,
    MessageRole,
    extract_say_text,
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


def test_to_chat_messages_maps_inner_to_system() -> None:
    messages = [
        Message(role=MessageRole.SYSTEM, content="sys"),
        Message(role=MessageRole.USER, content="hi"),
        Message(role=MessageRole.INNER, content="focus"),
        Message(role=MessageRole.ASSISTANT, content="ok"),
    ]
    chat = to_chat_messages(messages)
    assert chat[2] == {"role": "system", "content": "[inner voice] focus"}
    assert chat[3] == {"role": "assistant", "content": "ok"}


def test_is_context_overflow_error() -> None:
    assert is_context_overflow_error(RuntimeError("maximum context length exceeded"))
    assert not is_context_overflow_error(RuntimeError("connection reset"))
