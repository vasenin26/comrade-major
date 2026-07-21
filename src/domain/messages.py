from dataclasses import dataclass
from enum import StrEnum
from typing import Literal


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    INNER = "inner"


ChatRole = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class Message:
    role: MessageRole
    content: str


def extract_say_text(content: str) -> str | None:
    """Return spoken text if content starts with ``say`` (+ separator), else None."""
    stripped = content.lstrip()
    lower = stripped.lower()
    if not lower.startswith("say"):
        return None
    rest = stripped[3:]
    if not rest:
        return ""
    if rest[0] not in " \t\n\r:":
        return None
    return rest.lstrip(" \t\n\r:").strip()


def to_chat_messages(messages: list[Message]) -> list[dict[str, str]]:
    """Map domain messages to chat roles for Mind.think (inner → system note)."""
    result: list[dict[str, str]] = []
    for message in messages:
        if message.role == MessageRole.INNER:
            result.append(
                {
                    "role": "system",
                    "content": f"[inner voice] {message.content}",
                }
            )
        else:
            result.append({"role": message.role.value, "content": message.content})
    return result


def is_context_overflow_error(exc: BaseException) -> bool:
    """Heuristic: provider reported context/token limit exceeded."""
    text = str(exc).lower()
    markers = (
        "context length",
        "context_length",
        "maximum context",
        "max_tokens",
        "too many tokens",
        "token limit",
        "context window",
        "exceeds the model",
        "prompt is too long",
    )
    return any(marker in text for marker in markers)
