import asyncio
from collections.abc import Sequence

from src.domain.messages import Message, MessageRole, to_chat_messages


class ConversationStore:
    """Shared conversation context for parallel loops.

    Hold the asyncio.Lock only for short mutations — never across Mind.think.
    INNER is a single overwriteable slot injected into snapshot_chat, not a
    growing list of chat messages.
    """

    def __init__(self, system_prompt: str | None = None) -> None:
        self._lock = asyncio.Lock()
        self._messages: list[Message] = []
        self._inner_context: str = ""
        if system_prompt:
            self._messages.append(Message(role=MessageRole.SYSTEM, content=system_prompt))

    async def snapshot(self) -> list[Message]:
        async with self._lock:
            return list(self._messages)

    async def get_inner_context(self) -> str:
        async with self._lock:
            return self._inner_context

    async def set_inner_context(self, content: str) -> None:
        async with self._lock:
            self._inner_context = content.strip()

    async def snapshot_chat(self) -> list[dict[str, str]]:
        async with self._lock:
            messages = list(self._messages)
            inner = self._inner_context
        chat = to_chat_messages(messages)
        if not inner:
            return chat
        inner_block = {"role": "system", "content": f"[INNER]\n{inner}"}
        if chat and chat[0].get("role") == "system":
            return [chat[0], inner_block, *chat[1:]]
        return [inner_block, *chat]

    async def append(self, role: MessageRole, content: str) -> Message:
        message = Message(role=role, content=content)
        async with self._lock:
            self._messages.append(message)
        return message

    async def apply_patch(self, messages: Sequence[Message]) -> None:
        async with self._lock:
            self._messages.extend(messages)

    async def drop_oldest(self, count: int = 1) -> list[Message]:
        """Drop oldest non-system messages. Returns removed messages.

        Does not touch the INNER context slot.
        """
        if count < 1:
            return []
        removed: list[Message] = []
        async with self._lock:
            while count > 0 and self._messages:
                idx = next(
                    (
                        i
                        for i, msg in enumerate(self._messages)
                        if msg.role != MessageRole.SYSTEM
                    ),
                    None,
                )
                if idx is None:
                    break
                removed.append(self._messages.pop(idx))
                count -= 1
        return removed

    async def length(self) -> int:
        async with self._lock:
            return len(self._messages)
