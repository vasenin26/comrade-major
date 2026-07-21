import asyncio
from collections.abc import Sequence

from src.domain.messages import Message, MessageRole, to_chat_messages


class ConversationStore:
    """Shared conversation context for parallel loops.

    Hold the asyncio.Lock only for short mutations — never across Mind.think.
    """

    def __init__(self, system_prompt: str | None = None) -> None:
        self._lock = asyncio.Lock()
        self._messages: list[Message] = []
        if system_prompt:
            self._messages.append(Message(role=MessageRole.SYSTEM, content=system_prompt))

    async def snapshot(self) -> list[Message]:
        async with self._lock:
            return list(self._messages)

    async def snapshot_chat(self) -> list[dict[str, str]]:
        return to_chat_messages(await self.snapshot())

    async def append(self, role: MessageRole, content: str) -> Message:
        message = Message(role=role, content=content)
        async with self._lock:
            self._messages.append(message)
        return message

    async def apply_patch(self, messages: Sequence[Message]) -> None:
        async with self._lock:
            self._messages.extend(messages)

    async def drop_oldest(self, count: int = 1) -> list[Message]:
        """Drop oldest non-system messages. Returns removed messages."""
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
