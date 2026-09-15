import asyncio
from collections import deque
from collections.abc import Sequence
from time import monotonic

from src.domain.messages import Message, MessageRole, to_chat_messages


class ConversationStore:
    """Shared conversation context for parallel loops.

    Hold the condition lock only for short mutations — never across Mind.think.
    INNER is a single overwriteable slot; private thoughts are a rolling deque.
    Both are injected into snapshot_chat, not stored as chat messages.
    """

    def __init__(
        self,
        system_prompt: str | None = None,
        thought_history: int = 8,
    ) -> None:
        self._cond = asyncio.Condition()
        self._messages: list[Message] = []
        self._inner_context: str = ""
        self._private_thoughts: deque[str] = deque(maxlen=max(1, thought_history))
        self._revision: int = 0
        self._last_user_activity: float = monotonic()
        if system_prompt:
            self._messages.append(Message(role=MessageRole.SYSTEM, content=system_prompt))

    async def revision(self) -> int:
        async with self._cond:
            return self._revision

    async def seconds_since_user_activity(self) -> float:
        async with self._cond:
            return monotonic() - self._last_user_activity

    async def wait_until_changed(self, known: int, timeout: float) -> int:
        """Block until revision > known or timeout. Always returns current revision."""
        async with self._cond:
            if self._revision > known:
                return self._revision
            try:
                await asyncio.wait_for(
                    self._cond.wait_for(lambda: self._revision > known),
                    timeout=timeout,
                )
            except TimeoutError:
                pass
            return self._revision

    def _bump_unlocked(self) -> None:
        self._revision += 1
        self._cond.notify_all()

    async def snapshot(self) -> list[Message]:
        async with self._cond:
            return list(self._messages)

    async def get_inner_context(self) -> str:
        async with self._cond:
            return self._inner_context

    async def set_inner_context(self, content: str) -> None:
        async with self._cond:
            self._inner_context = content.strip()
            self._bump_unlocked()

    async def append_private_thought(self, content: str) -> None:
        stripped = content.strip()
        if not stripped:
            return
        async with self._cond:
            self._private_thoughts.append(stripped)

    async def private_thoughts(self) -> list[str]:
        async with self._cond:
            return list(self._private_thoughts)

    async def snapshot_chat(self) -> list[dict[str, str]]:
        async with self._cond:
            messages = list(self._messages)
            inner = self._inner_context
            thoughts = list(self._private_thoughts)
        chat = to_chat_messages(messages)
        extras: list[dict[str, str]] = []
        if inner:
            extras.append({"role": "system", "content": f"[INNER]\n{inner}"})
        if thoughts:
            joined = "\n---\n".join(thoughts)
            extras.append({"role": "system", "content": f"[THOUGHTS]\n{joined}"})
        if not extras:
            return chat
        if chat and chat[0].get("role") == "system":
            return [chat[0], *extras, *chat[1:]]
        return [*extras, *chat]

    async def append(self, role: MessageRole, content: str) -> Message:
        message = Message(role=role, content=content)
        async with self._cond:
            self._messages.append(message)
            if role == MessageRole.USER:
                self._last_user_activity = monotonic()
            self._bump_unlocked()
        return message

    async def apply_patch(self, messages: Sequence[Message]) -> None:
        async with self._cond:
            self._messages.extend(messages)
            if any(m.role == MessageRole.USER for m in messages):
                self._last_user_activity = monotonic()
            if messages:
                self._bump_unlocked()

    async def drop_oldest(self, count: int = 1) -> list[Message]:
        """Drop oldest non-system messages. Returns removed messages.

        Does not touch INNER or private thoughts.
        """
        if count < 1:
            return []
        removed: list[Message] = []
        async with self._cond:
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
            if removed:
                self._bump_unlocked()
        return removed

    async def length(self) -> int:
        async with self._cond:
            return len(self._messages)
