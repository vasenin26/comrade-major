import asyncio
import logging

from src.application.interfaces import MessageLog, Mind
from src.domain.conversation import ConversationStore
from src.domain.messages import is_context_overflow_error

logger = logging.getLogger(__name__)


class InnerThinkService:
    """Runs the inner mind and writes the result into the INNER context slot."""

    def __init__(
        self,
        store: ConversationStore,
        mind: Mind,
        message_log: MessageLog,
        system_prompt: str,
    ) -> None:
        self._store = store
        self._mind = mind
        self._message_log = message_log
        self._system_prompt = system_prompt
        self._lock = asyncio.Lock()

    async def ponder(self, topic: str | None = None) -> str:
        async with self._lock:
            return await self._ponder_unlocked(topic)

    async def _ponder_unlocked(self, topic: str | None) -> str:
        chat = await self._store.snapshot_chat()
        if topic is not None and topic.strip():
            nudge = f"Think about: {topic.strip()}"
        else:
            nudge = "Update the INNER notes for the primary mind."
        history = [
            {"role": "system", "content": self._system_prompt},
            *chat,
            {"role": "user", "content": nudge},
        ]
        try:
            note = await self._mind.think(history)
        except Exception as exc:
            if is_context_overflow_error(exc):
                raise
            logger.exception("Inner ponder failed: %s", exc)
            raise

        stripped = note.strip()
        if not stripped:
            return ""
        await self._store.set_inner_context(stripped)
        await self._message_log.append("inner", stripped)
        return stripped
