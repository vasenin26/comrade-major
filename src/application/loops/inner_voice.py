import asyncio
import logging

from src.application.interfaces import MessageLog, Mind
from src.domain.conversation import ConversationStore
from src.domain.messages import Message, MessageRole, is_context_overflow_error

logger = logging.getLogger(__name__)


class InnerVoiceLoop:
    """Background mind that patches the shared conversation context."""

    def __init__(
        self,
        store: ConversationStore,
        mind: Mind,
        message_log: MessageLog,
        system_prompt: str,
        context_trim_count: int = 2,
        error_backoff_seconds: float = 1.0,
    ) -> None:
        self._store = store
        self._mind = mind
        self._message_log = message_log
        self._system_prompt = system_prompt
        self._context_trim_count = context_trim_count
        self._error_backoff_seconds = error_backoff_seconds

    async def run(self) -> None:
        while True:
            try:
                await self._think_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if is_context_overflow_error(exc):
                    removed = await self._store.drop_oldest(self._context_trim_count)
                    logger.warning(
                        "Inner voice context overflow — dropped %s messages",
                        len(removed),
                    )
                    if not removed:
                        await asyncio.sleep(self._error_backoff_seconds)
                    continue
                logger.exception("Inner voice failed: %s", exc)
                await asyncio.sleep(self._error_backoff_seconds)

    async def _think_once(self) -> None:
        chat = await self._store.snapshot_chat()
        history = [
            {"role": "system", "content": self._system_prompt},
            *chat,
            {
                "role": "user",
                "content": "Inner voice: add a brief note for the primary mind.",
            },
        ]
        note = await self._mind.think(history)
        if not note.strip():
            await asyncio.sleep(0)
            return
        message = Message(role=MessageRole.INNER, content=note.strip())
        await self._store.apply_patch([message])
        await self._message_log.append(MessageRole.INNER.value, message.content)
        await asyncio.sleep(0)
