import asyncio
import logging

from src.application.inner_think import InnerThinkService
from src.domain.conversation import ConversationStore
from src.domain.messages import is_context_overflow_error

logger = logging.getLogger(__name__)


class InnerVoiceLoop:
    """Background mind that refreshes the INNER context slot on an interval."""

    def __init__(
        self,
        store: ConversationStore,
        inner_think: InnerThinkService,
        interval_seconds: float = 60.0,
        context_trim_count: int = 2,
        error_backoff_seconds: float = 1.0,
    ) -> None:
        self._store = store
        self._inner_think = inner_think
        self._interval_seconds = interval_seconds
        self._context_trim_count = context_trim_count
        self._error_backoff_seconds = error_backoff_seconds

    async def run(self) -> None:
        while True:
            try:
                await self._inner_think.ponder(topic=None)
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
                continue
            await asyncio.sleep(self._interval_seconds)
