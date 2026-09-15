import asyncio
import logging

from src.application.interfaces import ListeningGate, LongTermMemory, MessageLog
from src.domain.conversation import ConversationStore

logger = logging.getLogger(__name__)


class SleepConsolidationLoop:
    """Idle-gated worker that consolidates episodic buffer into long-term memory."""

    def __init__(
        self,
        store: ConversationStore,
        memory: LongTermMemory,
        message_log: MessageLog,
        *,
        idle_seconds: float = 120.0,
        poll_seconds: float = 10.0,
        listening_gate: ListeningGate | None = None,
        error_backoff_seconds: float = 5.0,
    ) -> None:
        self._store = store
        self._memory = memory
        self._message_log = message_log
        self._idle_seconds = idle_seconds
        self._poll_seconds = max(0.05, poll_seconds)
        self._listening_gate = listening_gate
        self._error_backoff_seconds = error_backoff_seconds

    async def run(self) -> None:
        while True:
            try:
                await asyncio.sleep(self._poll_seconds)
                if self._listening_gate is not None and not self._listening_gate.is_listening():
                    continue
                idle = await self._store.seconds_since_user_activity()
                if idle < self._idle_seconds:
                    continue
                processed = await self._memory.consolidate()
                if processed > 0:
                    await self._message_log.append(
                        "sleep",
                        f"consolidated {processed} episodes",
                    )
                    logger.info("Sleep consolidation processed %s episodes", processed)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Sleep consolidation failed: %s", exc)
                await asyncio.sleep(self._error_backoff_seconds)
