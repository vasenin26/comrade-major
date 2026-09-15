import asyncio
import logging

from src.application.inner_think import InnerThinkService
from src.application.interfaces import (
    AudioPlayer,
    LongTermMemory,
    MessageLog,
    Mind,
    TTSService,
)
from src.application.memory_context import build_recall_query, inject_memory_passages
from src.domain.conversation import ConversationStore
from src.domain.messages import (
    MessageRole,
    extract_say_text,
    extract_think_topic,
    is_context_overflow_error,
)

logger = logging.getLogger(__name__)


class PrimaryThinkingLoop:
    def __init__(
        self,
        store: ConversationStore,
        mind: Mind,
        message_log: MessageLog,
        tts: TTSService,
        audio_player: AudioPlayer,
        inner_think: InnerThinkService | None = None,
        memory: LongTermMemory | None = None,
        context_trim_count: int = 2,
        error_backoff_seconds: float = 1.0,
        max_think_rounds: int = 2,
        pulse_seconds: float = 15.0,
        memory_recall_limit: int = 5,
    ) -> None:
        self._store = store
        self._mind = mind
        self._message_log = message_log
        self._tts = tts
        self._audio_player = audio_player
        self._inner_think = inner_think
        self._memory = memory
        self._context_trim_count = context_trim_count
        self._error_backoff_seconds = error_backoff_seconds
        self._max_think_rounds = max_think_rounds
        self._pulse_seconds = pulse_seconds
        self._memory_recall_limit = memory_recall_limit

    async def run(self) -> None:
        known = await self._store.revision()
        while True:
            try:
                known = await self._store.wait_until_changed(
                    known, timeout=self._pulse_seconds
                )
                await self._think_once()
                known = await self._store.revision()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if is_context_overflow_error(exc):
                    removed = await self._store.drop_oldest(self._context_trim_count)
                    logger.warning(
                        "Context overflow — dropped %s oldest messages, retrying",
                        len(removed),
                    )
                    known = await self._store.revision()
                    if not removed:
                        await asyncio.sleep(self._error_backoff_seconds)
                    continue
                logger.exception("Primary thinking failed: %s", exc)
                await asyncio.sleep(self._error_backoff_seconds)
                known = await self._store.revision()

    async def _history_with_memory(self) -> list[dict[str, str]]:
        history = await self._store.snapshot_chat()
        if self._memory is None:
            return history
        query = await build_recall_query(self._store)
        if not query.strip():
            return history
        passages = await self._memory.recall(query, self._memory_recall_limit)
        return inject_memory_passages(history, passages)

    async def _think_once(self) -> None:
        history = await self._history_with_memory()
        reply = await self._mind.think(history)
        rounds = 0
        while self._inner_think is not None and rounds < self._max_think_rounds:
            topic = extract_think_topic(reply)
            if topic is None:
                break
            await self._inner_think.ponder(topic=topic)
            history = await self._history_with_memory()
            reply = await self._mind.think(history)
            rounds += 1

        if not reply.strip():
            return
        # Do not persist unresolved think calls as assistant chat messages
        if extract_think_topic(reply) is not None:
            return

        say_text = extract_say_text(reply)
        if say_text:
            await self._store.append(MessageRole.ASSISTANT, reply)
            await self._message_log.append(MessageRole.ASSISTANT.value, reply)
            await self._speak(say_text)
            return

        await self._store.append_private_thought(reply)
        await self._message_log.append("thought", reply)

    async def _speak(self, text: str) -> None:
        try:
            audio = await self._tts.synthesize(text)
            await self._audio_player.play(audio)
        except Exception:
            logger.exception("Failed to speak: %s", text[:80])
