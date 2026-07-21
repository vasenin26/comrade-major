import asyncio
import logging

from src.application.interfaces import AudioPlayer, MessageLog, Mind, TTSService
from src.domain.conversation import ConversationStore
from src.domain.messages import MessageRole, extract_say_text, is_context_overflow_error

logger = logging.getLogger(__name__)


class PrimaryThinkingLoop:
    def __init__(
        self,
        store: ConversationStore,
        mind: Mind,
        message_log: MessageLog,
        tts: TTSService,
        audio_player: AudioPlayer,
        context_trim_count: int = 2,
        error_backoff_seconds: float = 1.0,
    ) -> None:
        self._store = store
        self._mind = mind
        self._message_log = message_log
        self._tts = tts
        self._audio_player = audio_player
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
                        "Context overflow — dropped %s oldest messages, retrying",
                        len(removed),
                    )
                    if not removed:
                        await asyncio.sleep(self._error_backoff_seconds)
                    continue
                logger.exception("Primary thinking failed: %s", exc)
                await asyncio.sleep(self._error_backoff_seconds)

    async def _think_once(self) -> None:
        history = await self._store.snapshot_chat()
        reply = await self._mind.think(history)
        if not reply.strip():
            await asyncio.sleep(0)
            return
        await self._store.append(MessageRole.ASSISTANT, reply)
        await self._message_log.append(MessageRole.ASSISTANT.value, reply)

        say_text = extract_say_text(reply)
        if say_text:
            asyncio.create_task(self._speak(say_text), name="primary-say")
        await asyncio.sleep(0)

    async def _speak(self, text: str) -> None:
        try:
            audio = await self._tts.synthesize(text)
            await self._audio_player.play(audio)
        except Exception:
            logger.exception("Failed to speak: %s", text[:80])
