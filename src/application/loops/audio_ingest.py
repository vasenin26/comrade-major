import asyncio
import logging
import time

import numpy as np
import numpy.typing as npt

from src.application.interfaces import MessageLog, STTService, VADService
from src.domain.conversation import ConversationStore
from src.domain.messages import MessageRole

logger = logging.getLogger(__name__)


class AudioIngestLoop:
    """Buffers speech by VAD, transcribes utterances, appends user messages."""

    def __init__(
        self,
        store: ConversationStore,
        stt: STTService,
        vad: VADService,
        message_log: MessageLog,
        min_silence_ms: int,
        chunk_queue: asyncio.Queue[npt.NDArray[np.float32]],
    ) -> None:
        self._store = store
        self._stt = stt
        self._vad = vad
        self._message_log = message_log
        self._min_silence_sec = min_silence_ms / 1000.0
        self._chunk_queue = chunk_queue
        self._speech_chunks: list[npt.NDArray[np.float32]] = []
        self._silence_started_at: float | None = None

    async def run(self) -> None:
        while True:
            chunk = await self._chunk_queue.get()
            await self.handle_audio_chunk(chunk)

    async def handle_audio_chunk(self, audio_chunk: npt.NDArray[np.float32]) -> None:
        if self._vad.is_speech(audio_chunk):
            self._speech_chunks.append(audio_chunk)
            self._silence_started_at = None
            return

        if not self._speech_chunks:
            return

        now = time.monotonic()
        if self._silence_started_at is None:
            self._silence_started_at = now
            return

        if now - self._silence_started_at < self._min_silence_sec:
            return

        utterance = np.concatenate(self._speech_chunks)
        self._speech_chunks.clear()
        self._silence_started_at = None

        text = await self._stt.transcribe(utterance)
        if not text.strip():
            return

        await self._store.append(MessageRole.USER, text.strip())
        await self._message_log.append(MessageRole.USER.value, text.strip())
        logger.info("User utterance: %s", text.strip()[:120])
