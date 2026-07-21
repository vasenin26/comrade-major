import asyncio
import logging

import numpy as np
import numpy.typing as npt

from src.application.interfaces import LLMService, STTService, TTSService, VADService
from src.domain.chat import ChatSession

logger = logging.getLogger(__name__)


class AgentCoordinator:
    def __init__(
        self,
        stt_service: STTService,
        llm_service: LLMService,
        tts_service: TTSService,
        vad_service: VADService,
    ) -> None:
        self._stt = stt_service
        self._llm = llm_service
        self._tts = tts_service
        self._vad = vad_service
        self._chat = ChatSession()
        self._interrupt_event = asyncio.Event()

    @property
    def interrupt_event(self) -> asyncio.Event:
        return self._interrupt_event

    def request_interrupt(self) -> None:
        self._interrupt_event.set()

    def clear_interrupt(self) -> None:
        self._interrupt_event.clear()

    async def handle_audio_chunk(self, audio_chunk: npt.NDArray[np.float32]) -> None:
        if not self._vad.is_speech(audio_chunk):
            return

        if self._interrupt_event.is_set():
            logger.debug("Speech detected during playback — barge-in triggered")
            self.clear_interrupt()

        text = await self._stt.transcribe(audio_chunk)
        if not text.strip():
            return

        self._chat.add_user_message(text)
        reply = await self._llm.generate(text, self._chat.history)
        self._chat.add_assistant_message(reply)

        audio = await self._tts.synthesize(reply)
        await self._play_with_interrupt_check(audio)

    async def _play_with_interrupt_check(
        self,
        audio: npt.NDArray[np.float32],
        chunk_size: int = 1024,
    ) -> None:
        for offset in range(0, len(audio), chunk_size):
            if self._interrupt_event.is_set():
                break
            _ = audio[offset : offset + chunk_size]
            await asyncio.sleep(0)
