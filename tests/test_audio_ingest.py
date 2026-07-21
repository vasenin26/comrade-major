import asyncio

import numpy as np
import numpy.typing as npt
import pytest

from src.application.loops.audio_ingest import AudioIngestLoop
from src.domain.conversation import ConversationStore
from src.domain.messages import MessageRole


class FakeVAD:
    def __init__(self, speech_flags: list[bool]) -> None:
        self._flags = list(speech_flags)

    def is_speech(self, audio_chunk: npt.NDArray[np.float32]) -> bool:
        if not self._flags:
            return False
        return self._flags.pop(0)


class FakeSTT:
    def __init__(self, text: str = "hello from mic") -> None:
        self.text = text
        self.calls = 0

    async def transcribe(self, audio_data: npt.NDArray[np.float32]) -> str:
        self.calls += 1
        return self.text


class FakeLog:
    def __init__(self) -> None:
        self.entries: list[tuple[str, str]] = []

    async def append(self, role: str, content: str) -> None:
        self.entries.append((role, content))


@pytest.mark.asyncio
async def test_audio_ingest_commits_utterance_after_silence() -> None:
    store = ConversationStore()
    log = FakeLog()
    stt = FakeSTT()
    # speech, speech, silence (start timer), silence (flush after min_silence=0)
    vad = FakeVAD([True, True, False, False])
    queue: asyncio.Queue[npt.NDArray[np.float32]] = asyncio.Queue()
    loop = AudioIngestLoop(
        store=store,
        stt=stt,
        vad=vad,
        message_log=log,
        min_silence_ms=0,
        chunk_queue=queue,
    )

    chunk = np.zeros(4, dtype=np.float32)
    await loop.handle_audio_chunk(chunk)
    await loop.handle_audio_chunk(chunk)
    await loop.handle_audio_chunk(chunk)
    await loop.handle_audio_chunk(chunk)

    assert stt.calls == 1
    snap = await store.snapshot()
    assert snap[-1].role == MessageRole.USER
    assert snap[-1].content == "hello from mic"
    assert log.entries[-1] == ("user", "hello from mic")
