from collections.abc import AsyncIterator
from typing import Protocol

import numpy as np
import numpy.typing as npt


class STTService(Protocol):
    async def transcribe(self, audio_data: npt.NDArray[np.float32]) -> str: ...


class Mind(Protocol):
    """Domain-facing reasoning capability. Adapters wrap concrete model providers."""

    async def think(self, history: list[dict[str, str]]) -> str: ...

    async def stream(self, history: list[dict[str, str]]) -> AsyncIterator[str]: ...


class TTSService(Protocol):
    async def synthesize(self, text: str) -> npt.NDArray[np.float32]: ...


class VADService(Protocol):
    def is_speech(self, audio_chunk: npt.NDArray[np.float32]) -> bool: ...


class MessageLog(Protocol):
    async def append(self, role: str, content: str) -> None: ...


class LoopWorker(Protocol):
    async def run(self) -> None: ...


class AudioPlayer(Protocol):
    async def play(self, audio: npt.NDArray[np.float32]) -> None: ...
