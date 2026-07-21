from collections.abc import AsyncIterator
from typing import Protocol

import numpy as np
import numpy.typing as npt


class STTService(Protocol):
    async def transcribe(self, audio_data: npt.NDArray[np.float32]) -> str: ...


class LLMService(Protocol):
    """Контракт LLM. Application не зависит от конкретного провайдера."""

    async def generate(self, prompt: str, history: list[dict[str, str]]) -> str: ...

    async def stream(self, prompt: str, history: list[dict[str, str]]) -> AsyncIterator[str]: ...


class TTSService(Protocol):
    async def synthesize(self, text: str) -> npt.NDArray[np.float32]: ...


class VADService(Protocol):
    def is_speech(self, audio_chunk: npt.NDArray[np.float32]) -> bool: ...
