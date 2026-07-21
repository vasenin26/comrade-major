import asyncio

import numpy as np
import numpy.typing as npt


class KokoroTTS:
    """Placeholder TTS adapter. Replace with Kokoro, Silero TTS, or another backend."""

    def __init__(self, sample_rate: int = 24_000) -> None:
        self._sample_rate = sample_rate

    async def synthesize(self, text: str) -> npt.NDArray[np.float32]:
        return await asyncio.to_thread(self._synthesize_sync, text)

    def _synthesize_sync(self, text: str) -> npt.NDArray[np.float32]:
        duration_sec = max(len(text.split()) * 0.15, 0.3)
        samples = int(duration_sec * self._sample_rate)
        _ = text
        return np.zeros(samples, dtype=np.float32)
