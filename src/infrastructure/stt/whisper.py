import asyncio

import numpy as np
import numpy.typing as npt


class LocalWhisperSTT:
    def __init__(self, model_size: str = "base") -> None:
        self._model_size = model_size
        self._model = self._load_model()

    def _load_model(self) -> object:
        from faster_whisper import WhisperModel

        return WhisperModel(self._model_size, device="cpu", compute_type="int8")

    async def transcribe(self, audio_data: npt.NDArray[np.float32]) -> str:
        return await asyncio.to_thread(self._transcribe_sync, audio_data)

    def _transcribe_sync(self, audio_data: npt.NDArray[np.float32]) -> str:
        segments, _ = self._model.transcribe(audio_data, beam_size=5)
        return " ".join(segment.text.strip() for segment in segments)
