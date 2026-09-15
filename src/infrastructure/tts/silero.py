import asyncio
import logging
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt
import torch
import torchaudio  # type: ignore

logger = logging.getLogger(__name__)


class _SileroTtsModel(Protocol):
    def apply_tts(self, text: str, speaker: str, sample_rate: int) -> torch.Tensor: ...

    def to(self, device: torch.device) -> "_SileroTtsModel | None": ...

_SILERO_MODEL_ID = "v5_ru"
_SILERO_LANGUAGE = "ru"
_EMPTY_SILENCE_SEC = 0.3


class SileroTTS:
    """Local Russian TTS via Silero models (pip package `silero`)."""

    def __init__(
        self,
        sample_rate: int = 16_000,
        speaker: str = "aidar",
        device: str = "auto",
        synthesis_sample_rate: int = 24_000,
    ) -> None:
        self._output_sample_rate = sample_rate
        self._speaker = speaker
        self._synthesis_sample_rate = synthesis_sample_rate
        self._device = self._resolve_device(device)
        # Silero's packaged TTSModel.to() mutates in place and returns None.
        self._model = self._load_model()
        moved = self._model.to(self._device)
        if moved is not None:
            self._model = moved
        logger.info(
            "Silero TTS loaded (speaker=%s, synth_sr=%s, out_sr=%s, device=%s)",
            speaker,
            synthesis_sample_rate,
            sample_rate,
            self._device,
        )

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _load_model(self) -> _SileroTtsModel:
        # torch.hub breaks when this repo's top-level `src` package shadows
        # silero-models hubconf (`from src.silero import ...`).
        from silero import silero_tts

        model, _example = silero_tts(
            language=_SILERO_LANGUAGE,
            speaker=_SILERO_MODEL_ID,
        )
        return cast(_SileroTtsModel, model)

    async def synthesize(self, text: str) -> npt.NDArray[np.float32]:
        return await asyncio.to_thread(self._synthesize_sync, text)

    def _synthesize_sync(self, text: str) -> npt.NDArray[np.float32]:
        stripped = text.strip()
        if not stripped:
            samples = int(_EMPTY_SILENCE_SEC * self._output_sample_rate)
            return np.zeros(samples, dtype=np.float32)

        audio = self._model.apply_tts(
            text=stripped,
            speaker=self._speaker,
            sample_rate=self._synthesis_sample_rate,
        )
        waveform = self._to_mono_float_tensor(audio)
        if self._synthesis_sample_rate != self._output_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=self._synthesis_sample_rate,
                new_freq=self._output_sample_rate,
            )
        return waveform.squeeze().cpu().numpy().astype(np.float32, copy=False)

    @staticmethod
    def _to_mono_float_tensor(audio: object) -> torch.Tensor:
        if isinstance(audio, torch.Tensor):
            tensor = audio.detach().float()
        else:
            tensor = torch.as_tensor(audio, dtype=torch.float32)
        if tensor.ndim == 1:
            return tensor.unsqueeze(0)
        if tensor.ndim == 2 and tensor.shape[0] > 1:
            return tensor.mean(dim=0, keepdim=True)
        return tensor
