from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.infrastructure.tts.silero import SileroTTS


class FakeSileroModel:
    def apply_tts(self, text: str, speaker: str, sample_rate: int) -> torch.Tensor:
        _ = text, speaker, sample_rate
        t = torch.linspace(-1.0, 1.0, sample_rate // 10)
        return t.unsqueeze(0)

    def to(self, device: torch.device) -> "FakeSileroModel":
        _ = device
        return self


class FakeSileroModelToReturnsNone:
    """Mirrors real Silero packaged TTS: .to() mutates in place, returns None."""

    def apply_tts(self, text: str, speaker: str, sample_rate: int) -> torch.Tensor:
        _ = text, speaker, sample_rate
        return torch.zeros(1, sample_rate // 20)

    def to(self, device: torch.device) -> None:
        _ = device
        return None


@pytest.mark.asyncio
async def test_silero_synthesize_returns_audio() -> None:
    with patch.object(SileroTTS, "_load_model", return_value=FakeSileroModel()):
        tts = SileroTTS(
            sample_rate=16_000,
            speaker="xenia",
            device="cpu",
            synthesis_sample_rate=24_000,
        )
        audio = await tts.synthesize("привет")
    assert audio.dtype == np.float32
    assert len(audio) > 0
    assert float(np.abs(audio).max()) > 0.0


@pytest.mark.asyncio
async def test_silero_empty_text_is_silence() -> None:
    with patch.object(SileroTTS, "_load_model", return_value=FakeSileroModel()):
        tts = SileroTTS(sample_rate=16_000, device="cpu")
        audio = await tts.synthesize("   ")
    assert len(audio) == int(0.3 * 16_000)
    assert float(np.abs(audio).max()) == 0.0


@pytest.mark.asyncio
async def test_silero_to_returning_none_keeps_model() -> None:
    with patch.object(
        SileroTTS, "_load_model", return_value=FakeSileroModelToReturnsNone()
    ):
        tts = SileroTTS(sample_rate=16_000, device="cpu")
        audio = await tts.synthesize("тест")
    assert audio.dtype == np.float32
    assert len(audio) > 0
