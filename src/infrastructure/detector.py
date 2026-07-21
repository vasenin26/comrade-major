import numpy as np
import numpy.typing as npt
import torch


class SileroVAD:
    def __init__(self, threshold: float = 0.5, sample_rate: int = 16_000) -> None:
        self._threshold = threshold
        self._sample_rate = sample_rate
        self._model = self._load_model()

    def _load_model(self) -> object:
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        return model

    def is_speech(self, audio_chunk: npt.NDArray[np.float32]) -> bool:
        tensor = torch.from_numpy(audio_chunk)
        speech_prob = self._run_inference(tensor)
        return speech_prob >= self._threshold

    def _run_inference(self, audio_tensor: torch.Tensor) -> float:
        result = self._model(audio_tensor, self._sample_rate)
        if isinstance(result, torch.Tensor):
            return float(result.item())
        return float(result)
