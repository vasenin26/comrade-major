import asyncio
import logging
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import sounddevice as sd

from src.application.listening_gate import ListeningGate

logger = logging.getLogger(__name__)


class AudioIO:
    def __init__(
        self,
        sample_rate: int,
        channels: int = 1,
        listening_gate: ListeningGate | None = None,
    ) -> None:
        self._sample_rate = sample_rate
        self._channels = channels
        self._gate = listening_gate

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    async def stream_input(
        self,
        chunk_size: int,
        on_chunk: Callable[[npt.NDArray[np.float32]], None],
    ) -> None:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[npt.NDArray[np.float32]] = asyncio.Queue()

        def callback(
            indata: npt.NDArray[np.float32],
            _frames: int,
            _time: object,
            _status: sd.CallbackFlags,
        ) -> None:
            if _status:
                logger.warning("Audio input status: %s", _status)
            loop.call_soon_threadsafe(queue.put_nowait, indata.copy().flatten())

        with sd.InputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            blocksize=chunk_size,
            dtype="float32",
            callback=callback,
        ):
            while True:
                chunk = await queue.get()
                if self._gate is not None and not self._gate.is_listening():
                    continue
                on_chunk(chunk)

    async def play(self, audio: npt.NDArray[np.float32]) -> None:
        if self._gate is None:
            await asyncio.to_thread(
                sd.play,
                audio,
                self._sample_rate,
                blocking=True,
            )
            return
        async with self._gate.speaking():
            await asyncio.to_thread(
                sd.play,
                audio,
                self._sample_rate,
                blocking=True,
            )

    async def stop(self) -> None:
        await asyncio.to_thread(sd.stop)
