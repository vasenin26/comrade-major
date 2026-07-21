import asyncio
import logging
from collections.abc import AsyncIterator, Callable

import numpy as np
import numpy.typing as npt
import sounddevice as sd

logger = logging.getLogger(__name__)


class AudioIO:
    def __init__(self, sample_rate: int, channels: int = 1) -> None:
        self._sample_rate = sample_rate
        self._channels = channels

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

        def callback(indata: npt.NDArray[np.float32], _frames: int, _time: object, _status: sd.CallbackFlags) -> None:
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
                on_chunk(chunk)

    async def play(self, audio: npt.NDArray[np.float32]) -> None:
        await asyncio.to_thread(
            sd.play,
            audio,
            self._sample_rate,
            blocking=True,
        )

    async def stop(self) -> None:
        await asyncio.to_thread(sd.stop)
