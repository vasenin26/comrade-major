import asyncio
import logging

import numpy as np
import numpy.typing as npt

from config.settings import get_settings
from src.application.coordinator import AgentCoordinator
from src.infrastructure.audio_io import AudioIO
from src.infrastructure.detector import SileroVAD
from src.infrastructure.llm.factory import create_llm_service
from src.infrastructure.stt.whisper import LocalWhisperSTT
from src.infrastructure.tts.kokoro import KokoroTTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_coordinator() -> AgentCoordinator:
    settings = get_settings()

    stt_impl = LocalWhisperSTT(model_size=settings.whisper_model_size)
    llm_impl = create_llm_service(settings)
    tts_impl = KokoroTTS(sample_rate=settings.sample_rate)
    vad_impl = SileroVAD(threshold=settings.vad_threshold, sample_rate=settings.sample_rate)

    return AgentCoordinator(
        stt_service=stt_impl,
        llm_service=llm_impl,
        tts_service=tts_impl,
        vad_service=vad_impl,
    )


async def main() -> None:
    settings = get_settings()
    coordinator = build_coordinator()
    audio_io = AudioIO(sample_rate=settings.sample_rate)
    chunk_queue: asyncio.Queue[npt.NDArray[np.float32]] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    async def process_chunks() -> None:
        while True:
            chunk = await chunk_queue.get()
            await coordinator.handle_audio_chunk(chunk)

    processor = asyncio.create_task(process_chunks())
    logger.info("Voice agent initialized (sample_rate=%s)", settings.sample_rate)

    def on_chunk(chunk: npt.NDArray[np.float32]) -> None:
        loop.call_soon_threadsafe(chunk_queue.put_nowait, chunk)

    try:
        await audio_io.stream_input(chunk_size=512, on_chunk=on_chunk)
    finally:
        processor.cancel()


if __name__ == "__main__":
    asyncio.run(main())
