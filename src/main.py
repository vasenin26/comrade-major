import asyncio
import logging

import numpy as np
import numpy.typing as npt

from config.settings import get_settings
from config.types import MindRole
from src.application.loops import AudioIngestLoop, InnerVoiceLoop, PrimaryThinkingLoop
from src.application.runtime import AgentRuntime
from src.domain.conversation import ConversationStore
from src.infrastructure.audio_io import AudioIO
from src.infrastructure.detector import SileroVAD
from src.infrastructure.logging import FileMessageLog
from src.infrastructure.mind import create_mind
from src.infrastructure.stt.whisper import LocalWhisperSTT
from src.infrastructure.tts.kokoro import KokoroTTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_runtime(
    chunk_queue: asyncio.Queue[npt.NDArray[np.float32]],
) -> tuple[AgentRuntime, AudioIO]:
    settings = get_settings()

    store = ConversationStore(system_prompt=settings.primary_mind_system_prompt)
    message_log = FileMessageLog(log_dir=settings.log_dir)
    audio_io = AudioIO(sample_rate=settings.sample_rate)

    primary_mind = create_mind(settings, role=MindRole.PRIMARY)
    inner_voice = create_mind(settings, role=MindRole.INNER_VOICE)

    stt = LocalWhisperSTT(model_size=settings.whisper_model_size)
    tts = KokoroTTS(sample_rate=settings.sample_rate)
    vad = SileroVAD(threshold=settings.vad_threshold, sample_rate=settings.sample_rate)

    workers = [
        AudioIngestLoop(
            store=store,
            stt=stt,
            vad=vad,
            message_log=message_log,
            min_silence_ms=settings.vad_min_silence_ms,
            chunk_queue=chunk_queue,
        ),
        PrimaryThinkingLoop(
            store=store,
            mind=primary_mind,
            message_log=message_log,
            tts=tts,
            audio_player=audio_io,
            context_trim_count=settings.mind_context_trim_count,
        ),
        InnerVoiceLoop(
            store=store,
            mind=inner_voice,
            message_log=message_log,
            system_prompt=settings.inner_voice_system_prompt,
            context_trim_count=settings.mind_context_trim_count,
        ),
    ]
    return AgentRuntime(workers), audio_io


async def main() -> None:
    settings = get_settings()
    chunk_queue: asyncio.Queue[npt.NDArray[np.float32]] = asyncio.Queue()
    runtime, audio_io = build_runtime(chunk_queue)
    loop = asyncio.get_running_loop()

    await runtime.start()
    logger.info("Voice agent initialized (sample_rate=%s)", settings.sample_rate)

    def on_chunk(chunk: npt.NDArray[np.float32]) -> None:
        loop.call_soon_threadsafe(chunk_queue.put_nowait, chunk)

    try:
        await audio_io.stream_input(chunk_size=512, on_chunk=on_chunk)
    finally:
        await runtime.stop()


if __name__ == "__main__":
    asyncio.run(main())
