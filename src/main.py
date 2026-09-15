import asyncio
import logging

import numpy as np
import numpy.typing as npt

from config.settings import Settings, get_settings
from config.types import MindRole
from src.application.inner_think import InnerThinkService
from src.application.interfaces import LoopWorker, MessageLog
from src.application.loops import AudioIngestLoop, InnerVoiceLoop, PrimaryThinkingLoop
from src.application.runtime import AgentRuntime
from src.domain.conversation import ConversationStore
from src.infrastructure.audio_io import AudioIO
from src.infrastructure.detector import SileroVAD
from src.infrastructure.logging import CompositeMessageLog, FileMessageLog
from src.infrastructure.mind import create_mind
from src.infrastructure.monitor import BroadcastMessageLog, MonitorHub, run_monitor
from src.infrastructure.stt.whisper import LocalWhisperSTT
from src.infrastructure.tts.factory import create_tts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_runtime(
    chunk_queue: asyncio.Queue[npt.NDArray[np.float32]],
    settings: Settings,
    store: ConversationStore,
    message_log: MessageLog,
) -> tuple[AgentRuntime, AudioIO]:
    audio_io = AudioIO(sample_rate=settings.sample_rate)

    primary_mind = create_mind(settings, role=MindRole.PRIMARY)
    inner_voice = create_mind(settings, role=MindRole.INNER_VOICE)
    inner_think = InnerThinkService(
        store=store,
        mind=inner_voice,
        message_log=message_log,
        system_prompt=settings.inner_voice_system_prompt,
    )

    stt = LocalWhisperSTT(model_size=settings.whisper_model_size)
    tts = create_tts(settings)
    vad = SileroVAD(threshold=settings.vad_threshold, sample_rate=settings.sample_rate)

    workers: list[LoopWorker] = [
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
            inner_think=inner_think,
            context_trim_count=settings.mind_context_trim_count,
        ),
        InnerVoiceLoop(
            store=store,
            inner_think=inner_think,
            interval_seconds=settings.inner_voice_interval_seconds,
            context_trim_count=settings.mind_context_trim_count,
        ),
    ]
    return AgentRuntime(workers), audio_io


async def main() -> None:
    settings = get_settings()
    chunk_queue: asyncio.Queue[npt.NDArray[np.float32]] = asyncio.Queue()

    hub = MonitorHub()
    store = ConversationStore(system_prompt=settings.primary_mind_system_prompt)
    message_log = CompositeMessageLog(
        FileMessageLog(log_dir=settings.log_dir),
        BroadcastMessageLog(hub),
    )

    runtime, audio_io = build_runtime(chunk_queue, settings, store, message_log)
    loop = asyncio.get_running_loop()

    await runtime.start()
    ui_task = asyncio.create_task(
        run_monitor(hub, store, host=settings.ui_host, port=settings.ui_port),
        name="monitor-ui",
    )
    logger.info("Voice agent initialized (sample_rate=%s)", settings.sample_rate)

    def on_chunk(chunk: npt.NDArray[np.float32]) -> None:
        loop.call_soon_threadsafe(chunk_queue.put_nowait, chunk)

    try:
        await audio_io.stream_input(chunk_size=512, on_chunk=on_chunk)
    finally:
        ui_task.cancel()
        await asyncio.gather(ui_task, return_exceptions=True)
        await runtime.stop()


if __name__ == "__main__":
    asyncio.run(main())
