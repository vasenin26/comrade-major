import asyncio
import logging
from collections.abc import Sequence

from src.application.interfaces import LoopWorker

logger = logging.getLogger(__name__)


class AgentRuntime:
    """Starts independent loop workers as asyncio tasks."""

    def __init__(self, workers: Sequence[LoopWorker]) -> None:
        self._workers = list(workers)
        self._tasks: list[asyncio.Task[None]] = []

    async def start(self) -> None:
        if self._tasks:
            raise RuntimeError("AgentRuntime already started")
        for worker in self._workers:
            self._tasks.append(asyncio.create_task(worker.run(), name=type(worker).__name__))
        logger.info("AgentRuntime started %s workers", len(self._tasks))

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("AgentRuntime stopped")

    async def run_forever(self) -> None:
        await self.start()
        try:
            await asyncio.gather(*self._tasks)
        finally:
            await self.stop()
