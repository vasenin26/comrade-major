from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager


class ListeningGate:
    """Half-duplex gate: listening is off while speaking (+ hangover)."""

    def __init__(self, hangover_ms: int = 400) -> None:
        self._hangover_sec = max(0, hangover_ms) / 1000.0
        self._speak_lock = asyncio.Lock()
        self._speaking = False
        self._on_speak_start: list[Callable[[], None]] = []

    def is_listening(self) -> bool:
        return not self._speaking

    def register_on_speak_start(self, callback: Callable[[], None]) -> None:
        self._on_speak_start.append(callback)

    @asynccontextmanager
    async def speaking(self) -> AsyncIterator[None]:
        async with self._speak_lock:
            self._speaking = True
            for callback in list(self._on_speak_start):
                callback()
            try:
                yield
            finally:
                if self._hangover_sec > 0:
                    await asyncio.sleep(self._hangover_sec)
                self._speaking = False
