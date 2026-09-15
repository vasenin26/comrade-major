import asyncio

import pytest

from src.application.listening_gate import ListeningGate


@pytest.mark.asyncio
async def test_speaking_disables_listening_and_runs_hangover() -> None:
    gate = ListeningGate(hangover_ms=50)
    assert gate.is_listening()
    async with gate.speaking():
        assert not gate.is_listening()
    assert gate.is_listening()


@pytest.mark.asyncio
async def test_speak_start_callbacks() -> None:
    gate = ListeningGate(hangover_ms=0)
    calls = 0

    def on_start() -> None:
        nonlocal calls
        calls += 1

    gate.register_on_speak_start(on_start)
    async with gate.speaking():
        assert calls == 1
    assert calls == 1


@pytest.mark.asyncio
async def test_speaking_lock_serializes() -> None:
    gate = ListeningGate(hangover_ms=0)
    order: list[str] = []

    async def first() -> None:
        async with gate.speaking():
            order.append("a-enter")
            await asyncio.sleep(0.05)
            order.append("a-exit")

    async def second() -> None:
        await asyncio.sleep(0.01)
        async with gate.speaking():
            order.append("b-enter")
            order.append("b-exit")

    await asyncio.gather(first(), second())
    assert order == ["a-enter", "a-exit", "b-enter", "b-exit"]
