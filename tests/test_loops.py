import asyncio
from collections.abc import AsyncIterator

import numpy as np
import numpy.typing as npt
import pytest

from src.application.loops.inner_voice import InnerVoiceLoop
from src.application.loops.primary import PrimaryThinkingLoop
from src.application.runtime import AgentRuntime
from src.domain.conversation import ConversationStore
from src.domain.messages import MessageRole


class FakeMessageLog:
    def __init__(self) -> None:
        self.entries: list[tuple[str, str]] = []

    async def append(self, role: str, content: str) -> None:
        self.entries.append((role, content))


class ScriptedMind:
    def __init__(self, replies: list[str] | None = None, error: Exception | None = None) -> None:
        self.replies = list(replies or [])
        self.error = error
        self.calls = 0

    async def think(self, history: list[dict[str, str]]) -> str:
        self.calls += 1
        if self.error is not None and self.calls == 1:
            raise self.error
        if not self.replies:
            await asyncio.sleep(0.01)
            return "thought"
        return self.replies.pop(0)

    async def stream(self, history: list[dict[str, str]]) -> AsyncIterator[str]:
        yield await self.think(history)


class FakeTTS:
    def __init__(self) -> None:
        self.spoken: list[str] = []

    async def synthesize(self, text: str) -> npt.NDArray[np.float32]:
        self.spoken.append(text)
        return np.zeros(8, dtype=np.float32)


class FakePlayer:
    def __init__(self) -> None:
        self.played = 0

    async def play(self, audio: npt.NDArray[np.float32]) -> None:
        self.played += 1


@pytest.mark.asyncio
async def test_primary_say_triggers_tts() -> None:
    store = ConversationStore(system_prompt="sys")
    log = FakeMessageLog()
    tts = FakeTTS()
    player = FakePlayer()
    mind = ScriptedMind(replies=["say hello there", "silent"])

    loop = PrimaryThinkingLoop(
        store=store,
        mind=mind,
        message_log=log,
        tts=tts,
        audio_player=player,
    )
    task = asyncio.create_task(loop.run())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await asyncio.sleep(0.05)
    assert "hello there" in tts.spoken
    assert player.played >= 1
    snap = await store.snapshot()
    assert any(m.role == MessageRole.ASSISTANT and m.content.startswith("say ") for m in snap)


@pytest.mark.asyncio
async def test_primary_trims_on_context_overflow() -> None:
    store = ConversationStore(system_prompt="sys")
    await store.append(MessageRole.USER, "old")
    await store.append(MessageRole.ASSISTANT, "older")
    log = FakeMessageLog()
    mind = ScriptedMind(
        replies=["ok after trim"],
        error=RuntimeError("maximum context length exceeded"),
    )
    loop = PrimaryThinkingLoop(
        store=store,
        mind=mind,
        message_log=log,
        tts=FakeTTS(),
        audio_player=FakePlayer(),
        context_trim_count=1,
    )
    task = asyncio.create_task(loop.run())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert mind.calls >= 2
    contents = [m.content for m in await store.snapshot()]
    assert "old" not in contents or "older" not in contents
    assert "ok after trim" in contents


@pytest.mark.asyncio
async def test_runtime_primary_and_inner_parallel() -> None:
    store = ConversationStore(system_prompt="sys")
    log = FakeMessageLog()
    primary = ScriptedMind(replies=["primary-1", "primary-2", "primary-3"])
    inner = ScriptedMind(replies=["note-a", "note-b", "note-c"])

    runtime = AgentRuntime(
        [
            PrimaryThinkingLoop(
                store=store,
                mind=primary,
                message_log=log,
                tts=FakeTTS(),
                audio_player=FakePlayer(),
            ),
            InnerVoiceLoop(
                store=store,
                mind=inner,
                message_log=log,
                system_prompt="inner sys",
            ),
        ]
    )
    await runtime.start()
    await asyncio.sleep(0.1)
    await runtime.stop()

    assert primary.calls >= 1
    assert inner.calls >= 1
    roles = {m.role for m in await store.snapshot()}
    assert MessageRole.ASSISTANT in roles
    assert MessageRole.INNER in roles
    chat = await store.snapshot_chat()
    assert any(m["content"].startswith("[inner voice]") for m in chat)
