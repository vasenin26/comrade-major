import asyncio
from collections.abc import AsyncIterator

import numpy as np
import numpy.typing as npt
import pytest

from src.application.inner_think import InnerThinkService
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
        self.histories: list[list[dict[str, str]]] = []

    async def think(self, history: list[dict[str, str]]) -> str:
        self.calls += 1
        self.histories.append(history)
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
    def __init__(self, delay: float = 0.0) -> None:
        self.played = 0
        self.delay = delay
        self.play_started = asyncio.Event()
        self.in_play = False

    async def play(self, audio: npt.NDArray[np.float32]) -> None:
        self.in_play = True
        self.play_started.set()
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        self.played += 1
        self.in_play = False


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
        pulse_seconds=0.05,
    )
    task = asyncio.create_task(loop.run())
    await asyncio.sleep(0.12)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await asyncio.sleep(0.05)
    assert "hello there" in tts.spoken
    assert player.played >= 1
    snap = await store.snapshot()
    assert any(m.role == MessageRole.ASSISTANT and m.content.startswith("say ") for m in snap)


@pytest.mark.asyncio
async def test_primary_awaits_play_before_next_think() -> None:
    store = ConversationStore(system_prompt="sys")
    log = FakeMessageLog()
    tts = FakeTTS()
    player = FakePlayer(delay=0.1)
    mind = ScriptedMind(replies=["say first", "second-thought"])

    loop = PrimaryThinkingLoop(
        store=store,
        mind=mind,
        message_log=log,
        tts=tts,
        audio_player=player,
        pulse_seconds=0.05,
    )
    task = asyncio.create_task(loop.run())
    await player.play_started.wait()
    # While play is in progress, next think must not have started yet
    await asyncio.sleep(0.02)
    assert mind.calls == 1
    assert player.in_play
    await asyncio.sleep(0.15)
    assert mind.calls >= 2
    assert player.played >= 1
    assert "second-thought" in await store.private_thoughts()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_primary_private_reply_goes_to_thoughts_not_assistant() -> None:
    store = ConversationStore(system_prompt="sys")
    log = FakeMessageLog()
    mind = ScriptedMind(replies=["размышляю о тишине"])

    loop = PrimaryThinkingLoop(
        store=store,
        mind=mind,
        message_log=log,
        tts=FakeTTS(),
        audio_player=FakePlayer(),
        pulse_seconds=0.05,
    )
    task = asyncio.create_task(loop.run())
    await asyncio.sleep(0.12)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert MessageRole.ASSISTANT not in {m.role for m in await store.snapshot()}
    assert "размышляю о тишине" in await store.private_thoughts()
    assert ("thought", "размышляю о тишине") in log.entries
    chat = await store.snapshot_chat()
    assert any(m["content"].startswith("[THOUGHTS]\n") for m in chat)


@pytest.mark.asyncio
async def test_primary_think_tool_waits_and_updates_inner_slot() -> None:
    store = ConversationStore(system_prompt="sys")
    log = FakeMessageLog()
    tts = FakeTTS()
    primary = ScriptedMind(replies=["think how to answer", "say готово"])
    inner = ScriptedMind(replies=["нужно ответить кратко"])
    inner_think = InnerThinkService(
        store=store,
        mind=inner,
        message_log=log,
        system_prompt="inner sys",
    )
    loop = PrimaryThinkingLoop(
        store=store,
        mind=primary,
        message_log=log,
        tts=tts,
        audio_player=FakePlayer(),
        inner_think=inner_think,
        pulse_seconds=0.05,
    )
    task = asyncio.create_task(loop.run())
    await asyncio.sleep(0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert primary.calls >= 2
    assert inner.calls >= 1
    assert await store.get_inner_context() == "нужно ответить кратко"
    roles = {m.role for m in await store.snapshot()}
    assert MessageRole.INNER not in roles
    assert MessageRole.ASSISTANT in roles
    chat = await store.snapshot_chat()
    assert any(m["content"].startswith("[INNER]\n") for m in chat)
    assert "готово" in tts.spoken
    assert ("inner", "нужно ответить кратко") in log.entries


@pytest.mark.asyncio
async def test_primary_trims_on_context_overflow() -> None:
    store = ConversationStore(system_prompt="sys")
    await store.append(MessageRole.USER, "old")
    await store.append(MessageRole.ASSISTANT, "older")
    log = FakeMessageLog()
    mind = ScriptedMind(
        replies=["say ok after trim"],
        error=RuntimeError("maximum context length exceeded"),
    )
    loop = PrimaryThinkingLoop(
        store=store,
        mind=mind,
        message_log=log,
        tts=FakeTTS(),
        audio_player=FakePlayer(),
        context_trim_count=1,
        pulse_seconds=0.05,
    )
    task = asyncio.create_task(loop.run())
    await asyncio.sleep(0.15)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert mind.calls >= 2
    contents = [m.content for m in await store.snapshot()]
    assert "old" not in contents or "older" not in contents
    assert any(c.startswith("say ok after trim") for c in contents)


@pytest.mark.asyncio
async def test_primary_pulse_limits_calls_without_user() -> None:
    store = ConversationStore(system_prompt="sys")
    log = FakeMessageLog()
    mind = ScriptedMind(replies=["a", "b", "c", "d", "e", "f", "g", "h"])
    loop = PrimaryThinkingLoop(
        store=store,
        mind=mind,
        message_log=log,
        tts=FakeTTS(),
        audio_player=FakePlayer(),
        pulse_seconds=0.08,
    )
    task = asyncio.create_task(loop.run())
    await asyncio.sleep(0.2)
    calls = mind.calls
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # ~0.2s / 0.08s pulse ⇒ a few thinks, not a tight-loop flood
    assert 1 <= calls <= 5


@pytest.mark.asyncio
async def test_primary_wakes_quickly_on_new_user() -> None:
    store = ConversationStore(system_prompt="sys")
    log = FakeMessageLog()
    mind = ScriptedMind(replies=["first", "after-user", "more"])
    loop = PrimaryThinkingLoop(
        store=store,
        mind=mind,
        message_log=log,
        tts=FakeTTS(),
        audio_player=FakePlayer(),
        pulse_seconds=2.0,
    )
    task = asyncio.create_task(loop.run())
    await asyncio.sleep(0.05)
    calls_before = mind.calls
    await store.append(MessageRole.USER, "привет")
    await asyncio.sleep(0.1)
    calls_after = mind.calls
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert calls_before <= 1
    assert calls_after > calls_before


@pytest.mark.asyncio
async def test_runtime_primary_and_inner_parallel() -> None:
    store = ConversationStore(system_prompt="sys")
    log = FakeMessageLog()
    primary = ScriptedMind(replies=["primary-1", "primary-2", "primary-3"])
    inner = ScriptedMind(replies=["note-a", "note-b", "note-c"])
    inner_think = InnerThinkService(
        store=store,
        mind=inner,
        message_log=log,
        system_prompt="inner sys",
    )

    runtime = AgentRuntime(
        [
            PrimaryThinkingLoop(
                store=store,
                mind=primary,
                message_log=log,
                tts=FakeTTS(),
                audio_player=FakePlayer(),
                inner_think=inner_think,
                pulse_seconds=0.05,
            ),
            InnerVoiceLoop(
                store=store,
                inner_think=inner_think,
                interval_seconds=0.05,
            ),
        ]
    )
    await runtime.start()
    await asyncio.sleep(0.2)
    await runtime.stop()

    assert primary.calls >= 1
    assert inner.calls >= 1
    roles = {m.role for m in await store.snapshot()}
    assert MessageRole.INNER not in roles
    assert await store.get_inner_context() != ""
    assert await store.private_thoughts()
    chat = await store.snapshot_chat()
    assert any(m["content"].startswith("[INNER]\n") for m in chat)
    assert any(m["content"].startswith("[THOUGHTS]\n") for m in chat)


@pytest.mark.asyncio
async def test_inner_voice_loop_interval_between_ponders() -> None:
    store = ConversationStore(system_prompt="sys")
    log = FakeMessageLog()
    inner = ScriptedMind(replies=["a", "b", "c"])
    inner_think = InnerThinkService(
        store=store,
        mind=inner,
        message_log=log,
        system_prompt="inner sys",
    )
    loop = InnerVoiceLoop(
        store=store,
        inner_think=inner_think,
        interval_seconds=0.08,
    )
    task = asyncio.create_task(loop.run())
    await asyncio.sleep(0.05)
    calls_early = inner.calls
    await asyncio.sleep(0.12)
    calls_late = inner.calls
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert calls_early == 1
    assert calls_late >= 2
    assert MessageRole.INNER not in {m.role for m in await store.snapshot()}
