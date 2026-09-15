import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from src.application.loops.primary import PrimaryThinkingLoop
from src.application.loops.sleep import SleepConsolidationLoop
from src.application.memory_context import build_recall_query, inject_memory_passages
from src.domain.conversation import ConversationStore
from src.domain.messages import MessageRole
from src.infrastructure.memory.episode_store import EpisodeStore
from src.infrastructure.memory.hipporag_memory import (
    HippoRAGLongTermMemory,
    parse_durable_facts,
)
from src.infrastructure.memory.message_sink import MemoryMessageLog
from src.infrastructure.memory.noop import NoOpLongTermMemory


class FakeMessageLog:
    def __init__(self) -> None:
        self.entries: list[tuple[str, str]] = []

    async def append(self, role: str, content: str) -> None:
        self.entries.append((role, content))


class FakeMemory:
    def __init__(self) -> None:
        self.remembered: list[tuple[str, str]] = []
        self.consolidate_calls = 0
        self.recall_queries: list[str] = []
        self.passages: list[str] = ["Agent name is Comrade Major"]

    async def remember(self, role: str, content: str) -> None:
        self.remembered.append((role, content))

    async def consolidate(self) -> int:
        self.consolidate_calls += 1
        return 3

    async def recall(self, query: str, limit: int) -> list[str]:
        self.recall_queries.append(query)
        return self.passages[:limit]


class ScriptedMind:
    def __init__(self, replies: list[str] | None = None) -> None:
        self.replies = list(replies or ["thought"])
        self.histories: list[list[dict[str, str]]] = []

    async def think(self, history: list[dict[str, str]]) -> str:
        self.histories.append(history)
        if self.replies:
            return self.replies.pop(0)
        return "thought"

    async def stream(self, history: list[dict[str, str]]) -> AsyncIterator[str]:
        yield await self.think(history)


class FakeTTS:
    async def synthesize(self, text: str) -> npt.NDArray[np.float32]:
        return np.zeros(4, dtype=np.float32)


class FakePlayer:
    async def play(self, audio: npt.NDArray[np.float32]) -> None:
        return None


class FakeGate:
    def __init__(self, listening: bool = True) -> None:
        self._listening = listening

    def is_listening(self) -> bool:
        return self._listening

    def register_on_speak_start(self, callback: object) -> None:
        return None


def test_parse_durable_facts() -> None:
    assert parse_durable_facts("NONE") == []
    assert parse_durable_facts("1. Name is Major\n- Likes tea") == [
        "Name is Major",
        "Likes tea",
    ]


def test_inject_memory_passages() -> None:
    history = [
        {"role": "system", "content": "sys"},
        {"role": "system", "content": "[INNER]\nnote"},
        {"role": "user", "content": "hi"},
    ]
    out = inject_memory_passages(history, ["Agent is Major"])
    assert out[2]["content"].startswith("[MEMORY]")
    assert "Agent is Major" in out[2]["content"]
    assert out[3]["role"] == "user"


@pytest.mark.asyncio
async def test_build_recall_query() -> None:
    store = ConversationStore(system_prompt="sys")
    await store.append(MessageRole.USER, "меня зовут Иван")
    await store.append_private_thought("запомнить имя")
    query = await build_recall_query(store)
    assert "user: меня зовут Иван" in query
    assert "thought: запомнить имя" in query


@pytest.mark.asyncio
async def test_episode_store_pending_and_mark(tmp_path: Path) -> None:
    store = EpisodeStore(tmp_path / "episodes.db")
    await store.append("user", "hello")
    await store.append("assistant", "say hi")
    pending = await store.fetch_pending(10)
    assert len(pending) == 2
    await store.mark_consolidated([pending[0].id])
    remaining = await store.fetch_pending(10)
    assert len(remaining) == 1
    assert remaining[0].role == "assistant"


@pytest.mark.asyncio
async def test_memory_message_log_filters_roles() -> None:
    memory = FakeMemory()
    sink = MemoryMessageLog(memory)
    await sink.append("user", "hi")
    await sink.append("sleep", "consolidated 1")
    await sink.append("assistant", "say ok")
    assert memory.remembered == [("user", "hi"), ("assistant", "say ok")]


@pytest.mark.asyncio
async def test_primary_injects_memory_into_history() -> None:
    store = ConversationStore(system_prompt="sys")
    await store.append(MessageRole.USER, "привет")
    memory = FakeMemory()
    mind = ScriptedMind(replies=["private beat"])
    loop = PrimaryThinkingLoop(
        store=store,
        mind=mind,
        message_log=FakeMessageLog(),
        tts=FakeTTS(),
        audio_player=FakePlayer(),
        memory=memory,
        pulse_seconds=0.05,
        memory_recall_limit=3,
    )
    task = asyncio.create_task(loop.run())
    await asyncio.sleep(0.12)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert memory.recall_queries
    assert mind.histories
    memory_blocks = [
        m for m in mind.histories[0] if m.get("content", "").startswith("[MEMORY]")
    ]
    assert memory_blocks
    assert "Comrade Major" in memory_blocks[0]["content"]


@pytest.mark.asyncio
async def test_sleep_loop_waits_for_idle() -> None:
    store = ConversationStore(system_prompt="sys")
    await store.append(MessageRole.USER, "just now")
    memory = FakeMemory()
    log = FakeMessageLog()
    loop = SleepConsolidationLoop(
        store=store,
        memory=memory,
        message_log=log,
        idle_seconds=60.0,
        poll_seconds=0.05,
        listening_gate=FakeGate(True),
    )
    task = asyncio.create_task(loop.run())
    await asyncio.sleep(0.15)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert memory.consolidate_calls == 0


@pytest.mark.asyncio
async def test_sleep_loop_consolidates_when_idle() -> None:
    store = ConversationStore(system_prompt="sys")
    memory = FakeMemory()
    log = FakeMessageLog()
    loop = SleepConsolidationLoop(
        store=store,
        memory=memory,
        message_log=log,
        idle_seconds=0.01,
        poll_seconds=0.05,
        listening_gate=FakeGate(True),
        error_backoff_seconds=0.05,
    )
    task = asyncio.create_task(loop.run())
    for _ in range(40):
        if memory.consolidate_calls >= 1:
            break
        await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert memory.consolidate_calls >= 1
    assert any(role == "sleep" for role, _ in log.entries)


@pytest.mark.asyncio
async def test_consolidate_marks_even_without_facts(tmp_path: Path) -> None:
    episodes = EpisodeStore(tmp_path / "e.db")
    await episodes.append("user", "привет")
    mind = ScriptedMind(replies=["NONE"])
    memory = HippoRAGLongTermMemory(
        episodes=episodes,
        consolidator=mind,
        consolidate_system_prompt="extract",
        save_dir=tmp_path / "mem",
        llm_model_name="gpt-4o-mini",
        llm_base_url="https://api.openai.com/v1",
        api_key="test-key",
        embedding_model_name="text-embedding-3-small",
        embedding_base_url="https://api.openai.com/v1",
        consolidate_batch=10,
    )
    processed = await memory.consolidate()
    assert processed == 1
    assert await episodes.fetch_pending(10) == []


@pytest.mark.asyncio
async def test_noop_memory() -> None:
    mem = NoOpLongTermMemory()
    await mem.remember("user", "x")
    assert await mem.consolidate() == 0
    assert await mem.recall("q", 3) == []
