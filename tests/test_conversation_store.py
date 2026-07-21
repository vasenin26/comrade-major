import asyncio

import pytest

from src.domain.conversation import ConversationStore
from src.domain.messages import MessageRole


@pytest.mark.asyncio
async def test_snapshot_is_isolated_copy() -> None:
    store = ConversationStore(system_prompt="sys")
    await store.append(MessageRole.USER, "hi")
    snap = await store.snapshot()
    snap.append(snap[0])  # mutate copy
    assert await store.length() == 2


@pytest.mark.asyncio
async def test_drop_oldest_keeps_system() -> None:
    store = ConversationStore(system_prompt="sys")
    await store.append(MessageRole.USER, "u1")
    await store.append(MessageRole.ASSISTANT, "a1")
    await store.append(MessageRole.USER, "u2")
    removed = await store.drop_oldest(2)
    assert [m.content for m in removed] == ["u1", "a1"]
    remaining = await store.snapshot()
    assert [m.content for m in remaining] == ["sys", "u2"]


@pytest.mark.asyncio
async def test_concurrent_appends() -> None:
    store = ConversationStore()

    async def add(i: int) -> None:
        await store.append(MessageRole.USER, f"m{i}")

    await asyncio.gather(*(add(i) for i in range(20)))
    assert await store.length() == 20
