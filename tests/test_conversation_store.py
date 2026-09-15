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


@pytest.mark.asyncio
async def test_private_thoughts_rolling_in_snapshot_chat() -> None:
    store = ConversationStore(system_prompt="sys", thought_history=2)
    await store.append_private_thought("t1")
    await store.append_private_thought("t2")
    await store.append_private_thought("t3")
    assert await store.private_thoughts() == ["t2", "t3"]
    chat = await store.snapshot_chat()
    assert chat[0] == {"role": "system", "content": "sys"}
    thoughts = [m for m in chat if m["content"].startswith("[THOUGHTS]\n")]
    assert len(thoughts) == 1
    assert thoughts[0]["content"] == "[THOUGHTS]\nt2\n---\nt3"
    assert MessageRole.ASSISTANT not in {m.role for m in await store.snapshot()}


@pytest.mark.asyncio
async def test_revision_bumps_on_user_and_inner_not_on_private_thought() -> None:
    store = ConversationStore(system_prompt="sys")
    assert await store.revision() == 0
    await store.append(MessageRole.USER, "hi")
    rev_user = await store.revision()
    assert rev_user == 1
    await store.append_private_thought("secret")
    assert await store.revision() == rev_user
    await store.set_inner_context("note")
    assert await store.revision() == rev_user + 1


@pytest.mark.asyncio
async def test_wait_until_changed_pulse_timeout() -> None:
    store = ConversationStore()
    known = await store.revision()
    started = asyncio.get_running_loop().time()
    rev = await store.wait_until_changed(known, timeout=0.05)
    elapsed = asyncio.get_running_loop().time() - started
    assert rev == known
    assert elapsed >= 0.04


@pytest.mark.asyncio
async def test_wait_until_changed_wakes_on_user() -> None:
    store = ConversationStore()
    known = await store.revision()

    async def append_soon() -> None:
        await asyncio.sleep(0.02)
        await store.append(MessageRole.USER, "ping")

    task = asyncio.create_task(append_soon())
    rev = await store.wait_until_changed(known, timeout=1.0)
    await task
    assert rev > known
