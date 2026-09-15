import asyncio
import json
from pathlib import Path

import pytest

from src.domain.conversation import ConversationStore
from src.domain.messages import MessageRole
from src.infrastructure.logging.message_log import CompositeMessageLog, FileMessageLog
from src.infrastructure.monitor.hub import (
    BroadcastMessageLog,
    MonitorHub,
    context_fingerprint,
    poll_context,
)


class RecordingLog:
    def __init__(self) -> None:
        self.entries: list[tuple[str, str]] = []

    async def append(self, role: str, content: str) -> None:
        self.entries.append((role, content))


@pytest.mark.asyncio
async def test_file_message_log_appends_jsonl(tmp_path: Path) -> None:
    log = FileMessageLog(log_dir=str(tmp_path), filename="m.jsonl")
    await log.append("user", "hi")
    await log.append("assistant", "hello")

    lines = (tmp_path / "m.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["role"] == "user"
    assert first["content"] == "hi"
    assert "ts" in first


@pytest.mark.asyncio
async def test_composite_message_log_fans_out() -> None:
    first = RecordingLog()
    second = RecordingLog()
    composite = CompositeMessageLog(first, second)
    await composite.append("user", "hi")
    assert first.entries == [("user", "hi")]
    assert second.entries == [("user", "hi")]


@pytest.mark.asyncio
async def test_broadcast_message_log_publishes_to_hub() -> None:
    hub = MonitorHub()
    received: list[dict[str, object]] = []

    async def capture(event: dict[str, object]) -> None:
        received.append(event)

    await hub.subscribe(capture)
    log = BroadcastMessageLog(hub)
    await log.append("assistant", "thinking")

    assert len(hub.log_buffer) == 1
    assert hub.log_buffer[0]["role"] == "assistant"
    assert hub.log_buffer[0]["content"] == "thinking"
    assert received[0]["type"] == "log"


@pytest.mark.asyncio
async def test_poll_context_publishes_on_change() -> None:
    store = ConversationStore(system_prompt="sys")
    hub = MonitorHub()
    received: list[dict[str, object]] = []

    async def capture(event: dict[str, object]) -> None:
        received.append(event)

    await hub.subscribe(capture)

    await store.append(MessageRole.USER, "hello")
    messages = await store.snapshot()
    assert context_fingerprint(messages)[0] == 2

    task = asyncio.create_task(poll_context(store, hub, interval_seconds=0.05))
    await asyncio.sleep(0.12)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    context_events = [e for e in received if e.get("type") == "context"]
    assert context_events
    assert context_events[-1]["messages"][-1] == {"role": "user", "content": "hello"}


@pytest.mark.asyncio
async def test_poll_context_includes_inner_slot() -> None:
    store = ConversationStore(system_prompt="sys")
    hub = MonitorHub()
    received: list[dict[str, object]] = []

    async def capture(event: dict[str, object]) -> None:
        received.append(event)

    await hub.subscribe(capture)
    await store.set_inner_context("note")

    task = asyncio.create_task(poll_context(store, hub, interval_seconds=0.05))
    await asyncio.sleep(0.12)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    context_events = [e for e in received if e.get("type") == "context"]
    assert context_events
    messages = context_events[-1]["messages"]
    assert {"role": "inner", "content": "note"} in messages
