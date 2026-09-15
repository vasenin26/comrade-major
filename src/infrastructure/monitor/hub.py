from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

from src.domain.conversation import ConversationStore
from src.domain.messages import Message, MessageRole

Subscriber = Callable[[dict[str, Any]], Awaitable[None]]


class MonitorHub:
    """In-memory fan-out for monitor WebSocket clients."""

    def __init__(self, log_buffer_size: int = 200) -> None:
        self._subscribers: set[Subscriber] = set()
        self._log_buffer: deque[dict[str, Any]] = deque(maxlen=log_buffer_size)
        self._context: list[dict[str, str]] = []
        self._lock = asyncio.Lock()

    async def subscribe(self, subscriber: Subscriber) -> None:
        async with self._lock:
            self._subscribers.add(subscriber)

    async def unsubscribe(self, subscriber: Subscriber) -> None:
        async with self._lock:
            self._subscribers.discard(subscriber)

    async def publish_log(self, role: str, content: str, ts: str | None = None) -> None:
        event: dict[str, Any] = {
            "type": "log",
            "ts": ts or datetime.now(timezone.utc).isoformat(),
            "role": role,
            "content": content,
        }
        async with self._lock:
            self._log_buffer.append(event)
            subscribers = list(self._subscribers)
        await self._fan_out(subscribers, event)

    async def publish_context(self, messages: list[dict[str, str]]) -> None:
        event: dict[str, Any] = {"type": "context", "messages": messages}
        async with self._lock:
            self._context = messages
            subscribers = list(self._subscribers)
        await self._fan_out(subscribers, event)

    async def snapshot_for_client(self) -> list[dict[str, Any]]:
        async with self._lock:
            events: list[dict[str, Any]] = [
                {"type": "context", "messages": list(self._context)}
            ]
            events.extend(self._log_buffer)
            return events

    @property
    def log_buffer(self) -> list[dict[str, Any]]:
        return list(self._log_buffer)

    @property
    def context(self) -> list[dict[str, str]]:
        return list(self._context)

    async def _fan_out(
        self, subscribers: list[Subscriber], event: dict[str, Any]
    ) -> None:
        if not subscribers:
            return
        results = await asyncio.gather(
            *(subscriber(event) for subscriber in subscribers),
            return_exceptions=True,
        )
        for subscriber, result in zip(subscribers, results, strict=True):
            if isinstance(result, Exception):
                await self.unsubscribe(subscriber)


class BroadcastMessageLog:
    """MessageLog sink that publishes to MonitorHub."""

    def __init__(self, hub: MonitorHub) -> None:
        self._hub = hub

    async def append(self, role: str, content: str) -> None:
        await self._hub.publish_log(role, content)


def context_fingerprint(
    messages: list[Message], inner_context: str = ""
) -> tuple[int, str, str, str]:
    if not messages:
        return (0, "", "", inner_context)
    last = messages[-1]
    return (len(messages), last.role.value, last.content, inner_context)


async def poll_context(
    store: ConversationStore,
    hub: MonitorHub,
    interval_seconds: float = 0.5,
) -> None:
    """Push ConversationStore snapshots when they change (includes INNER slot)."""
    last_fp: tuple[int, str, str, str] | None = None
    while True:
        messages = await store.snapshot()
        inner = await store.get_inner_context()
        fp = context_fingerprint(messages, inner)
        if fp != last_fp:
            payload = [{"role": m.role.value, "content": m.content} for m in messages]
            if inner:
                # Place INNER after the first system message for monitor display
                insert_at = 1 if payload and payload[0]["role"] == MessageRole.SYSTEM.value else 0
                payload.insert(
                    insert_at,
                    {"role": MessageRole.INNER.value, "content": inner},
                )
            await hub.publish_context(payload)
            last_fp = fp
        await asyncio.sleep(interval_seconds)
