import asyncio
import json
import logging
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from src.application.interfaces import MessageLog

logger = logging.getLogger(__name__)


class FileMessageLog:
    """Append-only JSONL message log. Swap for a DB-backed MessageLog later."""

    def __init__(self, log_dir: str, filename: str = "messages.jsonl") -> None:
        self._path = Path(log_dir) / filename
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    async def append(self, role: str, content: str) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "role": role,
            "content": content,
        }
        line = json.dumps(record, ensure_ascii=False) + "\n"
        async with self._lock:
            await asyncio.to_thread(self._write_line, line)

    def _write_line(self, line: str) -> None:
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(line)


class CompositeMessageLog:
    """Fan-out append to multiple MessageLog sinks."""

    def __init__(self, *logs: MessageLog) -> None:
        self._logs: Sequence[MessageLog] = logs

    async def append(self, role: str, content: str) -> None:
        for log in self._logs:
            await log.append(role, content)
