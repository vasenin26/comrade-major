import json
from pathlib import Path

import pytest

from src.infrastructure.logging.message_log import FileMessageLog


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
