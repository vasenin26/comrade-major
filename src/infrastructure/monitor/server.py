from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from uvicorn import Config, Server

from src.domain.conversation import ConversationStore
from src.infrastructure.monitor.hub import MonitorHub, poll_context

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_monitor_app(hub: MonitorHub) -> FastAPI:
    app = FastAPI(title="Voice Agent Monitor")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()

        async def send_event(event: dict[str, object]) -> None:
            await websocket.send_json(event)

        for event in await hub.snapshot_for_client():
            await send_event(event)

        await hub.subscribe(send_event)
        try:
            while True:
                # Keep the socket open; client does not need to send.
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            await hub.unsubscribe(send_event)

    return app


async def run_monitor(
    hub: MonitorHub,
    store: ConversationStore,
    host: str,
    port: int,
    poll_interval_seconds: float = 0.5,
) -> None:
    app = create_monitor_app(hub)
    config = Config(app, host=host, port=port, log_level="info", loop="asyncio")
    server = Server(config)
    poller = asyncio.create_task(
        poll_context(store, hub, interval_seconds=poll_interval_seconds),
        name="monitor-context-poller",
    )
    logger.info("Monitor UI: http://%s:%s", host, port)
    try:
        await server.serve()
    finally:
        server.should_exit = True
        poller.cancel()
        await asyncio.gather(poller, return_exceptions=True)
