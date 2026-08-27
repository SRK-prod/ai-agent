"""FastAPI backend: owns the MeetingPipeline, pushes Answers to the overlay over
a WebSocket, and exposes health/enroll/metrics endpoints.

The pipeline auto-starts on app startup so, per the spec, "no manual
interaction is required once the meeting starts" -- just launch the backend
(`make run-backend` / `make run`) before joining the call.
"""

from __future__ import annotations

import asyncio
import dataclasses
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from meeting_copilot.pipeline.events import Answer, AudioHealth
from meeting_copilot.pipeline.orchestrator import MeetingPipeline
from meeting_copilot.speaker.enrollment import enroll_interactive
from meeting_copilot.utils.logging import get_logger

logger = get_logger()


class ConnectionManager:
    def __init__(self):
        self._connections: set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self._connections.discard(ws)

    async def broadcast_answer(self, answer: Answer) -> None:
        payload = dataclasses.asdict(answer)
        await self._broadcast({"type": "answer", "data": payload})

    async def broadcast_partial_answer(self, text_so_far: str) -> None:
        await self._broadcast({"type": "answer_partial", "data": {"text": text_so_far}})

    async def broadcast_audio_health(self, health: AudioHealth) -> None:
        # Only state/reason go to the overlay -- raw RMS/peak/callback counts are
        # debug detail, already logged server-side on every transition (see
        # MeetingPipeline._watchdog_loop), not something the overlay UI displays.
        await self._broadcast(
            {"type": "audio_health", "data": {"state": health.state, "reason": health.reason}}
        )

    async def _broadcast(self, message: dict) -> None:
        stale: list[WebSocket] = []
        for ws in self._connections:
            try:
                await ws.send_json(message)
            except Exception:  # noqa: BLE001 -- any send failure means this socket is dead
                stale.append(ws)
        for ws in stale:
            self.disconnect(ws)


connections = ConnectionManager()
_pipeline: MeetingPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    try:
        _pipeline = MeetingPipeline(
            on_answer=connections.broadcast_answer,
            on_partial_answer=connections.broadcast_partial_answer,
            on_audio_health=connections.broadcast_audio_health,
        )
        _pipeline.start()
        logger.info("MeetingPipeline auto-started on backend boot")
    except Exception:
        # Missing credentials/models (HF_TOKEN, claude login, ...) or an
        # unavailable audio device shouldn't take down /health, /metrics, /enroll -- log
        # loudly and keep serving; fix the underlying issue and restart the backend.
        logger.exception(
            "MeetingPipeline failed to start -- backend is still up (health/metrics/enroll "
            "work) but no live meeting processing will happen until this is fixed and the "
            "backend is restarted. See docs/troubleshooting.md."
        )
        _pipeline = None
    try:
        yield
    finally:
        if _pipeline:
            await _pipeline.stop()


app = FastAPI(title="meeting-copilot backend", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "pipeline": "running" if _pipeline else "not_started"}


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


class EnrollRequest(BaseModel):
    duration_seconds: float = 45.0


@app.post("/enroll")
async def enroll(req: EnrollRequest) -> dict:
    """Blocking mic recording -- only call this before a meeting starts,
    not while MeetingPipeline is actively capturing audio."""
    await asyncio.to_thread(enroll_interactive, req.duration_seconds)
    return {"status": "enrolled"}


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    await connections.connect(websocket)
    # Audio health is only broadcast on a state TRANSITION (see MeetingPipeline._watchdog_
    # loop), so a client connecting between two transitions would otherwise never learn the
    # current state until the next one happens to fire. Send it directly to just this new
    # connection so the overlay is never blind to an already-in-progress AUDIO_INPUT_LOST.
    if _pipeline is not None:
        health = _pipeline.audio_health()
        await websocket.send_json(
            {"type": "audio_health", "data": {"state": health.state, "reason": health.reason}}
        )
    try:
        while True:
            await websocket.receive_text()  # overlay currently only receives; ignore any input
    except WebSocketDisconnect:
        connections.disconnect(websocket)
