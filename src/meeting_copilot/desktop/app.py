"""Desktop overlay entrypoint: `make run-ui` / `python -m meeting_copilot.desktop.app`.

Connects to the backend's /ws endpoint on a background thread (its own
asyncio loop, separate from Qt's event loop) and forwards each Answer to the
overlay via a thread-safe Qt Signal.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading

import websockets
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from meeting_copilot.config import get_config
from meeting_copilot.desktop.hotkeys import HotkeyManager
from meeting_copilot.desktop.overlay import OverlayWindow
from meeting_copilot.utils.logging import configure_logging, get_logger

logger = get_logger()


class WebSocketBridge(QObject):
    answer_received = Signal(dict)
    partial_answer_received = Signal(str)
    audio_health_received = Signal(dict)


class BackendClient:
    def __init__(self, url: str, bridge: WebSocketBridge):
        self._url = url
        self._bridge = bridge
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        asyncio.run(self._listen_forever())

    async def _listen_forever(self) -> None:
        while True:
            try:
                async with websockets.connect(self._url) as ws:
                    logger.info(f"Connected to backend at {self._url}")
                    async for raw in ws:
                        message = json.loads(raw)
                        if message.get("type") == "answer":
                            self._bridge.answer_received.emit(message["data"])
                        elif message.get("type") == "answer_partial":
                            self._bridge.partial_answer_received.emit(message["data"]["text"])
                        elif message.get("type") == "audio_health":
                            self._bridge.audio_health_received.emit(message["data"])
            except Exception as exc:  # noqa: BLE001 -- any failure here just means retry
                logger.warning(f"Backend WebSocket connection lost ({exc}); retrying in 3s")
                await asyncio.sleep(3)


def main() -> None:
    configure_logging()
    cfg = get_config()

    app = QApplication(sys.argv)
    overlay = OverlayWindow(cfg.overlay)

    hotkeys = HotkeyManager(overlay, cfg.overlay)
    hotkeys.start()

    bridge = WebSocketBridge()
    bridge.answer_received.connect(overlay.show_answer)
    bridge.partial_answer_received.connect(overlay.show_partial_answer)
    bridge.audio_health_received.connect(overlay.show_audio_health)
    ws_url = f"ws://{cfg.secrets.host}:{cfg.secrets.port}/ws"
    BackendClient(ws_url, bridge).start()

    overlay.show()
    exit_code = app.exec()
    hotkeys.stop()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
