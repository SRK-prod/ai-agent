"""Single-process entrypoint for the PyInstaller-bundled .app.

In dev mode the backend and overlay are two separate processes
(scripts/start.sh). A packaged macOS .app is simpler as one process: the
FastAPI backend runs on a background thread, uvloop and all, while the Qt
event loop owns the main thread (PySide6/Cocoa requires that).
"""

from __future__ import annotations

import threading
import time

import uvicorn

from meeting_copilot.config import get_config
from meeting_copilot.desktop.app import main as run_overlay
from meeting_copilot.server.api import app as fastapi_app
from meeting_copilot.utils.logging import configure_logging, get_logger

logger = get_logger()


def _run_backend() -> None:
    secrets = get_config().secrets
    uvicorn.run(fastapi_app, host=secrets.host, port=secrets.port, loop="uvloop", log_config=None)


def main() -> None:
    configure_logging()
    backend_thread = threading.Thread(target=_run_backend, daemon=True)
    backend_thread.start()

    time.sleep(2)  # let the backend finish booting before the overlay connects
    logger.info("Launching overlay UI")
    run_overlay()


if __name__ == "__main__":
    main()
