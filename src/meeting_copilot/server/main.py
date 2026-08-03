"""Entrypoint: `uvicorn meeting_copilot.server.main:app --loop uvloop` (see Makefile `run-backend`)."""

from __future__ import annotations

from meeting_copilot.server.api import app
from meeting_copilot.utils.logging import configure_logging

configure_logging()

__all__ = ["app"]


if __name__ == "__main__":
    import uvicorn

    from meeting_copilot.config import get_config

    secrets = get_config().secrets
    uvicorn.run(app, host=secrets.host, port=secrets.port, loop="uvloop")
