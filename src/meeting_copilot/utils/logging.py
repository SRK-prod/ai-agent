"""loguru setup driven by configs/logging.yaml. Call configure_logging() once at process start."""

from __future__ import annotations

import sys
from functools import lru_cache
from typing import TYPE_CHECKING

import yaml
from loguru import logger

from meeting_copilot.paths import LOGGING_FILE, PROJECT_ROOT

if TYPE_CHECKING:
    from loguru import Logger

_DEFAULT_FORMAT = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"


@lru_cache
def configure_logging() -> Logger:
    with open(LOGGING_FILE) as f:
        cfg: dict = yaml.safe_load(f) or {}

    level: str = cfg.get("level", "INFO")
    fmt: str = cfg.get("format") or _DEFAULT_FORMAT

    logger.remove()

    if cfg.get("console", True):
        logger.add(sys.stderr, level=level, format=fmt)

    sink = cfg.get("sink")
    if sink:
        sink_path = PROJECT_ROOT / sink
        sink_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(sink_path),
            level=level,
            format=fmt,
            rotation=cfg.get("rotation", "50 MB"),
            retention=cfg.get("retention", "14 days"),
            enqueue=True,
        )

    return logger


def get_logger() -> Logger:
    """Return the configured loguru logger, configuring it on first use."""
    return configure_logging()
