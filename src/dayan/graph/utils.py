"""Logging helpers: stdlib-only, no side effects at import time.

Set ``DAYAN_LOG_DIR`` to also write logs to files; by default logs go to
stderr only.  Repeated calls with the same name return the same logger
without duplicating handlers.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

_FORMAT = "[%(levelname)s] %(name)s: %(message)s"


def get_logger(name: str, console_level: int = logging.INFO, file_level: int = logging.ERROR) -> logging.Logger:
    """Return a configured logger (idempotent)."""

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)

    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(logging.Formatter(_FORMAT))
    logger.addHandler(console)

    log_dir = os.environ.get("DAYAN_LOG_DIR")
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(Path(log_dir) / f"{name.replace('.', '_')}.log", encoding="utf-8")
        file_handler.setLevel(file_level)
        file_handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(file_handler)

    return logger
