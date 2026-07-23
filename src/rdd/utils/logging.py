"""Console + file logging, shared across stages."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED = False


def get_logger(name: str = "rdd") -> logging.Logger:
    return logging.getLogger(name)


def setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configure the root 'rdd' logger once. Safe to call repeatedly."""
    global _CONFIGURED
    logger = logging.getLogger("rdd")
    if _CONFIGURED:
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s [%(name)s] %(message)s", "%H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)  # stdout so headless captures work
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    _CONFIGURED = True
    return logger
