"""Consistent logging configuration for the aml_benchmark package."""
from __future__ import annotations

import logging
import sys


_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a named logger with a standardised stdout handler.

    Re-calling with the same *name* returns the existing logger without
    adding duplicate handlers.

    Parameters
    ----------
    name:
        Typically ``__name__`` of the calling module.
    level:
        Logging level (default: INFO).
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATE_FMT))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
