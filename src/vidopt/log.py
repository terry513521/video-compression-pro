"""Logging setup.

The reference threaded a ``logging_enabled: bool`` parameter through nearly every
function signature and printed emoji to stdout. Here logging is configured once and
modules just call ``get_logger(__name__)``.
"""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False

_FORMAT = "%(asctime)s %(levelname)-7s %(name)-22s %(message)s"
_DATEFMT = "%H:%M:%S"


def setup_logging(
    level: str = "INFO", *, log_file: str | os.PathLike[str] | None = None
) -> None:
    """Configure root logging. Safe to call more than once; later calls adjust level."""
    global _CONFIGURED

    root = logging.getLogger()
    numeric = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(numeric)

    if not _CONFIGURED:
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        root.addHandler(stream)
        _CONFIGURED = True

    if log_file is not None:
        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        root.addHandler(handler)

    # These are noisy at INFO and say nothing useful to a user of this tool.
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
