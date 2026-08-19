"""Logging setup.

The reference threaded a ``logging_enabled: bool`` parameter through nearly every
function signature and printed emoji to stdout. Here logging is configured once and
modules just call ``get_logger(__name__)``.

Every process (parent and spawn workers) writes to stderr **and** to a log file when
``log_file`` is set, so a redirected subprocess cannot silently drop worker INFO lines.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_CONFIGURED = False
_FILE_HANDLERS: set[str] = set()

_FORMAT = "%(asctime)s %(levelname)-7s %(name)-22s %(message)s"
_DATEFMT = "%H:%M:%S"


class FlushFileHandler(logging.FileHandler):
    """File handler that flushes every record so a crash does not lose the last lines."""

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


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
        path = Path(log_file).expanduser().resolve()
        key = str(path)
        if key not in _FILE_HANDLERS:
            path.parent.mkdir(parents=True, exist_ok=True)
            handler = FlushFileHandler(path, encoding="utf-8")
            handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
            root.addHandler(handler)
            _FILE_HANDLERS.add(key)

    # These are noisy at INFO and say nothing useful to a user of this tool.
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
