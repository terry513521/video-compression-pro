"""Lightweight progress event stream for desktop UI integration."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any


class ProgressEmitter:
    """Append-only JSONL event sink.

    Each event is one JSON object per line, flushed immediately so a desktop UI can tail
    the file and render real-time progress.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def emit(self, event: str, **payload: Any) -> None:
        row = {"ts": time.time(), "event": event, **payload}
        line = json.dumps(row, ensure_ascii=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()

