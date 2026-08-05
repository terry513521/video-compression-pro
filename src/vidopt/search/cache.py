"""Trial cache.

Encoding plus VMAF measurement is the dominant cost of dev mode — minutes per trial on
4K content. The reference cached nothing, so an interrupted run threw away everything.

Every (segment, encoder, params, vmaf-settings) outcome is stored in SQLite, keyed by the
segment's **content hash** rather than its path, so the cache survives moved or
regenerated working directories.

The cache also makes the three VMAF targets nearly free to add: targets 85, 89 and 93
explore overlapping CRF ranges, and a trial measured for one is reused by the others
without re-encoding.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

from ..encoding.params import EncodeParams
from ..log import get_logger

log = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trials (
    key            TEXT PRIMARY KEY,
    segment_hash   TEXT NOT NULL,
    encoder        TEXT NOT NULL,
    crf            REAL NOT NULL,
    aq_mode        INTEGER NOT NULL,
    aq_strength    REAL NOT NULL,
    vmaf_model     TEXT NOT NULL,
    n_subsample    INTEGER NOT NULL,
    ref_bytes      INTEGER NOT NULL,
    out_bytes      INTEGER NOT NULL,
    vmaf           REAL NOT NULL,
    encode_seconds REAL NOT NULL,
    vmaf_seconds   REAL NOT NULL,
    extra          TEXT
);
CREATE INDEX IF NOT EXISTS idx_trials_segment ON trials (segment_hash, encoder);
"""


@dataclass(frozen=True)
class TrialRecord:
    """One measured (params -> size, quality) outcome."""

    segment_hash: str
    encoder: str
    params: EncodeParams
    vmaf_model: str
    n_subsample: int
    ref_bytes: int
    out_bytes: int
    vmaf: float
    encode_seconds: float = 0.0
    vmaf_seconds: float = 0.0

    @property
    def rate(self) -> float:
        """out_bytes / ref_bytes — the compression rate the score function consumes."""
        return (self.out_bytes / self.ref_bytes) if self.ref_bytes > 0 else 1.0

    @property
    def ratio(self) -> float:
        rate = self.rate
        return (1.0 / rate) if rate > 0 else 0.0


def trial_key(
    segment_hash: str,
    encoder: str,
    params: EncodeParams,
    vmaf_model: str,
    n_subsample: int,
) -> str:
    p = params.rounded()
    return "|".join(
        [segment_hash, encoder, p.key(), vmaf_model, str(n_subsample)]
    )


class TrialCache:
    """SQLite-backed cache, safe for concurrent search workers.

    Dev mode runs one process per CPU worker, and every one of them writes here. Two
    settings make that safe:

    * **WAL journalling** lets readers proceed while one process writes, instead of
      serialising every access behind an exclusive lock.
    * **A generous busy timeout**, because a writer that arrives during another's commit
      should wait, not raise ``database is locked`` and lose a trial that cost minutes of
      encoding.

    A no-op instance is available via :meth:`disabled`.
    """

    BUSY_TIMEOUT_SECONDS = 60.0

    def __init__(self, db_path: str | Path) -> None:
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self.path),
            check_same_thread=False,
            timeout=self.BUSY_TIMEOUT_SECONDS,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(f"PRAGMA busy_timeout={int(self.BUSY_TIMEOUT_SECONDS * 1000)}")
        # NORMAL is durable enough here: a trial lost to a machine crash is simply
        # re-measured on the next run, and full syncs would dominate the write cost.
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self.hits = 0
        self.misses = 0

    @classmethod
    def disabled(cls) -> NullCache:
        return NullCache()

    def get(
        self,
        segment_hash: str,
        encoder: str,
        params: EncodeParams,
        vmaf_model: str,
        n_subsample: int,
    ) -> TrialRecord | None:
        key = trial_key(segment_hash, encoder, params, vmaf_model, n_subsample)
        with self._lock:
            row = self._conn.execute(
                "SELECT ref_bytes, out_bytes, vmaf, encode_seconds, vmaf_seconds "
                "FROM trials WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return TrialRecord(
            segment_hash=segment_hash,
            encoder=encoder,
            params=params.rounded(),
            vmaf_model=vmaf_model,
            n_subsample=n_subsample,
            ref_bytes=int(row[0]),
            out_bytes=int(row[1]),
            vmaf=float(row[2]),
            encode_seconds=float(row[3]),
            vmaf_seconds=float(row[4]),
        )

    def put(self, record: TrialRecord, extra: dict | None = None) -> None:
        key = trial_key(
            record.segment_hash,
            record.encoder,
            record.params,
            record.vmaf_model,
            record.n_subsample,
        )
        p = record.params.rounded()
        # Columns are named rather than positional: a schema change should be a clean
        # error, not a silently shifted row.
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO trials (
                    key, segment_hash, encoder, crf, aq_mode, aq_strength,
                    vmaf_model, n_subsample, ref_bytes, out_bytes, vmaf,
                    encode_seconds, vmaf_seconds, extra
                ) VALUES (
                    :key, :segment_hash, :encoder, :crf, :aq_mode, :aq_strength,
                    :vmaf_model, :n_subsample, :ref_bytes, :out_bytes, :vmaf,
                    :encode_seconds, :vmaf_seconds, :extra
                )
                """,
                {
                    "key": key,
                    "segment_hash": record.segment_hash,
                    "encoder": record.encoder,
                    "crf": p.crf,
                    "aq_mode": p.aq_mode,
                    "aq_strength": p.aq_strength,
                    "vmaf_model": record.vmaf_model,
                    "n_subsample": record.n_subsample,
                    "ref_bytes": record.ref_bytes,
                    "out_bytes": record.out_bytes,
                    "vmaf": record.vmaf,
                    "encode_seconds": record.encode_seconds,
                    "vmaf_seconds": record.vmaf_seconds,
                    "extra": json.dumps(extra or {}),
                },
            )
            self._conn.commit()

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> TrialCache:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class NullCache:
    """Cache interface that stores nothing. Used when ``search.cache`` is false."""

    hits = 0
    misses = 0

    def get(self, *args: object, **kwargs: object) -> None:
        return None

    def put(self, *args: object, **kwargs: object) -> None:
        return None

    def count(self) -> int:
        return 0

    def close(self) -> None:
        return None

    def __enter__(self) -> NullCache:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def open_cache(db_path: str | Path, enabled: bool) -> TrialCache | NullCache:
    if not enabled:
        log.info("trial cache disabled")
        return NullCache()
    cache = TrialCache(db_path)
    log.info("trial cache: %s (%d existing trial(s))", cache.path, cache.count())
    return cache
