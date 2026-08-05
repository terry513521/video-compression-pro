"""Dev mode: build the training corpus and fit the models.

    discover corpus -> segment -> extract features -> search -> dataset -> train

Every stage writes its artifacts into ``work_dir`` and can be re-run independently, so a
failure in training does not discard hours of search. The trial cache means a re-run of
the search itself costs only whatever it had not already measured.
"""

from __future__ import annotations

import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from ..config import Config
from ..encoding.encoders import get_encoder
from ..errors import SearchError, VidoptError
from ..features.extract import extract
from ..ffmpeg import toolchain
from ..ffmpeg.probe import content_hash, probe
from ..log import get_logger
from ..media.segment import Segment, segment_video
from ..modeling.dataset import Row, write_dataset
from ..modeling.train import TrainReport, train_all
from ..search.cache import open_cache
from ..search.optimizer import search_segment

log = get_logger(__name__)

# Worker processes are started with "spawn" explicitly rather than the platform default.
# Three reasons, in order of importance:
#
#   1. Windows only has spawn. Forcing it everywhere means the code path exercised in
#      development on Linux is the same one that runs in production on Windows, instead
#      of fork here and spawn there.
#   2. fork() in a process that has already initialised CUDA is unsafe — the child
#      inherits a broken context. The GPU configuration would hit that.
#   3. fork() copies whatever the parent happened to have loaded; spawn starts clean, so
#      a worker's state depends only on the payload it is given.
#
# The cost is that payloads must be picklable and workers pay ~0.5 s of start-up. Both
# are already true here: every worker takes a plain dict and rebuilds its own config.
_MP_CONTEXT = multiprocessing.get_context("spawn")


VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".webm", ".y4m", ".avi", ".m4v", ".ts"}


@dataclass
class DevResult:
    dataset_path: str
    n_sources: int
    n_segments: int
    n_rows: int
    n_infeasible: int
    reports: list[TrainReport]
    seconds: float


def discover_sources(inputs: list[str | Path], *, limit: int | None = None) -> list[Path]:
    """Expand files and directories into a sorted list of video paths."""
    found: list[Path] = []
    for item in inputs:
        path = Path(item).expanduser()
        if path.is_dir():
            found.extend(
                p for p in sorted(path.rglob("*")) if p.suffix.lower() in VIDEO_SUFFIXES
            )
        elif path.is_file():
            found.append(path)
        else:
            log.warning("input not found, skipping: %s", path)

    unique = sorted({p.resolve() for p in found})
    if limit is not None:
        unique = unique[:limit]
    return unique


def _segment_corpus(
    sources: list[Path], caps: toolchain.Capabilities, config: Config, work_dir: Path
) -> list[Segment]:
    """Segment every source. A failure on one source does not abort the corpus."""
    segments_root = work_dir / "segments"
    all_segments: list[Segment] = []

    for source in sources:
        try:
            info = probe(source, caps.ffprobe)
            out_dir = segments_root / source.stem
            produced = segment_video(source, out_dir, caps, config, info=info)
            all_segments.extend(produced)
        except VidoptError as exc:
            log.error("skipping %s: %s", source.name, exc)

    if not all_segments:
        raise SearchError(
            "segmentation produced nothing usable from the corpus. Check that the input "
            "files are readable video and that ffmpeg works (`vidopt doctor`)."
        )
    return all_segments


def _search_one(payload: dict) -> dict:
    """Worker entry point. Runs in a separate process, so it takes plain data.

    Re-deriving the toolchain and cache handles per process is intentional: SQLite
    connections and cached capability probes are not fork-safe to share.
    """
    from ..config import load_config  # re-imported inside the worker process

    config = load_config(payload["config_paths"], payload["config_overrides"])
    caps = toolchain.detect(config.ffmpeg.bin_dir)
    encoder = get_encoder(config.encoder.name)
    segment_path = Path(payload["segment_path"])

    with open_cache(config.paths.cache_db, config.search.cache) as cache:
        info = probe(segment_path, caps.ffprobe)
        features = extract(segment_path, info, config.features)
        results = search_segment(
            segment_path,
            payload["segment_hash"],
            info,
            list(config.search.targets),
            caps=caps,
            config=config,
            encoder=encoder,
            cache=cache,
            work_dir=Path(payload["work_dir"]) / "trials",
        )

    return {
        "segment_path": str(segment_path),
        "segment_hash": payload["segment_hash"],
        "source": payload["source"],
        "features": features.to_dict(),
        "results": {
            str(target): {
                "feasible": result.feasible,
                "vmaf": result.best.vmaf if result.best else 0.0,
                "rate": result.best.rate if result.best else 1.0,
                "ratio": result.best.ratio if result.best else 0.0,
                "score": result.score,
                "n_trials": result.n_trials,
                "params": result.best.params.to_dict() if result.best else {},
            }
            for target, result in results.items()
        },
    }


def run_dev(
    inputs: list[str | Path],
    config: Config,
    config_paths: list[str],
    config_overrides: list[str],
    *,
    limit: int | None = None,
    skip_training: bool = False,
) -> DevResult:
    """Execute the whole dev-mode workflow."""
    started = time.monotonic()

    caps = toolchain.detect(config.ffmpeg.bin_dir)
    encoder = get_encoder(config.encoder.name)
    toolchain.require(caps, encoder=encoder.ffmpeg_encoder, vmaf=True)

    work_dir = Path(config.paths.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    sources = discover_sources(inputs, limit=limit)
    if not sources:
        raise SearchError(f"no video files found in: {inputs}")
    log.info("corpus: %d source video(s)", len(sources))

    log.info("stage 1/4: segmenting")
    segments = _segment_corpus(sources, caps, config, work_dir)
    log.info("stage 1/4: %d segment(s) from %d source(s)", len(segments), len(sources))

    (work_dir / "segments.json").write_text(
        json.dumps([s.to_dict() for s in segments], indent=2), encoding="utf-8"
    )

    log.info("stage 2/4: hashing segments for the trial cache")
    payloads = [
        {
            "segment_path": segment.path,
            "segment_hash": content_hash(segment.path),
            "source": segment.source,
            "config_paths": config_paths,
            "config_overrides": config_overrides,
            "work_dir": str(work_dir),
        }
        for segment in segments
    ]

    workers = config.resolved_cpu_workers()
    if encoder.is_gpu:
        # GPU encodes serialise on the device; more processes than GPU slots only adds
        # contention and NVENC session-limit failures.
        workers = min(workers, max(1, config.jobs.gpu_workers))
    log.info(
        "stage 3/4: searching %d segment(s) x %d target(s) with %d worker(s)",
        len(segments), len(config.search.targets), workers,
    )

    records: list[dict] = []
    if workers <= 1:
        for payload in payloads:
            try:
                records.append(_search_one(payload))
            except VidoptError as exc:
                log.error(
                    "search failed for %s: %s",
                    Path(payload["segment_path"]).name, exc,
                )
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=_MP_CONTEXT) as pool:
            futures = {pool.submit(_search_one, p): p for p in payloads}
            for done in as_completed(futures):
                payload = futures[done]
                try:
                    records.append(done.result())
                except Exception as exc:  # noqa: BLE001 - worker errors arrive here
                    log.error(
                        "search failed for %s: %s",
                        Path(payload["segment_path"]).name, exc,
                    )

    if not records:
        raise SearchError("every segment failed to search; see the errors above")

    rows: list[Row] = []
    n_infeasible = 0
    for record in records:
        for target_text, result in record["results"].items():
            if not result["feasible"]:
                n_infeasible += 1
            params = result["params"] or {"crf": 0.0, "aq_mode": 0, "aq_strength": 0.0}
            rows.append(
                Row(
                    meta={
                        "source": record["source"],
                        "segment": record["segment_path"],
                        "segment_hash": record["segment_hash"],
                        "target": float(target_text),
                        "feasible": result["feasible"],
                        "vmaf": round(result["vmaf"], 3),
                        "rate": round(result["rate"], 6),
                        "ratio": round(result["ratio"], 3),
                        "score": round(result["score"], 5),
                        "encoder": config.encoder.name,
                        "n_trials": result["n_trials"],
                    },
                    features={k: round(v, 6) for k, v in record["features"].items()},
                    labels=params,
                )
            )

    dataset_path = write_dataset(rows, work_dir / "dataset.csv")
    log.info(
        "stage 3/4 complete: %d row(s), %d infeasible (%.0f%%)",
        len(rows), n_infeasible, 100.0 * n_infeasible / max(1, len(rows)),
    )

    reports: list[TrainReport] = []
    if skip_training:
        log.info("stage 4/4: skipped (--no-train)")
    else:
        log.info("stage 4/4: training")
        from ..modeling.dataset import read_dataset

        reports = train_all(read_dataset(dataset_path), config, config.paths.models_dir)

    elapsed = time.monotonic() - started
    result = DevResult(
        dataset_path=str(dataset_path),
        n_sources=len(sources),
        n_segments=len(segments),
        n_rows=len(rows),
        n_infeasible=n_infeasible,
        reports=reports,
        seconds=elapsed,
    )

    (work_dir / "dev_summary.json").write_text(
        json.dumps(
            {
                "dataset": result.dataset_path,
                "sources": result.n_sources,
                "segments": result.n_segments,
                "rows": result.n_rows,
                "infeasible": result.n_infeasible,
                "seconds": round(elapsed, 1),
                "encoder": config.encoder.name,
                "targets": list(config.search.targets),
                "models": [
                    {"target": r.target, "path": r.bundle_path, "metrics": r.metrics}
                    for r in reports
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info("dev mode finished in %.1fs", elapsed)
    return result
