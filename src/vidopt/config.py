"""Configuration schema and loading.

One YAML file describes every tunable in the pipeline. It is parsed into frozen
dataclasses and validated on load, so an impossible combination fails at second zero
rather than at minute forty of a search.

Layering, lowest precedence first:

    configs/default.yaml  ->  --config overlay(s)  ->  --set dotted.key=value
"""

from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigError

# --------------------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FfmpegConfig:
    """Where to find the binaries, and how hard to push them."""

    bin_dir: str | None = None
    """Explicit directory containing ffmpeg/ffprobe. None => search (see toolchain.py)."""

    threads: int = 0
    """-threads for encodes. 0 lets ffmpeg decide (usually best for a single job)."""

    loglevel: str = "error"


@dataclass(frozen=True)
class SegmentConfig:
    """Scene segmentation."""

    detector: str = "adaptive"
    """'adaptive' (PySceneDetect AdaptiveDetector) or 'content' (ContentDetector)."""

    adaptive_threshold: float = 3.0
    content_threshold: float = 27.0

    proxy_height: int = 360
    """Detection runs on a downscaled proxy of this height. 0 disables the proxy."""

    min_segment_seconds: float = 2.0
    """Cuts closer together than this are merged. Sub-second segments encode badly."""

    max_segment_seconds: float = 15.0
    """Longer runs are split, so that segments stay parallelisable."""

    fallback_segment_seconds: float = 6.0
    """Used when the detector finds no cuts at all."""

    container: str = "mkv"
    """Intermediate segment container. mkv tolerates copy-cut streams best."""


@dataclass(frozen=True)
class FeatureConfig:
    """Content analysis."""

    max_frames: int = 96
    """Frames sampled per segment. Higher = steadier features, slower."""

    analysis_width: int = 480
    """Frames are downscaled to this width before analysis. Texture/motion statistics
    are scale-sensitive, so this must stay fixed between dev and production."""

    canny_low: int = 100
    canny_high: int = 200


@dataclass(frozen=True)
class EncoderConfig:
    """Which encoder to search and deploy. Models are encoder-specific."""

    name: str = "libx265"
    preset: str = "medium"
    keyint_seconds: float = 2.0

    pix_fmt: str = "auto"
    """'auto' keeps the source pixel format when the encoder supports it (so 10-bit and
    4:4:4 input survive); an explicit name forces that format. See encoding/pixfmt.py."""
    extra_args: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class VmafConfig:
    """Quality measurement."""

    model: str = "vmaf_v0.6.1neg"
    """'vmaf_v0.6.1neg' resists sharpening tricks; 'vmaf_v0.6.1' is the classic model."""

    n_subsample_search: int = 2
    """Frame subsampling during search. 2 roughly halves cost for <0.5 VMAF of noise."""

    n_subsample_verify: int = 1
    """Never subsample when reporting a final number."""

    n_threads: int = 4
    use_cuda: bool = True
    """Use libvmaf_cuda when the toolchain reports it. Falls back to CPU otherwise."""

    pool: str = "harmonic_mean"
    """'harmonic_mean' (punishes bad frames — correct for a quality floor) or 'mean'."""


@dataclass(frozen=True)
class SearchConfig:
    """Parameter search (dev mode)."""

    targets: list[float] = field(default_factory=lambda: [85.0, 89.0, 93.0])
    """VMAF levels to label during search (one row per segment × target in the dataset).

    Training fits one model per encoder; ``--target`` is provided at compress time as
    a model input feature. Wider range improves interpolation/extrapolation stability.
    Override with ``--set search.targets='[85,89,93]'``."""

    strategy: str = "aq_then_crf"
    """How stage A proposes (crf, AQ) points.

    'aq_then_crf'  enumerate AQ, screen, then 1-D CRF solve (default)
    'coordinate'   AQ neighbour walk after the same screen
    'sample'       3-D space-filling design (see sampler)
    'bayes'        GP Bayesian optimisation after an initial design
    'tpe'          Tree-structured Parzen Estimator
    'cmaes'        diagonal CMA-style evolution strategy
    """

    sampler: str = "sobol"
    """Space-filling design for strategy='sample' and for bayes/tpe/cmaes init:
    'sobol', 'lhs', 'halton', 'random', 'grid'."""

    crf_solver: str = "bisect"
    """1-D solver for CRF at fixed AQ: 'bisect' (secant), 'brent', 'golden'."""

    n_explore: int = 12
    """Stage-A trial budget. Structured strategies treat this as a cap, not a 3-D count."""

    n_init: int = 0
    """Initial design size for bayes/tpe/cmaes. 0 => max(4, n_explore/2)."""

    n_refine_configs: int = 2
    """How many AQ settings from stage A get a 1-D CRF solve."""

    top_k_per_segment: int = 5
    """How many best feasible parameter points to keep per segment/target for training."""

    n_strength_steps: int = 5
    """Float AQ-strength grid size (ignored when the encoder uses integer strength)."""

    n_screen_crfs: int = 2
    """CRF probes per AQ during aq_then_crf / coordinate screening."""

    max_coordinate_rounds: int = 4
    """Neighbour-walk iterations for strategy='coordinate'."""

    max_bisect_iters: int = 6
    crf_tolerance: float = 0.5
    """Bisection stops once the CRF bracket is narrower than this."""

    seed: int = 20240817
    cache: bool = True


@dataclass(frozen=True)
class ModelConfig:
    """Training."""

    estimator: str = "hgb"
    """'hgb' (HistGradientBoosting) or 'rf' (RandomForest)."""

    crf_quantile: float = 0.15
    """Quantile loss level for the CRF head. Below 0.5 => deliberately conservative.

    Two effects make margin mandatory rather than optional: the search labels sit exactly
    on the target boundary, and whole-video VMAF is a harmonic mean over segments, which
    punishes a low segment more than an equal overshoot helps. See DESIGN.md section 7
    and the measured calibration in configs/default.yaml."""

    max_iter: int = 300
    learning_rate: float = 0.06
    max_depth: int | None = 4
    min_samples_leaf: int = 4
    min_training_rows: int = 8
    """Refuse to train a target with fewer feasible labelled segments than this."""
    val_fraction: float = 0.25
    seed: int = 20240817


@dataclass(frozen=True)
class JobsConfig:
    """Parallelism. Scale with ``--cpu-workers`` / ``--gpu-workers`` on the CLI."""

    cpu_workers: int = 4
    """Concurrent CPU encodes. 0 => auto (sized from cores). Default 4."""

    gpu_workers: int = 0
    """Concurrent GPU encodes. 0 = CPU-only (default). Raise for NVENC."""

    ffmpeg_threads_per_job: int = 2
    """Used to size the auto worker count so the box is not oversubscribed."""


@dataclass(frozen=True)
class PathsConfig:
    work_dir: str = "runs/current"
    cache_db: str = "runs/cache/trials.sqlite"
    models_dir: str = "models"


@dataclass(frozen=True)
class Config:
    ffmpeg: FfmpegConfig = field(default_factory=FfmpegConfig)
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    vmaf: VmafConfig = field(default_factory=VmafConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    jobs: JobsConfig = field(default_factory=JobsConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    log_level: str = "INFO"

    # ---------------- derived helpers ----------------

    def resolved_cpu_workers(self) -> int:
        if self.jobs.cpu_workers > 0:
            return self.jobs.cpu_workers
        cores = os.cpu_count() or 2
        per_job = max(1, self.jobs.ffmpeg_threads_per_job)
        return max(1, min(cores - 1, cores // per_job) or 1)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def dumps(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


_SECTIONS: dict[str, type] = {
    "ffmpeg": FfmpegConfig,
    "segment": SegmentConfig,
    "features": FeatureConfig,
    "encoder": EncoderConfig,
    "vmaf": VmafConfig,
    "search": SearchConfig,
    "model": ModelConfig,
    "jobs": JobsConfig,
    "paths": PathsConfig,
}


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _coerce(value: Any, annotation: Any) -> Any:
    """Coerce a YAML/CLI scalar to the dataclass field type."""
    text = str(annotation)
    if value is None:
        return None
    if "list" in text:
        if isinstance(value, list):
            return value
        raise ConfigError(f"expected a list, got {value!r}")
    if annotation is bool or text.startswith("bool"):
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}
    if "int" in text and "float" not in text:
        return int(value)
    if "float" in text:
        return float(value)
    return value if isinstance(value, str) else str(value)


def _build_section(name: str, cls: type, raw: Mapping[str, Any]) -> Any:
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(raw) - set(fields)
    if unknown:
        raise ConfigError(
            f"unknown key(s) in config section '{name}': {sorted(unknown)}. "
            f"Valid keys: {sorted(fields)}"
        )
    kwargs: dict[str, Any] = {}
    for key, value in raw.items():
        try:
            kwargs[key] = _coerce(value, fields[key].type)
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"invalid value for {name}.{key}: {value!r} ({exc})"
            ) from exc
    return cls(**kwargs)


def _apply_override(tree: dict[str, Any], dotted: str) -> None:
    if "=" not in dotted:
        raise ConfigError(f"--set expects key=value, got {dotted!r}")
    key, _, raw = dotted.partition("=")
    parts = key.strip().split(".")
    node = tree
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ConfigError(f"--set {key}: '{part}' is not a section")
    try:
        value: Any = yaml.safe_load(raw)
    except yaml.YAMLError:
        value = raw
    node[parts[-1]] = value


def load_config(
    paths: list[str | os.PathLike[str]] | None = None,
    overrides: list[str] | None = None,
) -> Config:
    """Load, merge, validate."""
    tree: dict[str, Any] = {}

    for path in paths or []:
        p = Path(path)
        if not p.is_file():
            raise ConfigError(f"config file not found: {p}")
        try:
            loaded = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"could not parse {p}: {exc}") from exc
        if not isinstance(loaded, dict):
            raise ConfigError(f"{p}: top level must be a mapping")
        tree = _deep_merge(tree, loaded)

    for override in overrides or []:
        _apply_override(tree, override)

    unknown_sections = set(tree) - set(_SECTIONS) - {"log_level"}
    if unknown_sections:
        raise ConfigError(
            f"unknown config section(s): {sorted(unknown_sections)}. "
            f"Valid sections: {sorted(_SECTIONS)}"
        )

    sections: dict[str, Any] = {}
    for name, cls in _SECTIONS.items():
        raw = tree.get(name) or {}
        if not isinstance(raw, Mapping):
            raise ConfigError(f"config section '{name}' must be a mapping")
        sections[name] = _build_section(name, cls, raw)

    config = Config(log_level=str(tree.get("log_level", "INFO")), **sections)
    validate(config)
    return config


def validate(config: Config) -> None:
    """Reject configurations that cannot possibly work."""
    from .encoding.encoders import ENCODERS  # local import: avoids a cycle
    from .search.optimizer import CRF_SOLVERS, STRATEGIES
    from .search.samplers import SAMPLERS

    if config.encoder.name not in ENCODERS:
        raise ConfigError(
            f"unknown encoder {config.encoder.name!r}. Available: {sorted(ENCODERS)}"
        )

    encoder = ENCODERS[config.encoder.name]
    if encoder.is_gpu and config.jobs.gpu_workers < 1:
        raise ConfigError(
            f"encoder {encoder.name!r} needs a GPU but jobs.gpu_workers is "
            f"{config.jobs.gpu_workers}. Set it to at least 1, or pick a CPU encoder."
        )

    if not config.search.targets:
        raise ConfigError("search.targets must list at least one VMAF target")
    for target in config.search.targets:
        if not 0.0 < target <= 100.0:
            raise ConfigError(f"search target out of range: {target}")

    strategy = str(config.search.strategy).strip().lower()
    if strategy not in STRATEGIES:
        raise ConfigError(
            f"unknown search.strategy {config.search.strategy!r}. "
            f"Available: {list(STRATEGIES)}"
        )

    sampler = str(config.search.sampler).strip().lower()
    if sampler not in SAMPLERS:
        raise ConfigError(
            f"unknown search.sampler {config.search.sampler!r}. "
            f"Available: {sorted(SAMPLERS)}"
        )

    solver = str(config.search.crf_solver).strip().lower()
    if solver not in CRF_SOLVERS:
        raise ConfigError(
            f"unknown search.crf_solver {config.search.crf_solver!r}. "
            f"Available: {list(CRF_SOLVERS)}"
        )

    if config.search.n_explore < 4:
        raise ConfigError("search.n_explore must be >= 4 to cover the AQ space")

    if config.search.n_init < 0:
        raise ConfigError("search.n_init must be >= 0 (0 = automatic)")

    if config.search.n_refine_configs < 1:
        raise ConfigError("search.n_refine_configs must be >= 1")

    if not 0.0 < config.model.crf_quantile < 1.0:
        raise ConfigError("model.crf_quantile must be strictly between 0 and 1")

    if config.model.estimator not in {"hgb", "rf"}:
        raise ConfigError(f"unknown model.estimator {config.model.estimator!r}")

    if not 0.0 < config.model.val_fraction < 0.9:
        raise ConfigError("model.val_fraction must be in (0, 0.9)")

    if config.vmaf.pool not in {"harmonic_mean", "mean"}:
        raise ConfigError(f"unknown vmaf.pool {config.vmaf.pool!r}")

    if config.segment.min_segment_seconds >= config.segment.max_segment_seconds:
        raise ConfigError(
            "segment.min_segment_seconds must be < segment.max_segment_seconds"
        )

    if config.features.max_frames < 8:
        raise ConfigError("features.max_frames must be >= 8 for stable statistics")


def configs_dir() -> Path:
    """Directory holding the shipped YAML configs.

    They live *inside* the package and are installed as package data, so they are found
    identically from an editable checkout, a wheel install, or a container — the repo
    layout is not assumed to exist at run time.
    """
    return Path(__file__).resolve().parent / "configs"


def default_config_path() -> Path:
    """The base configuration every run starts from."""
    path = configs_dir() / "default.yaml"
    if not path.is_file():
        raise ConfigError(
            f"the shipped default configuration is missing: {path}. "
            "The installation is incomplete; reinstall with "
            "`python scripts/setup.py`."
        )
    return path


def available_overlays() -> dict[str, Path]:
    """Shipped overlays, keyed by bare name (``gpu``) as well as file name."""
    overlays: dict[str, Path] = {}
    for path in sorted(configs_dir().glob("*.yaml")):
        if path.name == "default.yaml":
            continue
        overlays[path.stem] = path
        overlays[path.name] = path
    return overlays


def resolve_overlay(reference: str) -> Path:
    """Resolve a ``--config`` argument to a file.

    Accepts a filesystem path, or the bare name of a shipped overlay (``gpu``,
    ``gpu.yaml``). The name form is what makes ``--config gpu`` work from any working
    directory once the package is installed.
    """
    candidate = Path(reference).expanduser()
    if candidate.is_file():
        return candidate

    overlays = available_overlays()
    if reference in overlays:
        return overlays[reference]

    names = sorted({p.stem for p in overlays.values()})
    raise ConfigError(
        f"config overlay not found: {reference!r}.\n"
        f"Give a path to a YAML file, or one of the shipped overlays: {', '.join(names)}"
    )
