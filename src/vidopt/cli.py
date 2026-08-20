"""Command-line interface.

    vidopt doctor                     check the toolchain and configuration
    vidopt train CORPUS...            phase 1: search + train
    vidopt compress INPUT -o OUT      phase 2: compress with predicted parameters
    vidopt train DATASET              re-train from an existing dev dataset
    vidopt inspect                    show what models exist and how good they are
    vidopt score --vmaf V --ratio R   evaluate the objective function directly

``argparse`` rather than click/typer: it is stdlib, so the airgapped wheelhouse carries
one fewer package.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__
from .config import (
    Config,
    available_overlays,
    default_config_path,
    load_config,
    resolve_overlay,
)
from .errors import VidoptError
from .log import get_logger, setup_logging
from .progress import ProgressEmitter

log = get_logger(__name__)


# --------------------------------------------------------------------------------------
# Shared argument plumbing
# --------------------------------------------------------------------------------------


_ENCODER_CHOICES = (
    "libx265", "libx264", "libsvtav1",
    "hevc_nvenc", "av1_nvenc", "h264_nvenc",
)
_LEVEL_TO_VMAF: dict[int, float] = {1: 85.0, 2: 89.0, 3: 93.0}


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-c", "--config", action="append", default=[], metavar="PATH",
        help=(
            "Config overlay: a YAML path, or a shipped overlay name "
            "(cpu, gpu, quick). Applied over the built-in defaults. Repeatable."
        ),
    )
    parser.add_argument(
        "--set", action="append", default=[], dest="overrides", metavar="KEY=VALUE",
        help="Override one config key, e.g. --set search.n_explore=24. Repeatable.",
    )
    parser.add_argument(
        "--encoder", default=None, choices=_ENCODER_CHOICES, metavar="NAME",
        help=(
            "Encoder / codec for train and compress. "
            "CPU: libx265 (default), libx264, libsvtav1. "
            "GPU: hevc_nvenc, av1_nvenc, h264_nvenc. "
            "Models are encoder-specific — train and compress with the same name."
        ),
    )
    parser.add_argument(
        "--level", type=int, choices=sorted(_LEVEL_TO_VMAF), default=None, metavar="N",
        help=(
            "Quality level shorthand: 1=VMAF 85, 2=VMAF 89, 3=VMAF 93. "
            "For dev/train/compress workflows."
        ),
    )
    parser.add_argument(
        "--cpu-workers", type=int, default=None, metavar="N",
        help="Concurrent CPU encode jobs (default: 4). 0 = auto from core count.",
    )
    parser.add_argument(
        "--gpu-workers", type=int, default=None, metavar="N",
        help=(
            "Concurrent GPU encode jobs (default: 0 = CPU-only). "
            "Set >= 1 with a GPU encoder such as hevc_nvenc."
        ),
    )
    parser.add_argument(
        "--log-level", default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: from config).",
    )
    parser.add_argument(
        "--log-file", default=None, metavar="PATH",
        help="Also write logs to this file (default: <work_dir>/logs/vidopt.log).",
    )
    parser.add_argument(
        "--progress-jsonl", default=None, metavar="PATH",
        help=(
            "Append real-time progress events as JSON Lines for desktop UI consumption."
        ),
    )



def _cli_overrides(args: argparse.Namespace) -> list[str]:
    """Turn dedicated CLI flags into the same KEY=VALUE overrides workers rebuild."""
    overrides = list(args.overrides)
    if getattr(args, "encoder", None):
        overrides.append(f"encoder.name={args.encoder}")
    if getattr(args, "level", None) is not None:
        target = _LEVEL_TO_VMAF[int(args.level)]
        overrides.append(f"search.targets=[{target:g}]")
    if getattr(args, "cpu_workers", None) is not None:
        overrides.append(f"jobs.cpu_workers={args.cpu_workers}")
    if getattr(args, "gpu_workers", None) is not None:
        overrides.append(f"jobs.gpu_workers={args.gpu_workers}")
    return overrides


def _has_strategy_override(args: argparse.Namespace) -> bool:
    """True when the user explicitly set search.strategy via --set."""
    for item in getattr(args, "overrides", []) or []:
        key = str(item).split("=", 1)[0].strip()
        if key == "search.strategy":
            return True
    return False


def _resolve_config(args: argparse.Namespace) -> tuple[Config, list[str], list[str]]:
    """Build the config, and return the layer list so worker processes can rebuild it.

    Subprocesses cannot receive a frozen dataclass graph cheaply, and pickling the whole
    object would silently diverge from the file on disk. Passing the *layers* means every
    worker reconstructs exactly the same configuration.
    """
    paths = [str(default_config_path())] + [
        str(resolve_overlay(str(p))) for p in args.config
    ]
    overrides = _cli_overrides(args)
    config = load_config(paths, overrides)
    log_file = getattr(args, "log_file", None) or str(
        Path(config.paths.work_dir) / "logs" / "vidopt.log"
    )
    setup_logging(args.log_level or config.log_level, log_file=log_file)
    return config, paths, overrides


# --------------------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------------------


def cmd_doctor(args: argparse.Namespace) -> int:
    """Report toolchain status and whether the configured pipeline can run."""
    from .encoding.encoders import ENCODERS
    from .ffmpeg import toolchain

    config, _, _ = _resolve_config(args)
    caps = toolchain.detect(config.ffmpeg.bin_dir)

    print("=" * 74)
    print(f"vidopt {__version__} — toolchain report")
    print("=" * 74)
    print(caps.describe())
    print()

    print(f"configured encoder : {config.encoder.name}")
    print(f"vmaf model         : {config.vmaf.model}")
    print(f"vmaf pooling       : {config.vmaf.pool}")
    print(
        f"search strategy    : {config.search.strategy}  "
        f"sampler={config.search.sampler}  crf_solver={config.search.crf_solver}"
    )
    print(f"targets            : {', '.join(f'{t:g}' for t in config.search.targets)}")
    print(f"cpu workers        : {config.resolved_cpu_workers()}")
    print(f"gpu workers        : {config.jobs.gpu_workers}")
    print(f"models dir         : {config.paths.models_dir}")
    print()

    print("encoder availability:")
    problems: list[str] = []
    for name in sorted(ENCODERS):
        encoder = ENCODERS[name]
        present = caps.has_encoder(encoder.ffmpeg_encoder)
        if not present:
            status = "absent from this build"
        elif toolchain.encoder_works(caps.ffmpeg, encoder.ffmpeg_encoder):
            status = "OK"
        else:
            status = "present but unusable here (no device?)"
        marker = " <-- configured" if name == config.encoder.name else ""
        print(f"  {name:<14} {status}{marker}")
        if name == config.encoder.name and status != "OK":
            problems.append(f"configured encoder {name!r}: {status}")

    print()
    if not caps.has_libvmaf:
        problems.append("libvmaf filter missing: VMAF cannot be measured")
    else:
        print(
            "VMAF            : available"
            + (" (CUDA accelerated)" if caps.has_libvmaf_cuda else " (CPU)")
        )

    if problems:
        print()
        print("PROBLEMS:")
        for problem in problems:
            print(f"  - {problem}")
        print()
        print("Run `python scripts/setup.py` to install the vendored ffmpeg build,")
        print("or switch encoder with --set encoder.name=libx265")
        return 1

    print()
    print("All checks passed.")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from .pipeline.dev import run_dev

    # For threshold training, boundary search is a better default objective match:
    # maximise compression while respecting a VMAF floor.
    if not _has_strategy_override(args):
        args.overrides.append("search.strategy=boundary")
    config, paths, overrides = _resolve_config(args)
    progress = ProgressEmitter(args.progress_jsonl).emit if args.progress_jsonl else None
    result = run_dev(
        args.corpus,
        config,
        paths,
        overrides,
        limit=args.limit,
        skip_training=False,
        resume=args.resume,
        progress=progress,
    )

    print()
    print("=" * 74)
    print("training complete")
    print("=" * 74)
    print(f"sources    {result.n_sources}")
    print(f"segments   {result.n_segments}")
    print(f"rows       {result.n_rows}  ({result.n_infeasible} infeasible)")
    print(f"dataset    {result.dataset_path}")
    print(f"elapsed    {result.seconds:.1f}s")
    if result.reports:
        print()
        print("models:")
        for report in result.reports:
            print(f"  {report.summary()}")
            print(f"    -> {report.bundle_path}")
    return 0


def cmd_compress(args: argparse.Namespace) -> int:
    from .pipeline.production import compress

    config, paths, overrides = _resolve_config(args)
    target = _LEVEL_TO_VMAF[int(args.level)] if args.level is not None else args.target
    progress = ProgressEmitter(args.progress_jsonl).emit if args.progress_jsonl else None
    result = compress(
        args.input,
        args.output,
        target,
        config,
        paths,
        overrides,
        verify=args.verify,
        keep_work=args.keep_work,
        resume=args.resume,
        progress=progress,
    )

    print()
    print("=" * 74)
    print(result.summary())
    print("=" * 74)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    from .modeling.bundle import ModelBundle, list_bundles

    config, _, _ = _resolve_config(args)
    root = Path(args.models_dir or config.paths.models_dir)
    if not root.is_dir():
        print(f"no models directory at {root}. Run `vidopt train` first.")
        return 1

    found = list_bundles(root)
    if not found:
        print(f"no model bundles under {root}. Run `vidopt train` first.")
        return 1

    for directory in found:
        try:
            bundle = ModelBundle.load(directory)
        except VidoptError as exc:
            print(f"{directory}: UNUSABLE — {exc}")
            continue
        meta = bundle.metadata
        metrics = meta.metrics
        print("-" * 74)
        print(f"{directory}")
        if "vmaf_target" in meta.feature_names:
            trained = meta.training_targets or metrics.get("training_targets", [])
            if trained:
                text = f"{min(trained):g}..{max(trained):g}"
            else:
                text = "unknown"
            print(f"  encoder       {meta.encoder}   VMAF range {text}")
        else:
            print(f"  encoder       {meta.encoder}   target VMAF {meta.target:g}")
        print(
            f"  trained on    {meta.n_train} row(s) "
            f"from {len(meta.training_sources)} source(s)"
        )
        print(f"  created       {meta.created_at}")
        if metrics:
            hit = metrics.get("crf_hit_rate")
            print(f"  crf MAE       {metrics.get('crf_mae', float('nan')):.3f}")
            print(f"  crf R2        {metrics.get('crf_r2', float('nan')):.3f}")
            print(
                "  hit rate      "
                + ("n/a" if hit is None else f"{hit * 100:.0f}%")
                + "   (fraction of held-out segments that would meet the target)"
            )
            print(f"  aq_mode acc   {metrics.get('aq_mode_accuracy', float('nan')):.3f}")
            print(f"  crf quantile  {metrics.get('crf_quantile', 'n/a')}")
    print("-" * 74)
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    from .scoring import compression_score

    rate = args.rate if args.rate is not None else (1.0 / args.ratio)
    breakdown = compression_score(args.vmaf, rate, args.target)
    print(f"vmaf              {args.vmaf:.2f}")
    print(f"compression rate  {rate:.4f}  ({1.0 / rate:.2f}x)")
    print(f"target            {args.target:g}")
    print(f"compression term  {breakdown.compression_component:.4f}")
    print(f"quality term      {breakdown.quality_component:.4f}")
    print(f"score             {breakdown.score:.4f}")
    print(f"reason            {breakdown.reason}")
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    if args.list_overlays:
        print(f"built-in default: {default_config_path()}")
        print("shipped overlays (use the bare name with --config):")
        for name, path in sorted(
            {p.stem: p for p in available_overlays().values()}.items()
        ):
            print(f"  {name:<8} {path}")
        return 0

    config, paths, overrides = _resolve_config(args)
    # Provenance goes to stderr so stdout stays valid JSON and can be piped into jq.
    print(f"# layers: {', '.join(paths)}", file=sys.stderr)
    if overrides:
        print(f"# overrides: {', '.join(overrides)}", file=sys.stderr)
    print(config.dumps())
    return 0


# --------------------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vidopt",
        description="Scene-adaptive video compression with learned encoder parameters.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "typical workflow:\n"
            "  vidopt doctor --config cpu\n"
            "  vidopt train video/corpus --config cpu --encoder libx265 --level 2 --resume\n"
            "  vidopt compress in.mp4 -o out.mp4 --encoder libx265 --level 2 --verify\n"
            "\n"
            "scalability:\n"
            "  --cpu-workers N   concurrent CPU jobs (default 4; 0=auto)\n"
            "  --gpu-workers N   concurrent GPU jobs (default 0)\n"
            "  --encoder NAME    libx265|libx264|libsvtav1|hevc_nvenc|av1_nvenc|h264_nvenc\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"vidopt {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_doctor = sub.add_parser("doctor", help="check toolchain and configuration")
    _add_common(p_doctor)
    p_doctor.set_defaults(func=cmd_doctor)

    p_train = sub.add_parser(
        "train", help="phase 1: segment corpus, search parameters, and train models"
    )
    p_train.add_argument("corpus", nargs="+", help="Video files and/or directories.")
    p_train.add_argument("--limit", type=int, default=None, help="Use at most N sources.")
    p_train.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue an interrupted `vidopt train` in the same work directory. "
            "Reuses existing scene cuts and skips segments already searched."
        ),
    )
    _add_common(p_train)
    p_train.set_defaults(func=cmd_train)

    p_compress = sub.add_parser(
        "compress", help="phase 2: compress a video using the trained models"
    )
    p_compress.add_argument("input", help="Input video.")
    p_compress.add_argument("-o", "--output", required=True, help="Output video.")
    p_compress.add_argument(
        "--target", type=float, default=89.0, help="VMAF target (default: 89)."
    )
    p_compress.add_argument(
        "--verify", action="store_true",
        help="Measure the final VMAF and report the objective score.",
    )
    p_compress.add_argument(
        "--keep-work", action="store_true", help="Keep intermediate segments."
    )
    p_compress.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an interrupted compression from existing intermediate files in the "
            "work directory."
        ),
    )
    p_compress.add_argument("--json", action="store_true", help="Also print JSON.")
    _add_common(p_compress)
    p_compress.set_defaults(func=cmd_compress)

    p_inspect = sub.add_parser("inspect", help="show trained models and their metrics")
    p_inspect.add_argument("--models-dir", default=None)
    _add_common(p_inspect)
    p_inspect.set_defaults(func=cmd_inspect)

    p_score = sub.add_parser("score", help="evaluate the objective function directly")
    p_score.add_argument("--vmaf", type=float, required=True)
    group = p_score.add_mutually_exclusive_group(required=True)
    group.add_argument("--rate", type=float, help="out_bytes / ref_bytes")
    group.add_argument("--ratio", type=float, help="compression factor, e.g. 8 for 8x")
    p_score.add_argument("--target", type=float, default=89.0)
    _add_common(p_score)
    p_score.set_defaults(func=cmd_score)

    p_config = sub.add_parser("config", help="print the effective configuration")
    p_config.add_argument(
        "--list-overlays", action="store_true",
        help="List the shipped config overlays and where they live.",
    )
    _add_common(p_config)
    p_config.set_defaults(func=cmd_config)

    return parser


def main(argv: list[str] | None = None) -> int:
    # Before OpenCV/ffmpeg first init: drop H.264 MMCO chatter from stream-copied cuts.
    os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "8")
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging("INFO")
    try:
        return int(args.func(args))
    except VidoptError as exc:
        # Expected failure modes get a clean message, not a traceback.
        log.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        log.warning("interrupted")
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
