"""The dev-mode dataset: one row per (segment, VMAF target).

Stored as CSV. Datasets here are ~10^3 rows, so the bytes saved by a binary format are
irrelevant next to being able to inspect the labels with ``less`` or a spreadsheet when
a model behaves oddly. It also removes a dependency.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..errors import ModelError
from ..features.extract import FEATURE_NAMES
from ..log import get_logger

log = get_logger(__name__)

LABEL_NAMES: tuple[str, ...] = ("crf", "aq_mode", "aq_strength")

META_NAMES: tuple[str, ...] = (
    "source",
    "segment",
    "segment_hash",
    "target",
    "feasible",
    "vmaf",
    "rate",
    "ratio",
    "score",
    "encoder",
    "n_trials",
)

COLUMNS: tuple[str, ...] = META_NAMES + FEATURE_NAMES + LABEL_NAMES


@dataclass
class Row:
    """One training example."""

    meta: dict[str, object]
    features: dict[str, float]
    labels: dict[str, float]

    def as_record(self) -> dict[str, object]:
        return {**self.meta, **self.features, **self.labels}


def write_dataset(rows: list[Row], path: str | Path) -> Path:
    """Write rows to CSV, creating parent directories."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_record())
    log.info("wrote dataset: %s (%d row(s))", out, len(rows))
    return out


def read_dataset(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.is_file():
        raise ModelError(f"dataset not found: {p}")
    with p.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ModelError(f"dataset is empty: {p}")
    missing = set(COLUMNS) - set(rows[0])
    if missing:
        raise ModelError(
            f"dataset {p} is missing column(s): {sorted(missing)}. "
            "It was probably written by a different version; re-run `vidopt dev`."
        )
    return rows


def to_matrices(
    rows: list[dict[str, str]], target: float, *, feasible_only: bool = True
) -> tuple[np.ndarray, dict[str, np.ndarray], list[dict[str, str]]]:
    """Build (X, {label: y}, kept_rows) for one VMAF target.

    Infeasible rows — segments where no parameters reached the target — are excluded by
    default. Training on them would teach the model to aim at an unreachable point.
    """
    selected = [r for r in rows if abs(float(r["target"]) - target) < 1e-6]
    if feasible_only:
        selected = [r for r in selected if str(r["feasible"]).lower() in {"true", "1"}]

    if not selected:
        raise ModelError(
            f"no usable training rows for target {target:g}. "
            "Either dev mode found no feasible parameters, or the target is not in the "
            "dataset."
        )

    features = np.array(
        [[float(r[name]) for name in FEATURE_NAMES] for r in selected], dtype=np.float64
    )
    labels = {
        name: np.array([float(r[name]) for r in selected], dtype=np.float64)
        for name in LABEL_NAMES
    }
    return features, labels, selected


def group_labels(rows: list[dict[str, str]]) -> np.ndarray:
    """Source-video identity per row, for grouped train/validation splitting.

    Segments cut from the same source share content, so a random split leaks: the model
    sees near-duplicates of its validation set during training and the reported hit rate
    is optimistic. Splitting by source video avoids that.
    """
    return np.array([r["source"] for r in rows], dtype=object)
