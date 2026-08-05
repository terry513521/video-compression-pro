"""Model bundles: a versioned, self-describing artifact.

The reference shipped a bare ``pickle`` of an sklearn ``Pipeline`` containing seven
custom transformer classes. It could only be loaded with that exact source tree on
``sys.path`` — ``utils/__init__.py`` re-exported the classes purely to satisfy the
unpickler — and carried no record of what it was trained on or what features it expects.

A bundle here is a directory:

    models/libx265/target_89/
        estimators.joblib   the three fitted estimators
        metadata.json       schema version, encoder, target, feature order, metrics,
                            training corpus, config snapshot, creation time

Loading validates the schema version and that the feature order matches what the current
extractor produces, so a stale bundle fails loudly instead of silently reading features
in the wrong order.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from ..encoding.params import EncodeParams
from ..errors import ModelError
from ..features.extract import FEATURE_NAMES
from ..log import get_logger

log = get_logger(__name__)

SCHEMA_VERSION = 1


@dataclass
class BundleMetadata:
    """Everything needed to interpret, audit and reproduce a bundle."""

    schema_version: int
    encoder: str
    target: float
    feature_names: list[str]
    label_names: list[str]
    n_train: int
    n_val: int
    metrics: dict[str, Any] = field(default_factory=dict)
    feature_ranges: dict[str, list[float]] = field(default_factory=dict)
    """Per-feature [min, max] over the training rows. Used to detect input that the
    model has no evidence about -- tree ensembles cannot extrapolate, they clamp to the
    nearest trained leaf and return a confident-looking number regardless."""
    training_sources: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "encoder": self.encoder,
            "target": self.target,
            "feature_names": self.feature_names,
            "label_names": self.label_names,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "metrics": self.metrics,
            "feature_ranges": self.feature_ranges,
            "training_sources": self.training_sources,
            "config": self.config,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BundleMetadata:
        return cls(
            schema_version=int(data["schema_version"]),
            encoder=str(data["encoder"]),
            target=float(data["target"]),
            feature_names=list(data["feature_names"]),
            label_names=list(data["label_names"]),
            n_train=int(data.get("n_train", 0)),
            n_val=int(data.get("n_val", 0)),
            metrics=dict(data.get("metrics", {})),
            feature_ranges={
                k: [float(v[0]), float(v[1])]
                for k, v in (data.get("feature_ranges") or {}).items()
            },
            training_sources=list(data.get("training_sources", [])),
            config=dict(data.get("config", {})),
            created_at=str(data.get("created_at", "")),
        )


@dataclass
class ModelBundle:
    """Fitted estimators plus their metadata."""

    estimators: dict[str, Any]
    metadata: BundleMetadata

    # ---------------- persistence ----------------

    def save(self, directory: str | Path) -> Path:
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.estimators, out / "estimators.joblib", compress=3)
        meta = self.metadata
        if not meta.created_at:
            meta.created_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        (out / "metadata.json").write_text(
            json.dumps(meta.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
        log.info("saved model bundle: %s", out)
        return out

    @classmethod
    def load(cls, directory: str | Path) -> ModelBundle:
        path = Path(directory)
        meta_path = path / "metadata.json"
        est_path = path / "estimators.joblib"
        if not meta_path.is_file() or not est_path.is_file():
            raise ModelError(
                f"{path} is not a model bundle (expected metadata.json and "
                "estimators.joblib). Train one with `vidopt train`."
            )

        try:
            metadata = BundleMetadata.from_dict(
                json.loads(meta_path.read_text(encoding="utf-8"))
            )
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            raise ModelError(f"corrupt bundle metadata in {path}: {exc}") from exc

        if metadata.schema_version != SCHEMA_VERSION:
            raise ModelError(
                f"bundle {path} has schema version {metadata.schema_version}, "
                f"this build expects {SCHEMA_VERSION}. Retrain it."
            )

        if list(metadata.feature_names) != list(FEATURE_NAMES):
            raise ModelError(
                f"bundle {path} was trained on a different feature set.\n"
                f"  bundle:  {metadata.feature_names}\n"
                f"  current: {list(FEATURE_NAMES)}\n"
                "Feature order is positional, so this must not be ignored. Retrain."
            )

        estimators = joblib.load(est_path)
        return cls(estimators=estimators, metadata=metadata)

    # ---------------- inference ----------------

    def predict(self, features: np.ndarray) -> list[EncodeParams]:
        """Predict parameters for a batch of feature vectors.

        Args:
            features: Shape (n_samples, n_features), columns in ``FEATURE_NAMES`` order.

        Returns:
            One :class:`EncodeParams` per row, unclamped — the caller clamps into the
            encoder's space, since only the caller knows which encoder is in use.
        """
        matrix = np.atleast_2d(np.asarray(features, dtype=np.float64))
        expected = len(self.metadata.feature_names)
        if matrix.shape[1] != expected:
            raise ModelError(
                f"expected {expected} features, got {matrix.shape[1]}"
            )

        crf = np.asarray(self.estimators["crf"].predict(matrix), dtype=np.float64)
        aq_mode = np.asarray(self.estimators["aq_mode"].predict(matrix))
        strength = np.asarray(
            self.estimators["aq_strength"].predict(matrix), dtype=np.float64
        )

        return [
            EncodeParams(
                crf=float(crf[i]),
                aq_mode=int(round(float(aq_mode[i]))),
                aq_strength=float(strength[i]),
            )
            for i in range(matrix.shape[0])
        ]


    # ---------------- domain checking ----------------

    def out_of_domain(
        self, features: np.ndarray, *, tolerance: float = 0.15
    ) -> list[tuple[str, float, float, float]]:
        """Features that fall outside the range the model was trained on.

        Gradient-boosted trees do not extrapolate: presented with a 4K frame after being
        trained only on 720p, they return whatever the nearest trained leaf says, with no
        indication that the question was unanswerable. Since production input is
        arbitrary, that silent confidence is the dangerous failure mode — so it is
        detected and reported rather than hidden.

        Args:
            features: One row, in ``FEATURE_NAMES`` order.
            tolerance: Fraction of the trained span allowed outside it before a feature
                counts as out of domain. 0.15 ignores mild edge cases.

        Returns:
            ``(name, value, trained_min, trained_max)`` for each offending feature.
        """
        if not self.feature_ranges:
            return []

        row = np.atleast_2d(np.asarray(features, dtype=np.float64))[0]
        offenders: list[tuple[str, float, float, float]] = []

        for index, name in enumerate(self.metadata.feature_names):
            bounds = self.feature_ranges.get(name)
            if not bounds:
                continue
            low, high = bounds
            span = high - low
            slack = abs(span) * tolerance if span > 0 else max(abs(high), 1.0) * tolerance
            value = float(row[index])
            if value < low - slack or value > high + slack:
                offenders.append((name, value, low, high))
        return offenders

    @property
    def feature_ranges(self) -> dict[str, list[float]]:
        return self.metadata.feature_ranges


def bundle_dir(models_root: str | Path, encoder: str, target: float) -> Path:
    """Canonical location for a bundle: ``<root>/<encoder>/target_<T>``."""
    return Path(models_root) / encoder / f"target_{target:g}"


def find_bundle(models_root: str | Path, encoder: str, target: float) -> ModelBundle:
    """Load the bundle for an (encoder, target) pair, with a helpful error if absent."""
    directory = bundle_dir(models_root, encoder, target)
    if not directory.is_dir():
        available = sorted(
            p.relative_to(models_root).as_posix()
            for p in Path(models_root).glob("*/target_*")
            if p.is_dir()
        ) if Path(models_root).is_dir() else []
        raise ModelError(
            f"no model for encoder {encoder!r} at VMAF target {target:g} "
            f"(looked in {directory}).\n"
            + (
                f"Available: {', '.join(available)}"
                if available
                else "No models found at all. Run `vidopt dev` first."
            )
        )
    return ModelBundle.load(directory)
