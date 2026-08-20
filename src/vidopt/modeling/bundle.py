"""Model bundles: versioned, self-describing artifacts.

Schema v1 (default): ``models/<encoder>/target_<T>/`` with 8 scene features and separate
``crf.joblib``, ``aq_mode.joblib``, ``aq_strength.joblib`` heads.

Schema v2 (legacy unified): one bundle per encoder at ``models/<encoder>/`` with
``estimators.joblib`` and ``vmaf_target`` as a model input — still loadable.
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
from ..features.extract import FEATURE_NAMES, MODEL_FEATURE_NAMES, VMAF_TARGET_FEATURE, with_vmaf_target
from .dataset import LABEL_NAMES
from ..log import get_logger

log = get_logger(__name__)

SCHEMA_VERSION = 2
LEGACY_SCHEMA_VERSION = 1


@dataclass
class BundleMetadata:
    """Everything needed to interpret, audit and reproduce a bundle."""

    schema_version: int
    encoder: str
    feature_names: list[str]
    label_names: list[str]
    n_train: int
    n_val: int
    target: float | None = None
    """Legacy v1 only — fixed VMAF target the bundle was trained for."""
    training_targets: list[float] = field(default_factory=list)
    """VMAF targets present in the training labels (v2)."""
    metrics: dict[str, Any] = field(default_factory=dict)
    feature_ranges: dict[str, list[float]] = field(default_factory=dict)
    training_sources: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "schema_version": self.schema_version,
            "encoder": self.encoder,
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
        if self.target is not None:
            out["target"] = self.target
        if self.training_targets:
            out["training_targets"] = self.training_targets
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BundleMetadata:
        return cls(
            schema_version=int(data["schema_version"]),
            encoder=str(data["encoder"]),
            target=float(data["target"]) if "target" in data else None,
            feature_names=list(data["feature_names"]),
            label_names=list(data["label_names"]),
            n_train=int(data.get("n_train", 0)),
            n_val=int(data.get("n_val", 0)),
            training_targets=[float(t) for t in data.get("training_targets", [])],
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
        if self.metadata.schema_version == LEGACY_SCHEMA_VERSION:
            for name in LABEL_NAMES:
                joblib.dump(self.estimators[name], out / f"{name}.joblib", compress=3)
        else:
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
        if not meta_path.is_file():
            raise ModelError(
                f"{path} is not a model bundle (expected metadata.json). "
                "Train one with `vidopt train`."
            )

        try:
            metadata = BundleMetadata.from_dict(
                json.loads(meta_path.read_text(encoding="utf-8"))
            )
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            raise ModelError(f"corrupt bundle metadata in {path}: {exc}") from exc

        if metadata.schema_version not in (LEGACY_SCHEMA_VERSION, SCHEMA_VERSION):
            raise ModelError(
                f"bundle {path} has schema version {metadata.schema_version}, "
                f"but this build expects {LEGACY_SCHEMA_VERSION} or {SCHEMA_VERSION}. "
                "Retrain it."
            )

        if metadata.schema_version == SCHEMA_VERSION:
            expected = list(MODEL_FEATURE_NAMES)
        else:
            expected = list(FEATURE_NAMES)
        if list(metadata.feature_names) != expected:
            raise ModelError(
                f"bundle {path} was trained on a different feature set.\n"
                f"  bundle:  {metadata.feature_names}\n"
                f"  current: {expected}"
            )

        if metadata.schema_version == LEGACY_SCHEMA_VERSION:
            head_paths = {name: path / f"{name}.joblib" for name in LABEL_NAMES}
            if all(p.is_file() for p in head_paths.values()):
                estimators = {name: joblib.load(p) for name, p in head_paths.items()}
            elif (path / "estimators.joblib").is_file():
                estimators = joblib.load(path / "estimators.joblib")
            else:
                missing = [name for name, p in head_paths.items() if not p.is_file()]
                raise ModelError(
                    f"{path} is missing model head(s): {missing}. "
                    "Retrain with `vidopt train`."
                )
        else:
            est_path = path / "estimators.joblib"
            if not est_path.is_file():
                raise ModelError(
                    f"{path} is not a model bundle (expected estimators.joblib). "
                    "Train one with `vidopt train`."
                )
            estimators = joblib.load(est_path)
        return cls(estimators=estimators, metadata=metadata)

    # ---------------- inference ----------------

    def predict(
        self,
        features: np.ndarray,
        *,
        vmaf_target: float | None = None,
    ) -> list[EncodeParams]:
        """Predict parameters for a batch of feature vectors."""
        matrix = np.atleast_2d(np.asarray(features, dtype=np.float64))
        uses_vmaf = VMAF_TARGET_FEATURE in self.metadata.feature_names
        if uses_vmaf:
            if matrix.shape[1] == len(FEATURE_NAMES):
                if vmaf_target is None:
                    raise ModelError("this bundle requires vmaf_target at predict time")
                matrix = np.vstack(
                    [with_vmaf_target(matrix[i], vmaf_target) for i in range(matrix.shape[0])]
                )
            elif matrix.shape[1] != len(MODEL_FEATURE_NAMES):
                raise ModelError(
                    f"expected {len(FEATURE_NAMES)} or {len(MODEL_FEATURE_NAMES)} features, got {matrix.shape[1]}"
                )
        elif matrix.shape[1] != len(FEATURE_NAMES):
            raise ModelError(f"expected {len(FEATURE_NAMES)} features, got {matrix.shape[1]}")

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

    def out_of_domain(
        self,
        features: np.ndarray,
        *,
        vmaf_target: float | None = None,
        tolerance: float = 0.15,
    ) -> list[tuple[str, float, float, float]]:
        if not self.feature_ranges:
            return []

        matrix = np.atleast_2d(np.asarray(features, dtype=np.float64))
        uses_vmaf = VMAF_TARGET_FEATURE in self.metadata.feature_names
        if uses_vmaf and matrix.shape[1] == len(FEATURE_NAMES):
            if vmaf_target is None:
                return []
            row = with_vmaf_target(matrix[0], vmaf_target)
        else:
            row = matrix[0]

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

    # For v1 bundles, the directory already pins the VMAF target.


def bundle_dir(models_root: str | Path, encoder: str) -> Path:
    """Legacy unified v2 location: ``<root>/<encoder>/``."""
    return Path(models_root) / encoder


def legacy_bundle_dir(models_root: str | Path, encoder: str, target: float) -> Path:
    """Default v1 location: ``<root>/<encoder>/target_<T>/``."""
    return Path(models_root) / encoder / f"target_{target:g}"


def find_bundle(
    models_root: str | Path, encoder: str, target: float
) -> ModelBundle:
    """Load model by encoder and VMAF target (v1 first, unified v2 fallback)."""
    root = Path(models_root)
    legacy = legacy_bundle_dir(root, encoder, target)
    if (legacy / "metadata.json").is_file():
        return ModelBundle.load(legacy)

    v2 = bundle_dir(root, encoder)
    if (v2 / "metadata.json").is_file():
        bundle = ModelBundle.load(v2)
        trained = bundle.metadata.training_targets
        if trained:
            low, high = min(trained), max(trained)
            if target < low - 0.5 or target > high + 0.5:
                raise ModelError(
                    f"target {target:g} is outside training range {low:g}..{high:g} "
                    f"for encoder {encoder!r}"
                )
        return bundle

    raise ModelError(
        f"no model for encoder {encoder!r} target {target:g} "
        f"(looked in {legacy} and {v2}). Run `vidopt train` first."
    )


def list_bundles(models_root: str | Path) -> list[Path]:
    """All loadable bundle directories under ``models_root``."""
    root = Path(models_root)
    if not root.is_dir():
        return []
    found: list[Path] = []
    for meta in root.glob("**/target_*/metadata.json"):
        if meta.parent.is_dir():
            found.append(meta.parent)
    for meta in root.glob("*/metadata.json"):
        parent = meta.parent
        if parent.name.startswith("target_"):
            continue
        if (parent / "estimators.joblib").is_file() or all(
            (parent / f"{name}.joblib").is_file() for name in LABEL_NAMES
        ):
            found.append(parent)
    return sorted(dict.fromkeys(found))
