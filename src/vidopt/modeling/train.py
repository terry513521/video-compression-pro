"""Training.

Default layout: one bundle per encoder **and VMAF target** at
``models/<encoder>/target_<T>/`` (schema v1). Three heads per bundle: ``crf`` (regression),
``aq_mode`` (classification), ``aq_strength`` (regression). Scene features only — the
VMAF target is fixed by the directory name.

**The important design point is the CRF loss.** The objective drops to *zero* below
``target - 5`` (see ``scoring.py``), so the cost of prediction error is asymmetric:
predicting a CRF that is too high risks the cliff and loses everything, while predicting
too low merely wastes some bits. Squared error treats those two mistakes as equally bad,
which is simply the wrong loss for this problem.

The CRF head is therefore fit with **quantile loss below the median**
(``model.crf_quantile``, default 0.35), so the model is conservative by construction
rather than by a post-hoc fudge factor. The knob is a single documented number:

    q = 0.50   unbiased, maximum compression, highest risk of missing the target
    q = 0.35   default, small quality margin
    q = 0.20   cautious, for content unlike the training corpus

Validation reports the **VMAF hit rate** — the fraction of held-out segments whose
predicted CRF is at or below the CRF that was actually measured to meet the target —
alongside MAE and R2. A model with excellent R2 and a 60% hit rate is a bad model here,
and the report makes that visible.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score

from ..config import Config
from ..errors import ModelError
from ..features.extract import FEATURE_NAMES
from ..log import get_logger
from .bundle import LEGACY_SCHEMA_VERSION, BundleMetadata, ModelBundle, legacy_bundle_dir
from .dataset import LABEL_NAMES, group_labels, to_matrices, training_targets

log = get_logger(__name__)

MIN_TRAINING_ROWS = 8


@dataclass
class TrainReport:
    """Per-target training outcome, suitable for printing or serialising."""

    encoder: str
    target: float
    training_targets: list[float]
    n_train: int
    n_val: int
    metrics: dict[str, Any]
    bundle_path: str

    def summary(self) -> str:
        m = self.metrics
        hit = m.get("crf_hit_rate")
        hit_text = "n/a" if hit is None else f"{hit * 100:.0f}%"
        return (
            f"encoder {self.encoder}  target {self.target:g}  "
            f"train={self.n_train} val={self.n_val}  "
            f"crf MAE={m.get('crf_mae', float('nan')):.2f} "
            f"R2={m.get('crf_r2', float('nan')):.2f}  "
            f"hit-rate={hit_text}  "
            f"aq_mode acc={m.get('aq_mode_accuracy', float('nan')):.2f}"
        )


def _grouped_split(
    groups: np.ndarray, val_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split indices by source video, not at random.

    Segments from the same source share content; a random split would put near-duplicates
    on both sides and inflate the reported metrics.
    """
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)

    n_val_groups = max(1, int(round(len(unique) * val_fraction)))
    if n_val_groups >= len(unique):
        n_val_groups = max(1, len(unique) - 1)

    val_groups = set(shuffled[:n_val_groups].tolist())
    mask = np.array([g in val_groups for g in groups], dtype=bool)
    return np.where(~mask)[0], np.where(mask)[0]


def _make_regressor(config: Config, *, quantile: float | None) -> Any:
    m = config.model
    if m.estimator == "rf":
        # RandomForest has no quantile loss; the conservative behaviour is then supplied
        # by the caller's residual shift. Warn so the difference is not a surprise.
        if quantile is not None:
            log.warning(
                "estimator='rf' does not support quantile loss; the CRF head will be "
                "fit with squared error and will not be conservative. "
                "Use estimator='hgb' for the asymmetric loss."
            )
        return RandomForestRegressor(
            n_estimators=max(50, m.max_iter // 2),
            max_depth=m.max_depth,
            min_samples_leaf=m.min_samples_leaf,
            random_state=m.seed,
            n_jobs=-1,
        )

    kwargs: dict[str, Any] = {
        "max_iter": m.max_iter,
        "learning_rate": m.learning_rate,
        "max_depth": m.max_depth,
        "min_samples_leaf": m.min_samples_leaf,
        "random_state": m.seed,
        "early_stopping": False,
    }
    if quantile is not None:
        kwargs["loss"] = "quantile"
        kwargs["quantile"] = quantile
    return HistGradientBoostingRegressor(**kwargs)


def _make_classifier(config: Config) -> Any:
    m = config.model
    if m.estimator == "rf":
        return RandomForestClassifier(
            n_estimators=max(50, m.max_iter // 2),
            max_depth=m.max_depth,
            min_samples_leaf=m.min_samples_leaf,
            random_state=m.seed,
            n_jobs=-1,
        )
    return HistGradientBoostingClassifier(
        max_iter=m.max_iter,
        learning_rate=m.learning_rate,
        max_depth=m.max_depth,
        min_samples_leaf=m.min_samples_leaf,
        random_state=m.seed,
        early_stopping=False,
    )


class _ConstantEstimator:
    """Stand-in for a head whose training labels have only one value.

    Real estimators refuse to fit a single-class target; rather than special-casing that
    at every call site, the bundle gets an object with the same ``predict`` interface.
    """

    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: N803 - sklearn convention
        return np.full(np.atleast_2d(X).shape[0], self.value, dtype=np.float64)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<ConstantEstimator {self.value}>"


def _fit_head(
    name: str, X: np.ndarray, y: np.ndarray, config: Config  # noqa: N803
) -> Any:
    unique = np.unique(y)
    if unique.size == 1:
        log.info(
            "head %r has a single label value (%g); using a constant", name, unique[0]
        )
        return _ConstantEstimator(float(unique[0]))

    if name == "aq_mode":
        return _make_classifier(config).fit(X, y.astype(int))
    quantile = config.model.crf_quantile if name == "crf" else None
    return _make_regressor(config, quantile=quantile).fit(X, y)


def _evaluate(
    estimators: dict[str, Any],
    X_val: np.ndarray,  # noqa: N803
    y_val: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Validation metrics, headlined by the CRF hit rate."""
    metrics: dict[str, Any] = {}
    if X_val.shape[0] == 0:
        return {"note": "no validation rows (corpus too small to hold one out)"}

    crf_pred = np.asarray(estimators["crf"].predict(X_val), dtype=np.float64)
    crf_true = y_val["crf"]
    metrics["crf_mae"] = float(mean_absolute_error(crf_true, crf_pred))
    metrics["crf_bias"] = float(np.mean(crf_pred - crf_true))
    metrics["crf_r2"] = (
        float(r2_score(crf_true, crf_pred)) if len(crf_true) > 1 else float("nan")
    )

    # The metric that matters: a predicted CRF at or below the measured optimum would
    # have met the VMAF target, because VMAF is non-increasing in CRF. A small tolerance
    # absorbs quantisation.
    metrics["crf_hit_rate"] = float(np.mean(crf_pred <= crf_true + 0.5))
    metrics["crf_overshoot_p90"] = float(
        np.percentile(np.maximum(crf_pred - crf_true, 0.0), 90)
    )

    mode_pred = np.asarray(estimators["aq_mode"].predict(X_val)).astype(int)
    metrics["aq_mode_accuracy"] = float(
        accuracy_score(y_val["aq_mode"].astype(int), mode_pred)
    )

    strength_pred = np.asarray(
        estimators["aq_strength"].predict(X_val), dtype=np.float64
    )
    metrics["aq_strength_mae"] = float(
        mean_absolute_error(y_val["aq_strength"], strength_pred)
    )
    return metrics


def train_encoder_target(
    rows: list[dict[str, str]],
    config: Config,
    models_root: str | Path,
    target: float,
) -> TrainReport:
    """Train and save one v1 bundle for the configured encoder and VMAF target."""
    X, y, kept = to_matrices(rows, target=target, include_vmaf_target=False)

    if X.shape[0] < config.model.min_training_rows:
        raise ModelError(
            f"only {X.shape[0]} feasible row(s); need at least "
            f"{config.model.min_training_rows}. Add source videos to the dev corpus, "
            "or lower model.min_training_rows for a smoke run."
        )

    groups = group_labels(kept)
    train_idx, val_idx = _grouped_split(
        groups, config.model.val_fraction, config.model.seed
    )

    X_train, X_val = X[train_idx], X[val_idx]
    y_train = {name: y[name][train_idx] for name in LABEL_NAMES}
    y_val = {name: y[name][val_idx] for name in LABEL_NAMES}

    log.info(
        "encoder %s target %g: training on %d row(s), validating on %d",
        config.encoder.name,
        target,
        len(train_idx),
        len(val_idx),
    )

    started = time.monotonic()
    estimators = {
        name: _fit_head(name, X_train, y_train[name], config) for name in LABEL_NAMES
    }
    metrics = _evaluate(estimators, X_val, y_val)
    metrics["fit_seconds"] = round(time.monotonic() - started, 2)
    metrics["crf_quantile"] = config.model.crf_quantile
    metrics["estimator"] = config.model.estimator
    metrics["training_targets"] = [target]

    # Refit on everything once the metrics are recorded: the held-out rows are scarce
    # and valuable, and the metrics above already describe generalisation honestly.
    final_estimators = {
        name: _fit_head(name, X, y[name], config) for name in LABEL_NAMES
    }

    metadata = BundleMetadata(
        schema_version=LEGACY_SCHEMA_VERSION,
        encoder=config.encoder.name,
        target=target,
        feature_names=list(FEATURE_NAMES),
        label_names=list(LABEL_NAMES),
        n_train=int(X.shape[0]),
        n_val=int(len(val_idx)),
        training_targets=[target],
        metrics=metrics,
        # Recorded so production can tell when an input is outside anything the model
        # has evidence about -- see ModelBundle.out_of_domain.
        feature_ranges={
            name: [float(X[:, i].min()), float(X[:, i].max())]
            for i, name in enumerate(FEATURE_NAMES)
        },
        training_sources=sorted({str(g) for g in groups}),
        config={
            "encoder": config.encoder.name,
            "preset": config.encoder.preset,
            "vmaf_model": config.vmaf.model,
            "model": {
                "estimator": config.model.estimator,
                "crf_quantile": config.model.crf_quantile,
                "max_iter": config.model.max_iter,
                "learning_rate": config.model.learning_rate,
            },
            "features": {
                "max_frames": config.features.max_frames,
                "analysis_width": config.features.analysis_width,
            },
        },
    )

    directory = legacy_bundle_dir(models_root, config.encoder.name, target)
    ModelBundle(estimators=final_estimators, metadata=metadata).save(directory)

    report = TrainReport(
        encoder=config.encoder.name,
        target=target,
        training_targets=[target],
        n_train=int(X.shape[0]),
        n_val=int(len(val_idx)),
        metrics=metrics,
        bundle_path=str(directory),
    )
    log.info(report.summary())
    return report


def train_all(
    rows: list[dict[str, str]], config: Config, models_root: str | Path
) -> list[TrainReport]:
    """Train one v1 bundle per VMAF target present in the dataset."""
    reports: list[TrainReport] = []
    for target in training_targets(rows):
        reports.append(train_encoder_target(rows, config, models_root, target))
    return reports
