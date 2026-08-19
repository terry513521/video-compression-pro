"""Sequential / model-based search over the 3-D parameter cube.

These run after a small space-filling initial design. They do not replace the 1-D CRF
solver: they spend the remaining ``n_explore`` budget proposing new (crf, AQ) points,
then the usual CRF refinement still runs on the best AQ settings.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable

import numpy as np

from ..config import Config
from ..encoding.encoders import Encoder
from ..encoding.params import EncodeParams
from ..log import get_logger
from .cache import TrialRecord
from .samplers import get_sampler, params_from_unit, params_to_unit

log = get_logger(__name__)

EvaluateFn = Callable[[EncodeParams], TrialRecord | None]


def n_init_points(config: Config) -> int:
    n_explore = max(4, int(config.search.n_explore))
    configured = int(config.search.n_init)
    if configured > 0:
        return min(configured, n_explore)
    return min(n_explore, max(4, n_explore // 2))


def _fitness(trial: TrialRecord, target: float) -> float:
    from ..scoring import compression_score

    score = compression_score(trial.vmaf, trial.rate, target).score
    if score > 0:
        return score
    return 1e-6 * min(trial.vmaf, target)


def _keys(trials: list[TrialRecord]) -> set[str]:
    return {t.params.key() for t in trials}


def _initial_design(
    evaluate: EvaluateFn,
    encoder: Encoder,
    config: Config,
    n_init: int,
) -> list[TrialRecord]:
    sampler = get_sampler(config.search.sampler)
    points = sampler(encoder.space, n_init, config.search.seed)
    trials: list[TrialRecord] = []
    for params in points:
        record = evaluate(params)
        if record is not None:
            trials.append(record)
    return trials


def _candidate_pool(
    encoder: Encoder,
    config: Config,
    seen: set[str],
    rng: np.random.Generator,
    n: int = 96,
) -> list[EncodeParams]:
    sampler = get_sampler(config.search.sampler)
    extra = sampler(encoder.space, n, int(rng.integers(0, 2**31 - 1)))
    random = get_sampler("random")(encoder.space, n // 2, int(rng.integers(0, 2**31 - 1)))
    out: list[EncodeParams] = []
    for params in extra + random:
        if params.key() not in seen:
            out.append(params)
            seen.add(params.key())
    return out


def explore_bayes(
    evaluate: EvaluateFn,
    encoder: Encoder,
    config: Config,
    target: float,
) -> list[TrialRecord]:
    """Gaussian-process Bayesian optimisation (feasibility × high CRF)."""
    n_explore = max(4, int(config.search.n_explore))
    n_init = n_init_points(config)
    trials = _initial_design(evaluate, encoder, config, n_init)
    rng = np.random.default_rng(config.search.seed + 17)
    space = encoder.space

    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    except ImportError:  # pragma: no cover - sklearn is a hard dependency
        log.warning("sklearn missing; bayes falls back to the initial design")
        return trials

    while len(trials) < n_explore:
        X = np.vstack([params_to_unit(space, t.params) for t in trials])
        y = np.array([t.vmaf for t in trials], dtype=np.float64)
        kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=0.5)
        gp = GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, n_restarts_optimizer=2, random_state=0,
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp.fit(X, y)
        except Exception as exc:  # noqa: BLE001
            log.debug("GP fit failed (%s); stopping bayes early", exc)
            break

        seen = _keys(trials)
        candidates = _candidate_pool(encoder, config, seen, rng)
        if not candidates:
            break

        Xu = np.vstack([params_to_unit(space, p) for p in candidates])
        mu, sigma = gp.predict(Xu, return_std=True)
        sigma = np.maximum(sigma, 1e-6)
        # P(VMAF >= target) under a Gaussian posterior, times normalised CRF (compression).
        z = (mu - target) / sigma
        # erf-based CDF without scipy.stats in the hot loop
        cdf = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
        crf_u = Xu[:, 0]
        explore = 0.15 * (sigma / (sigma.max() + 1e-6))
        acq = cdf * crf_u + explore
        pick = candidates[int(np.argmax(acq))]
        record = evaluate(pick)
        if record is None:
            break
        trials.append(record)

    return trials


def explore_tpe(
    evaluate: EvaluateFn,
    encoder: Encoder,
    config: Config,
    target: float,
) -> list[TrialRecord]:
    """Tree-structured Parzen Estimator: sample where good trials were denser than bad."""
    n_explore = max(4, int(config.search.n_explore))
    n_init = n_init_points(config)
    trials = _initial_design(evaluate, encoder, config, n_init)
    rng = np.random.default_rng(config.search.seed + 23)
    space = encoder.space
    gamma = 0.25

    while len(trials) < n_explore:
        if len(trials) < 4:
            break
        ranked = sorted(trials, key=lambda t: _fitness(t, target), reverse=True)
        n_good = max(2, int(math.ceil(gamma * len(ranked))))
        good = ranked[:n_good]
        bad = ranked[n_good:] or ranked[-2:]
        G = np.vstack([params_to_unit(space, t.params) for t in good])
        B = np.vstack([params_to_unit(space, t.params) for t in bad])
        g_std = np.maximum(G.std(axis=0), 0.08)
        b_std = np.maximum(B.std(axis=0), 0.08)
        g_mean = G.mean(axis=0)
        b_mean = B.mean(axis=0)

        seen = _keys(trials)
        candidates = _candidate_pool(encoder, config, seen, rng)
        if not candidates:
            break

        def _log_gauss(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
            z = (x - mean) / std
            return -0.5 * np.sum(z * z + 2.0 * np.log(std), axis=1)

        Xu = np.vstack([params_to_unit(space, p) for p in candidates])
        # l(x)/g(x) ≈ density_good / density_bad
        ratio = _log_gauss(Xu, g_mean, g_std) - _log_gauss(Xu, b_mean, b_std)
        pick = candidates[int(np.argmax(ratio))]
        record = evaluate(pick)
        if record is None:
            break
        trials.append(record)

    return trials


def explore_cmaes(
    evaluate: EvaluateFn,
    encoder: Encoder,
    config: Config,
    target: float,
) -> list[TrialRecord]:
    """Diagonal CMA-style evolution strategy on the unit cube."""
    n_explore = max(4, int(config.search.n_explore))
    n_init = min(4, n_init_points(config))
    trials = _initial_design(evaluate, encoder, config, n_init)
    rng = np.random.default_rng(config.search.seed + 31)
    space = encoder.space
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    if trials:
        best = max(trials, key=lambda t: _fitness(t, target))
        mean = params_to_unit(space, best.params)
    sigma = 0.22
    diag = np.ones(3, dtype=np.float64)
    lam = 4
    mu = 2

    while len(trials) < n_explore:
        batch: list[tuple[float, np.ndarray, TrialRecord]] = []
        attempts = 0
        while len(batch) < lam and attempts < lam * 4 and len(trials) + len(batch) < n_explore:
            attempts += 1
            z = rng.normal(size=3)
            unit = np.clip(mean + sigma * diag * z, 0.0, 0.999)
            params = params_from_unit(space, unit)
            if params.key() in _keys(trials) or any(
                params.key() == t.params.key() for _, _, t in batch
            ):
                continue
            record = evaluate(params)
            if record is None:
                continue
            batch.append((_fitness(record, target), unit, record))
        if not batch:
            break
        for _, _, record in batch:
            trials.append(record)
        batch.sort(key=lambda item: item[0], reverse=True)
        keep = batch[:mu]
        weights = np.array([math.log(mu + 0.5) - math.log(i + 1) for i in range(len(keep))])
        weights /= weights.sum()
        new_mean = sum(w * u for w, (_, u, _) in zip(weights, keep, strict=True))
        mean = 0.8 * np.asarray(new_mean) + 0.2 * mean
        spread = np.stack([u for _, u, _ in keep], axis=0)
        diag = 0.8 * diag + 0.2 * np.maximum(spread.std(axis=0), 0.05)
        sigma = max(0.04, min(0.4, 0.95 * sigma))

    return trials


def explore_adaptive(
    evaluate: EvaluateFn,
    encoder: Encoder,
    config: Config,
    strategy: str,
    target: float,
) -> list[TrialRecord]:
    if strategy == "bayes":
        return explore_bayes(evaluate, encoder, config, target)
    if strategy == "tpe":
        return explore_tpe(evaluate, encoder, config, target)
    if strategy == "cmaes":
        return explore_cmaes(evaluate, encoder, config, target)
    raise ValueError(f"not an adaptive strategy: {strategy!r}")
