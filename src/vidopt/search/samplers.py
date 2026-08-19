"""Samplers for stage-A exploration of the parameter space.

At the budgets that matter here — tens of trials, since each one costs an encode plus a
VMAF run — low-discrepancy sequences (Sobol, Halton, Latin hypercube) cover the cube
more evenly than uniform random sampling. ``grid`` is deterministic. All share the same
``(space, n, seed) -> list[EncodeParams]`` interface.
"""

from __future__ import annotations

import itertools
import math
from typing import Protocol

import numpy as np
from scipy.stats import qmc

from ..encoding.params import EncodeParams, ParamSpace


class Sampler(Protocol):
    def __call__(self, space: ParamSpace, n: int, seed: int) -> list[EncodeParams]: ...


def params_from_unit(space: ParamSpace, point: np.ndarray) -> EncodeParams:
    """Map one point in [0, 1)^3 onto (crf, aq_mode, aq_strength)."""
    crf_u, mode_u, strength_u = (float(x) for x in np.asarray(point, dtype=np.float64)[:3])
    crf_u = min(max(crf_u, 0.0), 0.999999)
    mode_u = min(max(mode_u, 0.0), 0.999999)
    strength_u = min(max(strength_u, 0.0), 0.999999)
    n_modes = len(space.aq_modes)
    crf = space.crf_min + crf_u * (space.crf_max - space.crf_min)
    mode = space.aq_modes[min(n_modes - 1, int(mode_u * n_modes))]
    strength = space.aq_strength_min + strength_u * (
        space.aq_strength_max - space.aq_strength_min
    )
    return space.clamp(EncodeParams(crf=crf, aq_mode=mode, aq_strength=strength))


def params_to_unit(space: ParamSpace, params: EncodeParams) -> np.ndarray:
    """Inverse of :func:`params_from_unit` (aq_mode mapped to bin centre)."""
    p = space.clamp(params)
    crf_span = max(space.crf_max - space.crf_min, 1e-9)
    str_span = max(space.aq_strength_max - space.aq_strength_min, 1e-9)
    n_modes = len(space.aq_modes)
    try:
        mi = space.aq_modes.index(p.aq_mode)
    except ValueError:
        mi = min(range(n_modes), key=lambda i: abs(space.aq_modes[i] - p.aq_mode))
    return np.array([
        (p.crf - space.crf_min) / crf_span,
        (mi + 0.5) / n_modes,
        (p.aq_strength - space.aq_strength_min) / str_span,
    ], dtype=np.float64)


def _from_unit_cube(space: ParamSpace, points: np.ndarray) -> list[EncodeParams]:
    return [params_from_unit(space, row) for row in np.asarray(points, dtype=np.float64)]


def sobol(space: ParamSpace, n: int, seed: int) -> list[EncodeParams]:
    """Scrambled Sobol sequence over (crf, aq_mode, aq_strength)."""
    engine = qmc.Sobol(d=3, scramble=True, seed=seed)
    n_pow2 = 1 << max(2, math.ceil(math.log2(max(n, 2))))
    points = engine.random(n_pow2)[:n]
    return _dedupe(_from_unit_cube(space, points))


def latin_hypercube(space: ParamSpace, n: int, seed: int) -> list[EncodeParams]:
    """Latin hypercube: one sample per stratum on each axis, then scrambled."""
    engine = qmc.LatinHypercube(d=3, seed=seed)
    return _dedupe(_from_unit_cube(space, engine.random(max(n, 1))))


def halton(space: ParamSpace, n: int, seed: int) -> list[EncodeParams]:
    """Scrambled Halton sequence (another low-discrepancy construction)."""
    engine = qmc.Halton(d=3, scramble=True, seed=seed)
    return _dedupe(_from_unit_cube(space, engine.random(max(n, 1))))


def random_search(space: ParamSpace, n: int, seed: int) -> list[EncodeParams]:
    """Uniform random sampling. Baseline for comparison against QMC methods."""
    rng = np.random.default_rng(seed)
    return _dedupe(_from_unit_cube(space, rng.random((max(n, 1), 3))))


def grid(space: ParamSpace, n: int, seed: int) -> list[EncodeParams]:
    """Regular grid, sized so the product is close to ``n``.

    ``seed`` is accepted for interface compatibility and deliberately unused: a grid is
    deterministic.
    """
    del seed
    n_modes = len(space.aq_modes)
    per_axis = max(2, int(round((max(n, 4) / n_modes) ** 0.5)))
    crfs = np.linspace(space.crf_min, space.crf_max, per_axis)
    strengths = np.linspace(space.aq_strength_min, space.aq_strength_max, per_axis)
    combos = itertools.product(crfs, space.aq_modes, strengths)
    return _dedupe(
        [
            space.clamp(EncodeParams(crf=float(c), aq_mode=int(m), aq_strength=float(s)))
            for c, m, s in combos
        ]
    )[:n]


def _dedupe(items: list[EncodeParams]) -> list[EncodeParams]:
    """Drop duplicates created by clamping/quantisation, preserving order."""
    seen: set[str] = set()
    out: list[EncodeParams] = []
    for item in items:
        key = item.key()
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


SAMPLERS: dict[str, Sampler] = {
    "sobol": sobol,
    "lhs": latin_hypercube,
    "halton": halton,
    "random": random_search,
    "grid": grid,
}


def get_sampler(name: str) -> Sampler:
    key = str(name).strip().lower()
    try:
        return SAMPLERS[key]
    except KeyError:
        raise ValueError(
            f"unknown sampler {name!r}; available: {sorted(SAMPLERS)}"
        ) from None
