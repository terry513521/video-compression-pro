"""Samplers for stage-A exploration of the parameter space.

Sobol is the default. At the budgets that matter here — tens of trials, since each one
costs an encode plus a VMAF run — a scrambled Sobol sequence covers the space far more
evenly than uniform random sampling, which clumps and leaves gaps. ``random`` and
``grid`` exist behind the same interface so the choice can be measured rather than
asserted.
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


def _from_unit_cube(space: ParamSpace, points: np.ndarray) -> list[EncodeParams]:
    """Map points in [0,1)^3 onto the parameter space."""
    out: list[EncodeParams] = []
    n_modes = len(space.aq_modes)
    for crf_u, mode_u, strength_u in points:
        crf = space.crf_min + crf_u * (space.crf_max - space.crf_min)
        mode = space.aq_modes[min(n_modes - 1, int(mode_u * n_modes))]
        strength = space.aq_strength_min + strength_u * (
            space.aq_strength_max - space.aq_strength_min
        )
        out.append(
            space.clamp(EncodeParams(crf=crf, aq_mode=mode, aq_strength=strength))
        )
    return out


def sobol(space: ParamSpace, n: int, seed: int) -> list[EncodeParams]:
    """Scrambled Sobol sequence over (crf, aq_mode, aq_strength)."""
    engine = qmc.Sobol(d=3, scramble=True, seed=seed)
    # Sobol's balance properties hold for powers of two; draw the next power of two up
    # and truncate rather than requesting an arbitrary count.
    n_pow2 = 1 << max(2, math.ceil(math.log2(max(n, 2))))
    points = engine.random(n_pow2)[:n]
    return _dedupe(_from_unit_cube(space, points))


def random_search(space: ParamSpace, n: int, seed: int) -> list[EncodeParams]:
    """Uniform random sampling. Baseline for comparison against Sobol."""
    rng = np.random.default_rng(seed)
    return _dedupe(_from_unit_cube(space, rng.random((n, 3))))


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
    "random": random_search,
    "grid": grid,
}


def get_sampler(name: str) -> Sampler:
    try:
        return SAMPLERS[name]
    except KeyError:
        raise ValueError(
            f"unknown sampler {name!r}; available: {sorted(SAMPLERS)}"
        ) from None
