"""The objective function.

Reused from the reference implementation
(``vidaio-subnet/services/scoring/scoring_function.py``) with identical mathematics and
constants — this is the specification of "good" and changing it would change the problem.
Re-implemented here for clarity, typing, and to remove the plotting code that shared the
module.

Shape of the function, for a VMAF target ``T``:

* ``rate >= 0.80``          -> 0.0   (less than 1.25x compression: not a real result)
* ``vmaf < T - 5``          -> 0.0   (hard quality floor)
* ``T - 5 <= vmaf < T``     -> soft recovery zone, quadratic in position
* ``vmaf >= T``             -> 0.70 * compression + 0.30 * quality

Because compression carries 70% of the weight and quality saturates at the threshold,
the optimum sits *just above* ``T``. That is why the search is posed as
"minimise size subject to vmaf >= T" and why the CRF model is fit with an asymmetric
loss — see DESIGN.md sections 2 and 7.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

COMPRESSION_WEIGHT = 0.70
QUALITY_WEIGHT = 0.30
SOFT_THRESHOLD_MARGIN = 5.0
MIN_USEFUL_RATE = 0.80
NORMALIZATION = 1.12


@dataclass(frozen=True)
class ScoreBreakdown:
    """A score plus the parts it was built from, for reporting and debugging."""

    score: float
    compression_component: float
    quality_component: float
    reason: str

    @property
    def is_zero(self) -> bool:
        return self.score <= 0.0


def compression_ratio(rate: float) -> float:
    """Size ratio -> compression factor. ``rate`` is out_bytes / ref_bytes."""
    return (1.0 / rate) if rate > 0 else 0.0


def _compression_component_above(ratio: float) -> float:
    """Compression term used at or above the threshold."""
    if ratio <= 20.0:
        return ((ratio - 1.25) / 18.75) ** 0.9
    return 1.0 + 0.1 * math.log(ratio / 20.0)


def _compression_component_soft(ratio: float) -> float:
    """Compression term used inside the soft zone (steeper, and starts from 1x)."""
    if ratio <= 20.0:
        return ((ratio - 1.0) / 19.0) ** 1.5
    return 1.0 + 0.3 * math.log(ratio / 20.0)


def compression_score(
    vmaf: float,
    rate: float,
    target: float,
    *,
    compression_weight: float = COMPRESSION_WEIGHT,
    quality_weight: float = QUALITY_WEIGHT,
    soft_margin: float = SOFT_THRESHOLD_MARGIN,
) -> ScoreBreakdown:
    """Score an encode.

    Args:
        vmaf: Measured VMAF (0-100).
        rate: ``out_bytes / ref_bytes``. Lower is better.
        target: Required VMAF threshold, e.g. 85, 89 or 93.

    Returns:
        A :class:`ScoreBreakdown` whose ``score`` lies in [0, 1].
    """
    if abs(compression_weight + quality_weight - 1.0) > 1e-9:
        raise ValueError(
            f"weights must sum to 1.0, got {compression_weight + quality_weight}"
        )

    hard_cutoff = target - soft_margin

    # Case 0: no meaningful compression. Blocks "return the input" and near-copies.
    if rate >= MIN_USEFUL_RATE:
        ratio = compression_ratio(rate)
        return ScoreBreakdown(
            0.0, 0.0, 0.0,
            f"no meaningful compression (ratio {ratio:.2f}x, rate {rate:.3f}); "
            f"at least 1.25x required",
        )

    # Case 1: below the hard quality floor.
    if vmaf < hard_cutoff:
        return ScoreBreakdown(
            0.0, 0.0, 0.0, f"VMAF {vmaf:.2f} below hard cutoff {hard_cutoff:.2f}"
        )

    ratio = compression_ratio(rate)

    # Case 2: soft zone. Quality factor rises quadratically to 0.7 at the threshold,
    # which is exactly where case 3's quality component starts — the function is
    # continuous across the boundary.
    if vmaf < target:
        position = (vmaf - hard_cutoff) / soft_margin
        quality_factor = 0.7 * (position**2)
        component = _compression_component_soft(ratio)
        score = min(1.0, (component * quality_factor) / NORMALIZATION)
        return ScoreBreakdown(
            score, component, quality_factor,
            f"VMAF {vmaf:.2f} in soft zone (quality factor {quality_factor:.2f})",
        )

    # Case 3: at or above the threshold.
    max_excess = 100.0 - target
    excess = vmaf - target
    quality = 0.7 + 0.3 * (min(1.0, excess / max_excess) if max_excess > 0 else 1.0)
    component = _compression_component_above(ratio)
    score = min(
        1.0, (compression_weight * component + quality_weight * quality) / NORMALIZATION
    )
    suffix = "" if ratio < 10.0 else " (excellent compression)"
    return ScoreBreakdown(score, component, quality, f"success{suffix}")


def is_feasible(vmaf: float, rate: float, target: float) -> bool:
    """True when an encode meets the target *and* actually compressed something.

    This is the search constraint. Note it is strictly stronger than a non-zero score:
    a soft-zone result scores above zero but does not meet the target.
    """
    return vmaf >= target and rate < MIN_USEFUL_RATE
