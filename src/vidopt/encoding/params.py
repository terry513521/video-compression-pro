"""Encoder parameters and the space they are searched over.

The task names three parameters: ``crf``, ``aq-mode`` and ``aq-strength``. Those are the
*logical* knobs; each encoder translates them into its own flags (see ``encoders.py``).

This is deliberately NOT a cross-codec transfer function. The reference predicted a CQ
for ``av1_nvenc`` and converted it to five other codecs through hand-written anchor
tables whose own comments admitted they were untuned guesses. Here the search and the
model are run per encoder, so ``crf`` always means what that encoder means by it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace


@dataclass(frozen=True, order=True)
class EncodeParams:
    """One point in the search space."""

    crf: float
    aq_mode: int
    aq_strength: float

    def rounded(
        self, *, crf_decimals: int = 1, strength_decimals: int = 2
    ) -> EncodeParams:
        """Snap to the resolution the encoders actually honour.

        Keeps the cache key canonical: crf=28.0000001 and crf=28.0 must be one entry.
        """
        return replace(
            self,
            crf=round(float(self.crf), crf_decimals),
            aq_mode=int(self.aq_mode),
            aq_strength=round(float(self.aq_strength), strength_decimals),
        )

    def key(self) -> str:
        """Canonical string for cache keys and logs."""
        p = self.rounded()
        return f"crf={p.crf:g},aq_mode={p.aq_mode:d},aq_strength={p.aq_strength:g}"

    def to_dict(self) -> dict[str, float | int]:
        p = self.rounded()
        return {"crf": p.crf, "aq_mode": p.aq_mode, "aq_strength": p.aq_strength}

    def __str__(self) -> str:
        return self.key()


@dataclass(frozen=True)
class ParamSpace:
    """Bounds of the searchable space for one encoder."""

    crf_min: float
    crf_max: float
    aq_modes: tuple[int, ...]
    aq_strength_min: float
    aq_strength_max: float
    aq_strength_is_integer: bool = False
    """NVENC's aq-strength is an integer 1-15; x264/x265 take a float."""

    def clamp(self, params: EncodeParams) -> EncodeParams:
        """Project an arbitrary point into the space. Used after model prediction,
        where a regressor can extrapolate slightly outside the trained range."""
        crf = min(max(float(params.crf), self.crf_min), self.crf_max)

        mode = int(round(params.aq_mode))
        if mode not in self.aq_modes:
            mode = min(self.aq_modes, key=lambda m: abs(m - mode))

        strength = min(
            max(float(params.aq_strength), self.aq_strength_min), self.aq_strength_max
        )
        if self.aq_strength_is_integer:
            strength = float(int(round(strength)))

        return EncodeParams(crf=crf, aq_mode=mode, aq_strength=strength).rounded()

    def midpoint(self) -> EncodeParams:
        """A sane starting guess when there is no model and no history."""
        return self.clamp(
            EncodeParams(
                crf=(self.crf_min + self.crf_max) / 2.0,
                aq_mode=self.aq_modes[len(self.aq_modes) // 2],
                aq_strength=(self.aq_strength_min + self.aq_strength_max) / 2.0,
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "crf_min": self.crf_min,
            "crf_max": self.crf_max,
            "aq_modes": list(self.aq_modes),
            "aq_strength_min": self.aq_strength_min,
            "aq_strength_max": self.aq_strength_max,
            "aq_strength_is_integer": self.aq_strength_is_integer,
        }

    def dumps(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)
