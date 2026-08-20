"""vidopt — scene-adaptive video compression with learned encoder parameters.

Two workflows:

* **train mode** (``vidopt train``): segment a corpus by scene, extract per-segment
  features, search for the encoder parameters that minimise size subject to a VMAF
  target, and train a model per target.
* **production mode** (``vidopt compress``): segment an input, predict parameters per
  segment, encode in parallel and concatenate losslessly.

See DESIGN.md for the architecture and REFERENCE_ANALYSIS.md for what was inherited
from the reference implementations and what was redesigned.
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
