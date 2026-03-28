# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt


@dataclass
class Hypotheses:
    """Set of hypotheses consisting of evidence, locations, poses, and scales.

    The arrays are expected to have the same first dimension. Each index
    corresponds to a hypothesis. The scales array stores per-hypothesis scale
    factors (analogous to grid cell modules at different spacings). A scale
    of 1.0 means the query object is the same size as the stored graph.
    Scale > 1 means the query is larger; < 1 means smaller.
    """

    evidence: npt.NDArray[np.float64]
    locations: npt.NDArray[np.float64]
    poses: npt.NDArray[np.float64]
    possible: npt.NDArray[np.bool_]
    scales: Optional[npt.NDArray[np.float64]] = None
    states: Optional[npt.NDArray[np.int64]] = None


@dataclass
class ChannelHypotheses(Hypotheses):
    """A set of hypotheses for a single input channel."""

    input_channel: str = ""
