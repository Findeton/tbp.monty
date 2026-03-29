# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Structured metrics for Panda3D object recognition evaluation.

Captures per-episode and aggregate results from :class:`Panda3DEvalHarness`
runs. All results are plain dataclasses serializable to JSON/dict for
downstream analysis.

Usage::

    from tbp.monty.simulators.panda3d.metrics import EvalEpisodeResult, EvalRunResult

    episode = EvalEpisodeResult(
        object_name="011_banana",
        rotation=(0.0, 45.0, 0.0),
        detected_object="011_banana",
        correct=True,
        steps_to_converge=12,
        rotation_error_deg=3.2,
        max_evidence=8.5,
        wall_clock_seconds=1.3,
    )
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvalEpisodeResult:
    """Result of a single evaluation episode (one object at one rotation).

    Attributes
    ----------
    object_name : str
        Ground-truth YCB object name (e.g. ``"011_banana"``).
    rotation : tuple[float, float, float]
        Applied Euler rotation in degrees (rx, ry, rz).
    detected_object : str or None
        Object ID reported by the LM, or None if no convergence.
    correct : bool
        Whether ``detected_object`` matches ``object_name``.
    steps_to_converge : int or None
        Number of observation steps before the LM reached terminal state,
        or None if it did not converge within the episode.
    rotation_error_deg : float or None
        Angular error between detected and ground-truth rotation in degrees.
    max_evidence : float
        Peak evidence value across all hypotheses at episode end.
    wall_clock_seconds : float
        Wall-clock duration of this episode.
    """

    object_name: str
    rotation: tuple[float, float, float]
    detected_object: str | None
    correct: bool
    steps_to_converge: int | None
    rotation_error_deg: float | None
    max_evidence: float
    wall_clock_seconds: float
    mean_settling_iterations: float | None = None


@dataclass
class EvalRunResult:
    """Aggregate result of a full evaluation run across multiple episodes.

    Attributes
    ----------
    episodes : list[EvalEpisodeResult]
        Per-episode results.
    accuracy : float
        Fraction of episodes where ``correct`` is True.
    mean_steps_to_converge : float or None
        Mean steps across converged episodes. None if no episodes converged.
    mean_rotation_error_deg : float or None
        Mean rotation error across correct episodes. None if none correct.
    wall_clock_total : float
        Total wall-clock time for the entire run.
    config : dict
        All configuration parameters for reproducibility.
    """

    episodes: list[EvalEpisodeResult]
    accuracy: float
    mean_steps_to_converge: float | None
    mean_rotation_error_deg: float | None
    wall_clock_total: float
    config: dict = field(default_factory=dict)

    @staticmethod
    def from_episodes(
        episodes: list[EvalEpisodeResult],
        wall_clock_total: float,
        config: dict | None = None,
    ) -> "EvalRunResult":
        """Compute aggregate metrics from a list of episode results."""
        n = len(episodes)
        n_correct = sum(1 for e in episodes if e.correct)
        accuracy = n_correct / n if n > 0 else 0.0

        converged = [e.steps_to_converge for e in episodes if e.steps_to_converge is not None]
        mean_steps = sum(converged) / len(converged) if converged else None

        rot_errors = [e.rotation_error_deg for e in episodes if e.rotation_error_deg is not None and e.correct]
        mean_rot = sum(rot_errors) / len(rot_errors) if rot_errors else None

        return EvalRunResult(
            episodes=episodes,
            accuracy=accuracy,
            mean_steps_to_converge=mean_steps,
            mean_rotation_error_deg=mean_rot,
            wall_clock_total=wall_clock_total,
            config=config or {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return asdict(self)

    def save(self, path: str | Path) -> None:
        """Save results to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            f"Accuracy: {self.accuracy:.1%} ({sum(e.correct for e in self.episodes)}/{len(self.episodes)})",
        ]
        if self.mean_steps_to_converge is not None:
            lines.append(f"Mean steps to converge: {self.mean_steps_to_converge:.1f}")
        if self.mean_rotation_error_deg is not None:
            lines.append(f"Mean rotation error: {self.mean_rotation_error_deg:.1f} deg")
        lines.append(f"Total wall clock: {self.wall_clock_total:.1f}s")
        return "\n".join(lines)
