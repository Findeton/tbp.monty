# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Debug output for Panda3D temporal training.

Dumps rendered frames, depth maps, State features, and surprise curves
to an output directory so you can visually sanity-check what Monty is
seeing during sensorimotor training on animated 3D objects.

Output structure::

    output_dir/
        frames/
            frame_0000_rgba.png
            frame_0000_depth.png
            ...
        states/
            state_0000.json
            ...
        surprise_curve.png
        video.mp4          (if make_video() called)
        summary.json
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class TemporalTrainingDebugger:
    """Saves per-step debug output during temporal training.

    Parameters
    ----------
    output_dir : str or Path
        Directory to write output files.
    """

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.frames_dir = self.output_dir / "frames"
        self.states_dir = self.output_dir / "states"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.states_dir.mkdir(parents=True, exist_ok=True)

        self._surprise_history: List[float] = []
        self._step_metadata: List[Dict[str, Any]] = []

    def save_frame(
        self,
        step: int,
        rgba: np.ndarray,
        depth: np.ndarray,
        state=None,
        surprise: Optional[float] = None,
        action_name: Optional[str] = None,
    ) -> None:
        """Save RGBA frame, depth visualization, and state features.

        Parameters
        ----------
        step : int
            Current step number.
        rgba : np.ndarray
            RGBA image (H, W, 4) uint8.
        depth : np.ndarray
            Depth array (H, W) or (H, W, 1) float32.
        state : State or None
            The State object produced by CameraSM.
        surprise : float or None
            Surprise value from TemporalMemory.
        action_name : str or None
            Name of the action taken this step.
        """
        # Save RGBA as PNG
        rgba_path = self.frames_dir / f"frame_{step:04d}_rgba.png"
        _save_png(rgba_path, rgba)

        # Save depth as grayscale PNG (normalized for visualization)
        depth_2d = depth.squeeze() if depth.ndim == 3 else depth
        depth_path = self.frames_dir / f"frame_{step:04d}_depth.png"
        _save_depth_png(depth_path, depth_2d)

        # Save state features as JSON
        if state is not None:
            state_dict = _state_to_dict(state)
        else:
            state_dict = {}

        meta = {
            "step": step,
            "surprise": surprise,
            "action": action_name,
            "use_state": getattr(state, "use_state", None),
        }
        meta.update(state_dict)

        state_path = self.states_dir / f"state_{step:04d}.json"
        with open(state_path, "w") as f:
            json.dump(meta, f, indent=2, default=_json_default)

        if surprise is not None:
            self._surprise_history.append(surprise)
        self._step_metadata.append(meta)

    def save_surprise_curve(self) -> Path:
        """Plot and save the surprise curve over training steps.

        Returns the path to the saved plot.
        """
        plot_path = self.output_dir / "surprise_curve.png"

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(self._surprise_history, linewidth=1.5)
            ax.set_xlabel("Step")
            ax.set_ylabel("Surprise")
            ax.set_title("Temporal Memory Surprise Over Training")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            logger.info("Surprise curve saved to %s", plot_path)
        except ImportError:
            # Fallback: save raw data as JSON
            plot_path = self.output_dir / "surprise_curve.json"
            with open(plot_path, "w") as f:
                json.dump(self._surprise_history, f)
            logger.info(
                "matplotlib not available; surprise data saved to %s", plot_path
            )

        return plot_path

    def save_summary(self, extra: Optional[Dict[str, Any]] = None) -> Path:
        """Save a summary JSON of the training run."""
        summary_path = self.output_dir / "summary.json"
        surprises = self._surprise_history

        summary = {
            "total_steps": len(self._step_metadata),
            "total_surprise_entries": len(surprises),
            "mean_surprise": float(np.mean(surprises)) if surprises else None,
            "final_surprise": surprises[-1] if surprises else None,
            "min_surprise": float(np.min(surprises)) if surprises else None,
            "steps_with_use_state": sum(
                1 for m in self._step_metadata if m.get("use_state")
            ),
        }
        if extra:
            summary.update(extra)

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=_json_default)

        logger.info("Summary saved to %s", summary_path)
        return summary_path

    def make_video(self, fps: int = 10) -> Optional[Path]:
        """Compile RGBA frames into an MP4 video.

        Requires PIL (Pillow). Falls back gracefully if unavailable.

        Returns the path to the video, or None if creation failed.
        """
        video_path = self.output_dir / "video.mp4"

        # Collect frames in order
        frame_files = sorted(self.frames_dir.glob("frame_*_rgba.png"))
        if not frame_files:
            logger.warning("No frames to compile into video")
            return None

        try:
            from PIL import Image

            frames = [Image.open(f).convert("RGB") for f in frame_files]

            # Try imageio first (better MP4 support)
            try:
                import imageio.v3 as iio

                frame_arrays = [np.array(f) for f in frames]
                iio.imwrite(
                    str(video_path),
                    frame_arrays,
                    fps=fps,
                    codec="libx264",
                )
                logger.info("Video saved to %s (%d frames)", video_path, len(frames))
                return video_path
            except (ImportError, Exception):
                pass

            # Fallback: save as GIF
            gif_path = self.output_dir / "video.gif"
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=int(1000 / fps),
                loop=0,
            )
            logger.info("Video saved as GIF to %s (%d frames)", gif_path, len(frames))
            return gif_path

        except ImportError:
            logger.warning("Pillow not available; cannot create video")
            return None

    @property
    def surprise_history(self) -> List[float]:
        return list(self._surprise_history)


def _save_png(path: Path, rgba: np.ndarray) -> None:
    """Save an RGBA numpy array as a PNG file."""
    try:
        from PIL import Image

        img = Image.fromarray(rgba[:, :, :3] if rgba.shape[2] == 4 else rgba)
        img.save(path)
    except ImportError:
        # Minimal fallback using raw numpy save
        np.save(str(path).replace(".png", ".npy"), rgba)


def _save_depth_png(path: Path, depth_2d: np.ndarray) -> None:
    """Save a depth array as a grayscale PNG (normalized to [0, 255])."""
    valid = depth_2d[depth_2d < depth_2d.max() * 0.99]
    if len(valid) == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = float(valid.min()), float(valid.max())

    if vmax - vmin < 1e-6:
        vmax = vmin + 1.0

    normalized = np.clip((depth_2d - vmin) / (vmax - vmin), 0, 1)
    gray = (normalized * 255).astype(np.uint8)

    try:
        from PIL import Image

        img = Image.fromarray(gray, mode="L")
        img.save(path)
    except ImportError:
        np.save(str(path).replace(".png", ".npy"), depth_2d)


def _state_to_dict(state) -> Dict[str, Any]:
    """Convert a State object to a JSON-serializable dict."""
    d = {}
    if hasattr(state, "location") and state.location is not None:
        d["location"] = state.location.tolist() if hasattr(
            state.location, "tolist"
        ) else list(state.location)
    if hasattr(state, "morphological_features") and state.morphological_features:
        morph = {}
        for k, v in state.morphological_features.items():
            if hasattr(v, "tolist"):
                morph[k] = v.tolist()
            else:
                morph[k] = v
        d["morphological_features"] = morph
    if (
        hasattr(state, "non_morphological_features")
        and state.non_morphological_features
    ):
        non_morph = {}
        for k, v in state.non_morphological_features.items():
            if hasattr(v, "tolist"):
                non_morph[k] = v.tolist()
            else:
                non_morph[k] = v
        d["non_morphological_features"] = non_morph
    d["confidence"] = getattr(state, "confidence", None)
    d["use_state"] = getattr(state, "use_state", None)
    return d


def _json_default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return str(obj)
