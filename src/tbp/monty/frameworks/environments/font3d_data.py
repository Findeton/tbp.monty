# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""3D Font benchmark environment.

Renders characters (A-Z, 0-9) in multiple fonts as 3D extruded shapes using
distance-transform depth maps. Each character+font combination is a unique
object. Categories = character identity, instances = different fonts.

This creates genuine 3D geometry where curvature varies between characters
(serif vs sans-serif, thick vs thin strokes, curves vs angles) and the
matching pipeline's spatial+curvature features are fully exercised.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import quaternion as qt
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt, gaussian_filter

from tbp.monty.frameworks.actions.actions import Action
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.environments.environment import SimulatedEnvironment
from tbp.monty.frameworks.models.abstract_monty_classes import (
    AgentObservations,
    Observations,
    SensorObservation,
)
from tbp.monty.frameworks.models.motor_system_state import (
    AgentState,
    ProprioceptiveState,
    SensorState,
)
from tbp.monty.frameworks.sensors import SensorID

logger = logging.getLogger(__name__)

__all__ = ["Font3DEnvironment"]

# Default fonts to use (will be filtered to those that actually exist)
_DEFAULT_FONT_PATHS = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Times.ttc",
    "/System/Library/Fonts/Courier.ttc",
    "/System/Library/Fonts/Geneva.ttf",
    "/System/Library/Fonts/Avenir.ttc",
    "/System/Library/Fonts/Avenir Next.ttc",
    "/System/Library/Fonts/Avenir Next Condensed.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    # Linux fallbacks
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
]

_DEFAULT_CHARACTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")


def _find_available_fonts(font_paths=None, min_fonts=3):
    """Return list of font paths that exist on this system."""
    paths = font_paths or _DEFAULT_FONT_PATHS
    available = [p for p in paths if os.path.exists(p)]
    if len(available) < min_fonts:
        # Search system directories for more fonts
        for d in ["/System/Library/Fonts", "/Library/Fonts",
                  os.path.expanduser("~/Library/Fonts"),
                  "/usr/share/fonts"]:
            if os.path.exists(d):
                for f in sorted(os.listdir(d)):
                    if f.endswith((".ttf", ".otf")) and len(available) < 10:
                        p = os.path.join(d, f)
                        if p not in available:
                            available.append(p)
    return available


class Font3DEnvironment(SimulatedEnvironment):
    """Environment rendering 3D-extruded font characters.

    Args:
        characters: List of characters to use (default: A-Z + 0-9).
        font_paths: List of font file paths. Auto-detected if None.
        image_size: Size of rendered character images in pixels.
        font_size: Font size for rendering.
        patch_size: Size of observation patch.
        extrusion_scale: Height of 3D extrusion (0-1).
        smooth_sigma: Gaussian smoothing for continuous curvature.
        scan_step: Step size for spiral scan path (pixels per step).
    """

    def __init__(
        self,
        characters=None,
        font_paths=None,
        image_size=105,
        font_size=72,
        patch_size=10,
        extrusion_scale=0.3,
        smooth_sigma=1.0,
        scan_step=3,
    ):
        self.characters = characters or _DEFAULT_CHARACTERS
        self.font_paths = _find_available_fonts(font_paths)
        self.image_size = image_size
        self.font_size = font_size
        self.patch_size = patch_size
        self.extrusion_scale = extrusion_scale
        self.smooth_sigma = smooth_sigma
        self.scan_step = scan_step
        self.rotation = qt.from_rotation_vector([np.pi / 2, 0.0, 0.0])
        self.step_num = 0

        # Load fonts
        self._fonts = []
        for path in self.font_paths:
            try:
                font = ImageFont.truetype(path, self.font_size)
                self._fonts.append((path, font))
            except Exception:
                pass

        if not self._fonts:
            raise RuntimeError("No usable fonts found")

        self.font_names = [Path(p).stem for p, _ in self._fonts]
        self.n_fonts = len(self._fonts)
        self.n_characters = len(self.characters)

        logger.info(
            f"Font3D: {self.n_characters} chars x {self.n_fonts} fonts "
            f"= {self.n_characters * self.n_fonts} objects"
        )
        logger.info(f"Fonts: {self.font_names}")

        # State
        self.current_char_idx = 0
        self.current_font_idx = 0
        self._render_current()

        # Compatibility
        self._agents = [
            type("FakeAgent", (object,), {"action_space_type": "2d"})()
        ]

    def _render_current(self):
        """Render the current character in the current font."""
        char = self.characters[self.current_char_idx]
        _, font = self._fonts[self.current_font_idx]

        # Render character centered in image
        img = Image.new("L", (self.image_size, self.image_size), 255)
        draw = ImageDraw.Draw(img)

        # Get bounding box to center the character
        bbox = draw.textbbox((0, 0), char, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = (self.image_size - w) // 2 - bbox[0]
        y = (self.image_size - h) // 2 - bbox[1]
        draw.text((x, y), char, fill=0, font=font)

        self.current_image = np.array(img) > 128  # True = background

        # Compute 3D depth map
        stroke_mask = ~self.current_image
        if stroke_mask.any():
            dt = distance_transform_edt(stroke_mask.astype(float))
            max_dt = dt.max()
            if max_dt > 0:
                dt = dt / max_dt
            if self.smooth_sigma > 0:
                dt = gaussian_filter(dt, sigma=self.smooth_sigma)
                dt_max = dt.max()
                if dt_max > 0:
                    dt = dt / dt_max
            self._full_depth = np.where(stroke_mask, 1.0 - self.extrusion_scale * dt, 1.2)
        else:
            self._full_depth = np.full_like(self.current_image, 1.2, dtype=float)

        # Generate scan path (spiral from center)
        self.locations = self._generate_scan_path(stroke_mask)
        self.max_steps = max(len(self.locations) - 1, 1)
        self.step_num = 0

    def _generate_scan_path(self, stroke_mask):
        """Generate a scan path that visits stroke pixels."""
        # Find stroke pixel coordinates
        ys, xs = np.where(stroke_mask)
        if len(xs) == 0:
            return np.array([[self.image_size // 2, self.image_size // 2]])

        # Sort by angle from centroid (spiral-like traversal)
        cx, cy = xs.mean(), ys.mean()
        angles = np.arctan2(ys - cy, xs - cx)
        # Add distance for spiral effect
        dists = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        order = np.lexsort((dists, angles))

        # Subsample by step size
        coords = np.column_stack([ys[order], xs[order]])
        step = max(self.scan_step, 1)
        coords = coords[::step]

        # Filter to valid patch positions
        ps = self.patch_size
        valid = (
            (coords[:, 0] > ps)
            & (coords[:, 1] > ps)
            & (coords[:, 0] < self.image_size - ps)
            & (coords[:, 1] < self.image_size - ps)
        )
        coords = coords[valid]

        if len(coords) == 0:
            return np.array([[self.image_size // 2, self.image_size // 2]])
        return coords

    def switch_to_object(self, char_idx, font_idx, _version_idx=0):
        """Switch to a specific character+font combination."""
        self.current_char_idx = char_idx
        self.current_font_idx = font_idx
        self._render_current()

    def step(
        self, actions: Sequence[Action]
    ) -> tuple[Observations, ProprioceptiveState]:
        obs = self._observations()
        for action in actions:
            amount = 1
            if hasattr(action, "rotation_degrees"):
                amount = max(action.rotation_degrees, 1)
            self.step_num += int(amount)
            obs = self._observations()
        return obs, self.get_state()

    def _get_patch(self, array, loc):
        ps = self.patch_size
        loc = np.array(loc, dtype=int)
        return array[
            loc[0] - ps // 2 : loc[0] + ps // 2,
            loc[1] - ps // 2 : loc[1] + ps // 2,
        ]

    def _observations(self) -> Observations:
        loc = self.locations[self.step_num % (self.max_steps + 1)]
        patch = self._get_patch(self.current_image, loc)
        depth_patch = self._get_patch(self._full_depth, loc).astype(float)

        return Observations(
            {
                AgentID("agent_id_0"): AgentObservations(
                    {
                        SensorID("patch"): SensorObservation(
                            {
                                "depth": depth_patch,
                                "semantic": np.array(~patch, dtype=int),
                                "rgba": np.stack(
                                    [depth_patch, depth_patch, depth_patch], axis=2
                                ),
                            }
                        ),
                        SensorID("view_finder"): SensorObservation(
                            {
                                "depth": self._full_depth,
                                "semantic": np.array(
                                    ~self.current_image, dtype=int
                                ),
                            }
                        ),
                    }
                )
            }
        )

    def get_state(self) -> ProprioceptiveState:
        loc = self.locations[self.step_num % (self.max_steps + 1)]
        sensor_position = np.array([loc[0], loc[1], 0])
        return ProprioceptiveState(
            {
                AgentID("agent_id_0"): AgentState(
                    sensors={
                        SensorID("patch"): SensorState(
                            rotation=self.rotation,
                            position=sensor_position,
                        ),
                        SensorID("view_finder"): SensorState(
                            rotation=self.rotation,
                            position=sensor_position,
                        ),
                    },
                    rotation=self.rotation,
                    position=np.array([0, 0, 0]),
                )
            }
        )

    def reset(self) -> tuple[Observations, ProprioceptiveState]:
        self.step_num = 0
        return self._observations(), self.get_state()

    def close(self) -> None:
        pass
