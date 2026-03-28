# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Animation support for the Panda3D simulator.

Provides AnimatedObject for loading glTF/GLB models with skeletal
animations and advancing them frame-by-frame. Two animation modes:

1. **Keyframe**: Play animations embedded in glTF files (skeletal
   deformation from artist-authored keyframes).
2. **Programmatic**: Directly control joint transforms via
   ``controlJoint()`` for procedural animation.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from panda3d.core import NodePath

logger = logging.getLogger(__name__)


class AnimatedObject:
    """A 3D model with skeletal animation support.

    Wraps a Panda3D Actor loaded from a glTF/GLB file. Supports posing
    at specific animation frames or programmatically controlling joints.

    Parameters
    ----------
    model_path : str or Path
        Path to a .gltf or .glb file containing the model.
    base : ShowBase
        The Panda3D ShowBase instance (needed for Actor).
    skip_animations : bool
        If True, load geometry only (no animation data).
    """

    def __init__(
        self,
        model_path: str | Path,
        base,
        skip_animations: bool = False,
    ):
        from direct.actor.Actor import Actor

        try:
            from gltf import GltfSettings, load_model
        except ImportError:
            raise ValueError(
                "panda3d-gltf is required for animated models. "
                "Install with: pip install panda3d-gltf"
            )

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # panda3d-gltf loads to a PandaNode; Actor needs a BAM file.
        # Convert: glTF → PandaNode → BAM → Actor
        settings = GltfSettings(
            no_srgb=True,
            skip_animations=skip_animations,
        )
        model_root = load_model(str(model_path), gltf_settings=settings)
        model_np = NodePath(model_root)

        # Write to a temporary BAM file for Actor to load
        self._bam_dir = tempfile.mkdtemp(prefix="monty_anim_")
        self._bam_path = os.path.join(self._bam_dir, model_path.stem + ".bam")
        model_np.writeBamFile(self._bam_path)

        self._actor = Actor(self._bam_path)
        self._anim_names = list(self._actor.getAnimNames())
        self._controlled_joints: Dict[str, NodePath] = {}
        self._current_anim: Optional[str] = None
        self._current_frame: int = 0

        logger.info(
            "Loaded animated model %s: %d animations %s",
            model_path.name,
            len(self._anim_names),
            self._anim_names,
        )

    @property
    def node_path(self) -> NodePath:
        """The Actor's NodePath (for reparenting into the scene)."""
        return self._actor

    @property
    def animation_names(self) -> List[str]:
        """Names of available animations."""
        return list(self._anim_names)

    def get_num_frames(self, anim_name: Optional[str] = None) -> int:
        """Return the number of frames in the given animation.

        Parameters
        ----------
        anim_name : str or None
            Animation name. If None, uses the first animation.
        """
        anim_name = anim_name or self._default_anim()
        ctrl = self._actor.getAnimControl(anim_name)
        if ctrl is None:
            raise ValueError(f"Animation '{anim_name}' not found")
        return ctrl.getNumFrames()

    def get_frame_rate(self, anim_name: Optional[str] = None) -> float:
        """Return the frame rate of the given animation."""
        anim_name = anim_name or self._default_anim()
        ctrl = self._actor.getAnimControl(anim_name)
        if ctrl is None:
            raise ValueError(f"Animation '{anim_name}' not found")
        return ctrl.getFrameRate()

    def pose(self, frame: int, anim_name: Optional[str] = None) -> None:
        """Set the model to a specific animation frame.

        Parameters
        ----------
        frame : int
            Frame number (0-indexed).
        anim_name : str or None
            Animation name. If None, uses the first animation.
        """
        anim_name = anim_name or self._default_anim()
        self._actor.pose(anim_name, frame)
        self._current_anim = anim_name
        self._current_frame = frame

    def advance(self, n_frames: int = 1, anim_name: Optional[str] = None) -> int:
        """Advance the animation by n frames, wrapping at the end.

        Returns the new frame number.
        """
        anim_name = anim_name or self._default_anim()
        total = self.get_num_frames(anim_name)
        new_frame = (self._current_frame + n_frames) % total
        self.pose(new_frame, anim_name)
        return new_frame

    def control_joint(
        self, joint_name: str, part_name: str = "modelRoot"
    ) -> NodePath:
        """Get a NodePath that directly controls a joint's transform.

        Use this for programmatic (procedural) animation. Setting the
        position, rotation, or scale on the returned NodePath will
        override the joint's transform.

        Parameters
        ----------
        joint_name : str
            Name of the joint/bone to control.
        part_name : str
            Part name (default "modelRoot").

        Returns
        -------
        NodePath for the controlled joint.
        """
        if joint_name not in self._controlled_joints:
            ctrl_np = self._actor.controlJoint(None, part_name, joint_name)
            if ctrl_np is None:
                raise ValueError(f"Joint '{joint_name}' not found")
            self._controlled_joints[joint_name] = ctrl_np
        return self._controlled_joints[joint_name]

    def release_joint(
        self, joint_name: str, part_name: str = "modelRoot"
    ) -> None:
        """Release a previously controlled joint back to animation control."""
        if joint_name in self._controlled_joints:
            self._actor.releaseJoint(part_name, joint_name)
            del self._controlled_joints[joint_name]

    def get_joint_names(self, part_name: str = "modelRoot") -> List[str]:
        """Return names of all joints in the model."""
        joints = self._actor.getJoints(partName=part_name)
        return [j.getName() for j in joints]

    @property
    def current_frame(self) -> int:
        return self._current_frame

    @property
    def current_anim(self) -> Optional[str]:
        return self._current_anim

    def cleanup(self) -> None:
        """Release resources."""
        self._controlled_joints.clear()
        if self._actor is not None:
            self._actor.cleanup()
            self._actor = None
        # Clean up temp BAM file
        if self._bam_path and os.path.exists(self._bam_path):
            os.unlink(self._bam_path)
        if self._bam_dir and os.path.exists(self._bam_dir):
            try:
                os.rmdir(self._bam_dir)
            except OSError:
                pass

    def _default_anim(self) -> str:
        if not self._anim_names:
            raise ValueError("Model has no animations")
        return self._anim_names[0]
