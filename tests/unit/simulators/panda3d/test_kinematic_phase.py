# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Tests for motion-phase extraction on real animated Panda3D assets."""

from pathlib import Path
import unittest

from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.simulators.panda3d.agents import Panda3DAgent
from tbp.monty.simulators.panda3d.kinematic_phase import (
    extract_foot_cycle_phase_info,
    extract_joint_phase_info,
    infer_root_joint,
    select_foot_joints,
)
from tbp.monty.simulators.panda3d.simulator import Panda3DSimulator

try:
    import panda3d  # noqa: F401
    import gltf  # noqa: F401
except ImportError:
    raise unittest.SkipTest("Panda3D or panda3d-gltf not installed")


AGENT_ID = AgentID("test_cam")
SENSOR_ID = "sensor_0"
ASSET_DIR = Path(__file__).resolve().parent / "test_assets" / "animated"
FOX_PATH = ASSET_DIR / "Fox.glb"


def _assets_available():
    return FOX_PATH.is_file()


def _make_simulator(position=(0, 0, 0), resolution=(64, 64), **kwargs):
    agent = Panda3DAgent(
        agent_id=AGENT_ID,
        sensor_id=SENSOR_ID,
        position=position,
        rotation=(1.0, 0.0, 0.0, 0.0),
        resolution=resolution,
        fov=90.0,
    )
    return Panda3DSimulator(agents=[agent], near=0.01, far=20.0, **kwargs)


def _select_animation_name(animated_object):
    anim_names = list(animated_object.animation_names)
    if not anim_names:
        raise RuntimeError("Animated object has no animations")

    for name in anim_names:
        if "walk" in str(name).lower():
            return name

    return anim_names[0]


class TestKinematicPhaseHelpers(unittest.TestCase):
    def test_infer_root_joint_accepts_torso_named_skeleton(self):
        joints = [
            "Skeleton_torso_joint_1",
            "Skeleton_torso_joint_2",
            "leg_joint_L_1",
        ]

        self.assertEqual(
            infer_root_joint(joints),
            "Skeleton_torso_joint_1",
        )

    def test_select_foot_joints_prefers_distal_foot_names(self):
        joints = [
            "b_LeftLeg01_015",
            "b_LeftFoot01_017",
            "b_LeftFoot02_018",
            "b_RightLeg01_019",
            "b_RightFoot01_021",
            "b_RightFoot02_022",
        ]

        self.assertEqual(
            select_foot_joints(joints),
            ("b_LeftFoot02_018", "b_RightFoot02_022"),
        )

    def test_select_foot_joints_supports_dot_l_dot_r_names(self):
        joints = [
            "Foot.L",
            "Foot.L_end",
            "Foot.R",
            "Foot.R_end",
        ]

        self.assertEqual(
            select_foot_joints(joints),
            ("Foot.L_end", "Foot.R_end"),
        )

    def test_select_foot_joints_falls_back_to_leg_suffixes(self):
        joints = [
            "leg_joint_L_1",
            "leg_joint_L_3",
            "leg_joint_L_5",
            "leg_joint_R_1",
            "leg_joint_R_3",
            "leg_joint_R_5",
        ]

        self.assertEqual(
            select_foot_joints(joints),
            ("leg_joint_L_5", "leg_joint_R_5"),
        )


@unittest.skipUnless(_assets_available(), "Real 3D animated assets not found")
class TestRealAnimatedMeshPhaseExtraction(unittest.TestCase):
    def _load_fox_animation(self):
        sim = _make_simulator(asset_search_paths=[str(ASSET_DIR)])
        info = sim.add_object(str(FOX_PATH), animated=True)
        anim = sim.get_animated_object(info.object_id)
        return sim, anim, _select_animation_name(anim)

    def test_extract_joint_phase_info_on_real_fox(self):
        sim, anim, anim_name = self._load_fox_animation()
        try:
            phase_info = extract_joint_phase_info(
                anim,
                anim_name=anim_name,
                n_phase_bins=4,
            )
            self.assertEqual(
                len(phase_info["phase_by_frame"]),
                anim.get_num_frames(anim_name),
            )
            self.assertTrue(phase_info["joint_names"])
            self.assertIsNotNone(phase_info["root_joint_name"])
            self.assertGreaterEqual(phase_info["phase_span"], 0.0)
        finally:
            sim.close()

    def test_extract_foot_cycle_phase_info_on_real_fox(self):
        sim, anim, anim_name = self._load_fox_animation()
        try:
            phase_info = extract_foot_cycle_phase_info(
                anim,
                anim_name=anim_name,
                n_phase_bins=4,
            )
            self.assertEqual(
                len(phase_info["phase_by_frame"]),
                anim.get_num_frames(anim_name),
            )
            self.assertEqual(phase_info["phase_method"], "foot_cycle_angle")
            self.assertTrue(
                any("LeftFoot" in name for name in phase_info["joint_names"])
            )
            self.assertTrue(
                any("RightFoot" in name for name in phase_info["joint_names"])
            )
            self.assertGreaterEqual(len(set(phase_info["phase_by_frame"])), 3)
        finally:
            sim.close()


if __name__ == "__main__":
    unittest.main()