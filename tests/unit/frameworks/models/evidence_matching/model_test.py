# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import unittest
from unittest.mock import sentinel

from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.models.evidence_matching.model import (
    MontyForEvidenceGraphMatching,
)
from tbp.monty.frameworks.models.graph_matching import MontyForGraphMatching


class GraphMatchingMotorHandoffTest(unittest.TestCase):
    def test_pass_infos_to_motor_system_routes_processed_observations_per_agent(
        self,
    ) -> None:
        class _Policy:
            def __init__(self):
                self.percepts_by_agent = None

            def set_agent_processed_observations(self, percepts_by_agent):
                self.percepts_by_agent = percepts_by_agent

        class _MotorSystem:
            def __init__(self):
                self._policy = _Policy()

        class _SensorModule:
            def __init__(self, sensor_module_id):
                self.sensor_module_id = sensor_module_id

        class _TestMonty(MontyForGraphMatching):
            def __init__(self):
                self.step_type = "matching_step"
                self.sensor_modules = [
                    _SensorModule("patch_0"),
                    _SensorModule("patch_1"),
                ]
                self.sensor_module_outputs = [sentinel.percept_0, sentinel.percept_1]
                self.sm_to_agent_dict = {
                    "patch_0": AgentID("agent_id_0"),
                    "patch_1": AgentID("agent_id_1"),
                }
                self.motor_system = _MotorSystem()

        monty = _TestMonty()
        monty._pass_infos_to_motor_system()

        self.assertEqual(
            monty.motor_system._policy.percepts_by_agent,
            {
                AgentID("agent_id_0"): {"patch_0": sentinel.percept_0},
                AgentID("agent_id_1"): {"patch_1": sentinel.percept_1},
            },
        )


class EvidenceMatchingMotorHandoffTest(unittest.TestCase):
    def test_pass_infos_to_motor_system_forwards_all_goal_states_when_supported(
        self,
    ) -> None:
        class _Policy:
            def __init__(self):
                self.use_goal_state_driven_actions = True
                self.received_goal_states = None
                self.percepts_by_agent = None

            def set_agent_processed_observations(self, percepts_by_agent):
                self.percepts_by_agent = percepts_by_agent

            def set_driving_goal_states(self, goal_states):
                self.received_goal_states = list(goal_states)

        class _MotorSystem:
            def __init__(self):
                self._policy = _Policy()

        class _SensorModule:
            def __init__(self, sensor_module_id):
                self.sensor_module_id = sensor_module_id

        class _TestMonty(MontyForEvidenceGraphMatching):
            def __init__(self):
                self.step_type = "matching_step"
                self.sensor_modules = [_SensorModule("patch_0")]
                self.sensor_module_outputs = [sentinel.percept_0]
                self.sm_to_agent_dict = {"patch_0": AgentID("agent_id_0")}
                self.motor_system = _MotorSystem()
                self.gsg_outputs = [sentinel.goal_state_0, sentinel.goal_state_1]

        monty = _TestMonty()
        monty._pass_infos_to_motor_system()

        self.assertEqual(
            monty.motor_system._policy.received_goal_states,
            [sentinel.goal_state_0, sentinel.goal_state_1],
        )

    def test_pass_infos_to_motor_system_filters_to_authoritative_goal_sender(self):
        class _GoalState:
            def __init__(self, sender_id, confidence):
                self.sender_id = sender_id
                self.confidence = confidence
                self.use_state = True

        class _Policy:
            def __init__(self):
                self.use_goal_state_driven_actions = True
                self.received_goal_states = None
                self.percepts_by_agent = None

            def set_agent_processed_observations(self, percepts_by_agent):
                self.percepts_by_agent = percepts_by_agent

            def set_driving_goal_states(self, goal_states):
                self.received_goal_states = list(goal_states)

        class _MotorSystem:
            def __init__(self):
                self._policy = _Policy()

        class _SensorModule:
            def __init__(self, sensor_module_id):
                self.sensor_module_id = sensor_module_id

        class _TestMonty(MontyForEvidenceGraphMatching):
            def __init__(self):
                self.step_type = "matching_step"
                self.sensor_modules = [_SensorModule("patch_0")]
                self.sensor_module_outputs = [sentinel.percept_0]
                self.sm_to_agent_dict = {"patch_0": AgentID("agent_id_0")}
                self.motor_system = _MotorSystem()
                self.authoritative_goal_sender_ids = {"lm_parent"}
                self.gsg_outputs = [
                    _GoalState("lm_behavior", 0.7),
                    _GoalState("lm_parent", 0.6),
                ]

        monty = _TestMonty()
        monty._pass_infos_to_motor_system()

        self.assertEqual(len(monty.motor_system._policy.received_goal_states), 1)
        self.assertEqual(
            monty.motor_system._policy.received_goal_states[0].sender_id,
            "lm_parent",
        )

if __name__ == "__main__":
    unittest.main()
