# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, sentinel

from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.models.monty_base import MontyBase
from tbp.monty.frameworks.models.motor_system_state import AgentState, SensorState
from tbp.monty.frameworks.sensors import SensorID


class MontyBasePrivateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sm1 = MagicMock()
        self.sm1.sensor_module_id = "sm1"
        self.sm2 = MagicMock()
        self.sm2.sensor_module_id = "sm2"
        self.lm1 = MagicMock()
        self.lm2 = MagicMock()
        self.lm3 = MagicMock()
        self.motor_system = MagicMock()
        self.motor_system.motor_only_step = False
        self.monty_base = MontyBase(
            sensor_modules=[self.sm1, self.sm2],
            learning_modules=[self.lm1, self.lm2, self.lm3],
            motor_system=self.motor_system,
            sm_to_agent_dict={
                "sm1": AgentID("agent_id_0"),
                "sm2": AgentID("agent_id_1"),
            },
            sm_to_lm_matrix=[[], [], []],
            lm_to_lm_matrix=[[], [], []],
            lm_to_lm_vote_matrix=[[], [], []],
            min_eval_steps=10,
            min_train_steps=10,
            num_exploratory_steps=10,
            max_total_steps=100,
        )

    def test_aggregate_sensory_inputs_uses_per_sensor_agent_state(self) -> None:
        agent_state_0 = AgentState(
            sensors={
                SensorID("sm1"): SensorState(
                    position=(0.0, 0.0, 0.0),
                    rotation=sentinel.sensor_rotation_0,
                )
            },
            position=(1.0, 0.0, 0.0),
            rotation=sentinel.agent_rotation_0,
        )
        agent_state_1 = AgentState(
            sensors={
                SensorID("sm2"): SensorState(
                    position=(0.0, 0.0, 0.0),
                    rotation=sentinel.sensor_rotation_1,
                )
            },
            position=(0.0, 1.0, 0.0),
            rotation=sentinel.agent_rotation_1,
        )
        self.motor_system._state = {
            AgentID("agent_id_0"): agent_state_0,
            AgentID("agent_id_1"): agent_state_1,
        }
        self.lm1.get_output.return_value = sentinel.lm1_output
        self.lm2.get_output.return_value = sentinel.lm2_output
        self.lm3.get_output.return_value = sentinel.lm3_output

        observations = {
            AgentID("agent_id_0"): {"sm1": sentinel.raw_obs_1},
            AgentID("agent_id_1"): {"sm2": sentinel.raw_obs_2},
        }

        self.monty_base.aggregate_sensory_inputs(sentinel.ctx, observations)

        self.sm1.update_state.assert_called_once_with(agent_state_0)
        self.sm2.update_state.assert_called_once_with(agent_state_1)
        self.sm1.step.assert_called_once_with(sentinel.ctx, sentinel.raw_obs_1, False)
        self.sm2.step.assert_called_once_with(sentinel.ctx, sentinel.raw_obs_2, False)

    def test_pass_goal_states_collects_all_goals_from_learning_and_sensor_modules(
        self,
    ) -> None:
        self.monty_base.step_type = "matching_step"
        self.lm1.propose_goal_states.return_value = []
        self.lm2.propose_goal_states.return_value = [sentinel.lm2_goal]
        self.lm3.propose_goal_states.return_value = [
            sentinel.lm3_goal_1,
            sentinel.lm3_goal_2,
        ]
        self.sm1.propose_goal_states.return_value = []
        self.sm2.propose_goal_states.return_value = [
            sentinel.sm2_goal_1,
            sentinel.sm2_goal_2,
        ]
        self.monty_base._pass_goal_states()

        expected = set(
            {
                sentinel.lm2_goal,
                sentinel.lm3_goal_1,
                sentinel.lm3_goal_2,
                sentinel.sm2_goal_1,
                sentinel.sm2_goal_2,
            }
        )
        self.assertEqual(set(self.monty_base.gsg_outputs), expected)

    def test_pre_episode_resets_motor_system_before_modules(self) -> None:
        self.monty_base.pre_episode()

        self.motor_system.pre_episode.assert_called_once_with()
        self.lm1.pre_episode.assert_called_once_with()
        self.lm2.pre_episode.assert_called_once_with()
        self.lm3.pre_episode.assert_called_once_with()
        self.sm1.pre_episode.assert_called_once_with()
        self.sm2.pre_episode.assert_called_once_with()
