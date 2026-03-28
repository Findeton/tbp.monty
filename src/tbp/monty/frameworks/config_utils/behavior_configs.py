# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Configuration builders for behavior experiment testbed.

Builds 2-SM / 2-LM configurations where:
- SM 0 (CameraSM-like) → LM 0 (morphology column)
- SM 1 (ChangeDetectingSM) → LM 1 (behavior column)
- LMs vote with each other

This follows the TBP theory: morphology and behavior use the SAME
LM algorithm but with DIFFERENT SM inputs, in SEPARATE cortical columns.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from tbp.monty.frameworks.environments.behaviors import (
    door_opening,
    pendulum_swing,
    stapler_press,
    walking_gait,
)
from tbp.monty.frameworks.environments.synthetic_behavior_env import (
    SyntheticBehaviorEnvironment,
    make_behavior_sequences,
)
from tbp.monty.frameworks.models.states import State


# Level definitions for the 5-level testbed
TESTBED_LEVELS = {
    1: {
        "description": "3 simple repeated behaviors, no morphology variation",
        "behaviors": {
            "walking": walking_gait,
            "stapler": stapler_press,
            "door": door_opening,
        },
        "n_steps": 40,
    },
    2: {
        "description": (
            "Same behavior on 2 morphologies → behavior recognized"
            " on unseen morphology"
        ),
        "behaviors": {
            "walking": walking_gait,
            "pendulum": pendulum_swing,
        },
        "n_steps": 40,
    },
    3: {
        "description": "1 object with 2 behaviors at different speeds",
        "behaviors": {
            "walking_normal": walking_gait,
            "walking_fast": walking_gait,
        },
        "n_steps": 40,
    },
}


def build_behavior_testbed(
    level: int = 1,
    n_steps: Optional[int] = None,
) -> SyntheticBehaviorEnvironment:
    """Build a synthetic behavior environment for testing.

    Args:
        level: Difficulty level (1-5). See TESTBED_LEVELS.
        n_steps: Override number of steps per behavior.

    Returns:
        SyntheticBehaviorEnvironment with registered behavior sequences.
    """
    if level not in TESTBED_LEVELS:
        raise ValueError(f"Unknown level {level}. Available: {list(TESTBED_LEVELS)}")

    config = TESTBED_LEVELS[level]
    steps = n_steps or config["n_steps"]

    sequences = make_behavior_sequences(
        behaviors=config["behaviors"],
        n_steps=steps,
    )

    env = SyntheticBehaviorEnvironment(sequences=sequences)
    return env


def get_behavior_names(level: int = 1) -> List[str]:
    """Get behavior names for a testbed level."""
    if level not in TESTBED_LEVELS:
        return []
    return list(TESTBED_LEVELS[level]["behaviors"].keys())
