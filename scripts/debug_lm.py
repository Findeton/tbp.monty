"""Quick diagnostic: check why _auto_update_terminal_condition doesn't trigger exploratory."""
import sys
sys.path.insert(0, '/Users/felixrobles/workspace/tbp.monty/src')

def main():
    from tbp.monty.frameworks.models.cortical_column_torch.learning_module import CorticalColumnTorchLM
    from tbp.monty.frameworks.models.abstract_monty_classes import ExperimentMode
    from tbp.monty.simulators.panda3d.environment import Panda3DEnvironment
    from tbp.monty.simulators.panda3d.agents import Panda3DAgent
    from tbp.monty.frameworks.sensors import SensorID
    from tbp.monty.simulators.panda3d.transforms import Panda3DDepthNormalize
    from tbp.monty.frameworks.environment_utils.transforms import DepthTo3DLocations
    import numpy as np

    AGENT_ID = "agent_id_0"
    SENSOR_ID = SensorID("patch")

    # Create LM
    lm = CorticalColumnTorchLM(
        column_kwargs=dict(n_minicolumns=2048, n_cells_per_minicolumn=8, sparsity=0.03,
                           use_apical=False, seed=42),
        learning_module_id="lm_0",
    )
    lm.set_experiment_mode(ExperimentMode.TRAIN)
    lm.pre_episode(primary_target={"object": "011_banana", "quat_rotation": [1,0,0,0]})

    print(f"After pre_episode:")
    print(f"  _stepped: {lm._stepped}")
    print(f"  terminal_state: {lm.terminal_state}")
    print(f"  possible_matches: {lm.possible_matches}")
    print(f"  _mode: {lm._mode}")

    # Check _auto_update_terminal_condition
    print("\nCalling _auto_update_terminal_condition (with _stepped=False)...")
    lm._auto_update_terminal_condition()
    print(f"  terminal_state after: {lm.terminal_state}")

    # Simulate stepped=True
    lm._stepped = True
    print("\nCalling _auto_update_terminal_condition (with _stepped=True)...")
    lm._auto_update_terminal_condition()
    print(f"  terminal_state after: {lm.terminal_state}")

    print("\nColumn._evidence:", lm._column._evidence)
    print("\nget_possible_matches:", lm.get_possible_matches())
    print("\nall_no_match would be:", lm.terminal_state == "no_match")

if __name__ == "__main__":
    main()
