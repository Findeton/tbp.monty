from tests.integration.frameworks.models.test_predictive_hypothesis_torch_monty import FOX_PATH, _track14_kwargs
from tbp.monty.frameworks.environment_utils.transforms import DepthTo3DLocations
from tbp.monty.frameworks.models.cortical_column_torch.experiment import Panda3DTorchExperiment
from tbp.monty.simulators.panda3d.agents import Panda3DAgent
from tbp.monty.simulators.panda3d.simulator import Panda3DSimulator
from tbp.monty.simulators.panda3d.transforms import Panda3DDepthNormalize
from tbp.monty.frameworks.models.evidence_matching.model import MontyForEvidenceGraphMatching


def main():
    exp = Panda3DTorchExperiment(model_path=FOX_PATH, **_track14_kwargs())

    print('agent', flush=True)
    agent = Panda3DAgent(
        agent_id=exp._agent_id,
        sensor_id=str(exp._panda3d_sensor_id),
        resolution=exp._resolution,
        fov=exp._fov,
    )
    print('sim', flush=True)
    sim = Panda3DSimulator(
        agents=[agent],
        near=exp._near,
        far=exp._far,
        asset_search_paths=exp._asset_search_paths,
    )
    print('add_object', flush=True)
    info = sim.add_object(
        name=str(exp._model_path),
        position=exp._object_position,
        scale=exp._object_scale,
        animated=True,
    )
    print('get_animated_object', flush=True)
    exp._sim = sim
    exp._obj_id = info.object_id
    exp._anim_obj = sim.get_animated_object(info.object_id)
    print('sync_render', flush=True)
    exp._sync_render()
    print('depth_transform', flush=True)
    exp._depth_transform = Panda3DDepthNormalize(
        agent_id=exp._agent_id,
        near=exp._near,
        far=exp._far,
    )
    print('d3d_transform', flush=True)
    exp._d3d_transform = DepthTo3DLocations(
        agent_id=exp._agent_id,
        sensor_ids=[exp._panda3d_sensor_id],
        resolutions=[exp._resolution],
        hfov=exp._fov,
        world_coord=True,
        get_all_points=True,
    )

    print('build_camera_sensor_module', flush=True)
    sm_camera = exp._build_camera_sensor_module()
    print(type(sm_camera).__name__, flush=True)

    print('build_change_sensor_module', flush=True)
    sm_change = exp._build_change_sensor_module()
    print(type(sm_change).__name__, flush=True)

    child_kw = dict(
        n_minicolumns=2048,
        n_cells_per_minicolumn=8,
        sparsity=0.03,
        use_apical=True,
        seed=exp._seed,
    )
    child_kw.update(exp._column_kwargs)
    behavior_seed = int(child_kw.get('seed', exp._seed)) + 1
    morphology_child_kw = dict(child_kw)
    morphology_child_kw.update(exp._morphology_column_kwargs)
    behavior_child_kw = dict(child_kw)
    behavior_child_kw.update(exp._behavior_column_kwargs)
    behavior_child_kw.setdefault('seed', behavior_seed)

    print('build_lm_morphology', flush=True)
    lm_morph = exp._build_learning_module(
        learning_module_id='lm_morphology',
        column_kwargs=morphology_child_kw,
        lm_kwargs=exp._morphology_lm_kwargs,
    )
    print(type(lm_morph).__name__, flush=True)

    print('build_lm_behavior', flush=True)
    lm_behav = exp._build_learning_module(
        learning_module_id='lm_behavior',
        column_kwargs=behavior_child_kw,
        lm_kwargs=exp._behavior_lm_kwargs,
    )
    print(type(lm_behav).__name__, flush=True)

    parent_kw = dict(
        n_minicolumns=2048,
        n_cells_per_minicolumn=8,
        sparsity=0.03,
        use_apical=True,
        defer_context_auto_label=True,
        seed=behavior_seed + 1,
    )
    parent_kw.update(exp._parent_column_kwargs)
    parent_lm_kwargs = dict(
        expected_context_sender_ids=['lm_morphology', 'lm_behavior'],
        context_identity_weight=0.35,
        use_child_graph_context_labels=True,
    )
    parent_lm_kwargs.update(exp._parent_lm_kwargs)

    print('build_lm_parent', flush=True)
    lm_parent = exp._build_learning_module(
        learning_module_id='lm_parent',
        column_kwargs=parent_kw,
        lm_kwargs=parent_lm_kwargs,
    )
    print(type(lm_parent).__name__, flush=True)

    print('build_motor_system', flush=True)
    motor_system = exp._build_motor_system()
    print(type(motor_system).__name__, flush=True)

    print('build_monty', flush=True)
    monty = MontyForEvidenceGraphMatching(
        sensor_modules=[sm_camera, sm_change],
        learning_modules=[lm_morph, lm_behav, lm_parent],
        motor_system=motor_system,
        sm_to_agent_dict={
            exp.CAMERA_SM_ID: exp._agent_id,
            exp.CHANGE_SM_ID: exp._agent_id,
        },
        sm_to_lm_matrix=[[0], [1], []],
        lm_to_lm_matrix=[[], [], [0, 1]],
        lm_to_lm_vote_matrix=[[1], [0], []],
        min_eval_steps=9999,
        min_train_steps=9999,
        num_exploratory_steps=9999,
        max_total_steps=99999,
        conditional_voting=exp._conditional_voting,
        vote_confident_threshold=exp._vote_confident_threshold,
        hopfield_voting=exp._hopfield_voting,
        hopfield_surprise_threshold=exp._hopfield_surprise_threshold,
        predictive_voting=exp._predictive_voting,
        temporal_confusion_threshold=exp._temporal_confusion_threshold,
        vote_after_steps=exp._vote_after_steps,
        vote_cooldown_steps=exp._vote_cooldown_steps,
        authoritative_goal_sender_ids=exp._authoritative_goal_sender_ids,
    )
    print(type(monty).__name__, flush=True)

    exp.close()
    print('done', flush=True)


if __name__ == '__main__':
    main()
