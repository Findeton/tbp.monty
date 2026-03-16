from pathlib import Path
import shutil

import torch


def merge_state_dicts(paths):
    lm_dict = {}
    template_state = None

    for model_path in paths:
        state = torch.load(model_path, map_location="cpu")
        template_state = state
        for lm, lm_state in state["lm_dict"].items():
            agg = lm_dict.setdefault(
                lm,
                {
                    "graph_memory": {},
                    "target_to_graph_id": {},
                    "graph_id_to_target": {},
                },
            )
            agg["graph_memory"].update(lm_state["graph_memory"])
            agg["target_to_graph_id"].update(lm_state["target_to_graph_id"])
            agg["graph_id_to_target"].update(lm_state["graph_id_to_target"])

    merged_state = {k: v for k, v in template_state.items() if k != "lm_dict"}
    merged_state["lm_dict"] = lm_dict
    return merged_state


def main():
    rerun_base = Path(
        "/Users/felixrobles/tbp/results/monty/pretrained_models/my_trained_models/"
        "supervised_pre_training_objects_with_logos_lvl3_comp_models_rerun"
    )
    old_base = Path(
        "/Users/felixrobles/tbp/results/monty/pretrained_models/my_trained_models/"
        "supervised_pre_training_objects_with_logos_lvl3_comp_models"
    )
    only012 = Path(
        "/Users/felixrobles/tbp/results/monty/pretrained_models/my_trained_models/"
        "supervised_pre_training_objects_with_logos_lvl3_comp_models_only_012/"
        "pretrained/model.pt"
    )
    root_model = rerun_base / "pretrained" / "model.pt"

    if not root_model.exists():
        raise FileNotFoundError(root_model)
    if not only012.exists():
        raise FileNotFoundError(only012)

    old_episode_models = []
    for pdir in sorted(
        old_base.glob(
            "supervised_pre_training_objects_with_logos_lvl3_comp_models-"
            "parallel_train_episode_*"
        )
    ):
        model = pdir / "pretrained" / "model.pt"
        if model.exists():
            old_episode_models.append(model)

    state_paths = [root_model, only012] + old_episode_models
    merged_state = merge_state_dicts(state_paths)

    backup = rerun_base / "pretrained" / "model.before_full_25_merge.pt"
    if not backup.exists():
        shutil.copy2(root_model, backup)

    out_path = rerun_base / "pretrained" / "model.pt"
    torch.save(merged_state, out_path)

    print(f"WROTE {out_path}")
    for lm, lm_state in merged_state["lm_dict"].items():
        keys = sorted(lm_state["graph_memory"].keys())
        print(f"LM {lm} COUNT {len(keys)}")
        print(",".join(keys))


if __name__ == "__main__":
    main()