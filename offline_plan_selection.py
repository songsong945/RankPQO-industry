import os
import json
import time
import argparse

from model.model import RankPQOModel
from model_based_solutions import candidate_plan_selection
from train.train_pred_version import get_param_info, _meta_path, _plan_path, _param_path


# def _meta_path(base): return os.path.join(base, "meta_data.json")
# def _plan_path(base): return os.path.join(base, "plan_by_join_order.json")
# def _param_path(base): return os.path.join(base, "parameter.json")


def candidate_selection_by_cluster(data, model_path, device, k, output="selected_plans.json"):

    cluster_json_path = os.path.join(model_path, "cluster.json")
    with open(cluster_json_path, "r") as f:
        cluster_map = json.load(f)

    template_to_cluster = {}
    for cluster_id, template_list in cluster_map.items():
        for template_id in template_list:
            template_to_cluster[template_id] = str(cluster_id)

    total_time = 0
    all_folders = [f for f in os.listdir(data) if os.path.isdir(os.path.join(data, f))]

    for template_id in all_folders:
        if template_id not in template_to_cluster:
            print(f"Skipping {template_id} (not in cluster mapping)")
            continue

        cluster_id = template_to_cluster[template_id]
        model_path_file = os.path.join(model_path, cluster_id)

        path = os.path.join(data, template_id)

        with open(_meta_path(path), 'r') as f:
            meta = json.load(f)
        with open(_plan_path(path), 'r') as f:
            plan = json.load(f)
        with open(_param_path(path), 'r') as f:
            param = json.load(f)

        params, preprocess_info = get_param_info(meta)
        plans = list(plan.values())
        parameters = list(param.values())

        # Load model for this template’s cluster
        rank_PQO_model = RankPQOModel(None, template_id, preprocess_info, device=device)
        rank_PQO_model.load(model_path_file, first_layer=1)
        feature_generator = rank_PQO_model._feature_generator

        plans_trans = feature_generator.transform(plans)
        parameters_trans = feature_generator.transform_z(parameters, params, preprocess_info)

        start_time = time.time()
        selected_plans = candidate_plan_selection(rank_PQO_model, parameters_trans, plans_trans, k, device)
        end_time = time.time()

        selected_plan_dict = {}
        all_plan_keys = list(plan.keys())
        for idx in selected_plans:
            key = all_plan_keys[idx]
            selected_plan_dict[key] = plan[key]

        selected_plans_path = os.path.join(path, output)
        with open(selected_plans_path, 'w') as f:
            json.dump(selected_plan_dict, f, indent=4)

        print(f"Saved selected plans for {template_id} to {selected_plans_path}")
        total_time += (end_time - start_time)

    print(f"Total plan selection time: {total_time:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Candidate Plan Selection with Clustered Models")

    parser.add_argument("--function", type=str, default="candidate_selection", help="Function to run")
    parser.add_argument("--training_data", type=str, required=True, help="Path to the folder containing template subfolders")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained models and cluster.json")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use: 'cpu' or 'cuda'")
    parser.add_argument("--k", type=int, default=5, help="Number of candidate plans to select")
    parser.add_argument("--output", type=str, default="selected_plans.json", help="Output file name to save selected plans")

    args = parser.parse_args()

    print("Function: candidate_selection_by_cluster")
    print("Training data:", args.training_data)
    print("Model path:", args.model_path)
    print("Device:", args.device)
    print("Top-K:", args.k)
    print("Output file:", args.output)

    if args.function == "candidate_selection_by_cluster":
        candidate_selection_by_cluster(
            data=args.training_data,
            model_path=args.model_path,
            device=args.device,
            k=args.k,
            output=args.output
        )
    else:
        raise ValueError(f"Unsupported function name: {args.function}")

