import os
import json
import time
import argparse

from itertools import islice
from model.model import RankPQOModel
from model_based_solutions import best_plan_selection
from data_management.evaluate_cost_matrix_ob import connect_to_ob, generate_hint_from_plan, fetch_actual_latency
from train.train_pred_version import get_param_info, _meta_path, _param_path
from online_finetuning import online_round_update


def best_plan_prediction_grouped(data, model_path, device, group_dir, group_id, input_file, output_file):
    with open(os.path.join(model_path, "cluster.json")) as f:
        cluster_map = json.load(f)

    template_to_cluster = {
        template_id: str(cluster_id)
        for cluster_id, templates in cluster_map.items()
        for template_id in templates
    }

    connection = connect_to_ob()
    results = {}
    total_predict_time = 0.0
    total_exec_time = 0.0

    job_ids = sorted(os.listdir(group_dir))
    for job_id in job_ids:
        group_file = os.path.join(group_dir, job_id, f"group_{group_id}.json")
        if not os.path.exists(group_file):
            print(f"[Skip] {group_file} not found")
            continue

        if job_id not in template_to_cluster:
            print(f"[Skip] {job_id} not found in cluster.json")
            continue

        cluster_id = template_to_cluster[job_id]
        model_path_file = os.path.join(model_path, cluster_id)
        path = os.path.join(data, job_id)

        meta_path = _meta_path(path)
        if not os.path.exists(meta_path):
            print(f"[Skip] {job_id} missing meta_data.json")
            continue

        print(f"[Processing] {job_id}")

        with open(meta_path) as f:
            meta = json.load(f)
        with open(_param_path(path)) as f:
            param = json.load(f)
        with open(os.path.join(path, input_file)) as f:
            plan = json.load(f)

        template = meta["template"]
        params, preprocess_info = get_param_info(meta)

        plans = list(plan.values())
        plan_keys = list(plan.keys())

        rank_PQO_model = RankPQOModel(None, job_id, preprocess_info, device=device)
        rank_PQO_model.load(model_path_file, first_layer=1)
        feature_generator = rank_PQO_model._feature_generator
        plans_t = feature_generator.transform(plans)

        with open(group_file) as f:
            param_dict = json.load(f)

        results[job_id] = {}

        for param_key, param_values in islice(param_dict.items(), 1):
            parameter_trans = feature_generator.transform_z([param_values], params, preprocess_info)[0]

            start = time.time()
            best_idx = best_plan_selection(rank_PQO_model, parameter_trans, plans_t, device)
            end = time.time()
            predict_time = end - start
            total_predict_time += predict_time

            best_plan = plans[best_idx]
            plan_hint = generate_hint_from_plan(best_plan)
            query_with_hint = template.replace("SELECT", f"SELECT /*+ {plan_hint} */", 1)
            latency = fetch_actual_latency(connection, query_with_hint, param_values)
            total_exec_time += latency

            results[job_id][param_key] = {
                "best_plan_id": plan_keys[best_idx],
                "latency": latency,
                "predict_time": predict_time
            }

    summary = {
        "predict_time": total_predict_time,
        "execution_time": total_exec_time,
        "total": total_predict_time + total_exec_time
    }
    results["summary"] = summary

    print(f"[Group {group_id}] Total Time: {summary['total']:.3f}s")

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)



def best_plan_prediction_all_groups(data, model_path, device, group_dir, group_count, input_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    overall_time = 0.0
    for group_id in range(1, group_count + 1):
        output_file = os.path.join(output_dir, f"group_{group_id}_eval.json")
        best_plan_prediction_grouped(data, model_path, device, group_dir, group_id, input_file, output_file)
        with open(output_file) as f:
            result = json.load(f)
            overall_time += result.get("summary", {}).get("total", 0.0)
    print(f"[All Groups] Total Execution Time: {overall_time:.3f}s")


def best_plan_prediction_original_ob(data, group_dir, group_count, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    connection = connect_to_ob()
    overall_time = 0.0

    job_ids = sorted(os.listdir(group_dir)) 

    for t in range(1, group_count + 1):
        group_output = {}
        group_time = 0.0

        for job_id in job_ids:
            group_file = os.path.join(group_dir, job_id, f"group_{t}.json")
            if not os.path.exists(group_file):
                print(f"[Skip] {group_file} not found")
                continue

            path = os.path.join(data, job_id)
            meta_path = _meta_path(path)
            if not os.path.exists(meta_path):
                print(f"[Skip] {job_id} missing meta_data.json")
                continue

            print(f"[Processing] {job_id} ")

            with open(meta_path) as f:
                meta = json.load(f)

            template = meta["template"]
            with open(group_file) as f:
                param_dict = json.load(f)

            group_output[job_id] = {}

            for param_key, param_values in islice(param_dict.items(), 1):
                latency = fetch_actual_latency(connection, template, param_values)
                group_output[job_id][param_key] = {"latency": latency}
                group_time += latency

        group_output["summary"] = {"execution_time": group_time}
        overall_time += group_time

        output_file = os.path.join(output_dir, f"group_{t}_eval_ob.json")
        with open(output_file, "w") as f:
            json.dump(group_output, f, indent=2)
        print(f"[Group {t}] OB total execution time: {group_time:.3f}s")

    print(f"[All Groups - OB] Total execution time: {overall_time:.3f}s")




def best_plan_prediction_finetune_interleaved_lazy(
    data, model_path, device, group_dir, group_count,
    input_file, output_dir, k, tau_max, max_steps
):
    import time
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(model_path, "cluster.json")) as f:
        cluster_map = json.load(f)

    template_to_cluster = {
        template_id: str(cluster_id)
        for cluster_id, templates in cluster_map.items()
        for template_id in templates
    }

    total_predict_time = 0.0
    total_exec_time = 0.0
    total_finetune_time = 0.0

    for t in [1, 2, 3]:
        output_file = os.path.join(output_dir, f"group_{t}_eval_finetuned.json")
        best_plan_prediction_grouped(
            data, model_path, device, group_dir,
            group_id=t, input_file=input_file,
            output_file=output_file
        )
        with open(output_file) as f:
            result = json.load(f)
            total_predict_time += result.get("summary", {}).get("predict_time", 0.0)
            total_exec_time += result.get("summary", {}).get("execution_time", 0.0)

    for t in range(4, group_count + 1):
        # Predict group t
        print(f"\n[Time Slot {t}] Serving Queries")
        output_file = os.path.join(output_dir, f"group_{t}_eval_finetuned.json")
        best_plan_prediction_grouped(
            data, model_path, device, group_dir,
            group_id=t, input_file=input_file,
            output_file=output_file
        )
        with open(output_file) as f:
            result = json.load(f)
            total_predict_time += result.get("summary", {}).get("predict_time", 0.0)
            total_exec_time += result.get("summary", {}).get("execution_time", 0.0)

        # Finetune using t-3 and t-2 for next round
        print(f"\n[Time Slot {t-3} -> {t-2}] Online Finetune + Reselection")
        finetune_start = time.time()

        for template_id in template_to_cluster:
            cluster_id = template_to_cluster[template_id]
            model_cluster_path = os.path.join(model_path, cluster_id)
            template_path = os.path.join(data, template_id)

            meta = json.load(open(_meta_path(template_path)))
            params, preprocess_info = get_param_info(meta)
            param_types = [p["data_type"] for p in params]

            full_plan_path = os.path.join(template_path, "hybrid_plans.json")
            selected_plan_path = os.path.join(template_path, input_file)
            output_selected_plan_path = os.path.join(template_path, input_file)

            online_round_update(
                template_id,
                param_dir=group_dir,
                group_t=t - 3,
                group_t_plus_1=t - 2,
                model_path=model_cluster_path,
                selected_plan_path=selected_plan_path,
                full_plan_path=full_plan_path,
                param_types=param_types,
                preprocess_info=preprocess_info,
                params=params,
                k=k,
                tau_max=tau_max,
                max_steps=max_steps,
                device=device,
                output_selected_plan_path=output_selected_plan_path,
                reselect_strategy="prepend"
            )

        finetune_end = time.time()
        total_finetune_time += finetune_end - finetune_start

    print("[Finetune Interleaved Predict-Early Summary]")
    print(f"Total Finetune Time: {total_finetune_time:.3f}s")
    print(f"Total Predict Time: {total_predict_time:.3f}s")
    print(f"Total Execution Time: {total_exec_time:.3f}s")
    print(f"Total (Predict + Exec): {total_predict_time + total_exec_time:.3f}s")


def best_plan_prediction_finetune_interleaved_now(
    data, model_path, device, group_dir, group_count,
    input_file, output_dir, k, tau_max, max_steps
):
    import time
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(model_path, "cluster.json")) as f:
        cluster_map = json.load(f)

    template_to_cluster = {
        template_id: str(cluster_id)
        for cluster_id, templates in cluster_map.items()
        for template_id in templates
    }

    total_predict_time = 0.0
    total_exec_time = 0.0
    total_finetune_time = 0.0

    # Evaluate group 0 and 1 without finetune
    for t in [1, 2]:
        output_file = os.path.join(output_dir, f"group_{t}_eval_finetuned.json")
        best_plan_prediction_grouped(
            data, model_path, device, group_dir,
            group_id=t, input_file=input_file,
            output_file=output_file
        )
        with open(output_file) as f:
            result = json.load(f)
            total_predict_time += result.get("summary", {}).get("predict_time", 0.0)
            total_exec_time += result.get("summary", {}).get("execution_time", 0.0)

    for t in range(3, group_count + 1):
        print(f"\n[Time Slot {t-2} -> {t-1}] Online Finetune + Reselection")
        finetune_start = time.time()

        for template_id in template_to_cluster:
            cluster_id = template_to_cluster[template_id]
            model_cluster_path = os.path.join(model_path, cluster_id)
            template_path = os.path.join(data, template_id)

            meta = json.load(open(_meta_path(template_path)))
            params, preprocess_info = get_param_info(meta)
            param_types = [p["data_type"] for p in params]

            full_plan_path = os.path.join(template_path, "hybrid_plans.json")
            selected_plan_path = os.path.join(template_path, input_file)
            output_selected_plan_path = os.path.join(template_path, input_file)

            online_round_update(
                template_id,
                param_dir=group_dir,
                group_t=t - 2,
                group_t_plus_1=t - 1,
                model_path=model_cluster_path,
                selected_plan_path=selected_plan_path,
                full_plan_path=full_plan_path,
                param_types=param_types,
                preprocess_info=preprocess_info,
                params=params,
                k=k,
                tau_max=tau_max,
                max_steps=max_steps,
                device=device,
                output_selected_plan_path=output_selected_plan_path,
                reselect_strategy="prepend"
            )

        finetune_end = time.time()
        finetune_time = finetune_end - finetune_start
        total_finetune_time += finetune_time

        print(f"\n[Time Slot {t}] Serving Queries")
        output_file = os.path.join(output_dir, f"group_{t}_eval_finetuned.json")
        best_plan_prediction_grouped(
            data, model_path, device, group_dir,
            group_id=t, input_file=input_file,
            output_file=output_file
        )

        with open(output_file) as f:
            result = json.load(f)
            total_predict_time += result.get("summary", {}).get("predict_time", 0.0)
            total_exec_time += result.get("summary", {}).get("execution_time", 0.0)

    print("[Finetune Interleaved Predict-Now Summary]")
    print(f"Total Finetune Time: {total_finetune_time:.3f}s")
    print(f"Total Predict Time: {total_predict_time:.3f}s")
    print(f"Total Execution Time: {total_exec_time:.3f}s")
    print(f"Total (Predict + Exec): {total_finetune_time + total_predict_time + total_exec_time:.3f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grouped Evaluation Runner")
    parser.add_argument("--function", type=str, required=True)
    parser.add_argument("--training_data", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=False)
    parser.add_argument("--group_dir", type=str, required=True)
    parser.add_argument("--group_count", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max_plan_num", type=int, default=30)
    parser.add_argument("--tau_max", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--input_file", type=str, default="selected_plans.json")

    args = parser.parse_args()

    if args.function == "eval_all":
        best_plan_prediction_all_groups(args.training_data, args.model_path, args.device, args.group_dir, args.group_count, args.input_file, args.output_dir)
    elif args.function == "eval_ob":
        best_plan_prediction_original_ob(args.training_data, args.group_dir, args.group_count, args.output_dir)
    elif args.function == "eval_finetune_now":
        best_plan_prediction_finetune_interleaved_now(args.training_data, args.model_path, args.device, args.group_dir, args.group_count, args.input_file, args.output_dir, args.max_plan_num, args.tau_max, args.max_steps)
    elif args.function == "eval_finetune_lazy":
        best_plan_prediction_finetune_interleaved_lazy(args.training_data, args.model_path, args.device, args.group_dir, args.group_count, args.input_file, args.output_dir, args.max_plan_num, args.tau_max, args.max_steps)
    else:
        raise ValueError("Unsupported function: " + args.function)