import json
import os
import random
import numpy as np
from scipy.stats import entropy
from scipy.special import rel_entr
from collections import Counter
from math import ceil

from model_based_solutions import candidate_plan_selection
from model.model import RankPQOModel
from train.train_pred_version import _meta_path

from data_management.evaluate_cost_matrix_ob import connect_to_ob, generate_hint_from_plan, fetch_ob_cost

def generate_cost_dict_from_ob(connection, template: str, plan_dict: dict, param_dict: dict) -> dict:
    # connection = connect_to_ob()
    cost_dict = {}

    for param_key in param_dict:
        param_values = param_dict[param_key]
        plan_costs = {}
        for plan_key in plan_dict:
            # print(plan)
            plan = plan_dict[plan_key]
            hint = generate_hint_from_plan(plan)
            query_with_hint = f"EXPLAIN FORMAT=JSON " + template.replace("SELECT", f"SELECT /*+ {hint} */", 1)
            try:
                cost = fetch_ob_cost(connection, query_with_hint, param_values)
                plan_costs[plan_key] = cost
            except Exception as e:
                print(f"[Error] Cost fetch failed for param {param_key}, plan {plan_key}: {e}")
                continue

        cost_dict[str(param_key)] = plan_costs

    return cost_dict

def generate_pairwise_by_cost(cost_dict, plan_dict, param_dict):

    Z, X1, X2, Y1, Y2 = [], [], [], [], []

    for param_key, plan_costs in cost_dict.items():

        param_values = param_dict.get(param_key)
        if param_values is None:
            print(f"[Warning] Missing parameters for key: {param_key}")
            continue

        sorted_plans = sorted(plan_costs.items(), key=lambda x: x[1]) 
        n = len(sorted_plans)
        for i in range(n):
            for j in range(i + 1, n):
                plan_i_key, _ = sorted_plans[i]
                plan_j_key, _ = sorted_plans[j]

                if plan_i_key not in plan_dict or plan_j_key not in plan_dict:
                    continue

                plan_i = plan_dict[plan_i_key]
                plan_j = plan_dict[plan_j_key]

                Z.append(param_values)
                X1.append(plan_i)
                X2.append(plan_j)
                Y1.append(1)
                Y2.append(0)

    return Z, X1, X2, Y1, Y2


def compute_kl_divergence(P_dist, Q_dist):
    return np.sum(rel_entr(P_dist, Q_dist))

def estimate_distribution(values, bins=10, is_categorical=False):
    if is_categorical:
        counts = Counter(values)
        total = sum(counts.values())
        dist = np.array([counts[val] / total for val in sorted(counts)])
        return dist, sorted(counts)
    else:
        try:
            values = np.array(values, dtype=float)
        except ValueError:
            print(f"[WARN] Detected non-numeric values in feature assumed to be numeric, switching to categorical: {values[:5]}")
            return estimate_distribution(values, bins, is_categorical=True)

        hist, bin_edges = np.histogram(values, bins=bins, density=True)
        hist += 1e-8
        hist /= np.sum(hist)
        return hist, bin_edges



def compute_cluster_kl(param_data_t1, param_data_t2, param_types, bins=10):
    dim = len(param_data_t1[0])
    divergences = []

    for k in range(dim):
        values_t1 = [vec[k] for vec in param_data_t1]
        values_t2 = [vec[k] for vec in param_data_t2]
        is_cat = (param_types[k] == 'categorical')

        p_dist, p_space = estimate_distribution(values_t1, bins, is_cat)
        q_dist, _ = estimate_distribution(values_t2, bins if not is_cat else len(p_space), is_cat)

        if len(p_dist) > len(q_dist):
            q_dist = np.pad(q_dist, (0, len(p_dist) - len(q_dist)), constant_values=1e-8)
        elif len(q_dist) > len(p_dist):
            p_dist = np.pad(p_dist, (0, len(q_dist) - len(p_dist)), constant_values=1e-8)

        div_k = compute_kl_divergence(p_dist, q_dist)
        divergences.append(div_k)

    return np.mean(divergences)



def compute_finetune_steps(D_kl, N_max, tau_max):

    ratio = min(1.0, D_kl / tau_max)
    return int(N_max * ratio)


def compute_reselection_count(D_kl, k, tau_max):

    ratio = min(1.0, D_kl / tau_max)
    return ceil(k * ratio)


# def online_finetune_if_shifted(
#     param_path_t_prev, param_path_t_curr, 
#     param_types, model, preprocess_info, 
#     max_steps, tau_max, device
# ):

#     with open(param_path_t_prev) as f1:
#         param_data_1 = list(json.load(f1).values())

#     with open(param_path_t_curr) as f2:
#         param_data_2 = list(json.load(f2).values())

#     D_KL = compute_cluster_kl(param_data_1, param_data_2, param_types)
#     steps = compute_finetune_steps(D_KL, max_steps, tau_max)

#     if steps > 0:
#         print(f"[Finetune] D_KL = {D_KL:.4f}, performing {steps} steps")
#         feature_generator = model._feature_generator
#         X = feature_generator.transform_z(param_data_2, None, preprocess_info)
#         model.finetune_param_only(X, steps=steps)  # You must implement or expose this method in your model
#     else:
#         print(f"[Finetune] D_KL = {D_KL:.4f}, skipping update")

#     return D_KL, steps


# def online_reselect_candidates_if_shifted(
#     param_path_t_prev, param_path_t_curr, 
#     param_types, model, preprocess_info, 
#     plans, selected_plan_keys, 
#     k, tau_max, device,
#     reselect_strategy="default"
# ):
#     with open(param_path_t_prev) as f1:
#         param_data_1 = list(json.load(f1).values())

#     with open(param_path_t_curr) as f2:
#         param_data_2 = list(json.load(f2).values())

#     D_KL = compute_cluster_kl(param_data_1, param_data_2, param_types)
#     r = compute_reselection_count(D_KL, k, tau_max)

#     if r > 0:
#         print(f"[Reselect] D_KL = {D_KL:.4f}, replacing {r}/{k} candidates")

#         feature_generator = model._feature_generator
#         X_z = feature_generator.transform_z(param_data_2, None, preprocess_info)
#         all_plans = list(plans.values())
#         all_keys = list(plans.keys())
#         transformed_plans = feature_generator.transform(all_plans)

#         selected_ids = candidate_plan_selection(model, X_z, transformed_plans, r, device)
#         new_keys = [all_keys[i] for i in selected_ids]

#         if reselect_strategy == "prepend":
#             remaining_old = selected_plan_keys[:k - r]
#             new_selected_keys = new_keys + remaining_old
#         else:  # default is append
#             remaining_old = selected_plan_keys[:k - r]
#             new_selected_keys = remaining_old + new_keys
#     else:
#         print(f"[Reselect] D_KL = {D_KL:.4f}, no update needed")
#         new_selected_keys = selected_plan_keys

#     updated = {key: plans[key] for key in new_selected_keys}
#     return updated, D_KL, r


def online_round_update(
    template_id,
    param_dir,
    group_t,
    group_t_plus_1,
    model_path,
    selected_plan_path,
    full_plan_path,
    param_types,
    preprocess_info,
    params,
    k,
    tau_max,
    max_steps,
    device,
    output_selected_plan_path,
    reselect_strategy="random"
):
    path_t_prev = os.path.join(param_dir, template_id, f"group_{group_t}.json")
    path_t_curr = os.path.join(param_dir, template_id, f"group_{group_t_plus_1}.json")

    with open(full_plan_path) as f:
        all_plans = json.load(f)
    with open(selected_plan_path) as f:
        selected_plans = json.load(f)
    with open(path_t_prev) as f:
        param_dict = json.load(f)
    with open(path_t_curr) as f:
        param_dict_next = json.load(f)

    model = RankPQOModel(None, template_id, preprocess_info, device=device)
    model.load(model_path, first_layer=1)
    feature_generator = model._feature_generator

    connection = connect_to_ob()
    with open(_meta_path(os.path.dirname(full_plan_path))) as f:
        meta = json.load(f)
    template = meta["template"]

    cost_dict = generate_cost_dict_from_ob(connection, template, all_plans, param_dict)
    Z, X1, X2, Y1, Y2 = generate_pairwise_by_cost(cost_dict, all_plans, param_dict)

    D_KL_finetune = compute_cluster_kl(list(param_dict.values()), list(param_dict_next.values()), param_types)
    steps = compute_finetune_steps(D_KL_finetune, max_steps, tau_max)

    if steps > 0:
        print(f"[Finetune] D_KL = {D_KL_finetune:.4f}, performing {steps} steps")
        X1 = feature_generator.transform(X1)
        X2 = feature_generator.transform(X2)
        Z = feature_generator.transform_z(Z, params, preprocess_info)
        model.finetune_param_only(X1, X2, Y1, Y2, Z, steps=steps)
    else:
        print(f"[Finetune] D_KL = {D_KL_finetune:.4f}, skipping update")

    D_KL_reselect = compute_cluster_kl(list(param_dict.values()), list(param_dict_next.values()), param_types)
    r = compute_reselection_count(D_KL_reselect, k, tau_max)

    if r > 0:
        print(f"[Reselect] D_KL = {D_KL_reselect:.4f}, replacing {r}/{k} candidates")
        X_z = feature_generator.transform_z(list(param_dict_next.values()), params, preprocess_info)
        all_keys = list(all_plans.keys())
        all_values = list(all_plans.values())
        transformed_plans = feature_generator.transform(all_values)

        selected_ids = candidate_plan_selection(model, X_z, transformed_plans, r, device)
        new_keys = [all_keys[i] for i in selected_ids]

        selected_keys = list(selected_plans.keys())
        if reselect_strategy == "prepend":
            final_keys = new_keys + selected_keys[:k - r]
        else:
            import random
            keep_old = random.sample(selected_keys, k - r)
            final_keys = new_keys + keep_old
    else:
        print(f"[Reselect] D_KL = {D_KL_reselect:.4f}, no update needed")
        final_keys = list(selected_plans.keys())

    updated = {key: all_plans[key] for key in final_keys}

    with open(output_selected_plan_path, "w") as f:
        json.dump(updated, f, indent=2)

    model.save(model_path)

    return {
        "D_KL_finetune": D_KL_finetune,
        "finetune_steps": steps,
        "D_KL_reselect": D_KL_reselect,
        "reselect_count": r
    }



