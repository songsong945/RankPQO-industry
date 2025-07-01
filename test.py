import os
import json
import numpy as np
from online_finetuning import _meta_path, compute_cluster_kl
from train.train_pred_version import get_param_info

def infer_param_types(param_data):
    if not param_data:
        return []

    num_columns = len(param_data[0])
    param_types = []

    for col in range(num_columns):
        is_numeric = True
        for row in param_data:
            try:
                float(row[col])
            except (ValueError, TypeError):
                is_numeric = False
                break
        param_types.append("numeric" if is_numeric else "categorical")

    return param_types


def compute_kl_matrix(data_dir, param_dir, group_count):

    template_ids = [
        name for name in os.listdir(param_dir)
        if os.path.isdir(os.path.join(param_dir, name))
    ]

    kl_matrix = np.zeros((group_count, group_count))
    count_matrix = np.zeros((group_count, group_count))

    for template_id in template_ids:
        template_path = os.path.join(data_dir, template_id)
        meta_path = _meta_path(template_path)
        if not os.path.exists(meta_path):
            print(f"[Skip] Missing meta: {meta_path}")
            continue

        meta = json.load(open(_meta_path(template_path)))
        params, preprocess_info = get_param_info(meta)
        param_types = [p["data_type"] for p in params]

        for i in range(group_count):
            for j in range(group_count):
                if i == j:
                    continue
                path_i = os.path.join(param_dir, template_id, f"group_{i + 1}.json")
                path_j = os.path.join(param_dir, template_id, f"group_{j + 1}.json")

                if not os.path.exists(path_i) or not os.path.exists(path_j):
                    print(f"[Skip] {template_id} G{i+1} vs G{j+1}: file missing")
                    continue

                try:
                    with open(path_i) as f1, open(path_j) as f2:
                        data1 = list(json.load(f1).values())
                        data2 = list(json.load(f2).values())
                    if not data1 or not data2:
                        print(f"[Skip] {template_id} G{i+1} vs G{j+1}: empty")
                        continue

                    D_KL = compute_cluster_kl(data1, data2, param_types)
                    kl_matrix[i][j] += D_KL
                    count_matrix[i][j] += 1
                except Exception as e:
                    print(f"[Error] {template_id} G{i+1} vs G{j+1}: {e}")

    # Average
    with np.errstate(invalid='ignore'):
        avg_matrix = np.divide(kl_matrix, count_matrix, where=(count_matrix != 0))

    print("\nKL Divergence Matrix:")
    for row in avg_matrix:
        print([round(v, 4) if not np.isnan(v) else 0.0 for v in row])

    return avg_matrix

if __name__ == "__main__":
    data_dir = "./training_data/JOB_330_2"
    param_dir = "./data_management/parameters_330_2"
    group_count = 10

    print(compute_kl_matrix(data_dir, param_dir, group_count))
