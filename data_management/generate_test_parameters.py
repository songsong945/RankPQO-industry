import random
import numpy as np
import json
import os


def sample_int(left, right):
    return str(random.randint(left, right))


def sample_float(left, right):
    return str(round(np.random.uniform(left, right), 2))


def sample_text(distinct_values, range_min, range_max):
    non_empty_values = [value for value in distinct_values if value is not None and value != ""]
    sampled_values = non_empty_values[range_min:range_max] if len(non_empty_values) > range_max else non_empty_values
    return random.choice(sampled_values) if sampled_values else None


def generate_parameters(meta_data_path, num_samples=500, shift_interval=50, num_groups=10, output_dir="./parameter_groups"):
    with open(meta_data_path, 'r') as f:
        data = json.load(f)

    predicates = data["predicates"]
    os.makedirs(output_dir, exist_ok=True)
    
    for group in range(num_groups):
        param_combinations = {}
        
        # 生成同一组的left和right范围
        for predicate in predicates:
            if predicate["data_type"] == "float" and predicate["preprocess_type"] == "std_normalization":
                predicate["left"] = np.random.uniform(float(predicate["min"]), float(predicate["max"]) - 0.1)
                predicate["right"] = np.random.uniform(float(predicate["left"]), float(predicate["max"]))
            elif predicate["data_type"] == "int" and predicate["preprocess_type"] == "embedding":
                predicate["left"] = random.randint(int(float(predicate["min"])), int(float(predicate["max"])) - 1)
                predicate["right"] = random.randint(int(predicate["left"]), int(float(predicate["max"])))
            elif predicate["data_type"] == "text" and predicate["preprocess_type"] == "embedding":
                total_values = len(predicate["distinct_values"])
                if total_values > 10:
                    predicate["left"] = random.randint(0, max(total_values // 2, total_values - 11))
                    predicate["right"] = min(predicate["left"] + random.randint(5, 20), total_values)
                else:
                    predicate["left"] = 0
                    predicate["right"] = total_values

        for i in range(num_samples // num_groups):
            params_for_sample = []
            for predicate in predicates:
                data_type = predicate["data_type"]
                preprocess_type = predicate["preprocess_type"]
                if data_type == "int" and preprocess_type == "embedding":
                    params_for_sample.append(sample_int(predicate["left"], predicate["right"]))
                elif data_type == "float" and preprocess_type == "std_normalization":
                    params_for_sample.append(sample_float(predicate["left"], predicate["right"]))
                elif data_type == "text" and preprocess_type == "embedding":
                    params_for_sample.append(sample_text(predicate["distinct_values"], predicate["left"], predicate["right"]))
            param_combinations[f"parameter {i + 1}"] = params_for_sample
        
        output_path = os.path.join(output_dir, f"group_{group + 1}.json")
        with open(output_path, 'w') as f:
            json.dump(param_combinations, f, indent=4)
        
        print(f"Saved group {group + 1} parameters to {output_path}")


def generate_parameters_for_all(data_directory, output_dir="./parameters_1500"):
    for subdir, _, _ in os.walk(data_directory):
        meta_data_path = os.path.join(subdir, "meta_data.json")

        subdir_name = os.path.basename(subdir)
        sub_output_dir = os.path.join(output_dir, subdir_name)
        os.makedirs(sub_output_dir, exist_ok=True)

        if os.path.isfile(meta_data_path):
            print(f"Processing parameters for query at {subdir}")
            generate_parameters(meta_data_path, output_dir=sub_output_dir)


if __name__ == "__main__":
    meta_data_path = '../training_data/DSB_1500/'

    generate_parameters_for_all(meta_data_path)