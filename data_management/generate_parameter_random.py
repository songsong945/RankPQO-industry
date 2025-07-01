import logging
import random
import time

import numpy as np
import json
import os
import configure
import pymysql

random.seed(1234)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s')

def sample_int(min_val, max_val):
    return str(random.randint(int(float(min_val)), int(float(max_val))))


def sample_float(mean, variance):
    return str(round(np.random.normal(mean, np.sqrt(variance)), 2))

def sample_text(distinct_values):
    non_empty_values = [value for value in distinct_values if value is not None and value != ""]
    return random.choice(non_empty_values) if non_empty_values else None

def generate_parameters(meta_data_path, num_samples=200):
    with open(meta_data_path, 'r') as f:
        data = json.load(f)

    predicates = data["predicates"]
    template = data["template"]
    param_combinations = {}


    for i in range(num_samples):
        # j = 0
        # while True:
        #     j += 1
        params_for_sample = []
        for predicate in predicates:
            data_type = predicate["data_type"]
            preprocess_type = predicate["preprocess_type"]
            if data_type == "int" and preprocess_type == "embedding":
                params_for_sample.append(sample_int(predicate["min"], predicate["max"]))
            elif data_type == "float" and preprocess_type == "std_normalization":
                params_for_sample.append(sample_float(predicate["mean"], predicate["variance"]))
            elif data_type == "text" and preprocess_type == "embedding":
                params_for_sample.append(sample_text(predicate["distinct_values"]))

            # print(params_for_sample)

            # latency = fetch_actual_latency(connection, template, params_for_sample)
            # # logging.info(f"Executing time: {latency}s")

            # if latency > 7:
            #     break

        # logging.info(f"Sampling {j} times for {i}th vector")
        # logging.info(f"Executing time: {latency}s")

        param_combinations[f"parameter {i + 1}"] = params_for_sample

    return param_combinations

def enumerate_parameters_for_meta_data(meta_data_path, output_path, num_samples=1000):
    param_combinations = generate_parameters(meta_data_path, num_samples)
    with open(output_path, 'w') as f:
        json.dump(param_combinations, f, indent=4)


def generate_parameters_for_all(data_directory):
    # 遍历 data_directory 下的所有条目
    for subfolder in os.listdir(data_directory):
        subdir = os.path.join(data_directory, subfolder)
        # 确保该条目是文件夹
        if os.path.isdir(subdir):
            meta_data_path = os.path.join(subdir, "meta_data.json")
            parameter_output_path = os.path.join(subdir, "parameter_new.json")
            
            # 检查 meta_data.json 是否存在
            if os.path.isfile(meta_data_path):
                print(f"Processing parameters for query at {subdir}")
                enumerate_parameters_for_meta_data(meta_data_path, parameter_output_path)


if __name__ == "__main__":
    meta_data_path = '../training_data/DSB_1500/'
    generate_parameters_for_all(meta_data_path)
