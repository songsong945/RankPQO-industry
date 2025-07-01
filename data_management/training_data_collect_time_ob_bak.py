import os
import random
import json
import time
from multiprocessing import Pool

from evaluate_cost_matrix_ob import generate_hint_from_plan, fetch_actual_latency, connect_to_ob, fetch_actual_latency_timeout


def _plan_path(base):
    return os.path.join(base, "hybrid_plans.json")

def _param_path(base):
    return os.path.join(base, "parameter_new.json")


def collect_training_pair_k(candidate_plan, k):
    assert len(candidate_plan) >= 2
    X = set()

    for i in range(k):
        sampled_indices = random.sample(range(len(candidate_plan)), 2)
        s1 = candidate_plan[sampled_indices[0]]
        s2 = candidate_plan[sampled_indices[1]]
        X.add(s1)
        X.add(s2)
    return X


def collet_training_data_k(training_data_file, template_id, k):
    path = os.path.join(training_data_file, template_id)
    random.seed(42)

    data = {}

    with open(_plan_path(path), 'r') as f:
        plans = json.load(f)

    with open(_param_path(path), 'r') as f:
        param = json.load(f)

    param_keys = random.sample(list(param.keys()), min(200, len(param.keys())))
    # param_keys = param_keys[:160]
    param_num = len(param_keys)

    for param_key in param_keys:
        candidate_plan = list(plans.keys())
        data[param_key] = list(collect_training_pair_k(candidate_plan, int(k / param_num) + 1))

    with open(os.path.join(path, f"training_data_{k}.json"), 'w') as file:
        json.dump(data, file, indent=4)


def sample(training_data):
    all_folders = []
    for subdir, _, files in os.walk(training_data):
        print(subdir)
        if ("meta_data.json" in files and "hybrid_plans.json" in files
                and "parameter_new.json" in files):
            all_folders.append(os.path.basename(subdir))

    for folder in all_folders:
        for k in [1000, 2000, 3000, 4000, 5000]:
            collet_training_data_k(training_data, folder, k)


def evaluate_plans_for_parameters(connection, meta_data, plans, parameters_data, training_data):
    template = meta_data["template"]   # 获取 SQL 模板
    results = {}                       # 初始化结果字典
    # sampled_plan_keys = random.sample(list(plans.keys()), min(20, len(plans.keys())))  # 随机抽样 20 个计划
    # sampled_param_keys = list(parameters_data.keys())[:4]  # 取前 4 个参数向量
    sampled_param_keys = training_data.keys()

    for param_key in sampled_param_keys:
        param_values = parameters_data[param_key]  # 当前参数向量
        results[param_key] = {}
        sampled_plan_keys = training_data[param_key]
        for plan_key in sampled_plan_keys:
            plan = plans[plan_key]       # 选择执行计划
            plan_hint = generate_hint_from_plan(plan)  # 生成 hint
            # 修改：使用 OceanBase 的 EXPLAIN FORMAT=JSON 语法
            # query_with_hint = "EXPLAIN FORMAT=JSON " + template.replace("SELECT", f"SELECT /*+ {plan_hint} */", 1)
            query_with_hint = template.replace("SELECT", f"SELECT /*+ {plan_hint} */", 1)
            # query_with_hint = query_with_hint % tuple(param_values)  # 填入参数
            cost = fetch_actual_latency(connection, query_with_hint, param_values)  # 获取延迟
            print(cost)
            results[param_key][plan_key] = cost  # 保存结果
    return results                   # 返回评估结果



def evaluate_directory(subdir):
    connection = connect_to_ob()
    # print(connection.get_query_timeout() )

    k = 1000

    with open(os.path.join(subdir, "meta_data.json"), 'r') as f_meta:
        meta_data = json.load(f_meta)

    with open(os.path.join(subdir, "hybrid_plans.json"), 'r') as f_plans:
        plans = json.load(f_plans)

    with open(os.path.join(subdir, "parameter_new.json"), 'r') as f_params:
        parameters = json.load(f_params)

    with open(os.path.join(subdir, f"training_data_{k}.json"), 'r') as f_data:
        training_data = json.load(f_data)

    print(f"Processing {subdir}...")

    costs = evaluate_plans_for_parameters(connection, meta_data, plans, parameters, training_data)

    with open(os.path.join(subdir, f"cost_matrix_1k.json"), 'w') as f_costs:
        json.dump(costs, f_costs, indent=4)

    print(f"Finished {subdir}...")

    connection.close()

def evaluate_all_mutil_process(data_directory):
    directories_to_process = []
    k = 3000
    for subdir, _, files in os.walk(data_directory):
        if "meta_data.json" in files:
            directories_to_process.append(subdir)

    with Pool(processes=12) as pool:
        pool.map(evaluate_directory, directories_to_process)

if __name__ == "__main__":
    # meta_data_path = '../training_data/JOB_330/'
    # start_time = time.time()
    # evaluate_all_mutil_process(meta_data_path)
    # end_time = time.time()
    # print(end_time-start_time)
    # sample('../training_data/JOB_330/')
    evaluate_directory("../training_data/JOB_330/job_1")
