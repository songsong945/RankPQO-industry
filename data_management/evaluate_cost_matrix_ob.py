import os
import random
import time
import re
import json
from multiprocessing import Pool

# 修改：不再使用 psycopg2，改为 pymysql 连接 OceanBase
import pymysql

# 配置 OceanBase 连接参数（请根据实际环境修改）
OB_HOST = "127.0.0.1"
OB_PORT = 2881
OB_USER = "root"
OB_PASSWORD = "Y9dU0FkKYxjwrH1MCBR2"
OB_DATABASE = "JOB"

# 正则表达式用于提取执行时间（本例中暂不使用，可保留备用）
time_regex = re.compile(r'Execution Time: ([\d\.]+) ms')

# ----------------- 修改连接函数 -----------------
# 原 PostgreSQL 连接函数：connect_to_pg()
# 修改后：使用 pymysql 建立 OceanBase 连接
def connect_to_ob():
    connection = pymysql.connect(
        host=OB_HOST,
        port=OB_PORT,
        user=OB_USER,
        password=OB_PASSWORD,
        database=OB_DATABASE,
        charset="utf8mb4"
    )
    return connection

# ----------------- 修改提示生成函数 -----------------
# generate_hint_from_plan：原来基于 PostgreSQL 的 plan['Plan']、"Node Type"、"Relation Name"、"Alias"
# 修改后：使用 OceanBase 的 EXPLAIN 输出，根节点即计划，节点信息在 "OPERATOR" 和 "NAME"
def generate_hint_from_plan(plan):
    # 修改：OceanBase 的执行计划直接为 plan，而不嵌套在 plan['Plan'] 中
    node = plan  # 原来：node = plan['Plan']
    hints = []

    def traverse_node(node):
        # 修改：使用 OceanBase 的 "OPERATOR" 替代 PostgreSQL 的 "Node Type"
        node_type = node.get('OPERATOR', '')
        rels = []  # Flattened list of relation names/aliases
        leading = []  # Hierarchical structure for LEADING hint

        # 对于 OceanBase，去除多余空格
        node_type = node_type.replace(' ', '')
        # 如果有 "NAME" 则认为是扫描操作（类似于 Relation Name）
        if 'NAME' in node and node.get('NAME', '') != "":
            # 使用 NAME 作为 relation（如果有 alias，可在此处理）
            relation = node.get('NAME')
            # 针对扫描操作，可根据 OPERATOR 判断，如 "TABLEFULLSCAN" 或 "TABLERANGESCAN"
            if "Scan" in node_type or "SCAN" in node.get('OPERATOR', '').upper():
                hint = node_type + '(' + relation + ')'
                hints.append(hint)
            return [relation], relation
        else:
            # 如果存在子节点，遍历所有以 "CHILD_" 开头的键
            children = [node[key] for key in node if key.startswith("CHILD_")]
            for child in children:
                a, b = traverse_node(child)
                rels.extend(a)
                if b:
                    leading.append(b)
            # 对于连接操作，如 HashJoin、MergeJoin、NestLoopJoin（可以根据实际情况调整）
            if any(x in node_type for x in ['HashJoin', 'MergeJoin', 'NestLoopJoin']):
                join_hint = node_type + '(' + ' '.join(rels) + ')'
                hints.append(join_hint)
            return rels, leading

    _, leading_hierarchy = traverse_node(node)

    def pair_hierarchy(hierarchy):
        if isinstance(hierarchy, str):
            return hierarchy
        elif len(hierarchy) == 1:
            return pair_hierarchy(hierarchy[0])
        else:
            hierarchy = [pair_hierarchy(item) for item in hierarchy]
        return hierarchy

    leading_hierarchy = pair_hierarchy(leading_hierarchy)
    # 修改：生成 OceanBase 风格的 LEADING hint（逗号分隔）
    leading_hierarchy = str(leading_hierarchy).replace("'", "").replace(',', '')
    leading = 'LEADING(' + leading_hierarchy + ')'
    hints.append(leading)
    query_hint = "\n ".join(hints)
    return query_hint

def extract_est_time_us(result_rows):
    """
    从 EXPLAIN FORMAT=JSON 的 raw rows 提取 EST.TIME(us) 的值（单位：秒）
    """
    for row in result_rows:
        line = row[0].strip()
        if line.startswith('"EST.TIME(us)"') or '"EST.TIME(us)"' in line:
            try:
                value_str = line.split(":")[1].strip().rstrip(",")
                est_time_us = int(value_str)
                return est_time_us / 1e6  # 转换为秒
            except Exception as e:
                print(f"[Error parsing line]: {line} — {e}")
                return None
    return None


def fetch_ob_cost(connection, query_with_hint, parameters):

    cursor = connection.cursor()
    try:
        cursor.execute(query_with_hint, parameters)
        result = cursor.fetchall()
        if not result or not result[0]:
            raise ValueError("Empty EXPLAIN result")

        latency = extract_est_time_us(result)

        return latency

    except json.JSONDecodeError as e:
        print(f"[EXPLAIN PARSE ERROR] {e}\n")
        raise
    finally:
        cursor.close()




def fetch_actual_latency(connection, query_with_hint, parameters):
    # 创建游标
    cursor = connection.cursor()
    # 记录开始时间
    start_time = time.perf_counter()
    try:
        # 执行参数化查询
        cursor.execute(query_with_hint, parameters)
    except pymysql.err.OperationalError as e:
        # 检查错误代码是否为 4012（超时错误）
        if e.args[0] == 4012:
            # 超时错误，关闭游标后直接返回 50 分钟（3000 秒）的延迟
            cursor.close()
            return 50 * 60  # 50 分钟转换为秒
        else:
            # 如果不是超时错误，则关闭游标并重新抛出异常
            cursor.close()
            raise
    # 记录结束时间（此时 execute 已完成执行）
    end_time = time.perf_counter()
    # 关闭游标
    cursor.close()
    # 计算查询延迟
    latency = end_time - start_time
    return latency


def fetch_actual_latency_timeout(connection, query_with_hint, parameters, timeout):
    cursor = connection.cursor()
    # 设置 statement_timeout 命令在 OceanBase 中可能无效，直接忽略或使用 OceanBase 提供的超时设置
    # cursor.execute(f"SET statement_timeout TO {int(timeout * 1000)};")
    start_time = time.perf_counter()
    try:
        cursor.execute(query_with_hint, parameters)
    except Exception as e:
        connection.rollback()
        return timeout * 10
    end_time = time.perf_counter()
    cursor.close()
    latency = end_time - start_time
    return latency

# ----------------- 修改评估函数 -----------------
# 修改 get_counter_example、evaluate_plans_for_parameters 等函数中连接部分
# 将连接函数从 connect_to_pg() 修改为 connect_to_ob()
def get_counter_example(meta_data, plans, parameters_data):
    template = meta_data["template"]
    results = {}
    sampled_plan_keys = list(plans.keys())
    sampled_param_keys = list(parameters_data.keys())
    i = 0
    for param_key in sampled_param_keys:
        param_values = parameters_data[param_key]
        results[param_key] = {}
        for plan_key in sampled_plan_keys:
            connection = connect_to_ob()  # 修改：使用 OceanBase 连接
            plan = plans[plan_key]
            plan_hint = generate_hint_from_plan(plan)
            # 修改 EXPLAIN ANALYZE 为 OceanBase 的 EXPLAIN FORMAT=JSON （或其他适用语法）
            query_with_hint = f"/*+ {plan_hint} */ EXPLAIN FORMAT=JSON " + template
            # 使用 Python 格式化填入参数（假设模板中使用 %s 占位符）
            query_with_hint = query_with_hint % tuple(param_values)
            st = time.time()
            cost = fetch_actual_latency(connection, query_with_hint, param_values)
            ed = time.time()
            results[param_key][plan_key] = ed - st
            connection.close()
            i += 1
    return results

def evaluate_plans_for_parameters(connection, meta_data, plans, parameters_data):
    template = meta_data["template"]
    results = {}
    sampled_plan_keys = random.sample(list(plans.keys()), min(20, len(plans.keys())))
    sampled_param_keys = list(parameters_data.keys())[:4]
    i = 0
    for param_key in sampled_param_keys:
        param_values = parameters_data[param_key]
        results[param_key] = {}
        for plan_key in sampled_plan_keys:
            plan = plans[plan_key]
            plan_hint = generate_hint_from_plan(plan)
            query_with_hint = f"/*+ {plan_hint} */ EXPLAIN FORMAT=JSON " + template
            query_with_hint = query_with_hint % tuple(param_values)
            cost = fetch_actual_latency(connection, query_with_hint, param_values)
            results[param_key][plan_key] = cost
            i += 1
    return results

def evaluate_all(data_directory):
    connection = connect_to_ob()  # 修改：使用 OceanBase 连接
    for subdir, _, files in os.walk(data_directory):
        if "meta_data.json" in files and "plan.json" in files and "parameter.json" in files:
            with open(os.path.join(subdir, "meta_data.json"), 'r') as f_meta:
                meta_data = json.load(f_meta)
            with open(os.path.join(subdir, "plan.json"), 'r') as f_plans:
                plans = json.load(f_plans)
            with open(os.path.join(subdir, "parameter.json"), 'r') as f_params:
                parameters = json.load(f_params)
            print(f"Processing {subdir}...")
            costs = evaluate_plans_for_parameters(connection, meta_data, plans, parameters)
            with open(os.path.join(subdir, "cost_test.json"), 'w') as f_costs:
                json.dump(costs, f_costs, indent=4)
    connection.close()

def evaluate_directory(subdir):
    connection = connect_to_ob()  # 修改：使用 OceanBase 连接
    with open(os.path.join(subdir, "meta_data.json"), 'r') as f_meta:
        meta_data = json.load(f_meta)
    with open(os.path.join(subdir, "hybrid_plans.json"), 'r') as f_plans:
        plans = json.load(f_plans)
    with open(os.path.join(subdir, "parameters.json"), 'r') as f_params:
        parameters = json.load(f_params)
    print(f"Processing {subdir}...")
    costs = evaluate_plans_for_parameters(connection, meta_data, plans, parameters)
    with open(os.path.join(subdir, "cost_matrix_2.json"), 'w') as f_costs:
        json.dump(costs, f_costs, indent=4)
    print(f"Finished {subdir}...")
    connection.close()

def evaluate_all_mutil_process(data_directory):
    directories_to_process = []
    for subdir, _, files in os.walk(data_directory):
        if "meta_data.json" in files:
            directories_to_process.append(subdir)

    # print(directories_to_process)
    with Pool(processes=12) as pool:
        pool.map(evaluate_directory, directories_to_process)

if __name__ == "__main__":
    meta_data_path = '../training_data/TPCDS/'
    start_time = time.time()
    evaluate_all_mutil_process(meta_data_path)
    end_time = time.time()
    print(end_time - start_time)
