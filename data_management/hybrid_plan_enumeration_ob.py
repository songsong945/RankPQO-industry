import hashlib
import json
import os
import random
import re
import time
from multiprocessing import Pool

import pymysql  # 使用 pymysql 连接 OceanBase

# 显式定义 OceanBase 的连接参数
# 请根据实际环境修改以下参数
OB_HOST = "127.0.0.1"
OB_PORT = 2881
OB_USER = "root"
OB_PASSWORD = 'Y9dU0FkKYxjwrH1MCBR2'
OB_DATABASE = "DSB"

def compute_hash(representation):
    m = hashlib.md5()
    m.update(str(representation).encode('utf-8'))
    return m.hexdigest()

# 修改 get_structural_representation：原来针对 PostgreSQL 的 "Node Type" 和 "Plans"，
# 现在改为使用 OceanBase 的 "OPERATOR" 作为节点类型，且子节点使用 "CHILD_1", "CHILD_2", ... 代替。
def get_structural_representation(plan, depth=0):
    # 原注释保留：获取节点类型，原代码使用 plan['Node Type']
    # 修改：使用 OceanBase 的 "OPERATOR"
    node_type = plan.get('OPERATOR', 'unknown')  # 修改：由 "Node Type" 改为 "OPERATOR"
    
    # 查找所有子节点，OceanBase 的子节点以 "CHILD_" 开头
    children = [plan[key] for key in plan if key.startswith("CHILD_")]
    
    if not children:
        # 叶子节点：获取表名，原代码使用 plan.get('Relation Name', 'unknown')
        # 修改：OceanBase 输出中通常在 "NAME" 中记录相关表名（有时可能为空字符串）
        table_name = plan.get('NAME', 'unknown')
        return [(node_type, table_name, depth)]
    else:
        sub_structure = []
        for child in children:
            sub_structure.extend(get_structural_representation(child, depth + 1))
        return [(node_type, depth)] + sub_structure

# 修改后的 OceanBase 连接函数，不再依赖 configure 模块
def connect_to_ob():
    connection = pymysql.connect(
        host=OB_HOST,              # 修改：显式指定 OceanBase 主机地址
        port=OB_PORT,              # 修改：显式指定 OceanBase 端口，例如10400
        user=OB_USER,              # 修改：显式指定用户名，例如 "root@sys"
        password=OB_PASSWORD,      # 修改：显式指定密码
        database=OB_DATABASE       # 修改：显式指定数据库名称，例如 "JOB"
        # charset="utf8mb4",
        # local_infile=True
    )
    return connection

# fetch_execution_plan 保留原有注释，并使用参数化查询
def fetch_execution_plan(connection, template, parameters):
    cursor = connection.cursor()
    # OceanBase不需要设置 PostgreSQL 特有的参数，此处去掉 SET 命令

    # 修改为 OceanBase 支持的 EXPLAIN 语法，格式为 "EXPLAIN FORMAT=JSON"
    full_query = "EXPLAIN FORMAT=JSON " + template  # 原注释保留
    # 使用参数化查询，自动处理参数转义（避免SQL注入及语法问题）
    cursor.execute(full_query, parameters)
    
    results = cursor.fetchall()
    cursor.close()
    
    # 拼接所有行（假设每行是一个字符串部分）
    full_result = "".join(row[0] for row in results)
    # 解析完整的 JSON 字符串
    plan = json.loads(full_result)
    # 原注释：假设 OceanBase 返回的 JSON 结构与 PostgreSQL 类似（计划树位于 plan[0][0]）
    # 修改：OceanBase 的 EXPLAIN 输出直接返回整个计划，不再嵌套在额外的列表或字典中
    return plan  # 修改：直接返回解析后的 plan

def sample_join_order(graph, cardinality=None):
    # 确保 graph 不是空字典
    if not graph:
        raise ValueError("Graph is empty. Please check the input graph.")

    # 初始化 selection_probability，处理 cardinality 或默认值
    if not cardinality:
        selection_probability = {k: 1 for k in graph.keys()}
    else:
        selection_probability = {k: cardinality.get(k, 1) for k in graph.keys()}

        # 归一化处理 selection_probability，使得概率之和为 1
        total = sum(selection_probability.values())
        if total > 0:
            selection_probability = {k: v / total for k, v in selection_probability.items()}
        else:
            selection_probability = {k: 1 for k in graph.keys()}  # 如果所有值为 0，设置为 1

    visited = set()

    # 计算每个节点的权重
    start_weights = [selection_probability.get(k, 1) for k in graph.keys()]
    if sum(start_weights) == 0:
        start_weights = [1] * len(graph)  # 如果所有权重之和为 0，给每个节点权重 1
    
    # 检查 start_weights 是否正确
    if len(start_weights) != len(graph):
        raise ValueError("Length of start_weights does not match number of graph keys.")
    
    # 从 graph 中随机选择一个节点作为起始节点
    start_node = random.choices(list(graph.keys()), weights=start_weights, k=1)[0]
    join_order = [start_node]
    visited.add(start_node)

    # 遍历 graph，直到遍历完所有节点
    while len(join_order) < len(graph):
        current_neighbors = [node for node in graph[join_order[-1]] if node not in visited]

        # 如果当前节点没有未访问的邻居，选择所有未访问的邻居
        if not current_neighbors:
            all_neighbors = set()
            for visited_node in join_order:
                all_neighbors.update([node for node in graph[visited_node] if node not in visited])

            if not all_neighbors:
                break  # 如果没有未访问的邻居，退出循环

            # 计算所有未访问邻居的权重
            neighbor_weights = [selection_probability.get(k, 1) for k in all_neighbors]
            if sum(neighbor_weights) == 0:
                neighbor_weights = [1] * len(all_neighbors)

            next_node = random.choices(list(all_neighbors), weights=neighbor_weights, k=1)[0]
        else:
            # 对当前节点的未访问邻居进行权重计算
            neighbor_weights = [selection_probability.get(k, 1) for k in current_neighbors]
            if sum(neighbor_weights) == 0:
                neighbor_weights = [1] * len(current_neighbors)
            next_node = random.choices(current_neighbors, weights=neighbor_weights, k=1)[0]

        # 将选择的邻居节点添加到 join_order 中，并标记为已访问
        join_order.append(next_node)
        visited.add(next_node)

    return join_order

def extract_join_graph_from_sql(sql):
    pattern = r"(\w+)\.\w+\s+=\s+(\w+)\.\w+"
    matches = re.findall(pattern, sql)
    graph = {}
    for table1, table2 in matches:
        if table1 not in graph:
            graph[table1] = []
        if table2 not in graph:
            graph[table2] = []
        if table2 not in graph[table1]:
            graph[table1].append(table2)
        if table1 not in graph[table2]:
            graph[table2].append(table1)
    return graph

def extract_join_graph_from_sql_dsb(sql):
    pattern = r"(\w+)\s*=\s*(\w+)"
    matches = re.findall(pattern, sql)
    graph = {}
    for table1, table2 in matches:
        if table1 not in graph:
            graph[table1] = []
        if table2 not in graph:
            graph[table2] = []
        if table2 not in graph[table1]:
            graph[table1].append(table2)
        if table1 not in graph[table2]:
            graph[table2].append(table1)
    return graph

# 修改 get_cardinality_from_plan：原来针对 PostgreSQL 输出使用 "Plans"、"Alias"、"Plan Rows"
# 现在改为使用 OceanBase 的子节点键 "CHILD_"，以及 "EST.ROWS"（作为扫描节点的基数），"NAME" 作为表名
def get_cardinality_from_plan(plan):
    cardinality = {}
    # 修改：检测 OceanBase 的子节点，以 "CHILD_" 开头
    children = [plan[key] for key in plan if key.startswith("CHILD_")]
    if children:
        for child in children:
            cardinality.update(get_cardinality_from_plan(child))
    else:
        # 判断是否为扫描操作：对于 OceanBase，扫描操作的 OPERATOR 一般包含 "SCAN"
        # 修改：使用 "OPERATOR" 而非 "Node Type"，且基数取 "EST.ROWS"
        if "SCAN" in plan.get("OPERATOR", ""):
            # 使用 "NAME" 作为标识（类似于表名）
            alias = plan.get("NAME", "unknown")
            rows = plan.get("EST.ROWS", 0)
            cardinality[alias] = rows
    return cardinality

# 修改 generate_plans_by_join_order：插入 Hint 后需将 Hint 插入到 SELECT 关键字后面，
# 并修改 hint 格式为 OceanBase 的要求（使用逗号分隔）。
def generate_plans_by_join_order(connection, template, params, plan, k):
    plans = []
    #join_graph = extract_join_graph_from_sql(template)
    join_graph = extract_join_graph_from_sql_dsb(template)
    cardinality = get_cardinality_from_plan(plan)
    for i in range(k):
        join_order = sample_join_order(join_graph, cardinality)
        # 修改 hint 格式为 OceanBase 的要求，使用逗号分隔
        hint = "/*+ LEADING(" + ", ".join(join_order) + ") */"  # 原注释保留
        # 将 hint 插入到 SELECT 关键字后面（不论大小写）
        new_template = re.sub(r"^(select)", r"\1 " + hint, template, flags=re.IGNORECASE)
        # print(new_template)
        plan_with_hint = fetch_execution_plan(connection, new_template, params)
        plans.append(plan_with_hint)
    # print("generate_plans_by_join_order -> result length")
    # print(len(plans))
    return plans

def generate_plans(meta_data_path, parameter_path, k1=10, k2=50):
    with open(meta_data_path, 'r') as f:
        meta_data = json.load(f)
    with open(parameter_path, 'r') as f:
        parameters_list = json.load(f)
    connection = connect_to_ob()  # 使用 OceanBase 连接
    plans = {}
    seen_hashes = set()
    idx = 0
    i = 0
    for params in enumerate(parameters_list.values()):
        i += 1
        if i > k1:
            break
        # 修改调用 fetch_execution_plan 后直接使用 plan（OceanBase格式），而不再取 plan['Plan']
        plan = fetch_execution_plan(connection, meta_data['template'], params[1])
        # 修改调用 get_structural_representation：不再依赖 PostgreSQL 的 "Node Type"，已在函数中修改为使用 "OPERATOR"
        representation = get_structural_representation(plan)
        hash_val = compute_hash(representation)
        if hash_val in seen_hashes:
            continue
        seen_hashes.add(hash_val)
        idx += 1
        plans[f"plan {idx}"] = plan
        altered_plans = generate_plans_by_join_order(connection, meta_data['template'], params[1], plan, k2)
        for plan in altered_plans:
            representation = get_structural_representation(plan)
            hash_val = compute_hash(representation)
            if hash_val in seen_hashes:
                continue
            seen_hashes.add(hash_val)
            idx += 1
            plans[f"plan {idx}"] = plan
    connection.close()
    return plans

def process_directory(subdir):
    meta_data_path = os.path.join(subdir, "meta_data.json")
    parameter_path = os.path.join(subdir, "parameter_new.json")
    if os.path.isfile(meta_data_path) and os.path.isfile(parameter_path):
        print(f"Processing: {meta_data_path}")
        plans = generate_plans(meta_data_path, parameter_path)
        print(len(plans))
        with open(os.path.join(subdir, "hybrid_plans.json"), 'w') as f:
            json.dump(plans, f, indent=4)

def save_execution_plans_for_all_multiprocess(data_directory, num_processes=6):
    dirs = [x[0] for x in os.walk(data_directory)]
    print(dirs)
    # for dir in dirs:
    #     process_directory(dir)
    with Pool(num_processes) as pool:
        pool.map(process_directory, dirs)

if __name__ == "__main__":
    # 指定数据目录，子目录中应包含 meta_data.json 和 parameter_new.json 文件
    meta_data_path = '../training_data/DSB_150/'
    # start_time = time.time()
    # save_execution_plans_for_all_multiprocess(meta_data_path)
    # end_time = time.time()
    # print(end_time - start_time)
    # dirs = [x[0] for x in os.walk(meta_data_path)]
    dirs = ['../training_data/DSB_150/dsb_85', '../training_data/DSB_150/dsb_106', '../training_data/DSB_150/dsb_73', '../training_data/DSB_150/dsb_72', '../training_data/DSB_150/dsb_54', '../training_data/DSB_150/dsb_29', '../training_data/DSB_150/dsb_59', '../training_data/DSB_150/dsb_76', '../training_data/DSB_150/dsb_12', '../training_data/DSB_150/dsb_142', '../training_data/DSB_150/dsb_134', '../training_data/DSB_150/dsb_53', '../training_data/DSB_150/dsb_7', '../training_data/DSB_150/dsb_135', '../training_data/DSB_150/dsb_148', '../training_data/DSB_150/dsb_66', '../training_data/DSB_150/dsb_32', '../training_data/DSB_150/dsb_25', '../training_data/DSB_150/dsb_143', '../training_data/DSB_150/dsb_44', '../training_data/DSB_150/dsb_69', '../training_data/DSB_150/dsb_71', '../training_data/DSB_150/dsb_37', '../training_data/DSB_150/dsb_81', '../training_data/DSB_150/dsb_77']
    print(dirs)
    for dir in dirs:
        process_directory(dir)
