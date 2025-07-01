import hashlib
import itertools
import json
import os
import random
import re
import time
from multiprocessing import Pool

import psycopg2
import configure


def compute_hash(representation):
    m = hashlib.md5()
    m.update(str(representation).encode('utf-8'))
    return m.hexdigest()


def get_structural_representation(plan, depth=0):
    node_type = plan['Node Type']

    if 'Plans' not in plan:
        # 叶子节点
        table_name = plan.get('Relation Name', 'unknown')
        return [(node_type, table_name, depth)]
    else:
        # 内部节点
        sub_structure = [item for subplan in plan['Plans'] for item in
                         get_structural_representation(subplan, depth + 1)]
        return [(node_type, depth)] + sub_structure


def connect_to_pg():
    connection = psycopg2.connect(
        dbname=configure.dbname,
        user=configure.user,
        password=configure.password,
        host=configure.host,
        port=configure.port
    )
    return connection


def fetch_execution_plan(connection, template, parameters):
    cursor = connection.cursor()

    cursor.execute("SET max_parallel_workers_per_gather TO 0;")

    query = cursor.mogrify(template, parameters).decode()
    cursor.execute(f"EXPLAIN (FORMAT JSON) {query}")
    plan = cursor.fetchone()
    cursor.close()
    return plan[0][0]


def extract_tables_from_node(node):
    """
    Recursively extract table aliases from a plan node.
    """
    tables = []

    if "Relation Name" in node:
        tables.append(node["Alias"])

    if "Plans" in node:
        for child in node["Plans"]:
            tables.extend(extract_tables_from_node(child))

    return tables

def generate_join_order_hints(plan, k):
    if isinstance(plan, str):
        plan = json.loads(plan)

    tables = extract_tables_from_node(plan["Plan"])

    table_permutations = itertools.permutations(tables)

    print("----")

    # Sample k permutations from the generator
    sampled_permutations = list(itertools.islice(table_permutations, k))

    hints = []
    for perm in sampled_permutations:
        hint = "/*+ Leading(" + " ".join(perm) + ") */"
        hints.append(hint)

    return hints


def sample_join_order(graph, cardinality=None):
    if not cardinality:
        selection_probability = {k: 1 for k in graph.keys()}
    else:
        total_cardinality = sum(cardinality.values())
        selection_probability = {k: v / total_cardinality for k, v in cardinality.items()}

    visited = set()
    start_node = random.choices(list(graph.keys()), weights=[selection_probability[k] for k in graph.keys()], k=1)[0]
    join_order = [start_node]
    visited.add(start_node)

    while len(join_order) < len(graph):
        current_neighbors = [node for node in graph[join_order[-1]] if node not in visited]

        if not current_neighbors:
            all_neighbors = set()
            for visited_node in join_order:
                all_neighbors.update([node for node in graph[visited_node] if node not in visited])

            if not all_neighbors:
                break

            next_node = \
                random.choices(list(all_neighbors), weights=[selection_probability[k] for k in all_neighbors], k=1)[0]
        else:
            next_node = \
                random.choices(current_neighbors, weights=[selection_probability[k] for k in current_neighbors], k=1)[0]

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


def get_cardinality_from_plan(plan):
    cardinality = {}

    if "Plans" in plan:
        for sub_plan in plan["Plans"]:
            cardinality.update(get_cardinality_from_plan(sub_plan))
    else:
        if "Node Type" in plan and "Scan" in plan["Node Type"]:
            alias = plan["Alias"]
            rows = plan["Plan Rows"]
            cardinality[alias] = rows
    return cardinality


def generate_plans_by_join_order(connection, template, params, plan, k):
    plans = []
    join_graph = extract_join_graph_from_sql(template)
    #cardinality = get_cardinality_from_plan(plan)
    # for i in range(k):
    #     #join_order = sample_join_order(join_graph, cardinality)
    #     join_order = sample_join_order(join_graph)
    #     hint = "/*+ Leading(" + " ".join(join_order) + ") */"
    #     plan_with_hint = fetch_execution_plan(connection, hint + " " + template, params)
    #     plans.append(plan_with_hint)

    tables = join_graph.keys()

    table_permutations = itertools.permutations(tables)

    # Sample k permutations from the generator
    sampled_permutations = list(itertools.islice(table_permutations, k))
    for join_order in sampled_permutations:
        #join_order = sample_join_order(join_graph, cardinality)
        hint = "/*+ Leading(" + " ".join(join_order) + ") */"
        plan_with_hint = fetch_execution_plan(connection, hint + " " + template, params)
        plans.append(plan_with_hint)
    return plans


def generate_plans(meta_data_path, parameter_path, k=50):
    with open(meta_data_path, 'r') as f:
        meta_data = json.load(f)
    with open(parameter_path, 'r') as f:
        parameters_list = json.load(f)

    connection = connect_to_pg()
    plans = {}
    seen_hashes = set()
    idx = 0
    for params in enumerate(parameters_list.values()):
        plan = fetch_execution_plan(connection, meta_data['template'], params[1])

        representation = get_structural_representation(plan['Plan'])
        hash_val = compute_hash(representation)
        if hash_val in seen_hashes:
            continue
        seen_hashes.add(hash_val)
        idx += 1
        plans[f"plan {idx}"] = plan

        altered_plans = generate_plans_by_join_order(connection, meta_data['template'], params[1], plan, k)
        for plan in altered_plans:
            representation = get_structural_representation(plan['Plan'])
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

        with open(os.path.join(subdir, "all_plans_by_hybrid_naive_random.json"), 'w') as f:
            json.dump(plans, f, indent=4)

def save_execution_plans_for_all_multiprocess(data_directory, num_processes=8):
    dirs = [x[0] for x in os.walk(data_directory) if 'a' in os.path.basename(x[0])]

    with Pool(num_processes) as pool:
        pool.map(process_directory, dirs)


if __name__ == "__main__":
    meta_data_path = '../training_data/JOB/'
    # meta_data_path = '../training_data/example_one/'
    start_time = time.time()
    save_execution_plans_for_all_multiprocess(meta_data_path)
    end_time = time.time()
    print(end_time-start_time)
