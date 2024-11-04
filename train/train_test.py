import argparse
import json
import os
import random
import sys
import time

import torch

sys.path.append('..')

from model.feature import FeatureGenerator
from model.model import RankPQOModel

k1 = [7, 9, 11, 13, 15]
k2 = [15, 20, 25, 30, 32]
k3 = [7, 9, 11, 13, 15, 16, 17, 18, 19, 20]  # 1k - 10k
k_train = [7, 9, 11, 13, 15, 20, 25, 30, 32]
k_test = [3, 4, 5, 6, 7, 9, 11, 13, 15]  # 200, 400, 600, 800, 1k, 2k, 3k, 4k, 5k
k4 = [2, 5, 7, 15, 20, 25]  # 100, 500, 1k, 5k, 10k, 15k

random.seed(42)


def _param_path(base):
    return os.path.join(base, "parameter_new.json")


def _cost_path(base):
    return os.path.join(base, "latency_matrix_pg_200.json")


def _meta_path(base):
    return os.path.join(base, "meta_data.json")


def _plan_path(base):
    return os.path.join(base, "all_plans_by_hybrid_new.json")


def get_param_info(meta):
    params, preprocess_info = [], []

    for data in meta["predicates"]:
        param_data = {}
        preprocess_info_data = {}

        if data["data_type"] in ["int", "float", "text"]:
            param_data["data_type"] = data["data_type"]

            if data["data_type"] == "int" and "min" in data and "max" in data:
                param_data["min"] = data["min"]
                param_data["max"] = data["max"]

            if "distinct_values" in data:
                param_data["distinct_values"] = data["distinct_values"]

            params.append(param_data)

        if data["preprocess_type"] in ["one_hot", "std_normalization", "embedding"]:
            preprocess_info_data["type"] = data["preprocess_type"]

            if data["preprocess_type"] == "one_hot" and "max_len" in data:
                preprocess_info_data["max_len"] = data["max_len"]

            if data["preprocess_type"] == "std_normalization" and "mean" in data and "variance" in data:
                preprocess_info_data["mean"] = data["mean"]
                preprocess_info_data["variance"] = data["variance"]

            if data["preprocess_type"] == "embedding" and "max_len" in data:
                preprocess_info_data["max_len"] = data["max_len"]

            preprocess_info.append(preprocess_info_data)

    return params, preprocess_info


def get_training_pair(candidate_plan, plan, param_key, cost):
    assert len(candidate_plan) >= 2
    X1, X2, Y1, Y2 = [], [], [], []

    i = 0
    while i < len(candidate_plan) - 1:
        s1 = candidate_plan[i]
        j = i + 1
        while j < len(candidate_plan):
            s2 = candidate_plan[j]
            X1.append(plan[s1])
            Y1.append(cost[param_key][s1])
            X2.append(plan[s2])
            Y2.append(cost[param_key][s2])
            j += 1
        i += 1
        # if i > k2[4]:
        #     break
    return X1, X2, Y1, Y2


def get_training_pair2(plan_combinations, plan, param_key, cost):
    X1, X2, Y1, Y2 = [], [], [], []

    for plan_pair in plan_combinations:
        plan1 = plan_pair[0]
        plan2 = plan_pair[1]

        X1.append(plan[plan1])
        Y1.append(cost[param_key][plan1])
        X2.append(plan[plan2])
        Y2.append(cost[param_key][plan2])

    return X1, X2, Y1, Y2


def load_training_data(training_data_file, template_id):
    path = os.path.join(training_data_file, template_id)
    Z, X1, X2, Y1, Y2, params, preprocess_info = [], [], [], [], [], [], []
    random.seed(42)

    with open(_param_path(path), 'r') as f:
        param = json.load(f)

    with open(_cost_path(path), 'r') as f:
        cost = json.load(f)

    with open(_meta_path(path), 'r') as f:
        meta = json.load(f)

    with open(_plan_path(path), 'r') as f:
        plan = json.load(f)

    keys_list = list(cost.keys())[:160]

    param_keys = random.sample(keys_list, min(50, len(keys_list)))

    for param_key in param_keys:
        param_values = param[param_key]
        candidate_plan = random.sample(list(cost[param_key].keys()), min(k4[4], len(cost[param_key].keys())))
        x1, x2, y1, y2 = get_training_pair(candidate_plan, plan, param_key, cost)
        Z += [list(param_values) for _ in range(len(x1))]
        X1 += x1
        X2 += x2
        Y1 += y1
        Y2 += y2

    params, preprocess_info = get_param_info(meta)

    return Z, X1, X2, Y1, Y2, params, preprocess_info


def generate_combinations(elements, sample_size):
    sampled = set()
    covered_elements = set()
    duplicate = False
    while len(sampled) < sample_size:
        if len(covered_elements) < len(elements):
            element1 = random.choice(list(set(elements) - covered_elements))
            element2 = random.choice(list(set(elements) - {element1}))
        else:
            duplicate = True
            sampled = list(sampled)
            element1, element2 = random.sample(elements, 2)

        combination = tuple(sorted((element1, element2)))
        if duplicate:
            sampled.append(combination)
        elif combination not in sampled:
            sampled.add(combination)
            covered_elements.update(combination)

    return list(sampled)

def write_combinations_to_file(combinations, file_path):
    with open(file_path, 'w') as file:
        for comb in combinations:
            line = f"{comb[0]}, {comb[1]}\n"
            file.write(line)


def read_combinations_from_file(file_path):
    combinations = []
    with open(file_path, 'r') as file:
        for line in file:
            elements = line.strip().split(', ')
            combinations.append(tuple(elements))
    return combinations


def process_combinations(file_path, elements, sample_size):
    if not os.path.exists(file_path):
        combinations = generate_combinations(elements, sample_size)
        write_combinations_to_file(combinations, file_path)
        return combinations
    else:
        return read_combinations_from_file(file_path)

# 确保文件夹存在
def ensure_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def load_training_test_data(training_data_file, template_id, k):
    path = os.path.join(training_data_file, template_id)
    Z, X1, X2, Y1, Y2, Z_t, X1_t, X2_t, Y1_t, Y2_t, params, preprocess_info = [], [], [], [], [], [], [], [], [], [], [], []
    random.seed(42)

    with open(_param_path(path), 'r') as f:
        param = json.load(f)

    with open(_cost_path(path), 'r') as f:
        cost = json.load(f)

    with open(_meta_path(path), 'r') as f:
        meta = json.load(f)

    with open(_plan_path(path), 'r') as f:
        plan = json.load(f)

    train_keys_list = list(cost.keys())[:160]
    test_keys_list = list(cost.keys())[160:]

    for param_key in train_keys_list:
        param_values = param[param_key]
        #plan_combinations = generate_combinations(list(cost[param_key].keys()), k)
        ensure_directory(f'{path}/plan_combinations/')
        plan_combinations = process_combinations(f'{path}/plan_combinations/{param_key}_{k}.txt', list(cost[param_key].keys()), k)
        x1, x2, y1, y2 = get_training_pair2(plan_combinations, plan, param_key, cost)
        Z += [list(param_values) for _ in range(len(x1))]
        X1 += x1
        X2 += x2
        Y1 += y1
        Y2 += y2

    for param_key in test_keys_list:
        param_values = param[param_key]
        #plan_combinations = generate_combinations(list(cost[param_key].keys()), k)
        plan_combinations = process_combinations(f'{path}/plan_combinations/{param_key}_{k}.txt', list(cost[param_key].keys()), k)
        x1, x2, y1, y2 = get_training_pair2(plan_combinations, plan, param_key, cost)
        Z_t += [list(param_values) for _ in range(len(x1))]
        X1_t += x1
        X2_t += x2
        Y1_t += y1
        Y2_t += y2

    params, preprocess_info = get_param_info(meta)

    return Z, X1, X2, Y1, Y2, Z_t, X1_t, X2_t, Y1_t, Y2_t, params, preprocess_info


def load_X_all(training_data_file, template_id):
    path = os.path.join(training_data_file, template_id)
    X = []

    with open(_cost_path(path), 'r') as f:
        cost = json.load(f)

    with open(_plan_path(path), 'r') as f:
        plan = json.load(f)

    for param_key in cost.keys():
        for plan_key in cost[param_key].keys():
            X.append(plan[plan_key])

    return X


def load_test_data(training_data_file, template_id):
    path = os.path.join(training_data_file, template_id)
    Z, X1, X2, Y1, Y2 = [], [], [], [], []
    random.seed(42)

    with open(_param_path(path), 'r') as f:
        param = json.load(f)

    with open(_cost_path(path), 'r') as f:
        cost = json.load(f)

    with open(_meta_path(path), 'r') as f:
        meta = json.load(f)

    with open(_plan_path(path), 'r') as f:
        plan = json.load(f)

    keys_list = list(cost.keys())[160:]

    param_keys = random.sample(keys_list, min(10, len(keys_list)))

    for param_key in param_keys:
        param_values = param[param_key]
        candidate_plan = random.sample(list(cost[param_key].keys()), min(k4[4], len(cost[param_key].keys())))
        x1, x2, y1, y2 = get_training_pair(candidate_plan, plan, param_key, cost)
        Z += [list(param_values) for _ in range(len(x1))]
        X1 += x1
        X2 += x2
        Y1 += y1
        Y2 += y2

    return Z, X1, X2, Y1, Y2


def training_pairwise(training_data_file, model_path, template_id, device, pre_trained=0, first_layer=0):
    Z, X1, X2, Y1, Y2, params, preprocess_info = load_training_data(training_data_file, template_id)

    rank_PQO_model = None
    if pre_trained:
        rank_PQO_model = RankPQOModel(None, template_id, preprocess_info, device=device)
        rank_PQO_model.load(model_path, first_layer)
        feature_generator = rank_PQO_model._feature_generator
    else:
        feature_generator = FeatureGenerator()
        feature_generator.fit_pred_model(X1 + X2)

    X1 = feature_generator.transform(X1)
    X2 = feature_generator.transform(X2)
    Z = feature_generator.transform_z(Z, params, preprocess_info)
    print("Training data set size = " + str(len(X1)))

    if not pre_trained:
        assert rank_PQO_model is None
        rank_PQO_model = RankPQOModel(feature_generator, template_id, preprocess_info, device=device)
    rank_PQO_model.fit_with_test(X1, X2, Y1, Y2, Z, pre_trained)

    print("saving model...")
    rank_PQO_model.save(model_path)


def training_no_share(training_data, model_path, device, epochs, data_size):
    all_folders = []
    for subdir, _, files in os.walk(training_data):
        if 'a' in os.path.basename(subdir) and "meta_data.json" in files:
            all_folders.append(os.path.basename(subdir))

    print(all_folders)
    io_time = 0
    training_time = 0
    pre_trained = 0

    X_all = None
    for folder in all_folders:
        X = load_X_all(training_data, folder)
        if X_all is None:
            X_all = X
        else:
            X_all += X

    for folder in all_folders:
        print(f"Training for folder: {folder}")
        model_path_file = os.path.join(model_path, folder)

        t1 = time.time()
        (Z, X1, X2, Y1, Y2, Z_t, X1_t, X2_t, Y1_t, Y2_t,
         params, preprocess_info) = load_training_test_data(training_data, folder, data_size)
        t2 = time.time()

        feature_generator = FeatureGenerator(False, True)
        feature_generator.fit_pred_model(X_all)

        X1 = feature_generator.transform(X1)
        X2 = feature_generator.transform(X2)
        Z = feature_generator.transform_z(Z, params, preprocess_info)
        X1_t = feature_generator.transform(X1_t)
        X2_t = feature_generator.transform(X2_t)
        Z_t = feature_generator.transform_z(Z_t, params, preprocess_info)
        print("Training data set size = " + str(len(X1)))

        rank_PQO_model = RankPQOModel(feature_generator, folder, preprocess_info, device=device)

        rank_PQO_model.fit_with_test2(X1, X2, Y1, Y2, Z, X1_t, X2_t, Y1_t, Y2_t, Z_t, pre_trained, 16, epochs)

        rank_PQO_model.save(model_path_file)

        print(f"Finished training for folder: {folder}")

        t3 = time.time()

        io_time += (t2 - t1)
        training_time += (t3 - t2)

    total_train_acc, total_test_acc, total_train_loss, total_test_loss = 0, 0, 0, 0
    for folder in all_folders:
        template_id = folder
        model_path_file = os.path.join(model_path, folder)

        (Z, X1, X2, Y1, Y2, Z_t, X1_t, X2_t, Y1_t, Y2_t,
         params, preprocess_info) = load_training_test_data(training_data, folder, data_size)

        rank_PQO_model = RankPQOModel(None, template_id, preprocess_info, device=device)
        rank_PQO_model.load(model_path_file, fist_layer=1)
        feature_generator = rank_PQO_model._feature_generator
        X1 = feature_generator.transform(X1)
        X2 = feature_generator.transform(X2)
        Z = feature_generator.transform_z(Z, params, preprocess_info)
        X1_t = feature_generator.transform(X1_t)
        X2_t = feature_generator.transform(X2_t)
        Z_t = feature_generator.transform_z(Z_t, params, preprocess_info)
        print(f"Test results for {folder}")
        train_acc, test_acc, train_loss, test_loss = rank_PQO_model.test2(X1, X2, Y1, Y2, Z, X1_t, X2_t, Y1_t, Y2_t,
                                                                          Z_t, 16)

        total_train_acc += train_acc
        total_test_acc += test_acc
        total_train_loss += train_loss
        total_test_loss += test_loss

    print(f'training accuracy: {total_train_acc / 33}')
    print(f'test accuracy: {total_test_acc / 33}')
    print(f'training loss: {total_train_loss / 33}')
    print(f'test loss: {total_test_loss / 33}')
    print(f'io time: {io_time}')
    print(f'train time: {training_time}')


def training_baseline(training_data, model_path, device):
    # all_folders = [f for f in os.listdir(training_data) if os.path.isdir(os.path.join(training_data, f))]
    all_folders = []
    for subdir, _, files in os.walk(training_data):
        if ('a' in os.path.basename(subdir) and "meta_data.json" in files and "all_plans_by_hybrid_new.json" in files
                and "parameter_new.json" in files and "latency_matrix_pg.json" in files):
            all_folders.append(os.path.basename(subdir))

    print(all_folders)

    io_time = 0
    training_time = 0

    X_all = None
    for folder in all_folders:
        Z, X1, X2, Y1, Y2, params, preprocess_info = load_training_data(training_data, folder)
        if X_all is None:
            X_all = (X1 + X2)
        else:
            X_all += (X1 + X2)

    pre_trained = 0
    for folder in all_folders:
        print(f"Training for folder: {folder}")

        t1 = time.time()

        Z, X1, X2, Y1, Y2, params, preprocess_info = load_training_data(training_data, folder)

        t2 = time.time()

        if pre_trained:
            rank_PQO_model = RankPQOModel(None, folder, preprocess_info, device=device)
            rank_PQO_model.load(model_path)
            feature_generator = rank_PQO_model._feature_generator
        else:
            feature_generator = FeatureGenerator(False, True)
            feature_generator.fit_pred_model(X_all)

        X1 = feature_generator.transform(X1)
        X2 = feature_generator.transform(X2)
        Z = feature_generator.transform_z(Z, params, preprocess_info)
        print("Training data set size = " + str(len(X1)))

        if not pre_trained:
            rank_PQO_model = RankPQOModel(feature_generator, folder, preprocess_info, device=device)

        rank_PQO_model.fit_with_test(X1, X2, Y1, Y2, Z, pre_trained, 16, 50)

        rank_PQO_model.save(model_path)

        pre_trained = 1

        print(f"Finished training for folder: {folder}")

        t3 = time.time()

        io_time += (t2 - t1)
        training_time += (t3 - t2)

    for folder in all_folders:
        template_id = folder

        Z, X1, X2, Y1, Y2, params, preprocess_info = load_training_data(training_data, template_id)
        rank_PQO_model = RankPQOModel(None, template_id, preprocess_info, device=device)
        rank_PQO_model.load(model_path, fist_layer=1)
        feature_generator = rank_PQO_model._feature_generator
        X1 = feature_generator.transform(X1)
        X2 = feature_generator.transform(X2)
        Z = feature_generator.transform_z(Z, params, preprocess_info)
        print(f"Test results for {folder}")
        rank_PQO_model.test(X1, X2, Y1, Y2, Z, 16)

    print(f'io time: {io_time}')
    print(f'train time: {training_time}')


def alternating_training(training_data, model_path, device, epochs, epoch_step, data_size):
    all_folders = []
    for subdir, _, files in os.walk(training_data):
        if 'a' in os.path.basename(subdir) and "meta_data.json" in files:
            all_folders.append(os.path.basename(subdir))

    print(all_folders)
    io_time = 0
    training_time = 0

    all_epochs = [epoch_step for _ in range(epochs // epoch_step)] + (
        [epochs % epoch_step] if epochs % epoch_step else [])

    pre_trained = 0
    epoch_counter = 0
    first_layer = 0

    X_all = None
    for folder in all_folders:
        X = load_X_all(training_data, folder)
        if X_all is None:
            X_all = X
        else:
            X_all += X

    for epoch in all_epochs:
        epoch_counter += epoch
        print(f"Training for total epochs: {epoch_counter}/{epochs}")

        total_train_acc, total_test_acc = 0, 0
        total_train_loss, total_test_loss = 0, 0

        for folder in all_folders:
            # print(f"Training for folder: {folder}")

            t1 = time.time()

            (Z, X1, X2, Y1, Y2, Z_t, X1_t, X2_t, Y1_t, Y2_t,
             params, preprocess_info) = load_training_test_data(training_data, folder, data_size)

            t2 = time.time()

            if pre_trained:
                rank_PQO_model = RankPQOModel(None, folder, preprocess_info, device=device)
                rank_PQO_model.load(model_path, first_layer)
                feature_generator = rank_PQO_model._feature_generator
            else:
                feature_generator = FeatureGenerator(False, True)
                feature_generator.fit_pred_model(X_all)

            X1 = feature_generator.transform(X1)
            X2 = feature_generator.transform(X2)
            Z = feature_generator.transform_z(Z, params, preprocess_info)
            X1_t = feature_generator.transform(X1_t)
            X2_t = feature_generator.transform(X2_t)
            Z_t = feature_generator.transform_z(Z_t, params, preprocess_info)

            # print("Training data set size = " + str(len(X1)))

            if not pre_trained:
                rank_PQO_model = RankPQOModel(feature_generator, folder, preprocess_info, device=device)

            train_acc, test_acc, train_loss, test_loss = rank_PQO_model.fit_with_test2(X1, X2, Y1, Y2, Z, X1_t, X2_t,
                                                                                       Y1_t, Y2_t, Z_t, pre_trained, 16,
                                                                                       epoch)

            total_train_acc += train_acc
            total_test_acc += test_acc
            total_train_loss += train_loss
            total_test_loss += test_loss
            rank_PQO_model.save(model_path)

            pre_trained = 1

            # print(f"Finished training for folder: {folder}")
            t3 = time.time()

            io_time += (t2 - t1)
            training_time += (t3 - t2)

        first_layer = 1
        random.shuffle(all_folders)
        # print(all_folders)
        # print(f'training accuracy: {total_train_acc/(33 * epoch)}')
        # print(f'test accuracy: {total_test_acc / (33 * epoch)}')
        # print(f'training loss: {total_train_loss / (33 * epoch)}')
        # print(f'test loss: {total_test_loss / (33 * epoch)}')

    total_train_acc, total_test_acc, total_train_loss, total_test_loss = 0, 0, 0, 0
    for folder in all_folders:
        template_id = folder

        (Z, X1, X2, Y1, Y2, Z_t, X1_t, X2_t, Y1_t, Y2_t,
         params, preprocess_info) = load_training_test_data(training_data, folder, data_size)
        rank_PQO_model = RankPQOModel(None, template_id, preprocess_info, device=device)
        rank_PQO_model.load(model_path, fist_layer=1)
        feature_generator = rank_PQO_model._feature_generator
        X1 = feature_generator.transform(X1)
        X2 = feature_generator.transform(X2)
        Z = feature_generator.transform_z(Z, params, preprocess_info)
        X1_t = feature_generator.transform(X1_t)
        X2_t = feature_generator.transform(X2_t)
        Z_t = feature_generator.transform_z(Z_t, params, preprocess_info)
        print(f"Test results for {folder}")
        train_acc, test_acc, train_loss, test_loss = rank_PQO_model.test2(X1, X2, Y1, Y2, Z, X1_t, X2_t, Y1_t, Y2_t,
                                                                          Z_t, 16)
        total_train_acc += train_acc
        total_test_acc += test_acc
        total_train_loss += train_loss
        total_test_loss += test_loss

    print(f'training accuracy: {total_train_acc / 33}')
    print(f'test accuracy: {total_test_acc / 33}')
    print(f'training loss: {total_train_loss / 33}')
    print(f'test loss: {total_test_loss / 33}')

    print(f'io time: {io_time}')
    print(f'train time: {training_time}')


def test(training_data, model_path, device, is_share):
    # all_folders = [f for f in os.listdir(training_data) if os.path.isdir(os.path.join(training_data, f))]
    # all_folders = [
    #     "30c", "16c", "21c", "9a", "13b", "13c", "15a", "4a", "23b", "33a", "27b", "2b", "31a"
    # ]
    all_folders = []
    for subdir, _, files in os.walk(training_data):
        if ('a' in os.path.basename(subdir) and "meta_data.json" in files and "plan_by_join_order.json" in files
                and "parameter.json" in files and "latency_matrix.json" in files):
            all_folders.append(os.path.basename(subdir))
    for folder in all_folders:
        template_id = folder
        if is_share:
            model_path_file = model_path
        else:
            model_path_file = os.path.join(model_path, folder)

        Z, X1, X2, Y1, Y2, params, preprocess_info = load_training_data(training_data, template_id)
        rank_PQO_model = RankPQOModel(None, template_id, preprocess_info, device=device)
        rank_PQO_model.load(model_path_file, fist_layer=1)
        feature_generator = rank_PQO_model._feature_generator
        X1 = feature_generator.transform(X1)
        X2 = feature_generator.transform(X2)
        Z = feature_generator.transform_z(Z, params, preprocess_info)
        print(f"Test results for {folder}")
        rank_PQO_model.test(X1, X2, Y1, Y2, Z, 16)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Model training helper")
    parser.add_argument("--function", type=str)
    parser.add_argument("--training_data", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--template_id", type=str)
    parser.add_argument("--pre_trained", type=int)
    parser.add_argument("--first_layer", type=int)
    parser.add_argument("--is_share", type=int)
    parser.add_argument("--data_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--epoch_step", type=int)
    parser.add_argument("--device", type=str, default='cpu')

    args = parser.parse_args()

    function = None
    if args.function is not None:
        function = args.function
    print("function:", function)

    training_data = None
    if args.training_data is not None:
        training_data = args.training_data
    print("training_data:", training_data)

    model_path = None
    if args.model_path is not None:
        model_path = args.model_path
    print("model_path:", model_path)

    template_id = False
    if args.template_id is not None:
        template_id = args.template_id
    print("template_id:", template_id)

    pre_trained = False
    if args.pre_trained is not None:
        pre_trained = args.pre_trained
    print("pre_trained:", pre_trained)

    first_layer = False
    if args.first_layer is not None:
        first_layer = args.first_layer
    print("first_layer:", first_layer)

    data_size = 0
    if args.data_size is not None:
        data_size = args.data_size
    print("data_size:", data_size)

    epochs = 10
    if args.epochs is not None:
        epochs = args.epochs
    print("epochs:", epochs)

    epoch_step = 1
    if args.epoch_step is not None:
        epoch_step = args.epoch_step
    print("epoch_step:", epoch_step)

    is_share = False
    if args.is_share is not None:
        is_share = args.is_share
    print("is_share:", is_share)

    print("Device: ", args.device)

    if function == "training_baseline":
        training_baseline(training_data, model_path, args.device)
    elif function == "alternating_training":
        alternating_training(training_data, model_path, args.device, epochs, epoch_step, data_size)
    elif function == "training_no_share":
        training_no_share(training_data, model_path, args.device, epochs, data_size)
    elif function == "test":
        test(training_data, model_path, args.device, is_share)
    else:
        training_pairwise(training_data, model_path, template_id, args.device, pre_trained, first_layer)
