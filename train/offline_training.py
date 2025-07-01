import json
import os
import random
import numpy as np
from sklearn.cluster import KMeans
import time
import argparse

from industry.template_cluster import template_cluster_baseline, template_cluster_model_based
from model.feature_ob import FeatureGenerator
from model.model import RankPQOModel
from train.train_pred_version import load_training_data_k

def training_one_model(training_data, all_folders, model_path, device, epochs, epoch_step, data_size):

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
        Z, X1, X2, Y1, Y2, params, preprocess_info = load_training_data_k(training_data, folder, data_size)
        if X_all is None:
            X_all = (X1 + X2)
        else:
            X_all += (X1 + X2)
    for epoch in all_epochs:
        epoch_counter += epoch
        print(f"Training for total epochs: {epoch_counter}/{epochs}")

        for folder in all_folders:
            print(f"Training for folder: {folder}")

            t1 = time.time()

            Z, X1, X2, Y1, Y2, params, preprocess_info = load_training_data_k(training_data, folder, data_size)

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

            print("Training data set size = " + str(len(X1)))

            if not pre_trained:
                rank_PQO_model = RankPQOModel(feature_generator, folder, preprocess_info, device=device)

            rank_PQO_model.fit_with_test(X1, X2, Y1, Y2, Z, pre_trained, 16, epoch)

            rank_PQO_model.save(model_path)

            pre_trained = 1

            print(f"Finished training for folder: {folder}")
            t3 = time.time()

            io_time += (t2 - t1)
            training_time += (t3 - t2)

        first_layer = 1
        random.shuffle(all_folders)
        print(all_folders)

    for folder in all_folders:
        template_id = folder

        Z, X1, X2, Y1, Y2, params, preprocess_info = load_training_data_k(training_data, folder, data_size)
        rank_PQO_model = RankPQOModel(None, template_id, preprocess_info, device=device)
        rank_PQO_model.load(model_path, first_layer=1)
        feature_generator = rank_PQO_model._feature_generator
        X1 = feature_generator.transform(X1)
        X2 = feature_generator.transform(X2)
        Z = feature_generator.transform_z(Z, params, preprocess_info)
        print(f"Test results for {folder}")
        rank_PQO_model.test(X1, X2, Y1, Y2, Z, 16)

    print(f'io time: {io_time}')
    print(f'train time: {training_time}')



def finetune_one_model(training_data, all_folders, input_model_path, output_model_path, device, max_epochs, epoch_step, data_size, fine_tune_ratio=0.2):
    print(all_folders)
    io_time = 0
    training_time = 0
    fine_tune_epochs = int(max_epochs * fine_tune_ratio)

    all_epochs = [epoch_step for _ in range(fine_tune_epochs // epoch_step)] + (
        [fine_tune_epochs % epoch_step] if fine_tune_epochs % epoch_step else [])

    epoch_counter = 0
    X_all = None
    for folder in all_folders:
        Z, X1, X2, Y1, Y2, params, preprocess_info = load_training_data_k(training_data, folder, data_size)
        if X_all is None:
            X_all = (X1 + X2)
        else:
            X_all += (X1 + X2)

    for epoch in all_epochs:
        epoch_counter += epoch
        print(f"Fine-tuning for total epochs: {epoch_counter}/{fine_tune_epochs}")

        for folder in all_folders:
            print(f"Fine-tuning for folder: {folder}")

            t1 = time.time()

            Z, X1, X2, Y1, Y2, params, preprocess_info = load_training_data_k(training_data, folder, data_size)

            t2 = time.time()

            rank_PQO_model = RankPQOModel(None, folder, preprocess_info, device=device)
            rank_PQO_model.load(input_model_path, first_layer=1)
            feature_generator = rank_PQO_model._feature_generator

            X1 = feature_generator.transform(X1)
            X2 = feature_generator.transform(X2)
            Z = feature_generator.transform_z(Z, params, preprocess_info)

            print("Training data set size = " + str(len(X1)))

            rank_PQO_model.fit_with_test(X1, X2, Y1, Y2, Z, 1, 16, epoch)

            rank_PQO_model.save(output_model_path)

            print(f"Finished fine-tuning for folder: {folder}")
            t3 = time.time()

            io_time += (t2 - t1)
            training_time += (t3 - t2)

        random.shuffle(all_folders)
        print(all_folders)

    for folder in all_folders:
        template_id = folder

        Z, X1, X2, Y1, Y2, params, preprocess_info = load_training_data_k(training_data, folder, 3000)
        rank_PQO_model = RankPQOModel(None, template_id, preprocess_info, device=device)
        rank_PQO_model.load(output_model_path, first_layer=1)
        feature_generator = rank_PQO_model._feature_generator
        X1 = feature_generator.transform(X1)
        X2 = feature_generator.transform(X2)
        Z = feature_generator.transform_z(Z, params, preprocess_info)
        print(f"Test results for {folder}")
        rank_PQO_model.test(X1, X2, Y1, Y2, Z, 16)

    print(f'io time: {io_time}')
    print(f'train time: {training_time}')



def hierarchical_training_and_clustering(training_data, model_path, device, epochs, epoch_step, k, data_size):
    templates = []
    all_folders = []

    for subdir, _, files in os.walk(training_data):
        if ("meta_data.json" in files):
            
            folder = os.path.basename(subdir)
            meta_path = os.path.join(subdir, "meta_data.json")

            with open(meta_path, "r") as f:
                meta = json.load(f)
                template = meta.get("template", "")
                if template:
                    templates.append(template)
                    all_folders.append(folder)

    start = time.time()

    training_one_model(
        training_data=training_data,
        all_folders=all_folders,
        model_path=model_path,
        device=device,
        epochs=epochs,
        epoch_step=epoch_step,
        data_size=data_size
    )

    clusters = template_cluster_model_based(
        all_folders, training_data, model_path, device, k
    )

    cluster_mapping = {}
    for cluster_id, folder_list in clusters.items():
        
        cluster_mapping[str(cluster_id)] = folder_list


        print(f"Fine-tuning model for cluster {cluster_id}")

        cluster_model_dir = os.path.join(model_path, str(cluster_id))
        os.makedirs(cluster_model_dir, exist_ok=True)

        finetune_one_model(
            training_data=training_data,
            all_folders=folder_list,
            input_model_path=model_path,
            output_model_path=cluster_model_dir,
            device=device,
            max_epochs=epochs,
            epoch_step=epoch_step,
            data_size=data_size,
            fine_tune_ratio=0.2
        )
    
    end = time.time()

    cluster_json_path = os.path.join(model_path, "cluster.json")
    with open(cluster_json_path, "w") as f:
        json.dump(cluster_mapping, f, indent=2)
    print(f"Cluster mapping saved to {cluster_json_path}")
    print(f"total time is {end - start} s")



def edited_based_clustering_and_training(training_data, model_path, device, epochs, epoch_step, k, data_size):
    all_folders = []
    for subdir, _, files in os.walk(training_data):
        if ("meta_data.json" in files):
            all_folders.append(os.path.basename(subdir))

    clusters = template_cluster_baseline(all_folders, training_data, k)

    cluster_mapping = {}

    for cluster_id, folder_list in clusters.items():
        print(f"Processing cluster {cluster_id} with {len(folder_list)} folders")

        cluster_model_path = os.path.join(model_path, str(cluster_id))
        os.makedirs(cluster_model_path, exist_ok=True)

        training_one_model(
            training_data=training_data,
            all_folders=folder_list,
            model_path=cluster_model_path,
            device=device,
            epochs=epochs,
            epoch_step=epoch_step,
            data_size=data_size,
        )

        cluster_mapping[str(cluster_id)] = folder_list

    cluster_json_path = os.path.join(model_path, "cluster.json")
    with open(cluster_json_path, "w") as f:
        json.dump(cluster_mapping, f, indent=2)

    print(f"Cluster mapping saved to {cluster_json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Model training helper")

    parser.add_argument("--function", type=str, required=True, help="Function to run")
    parser.add_argument("--training_data", type=str, required=True, help="Path to training data")
    parser.add_argument("--model_path", type=str, required=True, help="Path to save models")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use, e.g., 'cuda' or 'cpu'")
    parser.add_argument("--epochs", type=int, default=10, help="Total number of epochs")
    parser.add_argument("--epoch_step", type=int, default=1, help="Epoch step size")
    parser.add_argument("--cluster", type=int, default=10, help="Number of clusters for template clustering")
    parser.add_argument("--data_size", type=int, default=300, help="Training data size per folder")

    args = parser.parse_args()

    print("Function:", args.function)
    print("Training data:", args.training_data)
    print("Model path:", args.model_path)
    print("Device:", args.device)
    print("Epochs:", args.epochs)
    print("Epoch step:", args.epoch_step)
    print("Num clusters:", args.cluster)
    print("Data size:", args.data_size)

    if args.function == "edited_cluster_train":
        edited_based_clustering_and_training(
            training_data=args.training_data,
            model_path=args.model_path,
            device=args.device,
            epochs=args.epochs,
            epoch_step=args.epoch_step,
            k=args.cluster,
            data_size=args.data_size
        )

    elif args.function == "hierarchical_cluster_train":
        hierarchical_training_and_clustering(
            training_data=args.training_data,
            model_path=args.model_path,
            device=args.device,
            epochs=args.epochs,
            epoch_step=args.epoch_step,
            k=args.cluster,
            data_size=args.data_size
        )

    else:
        raise ValueError(f"Unsupported function name: {args.function}")


