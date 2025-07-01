from industry.template_cluster_baseline import calculate_edit_distance_semantic
from industry.template_cluster_by_model import calculate_template_embedding_by_model, calculate_distance
from train.train_pred_version import get_param_info
from model.model import RankPQOModel
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
import numpy as np
import os
import json


def template_distance_baseline(templates):
    num_templates = len(templates)
    distance_matrix = [[0] * num_templates for _ in range(num_templates)]

    for i in range(num_templates):
        for j in range(num_templates):
            if i == j:
                distance_matrix[i][j] = 0
            elif i < j:
                distance = calculate_edit_distance_semantic(templates[i], templates[j])
                distance_matrix[i][j] = distance
                distance_matrix[j][i] = distance

    return distance_matrix

def template_distance_model(templates, model, parameters, plans, device):
    num_templates = len(templates)
    distance_matrix = [[0] * num_templates for _ in range(num_templates)]

    template_embeddings = []

    for i, template in enumerate(templates):
        template_embeddings.append(calculate_template_embedding_by_model(model, parameters[i], plans[i], device))

    for i in range(num_templates):
        for j in range(num_templates):
            if i == j:
                distance_matrix[i][j] = 0
            elif i < j:
                distance = calculate_distance(template_embeddings[i], template_embeddings[j])
                distance_matrix[i][j] = distance
                distance_matrix[j][i] = distance

    return distance_matrix

def template_embedding_model(
    templates, model, parameters, plans, device,
    feature_generator, preprocess_info
):
    template_embeddings = []

    for i in range(len(templates)):
        embedding = calculate_template_embedding_by_model(
            model=model,
            parameters=parameters[i],
            plans=plans[i],
            device=device,
            feature_generator=feature_generator,
            preprocess_info=preprocess_info
        )
        template_embeddings.append(embedding)

    return template_embeddings



def template_cluster_baseline(all_folders, training_data, k):
    templates = []

    print(f"[DEBUG] training data path: {training_data}")

    for folder in all_folders:
        meta_data_path = os.path.join(training_data, folder, "meta_data.json")
        if os.path.exists(meta_data_path):
            with open(meta_data_path, "r") as f:
                meta_data = json.load(f)
                template = meta_data.get("template", "")
                templates.append(template)

    print(f"[DEBUG] Found {len(templates)} templates")

    distance_matrix = np.array(template_distance_baseline(templates))

    embedding = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    transformed_data = embedding.fit_transform(distance_matrix)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(transformed_data)

    clusters = {}
    for idx, label in enumerate(kmeans.labels_):
        folder = all_folders[idx]
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(folder)

    return clusters



def template_cluster_model_based(all_folders, training_data, model_path, device, k):

    template_embeddings = []

    for folder in all_folders:
        base_path = os.path.join(training_data, folder)

        with open(os.path.join(base_path, "meta_data.json")) as f:
            meta = json.load(f)

        with open(os.path.join(base_path, "parameter_new.json")) as f:
            param_dict = json.load(f)
            param_list = list(param_dict.values())

        with open(os.path.join(base_path, "hybrid_plans.json")) as f:
            plan_dict = json.load(f)
            plan_list = list(plan_dict.values())
        
        params, preprocess_info = get_param_info(meta)

        rank_PQO_model = RankPQOModel(None, folder, preprocess_info, device=device)
        rank_PQO_model.load(model_path, first_layer=1)
        feature_generator = rank_PQO_model._feature_generator

        plans = feature_generator.transform(plan_list)
        parameters = feature_generator.transform_z(param_list, params, preprocess_info)

        embedding = calculate_template_embedding_by_model(
            model=rank_PQO_model,
            parameters=parameters,
            plans=plans,
            device=device
        )
        template_embeddings.append(embedding.detach().cpu().numpy())


    template_embeddings = np.array(template_embeddings)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(template_embeddings)

    clusters = {}
    for idx, label in enumerate(kmeans.labels_):
        folder = all_folders[idx]
        clusters.setdefault(label, []).append(folder)

    return clusters
