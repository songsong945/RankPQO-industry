from industry.template_cluster_baseline import calculate_edit_distance_semantic
from industry.template_cluster_by_model import calculate_template_embedding_by_model, calculate_distance


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


def template_cluster_baseline(templates):


def template_cluster_model_based(templates):
