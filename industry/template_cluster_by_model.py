import torch
import numpy as np
from scipy.spatial.distance import cosine, euclidean


def calculate_template_embedding_by_model(model, parameters, plans, device):
    """
    Calculate the embedding for a given template using the model.

    Args:
        model: A model with parameter_net and plan_net.
        parameters: A list or batch of parameter vectors for the template.
        plans: List of plans for the template.
        device: Torch device (e.g., 'cpu' or 'cuda').

    Returns:
        Tensor: The embedding vector for the template.
    """
    # Process parameters (batch of parameter vectors)
    parameter_array = np.array(parameters)
    parameter_x = torch.tensor(parameter_array, dtype=torch.float32).to(device)
    param_embeddings = model.parameter_net(parameter_x)
    avg_param_embedding = torch.mean(param_embeddings, dim=0)

    # Process plans (same as before)
    plan_x = model.plan_net.build_trees(plans)  # Preprocess plans
    plan_embeddings = model.plan_net(plan_x)  # Forward pass through plan_net
    avg_plan_embedding = torch.mean(plan_embeddings, dim=0)  # Mean across plans

    # Combine parameter and plan embeddings (e.g., concatenate or add)
    template_embedding = torch.cat((avg_param_embedding, avg_plan_embedding), dim=-1)

    return template_embedding


def calculate_distance(embedding1, embedding2):
    """
    Calculate the distance and similarity between two embeddings.

    Args:
        embedding1: The first embedding vector.
        embedding2: The second embedding vector.

    Returns:
        Tuple[float, float]: Euclidean distance and cosine similarity.
    """
    # Convert to numpy arrays for compatibility
    embedding1 = embedding1.cpu().detach().numpy()
    embedding2 = embedding2.cpu().detach().numpy()

    distance = euclidean(embedding1, embedding2)

    return distance


