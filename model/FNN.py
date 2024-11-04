import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class OneHotNN(nn.Module):
    def __init__(self, num_classes):
        super(OneHotNN, self).__init__()
        self.num_classes = num_classes

    def forward(self, x):
        # Batch x 1
        one_hot = torch.zeros(x.size(0), self.num_classes, device=x.device)
        one_hot.scatter_(1, x.long(), 1)
        return one_hot

class ParameterEmbeddingNet(nn.Module):
    def __init__(self, template_id, preprocessing_infos, embed_dim=16):
        super(ParameterEmbeddingNet, self).__init__()

        self.id = template_id
        self.embed_dim = embed_dim

        layers = []
        self.length = len(preprocessing_infos)
        embed_len = 0

        for info in preprocessing_infos:
            if info["type"] == "one_hot":
                layers.append(OneHotNN(info['max_len']))
                embed_len += info['max_len']
            elif info["type"] == "std_normalization":
                layers.append(nn.Identity())
                embed_len += 1
            elif info["type"] == "embedding":
                layers.append(nn.Embedding(info["max_len"], embed_dim))
                embed_len += embed_dim
            else:
                raise ValueError(f"Unknown preprocessing type: {info['type']}")

        self.embed_layers = nn.ModuleList(layers)
        self.embed_len = embed_len

        self.fc1 = nn.Linear(embed_len, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        batch_size = x.size(0)
        x_l = torch.split(x, 1, dim=-1)  # list of Batch x 1
        embedded = []
        for x_i, e in zip(x_l, self.embed_layers):
            if not isinstance(e, nn.Identity):
                embedded.append(e(x_i.long()).view(batch_size, -1))
            else:
                embedded.append(e(x_i))

        embedded = torch.concat(embedded, -1)

        x = F.relu(self.fc1(embedded))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class FNN():
    def __init__(self, feature_generator, template_id, preprocessing_infos, device, is_predict=True) -> None: