import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


class CategoricalActor(nn.Module):
    def __init__(self, obs_space_dims: int, action_space_dims: int, layer_sizes: list[int], activation: any):
        super().__init__()
        layer_sizes = [obs_space_dims] + layer_sizes
        layers = []
        for i in range(len(layer_sizes) - 1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            nn.init.orthogonal_(layer.weight)
            layers.append(layer)
            layers.append(activation())
        output_layer = nn.Linear(layer_sizes[-1], action_space_dims)
        nn.init.orthogonal_(output_layer.weight)
        layers.append(output_layer)
        self.net = nn.Sequential(*layers)

    def __call__(self, obs: torch.tensor) -> torch.distributions.Distribution:
        logits = self.net(obs)
        return Categorical(logits=logits)
