import copy

import torch
import torch.nn as nn

from utils import torch_utils


class Actor(nn.Module):
    def __init__(self, obs_space_dims: int, action_space_dims: int, layer_sizes: list[int], activation: any, limit: float):
        super().__init__()
        self.obs_space_dims, self.action_space_dims = obs_space_dims, action_space_dims
        layer_sizes = [obs_space_dims] + layer_sizes + [action_space_dims]
        layers = torch_utils.make_mlp_layers(layer_sizes, activation)
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        self.limit = limit

    def __call__(self, obs: torch.tensor) -> torch.tensor:
        return self.limit * self.net(obs)

    def deepcopy(self):
        copied_model = Actor(self.obs_space_dims, self.action_space_dims, [1], nn.ReLU, self.limit)
        copied_model.net = copy.deepcopy(self.net)
        for p in copied_model.parameters():
            p.requires_grad = False
        return copied_model
