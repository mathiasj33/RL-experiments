import copy

import torch
import torch.nn as nn

from utils import torch_utils


class Critic(nn.Module):

    def __init__(self, obs_space_dims: int, action_space_dims: int, layer_sizes: list[int], activation: any):
        super().__init__()
        self.obs_space_dims, self.action_space_dims = obs_space_dims, action_space_dims
        layer_sizes = [obs_space_dims + action_space_dims] + layer_sizes + [1]
        layers = torch_utils.make_mlp_layers(layer_sizes, activation)
        self.net = nn.Sequential(*layers)

    def __call__(self, obs: torch.tensor, action: torch.tensor) -> torch.tensor:
        if len(obs.shape) > 1:  # batch
            concat = torch.concat((obs, action), dim=1)
        else:
            concat = torch.concat((obs, action))
        return self.net(concat)

    def deepcopy(self):
        copied_model = Critic(self.obs_space_dims, self.action_space_dims, [1], nn.ReLU)
        copied_model.net = copy.deepcopy(self.net)
        copied_model.freeze_grads()
        return copied_model

    def freeze_grads(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_grads(self):
        for p in self.parameters():
            p.requires_grad = True
