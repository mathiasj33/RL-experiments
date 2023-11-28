import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from utils import torch_utils


class GaussianActor(nn.Module):
    def __init__(self, obs_space_dims: int, action_space_dims: int, layer_sizes: list[int], activation: any):
        super().__init__()
        layer_sizes = [obs_space_dims] + layer_sizes + [action_space_dims]
        layers = torch_utils.make_mlp_layers(layer_sizes, activation)
        self.mean_net = nn.Sequential(*layers)
        log_stddev = torch.zeros(action_space_dims, dtype=torch.float32)
        self.log_stddev = nn.Parameter(log_stddev)

    def __call__(self, obs: torch.tensor) -> torch.distributions.Distribution:
        means = self.mean_net(obs)
        stddevs = torch.exp(self.log_stddev)
        return Normal(means, stddevs)
