import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from policy_network import PolicyNetwork

class GaussianPolicyNetwork(PolicyNetwork):

    def __init__(self, obs_space_dims: int, action_space_dims: int, hidden1: int, hidden2: int):
        super().__init__()
        self.mean_net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden1),
            nn.Tanh(),
            nn.Linear(hidden1, hidden2),
            nn.Tanh(),
            nn.Linear(hidden2, action_space_dims)
        )
        log_std = -0.5 * torch.ones(action_space_dims, dtype=torch.float32)
        self.log_stddev = nn.Parameter(log_std)

    def __call__(self, obs: torch.tensor) -> torch.distributions.Distribution:
        means = self.mean_net(obs)
        stddevs = torch.exp(self.log_stddev)
        return Normal(means, stddevs)
