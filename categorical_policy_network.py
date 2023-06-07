import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from policy_network import PolicyNetwork


class CategoricalPolicyNetwork(PolicyNetwork):

    def __init__(self, obs_space_dims: int, action_space_dims: int, hidden1: int, hidden2: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_space_dims, hidden1),
            nn.Tanh(),
            nn.Linear(hidden1, hidden2),
            nn.Tanh(),
            nn.Linear(hidden2, action_space_dims)
        )

    def __call__(self, obs: torch.tensor) -> torch.distributions.Distribution:
        logits = self.network(obs)
        return Categorical(logits=logits)
