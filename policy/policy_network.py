from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class PolicyNetwork(ABC, nn.Module):
    @abstractmethod
    def __call__(self, obs: torch.tensor) -> torch.distributions.Distribution:
        pass
