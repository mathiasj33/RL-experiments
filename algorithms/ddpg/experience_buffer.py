from dataclasses import dataclass

import numpy as np
import torch

from utils import torch_utils


@dataclass
class ExperienceBatch:
    state: torch.tensor
    action: torch.tensor
    reward: torch.tensor
    next_state: torch.tensor
    done: torch.tensor


class ExperienceBuffer:
    def __init__(self, obs_space_dims: int, action_space_dims: int, size: int = int(1e6)):
        self.device = torch_utils.get_device()
        self.state_data = np.zeros((size, obs_space_dims), dtype=np.float32)
        self.action_data = np.zeros((size, action_space_dims), dtype=np.float32)
        self.reward_data = np.zeros(size, dtype=np.float32)
        self.next_state_data = np.zeros((size, obs_space_dims), dtype=np.float32)
        self.done_data = np.zeros(size, dtype=np.int32)
        self.size = size
        self.full = False
        self.current = 0

    def sample(self, n: int) -> ExperienceBatch:
        max_index = self.size if self.full else self.current
        indices = np.random.randint(low=0, high=max_index, size=(n,))
        batch = dict(state=self.state_data[indices], action=self.action_data[indices], reward=self.reward_data[indices],
                     next_state=self.next_state_data[indices], done=self.done_data[indices])
        batch = {k: torch.tensor(v, dtype=torch.long if k == 'done' else torch.float32).to(self.device) for k, v in batch.items()}
        return ExperienceBatch(**batch)

    def store(self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool):
        self.state_data[self.current] = obs
        self.action_data[self.current] = action
        self.reward_data[self.current] = reward
        self.next_state_data[self.current] = next_obs
        self.done_data[self.current] = int(done)
        self.current += 1
        if self.current >= self.size:
            self.full = True
            self.current = 0
