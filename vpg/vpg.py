import sys

import torch
import numpy as np
from gymnasium.core import Env
from gymnasium.spaces import Box
from tqdm import tqdm

from categorical_actor import CategoricalActor
from config import VPGConfig
from gaussian_actor import GaussianActor


class VPG:
    def __init__(self, config: VPGConfig, env: Env):
        self.env = env
        obs_space_dims = env.observation_space.shape[0]
        action_space_dims = env.action_space.shape[0] if isinstance(env.action_space, Box) else env.action_space.n
        if config.discrete:
            self.actor = CategoricalActor(obs_space_dims, action_space_dims, config.layer_sizes, config.activation)
        else:
            self.actor = GaussianActor(obs_space_dims, action_space_dims, config.layer_sizes, config.activation)
        self.optimiser = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.config = config

    def train(self, returns: list[float]):
        pbar = tqdm(range(self.config.num_episodes))
        max_ret = -sys.maxsize
        for _ in pbar:
            obs, _ = self.env.reset()
            done = False
            log_probs = []
            rewards = []
            while not done:
                dist = self.actor(torch.tensor(obs, dtype=torch.float32))
                action = dist.sample()
                obs, reward, terminated, truncated, _ = self.env.step(np.array(action))
                log_probs.append(dist.log_prob(action))
                rewards.append(float(reward))
                done = terminated or truncated

            running_return = 0
            discounted_returns = []
            for r in rewards[::-1]:
                running_return = r + self.config.gamma * running_return
                discounted_returns.append(running_return)
            discounted_returns.reverse()
            loss = 0
            for log_prob, ret in zip(log_probs, discounted_returns):
                loss -= log_prob.sum() * ret  # multiply diagonal Gaussians - sum their logprobs

            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()

            ret = sum(rewards)
            returns.append(ret)
            pbar.set_postfix({'return': ret})

            if ret > max_ret:
                max_ret = ret
                torch.save(self.actor.state_dict(), f'models/vpg/{self.env.spec.id}.pth')
