import sys
import time

import torch
import numpy as np
from gymnasium.core import Env
from gymnasium.spaces import Box
from tqdm import tqdm

from categorical_actor import CategoricalActor
from config import VPGConfig
from gaussian_actor import GaussianActor
from utils.logger import Logger


class VPG:
    def __init__(self, config: VPGConfig, env: Env, logger: Logger):
        self.env = env
        obs_space_dims = env.observation_space.shape[0]
        action_space_dims = env.action_space.shape[0] if isinstance(env.action_space, Box) else env.action_space.n
        if config.discrete:
            self.actor = CategoricalActor(obs_space_dims, action_space_dims, config.layer_sizes, config.activation)
        else:
            self.actor = GaussianActor(obs_space_dims, action_space_dims, config.layer_sizes, config.activation)
        self.optimiser = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.config = config
        self.logger = logger

    def compute_loss(self, rewards, log_probs):
        running_return = 0
        discounted_returns = []
        for r in rewards[::-1]:
            running_return = r + self.config.gamma * running_return
            discounted_returns.append(running_return)
        discounted_returns.reverse()
        loss = 0
        for log_prob, ret in zip(log_probs, discounted_returns):
            loss -= log_prob.sum() * ret  # multiply diagonal Gaussians - sum their logprobs
        return loss

    def train(self):
        pbar = tqdm(range(self.config.num_episodes))
        total_steps = 0
        max_ret = -sys.maxsize
        for episode in pbar:
            start_time = time.time()
            obs, _ = self.env.reset()
            done = False
            log_probs = []
            rewards = []
            episode_length = 0
            while not done:
                dist = self.actor(torch.tensor(obs, dtype=torch.float32))
                action = dist.sample()
                obs, reward, terminated, truncated, _ = self.env.step(np.array(action))
                log_probs.append(dist.log_prob(action))
                rewards.append(float(reward))
                done = terminated or truncated
                total_steps += 1
                episode_length += 1

            loss = self.compute_loss(rewards, log_probs)
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()

            ret = sum(rewards)
            pbar.set_postfix({'return': ret, 'loss': loss.item()})
            self.logger.store(
                Episode=episode,
                EpisodeLength=episode_length,
                TotalSteps=total_steps,
                Return=ret,
                Loss=loss.item(),
                Time=time.time() - start_time
            )
            self.logger.log()

            if ret > max_ret:
                max_ret = ret
                self.logger.save_model(self.actor)
