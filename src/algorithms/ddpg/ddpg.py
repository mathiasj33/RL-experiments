import time

import numpy as np
import torch
from gymnasium.core import Env
from torch.distributions import Uniform, Normal
from torch.nn import MSELoss
from tqdm import tqdm

from actor import Actor
from algorithms.ddpg.config import DDPGConfig
from critic import Critic
from experience_buffer import ExperienceBuffer, ExperienceBatch
from utils import torch_utils
from utils.logger import Logger
from utils.torch_utils import polyak_average


class DDPG:
    def __init__(self, config: DDPGConfig, env: Env, logger: Logger):
        self.device = torch_utils.get_device()
        self.env = env
        obs_space_dims = env.observation_space.shape[0]
        self.action_space_dims = env.action_space.shape[0]
        self.actor = Actor(obs_space_dims, self.action_space_dims, config.actor_layers, config.actor_activation,
                           config.high_clip).to(self.device)
        self.actor_target = self.actor.deepcopy().to(self.device)
        self.actor_optimiser = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic = Critic(obs_space_dims, self.action_space_dims, config.critic_layers, config.critic_activation).to(self.device)
        self.critic_target = self.critic.deepcopy().to(self.device)
        self.critic_optimiser = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.buffer = ExperienceBuffer(obs_space_dims, self.action_space_dims)
        self.config = config
        self.logger = logger
        self.uniform_dist = Uniform(torch.tensor(self.config.low_clip).to(self.device), torch.tensor(self.config.high_clip).to(self.device))
        self.noise_dist = Normal(torch.tensor(0.).to(self.device), torch.tensor(self.config.noise_variance).to(self.device))

    def select_action(self, obs: np.ndarray, step: int) -> torch.tensor:
        if step <= self.config.warmup_steps:
            return self.uniform_dist.sample(torch.Size((self.action_space_dims,)))
        action = self.actor(torch.tensor(obs, dtype=torch.float32).to(self.device))
        action += self.noise_dist.sample(torch.Size((self.action_space_dims,)))
        return torch.clamp(action, min=self.config.low_clip, max=self.config.high_clip)

    def update(self, batch: ExperienceBatch):
        with torch.no_grad():
            best_next_actions = self.actor_target(batch.next_state).detach()
            ys = batch.reward + self.config.gamma * (1 - batch.done) * \
                 self.critic_target(batch.next_state, best_next_actions).flatten().detach()

        criterion = MSELoss()
        critic_loss = criterion(self.critic(batch.state, batch.action).flatten(), ys)
        self.critic_optimiser.zero_grad()
        critic_loss.backward()
        self.critic_optimiser.step()

        self.critic.freeze_grads()
        actor_loss = -self.critic(batch.state, self.actor(batch.state)).mean()
        self.critic.unfreeze_grads()
        self.actor_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_optimiser.step()

        self.logger.store(CriticLoss=critic_loss.item(), ActorLoss=actor_loss.item())
        with torch.no_grad():
            polyak_average(self.critic_target, self.critic, self.config.polyak)
            polyak_average(self.actor_target, self.actor, self.config.polyak)

    def train(self):
        pbar = tqdm(range(1, self.config.num_epochs + 1))
        max_ret = 0
        step = 0
        for epoch in pbar:
            obs, _ = self.env.reset()  # obs: (n_obs,)
            start_time = time.time()
            ret = length = 0
            for _ in range(self.config.steps_per_epoch):
                step += 1
                with torch.no_grad():
                    action = self.select_action(obs, step)
                    next_obs, reward, terminated, truncated, _ = self.env.step(action.detach().cpu().numpy())
                    self.buffer.store(obs, action.detach().cpu().numpy(), float(reward), next_obs, terminated)
                    obs = next_obs
                    ret += float(reward)
                    length += 1
                    if terminated or truncated:
                        self.logger.store(EpisodeReturn=ret, EpisodeLength=length)
                        obs, _ = self.env.reset()
                        ret, length = 0, 0

                if step == self.config.warmup_steps:
                    print('Finished warming up.')
                if step > self.config.update_after and step % self.config.update_every == 0:
                    for _ in range(self.config.update_every):
                        batch = self.buffer.sample(self.config.batch_size)
                        self.update(batch)

            self.logger.store(Epoch=epoch, TotalSteps=step)
            self.test_agent()
            self.logger.store(Time=time.time() - start_time)
            test_return_mean = np.mean(self.logger.get('TestEpisodeReturn'))
            pbar.set_postfix({'return': test_return_mean})
            if test_return_mean > max_ret:
                max_ret = test_return_mean
                self.logger.save_model(self.actor, 'actor')
                self.logger.save_model(self.critic, 'critic')
            self.logger.log()

    def test_agent(self):
        for _ in range(self.config.num_test_episodes):
            obs, _ = self.env.reset()
            ret = length = 0
            with torch.no_grad():
                done = False
                while not done:
                    action = self.actor(torch.tensor(obs, dtype=torch.float32).to(self.device))
                    obs, reward, terminated, truncated, _ = self.env.step(action.detach().cpu().numpy())
                    ret += float(reward)
                    length += 1
                    done = terminated or truncated
            self.logger.store(TestEpisodeReturn=ret, TestEpisodeLength=length)
