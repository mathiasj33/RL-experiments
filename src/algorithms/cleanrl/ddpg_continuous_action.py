import random
import time
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gymnasium.wrappers import FlattenObservation, ClipAction, TimeLimit
from stable_baselines3.common.buffers import ReplayBuffer

from algorithms.ddpg.actor import Actor
from algorithms.ddpg.config import inverted_pendulum, DDPGConfig, dmc_cartpole
from algorithms.ddpg.critic import Critic
from utils.logger import LogMetadata, Logger
from utils.no_logger import NoLogger
from utils.wandb_logger import WandbLogger


def make_env(config: DDPGConfig, experiment: str, seed: int, epoch: int, capture_video=False):
    def thunk():
        if capture_video and not config.env_name.startswith('dm_control'):
            env = gym.make(config.env_name, render_mode='rgb_array')
            env = gym.wrappers.RecordVideo(env, f'videos/ddpg/{config.env_name}/{experiment}/epoch{epoch}',
                                           disable_logger=True, step_trigger=lambda x: x == 0)
        else:
            env = gym.make(config.env_name)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        if config.env_name.startswith('dm_control'):
            env = FlattenObservation(env)
        env = ClipAction(env)
        if config.max_episode_steps is not None:
            env = TimeLimit(env, max_episode_steps=config.max_episode_steps)
        return env

    return thunk


def evaluate(
        config: DDPGConfig,
        experiment: str,
        epoch: int,
        make_env: Callable,
        eval_episodes: int,
        logger: Logger,
        Model: tuple[nn.Module, nn.Module],
        device: torch.device = torch.device("cpu"),
):
    envs = gym.vector.SyncVectorEnv([make_env(config, experiment, 0, epoch, capture_video=True)])
    actor, qf = Model

    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        with torch.no_grad():
            actions = actor(torch.Tensor(obs).to(device))

        next_obs, _, _, _, infos = envs.step(actions.cpu().numpy())
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
                logger.store(TestEpisodeReturn=info["episode"]["r"][0], TestEpisodeLength=info["episode"]["l"][0])
        obs = next_obs


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )
    config = dmc_cartpole
    experiment = 'server-clean'
    seed = 0
    logger = WandbLogger(LogMetadata(
        algorithm='ddpg', env=config.env_name, experiment=experiment, seed=seed, config=config
    ), keys={'EpisodeReturn', 'EpisodeLength', 'Epoch', 'TotalSteps', 'Time', 'CriticLoss', 'ActorLoss',
             'TestEpisodeReturn', 'TestEpisodeLength'},
        stds={'EpisodeReturn', 'EpisodeLength', 'TestEpisodeReturn', 'TestEpisodeLength'})

    # TRY NOT TO MODIFY: seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(config, experiment, seed, 0)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    obs_space_dims = envs.single_observation_space.shape[0]
    action_space_dims = envs.single_action_space.shape[0]
    actor = Actor(obs_space_dims, action_space_dims, config.actor_layers, config.actor_activation, config.high_clip).to(
        device)
    qf1 = Critic(obs_space_dims, action_space_dims, config.critic_layers, config.critic_activation).to(device)
    qf1_target = Critic(obs_space_dims, action_space_dims, config.critic_layers, config.critic_activation).to(device)
    target_actor = Actor(obs_space_dims, action_space_dims, config.actor_layers, config.actor_activation, config.high_clip).to(
        device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()), lr=config.critic_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=config.actor_lr)

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        int(1e6),
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    max_ret = 0
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=seed)
    for global_step in range(config.num_epochs * config.steps_per_epoch):
        # ALGO LOGIC: put action logic here
        if global_step < config.warmup_steps:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                actions = actor(torch.Tensor(obs).to(device)).cpu()
                actions += torch.normal(0, config.noise_variance, size=actions.shape)
                actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                logger.store(EpisodeReturn=info["episode"]["r"], EpisodeLength=info["episode"]["l"])
                break

        # TRY NOT TO MODIFY: save data to replay buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > config.update_after:
            data = rb.sample(config.batch_size)
            with torch.no_grad():
                next_state_actions = target_actor(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * config.gamma * (
                    qf1_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

            # optimize the model
            q_optimizer.zero_grad()
            qf1_loss.backward()
            q_optimizer.step()

            # Important: we ignore the update frequency here
            actor_loss = -qf1(data.observations, actor(data.observations)).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # update the target network
            tau = 1 - config.polyak
            for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            logger.store(CriticLoss=qf1_a_values.mean().item(), ActorLoss=actor_loss.item())

        if (global_step + 1) % config.steps_per_epoch == 0:
            epoch = (global_step + 1) // config.steps_per_epoch
            logger.store(Epoch=epoch, TotalSteps=global_step)

            evaluate(
                config,
                experiment,
                epoch,
                make_env,
                config.num_test_episodes,
                logger,
                Model=(actor, qf1),
                device=device,
            )

            logger.store(Time=time.time() - start_time)
            start_time = time.time()
            test_return_mean = np.mean(logger.get('TestEpisodeReturn'))
            print(f'test_return_mean={test_return_mean}')
            if test_return_mean > max_ret:
                max_ret = test_return_mean
                logger.save_model(actor, 'actor')
                logger.save_model(qf1, 'critic')
            logger.log()

    envs.close()
    logger.finish()
