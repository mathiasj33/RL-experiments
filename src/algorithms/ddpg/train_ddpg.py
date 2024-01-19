import functools
import random
from multiprocessing import Pool

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.wrappers import ClipAction, TimeLimit, FlattenObservation, RecordVideo

from config import DDPGConfig, inverted_pendulum
from ddpg import DDPG
from utils.logger import LogMetadata, Logger
from utils.no_logger import NoLogger
from utils.wandb_logger import WandbLogger


def main():
    experiment = 'default'
    config = dmc_cartpole
    parallel = False
    seeds = range(1)
    if parallel:
        with Pool(processes=min(5, len(seeds))) as pool:
            pool.map(functools.partial(run_seed, experiment, config), seeds)
    else:
        for seed in seeds:
            run_seed(experiment, config, seed)


def run_seed(experiment, config, seed):
    print(f'Running seed {seed}...')
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    logger = NoLogger(LogMetadata(
        algorithm='ddpg', env=config.env_name, experiment=experiment, seed=seed, config=config
    ), keys={'EpisodeReturn', 'EpisodeLength', 'Epoch', 'TotalSteps', 'Time', 'CriticLoss', 'ActorLoss',
             'TestEpisodeReturn', 'TestEpisodeLength'},
        stds={'EpisodeReturn', 'EpisodeLength', 'TestEpisodeReturn', 'TestEpisodeLength'})
    video_path = f'videos/ddpg/{config.env_name}/{experiment}/seed_{seed}'
    train(config, seed, video_path, logger)
    logger.finish()


def train(config: DDPGConfig, seed: int, video_path: str, logger: Logger):
    ddpg = DDPG(config,
                lambda: make_env(config),
                lambda: make_test_env(config, video_path, record_video=True),
                logger)
    try:
        ddpg.train(seed)
    except KeyboardInterrupt:
        print('Interrupted.')


def make_env(config: DDPGConfig, render_mode=None) -> gym.Env:
    env = gym.make(config.env_name, render_mode=render_mode)
    env = ClipAction(env)
    if isinstance(env.observation_space, spaces.Dict):
        env = FlattenObservation(env)
    if config.max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=config.max_episode_steps)
    return env


def make_test_env(config: DDPGConfig, video_path: str, record_video: bool) -> gym.Env:
    env = make_env(config, render_mode='rgb_array')
    if record_video:
        env = RecordVideo(env, video_path, disable_logger=True,
                          episode_trigger=lambda i: i % config.num_test_episodes == 0)
    return env


if __name__ == '__main__':
    main()
