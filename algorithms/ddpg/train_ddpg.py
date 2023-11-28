import functools
import random
from multiprocessing import Pool

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import ClipAction, TimeLimit

from config import DDPGConfig, inverted_pendulum, half_cheetah_small, half_cheetah
from ddpg import DDPG
from utils.logger import LogMetadata, Logger
from utils.wandb_logger import WandbLogger


def main():
    experiment = 'default'
    config = half_cheetah
    parallel = False
    seeds = range(3)
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

    logger = WandbLogger(LogMetadata(
        algorithm='ddpg', env=config.env_name, experiment=experiment, seed=seed, config=config
    ), keys={'EpisodeReturn', 'EpisodeLength', 'Epoch', 'TotalSteps', 'Time', 'CriticLoss', 'ActorLoss',
             'TestEpisodeReturn', 'TestEpisodeLength'},
        stds={'EpisodeReturn', 'EpisodeLength', 'TestEpisodeReturn', 'TestEpisodeLength'})
    train(config, seed, logger)
    logger.finish()
    # if show_plots:
    #     df = utils.plot.load_data(config.env_name, algorithm_and_experiments={'ddpg': [experiment]})
    #     sns.relplot(df, x='Episode', y='SmoothedReturn', kind='line', errorbar='sd')
    #     plt.show()


def train(config: DDPGConfig, seed: int, logger: Logger):
    env = gym.make(config.env_name)
    env = ClipAction(env)
    if config.max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=config.max_episode_steps)
    env.reset(seed=seed)
    ddpg = DDPG(config, env, logger)
    try:
        ddpg.train()
    except KeyboardInterrupt:
        print('Interrupted.')
    env.close()


if __name__ == '__main__':
    main()
