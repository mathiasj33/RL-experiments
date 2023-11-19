import random

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from gymnasium.wrappers import ClipAction

import utils.plot
from config import VPGConfig, inverted_pendulum_config, cartpole_config
from utils.no_logger import NoLogger
from utils.file_logger import FileLogger
from utils.logger import LogMetadata, Logger
from utils.wandb_logger import WandbLogger
from vpg import VPG

sns.set_theme()


def main():
    experiment = 'default'
    config = inverted_pendulum_config
    show_plots = False
    seeds = range(5)
    for seed in seeds:
        print(f'Running seed {seed}...')
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        logger = WandbLogger(LogMetadata(
            algorithm='vpg',
            env=config.env_name,
            experiment=experiment,
            seed=seed,
            config=config
        ), keys={'Episode', 'EpisodeLength', 'Return', 'Loss', 'TotalSteps', 'Time'}, stds={'Return', 'Loss'})
        train(config, seed, logger)
        logger.finish()
        if show_plots:
            df = utils.plot.load_data(config.env_name, algorithm_and_experiments={'vpg': [experiment]})
            sns.relplot(df, x='Episode', y='SmoothedReturn', kind='line', errorbar='sd')
            plt.show()


def train(config: VPGConfig, seed: int, logger: Logger):
    env = gym.make(config.env_name)
    if not config.discrete:
        env = ClipAction(env)
    env.reset(seed=seed)
    vpg = VPG(config, env, logger)
    try:
        vpg.train()
    except KeyboardInterrupt:
        print('Interrupted.')
    env.close()


if __name__ == '__main__':
    main()
