import random
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from gymnasium.wrappers import ClipAction, RecordEpisodeStatistics
import os

from config import VPGConfig, inverted_pendulum_config, cartpole_config, acrobot_config
from vpg import VPG

def main():
    experiment_name = 'default'
    config = inverted_pendulum_config
    plot = False
    seeds = range(1, 5)
    for seed in seeds:
        print(f'Running seed {seed}...')
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        returns = train(config, seed)
        path = f'results/vpg/{config.env_name}/{experiment_name}'
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(f'{path}/seed{seed}.npy', returns)
        print('Saved results.')
        if plot:
            plt.plot(returns)
            plt.show()


def train(config: VPGConfig, seed: int) -> list[int]:
    env = gym.make(config.env_name)
    env = RecordEpisodeStatistics(env)
    if not config.discrete:
        env = ClipAction(env)
    env.reset(seed=seed)
    vpg = VPG(config, env)
    returns = []
    try:
        vpg.train(returns)
    except KeyboardInterrupt:
        print('Interrupted.')
    env.close()
    return returns


if __name__ == '__main__':
    main()
